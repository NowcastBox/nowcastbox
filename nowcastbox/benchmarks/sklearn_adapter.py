r"""Adapter turning any scikit-learn-compatible regressor into a benchmark (plan I11).

The higher-frequency predictors are completed at the ragged edge by iterated AR
forecasts and aggregated to the target frequency (as in a bridge equation, Baffigi,
Golinelli & Parigi, 2004); the resulting target-frequency features (optionally with
lags of the features and of the target) are passed to the regressor's
``fit(X, y)`` / ``predict(X)``. Any object following the scikit-learn estimator API
(Buitinck et al., 2013) works: linear models, regularised regressions, random
forests, gradient boosting, pipelines... scikit-learn itself is optional: it is only
imported (lazily) to clone the estimator; otherwise :func:`copy.deepcopy` is used.

References
----------
Buitinck, L. et al. (2013). API design for machine learning software: experiences from
the scikit-learn project. *ECML PKDD Workshop: Languages for Data Mining and Machine
Learning*, 108-122.

Baffigi, A., Golinelli, R. & Parigi, G. (2004). Bridge models to forecast the euro area
GDP. *International Journal of Forecasting*, 20(3), 447-460.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from nowcastbox.benchmarks._utils import check_int, resolve_predictors
from nowcastbox.core.base import BaseBenchmark
from nowcastbox.core.data import MixedFrequencyData
from nowcastbox.core.exceptions import NowcastDataError
from nowcastbox.core.frequency import AggregationType
from nowcastbox.models.bridge import aggregate_to_target, ar_extend, resolve_aggregation_weights

__all__ = ["SklearnBenchmark", "clone_estimator"]


def clone_estimator(estimator: Any) -> Any:
    """Unfitted copy of a scikit-learn-compatible estimator.

    Uses :func:`sklearn.base.clone` when scikit-learn is installed and the object
    supports it, and :func:`copy.deepcopy` otherwise.

    Parameters
    ----------
    estimator : object
        Estimator with ``fit``/``predict``.

    Returns
    -------
    object
        Copy of ``estimator``.

    Examples
    --------
    >>> from sklearn.linear_model import Ridge
    >>> clone_estimator(Ridge(alpha=2.0)).alpha
    2.0
    """
    try:
        from sklearn.base import clone
    except ImportError:
        return copy.deepcopy(estimator)
    try:
        return clone(estimator)
    except (TypeError, ValueError, RuntimeError):
        return copy.deepcopy(estimator)


class SklearnBenchmark(BaseBenchmark):
    """Benchmark wrapping a scikit-learn-compatible regressor.

    Parameters
    ----------
    estimator : object
        Regressor with ``fit(X, y)`` and ``predict(X)`` (cloned at every fit).
    predictors : sequence of str, optional
        Predictors (default: every series except the target).
    aggregation : str, AggregationType or sequence of float, default "average"
        Aggregation of higher-frequency predictors to the target frequency
        (``"average"``, ``"flow"``, ``"stock"``, ``"mariano_murasawa"`` or explicit
        weights, most recent period first).
    regressor_lags : int, default 0
        Lags (target periods) of each aggregated predictor added as features.
    target_lags : int, default 0
        Lags of the target added as features (iterated with own predictions when not
        observed).
    ar_lags : int, default 1
        AR order completing the predictors at the ragged edge.
    label : str, optional
        Display name (default ``"Sklearn(<EstimatorClass>)"``).

    Attributes
    ----------
    estimator_ : object
        Fitted clone of ``estimator``.
    feature_names_ : list of str
        Names of the feature columns.
    n_obs_ : int
        Number of training rows.

    Raises
    ------
    TypeError
        At fit time, if ``estimator`` lacks ``fit`` or ``predict``.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from sklearn.linear_model import LinearRegression
    >>> from nowcastbox.benchmarks import SklearnBenchmark
    >>> rng = np.random.default_rng(0)
    >>> idx = pd.period_range("2000-01", periods=60, freq="M")
    >>> x = rng.standard_normal(60)
    >>> y = (1.0 + 2.0 * pd.Series(x, index=idx).rolling(3).mean()).where(idx.month % 3 == 0)
    >>> frame = pd.DataFrame({"x": x, "y": y}, index=idx)
    >>> bench = SklearnBenchmark(LinearRegression()).fit(frame, "y", frequency={"x": "M", "y": "Q"})
    >>> bench.name
    'Sklearn(LinearRegression)'
    >>> round(float(bench.estimator_.coef_[0]), 6)
    2.0
    """

    def __init__(
        self,
        estimator: Any,
        predictors: Sequence[str] | None = None,
        aggregation: str | AggregationType | Sequence[float] = "average",
        regressor_lags: int = 0,
        target_lags: int = 0,
        ar_lags: int = 1,
        label: str | None = None,
    ) -> None:
        self.estimator = estimator
        self.predictors = predictors
        self.aggregation = aggregation
        self.regressor_lags = regressor_lags
        self.target_lags = target_lags
        self.ar_lags = ar_lags
        self.label = label

    _panel: MixedFrequencyData | None = None
    _names: list[str] | None = None
    _estimator: Any = None
    _n_obs: int = 0
    _tname: str = ""
    _feature_names: list[str] | None = None

    @property
    def name(self) -> str:
        """Display name (``label`` or ``"Sklearn(<EstimatorClass>)"``)."""
        if self.label is not None:
            return str(self.label)
        return f"Sklearn({type(self.estimator).__name__})"

    def _validate(self) -> None:
        for attr in ("fit", "predict"):
            if not callable(getattr(self.estimator, attr, None)):
                raise TypeError(
                    f"estimator must implement fit(X, y) and predict(X); "
                    f"{type(self.estimator).__name__} has no {attr!r}."
                )
        check_int("regressor_lags", self.regressor_lags, 0)
        check_int("target_lags", self.target_lags, 0)
        check_int("ar_lags", self.ar_lags, 0)

    def _features(self, periods: pd.PeriodIndex) -> pd.DataFrame:
        """Aggregated predictors (and their lags) on ``periods`` (contiguous)."""
        assert self._panel is not None  # noqa: S101
        assert self._names is not None  # noqa: S101
        target_freq = self._panel.metadata[self._tname].frequency
        columns: dict[str, pd.Series] = {}
        for name in self._names:
            freq = self._panel.metadata[name].frequency
            native = self._panel.to_native(name)
            end = periods[-1].asfreq(freq.pandas_freq, how="E")
            grid = pd.period_range(native.index[0], max(end, native.index[-1]), freq=end.freq)
            values = native.reindex(grid).to_numpy(dtype=float)
            last = int(np.flatnonzero(np.isfinite(values))[-1])
            values = ar_extend(values, len(grid) - last - 1, int(self.ar_lags))
            weights = resolve_aggregation_weights(self.aggregation, freq, target_freq)
            columns[name] = aggregate_to_target(pd.Series(values, index=grid), weights, periods)
        frame = pd.DataFrame(columns, index=periods)
        lagged = [
            frame.shift(j).add_suffix(f"_lag{j}") for j in range(1, int(self.regressor_lags) + 1)
        ]
        return pd.concat([frame, *lagged], axis=1)

    def _with_target_lags(self, frame: pd.DataFrame, history: pd.Series) -> pd.DataFrame:
        out = frame.copy()
        for j in range(1, int(self.target_lags) + 1):
            out[f"{history.name}_lag{j}"] = history.shift(j).to_numpy()
        return out

    def _periods(self, end: pd.Period | None = None) -> pd.PeriodIndex:
        assert self._panel is not None  # noqa: S101
        freq = self._panel.metadata[self._tname].frequency.pandas_freq
        first = self._panel.start.asfreq(freq)
        last = self._panel.end.asfreq(freq)
        return pd.period_range(first, last if end is None else max(last, end), freq=freq)

    def _fit(self, data: MixedFrequencyData, target: str) -> None:
        self._validate()
        self._names = resolve_predictors(data, target, self.predictors)
        self._panel = data.select([c for c in data.columns if c in {target, *self._names}])
        self._tname = target
        periods = self._periods()
        y = data.to_native(target).reindex(periods).rename(target)
        X = self._with_target_lags(self._features(periods), y)
        ok = y.notna().to_numpy() & np.isfinite(X.to_numpy(dtype=float)).all(axis=1)
        if int(ok.sum()) < 2:
            raise NowcastDataError(
                f"Too few complete observations ({int(ok.sum())}) to train the estimator."
            )
        estimator = clone_estimator(self.estimator)
        estimator.fit(X.to_numpy(dtype=float)[ok], y.to_numpy(dtype=float)[ok])
        self._estimator = estimator
        self._feature_names = list(X.columns)
        self._n_obs = int(ok.sum())

    def _predict_frame(self, X: pd.DataFrame) -> np.ndarray:
        values = X.to_numpy(dtype=float)
        out = np.full(values.shape[0], np.nan)
        ok = np.isfinite(values).all(axis=1)
        if ok.any():
            out[ok] = np.asarray(self._estimator.predict(values[ok]), dtype=float).ravel()
        return out

    def _predict(self, periods: pd.PeriodIndex) -> pd.Series:
        assert self._panel is not None  # noqa: S101
        full = self._periods(periods.max() if len(periods) else None)
        features = self._features(full)
        history = self._panel.to_native(self._tname).reindex(full).rename(self._tname)
        if int(self.target_lags) == 0:
            preds = pd.Series(self._predict_frame(features), index=full)
        else:
            preds = self._iterate(features, history)
        return pd.Series(preds.reindex(periods).to_numpy(dtype=float), index=periods)

    def _iterate(self, features: pd.DataFrame, history: pd.Series) -> pd.Series:
        """Predictions with target lags; unobserved lags replaced by predictions."""
        filled = history.astype(float).copy()
        preds = pd.Series(np.nan, index=features.index)
        X = self._with_target_lags(features, filled)
        preds[:] = self._predict_frame(X)
        last = filled.last_valid_index()
        start = 0 if last is None else int(np.flatnonzero(filled.index == last)[0]) + 1
        for i in range(start, len(filled)):
            X = self._with_target_lags(features.iloc[: i + 1], filled.iloc[: i + 1])
            value = self._predict_frame(X.iloc[[i]])[0]
            preds.iloc[i] = value
            filled.iloc[i] = value
        return preds

    @property
    def estimator_(self) -> Any:
        """Fitted clone of the estimator."""
        self._check_fitted()
        return self._estimator

    @property
    def feature_names_(self) -> list[str]:
        """Feature column names."""
        self._check_fitted()
        assert self._feature_names is not None  # noqa: S101
        return list(self._feature_names)

    @property
    def n_obs_(self) -> int:
        """Number of training rows."""
        self._check_fitted()
        return self._n_obs
