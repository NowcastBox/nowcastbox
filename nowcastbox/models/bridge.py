r"""Bridge equations: regressions of a low-frequency target on aggregated indicators.

A bridge equation links a low-frequency target (e.g. quarterly GDP growth) to
indicators observed at a higher frequency, after aggregating them to the target
frequency (Baffigi, Golinelli & Parigi, 2004; Giannone, Reichlin & Small, 2008,
eq. for :math:`y^Q_t`):

.. math::

    y_t = \alpha + \sum_{k} \sum_{l=0}^{L} \beta_{k,l}\, \bar x^{(k)}_{t-l}
          + \sum_{j=1}^{J} \phi_j\, y_{t-j} + e_t,
    \qquad \bar x^{(k)}_t = \sum_{i} w_i\, x^{(k)}_{\tau(t) - i},

where :math:`\tau(t)` is the last high-frequency period of target period :math:`t`
and :math:`w` the aggregation weights (average, sum, end of period or
Mariano-Murasawa). Indicators not yet released at the end of the sample (the ragged
edge) are extended with iterated AR(:math:`p`) forecasts before aggregation, the
standard bridge-model practice (Baffigi et al., 2004).

This module provides

* :class:`BridgeRegression` / :func:`fit_bridge_regression` - the OLS regression
  itself (statsmodels OLS), reused by :class:`~nowcastbox.models.TwoStepDFM` with the
  aggregated factors as regressors;
* :class:`BridgeEquation` - a complete nowcaster (``fit(data, target)``) built on the
  contracts of :mod:`nowcastbox.core`;
* :func:`ar_extend` - iterated AR(:math:`p`) extension of a series.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from nowcastbox._logging import get_logger
from nowcastbox.core.base import BaseNowcaster
from nowcastbox.core.data import FrequencySpec, MixedFrequencyData
from nowcastbox.core.exceptions import NowcastDataError
from nowcastbox.core.frequency import (
    AggregationType,
    Frequency,
    aggregation_ratio,
)
from nowcastbox.core.results import NowcastResults, build_nowcast_frame
from nowcastbox.preprocessing.aggregation import rolling_aggregate

__all__ = [
    "BridgeEquation",
    "BridgeRegression",
    "BridgeResults",
    "aggregate_to_target",
    "ar_extend",
    "fit_bridge_regression",
    "resolve_aggregation_weights",
]

logger = get_logger(__name__)

_COV_TYPES = ("nonrobust", "HC0", "HC1", "HC2", "HC3", "HAC")
_FILL_METHODS = ("ar", "none")


# ---------------------------------------------------------------------- regression
@dataclass(frozen=True, eq=False)
class BridgeRegression:
    r"""Fitted bridge regression :math:`y_t = \alpha + \beta' x_t + e_t` (OLS).

    Parameters
    ----------
    target : str
        Name of the dependent variable.
    params : pandas.Series
        Coefficients (``"const"`` first when an intercept is used).
    ols : statsmodels RegressionResults
        Full statsmodels results (standard errors, tests, ``summary()``).
    exog_names : list of str
        Regressor names (without the constant).
    add_constant : bool
        Whether an intercept was estimated.
    sample : pandas.PeriodIndex
        Periods used in the estimation.
    cov_type : str
        Covariance estimator of the coefficients.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from nowcastbox.models.bridge import fit_bridge_regression
    >>> idx = pd.period_range("2000Q1", periods=40, freq="Q")
    >>> x = pd.DataFrame({"f1": np.sin(np.arange(40.0))}, index=idx)
    >>> y = pd.Series(1.0 + 2.0 * x["f1"], name="gdp")
    >>> reg = fit_bridge_regression(y, x)
    >>> reg.params.round(6).tolist()
    [1.0, 2.0]
    """

    target: str
    params: pd.Series
    ols: Any
    exog_names: list[str]
    add_constant: bool
    sample: pd.PeriodIndex
    cov_type: str = "nonrobust"

    @property
    def n_obs(self) -> int:
        """Number of observations used in the regression."""
        return len(self.sample)

    @property
    def rsquared(self) -> float:
        """Coefficient of determination."""
        return float(self.ols.rsquared)

    @property
    def sigma(self) -> float:
        r"""Residual standard error :math:`\sqrt{e'e/(n-k)}`."""
        return float(np.sqrt(self.ols.scale))

    @property
    def bse(self) -> pd.Series:
        """Standard errors of the coefficients."""
        return pd.Series(np.asarray(self.ols.bse, dtype=float), index=self.params.index)

    @property
    def residuals(self) -> pd.Series:
        """In-sample residuals indexed by :attr:`sample`."""
        return pd.Series(np.asarray(self.ols.resid, dtype=float), index=self.sample, name="resid")

    @property
    def fitted_values(self) -> pd.Series:
        """In-sample fitted values indexed by :attr:`sample`."""
        return pd.Series(
            np.asarray(self.ols.fittedvalues, dtype=float), index=self.sample, name=self.target
        )

    def predict(self, regressors: pd.DataFrame) -> pd.Series:
        """Predict the target from a regressor matrix.

        Parameters
        ----------
        regressors : pandas.DataFrame
            Must contain the columns :attr:`exog_names`. Rows with any missing
            regressor get NaN.

        Returns
        -------
        pandas.Series
            Predictions on the index of ``regressors``.

        Raises
        ------
        NowcastDataError
            If regressors are missing.

        Examples
        --------
        >>> import numpy as np, pandas as pd
        >>> idx = pd.period_range("2000Q1", periods=20, freq="Q")
        >>> x = pd.DataFrame({"a": np.arange(20.0)}, index=idx)
        >>> reg = fit_bridge_regression(pd.Series(3.0 * x["a"] + 1, name="y"), x)
        >>> round(float(reg.predict(pd.DataFrame({"a": [10.0]})).iloc[0]), 8)
        31.0
        """
        missing = [c for c in self.exog_names if c not in regressors.columns]
        if missing:
            raise NowcastDataError(f"Regressors {missing} are missing.")
        x = regressors.loc[:, self.exog_names].to_numpy(dtype=float)
        beta = self.params.to_numpy(dtype=float)
        if self.add_constant:
            out = beta[0] + x @ beta[1:]
        else:
            out = x @ beta
        return pd.Series(out, index=regressors.index, name=self.target)

    def summary(self) -> str:
        """Text summary of the regression (statsmodels table).

        Returns
        -------
        str
            Summary table.

        Examples
        --------
        >>> import numpy as np, pandas as pd
        >>> idx = pd.period_range("2000Q1", periods=20, freq="Q")
        >>> rng = np.random.default_rng(0)
        >>> x = pd.DataFrame({"a": rng.standard_normal(20)}, index=idx)
        >>> reg = fit_bridge_regression(pd.Series(x["a"] + rng.standard_normal(20), name="y"), x)
        >>> "OLS Regression Results" in reg.summary()
        True
        """
        return str(self.ols.summary())

    def __repr__(self) -> str:
        return (
            f"BridgeRegression(target={self.target!r}, regressors={self.exog_names}, "
            f"n_obs={self.n_obs}, rsquared={self.rsquared:.4f})"
        )


def fit_bridge_regression(
    y: pd.Series,
    regressors: pd.DataFrame,
    *,
    add_constant: bool = True,
    cov_type: str = "nonrobust",
    cov_kwds: dict[str, Any] | None = None,
) -> BridgeRegression:
    """Estimate a bridge regression by OLS on the periods where all data are observed.

    Parameters
    ----------
    y : pandas.Series
        Target on its (native) index; NaN = not observed.
    regressors : pandas.DataFrame
        Regressors on an index comparable with ``y``.
    add_constant : bool, default True
        Include an intercept (named ``"const"``).
    cov_type : str, default "nonrobust"
        statsmodels covariance type (``"nonrobust"``, ``"HC0"``..``"HC3"``, ``"HAC"``).
    cov_kwds : dict, optional
        Extra options of the covariance estimator (e.g. ``{"maxlags": 4}`` for HAC).

    Returns
    -------
    BridgeRegression
        Fitted regression.

    Raises
    ------
    NowcastDataError
        If there are fewer usable observations than coefficients + 1, or the
        regressor matrix is rank deficient.
    ValueError
        If ``cov_type`` is unknown.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> idx = pd.period_range("2000Q1", periods=30, freq="Q")
    >>> x = pd.DataFrame({"a": np.cos(np.arange(30.0))}, index=idx)
    >>> y = pd.Series(0.5 - x["a"], name="y")
    >>> y.iloc[-2:] = np.nan
    >>> fit_bridge_regression(y, x).n_obs
    28
    """
    if cov_type not in _COV_TYPES:
        raise ValueError(f"cov_type must be one of {_COV_TYPES}, got {cov_type!r}.")
    name = str(y.name) if y.name is not None else "y"
    exog_names = [str(c) for c in regressors.columns]
    frame = pd.concat([y.rename("__y__"), regressors], axis=1, join="inner")
    frame = frame.dropna()
    n_params = len(exog_names) + int(add_constant)
    if n_params == 0:
        raise NowcastDataError("The bridge regression needs at least one regressor or a constant.")
    if len(frame) < n_params + 1:
        raise NowcastDataError(
            f"Only {len(frame)} periods with the target and all regressors observed; "
            f"the bridge regression needs at least {n_params + 1}."
        )
    endog = frame["__y__"].to_numpy(dtype=float)
    exog = frame.loc[:, exog_names].to_numpy(dtype=float)
    if add_constant:
        exog = np.column_stack([np.ones(len(frame)), exog])
    if np.linalg.matrix_rank(exog) < exog.shape[1]:
        raise NowcastDataError(
            "The bridge regressors are collinear (rank-deficient design matrix)."
        )
    names = (["const"] if add_constant else []) + exog_names
    # lazy: statsmodels takes ~0.7 s to import and is only needed here
    from statsmodels.regression.linear_model import OLS

    model = OLS(endog, exog)
    ols = model.fit(cov_type=cov_type, cov_kwds=cov_kwds or {})
    index = frame.index
    if not isinstance(index, pd.PeriodIndex):
        index = pd.PeriodIndex(index)
    params = pd.Series(np.asarray(ols.params, dtype=float), index=names, name=name)
    return BridgeRegression(
        target=name,
        params=params,
        ols=ols,
        exog_names=exog_names,
        add_constant=add_constant,
        sample=index,
        cov_type=cov_type,
    )


# ---------------------------------------------------------------------- helpers
def resolve_aggregation_weights(
    aggregation: str | AggregationType | Sequence[float] | np.ndarray | None,
    high: Frequency,
    low: Frequency,
    *,
    default: AggregationType = AggregationType.GROWTH_RATE,
    normalize: str = "ratio",
) -> np.ndarray:
    r"""Aggregation weights from a name, an :class:`AggregationType` or explicit values.

    Parameters
    ----------
    aggregation : str, AggregationType, sequence of float or None
        ``"flow"``, ``"average"``, ``"stock"``, ``"mariano_murasawa"`` (aliases of
        :meth:`AggregationType.from_value`), explicit weights (most recent period
        first) or ``None`` for ``default``.
    high, low : Frequency
        High (regressor) and low (target) frequencies.
    default : AggregationType, default GROWTH_RATE
        Used when ``aggregation`` is None.
    normalize : {"ratio", "sum", "none"}, default "ratio"
        Normalisation of named weights: Mariano-Murasawa weights divided by the
        frequency ratio (``"ratio"``, :math:`\tfrac13(1,2,3,2,1)`), all weights
        divided by their sum (``"sum"``, :math:`\tfrac19(1,2,3,2,1)`), or raw.
        Explicit weights are never rescaled.

    Returns
    -------
    numpy.ndarray
        Weights, most recent period first. ``[1.0]`` when ``high == low``.

    Raises
    ------
    ValueError
        If the weights are invalid or the frequency ratio is not fixed.

    Examples
    --------
    >>> from nowcastbox.core.frequency import Frequency
    >>> from nowcastbox.models.bridge import resolve_aggregation_weights
    >>> M, Q = Frequency.MONTHLY, Frequency.QUARTERLY
    >>> resolve_aggregation_weights(None, M, Q, normalize="sum").round(4).tolist()
    [0.1111, 0.2222, 0.3333, 0.2222, 0.1111]
    >>> resolve_aggregation_weights("average", M, Q).round(4).tolist()
    [0.3333, 0.3333, 0.3333]
    """
    if normalize not in ("ratio", "sum", "none"):
        raise ValueError(f"normalize must be 'ratio', 'sum' or 'none', got {normalize!r}.")
    if high == low:
        return np.ones(1)
    ratio = aggregation_ratio(high, low)
    if aggregation is None or isinstance(aggregation, str | AggregationType):
        kind = default if aggregation is None else AggregationType.from_value(aggregation)
        w = kind.weights(ratio, normalize=normalize == "ratio")
        if normalize == "sum":
            w = w / w.sum()
        return np.asarray(w, dtype=float)
    w = np.asarray(aggregation, dtype=float).ravel()
    if w.size == 0 or not bool(np.isfinite(w).all()) or not w.any():
        raise ValueError("Explicit aggregation weights must be finite and not all zero.")
    return w


def aggregate_to_target(
    native: pd.Series, weights: np.ndarray, target_periods: pd.PeriodIndex
) -> pd.Series:
    r"""Aggregate a native high-frequency series to target periods.

    Computes :math:`\bar x_t = \sum_i w_i x_{\tau(t)-i}` with :math:`\tau(t)` the last
    high-frequency period of target period :math:`t`.

    Parameters
    ----------
    native : pandas.Series
        Series on a contiguous PeriodIndex at its native (higher or equal) frequency.
    weights : numpy.ndarray
        Weights, most recent period first.
    target_periods : pandas.PeriodIndex
        Periods of the target frequency.

    Returns
    -------
    pandas.Series
        Aggregates indexed by ``target_periods`` (NaN where data are incomplete or
        the window falls outside ``native``).

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from nowcastbox.models.bridge import aggregate_to_target
    >>> m = pd.Series(np.arange(1.0, 7.0), index=pd.period_range("2020-01", periods=6, freq="M"))
    >>> q = pd.period_range("2020Q1", periods=2, freq="Q")
    >>> aggregate_to_target(m, np.full(3, 1 / 3), q).round(10).tolist()
    [2.0, 5.0]
    """
    index = native.index
    if not isinstance(index, pd.PeriodIndex):
        raise NowcastDataError("native must be indexed by a PeriodIndex.")
    filtered = rolling_aggregate(native.to_numpy(dtype=float), weights)
    ends = target_periods.asfreq(index.freqstr, how="E")
    pos = index.get_indexer(ends)
    out = np.full(len(target_periods), np.nan)
    ok = pos >= 0
    out[ok] = filtered[pos[ok]]
    return pd.Series(out, index=target_periods, name=native.name)


def ar_extend(values: np.ndarray, n_ahead: int, ar_lags: int = 1) -> np.ndarray:
    r"""Extend a series with iterated AR(:math:`p`) forecasts.

    The AR model :math:`x_t = c + \sum_{j=1}^p \rho_j x_{t-j} + u_t` is estimated by
    OLS on all periods where :math:`x_t` and its ``p`` lags are observed; forecasts
    start after the last observation.

    Parameters
    ----------
    values : numpy.ndarray
        1-D series (NaN = missing). Values after the last observation are ignored.
    n_ahead : int
        Number of forecasts appended after the last observation.
    ar_lags : int, default 1
        AR order :math:`p` (0 = forecast with the sample mean).

    Returns
    -------
    numpy.ndarray
        Array of length ``last_observed + 1 + n_ahead``: the observed history followed
        by the forecasts.

    Raises
    ------
    NowcastDataError
        If the series has too few complete observations for the AR model.

    Examples
    --------
    >>> import numpy as np
    >>> from nowcastbox.models.bridge import ar_extend
    >>> ar_extend(np.array([1.0, 2.0, 3.0, 4.0, np.nan]), 2, ar_lags=1).tolist()
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    """
    x = np.asarray(values, dtype=float).ravel()
    valid = np.flatnonzero(np.isfinite(x))
    if valid.size == 0:
        raise NowcastDataError("Cannot extend a series without observations.")
    x = x[: valid[-1] + 1]
    out = np.concatenate([x, np.full(max(int(n_ahead), 0), np.nan)])
    if n_ahead <= 0:
        return out
    p = int(ar_lags)
    if p == 0:
        out[x.size :] = np.nanmean(x)
        return out
    rows = np.column_stack([x[p - j : x.size - j] for j in range(0, p + 1)])
    rows = rows[np.isfinite(rows).all(axis=1)]
    if rows.shape[0] < p + 2:
        raise NowcastDataError(
            f"Too few complete observations ({rows.shape[0]}) to estimate an AR({p}) "
            "for the ragged-edge extension."
        )
    design = np.column_stack([np.ones(rows.shape[0]), rows[:, 1:]])
    coef, *_ = np.linalg.lstsq(design, rows[:, 0], rcond=None)
    for t in range(x.size, out.size):
        lags = out[t - p : t][::-1]
        if not bool(np.isfinite(lags).all()):
            lags = np.where(np.isfinite(lags), lags, np.nanmean(x))
        out[t] = coef[0] + float(lags @ coef[1:])
    return out


def _target_periods(
    data: MixedFrequencyData, target_freq: Frequency, horizon: int
) -> pd.PeriodIndex:
    """Target periods spanned by the panel plus ``horizon`` extra target periods."""
    pf = target_freq.pandas_freq
    first = data.start.asfreq(pf)
    last = data.end.asfreq(pf) + horizon
    return pd.period_range(first, last, freq=pf)


def _add_lags(frame: pd.DataFrame, lags: int) -> pd.DataFrame:
    """Append lags ``1..lags`` of every column (named ``<col>_lag<j>``)."""
    if lags <= 0:
        return frame
    parts = [frame]
    for j in range(1, lags + 1):
        parts.append(frame.shift(j).add_suffix(f"_lag{j}"))
    return pd.concat(parts, axis=1)


def _recursive_predict(
    regression: BridgeRegression,
    regressors: pd.DataFrame,
    observed: pd.Series,
    target_lags: int,
) -> pd.Series:
    """Predict with lagged targets, replacing unobserved lags by earlier predictions."""
    if target_lags == 0:
        return regression.predict(regressors)
    name = regression.target
    exog = regressors.copy()
    history = observed.astype(float).copy()
    preds = pd.Series(np.nan, index=regressors.index, name=name)
    for i, period in enumerate(regressors.index):
        for j in range(1, target_lags + 1):
            exog.iloc[i, exog.columns.get_loc(f"{name}_lag{j}")] = (
                history.iloc[i - j] if i - j >= 0 else np.nan
            )
        pred = float(regression.predict(exog.iloc[[i]]).iloc[0])
        preds.loc[period] = pred
        if not np.isfinite(history.iloc[i]):
            history.iloc[i] = pred
    return preds


# ---------------------------------------------------------------------- results
@dataclass(frozen=True, kw_only=True, eq=False, repr=False)
class BridgeResults(NowcastResults):
    """Results of :class:`BridgeEquation`.

    Parameters
    ----------
    bridge : BridgeRegression, optional
        Fitted OLS regression (``bridge.summary()``, ``bridge.ols``).
    regressors : pandas.DataFrame, optional
        Target-frequency regressor matrix (aggregated, ragged edge filled, lags
        included) used for estimation and prediction.

    Notes
    -----
    Every field of :class:`~nowcastbox.core.results.NowcastResults` is also accepted (keyword-only).

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from nowcastbox.models import BridgeEquation
    >>> rng = np.random.default_rng(0)
    >>> idx = pd.period_range("2000-01", periods=60, freq="M")
    >>> x = rng.standard_normal(60)
    >>> y = pd.Series(x, index=idx).rolling(3).mean().where(idx.month % 3 == 0)
    >>> df = pd.DataFrame({"x": x, "y": y}, index=idx)
    >>> res = BridgeEquation(aggregation="average").fit(df, "y", frequency={"x": "M", "y": "Q"})
    >>> round(float(res.bridge.params["x"]), 6)
    1.0
    """

    bridge: BridgeRegression | None = None
    regressors: pd.DataFrame | None = field(default=None)

    @property
    def coefficients(self) -> pd.Series:
        """Bridge coefficients (``bridge.params``).

        Raises
        ------
        ValueError
            If the results hold no bridge regression.
        """
        if self.bridge is None:
            raise ValueError("These results hold no bridge regression.")
        return self.bridge.params.copy()

    def _summary_sections(self) -> list[tuple[str, list[str]]]:
        sections = super()._summary_sections()
        if self.bridge is not None:
            lines = [
                f"  {'Observations':<22}{self.bridge.n_obs}",
                f"  {'R-squared':<22}{self.bridge.rsquared:.4f}",
                f"  {'Residual std':<22}{self.bridge.sigma:.4f}",
            ]
            lines += [f"  {k:<22}{v:.4f}" for k, v in self.bridge.params.items()]
            sections.append(("Bridge equation", lines))
        return sections


# ---------------------------------------------------------------------- estimator
class BridgeEquation(BaseNowcaster):
    r"""Bridge-equation nowcaster.

    Every predictor is (i) extended past its last observation with iterated
    AR(``ar_lags``) forecasts up to the end of the forecast horizon, (ii) aggregated
    to the target frequency with the chosen weights and (iii) used, with optional
    lags, in an OLS regression of the target (optionally with its own lags). Nowcasts
    and forecasts are the regression predictions for the periods where the target is
    not observed.

    Parameters
    ----------
    aggregation : str, AggregationType or sequence of float, default "average"
        Aggregation of higher-frequency predictors to the target frequency
        (``"average"``, ``"flow"``, ``"stock"``, ``"mariano_murasawa"`` - normalised
        by the frequency ratio - or explicit weights, most recent period first).
    regressor_lags : int, default 0
        Lags (in target periods) of each aggregated predictor.
    target_lags : int, default 0
        Autoregressive lags of the target; unobserved lags are replaced by earlier
        predictions (iterated forecasts).
    add_constant : bool, default True
        Include an intercept.
    fill_method : {"ar", "none"}, default "ar"
        Ragged-edge treatment of predictors: iterated AR forecasts or none (periods
        with incomplete predictors then get no estimate).
    ar_lags : int, default 1
        AR order of the ragged-edge extension (0 = sample mean).
    horizon : int, default 1
        Number of target periods to forecast after the target period containing the
        last base period of the panel.
    cov_type : str, default "nonrobust"
        Covariance estimator of the OLS coefficients.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from nowcastbox.models import BridgeEquation
    >>> rng = np.random.default_rng(1)
    >>> idx = pd.period_range("2000-01", periods=63, freq="M")
    >>> x = rng.standard_normal(63)
    >>> y = 0.5 + pd.Series(x, index=idx).rolling(3).mean()
    >>> y = y.where(idx.month % 3 == 0)
    >>> y.iloc[-1] = np.nan
    >>> df = pd.DataFrame({"x": x, "y": y}, index=idx)
    >>> res = BridgeEquation(horizon=0).fit(df, "y ~ x", frequency={"x": "M", "y": "Q"})
    >>> str(res.nowcast.index[-1])
    '2005Q1'
    >>> bool(np.isclose(res.get_nowcast(), 0.5 + x[-3:].mean()))
    True
    """

    def __init__(
        self,
        aggregation: str | AggregationType | Sequence[float] = "average",
        regressor_lags: int = 0,
        target_lags: int = 0,
        add_constant: bool = True,
        fill_method: str = "ar",
        ar_lags: int = 1,
        horizon: int = 1,
        cov_type: str = "nonrobust",
    ) -> None:
        self.aggregation = aggregation
        self.regressor_lags = regressor_lags
        self.target_lags = target_lags
        self.add_constant = add_constant
        self.fill_method = fill_method
        self.ar_lags = ar_lags
        self.horizon = horizon
        self.cov_type = cov_type

    def _validate_params(self) -> None:
        for name in ("regressor_lags", "target_lags", "ar_lags", "horizon"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int | np.integer) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer, got {value!r}.")
        if self.fill_method not in _FILL_METHODS:
            raise ValueError(
                f"fill_method must be one of {_FILL_METHODS}, got {self.fill_method!r}."
            )
        if self.cov_type not in _COV_TYPES:
            raise ValueError(f"cov_type must be one of {_COV_TYPES}, got {self.cov_type!r}.")
        if not isinstance(self.add_constant, bool):
            raise ValueError(f"add_constant must be a bool, got {self.add_constant!r}.")
        if not (
            self.aggregation is None
            or isinstance(self.aggregation, str | AggregationType)
            or np.ndim(self.aggregation) == 1
        ):
            raise ValueError(f"Invalid aggregation {self.aggregation!r}.")

    def _regressor_frame(
        self, data: MixedFrequencyData, target: str, periods: pd.PeriodIndex
    ) -> pd.DataFrame:
        target_freq = data.metadata[target].frequency
        columns: dict[str, pd.Series] = {}
        for col in data.columns:
            if col == target:
                continue
            freq = data.metadata[col].frequency
            if freq.is_lower_than(target_freq):
                raise NowcastDataError(
                    f"Predictor {col!r} ({freq.label}) has a lower frequency than the "
                    f"target ({target_freq.label})."
                )
            native = data.to_native(col)
            end = periods[-1].asfreq(freq.pandas_freq, how="E")
            grid = pd.period_range(native.index[0], end, freq=freq.pandas_freq)
            values = native.reindex(grid).to_numpy(dtype=float)
            if self.fill_method == "ar":
                last = (
                    int(np.flatnonzero(np.isfinite(values))[-1])
                    if np.isfinite(values).any()
                    else -1
                )
                if last < 0:
                    raise NowcastDataError(f"Predictor {col!r} has no observations.")
                n_ahead = len(grid) - last - 1
                values = ar_extend(values, n_ahead, int(self.ar_lags))
            series = pd.Series(values, index=grid, name=col)
            weights = resolve_aggregation_weights(
                self.aggregation, freq, target_freq, normalize="ratio"
            )
            columns[col] = aggregate_to_target(series, weights, periods)
        frame = pd.DataFrame(columns, index=periods)
        return _add_lags(frame, int(self.regressor_lags))

    def fit(
        self,
        data: MixedFrequencyData | pd.DataFrame,
        target: str,
        *,
        frequency: FrequencySpec | None = None,
        **fit_kwargs: Any,
    ) -> BridgeResults:
        """Estimate the bridge equation and nowcast ``target``.

        Parameters
        ----------
        data : MixedFrequencyData or pandas.DataFrame
            Panel on the base grid.
        target : str
            Target name or formula (``"gdp ~ ip + pmi"``).
        frequency : frequency specification, optional
            Per-series frequencies for DataFrame input.
        **fit_kwargs
            Not used (an error is raised for unknown options).

        Returns
        -------
        BridgeResults
            Estimation results (also stored in :attr:`results_`).

        Raises
        ------
        NowcastDataError
            If the data cannot support the regression.
        ValueError
            If a hyper-parameter is invalid.

        Examples
        --------
        >>> import numpy as np, pandas as pd
        >>> idx = pd.period_range("2000-01", periods=36, freq="M")
        >>> x = np.sin(np.arange(36.0))
        >>> y = pd.Series(x, index=idx).rolling(3).mean().where(idx.month % 3 == 0)
        >>> df = pd.DataFrame({"x": x, "y": y}, index=idx)
        >>> res = BridgeEquation().fit(df, "y", frequency={"x": "M", "y": "Q"})
        >>> round(float(res.coefficients["x"]), 6)
        1.0
        """
        results = super().fit(data, target, frequency=frequency, **fit_kwargs)
        assert isinstance(results, BridgeResults)  # noqa: S101
        return results

    def _fit(self, data: MixedFrequencyData, target: str, **fit_kwargs: Any) -> BridgeResults:
        if fit_kwargs:
            raise TypeError(f"Unexpected fit options {sorted(fit_kwargs)}.")
        started = time.perf_counter()
        target_freq = data.metadata[target].frequency
        if data.n_series < 2 and self.target_lags == 0:
            raise NowcastDataError(
                "BridgeEquation needs at least one predictor (or target_lags > 0)."
            )
        periods = _target_periods(data, target_freq, int(self.horizon))
        observed = data.to_native(target).reindex(periods)
        if int(observed.notna().sum()) == 0:
            raise NowcastDataError(f"The target {target!r} has no observations.")
        regressors = self._regressor_frame(data, target, periods)
        for j in range(1, int(self.target_lags) + 1):
            regressors[f"{target}_lag{j}"] = observed.shift(j)
        regression = fit_bridge_regression(
            observed.rename(target),
            regressors,
            add_constant=self.add_constant,
            cov_type=self.cov_type,
        )
        estimate = _recursive_predict(regression, regressors, observed, int(self.target_lags))
        frame = build_nowcast_frame(observed, estimate)
        logger.debug("BridgeEquation fitted on %d periods", regression.n_obs)
        return BridgeResults(
            target=target,
            nowcast=frame,
            model_name="BridgeEquation",
            model_params=self.get_params(deep=False),
            params={"coefficients": regression.params.copy(), "sigma": regression.sigma},
            data=data,
            info={"fit_time": time.perf_counter() - started, "n_obs": regression.n_obs},
            bridge=regression,
            regressors=regressors,
        )
