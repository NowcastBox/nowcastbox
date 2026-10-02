r"""Combination of bridge equations as a benchmark.

Bańbura, Belousova, Bodnár & Tóth (2023) nowcast with the average of all small bridge
equations (1-2 monthly and 0-1 quarterly indicators), each completed at the ragged edge
by extrapolated indicators. The estimation is delegated to
:class:`nowcastbox.models.BridgeCombination`; this class only adapts it to the
:class:`~nowcastbox.core.base.BaseBenchmark` contract (``fit`` / ``predict``), so that it
can enter :class:`~nowcastbox.evaluation.PseudoRealTimeBacktest` next to the other
benchmarks. Each vintage is fitted on its own information set only, so the
equation MSEs, the inverse-MSE weights and the trimming never look ahead.

References
----------
Bańbura, M., Belousova, I., Bodnár, K. & Tóth, M. B. (2023). Nowcasting employment in
the euro area. ECB Working Paper No. 2815.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from nowcastbox.benchmarks._utils import check_int, ordinals, resolve_predictors
from nowcastbox.core.base import BaseBenchmark
from nowcastbox.core.data import MixedFrequencyData
from nowcastbox.core.frequency import AggregationType
from nowcastbox.models.bridge_combination import BridgeCombination, BridgeCombinationResults
from nowcastbox.models.extrapolation import Extrapolator

__all__ = ["BridgeCombinationBenchmark"]


class BridgeCombinationBenchmark(BaseBenchmark):
    """Combination of all small bridge equations as a benchmark.

    Parameters
    ----------
    predictors : sequence of str, optional
        Candidate indicators (default: every series except the target).
    max_monthly, max_quarterly, min_monthly : int, default 2, 1, 1
        Size limits of the indicator subsets (see
        :class:`~nowcastbox.models.BridgeCombination`).
    target_lags, regressor_lags : int, default 0
        Target and indicator lags of every equation.
    aggregation : str, AggregationType or sequence of float, default "average"
        Aggregation of higher-frequency indicators.
    add_constant : bool, default True
        Include an intercept.
    combine : {"mean", "median", "inverse_mse"}, default "mean"
        Combination of the equation predictions.
    mse : {"in_sample", "out_of_sample"}, default "in_sample"
        Accuracy measure of the inverse-MSE weights and of the trimming.
    mse_window : int, optional
        Number of most recent target periods of the MSE.
    discount : float, default 1.0
        Discount factor of past squared errors.
    min_train : int, default 8
        Minimum estimation sample of the pseudo out-of-sample errors.
    trim : float, default 0.0
        Share of the worst equations (by MSE) discarded.
    extrapolation : str, callable or None, default "ar"
        Indicator extrapolation (:mod:`nowcastbox.models.extrapolation`; e.g. ``"ar"``
        or ``"bvar"``). A named extrapolator is built once per ``fit`` and reused when
        ``predict`` refits for a longer horizon, so the ``"bvar"`` VAR is estimated
        once per vintage.
    ar_lags : int, default 1
        AR order of the ``"ar"`` extrapolation.
    extrapolation_options : mapping, optional
        Extra options of a named extrapolator.

    Attributes
    ----------
    results_ : nowcastbox.models.BridgeCombinationResults
        Results of the underlying model (longest horizon requested so far).
    weights_ : pandas.Series
        Combination weights by equation id.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from nowcastbox.benchmarks import BridgeCombinationBenchmark
    >>> rng = np.random.default_rng(1)
    >>> idx = pd.period_range("2000-01", periods=90, freq="M")
    >>> x = rng.standard_normal((90, 2))
    >>> y = pd.Series(x[:, 0], index=idx).rolling(3).mean().where(idx.month % 3 == 0)
    >>> frame = pd.DataFrame({"a": x[:, 0], "b": x[:, 1], "y": y}, index=idx)
    >>> freq = {"a": "M", "b": "M", "y": "Q"}
    >>> bench = BridgeCombinationBenchmark().fit(frame.iloc[:-2], "y", frequency=freq)
    >>> len(bench.weights_)
    3
    >>> bool(np.isfinite(bench.predict(["2007Q2", "2007Q3"])).all())
    True
    """

    def __init__(
        self,
        predictors: Sequence[str] | None = None,
        max_monthly: int = 2,
        max_quarterly: int = 1,
        min_monthly: int = 1,
        target_lags: int = 0,
        regressor_lags: int = 0,
        aggregation: str | AggregationType | Sequence[float] = "average",
        add_constant: bool = True,
        combine: str = "mean",
        mse: str = "in_sample",
        mse_window: int | None = None,
        discount: float = 1.0,
        min_train: int = 8,
        trim: float = 0.0,
        extrapolation: str | Extrapolator | None = "ar",
        ar_lags: int = 1,
        extrapolation_options: Mapping[str, Any] | None = None,
    ) -> None:
        self.predictors = predictors
        self.max_monthly = max_monthly
        self.max_quarterly = max_quarterly
        self.min_monthly = min_monthly
        self.target_lags = target_lags
        self.regressor_lags = regressor_lags
        self.aggregation = aggregation
        self.add_constant = add_constant
        self.combine = combine
        self.mse = mse
        self.mse_window = mse_window
        self.discount = discount
        self.min_train = min_train
        self.trim = trim
        self.extrapolation = extrapolation
        self.ar_lags = ar_lags
        self.extrapolation_options = extrapolation_options

    _panel: MixedFrequencyData | None = None
    _results: BridgeCombinationResults | None = None
    _horizon: int = -1
    _extrapolator: Extrapolator | None = None

    def _model(self, horizon: int) -> BridgeCombination:
        params = self.get_params(deep=False)
        params.pop("predictors")
        model = BridgeCombination(**params, horizon=horizon)
        if self._extrapolator is not None:
            # one extrapolator instance per fit: refits for longer horizons reuse what
            # it caches (e.g. the estimated VAR of the "bvar" extrapolator)
            model.set_params(extrapolation=self._extrapolator, extrapolation_options=None)
        return model

    def _fit(self, data: MixedFrequencyData, target: str) -> None:
        check_int("ar_lags", self.ar_lags, 0)
        self._extrapolator = None
        built = self._model(0).extrapolator()
        self._extrapolator = built if isinstance(self.extrapolation, str) else None
        names = resolve_predictors(data, target, self.predictors)
        self._panel = data.select([c for c in data.columns if c in {target, *names}])
        self._run(target, 0)

    def _run(self, target: str, horizon: int) -> BridgeCombinationResults:
        assert self._panel is not None  # noqa: S101
        if horizon > self._horizon or self._results is None:
            self._results = self._model(horizon).fit(self._panel, target)
            self._horizon = horizon
        return self._results

    def _predict(self, periods: pd.PeriodIndex) -> pd.Series:
        assert self._panel is not None  # noqa: S101
        last = self._panel.end.asfreq(periods.freqstr)
        horizon = max(int(np.max(ordinals(periods) - last.ordinal, initial=0)), 0)
        estimate = self._run(self.target_, horizon).estimate.reindex(periods)
        return pd.Series(estimate.to_numpy(dtype=float), index=periods)

    @property
    def results_(self) -> BridgeCombinationResults:
        """Results of the underlying :class:`~nowcastbox.models.BridgeCombination`."""
        self._check_fitted()
        assert self._results is not None  # noqa: S101
        return self._results

    @property
    def weights_(self) -> pd.Series:
        """Combination weights by equation id."""
        return self.results_.weights

    def _reset(self) -> None:
        super()._reset()
        self._panel = None
        self._results = None
        self._horizon = -1
        self._extrapolator = None
