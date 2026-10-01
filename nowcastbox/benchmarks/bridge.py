r"""Bridge-equation benchmark.

Bridge equations (Baffigi, Golinelli & Parigi, 2004; Diron, 2008) regress the
low-frequency target on its higher-frequency indicators aggregated to the target
frequency,

.. math::

    y_t = \alpha + \sum_k \sum_{l=0}^{L} \beta_{k,l}\, \bar x^{(k)}_{t-l}
          + \sum_{j=1}^{J} \phi_j\, y_{t-j} + e_t,

where indicators not yet released for the current quarter are first completed by
iterated AR(:math:`p`) forecasts of the monthly series (the standard bridge-model
treatment of the ragged edge). The estimation is delegated to
:class:`nowcastbox.models.BridgeEquation`; this class only adapts it to the
:class:`~nowcastbox.core.base.BaseBenchmark` contract (``fit`` / ``predict``).

References
----------
Baffigi, A., Golinelli, R. & Parigi, G. (2004). Bridge models to forecast the euro area
GDP. *International Journal of Forecasting*, 20(3), 447-460.

Diron, M. (2008). Short-term forecasts of euro area real GDP growth: an assessment of
real-time performance based on vintage data. *Journal of Forecasting*, 27(5), 371-390.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from nowcastbox.benchmarks._utils import check_int, ordinals, resolve_predictors
from nowcastbox.core.base import BaseBenchmark
from nowcastbox.core.data import MixedFrequencyData
from nowcastbox.core.frequency import AggregationType
from nowcastbox.core.results import NowcastResults
from nowcastbox.models.bridge import BridgeEquation

__all__ = ["BridgeBenchmark"]


class BridgeBenchmark(BaseBenchmark):
    """Bridge equation with AR-completed indicators as a benchmark.

    Parameters
    ----------
    predictors : sequence of str, optional
        Indicators (default: every series except the target).
    aggregation : str, AggregationType or sequence of float, default "average"
        Aggregation of the indicators to the target frequency (see
        :class:`~nowcastbox.models.BridgeEquation`).
    regressor_lags : int, default 0
        Lags (target periods) of each aggregated indicator.
    target_lags : int, default 0
        Autoregressive lags of the target (iterated when unobserved).
    ar_lags : int, default 1
        Order of the AR models completing the indicators.
    add_constant : bool, default True
        Include an intercept.

    Attributes
    ----------
    results_ : nowcastbox.models.BridgeResults
        Results of the underlying bridge equation (longest horizon requested so far).
    coef_ : pandas.Series
        Bridge coefficients.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from nowcastbox.benchmarks import BridgeBenchmark
    >>> rng = np.random.default_rng(1)
    >>> idx = pd.period_range("2000-01", periods=60, freq="M")
    >>> x = rng.standard_normal(60)
    >>> y = (1.0 + 2.0 * pd.Series(x, index=idx).rolling(3).mean()).where(idx.month % 3 == 0)
    >>> frame = pd.DataFrame({"x": x, "y": y}, index=idx)
    >>> bench = BridgeBenchmark().fit(frame.iloc[:-2], "y", frequency={"x": "M", "y": "Q"})
    >>> np.round(bench.coef_.to_numpy(), 6).tolist()
    [1.0, 2.0]
    >>> bool(np.isfinite(bench.predict(["2004Q4", "2005Q1"])).all())
    True
    """

    def __init__(
        self,
        predictors: Sequence[str] | None = None,
        aggregation: str | AggregationType | Sequence[float] = "average",
        regressor_lags: int = 0,
        target_lags: int = 0,
        ar_lags: int = 1,
        add_constant: bool = True,
    ) -> None:
        self.predictors = predictors
        self.aggregation = aggregation
        self.regressor_lags = regressor_lags
        self.target_lags = target_lags
        self.ar_lags = ar_lags
        self.add_constant = add_constant

    _panel: MixedFrequencyData | None = None
    _results: NowcastResults | None = None
    _horizon: int = -1

    def _equation(self, horizon: int) -> BridgeEquation:
        return BridgeEquation(
            aggregation=self.aggregation,
            regressor_lags=self.regressor_lags,
            target_lags=self.target_lags,
            add_constant=self.add_constant,
            fill_method="ar",
            ar_lags=self.ar_lags,
            horizon=horizon,
        )

    def _fit(self, data: MixedFrequencyData, target: str) -> None:
        check_int("ar_lags", self.ar_lags, 0)
        names = resolve_predictors(data, target, self.predictors)
        self._panel = data.select([c for c in data.columns if c in {target, *names}])
        self._run(target, 0)

    def _run(self, target: str, horizon: int) -> NowcastResults:
        assert self._panel is not None  # noqa: S101
        if horizon > self._horizon or self._results is None:
            self._results = self._equation(horizon).fit(self._panel, target)
            self._horizon = horizon
        return self._results

    def _predict(self, periods: pd.PeriodIndex) -> pd.Series:
        assert self._panel is not None  # noqa: S101
        freq = periods.freqstr
        last = self._panel.end.asfreq(freq)
        horizon = max(int(np.max((ordinals(periods) - last.ordinal), initial=0)), 0)
        results = self._run(self.target_, horizon)
        estimate = results.estimate.reindex(periods)
        return pd.Series(estimate.to_numpy(dtype=float), index=periods)

    @property
    def results_(self) -> NowcastResults:
        """Results of the underlying :class:`~nowcastbox.models.BridgeEquation`."""
        self._check_fitted()
        assert self._results is not None  # noqa: S101
        return self._results

    @property
    def coef_(self) -> pd.Series:
        """Bridge coefficients."""
        return self.results_.params["coefficients"].copy()

    def _reset(self) -> None:
        super()._reset()
        self._panel = None
        self._results = None
        self._horizon = -1
