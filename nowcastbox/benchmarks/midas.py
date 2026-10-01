r"""MIDAS benchmarks: unrestricted (U-MIDAS, OLS) and restricted (exponential Almon / Beta, NLS).

Mixed-data sampling regressions (Ghysels, Santa-Clara & Valkanov, 2004; Ghysels,
Sinko & Valkanov, 2007) project the low-frequency target directly on high-frequency
lags of the indicators:

.. math::

    y_t = c + \sum_{i} \beta_i \sum_{j=0}^{K_i-1} w_{ij}(\theta_i)\,
          x^{(i)}_{\tau_i(t) - s_i - j}
          + \sum_{l=0}^{L-1} \phi_l\, y_{t-h-l} + e_t ,

where :math:`\tau_i(t)` is the last high-frequency period of target period :math:`t`
(e.g. the third month of a quarter). In **U-MIDAS** (Foroni, Marcellino &
Schumacher, 2015) every lag has its own coefficient (:math:`\beta_i w_{ij}` free,
OLS). In **MIDAS** the lag weights follow a two-parameter polynomial normalised to sum
one, estimated by non-linear least squares:

* exponential Almon: :math:`w_j \propto \exp(\theta_1 j + \theta_2 j^2)`,
  :math:`j = 1, \dots, K`;
* Beta: :math:`w_j \propto x_j^{\theta_1 - 1} (1 - x_j)^{\theta_2 - 1}` with
  :math:`x_j` equally spaced on :math:`[\varepsilon, 1 - \varepsilon]`.

The linear parameters are concentrated out (variable projection) and the profile sum
of squared residuals is minimised over :math:`\theta` with L-BFGS-B, started
from the best of a few candidate values of :math:`\theta`.

**Ragged edge.** With ``ragged_edge="realign"`` (default) the regressors of the
equation used to predict period :math:`p` are shifted by :math:`s_i`, the number of
high-frequency periods of :math:`p` not yet observed for indicator :math:`i`, and the
equation is re-estimated with that shift ("vertical realignment"; Marcellino &
Schumacher, 2010; Clements & Galvão, 2008). With
``ragged_edge="ar"`` the indicators are instead completed by iterated AR forecasts and
:math:`s_i = 0`. Target lags enter directly at horizon :math:`h` (number of target
periods between the last observed target value and :math:`p`), as in the direct
AR-MIDAS of Clements & Galvão (2008).

References
----------
Ghysels, E., Santa-Clara, P. & Valkanov, R. (2004). The MIDAS touch: Mixed data
sampling regression models. Working paper, UCLA/UNC.

Ghysels, E., Sinko, A. & Valkanov, R. (2007). MIDAS regressions: Further results and
new directions. *Econometric Reviews*, 26(1), 53-90.

Foroni, C., Marcellino, M. & Schumacher, C. (2015). Unrestricted mixed data sampling
(MIDAS): MIDAS regressions with unrestricted lag polynomials. *Journal of the Royal
Statistical Society A*, 178(1), 57-82.

Clements, M. P. & Galvão, A. B. (2008). Macroeconomic forecasting with mixed-frequency
data: Forecasting output growth in the United States. *Journal of Business & Economic
Statistics*, 26(4), 546-554.

Marcellino, M. & Schumacher, C. (2010). Factor MIDAS for nowcasting and forecasting with
ragged-edge data. *Oxford Bulletin of Economics and Statistics*, 72(4), 518-550.
"""

from __future__ import annotations

import warnings
from abc import abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from nowcastbox.benchmarks._utils import (
    check_int,
    check_ragged_edge,
    last_valid,
    lstsq,
    native_grid,
    ordinals,
    period_positions,
    resolve_predictors,
)
from nowcastbox.core.base import BaseBenchmark
from nowcastbox.core.data import MixedFrequencyData
from nowcastbox.core.exceptions import ConvergenceWarning, NowcastDataError
from nowcastbox.core.frequency import Frequency, aggregation_ratio
from nowcastbox.models.bridge import ar_extend

__all__ = ["MIDAS", "UMIDAS", "beta_weights", "exp_almon_weights"]

_POLYNOMIALS = ("exp_almon", "beta")
_BETA_EPS = 1e-6
_BOUNDS = {"exp_almon": ((-10.0, 10.0), (-2.0, 2.0)), "beta": ((0.1, 50.0), (0.1, 50.0))}
_STARTS = {
    "exp_almon": ((0.0, 0.0), (-0.5, 0.0), (0.5, -0.1), (0.0, -0.1)),
    "beta": ((1.0, 1.0), (1.0, 3.0), (1.0, 10.0), (2.0, 2.0)),
}

SpecKey = tuple[tuple[int, ...], int]


def _softmax(log_w: np.ndarray) -> np.ndarray:
    w = np.exp(log_w - log_w.max())
    return w / w.sum()


def exp_almon_weights(theta1: float, theta2: float, n_lags: int) -> np.ndarray:
    r"""Normalised exponential Almon lag weights.

    .. math:: w_j = \frac{\exp(\theta_1 j + \theta_2 j^2)}{\sum_{k=1}^{K}
              \exp(\theta_1 k + \theta_2 k^2)}, \qquad j = 1, \dots, K.

    Parameters
    ----------
    theta1, theta2 : float
        Shape parameters (``0, 0`` gives equal weights).
    n_lags : int
        Number of lags :math:`K` (most recent first).

    Returns
    -------
    numpy.ndarray
        Weights summing to one.

    Raises
    ------
    ValueError
        If ``n_lags < 1``.

    Examples
    --------
    >>> from nowcastbox.benchmarks import exp_almon_weights
    >>> exp_almon_weights(0.0, 0.0, 4).tolist()
    [0.25, 0.25, 0.25, 0.25]
    >>> bool((np.diff(exp_almon_weights(0.0, -0.2, 6)) < 0).all())
    True
    """
    k = check_int("n_lags", n_lags, 1)
    j = np.arange(1, k + 1, dtype=float)
    return _softmax(theta1 * j + theta2 * j**2)


def beta_weights(theta1: float, theta2: float, n_lags: int) -> np.ndarray:
    r"""Normalised Beta lag weights (Ghysels, Sinko & Valkanov, 2007).

    .. math:: w_j = \frac{f(x_j; \theta_1, \theta_2)}{\sum_k f(x_k; \theta_1, \theta_2)},
              \qquad f(x; a, b) = x^{a-1} (1 - x)^{b-1},

    with :math:`x_1 = \varepsilon, \dots, x_K = 1 - \varepsilon` equally spaced
    (:math:`\varepsilon = 10^{-6}`).

    Parameters
    ----------
    theta1, theta2 : float
        Positive shape parameters (``1, 1`` gives equal weights; ``1, b > 1`` declining
        weights).
    n_lags : int
        Number of lags :math:`K` (most recent first).

    Returns
    -------
    numpy.ndarray
        Weights summing to one.

    Raises
    ------
    ValueError
        If ``n_lags < 1`` or a shape parameter is not positive.

    Examples
    --------
    >>> from nowcastbox.benchmarks import beta_weights
    >>> beta_weights(1.0, 1.0, 3).round(6).tolist()
    [0.333333, 0.333333, 0.333333]
    >>> bool((np.diff(beta_weights(1.0, 5.0, 6)) < 0).all())
    True
    """
    k = check_int("n_lags", n_lags, 1)
    if not (theta1 > 0 and theta2 > 0):
        raise ValueError(f"Beta shape parameters must be positive, got {(theta1, theta2)}.")
    if k == 1:
        return np.ones(1)
    x = np.linspace(_BETA_EPS, 1.0 - _BETA_EPS, k)
    return _softmax((theta1 - 1.0) * np.log(x) + (theta2 - 1.0) * np.log1p(-x))


_WEIGHT_FUNCTIONS = {"exp_almon": exp_almon_weights, "beta": beta_weights}


@dataclass
class _Indicator:
    """High-frequency indicator on its native grid (up to its last observation)."""

    name: str
    values: np.ndarray
    start: pd.Period
    frequency: Frequency
    n_lags: int

    @property
    def last(self) -> int:
        return self.values.size - 1


class _MidasBase(BaseBenchmark):
    """Shared machinery of :class:`UMIDAS` and :class:`MIDAS`."""

    predictors: Sequence[str] | None
    n_lags: int | None
    target_lags: int
    ragged_edge: str
    ar_lags: int

    _default_n_lags_factor: int = 1
    _indicators: list[_Indicator] | None = None
    _extended: dict[str, np.ndarray] | None = None
    _y: np.ndarray | None = None
    _y_start: pd.Period | None = None
    _fits: dict[SpecKey, np.ndarray] | None = None
    _default_key: SpecKey | None = None

    # ------------------------------------------------------------------ setup
    def _validate(self) -> None:
        check_int("target_lags", self.target_lags, 0)
        check_int("ar_lags", self.ar_lags, 0)
        check_ragged_edge(self.ragged_edge)
        if self.n_lags is not None:
            check_int("n_lags", self.n_lags, 1)

    def _lags_for(self, frequency: Frequency, target_freq: Frequency) -> int:
        if self.n_lags is not None:
            return int(self.n_lags)
        return self._default_n_lags_factor * aggregation_ratio(frequency, target_freq)

    def _fit(self, data: MixedFrequencyData, target: str) -> None:
        self._validate()
        target_freq = data.metadata[target].frequency
        indicators = []
        for name in resolve_predictors(data, target, self.predictors):
            series = native_grid(data, name)
            values = series.to_numpy(dtype=float)
            freq = data.metadata[name].frequency
            index = series.index
            assert isinstance(index, pd.PeriodIndex)  # noqa: S101
            indicators.append(
                _Indicator(
                    name,
                    values[: last_valid(values) + 1],
                    index[0],
                    freq,
                    self._lags_for(freq, target_freq),
                )
            )
        y_series = native_grid(data, target)
        y = y_series.to_numpy(dtype=float)
        last = last_valid(y)
        if last < 0:
            raise NowcastDataError(f"The target {target!r} has no observations.")
        self._indicators = indicators
        self._extended = {}
        self._y = y[: last + 1]
        y_index = y_series.index
        assert isinstance(y_index, pd.PeriodIndex)  # noqa: S101
        self._y_start = y_index[0]
        self._fits = {}
        first_future = pd.PeriodIndex([y_index[0] + (last + 1)])
        self._default_key = self._keys(first_future)[0]
        self._fitted(self._default_key, target)

    def _reset(self) -> None:
        super()._reset()
        self._fits = None
        self._indicators = None

    # ------------------------------------------------------------------ design
    def _keys(self, periods: pd.PeriodIndex) -> list[SpecKey]:
        """Specification (shifts, direct horizon) used to predict each period."""
        assert self._indicators is not None  # noqa: S101
        assert self._y is not None  # noqa: S101
        assert self._y_start is not None  # noqa: S101
        n = len(periods)
        shifts = np.zeros((n, len(self._indicators)), dtype=np.int64)
        if self.ragged_edge == "realign":
            for i, ind in enumerate(self._indicators):
                pos = period_positions(ind.start, ind.frequency, periods)
                shifts[:, i] = np.maximum(pos - ind.last, 0)
        horizon = np.maximum(ordinals(periods) - self._y_start.ordinal - (self._y.size - 1), 1)
        if self.target_lags == 0:
            horizon = np.zeros(n, dtype=np.int64)
        return [(tuple(int(s) for s in shifts[r]), int(horizon[r])) for r in range(n)]

    def _indicator_values(self, ind: _Indicator, needed: int) -> np.ndarray:
        """Indicator values, AR-extended to ``needed`` periods in ``ragged_edge="ar"``."""
        if self.ragged_edge != "ar" or needed <= ind.values.size:
            return ind.values
        assert self._extended is not None  # noqa: S101
        cached = self._extended.get(ind.name)
        if cached is None or cached.size < needed:
            cached = ar_extend(ind.values, needed - ind.values.size, int(self.ar_lags))
            self._extended[ind.name] = cached
        return cached

    def _blocks(self, periods: pd.PeriodIndex, key: SpecKey) -> list[np.ndarray]:
        """High-frequency lag matrices ``(n, K_i)`` of every indicator."""
        assert self._indicators is not None  # noqa: S101
        blocks = []
        for ind, shift in zip(self._indicators, key[0], strict=True):
            base = period_positions(ind.start, ind.frequency, periods) - shift
            idx = base[:, None] - np.arange(ind.n_lags)[None, :]
            values = self._indicator_values(ind, int(idx.max(initial=0)) + 1)
            blocks.append(_take(values, idx))
        return blocks

    def _ar_block(self, periods: pd.PeriodIndex, key: SpecKey) -> np.ndarray:
        """Direct target lags ``y_{t-h-l}``, ``l = 0..L-1``."""
        assert self._y is not None  # noqa: S101
        assert self._y_start is not None  # noqa: S101
        pos = np.asarray(ordinals(periods) - self._y_start.ordinal, dtype=np.int64)
        idx = pos[:, None] - key[1] - np.arange(int(self.target_lags))[None, :]
        return _take(self._y, idx)

    def _sample(self, key: SpecKey) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
        """Estimation sample (complete rows) of one specification."""
        assert self._y is not None  # noqa: S101
        assert self._y_start is not None  # noqa: S101
        periods = pd.period_range(self._y_start, periods=self._y.size, freq=self._y_start.freq)
        blocks = self._blocks(periods, key)
        ar = self._ar_block(periods, key)
        ok = np.isfinite(self._y) & np.isfinite(ar).all(axis=1)
        for block in blocks:
            ok &= np.isfinite(block).all(axis=1)
        return [b[ok] for b in blocks], ar[ok], self._y[ok]

    def _fitted(self, key: SpecKey, target: str) -> np.ndarray:
        assert self._fits is not None  # noqa: S101
        if key not in self._fits:
            blocks, ar, y = self._sample(key)
            n_params = self._n_params(blocks, ar)
            if y.size < n_params + 1:
                raise NowcastDataError(
                    f"Too few complete observations ({y.size}) for a {type(self).__name__} "
                    f"of {target!r} with {n_params} parameters."
                )
            self._fits[key] = self._estimate(blocks, ar, y)
        return self._fits[key]

    def _predict(self, periods: pd.PeriodIndex) -> pd.Series:
        keys = self._keys(periods)
        out = np.full(len(periods), np.nan)
        for key in dict.fromkeys(keys):
            rows = np.flatnonzero([k == key for k in keys])
            sub = periods[rows]
            params = self._fitted(key, self.target_)
            out[rows] = self._combine(params, self._blocks(sub, key), self._ar_block(sub, key))
        return pd.Series(out, index=periods)

    # ------------------------------------------------------------------ hooks
    @abstractmethod
    def _n_params(self, blocks: list[np.ndarray], ar: np.ndarray) -> int:
        """Number of estimated parameters."""

    @abstractmethod
    def _estimate(self, blocks: list[np.ndarray], ar: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Estimate the parameter vector on complete rows."""

    @abstractmethod
    def _combine(self, params: np.ndarray, blocks: list[np.ndarray], ar: np.ndarray) -> np.ndarray:
        """Predictions (NaN rows propagate)."""

    # ------------------------------------------------------------------ attributes
    def _default_params(self) -> np.ndarray:
        self._check_fitted()
        assert self._fits is not None  # noqa: S101
        assert self._default_key is not None  # noqa: S101
        return self._fits[self._default_key]

    def _ar_names(self) -> list[str]:
        assert self._default_key is not None  # noqa: S101
        h = self._default_key[1]
        return [f"{self.target_}_L{h + l}" for l in range(int(self.target_lags))]

    @property
    def shifts_(self) -> pd.Series:
        """Regressor shifts :math:`s_i` of the nowcast of the first unobserved period."""
        self._check_fitted()
        assert self._default_key is not None  # noqa: S101
        assert self._indicators is not None  # noqa: S101
        names = [ind.name for ind in self._indicators]
        return pd.Series(self._default_key[0], index=names, name="shift", dtype=int)


def _take(values: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """``values[idx]`` with NaN outside ``0..len-1``."""
    valid = (idx >= 0) & (idx < values.size)
    return np.where(valid, values[np.clip(idx, 0, max(values.size - 1, 0))], np.nan)


def _linear_design(blocks: list[np.ndarray], ar: np.ndarray) -> np.ndarray:
    n = ar.shape[0]
    return np.column_stack([np.ones(n), *blocks, ar])


class UMIDAS(_MidasBase):
    r"""Unrestricted MIDAS regression (Foroni, Marcellino & Schumacher, 2015), OLS.

    Parameters
    ----------
    predictors : sequence of str, optional
        High-frequency indicators (default: every series except the target).
    n_lags : int, optional
        Number of high-frequency lags of each indicator (in its native periods,
        most recent = lag 0). Default: the frequency ratio (3 for monthly indicators
        of a quarterly target).
    target_lags : int, default 0
        Number of autoregressive target lags (direct, at the forecast horizon).
    ragged_edge : {"realign", "ar"}, default "realign"
        Treatment of unreleased indicator values (see the module notes).
    ar_lags : int, default 1
        AR order used to complete indicators when ``ragged_edge="ar"``.

    Attributes
    ----------
    coef_ : pandas.Series
        Coefficients of the equation used for the nowcast of the first period after
        the last target observation (``const``, ``<x>_lag<j>``, ``<y>_L<h>``).
    shifts_ : pandas.Series
        Shifts :math:`s_i` of that equation.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from nowcastbox.benchmarks import UMIDAS
    >>> rng = np.random.default_rng(0)
    >>> idx = pd.period_range("2000-01", periods=150, freq="M")
    >>> x = rng.standard_normal(150)
    >>> y = 0.5 + 1.0 * x + 0.5 * np.roll(x, 1) + 0.25 * np.roll(x, 2)
    >>> y = pd.Series(y, index=idx).where(idx.month % 3 == 0)
    >>> y.iloc[-1] = np.nan  # last quarter not yet released: a nowcast with shift 0
    >>> frame = pd.DataFrame({"x": x, "y": y}, index=idx)
    >>> bench = UMIDAS().fit(frame, "y", frequency={"x": "M", "y": "Q"})
    >>> np.round(bench.coef_.to_numpy(), 6).tolist()
    [0.5, 1.0, 0.5, 0.25]
    """

    def __init__(
        self,
        predictors: Sequence[str] | None = None,
        n_lags: int | None = None,
        target_lags: int = 0,
        ragged_edge: str = "realign",
        ar_lags: int = 1,
    ) -> None:
        self.predictors = predictors
        self.n_lags = n_lags
        self.target_lags = target_lags
        self.ragged_edge = ragged_edge
        self.ar_lags = ar_lags

    def _n_params(self, blocks: list[np.ndarray], ar: np.ndarray) -> int:
        return 1 + sum(b.shape[1] for b in blocks) + ar.shape[1]

    def _estimate(self, blocks: list[np.ndarray], ar: np.ndarray, y: np.ndarray) -> np.ndarray:
        coef, _ = lstsq(_linear_design(blocks, ar), y)
        return coef

    def _combine(self, params: np.ndarray, blocks: list[np.ndarray], ar: np.ndarray) -> np.ndarray:
        return _linear_design(blocks, ar) @ params

    @property
    def coef_(self) -> pd.Series:
        """OLS coefficients of the default (nowcast) equation."""
        params = self._default_params()
        assert self._indicators is not None  # noqa: S101
        names = ["const"]
        for ind in self._indicators:
            names += [f"{ind.name}_lag{j}" for j in range(ind.n_lags)]
        return pd.Series(params, index=names + self._ar_names(), name="coef")


class MIDAS(_MidasBase):
    r"""MIDAS regression with a parametric lag polynomial, estimated by NLS.

    Parameters
    ----------
    predictors : sequence of str, optional
        High-frequency indicators (default: every series except the target).
    polynomial : {"exp_almon", "beta"}, default "exp_almon"
        Lag polynomial (:func:`exp_almon_weights`, :func:`beta_weights`).
    n_lags : int, optional
        Number of high-frequency lags :math:`K` of each indicator. Default: three
        times the frequency ratio (9 months for a quarterly target).
    target_lags : int, default 0
        Number of autoregressive target lags (direct, at the forecast horizon).
    ragged_edge : {"realign", "ar"}, default "realign"
        Treatment of unreleased indicator values (see the module notes).
    ar_lags : int, default 1
        AR order used to complete indicators when ``ragged_edge="ar"``.
    max_iter : int, default 500
        Maximum iterations of each L-BFGS-B run.

    Attributes
    ----------
    coef_ : pandas.Series
        Linear coefficients (``const``, slope :math:`\beta_i` of each indicator, target
        lags) of the default (nowcast) equation.
    theta_ : pandas.DataFrame
        Polynomial parameters (rows: indicators; columns ``theta1``, ``theta2``).
    lag_weights_ : dict of str to numpy.ndarray
        Estimated lag weights of each indicator (most recent first).

    Warns
    -----
    ConvergenceWarning
        If the optimiser did not converge.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from nowcastbox.benchmarks import MIDAS, exp_almon_weights
    >>> rng = np.random.default_rng(0)
    >>> n = 600
    >>> idx = pd.period_range("1960-01", periods=n, freq="M")
    >>> x = rng.standard_normal(n)
    >>> w = exp_almon_weights(0.3, -0.1, 6)
    >>> signal = np.convolve(x, w)[:n]
    >>> y = pd.Series(1.0 + 2.0 * signal + 0.05 * rng.standard_normal(n), index=idx)
    >>> y.iloc[-1] = np.nan
    >>> frame = pd.DataFrame({"x": x, "y": y.where(idx.month % 3 == 0)}, index=idx)
    >>> bench = MIDAS(n_lags=6).fit(frame, "y", frequency={"x": "M", "y": "Q"})
    >>> np.round(bench.coef_.to_numpy(), 1).tolist()
    [1.0, 2.0]
    >>> bool(np.abs(bench.lag_weights_["x"] - w).max() < 0.05)
    True
    """

    _default_n_lags_factor = 3

    def __init__(
        self,
        predictors: Sequence[str] | None = None,
        polynomial: str = "exp_almon",
        n_lags: int | None = None,
        target_lags: int = 0,
        ragged_edge: str = "realign",
        ar_lags: int = 1,
        max_iter: int = 500,
    ) -> None:
        self.predictors = predictors
        self.polynomial = polynomial
        self.n_lags = n_lags
        self.target_lags = target_lags
        self.ragged_edge = ragged_edge
        self.ar_lags = ar_lags
        self.max_iter = max_iter

    def _validate(self) -> None:
        super()._validate()
        if self.polynomial not in _POLYNOMIALS:
            raise ValueError(f"polynomial must be one of {_POLYNOMIALS}, got {self.polynomial!r}.")
        check_int("max_iter", self.max_iter, 1)

    def _weights(self, theta: np.ndarray, blocks: list[np.ndarray]) -> list[np.ndarray]:
        func = _WEIGHT_FUNCTIONS[self.polynomial]
        return [
            func(float(theta[2 * i]), float(theta[2 * i + 1]), b.shape[1])
            for i, b in enumerate(blocks)
        ]

    def _aggregated(
        self, theta: np.ndarray, blocks: list[np.ndarray], ar: np.ndarray
    ) -> np.ndarray:
        weights = self._weights(theta, blocks)
        return _linear_design([b @ w[:, None] for b, w in zip(blocks, weights, strict=True)], ar)

    def _n_params(self, blocks: list[np.ndarray], ar: np.ndarray) -> int:
        return 1 + 3 * len(blocks) + ar.shape[1]

    def _profile(
        self, theta: np.ndarray, blocks: list[np.ndarray], ar: np.ndarray, y: np.ndarray
    ) -> float:
        return lstsq(self._aggregated(theta, blocks, ar), y)[1]

    def _estimate(self, blocks: list[np.ndarray], ar: np.ndarray, y: np.ndarray) -> np.ndarray:
        bounds = list(_BOUNDS[self.polynomial]) * len(blocks)
        starts = [
            np.tile(np.asarray(s, dtype=float), len(blocks)) for s in _STARTS[self.polynomial]
        ]
        x0 = min(starts, key=lambda theta: self._profile(theta, blocks, ar, y))
        # L-BFGS-B stops on absolute gradient/step tolerances: minimise the scale-free
        # SSR / SST so that theta (and the nowcast) do not depend on the units of y.
        centred = y - y.mean()
        y_scaled = y / max(float(np.sqrt(centred @ centred)), np.finfo(float).tiny)
        res = minimize(
            self._profile,
            x0,
            args=(blocks, ar, y_scaled),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": int(self.max_iter), "ftol": 1e-12, "gtol": 1e-8},
        )
        if not bool(res.success):
            warnings.warn(
                f"MIDAS: the NLS optimiser did not converge ({res.message}).",
                ConvergenceWarning,
                stacklevel=4,
            )
        best_theta = np.asarray(res.x, dtype=float)
        coef, _ = lstsq(self._aggregated(best_theta, blocks, ar), y)
        return np.concatenate([best_theta, coef])

    def _combine(self, params: np.ndarray, blocks: list[np.ndarray], ar: np.ndarray) -> np.ndarray:
        n_theta = 2 * len(blocks)
        return self._aggregated(params[:n_theta], blocks, ar) @ params[n_theta:]

    @property
    def coef_(self) -> pd.Series:
        """Linear coefficients of the default (nowcast) equation."""
        params = self._default_params()
        assert self._indicators is not None  # noqa: S101
        names = ["const", *[ind.name for ind in self._indicators], *self._ar_names()]
        return pd.Series(params[2 * len(self._indicators) :], index=names, name="coef")

    @property
    def theta_(self) -> pd.DataFrame:
        """Lag-polynomial parameters of each indicator."""
        params = self._default_params()
        assert self._indicators is not None  # noqa: S101
        m = len(self._indicators)
        return pd.DataFrame(
            params[: 2 * m].reshape(m, 2),
            index=[ind.name for ind in self._indicators],
            columns=["theta1", "theta2"],
        )

    @property
    def lag_weights_(self) -> dict[str, np.ndarray]:
        """Estimated lag weights of each indicator (most recent first)."""
        theta = self.theta_.to_numpy(dtype=float)
        assert self._indicators is not None  # noqa: S101
        func = _WEIGHT_FUNCTIONS[self.polynomial]
        return {
            ind.name: func(float(theta[i, 0]), float(theta[i, 1]), ind.n_lags)
            for i, ind in enumerate(self._indicators)
        }
