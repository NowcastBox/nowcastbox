r"""Autoregressive benchmark AR(:math:`p`) on the target at its native frequency.

The target (quarterly GDP growth, or a monthly target) follows

.. math::

    y_t = c + \sum_{j=1}^{p} \phi_j\, y_{t-j} + u_t,

estimated by OLS on every period where :math:`y_t` and its :math:`p` lags are
observed. Forecasts for :math:`h` periods after the last observation are iterated
(:math:`\hat y_{T+h} = c + \sum_j \phi_j \hat y_{T+h-j}`); for observed periods the
benchmark returns the one-step-ahead fitted value. The order can be chosen by AIC or
BIC (Lütkepohl, 2005, §4.3) on a common estimation sample. This is the standard
benchmark of the nowcasting literature (Giannone, Reichlin & Small, 2008; Bańbura et
al., 2013).

References
----------
Lütkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*. Springer.

Giannone, D., Reichlin, L. & Small, D. (2008). Nowcasting: The real-time informational
content of macroeconomic data. *Journal of Monetary Economics*, 55(4), 665-676.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nowcastbox.benchmarks._base import UnivariateBenchmark
from nowcastbox.benchmarks._utils import check_int, lstsq
from nowcastbox.core.exceptions import NowcastDataError

__all__ = ["AR"]

_TRENDS = ("c", "n")
_CRITERIA = ("aic", "bic")


def _lag_rows(values: np.ndarray, p: int, first: int) -> tuple[np.ndarray, np.ndarray]:
    """Dependent variable and lag matrix for ``t = first..T`` (complete rows only)."""
    n = values.size
    if first >= n:
        return np.empty(0), np.empty((0, p))
    y = values[first:]
    lags = (
        np.column_stack([values[first - j : n - j] for j in range(1, p + 1)])
        if p
        else np.empty((y.size, 0))
    )
    ok = np.isfinite(y) & np.isfinite(lags).all(axis=1)
    return y[ok], lags[ok]


class AR(UnivariateBenchmark):
    r"""AR(:math:`p`) benchmark on the target's native frequency.

    Parameters
    ----------
    p : int or {"aic", "bic"}, default 1
        Autoregressive order, or an information criterion selecting it in
        ``0..max_p``.
    max_p : int, default 4
        Largest order considered when ``p`` is a criterion.
    trend : {"c", "n"}, default "c"
        Include a constant (``"c"``) or not (``"n"``).

    Attributes
    ----------
    order_ : int
        Selected order.
    coef_ : pandas.Series
        ``const`` (if any) and ``ar.L1 .. ar.Lp``.
    sigma2_ : float
        Residual variance (``SSR / (n - k)``).
    n_obs_ : int
        Number of observations in the estimation.

    Notes
    -----
    Missing lags at the forecast origin (gaps in the target) are replaced by the
    sample mean when iterating forecasts. With ``p`` a criterion, the order minimises
    :math:`n \log(\mathrm{SSR}/n) + k\, c_n` with :math:`c_n = 2` (AIC) or
    :math:`\log n` (BIC) over the sample starting at observation ``max_p``; the
    chosen model is then re-estimated on all usable observations.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from nowcastbox.benchmarks import AR
    >>> rng = np.random.default_rng(0)
    >>> y = np.zeros(80)
    >>> for t in range(1, 80):
    ...     y[t] = 0.5 + 0.6 * y[t - 1] + 0.1 * rng.standard_normal()
    >>> idx = pd.period_range("2000Q1", periods=80, freq="Q")
    >>> model = AR(p=1).fit(pd.DataFrame({"gdp": y}, index=idx), "gdp", frequency="Q")
    >>> np.round(model.coef_.to_numpy(), 1).tolist()
    [0.5, 0.6]
    >>> forecast = model.predict(["2020Q1", "2020Q2"])
    >>> c, phi = model.coef_
    >>> bool(np.isclose(forecast.iloc[1], c * (1 + phi) + phi**2 * y[-1]))
    True
    """

    def __init__(self, p: int | str = 1, max_p: int = 4, trend: str = "c") -> None:
        self.p = p
        self.max_p = max_p
        self.trend = trend

    _coef: np.ndarray | None = None
    _order: int = 0
    _sigma2: float = float("nan")
    _n_obs: int = 0
    _mean: float = float("nan")

    def _check_params(self) -> None:
        if self.trend not in _TRENDS:
            raise ValueError(f"trend must be one of {_TRENDS}, got {self.trend!r}.")
        check_int("max_p", self.max_p, 0)
        if isinstance(self.p, str):
            if self.p not in _CRITERIA:
                raise ValueError(f"p must be an integer or one of {_CRITERIA}, got {self.p!r}.")
        else:
            check_int("p", self.p, 0)

    def _design(self, lags: np.ndarray) -> np.ndarray:
        if self.trend == "c":
            return np.column_stack([np.ones(lags.shape[0]), lags])
        return lags

    def _select_order(self, values: np.ndarray) -> int:
        penalty_kind = str(self.p)
        best, best_ic = 0, np.inf
        for p in range(int(self.max_p) + 1):
            y, lags = _lag_rows(values, p, int(self.max_p))
            design = self._design(lags)
            k = design.shape[1]
            if y.size <= k or k == 0:
                continue
            _, ssr = lstsq(design, y)
            c_n = 2.0 if penalty_kind == "aic" else float(np.log(y.size))
            ic = y.size * np.log(max(ssr, 1e-300) / y.size) + k * c_n
            if ic < best_ic:
                best, best_ic = p, ic
        return best

    def _fit_history(self, values: np.ndarray) -> None:
        self._check_params()
        order = self._select_order(values) if isinstance(self.p, str) else int(self.p)
        y, lags = _lag_rows(values, order, order)
        design = self._design(lags)
        k = design.shape[1]
        if y.size < k + 1:
            raise NowcastDataError(
                f"Too few complete observations ({y.size}) to estimate an AR({order})."
            )
        if k == 0:
            coef, ssr = np.empty(0), float(y @ y)
        else:
            coef, ssr = lstsq(design, y)
        self._order = order
        self._coef = coef
        self._n_obs = int(y.size)
        self._sigma2 = ssr / max(y.size - k, 1)
        self._mean = float(np.nanmean(values))

    def _split(self) -> tuple[float, np.ndarray]:
        assert self._coef is not None  # noqa: S101
        if self.trend == "c":
            return float(self._coef[0]), self._coef[1:]
        return 0.0, self._coef

    def _forecast_path(self, horizon: int) -> np.ndarray:
        """History followed by ``horizon`` iterated forecasts (NaN lags -> mean)."""
        assert self._history is not None  # noqa: S101
        const, phi = self._split()
        path = np.concatenate([self._history, np.full(horizon, np.nan)])
        start = self._history.size
        for t in range(start, path.size):
            lags = path[t - phi.size : t][::-1] if phi.size else np.empty(0)
            lags = np.where(np.isfinite(lags), lags, self._mean)
            path[t] = const + float(lags @ phi)
        return path

    def _predict_positions(self, positions: np.ndarray) -> np.ndarray:
        assert self._history is not None  # noqa: S101
        const, phi = self._split()
        last = self._history.size - 1
        horizon = max(int(positions.max(initial=last)) - last, 0)
        path = self._forecast_path(horizon)
        out = np.full(positions.size, np.nan)
        for i, k in enumerate(positions):
            if k > last:
                out[i] = path[k]
            elif k >= phi.size:
                lags = self._history[k - phi.size : k][::-1] if phi.size else np.empty(0)
                out[i] = const + float(lags @ phi)
        return out

    @property
    def order_(self) -> int:
        """Selected autoregressive order."""
        self._check_fitted()
        return self._order

    @property
    def coef_(self) -> pd.Series:
        """Estimated coefficients (``const``, ``ar.L1`` ...)."""
        self._check_fitted()
        assert self._coef is not None  # noqa: S101
        names = (["const"] if self.trend == "c" else []) + [
            f"ar.L{j}" for j in range(1, self._order + 1)
        ]
        return pd.Series(self._coef, index=names, name="coef", dtype=float)

    @property
    def sigma2_(self) -> float:
        """Residual variance."""
        self._check_fitted()
        return self._sigma2

    @property
    def n_obs_(self) -> int:
        """Number of observations used in the estimation."""
        self._check_fitted()
        return self._n_obs
