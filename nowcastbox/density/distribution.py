r"""Predictive distribution of a nowcast (innovation I5).

:class:`NowcastDistribution` represents, for one or several target periods, the
predictive distribution of the target as a finite **Gaussian mixture**

.. math::

    p_t(y) = \sum_{k=1}^{K} w_k\, \frac{1}{\sigma_{t,k}}
             \varphi\!\left(\frac{y - \mu_{t,k}}{\sigma_{t,k}}\right),
    \qquad \sum_k w_k = 1 .

With :math:`K = 1` it is the Gaussian distribution implied by the Kalman smoother
(filtering/smoothing uncertainty only); with :math:`K = B` bootstrap replications it is
the mixture over re-estimated parameters that also carries parameter uncertainty
(Hamilton, 1986; Pfeffermann & Tiller, 2005). Densities, distribution functions,
moments and random draws are exact for the mixture; quantiles are obtained by
bisection of the (monotone) mixture distribution function.

References
----------
Hamilton, J. D. (1986). A standard error for the estimated state vector of a
state-space model. *Journal of Econometrics*, 33(3), 387-397.

Pfeffermann, D. & Tiller, R. (2005). Bootstrap approximation to prediction MSE for
state-space models with estimated parameters. *Journal of Time Series Analysis*,
26(6), 893-916.

Gneiting, T. & Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and
estimation. *Journal of the American Statistical Association*, 102(477), 359-378.

Britton, E., Fisher, P. & Whitley, J. (1998). The Inflation Report projections:
understanding the fan chart. *Bank of England Quarterly Bulletin*, 38(1), 30-37.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import scipy.special
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    from matplotlib.axes import Axes

__all__ = ["DEFAULT_LEVELS", "NowcastDistribution"]

FloatArray = NDArray[np.float64]

DEFAULT_LEVELS: tuple[float, ...] = (0.5, 0.68, 0.9)
"""Central interval levels reported by :meth:`NowcastDistribution.to_frame`."""

_BISECTION_ITERATIONS = 100
_LOG_SQRT_2PI = 0.5 * float(np.log(2.0 * np.pi))


def _level_label(level: float) -> str:
    """Column suffix of a central interval level (``0.68 -> "68"``, ``0.955 -> "95.5"``)."""
    return f"{100.0 * level:.10g}"


def _check_levels(levels: float | Iterable[float]) -> list[float]:
    """Validate interval levels in ``(0, 1)``."""
    values = [float(levels)] if np.isscalar(levels) else [float(v) for v in levels]  # type: ignore[arg-type]
    if not values:
        raise ValueError("At least one interval level is required.")
    for v in values:
        if not 0.0 < v < 1.0:
            raise ValueError(f"Interval levels must be in (0, 1), got {v}.")
    return values


def _check_probabilities(q: float | Iterable[float]) -> FloatArray:
    """Validate probabilities in ``(0, 1)`` (1-D array)."""
    arr = np.atleast_1d(np.asarray(q, dtype=np.float64))
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("Probabilities must be a scalar or a non-empty 1-D sequence.")
    if not bool(np.all((arr > 0.0) & (arr < 1.0))):
        raise ValueError(f"Probabilities must lie in (0, 1), got {arr.tolist()}.")
    return arr


class NowcastDistribution:
    r"""Predictive distribution of a nowcast for one or more target periods.

    A Gaussian mixture per period with common component weights (see the module
    docstring). Use :func:`nowcastbox.density.nowcast_distribution` to build it from
    fitted model results.

    Parameters
    ----------
    index : pandas.PeriodIndex or sequence of Period/str
        Target periods (native frequency of the target).
    locs : array_like, shape (n_periods,) or (n_periods, n_components)
        Component means :math:`\mu_{t,k}`.
    scales : array_like, same shape as ``locs``
        Component standard deviations :math:`\sigma_{t,k} > 0`.
    weights : array_like, shape (n_components,), optional
        Component weights (non-negative, normalised to sum one). Default: equal.
    target : str, default ""
        Name of the target series.
    point : array_like, shape (n_periods,), optional
        Model point nowcast (default: the mixture mean).
    variance_decomposition : pandas.DataFrame, optional
        Columns ``filtering``, ``parameter`` and ``total`` (indexed like ``index``):
        contributions of the smoothing uncertainty and of the parameter uncertainty
        to the predictive variance.
    info : mapping, optional
        Extra information (bootstrap diagnostics ...).

    Raises
    ------
    ValueError
        On inconsistent shapes, non-finite values, non-positive scales or invalid
        weights.

    See Also
    --------
    nowcastbox.density.nowcast_distribution : Build the distribution from results.
    nowcastbox.evaluation.scoring : CRPS, log score, PIT and coverage tests.

    Examples
    --------
    >>> from nowcastbox.density import NowcastDistribution
    >>> dist = NowcastDistribution(["2020Q1", "2020Q2"], [0.5, 1.0], [0.2, 0.4], target="gdp")
    >>> dist.is_gaussian, len(dist)
    (True, 2)
    >>> dist.interval(0.9).round(3)
            lower  upper
    period
    2020Q1  0.171  0.829
    2020Q2  0.342  1.658
    >>> dist["2020Q1"].cdf(0.5).tolist()
    [0.5]
    """

    def __init__(
        self,
        index: pd.PeriodIndex | Sequence[pd.Period | str],
        locs: ArrayLike,
        scales: ArrayLike,
        *,
        weights: ArrayLike | None = None,
        target: str = "",
        point: ArrayLike | None = None,
        variance_decomposition: pd.DataFrame | None = None,
        info: Mapping[str, Any] | None = None,
    ) -> None:
        self._index = self._coerce_index(index)
        n = len(self._index)
        loc_arr = self._as_matrix(locs, "locs", n)
        scale_arr = self._as_matrix(scales, "scales", n)
        if loc_arr.shape != scale_arr.shape:
            raise ValueError(
                f"locs and scales must have the same shape, got {loc_arr.shape} and "
                f"{scale_arr.shape}."
            )
        if not bool(np.all(scale_arr > 0.0)):
            raise ValueError("All component scales must be strictly positive.")
        self._locs = loc_arr
        self._scales = scale_arr
        self._weights = self._as_weights(weights, loc_arr.shape[1])
        self._target = str(target)
        if point is None:
            self._point = self._locs @ self._weights
        else:
            pt = np.asarray(point, dtype=np.float64).reshape(-1)
            if pt.shape != (n,) or not bool(np.isfinite(pt).all()):
                raise ValueError(f"point must be a finite array of length {n}.")
            self._point = pt
        self._decomposition = None if variance_decomposition is None else variance_decomposition
        self._info: dict[str, Any] = dict(info or {})
        for arr in (self._locs, self._scales, self._weights, self._point):
            arr.setflags(write=False)

    # ------------------------------------------------------------------ validation
    @staticmethod
    def _coerce_index(index: pd.PeriodIndex | Sequence[pd.Period | str]) -> pd.PeriodIndex:
        if isinstance(index, pd.PeriodIndex):
            out = index.copy()
        else:
            items = list(index)
            if not items:
                raise ValueError("index must contain at least one period.")
            try:
                out = pd.PeriodIndex([pd.Period(p) for p in items])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"index must contain periods, got {items!r}.") from exc
        if len(out) == 0:
            raise ValueError("index must contain at least one period.")
        if out.has_duplicates:
            raise ValueError("index must not contain duplicated periods.")
        out.name = "period"
        return out

    @staticmethod
    def _as_matrix(values: ArrayLike, name: str, n: int) -> FloatArray:
        arr = np.array(values, dtype=np.float64)
        if arr.ndim == 0:
            arr = arr.reshape(1, 1)
        elif arr.ndim == 1:
            arr = arr.reshape(-1, 1) if arr.shape[0] == n else arr.reshape(1, -1)
        if arr.ndim != 2 or arr.shape[0] != n or arr.shape[1] == 0:
            raise ValueError(f"{name} must have shape ({n},) or ({n}, n_components).")
        if not bool(np.isfinite(arr).all()):
            raise ValueError(f"{name} must be finite.")
        return arr

    @staticmethod
    def _as_weights(weights: ArrayLike | None, k: int) -> FloatArray:
        if weights is None:
            return np.full(k, 1.0 / k)
        w = np.array(weights, dtype=np.float64).reshape(-1)
        if w.shape != (k,) or not bool(np.isfinite(w).all()) or bool((w < 0).any()):
            raise ValueError(f"weights must be {k} finite non-negative numbers.")
        total = float(w.sum())
        if total <= 0.0:
            raise ValueError("weights must not all be zero.")
        return w / total

    # ------------------------------------------------------------------ attributes
    @property
    def index(self) -> pd.PeriodIndex:
        """Target periods."""
        return self._index.copy()

    @property
    def target(self) -> str:
        """Name of the target series."""
        return self._target

    @property
    def locs(self) -> FloatArray:
        """Component means, shape ``(n_periods, n_components)`` (read-only)."""
        return self._locs

    @property
    def scales(self) -> FloatArray:
        """Component standard deviations, shape ``(n_periods, n_components)`` (read-only)."""
        return self._scales

    @property
    def weights(self) -> FloatArray:
        """Component weights, shape ``(n_components,)`` (read-only)."""
        return self._weights

    @property
    def n_periods(self) -> int:
        """Number of target periods."""
        return int(self._locs.shape[0])

    @property
    def n_components(self) -> int:
        """Number of mixture components (1 for a Gaussian distribution)."""
        return int(self._locs.shape[1])

    @property
    def is_gaussian(self) -> bool:
        """Whether every period has a single Gaussian component."""
        return self.n_components == 1

    @property
    def info(self) -> dict[str, Any]:
        """Extra information (copy)."""
        return dict(self._info)

    @property
    def variance_decomposition(self) -> pd.DataFrame:
        """Filtering / parameter / total variance per period.

        Returns
        -------
        pandas.DataFrame
            Columns ``filtering``, ``parameter``, ``total``. When none was given, the
            within-component variance is reported as ``filtering`` and the
            between-component variance as ``parameter``.
        """
        if self._decomposition is not None:
            return self._decomposition.copy()
        within = (self._scales**2) @ self._weights
        between = self.variance.to_numpy() - within
        return pd.DataFrame(
            {"filtering": within, "parameter": np.maximum(between, 0.0), "total": within + between},
            index=self.index,
        )

    def __len__(self) -> int:
        return self.n_periods

    def __repr__(self) -> str:
        kind = "Gaussian" if self.is_gaussian else f"mixture of {self.n_components}"
        span = f"{self._index[0]}" if len(self) == 1 else f"{self._index[0]}..{self._index[-1]}"
        name = f"{self._target!r}, " if self._target else ""
        return f"NowcastDistribution({name}{span}, {kind})"

    # ------------------------------------------------------------------ selection
    def _position(self, period: pd.Period | str | int) -> int:
        if isinstance(period, int | np.integer) and not isinstance(period, bool):
            pos = int(period)
            if not -self.n_periods <= pos < self.n_periods:
                raise KeyError(f"Position {pos} out of range for {self.n_periods} periods.")
            return pos % self.n_periods
        freq = self._index.freqstr
        key = period.asfreq(freq) if isinstance(period, pd.Period) else pd.Period(str(period), freq)
        matches = np.flatnonzero(self._index == key)
        if matches.size == 0:
            raise KeyError(f"Period {key} is not in the distribution index.")
        return int(matches[0])

    def __getitem__(self, period: pd.Period | str | int) -> NowcastDistribution:
        """Distribution of a single period (label or integer position)."""
        return self.select([period])

    def select(self, periods: Iterable[pd.Period | str | int]) -> NowcastDistribution:
        """Distribution restricted to some periods.

        Parameters
        ----------
        periods : iterable of Period, str or int
            Period labels or integer positions.

        Returns
        -------
        NowcastDistribution
            New object with the selected periods.

        Raises
        ------
        KeyError
            If a period is not in :attr:`index`.

        Examples
        --------
        >>> from nowcastbox.density import NowcastDistribution
        >>> d = NowcastDistribution(["2020Q1", "2020Q2"], [0.0, 1.0], [1.0, 1.0])
        >>> d.select(["2020Q2"]).mean.tolist()
        [1.0]
        """
        pos = [self._position(p) for p in periods]
        dec = None if self._decomposition is None else self._decomposition.iloc[pos]
        return NowcastDistribution(
            self._index[pos],
            self._locs[pos],
            self._scales[pos],
            weights=self._weights,
            target=self._target,
            point=self._point[pos],
            variance_decomposition=dec,
            info=self._info,
        )

    # ------------------------------------------------------------------ moments
    @property
    def point(self) -> pd.Series:
        """Model point nowcast (equals :attr:`mean` unless given explicitly)."""
        return pd.Series(self._point, index=self.index, name="point")

    @property
    def mean(self) -> pd.Series:
        """Mean of the predictive distribution."""
        return pd.Series(self._locs @ self._weights, index=self.index, name="mean")

    @property
    def variance(self) -> pd.Series:
        """Variance of the predictive distribution (law of total variance)."""
        m = self._locs @ self._weights
        second = (self._scales**2 + self._locs**2) @ self._weights
        return pd.Series(np.maximum(second - m**2, 0.0), index=self.index, name="variance")

    @property
    def std(self) -> pd.Series:
        """Standard deviation of the predictive distribution."""
        return pd.Series(np.sqrt(self.variance.to_numpy()), index=self.index, name="std")

    @property
    def median(self) -> pd.Series:
        """Median of the predictive distribution."""
        return self.quantiles(0.5).iloc[:, 0].rename("median")

    # ------------------------------------------------------------------ pdf / cdf
    def _broadcast(self, x: ArrayLike) -> tuple[FloatArray, tuple[int, ...]]:
        arr = np.asarray(x, dtype=np.float64)
        shape = np.broadcast_shapes(arr.shape, (self.n_periods,))
        return np.broadcast_to(arr, shape), shape

    def cdf(self, x: ArrayLike) -> FloatArray:
        """Distribution function :math:`F_t(x)`.

        Parameters
        ----------
        x : array_like
            Points, broadcast against the period axis (the last axis, of length
            :attr:`n_periods`). For a single-period distribution any shape works.

        Returns
        -------
        numpy.ndarray
            :math:`F_t(x)` with the broadcast shape.

        Examples
        --------
        >>> from nowcastbox.density import NowcastDistribution
        >>> d = NowcastDistribution(["2020Q1"], [0.0], [1.0])
        >>> d.cdf([-1.96, 0.0, 1.96]).round(3).tolist()
        [0.025, 0.5, 0.975]
        """
        xb, _ = self._broadcast(x)
        z = (xb[..., None] - self._locs) / self._scales
        return np.asarray(scipy.special.ndtr(z) @ self._weights, dtype=np.float64)

    def logpdf(self, x: ArrayLike) -> FloatArray:
        r"""Log predictive density :math:`\log p_t(x)`.

        Parameters
        ----------
        x : array_like
            Points, broadcast against the period axis (see :meth:`cdf`).

        Returns
        -------
        numpy.ndarray
            Log density with the broadcast shape.

        Examples
        --------
        >>> import numpy as np
        >>> from nowcastbox.density import NowcastDistribution
        >>> d = NowcastDistribution(["2020Q1"], [0.0], [1.0])
        >>> bool(np.isclose(d.logpdf(0.0)[0], -0.5 * np.log(2 * np.pi)))
        True
        """
        xb, _ = self._broadcast(x)
        z = (xb[..., None] - self._locs) / self._scales
        log_comp = -0.5 * z**2 - np.log(self._scales) - _LOG_SQRT_2PI
        with np.errstate(divide="ignore"):
            log_w = np.log(self._weights)
        return np.asarray(scipy.special.logsumexp(log_comp + log_w, axis=-1), dtype=np.float64)

    def pdf(self, x: ArrayLike) -> FloatArray:
        """Predictive density :math:`p_t(x)`.

        Parameters
        ----------
        x : array_like
            Points, broadcast against the period axis (see :meth:`cdf`).

        Returns
        -------
        numpy.ndarray
            Density with the broadcast shape.

        Examples
        --------
        >>> from nowcastbox.density import NowcastDistribution
        >>> d = NowcastDistribution(["2020Q1"], [0.0], [1.0])
        >>> round(float(d.pdf(0.0)[0]), 4)
        0.3989
        """
        return np.exp(self.logpdf(x))

    # ------------------------------------------------------------------ quantiles
    def _ppf(self, probs: FloatArray) -> FloatArray:
        """Quantiles, shape ``(n_periods, len(probs))``."""
        if self.is_gaussian:
            return self._locs + self._scales * scipy.special.ndtri(probs)[None, :]
        span = float(scipy.special.ndtri(min(probs.min(), 1.0 - probs.max()))) - 1.0
        lo = (self._locs + span * self._scales).min(axis=1)[:, None] * np.ones(probs.size)
        hi = (self._locs - span * self._scales).max(axis=1)[:, None] * np.ones(probs.size)
        locs = self._locs[:, None, :]
        scales = self._scales[:, None, :]
        for _ in range(_BISECTION_ITERATIONS):
            mid = 0.5 * (lo + hi)
            value = scipy.special.ndtr((mid[..., None] - locs) / scales) @ self._weights
            below = value < probs[None, :]
            lo = np.where(below, mid, lo)
            hi = np.where(below, hi, mid)
        return 0.5 * (lo + hi)

    def ppf(self, q: float | Iterable[float]) -> FloatArray:
        """Quantile function :math:`F_t^{-1}(q)`.

        Parameters
        ----------
        q : float or sequence of float
            Probabilities in ``(0, 1)``.

        Returns
        -------
        numpy.ndarray, shape (n_periods, len(q))
            Quantiles (exact for Gaussian distributions, bisection to machine
            precision for mixtures).

        Raises
        ------
        ValueError
            If a probability is outside ``(0, 1)``.

        Examples
        --------
        >>> from nowcastbox.density import NowcastDistribution
        >>> d = NowcastDistribution(["2020Q1"], [1.0], [2.0])
        >>> d.ppf([0.5]).tolist()
        [[1.0]]
        """
        return self._ppf(_check_probabilities(q))

    def quantiles(self, q: float | Iterable[float]) -> pd.DataFrame:
        """Quantiles of every period.

        Parameters
        ----------
        q : float or sequence of float
            Probabilities in ``(0, 1)``.

        Returns
        -------
        pandas.DataFrame
            Index = periods, one column per probability.

        Raises
        ------
        ValueError
            If a probability is outside ``(0, 1)``.

        Examples
        --------
        >>> from nowcastbox.density import NowcastDistribution
        >>> d = NowcastDistribution(["2020Q1"], [0.0], [1.0])
        >>> d.quantiles([0.05, 0.95]).round(3).to_numpy().tolist()
        [[-1.645, 1.645]]
        """
        probs = _check_probabilities(q)
        return pd.DataFrame(self._ppf(probs), index=self.index, columns=pd.Index(probs, name="q"))

    def interval(self, level: float = 0.9) -> pd.DataFrame:
        """Central (equal-tailed) prediction interval.

        Parameters
        ----------
        level : float, default 0.9
            Nominal coverage in ``(0, 1)``.

        Returns
        -------
        pandas.DataFrame
            Columns ``lower`` and ``upper`` (quantiles ``(1 - level)/2`` and
            ``(1 + level)/2``).

        Raises
        ------
        ValueError
            If ``level`` is outside ``(0, 1)``.

        Examples
        --------
        >>> from nowcastbox.density import NowcastDistribution
        >>> d = NowcastDistribution(["2020Q1"], [0.0], [1.0])
        >>> d.interval(0.95).round(2).to_numpy().tolist()
        [[-1.96, 1.96]]
        """
        (lvl,) = _check_levels(level)
        q = self._ppf(np.array([0.5 * (1.0 - lvl), 0.5 * (1.0 + lvl)]))
        return pd.DataFrame(q, index=self.index, columns=["lower", "upper"])

    # ------------------------------------------------------------------ sampling
    def sample(
        self, n: int, *, random_state: int | np.random.Generator | None = None
    ) -> pd.DataFrame:
        """Draw from the predictive distribution.

        Draws are independent across periods (the joint distribution across periods is
        not modelled).

        Parameters
        ----------
        n : int
            Number of draws per period (``>= 1``).
        random_state : int, numpy.random.Generator or None
            Seed or generator.

        Returns
        -------
        pandas.DataFrame
            Shape ``(n, n_periods)``; columns = periods.

        Raises
        ------
        ValueError
            If ``n`` is not a positive integer.

        Examples
        --------
        >>> from nowcastbox.density import NowcastDistribution
        >>> d = NowcastDistribution(["2020Q1"], [0.0], [1.0])
        >>> d.sample(1000, random_state=0).shape
        (1000, 1)
        """
        if isinstance(n, bool) or not isinstance(n, int | np.integer) or n < 1:
            raise ValueError(f"n must be a positive integer, got {n!r}.")
        rng = np.random.default_rng(random_state)
        comp = rng.choice(self.n_components, size=(int(n), self.n_periods), p=self._weights)
        rows = np.arange(self.n_periods)[None, :]
        draws = self._locs[rows, comp] + self._scales[rows, comp] * rng.standard_normal(comp.shape)
        return pd.DataFrame(draws, columns=self.index)

    # ------------------------------------------------------------------ export
    def to_frame(self, levels: float | Iterable[float] = DEFAULT_LEVELS) -> pd.DataFrame:
        """Summary table: point, mean, std, median and central interval bounds.

        Parameters
        ----------
        levels : float or iterable of float, default (0.5, 0.68, 0.9)
            Interval levels; columns ``lower_<100*level>``/``upper_<100*level>``.

        Returns
        -------
        pandas.DataFrame
            One row per period.

        Raises
        ------
        ValueError
            If a level is outside ``(0, 1)``.

        Examples
        --------
        >>> from nowcastbox.density import NowcastDistribution
        >>> d = NowcastDistribution(["2020Q1"], [0.0], [1.0])
        >>> list(d.to_frame(levels=[0.9]).columns)
        ['point', 'mean', 'std', 'median', 'lower_90', 'upper_90']
        """
        lv = _check_levels(levels)
        probs = np.array([0.5] + [p for v in lv for p in (0.5 * (1 - v), 0.5 * (1 + v))])
        q = self._ppf(probs)
        frame = pd.DataFrame(
            {
                "point": self._point,
                "mean": self.mean.to_numpy(),
                "std": self.std.to_numpy(),
                "median": q[:, 0],
            },
            index=self.index,
        )
        for i, v in enumerate(lv):
            label = _level_label(v)
            frame[f"lower_{label}"] = q[:, 1 + 2 * i]
            frame[f"upper_{label}"] = q[:, 2 + 2 * i]
        return frame

    def fan_chart_frame(self, levels: float | Iterable[float] = DEFAULT_LEVELS) -> pd.DataFrame:
        """Long-format bands for fan charts (backend-agnostic plot hook).

        Parameters
        ----------
        levels : float or iterable of float, default (0.5, 0.68, 0.9)
            Interval levels.

        Returns
        -------
        pandas.DataFrame
            Columns ``period``, ``level``, ``lower``, ``upper``, ``median``, sorted
            from the widest to the narrowest band (the drawing order of a fan chart).

        Examples
        --------
        >>> from nowcastbox.density import NowcastDistribution
        >>> d = NowcastDistribution(["2020Q1"], [0.0], [1.0])
        >>> d.fan_chart_frame([0.5, 0.9])["level"].tolist()
        [0.9, 0.5]
        """
        lv = sorted(_check_levels(levels), reverse=True)
        median = self.median.to_numpy()
        parts = []
        for v in lv:
            band = self.interval(v)
            parts.append(
                pd.DataFrame(
                    {
                        "period": self._index,
                        "level": v,
                        "lower": band["lower"].to_numpy(),
                        "upper": band["upper"].to_numpy(),
                        "median": median,
                    }
                )
            )
        return pd.concat(parts, ignore_index=True)

    def plot(
        self,
        *,
        levels: float | Iterable[float] = DEFAULT_LEVELS,
        history: pd.Series | None = None,
        ax: Axes | None = None,
        color: str = "C0",
        title: str | None = None,
    ) -> Axes:
        """Fan chart of the predictive distribution (matplotlib).

        Parameters
        ----------
        levels : float or iterable of float, default (0.5, 0.68, 0.9)
            Interval levels drawn as nested bands (darker = narrower).
        history : pandas.Series, optional
            Observed target values (PeriodIndex of the same frequency) drawn as a line.
        ax : matplotlib.axes.Axes, optional
            Axes to draw on (default: a new figure).
        color : str, default "C0"
            Colour of the bands and of the median line.
        title : str, optional
            Axes title (default: ``"Nowcast density: <target>"``).

        Returns
        -------
        matplotlib.axes.Axes
            The axes.

        Examples
        --------
        >>> import matplotlib
        >>> matplotlib.use("Agg")
        >>> from nowcastbox.density import NowcastDistribution
        >>> d = NowcastDistribution(["2020Q1", "2020Q2"], [0.0, 0.5], [1.0, 1.5])
        >>> ax = d.plot()
        >>> len(ax.collections)
        3
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 4))
        bands = self.fan_chart_frame(levels)
        x = self._index.to_timestamp(how="end").normalize()
        lv = sorted(set(bands["level"]), reverse=True)
        for i, v in enumerate(lv):
            band = bands[bands["level"] == v]
            alpha = 0.15 + 0.5 * (i + 1) / (len(lv) + 1)
            ax.fill_between(
                x,
                band["lower"].to_numpy(),
                band["upper"].to_numpy(),
                color=color,
                alpha=alpha,
                linewidth=0,
                label=f"{_level_label(v)}%",
            )
        ax.plot(x, self.median.to_numpy(), color=color, marker="o", label="median")
        if history is not None:
            hist = history.dropna()
            if isinstance(hist.index, pd.PeriodIndex) and len(hist):
                hx = hist.index.to_timestamp(how="end").normalize()
                ax.plot(hx, hist.to_numpy(dtype=float), color="black", label="observed")
        ax.set_title(title if title is not None else f"Nowcast density: {self._target}".strip())
        ax.legend(loc="best")
        return ax
