r"""Temporal aggregation constraints and frequency conversion utilities.

Aggregation constraints
-----------------------
A low-frequency observation :math:`y_t` relates to a latent high-frequency series
:math:`x_t` through weights on its current and lagged values,

.. math::

    y_t = \sum_{j=0}^{L-1} w_j\, x_{t-j},

where :math:`t` is the last high-frequency period of the low-frequency period. With
:math:`k` high-frequency periods per low-frequency period:

* flows (sums): :math:`w = (1, \dots, 1)` of length :math:`k`;
* averages: :math:`w = (1, \dots, 1)/k`;
* stocks (end-of-period values): :math:`w = (1, 0, \dots, 0)`;
* growth rates of flows/averages, approximated with high-frequency growth rates
  (Mariano & Murasawa, 2003): triangular weights of length :math:`2k-1`,
  :math:`(1, 2, \dots, k, \dots, 2, 1)` - :math:`(1, 2, 3, 2, 1)` for months within
  quarters, times :math:`1/k` when normalised (Bańbura & Modugno, 2014, eq. for
  :math:`y^Q_t`).

In a factor model :math:`x_t = \lambda' f_t + e_t`, the low-frequency series loads on
:math:`(f_t, \dots, f_{t-L+1})` with loadings :math:`(w_0\lambda, \dots,
w_{L-1}\lambda)`. :func:`loading_constraints` returns the linear restrictions
:math:`R\,\mathrm{vec}(\Lambda) = q` that impose this structure on freely estimated
loadings (the restriction matrix of Bańbura & Modugno, 2014). Everything is generic
in the pair of frequencies (plan innovation I1).

Calendar-aware aggregation (innovation I1)
------------------------------------------
When the number of high-frequency periods per low-frequency period varies over the
calendar (days per month, weeks per month or quarter...), the weights depend on the
period: :class:`CalendarAggregation` and :func:`calendar_weight_matrix` give, for every
period of a base grid, the exact weights of
:meth:`~nowcastbox.core.frequency.AggregationType.calendar_weights` (sum for flows, mean
for averages, last value for stocks, and the growth rate of the low-frequency average
for ``"mariano_murasawa"``: :math:`(1, 2, \dots, n, \dots, 2, 1)`-shaped weights over
the current and previous periods, each part divided by its own length). In a factor
model the low-frequency series then loads on :math:`\sum_l w_{t,l} f_{t-l}` with a
**time-varying** design row (Bańbura, Giannone & Reichlin, 2011, sec. 4; Bańbura &
Rünstler, 2011), which :class:`~nowcastbox.statespace.StateSpace` supports through
``obs_index``.

Frequency conversion
--------------------
:func:`to_lower_frequency` / :func:`month_to_quarter` and :func:`to_higher_frequency`
/ :func:`quarter_to_month` convert native series between frequencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

from nowcastbox.core.data import FrequencySpec
from nowcastbox.core.exceptions import NowcastDataError
from nowcastbox.core.frequency import (
    AggregationType,
    AggregationTypeLike,
    Frequency,
    FrequencyLike,
    aggregation_ratio,
    base_to_native,
    calendar_position,
    is_fixed_ratio,
    max_periods_per,
    native_period_bounds,
)
from nowcastbox.preprocessing._utils import PanelLike, as_panel, return_like

__all__ = [
    "CalendarAggregation",
    "HigherFrequencyMethod",
    "LowerFrequencyMethod",
    "TemporalAggregation",
    "aggregate_panel",
    "aggregation_weights",
    "calendar_aggregation",
    "calendar_weight_matrix",
    "loading_constraints",
    "mariano_murasawa_weights",
    "month_to_quarter",
    "quarter_to_month",
    "rolling_aggregate",
    "temporal_aggregation",
    "to_higher_frequency",
    "to_lower_frequency",
]

LowerFrequencyMethod = Literal["mean", "sum", "first", "middle", "last", "mariano_murasawa"]
"""Aggregation methods of :func:`to_lower_frequency` (an int position is also valid)."""

HigherFrequencyMethod = Literal["end", "start", "middle", "repeat", "divide", "linear", "spline"]
"""Disaggregation methods of :func:`to_higher_frequency`."""


# ---------------------------------------------------------------------- weights
def mariano_murasawa_weights(ratio: int = 3, *, normalize: bool = False) -> np.ndarray:
    r"""Triangular Mariano-Murasawa weights :math:`(1, 2, \dots, k, \dots, 2, 1)`.

    Parameters
    ----------
    ratio : int, default 3
        Number :math:`k` of high-frequency periods per low-frequency period.
    normalize : bool, default False
        Divide by :math:`k` (the growth-rate approximation
        :math:`\tfrac13(1,2,3,2,1)` for quarters/months).

    Returns
    -------
    numpy.ndarray
        Weights on :math:`(x_t, x_{t-1}, \dots, x_{t-2k+2})`.

    Examples
    --------
    >>> from nowcastbox.preprocessing.aggregation import mariano_murasawa_weights
    >>> mariano_murasawa_weights(3).tolist()
    [1.0, 2.0, 3.0, 2.0, 1.0]
    >>> mariano_murasawa_weights(4).tolist()
    [1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0]
    """
    return AggregationType.GROWTH_RATE.weights(ratio, normalize=normalize)


def aggregation_weights(
    high: FrequencyLike,
    low: FrequencyLike,
    aggregation: AggregationTypeLike = AggregationType.GROWTH_RATE,
    *,
    normalize: bool = False,
) -> np.ndarray:
    """Aggregation weights linking two frequencies.

    Parameters
    ----------
    high : Frequency, str or int
        High (latent) frequency, e.g. ``"M"``.
    low : Frequency, str or int
        Low (observed) frequency, e.g. ``"Q"``.
    aggregation : AggregationType or str, default "mariano_murasawa"
        ``"flow"``, ``"average"``, ``"stock"`` or ``"mariano_murasawa"``.
    normalize : bool, default False
        Normalise Mariano-Murasawa weights by the ratio.

    Returns
    -------
    numpy.ndarray
        Weights, most recent period first.

    Raises
    ------
    ValueError
        If the frequency ratio is not fixed (use :func:`calendar_weight_matrix`) or
        ``aggregation`` is unknown.

    Examples
    --------
    >>> from nowcastbox.preprocessing.aggregation import aggregation_weights
    >>> aggregation_weights("M", "Q", "average").round(4).tolist()
    [0.3333, 0.3333, 0.3333]
    >>> len(aggregation_weights("Q", "A"))
    7
    """
    return temporal_aggregation(high, low, aggregation, normalize=normalize).weights


def loading_constraints(
    weights: np.ndarray | list[float], n_factors: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    r"""Linear restrictions on loadings implied by aggregation weights.

    For loadings :math:`(\lambda_0, \dots, \lambda_{L-1})` of a low-frequency series on
    :math:`(f_t, \dots, f_{t-L+1})` (each :math:`\lambda_j` of length ``n_factors``),
    the aggregation structure :math:`\lambda_j = (w_j / w_0)\,\lambda_0` is written as
    :math:`R\,\lambda = q` with rows :math:`w_j \lambda_0 - w_0 \lambda_j = 0`,
    :math:`j = 1, \dots, L-1` (Bańbura & Modugno, 2014).

    Parameters
    ----------
    weights : array-like
        Aggregation weights with a non-zero first element.
    n_factors : int, default 1
        Number of factors :math:`r`; the scalar restriction matrix is expanded with a
        Kronecker product :math:`R \otimes I_r`.

    Returns
    -------
    R : numpy.ndarray
        Restriction matrix of shape ``((L-1) r, L r)``.
    q : numpy.ndarray
        Right-hand side (zeros) of length ``(L-1) r``.

    Raises
    ------
    ValueError
        If ``weights`` is empty, not finite, has a zero first element, or
        ``n_factors`` is not a positive integer.

    Examples
    --------
    >>> from nowcastbox.preprocessing.aggregation import loading_constraints
    >>> R, q = loading_constraints([1, 2, 3, 2, 1])
    >>> R.astype(int).tolist()[:2]
    [[2, -1, 0, 0, 0], [3, 0, -1, 0, 0]]
    >>> loading_constraints([1, 2, 3, 2, 1], n_factors=2)[0].shape
    (8, 10)
    """
    w = np.asarray(weights, dtype=float).ravel()
    if w.size == 0 or not bool(np.isfinite(w).all()):
        raise ValueError("weights must be a non-empty array of finite numbers.")
    if w[0] == 0:
        raise ValueError("The first aggregation weight must be non-zero.")
    if isinstance(n_factors, bool) or not isinstance(n_factors, int | np.integer):
        raise ValueError(f"n_factors must be a positive integer, got {n_factors!r}.")
    if n_factors < 1:
        raise ValueError(f"n_factors must be a positive integer, got {n_factors!r}.")
    n = w.size
    R = np.zeros((n - 1, n))
    R[:, 0] = w[1:]
    R[np.arange(n - 1), np.arange(1, n)] = -w[0]
    R_full = np.kron(R, np.eye(int(n_factors)))
    return R_full, np.zeros(R_full.shape[0])


@dataclass(frozen=True)
class TemporalAggregation:
    """Aggregation constraint between a high and a low frequency.

    Parameters
    ----------
    high : Frequency
        High (latent) frequency.
    low : Frequency
        Low (observed) frequency.
    aggregation : AggregationType
        Type of aggregation.
    normalize : bool, default False
        Whether Mariano-Murasawa weights are divided by the ratio.

    Attributes
    ----------
    ratio : int
        High-frequency periods per low-frequency period.
    weights : numpy.ndarray
        Weights on :math:`(x_t, x_{t-1}, ...)`.
    n_lags : int
        Number of lags in the constraint (``len(weights) - 1``).

    Examples
    --------
    >>> from nowcastbox.preprocessing.aggregation import temporal_aggregation
    >>> agg = temporal_aggregation("M", "Q")
    >>> agg.ratio, agg.n_lags, agg.weights.tolist()
    (3, 4, [1.0, 2.0, 3.0, 2.0, 1.0])
    """

    high: Frequency
    low: Frequency
    aggregation: AggregationType
    normalize: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "high", Frequency.from_value(self.high))
        object.__setattr__(self, "low", Frequency.from_value(self.low))
        object.__setattr__(self, "aggregation", AggregationType.from_value(self.aggregation))
        aggregation_ratio(self.high, self.low)  # validates the pair

    @property
    def ratio(self) -> int:
        """High-frequency periods per low-frequency period."""
        return aggregation_ratio(self.high, self.low)

    @property
    def weights(self) -> np.ndarray:
        """Aggregation weights, most recent period first."""
        return self.aggregation.weights(self.ratio, normalize=self.normalize)

    @property
    def n_lags(self) -> int:
        """Number of high-frequency lags in the constraint."""
        return len(self.weights) - 1

    def loading_constraints(self, n_factors: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """Restrictions on loadings (see :func:`loading_constraints`).

        Parameters
        ----------
        n_factors : int, default 1
            Number of factors.

        Returns
        -------
        R, q : numpy.ndarray
            Restriction matrix and right-hand side.

        Examples
        --------
        >>> from nowcastbox.preprocessing.aggregation import temporal_aggregation
        >>> temporal_aggregation("M", "Q", "stock").loading_constraints()[0].tolist()
        [[0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
        """
        return loading_constraints(self.weights, n_factors)

    def apply(self, values: Any) -> Any:
        """Apply the weights as a rolling filter (see :func:`rolling_aggregate`).

        Parameters
        ----------
        values : numpy.ndarray, pandas.Series or pandas.DataFrame
            High-frequency values.

        Returns
        -------
        same type as ``values``
            Filtered values.

        Examples
        --------
        >>> import numpy as np
        >>> from nowcastbox.preprocessing.aggregation import temporal_aggregation
        >>> temporal_aggregation("M", "Q", "flow").apply(np.arange(5.0)).tolist()
        [nan, nan, 3.0, 6.0, 9.0]
        """
        return rolling_aggregate(values, self.weights)


def temporal_aggregation(
    high: FrequencyLike,
    low: FrequencyLike,
    aggregation: AggregationTypeLike = AggregationType.GROWTH_RATE,
    *,
    normalize: bool = False,
) -> TemporalAggregation:
    """Build a :class:`TemporalAggregation` for a pair of frequencies.

    Parameters
    ----------
    high : Frequency, str or int
        High (latent) frequency.
    low : Frequency, str or int
        Low (observed) frequency.
    aggregation : AggregationType or str, default "mariano_murasawa"
        Type of aggregation.
    normalize : bool, default False
        Normalise Mariano-Murasawa weights.

    Returns
    -------
    TemporalAggregation
        The aggregation constraint.

    Raises
    ------
    ValueError
        If the ratio between the frequencies is not fixed or ``low`` is higher
        than ``high``.

    Examples
    --------
    >>> from nowcastbox.preprocessing.aggregation import temporal_aggregation
    >>> temporal_aggregation("M", "A", "average").ratio
    12
    """
    return TemporalAggregation(
        Frequency.from_value(high),
        Frequency.from_value(low),
        AggregationType.from_value(aggregation),
        normalize,
    )


@dataclass(frozen=True)
class CalendarAggregation:
    r"""Calendar-aware aggregation constraint between a high and a low frequency.

    Works for every pair of frequencies, including those whose ratio varies over the
    calendar (D -> M, W -> M, W -> Q, D -> Q, W -> A...; weeks belong to the period
    containing their last day). For the period :math:`T` ending at base period
    :math:`t` the weights are
    :meth:`~nowcastbox.core.frequency.AggregationType.calendar_weights`
    ``(n_T, n_{T-1})``, with :math:`n_T` the number of high-frequency periods of
    :math:`T`. At a base period :math:`t` that is not the last one of its period, the
    weights are those of the period *truncated at* :math:`t` (running sum, mean or
    growth rate of the period so far), so the aggregate is defined in every period.

    Parameters
    ----------
    high : Frequency
        High (latent, base) frequency.
    low : Frequency
        Low (observed) frequency (not higher than ``high``).
    aggregation : AggregationType
        Type of aggregation.

    Attributes
    ----------
    max_periods : int
        Largest number of high-frequency periods in one low-frequency period.
    n_weights : int
        Length of the weight vectors (``2 max_periods - 1`` for growth rates).
    reference_weights : numpy.ndarray
        Weights of a period of ``max_periods`` preceded by another one (the longest
        weight vector).

    Raises
    ------
    ValueError
        If ``low`` is higher than ``high``.

    Examples
    --------
    >>> import pandas as pd
    >>> from nowcastbox.preprocessing.aggregation import calendar_aggregation
    >>> agg = calendar_aggregation("W", "M", "average")
    >>> agg.max_periods, agg.n_weights
    (5, 5)
    >>> idx = pd.period_range("2020-01-06", periods=8, freq="W")
    >>> agg.weight_matrix(idx)[2].round(2).tolist()  # last week of January (4 weeks)
    [0.25, 0.25, 0.25, 0.25, 0.0]
    """

    high: Frequency
    low: Frequency
    aggregation: AggregationType

    def __post_init__(self) -> None:
        object.__setattr__(self, "high", Frequency.from_value(self.high))
        object.__setattr__(self, "low", Frequency.from_value(self.low))
        object.__setattr__(self, "aggregation", AggregationType.from_value(self.aggregation))
        max_periods_per(self.high, self.low)  # validates the pair

    @property
    def is_fixed(self) -> bool:
        """Whether the ratio between the frequencies is fixed (weights do not vary)."""
        return is_fixed_ratio(self.high, self.low)

    @property
    def max_periods(self) -> int:
        """Largest number of high-frequency periods in one low-frequency period."""
        return max_periods_per(self.high, self.low)

    @property
    def n_weights(self) -> int:
        """Length of the weight vectors (``2 max_periods - 1`` for growth rates)."""
        k = self.max_periods
        return 2 * k - 1 if self.aggregation is AggregationType.GROWTH_RATE else k

    @property
    def reference_weights(self) -> np.ndarray:
        """Weights of a ``max_periods`` period preceded by another one (longest vector)."""
        k = self.max_periods
        return self.aggregation.calendar_weights(k, k)

    def weight_matrix(self, index: pd.PeriodIndex) -> np.ndarray:
        r"""Weights of every base period of ``index`` (see the class description).

        Parameters
        ----------
        index : pandas.PeriodIndex
            Contiguous grid at frequency :attr:`high`.

        Returns
        -------
        numpy.ndarray, shape (len(index), n_weights)
            Row ``t``: weights on :math:`(x_t, x_{t-1}, \dots)` of the low-frequency
            period containing ``t`` truncated at ``t`` (zero-padded).

        Raises
        ------
        ValueError
            If ``index`` is not at frequency :attr:`high`.

        Examples
        --------
        >>> import pandas as pd
        >>> from nowcastbox.preprocessing.aggregation import calendar_aggregation
        >>> idx = pd.period_range("2020-01", periods=3, freq="M")
        >>> calendar_aggregation("M", "Q", "flow").weight_matrix(idx).tolist()
        [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]]
        """
        if Frequency.from_index(index) != self.high:
            raise ValueError(f"index must be at {self.high.label} frequency, got {index.freqstr}.")
        n_rows = len(index)
        out = np.zeros((n_rows, self.n_weights))
        if n_rows == 0:
            return out
        position, _ = calendar_position(index, self.low)
        previous = _previous_lengths(index, self.low, self.high)
        cache: dict[tuple[int, int], np.ndarray] = {}
        for t in range(n_rows):
            key = (int(position[t]) + 1, int(previous[t]))
            w = cache.get(key)
            if w is None:
                w = self.aggregation.calendar_weights(*key)
                cache[key] = w
            out[t, : w.size] = w
        return out

    def apply(self, values: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
        r"""Time-varying rolling aggregate :math:`y_t = \sum_l w_{t,l} x_{t-l}`.

        Parameters
        ----------
        values : pandas.Series or pandas.DataFrame
            High-frequency data on a contiguous :class:`pandas.PeriodIndex` at
            frequency :attr:`high`.

        Returns
        -------
        pandas.Series or pandas.DataFrame
            Aggregates (running aggregate of the current low-frequency period; NaN where
            a value with a non-zero weight is missing or before the start).

        Raises
        ------
        NowcastDataError
            If ``values`` is not indexed by a PeriodIndex.

        Examples
        --------
        >>> import pandas as pd
        >>> from nowcastbox.preprocessing.aggregation import calendar_aggregation
        >>> d = pd.Series(1.0, index=pd.period_range("2021-02-01", periods=31, freq="D"))
        >>> out = calendar_aggregation("D", "M", "flow").apply(d)
        >>> float(out["2021-02-28"]), float(out["2021-03-03"])
        (28.0, 3.0)
        """
        data = _require_native(values, "values")
        if isinstance(data, pd.DataFrame):
            return pd.DataFrame({c: self.apply(data[c]) for c in data.columns})
        index = data.index
        assert isinstance(index, pd.PeriodIndex)  # noqa: S101 - checked above
        full = pd.period_range(index[0], index[-1], freq=index.freq)
        x = data.reindex(full).to_numpy(dtype=float)
        W = self.weight_matrix(full)
        L = W.shape[1]
        padded = np.concatenate([np.full(L - 1, np.nan), x])
        lagged = np.lib.stride_tricks.sliding_window_view(padded, L)[:, ::-1]
        used = W != 0.0
        with np.errstate(invalid="ignore"):
            out = np.where(used, lagged, 0.0)
        bad = (used & np.isnan(lagged)).any(axis=1)
        result = np.where(bad, np.nan, np.einsum("tl,tl->t", W, out))
        return pd.Series(result, index=full, name=data.name)


def _previous_lengths(index: pd.PeriodIndex, low: Frequency, high: Frequency) -> np.ndarray:
    """Number of high-frequency periods of the low-frequency period before each one."""
    previous = pd.PeriodIndex(base_to_native(index, low) - 1)
    first, last = native_period_bounds(previous, high)
    return np.asarray(_ordinals(last) - _ordinals(first) + 1, dtype=np.int64)


def calendar_aggregation(
    high: FrequencyLike,
    low: FrequencyLike,
    aggregation: AggregationTypeLike = AggregationType.GROWTH_RATE,
) -> CalendarAggregation:
    """Build a :class:`CalendarAggregation` for any pair of frequencies.

    Parameters
    ----------
    high : Frequency, str or int
        High (latent) frequency, e.g. ``"W"``.
    low : Frequency, str or int
        Low (observed) frequency, e.g. ``"Q"``.
    aggregation : AggregationType or str, default "mariano_murasawa"
        ``"flow"``, ``"average"``, ``"stock"`` or ``"mariano_murasawa"``.

    Returns
    -------
    CalendarAggregation
        The calendar-aware aggregation constraint.

    Raises
    ------
    ValueError
        If ``low`` is higher than ``high`` or ``aggregation`` is unknown.

    Examples
    --------
    >>> from nowcastbox.preprocessing.aggregation import calendar_aggregation
    >>> calendar_aggregation("W", "Q").n_weights
    27
    """
    return CalendarAggregation(
        Frequency.from_value(high),
        Frequency.from_value(low),
        AggregationType.from_value(aggregation),
    )


def calendar_weight_matrix(
    index: pd.PeriodIndex,
    low: FrequencyLike,
    aggregation: AggregationTypeLike = AggregationType.GROWTH_RATE,
) -> np.ndarray:
    """Calendar-aware aggregation weights of every period of a base grid.

    Shortcut for ``calendar_aggregation(index_frequency, low, aggregation)
    .weight_matrix(index)``; the rows of the storage slots of ``low`` (see
    :func:`~nowcastbox.core.frequency.is_period_end`) are the weights of complete
    low-frequency periods.

    Parameters
    ----------
    index : pandas.PeriodIndex
        Base grid (e.g. weekly).
    low : Frequency, str or int
        Low frequency.
    aggregation : AggregationType or str, default "mariano_murasawa"
        Type of aggregation.

    Returns
    -------
    numpy.ndarray, shape (len(index), n_weights)
        Weights, most recent period first, zero-padded.

    Raises
    ------
    ValueError
        If ``low`` is higher than the grid frequency.

    Examples
    --------
    >>> import pandas as pd
    >>> from nowcastbox.preprocessing.aggregation import calendar_weight_matrix
    >>> idx = pd.period_range("2021-01-25", periods=5, freq="W")  # last week ends Feb 28
    >>> calendar_weight_matrix(idx, "M", "flow")[-1].tolist()
    [1.0, 1.0, 1.0, 1.0, 0.0]
    """
    return calendar_aggregation(Frequency.from_index(index), low, aggregation).weight_matrix(index)


def rolling_aggregate(values: Any, weights: np.ndarray | list[float]) -> Any:
    r"""Rolling weighted sum :math:`y_t = \sum_j w_j x_{t-j}`.

    Parameters
    ----------
    values : numpy.ndarray, pandas.Series or pandas.DataFrame
        Values on a contiguous high-frequency grid (columns filtered separately).
    weights : array-like
        Weights, most recent period first.

    Returns
    -------
    same type as ``values``
        Filtered values; NaN for the first periods (as many as the position of the
        last non-zero weight) and wherever a value with a non-zero weight is missing.

    Raises
    ------
    ValueError
        If ``weights`` is empty, not finite or all zero.

    Examples
    --------
    >>> import numpy as np
    >>> from nowcastbox.preprocessing.aggregation import rolling_aggregate
    >>> rolling_aggregate(np.ones(6), [1, 2, 3, 2, 1]).tolist()
    [nan, nan, nan, nan, 9.0, 9.0]
    """
    w = np.asarray(weights, dtype=float).ravel()
    if w.size == 0 or not bool(np.isfinite(w).all()):
        raise ValueError("weights must be a non-empty array of finite numbers.")
    if not w.any():
        raise ValueError("At least one weight must be non-zero.")
    if isinstance(values, pd.DataFrame):
        return values.apply(lambda col: rolling_aggregate(col, w))
    if isinstance(values, pd.Series):
        out = rolling_aggregate(values.to_numpy(dtype=float), w)
        return pd.Series(out, index=values.index, name=values.name)
    x = np.asarray(values, dtype=float)
    if x.ndim == 2:
        return np.column_stack([rolling_aggregate(x[:, j], w) for j in range(x.shape[1])])
    w = w[: int(np.flatnonzero(w)[-1]) + 1]  # trailing zero weights need no data
    used = w != 0
    n, L = x.size, w.size
    out = np.full(n, np.nan)
    if n >= L:
        windows = np.lib.stride_tricks.sliding_window_view(x, L)[:, ::-1][:, used]
        out[L - 1 :] = windows @ w[used]
    return out


# ---------------------------------------------------------------------- panels
def aggregate_panel(
    data: PanelLike,
    aggregation: AggregationTypeLike | None = None,
    *,
    low_frequency: FrequencyLike | None = None,
    normalize: bool = True,
    columns: list[str] | None = None,
    frequency: FrequencySpec | None = None,
) -> PanelLike:
    r"""Filter the base-frequency series with low-frequency aggregation weights.

    Each base-frequency series :math:`x_t` is replaced by
    :math:`\sum_j w_j x_{t-j}`, making it comparable with the lower-frequency series
    of the panel (e.g. monthly growth rates turned into quarterly growth rates with
    the Mariano-Murasawa weights). Lower-frequency series are left unchanged.

    Parameters
    ----------
    data : pandas.DataFrame or MixedFrequencyData
        Panel on a base grid.
    aggregation : AggregationType or str, optional
        Aggregation for every filtered series. When omitted, each series uses its
        ``aggregation`` metadata, defaulting to Mariano-Murasawa.
    low_frequency : Frequency, str or int, optional
        Target low frequency; defaults to the lowest frequency in the panel.
    normalize : bool, default True
        Normalise Mariano-Murasawa weights by the ratio (:math:`\tfrac13(1,2,3,2,1)`).
        Pairs whose ratio varies over the calendar (e.g. weekly series and a quarterly
        target) always use the exact weights of :class:`CalendarAggregation`.
    columns : list of str, optional
        Base-frequency series to filter (default: all of them).
    frequency : frequency specification, optional
        Per-series frequencies for DataFrame input.

    Returns
    -------
    pandas.DataFrame or MixedFrequencyData
        Filtered panel of the input type.

    Raises
    ------
    ValueError
        If no lower frequency is available or the ratio is not fixed.
    NowcastDataError
        If ``columns`` contains unknown or non-base-frequency series.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from nowcastbox.preprocessing.aggregation import aggregate_panel
    >>> idx = pd.period_range("2020-01", periods=6, freq="M")
    >>> df = pd.DataFrame({"m": np.ones(6), "q": [np.nan, np.nan, 1, np.nan, np.nan, 2]}, index=idx)
    >>> aggregate_panel(df, frequency={"m": "M", "q": "Q"})["m"].tolist()
    [nan, nan, nan, nan, 3.0, 3.0]
    """
    panel, was_frame = as_panel(data, frequency)
    base = panel.base_frequency
    if low_frequency is None:
        lows = [m.frequency for m in panel.metadata.values() if m.frequency.is_lower_than(base)]
        if not lows:
            raise ValueError("The panel has a single frequency; pass low_frequency.")
        low = min(lows, key=lambda f: f.periods_per_year)
    else:
        low = Frequency.from_value(low_frequency)
    base_cols = panel.columns_with_frequency(base)
    cols = base_cols if columns is None else list(columns)
    bad = sorted(set(cols) - set(base_cols))
    if bad:
        raise NowcastDataError(f"Only {base.label} series can be aggregated; got {bad}.")
    frame = panel.to_frame()
    for c in cols:
        agg = aggregation if aggregation is not None else panel.metadata[c].aggregation
        kind = AggregationType.GROWTH_RATE if agg is None else agg
        if is_fixed_ratio(base, low):
            spec = temporal_aggregation(base, low, kind, normalize=normalize)
            frame[c] = spec.apply(frame[c].to_numpy(dtype=float))
        else:
            frame[c] = calendar_aggregation(base, low, kind).apply(frame[c])
    return return_like(panel.with_data(frame), was_frame)


# ---------------------------------------------------------------------- conversions
def _require_native(x: Any, what: str = "x") -> pd.Series | pd.DataFrame:
    if not isinstance(x, pd.Series | pd.DataFrame):
        raise NowcastDataError(f"{what} must be a pandas Series or DataFrame.")
    if not isinstance(x.index, pd.PeriodIndex):
        raise NowcastDataError(f"{what} must be indexed by a pandas.PeriodIndex.")
    if x.index.has_duplicates:
        raise NowcastDataError(f"{what} has duplicated periods.")
    if len(x.index) == 0:
        raise NowcastDataError(f"{what} is empty.")
    return x.sort_index().astype(float)


def _ordinals(index: pd.PeriodIndex) -> np.ndarray:
    return np.asarray(index.asi8, dtype=np.int64)  # pyright: ignore[reportAttributeAccessIssue]


def _positions(index: pd.PeriodIndex, low: Frequency, high: Frequency) -> tuple[Any, Any]:
    """Low period of each high period, 0-based position inside it and its length.

    Calendar aware: a week belongs to the period containing its last day.
    """
    del high  # the high frequency is the frequency of ``index``
    return base_to_native(index, low), calendar_position(index, low)


def _check_how_lower(how: Any) -> None:
    valid = ("mean", "sum", "first", "middle", "last", "mariano_murasawa")
    if isinstance(how, bool) or not (
        how in valid or (isinstance(how, int | np.integer) and how >= 1)
    ):
        raise ValueError(f"how must be one of {valid} or a positive integer, got {how!r}.")


def _lower_series(
    s: pd.Series, low: Frequency, high: Frequency, how: Any, complete: bool
) -> pd.Series:
    full = pd.period_range(s.index[0], s.index[-1], freq=high.pandas_freq)
    s = s.reindex(full)
    low_idx, (pos, length) = _positions(full, low, high)
    out_index = pd.period_range(low_idx[0], low_idx[-1], freq=low.pandas_freq)
    frame = pd.DataFrame({"x": s.to_numpy(), "low": low_idx, "pos": pos, "len": length})
    if how == "mariano_murasawa":
        if is_fixed_ratio(high, low):
            frame["x"] = rolling_aggregate(
                s.to_numpy(), mariano_murasawa_weights(aggregation_ratio(high, low), normalize=True)
            )
        else:
            frame["x"] = calendar_aggregation(high, low).apply(s).to_numpy()
        how = "last"
    groups = frame.groupby("low", sort=True)
    if how in ("mean", "sum"):
        agg = groups["x"].agg(how)
        n_obs = groups["x"].count()
        expected = groups["len"].first()
        has_data = n_obs > 0
        keep = (n_obs == expected) if complete else has_data
        result = agg.where(keep & has_data)
    else:
        if how == "first":
            target = np.zeros_like(pos)
        elif how == "middle":
            target = (length - 1) // 2
        elif how == "last":
            target = length - 1
        else:
            if int(how) > int(length.max()):
                raise ValueError(
                    f"Position {how} exceeds the {int(length.max())} {high.label} periods "
                    f"of a {low.label} period."
                )
            target = np.full_like(pos, int(how) - 1)
        picked = frame.loc[pos == target]
        result = picked.set_index("low")["x"]
    out = pd.Series(result, dtype=float).reindex(out_index)
    out.index.name = s.index.name
    out.name = s.name
    return out


def to_lower_frequency(
    x: pd.Series | pd.DataFrame,
    frequency: FrequencyLike,
    how: LowerFrequencyMethod | int = "mean",
    *,
    complete: bool = True,
) -> pd.Series | pd.DataFrame:
    r"""Convert a native high-frequency series to a lower frequency.

    Parameters
    ----------
    x : pandas.Series or pandas.DataFrame
        Data on a native :class:`pandas.PeriodIndex` (e.g. monthly).
    frequency : Frequency, str or int
        Target lower frequency (e.g. ``"Q"``).
    how : {"mean", "sum", "first", "middle", "last", "mariano_murasawa"} or int
        Average, sum, the first/middle/last high-frequency value of each period, the
        value at a 1-based position (e.g. ``2`` = second month of the quarter), or
        the normalised Mariano-Murasawa combination
        :math:`\tfrac1k(1, 2, \dots, k, \dots, 2, 1)` of the current and previous
        period (for growth rates; requires a fixed frequency ratio).
    complete : bool, default True
        For ``"mean"``/``"sum"``: return NaN for periods with a missing
        high-frequency value (including incomplete periods at the sample borders).
        With False, the available values are aggregated.

    Returns
    -------
    pandas.Series or pandas.DataFrame
        Data on the lower-frequency :class:`pandas.PeriodIndex`.

    Raises
    ------
    NowcastDataError
        If ``x`` is not indexed by a PeriodIndex.
    ValueError
        If ``frequency`` is not lower than the data frequency or ``how`` is invalid.

    Examples
    --------
    >>> import pandas as pd
    >>> from nowcastbox.preprocessing.aggregation import to_lower_frequency
    >>> m = pd.Series(range(1, 7), index=pd.period_range("2020-01", periods=6, freq="M"))
    >>> to_lower_frequency(m, "Q", "sum").tolist()
    [6.0, 15.0]
    >>> to_lower_frequency(m, "Q", 2).tolist()
    [2.0, 5.0]
    """
    data = _require_native(x)
    _check_how_lower(how)
    high = Frequency.from_index(data.index)  # type: ignore[arg-type]
    low = Frequency.from_value(frequency)
    if not low.is_lower_than(high):
        raise ValueError(f"{low.label} is not lower than the {high.label} data frequency.")
    if isinstance(data, pd.DataFrame):
        return pd.DataFrame(
            {c: _lower_series(data[c], low, high, how, complete) for c in data.columns}
        )
    return _lower_series(data, low, high, how, complete)


def month_to_quarter(
    x: pd.Series | pd.DataFrame,
    how: LowerFrequencyMethod | int = "mean",
    *,
    complete: bool = True,
) -> pd.Series | pd.DataFrame:
    """Convert monthly data to quarterly (see :func:`to_lower_frequency`).

    Parameters
    ----------
    x : pandas.Series or pandas.DataFrame
        Monthly data (``PeriodIndex`` with ``freq="M"``).
    how : {"mean", "sum", "first", "middle", "last", "mariano_murasawa"} or int
        Aggregation; an integer 1-3 selects that month of the quarter.
    complete : bool, default True
        Require the three months for ``"mean"``/``"sum"``.

    Returns
    -------
    pandas.Series or pandas.DataFrame
        Quarterly data.

    Raises
    ------
    NowcastDataError
        If ``x`` is not monthly.

    Examples
    --------
    >>> import pandas as pd
    >>> from nowcastbox.preprocessing.aggregation import month_to_quarter
    >>> m = pd.Series([1.0, 2, 3, 4, 5, 6], index=pd.period_range("2020-01", periods=6, freq="M"))
    >>> month_to_quarter(m).tolist()
    [2.0, 5.0]
    >>> month_to_quarter(m, "last").index.astype(str).tolist()
    ['2020Q1', '2020Q2']
    """
    data = _require_native(x)
    if Frequency.from_index(data.index) != Frequency.MONTHLY:  # type: ignore[arg-type]
        raise NowcastDataError("month_to_quarter needs monthly data (PeriodIndex freq 'M').")
    return to_lower_frequency(data, Frequency.QUARTERLY, how, complete=complete)


def _check_how_higher(how: str) -> None:
    valid = ("end", "start", "middle", "repeat", "divide", "linear", "spline")
    if how not in valid:
        raise ValueError(f"how must be one of {valid}, got {how!r}.")


def _higher_series(s: pd.Series, high: Frequency, low: Frequency, how: str) -> pd.Series:
    native = s.reindex(pd.period_range(s.index[0], s.index[-1], freq=low.pandas_freq))
    first, last = native_period_bounds(native.index, high)  # type: ignore[arg-type]
    out_index = pd.period_range(first[0], last[-1], freq=high.pandas_freq)
    low_idx, (pos, length) = _positions(out_index, low, high)
    values = native.reindex(low_idx).to_numpy(dtype=float)
    anchor = {"start": pos == 0, "middle": pos == (length - 1) // 2}.get(how, pos == length - 1)
    if how == "repeat":
        result = values
    elif how == "divide":
        result = values / length
    else:
        result = np.where(anchor, values, np.nan)
        if how in ("linear", "spline"):
            result = _interpolate(result, how)
    return pd.Series(result, index=out_index, name=s.name)


def _interpolate(y: np.ndarray, how: str) -> np.ndarray:
    obs = ~np.isnan(y)
    pos = np.arange(y.size)
    if int(obs.sum()) < 2:
        return y
    first, last = pos[obs][0], pos[obs][-1]
    inside = (pos > first) & (pos < last) & ~obs
    out = y.copy()
    if how == "linear":
        out[inside] = np.interp(pos[inside], pos[obs], y[obs])
    else:
        out[inside] = CubicSpline(pos[obs], y[obs])(pos[inside])
    return out


def to_higher_frequency(
    x: pd.Series | pd.DataFrame,
    frequency: FrequencyLike,
    how: HigherFrequencyMethod = "end",
) -> pd.Series | pd.DataFrame:
    """Convert a native low-frequency series to a higher frequency.

    Parameters
    ----------
    x : pandas.Series or pandas.DataFrame
        Data on a native :class:`pandas.PeriodIndex` (e.g. quarterly).
    frequency : Frequency, str or int
        Target higher frequency (e.g. ``"M"``).
    how : {"end", "start", "middle", "repeat", "divide", "linear", "spline"}
        ``"end"`` places each value in the last high-frequency period (the storage
        convention of :class:`~nowcastbox.core.data.MixedFrequencyData`), ``"start"``
        / ``"middle"`` in the first / middle one (other periods NaN); ``"repeat"``
        copies the value to every period; ``"divide"`` spreads a flow evenly;
        ``"linear"`` / ``"spline"`` interpolate between the end-of-period values
        (no extrapolation beyond the first and last values).

    Returns
    -------
    pandas.Series or pandas.DataFrame
        Data on the higher-frequency :class:`pandas.PeriodIndex`.

    Raises
    ------
    NowcastDataError
        If ``x`` is not indexed by a PeriodIndex.
    ValueError
        If ``frequency`` is not higher than the data frequency or ``how`` is invalid.

    Examples
    --------
    >>> import pandas as pd
    >>> from nowcastbox.preprocessing.aggregation import to_higher_frequency
    >>> q = pd.Series([3.0, 6.0], index=pd.period_range("2020Q1", periods=2, freq="Q"))
    >>> to_higher_frequency(q, "M", "divide").tolist()
    [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]
    """
    data = _require_native(x)
    _check_how_higher(how)
    low = Frequency.from_index(data.index)  # type: ignore[arg-type]
    high = Frequency.from_value(frequency)
    if not high.is_higher_than(low):
        raise ValueError(f"{high.label} is not higher than the {low.label} data frequency.")
    if isinstance(data, pd.DataFrame):
        return pd.DataFrame({c: _higher_series(data[c], high, low, how) for c in data.columns})
    return _higher_series(data, high, low, how)


def quarter_to_month(
    x: pd.Series | pd.DataFrame, how: HigherFrequencyMethod = "end"
) -> pd.Series | pd.DataFrame:
    """Convert quarterly data to monthly (see :func:`to_higher_frequency`).

    Parameters
    ----------
    x : pandas.Series or pandas.DataFrame
        Quarterly data (``PeriodIndex`` with ``freq="Q"``).
    how : {"end", "start", "middle", "repeat", "divide", "linear", "spline"}
        Disaggregation method; ``"end"`` (default) gives the storage layout of
        :class:`~nowcastbox.core.data.MixedFrequencyData`.

    Returns
    -------
    pandas.Series or pandas.DataFrame
        Monthly data.

    Raises
    ------
    NowcastDataError
        If ``x`` is not quarterly.

    Examples
    --------
    >>> import pandas as pd
    >>> from nowcastbox.preprocessing.aggregation import quarter_to_month
    >>> q = pd.Series([3.0, 6.0], index=pd.period_range("2020Q1", periods=2, freq="Q"))
    >>> quarter_to_month(q).tolist()
    [nan, nan, 3.0, nan, nan, 6.0]
    >>> quarter_to_month(q, "linear").tolist()
    [nan, nan, 3.0, 4.0, 5.0, 6.0]
    """
    data = _require_native(x)
    if Frequency.from_index(data.index) != Frequency.QUARTERLY:  # type: ignore[arg-type]
        raise NowcastDataError("quarter_to_month needs quarterly data (PeriodIndex freq 'Q').")
    return to_higher_frequency(data, Frequency.MONTHLY, how)
