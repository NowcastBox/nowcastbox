"""Observation frequencies and temporal aggregation types.

This module is the single source of truth for everything calendar-related in
nowcastbox:

* :class:`Frequency` - the frequencies a series can have (daily, weekly, monthly,
  quarterly, annual), with periods per year and pandas period aliases.
* :class:`AggregationType` - how a low-frequency observation relates to the latent
  high-frequency series (flow, stock, average, Mariano-Murasawa growth rate), with the
  corresponding aggregation weights.
* Helpers to move between a series' *native* period grid (e.g. ``2020Q1``) and the
  *base* grid on which :class:`~nowcastbox.core.data.MixedFrequencyData` stores every
  series (e.g. monthly: ``2020-03``).

Storage convention
------------------
A low-frequency observation is stored on the **last base period** of its native
period: with a monthly base grid, ``2020Q1`` is stored in ``2020-03`` and the year 2020
in ``2020-12``. All other base periods of that series are structurally empty (NaN).

Calendar convention (plan innovation I1)
----------------------------------------
Pairs of frequencies whose ratio varies over the calendar (days per month, weeks per
quarter...) are handled with calendar-aware helpers (:func:`native_period_bounds`,
:func:`calendar_position`, :func:`max_periods_per`). A base period belongs to the
native period that contains its **last day** (the pandas ``asfreq`` convention): the
week ``2020-01-27/2020-02-02`` belongs to February 2020 and to 2020Q1. Weeks are
pandas ``"W"`` (``W-SUN``) periods. The storage slot of a native period is the last
base period that belongs to it (e.g. the last week ending in the month).

The design is frequency-agnostic: nothing here hard-codes the monthly/quarterly pair.
"""

from __future__ import annotations

from enum import Enum
from numbers import Integral, Real
from typing import TypeAlias

import numpy as np
import pandas as pd

__all__ = [
    "AggregationType",
    "AggregationTypeLike",
    "Frequency",
    "FrequencyLike",
    "aggregation_ratio",
    "base_to_native",
    "calendar_position",
    "infer_frequency",
    "is_fixed_ratio",
    "is_period_end",
    "max_periods_per",
    "native_period_bounds",
    "native_to_base",
    "period_to_base",
]


class Frequency(str, Enum):
    """Observation frequency of a time series.

    Members compare equal to their one-letter code (``Frequency.MONTHLY == "M"``)
    and can be built from many spellings with :meth:`from_value`.

    Attributes
    ----------
    DAILY, WEEKLY, MONTHLY, QUARTERLY, ANNUAL
        Supported frequencies, ordered from highest to lowest.

    Examples
    --------
    >>> from nowcastbox.core.frequency import Frequency
    >>> Frequency.from_value("quarterly")
    <Frequency.QUARTERLY: 'Q'>
    >>> Frequency.from_value(12)
    <Frequency.MONTHLY: 'M'>
    >>> Frequency.QUARTERLY.periods_per_year
    4
    >>> Frequency.QUARTERLY.is_lower_than(Frequency.MONTHLY)
    True
    """

    DAILY = "D"
    WEEKLY = "W"
    MONTHLY = "M"
    QUARTERLY = "Q"
    ANNUAL = "A"

    @property
    def periods_per_year(self) -> int:
        """Nominal number of periods in one year (365, 52, 12, 4 or 1)."""
        return _PERIODS_PER_YEAR[self]

    @property
    def pandas_freq(self) -> str:
        """Pandas *period* alias (``"D"``, ``"W"``, ``"M"``, ``"Q"``, ``"Y"``)."""
        return _PANDAS_FREQ[self]

    @property
    def label(self) -> str:
        """Human-readable lower-case name, e.g. ``"quarterly"``."""
        return self.name.lower()

    def is_lower_than(self, other: FrequencyLike) -> bool:
        """Return True if this frequency is strictly lower (coarser) than ``other``.

        Parameters
        ----------
        other : Frequency, str or int
            Frequency to compare with (anything accepted by :meth:`from_value`).

        Returns
        -------
        bool
            Whether ``self`` has fewer periods per year than ``other``.

        Examples
        --------
        >>> Frequency.ANNUAL.is_lower_than("M")
        True
        """
        return self.periods_per_year < Frequency.from_value(other).periods_per_year

    def is_higher_than(self, other: FrequencyLike) -> bool:
        """Return True if this frequency is strictly higher (finer) than ``other``.

        Parameters
        ----------
        other : Frequency, str or int
            Frequency to compare with.

        Returns
        -------
        bool
            Whether ``self`` has more periods per year than ``other``.

        Examples
        --------
        >>> Frequency.MONTHLY.is_higher_than("Q")
        True
        """
        return self.periods_per_year > Frequency.from_value(other).periods_per_year

    @classmethod
    def from_value(cls, value: FrequencyLike) -> Frequency:
        """Build a :class:`Frequency` from a flexible specification.

        Parameters
        ----------
        value : Frequency, str or int
            * a :class:`Frequency` member (returned unchanged);
            * a string, case-insensitive: one-letter codes (``"D"``, ``"W"``, ``"M"``,
              ``"Q"``, ``"A"``/``"Y"``), names (``"monthly"``, ``"quarter"``,
              ``"annual"``...), or pandas aliases (``"Q-DEC"``, ``"W-SUN"``, ``"YE"``...);
            * an integer number of periods per year (365, 52, 12, 4, 1; 260 and 252
              are also read as daily), the convention of R ``ts`` objects.

        Returns
        -------
        Frequency
            The parsed frequency.

        Raises
        ------
        ValueError
            If ``value`` cannot be interpreted as a frequency.

        Examples
        --------
        >>> Frequency.from_value("Q-DEC")
        <Frequency.QUARTERLY: 'Q'>
        >>> Frequency.from_value(4)
        <Frequency.QUARTERLY: 'Q'>
        """
        if isinstance(value, Frequency):
            return value
        if isinstance(value, str):
            return _parse_frequency_string(value)
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"Cannot interpret {value!r} as a frequency.")
        if not isinstance(value, Integral) and not float(value).is_integer():
            raise ValueError(f"Cannot interpret {value!r} as a frequency.")
        n = int(value)
        if n not in _FROM_PERIODS_PER_YEAR:
            raise ValueError(
                f"Unsupported number of periods per year {n}; "
                f"expected one of {sorted(_FROM_PERIODS_PER_YEAR)}."
            )
        return _FROM_PERIODS_PER_YEAR[n]

    @classmethod
    def from_index(cls, index: pd.PeriodIndex) -> Frequency:
        """Return the frequency of a :class:`pandas.PeriodIndex`.

        Parameters
        ----------
        index : pandas.PeriodIndex
            Period index.

        Returns
        -------
        Frequency
            Frequency matching ``index.freqstr``.

        Raises
        ------
        TypeError
            If ``index`` is not a :class:`pandas.PeriodIndex`.
        ValueError
            If the index frequency is not supported.

        Examples
        --------
        >>> import pandas as pd
        >>> Frequency.from_index(pd.period_range("2020Q1", periods=4, freq="Q"))
        <Frequency.QUARTERLY: 'Q'>
        """
        if not isinstance(index, pd.PeriodIndex):
            raise TypeError(f"Expected a pandas.PeriodIndex, got {type(index).__name__}.")
        return _parse_frequency_string(index.freqstr)


FrequencyLike: TypeAlias = Frequency | str | int

_PERIODS_PER_YEAR: dict[Frequency, int] = {
    Frequency.DAILY: 365,
    Frequency.WEEKLY: 52,
    Frequency.MONTHLY: 12,
    Frequency.QUARTERLY: 4,
    Frequency.ANNUAL: 1,
}

_PANDAS_FREQ: dict[Frequency, str] = {
    Frequency.DAILY: "D",
    Frequency.WEEKLY: "W",
    Frequency.MONTHLY: "M",
    Frequency.QUARTERLY: "Q",
    Frequency.ANNUAL: "Y",
}

_FROM_PERIODS_PER_YEAR: dict[int, Frequency] = {
    365: Frequency.DAILY,
    366: Frequency.DAILY,
    260: Frequency.DAILY,
    252: Frequency.DAILY,
    52: Frequency.WEEKLY,
    12: Frequency.MONTHLY,
    4: Frequency.QUARTERLY,
    1: Frequency.ANNUAL,
}

_STRING_ALIASES: dict[str, Frequency] = {
    "d": Frequency.DAILY,
    "b": Frequency.DAILY,
    "day": Frequency.DAILY,
    "daily": Frequency.DAILY,
    "w": Frequency.WEEKLY,
    "week": Frequency.WEEKLY,
    "weekly": Frequency.WEEKLY,
    "m": Frequency.MONTHLY,
    "me": Frequency.MONTHLY,
    "ms": Frequency.MONTHLY,
    "month": Frequency.MONTHLY,
    "monthly": Frequency.MONTHLY,
    "q": Frequency.QUARTERLY,
    "qe": Frequency.QUARTERLY,
    "qs": Frequency.QUARTERLY,
    "quarter": Frequency.QUARTERLY,
    "quarterly": Frequency.QUARTERLY,
    "a": Frequency.ANNUAL,
    "y": Frequency.ANNUAL,
    "ye": Frequency.ANNUAL,
    "ys": Frequency.ANNUAL,
    "as": Frequency.ANNUAL,
    "year": Frequency.ANNUAL,
    "yearly": Frequency.ANNUAL,
    "annual": Frequency.ANNUAL,
    "annually": Frequency.ANNUAL,
}


def _parse_frequency_string(value: str) -> Frequency:
    key = value.strip().lower()
    if key in _STRING_ALIASES:
        return _STRING_ALIASES[key]
    # pandas anchored aliases such as "Q-DEC", "W-SUN", "A-DEC", "Y-DEC", "QE-DEC"
    head = key.split("-", 1)[0]
    if "-" in key and head in _STRING_ALIASES:
        return _STRING_ALIASES[head]
    raise ValueError(f"Cannot interpret {value!r} as a frequency.")


_FIXED_RATIOS: dict[tuple[Frequency, Frequency], int] = {
    (Frequency.DAILY, Frequency.WEEKLY): 7,
    (Frequency.MONTHLY, Frequency.QUARTERLY): 3,
    (Frequency.MONTHLY, Frequency.ANNUAL): 12,
    (Frequency.QUARTERLY, Frequency.ANNUAL): 4,
}

# maximum number of high-frequency periods per low-frequency period for the pairs whose
# ratio varies over the calendar (weeks are assigned to the period of their last day)
_MAX_VARIABLE_RATIOS: dict[tuple[Frequency, Frequency], int] = {
    (Frequency.DAILY, Frequency.MONTHLY): 31,
    (Frequency.DAILY, Frequency.QUARTERLY): 92,
    (Frequency.DAILY, Frequency.ANNUAL): 366,
    (Frequency.WEEKLY, Frequency.MONTHLY): 5,
    (Frequency.WEEKLY, Frequency.QUARTERLY): 14,
    (Frequency.WEEKLY, Frequency.ANNUAL): 53,
}


def aggregation_ratio(high: FrequencyLike, low: FrequencyLike) -> int:
    """Number of ``high``-frequency periods in one ``low``-frequency period.

    Only pairs with a *fixed* ratio are supported (D/W = 7, M/Q = 3, M/A = 12, Q/A = 4
    and any frequency with itself = 1). Pairs whose ratio varies over the calendar
    (e.g. days per month, weeks per quarter) raise, because they need calendar-aware
    aggregation weights (:func:`calendar_position`, :func:`max_periods_per`).

    Parameters
    ----------
    high : Frequency, str or int
        Higher (finer) frequency.
    low : Frequency, str or int
        Lower (coarser) frequency.

    Returns
    -------
    int
        The fixed ratio.

    Raises
    ------
    ValueError
        If ``low`` is higher than ``high`` or the ratio is not fixed.

    Examples
    --------
    >>> from nowcastbox.core.frequency import aggregation_ratio
    >>> aggregation_ratio("M", "Q")
    3
    >>> aggregation_ratio("Q", "Q")
    1
    """
    high_f = Frequency.from_value(high)
    low_f = Frequency.from_value(low)
    if high_f == low_f:
        return 1
    if low_f.is_higher_than(high_f):
        raise ValueError(f"{low_f.label} is higher than {high_f.label}; swap the arguments.")
    try:
        return _FIXED_RATIOS[(high_f, low_f)]
    except KeyError:
        raise ValueError(
            f"The number of {high_f.label} periods per {low_f.label} period is not fixed; "
            "calendar-aware aggregation is required."
        ) from None


def is_fixed_ratio(high: FrequencyLike, low: FrequencyLike) -> bool:
    """Whether ``low`` periods always contain the same number of ``high`` periods.

    Parameters
    ----------
    high : Frequency, str or int
        Higher (finer) frequency.
    low : Frequency, str or int
        Lower (coarser) frequency.

    Returns
    -------
    bool
        True for equal frequencies and fixed-ratio pairs (D/W, M/Q, M/A, Q/A).

    Raises
    ------
    ValueError
        If ``low`` is higher than ``high``.

    Examples
    --------
    >>> from nowcastbox.core.frequency import is_fixed_ratio
    >>> is_fixed_ratio("M", "Q"), is_fixed_ratio("W", "Q")
    (True, False)
    """
    high_f = Frequency.from_value(high)
    low_f = Frequency.from_value(low)
    if low_f.is_higher_than(high_f):
        raise ValueError(f"{low_f.label} is higher than {high_f.label}; swap the arguments.")
    return high_f == low_f or (high_f, low_f) in _FIXED_RATIOS


def max_periods_per(high: FrequencyLike, low: FrequencyLike) -> int:
    """Largest possible number of ``high`` periods in one ``low`` period.

    Equal to :func:`aggregation_ratio` for fixed-ratio pairs; for calendar pairs the
    maximum over the calendar (31 days per month, 5 weeks per month, 14 weeks per
    quarter, 53 weeks per year, 92 days per quarter, 366 days per year), with weeks
    assigned to the period containing their last day.

    Parameters
    ----------
    high : Frequency, str or int
        Higher (finer) frequency.
    low : Frequency, str or int
        Lower (coarser) frequency.

    Returns
    -------
    int
        Maximum number of high-frequency periods.

    Raises
    ------
    ValueError
        If ``low`` is higher than ``high``.

    Examples
    --------
    >>> from nowcastbox.core.frequency import max_periods_per
    >>> max_periods_per("W", "Q"), max_periods_per("M", "Q")
    (14, 3)
    """
    high_f = Frequency.from_value(high)
    low_f = Frequency.from_value(low)
    if is_fixed_ratio(high_f, low_f):
        return aggregation_ratio(high_f, low_f)
    return _MAX_VARIABLE_RATIOS[(high_f, low_f)]


def native_period_bounds(
    native: pd.PeriodIndex, base: FrequencyLike
) -> tuple[pd.PeriodIndex, pd.PeriodIndex]:
    """First and last base periods belonging to each native period (calendar aware).

    A base period belongs to the native period containing its last day, so a week
    straddling two months belongs to the later one. The last base period is the
    storage slot of the native period.

    Parameters
    ----------
    native : pandas.PeriodIndex
        Native periods (e.g. quarters).
    base : Frequency, str or int
        Base frequency (at least as fine as ``native``).

    Returns
    -------
    first, last : pandas.PeriodIndex
        Base periods (same length as ``native``).

    Raises
    ------
    ValueError
        If the base frequency is lower than the native frequency.

    Examples
    --------
    >>> import pandas as pd
    >>> from nowcastbox.core.frequency import native_period_bounds
    >>> first, last = native_period_bounds(pd.PeriodIndex(["2020-02"], freq="M"), "W")
    >>> str(first[0]), str(last[0])
    ('2020-01-27/2020-02-02', '2020-02-17/2020-02-23')
    """
    base_f = Frequency.from_value(base)
    own = Frequency.from_index(native)
    if base_f.is_lower_than(own):
        raise ValueError(f"Cannot map a {own.label} index onto a {base_f.label} base grid.")
    freq = base_f.pandas_freq
    first = pd.PeriodIndex(native.to_timestamp(how="start").to_period(freq), freq=freq)
    last = pd.PeriodIndex(native.to_timestamp(how="end").to_period(freq), freq=freq)
    # a base period belongs to the native period containing its last day
    own_freq = native.freqstr
    first_owner = first.asfreq(own_freq)
    last_owner = last.asfreq(own_freq)
    first = first + np.asarray(first_owner < native, dtype=np.int64)
    last = last - np.asarray(last_owner > native, dtype=np.int64)
    return pd.PeriodIndex(first, freq=freq), pd.PeriodIndex(last, freq=freq)


def calendar_position(index: pd.PeriodIndex, freq: FrequencyLike) -> tuple[np.ndarray, np.ndarray]:
    """Position of every base period inside its native period and the period length.

    Parameters
    ----------
    index : pandas.PeriodIndex
        Base-grid index.
    freq : Frequency, str or int
        Native (lower or equal) frequency.

    Returns
    -------
    position : numpy.ndarray of int
        0-based position of each base period inside the native period containing it.
    length : numpy.ndarray of int
        Number of base periods of that native period (the whole period, also when it
        extends beyond ``index``).

    Raises
    ------
    ValueError
        If ``freq`` is higher than the index frequency.

    Examples
    --------
    >>> import pandas as pd
    >>> from nowcastbox.core.frequency import calendar_position
    >>> idx = pd.period_range("2020-01-27", periods=6, freq="W")
    >>> pos, length = calendar_position(idx, "M")
    >>> pos.tolist(), length.tolist()
    ([0, 1, 2, 3, 0, 1], [4, 4, 4, 4, 5, 5])
    """
    native = base_to_native(index, freq)
    first, last = native_period_bounds(native, Frequency.from_index(index))
    position = _ordinals(index) - _ordinals(first)
    length = _ordinals(last) - _ordinals(first) + 1
    return position.astype(np.int64), length.astype(np.int64)


def _ordinals(index: pd.PeriodIndex) -> np.ndarray:
    return np.asarray(index.asi8, dtype=np.int64)  # pyright: ignore[reportAttributeAccessIssue]


def period_to_base(period: pd.Period | str, base: FrequencyLike, *, how: str = "end") -> pd.Period:
    """Map a period of any frequency onto the base grid.

    Parameters
    ----------
    period : pandas.Period or str
        Period to map, e.g. ``pd.Period("2020Q1")`` or ``"2020Q1"``.
    base : Frequency, str or int
        Frequency of the base grid (must not be lower than the period's frequency).
    how : {"end", "start"}, default "end"
        Return the last (storage convention) or first base period.

    Returns
    -------
    pandas.Period
        Base-grid period.

    Raises
    ------
    ValueError
        If ``how`` is invalid or the base frequency is lower than the period's.

    Examples
    --------
    >>> from nowcastbox.core.frequency import period_to_base
    >>> period_to_base("2020Q1", "M")
    Period('2020-03', 'M')
    >>> period_to_base("2020Q1", "M", how="start")
    Period('2020-01', 'M')
    """
    if how not in {"end", "start"}:
        raise ValueError(f"how must be 'end' or 'start', got {how!r}.")
    p = pd.Period(period)
    base_f = Frequency.from_value(base)
    own = _parse_frequency_string(p.freqstr)
    if base_f.is_lower_than(own):
        raise ValueError(
            f"Cannot map a {own.label} period onto a {base_f.label} base grid "
            "(the base grid must be at least as fine as the series)."
        )
    if own == base_f:
        return p
    first, last = native_period_bounds(pd.PeriodIndex([p]), base_f)
    return last[0] if how == "end" else first[0]


def base_to_native(index: pd.PeriodIndex, freq: FrequencyLike) -> pd.PeriodIndex:
    """Map each base-grid period to the period of frequency ``freq`` that contains it.

    Parameters
    ----------
    index : pandas.PeriodIndex
        Base-grid index.
    freq : Frequency, str or int
        Target (lower or equal) frequency.

    Returns
    -------
    pandas.PeriodIndex
        Index of the same length at frequency ``freq``.

    Raises
    ------
    ValueError
        If ``freq`` is higher than the index frequency.

    Examples
    --------
    >>> import pandas as pd
    >>> from nowcastbox.core.frequency import base_to_native
    >>> idx = pd.period_range("2020-01", periods=4, freq="M")
    >>> list(base_to_native(idx, "Q").astype(str))
    ['2020Q1', '2020Q1', '2020Q1', '2020Q2']
    """
    target = Frequency.from_value(freq)
    base = Frequency.from_index(index)
    if target.is_higher_than(base):
        raise ValueError(f"Cannot map a {base.label} index to the higher frequency {target.label}.")
    return index.asfreq(target.pandas_freq)


def native_to_base(index: pd.PeriodIndex, base: FrequencyLike) -> pd.PeriodIndex:
    """Map a native-frequency index onto the base grid (storage convention: period end).

    Parameters
    ----------
    index : pandas.PeriodIndex
        Index at the series' own frequency, e.g. quarterly.
    base : Frequency, str or int
        Frequency of the base grid.

    Returns
    -------
    pandas.PeriodIndex
        Index of base-grid periods (last base period of each native period).

    Raises
    ------
    ValueError
        If the base frequency is lower than the index frequency.

    Examples
    --------
    >>> import pandas as pd
    >>> from nowcastbox.core.frequency import native_to_base
    >>> q = pd.period_range("2020Q1", periods=2, freq="Q")
    >>> list(native_to_base(q, "M").astype(str))
    ['2020-03', '2020-06']
    """
    return native_period_bounds(index, base)[1]


def is_period_end(index: pd.PeriodIndex, freq: FrequencyLike) -> np.ndarray:
    """Flag the base-grid periods that are storage slots for frequency ``freq``.

    Parameters
    ----------
    index : pandas.PeriodIndex
        Base-grid index.
    freq : Frequency, str or int
        Series frequency (lower than or equal to the index frequency).

    Returns
    -------
    numpy.ndarray of bool
        ``True`` where the base period is the last base period of a ``freq`` period.

    Raises
    ------
    ValueError
        If ``freq`` is higher than the index frequency.

    Examples
    --------
    >>> import pandas as pd
    >>> from nowcastbox.core.frequency import is_period_end
    >>> idx = pd.period_range("2020-01", periods=6, freq="M")
    >>> is_period_end(idx, "Q").tolist()
    [False, False, True, False, False, True]
    """
    position, length = calendar_position(index, freq)
    return np.asarray(position == length - 1, dtype=bool)


def infer_frequency(series: pd.Series, *, min_observations: int = 2) -> Frequency:
    """Infer the frequency of a series stored on a base grid from its observation pattern.

    Returns the *lowest* frequency (lower than the base) such that every observed value
    lies on a storage slot of that frequency (calendar-aware for weekly and daily base
    grids). Series with fewer than ``min_observations`` values are assigned the base
    frequency.

    Parameters
    ----------
    series : pandas.Series
        Series indexed by a base-grid :class:`pandas.PeriodIndex`.
    min_observations : int, default 2
        Minimum number of non-missing values needed to infer a lower frequency.

    Returns
    -------
    Frequency
        Inferred frequency.

    Raises
    ------
    TypeError
        If the series is not indexed by a :class:`pandas.PeriodIndex`.

    Notes
    -----
    This is a heuristic: a monthly series that happens to be observed only in
    quarter-end months is classified as quarterly. Pass frequencies explicitly
    whenever they are known.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from nowcastbox.core.frequency import infer_frequency
    >>> idx = pd.period_range("2020-01", periods=6, freq="M")
    >>> infer_frequency(pd.Series([np.nan, np.nan, 1.0, np.nan, np.nan, 2.0], index=idx))
    <Frequency.QUARTERLY: 'Q'>
    """
    if not isinstance(series.index, pd.PeriodIndex):
        raise TypeError("infer_frequency requires a Series indexed by a pandas.PeriodIndex.")
    index = series.index
    base = Frequency.from_index(index)
    observed = series.notna().to_numpy()
    if int(observed.sum()) < min_observations:
        return base
    candidates = sorted(
        (f for f in Frequency if f.is_lower_than(base)), key=lambda f: f.periods_per_year
    )
    for freq in candidates:
        if bool(np.all(is_period_end(index, freq)[observed])):
            return freq
    return base


class AggregationType(str, Enum):
    r"""Link between a low-frequency observation and the latent high-frequency series.

    Let :math:`k` be the number of high-frequency periods per low-frequency period
    (:func:`aggregation_ratio`) and :math:`t` the last high-frequency period of the
    low-frequency period. The low-frequency value is
    :math:`y_t = \sum_j w_j x_{t-j}` with weights given by :meth:`weights`:

    * ``FLOW``: sum, :math:`w = (1, \dots, 1)` (length :math:`k`);
    * ``AVERAGE``: mean, :math:`w = (1, \dots, 1)/k`;
    * ``STOCK``: end-of-period value, :math:`w = (1, 0, \dots, 0)`;
    * ``GROWTH_RATE`` (``"mariano_murasawa"``): growth rate of a low-frequency
      flow/average expressed with high-frequency growth rates (Mariano & Murasawa,
      2003), triangular weights of length :math:`2k-1`; for :math:`k=3`,
      :math:`w = (1, 2, 3, 2, 1)` (times :math:`1/3` when ``normalize=True``).

    Examples
    --------
    >>> from nowcastbox.core.frequency import AggregationType
    >>> AggregationType.from_value("mm").weights(3).tolist()
    [1.0, 2.0, 3.0, 2.0, 1.0]
    >>> AggregationType.STOCK.weights(3).tolist()
    [1.0, 0.0, 0.0]
    """

    FLOW = "flow"
    STOCK = "stock"
    AVERAGE = "average"
    GROWTH_RATE = "mariano_murasawa"

    @classmethod
    def from_value(cls, value: AggregationTypeLike) -> AggregationType:
        """Parse an aggregation type from a member or a string alias.

        Parameters
        ----------
        value : AggregationType or str
            Member, value or alias (``"sum"``, ``"flow"``, ``"stock"``, ``"last"``,
            ``"end"``, ``"mean"``, ``"average"``, ``"mm"``, ``"mariano_murasawa"``,
            ``"mariano-murasawa"``, ``"growth"``, ``"growth_rate"``), case-insensitive.

        Returns
        -------
        AggregationType
            Parsed member.

        Raises
        ------
        ValueError
            If ``value`` is not recognised.

        Examples
        --------
        >>> AggregationType.from_value("sum")
        <AggregationType.FLOW: 'flow'>
        """
        if isinstance(value, AggregationType):
            return value
        if isinstance(value, str):
            key = value.strip().lower().replace("-", "_").replace(" ", "_")
            if key in _AGGREGATION_ALIASES:
                return _AGGREGATION_ALIASES[key]
        raise ValueError(
            f"Unknown aggregation type {value!r}; expected one of {sorted(_AGGREGATION_ALIASES)}."
        )

    def n_lags(self, ratio: int) -> int:
        """Number of high-frequency lags entering the aggregation (``len(weights) - 1``).

        Parameters
        ----------
        ratio : int
            High-frequency periods per low-frequency period.

        Returns
        -------
        int
            Number of lags.

        Examples
        --------
        >>> AggregationType.GROWTH_RATE.n_lags(3)
        4
        """
        return len(self.weights(ratio)) - 1

    def weights(self, ratio: int, *, normalize: bool = False) -> np.ndarray:
        r"""Aggregation weights on :math:`(x_t, x_{t-1}, \dots)`.

        Parameters
        ----------
        ratio : int
            High-frequency periods per low-frequency period (e.g. 3 for M -> Q).
        normalize : bool, default False
            For ``GROWTH_RATE`` only: divide the triangular weights by ``ratio``
            (Mariano-Murasawa approximation of the low-frequency growth rate, e.g.
            :math:`\tfrac13(1,2,3,2,1)`). Ignored for the other types.

        Returns
        -------
        numpy.ndarray
            1-D float array of weights, most recent period first.

        Raises
        ------
        ValueError
            If ``ratio`` is not a positive integer.

        Examples
        --------
        >>> AggregationType.GROWTH_RATE.weights(3, normalize=True).round(4).tolist()
        [0.3333, 0.6667, 1.0, 0.6667, 0.3333]
        >>> AggregationType.AVERAGE.weights(4).tolist()
        [0.25, 0.25, 0.25, 0.25]
        """
        if isinstance(ratio, bool) or not isinstance(ratio, Integral) or ratio < 1:
            raise ValueError(f"ratio must be a positive integer, got {ratio!r}.")
        k = int(ratio)
        if self is AggregationType.FLOW:
            return np.ones(k)
        if self is AggregationType.AVERAGE:
            return np.full(k, 1.0 / k)
        if self is AggregationType.STOCK:
            w = np.zeros(k)
            w[0] = 1.0
            return w
        j = np.arange(2 * k - 1)
        tri = (k - np.abs(j - (k - 1))).astype(float)
        return tri / k if normalize else tri

    def calendar_weights(self, n_current: int, n_previous: int | None = None) -> np.ndarray:
        r"""Exact weights for a low-frequency period of ``n_current`` high-frequency periods.

        Calendar-aware counterpart of :meth:`weights` for pairs whose ratio varies
        (e.g. 4 or 5 weeks per month). With :math:`n = ` ``n_current`` and
        :math:`n' = ` ``n_previous`` (length of the previous low-frequency period):

        * ``FLOW``: :math:`(1, \dots, 1)` (length :math:`n`, the sum);
        * ``AVERAGE``: :math:`(1, \dots, 1)/n` (the mean);
        * ``STOCK``: :math:`(1, 0, \dots, 0)` (the last value);
        * ``GROWTH_RATE``: growth rate of the low-frequency (geometric) average written
          with high-frequency growth rates :math:`x_s = \tilde x_s - \tilde x_{s-1}`,
          :math:`y_T = \tfrac1n\sum_{j\in T}\tilde x_j - \tfrac1{n'}\sum_{j\in T-1}\tilde x_j`
          (Mariano & Murasawa, 2003, generalised to unequal periods): weight
          :math:`(l+1)/n` on lag :math:`l < n` and :math:`(n' - v)/n'` on lag
          :math:`n - 1 + v`, :math:`v = 1, \dots, n'-1` (length :math:`n + n' - 1`).

        With :math:`n = n' = k` the result equals ``weights(k, normalize=True)``.

        Parameters
        ----------
        n_current : int
            High-frequency periods in the current low-frequency period (positive).
        n_previous : int, optional
            High-frequency periods in the previous one (``GROWTH_RATE`` only; defaults
            to ``n_current``).

        Returns
        -------
        numpy.ndarray
            Weights on :math:`(x_t, x_{t-1}, \dots)`, most recent period first.

        Raises
        ------
        ValueError
            If a length is not a positive integer.

        Examples
        --------
        >>> AggregationType.GROWTH_RATE.calendar_weights(3).round(4).tolist()
        [0.3333, 0.6667, 1.0, 0.6667, 0.3333]
        >>> AggregationType.GROWTH_RATE.calendar_weights(2, 3).round(4).tolist()
        [0.5, 1.0, 0.6667, 0.3333]
        >>> AggregationType.AVERAGE.calendar_weights(4).tolist()
        [0.25, 0.25, 0.25, 0.25]
        """
        n = _positive_int(n_current, "n_current")
        n_prev = n if n_previous is None else _positive_int(n_previous, "n_previous")
        if self is AggregationType.FLOW:
            return np.ones(n)
        if self is AggregationType.AVERAGE:
            return np.full(n, 1.0 / n)
        if self is AggregationType.STOCK:
            w = np.zeros(n)
            w[0] = 1.0
            return w
        current = np.arange(1, n + 1, dtype=float) / n
        previous = (n_prev - np.arange(1, n_prev, dtype=float)) / n_prev
        return np.concatenate([current, previous])


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    return int(value)


AggregationTypeLike: TypeAlias = AggregationType | str

_AGGREGATION_ALIASES: dict[str, AggregationType] = {
    "flow": AggregationType.FLOW,
    "sum": AggregationType.FLOW,
    "stock": AggregationType.STOCK,
    "last": AggregationType.STOCK,
    "end": AggregationType.STOCK,
    "average": AggregationType.AVERAGE,
    "mean": AggregationType.AVERAGE,
    "avg": AggregationType.AVERAGE,
    "mariano_murasawa": AggregationType.GROWTH_RATE,
    "mm": AggregationType.GROWTH_RATE,
    "growth": AggregationType.GROWTH_RATE,
    "growth_rate": AggregationType.GROWTH_RATE,
}
