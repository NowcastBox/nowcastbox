"""Helpers that turn raw (date, value) records into period-indexed pandas objects.

Every connector returns series indexed by a :class:`pandas.PeriodIndex` at the
series' *native* frequency (inferred from the observation dates unless given), and
combines several series either at their common native frequency or on a requested
base grid using the storage convention of :mod:`nowcastbox.core` (a lower-frequency
value is stored in the last base period of its native period).
"""

from __future__ import annotations

import datetime as dt
import math
import warnings
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from functools import reduce

import numpy as np
import pandas as pd

from nowcastbox.core.exceptions import DataQualityWarning, NowcastDataError
from nowcastbox.core.frequency import Frequency, FrequencyLike, native_to_base

__all__ = [
    "DateLike",
    "build_series",
    "combine_series",
    "filter_period_range",
    "infer_native_frequency",
    "normalize_codes",
    "resolve_native_frequency",
    "to_timestamp",
]

DateLike = str | dt.date | dt.datetime | pd.Timestamp | pd.Period


def to_timestamp(value: DateLike | None, *, end: bool = False) -> pd.Timestamp | None:
    """Convert a date-like value to a (normalized) timestamp.

    Parameters
    ----------
    value : str, date, datetime, Timestamp, Period or None
        Value to convert. Periods (and period-like strings such as ``"2020Q1"`` or
        ``"2020-03"``) map to their first day, or to their last day when ``end=True``.
    end : bool, default False
        Whether ``value`` is the end of a range.

    Returns
    -------
    pandas.Timestamp or None
        Day-precision timestamp (``None`` when ``value`` is ``None``).

    Raises
    ------
    ValueError
        If ``value`` cannot be parsed.
    """
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if len(text) == 10 and text.count("-") == 2:
            return pd.Timestamp(text).normalize()
        try:
            period = pd.Period(text)
        except (ValueError, TypeError):
            raise ValueError(f"Cannot interpret {value!r} as a date.") from None
        value = period
    if isinstance(value, pd.Period):
        stamp = value.end_time if end else value.start_time
        return pd.Timestamp(stamp).normalize()
    if isinstance(value, dt.date | dt.datetime | pd.Timestamp):
        return pd.Timestamp(value).normalize()
    raise ValueError(f"Cannot interpret {value!r} as a date.")


def check_range(start: pd.Timestamp | None, end: pd.Timestamp | None) -> None:
    """Raise ``ValueError`` when ``start`` is after ``end``."""
    if start is not None and end is not None and start > end:
        raise ValueError(f"start ({start.date()}) is after end ({end.date()}).")


def _month_number(dates: pd.DatetimeIndex) -> np.ndarray:
    return np.asarray(dates.year) * 12 + np.asarray(dates.month)


def infer_native_frequency(dates: Iterable[pd.Timestamp] | pd.DatetimeIndex) -> Frequency:
    """Infer the native frequency of a series from its observation dates.

    If every date falls on the same day of month (all first days, or all month
    ends), the frequency is read from the greatest common divisor of the month gaps
    (1 → monthly, 3 → quarterly, 12 → annual). Otherwise, gaps that are all
    multiples of seven days mean weekly data, anything else daily data.

    Parameters
    ----------
    dates : iterable of Timestamp or DatetimeIndex
        Observation dates (at least two distinct dates).

    Returns
    -------
    Frequency
        Inferred frequency.

    Raises
    ------
    NowcastDataError
        With fewer than two distinct dates or an unsupported spacing (e.g. semiannual).
    """
    index = pd.DatetimeIndex(list(dates)).normalize().unique().sort_values()
    if len(index) < 2:
        raise NowcastDataError(
            "Cannot infer the frequency from fewer than two observations; pass "
            "native_frequency explicitly."
        )
    days = np.asarray(index.day)
    month_end = np.asarray(index.is_month_end)
    if np.all(days == days[0]) or np.all(month_end):
        months = _month_number(index)
        gaps = np.diff(months)
        if np.all(gaps > 0):
            step = reduce(math.gcd, (int(g) for g in gaps))
            mapping = {1: Frequency.MONTHLY, 3: Frequency.QUARTERLY, 12: Frequency.ANNUAL}
            if step in mapping:
                return mapping[step]
            raise NowcastDataError(
                f"Unsupported spacing of {step} months between observations; pass "
                "native_frequency explicitly or aggregate the series."
            )
    day_gaps = np.diff(index.to_numpy().astype("datetime64[D]").astype(np.int64))
    if np.all(day_gaps % 7 == 0):
        return Frequency.WEEKLY
    return Frequency.DAILY


def resolve_native_frequency(
    native_frequency: FrequencyLike | Mapping[str, FrequencyLike] | None, name: str
) -> Frequency | None:
    """Pick the native frequency of series ``name`` from a scalar or a mapping."""
    if native_frequency is None:
        return None
    if isinstance(native_frequency, Mapping):
        value = native_frequency.get(name)
        return None if value is None else Frequency.from_value(value)
    return Frequency.from_value(native_frequency)


def build_series(
    dates: Sequence[pd.Timestamp] | pd.DatetimeIndex,
    values: Sequence[object],
    name: str,
    *,
    native_frequency: FrequencyLike | None = None,
    missing_tokens: Iterable[str] = ("",),
) -> pd.Series:
    """Build a float series indexed by a sorted, unique native-frequency PeriodIndex.

    Parameters
    ----------
    dates : sequence of Timestamp
        Observation dates.
    values : sequence
        Raw values (numbers or numeric strings, decimal point ``.``).
    name : str
        Series name.
    native_frequency : Frequency, str or int, optional
        Frequency of the series; inferred from ``dates`` when omitted.
    missing_tokens : iterable of str, default ("",)
        Strings that denote a missing value. Other non-numeric strings also become
        ``NaN`` but trigger a :class:`DataQualityWarning`.

    Returns
    -------
    pandas.Series
        ``float64`` series.

    Raises
    ------
    NowcastDataError
        If ``dates`` and ``values`` differ in length, or two different values fall in
        the same period.
    """
    if len(dates) != len(values):
        raise NowcastDataError(f"{name}: {len(dates)} dates but {len(values)} values.")
    if len(dates) == 0:
        index = pd.PeriodIndex([], freq=Frequency.from_value(native_frequency or "M").pandas_freq)
        return pd.Series([], index=index, dtype="float64", name=name)
    stamps = pd.DatetimeIndex(pd.to_datetime(list(dates))).normalize()
    freq = (
        Frequency.from_value(native_frequency)
        if native_frequency is not None
        else infer_native_frequency(stamps)
    )
    raw = pd.Series(list(values), dtype="object")
    tokens = {t.strip() for t in missing_tokens}
    is_token = raw.map(lambda v: v is None or (isinstance(v, str) and v.strip() in tokens))
    numeric = pd.to_numeric(raw.where(~is_token, None), errors="coerce").astype("float64")
    bad = numeric.isna() & ~is_token & raw.notna()
    if bool(bad.any()):
        examples = ", ".join(repr(v) for v in raw[bad].unique()[:3])
        warnings.warn(
            f"{name}: {int(bad.sum())} non-numeric value(s) ({examples}) set to NaN.",
            DataQualityWarning,
            stacklevel=3,
        )
    periods = pd.PeriodIndex(stamps.to_period(freq.pandas_freq))
    series = pd.Series(numeric.to_numpy(), index=periods, name=name)
    if series.index.has_duplicates:
        distinct = series.groupby(level=0).nunique()
        if bool((distinct > 1).any()):
            clash = distinct[distinct > 1].index[0]
            raise NowcastDataError(
                f"{name}: several different values for period {clash} — the native "
                f"frequency {freq.label} looks wrong; pass native_frequency explicitly."
            )
        series = series.groupby(level=0).max().rename(name)
    return series.sort_index().astype("float64")


def _period_index(series: pd.Series, name: str) -> pd.PeriodIndex:
    """Return the PeriodIndex of ``series`` (``TypeError`` for any other index)."""
    if not isinstance(series.index, pd.PeriodIndex):
        raise TypeError(f"{name!r} must be indexed by a pandas.PeriodIndex.")
    return series.index


def filter_period_range(
    series: pd.Series, start: pd.Timestamp | None, end: pd.Timestamp | None
) -> pd.Series:
    """Keep the periods whose start lies in ``[start, end]`` (open bounds when ``None``)."""
    if len(series) == 0:
        return series
    starts = _period_index(series, "series").start_time.normalize()
    mask = np.ones(len(series), dtype=bool)
    if start is not None:
        mask &= np.asarray(starts >= start)
    if end is not None:
        mask &= np.asarray(starts <= end)
    return series[mask]


def _complete_index(index: pd.PeriodIndex) -> pd.PeriodIndex:
    if len(index) == 0 or index.freqstr.startswith("D"):
        return index
    return pd.period_range(index.min(), index.max(), freq=index.freq)


def combine_series(
    series: Mapping[str, pd.Series], *, base_frequency: FrequencyLike | None = None
) -> pd.DataFrame:
    """Combine period-indexed series into one DataFrame.

    Parameters
    ----------
    series : mapping of str to pandas.Series
        Series indexed by native-frequency PeriodIndexes (column order is kept).
    base_frequency : Frequency, str or int, optional
        Frequency of the output grid. When omitted, all series must share the same
        frequency. Otherwise each series is placed on the base grid in the last base
        period of its native period (quarterly → third month), the convention of
        :class:`~nowcastbox.core.data.MixedFrequencyData`.

    Returns
    -------
    pandas.DataFrame
        ``float64`` frame. The index is contiguous between the first and last
        observed period, except for daily data, which keeps only observed days.

    Raises
    ------
    NowcastDataError
        Mixed frequencies without ``base_frequency``, or a series whose frequency is
        higher than ``base_frequency``.
    """
    if not series:
        raise NowcastDataError("No series to combine.")
    indexes = {name: _period_index(s, name) for name, s in series.items()}
    freqs = {name: Frequency.from_index(index) for name, index in indexes.items()}
    if base_frequency is None:
        distinct = sorted({f.label for f in freqs.values()})
        if len(distinct) > 1:
            detail = ", ".join(f"{n}: {f.label}" for n, f in freqs.items())
            raise NowcastDataError(
                f"Series have different native frequencies ({detail}); pass "
                "base_frequency (e.g. 'M') to place them on a common grid."
            )
        target = next(iter(freqs.values()))
        aligned = dict(series)
    else:
        target = Frequency.from_value(base_frequency)
        aligned = {}
        for name, s in series.items():
            if freqs[name].is_higher_than(target):
                raise NowcastDataError(
                    f"Series {name!r} is {freqs[name].label}, higher than the base frequency "
                    f"{target.label}; aggregate it first (nowcastbox.preprocessing)."
                )
            aligned[name] = (
                s.set_axis(native_to_base(indexes[name], target)) if freqs[name] != target else s
            )
    frame = pd.concat(
        [s.rename(name) for name, s in aligned.items()], axis=1, join="outer", sort=True
    )
    if frame.columns.has_duplicates:
        raise NowcastDataError(f"Duplicated series names: {list(frame.columns)}.")
    frame.index = pd.PeriodIndex(frame.index, freq=target.pandas_freq)
    frame = frame.reindex(_complete_index(frame.index)).astype("float64")
    frame.index.name = "period"
    return frame


def normalize_codes(
    codes: Hashable | Sequence[Hashable] | Mapping[str, Hashable],
    *,
    default_name: Callable[[Hashable], str],
) -> dict[str, str]:
    """Normalize a code, list of codes or ``{name: code}`` mapping to ``{name: code}``.

    Parameters
    ----------
    codes : scalar, sequence or mapping
        Series identifiers.
    default_name : callable
        Function ``code -> name`` used for unnamed codes.

    Returns
    -------
    dict of str to str
        Ordered mapping of series name to code.

    Raises
    ------
    ValueError
        Empty input or duplicated names.
    TypeError
        Names that are not non-empty strings.
    """
    if isinstance(codes, Mapping):
        pairs = list(codes.items())
    elif isinstance(codes, str | int) or not isinstance(codes, Iterable):
        pairs = [(default_name(codes), codes)]
    else:
        pairs = [(default_name(c), c) for c in codes]
    if not pairs:
        raise ValueError("At least one series code is required.")
    result: dict[str, str] = {}
    for name, code in pairs:
        if not isinstance(name, str) or not name.strip():
            raise TypeError(f"Series names must be non-empty strings, got {name!r}.")
        if name in result:
            raise ValueError(f"Duplicated series name {name!r}.")
        if isinstance(code, bool) or code is None or str(code).strip() == "":
            raise ValueError(f"Invalid series code {code!r} for {name!r}.")
        result[name] = str(code).strip()
    return result
