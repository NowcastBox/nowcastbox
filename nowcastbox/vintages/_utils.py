"""Private helpers shared by the :mod:`nowcastbox.vintages` modules."""

from __future__ import annotations

import datetime as _dt
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import TypeAlias

import numpy as np
import pandas as pd

from nowcastbox.core.exceptions import NowcastDataError
from nowcastbox.core.frequency import Frequency, FrequencyLike

__all__ = [
    "MIN_RELEASE_DELAY",
    "DateLike",
    "PeriodLike",
    "coerce_delay",
    "coerce_frequency_map",
    "grid_bound",
    "native_period_end",
    "to_date",
    "to_period",
]

MIN_RELEASE_DELAY = -366
"""Release delays (days after the end of the period) must be larger than this value."""

DateLike: TypeAlias = pd.Timestamp | str | _dt.date | np.datetime64
PeriodLike: TypeAlias = pd.Period | str | pd.Timestamp | _dt.date


def to_date(value: object, name: str = "date") -> pd.Timestamp:
    """Parse ``value`` as a calendar date (time of day and time zone dropped).

    Parameters
    ----------
    value : date-like
        Timestamp, ``datetime.date``, ``numpy.datetime64`` or ISO string.
    name : str, default "date"
        Argument name used in error messages.

    Returns
    -------
    pandas.Timestamp
        Normalised (midnight) naive timestamp.

    Raises
    ------
    ValueError
        If ``value`` cannot be parsed or is missing.

    Examples
    --------
    >>> to_date("2020-03-31 15:00")
    Timestamp('2020-03-31 00:00:00')
    """
    if value is None or isinstance(value, bool):
        raise ValueError(f"{name} must be a date, got {value!r}.")
    if isinstance(value, pd.Period):
        raise ValueError(f"{name} must be a date (not a period), got {value!r}.")
    try:
        ts = pd.Timestamp(value)  # pyright: ignore[reportArgumentType]
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Cannot parse {name}={value!r} as a date.") from exc
    if pd.isna(ts):
        raise ValueError(f"{name} must be a date, got a missing value.")
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts.normalize()


def to_period(value: object, frequency: Frequency | None, name: str = "period") -> pd.Period:
    """Parse ``value`` as a period of ``frequency``.

    Parameters
    ----------
    value : period-like
        ``pandas.Period``, string (``"2020Q1"``, ``"2020-03"``) or date-like.
    frequency : Frequency or None
        Target frequency. ``None`` keeps the frequency of a parsed period (dates then
        raise because their frequency is ambiguous).
    name : str, default "period"
        Argument name used in error messages.

    Returns
    -------
    pandas.Period
        Period at ``frequency``.

    Raises
    ------
    NowcastDataError
        If the value cannot be parsed, is of lower frequency than ``frequency`` or is a
        date without a frequency.

    Examples
    --------
    >>> to_period("2020-02", Frequency.QUARTERLY)
    Period('2020Q1', 'Q-DEC')
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        raise NowcastDataError(f"{name} is missing.")
    if isinstance(value, pd.Period):
        period = value
    elif isinstance(value, str):
        try:
            period = (
                pd.Period(value, freq=frequency.pandas_freq)
                if frequency is not None and _looks_native(value, frequency)
                else pd.Period(value)
            )
        except (ValueError, TypeError) as exc:
            raise NowcastDataError(f"Cannot parse {name}={value!r} as a period.") from exc
    else:
        if frequency is None:
            raise NowcastDataError(
                f"{name}={value!r} is a date; a frequency is required to map it to a period."
            )
        try:
            period = pd.Period(pd.Timestamp(value), freq=frequency.pandas_freq)  # pyright: ignore[reportArgumentType]
        except (ValueError, TypeError) as exc:
            raise NowcastDataError(f"Cannot parse {name}={value!r} as a period.") from exc
    if frequency is None:
        return period
    own = Frequency.from_value(period.freqstr)
    if own.is_lower_than(frequency):
        raise NowcastDataError(
            f"{name}={period} is {own.label}, lower than the series frequency "
            f"({frequency.label}); it does not identify a single reference period."
        )
    return period.asfreq(frequency.pandas_freq, how="E")


def _looks_native(value: str, frequency: Frequency) -> bool:
    """Whether a string should be parsed directly at ``frequency``."""
    text = value.strip().upper()
    if frequency is Frequency.QUARTERLY:
        return "Q" in text
    if frequency is Frequency.WEEKLY:
        return "/" in text
    if frequency is Frequency.ANNUAL:
        return text.isdigit() and len(text) == 4
    return False


def native_period_end(period: pd.Period) -> pd.Timestamp:
    """Last calendar day of ``period`` (midnight).

    Parameters
    ----------
    period : pandas.Period
        Reference period.

    Returns
    -------
    pandas.Timestamp
        Normalised end date.

    Examples
    --------
    >>> native_period_end(pd.Period("2020Q1"))
    Timestamp('2020-03-31 00:00:00')
    """
    return pd.Timestamp(period.end_time).normalize()


def coerce_delay(value: object, series: str) -> int | None:
    """Validate a release delay (integer number of days or missing).

    Delays are counted from the last day of the reference period. A negative delay
    means that the value is published *before* the period ends (e.g. regional
    business surveys collected in the first half of the month); it must be larger
    than :data:`MIN_RELEASE_DELAY` (one year before the end of the period).

    Parameters
    ----------
    value : object
        Candidate delay.
    series : str
        Series name for error messages.

    Returns
    -------
    int or None
        The delay, or ``None`` when missing.

    Raises
    ------
    ValueError
        If the delay is not an integer or is not larger than :data:`MIN_RELEASE_DELAY`.

    Examples
    --------
    >>> coerce_delay(30.0, "ip")
    30
    >>> coerce_delay(-14, "empire_state")
    -14
    """
    if value is None or value is pd.NA or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"Release delay of {series!r} must be an integer, got {value!r}.")
    as_float = float(value)
    if not as_float.is_integer():
        raise ValueError(f"Release delay of {series!r} must be an integer, got {value!r}.")
    if as_float <= MIN_RELEASE_DELAY:
        raise ValueError(
            f"Release delay of {series!r} must be an integer larger than {MIN_RELEASE_DELAY} days "
            f"(released at most one year before the end of the period), got {value!r}."
        )
    return int(as_float)


def coerce_frequency_map(
    frequencies: Mapping[str, FrequencyLike] | pd.Series | None,
) -> dict[str, Frequency]:
    """Convert a frequency mapping to ``{name: Frequency}``.

    Parameters
    ----------
    frequencies : mapping or Series, optional
        Frequencies by series name.

    Returns
    -------
    dict of str to Frequency
        Parsed frequencies (empty when ``None``).

    Raises
    ------
    ValueError
        If ``frequencies`` is not a mapping or contains unknown frequencies.

    Examples
    --------
    >>> coerce_frequency_map({"gdp": 4})
    {'gdp': <Frequency.QUARTERLY: 'Q'>}
    """
    if frequencies is None:
        return {}
    if isinstance(frequencies, (pd.Series, Mapping)):
        items = frequencies.items()
    else:
        raise ValueError(
            f"frequencies must be a mapping or Series, got {type(frequencies).__name__}."
        )
    return {str(k): Frequency.from_value(v) for k, v in items}  # pyright: ignore[reportArgumentType]


def grid_bound(value: object, base: Frequency, how: str) -> pd.Period:
    """Parse a grid boundary (``start``/``end``) onto the base grid.

    Parameters
    ----------
    value : period-like or date-like
        Boundary. Lower-frequency periods map to their first (``how="start"``) or last
        (``how="end"``) base period.
    base : Frequency
        Base frequency.
    how : {"start", "end"}
        Which end of a lower-frequency period to use.

    Returns
    -------
    pandas.Period
        Base-grid period.

    Raises
    ------
    NowcastDataError
        If the value cannot be parsed.

    Examples
    --------
    >>> grid_bound("2020Q2", Frequency.MONTHLY, "end")
    Period('2020-06', 'M')
    """
    if isinstance(value, pd.Period):
        period = value
    else:
        try:
            period = pd.Period(value)  # pyright: ignore[reportArgumentType]
        except (ValueError, TypeError) as exc:
            raise NowcastDataError(f"Cannot parse grid boundary {value!r}.") from exc
    return period.asfreq(base.pandas_freq, how="S" if how == "start" else "E")


def positional_or_named(
    values: Sequence[object] | np.ndarray, columns: Sequence[str], what: str
) -> dict[str, object]:
    """Align a positional sequence with ``columns``.

    Parameters
    ----------
    values : sequence
        One value per column.
    columns : sequence of str
        Column names.
    what : str
        Name used in error messages.

    Returns
    -------
    dict
        ``{column: value}``.

    Raises
    ------
    ValueError
        If the lengths differ.

    Examples
    --------
    >>> positional_or_named([1, 2], ["a", "b"], "delay")
    {'a': 1, 'b': 2}
    """
    seq = list(values)
    if len(seq) != len(columns):
        raise ValueError(f"{what} has {len(seq)} entries but the data have {len(columns)} series.")
    return dict(zip(columns, seq, strict=True))
