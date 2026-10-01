"""Pseudo real-time vintages.

A *pseudo real-time* vintage is the final (latest available) dataset with every value
that had not yet been published at the vintage date replaced by NaN. Publication dates
come from a :class:`~nowcastbox.vintages.calendar.ReleaseCalendar` (stylised delays or
explicit release dates). Data revisions are ignored; for genuine real-time data use
:class:`~nowcastbox.vintages.vintage_store.VintageStore`.

This reproduces the "ragged edge" faced by a forecaster at each date and is the
standard design for evaluating nowcasting models (Giannone, Reichlin & Small, 2008;
Bańbura & Rünstler, 2011; Bańbura, Giannone, Modugno & Reichlin, 2013).

References
----------
Giannone, D., Reichlin, L. & Small, D. (2008). Nowcasting: The real-time informational
content of macroeconomic data. *Journal of Monetary Economics*, 55(4), 665-676.

Bańbura, M., Giannone, D., Modugno, M. & Reichlin, L. (2013). Now-casting and the
real-time data flow. In *Handbook of Economic Forecasting*, vol. 2A, 195-237.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping, Sequence
from numbers import Integral
from typing import NamedTuple, TypeAlias

import numpy as np
import pandas as pd

from nowcastbox._logging import get_logger
from nowcastbox.core.data import FrequencySpec, MixedFrequencyData, as_mixed_frequency_data
from nowcastbox.vintages._utils import DateLike, positional_or_named, to_date
from nowcastbox.vintages.calendar import ReleaseCalendar

__all__ = [
    "DelayLike",
    "StepLike",
    "Vintage",
    "generate_vintages",
    "pseudo_real_time",
    "vintage_dates",
]

logger = get_logger(__name__)

DelayLike: TypeAlias = (
    ReleaseCalendar | Mapping[str, object] | pd.Series | Sequence[object] | np.ndarray | int
)
StepLike: TypeAlias = str | int | pd.Timedelta | pd.DateOffset

PanelLike: TypeAlias = MixedFrequencyData | pd.DataFrame


class Vintage(NamedTuple):
    """A dated dataset, as yielded by :func:`generate_vintages`.

    Attributes
    ----------
    date : pandas.Timestamp
        Information date.
    data : MixedFrequencyData or pandas.DataFrame
        The dataset as observed at ``date`` (same type as the input panel).

    Examples
    --------
    >>> import pandas as pd
    >>> v = Vintage(pd.Timestamp("2020-01-31"), pd.DataFrame())
    >>> date, data = v
    >>> str(date.date())
    '2020-01-31'
    """

    date: pd.Timestamp
    data: MixedFrequencyData | pd.DataFrame


def _resolve_calendar(mfd: MixedFrequencyData, delay: DelayLike | None) -> ReleaseCalendar:
    """Turn any accepted delay specification into a calendar covering ``mfd``."""
    if delay is None:
        return ReleaseCalendar.from_data(mfd)
    if isinstance(delay, ReleaseCalendar):
        return delay
    columns = mfd.columns
    if isinstance(delay, Integral) and not isinstance(delay, bool):
        mapping: dict[str, object] = dict.fromkeys(columns, int(delay))
    elif isinstance(delay, pd.Series):
        labels = [str(k) for k in delay.index]
        unique = len(labels) == len(set(labels))
        if unique and set(labels) <= set(columns):
            mapping = dict(zip(labels, delay.tolist(), strict=True))
        elif unique and set(columns) <= set(labels):
            # A named delay vector of a larger panel (e.g. ``Dataset.delay`` after
            # ``prepare_panel`` dropped sparse series): select by name.
            by_name = dict(zip(labels, delay.tolist(), strict=True))
            mapping = {c: by_name[c] for c in columns}
        else:
            mapping = positional_or_named(delay.tolist(), columns, "delay")
    elif isinstance(delay, Mapping):
        mapping = {str(k): v for k, v in delay.items()}
    elif isinstance(delay, (Sequence, np.ndarray)) and not isinstance(delay, str):
        mapping = positional_or_named(list(delay), columns, "delay")
    else:
        raise ValueError(
            "delay must be a ReleaseCalendar, a mapping/Series of days, a sequence aligned "
            f"with the columns or an int; got {type(delay).__name__}."
        )
    return ReleaseCalendar.from_data(mfd, mapping)


def _coerce_panel(data: PanelLike, frequency: FrequencySpec | None) -> MixedFrequencyData:
    if isinstance(data, MixedFrequencyData) and frequency is not None:
        raise ValueError("frequency can only be given with a DataFrame input.")
    return as_mixed_frequency_data(data, frequency)


def _pick_delay(delay: DelayLike | None, calendar: DelayLike | None) -> DelayLike | None:
    if delay is not None and calendar is not None:
        raise ValueError("Pass either delay or calendar, not both.")
    return calendar if calendar is not None else delay


def _apply(mfd: MixedFrequencyData, mask: np.ndarray, as_frame: bool) -> PanelLike:
    frame = mfd.data.where(mask)
    if as_frame:
        return frame
    return mfd.with_data(frame)


def pseudo_real_time(
    data: PanelLike,
    delay: DelayLike | None = None,
    vintage: DateLike | None = None,
    *,
    calendar: DelayLike | None = None,
    frequency: FrequencySpec | None = None,
) -> PanelLike:
    """Dataset as it would have been observed at date ``vintage``.

    Every observation whose release date (see
    :class:`~nowcastbox.vintages.calendar.ReleaseCalendar`) is after ``vintage`` is set
    to NaN; the period grid is unchanged. No revisions are modelled.

    Parameters
    ----------
    data : MixedFrequencyData or pandas.DataFrame
        Final (latest) dataset on the base grid. DataFrames must have a monthly
        :class:`pandas.PeriodIndex` (quarterly values in the 3rd month of the quarter).
    delay : ReleaseCalendar, mapping, Series, sequence or int, optional
        Release rule: a calendar, delays in days by series name (mapping or Series), a
        sequence aligned with the columns (as in R's ``ts`` legends) or one delay for all
        series. ``None`` uses the ``release_delays`` metadata of ``data``.
    vintage : date-like
        Information date (inclusive: values released on that day are kept).
    calendar : ReleaseCalendar or delay specification, optional
        Alias of ``delay`` (pass one of them).
    frequency : frequency specification, optional
        Per-series frequencies for DataFrame input (inferred when omitted).

    Returns
    -------
    MixedFrequencyData or pandas.DataFrame
        Same type as ``data``.

    Raises
    ------
    ValueError
        If ``vintage`` is missing, both ``delay`` and ``calendar`` are given, or the
        delay specification is malformed.
    NowcastDataError
        If a series has no release rule or an observed value has no release date.

    See Also
    --------
    generate_vintages : Sequence of vintages.
    nowcastbox.core.data.MixedFrequencyData.as_of : Delay-only special case.

    Examples
    --------
    >>> import pandas as pd
    >>> from nowcastbox.vintages import pseudo_real_time
    >>> idx = pd.period_range("2020-01", periods=6, freq="M")
    >>> frame = pd.DataFrame(
    ...     {"ip": [1.0, 2, 3, 4, 5, 6], "gdp": [None, None, 0.5, None, None, 0.7]}, index=idx
    ... )
    >>> v = pseudo_real_time(frame, delay={"ip": 30, "gdp": 60}, vintage="2020-06-15")
    >>> v["ip"].tolist()
    [1.0, 2.0, 3.0, 4.0, nan, nan]
    >>> v["gdp"].dropna().tolist()
    [0.5]
    """
    if vintage is None:
        raise ValueError("vintage (information date) is required.")
    mfd = _coerce_panel(data, frequency)
    cal = _resolve_calendar(mfd, _pick_delay(delay, calendar))
    date = to_date(vintage, name="vintage")
    mask = cal.release_mask(mfd, date).to_numpy()
    return _apply(mfd, mask, as_frame=isinstance(data, pd.DataFrame))


_STEP_RE = re.compile(r"^\s*(\d*)\s*([A-Za-z]+)\s*$")
_STEP_UNITS: dict[str, tuple[str, int]] = {
    "D": ("days", 1),
    "B": ("bdays", 1),
    "W": ("days", 7),
    "M": ("months", 1),
    "ME": ("months", 1),
    "MS": ("months", 1),
    "Q": ("months", 3),
    "QE": ("months", 3),
    "QS": ("months", 3),
    "Y": ("months", 12),
    "YE": ("months", 12),
    "YS": ("months", 12),
    "A": ("months", 12),
}


def _parse_step(step: StepLike) -> pd.DateOffset | pd.Timedelta | tuple[str, int]:
    if isinstance(step, bool):
        raise ValueError(f"Invalid step {step!r}.")
    if isinstance(step, Integral):
        if int(step) <= 0:
            raise ValueError(f"step must be positive, got {step!r}.")
        return pd.Timedelta(days=int(step))
    if isinstance(step, pd.Timedelta):
        if step <= pd.Timedelta(0):
            raise ValueError(f"step must be positive, got {step!r}.")
        return step
    if isinstance(step, pd.DateOffset):
        return step
    if isinstance(step, str):
        match = _STEP_RE.match(step)
        if match is not None:
            n = int(match.group(1) or 1)
            unit = match.group(2).upper()
            if unit in _STEP_UNITS and n > 0:
                kind, mult = _STEP_UNITS[unit]
                return (kind, n * mult)
    raise ValueError(
        f"Invalid step {step!r}; use 'release', an int (days), a Timedelta, a DateOffset or "
        "a string such as 'D', '2W', 'M', 'Q', 'B'."
    )


def vintage_dates(
    start: DateLike,
    end: DateLike,
    step: StepLike = "M",
    *,
    calendar: ReleaseCalendar | None = None,
    data: PanelLike | None = None,
) -> pd.DatetimeIndex:
    """Information dates of a sequence of vintages.

    Parameters
    ----------
    start, end : date-like
        First and last admissible dates (inclusive). ``start`` is always the first date.
    step : str, int, Timedelta or DateOffset, default "M"
        Spacing. Calendar steps (``"M"``, ``"Q"``, ``"Y"``, optionally with a multiple
        such as ``"2M"``) are anchored at ``start``: date :math:`k` is
        ``start + k * step`` computed from ``start`` (no drift at month ends; day 31 maps
        to the last day of shorter months). ``"D"``, ``"W"`` and ints are fixed numbers
        of days, ``"B"`` business days. ``"release"`` uses ``start`` followed by every
        date with a release in ``(start, end]`` according to ``calendar``.
    calendar : ReleaseCalendar, optional
        Required by ``step="release"``.
    data : MixedFrequencyData or pandas.DataFrame, optional
        With ``step="release"``, only releases of observed values of this panel count.

    Returns
    -------
    pandas.DatetimeIndex
        Sorted vintage dates.

    Raises
    ------
    ValueError
        If ``start > end``, the step is invalid or ``step="release"`` lacks a calendar.

    Examples
    --------
    >>> from nowcastbox.vintages import vintage_dates
    >>> vintage_dates("2020-01-31", "2020-04-30", "M").strftime("%Y-%m-%d").tolist()
    ['2020-01-31', '2020-02-29', '2020-03-31', '2020-04-30']
    """
    d0 = to_date(start, name="start")
    d1 = to_date(end, name="end")
    if d0 > d1:
        raise ValueError(f"start ({d0.date()}) must not be after end ({d1.date()}).")
    if isinstance(step, str) and step.strip().lower() == "release":
        if calendar is None:
            raise ValueError("step='release' requires a calendar.")
        releases = calendar.release_dates(d0, d1, data)
        return pd.DatetimeIndex([d0, *releases], name="vintage")
    parsed = _parse_step(step)
    dates: list[pd.Timestamp] = []
    if isinstance(parsed, tuple) and parsed[0] == "bdays":
        idx = pd.bdate_range(d0, d1, freq=f"{parsed[1]}B")
        dates = [d0, *[d for d in idx if d > d0]]
        return pd.DatetimeIndex(dates, name="vintage")
    k = 0
    while True:
        if isinstance(parsed, tuple):
            kind, n = parsed
            offset = pd.DateOffset(months=k * n) if kind == "months" else pd.Timedelta(days=k * n)
            date = d0 + offset
        elif isinstance(parsed, pd.Timedelta):
            date = d0 + k * parsed
        else:
            date = d0 + k * parsed  # DateOffset multiples are anchored at start
        date = pd.Timestamp(date).normalize()
        if date > d1:
            break
        if dates and date <= dates[-1]:
            raise ValueError(f"step {step!r} does not advance the date.")
        dates.append(date)
        k += 1
    return pd.DatetimeIndex(dates, name="vintage")


def generate_vintages(
    data: PanelLike,
    calendar: DelayLike | None = None,
    start: DateLike | None = None,
    end: DateLike | None = None,
    step: StepLike = "M",
    *,
    frequency: FrequencySpec | None = None,
    skip_unchanged: bool = False,
    delay: DelayLike | None = None,
) -> Iterator[Vintage]:
    """Iterate over pseudo real-time vintages between two dates.

    Parameters
    ----------
    data : MixedFrequencyData or pandas.DataFrame
        Final dataset on the base grid.
    calendar : ReleaseCalendar or delay specification, optional
        Release rule (see :func:`pseudo_real_time`); ``None`` uses the metadata.
    start, end : date-like
        First and last vintage dates (both required).
    step : str, int, Timedelta or DateOffset, default "M"
        Spacing of the vintages (see :func:`vintage_dates`); ``"release"`` yields one
        vintage per release date.
    frequency : frequency specification, optional
        Per-series frequencies for DataFrame input.
    skip_unchanged : bool, default False
        Skip vintages identical to the previously yielded one.
    delay : ReleaseCalendar or delay specification, optional
        Alias of ``calendar`` (the keyword of :func:`pseudo_real_time`; pass one of
        them).

    Yields
    ------
    Vintage
        ``(date, data)`` named tuples; ``data`` has the type of the input.

    Raises
    ------
    ValueError
        If ``start``/``end`` are missing, both ``delay`` and ``calendar`` are given or
        the step is invalid.
    NowcastDataError
        If a series has no release rule.

    Examples
    --------
    >>> import pandas as pd
    >>> from nowcastbox.vintages import generate_vintages
    >>> idx = pd.period_range("2020-01", periods=4, freq="M")
    >>> frame = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]}, index=idx)
    >>> [int(v.data["a"].count()) for v in generate_vintages(frame, 5, "2020-02-01", "2020-05-01")]
    [0, 1, 2, 3]
    """
    if start is None or end is None:
        raise ValueError("start and end are required.")
    mfd = _coerce_panel(data, frequency)
    cal = _resolve_calendar(mfd, _pick_delay(delay, calendar))
    as_frame = isinstance(data, pd.DataFrame)
    dates = vintage_dates(start, end, step, calendar=cal, data=mfd)
    logger.debug("Generating %d pseudo real-time vintages.", len(dates))
    observed = mfd.observation_mask().to_numpy()
    previous: np.ndarray | None = None
    for date in dates:
        mask = cal.release_mask(mfd, date).to_numpy()
        visible = mask & observed
        if skip_unchanged and previous is not None and np.array_equal(visible, previous):
            continue
        previous = visible
        yield Vintage(pd.Timestamp(date), _apply(mfd, mask, as_frame))
