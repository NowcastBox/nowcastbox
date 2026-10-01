r"""Real-time (revised) data vintages.

:class:`VintageStore` keeps genuine real-time data in *long* format, one row per
``(series, reference_period, vintage_date, value)``: the value of ``series`` for
``reference_period`` as published (or revised) on ``vintage_date``. This is the layout
of real-time databases such as ALFRED and the Philadelphia Fed's Real-Time Data Set
for Macroeconomists (Croushore & Stark, 2001), where a row is only needed when a value
is first released or changes.

The value known at date :math:`v` for period :math:`p` is the one with the latest
``vintage_date`` :math:`\le v`. :meth:`VintageStore.as_of` returns the whole panel as
known at :math:`v` on the base grid (quarterly values in the third month), ready for
:class:`~nowcastbox.core.data.MixedFrequencyData` and the estimators.

Revision analysis (:meth:`VintageStore.revisions`, :meth:`VintageStore.revision_summary`)
follows the real-time literature: the revision of release :math:`k` is
:math:`r_{p,k} = x_{p}^{(k)} - x_{p}^{(k-1)}`, and the total revision is
:math:`x_p^{latest} - x_p^{(0)}`; summary statistics (mean, mean absolute, RMS,
noise-to-signal ratio :math:`\sigma_r / \sigma_{x^{latest}}`) are those of Aruoba (2008).

References
----------
Croushore, D. & Stark, T. (2001). A real-time data set for macroeconomists. *Journal of
Econometrics*, 105(1), 111-130.

Aruoba, S. B. (2008). Data revisions are not well behaved. *Journal of Money, Credit and
Banking*, 40(2-3), 319-340.

Mankiw, N. G. & Shapiro, M. D. (1986). News or noise? An analysis of GNP revisions.
*Survey of Current Business*, 66(5), 20-25.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nowcastbox._logging import get_logger
from nowcastbox.core.data import FrequencySpec, MixedFrequencyData, as_mixed_frequency_data
from nowcastbox.core.exceptions import DataQualityWarning, NowcastDataError
from nowcastbox.core.frequency import Frequency, FrequencyLike, native_to_base
from nowcastbox.vintages._utils import (
    DateLike,
    coerce_frequency_map,
    grid_bound,
    to_date,
    to_period,
)
from nowcastbox.vintages.calendar import ReleaseCalendar

__all__ = ["RECORD_COLUMNS", "VintageStore"]

logger = get_logger(__name__)

#: Required columns of the long format.
RECORD_COLUMNS: tuple[str, ...] = ("series", "reference_period", "vintage_date", "value")
_FILE_COLUMNS: tuple[str, ...] = (
    "series",
    "frequency",
    "reference_period",
    "vintage_date",
    "value",
)


def _same(a: float, b: float) -> bool:
    return (np.isnan(a) and np.isnan(b)) or a == b


def _check_columns(records: object) -> pd.DataFrame:
    if not isinstance(records, pd.DataFrame):
        raise NowcastDataError(f"records must be a pandas DataFrame, got {type(records).__name__}.")
    cols = [str(c) for c in records.columns]
    missing = [c for c in RECORD_COLUMNS if c not in cols]
    if missing:
        raise NowcastDataError(f"records lack columns {missing}.")
    unknown = sorted(set(cols) - set(_FILE_COLUMNS))
    if unknown:
        raise NowcastDataError(f"records have unknown columns {unknown}.")
    if records.empty:
        raise NowcastDataError("records are empty.")
    return records.reset_index(drop=True)


def _series_names(frame: pd.DataFrame) -> np.ndarray:
    names = frame["series"]
    if names.isna().any() or (names.astype(str).str.strip() == "").any():
        raise NowcastDataError("Series names must be non-empty strings.")
    return names.astype(str).to_numpy(dtype=object)


def _frequencies_from_column(
    frame: pd.DataFrame, names: np.ndarray, frequencies: dict[str, Frequency]
) -> None:
    if "frequency" not in frame:
        return
    for name, group in frame.groupby(names, sort=False):
        if str(name) in frequencies:
            continue
        values = {Frequency.from_value(f) for f in group["frequency"].dropna()}
        if len(values) > 1:
            raise NowcastDataError(f"Series {name!r} has several frequencies.")
        if values:
            frequencies[str(name)] = values.pop()


def _storage_slots(periods: Sequence[pd.Period], base: Frequency) -> pd.PeriodIndex:
    """Base-grid storage slot of every reference period (calendar aware).

    The slot is the last base period *ending* in the reference period (the convention
    of :class:`~nowcastbox.core.data.MixedFrequencyData`): on a weekly grid a quarter
    is stored in its last full week, not in the week containing its last day.
    """
    slots = np.empty(len(periods), dtype=object)
    groups: dict[str, list[int]] = {}
    for i, period in enumerate(periods):
        groups.setdefault(period.freqstr, []).append(i)
    for freqstr, positions in groups.items():
        index = pd.PeriodIndex([periods[i] for i in positions], freq=freqstr)
        slots[positions] = list(native_to_base(index, base))
    return pd.PeriodIndex(list(slots), freq=base.pandas_freq)


def _parse_periods(
    frame: pd.DataFrame, names: np.ndarray, frequencies: dict[str, Frequency], base: Frequency
) -> tuple[np.ndarray, dict[str, Frequency]]:
    periods = np.empty(len(frame), dtype=object)
    ref = frame["reference_period"].to_numpy(dtype=object)
    out_freqs: dict[str, Frequency] = {}
    for name in dict.fromkeys(names):
        rows = np.flatnonzero(names == name)
        freq = frequencies.get(str(name))
        parsed = [to_period(ref[i], freq, name=f"reference period of {name!r}") for i in rows]
        if freq is None:
            found = {Frequency.from_value(p.freqstr) for p in parsed}
            if len(found) != 1:
                raise NowcastDataError(
                    f"Reference periods of {name!r} mix frequencies "
                    f"{sorted(f.value for f in found)}; pass frequencies=."
                )
            freq = found.pop()
        if freq.is_higher_than(base):
            raise NowcastDataError(
                f"Series {name!r} is {freq.label}, higher than the base frequency ({base.label})."
            )
        out_freqs[str(name)] = freq
        for i, p in zip(rows, parsed, strict=True):
            periods[i] = p.asfreq(freq.pandas_freq)
    return periods, out_freqs


def _parse_vintage_dates(column: pd.Series) -> np.ndarray:
    try:
        vintages = pd.to_datetime(column)
    except (ValueError, TypeError) as exc:
        raise NowcastDataError("vintage_date cannot be parsed as dates.") from exc
    if vintages.isna().any():
        raise NowcastDataError("vintage_date has missing values.")
    if vintages.dt.tz is not None:
        vintages = vintages.dt.tz_localize(None)
    return vintages.dt.normalize().astype("datetime64[ns]").to_numpy()


def _parse_values(column: pd.Series) -> np.ndarray:
    try:
        values = pd.to_numeric(column, errors="raise").astype("float64").to_numpy()
    except (ValueError, TypeError) as exc:
        raise NowcastDataError("value must be numeric.") from exc
    if np.isinf(values).any():
        raise NowcastDataError("value contains infinite entries.")
    return values


def _check_duplicates(out: pd.DataFrame) -> None:
    dup = out.duplicated(["series", "_slot", "vintage_date"])
    if dup.any():
        first = out[dup].iloc[0]
        raise NowcastDataError(
            f"Duplicated record: series={first['series']!r}, "
            f"reference_period={first['reference_period']}, "
            f"vintage_date={pd.Timestamp(first['vintage_date']).date()} "
            f"({int(dup.sum())} duplicates)."
        )


def _sort_by_series(records: pd.DataFrame, order: Sequence[str]) -> pd.DataFrame:
    """Stable sort of ``records`` so that series appear in ``order``."""
    rank = records["series"].map({name: i for i, name in enumerate(order)})
    return records.iloc[np.argsort(rank.to_numpy(), kind="stable")].reset_index(drop=True)


def _panel_records(
    mfd: MixedFrequencyData,
    date: pd.Timestamp,
    known: dict[tuple[str, pd.Period], float],
    freqs: dict[str, Frequency],
    drop_unchanged: bool,
) -> list[tuple[str, pd.Period, pd.Timestamp, float]]:
    """Records contributed by one wide vintage (updates ``known`` and ``freqs``)."""
    rows: list[tuple[str, pd.Period, pd.Timestamp, float]] = []
    seen: set[tuple[str, pd.Period]] = set()
    for name in mfd.columns:
        freq = mfd.metadata[name].frequency
        if freqs.setdefault(name, freq) is not freq:
            raise NowcastDataError(f"Series {name!r} changes frequency across vintages.")
        native = mfd.to_native(name, dropna=True)
        for period, raw in native.items():
            key = (name, pd.Period(period))  # pyright: ignore[reportArgumentType]
            value = float(raw)
            seen.add(key)
            if drop_unchanged and key in known and _same(known[key], value):
                continue
            rows.append((key[0], key[1], date, value))
            known[key] = value
    for key, value in list(known.items()):
        if key not in seen and not np.isnan(value):
            rows.append((key[0], key[1], date, np.nan))
            known[key] = np.nan
    return rows


class VintageStore:
    """Store of real-time data vintages in long format.

    Parameters
    ----------
    records : pandas.DataFrame
        Long table with columns ``series``, ``reference_period``, ``vintage_date`` and
        ``value`` (optionally ``frequency``). ``reference_period`` may hold
        :class:`pandas.Period` objects, strings (``"2020Q1"``, ``"2020-03"``) or dates
        (then a frequency is required). ``value`` may be NaN, meaning the value was
        withdrawn in that vintage.
    frequencies : mapping of str to frequency or pandas.Series, optional
        Native frequency of each series. Otherwise taken from a ``frequency`` column or
        inferred from the reference periods.
    base_frequency : frequency, default "M"
        Frequency of the base grid used by :meth:`as_of`.

    Raises
    ------
    NowcastDataError
        If columns are missing or unknown, the table is empty, values are infinite or
        non-numeric, dates/periods cannot be parsed, a series mixes frequencies, or a
        ``(series, reference_period, vintage_date)`` triple is duplicated.

    See Also
    --------
    nowcastbox.vintages.pseudo_real_time : Vintages without revisions.

    Examples
    --------
    >>> import pandas as pd
    >>> from nowcastbox.vintages import VintageStore
    >>> records = pd.DataFrame(
    ...     {
    ...         "series": ["gdp", "gdp", "gdp"],
    ...         "reference_period": ["2020Q1", "2020Q1", "2020Q2"],
    ...         "vintage_date": ["2020-05-29", "2020-08-31", "2020-08-31"],
    ...         "value": [-1.5, -2.5, -9.7],
    ...     }
    ... )
    >>> store = VintageStore(records)
    >>> store.as_of("2020-06-30")["gdp"].dropna().tolist()
    [-1.5]
    >>> store.as_of("2020-09-30")["gdp"].dropna().tolist()
    [-2.5, -9.7]
    """

    def __init__(
        self,
        records: pd.DataFrame,
        *,
        frequencies: Mapping[str, FrequencyLike] | pd.Series | None = None,
        base_frequency: FrequencyLike = "M",
    ) -> None:
        self._base = Frequency.from_value(base_frequency)
        self._records, self._frequencies = self._normalise(
            records, coerce_frequency_map(frequencies)
        )
        self._series: list[str] = list(dict.fromkeys(self._records["series"]))

    # ------------------------------------------------------------------ validation
    def _normalise(
        self, records: pd.DataFrame, frequencies: dict[str, Frequency]
    ) -> tuple[pd.DataFrame, dict[str, Frequency]]:
        frame = _check_columns(records)
        names = _series_names(frame)
        _frequencies_from_column(frame, names, frequencies)
        unknown_freq = sorted(set(frequencies) - set(names))
        if unknown_freq:
            raise NowcastDataError(f"frequencies given for unknown series {unknown_freq}.")
        periods, out_freqs = _parse_periods(frame, names, frequencies, self._base)
        slots = _storage_slots(list(periods), self._base)
        out = pd.DataFrame(
            {
                "series": pd.Series(names, dtype=object),
                "reference_period": pd.Series(periods, dtype=object),
                "vintage_date": _parse_vintage_dates(frame["vintage_date"]),
                "value": _parse_values(frame["value"]),
                "_slot": slots,
            }
        )
        _check_duplicates(out)
        order = {n: i for i, n in enumerate(dict.fromkeys(names))}
        out["_order"] = out["series"].map(order)
        out = out.sort_values(["_order", "_slot", "vintage_date"], kind="stable")
        out = out.drop(columns="_order").reset_index(drop=True)
        return out, out_freqs

    # ------------------------------------------------------------------ properties
    @property
    def series(self) -> list[str]:
        """Names of the stored series (first-appearance order)."""
        return list(self._series)

    @property
    def base_frequency(self) -> Frequency:
        """Frequency of the base grid used by :meth:`as_of`."""
        return self._base

    @property
    def frequencies(self) -> pd.Series:
        """Native frequency of each series."""
        return pd.Series(
            [self._frequencies[n] for n in self._series],
            index=pd.Index(self._series),
            dtype=object,
            name="frequency",
        )

    @property
    def records(self) -> pd.DataFrame:
        """Copy of the normalised long table (``reference_period`` as Period objects)."""
        return self._records[list(RECORD_COLUMNS)].copy()

    @property
    def n_records(self) -> int:
        """Number of stored records."""
        return len(self._records)

    @property
    def first_vintage(self) -> pd.Timestamp:
        """Earliest vintage date."""
        return pd.Timestamp(self._records["vintage_date"].min())

    @property
    def last_vintage(self) -> pd.Timestamp:
        """Latest vintage date."""
        return pd.Timestamp(self._records["vintage_date"].max())

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, series: object) -> bool:
        return series in self._series

    def __repr__(self) -> str:
        return (
            f"VintageStore(n_series={len(self._series)}, n_records={len(self._records)}, "
            f"n_vintages={self._records['vintage_date'].nunique()}, "
            f"vintages={self.first_vintage.date()}..{self.last_vintage.date()})"
        )

    def _check_series(self, series: Sequence[str] | str | None) -> list[str]:
        if series is None:
            return list(self._series)
        names = [series] if isinstance(series, str) else list(series)
        unknown = sorted({n for n in names if n not in self._series})
        if unknown:
            raise NowcastDataError(f"Series {unknown} are not in the vintage store.")
        return names

    def vintage_dates(self, series: Sequence[str] | str | None = None) -> pd.DatetimeIndex:
        """Distinct vintage dates.

        Parameters
        ----------
        series : str or sequence of str, optional
            Restrict to these series.

        Returns
        -------
        pandas.DatetimeIndex
            Sorted unique dates.

        Examples
        --------
        >>> import pandas as pd
        >>> rec = pd.DataFrame(
        ...     {
        ...         "series": ["a"],
        ...         "reference_period": ["2020-01"],
        ...         "vintage_date": ["2020-02-15"],
        ...         "value": [1.0],
        ...     }
        ... )
        >>> VintageStore(rec).vintage_dates().strftime("%Y-%m-%d").tolist()
        ['2020-02-15']
        """
        names = self._check_series(series)
        sub = self._records[self._records["series"].isin(names)]
        return pd.DatetimeIndex(np.unique(sub["vintage_date"].to_numpy()), name="vintage_date")

    # ------------------------------------------------------------------ queries
    def _grid(self, names: list[str], start: object, end: object) -> pd.PeriodIndex:
        sub = self._records[self._records["series"].isin(names)]
        first = grid_bound(start, self._base, "start") if start is not None else sub["_slot"].min()
        last = grid_bound(end, self._base, "end") if end is not None else sub["_slot"].max()
        if first > last:
            raise ValueError(f"start ({first}) must not be after end ({last}).")
        return pd.period_range(first, last, freq=self._base.pandas_freq)

    def _known_at(self, date: pd.Timestamp, names: list[str]) -> pd.DataFrame:
        rec = self._records
        sub = rec[rec["series"].isin(names) & (rec["vintage_date"] <= date)]
        return sub.drop_duplicates(["series", "_slot"], keep="last")

    def _wide(self, table: pd.DataFrame, names: list[str], grid: pd.PeriodIndex) -> pd.DataFrame:
        frame = pd.DataFrame(np.nan, index=grid, columns=pd.Index(names), dtype="float64")
        positions = grid.get_indexer(pd.PeriodIndex(table["_slot"], freq=grid.freq))
        inside = positions >= 0
        cols = pd.Index(names).get_indexer(pd.Index(table["series"]))
        values = frame.to_numpy(copy=True)
        values[positions[inside], cols[inside]] = table["value"].to_numpy()[inside]
        return pd.DataFrame(values, index=grid, columns=pd.Index(names))

    def as_of(
        self,
        date: DateLike,
        *,
        series: Sequence[str] | str | None = None,
        start: object = None,
        end: object = None,
        as_mixed: bool = False,
        **metadata: Any,
    ) -> pd.DataFrame | MixedFrequencyData:
        """Panel as known at ``date`` (latest vintage on or before ``date``).

        Parameters
        ----------
        date : date-like
            Information date (inclusive).
        series : str or sequence of str, optional
            Series to return (default: all, in store order).
        start, end : period-like, optional
            Grid boundaries; default: the first and last slot stored for the selected
            series in **any** vintage, so that grids of different vintages coincide.
        as_mixed : bool, default False
            Return a :class:`~nowcastbox.core.data.MixedFrequencyData` (with the stored
            frequencies) instead of a DataFrame.
        **metadata
            Extra keyword arguments of :class:`~nowcastbox.core.data.MixedFrequencyData`
            (``release_delays``, ``blocks``, ...); only with ``as_mixed=True``.

        Returns
        -------
        pandas.DataFrame or MixedFrequencyData
            Wide panel on the base grid (:class:`pandas.PeriodIndex`); lower-frequency
            values in the last base period of their period; NaN where nothing was known.

        Raises
        ------
        NowcastDataError
            If a series is unknown.
        ValueError
            If ``metadata`` is given without ``as_mixed`` or ``start > end``.

        Examples
        --------
        >>> import pandas as pd
        >>> rec = pd.DataFrame(
        ...     {
        ...         "series": ["a", "a"],
        ...         "reference_period": ["2020-01", "2020-01"],
        ...         "vintage_date": ["2020-02-15", "2020-03-15"],
        ...         "value": [1.0, 1.2],
        ...     }
        ... )
        >>> VintageStore(rec).as_of("2020-03-01")["a"].tolist()
        [1.0]
        """
        if metadata and not as_mixed:
            raise ValueError("Metadata keyword arguments require as_mixed=True.")
        names = self._check_series(series)
        when = to_date(date)
        grid = self._grid(names, start, end)
        frame = self._wide(self._known_at(when, names), names, grid)
        if not as_mixed:
            return frame
        freqs = {n: self._frequencies[n] for n in names}
        return MixedFrequencyData(frame, freqs, base_frequency=self._base, **metadata)

    def latest(self, **kwargs: Any) -> pd.DataFrame | MixedFrequencyData:
        """Panel as known at the last vintage (see :meth:`as_of`).

        Parameters
        ----------
        **kwargs
            Keyword arguments of :meth:`as_of`.

        Returns
        -------
        pandas.DataFrame or MixedFrequencyData
            Latest panel.

        Examples
        --------
        >>> import pandas as pd
        >>> rec = pd.DataFrame(
        ...     {
        ...         "series": ["a", "a"],
        ...         "reference_period": ["2020-01", "2020-01"],
        ...         "vintage_date": ["2020-02-15", "2020-03-15"],
        ...         "value": [1.0, 1.2],
        ...     }
        ... )
        >>> VintageStore(rec).latest()["a"].tolist()
        [1.2]
        """
        return self.as_of(self.last_vintage, **kwargs)

    def vintage_matrix(self, series: str) -> pd.DataFrame:
        """Real-time data matrix of one series (Croushore & Stark, 2001).

        Parameters
        ----------
        series : str
            Series name.

        Returns
        -------
        pandas.DataFrame
            Rows: reference periods (native :class:`pandas.PeriodIndex`); columns: vintage
            dates of the series; cell :math:`(p, v)`: value of :math:`p` known at
            :math:`v` (NaN before its first release or after a withdrawal).

        Raises
        ------
        NowcastDataError
            If the series is unknown.

        Examples
        --------
        >>> import pandas as pd
        >>> rec = pd.DataFrame(
        ...     {
        ...         "series": ["a", "a", "a"],
        ...         "reference_period": ["2020-01", "2020-01", "2020-02"],
        ...         "vintage_date": ["2020-02-15", "2020-03-15", "2020-03-15"],
        ...         "value": [1.0, 1.2, 2.0],
        ...     }
        ... )
        >>> VintageStore(rec).vintage_matrix("a").to_numpy().tolist()
        [[1.0, 1.2], [nan, 2.0]]
        """
        self._check_series(series)
        freq = self._frequencies[series]
        sub = self._records[self._records["series"] == series]
        periods = pd.PeriodIndex(sorted(set(sub["reference_period"])), freq=freq.pandas_freq)
        vintages = pd.DatetimeIndex(np.unique(sub["vintage_date"].to_numpy()), name="vintage_date")
        rows = periods.get_indexer(
            pd.PeriodIndex(list(sub["reference_period"]), freq=freq.pandas_freq)
        )
        cols = vintages.get_indexer(sub["vintage_date"])
        vals = np.full((len(periods), len(vintages)), np.nan)
        has = np.zeros_like(vals, dtype=bool)
        vals[rows, cols] = sub["value"].to_numpy()
        has[rows, cols] = True
        for j in range(1, len(vintages)):
            vals[:, j] = np.where(has[:, j], vals[:, j], vals[:, j - 1])
        periods.name = "reference_period"
        return pd.DataFrame(vals, index=periods, columns=vintages)

    def revisions(self, series: str, *, include_unchanged: bool = False) -> pd.DataFrame:
        """Release history of one series: first release and every subsequent change.

        Parameters
        ----------
        series : str
            Series name.
        include_unchanged : bool, default False
            Keep records that repeat the previously known value.

        Returns
        -------
        pandas.DataFrame
            Columns ``reference_period``, ``vintage_date``, ``release_number`` (0 for
            the first release, :math:`k` for the :math:`k`-th change), ``value``,
            ``previous_value`` and ``revision`` (``value - previous_value``; NaN for the
            first release), sorted by period and vintage.

        Raises
        ------
        NowcastDataError
            If the series is unknown.

        Examples
        --------
        >>> import pandas as pd
        >>> rec = pd.DataFrame(
        ...     {
        ...         "series": ["a", "a"],
        ...         "reference_period": ["2020-01", "2020-01"],
        ...         "vintage_date": ["2020-02-15", "2020-03-15"],
        ...         "value": [1.0, 1.25],
        ...     }
        ... )
        >>> VintageStore(rec).revisions("a")["revision"].tolist()
        [nan, 0.25]
        """
        self._check_series(series)
        sub = self._records[self._records["series"] == series]
        rows: list[dict[str, Any]] = []
        for period, group in sub.groupby("_slot", sort=True):
            del period
            previous = np.nan
            number = -1
            for _, rec in group.iterrows():
                value = float(rec["value"])
                changed = number < 0 or not _same(value, previous)
                if changed:
                    number += 1
                elif not include_unchanged:
                    continue
                rows.append(
                    {
                        "reference_period": rec["reference_period"],
                        "vintage_date": rec["vintage_date"],
                        "release_number": number,
                        "value": value,
                        "previous_value": np.nan if number == 0 and changed else previous,
                        "revision": np.nan if number == 0 else value - previous,
                    }
                )
                previous = value
        out = pd.DataFrame(
            rows,
            columns=[
                "reference_period",
                "vintage_date",
                "release_number",
                "value",
                "previous_value",
                "revision",
            ],
        )
        out["reference_period"] = out["reference_period"].astype(object)
        out["release_number"] = out["release_number"].astype("int64")
        return out

    def nth_release(
        self,
        n: int = 0,
        *,
        series: Sequence[str] | str | None = None,
        start: object = None,
        end: object = None,
    ) -> pd.DataFrame:
        """Panel of the :math:`n`-th release of every observation.

        Parameters
        ----------
        n : int, default 0
            Release number: 0 = first release, 1 = first revision, ...; negative values
            count from the latest (-1 = latest value).
        series : str or sequence of str, optional
            Series to return.
        start, end : period-like, optional
            Grid boundaries (see :meth:`as_of`).

        Returns
        -------
        pandas.DataFrame
            Wide panel on the base grid; NaN where the observation has fewer releases.

        Raises
        ------
        ValueError
            If ``n`` is not an integer.

        Examples
        --------
        >>> import pandas as pd
        >>> rec = pd.DataFrame(
        ...     {
        ...         "series": ["a", "a"],
        ...         "reference_period": ["2020-01", "2020-01"],
        ...         "vintage_date": ["2020-02-15", "2020-03-15"],
        ...         "value": [1.0, 1.25],
        ...     }
        ... )
        >>> (
        ...     VintageStore(rec).nth_release(0)["a"].tolist(),
        ...     VintageStore(rec).nth_release(-1)["a"].tolist(),
        ... )
        ([1.0], [1.25])
        """
        if isinstance(n, bool) or not isinstance(n, (int, np.integer)):
            raise ValueError(f"n must be an integer, got {n!r}.")
        names = self._check_series(series)
        grid = self._grid(names, start, end)
        parts = []
        for name in names:
            hist = self.revisions(name)
            if n >= 0:
                picked = hist[hist["release_number"] == n]
            else:
                ranks = hist.groupby([p.ordinal for p in hist["reference_period"]], sort=False)[
                    "release_number"
                ].transform("max")
                picked = hist[hist["release_number"] == ranks + 1 + n]
            parts.append(
                pd.DataFrame(
                    {
                        "series": name,
                        "_slot": _storage_slots(list(picked["reference_period"]), self._base),
                        "value": picked["value"].to_numpy(),
                    }
                )
            )
        return self._wide(pd.concat(parts, ignore_index=True), names, grid)

    def revision_summary(self, series: Sequence[str] | str | None = None) -> pd.DataFrame:
        """Summary statistics of total revisions (latest minus first release).

        Parameters
        ----------
        series : str or sequence of str, optional
            Series to summarise (default: all).

        Returns
        -------
        pandas.DataFrame
            One row per series with ``n_periods``, ``n_revised`` (periods revised at
            least once), ``mean_n_revisions``, ``mean_revision``, ``mean_abs_revision``,
            ``rms_revision``, ``std_revision``, ``max_abs_revision`` and
            ``noise_to_signal`` (std of total revisions over std of latest values,
            Aruoba, 2008). Statistics use periods whose first and latest values are both
            non-missing; undefined statistics are NaN.

        Examples
        --------
        >>> import pandas as pd
        >>> rec = pd.DataFrame(
        ...     {
        ...         "series": ["a", "a", "a"],
        ...         "reference_period": ["2020-01", "2020-01", "2020-02"],
        ...         "vintage_date": ["2020-02-15", "2020-03-15", "2020-03-15"],
        ...         "value": [1.0, 1.5, 2.0],
        ...     }
        ... )
        >>> VintageStore(rec).revision_summary().loc["a", "mean_revision"]
        np.float64(0.25)
        """
        names = self._check_series(series)
        rows = {}
        for name in names:
            hist = self.revisions(name)
            key = [p.ordinal for p in hist["reference_period"]]
            grouped = hist.groupby(key, sort=False)
            first = grouped["value"].agg(lambda v: v.iloc[0])
            latest = grouped["value"].agg(lambda v: v.iloc[-1])
            n_rev = grouped["release_number"].max()
            total = (latest - first).dropna()
            ok_latest = latest[total.index]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                std_r = float(total.std(ddof=1)) if len(total) > 1 else np.nan
                std_x = float(ok_latest.std(ddof=1)) if len(ok_latest) > 1 else np.nan
                rows[name] = {
                    "n_periods": len(first),
                    "n_revised": int((n_rev > 0).sum()),
                    "mean_n_revisions": float(n_rev.mean()),
                    "mean_revision": float(total.mean()) if len(total) else np.nan,
                    "mean_abs_revision": float(total.abs().mean()) if len(total) else np.nan,
                    "rms_revision": float(np.sqrt((total**2).mean())) if len(total) else np.nan,
                    "std_revision": std_r,
                    "max_abs_revision": float(total.abs().max()) if len(total) else np.nan,
                    "noise_to_signal": std_r / std_x if std_x and std_x > 0 else np.nan,
                }
        frame = pd.DataFrame.from_dict(rows, orient="index")
        frame.index = pd.Index(frame.index, name="series")
        return frame

    # ------------------------------------------------------------------ derivation
    def to_frame(self) -> pd.DataFrame:
        """Serialisable long table.

        Returns
        -------
        pandas.DataFrame
            Columns ``series``, ``frequency`` (code such as ``"M"``/``"Q"``),
            ``reference_period`` (string), ``vintage_date`` and ``value``.

        Examples
        --------
        >>> import pandas as pd
        >>> rec = pd.DataFrame(
        ...     {
        ...         "series": ["g"],
        ...         "reference_period": ["2020Q1"],
        ...         "vintage_date": ["2020-05-29"],
        ...         "value": [1.0],
        ...     }
        ... )
        >>> VintageStore(rec).to_frame()[["frequency", "reference_period"]].values.tolist()
        [['Q', '2020Q1']]
        """
        rec = self._records
        return pd.DataFrame(
            {
                "series": rec["series"].astype(str),
                "frequency": [self._frequencies[n].value for n in rec["series"]],
                "reference_period": [str(p) for p in rec["reference_period"]],
                "vintage_date": rec["vintage_date"].to_numpy(),
                "value": rec["value"].to_numpy(),
            }
        )

    def select(self, series: Sequence[str] | str) -> VintageStore:
        """Store restricted to some series.

        Parameters
        ----------
        series : str or sequence of str
            Series to keep.

        Returns
        -------
        VintageStore
            New store.

        Examples
        --------
        >>> import pandas as pd
        >>> rec = pd.DataFrame(
        ...     {
        ...         "series": ["a", "b"],
        ...         "reference_period": ["2020-01", "2020-01"],
        ...         "vintage_date": ["2020-02-15", "2020-02-15"],
        ...         "value": [1.0, 2.0],
        ...     }
        ... )
        >>> VintageStore(rec).select("b").series
        ['b']
        """
        names = self._check_series(series)
        frame = self.to_frame()
        return VintageStore(frame[frame["series"].isin(names)], base_frequency=self._base)

    def add(self, records: pd.DataFrame | VintageStore) -> VintageStore:
        """New store with extra records appended.

        Parameters
        ----------
        records : pandas.DataFrame or VintageStore
            Records in the long format.

        Returns
        -------
        VintageStore
            Combined store (duplicated triples raise).

        Raises
        ------
        NowcastDataError
            If records duplicate existing ones or frequencies conflict.

        Examples
        --------
        >>> import pandas as pd
        >>> rec = pd.DataFrame(
        ...     {
        ...         "series": ["a"],
        ...         "reference_period": ["2020-01"],
        ...         "vintage_date": ["2020-02-15"],
        ...         "value": [1.0],
        ...     }
        ... )
        >>> more = rec.assign(vintage_date="2020-03-15", value=1.1)
        >>> len(VintageStore(rec).add(more))
        2
        """
        other = (
            records
            if isinstance(records, VintageStore)
            else VintageStore(records, base_frequency=self._base)
        )
        for name in set(other.series) & set(self._series):
            if other._frequencies[name] is not self._frequencies[name]:
                raise NowcastDataError(f"Frequency of {name!r} differs between stores.")
        return VintageStore(
            pd.concat([self.to_frame(), other.to_frame()], ignore_index=True),
            base_frequency=self._base,
        )

    def equals(self, other: object) -> bool:
        """Whether two stores hold the same records.

        Parameters
        ----------
        other : object
            Object to compare.

        Returns
        -------
        bool
            ``True`` for identical records, frequencies and base frequency.

        Examples
        --------
        >>> import pandas as pd
        >>> rec = pd.DataFrame(
        ...     {
        ...         "series": ["a"],
        ...         "reference_period": ["2020-01"],
        ...         "vintage_date": ["2020-02-15"],
        ...         "value": [1.0],
        ...     }
        ... )
        >>> VintageStore(rec).equals(VintageStore(rec))
        True
        """
        if not isinstance(other, VintageStore) or other._base is not self._base:
            return False
        a = self.to_frame().sort_values(["series", "reference_period", "vintage_date"])
        b = other.to_frame().sort_values(["series", "reference_period", "vintage_date"])
        return bool(a.reset_index(drop=True).equals(b.reset_index(drop=True)))

    # ------------------------------------------------------------------ constructors
    @classmethod
    def from_vintages(
        cls,
        vintages: Mapping[DateLike, pd.DataFrame | MixedFrequencyData],
        *,
        frequencies: FrequencySpec | None = None,
        drop_unchanged: bool = True,
    ) -> VintageStore:
        """Build a store from a collection of wide vintages.

        Parameters
        ----------
        vintages : mapping of date to DataFrame or MixedFrequencyData
            Each value is the full panel as published on its date (base grid).
        frequencies : frequency specification, optional
            Per-series frequencies for DataFrame vintages (inferred when omitted).
        drop_unchanged : bool, default True
            Store only first releases and changes (the ALFRED convention). A value that
            disappears from a later vintage is recorded as NaN (withdrawn).

        Returns
        -------
        VintageStore
            Store such that ``store.as_of(date)`` reproduces each vintage (on the
            union grid).

        Raises
        ------
        NowcastDataError
            If ``vintages`` is empty, panels use different base frequencies or a series
            changes frequency.

        Examples
        --------
        >>> import pandas as pd
        >>> idx = pd.period_range("2020-01", periods=2, freq="M")
        >>> v1 = pd.DataFrame({"a": [1.0, None]}, index=idx)
        >>> v2 = pd.DataFrame({"a": [1.1, 2.0]}, index=idx)
        >>> store = VintageStore.from_vintages(
        ...     {"2020-02-15": v1, "2020-03-15": v2}, frequencies="M"
        ... )
        >>> store.as_of("2020-02-20")["a"].tolist()
        [1.0, nan]
        """
        if not vintages:
            raise NowcastDataError("vintages is empty.")
        parsed = sorted(
            ((to_date(d, name="vintage date"), v) for d, v in vintages.items()),
            key=lambda item: item[0],
        )
        dates = [d for d, _ in parsed]
        if len(set(dates)) != len(dates):
            raise NowcastDataError("Duplicated vintage dates.")
        known: dict[tuple[str, pd.Period], float] = {}
        freqs: dict[str, Frequency] = {}
        bases: set[Frequency] = set()
        order: list[str] = []
        rows: list[tuple[str, pd.Period, pd.Timestamp, float]] = []
        for date, panel in parsed:
            with warnings.catch_warnings():
                # early vintages routinely contain series with nothing released yet
                warnings.filterwarnings(
                    "ignore", "Series without any observation", DataQualityWarning
                )
                mfd = as_mixed_frequency_data(
                    panel, None if isinstance(panel, MixedFrequencyData) else frequencies
                )
            bases.add(mfd.base_frequency)
            if len(bases) > 1:
                raise NowcastDataError("Vintages use different base frequencies.")
            rows += _panel_records(mfd, date, known, freqs, drop_unchanged)
            order += [c for c in mfd.columns if c not in order]
        if not rows:
            raise NowcastDataError("The vintages contain no observations.")
        records = pd.DataFrame(rows, columns=list(RECORD_COLUMNS))
        records["reference_period"] = records["reference_period"].astype(object)
        records = _sort_by_series(records, order)
        return cls(records, frequencies=freqs, base_frequency=bases.pop())

    @classmethod
    def from_calendar(
        cls,
        data: MixedFrequencyData | pd.DataFrame,
        calendar: ReleaseCalendar | Mapping[str, object] | pd.Series | None = None,
        *,
        frequency: FrequencySpec | None = None,
    ) -> VintageStore:
        """Store without revisions: each final value dated by its release date.

        ``store.as_of(v)`` then equals ``pseudo_real_time(data, calendar, v)``.

        Parameters
        ----------
        data : MixedFrequencyData or pandas.DataFrame
            Final dataset.
        calendar : ReleaseCalendar, mapping or Series, optional
            Release rule (``None`` uses the ``release_delays`` metadata).
        frequency : frequency specification, optional
            Frequencies for DataFrame input.

        Returns
        -------
        VintageStore
            The store.

        Raises
        ------
        NowcastDataError
            If release dates are missing or the data have no observation.

        Examples
        --------
        >>> import pandas as pd
        >>> idx = pd.period_range("2020-01", periods=2, freq="M")
        >>> store = VintageStore.from_calendar(
        ...     pd.DataFrame({"a": [1.0, 2.0]}, index=idx), {"a": 20}
        ... )
        >>> store.vintage_dates().strftime("%Y-%m-%d").tolist()
        ['2020-02-20', '2020-03-20']
        """
        mfd = as_mixed_frequency_data(data, frequency)
        if calendar is None:
            cal = ReleaseCalendar.from_data(mfd)
        elif isinstance(calendar, ReleaseCalendar):
            cal = calendar
        else:
            cal = ReleaseCalendar.from_data(mfd, calendar)
        table = cal.schedule(mfd, observed_only=True)
        if table.empty:
            raise NowcastDataError("The data contain no observations.")
        records = pd.DataFrame(
            {
                "series": table["series"],
                "reference_period": table["reference_period"],
                "vintage_date": table["release_date"],
                "value": table["value"],
            }
        )
        records = _sort_by_series(records, mfd.columns)
        freqs = {n: m.frequency for n, m in mfd.metadata.items()}
        return cls(records, frequencies=freqs, base_frequency=mfd.base_frequency)

    # ------------------------------------------------------------------ I/O
    def to_csv(self, path: str | Path, **kwargs: Any) -> None:
        """Write the long table (see :meth:`to_frame`) to CSV.

        Parameters
        ----------
        path : str or Path
            Destination file.
        **kwargs
            Passed to :meth:`pandas.DataFrame.to_csv` (``index`` is always False).

        Examples
        --------
        >>> import pandas as pd, tempfile, os
        >>> rec = pd.DataFrame(
        ...     {
        ...         "series": ["a"],
        ...         "reference_period": ["2020-01"],
        ...         "vintage_date": ["2020-02-15"],
        ...         "value": [1.0],
        ...     }
        ... )
        >>> path = os.path.join(tempfile.mkdtemp(), "v.csv")
        >>> VintageStore(rec).to_csv(path)
        >>> VintageStore.from_csv(path).equals(VintageStore(rec))
        True
        """
        kwargs.pop("index", None)
        self.to_frame().to_csv(path, index=False, date_format="%Y-%m-%d", **kwargs)

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        *,
        frequencies: Mapping[str, FrequencyLike] | pd.Series | None = None,
        base_frequency: FrequencyLike = "M",
        **kwargs: Any,
    ) -> VintageStore:
        """Read a store written by :meth:`to_csv` (or any long CSV).

        Parameters
        ----------
        path : str or Path
            Source file.
        frequencies : mapping or Series, optional
            Frequencies (otherwise from the ``frequency`` column or inferred).
        base_frequency : frequency, default "M"
            Base grid frequency.
        **kwargs
            Passed to :func:`pandas.read_csv`.

        Returns
        -------
        VintageStore
            The store.

        Examples
        --------
        See :meth:`to_csv`.
        """
        frame = pd.read_csv(
            path, dtype={"series": str, "reference_period": str, "frequency": str}, **kwargs
        )
        return cls(frame, frequencies=frequencies, base_frequency=base_frequency)

    def to_parquet(self, path: str | Path, **kwargs: Any) -> None:
        """Write the long table to Parquet (requires ``pyarrow`` or ``fastparquet``).

        Parameters
        ----------
        path : str or Path
            Destination file.
        **kwargs
            Passed to :meth:`pandas.DataFrame.to_parquet` (``index`` is always False).

        Raises
        ------
        ImportError
            If no Parquet engine is installed.

        Examples
        --------
        >>> import pandas as pd, tempfile, os
        >>> rec = pd.DataFrame(
        ...     {
        ...         "series": ["a"],
        ...         "reference_period": ["2020-01"],
        ...         "vintage_date": ["2020-02-15"],
        ...         "value": [1.0],
        ...     }
        ... )
        >>> path = os.path.join(tempfile.mkdtemp(), "v.parquet")
        >>> VintageStore(rec).to_parquet(path)
        >>> VintageStore.from_parquet(path).equals(VintageStore(rec))
        True
        """
        kwargs.pop("index", None)
        try:
            self.to_frame().to_parquet(path, index=False, **kwargs)
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError("Writing Parquet requires 'pyarrow' (pip install pyarrow).") from exc

    @classmethod
    def from_parquet(
        cls,
        path: str | Path,
        *,
        frequencies: Mapping[str, FrequencyLike] | pd.Series | None = None,
        base_frequency: FrequencyLike = "M",
        **kwargs: Any,
    ) -> VintageStore:
        """Read a store written by :meth:`to_parquet`.

        Parameters
        ----------
        path : str or Path
            Source file.
        frequencies : mapping or Series, optional
            Frequencies (otherwise from the ``frequency`` column or inferred).
        base_frequency : frequency, default "M"
            Base grid frequency.
        **kwargs
            Passed to :func:`pandas.read_parquet`.

        Returns
        -------
        VintageStore
            The store.

        Raises
        ------
        ImportError
            If no Parquet engine is installed.

        Examples
        --------
        See :meth:`to_parquet`.
        """
        try:
            frame = pd.read_parquet(path, **kwargs)
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError("Reading Parquet requires 'pyarrow' (pip install pyarrow).") from exc
        return cls(frame, frequencies=frequencies, base_frequency=base_frequency)
