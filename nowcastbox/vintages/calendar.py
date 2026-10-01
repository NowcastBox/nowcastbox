r"""Release calendars: the publication date of every observation.

A :class:`ReleaseCalendar` answers the question *"on which date did the value of series*
:math:`i` *for reference period* :math:`p` *become public?"*. Two rules are supported and
can be combined per series:

* **Stylised delays** (Giannone, Reichlin & Small, 2008; Bańbura & Rünstler, 2011): the
  observation of period :math:`p` is released ``release_delay`` days after the last day
  of :math:`p`,

  .. math:: d_i(p) = \operatorname{end}(p) + \delta_i .

  This is the rule used by :meth:`MixedFrequencyData.as_of
  <nowcastbox.core.data.MixedFrequencyData.as_of>`.
* **Explicit release dates** (e.g. a statistical office's published calendar): a table
  ``(series, reference period) -> release date``. When a series has both, the explicit
  date wins and the delay is the fallback for periods not in the table.

An observation is *available* at an information date :math:`v` iff
:math:`d_i(p) \le v` (inclusive). Information sets are therefore nested,
:math:`\Omega_v \subseteq \Omega_{v'}` for :math:`v \le v'`, and the releases
between two vintages, :math:`\{(i, p) : v < d_i(p) \le v'\}`, are exactly the data
whose *news* drive the nowcast revision (Bańbura & Modugno, 2014).

References
----------
Giannone, D., Reichlin, L. & Small, D. (2008). Nowcasting: The real-time informational
content of macroeconomic data. *Journal of Monetary Economics*, 55(4), 665-676.

Bańbura, M. & Rünstler, G. (2011). A look into the factor model black box: Publication
lags and the role of hard and soft data in forecasting GDP. *International Journal of
Forecasting*, 27(2), 333-346.

Bańbura, M. & Modugno, M. (2014). Maximum likelihood estimation of factor models on
datasets with arbitrary pattern of missing data. *Journal of Applied Econometrics*,
29(1), 133-160.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from nowcastbox._logging import get_logger
from nowcastbox.core.data import MixedFrequencyData, as_mixed_frequency_data
from nowcastbox.core.exceptions import DataQualityWarning, NowcastDataError
from nowcastbox.core.frequency import Frequency, FrequencyLike
from nowcastbox.vintages._utils import (
    DateLike,
    PeriodLike,
    coerce_delay,
    coerce_frequency_map,
    native_period_end,
    to_date,
    to_period,
)

__all__ = ["RELEASE_COLUMNS", "ReleaseCalendar"]

logger = get_logger(__name__)

#: Columns of the long release tables returned by :class:`ReleaseCalendar`.
RELEASE_COLUMNS: tuple[str, ...] = ("release_date", "series", "reference_period")

DelaySpec = Mapping[str, object] | pd.Series


class ReleaseCalendar:
    r"""Per-series publication calendar.

    Parameters
    ----------
    delays : mapping of str to int or pandas.Series, optional
        Release delay of each series in **days after the end of the reference period**
        (integers; a negative delay means the value is published before the period
        ends, e.g. mid-month surveys; missing values mean "no delay rule").
    release_dates : mapping or pandas.DataFrame, optional
        Explicit release dates. Either ``{series: {reference_period: date}}`` (inner
        mappings may be :class:`pandas.Series` indexed by period) or a long DataFrame
        with columns ``series``, ``reference_period`` and ``release_date``. Explicit
        dates override the delay rule for the periods they cover.
    frequencies : mapping of str to frequency or pandas.Series, optional
        Native frequency of each series. Needed by delay-based queries that do not
        receive a data panel (:meth:`available_at`, :meth:`releases_between` without
        ``data``). For explicit-only series it is inferred from the periods.

    Raises
    ------
    ValueError
        If no series is given or a delay is invalid.
    NowcastDataError
        If explicit release dates are malformed (unparseable or duplicated periods,
        inconsistent frequencies).

    See Also
    --------
    nowcastbox.vintages.pseudo_real_time : Apply a calendar to a data panel.
    nowcastbox.core.data.MixedFrequencyData.as_of : Delay-only special case.

    Notes
    -----
    The release date of the observation of reference period :math:`p` of series
    :math:`i` is :math:`d_i(p) = \operatorname{end}(p) + \delta_i` (delay rule), where
    :math:`\operatorname{end}(p)` is the last calendar day of :math:`p` (31 March for
    2020Q1), or the explicit date when one is given. An observation belongs to the
    information set of date :math:`v` iff :math:`d_i(p) \le v`.

    Examples
    --------
    >>> import pandas as pd
    >>> from nowcastbox.vintages import ReleaseCalendar
    >>> cal = ReleaseCalendar({"ip": 40, "gdp": 60}, frequencies={"ip": "M", "gdp": "Q"})
    >>> cal.release_date("gdp", "2020Q1")
    Timestamp('2020-05-30 00:00:00')
    >>> cal.available_at("2020-06-15").astype(str).to_dict()
    {'ip': '2020-04', 'gdp': '2020Q1'}
    """

    def __init__(
        self,
        delays: DelaySpec | None = None,
        release_dates: Mapping[str, object] | pd.DataFrame | None = None,
        *,
        frequencies: Mapping[str, FrequencyLike] | pd.Series | None = None,
    ) -> None:
        self._delays: dict[str, int] = self._parse_delays(delays)
        self._frequencies: dict[str, Frequency] = coerce_frequency_map(frequencies)
        self._explicit: dict[str, pd.Series] = self._parse_release_dates(release_dates)
        names = list(self._delays)
        names += [n for n in self._explicit if n not in self._delays]
        if not names:
            raise ValueError("A ReleaseCalendar needs at least one series (delays or dates).")
        self._series: list[str] = names

    # ------------------------------------------------------------------ parsing
    @staticmethod
    def _parse_delays(delays: DelaySpec | None) -> dict[str, int]:
        if delays is None:
            return {}
        if not isinstance(delays, (Mapping, pd.Series)):
            raise ValueError(
                f"delays must be a mapping or a pandas Series, got {type(delays).__name__}."
            )
        out: dict[str, int] = {}
        for key, value in delays.items():
            name = str(key)
            delay = coerce_delay(value, name)
            if delay is not None:
                out[name] = delay
        return out

    def _parse_release_dates(
        self, release_dates: Mapping[str, object] | pd.DataFrame | None
    ) -> dict[str, pd.Series]:
        if release_dates is None:
            return {}
        if isinstance(release_dates, pd.DataFrame):
            required = {"series", "reference_period", "release_date"}
            missing = sorted(required - set(map(str, release_dates.columns)))
            if missing:
                raise NowcastDataError(f"release_dates frame lacks columns {missing}.")
            grouped: dict[str, object] = {
                str(name): pd.Series(
                    list(group["release_date"]), index=list(group["reference_period"])
                )
                for name, group in release_dates.groupby("series", sort=False)
            }
        elif isinstance(release_dates, Mapping):
            grouped = {str(k): v for k, v in release_dates.items()}
        else:
            raise NowcastDataError(
                "release_dates must be a mapping or a long DataFrame, got "
                f"{type(release_dates).__name__}."
            )
        return {name: self._parse_series_dates(name, table) for name, table in grouped.items()}

    def _parse_series_dates(self, name: str, table: object) -> pd.Series:
        if isinstance(table, (pd.Series, Mapping)):
            items = list(table.items())
        else:
            raise NowcastDataError(
                f"Release dates of {name!r} must be a mapping or Series, got "
                f"{type(table).__name__}."
            )
        if not items:
            raise NowcastDataError(f"No release dates given for {name!r}.")
        freq = self._frequencies.get(name)
        periods = [to_period(p, freq, name=f"reference period of {name!r}") for p, _ in items]
        if freq is None:
            freqs = {Frequency.from_value(p.freqstr) for p in periods}
            if len(freqs) != 1:
                raise NowcastDataError(
                    f"Reference periods of {name!r} mix frequencies {sorted(f.value for f in freqs)}."
                )
            freq = freqs.pop()
            self._frequencies[name] = freq
            periods = [p.asfreq(freq.pandas_freq) for p in periods]
        dates = [to_date(d, name=f"release date of {name!r}") for _, d in items]
        index = pd.PeriodIndex(periods, freq=freq.pandas_freq)
        if index.has_duplicates:
            dup = sorted({str(p) for p in index[index.duplicated()]})
            raise NowcastDataError(f"Duplicated reference periods for {name!r}: {dup}.")
        out = pd.Series(pd.DatetimeIndex(dates), index=index, name=name).sort_index()
        early = [str(p) for p, d in out.items() if d < native_period_end(p)]  # pyright: ignore[reportArgumentType]
        if early:
            warnings.warn(
                f"Release dates of {name!r} precede the end of the reference period for "
                f"{early[:5]}{'...' if len(early) > 5 else ''}.",
                DataQualityWarning,
                stacklevel=4,
            )
        return out

    # ------------------------------------------------------------------ constructors
    @classmethod
    def from_data(
        cls,
        data: MixedFrequencyData | pd.DataFrame,
        release_delays: DelaySpec | None = None,
        *,
        frequency: object = None,
    ) -> ReleaseCalendar:
        """Build a delay calendar from the metadata of a panel.

        Parameters
        ----------
        data : MixedFrequencyData or pandas.DataFrame
            Panel. Its ``release_delays`` metadata and frequencies are used.
        release_delays : mapping or Series, optional
            Delays (days) overriding the metadata.
        frequency : frequency specification, optional
            Per-series frequencies (DataFrame input only).

        Returns
        -------
        ReleaseCalendar
            Calendar covering every series of ``data``.

        Raises
        ------
        NowcastDataError
            If a series has no release delay or overrides name unknown series.

        Examples
        --------
        >>> import pandas as pd
        >>> from nowcastbox.core import MixedFrequencyData
        >>> idx = pd.period_range("2020-01", periods=3, freq="M")
        >>> mfd = MixedFrequencyData(pd.DataFrame({"a": [1.0, 2.0, 3.0]}, index=idx), "M")
        >>> ReleaseCalendar.from_data(mfd, {"a": 15}).delays.to_dict()
        {'a': 15}
        """
        mfd = as_mixed_frequency_data(data, frequency)  # pyright: ignore[reportArgumentType]
        delays: dict[str, object] = {
            name: meta.release_delay for name, meta in mfd.metadata.items()
        }
        if release_delays is not None:
            overrides = cls._parse_delays(release_delays)
            unknown = sorted(set(overrides) - set(delays))
            if unknown:
                raise NowcastDataError(f"release_delays given for unknown series {unknown}.")
            delays.update(overrides)
        missing = [name for name, d in delays.items() if d is None]
        if missing:
            raise NowcastDataError(
                f"No release delay for series {missing}; pass delays or a ReleaseCalendar."
            )
        return cls(delays, frequencies={n: m.frequency for n, m in mfd.metadata.items()})

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        *,
        series_column: str = "series",
        period_column: str = "reference_period",
        date_column: str = "release_date",
        delays: DelaySpec | None = None,
        frequencies: Mapping[str, FrequencyLike] | pd.Series | None = None,
    ) -> ReleaseCalendar:
        """Build a calendar of explicit release dates from a long table.

        Parameters
        ----------
        frame : pandas.DataFrame
            One row per release.
        series_column, period_column, date_column : str
            Column names of the series, reference period and release date.
        delays : mapping or Series, optional
            Fallback delays for periods absent from the table.
        frequencies : mapping or Series, optional
            Native frequencies (inferred from the periods when omitted).

        Returns
        -------
        ReleaseCalendar
            The calendar.

        Raises
        ------
        NowcastDataError
            If columns are missing or the table is malformed.

        Examples
        --------
        >>> import pandas as pd
        >>> table = pd.DataFrame(
        ...     {"series": ["gdp"], "reference_period": ["2020Q1"], "release_date": ["2020-05-29"]}
        ... )
        >>> ReleaseCalendar.from_frame(table).release_date("gdp", "2020Q1")
        Timestamp('2020-05-29 00:00:00')
        """
        missing = [c for c in (series_column, period_column, date_column) if c not in frame]
        if missing:
            raise NowcastDataError(f"frame lacks columns {missing}.")
        renamed = frame[[series_column, period_column, date_column]].copy()
        renamed.columns = ["series", "reference_period", "release_date"]
        return cls(delays, renamed, frequencies=frequencies)

    # ------------------------------------------------------------------ properties
    @property
    def series(self) -> list[str]:
        """Names of the series covered by the calendar."""
        return list(self._series)

    @property
    def delays(self) -> pd.Series:
        """Release delays in days (``Int64``; ``<NA>`` for explicit-only series)."""
        return pd.Series(
            [self._delays.get(n) for n in self._series],
            index=pd.Index(self._series),
            dtype="Int64",
            name="release_delay",
        )

    @property
    def frequencies(self) -> pd.Series:
        """Known native frequencies (``None`` when still unknown)."""
        return pd.Series(
            [self._frequencies.get(n) for n in self._series],
            index=pd.Index(self._series),
            dtype=object,
            name="frequency",
        )

    def has_explicit_dates(self, series: str) -> bool:
        """Whether ``series`` has explicit release dates.

        Parameters
        ----------
        series : str
            Series name.

        Returns
        -------
        bool
            ``True`` if an explicit release table was given for ``series``.

        Examples
        --------
        >>> ReleaseCalendar({"a": 5}, frequencies={"a": "M"}).has_explicit_dates("a")
        False
        """
        self._check_series([series])
        return series in self._explicit

    def explicit_dates(self, series: str) -> pd.Series:
        """Explicit release dates of ``series`` (indexed by reference period).

        Parameters
        ----------
        series : str
            Series name.

        Returns
        -------
        pandas.Series
            Release dates (empty when the series follows the delay rule only).

        Examples
        --------
        >>> cal = ReleaseCalendar(release_dates={"a": {"2020-01": "2020-02-10"}})
        >>> cal.explicit_dates("a").astype(str).to_dict()
        {Period('2020-01', 'M'): '2020-02-10'}
        """
        self._check_series([series])
        table = self._explicit.get(series)
        if table is None:
            return pd.Series([], dtype="datetime64[ns]", name=series)
        return table.copy()

    def __contains__(self, series: object) -> bool:
        return series in self._series

    def __len__(self) -> int:
        return len(self._series)

    def __repr__(self) -> str:
        n_explicit = len(self._explicit)
        return (
            f"ReleaseCalendar(n_series={len(self._series)}, delay_rules={len(self._delays)}, "
            f"explicit_tables={n_explicit})"
        )

    def to_frame(self) -> pd.DataFrame:
        """Summary table: one row per series.

        Returns
        -------
        pandas.DataFrame
            Columns ``frequency``, ``release_delay``, ``n_explicit_dates``,
            ``first_explicit_period`` and ``last_explicit_period``.

        Examples
        --------
        >>> ReleaseCalendar({"a": 5}, frequencies={"a": "M"}).to_frame()["release_delay"].tolist()
        [5]
        """
        rows = []
        for name in self._series:
            table = self._explicit.get(name)
            freq = self._frequencies.get(name)
            rows.append(
                {
                    "frequency": None if freq is None else freq.value,
                    "release_delay": self._delays.get(name),
                    "n_explicit_dates": 0 if table is None else len(table),
                    "first_explicit_period": None if table is None else table.index[0],
                    "last_explicit_period": None if table is None else table.index[-1],
                }
            )
        frame = pd.DataFrame(rows, index=pd.Index(self._series, name="series"))
        frame["release_delay"] = frame["release_delay"].astype("Int64")
        return frame

    def with_delays(self, delays: DelaySpec) -> ReleaseCalendar:
        """Return a copy with delay rules added or replaced.

        Parameters
        ----------
        delays : mapping or Series
            New delays (days).

        Returns
        -------
        ReleaseCalendar
            New calendar.

        Examples
        --------
        >>> cal = ReleaseCalendar({"a": 5}, frequencies={"a": "M"}).with_delays({"b": 9})
        >>> cal.series
        ['a', 'b']
        """
        new = self._copy()
        added = self._parse_delays(delays)
        new._delays.update(added)
        new._series += [n for n in added if n not in new._series]
        return new

    def with_frequencies(
        self, frequencies: Mapping[str, FrequencyLike] | pd.Series
    ) -> ReleaseCalendar:
        """Return a copy with native frequencies set.

        Parameters
        ----------
        frequencies : mapping or Series
            Frequencies by series name (unknown names raise).

        Returns
        -------
        ReleaseCalendar
            New calendar.

        Raises
        ------
        NowcastDataError
            If a name is not in the calendar or conflicts with explicit-date periods.

        Examples
        --------
        >>> ReleaseCalendar({"a": 5}).with_frequencies({"a": "Q"}).frequencies.astype(str).tolist()
        ['Q']
        """
        parsed = coerce_frequency_map(frequencies)
        self._check_series(list(parsed))
        new = self._copy()
        for name, freq in parsed.items():
            table = new._explicit.get(name)
            if (
                table is not None
                and Frequency.from_value(pd.PeriodIndex(table.index).freqstr) is not freq
            ):
                raise NowcastDataError(
                    f"Frequency {freq.value!r} of {name!r} conflicts with its release table."
                )
            new._frequencies[name] = freq
        return new

    def _copy(self) -> ReleaseCalendar:
        new = object.__new__(ReleaseCalendar)
        new._delays = dict(self._delays)
        new._frequencies = dict(self._frequencies)
        new._explicit = {k: v.copy() for k, v in self._explicit.items()}
        new._series = list(self._series)
        return new

    # ------------------------------------------------------------------ helpers
    def _check_series(self, names: Iterable[str]) -> None:
        unknown = sorted({n for n in names if n not in self._series})
        if unknown:
            raise NowcastDataError(f"Series {unknown} are not in the release calendar.")

    def _frequency(self, series: str, data: MixedFrequencyData | None = None) -> Frequency:
        if data is not None and series in data.metadata:
            freq = data.metadata[series].frequency
            known = self._frequencies.get(series)
            if known is not None and known is not freq:
                raise NowcastDataError(
                    f"Series {series!r} is {freq.label} in the data but {known.label} in the "
                    "release calendar."
                )
            return freq
        freq = self._frequencies.get(series)
        if freq is None:
            raise NowcastDataError(
                f"Native frequency of {series!r} is unknown; pass frequencies= to the "
                "calendar or a data panel."
            )
        return freq

    def _dates_for(
        self, series: str, periods: pd.PeriodIndex, ends: np.ndarray | None = None
    ) -> pd.DatetimeIndex:
        """Vectorised release dates (NaT where unknown).

        ``ends`` (optional) are the precomputed last days of ``periods`` as
        ``datetime64[ns]`` (shared by every series of one frequency in
        :meth:`release_mask`).
        """
        delay = self._delays.get(series)
        table = self._explicit.get(series)
        if not len(periods):
            return pd.DatetimeIndex([], dtype="datetime64[ns]")
        fallback = None
        if delay is not None:
            if ends is None:
                ends = _period_ends(periods)
            fallback = ends + np.timedelta64(delay, "D")
            if table is None:
                return pd.DatetimeIndex(fallback)
        result = np.full(len(periods), np.datetime64("NaT"), dtype="datetime64[ns]")
        if table is not None:
            result = table.reindex(periods).to_numpy(dtype="datetime64[ns]")
        if fallback is not None:
            result = np.where(np.isnat(result), fallback, result)
        return pd.DatetimeIndex(result)

    # ------------------------------------------------------------------ queries
    def release_date(self, series: str, reference_period: PeriodLike) -> pd.Timestamp:
        """Release date of one observation.

        Parameters
        ----------
        series : str
            Series name.
        reference_period : period-like
            Reference period (``"2020Q1"``, ``"2020-03"``, ``pandas.Period``...).

        Returns
        -------
        pandas.Timestamp
            Publication date.

        Raises
        ------
        NowcastDataError
            If the series is unknown, its frequency is unknown, or the period has no
            explicit date and the series no delay rule.

        Examples
        --------
        >>> cal = ReleaseCalendar({"ip": 45}, frequencies={"ip": "M"})
        >>> cal.release_date("ip", "2020-01")
        Timestamp('2020-03-16 00:00:00')
        """
        self._check_series([series])
        freq = self._frequency(series)
        period = to_period(reference_period, freq, name="reference_period")
        date = self._dates_for(series, pd.PeriodIndex([period]))[0]
        if pd.isna(date):
            raise NowcastDataError(
                f"No release date for {series!r} in {period}: not in its release table and "
                "no delay rule."
            )
        return pd.Timestamp(date)

    def release_dates_for(
        self, series: str, reference_periods: Sequence[PeriodLike] | pd.PeriodIndex
    ) -> pd.Series:
        """Release dates of several reference periods of one series.

        Parameters
        ----------
        series : str
            Series name.
        reference_periods : sequence of period-like or PeriodIndex
            Reference periods.

        Returns
        -------
        pandas.Series
            Release dates indexed by reference period (``NaT`` when unknown).

        Examples
        --------
        >>> cal = ReleaseCalendar({"gdp": 60}, frequencies={"gdp": "Q"})
        >>> cal.release_dates_for("gdp", ["2020Q1", "2020Q2"]).dt.strftime("%Y-%m-%d").tolist()
        ['2020-05-30', '2020-08-29']
        """
        self._check_series([series])
        freq = self._frequency(series)
        periods = pd.PeriodIndex(
            [to_period(p, freq, name="reference_period") for p in reference_periods],
            freq=freq.pandas_freq,
        )
        return pd.Series(self._dates_for(series, periods), index=periods, name=series)

    def release_mask(
        self, data: MixedFrequencyData | pd.DataFrame, vintage: DateLike
    ) -> pd.DataFrame:
        """Mask of the cells of a panel released on or before ``vintage``.

        Parameters
        ----------
        data : MixedFrequencyData or pandas.DataFrame
            Panel on the base grid (a DataFrame is coerced with inferred frequencies).
        vintage : date-like
            Information date (inclusive).

        Returns
        -------
        pandas.DataFrame of bool
            Same shape as the panel; ``True`` where the reference period of the cell has
            been released. Non-slot cells of low-frequency series carry the flag of their
            native period (they are NaN anyway).

        Raises
        ------
        NowcastDataError
            If a series of ``data`` is not in the calendar, or an observed value has no
            known release date.

        Examples
        --------
        >>> import pandas as pd
        >>> idx = pd.period_range("2020-01", periods=3, freq="M")
        >>> frame = pd.DataFrame({"a": [1.0, 2.0, 3.0]}, index=idx)
        >>> cal = ReleaseCalendar({"a": 10}, frequencies={"a": "M"})
        >>> cal.release_mask(frame, "2020-03-10")["a"].tolist()
        [True, True, False]
        """
        mfd = as_mixed_frequency_data(data)
        self._check_series(mfd.columns)
        date = np.datetime64(to_date(vintage, name="vintage"), "ns")
        observed = mfd.observation_mask().to_numpy()
        out = np.zeros(observed.shape, dtype=bool)
        natives: dict[Frequency, tuple[pd.PeriodIndex, np.ndarray]] = {}
        for j, name in enumerate(mfd.columns):
            freq = self._frequency(name, mfd)
            if freq not in natives:
                native_index = mfd.index.asfreq(freq.pandas_freq)
                natives[freq] = (native_index, _period_ends(native_index))
            native, ends = natives[freq]
            dates = self._dates_for(name, native, ends).to_numpy()
            unknown = np.isnat(dates) & observed[:, j]
            if unknown.any():
                bad = sorted({str(p) for p in native[unknown]})
                raise NowcastDataError(
                    f"No release date for observed values of {name!r} in {bad[:5]}"
                    f"{'...' if len(bad) > 5 else ''}."
                )
            out[:, j] = ~np.isnat(dates) & (dates <= date)
        return pd.DataFrame(out, index=mfd.index, columns=pd.Index(mfd.columns))

    def available_at(self, date: DateLike, *, series: Sequence[str] | None = None) -> pd.Series:
        r"""Last reference period of each series released on or before ``date``.

        Parameters
        ----------
        date : date-like
            Information date (inclusive).
        series : sequence of str, optional
            Subset of series (default: all).

        Returns
        -------
        pandas.Series
            Object series of :class:`pandas.Period` (``None`` when nothing of a series
            is released yet), indexed by series name.

        Raises
        ------
        NowcastDataError
            If a delay-only series has no known frequency.

        Notes
        -----
        For the delay rule the answer is the period :math:`p^*` containing
        :math:`v - \delta` if :math:`\operatorname{end}(p^*) + \delta \le v`, otherwise
        :math:`p^* - 1`. Explicit tables are scanned directly; with both rules the larger
        of the two candidates is returned.

        Examples
        --------
        >>> cal = ReleaseCalendar({"gdp": 60}, frequencies={"gdp": "Q"})
        >>> str(cal.available_at("2020-05-30")["gdp"]), str(cal.available_at("2020-05-29")["gdp"])
        ('2020Q1', '2019Q4')
        """
        names = list(self._series if series is None else series)
        self._check_series(names)
        v = to_date(date)
        out: dict[str, pd.Period | None] = {}
        for name in names:
            out[name] = self._last_released(name, v)
        return pd.Series(out, dtype=object, name="last_released")

    def _last_released(self, name: str, v: pd.Timestamp) -> pd.Period | None:
        candidates: list[pd.Period] = []
        table = self._explicit.get(name)
        explicit_set: set[pd.Period] = set()
        if table is not None:
            explicit_set = set(table.index)
            released = table[table <= v]
            if len(released):
                candidates.append(released.index.max())
        delay = self._delays.get(name)
        if delay is not None:
            freq = self._frequency(name)
            p = pd.Period(v - pd.Timedelta(days=delay), freq=freq.pandas_freq)
            if native_period_end(p) + pd.Timedelta(days=delay) > v:
                p = p - 1
            # periods with explicit dates are governed by their table
            for _ in range(len(explicit_set) + 1):
                if p not in explicit_set:
                    break
                p = p - 1
            candidates.append(p)
        return max(candidates) if candidates else None

    def schedule(
        self, data: MixedFrequencyData | pd.DataFrame, *, observed_only: bool = False
    ) -> pd.DataFrame:
        """Release date of every native period of every series of a panel.

        Parameters
        ----------
        data : MixedFrequencyData or pandas.DataFrame
            Panel on the base grid.
        observed_only : bool, default False
            Keep only cells holding a value.

        Returns
        -------
        pandas.DataFrame
            Long table with columns ``release_date``, ``series``, ``reference_period``
            (native :class:`pandas.Period`), ``slot`` (base-grid period where the value
            is stored) and ``value``; sorted by release date, then series order and period.
            Periods without a known release date are dropped (unless observed, which
            raises).

        Raises
        ------
        NowcastDataError
            If a series is not in the calendar or an observed value has no release date.

        Examples
        --------
        >>> import pandas as pd
        >>> idx = pd.period_range("2020-01", periods=6, freq="M")
        >>> frame = pd.DataFrame({"gdp": [None, None, 1.0, None, None, 2.0]}, index=idx)
        >>> cal = ReleaseCalendar({"gdp": 60})
        >>> cal.schedule(frame)["release_date"].dt.strftime("%Y-%m-%d").tolist()
        ['2020-05-30', '2020-08-29']
        """
        mfd = as_mixed_frequency_data(data)
        self._check_series(mfd.columns)
        observed = mfd.observation_mask()
        slots = mfd.slot_mask()
        values = mfd.data
        parts: list[pd.DataFrame] = []
        order = {name: i for i, name in enumerate(mfd.columns)}
        for name in mfd.columns:
            freq = self._frequency(name, mfd)
            keep = slots[name].to_numpy()
            if observed_only:
                keep = keep & observed[name].to_numpy()
            slot_index = mfd.index[keep]
            native = slot_index.asfreq(freq.pandas_freq)
            dates = self._dates_for(name, native)
            obs = observed[name].to_numpy()[keep]
            unknown = np.isnat(dates.to_numpy()) & obs
            if unknown.any():
                bad = sorted({str(p) for p in native[unknown]})
                raise NowcastDataError(
                    f"No release date for observed values of {name!r} in {bad[:5]}."
                )
            part = pd.DataFrame(
                {
                    "release_date": dates,
                    "series": name,
                    "reference_period": list(native),
                    "slot": slot_index,
                    "value": values[name].to_numpy()[keep],
                    "_order": order[name],
                }
            )
            parts.append(part[part["release_date"].notna()])
        return _finalise(parts, extra=("slot", "value"))

    def releases_between(
        self,
        start: DateLike,
        end: DateLike,
        data: MixedFrequencyData | pd.DataFrame | None = None,
        *,
        observed_only: bool = True,
        series: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        r"""Releases published after ``start`` and on or before ``end``.

        These are the data releases separating the information sets of two vintages,
        i.e. the sources of *news* between :math:`\Omega_{start}` and
        :math:`\Omega_{end}`.

        Parameters
        ----------
        start : date-like
            Old information date (exclusive).
        end : date-like
            New information date (inclusive).
        data : MixedFrequencyData or pandas.DataFrame, optional
            Restrict to the periods of this panel and add ``slot`` and ``value`` columns.
        observed_only : bool, default True
            With ``data``, keep only releases whose cell holds a value.
        series : sequence of str, optional
            Subset of series.

        Returns
        -------
        pandas.DataFrame
            Long table with columns ``release_date``, ``series``, ``reference_period``
            (plus ``slot`` and ``value`` with ``data``), sorted by release date.

        Raises
        ------
        ValueError
            If ``start`` is after ``end``.
        NowcastDataError
            If a series is unknown or has no known frequency.

        Examples
        --------
        >>> cal = ReleaseCalendar({"ip": 40, "gdp": 60}, frequencies={"ip": "M", "gdp": "Q"})
        >>> rel = cal.releases_between("2020-05-01", "2020-06-01")
        >>> [(s, str(p)) for s, p in zip(rel["series"], rel["reference_period"])]
        [('ip', '2020-03'), ('gdp', '2020Q1')]
        """
        d0 = to_date(start, name="start")
        d1 = to_date(end, name="end")
        if d0 > d1:
            raise ValueError(f"start ({d0.date()}) must not be after end ({d1.date()}).")
        if data is not None:
            mfd = as_mixed_frequency_data(data)
            if series is not None:
                mfd = mfd.select(list(series))
            table = self.schedule(mfd, observed_only=observed_only)
            within = (table["release_date"] > d0) & (table["release_date"] <= d1)
            return table[within].reset_index(drop=True)
        names = list(self._series if series is None else series)
        self._check_series(names)
        parts = [self._releases_without_data(name, i, d0, d1) for i, name in enumerate(names)]
        return _finalise(parts)

    def _releases_without_data(
        self, name: str, order: int, d0: pd.Timestamp, d1: pd.Timestamp
    ) -> pd.DataFrame:
        periods: list[pd.Period] = []
        dates: list[pd.Timestamp] = []
        table = self._explicit.get(name)
        explicit_set: set[pd.Period] = set()
        if table is not None:
            explicit_set = set(table.index)
            hit = table[(table > d0) & (table <= d1)]
            periods += list(hit.index)
            dates += list(hit)
        delay = self._delays.get(name)
        if delay is not None:
            freq = self._frequency(name)
            shift = pd.Timedelta(days=delay)
            first = pd.Period(d0 - shift, freq=freq.pandas_freq)
            last = pd.Period(d1 - shift, freq=freq.pandas_freq)
            for p in pd.period_range(first, last, freq=freq.pandas_freq):
                if p in explicit_set:
                    continue
                d = native_period_end(p) + shift
                if d0 < d <= d1:
                    periods.append(p)
                    dates.append(d)
        return pd.DataFrame(
            {
                "release_date": pd.DatetimeIndex(dates, dtype="datetime64[ns]"),
                "series": name,
                "reference_period": pd.Series(periods, dtype=object),
                "_order": order,
            }
        )

    def release_dates(
        self,
        start: DateLike,
        end: DateLike,
        data: MixedFrequencyData | pd.DataFrame | None = None,
        *,
        observed_only: bool = True,
    ) -> pd.DatetimeIndex:
        """Distinct dates with at least one release in ``(start, end]``.

        Parameters
        ----------
        start : date-like
            Exclusive lower bound.
        end : date-like
            Inclusive upper bound.
        data : MixedFrequencyData or pandas.DataFrame, optional
            Restrict to the releases of this panel.
        observed_only : bool, default True
            With ``data``, only count releases of observed values.

        Returns
        -------
        pandas.DatetimeIndex
            Sorted unique release dates (the "daily data flow").

        Examples
        --------
        >>> cal = ReleaseCalendar({"a": 10, "b": 10}, frequencies={"a": "M", "b": "M"})
        >>> cal.release_dates("2020-01-01", "2020-03-31").strftime("%m-%d").tolist()
        ['01-10', '02-10', '03-10']
        """
        table = self.releases_between(start, end, data, observed_only=observed_only)
        return pd.DatetimeIndex(np.unique(table["release_date"].to_numpy()), name="release_date")


def _finalise(parts: list[pd.DataFrame], extra: tuple[str, ...] = ()) -> pd.DataFrame:
    columns = [*RELEASE_COLUMNS, *extra]
    parts = [p for p in parts if len(p)]
    if not parts:
        empty = pd.DataFrame({c: pd.Series([], dtype=object) for c in columns})
        empty["release_date"] = pd.Series([], dtype="datetime64[ns]")
        if "value" in empty:
            empty["value"] = pd.Series([], dtype=float)
        return empty
    table = pd.concat(parts, ignore_index=True)
    table["_ordinal"] = [p.ordinal for p in table["reference_period"]]
    table = table.sort_values(["release_date", "_order", "_ordinal"], kind="stable")
    table = table[columns].reset_index(drop=True)
    table["reference_period"] = table["reference_period"].astype(object)
    return table


def _period_ends(periods: pd.PeriodIndex) -> np.ndarray:
    """Last calendar day of each period as ``datetime64[ns]`` (computed per unique period)."""
    codes, uniques = pd.factorize(periods)
    distinct = pd.PeriodIndex(uniques)
    ends = distinct.to_timestamp(how="end").normalize().to_numpy(dtype="datetime64[ns]")
    return ends[codes]
