"""The :class:`Dataset` container returned by every ``load_*`` function.

A dataset bundles a :class:`~nowcastbox.core.data.MixedFrequencyData` panel (values in
**levels**, i.e. before any stationarity transformation, on a monthly grid with
quarterly values in the third month of the quarter) with a *legend*: one row per series
with its description, primary source and code, native frequency, named transformation,
legacy numeric code, typical publication delay, factor blocks, category and units. The
same information is also stored in the panel's :class:`~nowcastbox.core.data.SeriesMetadata`
so the panel can be passed straight to the estimators, to
:func:`~nowcastbox.preprocessing.prepare_panel` and to the vintage tools.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from nowcastbox.core.data import MixedFrequencyData
from nowcastbox.core.exceptions import NowcastDataError
from nowcastbox.vintages.calendar import ReleaseCalendar

__all__ = ["LEGEND_COLUMNS", "Dataset", "split_blocks"]

LEGEND_COLUMNS: tuple[str, ...] = (
    "name",
    "description",
    "source",
    "source_code",
    "frequency",
    "transform",
    "legacy_code",
    "delay_days",
    "blocks",
    "category",
    "units",
)
"""Required legend columns, in display order (extra columns are kept after them)."""

_BLOCK_SEP = ";"


def split_blocks(value: object) -> tuple[str, ...]:
    """Split a legend ``blocks`` cell (``"global;real"``) into block names.

    Parameters
    ----------
    value : str, sequence of str or None
        Cell value. Missing values and empty strings mean "no block".

    Returns
    -------
    tuple of str
        Block names, stripped, in order.

    Examples
    --------
    >>> from nowcastbox.datasets.dataset import split_blocks
    >>> split_blocks("global; real")
    ('global', 'real')
    >>> split_blocks(None)
    ()
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ()
    if isinstance(value, str):
        parts = value.split(_BLOCK_SEP)
    elif isinstance(value, Sequence):
        parts = [str(v) for v in value]
    else:
        raise ValueError(f"Cannot interpret {value!r} as a list of blocks.")
    return tuple(p.strip() for p in parts if p.strip())


def _normalise_legend(legend: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    missing = [c for c in LEGEND_COLUMNS if c not in legend.columns]
    if missing:
        raise NowcastDataError(f"Legend is missing the columns {missing}.")
    out = legend.copy()
    out["name"] = out["name"].astype(str)
    if out["name"].duplicated().any():
        dup = sorted(out.loc[out["name"].duplicated(), "name"])
        raise NowcastDataError(f"Duplicated series names in the legend: {dup}.")
    out = out.set_index(pd.Index(out["name"], name="series"))
    absent = [c for c in columns if c not in out.index]
    extra = [n for n in out.index if n not in set(columns)]
    if absent or extra:
        raise NowcastDataError(
            f"Legend and data disagree: no legend for {absent}; no data for {extra}."
        )
    out = out.loc[list(columns)]
    out["delay_days"] = pd.array(pd.to_numeric(out["delay_days"], errors="raise"), dtype="Int64")
    for col in ("description", "source", "source_code", "legacy_code", "units", "blocks"):
        out[col] = out[col].fillna("").astype(str)
    out["transform"] = out["transform"].astype(object).where(out["transform"].notna(), None)
    order = list(LEGEND_COLUMNS) + [c for c in out.columns if c not in LEGEND_COLUMNS]
    return out[order]


class Dataset:
    """A built-in dataset: panel in levels + legend + provenance.

    Parameters
    ----------
    name : str
        Dataset identifier (e.g. ``"brazil_nowcast"``).
    data : pandas.DataFrame
        Values in levels on a monthly :class:`pandas.PeriodIndex`; lower-frequency
        series hold values only in the last month of their period.
    legend : pandas.DataFrame
        One row per column of ``data`` with at least the columns in
        :data:`LEGEND_COLUMNS`. ``blocks`` holds ``";"``-separated block names.
    title : str, default ""
        Short human-readable title.
    description : str, default ""
        Longer description.
    target : str, optional
        Default nowcasting target (a column of ``data``).
    metadata : mapping, optional
        Remaining provenance (sources, license, citation, build information, ...).
        ``license``, ``citation``, ``url`` and ``notes`` keys are exposed as properties.

    Raises
    ------
    NowcastDataError
        If the legend and data disagree, required legend columns are missing, the
        target is not a column, or the panel violates the
        :class:`~nowcastbox.core.data.MixedFrequencyData` contract.

    See Also
    --------
    nowcastbox.datasets.list_datasets : Catalogue of the built-in datasets.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from nowcastbox.datasets import Dataset
    >>> idx = pd.period_range("2020-01", periods=6, freq="M")
    >>> data = pd.DataFrame(
    ...     {"ip": np.arange(6.0) + 100, "gdp": [np.nan, np.nan, 1.0, np.nan, np.nan, 2.0]},
    ...     index=idx,
    ... )
    >>> legend = pd.DataFrame(
    ...     {
    ...         "name": ["ip", "gdp"],
    ...         "description": ["Industry", "GDP"],
    ...         "source": ["demo", "demo"],
    ...         "source_code": ["1", "2"],
    ...         "frequency": ["M", "Q"],
    ...         "transform": ["dlog", "qoq"],
    ...         "legacy_code": ["", "7"],
    ...         "delay_days": [40, 60],
    ...         "blocks": ["global;real", "global"],
    ...         "category": ["hard", "hard"],
    ...         "units": ["index", "index"],
    ...     }
    ... )
    >>> ds = Dataset("demo", data, legend, target="gdp")
    >>> ds.frequency.to_dict()
    {'ip': 'M', 'gdp': 'Q'}
    >>> ds.blocks
            global  real
    series
    ip           1     1
    gdp          1     0
    >>> ds.data.quarterly_columns
    ['gdp']
    """

    __slots__ = ("_data", "_description", "_legend", "_metadata", "_name", "_target", "_title")

    def __init__(
        self,
        name: str,
        data: pd.DataFrame,
        legend: pd.DataFrame,
        *,
        title: str = "",
        description: str = "",
        target: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(data, pd.DataFrame):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise NowcastDataError(f"data must be a DataFrame, got {type(data).__name__}.")
        columns = [str(c) for c in data.columns]
        table = _normalise_legend(legend, columns)
        if target is not None and target not in columns:
            raise NowcastDataError(f"Target {target!r} is not a series of dataset {name!r}.")
        frame = data.copy()
        frame.columns = columns
        meta = {
            n: {
                "frequency": row["frequency"],
                "transform": row["transform"],
                "release_delay": None if pd.isna(row["delay_days"]) else int(row["delay_days"]),
                "blocks": split_blocks(row["blocks"]),
                "category": row["category"] if row["category"] else None,
                "description": row["description"],
                "units": row["units"],
            }
            for n, row in table.iterrows()
        }
        self._data = MixedFrequencyData(frame, metadata=meta)  # type: ignore[arg-type]
        self._legend = table
        self._name = str(name)
        self._title = str(title)
        self._description = str(description)
        self._target = target
        self._metadata: dict[str, Any] = copy.deepcopy(dict(metadata or {}))

    # ------------------------------------------------------------------ basic
    @property
    def name(self) -> str:
        """Dataset identifier."""
        return self._name

    @property
    def title(self) -> str:
        """Short title."""
        return self._title

    @property
    def description(self) -> str:
        """Long description."""
        return self._description

    @property
    def target(self) -> str | None:
        """Default nowcasting target (``None`` if the dataset has none)."""
        return self._target

    @property
    def data(self) -> MixedFrequencyData:
        """The panel in levels, with per-series metadata (immutable object)."""
        return self._data

    @property
    def legend(self) -> pd.DataFrame:
        """Legend table indexed by series name (a copy)."""
        return self._legend.copy()

    @property
    def columns(self) -> list[str]:
        """Series names, in panel order."""
        return list(self._legend.index)

    @property
    def n_series(self) -> int:
        """Number of series."""
        return len(self._legend)

    @property
    def metadata(self) -> dict[str, Any]:
        """Provenance metadata (a deep copy)."""
        return copy.deepcopy(self._metadata)

    @property
    def license(self) -> str:
        """Data license / terms of use of the primary sources."""
        return str(self._metadata.get("license", ""))

    @property
    def citation(self) -> str:
        """How to cite the data."""
        return str(self._metadata.get("citation", ""))

    @property
    def url(self) -> str:
        """Main primary-source URL."""
        return str(self._metadata.get("url", ""))

    @property
    def notes(self) -> str:
        """Construction notes and caveats."""
        return str(self._metadata.get("notes", ""))

    # ------------------------------------------------------------------ legend views
    @property
    def frequency(self) -> pd.Series:
        """Native frequency code of each series (``"M"``, ``"Q"``)."""
        return self._legend["frequency"].copy()

    @property
    def transform(self) -> pd.Series:
        """Named transformation of each series (see :mod:`nowcastbox.preprocessing`)."""
        return self._legend["transform"].copy()

    @property
    def delay(self) -> pd.Series:
        """Typical publication delay in days after the end of the reference period."""
        return self._legend["delay_days"].copy()

    @property
    def categories(self) -> pd.Series:
        """Category of each series (``"hard"``, ``"soft"``, ``"financial"``, ``"other"``)."""
        return self._legend["category"].copy()

    @property
    def block_names(self) -> list[str]:
        """Block names in order of first appearance in the legend."""
        names: dict[str, None] = {}
        for cell in self._legend["blocks"]:
            for b in split_blocks(cell):
                names.setdefault(b, None)
        return list(names)

    @property
    def blocks(self) -> pd.DataFrame:
        """0/1 block-membership matrix (series x block), NY Fed layout."""
        names = self.block_names
        rows = [[int(b in split_blocks(cell)) for b in names] for cell in self._legend["blocks"]]
        return pd.DataFrame(rows, index=self._legend.index.copy(), columns=names, dtype=int)

    # ------------------------------------------------------------------ derived
    def calendar(self) -> ReleaseCalendar:
        """Release calendar built from the typical delays of the legend.

        Returns
        -------
        ReleaseCalendar
            Delay-based calendar of the series that have a delay.

        Raises
        ------
        ValueError
            If no series has a delay.

        Examples
        --------
        >>> from nowcastbox.datasets import load_nyfed
        >>> cal = load_nyfed().calendar()
        >>> int(cal.delays["PAYEMS"])
        5
        """
        delays = self._legend["delay_days"].dropna().astype(int)
        freqs = self._legend.loc[delays.index, "frequency"]
        return ReleaseCalendar(delays, frequencies=freqs)

    def _derive(self, legend: pd.DataFrame, data: pd.DataFrame) -> Dataset:
        target = self._target if self._target in legend.index else None
        return Dataset(
            self._name,
            data,
            legend.reset_index(drop=True),
            title=self._title,
            description=self._description,
            target=target,
            metadata=self._metadata,
        )

    def select(self, columns: Sequence[str]) -> Dataset:
        """Return a dataset with a subset of the series.

        Parameters
        ----------
        columns : sequence of str
            Series to keep, in the desired order.

        Returns
        -------
        Dataset
            New dataset (the target is kept only if selected).

        Raises
        ------
        NowcastDataError
            For unknown or duplicated names, or an empty selection.

        Examples
        --------
        >>> from nowcastbox.datasets import load_nyfed
        >>> load_nyfed().select(["GDPC1", "INDPRO"]).columns
        ['GDPC1', 'INDPRO']
        """
        cols = [str(c) for c in columns]
        if not cols:
            raise NowcastDataError("Select at least one series.")
        unknown = [c for c in cols if c not in self._legend.index]
        if unknown:
            raise NowcastDataError(f"Unknown series {unknown} in dataset {self._name!r}.")
        if len(set(cols)) != len(cols):
            raise NowcastDataError(f"Duplicated series in selection {cols}.")
        return self._derive(self._legend.loc[cols], self._data.data[cols])

    def truncate(self, start: object = None, end: object = None) -> Dataset:
        """Return a dataset restricted to ``[start, end]`` (inclusive).

        Parameters
        ----------
        start, end : period-like, optional
            Bounds accepted by :meth:`MixedFrequencyData.truncate` (``"2005-01"``,
            ``"2019Q4"``, ...).

        Returns
        -------
        Dataset
            New dataset.

        Raises
        ------
        NowcastDataError
            If the range is empty or invalid.

        Examples
        --------
        >>> from nowcastbox.datasets import load_nyfed
        >>> str(load_nyfed().truncate("2000-01", "2009Q4").data.end)
        '2009-12'
        """
        panel = self._data.truncate(start, end)  # type: ignore[arg-type]
        return self._derive(self._legend, panel.data)

    def to_frame(self) -> pd.DataFrame:
        """Return the panel values (levels) as a DataFrame copy.

        Returns
        -------
        pandas.DataFrame
            Monthly PeriodIndex, one column per series.

        Examples
        --------
        >>> from nowcastbox.datasets import load_nyfed
        >>> load_nyfed().to_frame().index.freqstr
        'M'
        """
        return self._data.data

    def summary(self) -> str:
        """Return a plain-text summary (provenance, sample, frequencies, blocks).

        Returns
        -------
        str
            Multi-line summary.

        Examples
        --------
        >>> from nowcastbox.datasets import load_nyfed
        >>> print(load_nyfed().summary().splitlines()[0])
        Dataset 'nyfed': NY Fed Staff Nowcast replication panel (US)
        """
        freq_counts = self._legend["frequency"].value_counts()
        cat_counts = self._legend["category"].replace("", "unset").value_counts()
        lines = [
            f"Dataset {self._name!r}: {self._title}",
            f"  Sample        : {self._data.start} to {self._data.end} "
            f"({self._data.n_periods} months)",
            f"  Series        : {self.n_series} ("
            + ", ".join(f"{k}: {v}" for k, v in freq_counts.items())
            + ")",
            "  Categories    : " + ", ".join(f"{k}: {v}" for k, v in cat_counts.items()),
            "  Blocks        : " + (", ".join(self.block_names) or "none"),
            f"  Target        : {self._target or 'none'}",
            f"  License       : {self.license or 'n/a'}",
        ]
        sources = self._metadata.get("sources") or []
        if sources:
            lines.append("  Sources       : " + "; ".join(str(s) for s in sources))
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"Dataset(name={self._name!r}, n_series={self.n_series}, "
            f"start={self._data.start}, end={self._data.end}, target={self._target!r})"
        )
