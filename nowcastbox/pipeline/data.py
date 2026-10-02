"""Data stage of the pipeline: load, apply the vintage and preprocess.

:func:`load_data` turns the ``data`` section of a :class:`~nowcastbox.pipeline.spec.NowcastSpec`
into a :class:`~nowcastbox.core.data.MixedFrequencyData` panel in levels (built-in
dataset, CSV/Parquet file, Excel workbook or the BCB/IBGE/IPEA/FRED connectors);
:func:`apply_vintage` keeps the information available at the spec's vintage (release
rule of :meth:`MixedFrequencyData.as_of`); :func:`preprocess` applies the
transformations and cleaning of :func:`~nowcastbox.preprocessing.prepare_panel`;
:func:`data_hash` fingerprints a data vintage for the snapshots.

Excel workbooks (optional extra ``nowcastbox[excel]``, which installs ``openpyxl``)
follow the layout of the ECB Nowcasting Toolbox templates (Linzenich & Meunier, 2024):
one sheet per frequency (``monthly``, ``quarterly``, ``annual``; a date column followed
by one column per series) and a ``metadata`` sheet with one row per series
(:data:`EXCEL_METADATA_COLUMNS`). :func:`read_excel_panel` and
:func:`write_excel_panel` convert between workbooks and
:class:`~nowcastbox.core.data.MixedFrequencyData`; :func:`write_run_excel` exports the
results of a pipeline run.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping
from importlib import resources
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from nowcastbox._logging import get_logger
from nowcastbox.core.data import MixedFrequencyData, SeriesMetadata
from nowcastbox.core.frequency import Frequency
from nowcastbox.pipeline.spec import ConnectorSeries, DataSpec, NowcastSpec, SpecError

__all__ = [
    "CONNECTORS",
    "EXCEL_DATA_SHEETS",
    "EXCEL_METADATA_COLUMNS",
    "apply_vintage",
    "data_hash",
    "example_workbook_path",
    "frame_hash",
    "load_data",
    "preprocess",
    "read_excel_panel",
    "read_panel_file",
    "write_excel_panel",
    "write_run_excel",
]

logger = get_logger(__name__)


def _fetch_bcb(series: ConnectorSeries, start: Any, end: Any) -> pd.DataFrame:
    from nowcastbox.data_sources import fetch_sgs

    return fetch_sgs(
        {series.name: str(series.code)},
        start,
        end,
        native_frequency=series.frequency,
        base_frequency="M",
        **series.options,
    )


def _fetch_ipea(series: ConnectorSeries, start: Any, end: Any) -> pd.DataFrame:
    from nowcastbox.data_sources import fetch_ipeadata

    return fetch_ipeadata(
        {series.name: str(series.code)},
        start,
        end,
        native_frequency=series.frequency,
        base_frequency="M",
        **series.options,
    )


def _fetch_fred(series: ConnectorSeries, start: Any, end: Any) -> pd.DataFrame:
    from nowcastbox.data_sources import fetch_fred

    return fetch_fred(
        {series.name: str(series.code)},
        start,
        end,
        native_frequency=series.frequency,
        base_frequency="M",
        **series.options,
    )


def _fetch_ibge(series: ConnectorSeries, start: Any, end: Any) -> pd.DataFrame:
    from nowcastbox.data_sources import fetch_sidra

    return fetch_sidra(
        str(series.table),
        series.variable,
        cast("Any", series.classifications or None),
        start=start,
        end=end,
        name=series.name,
        native_frequency=series.frequency,
        base_frequency="M",
        **series.options,
    )


CONNECTORS: dict[str, Callable[[ConnectorSeries, Any, Any], pd.DataFrame]] = {
    "bcb": _fetch_bcb,
    "ibge": _fetch_ibge,
    "ipea": _fetch_ipea,
    "fred": _fetch_fred,
}
"""Download function per connector (``(series, start, end) -> monthly DataFrame``)."""


# ---------------------------------------------------------------------------- load
def load_data(spec: NowcastSpec) -> MixedFrequencyData:
    """Load the panel (in levels) described by ``spec.data``.

    Parameters
    ----------
    spec : NowcastSpec
        Validated spec.

    Returns
    -------
    MixedFrequencyData
        Monthly panel with the target first, the requested columns, sample bounds and
        metadata overrides applied.

    Raises
    ------
    SpecError
        If the target or a requested column is missing, or the data cannot be read.
    NowcastDataError
        If the data violate the panel contract.

    Examples
    --------
    >>> from nowcastbox.pipeline import NowcastSpec
    >>> spec = NowcastSpec.from_dict(
    ...     {"target": "gdp", "data": {"source": "simulated_dfm", "columns": ["x01", "x02"]}}
    ... )
    >>> load_data(spec).columns
    ['gdp', 'x01', 'x02']
    """
    data = spec.data
    loaders: dict[str, Callable[[DataSpec], MixedFrequencyData]] = {
        "dataset": _load_dataset,
        "file": _load_file,
        "excel": _load_excel,
        "connectors": _load_connectors,
    }
    panel = loaders[data.kind](data)
    panel = _select(panel, spec.target_name, data)
    return _with_overrides(panel, data)


def _load_dataset(data: DataSpec) -> MixedFrequencyData:
    from nowcastbox import datasets

    ds = datasets.load_dataset(data.source)  # nosec B615 - built-in local dataset, not the HF Hub
    if not isinstance(ds, datasets.Dataset):
        raise SpecError([("data.source", f"dataset {data.source!r} is not a panel")])
    return ds.data


def read_panel_file(path: str | Path, index_column: str | None = None) -> pd.DataFrame:
    r"""Read a CSV or Parquet panel onto a monthly :class:`pandas.PeriodIndex`.

    Parameters
    ----------
    path : str or pathlib.Path
        ``.csv`` (or ``.csv.gz``) or ``.parquet`` file with a date column and one
        column per series.
    index_column : str, optional
        Date column (default: the first column, or the stored index of a Parquet
        file).

    Returns
    -------
    pandas.DataFrame
        ``float64`` values on a monthly PeriodIndex (dates are mapped to their month;
        quarterly labels such as ``2020Q1`` to the last month of the quarter).

    Raises
    ------
    SpecError
        If the index column is missing or dates cannot be parsed.

    Examples
    --------
    >>> import tempfile, pathlib
    >>> tmp = pathlib.Path(tempfile.mkdtemp()) / "p.csv"
    >>> _ = tmp.write_text("date,ip\n2020-01-31,1.0\n2020-02-29,2.0\n")
    >>> read_panel_file(tmp).index.astype(str).tolist()
    ['2020-01', '2020-02']
    """
    file = Path(path)
    if file.suffix == ".parquet":
        frame = pd.read_parquet(file)
        if index_column is not None and index_column == frame.index.name:
            pass
        elif index_column is not None or not isinstance(
            frame.index, pd.PeriodIndex | pd.DatetimeIndex
        ):
            frame = _set_index(frame, index_column)
    else:
        frame = _set_index(pd.read_csv(file), index_column)
    frame.index = _monthly_index(frame.index)
    frame.columns = [str(c) for c in frame.columns]
    return frame.apply(pd.to_numeric, errors="coerce").astype(float).sort_index()


def _set_index(frame: pd.DataFrame, index_column: str | None) -> pd.DataFrame:
    column = index_column if index_column is not None else str(frame.columns[0])
    if column not in frame.columns:
        raise SpecError([("data.index_column", f"column {column!r} not found in the file")])
    return frame.set_index(column)


def _monthly_index(index: pd.Index) -> pd.PeriodIndex:
    if isinstance(index, pd.PeriodIndex):
        return index.asfreq("M", how="E")
    if isinstance(index, pd.DatetimeIndex):
        return index.to_period("M")
    labels = [str(v) for v in index]
    try:
        return pd.PeriodIndex([pd.Period(v).asfreq("M", how="E") for v in labels], freq="M")
    except (ValueError, TypeError) as err:
        raise SpecError([("data.path", f"cannot parse the dates of the file: {err}")]) from err


def _load_file(data: DataSpec) -> MixedFrequencyData:
    if data.path is None:
        raise SpecError([("data.path", "is required")])
    frame = read_panel_file(data.path, data.index_column)
    if frame.index.has_duplicates:
        raise SpecError([("data.path", "the file has duplicated dates")])
    full = pd.period_range(frame.index.min(), frame.index.max(), freq="M")
    frame = frame.reindex(full)
    legend = _read_legend(data.legend) if data.legend is not None else {}
    overrides = {
        "frequency": data.frequency,
        "transform": data.transform,
        "delay": data.delay,
        "blocks": data.blocks,
        "category": data.categories,
    }
    for key, mapping in overrides.items():
        if mapping:
            legend[key] = {**legend.get(key, {}), **{k: _listify(v) for k, v in mapping.items()}}
    unknown = sorted({n for m in legend.values() for n in m} - set(frame.columns))
    if unknown:
        raise SpecError([("data", f"metadata given for series not in the file: {unknown}")])
    return MixedFrequencyData(
        frame,
        frequencies=legend.get("frequency"),
        transforms=legend.get("transform"),
        release_delays=legend.get("delay"),
        blocks=legend.get("blocks"),
        categories=legend.get("category"),
    )


def _listify(value: Any) -> Any:
    return list(value) if isinstance(value, tuple) else value


def _read_legend(path: Path) -> dict[str, dict[str, Any]]:
    """Metadata mappings from a legend CSV (``name`` + optional columns)."""
    table = pd.read_csv(path, dtype=str)
    if "name" not in table.columns:
        raise SpecError([("data.legend", "the legend needs a 'name' column")])
    table = table.set_index("name")
    out: dict[str, dict[str, Any]] = {}
    columns = {
        "frequency": "frequency",
        "transform": "transform",
        "delay_days": "delay",
        "delay": "delay",
        "blocks": "blocks",
        "category": "category",
    }
    for column, key in columns.items():
        if column not in table.columns:
            continue
        values = table[column].dropna()
        if key == "delay":
            out[key] = {k: int(float(v)) for k, v in values.items()}
        elif key == "blocks":
            out[key] = {
                k: [b.strip() for b in str(v).split(";") if b.strip()] for k, v in values.items()
            }
        else:
            out[key] = {k: str(v) for k, v in values.items()}
    return out


def _load_connectors(data: DataSpec) -> MixedFrequencyData:
    frames = []
    for series in data.series:
        fetch = CONNECTORS[series.source]
        logger.info("downloading %s from %s", series.name, series.source)
        frame = fetch(series, data.start, data.end)
        frames.append(
            frame[[series.name]]
            if series.name in frame.columns
            else frame.iloc[:, :1].set_axis([series.name], axis=1)
        )
    panel = pd.concat(frames, axis=1).sort_index()
    full = pd.period_range(panel.index.min(), panel.index.max(), freq="M")
    panel = panel.reindex(full).astype(float)
    return MixedFrequencyData(
        panel,
        frequencies={s.name: s.frequency for s in data.series if s.frequency},
        transforms={s.name: s.transform for s in data.series if s.transform is not None},
        release_delays={s.name: s.delay for s in data.series if s.delay is not None},
        blocks={s.name: list(s.blocks) for s in data.series if s.blocks},
        categories={s.name: s.category for s in data.series if s.category},
    )


def _select(panel: MixedFrequencyData, target: str, data: DataSpec) -> MixedFrequencyData:
    if target not in panel.columns:
        raise SpecError([("target", f"{target!r} is not a series of the data")])
    if data.columns is not None:
        missing = [c for c in data.columns if c not in panel.columns]
        if missing:
            raise SpecError([("data.columns", f"series not in the data: {missing}")])
        cols = [target, *[c for c in data.columns if c != target]]
    else:
        cols = [target, *[c for c in panel.columns if c != target]]
    panel = panel.select(cols)
    if data.start is not None or data.end is not None:
        panel = panel.truncate(data.start, data.end)
    return panel


def _with_overrides(panel: MixedFrequencyData, data: DataSpec) -> MixedFrequencyData:
    """Apply ``data.frequency/transform/delay/blocks/categories`` (dataset sources)."""
    fields = {
        "frequency": data.frequency,
        "transform": data.transform,
        "release_delay": data.delay,
        "blocks": data.blocks,
        "category": data.categories,
    }
    if data.kind == "file":  # already applied when the file was read (CSV/Parquet)
        return panel
    changes: dict[str, dict[str, Any]] = {}
    for field, mapping in fields.items():
        for name, value in (mapping or {}).items():
            if name not in panel.columns:
                key = {"release_delay": "delay", "category": "categories"}.get(field, field)
                raise SpecError([(f"data.{key}.{name}", "is not a series of the data")])
            changes.setdefault(name, {})[field] = tuple(value) if field == "blocks" else value
    if not changes:
        return panel
    metadata: dict[str, SeriesMetadata | dict[str, Any]] = {}
    for name in panel.columns:
        meta = panel.metadata[name].to_dict()
        meta.update(changes.get(name, {}))
        meta.pop("name", None)
        metadata[name] = meta
    return MixedFrequencyData(
        panel.to_frame(), metadata=metadata, base_frequency=panel.base_frequency
    )


# ---------------------------------------------------------------------------- excel
EXCEL_DATA_SHEETS: dict[str, Frequency] = {
    "monthly": Frequency.MONTHLY,
    "quarterly": Frequency.QUARTERLY,
    "annual": Frequency.ANNUAL,
}
"""Data sheets of an Excel workbook and the native frequency of their series."""

EXCEL_METADATA_COLUMNS: tuple[str, ...] = (
    "series",
    "frequency",
    "transform",
    "delay_days",
    "blocks",
    "category",
    "description",
    "aggregation",
    "units",
    "transform_applied",
)
"""Columns of the ``metadata`` sheet (one row per series; only ``series`` is required).

``transform`` takes a code 0-7 (:func:`~nowcastbox.preprocessing.transforms.transform_from_code`)
or a transformation name, ``delay_days`` the publication delay in days after the end of
the reference period and ``blocks`` the factor blocks separated by ``;``.
"""

_METADATA_ALIASES = {"name": "series", "delay": "delay_days", "release_delay": "delay_days"}
_README = (
    "nowcastbox Excel data template",
    "",
    "Sheets 'monthly', 'quarterly' and 'annual': first column 'date', then one column per series.",
    "Dates: 2020-01 (monthly), 2020Q1 (quarterly), 2020 (annual) or Excel dates (mapped to "
    "their period).",
    "Leave unreleased or missing values empty. Unused sheets can stay empty.",
    "Sheet 'metadata': one row per series (only 'series' is required).",
    "  frequency: M, Q or A (must match the sheet); transform: code 0-7 or a name "
    "(level, dlog, qoq, ...)",
    "  delay_days: publication delay in days after the end of the period; blocks: global;real;soft",
    "  category: hard, soft, financial or other; aggregation: flow, stock, average or "
    "mariano_murasawa",
    "Use it in a spec: data: {source: excel, path: this_file.xlsx}",
)


def _require_openpyxl() -> None:
    try:
        import openpyxl  # noqa: F401  # pyright: ignore[reportUnusedImport]
    except ImportError as err:
        raise ImportError(
            "Excel workbooks need the optional dependency 'openpyxl': "
            "pip install 'nowcastbox[excel]'"
        ) from err


def example_workbook_path() -> Path:
    """Path of the bundled example Excel workbook (``pipeline/templates/example.xlsx``).

    Returns
    -------
    pathlib.Path
        Workbook with ``monthly``, ``quarterly``, ``metadata`` and ``readme`` sheets
        (a subset of the simulated dataset).

    Examples
    --------
    >>> example_workbook_path().name
    'example.xlsx'
    """
    return Path(str(resources.files("nowcastbox.pipeline") / "templates" / "example.xlsx"))


def read_excel_panel(
    path: str | Path, sheets: Mapping[str, str | None] | None = None
) -> MixedFrequencyData:
    """Read a mixed-frequency panel from an Excel workbook.

    Parameters
    ----------
    path : str or pathlib.Path
        ``.xlsx`` workbook.
    sheets : mapping, optional
        Sheet name of each role (``monthly``, ``quarterly``, ``annual``, ``metadata``);
        default: the role names. Roles absent from the workbook are skipped unless they
        are given explicitly; ``None`` disables a role.

    Returns
    -------
    MixedFrequencyData
        Monthly panel; series take the frequency of their sheet, lower-frequency values
        are stored in the last month of their period, and the metadata sheet fills
        the :class:`~nowcastbox.core.data.SeriesMetadata` and sets the column order
        (series without metadata follow, in sheet order).

    Raises
    ------
    ImportError
        If ``openpyxl`` is not installed.
    SpecError
        If a sheet is missing or malformed (dates, values, metadata).

    Examples
    --------
    >>> panel = read_excel_panel(example_workbook_path())
    >>> panel.columns[:2], panel.metadata["gdp"].frequency.value
    (['gdp', 'x01'], 'Q')
    """
    _require_openpyxl()
    file = Path(path)
    roles = _sheet_roles(sheets)
    with pd.ExcelFile(file, engine="openpyxl") as book:
        available = [str(n) for n in book.sheet_names]
        explicit = set(sheets or {})
        missing = [
            name for role, name in roles.items() if role in explicit and name not in available
        ]
        if missing:
            raise SpecError([("data.sheets", f"sheets not found in {file.name}: {missing}")])
        frames = [
            _read_data_sheet(book, roles[role], EXCEL_DATA_SHEETS[role])
            for role in EXCEL_DATA_SHEETS
            if roles.get(role) in available
        ]
        meta_name = roles.get("metadata")
        meta = (
            _read_metadata_sheet(book, meta_name)
            if meta_name is not None and meta_name in available
            else {}
        )
    return _assemble_excel(frames, meta, file.name)


def _sheet_roles(sheets: Mapping[str, str | None] | None) -> dict[str, str]:
    allowed = (*EXCEL_DATA_SHEETS, "metadata")
    unknown = [k for k in (sheets or {}) if k not in allowed]
    if unknown:
        raise SpecError([("data.sheets", f"unknown sheet roles {unknown}; use {list(allowed)}")])
    roles: dict[str, str | None] = {role: role for role in allowed}
    roles.update(sheets or {})
    return {k: str(v) for k, v in roles.items() if v is not None}


def _read_data_sheet(
    book: pd.ExcelFile, sheet: str, freq: Frequency
) -> tuple[Frequency, pd.DataFrame]:
    """``(frequency, values on base slots)`` of one data sheet."""
    raw = book.parse(sheet)
    raw = raw.dropna(how="all")
    if raw.shape[1] == 0:
        return freq, pd.DataFrame(index=pd.PeriodIndex([], freq="M"))
    frame = raw.set_index(raw.columns[0])
    frame.columns = [str(c).strip() for c in frame.columns]
    frame = frame.loc[:, [not c.startswith("Unnamed:") for c in frame.columns]]
    try:
        native = pd.PeriodIndex([_to_period(v, freq) for v in frame.index], freq=freq.pandas_freq)
    except (ValueError, TypeError) as err:
        raise SpecError([(f"data.sheets.{sheet}", f"cannot parse the dates: {err}")]) from err
    if native.has_duplicates:
        raise SpecError([(f"data.sheets.{sheet}", "duplicated dates")])
    try:
        values = frame.apply(pd.to_numeric).astype(float)
    except (ValueError, TypeError) as err:
        raise SpecError([(f"data.sheets.{sheet}", f"non-numeric values: {err}")]) from err
    values.index = native.asfreq("M", how="E")
    return freq, values


def _to_period(value: Any, freq: Frequency) -> pd.Period:
    if isinstance(value, float | int | np.integer | np.floating) and not isinstance(value, bool):
        value = str(int(value))  # an annual sheet with years stored as numbers
    if isinstance(value, pd.Timestamp | np.datetime64) or hasattr(value, "year"):
        return pd.Period(pd.Timestamp(value), freq=freq.pandas_freq)
    return pd.Period(str(value).strip(), freq=freq.pandas_freq)


def _read_metadata_sheet(book: pd.ExcelFile, sheet: str) -> dict[str, dict[str, Any]]:
    table = book.parse(sheet, dtype=object).dropna(how="all")
    table.columns = [_METADATA_ALIASES.get(str(c).strip(), str(c).strip()) for c in table.columns]
    if "series" not in table.columns:
        raise SpecError([(f"data.sheets.{sheet}", "the metadata sheet needs a 'series' column")])
    unknown = [c for c in table.columns if c not in EXCEL_METADATA_COLUMNS]
    if unknown:
        raise SpecError(
            [(f"data.sheets.{sheet}", f"unknown columns {unknown}; use {EXCEL_METADATA_COLUMNS}")]
        )
    out: dict[str, dict[str, Any]] = {}
    for position, (_, row) in enumerate(table.iterrows(), start=2):
        if _blank(row["series"]):
            raise SpecError([(f"data.sheets.{sheet}", f"row {position} has no series name")])
        name = str(row["series"]).strip()
        fields = {k: v for k, v in row.items() if k != "series" and not _blank(v)}
        try:
            out[name] = {str(k): _metadata_value(str(k), v) for k, v in fields.items()}
        except ValueError as err:
            raise SpecError([(f"data.sheets.{sheet}.{name}", str(err))]) from err
    return out


def _blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return isinstance(value, str) and not value.strip()


def _metadata_value(column: str, value: Any) -> Any:
    """Typed metadata value of one cell (``ValueError`` when invalid)."""
    if column == "transform":
        return _transform_value(value)
    if column == "delay_days":
        number = float(value)
        if not number.is_integer():
            raise ValueError(f"delay_days must be a whole number of days, got {value!r}")
        return int(number)
    if column == "blocks":
        return tuple(b.strip() for b in str(value).replace(",", ";").split(";") if b.strip())
    if column == "transform_applied":
        return value if isinstance(value, bool) else str(value).strip().lower() in ("true", "1")
    return str(value).strip()


def _transform_value(value: Any) -> int | str:
    from nowcastbox.preprocessing.transforms import transform_from_code

    text = str(value).strip()
    try:
        number = float(text)
    except ValueError:
        return text  # a transformation name (validated by the preprocessing)
    transform_from_code(cast("int", number))  # ValueError for codes outside 0-7
    return int(number)


def _assemble_excel(
    frames: list[tuple[Frequency, pd.DataFrame]], meta: dict[str, dict[str, Any]], where: str
) -> MixedFrequencyData:
    columns: dict[str, pd.Series] = {}
    frequencies: dict[str, str] = {}
    index = pd.PeriodIndex([], freq="M")
    for freq, frame in frames:
        index = index.union(frame.index)
        for name in frame.columns:
            if name in columns:
                raise SpecError([("data.sheets", f"series {name!r} appears in several sheets")])
            columns[name] = frame[name]
            frequencies[name] = freq.value
    if not columns:
        raise SpecError([("data.path", f"{where} has no series")])
    unknown = sorted(set(meta) - set(columns))
    if unknown:
        raise SpecError(
            [("data.sheets.metadata", f"metadata for series not in the data: {unknown}")]
        )
    for name, fields in meta.items():
        _check_sheet_frequency(name, fields.pop("frequency", None), frequencies[name])
        if "delay_days" in fields:
            fields["release_delay"] = fields.pop("delay_days")
    grid = pd.period_range(index.min(), index.max(), freq="M")
    order = [n for n in meta if n in columns] + [n for n in columns if n not in meta]
    panel = pd.DataFrame({k: columns[k].reindex(grid) for k in order}, index=grid)
    return MixedFrequencyData(
        panel,
        frequencies=frequencies,
        metadata=cast("dict[str, SeriesMetadata | dict[str, Any]]", meta),
    )


def _check_sheet_frequency(name: str, given: Any, sheet: str) -> None:
    """The ``frequency`` of a metadata row must match the sheet holding the series."""
    if given is None:
        return
    where = f"data.sheets.metadata.{name}"
    try:
        value = Frequency.from_value(given).value
    except ValueError as err:
        raise SpecError([(where, str(err))]) from err
    if value != sheet:
        raise SpecError([(where, f"frequency {given!r} differs from its sheet ({sheet})")])


def write_excel_panel(path: str | Path, data: MixedFrequencyData | None = None) -> Path:
    """Write a panel (or an empty template) to an Excel workbook.

    Parameters
    ----------
    path : str or pathlib.Path
        Output ``.xlsx`` file (parent folders are created).
    data : MixedFrequencyData, optional
        Monthly panel. ``None`` writes an empty template (headers only) with a
        ``readme`` sheet.

    Returns
    -------
    pathlib.Path
        The written file; :func:`read_excel_panel` gives back an equal panel.

    Raises
    ------
    ImportError
        If ``openpyxl`` is not installed.
    ValueError
        If the panel is not on a monthly grid.

    Examples
    --------
    >>> import tempfile, pathlib
    >>> from nowcastbox.datasets import load_simulated_dfm
    >>> panel = load_simulated_dfm().data.select(["gdp", "x01", "x02"])
    >>> file = write_excel_panel(pathlib.Path(tempfile.mkdtemp()) / "panel.xlsx", panel)
    >>> read_excel_panel(file).equals(panel)
    True
    """
    _require_openpyxl()
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    if data is not None and data.base_frequency is not Frequency.MONTHLY:
        raise ValueError("Excel workbooks hold monthly panels (with quarterly/annual series).")
    sheets = _empty_sheets() if data is None else _panel_sheets(data)
    with pd.ExcelWriter(file, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False)
    return file


def _empty_sheets() -> dict[str, pd.DataFrame]:
    sheets = {role: pd.DataFrame(columns=["date"]) for role in ("monthly", "quarterly")}
    sheets["metadata"] = pd.DataFrame(columns=list(EXCEL_METADATA_COLUMNS))
    sheets["readme"] = pd.DataFrame({"nowcastbox": list(_README)})
    return sheets


def _panel_sheets(data: MixedFrequencyData) -> dict[str, pd.DataFrame]:
    sheets: dict[str, pd.DataFrame] = {}
    for role, freq in EXCEL_DATA_SHEETS.items():
        cols = data.columns_with_frequency(freq)
        if freq is Frequency.MONTHLY:
            frame = data.to_frame()[cols]
        elif cols:
            frame = pd.concat([data.to_native(c) for c in cols], axis=1)
        else:
            continue
        out = frame.reset_index(drop=True)
        out.insert(0, "date", [str(p) for p in frame.index])
        sheets[role] = out
    rows = []
    for name in data.columns:
        meta = data.metadata[name].to_dict()
        rows.append(
            {
                "series": name,
                "frequency": meta["frequency"],
                "transform": meta["transform"],
                "delay_days": meta["release_delay"],
                "blocks": ";".join(meta["blocks"]),
                "category": meta["category"],
                "description": meta["description"],
                "aggregation": meta["aggregation"],
                "units": meta["units"],
                "transform_applied": meta["transform_applied"],
            }
        )
    sheets["metadata"] = pd.DataFrame(rows, columns=list(EXCEL_METADATA_COLUMNS))
    sheets["readme"] = pd.DataFrame({"nowcastbox": list(_README)})
    return sheets


def write_run_excel(run: Any, path: str | Path) -> Path:
    """Export the results of a pipeline run to an Excel workbook (``outputs.excel``).

    Parameters
    ----------
    run : PipelineRun
        Run of :func:`~nowcastbox.pipeline.run_pipeline` (any object with the same
        attributes works: ``results`` is required; ``spec``, ``data``,
        ``distribution``, ``news``, ``diagnostics``, ``backtest``,
        ``backtest_metrics``, ``empirical_bands``, ``heatmap`` and ``alternatives`` are
        used when present).
    path : str or pathlib.Path
        Output ``.xlsx`` file (parent folders are created).

    Returns
    -------
    pathlib.Path
        The written file, with sheets ``nowcast``, ``loadings``/``factors`` (factor
        models), ``density``, ``news``, ``diagnostics``, ``backtest``/``backtest_rmsfe``/
        ``backtest_metrics``, ``empirical_bands``, ``heatmap``,
        ``alternatives``/``alternatives_range``, ``data`` and ``info`` as available.

    Raises
    ------
    ImportError
        If ``openpyxl`` is not installed.

    Examples
    --------
    >>> import tempfile, pathlib
    >>> from nowcastbox.pipeline import run_pipeline
    >>> run = run_pipeline(
    ...     {
    ...         "target": "gdp",
    ...         "data": {"source": "simulated_dfm", "columns": ["x01", "x02", "x03"]},
    ...         "preprocessing": False,
    ...         "model": {"type": "TwoStepDFM", "factors": 1},
    ...     }
    ... )
    >>> file = write_run_excel(run, pathlib.Path(tempfile.mkdtemp()) / "run.xlsx")
    >>> import pandas as pd
    >>> pd.ExcelFile(file).sheet_names[:3]
    ['nowcast', 'loadings', 'factors']
    """
    _require_openpyxl()
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    tables = _run_tables(run)
    with pd.ExcelWriter(file, engine="openpyxl") as writer:
        for name, frame in tables.items():
            _excel_ready(frame).to_excel(writer, sheet_name=name[:31])
    return file


def _run_tables(run: Any) -> dict[str, pd.DataFrame]:
    results = run.results
    tables: dict[str, pd.DataFrame] = {"nowcast": results.nowcast}
    optional: list[tuple[str, Any, Callable[[Any], pd.DataFrame]]] = [
        ("loadings", results.loadings, lambda x: x),
        ("factors", results.factors, lambda x: x),
        ("density", getattr(run, "distribution", None), lambda x: x.to_frame()),
        ("news", getattr(run, "news", None), lambda x: x.to_frame(by=_news_by(run))),
        ("diagnostics", getattr(run, "diagnostics", None), lambda x: x.series_overview()),
        ("backtest", getattr(run, "backtest", None), lambda x: x.to_frame()),
        ("backtest_rmsfe", getattr(run, "backtest", None), lambda x: x.rmsfe_by_horizon()),
        ("backtest_metrics", getattr(run, "backtest_metrics", None), lambda x: x),
        ("empirical_bands", getattr(run, "empirical_bands", None), lambda x: x.to_frame()),
        ("heatmap", getattr(run, "heatmap", None), lambda x: x.zscores),
        ("alternatives", getattr(run, "alternatives", None), lambda x: x.table()),
        ("alternatives_range", getattr(run, "alternatives", None), lambda x: x.range()),
        ("data", getattr(run, "data", None), lambda x: x.to_frame()),
    ]
    for name, value, convert in optional:
        if value is not None:
            tables[name] = convert(value)
    info = run.to_dict() if hasattr(run, "to_dict") else {}
    tables["info"] = pd.DataFrame(
        {"value": [str(v) for v in info.values()]}, index=pd.Index(list(info), name="key")
    )
    return tables


def _news_by(run: Any) -> str:
    options = getattr(getattr(getattr(run, "spec", None), "outputs", None), "news", None)
    return "series" if options is None else str(options.by)


def _excel_ready(frame: pd.DataFrame) -> pd.DataFrame:
    """Copy with period labels as text (Excel has no period type)."""
    out = frame.copy()
    if isinstance(out.index, pd.PeriodIndex):
        out.index = pd.Index([str(p) for p in out.index], name=out.index.name)
    if isinstance(out.columns, pd.PeriodIndex):
        out.columns = [str(p) for p in out.columns]
    return out


def _load_excel(data: DataSpec) -> MixedFrequencyData:
    if data.path is None:
        raise SpecError([("data.path", "is required")])
    return read_excel_panel(data.path, data.sheets)


# ---------------------------------------------------------------------------- vintage
def apply_vintage(
    panel: MixedFrequencyData, vintage: pd.Timestamp, *, explicit: bool = True
) -> MixedFrequencyData:
    """Keep the observations released by ``vintage`` and end the grid at its month.

    Parameters
    ----------
    panel : MixedFrequencyData
        Panel in levels with ``release_delay`` metadata.
    vintage : pandas.Timestamp
        Information-set date.
    explicit : bool, default True
        ``True`` when the spec asked for a dated vintage: missing delays are an error.
        ``False`` (``vintage: today``): without delays the data are used as loaded.

    Returns
    -------
    MixedFrequencyData
        Pseudo real-time panel (:meth:`MixedFrequencyData.as_of`).

    Raises
    ------
    SpecError
        If delays are missing for a dated vintage, or the vintage precedes the data.

    Examples
    --------
    >>> import pandas as pd
    >>> from nowcastbox.datasets import load_simulated_dfm
    >>> panel = load_simulated_dfm().data
    >>> str(apply_vintage(panel, pd.Timestamp("2015-06-15")).end)
    '2015-06'
    """
    missing = [c for c in panel.columns if panel.metadata[c].release_delay is None]
    if missing:
        if explicit:
            raise SpecError(
                [("data.delay", f"publication delays are needed for a dated vintage: {missing}")]
            )
        logger.info("no publication delays for %s: data used as loaded", missing)
        return panel
    month = pd.Period(vintage, freq="M")
    if month < panel.start:
        raise SpecError([("vintage", f"{vintage.date()} is before the data start {panel.start}")])
    out = panel.as_of(vintage)
    if month < out.end:
        out = out.truncate(end=month)
    return out


# ---------------------------------------------------------------------------- preprocess
def preprocess(panel: MixedFrequencyData, spec: NowcastSpec) -> MixedFrequencyData:
    """Apply the ``preprocessing`` section of ``spec`` to a panel in levels.

    Parameters
    ----------
    panel : MixedFrequencyData
        Panel (after the vintage).
    spec : NowcastSpec
        Validated spec.

    Returns
    -------
    MixedFrequencyData
        Model-ready panel (the target is always kept).

    Raises
    ------
    SpecError
        If ``preprocessing.transform`` names unknown series.
    NowcastDataError
        From :func:`~nowcastbox.preprocessing.prepare_panel`.

    Examples
    --------
    >>> from nowcastbox.pipeline import NowcastSpec
    >>> spec = NowcastSpec.from_dict(
    ...     {"target": "gdp", "data": {"source": "simulated_dfm"}, "preprocessing": False}
    ... )
    >>> raw = load_data(spec)
    >>> preprocess(raw, spec).equals(raw)
    True
    """
    from nowcastbox.preprocessing import prepare_panel

    prep = spec.preprocessing
    if not prep.enabled:
        return panel
    transform: Any
    if prep.transform is True:
        transform = None
    elif prep.transform is False:
        transform = "level"
    else:
        transform = _transform_mapping(panel, prep.transform)
    out = prepare_panel(panel, transform, keep=[spec.target_name], **prep.options)
    return cast("MixedFrequencyData", out)


def _transform_mapping(panel: MixedFrequencyData, overrides: Mapping[str, Any]) -> dict[str, Any]:
    unknown = [k for k in overrides if k not in panel.columns]
    if unknown:
        raise SpecError([("preprocessing.transform", f"series not in the data: {unknown}")])
    base = {c: panel.metadata[c].transform for c in panel.columns}
    base.update(overrides)
    return {c: ("level" if t is None else t) for c, t in base.items()}


# ---------------------------------------------------------------------------- hashing
def frame_hash(frame: pd.DataFrame) -> str:
    """SHA-256 of a DataFrame's index, columns and values (full precision).

    Parameters
    ----------
    frame : pandas.DataFrame
        Table to fingerprint.

    Returns
    -------
    str
        Hex digest.

    Examples
    --------
    >>> import pandas as pd
    >>> a = pd.DataFrame({"x": [1.0, 2.0]})
    >>> frame_hash(a) == frame_hash(a.copy()), frame_hash(a) == frame_hash(a + 1)
    (True, False)
    """
    text = frame.to_csv(float_format="%.17g", lineterminator="\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def data_hash(panel: MixedFrequencyData) -> str:
    """SHA-256 fingerprint of a data vintage (values and metadata).

    Parameters
    ----------
    panel : MixedFrequencyData
        Panel.

    Returns
    -------
    str
        Hex digest; equal panels give equal digests.

    Examples
    --------
    >>> from nowcastbox.datasets import load_simulated_dfm
    >>> panel = load_simulated_dfm().data
    >>> data_hash(panel) == data_hash(panel.copy())
    True
    """
    meta = panel.metadata_frame().astype(str)
    return hashlib.sha256(
        (frame_hash(panel.to_frame()) + frame_hash(meta)).encode("ascii")
    ).hexdigest()
