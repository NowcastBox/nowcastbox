"""Data stage of the pipeline: load, apply the vintage and preprocess.

:func:`load_data` turns the ``data`` section of a :class:`~nowcastbox.pipeline.spec.NowcastSpec`
into a :class:`~nowcastbox.core.data.MixedFrequencyData` panel in levels (built-in
dataset, CSV/Parquet file or the BCB/IBGE/IPEA/FRED connectors);
:func:`apply_vintage` keeps the information available at the spec's vintage (release
rule of :meth:`MixedFrequencyData.as_of`); :func:`preprocess` applies the
transformations and cleaning of :func:`~nowcastbox.preprocessing.prepare_panel`;
:func:`data_hash` fingerprints a data vintage for the snapshots.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

import pandas as pd

from nowcastbox._logging import get_logger
from nowcastbox.core.data import MixedFrequencyData, SeriesMetadata
from nowcastbox.pipeline.spec import ConnectorSeries, DataSpec, NowcastSpec, SpecError

__all__ = [
    "CONNECTORS",
    "apply_vintage",
    "data_hash",
    "frame_hash",
    "load_data",
    "preprocess",
    "read_panel_file",
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
    if data.kind == "dataset":
        panel = _load_dataset(data)
    elif data.kind == "file":
        panel = _load_file(data)
    else:
        panel = _load_connectors(data)
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
    if data.kind == "file":  # already applied when the file was read
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
