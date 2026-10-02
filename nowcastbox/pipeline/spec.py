"""Declarative nowcast specification (innovation I10, plan section 6.2).

A :class:`NowcastSpec` describes a production nowcast end to end: the target, where the
data come from (a built-in dataset, a CSV/Parquet file, an Excel workbook or the
BCB/IBGE/IPEA/FRED connectors), the information set (``vintage``), the preprocessing, the model and the
outputs (nowcast table, news against the previous snapshot, density, diagnostics, HTML
report, optional backtest), plus the directory of versioned snapshots. Specs are parsed
from YAML (or a plain mapping) and validated up front; every problem is reported with
its location in the spec (``model.factors.global: ...``) and close-match suggestions
for misspelled keys.

Example (``nowcastbox init`` writes a commented version)::

    name: pib_brazil
    target: pib
    data:
      source: brazil_nowcast        # built-in dataset, csv, parquet, excel or connectors
      vintage: today
    model:
      type: MixedFreqDFM
      factors: {global: 1, real: 1, soft: 1}
      idiosyncratic: student_t
      long_run_mean: time_varying
    outputs: [nowcast, news, density, report_html]
    snapshot_dir: ./snapshots
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import difflib
import inspect
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from nowcastbox.core.exceptions import NowcastBoxError
from nowcastbox.pipeline._common import plain as _plain
from nowcastbox.pipeline.selection import (
    PRESELECT_EXCLUDED,
    ROBUSTNESS_KEYS,
    SEARCH_BACKTEST_KEYS,
    SEARCH_KEYS,
    SEARCH_OPTIONS,
    PreselectSpec,
    SearchSpec,
    SelectionSpec,
)

__all__ = [
    "CONNECTOR_SOURCES",
    "EXCEL_SOURCE",
    "FILE_SOURCES",
    "MODEL_TYPES",
    "OUTPUT_NAMES",
    "AlternativesOutput",
    "BacktestOutput",
    "ConnectorSeries",
    "DataSpec",
    "DensityOutput",
    "DiagnosticsOutput",
    "EmpiricalBandsOutput",
    "ExcelOutput",
    "HeatmapOutput",
    "ModelSpec",
    "NewsOutput",
    "NowcastSpec",
    "OutputsSpec",
    "PreprocessingSpec",
    "ReportOutput",
    "SpecError",
    "load_spec",
]

MODEL_TYPES: tuple[str, ...] = ("MixedFreqDFM", "TwoStepDFM", "BridgeCombination", "LargeBVAR")
"""Estimators a spec can name (``model.type``)."""

FILE_SOURCES: tuple[str, ...] = ("csv", "parquet")
"""File-based data sources (``data.source``)."""

EXCEL_SOURCE = "excel"
"""Excel workbook data source (``data.source: excel``; optional extra ``[excel]``)."""

_EXCEL_SUFFIXES = (".xlsx", ".xlsm")
_EXCEL_ROLES = ("monthly", "quarterly", "annual", "metadata")

CONNECTOR_SOURCES: tuple[str, ...] = ("bcb", "ibge", "ipea", "fred")
"""Connectors usable in ``data.series.<name>.source``."""

OUTPUT_NAMES: tuple[str, ...] = (
    "nowcast",
    "news",
    "density",
    "diagnostics",
    "report_html",
    "backtest",
    "empirical_bands",
    "heatmap",
    "alternatives",
    "excel",
)
"""Outputs a spec can request (``outputs``)."""

_MODEL_ALIASES: dict[str, str] = {
    "mixedfreqdfm": "MixedFreqDFM",
    "mixed_freq_dfm": "MixedFreqDFM",
    "em": "MixedFreqDFM",
    "twostepdfm": "TwoStepDFM",
    "two_step_dfm": "TwoStepDFM",
    "two_step": "TwoStepDFM",
    "twostep": "TwoStepDFM",
    "bridgecombination": "BridgeCombination",
    "bridge_combination": "BridgeCombination",
    "largebvar": "LargeBVAR",
    "large_bvar": "LargeBVAR",
    "bvar": "LargeBVAR",
}
_METHODS = {
    "MixedFreqDFM": "em",
    "TwoStepDFM": "two_step",
    "BridgeCombination": "bridge_combination",
    "LargeBVAR": "large_bvar",
}
_FACTORLESS = ("BridgeCombination", "LargeBVAR")
_FACTOR_KEYS = ("factors", "n_factors", "factor_lags", "blocks", "n_shocks", "robust", "rmax")
_OUTPUT_ALIASES: dict[str, str] = {
    "report": "report_html",
    "html": "report_html",
    "xlsx": "excel",
    "bands": "empirical_bands",
    "zscores": "heatmap",
    "alternative_models": "alternatives",
}
_EM_ONLY = ("idiosyncratic", "long_run_mean", "outliers", "covid", "robust")
_TOP_KEYS = (
    "name",
    "description",
    "target",
    "data",
    "vintage",
    "preprocessing",
    "model",
    "outputs",
    "selection",
    "snapshot_dir",
    "random_state",
)
_DATA_COMMON = (
    "source",
    "vintage",
    "columns",
    "start",
    "end",
    "frequency",
    "transform",
    "delay",
    "blocks",
    "categories",
)
_DATA_FILE = ("path", "index_column", "legend")
_DATA_EXCEL = ("path", "sheets")
_DATA_CONNECTORS = ("series",)
_SERIES_KEYS = (
    "source",
    "code",
    "table",
    "variable",
    "classifications",
    "frequency",
    "transform",
    "delay",
    "blocks",
    "category",
    "options",
)
_MODEL_KEYS = (
    "type",
    "factors",
    "n_factors",
    "factor_lags",
    "blocks",
    "horizon",
    "n_shocks",
    "robust",
    "rmax",
    "criterion",
    "kwargs",
)
_PREP_KEYS = ("enabled", "transform")
_PREP_EXCLUDED = ("data", "transform", "frequency", "return_report", "keep")
_REPORT_KEYS = ("path", "plotlyjs", "title", "author", "notes", "n_periods")
_BACKTEST_KEYS = (
    "start",
    "end",
    "step",
    "benchmarks",
    "target_offsets",
    "window",
    "window_length",
    "refit_every",
    "model",
    "metrics",
    "periods",
)
_BACKTEST_METRICS = ("rmsfe", "mse", "mae", "bias", "fda", "n")
_PERIOD_SHORTCUTS = ("covid", "ex-covid")
_HEATMAP_KEYS = ("by", "smooth", "window", "last")
_BANDS_KEYS = ("method", "window", "levels", "outliers", "min_errors", "availability")
_BANDS_METHODS = ("mae", "rmse", "quantile")
_ALTERNATIVES_KEYS = ("by", "drop", "refit")
_BENCHMARK_TYPES = ("AR", "RandomWalk", "HistoricalMean", "UMIDAS", "MIDAS", "BridgeBenchmark")
_DATE_RE = re.compile(r"^\d{4}-\d{2}(-\d{2})?$")


# ---------------------------------------------------------------------------- errors
class SpecError(NowcastBoxError, ValueError):
    """Invalid nowcast specification.

    Parameters
    ----------
    issues : sequence of (str, str)
        ``(location, message)`` pairs; the location is a dotted path in the spec
        (``"model.factors.global"``), empty for the document itself.
    source : str, optional
        Where the spec came from (file name), shown in the message.

    Attributes
    ----------
    issues : list of (str, str)
        Every problem found.

    Examples
    --------
    >>> err = SpecError([("model.type", "unknown model 'DFM'")], source="spec.yaml")
    >>> print(err)
    Invalid nowcast spec (spec.yaml): 1 problem
      - model.type: unknown model 'DFM'
    """

    def __init__(self, issues: Sequence[tuple[str, str]], source: str | None = None) -> None:
        self.issues = list(issues)
        self.source = source
        where = f" ({source})" if source else ""
        n = len(self.issues)
        lines = [f"Invalid nowcast spec{where}: {n} problem{'s' if n != 1 else ''}"]
        lines += [f"  - {loc + ': ' if loc else ''}{msg}" for loc, msg in self.issues]
        super().__init__("\n".join(lines))


class _Issues:
    """Collects ``(path, message)`` problems while parsing."""

    def __init__(self) -> None:
        self.items: list[tuple[str, str]] = []

    def add(self, path: str, message: str) -> None:
        self.items.append((path, message))

    def check_keys(self, mapping: Mapping[str, Any], allowed: Iterable[str], path: str) -> None:
        allowed = list(allowed)
        for key in mapping:
            if key not in allowed:
                self.add(_join(path, str(key)), f"unknown key{_suggest(str(key), allowed)}")


def _join(path: str, key: str) -> str:
    return f"{path}.{key}" if path else key


def _suggest(word: str, choices: Iterable[str]) -> str:
    choices = list(choices)
    close = difflib.get_close_matches(word, choices, n=1, cutoff=0.6)
    if close:
        return f" (did you mean {close[0]!r}?)"
    return f"; allowed: {', '.join(sorted(choices))}"


# ---------------------------------------------------------------------------- scalars
def _mapping(value: Any, path: str, issues: _Issues) -> dict[str, Any] | None:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        issues.add(path, f"must be a mapping, got {type(value).__name__}")
        return None
    return {str(k): v for k, v in value.items()}


def _string(value: Any, path: str, issues: _Issues, *, required: bool = False) -> str | None:
    if value is None:
        if required:
            issues.add(path, "is required")
        return None
    if isinstance(value, bool) or not isinstance(value, str | int | float):
        issues.add(path, f"must be a string, got {type(value).__name__}")
        return None
    text = str(value).strip()
    if not text and required:
        issues.add(path, "must not be empty")
        return None
    return text or None


def _integer(
    value: Any, path: str, issues: _Issues, *, minimum: int = 0, default: int | None = None
) -> int | None:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        issues.add(path, f"must be an integer, got {value!r}")
        return default
    if value < minimum:
        issues.add(path, f"must be >= {minimum}, got {value}")
        return default
    return int(value)


def _boolean(value: Any, path: str, issues: _Issues, *, default: bool) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        issues.add(path, f"must be true or false, got {value!r}")
        return default
    return value


def _date(value: Any, path: str, issues: _Issues) -> str | None:
    """ISO date string (``YYYY-MM`` or ``YYYY-MM-DD``) from YAML scalars."""
    if value is None:
        return None
    if isinstance(value, _dt.datetime):
        return value.date().isoformat()
    if isinstance(value, _dt.date):
        return value.isoformat()
    text = str(value).strip()
    try:
        pd.Timestamp(text)
    except (ValueError, TypeError):
        issues.add(path, f"is not a date (use YYYY-MM-DD or YYYY-MM), got {value!r}")
        return None
    return text


def _period(value: Any, path: str, issues: _Issues) -> str | None:
    """Period-like bound (``"2010-01"``, ``"2019Q4"``, ``2010-01-01``)."""
    if value is None:
        return None
    if isinstance(value, _dt.date):
        return _date(value, path, issues)
    text = str(value).strip()
    try:
        pd.Period(text)
    except (ValueError, TypeError):
        issues.add(path, f"is not a period (e.g. '2010-01' or '2019Q4'), got {value!r}")
        return None
    return text


def _str_list(value: Any, path: str, issues: _Issues) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, Sequence) or not all(isinstance(v, str | int) for v in value):
        issues.add(path, "must be a list of names")
        return None
    return tuple(str(v) for v in value)


def _known_kwargs(func: Callable[..., Any], exclude: Iterable[str] = ()) -> list[str]:
    params = inspect.signature(func).parameters
    skip = {"self", *exclude}
    return [
        n
        for n, p in params.items()
        if n not in skip and p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)
    ]


# ---------------------------------------------------------------------------- data
@dataclasses.dataclass(frozen=True)
class ConnectorSeries:
    """One series downloaded from a public API (``data.series.<name>``).

    Parameters
    ----------
    name : str
        Column name in the panel.
    source : {"bcb", "ibge", "ipea", "fred"}
        Connector (:mod:`nowcastbox.data_sources`).
    code : str, optional
        BCB/SGS code, IPEADATA code or FRED series id.
    table, variable : str, optional
        IBGE/SIDRA table and variable.
    classifications : dict, optional
        IBGE/SIDRA classifications (``{11255: 90707}``).
    frequency : str, optional
        Native frequency (``"M"``, ``"Q"``); inferred when omitted.
    transform : optional
        Stationarity transformation (name or code 0-7).
    delay : int, optional
        Publication delay in days (needed for ``vintage`` dates and news).
    blocks : tuple of str
        Factor blocks.
    category : str, optional
        ``hard``, ``soft``, ``financial`` or ``other``.
    options : dict
        Extra keyword arguments of the connector function.

    Examples
    --------
    >>> ConnectorSeries("ipca", "bcb", code="433", frequency="M").source
    'bcb'
    """

    name: str
    source: str
    code: str | None = None
    table: str | None = None
    variable: str | None = None
    classifications: dict[str, Any] = dataclasses.field(default_factory=dict)
    frequency: str | None = None
    transform: Any = None
    delay: int | None = None
    blocks: tuple[str, ...] = ()
    category: str | None = None
    options: dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """YAML-ready mapping (without ``name`` and empty fields).

        Returns
        -------
        dict
            Fields of the series.

        Examples
        --------
        >>> ConnectorSeries("ipca", "bcb", code="433").to_dict()
        {'source': 'bcb', 'code': '433'}
        """
        out = {
            k: _plain(v)
            for k, v in dataclasses.asdict(self).items()
            if k != "name" and v not in (None, (), {}, [])
        }
        return out


@dataclasses.dataclass(frozen=True)
class DataSpec:
    """Where the data come from and which part of it is used (``data``).

    Parameters
    ----------
    source : str
        Built-in dataset name (see :func:`nowcastbox.datasets.list_datasets`),
        ``"csv"``, ``"parquet"``, ``"excel"`` or ``"connectors"``.
    path : pathlib.Path, optional
        Data file (``csv``/``parquet``/``excel``), resolved against the spec directory.
    index_column : str, optional
        Column with the dates of a file (default: the first column).
    legend : pathlib.Path, optional
        CSV with one row per series (``name`` plus any of ``frequency``,
        ``transform``, ``delay_days``, ``blocks`` (``;``-separated), ``category``).
    sheets : dict of str to str, optional
        Excel sheet of each role (``monthly``, ``quarterly``, ``annual``,
        ``metadata``; default: the role names), see
        :func:`~nowcastbox.pipeline.data.read_excel_panel`.
    series : tuple of ConnectorSeries
        Series to download (``connectors``).
    columns : tuple of str, optional
        Subset of series (the target is always kept).
    start, end : str, optional
        Sample bounds.
    frequency, transform, delay, blocks, categories : dict, optional
        Per-series metadata overrides.

    Examples
    --------
    >>> DataSpec(source="brazil_nowcast").kind
    'dataset'
    """

    source: str
    path: Path | None = None
    index_column: str | None = None
    legend: Path | None = None
    sheets: dict[str, str | None] | None = None
    series: tuple[ConnectorSeries, ...] = ()
    columns: tuple[str, ...] | None = None
    start: str | None = None
    end: str | None = None
    frequency: dict[str, str] | None = None
    transform: dict[str, Any] | None = None
    delay: dict[str, int] | None = None
    blocks: dict[str, tuple[str, ...]] | None = None
    categories: dict[str, str] | None = None

    @property
    def kind(self) -> str:
        """``"dataset"``, ``"file"``, ``"excel"`` or ``"connectors"``."""
        if self.source in FILE_SOURCES:
            return "file"
        if self.source == EXCEL_SOURCE:
            return "excel"
        if self.source == "connectors":
            return "connectors"
        return "dataset"

    def to_dict(self) -> dict[str, Any]:
        """YAML-ready mapping (empty fields omitted).

        Returns
        -------
        dict
            The ``data`` section.

        Examples
        --------
        >>> DataSpec(source="nyfed", start="2010-01").to_dict()
        {'source': 'nyfed', 'start': '2010-01'}
        """
        out: dict[str, Any] = {}
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if field.name == "series":
                if value:
                    out["series"] = {s.name: s.to_dict() for s in value}
            elif value is not None:
                out[field.name] = _plain(value)
        return out


def _parse_data(raw: Any, issues: _Issues, base_dir: Path | None) -> tuple[DataSpec | None, Any]:
    """``(DataSpec, vintage value)`` from the ``data`` section."""
    data = _mapping(raw, "data", issues)
    if data is None:
        return None, None
    source = _string(data.get("source"), "data.source", issues, required=True)
    if source is None:
        return None, data.get("vintage")
    source, path = _normalise_source(source, data)
    allowed: list[str] = list(_DATA_COMMON)
    if source in FILE_SOURCES:
        allowed += list(_DATA_FILE)
    elif source == EXCEL_SOURCE:
        allowed += list(_DATA_EXCEL)
    elif source == "connectors":
        allowed += list(_DATA_CONNECTORS)
    issues.check_keys(data, allowed, "data")
    fields: dict[str, Any] = {
        "source": source,
        "columns": _str_list(data.get("columns"), "data.columns", issues),
        "start": _period(data.get("start"), "data.start", issues),
        "end": _period(data.get("end"), "data.end", issues),
        **_parse_overrides(data, issues),
    }
    if source in FILE_SOURCES:
        fields.update(_parse_file_fields(data, path, issues, base_dir))
    elif source == EXCEL_SOURCE:
        fields.update(_parse_excel_fields(data, path, issues, base_dir))
    elif source == "connectors":
        fields["series"] = _parse_series(data.get("series"), issues)
    else:
        _check_dataset(source, fields, issues)
    return DataSpec(**fields), data.get("vintage")


def _normalise_source(source: str, data: Mapping[str, Any]) -> tuple[str, Any]:
    """Accept ``source: path/to/file.csv`` (or ``.xlsx``) as a shortcut for ``source: csv``."""
    lowered = source.lower()
    for ext in FILE_SOURCES:
        if lowered.endswith(f".{ext}"):
            return ext, data.get("path", source)
    if lowered.endswith(_EXCEL_SUFFIXES):
        return EXCEL_SOURCE, data.get("path", source)
    if lowered in (*FILE_SOURCES, EXCEL_SOURCE, "connectors"):
        return lowered, data.get("path")
    return source, data.get("path")


def _parse_overrides(data: Mapping[str, Any], issues: _Issues) -> dict[str, Any]:
    """Per-series metadata overrides (``frequency``, ``transform``, ``delay`` ...)."""
    out: dict[str, Any] = {}
    for key in ("frequency", "transform", "categories"):
        value = _mapping(data.get(key), f"data.{key}", issues)
        out[key] = (
            {k: (v if key == "transform" else str(v)) for k, v in value.items()} if value else None
        )
    delays = _mapping(data.get("delay"), "data.delay", issues)
    if delays:
        out["delay"] = {
            k: d
            for k, v in delays.items()
            if (d := _integer(v, f"data.delay.{k}", issues)) is not None
        }
    else:
        out["delay"] = None
    blocks = _mapping(data.get("blocks"), "data.blocks", issues)
    if blocks:
        out["blocks"] = {
            k: b
            for k, v in blocks.items()
            if (b := _str_list(v, f"data.blocks.{k}", issues)) is not None
        }
    else:
        out["blocks"] = None
    return out


def _resolve_path(value: Any, path: str, issues: _Issues, base_dir: Path | None) -> Path | None:
    text = _string(value, path, issues, required=True)
    if text is None:
        return None
    file = Path(text).expanduser()
    if not file.is_absolute() and base_dir is not None:
        file = base_dir / file
    if not file.is_file():
        issues.add(path, f"file not found: {file}")
    return file


def _parse_file_fields(
    data: Mapping[str, Any], path: Any, issues: _Issues, base_dir: Path | None
) -> dict[str, Any]:
    legend = data.get("legend")
    return {
        "path": _resolve_path(path, "data.path", issues, base_dir),
        "index_column": _string(data.get("index_column"), "data.index_column", issues),
        "legend": None
        if legend is None
        else _resolve_path(legend, "data.legend", issues, base_dir),
    }


def _parse_excel_fields(
    data: Mapping[str, Any], path: Any, issues: _Issues, base_dir: Path | None
) -> dict[str, Any]:
    """``path`` and ``sheets`` of an Excel source."""
    sheets = _mapping(data.get("sheets"), "data.sheets", issues)
    parsed: dict[str, str | None] | None = None
    if sheets:
        issues.check_keys(sheets, _EXCEL_ROLES, "data.sheets")
        parsed = {
            str(k): (None if v in (None, False) else str(v))
            for k, v in sheets.items()
            if k in _EXCEL_ROLES
        }
    return {"path": _resolve_path(path, "data.path", issues, base_dir), "sheets": parsed}


def _parse_series(raw: Any, issues: _Issues) -> tuple[ConnectorSeries, ...]:
    series = _mapping(raw, "data.series", issues)
    if series is None:
        return ()
    if not series:
        issues.add("data.series", "connectors need at least one series")
        return ()
    out = []
    for name, entry in series.items():
        item = _parse_one_series(name, entry, issues)
        if item is not None:
            out.append(item)
    return tuple(out)


def _parse_one_series(name: str, entry: Any, issues: _Issues) -> ConnectorSeries | None:
    path = f"data.series.{name}"
    spec = _mapping(entry, path, issues)
    if spec is None:
        return None
    issues.check_keys(spec, _SERIES_KEYS, path)
    source = _string(spec.get("source"), f"{path}.source", issues, required=True)
    if source is None:
        return None
    source = source.lower()
    if source not in CONNECTOR_SOURCES:
        issues.add(
            f"{path}.source", f"unknown connector {source!r}{_suggest(source, CONNECTOR_SOURCES)}"
        )
        return None
    fields = {
        "code": _string(spec.get("code"), f"{path}.code", issues),
        "table": _string(spec.get("table"), f"{path}.table", issues),
        "variable": _string(spec.get("variable"), f"{path}.variable", issues),
    }
    needed = ("table",) if source == "ibge" else ("code",)
    for key in needed:
        if fields[key] is None:
            issues.add(f"{path}.{key}", f"is required for source {source!r}")
    return ConnectorSeries(
        name=name,
        source=source,
        classifications=_mapping(spec.get("classifications"), f"{path}.classifications", issues)
        or {},
        frequency=_string(spec.get("frequency"), f"{path}.frequency", issues),
        transform=spec.get("transform"),
        delay=_integer(spec.get("delay"), f"{path}.delay", issues),
        blocks=_str_list(spec.get("blocks"), f"{path}.blocks", issues) or (),
        category=_string(spec.get("category"), f"{path}.category", issues),
        options=_mapping(spec.get("options"), f"{path}.options", issues) or {},
        **fields,
    )


def _check_dataset(name: str, fields: Mapping[str, Any], issues: _Issues) -> None:
    """The dataset exists, is a panel and holds the requested columns."""
    from nowcastbox.datasets import dataset_info, list_datasets

    catalogue = list_datasets()
    if name not in catalogue.index:
        issues.add(
            "data.source",
            f"unknown data source {name!r}{_suggest(name, list(catalogue.index))} "
            "(built-in dataset names, 'csv', 'parquet' or 'connectors')",
        )
        return
    if name in ("brazil_calendar", "brazil_vintages"):
        issues.add("data.source", f"dataset {name!r} is not a panel of series")
        return
    names = [s["name"] for s in dataset_info(name).get("series") or [] if "name" in s]
    if names and fields.get("columns"):
        for col in fields["columns"]:
            if col not in names:
                issues.add(
                    "data.columns", f"{col!r} is not a series of {name}{_suggest(col, names)}"
                )


# ---------------------------------------------------------------------------- preprocessing
@dataclasses.dataclass(frozen=True)
class PreprocessingSpec:
    """Preprocessing (``preprocessing``): transformations and cleaning.

    Parameters
    ----------
    enabled : bool, default True
        ``False`` uses the data exactly as loaded.
    transform : bool or dict, default True
        ``True``: the transformations recorded in the series metadata (dataset legend,
        ``data.transform``); ``False``: none; mapping: per-series overrides.
    options : dict
        Options of :func:`~nowcastbox.preprocessing.prepare_panel` (``max_na_prop``,
        ``replace_outliers``, ``na_method`` ...).

    Examples
    --------
    >>> PreprocessingSpec(options={"max_na_prop": 0.5}).to_dict()
    {'enabled': True, 'transform': True, 'max_na_prop': 0.5}
    """

    enabled: bool = True
    transform: bool | dict[str, Any] = True
    options: dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """YAML-ready mapping.

        Returns
        -------
        dict
            The ``preprocessing`` section.

        Examples
        --------
        >>> PreprocessingSpec(enabled=False).to_dict()
        {'enabled': False, 'transform': True}
        """
        return {
            "enabled": self.enabled,
            "transform": _plain(self.transform),
            **_plain(self.options),
        }


def _parse_preprocessing(raw: Any, issues: _Issues) -> PreprocessingSpec:
    if isinstance(raw, bool):
        return PreprocessingSpec(enabled=raw)
    prep = _mapping(raw, "preprocessing", issues)
    if prep is None:
        return PreprocessingSpec()
    from nowcastbox.preprocessing import prepare_panel

    options_allowed = _known_kwargs(prepare_panel, _PREP_EXCLUDED)
    issues.check_keys(prep, [*_PREP_KEYS, *options_allowed], "preprocessing")
    transform = prep.get("transform", True)
    if not isinstance(transform, bool | Mapping):
        issues.add(
            "preprocessing.transform", "must be true, false or a mapping series -> transform"
        )
        transform = True
    return PreprocessingSpec(
        enabled=_boolean(prep.get("enabled"), "preprocessing.enabled", issues, default=True),
        transform=dict(transform) if isinstance(transform, Mapping) else transform,
        options={k: v for k, v in prep.items() if k in options_allowed},
    )


# ---------------------------------------------------------------------------- model
@dataclasses.dataclass(frozen=True)
class ModelSpec:
    """Estimator (``model``).

    Parameters
    ----------
    type : {"MixedFreqDFM", "TwoStepDFM", "BridgeCombination", "LargeBVAR"}
        Estimator class (``bridge_combination`` is an alias of
        :class:`~nowcastbox.models.BridgeCombination`, which takes no factor settings:
        its options are ``max_monthly``, ``combine``, ``extrapolation``...;
        ``large_bvar``/``bvar`` of :class:`~nowcastbox.models.LargeBVAR`, whose options
        are ``lags``, ``prior``, ``prior_mean``, ``n_draws``...).
    n_factors : int, dict or "auto"
        Number of factors (per block with a mapping, plan §6.2 ``factors``);
        ``"auto"``: Bai & Ng (2002) criterion.
    factor_lags : int, default 1
        Order of the factor VAR.
    blocks : optional
        Block structure (``"data"`` uses the series metadata; default when
        ``n_factors`` is a mapping).
    horizon : int, default 1
        Target periods forecast after the current one.
    n_shocks : int or "auto", optional
        Dynamic shocks of the two-step model.
    rmax : int, default 8
        Largest number of factors considered with ``n_factors="auto"``.
    criterion : str, default "IC2"
        Bai-Ng criterion with ``n_factors="auto"``.
    options : dict
        Further estimator arguments (``idiosyncratic``, ``long_run_mean``, ``outliers``,
        ``covid``, ``max_iter`` ...); ``robust: true`` expands to
        ``idiosyncratic="student_t", outliers="auto"``.

    Examples
    --------
    >>> ModelSpec(type="MixedFreqDFM", n_factors={"global": 1}).method
    'em'
    """

    type: str = "MixedFreqDFM"
    n_factors: int | dict[str, int] | str = 1
    factor_lags: int = 1
    blocks: Any = None
    horizon: int = 1
    n_shocks: int | str | None = None
    rmax: int = 8
    criterion: str = "IC2"
    options: dict[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def method(self) -> str:
        """``"em"``, ``"two_step"`` (:func:`nowcastbox.nowcast`), ``"bridge_combination"`` or ``"large_bvar"``."""
        return _METHODS[self.type]

    @property
    def estimator_class(self) -> type:
        """The estimator class named by :attr:`type`."""
        from nowcastbox import models

        return getattr(models, self.type)

    def to_dict(self) -> dict[str, Any]:
        """YAML-ready mapping.

        Returns
        -------
        dict
            The ``model`` section (``factors`` holds :attr:`n_factors`).

        Examples
        --------
        >>> ModelSpec(type="TwoStepDFM", n_factors=2).to_dict()["factors"]
        2
        >>> ModelSpec(type="BridgeCombination", options={"combine": "median"}).to_dict()
        {'type': 'BridgeCombination', 'horizon': 1, 'combine': 'median'}
        >>> ModelSpec(type="LargeBVAR", horizon=0, options={"lags": 2}).to_dict()
        {'type': 'LargeBVAR', 'horizon': 0, 'lags': 2}
        """
        if self.type in _FACTORLESS:
            return {"type": self.type, "horizon": self.horizon, **_plain(self.options)}
        out: dict[str, Any] = {
            "type": self.type,
            "factors": _plain(self.n_factors),
            "factor_lags": self.factor_lags,
            "horizon": self.horizon,
        }
        if self.blocks is not None:
            out["blocks"] = _plain(self.blocks)
        if self.n_shocks is not None:
            out["n_shocks"] = self.n_shocks
        if self.n_factors == "auto":
            out.update(rmax=self.rmax, criterion=self.criterion)
        out.update(_plain(self.options))
        return out


def _parse_model(raw: Any, issues: _Issues) -> ModelSpec | None:
    model = _mapping(raw, "model", issues)
    if model is None:
        return None
    name = _string(model.get("type", "MixedFreqDFM"), "model.type", issues, required=True)
    if name is None:
        return None
    key = name.strip().lower().replace("-", "_")
    if key not in _MODEL_ALIASES:
        issues.add("model.type", f"unknown model {name!r}{_suggest(name, MODEL_TYPES)}")
        return None
    mtype = _MODEL_ALIASES[key]
    spec = ModelSpec(type=mtype)
    extra = _mapping(model.pop("kwargs", None), "model.kwargs", issues) or {}
    options = {**{k: v for k, v in model.items() if k not in _MODEL_KEYS}, **extra}
    allowed = _known_kwargs(spec.estimator_class.__init__)
    structural = ("n_factors", "factor_lags", "blocks", "horizon", "n_shocks")
    if mtype in _FACTORLESS:
        return _parse_factorless(mtype, model, options, allowed, issues)
    issues.check_keys(options, [a for a in allowed if a not in structural] + ["robust"], "model")
    if "factors" in model and "n_factors" in model:
        issues.add("model.n_factors", "give either 'factors' or 'n_factors', not both")
    n_factors = _parse_factors(model.get("factors", model.get("n_factors", 1)), issues)
    options = _robust_options(mtype, options, model.get("robust"), issues)
    blocks = model.get("blocks")
    if mtype == "TwoStepDFM":
        _check_two_step(n_factors, blocks, options, issues)
    if blocks is None and isinstance(n_factors, dict):
        blocks = "data"
    return dataclasses.replace(
        spec,
        n_factors=n_factors,
        factor_lags=_integer(
            model.get("factor_lags"), "model.factor_lags", issues, minimum=1, default=1
        )
        or 1,
        blocks=blocks,
        horizon=_integer(model.get("horizon"), "model.horizon", issues, default=1) or 0,
        n_shocks=_parse_shocks(model.get("n_shocks"), mtype, issues),
        rmax=_integer(model.get("rmax"), "model.rmax", issues, minimum=1, default=8) or 8,
        criterion=_string(model.get("criterion"), "model.criterion", issues) or "IC2",
        options=options,
    )


def _parse_factorless(
    mtype: str,
    model: Mapping[str, Any],
    options: dict[str, Any],
    allowed: list[str],
    issues: _Issues,
) -> ModelSpec:
    """``bridge_combination`` / ``large_bvar``: no factor settings, estimator options only."""
    for key in (*_FACTOR_KEYS, "criterion"):
        if key in model:
            issues.add(f"model.{key}", f"is not an option of {mtype} (no factors)")
    issues.check_keys(options, [a for a in allowed if a != "horizon"], "model")
    horizon = _integer(model.get("horizon"), "model.horizon", issues, minimum=0, default=1)
    return ModelSpec(
        type=mtype,
        n_factors=0,
        horizon=1 if horizon is None else horizon,
        options={k: v for k, v in options.items() if k in allowed},
    )


def _parse_factors(value: Any, issues: _Issues) -> int | dict[str, int] | str:
    path = "model.factors"
    if isinstance(value, str) and value.strip().lower() == "auto":
        return "auto"
    if isinstance(value, Mapping):
        if not value:
            issues.add(path, "the mapping block -> number of factors is empty")
        out = {}
        for block, n in value.items():
            k = _integer(n, f"{path}.{block}", issues, minimum=1)
            if k is not None:
                out[str(block)] = k
        return out
    n = _integer(value, path, issues, minimum=1)
    return 1 if n is None else n


def _parse_shocks(value: Any, mtype: str, issues: _Issues) -> int | str | None:
    if value is None:
        return None
    if mtype != "TwoStepDFM":
        issues.add("model.n_shocks", "is only used by TwoStepDFM")
        return None
    if value == "auto":
        return "auto"
    return _integer(value, "model.n_shocks", issues, minimum=1)


def _robust_options(
    mtype: str, options: dict[str, Any], robust: Any, issues: _Issues
) -> dict[str, Any]:
    flag = _boolean(robust, "model.robust", issues, default=False)
    options = {k: v for k, v in options.items() if k != "robust"}
    if flag and mtype == "MixedFreqDFM":
        options.setdefault("idiosyncratic", "student_t")
        options.setdefault("outliers", "auto")
    elif flag:
        issues.add("model.robust", "robust estimation is only available for MixedFreqDFM")
    return options


def _check_two_step(
    n_factors: Any, blocks: Any, options: Mapping[str, Any], issues: _Issues
) -> None:
    if isinstance(n_factors, dict):
        issues.add("model.factors", "TwoStepDFM takes an integer number of factors, not blocks")
    if blocks is not None:
        issues.add("model.blocks", "blocks are only supported by MixedFreqDFM")
    for key in _EM_ONLY:
        if key in options:
            issues.add(f"model.{key}", "is only available for MixedFreqDFM")


# ---------------------------------------------------------------------------- outputs
@dataclasses.dataclass(frozen=True)
class NewsOutput:
    """News decomposition output (``outputs.news``).

    Parameters
    ----------
    against : str, default "previous_snapshot"
        ``"previous_snapshot"`` (the latest snapshot of the same spec name) or a date:
        the pseudo real-time vintage of the same data at that date.
    by : str, default "series"
        Grouping of the stored table (``series``, ``category``, ``block`` ...).

    Examples
    --------
    >>> NewsOutput().against
    'previous_snapshot'
    """

    against: str = "previous_snapshot"
    by: str = "series"


@dataclasses.dataclass(frozen=True)
class DensityOutput:
    """Density nowcast output (``outputs.density``, innovation I5).

    Parameters
    ----------
    n_boot : int, default 0
        Bootstrap replications with re-estimation (``0``: filtering uncertainty).

    Examples
    --------
    >>> DensityOutput(n_boot=50).n_boot
    50
    """

    n_boot: int = 0


@dataclasses.dataclass(frozen=True)
class DiagnosticsOutput:
    """DFM diagnostics output (``outputs.diagnostics``, innovation I9).

    Parameters
    ----------
    options : dict
        Options of :func:`~nowcastbox.diagnostics.run_diagnostics`.

    Examples
    --------
    >>> DiagnosticsOutput({"trim": 0.2}).options
    {'trim': 0.2}
    """

    options: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class ReportOutput:
    """HTML report output (``outputs.report_html``).

    Parameters
    ----------
    path : pathlib.Path, optional
        Extra location of the report (the snapshot always gets ``report.html``).
    plotlyjs : {"inline", "cdn"}, default "inline"
        How Plotly is embedded (``inline`` works offline).
    title, author, notes : str, optional
        Report header fields.
    n_periods : int, default 12
        Target periods shown in the path chart.

    Examples
    --------
    >>> ReportOutput(plotlyjs="cdn").plotlyjs
    'cdn'
    """

    path: Path | None = None
    plotlyjs: str = "inline"
    title: str | None = None
    author: str | None = None
    notes: str | None = None
    n_periods: int = 12


@dataclasses.dataclass(frozen=True)
class BacktestOutput:
    """Pseudo real-time backtest output (``outputs.backtest``).

    Parameters
    ----------
    start, end : str
        First and last vintage dates.
    step : str, default "M"
        Vintage frequency.
    benchmarks : tuple of (str, dict)
        Benchmarks of :mod:`nowcastbox.benchmarks` with their arguments
        (default: ``AR(p=1)``).
    target_offsets : tuple of int, default (-1, 0, 1)
        Backcast/nowcast/forecast horizons evaluated.
    window : str, default "expanding"
        Estimation window.
    window_length : int, optional
        Length of a rolling window.
    refit_every : int, default 1
        Re-estimate every k vintages.
    model : dict
        Overrides of the model options in the backtest (e.g. ``{max_iter: 50}``).
    metrics : tuple of str, default ("rmsfe", "mae", "bias", "n")
        Metrics of the stored accuracy table (``backtest_metrics``), see
        :meth:`~nowcastbox.evaluation.BacktestResults.metrics`; ``"fda"`` adds the
        forecast directional accuracy.
    periods : str or dict, optional
        Sub-periods of the accuracy table: ``"covid"``, ``"ex-covid"`` or
        ``{label: [first, last]}`` with target periods (``null`` = open end).

    Examples
    --------
    >>> BacktestOutput("2019-01-01", "2019-12-01").benchmarks
    (('AR', {'p': 1}),)
    """

    start: str
    end: str
    step: str = "M"
    benchmarks: tuple[tuple[str, dict[str, Any]], ...] = (("AR", {"p": 1}),)
    target_offsets: tuple[int, ...] = (-1, 0, 1)
    window: str = "expanding"
    window_length: int | None = None
    refit_every: int = 1
    model: dict[str, Any] = dataclasses.field(default_factory=dict)
    metrics: tuple[str, ...] = ("rmsfe", "mae", "bias", "n")
    periods: str | dict[str, list[str | None]] | None = None


@dataclasses.dataclass(frozen=True)
class EmpiricalBandsOutput:
    """Empirical error bands around the nowcast (``outputs.empirical_bands``).

    Built by :func:`~nowcastbox.density.empirical_bands` from the errors of the
    ``backtest`` output at the same horizon (needs ``outputs.backtest``).

    Parameters
    ----------
    method : {"mae", "rmse", "quantile"}, default "mae"
        Scale of the past errors (ECB convention: MAE).
    window : str or None, default "10Y"
        Rolling window of past errors (``None``: every past error).
    levels : tuple of float, default (0.575, 0.68, 0.9)
        Band levels.
    outliers : {"exclude", "winsorize"}, optional
        Treatment of outlying past errors.
    min_errors : int, default 8
        Minimum number of past errors at a horizon.
    availability : {"release", "period_end"}, default "release"
        When a past error counts as known.

    Examples
    --------
    >>> EmpiricalBandsOutput().levels
    (0.575, 0.68, 0.9)
    """

    method: str = "mae"
    window: str | None = "10Y"
    levels: tuple[float, ...] = (0.575, 0.68, 0.9)
    outliers: str | None = None
    min_errors: int = 8
    availability: str = "release"


@dataclasses.dataclass(frozen=True)
class HeatmapOutput:
    """Z-score heatmap of the indicators (``outputs.heatmap``).

    Built by :func:`~nowcastbox.diagnostics.indicator_zscores` on the model-ready panel
    of the vintage (moments use only data available at the vintage).

    Parameters
    ----------
    by : str, optional
        Grouping (``"category"``, ``"block"``, ``"frequency"``; default: by series).
    smooth : str or None, default "mm"
        Smoothing of the monthly series (Mariano-Murasawa 1-2-3-2-1 weights).
    window : int, optional
        Rolling window of the moments, in base periods (default: whole sample).
    last : int, default 24
        Number of most recent periods drawn in the report.

    Examples
    --------
    >>> HeatmapOutput(by="category").last
    24
    """

    by: str | None = None
    smooth: str | None = "mm"
    window: int | None = None
    last: int = 24


@dataclasses.dataclass(frozen=True)
class AlternativesOutput:
    """Nowcasts of alternative models without one or two groups (``outputs.alternatives``).

    Built by :func:`~nowcastbox.experiment.alternative_models`.

    Parameters
    ----------
    by : str, default "category"
        Grouping of the predictors (``"category"`` or ``"block"``).
    drop : tuple of int, default (1, 2)
        Numbers of groups removed at a time.
    refit : bool, default True
        Re-estimate every alternative (``False``: re-filter with the base parameters).

    Examples
    --------
    >>> AlternativesOutput(refit=False).drop
    (1, 2)
    """

    by: str = "category"
    drop: tuple[int, ...] = (1, 2)
    refit: bool = True


@dataclasses.dataclass(frozen=True)
class ExcelOutput:
    """Excel export of the run (``outputs.excel``; optional extra ``[excel]``).

    Parameters
    ----------
    path : pathlib.Path, optional
        Extra location of the workbook (resolved against the spec directory); the
        snapshot, when written, always gets ``results.xlsx``. See
        :func:`~nowcastbox.pipeline.data.write_run_excel` for the sheets.

    Examples
    --------
    >>> ExcelOutput().path is None
    True
    """

    path: Path | None = None


@dataclasses.dataclass(frozen=True)
class OutputsSpec:
    """Requested outputs (``outputs``); ``None`` means "not requested".

    Parameters
    ----------
    news : NewsOutput, optional
    density : DensityOutput, optional
    diagnostics : DiagnosticsOutput, optional
    report_html : ReportOutput, optional
    backtest : BacktestOutput, optional
    empirical_bands : EmpiricalBandsOutput, optional
    heatmap : HeatmapOutput, optional
    alternatives : AlternativesOutput, optional
    excel : ExcelOutput, optional

    Examples
    --------
    >>> OutputsSpec(density=DensityOutput()).names
    ('nowcast', 'density')
    """

    news: NewsOutput | None = None
    density: DensityOutput | None = None
    diagnostics: DiagnosticsOutput | None = None
    report_html: ReportOutput | None = None
    backtest: BacktestOutput | None = None
    empirical_bands: EmpiricalBandsOutput | None = None
    heatmap: HeatmapOutput | None = None
    alternatives: AlternativesOutput | None = None
    excel: ExcelOutput | None = None

    @property
    def names(self) -> tuple[str, ...]:
        """Requested outputs (``nowcast`` is always produced)."""
        return ("nowcast", *(n for n in OUTPUT_NAMES[1:] if getattr(self, n) is not None))

    def to_dict(self) -> dict[str, Any]:
        """YAML-ready mapping.

        Returns
        -------
        dict
            ``{output: options}`` for every requested output.

        Examples
        --------
        >>> OutputsSpec(density=DensityOutput(n_boot=5)).to_dict()
        {'nowcast': True, 'density': {'n_boot': 5}}
        """
        out: dict[str, Any] = {"nowcast": True}
        for name in OUTPUT_NAMES[1:]:
            value = getattr(self, name)
            if value is None:
                continue
            fields = dataclasses.asdict(value)
            if name == "diagnostics":
                fields = fields["options"]
            if name == "backtest":
                fields["benchmarks"] = [{"type": t, **kw} for t, kw in value.benchmarks]
            fields.update(_none_sentinels(name, fields))
            out[name] = _plain({k: v for k, v in fields.items() if v is not None}) or True
        return out


def _none_sentinels(name: str, fields: Mapping[str, Any]) -> dict[str, Any]:
    """YAML spellings of the meaningful ``None`` options (``window: all``, ``smooth: none``)."""
    if name == "empirical_bands" and fields.get("window") is None:
        return {"window": "all"}
    if name == "heatmap" and fields.get("smooth") is None:
        return {"smooth": "none"}
    return {}


def _parse_outputs(raw: Any, issues: _Issues, base_dir: Path | None) -> OutputsSpec:
    entries = _output_entries(raw, issues)
    parsers: dict[str, Callable[[dict[str, Any], str], Any]] = {
        "news": lambda o, p: _parse_news(o, p, issues),
        "density": lambda o, p: _parse_density(o, p, issues),
        "diagnostics": lambda o, p: _parse_diagnostics(o, p, issues),
        "report_html": lambda o, p: _parse_report(o, p, issues, base_dir),
        "backtest": lambda o, p: _parse_backtest(o, p, issues),
        "excel": lambda o, p: _parse_excel_output(o, p, issues, base_dir),
        "empirical_bands": lambda o, p: _parse_bands(o, p, issues),
        "heatmap": lambda o, p: _parse_heatmap(o, p, issues),
        "alternatives": lambda o, p: _parse_alternatives(o, p, issues),
    }
    fields: dict[str, Any] = {}
    for name, (options, path) in entries.items():
        if name in parsers:
            fields[name] = parsers[name](options, path)
    return OutputsSpec(**fields)


def _output_items(raw: Any, issues: _Issues) -> list[tuple[str, Any, str]]:
    """``[(name, value, path)]`` from the list or mapping form of ``outputs``."""
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = [raw]
    if isinstance(raw, Mapping):
        return [(str(k), v, f"outputs.{k}") for k, v in raw.items()]
    if not isinstance(raw, Sequence):
        issues.add("outputs", "must be a list of output names or a mapping name -> options")
        return []
    items: list[tuple[str, Any, str]] = []
    for i, entry in enumerate(raw):
        if isinstance(entry, Mapping) and len(entry) == 1:
            ((k, v),) = entry.items()
            items.append((str(k), v, f"outputs[{i}].{k}"))
        else:
            items.append((str(entry), True, f"outputs[{i}]"))
    return items


def _output_entries(raw: Any, issues: _Issues) -> dict[str, tuple[dict[str, Any], str]]:
    """``{canonical name: (options, path)}`` of the requested outputs."""
    out: dict[str, tuple[dict[str, Any], str]] = {}
    for name, value, path in _output_items(raw, issues):
        canonical = _OUTPUT_ALIASES.get(name, name)
        if canonical not in OUTPUT_NAMES:
            issues.add(path, f"unknown output {name!r}{_suggest(name, OUTPUT_NAMES)}")
        elif value is True or value is None:
            out[canonical] = ({}, path)
        elif value is not False:
            options = _mapping(value, path, issues)
            if options is not None:
                out[canonical] = (options, path)
    return out


def _parse_news(options: dict[str, Any], path: str, issues: _Issues) -> NewsOutput:
    issues.check_keys(options, ("against", "by"), path)
    against = options.get("against", "previous_snapshot")
    if against != "previous_snapshot":
        against = _date(against, f"{path}.against", issues) or "previous_snapshot"
    return NewsOutput(against=str(against), by=str(options.get("by", "series")))


def _parse_density(options: dict[str, Any], path: str, issues: _Issues) -> DensityOutput:
    issues.check_keys(options, ("n_boot",), path)
    return DensityOutput(n_boot=_integer(options.get("n_boot"), f"{path}.n_boot", issues) or 0)


def _parse_diagnostics(options: dict[str, Any], path: str, issues: _Issues) -> DiagnosticsOutput:
    from nowcastbox.diagnostics import run_diagnostics

    issues.check_keys(options, _known_kwargs(run_diagnostics, ("results", "data")), path)
    return DiagnosticsOutput(options=dict(options))


def _parse_report(
    options: dict[str, Any], path: str, issues: _Issues, base_dir: Path | None
) -> ReportOutput:
    issues.check_keys(options, _REPORT_KEYS, path)
    plotlyjs = str(options.get("plotlyjs", "inline"))
    if plotlyjs not in ("inline", "cdn"):
        issues.add(f"{path}.plotlyjs", f"must be 'inline' or 'cdn', got {plotlyjs!r}")
        plotlyjs = "inline"
    target = _string(options.get("path"), f"{path}.path", issues)
    file = None
    if target is not None:
        file = Path(target).expanduser()
        if not file.is_absolute() and base_dir is not None:
            file = base_dir / file
    return ReportOutput(
        path=file,
        plotlyjs=plotlyjs,
        title=_string(options.get("title"), f"{path}.title", issues),
        author=_string(options.get("author"), f"{path}.author", issues),
        notes=_string(options.get("notes"), f"{path}.notes", issues),
        n_periods=_integer(
            options.get("n_periods"), f"{path}.n_periods", issues, minimum=1, default=12
        )
        or 12,
    )


def _parse_excel_output(
    options: dict[str, Any], path: str, issues: _Issues, base_dir: Path | None
) -> ExcelOutput:
    issues.check_keys(options, ("path",), path)
    target = _string(options.get("path"), f"{path}.path", issues)
    if target is None:
        return ExcelOutput()
    file = Path(target).expanduser()
    if not file.is_absolute() and base_dir is not None:
        file = base_dir / file
    return ExcelOutput(path=file)


def _parse_backtest(options: dict[str, Any], path: str, issues: _Issues) -> BacktestOutput | None:
    issues.check_keys(options, _BACKTEST_KEYS, path)
    start = _date(options.get("start"), f"{path}.start", issues)
    end = _date(options.get("end"), f"{path}.end", issues)
    for key, value in (("start", start), ("end", end)):
        if value is None and options.get(key) is None:
            issues.add(f"{path}.{key}", "is required (first/last vintage date)")
    offsets = options.get("target_offsets", (-1, 0, 1))
    if not isinstance(offsets, Sequence) or not all(
        isinstance(o, int) and not isinstance(o, bool) for o in offsets
    ):
        issues.add(f"{path}.target_offsets", "must be a list of integers")
        offsets = (-1, 0, 1)
    fields = {
        "step": str(options.get("step", "M")),
        "benchmarks": _parse_benchmarks(options.get("benchmarks"), f"{path}.benchmarks", issues),
        "target_offsets": tuple(int(o) for o in offsets),
        "window": str(options.get("window", "expanding")),
        "window_length": _integer(
            options.get("window_length"), f"{path}.window_length", issues, minimum=1
        ),
        "refit_every": _integer(
            options.get("refit_every"), f"{path}.refit_every", issues, minimum=1, default=1
        )
        or 1,
        "model": _mapping(options.get("model"), f"{path}.model", issues) or {},
        "metrics": _parse_metrics(options.get("metrics"), f"{path}.metrics", issues),
        "periods": _parse_periods(options.get("periods"), f"{path}.periods", issues),
    }
    if start is None or end is None:
        return None
    return BacktestOutput(start=start, end=end, **fields)


def _parse_metrics(value: Any, path: str, issues: _Issues) -> tuple[str, ...]:
    """``outputs.backtest.metrics``: names of :meth:`BacktestResults.metrics`."""
    default = ("rmsfe", "mae", "bias", "n")
    names = _str_list(value, path, issues)
    if not names:
        return default
    unknown = [n for n in names if n not in _BACKTEST_METRICS]
    if unknown:
        issues.add(path, f"unknown metrics {unknown}; allowed: {', '.join(_BACKTEST_METRICS)}")
        return default
    return names


def _parse_periods(
    value: Any, path: str, issues: _Issues
) -> str | dict[str, list[str | None]] | None:
    """``outputs.backtest.periods``: a shortcut or ``{label: [first, last]}``."""
    if value is None:
        return None
    if isinstance(value, str):
        if value.lower() not in _PERIOD_SHORTCUTS:
            issues.add(path, f"must be 'covid', 'ex-covid' or a mapping, got {value!r}")
            return None
        return value.lower()
    periods = _mapping(value, path, issues)
    if not periods:
        return None
    out: dict[str, list[str | None]] = {}
    for label, bounds in periods.items():
        where = f"{path}.{label}"
        if not isinstance(bounds, Sequence) or isinstance(bounds, str) or len(bounds) != 2:
            issues.add(where, "must be a pair [first, last] of target periods (null = open)")
            continue
        out[label] = [_period(b, where, issues) for b in bounds]
    return out or None


def _choice(value: Any, path: str, issues: _Issues, choices: Sequence[str], default: Any) -> Any:
    if value is None:
        return default
    if value not in choices:
        issues.add(path, f"must be one of {', '.join(choices)}, got {value!r}")
        return default
    return value


def _parse_bands(options: dict[str, Any], path: str, issues: _Issues) -> EmpiricalBandsOutput:
    issues.check_keys(options, _BANDS_KEYS, path)
    levels = options.get("levels", (0.575, 0.68, 0.9))
    if isinstance(levels, int | float) and not isinstance(levels, bool):
        levels = [levels]
    if not isinstance(levels, Sequence) or not all(
        isinstance(v, int | float) and not isinstance(v, bool) and 0 < v < 1 for v in levels
    ):
        issues.add(f"{path}.levels", "must be a list of numbers in (0, 1)")
        levels = (0.575, 0.68, 0.9)
    window = options.get("window", "10Y")
    return EmpiricalBandsOutput(
        method=_choice(options.get("method"), f"{path}.method", issues, _BANDS_METHODS, "mae"),
        window=None if window in (None, False, "all") else str(window),
        levels=tuple(float(v) for v in levels),
        outliers=_choice(
            options.get("outliers"), f"{path}.outliers", issues, ("exclude", "winsorize"), None
        ),
        min_errors=_integer(
            options.get("min_errors"), f"{path}.min_errors", issues, minimum=1, default=8
        )
        or 8,
        availability=_choice(
            options.get("availability"),
            f"{path}.availability",
            issues,
            ("release", "period_end"),
            "release",
        ),
    )


def _parse_heatmap(options: dict[str, Any], path: str, issues: _Issues) -> HeatmapOutput:
    issues.check_keys(options, _HEATMAP_KEYS, path)
    by = _choice(
        options.get("by"), f"{path}.by", issues, ("series", "category", "block", "frequency"), None
    )
    smooth = options.get("smooth", "mm")
    if smooth in (None, False, "none"):
        smooth = None
    else:
        smooth = _choice(smooth, f"{path}.smooth", issues, ("mm",), "mm")
    return HeatmapOutput(
        by=None if by == "series" else by,
        smooth=smooth,
        window=_integer(options.get("window"), f"{path}.window", issues, minimum=2),
        last=_integer(options.get("last"), f"{path}.last", issues, minimum=1, default=24) or 24,
    )


def _parse_alternatives(options: dict[str, Any], path: str, issues: _Issues) -> AlternativesOutput:
    issues.check_keys(options, _ALTERNATIVES_KEYS, path)
    drop = options.get("drop", (1, 2))
    if isinstance(drop, int) and not isinstance(drop, bool):
        drop = [drop]
    if (
        not isinstance(drop, Sequence)
        or not drop
        or not all(isinstance(d, int) and not isinstance(d, bool) and d >= 1 for d in drop)
    ):
        issues.add(f"{path}.drop", "must be a positive integer or a list of them")
        drop = (1, 2)
    return AlternativesOutput(
        by=_choice(options.get("by"), f"{path}.by", issues, ("category", "block"), "category"),
        drop=tuple(int(d) for d in drop),
        refit=_boolean(options.get("refit"), f"{path}.refit", issues, default=True),
    )


def _parse_benchmarks(
    raw: Any, path: str, issues: _Issues
) -> tuple[tuple[str, dict[str, Any]], ...]:
    if raw is None:
        return (("AR", {"p": 1}),)
    if isinstance(raw, str | Mapping) or not isinstance(raw, Sequence):
        raw = [raw]
    out = []
    for i, entry in enumerate(raw):
        where = f"{path}[{i}]"
        if isinstance(entry, str):
            name, kwargs = entry, {}
        elif isinstance(entry, Mapping) and "type" in entry:
            name = str(entry["type"])
            kwargs = {str(k): v for k, v in entry.items() if k != "type"}
        else:
            issues.add(where, "must be a benchmark name or a mapping with a 'type' key")
            continue
        if name not in _BENCHMARK_TYPES:
            issues.add(where, f"unknown benchmark {name!r}{_suggest(name, _BENCHMARK_TYPES)}")
            continue
        out.append((name, kwargs))
    return tuple(out)


# ---------------------------------------------------------------------------- selection
def _parse_selection(raw: Any, issues: _Issues, base_dir: Path | None) -> SelectionSpec:
    """Parse and validate the ``selection`` section of a spec.

    Parameters
    ----------
    raw : mapping or None
        The ``selection`` section.
    issues : _Issues
        Problem collector of the spec parser.
    base_dir : pathlib.Path, optional
        Directory for a relative ``checkpoint``.

    Returns
    -------
    SelectionSpec
        Parsed stages (empty when ``raw`` is ``None`` or invalid).

    Examples
    --------
    >>> issues = _Issues()
    >>> _parse_selection({"preselect": {"top": 5}}, issues, None).preselect.options
    {'top': 5}
    >>> _parse_selection({"preselect": {"tops": 5}}, issues, None).empty, len(issues.items)
    (False, 1)
    """
    if raw is None:
        return SelectionSpec()
    section = _mapping(raw, "selection", issues)
    if section is None:
        return SelectionSpec()
    issues.check_keys(section, ("preselect", "search"), "selection")
    pre = None
    if section.get("preselect") is not None:
        pre = _parse_preselect(section["preselect"], issues)
    search = None
    if section.get("search") is not None:
        search = _parse_search(section["search"], issues, base_dir)
    return SelectionSpec(preselect=pre, search=search)


def _parse_preselect(raw: Any, issues: _Issues) -> PreselectSpec | None:
    from nowcastbox.selection import preselect

    path = "selection.preselect"
    options = {} if raw is True else _mapping(raw, path, issues)
    if options is None:
        return None
    allowed = _known_kwargs(preselect, PRESELECT_EXCLUDED)
    issues.check_keys(options, [*allowed, "apply"], path)
    apply = _boolean(options.get("apply"), f"{path}.apply", issues, default=True)
    return PreselectSpec(
        options={k: v for k, v in options.items() if k in allowed},
        apply=apply,
    )


def _parse_search(raw: Any, issues: _Issues, base_dir: Path | None) -> SearchSpec | None:
    path = "selection.search"
    search = _mapping(raw, path, issues)
    if search is None:
        return None
    issues.check_keys(search, SEARCH_KEYS, path)
    space = _parse_space(search.get("space"), f"{path}.space", issues)
    backtest, benchmarks = _parse_search_backtest(search.get("backtest"), issues)
    checkpoint = _absolute(
        _string(search.get("checkpoint"), f"{path}.checkpoint", issues),
        base_dir if base_dir is not None else Path.cwd(),
    )
    if space is None:
        return None
    return SearchSpec(
        space=space,
        n_draws=_parse_draws(search, issues),
        ranking=search.get("ranking"),
        backtest=backtest,
        benchmarks=benchmarks,
        options={k: search[k] for k in SEARCH_OPTIONS if search.get(k) is not None},
        checkpoint=checkpoint,
        robustness=_parse_robustness(search.get("covid_robustness"), issues),
        apply=_boolean(search.get("apply"), f"{path}.apply", issues, default=False),
    )


def _parse_space(raw: Any, path: str, issues: _Issues) -> dict[str, Any] | None:
    from nowcastbox.selection import ParameterSpace

    if raw is None:
        issues.add(path, "is required (e.g. {n_factors: [1, 2], n_series: [10, 30]})")
        return None
    space = _mapping(raw, path, issues)
    if space is None:
        return None
    try:
        ParameterSpace(space)
    except (TypeError, ValueError) as err:
        issues.add(path, str(err))
        return None
    return dict(space)


def _parse_draws(search: Mapping[str, Any], issues: _Issues) -> int | None:
    if "n_draws" in search and search["n_draws"] is None:
        return None
    return _integer(search.get("n_draws"), "selection.search.n_draws", issues, minimum=1) or 100


def _parse_search_backtest(
    raw: Any, issues: _Issues
) -> tuple[dict[str, Any], tuple[tuple[str, dict[str, Any]], ...]]:
    path = "selection.search.backtest"
    if raw is None:
        return {}, ()
    options = _mapping(raw, path, issues)
    if options is None:
        return {}, ()
    issues.check_keys(options, SEARCH_BACKTEST_KEYS, path)
    benchmarks: tuple[tuple[str, dict[str, Any]], ...] = ()
    if options.get("benchmarks") is not None:
        benchmarks = _parse_benchmarks(options["benchmarks"], f"{path}.benchmarks", issues)
    kept = {
        k: (str(v) if k in ("start", "end") else v)
        for k, v in options.items()
        if k in SEARCH_BACKTEST_KEYS and k != "benchmarks" and v is not None
    }
    return kept, benchmarks


def _parse_robustness(raw: Any, issues: _Issues) -> dict[str, Any] | None:
    path = "selection.search.covid_robustness"
    if raw is None or raw is False:
        return None
    options = {} if raw is True else _mapping(raw, path, issues)
    if options is None:
        return None
    issues.check_keys(options, ROBUSTNESS_KEYS, path)
    out = {k: v for k, v in options.items() if k in ROBUSTNESS_KEYS}
    if out.get("evaluate_from") is not None:
        out["evaluate_from"] = str(out["evaluate_from"])
    if isinstance(out.get("treatments"), list | tuple):
        out["treatments"] = tuple(out["treatments"])
    return out


# ---------------------------------------------------------------------------- spec
@dataclasses.dataclass(frozen=True)
class NowcastSpec:
    """Validated, declarative description of a production nowcast (I10).

    Build it with :meth:`from_yaml`, :meth:`from_dict` or :func:`load_spec`; run it
    with :func:`nowcastbox.pipeline.run_pipeline`.

    Parameters
    ----------
    target : str
        Target series (or a formula such as ``"pib ~ ."``).
    data : DataSpec
        Data source.
    model : ModelSpec
        Estimator.
    name : str, optional
        Identifier of the nowcast (snapshots are grouped by it). Default: the spec
        file stem or the target.
    description : str, optional
        Free text.
    vintage : str, default "today"
        Information set: ``"today"`` or a date (pseudo real-time vintage built from
        the publication delays, :meth:`MixedFrequencyData.as_of`).
    preprocessing : PreprocessingSpec
        Transformations and cleaning.
    outputs : OutputsSpec
        Requested outputs.
    selection : SelectionSpec
        Model-building stages run before the nowcast (``selection.preselect``,
        ``selection.search``; see :mod:`nowcastbox.pipeline.selection`).
    snapshot_dir : pathlib.Path, optional
        Directory of versioned snapshots (none written when omitted).
    random_state : int, optional
        Seed for bootstrap-based outputs.
    base_dir : pathlib.Path, optional
        Directory relative paths were resolved against.
    source : str, optional
        File the spec was read from.

    Examples
    --------
    >>> spec = NowcastSpec.from_dict(
    ...     {
    ...         "target": "gdp",
    ...         "data": {"source": "simulated_dfm"},
    ...         "model": {"type": "TwoStepDFM", "factors": 1},
    ...         "outputs": ["nowcast", "density"],
    ...     }
    ... )
    >>> spec.name, spec.model.method, spec.outputs.names
    ('gdp', 'two_step', ('nowcast', 'density'))
    >>> NowcastSpec.from_dict({"target": "gdp", "data": {"source": "simulated"}})
    Traceback (most recent call last):
    ...
    nowcastbox.pipeline.spec.SpecError: Invalid nowcast spec: 1 problem
      - data.source: unknown data source 'simulated' (did you mean 'simulated_dfm'?) ...
    """

    target: str
    data: DataSpec
    model: ModelSpec = dataclasses.field(default_factory=ModelSpec)
    name: str = ""
    description: str | None = None
    vintage: str = "today"
    preprocessing: PreprocessingSpec = dataclasses.field(default_factory=PreprocessingSpec)
    outputs: OutputsSpec = dataclasses.field(default_factory=OutputsSpec)
    selection: SelectionSpec = dataclasses.field(default_factory=SelectionSpec)
    snapshot_dir: Path | None = None
    random_state: int | None = None
    base_dir: Path | None = None
    source: str | None = None

    # ------------------------------------------------------------------ builders
    @classmethod
    def from_dict(
        cls,
        mapping: Mapping[str, Any],
        *,
        base_dir: str | Path | None = None,
        source: str | None = None,
        name: str | None = None,
    ) -> NowcastSpec:
        """Parse and validate a spec mapping.

        Parameters
        ----------
        mapping : mapping
            Parsed YAML document.
        base_dir : str or pathlib.Path, optional
            Directory for relative paths (default: the working directory).
        source : str, optional
            Origin shown in error messages.
        name : str, optional
            Default name when the document has none.

        Returns
        -------
        NowcastSpec
            Validated spec.

        Raises
        ------
        SpecError
            Listing every problem with its location in the spec.

        Examples
        --------
        >>> spec = NowcastSpec.from_dict(
        ...     {"target": "gdp", "data": {"source": "simulated_dfm"}, "vintage": "2019-06-15"}
        ... )
        >>> spec.vintage, spec.model.type
        ('2019-06-15', 'MixedFreqDFM')
        """
        issues = _Issues()
        if not isinstance(mapping, Mapping):
            raise SpecError([("", "the spec must be a mapping (YAML document with keys)")], source)
        doc = {str(k): v for k, v in mapping.items()}
        issues.check_keys(doc, _TOP_KEYS, "")
        base = Path(base_dir).expanduser() if base_dir is not None else Path.cwd()
        target = _string(doc.get("target"), "target", issues, required=True)
        if "data" in doc:
            data, data_vintage = _parse_data(doc["data"], issues, base)
        else:
            data, data_vintage = None, None
            issues.add("data", "is required (e.g. data: {source: brazil_nowcast})")
        vintage = _parse_vintage(doc.get("vintage"), data_vintage, issues)
        model = _parse_model(doc.get("model"), issues)
        _check_target(target, data, issues)
        snapshot = _string(doc.get("snapshot_dir"), "snapshot_dir", issues)
        spec_name = _string(doc.get("name"), "name", issues) or name or target or "nowcast"
        if not re.fullmatch(r"[\w.\-]+", spec_name):
            issues.add("name", f"use letters, digits, '_', '-' or '.', got {spec_name!r}")
        outputs = _parse_outputs(doc.get("outputs"), issues, base)
        selection = _parse_selection(doc.get("selection"), issues, base)
        preprocessing = _parse_preprocessing(doc.get("preprocessing"), issues)
        random_state = _integer(doc.get("random_state"), "random_state", issues)
        _check_outputs(outputs, model, issues)
        if issues.items or target is None or data is None or model is None:
            raise SpecError(issues.items, source)
        return cls(
            target=target,
            data=data,
            model=model,
            name=spec_name,
            description=_string(doc.get("description"), "description", issues),
            vintage=vintage,
            preprocessing=preprocessing,
            outputs=outputs,
            selection=selection,
            snapshot_dir=_absolute(snapshot, base),
            random_state=random_state,
            base_dir=base,
            source=source,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> NowcastSpec:
        """Read and validate a YAML spec file.

        Relative paths in the spec are resolved against the file's directory (against
        the working directory for the templates bundled with the package, so that
        running a template never writes into the installed package) and the default
        ``name`` is the file stem.

        Parameters
        ----------
        path : str or pathlib.Path
            YAML file.

        Returns
        -------
        NowcastSpec
            Validated spec.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        SpecError
            On YAML syntax errors or invalid content.

        Examples
        --------
        >>> from nowcastbox.pipeline import template_path
        >>> NowcastSpec.from_yaml(template_path("simulated")).data.source
        'simulated_dfm'
        """
        file = Path(path).expanduser()
        try:
            found = file.is_file()
        except OSError:  # e.g. YAML text passed as a path: name too long
            found = False
        if not found:
            hint = " (for YAML text use NowcastSpec.from_string)" if "\n" in str(path) else ""
            raise FileNotFoundError(f"Spec file not found: {str(file)[:200]}{hint}")
        folder = file.resolve().parent
        if _inside_package(folder):
            # Bundled templates: never resolve snapshot/report paths into the installed
            # package; use the working directory as for a spec given as text.
            folder = Path.cwd()
        return cls.from_string(
            file.read_text(encoding="utf-8"),
            base_dir=folder,
            source=str(file),
            name=file.stem,
        )

    @classmethod
    def from_string(
        cls,
        text: str,
        *,
        base_dir: str | Path | None = None,
        source: str | None = None,
        name: str | None = None,
    ) -> NowcastSpec:
        r"""Parse and validate a YAML document.

        Parameters
        ----------
        text : str
            YAML text.
        base_dir : str or pathlib.Path, optional
            Directory for relative paths.
        source : str, optional
            Origin shown in error messages.
        name : str, optional
            Default name.

        Returns
        -------
        NowcastSpec
            Validated spec.

        Raises
        ------
        SpecError
            On YAML syntax errors or invalid content.

        Examples
        --------
        >>> text = "target: gdp\ndata: {source: simulated_dfm}\nmodel: {factors: 1}\n"
        >>> NowcastSpec.from_string(text).model.n_factors
        1
        """
        try:
            doc = yaml.safe_load(text)
        except yaml.YAMLError as err:
            mark = getattr(err, "problem_mark", None)
            where = f"line {mark.line + 1}, column {mark.column + 1}" if mark else ""
            problem = getattr(err, "problem", None) or str(err)
            raise SpecError([(where, f"invalid YAML: {problem}")], source) from err
        if doc is None:
            raise SpecError([("", "the spec is empty")], source)
        return cls.from_dict(doc, base_dir=base_dir, source=source, name=name)

    # ------------------------------------------------------------------ export
    def to_dict(self) -> dict[str, Any]:
        """Canonical YAML-ready mapping (round-trips through :meth:`from_dict`).

        Returns
        -------
        dict
            Normalised spec (absolute paths, expanded defaults).

        Examples
        --------
        >>> spec = NowcastSpec.from_dict({"target": "gdp", "data": {"source": "simulated_dfm"}})
        >>> sorted(spec.to_dict())[:4]
        ['data', 'model', 'name', 'outputs']
        """
        out: dict[str, Any] = {"name": self.name, "target": self.target}
        if self.description:
            out["description"] = self.description
        out["data"] = self.data.to_dict()
        out["vintage"] = self.vintage
        out["preprocessing"] = self.preprocessing.to_dict()
        out["model"] = self.model.to_dict()
        out["outputs"] = self.outputs.to_dict()
        if not self.selection.empty:
            out["selection"] = self.selection.to_dict()
        if self.snapshot_dir is not None:
            out["snapshot_dir"] = str(self.snapshot_dir)
        if self.random_state is not None:
            out["random_state"] = self.random_state
        return out

    def to_yaml(self, path: str | Path | None = None) -> str:
        """Serialise :meth:`to_dict` as YAML.

        Parameters
        ----------
        path : str or pathlib.Path, optional
            File to write.

        Returns
        -------
        str
            YAML text.

        Examples
        --------
        >>> spec = NowcastSpec.from_dict({"target": "gdp", "data": {"source": "simulated_dfm"}})
        >>> NowcastSpec.from_string(spec.to_yaml()).target
        'gdp'
        """
        text = yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    def replace(self, **changes: Any) -> NowcastSpec:
        """Copy with some fields changed (no re-validation).

        Parameters
        ----------
        **changes
            Field values.

        Returns
        -------
        NowcastSpec
            New spec.

        Examples
        --------
        >>> spec = NowcastSpec.from_dict({"target": "gdp", "data": {"source": "simulated_dfm"}})
        >>> spec.replace(vintage="2019-01-01").vintage
        '2019-01-01'
        """
        return dataclasses.replace(self, **changes)

    @property
    def target_name(self) -> str:
        """Target series name (left-hand side of a formula target)."""
        return self.target.split("~", 1)[0].strip()

    def vintage_timestamp(self, today: pd.Timestamp | None = None) -> pd.Timestamp:
        """Information-set date of the spec.

        Parameters
        ----------
        today : pandas.Timestamp, optional
            Date used for ``vintage: today`` (default: the current date).

        Returns
        -------
        pandas.Timestamp
            Normalised date.

        Examples
        --------
        >>> spec = NowcastSpec.from_dict(
        ...     {"target": "gdp", "data": {"source": "simulated_dfm"}, "vintage": "2019-06"}
        ... )
        >>> str(spec.vintage_timestamp().date())
        '2019-06-01'
        """
        if self.vintage == "today":
            return (today if today is not None else pd.Timestamp.today()).normalize()
        return pd.Timestamp(self.vintage).normalize()


def _parse_vintage(top: Any, nested: Any, issues: _Issues) -> str:
    if top is not None and nested is not None:
        issues.add("vintage", "given both at the top level and in 'data'; keep one")
    value = top if top is not None else nested
    path = "vintage" if top is not None else "data.vintage"
    if value is None or (isinstance(value, str) and value.strip().lower() == "today"):
        return "today"
    text = _date(value, path, issues)
    if text is not None and not _DATE_RE.match(text):
        issues.add(path, f"use 'today' or a date YYYY-MM-DD, got {value!r}")
    return text or "today"


def _check_target(target: str | None, data: DataSpec | None, issues: _Issues) -> None:
    if target is None or data is None:
        return
    name = target.split("~", 1)[0].strip()
    if data.kind == "connectors" and data.series and name not in {s.name for s in data.series}:
        issues.add("target", f"{name!r} is not one of the series in data.series")
    if data.kind == "dataset":
        from nowcastbox.datasets import dataset_info, list_datasets

        if data.source not in list_datasets().index:
            return
        names = [s["name"] for s in dataset_info(data.source).get("series") or [] if "name" in s]
        if names and name not in names:
            issues.add(
                "target", f"{name!r} is not a series of {data.source}{_suggest(name, names)}"
            )


def _check_outputs(outputs: OutputsSpec, model: ModelSpec | None, issues: _Issues) -> None:
    if outputs.empirical_bands is not None and outputs.backtest is None:
        issues.add(
            "outputs.empirical_bands", "needs outputs.backtest (the bands use its past errors)"
        )
    if model is not None and model.type == "BridgeCombination":
        _check_bridge_outputs(outputs, issues)
    if model is not None and model.type == "LargeBVAR":
        _check_bvar_outputs(outputs, issues)
    if model is None or model.type != "TwoStepDFM":
        return
    aggregate = model.options.get("aggregate", "factors")
    if outputs.news is not None and aggregate != "factors":
        issues.add(
            "outputs.news",
            "news needs a state-space model: MixedFreqDFM or TwoStepDFM(aggregate='factors')",
        )


def _check_bridge_outputs(outputs: OutputsSpec, issues: _Issues) -> None:
    if outputs.news is not None:
        issues.add(
            "outputs.news",
            "news needs a state-space model (BridgeCombination: use nowcast_change instead)",
        )
    if outputs.density is not None:
        issues.add(
            "outputs.density",
            "BridgeCombination has no model-based density; use outputs.empirical_bands",
        )


def _check_bvar_outputs(outputs: OutputsSpec, issues: _Issues) -> None:
    if outputs.density is not None and outputs.density.n_boot:
        issues.add(
            "outputs.density.n_boot",
            "LargeBVAR densities come from posterior draws: set model.n_draws, not n_boot",
        )


_PACKAGE_DIR = Path(__file__).resolve().parents[1]


def _inside_package(folder: Path) -> bool:
    """Whether ``folder`` lies inside the installed ``nowcastbox`` package."""
    try:
        folder.resolve().relative_to(_PACKAGE_DIR)
    except ValueError:
        return False
    return True


def _absolute(text: str | None, base: Path) -> Path | None:
    if text is None:
        return None
    path = Path(text).expanduser()
    return path if path.is_absolute() else base / path


def load_spec(spec: NowcastSpec | Mapping[str, Any] | str | Path) -> NowcastSpec:
    """Coerce a spec file, YAML text, mapping or spec into a :class:`NowcastSpec`.

    Parameters
    ----------
    spec : NowcastSpec, mapping, str or pathlib.Path
        A spec, a parsed mapping, a path to a YAML file, or YAML text (a string with
        a newline or a ``:``, which is not an existing file).

    Returns
    -------
    NowcastSpec
        Validated spec.

    Raises
    ------
    SpecError
        On invalid content.
    FileNotFoundError
        If a path does not exist.
    TypeError
        On unsupported input types.

    Examples
    --------
    >>> load_spec({"target": "gdp", "data": {"source": "simulated_dfm"}}).target
    'gdp'
    """
    if isinstance(spec, NowcastSpec):
        return spec
    if isinstance(spec, Mapping):
        return NowcastSpec.from_dict(spec)
    if isinstance(spec, Path):
        return NowcastSpec.from_yaml(spec)
    if isinstance(spec, str):
        if "\n" in spec or (":" in spec and not Path(spec).exists()):
            return NowcastSpec.from_string(spec)
        return NowcastSpec.from_yaml(spec)
    raise TypeError(f"Cannot build a NowcastSpec from {type(spec).__name__}.")
