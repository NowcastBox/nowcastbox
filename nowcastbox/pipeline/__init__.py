"""Declarative production pipeline (innovation I10, plan §6.2).

- :class:`NowcastSpec` — YAML/mapping specification, validated with located error
  messages (:class:`SpecError`);
- :class:`SelectionSpec` — optional model building before the nowcast (``selection``:
  pre-selection of the indicators and specification search with a Covid robustness
  step, as in the ECB toolbox; :mod:`nowcastbox.pipeline.selection`);
- :func:`run_pipeline` — data -> vintage -> preprocessing -> [selection] -> model -> outputs (news,
  density, diagnostics, backtest with sub-period and directional accuracy, empirical
  error bands, indicator heatmap, alternative models, HTML report, Excel workbook) ->
  snapshot, returning a :class:`PipelineRun`;
- Excel data templates (:func:`read_excel_panel`, :func:`write_excel_panel`,
  :func:`write_run_excel`, :func:`example_workbook_path`; optional extra ``[excel]``);
- :class:`SnapshotStore` — versioned snapshots (spec, data hash, nowcast table, model
  parameters, news against the previous snapshot, report), with listing, loading,
  nowcast history and diffs;
- bundled example specs (:func:`list_templates`, :func:`template_path`), used by
  ``nowcastbox init``.

Examples
--------
>>> from nowcastbox.pipeline import NowcastSpec, template_path
>>> spec = NowcastSpec.from_yaml(template_path("brazil_pib"))
>>> spec.model.n_factors
{'global': 1, 'real': 1, 'soft': 1}
"""

from nowcastbox.pipeline.data import (
    CONNECTORS,
    EXCEL_DATA_SHEETS,
    EXCEL_METADATA_COLUMNS,
    apply_vintage,
    data_hash,
    example_workbook_path,
    frame_hash,
    load_data,
    preprocess,
    read_excel_panel,
    read_panel_file,
    write_excel_panel,
    write_run_excel,
)
from nowcastbox.pipeline.examples import (
    DEFAULT_TEMPLATE,
    list_templates,
    template_path,
    template_text,
)
from nowcastbox.pipeline.runner import PipelineRun, run_pipeline
from nowcastbox.pipeline.selection import PreselectSpec, SearchSpec, SelectionSpec
from nowcastbox.pipeline.snapshots import (
    MANIFEST,
    Snapshot,
    SnapshotDiff,
    SnapshotStore,
    headline_period,
    jsonable,
)
from nowcastbox.pipeline.spec import (
    CONNECTOR_SOURCES,
    EXCEL_SOURCE,
    FILE_SOURCES,
    MODEL_TYPES,
    OUTPUT_NAMES,
    AlternativesOutput,
    BacktestOutput,
    ConnectorSeries,
    DataSpec,
    DensityOutput,
    DiagnosticsOutput,
    EmpiricalBandsOutput,
    ExcelOutput,
    HeatmapOutput,
    ModelSpec,
    NewsOutput,
    NowcastSpec,
    OutputsSpec,
    PreprocessingSpec,
    ReportOutput,
    SpecError,
    load_spec,
)

__all__ = [
    "CONNECTORS",
    "CONNECTOR_SOURCES",
    "DEFAULT_TEMPLATE",
    "EXCEL_DATA_SHEETS",
    "EXCEL_METADATA_COLUMNS",
    "EXCEL_SOURCE",
    "FILE_SOURCES",
    "MANIFEST",
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
    "PipelineRun",
    "PreprocessingSpec",
    "PreselectSpec",
    "ReportOutput",
    "SearchSpec",
    "SelectionSpec",
    "Snapshot",
    "SnapshotDiff",
    "SnapshotStore",
    "SpecError",
    "apply_vintage",
    "data_hash",
    "example_workbook_path",
    "frame_hash",
    "headline_period",
    "jsonable",
    "list_templates",
    "load_data",
    "load_spec",
    "preprocess",
    "read_excel_panel",
    "read_panel_file",
    "run_pipeline",
    "template_path",
    "template_text",
    "write_excel_panel",
    "write_run_excel",
]
