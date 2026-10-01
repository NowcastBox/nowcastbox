"""Declarative production pipeline (innovation I10, plan §6.2).

- :class:`NowcastSpec` — YAML/mapping specification, validated with located error
  messages (:class:`SpecError`);
- :func:`run_pipeline` — data -> vintage -> preprocessing -> model -> outputs (news,
  density, diagnostics, backtest, HTML report) -> snapshot, returning a
  :class:`PipelineRun`;
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
    apply_vintage,
    data_hash,
    frame_hash,
    load_data,
    preprocess,
    read_panel_file,
)
from nowcastbox.pipeline.examples import (
    DEFAULT_TEMPLATE,
    list_templates,
    template_path,
    template_text,
)
from nowcastbox.pipeline.runner import PipelineRun, run_pipeline
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
    FILE_SOURCES,
    MODEL_TYPES,
    OUTPUT_NAMES,
    BacktestOutput,
    ConnectorSeries,
    DataSpec,
    DensityOutput,
    DiagnosticsOutput,
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
    "FILE_SOURCES",
    "MANIFEST",
    "MODEL_TYPES",
    "OUTPUT_NAMES",
    "BacktestOutput",
    "ConnectorSeries",
    "DataSpec",
    "DensityOutput",
    "DiagnosticsOutput",
    "ModelSpec",
    "NewsOutput",
    "NowcastSpec",
    "OutputsSpec",
    "PipelineRun",
    "PreprocessingSpec",
    "ReportOutput",
    "Snapshot",
    "SnapshotDiff",
    "SnapshotStore",
    "SpecError",
    "apply_vintage",
    "data_hash",
    "frame_hash",
    "headline_period",
    "jsonable",
    "list_templates",
    "load_data",
    "load_spec",
    "preprocess",
    "read_panel_file",
    "run_pipeline",
    "template_path",
    "template_text",
]
