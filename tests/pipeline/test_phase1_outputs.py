"""Pipeline wiring of the ECB-parity outputs: backtest metrics/periods, empirical bands,
indicator heatmap, alternative models and the Excel workbook."""

from __future__ import annotations

import pandas as pd
import pytest

from nowcastbox.pipeline import NowcastSpec, SpecError, run_pipeline
from nowcastbox.pipeline.spec import (
    AlternativesOutput,
    EmpiricalBandsOutput,
    HeatmapOutput,
    OutputsSpec,
)
from tests.pipeline.test_runner import base_spec

pytestmark = [
    pytest.mark.filterwarnings("ignore::nowcastbox.core.exceptions.ConvergenceWarning"),
    pytest.mark.filterwarnings("ignore::nowcastbox.core.exceptions.DataQualityWarning"),
]

CATEGORIES = {
    "x01": "hard",
    "x02": "hard",
    "x03": "soft",
    "x04": "soft",
    "x05": "financial",
    "x06": "financial",
}
BACKTEST = {
    "start": "2015-01-01",
    "end": "2019-10-01",
    "target_offsets": [0],
    "metrics": ["rmsfe", "fda", "n"],
    "periods": {"early": ["2015Q1", "2017Q4"], "late": ["2018Q1", None]},
}


def phase1_spec(snapshot_dir=None, **outputs):
    spec = base_spec(snapshot_dir, model={"type": "TwoStepDFM", "factors": 1})
    spec["data"] = {**spec["data"], "categories": CATEGORIES}
    spec["outputs"] = outputs
    return spec


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    root = tmp_path_factory.mktemp("phase1")
    spec = phase1_spec(
        root,
        backtest=BACKTEST,
        empirical_bands={"min_errors": 4, "levels": [0.575, 0.9], "outliers": "winsorize"},
        heatmap={"by": "category", "last": 12},
        alternatives={"refit": False, "drop": 1},
        report_html={"plotlyjs": "cdn"},
        excel={"path": str(root / "out" / "run.xlsx")},
    )
    return run_pipeline(spec)


# ---------------------------------------------------------------------- spec
def test_spec_round_trip():
    spec = NowcastSpec.from_dict(
        phase1_spec(
            backtest={**BACKTEST, "periods": "covid"},
            bands={"method": "quantile", "window": None},
            zscores={"smooth": False, "window": 24},
            alternative_models=True,
        )
    )
    out = spec.outputs
    assert out.backtest is not None and out.backtest.periods == "covid"
    assert out.backtest.metrics == ("rmsfe", "fda", "n")
    assert out.empirical_bands == EmpiricalBandsOutput(method="quantile", window=None)
    assert out.heatmap == HeatmapOutput(smooth=None, window=24)
    assert out.alternatives == AlternativesOutput()
    again = NowcastSpec.from_dict(spec.to_dict())
    assert again.outputs == out
    assert out.names[-3:] == ("empirical_bands", "heatmap", "alternatives")


def test_spec_defaults():
    out = NowcastSpec.from_dict(
        phase1_spec(backtest={"start": "2019-01", "end": "2019-03"})
    ).outputs
    assert out.backtest is not None
    assert out.backtest.metrics == ("rmsfe", "mae", "bias", "n") and out.backtest.periods is None
    heat = NowcastSpec.from_dict(phase1_spec(heatmap={"by": "series", "smooth": "mm"})).outputs
    assert heat.heatmap == HeatmapOutput(by=None)
    assert OutputsSpec(heatmap=HeatmapOutput()).names == ("nowcast", "heatmap")


@pytest.mark.parametrize(
    ("outputs", "where"),
    [
        ({"empirical_bands": True}, "outputs.empirical_bands"),
        ({"backtest": {**BACKTEST, "metrics": ["rmsfe", "hit"]}}, "metrics"),
        ({"backtest": {**BACKTEST, "periods": "pre-covid"}}, "periods"),
        ({"backtest": {**BACKTEST, "periods": {"a": "2019Q1"}}}, "periods.a"),
        ({"backtest": {**BACKTEST, "periods": {"a": ["2019Qx", None]}}}, "periods.a"),
        ({"backtest": {**BACKTEST, "periods": 3}}, "periods"),
        ({"backtest": BACKTEST, "empirical_bands": {"method": "sd"}}, "method"),
        ({"backtest": BACKTEST, "empirical_bands": {"levels": [1.5]}}, "levels"),
        ({"backtest": BACKTEST, "empirical_bands": {"levels": "x"}}, "levels"),
        ({"backtest": BACKTEST, "empirical_bands": {"outliers": "drop"}}, "outliers"),
        ({"backtest": BACKTEST, "empirical_bands": {"availability": "now"}}, "availability"),
        ({"heatmap": {"by": "sector"}}, "heatmap.by"),
        ({"heatmap": {"smooth": "hp"}}, "heatmap.smooth"),
        ({"heatmap": {"window": 1}}, "heatmap.window"),
        ({"alternatives": {"drop": [0]}}, "alternatives.drop"),
        ({"alternatives": {"drop": []}}, "alternatives.drop"),
        ({"alternatives": {"by": "sector"}}, "alternatives.by"),
        ({"alternatives": {"refit": "yes"}}, "alternatives.refit"),
        ({"alternatives": {"groups": 1}}, "alternatives"),
    ],
)
def test_spec_errors(outputs, where):
    with pytest.raises(SpecError) as info:
        NowcastSpec.from_dict(phase1_spec(**outputs))
    assert any(where in loc for loc, _ in info.value.issues)


def test_scalar_levels_and_drop():
    out = NowcastSpec.from_dict(
        phase1_spec(
            backtest=BACKTEST,
            empirical_bands={"levels": 0.9, "window": "all"},
            alternatives={"drop": 2},
        )
    ).outputs
    assert out.empirical_bands is not None and out.empirical_bands.levels == (0.9,)
    assert out.empirical_bands.window is None
    assert out.alternatives is not None and out.alternatives.drop == (2,)


# ---------------------------------------------------------------------- runner
def test_backtest_metrics_by_sub_period(run):
    metrics = run.backtest_metrics
    assert metrics is not None
    assert list(metrics.index.get_level_values("period").unique()) == ["early", "late"]
    assert "fda" in metrics.columns.get_level_values(0)
    expected = run.backtest.metrics(metrics=("rmsfe", "fda", "n"), periods=BACKTEST["periods"])
    pd.testing.assert_frame_equal(metrics, expected)


def test_empirical_bands_use_the_vintage(run):
    bands = run.empirical_bands
    assert bands is not None and bands.levels == (0.575, 0.9)
    assert str(run.headline_period) in [str(p) for p in bands.index]
    assert "emp. bands  : 90%" in run.summary()


def test_heatmap_and_alternatives(run):
    assert run.heatmap is not None and "gdp" not in run.heatmap.series
    assert run.heatmap.groups is not None
    assert run.alternatives is not None and run.alternatives.n_models == 3
    assert "alternatives: 3 models" in run.summary()
    assert "heatmap     : z-scores of 6 indicators" in run.summary()


def test_report_sections(run):
    html = run.report_html
    assert html is not None
    for text in (
        "57.5% empirical band",
        "Nowcasts of alternative models",
        "data already released",
        "Backtest accuracy",
        "Indicator z-scores",
    ):
        assert text in html


def test_snapshot_and_excel(run):
    snap = run.snapshot
    assert snap is not None
    for name in ("backtest_metrics", "empirical_bands", "heatmap", "heatmap_groups"):
        assert snap.table(name) is not None, name
    assert snap.table("alternatives_range") is not None
    assert len(run.excel_paths) == 2 and all(p.exists() for p in run.excel_paths)
    sheets = pd.ExcelFile(run.excel_paths[0]).sheet_names
    for name in ("backtest_metrics", "empirical_bands", "heatmap", "alternatives"):
        assert name in sheets
    assert "excel       :" in run.summary()
    assert run.to_dict()["excel_paths"] == [str(p) for p in run.excel_paths]


def test_alternatives_refit_and_em_horizon(tmp_path):
    spec = phase1_spec(alternatives={"drop": [1]})
    spec["model"] = {"type": "MixedFreqDFM", "factors": 1, "max_iter": 5}
    out = run_pipeline(spec)
    assert out.alternatives is not None and out.alternatives.refit
    assert out.alternatives.n_models == 3


def test_excel_without_target_is_a_warning():
    out = run_pipeline(phase1_spec(excel=True))
    assert out.excel_paths == ()
    assert any("excel output skipped" in w for w in out.warnings)


def test_excel_without_openpyxl_is_a_warning(tmp_path, monkeypatch):
    import nowcastbox.pipeline.data as data_module

    def missing(*args, **kwargs):
        raise ImportError("openpyxl is required")

    monkeypatch.setattr(data_module, "write_run_excel", missing)
    out = run_pipeline(phase1_spec(excel={"path": str(tmp_path / "x.xlsx")}))
    assert out.excel_paths == ()
    assert any("openpyxl is required" in w for w in out.warnings)


def test_bands_skipped_when_backtest_failed(tmp_path):
    spec = phase1_spec(
        backtest={**BACKTEST, "start": "2030-01-01", "end": "2030-02-01"},
        empirical_bands=True,
    )
    out = run_pipeline(spec)
    assert out.empirical_bands is None
    assert any("empirical_bands output skipped" in w for w in out.warnings)


def test_summary_skips_periods_without_bands_or_alternatives(run):
    import dataclasses

    other = dataclasses.replace(run, headline_period=pd.Period("2030Q1", "Q"))
    assert other._empirical_band_line() == [] and other._alternatives_line() == []
    none = dataclasses.replace(run, headline_period=None)
    assert none._empirical_band_line() == [] and none._alternatives_line() == []


def test_bands_stage_without_backtest():
    from types import SimpleNamespace

    from nowcastbox.pipeline.runner import _empirical_bands

    state = SimpleNamespace(backtest=None, notes=[], bands=None)
    _empirical_bands(state, EmpiricalBandsOutput())  # type: ignore[arg-type]
    assert state.bands is None and "the backtest output failed" in state.notes[0]


def test_heatmap_by_series_in_snapshot(tmp_path):
    out = run_pipeline(phase1_spec(tmp_path, heatmap={"smooth": "none"}))
    assert out.heatmap is not None and out.heatmap.groups is None
    assert out.snapshot is not None and out.snapshot.table("heatmap") is not None
    assert out.snapshot.table("heatmap_groups") is None
