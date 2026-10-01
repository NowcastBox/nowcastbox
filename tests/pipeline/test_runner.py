"""Tests of nowcastbox.pipeline.runner (end-to-end pipeline runs)."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

import nowcastbox as nb
from nowcastbox.pipeline import NowcastSpec, PipelineRun, SpecError, run_pipeline

pytestmark = [
    pytest.mark.filterwarnings("ignore::nowcastbox.core.exceptions.ConvergenceWarning"),
    pytest.mark.filterwarnings("ignore::nowcastbox.core.exceptions.DataQualityWarning"),
]

COLUMNS = ["x01", "x02", "x03", "x04", "x05", "x06"]


def base_spec(snapshot_dir=None, **changes):
    spec = {
        "name": "sim",
        "target": "gdp",
        "data": {"source": "simulated_dfm", "columns": COLUMNS, "start": "2008-01"},
        "vintage": "2019-11-15",
        "preprocessing": False,
        "model": {"type": "MixedFreqDFM", "factors": 1, "max_iter": 30},
        "random_state": 0,
    }
    if snapshot_dir is not None:
        spec["snapshot_dir"] = str(snapshot_dir)
    spec.update(changes)
    return spec


@dataclass
class Runs:
    first: PipelineRun
    second: PipelineRun
    root: object


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("snapshots")
    outputs = {
        "nowcast": True,
        "news": True,
        "density": {"n_boot": 0},
        "diagnostics": {"warn": False},
        "report_html": {"plotlyjs": "cdn", "author": "tests", "notes": "n"},
    }
    first = run_pipeline(base_spec(root, outputs=outputs))
    second = run_pipeline(base_spec(root, outputs=outputs, vintage="2019-12-20"))
    return Runs(first, second, root)


def test_first_run(runs):
    run = runs.first
    assert isinstance(run, PipelineRun)
    assert run.vintage == pd.Timestamp("2019-11-15")
    assert str(run.data.end) == "2019-11"
    assert run.panel.columns == ["gdp", *COLUMNS]
    assert len(run.data_hash) == 64
    assert str(run.headline_period) == "2019Q4"
    assert np.isfinite(run.headline)
    assert run.distribution is not None
    assert {"lower_68", "upper_90", "median"} <= set(run.nowcast.columns)
    assert run.news is None  # first snapshot: nothing to compare with
    assert any("no previous snapshot" in w for w in run.warnings)
    assert run.diagnostics is not None
    assert run.report_html is not None and "<html" in run.report_html.lower()
    assert run.snapshot is not None and run.previous is None
    assert run.report_path == run.snapshot.report_path
    assert set(run.timings) >= {"data", "estimation", "density", "report_html", "snapshot"}
    files = set(run.snapshot.files)
    assert {
        "manifest.json",
        "spec.yaml",
        "nowcast.csv",
        "panel.csv",
        "params.json",
        "density.csv",
        "diagnostics.csv",
        "diagnostics.txt",
        "loadings.csv",
        "factors.csv",
        "report.html",
    } <= files
    assert run.snapshot.headline == pytest.approx(run.headline)
    assert run.snapshot.data_hash == run.data_hash
    params = run.snapshot.params()
    assert params["model_name"] == "MixedFreqDFM" and params["model_params"]["max_iter"] == 30
    assert run.snapshot.spec()["vintage"] == "2019-11-15"


def test_second_run_news_against_previous_snapshot(runs):
    first, second = runs.first, runs.second
    assert second.previous is not None and second.previous.id == first.snapshot.id
    news = second.news
    assert news is not None and news.check_identity()
    assert second.news_reference == first.snapshot.id
    assert news.n_releases > 0
    assert news.new_nowcast == pytest.approx(second.headline)
    snap = second.snapshot
    assert snap is not None and snap.previous == first.snapshot.id
    assert snap.news() is not None
    assert snap.manifest["news"]["against"] == first.snapshot.id
    text = second.summary()
    assert "news        :" in text and f"against {first.snapshot.id}" in text
    assert "diagnostics :" in text and "snapshot    :" in text and "report      :" in text
    assert "[68%:" in text
    info = second.to_dict()
    json.dumps(info)
    assert info["news"]["against"] == first.snapshot.id
    assert info["snapshot"] == snap.id


def test_snapshot_history_and_diff(runs):
    store = nb.pipeline.SnapshotStore(runs.root)
    hist = store.history("sim")
    assert len(hist) == 2
    assert hist["value"].tolist() == pytest.approx([runs.first.headline, runs.second.headline])
    diff = store.diff("latest~1", "latest")
    assert diff.data_changed
    assert diff.data_revisions["new"] > 0
    assert diff.spec_changes["vintage"] == ("2019-11-15", "2019-12-20")
    assert diff.headline_change == pytest.approx(runs.second.headline - runs.first.headline)


def test_no_snapshot_and_minimal_outputs():
    run = run_pipeline(base_spec(), today="2019-10-10")
    assert run.snapshot is None and run.report_path is None
    assert run.distribution is None and run.news is None and run.diagnostics is None
    assert "Nowcast pipeline 'sim'" in run.summary()
    assert run.to_dict()["snapshot"] is None


def test_snapshot_disabled_and_override(tmp_path):
    spec = base_spec(tmp_path / "a")
    run = run_pipeline(spec, snapshot=False)
    assert run.snapshot is None
    run = run_pipeline(base_spec(), snapshot_dir=tmp_path / "b")
    assert run.snapshot is not None and run.snapshot.path.parent == tmp_path / "b"


def test_vintage_override_and_today():
    run = run_pipeline(base_spec(), vintage="2019-06-10")
    assert run.vintage == pd.Timestamp("2019-06-10") and run.spec.vintage == "2019-06-10"
    run = run_pipeline(base_spec(vintage="today"), today="2019-03-05")
    assert run.vintage == pd.Timestamp("2019-03-05")
    assert str(run.data.end) == "2019-03"


def test_news_against_date_and_report_path(tmp_path):
    path = tmp_path / "out" / "report.html"
    spec = base_spec(
        outputs={
            "news": {"against": "2019-10-15", "by": "category"},
            "report_html": {"path": str(path), "plotlyjs": "cdn"},
        },
        snapshot_dir=str(tmp_path / "snaps"),
    )
    run = run_pipeline(spec)
    assert run.news is not None and run.news.check_identity()
    assert run.news_reference == "2019-10-15"
    assert path.is_file() and run.report_path == path
    assert run.snapshot is not None and run.snapshot.report_path is not None
    news = run.snapshot.news()
    assert news is not None


def test_news_against_date_after_vintage_is_a_spec_error():
    spec = base_spec(outputs={"news": {"against": "2020-01-01"}})
    with pytest.raises(SpecError, match="not before the vintage"):
        run_pipeline(spec)


def test_news_failure_is_recorded(tmp_path):
    root = tmp_path / "snaps"
    run_pipeline(base_spec(root, data={"source": "simulated_dfm", "columns": ["x01", "x02"]}))
    run = run_pipeline(base_spec(root, outputs=["news"]))
    assert run.news is None
    assert any(w.startswith("news output skipped: NowcastDataError") for w in run.warnings)


def test_news_skipped_without_stored_panel(tmp_path, monkeypatch):
    root = tmp_path / "snaps"
    run_pipeline(base_spec(root))
    monkeypatch.setattr(nb.pipeline.Snapshot, "panel", lambda self: None)
    run = run_pipeline(base_spec(root, outputs=["news"]))
    assert run.news is None


def test_two_step_auto_factors_with_preprocessing():
    spec = base_spec(
        data={"source": "simulated_dfm", "start": "2005-01"},
        preprocessing={"transform": True, "max_na_prop": 0.9},
        model={"type": "TwoStepDFM", "factors": "auto", "rmax": 3},
        outputs=["density"],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run = run_pipeline(spec)
    assert run.results.model_name == "TwoStepDFM"
    assert "selection" in run.results.info
    assert run.distribution is not None


def brazil_spec(factors, columns, **changes):
    spec = {
        "name": "br",
        "target": "pib",
        "data": {"source": "brazil_nowcast", "columns": columns, "start": "2012-01"},
        "vintage": "2023-12-15",
        "model": {"factors": factors, "max_iter": 15},
        "outputs": [],
    }
    spec.update(changes)
    return spec


BR_COLUMNS = ["ibc_br", "ipca", "selic", "focus_pib"]


def test_blocks_restricted_to_listed_factors():
    full = nb.load_brazil_nowcast()
    blocks = {c: full.data.metadata[c].blocks for c in BR_COLUMNS}
    assert any(len(b) > 1 for b in blocks.values())
    run = run_pipeline(brazil_spec({"global": 1}, BR_COLUMNS))
    assert run.results.model_params["blocks"] == {c: ["global"] for c in ["pib", *BR_COLUMNS]}
    assert np.isfinite(run.headline)


def test_blocks_errors():
    with pytest.raises(SpecError, match="have no series in the data"):
        run_pipeline(brazil_spec({"global": 1, "labor": 1}, BR_COLUMNS))
    with pytest.raises(SpecError, match="load on none of the blocks"):
        run_pipeline(brazil_spec({"real": 1}, BR_COLUMNS))


def test_explicit_blocks_pass_through():
    run = run_pipeline(
        brazil_spec(
            {"global": 1},
            ["ibc_br", "ipca"],
            model={
                "factors": {"global": 1},
                "blocks": {"pib": "global", "ibc_br": "global", "ipca": "global"},
                "max_iter": 10,
            },
        )
    )
    assert run.results.model_params["blocks"]["pib"] == "global"


def test_backtest_output():
    spec = base_spec(
        model={"type": "TwoStepDFM", "factors": 1},
        outputs={
            "backtest": {
                "start": "2019-01-01",
                "end": "2019-03-01",
                "benchmarks": ["RandomWalk", {"type": "AR", "p": 1}],
                "target_offsets": [0],
            },
            "report_html": {"plotlyjs": "cdn"},
        },
    )
    run = run_pipeline(spec)
    assert run.backtest is not None
    table = run.backtest.rmsfe_by_horizon()
    assert {"TwoStepDFM", "RandomWalk", "AR"} <= set(table.columns)
    assert "backtest    : RMSFE by horizon" in run.summary()


def test_backtest_em_with_snapshot(tmp_path):
    spec = base_spec(
        tmp_path,
        outputs={
            "backtest": {
                "start": "2019-02-01",
                "end": "2019-03-01",
                "target_offsets": [0],
                "model": {"max_iter": 5},
            }
        },
    )
    run = run_pipeline(spec)
    assert run.backtest is not None
    assert run.snapshot is not None and run.snapshot.table("backtest_rmsfe") is not None


def test_backtest_without_delays_is_recorded(tmp_path):
    frame = nb.load_simulated_dfm().data.select(["gdp", "x01", "x02"]).to_frame()
    out = frame.copy()
    out.index = out.index.astype(str)
    out.index.name = "date"
    path = tmp_path / "p.csv"
    out.to_csv(path)
    spec = {
        "target": "gdp",
        "data": {"source": "csv", "path": str(path), "frequency": {"gdp": "Q"}},
        "preprocessing": False,
        "model": {"type": "TwoStepDFM", "factors": 1},
        "outputs": {"backtest": {"start": "2019-01-01", "end": "2019-02-01"}},
    }
    with pytest.raises(SpecError, match="publication delays for every series"):
        run_pipeline(spec)


def test_optional_output_failure_is_recorded(monkeypatch):
    def boom(self, **kwargs):
        raise RuntimeError("kaput")

    monkeypatch.setattr(nb.NowcastResults, "diagnostics", boom)
    run = run_pipeline(base_spec(outputs=["diagnostics"]))
    assert run.diagnostics is None
    assert "diagnostics output skipped: RuntimeError: kaput" in run.warnings


def test_summary_without_bands_or_headline(monkeypatch):
    import nowcastbox.pipeline.runner as runner

    run = run_pipeline(base_spec(outputs=["density"]))
    frame = run.results.nowcast.copy()
    frame.loc[run.headline_period, "lower_68"] = np.nan
    no_band = run.results.replace(nowcast=frame)
    text = PipelineRun(**{**run.__dict__, "results": no_band}).summary()
    assert "[68%" not in text
    monkeypatch.setattr(runner, "headline_period", lambda table: None)
    run = run_pipeline(base_spec())
    assert run.headline_period is None and np.isnan(run.headline)
    assert "nowcast     : None = n/a" in run.summary()


def test_run_from_yaml_file(tmp_path):
    path = tmp_path / "spec.yaml"
    NowcastSpec.from_dict(base_spec()).to_yaml(path)
    run = run_pipeline(path)
    assert run.spec.source == str(path)


def test_snapshot_false_still_reads_previous(runs):
    outputs = {"nowcast": True, "news": True}
    run = run_pipeline(base_spec(runs.root, outputs=outputs, vintage="2019-12-20"), snapshot=False)
    assert run.snapshot is None
    assert run.previous is not None
    assert run.news is not None and run.news.check_identity()
