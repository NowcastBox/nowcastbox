"""Pipeline wiring of the ECB-parity model building (0.2.0): ``selection.preselect``,
``selection.search`` (with the Covid robustness step) and ``model: bridge_combination``."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from nowcastbox.datasets import load_simulated_dfm
from nowcastbox.models import BridgeCombination, TwoStepDFM
from nowcastbox.pipeline import (
    NowcastSpec,
    PreselectSpec,
    SelectionSpec,
    SpecError,
    run_pipeline,
)
from nowcastbox.pipeline.runner import selection_tables
from nowcastbox.pipeline.selection import (
    best_model,
    funnel_model,
    run_preselection,
    run_search,
    search_ranking,
)
from nowcastbox.pipeline.spec import ModelSpec
from nowcastbox.reports import NowcastReport
from nowcastbox.selection import SpecifiedModel
from tests.pipeline.test_runner import base_spec

pytestmark = [
    pytest.mark.filterwarnings("ignore::nowcastbox.core.exceptions.ConvergenceWarning"),
    pytest.mark.filterwarnings("ignore::nowcastbox.core.exceptions.DataQualityWarning"),
]

TWO_STEP = {"type": "TwoStepDFM", "factors": 1}
SEARCH_BACKTEST = {"start": "2018-06-15", "end": "2019-06-15"}
OUTPUT_BACKTEST = {"start": "2019-01-15", "end": "2019-06-15", "target_offsets": [0]}


def spec_with(selection=None, model=None, **changes):
    spec = base_spec(model=model or TWO_STEP, **changes)
    spec["data"] = {"source": "simulated_dfm", "start": "2008-01"}
    if selection is not None:
        spec["selection"] = selection
    return spec


def parse(selection=None, model=None, **changes):
    return NowcastSpec.from_dict(spec_with(selection, model, **changes))


def problems(selection=None, model=None, **changes):
    with pytest.raises(SpecError) as err:
        parse(selection, model, **changes)
    return [path for path, _ in err.value.issues]


# ---------------------------------------------------------------------------- spec
class TestBridgeModelSpec:
    @pytest.mark.parametrize(
        "name", ["bridge_combination", "BridgeCombination", "bridge-combination"]
    )
    def test_aliases(self, name):
        spec = parse(model={"type": name, "max_monthly": 1, "combine": "median"})
        assert spec.model.type == "BridgeCombination"
        assert spec.model.method == "bridge_combination"
        assert spec.model.estimator_class is BridgeCombination
        assert spec.model.options == {"max_monthly": 1, "combine": "median"}

    def test_round_trip(self):
        spec = parse(model={"type": "bridge_combination", "horizon": 0, "trim": 0.1})
        assert spec.model.to_dict() == {"type": "BridgeCombination", "horizon": 0, "trim": 0.1}
        again = NowcastSpec.from_dict(spec.to_dict())
        assert again.model == spec.model

    def test_factor_settings_rejected(self):
        paths = problems(
            model={"type": "bridge_combination", "factors": 2, "criterion": "IC1", "max_iter": 3}
        )
        assert {"model.factors", "model.criterion", "model.max_iter"} <= set(paths)

    def test_negative_horizon(self):
        assert "model.horizon" in problems(model={"type": "bridge_combination", "horizon": -1})

    def test_state_space_outputs_rejected(self):
        paths = problems(model={"type": "bridge_combination"}, outputs=["news", "density"])
        assert {"outputs.news", "outputs.density"} <= set(paths)

    def test_dfm_estimator_class_unchanged(self):
        assert ModelSpec(type="TwoStepDFM").estimator_class is TwoStepDFM


class TestSelectionSpec:
    def test_absent_is_empty(self):
        spec = parse()
        assert spec.selection.empty
        assert "selection" not in spec.to_dict()

    def test_preselect_true(self):
        spec = parse({"preselect": True})
        assert spec.selection.preselect == PreselectSpec()

    def test_preselect_options(self):
        spec = parse(
            {"preselect": {"methods": ["sis"], "top": 4, "x_lags": [0, 1], "apply": False}}
        )
        pre = spec.selection.preselect
        assert pre.options == {"methods": ["sis"], "top": 4, "x_lags": [0, 1]}
        assert not pre.apply
        assert pre.ranking_options == {"methods": ["sis"], "x_lags": [0, 1]}

    @pytest.mark.parametrize(
        ("selection", "path"),
        [
            ({"preselect": {"as_of": "2019-01-01"}}, "selection.preselect"),
            ({"preselect": {"apply": "yes"}}, "selection.preselect.apply"),
            ({"preselect": 3}, "selection.preselect"),
            ({"presel": {}}, "selection"),
            ("preselect", "selection"),
            ({"search": {}}, "selection.search.space"),
            ({"search": {"space": {"n_factors": []}}}, "selection.search.space"),
            ({"search": {"space": [1, 2]}}, "selection.search.space"),
            ({"search": 3}, "selection.search"),
            ({"search": {"space": {"n_factors": [1]}, "n_draws": 0}}, "selection.search.n_draws"),
            ({"search": {"space": {"n_factors": [1]}, "draws": 2}}, "selection.search"),
            (
                {"search": {"space": {"n_factors": [1]}, "backtest": {"stat": "2019"}}},
                "selection.search.backtest",
            ),
            (
                {"search": {"space": {"n_factors": [1]}, "backtest": 2}},
                "selection.search.backtest",
            ),
            (
                {"search": {"space": {"n_factors": [1]}, "backtest": {"benchmarks": ["ARX"]}}},
                "selection.search.backtest.benchmarks[0]",
            ),
            (
                {"search": {"space": {"n_factors": [1]}, "covid_robustness": {"best": 2}}},
                "selection.search.covid_robustness",
            ),
            (
                {"search": {"space": {"n_factors": [1]}, "covid_robustness": 2}},
                "selection.search.covid_robustness",
            ),
            ({"search": {"space": {"n_factors": [1]}, "apply": 1}}, "selection.search.apply"),
        ],
    )
    def test_invalid(self, selection, path):
        assert any(p == path or p.startswith(f"{path}.") for p in problems(selection))

    def test_search_options(self, tmp_path):
        spec = NowcastSpec.from_dict(
            spec_with(
                {
                    "search": {
                        "space": {"n_factors": [1, 2], "n_series": [3, 6]},
                        "n_draws": None,
                        "backtest": {
                            "start": pd.Timestamp("2018-06-15").date(),
                            "end": "2019-06-15",
                            "refit_every": 2,
                            "benchmarks": ["AR"],
                        },
                        "score": {"rmsfe": 0.7, "fda": 0.3},
                        "periods": "ex-covid",
                        "checkpoint": "search.parquet",
                        "covid_robustness": {
                            "top": 2,
                            "treatments": ["none", "outliers"],
                            "evaluate_from": pd.Period("2019Q1"),
                        },
                        "apply": True,
                    }
                }
            ),
            base_dir=tmp_path,
        )
        search = spec.selection.search
        assert search.n_draws is None
        assert search.backtest == {"start": "2018-06-15", "end": "2019-06-15", "refit_every": 2}
        assert search.benchmarks == (("AR", {}),)
        assert search.options == {"score": {"rmsfe": 0.7, "fda": 0.3}, "periods": "ex-covid"}
        assert search.checkpoint == tmp_path / "search.parquet"
        assert search.robustness == {
            "top": 2,
            "treatments": ("none", "outliers"),
            "evaluate_from": "2019Q1",
        }
        assert search.apply
        doc = spec.to_dict()["selection"]["search"]
        assert doc["backtest"]["benchmarks"] == [{"type": "AR"}]
        assert doc["checkpoint"] == str(tmp_path / "search.parquet")
        again = NowcastSpec.from_dict(spec.to_dict())
        assert again.selection.search.robustness == search.robustness
        assert again.selection.search.space == search.space

    def test_search_defaults(self):
        search = parse({"search": {"space": {"n_factors": [1, 2]}, "covid_robustness": True}})
        stage = search.selection.search
        assert stage.n_draws == 100
        assert stage.robustness == {}
        assert not stage.apply
        assert stage.to_dict() == {
            "space": {"n_factors": [1, 2]},
            "n_draws": 100,
            "backtest": {},
            "covid_robustness": {},
            "apply": False,
        }
        off = parse({"search": {"space": {"n_factors": [1]}, "covid_robustness": False}})
        assert off.selection.search.robustness is None

    def test_ranking_kept(self):
        spec = parse({"search": {"space": {"n_series": [2, 4]}, "ranking": ["x01", "x02"]}})
        assert spec.selection.search.ranking == ["x01", "x02"]
        assert spec.to_dict()["selection"]["search"]["ranking"] == ["x01", "x02"]

    def test_null_sections(self):
        spec = parse({"preselect": None, "search": None})
        assert spec.selection == SelectionSpec()


class TestSearchRanking:
    def test_none_without_stage_or_funnel(self):
        assert search_ranking(parse()) is None
        assert search_ranking(parse({"search": {"space": {"n_factors": [1]}}})) is None

    def test_explicit(self):
        spec = parse({"search": {"space": {"n_series": [2]}, "ranking": "preselect"}})
        assert search_ranking(spec) == "preselect"

    def test_from_preselect_options(self):
        spec = parse(
            {
                "preselect": {"methods": ["sis"], "top": 5},
                "search": {"space": {"n_series": [2, 4]}},
            }
        )
        assert search_ranking(spec) == {"methods": ["sis"]}
        bare = parse({"preselect": {"top": 5}, "search": {"space": {"n_series": [2, 4]}}})
        assert search_ranking(bare) == "preselect"


# ---------------------------------------------------------------------------- helpers
class TestHelpers:
    def test_run_preselection_needs_stage(self):
        with pytest.raises(ValueError, match=r"no selection\.preselect"):
            run_preselection(parse(), load_simulated_dfm().data)

    def test_run_search_needs_stage(self):
        with pytest.raises(ValueError, match=r"no selection\.search"):
            run_search(parse(), load_simulated_dfm().data, TwoStepDFM(), None)

    def test_run_search_needs_delays(self):
        spec = parse({"search": {"space": {"n_factors": [1]}}})
        data = load_simulated_dfm().data
        data = data.with_metadata("x01", release_delay=None)
        with pytest.raises(ValueError, match="publication delays"):
            run_search(spec, data, TwoStepDFM(), None)

    def test_run_search_passes_fit_kwargs(self):
        from nowcastbox.models import MixedFreqDFM

        spec = parse({"search": {"space": {"n_factors": [1]}, "backtest": SEARCH_BACKTEST}})
        data = load_simulated_dfm().data.select(["gdp", "x01", "x02", "x03"])
        out, rob = run_search(spec, data, MixedFreqDFM(max_iter=3), None, {"horizon": 1})
        assert rob is None
        assert out.table()["status"].iloc[0] == "ok"

    def test_funnel_model(self):
        spec = parse({"preselect": {"methods": ["sis"], "top": 3}})
        data = load_simulated_dfm().data
        pre = run_preselection(spec, data)
        model = funnel_model(spec, TwoStepDFM(n_factors=2), pre)
        assert isinstance(model, SpecifiedModel)
        assert model.model is TwoStepDFM
        assert model.params["n_factors"] == 2
        assert model.ranking == {"methods": ["sis"]}
        res = model.fit(data, "gdp")
        assert model.columns_ == list(pre.selected)
        assert res.nowcast["out_of_sample"].notna().any()

    def test_funnel_model_without_stage(self):
        spec = parse({"preselect": {"methods": ["sis"], "top": 3}})
        pre = run_preselection(spec, load_simulated_dfm().data)
        model = funnel_model(parse(), TwoStepDFM(), pre)
        assert model.ranking == {}

    def test_best_model_prefers_robustness(self):
        search = SimpleNamespace(best_model=lambda: "search")
        robustness = SimpleNamespace(best_model=lambda: "robust")
        assert best_model(search, None) == "search"  # type: ignore[arg-type]
        assert best_model(search, robustness) == "robust"  # type: ignore[arg-type]

    def test_selection_tables_empty(self):
        empty = SimpleNamespace(preselection=None, search=None, robustness=None)
        assert selection_tables(empty) == {}

    def test_report_rejects_other_objects(self, small_results):
        with pytest.raises(TypeError, match="selection items"):
            NowcastReport(small_results, selection=[object()], plotlyjs="cdn").to_html()


@pytest.fixture(scope="module")
def small_results():
    return TwoStepDFM(n_factors=1).fit(load_simulated_dfm().data.select(["gdp", "x01"]), "gdp")


# ---------------------------------------------------------------------------- runs
@pytest.fixture(scope="module")
def preselect_run():
    spec = spec_with(
        {"preselect": {"methods": ["tstat", "sis"], "top": 5}},
        outputs={"nowcast": None, "backtest": OUTPUT_BACKTEST, "report_html": {"plotlyjs": "cdn"}},
    )
    return run_pipeline(spec)


class TestPreselectRun:
    def test_model_uses_selected_indicators(self, preselect_run):
        run = preselect_run
        assert run.preselection.n_selected == 5
        assert sorted(run.results.data.columns) == sorted([*run.preselection.selected, "gdp"])
        assert run.panel.n_series == 21
        assert "preselect" in run.timings

    def test_backtest_repeats_preselection(self, preselect_run):
        assert "SpecifiedModel" in preselect_run.backtest.models

    def test_summary_report_and_tables(self, preselect_run):
        run = preselect_run
        assert "preselect   : 5 indicators" in run.summary()
        assert "Pre-selected indicators (5)" in run.report_html
        assert list(selection_tables(run)) == ["preselection"]

    def test_not_applied(self):
        run = run_pipeline(spec_with({"preselect": {"methods": ["sis"], "top": 3, "apply": False}}))
        assert run.preselection.n_selected == 3
        assert run.results.data.n_series == 21

    def test_failure_keeps_all_indicators(self):
        run = run_pipeline(spec_with({"preselect": {"methods": ["sis"], "min_obs": 10_000}}))
        assert run.preselection is None
        assert run.results.data.n_series == 21
        assert any("preselect skipped" in w for w in run.warnings)


@pytest.fixture(scope="module")
def search_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("phase2")
    spec = spec_with(
        {
            "preselect": {"methods": ["tstat", "sis"], "top": 6},
            "search": {
                "space": {"n_factors": [1, 2], "n_series": [3, 6]},
                "n_draws": None,
                "backtest": {**SEARCH_BACKTEST, "benchmarks": ["AR"]},
                "covid_robustness": {"top": 2, "treatments": ["none", "outliers"]},
                "apply": True,
            },
        },
        outputs={
            "nowcast": None,
            "backtest": OUTPUT_BACKTEST,
            "report_html": {"plotlyjs": "cdn"},
            "excel": {"path": str(root / "run.xlsx")},
        },
        snapshot_dir=str(root / "snapshots"),
    )
    return run_pipeline(spec)


class TestSearchRun:
    def test_search_and_robustness(self, search_run):
        run = search_run
        assert len(run.search.evaluations) == 4
        assert set(run.robustness.table()["treatment"]) <= {"none", "outliers"}
        assert "search" in run.timings

    def test_best_model_applied(self, search_run):
        run = search_run
        best = run.robustness.best()
        n_series = run.search.specs[best["draw"]]["n_series"]
        assert run.results.data.n_series == n_series + 1
        assert run.backtest.models[0] == "SpecifiedModel"

    def test_outputs(self, search_run):
        run = search_run
        text = run.summary()
        assert "search      : 4 specifications" in text
        assert "robustness  : best treatment" in text
        assert "Specification search: best specifications" in run.report_html
        assert "Covid robustness" in run.report_html
        files = {p.name for p in run.snapshot.path.iterdir()}
        assert {"preselection.csv", "search.csv", "robustness.csv"} <= files
        sheets = pd.ExcelFile(run.excel_paths[0]).sheet_names
        assert {"preselection", "search", "robustness"} <= set(sheets)

    def test_not_applied_keeps_spec_model(self):
        run = run_pipeline(
            spec_with(
                {
                    "search": {
                        "space": {"n_factors": [1, 2]},
                        "n_draws": 1,
                        "backtest": SEARCH_BACKTEST,
                    }
                }
            )
        )
        assert len(run.search.evaluations) == 1
        assert run.robustness is None
        assert run.results.data.n_series == 21
        assert "robustness" not in run.summary()

    def test_failure_is_a_warning(self):
        run = run_pipeline(
            spec_with({"search": {"space": {"n_fators": [1]}, "backtest": SEARCH_BACKTEST}})
        )
        assert run.search is None
        assert any("search skipped" in w for w in run.warnings)


class TestBridgeCombinationRun:
    def test_fit_and_backtest(self, tmp_path):
        run = run_pipeline(
            spec_with(
                model={"type": "bridge_combination", "max_monthly": 1, "combine": "median"},
                outputs={"nowcast": None, "backtest": OUTPUT_BACKTEST},
                snapshot_dir=str(tmp_path),
            )
        )
        assert run.snapshot is not None
        assert "loadings" not in run.snapshot.manifest.get("tables", [])
        assert run.results.model_name == "BridgeCombination"
        assert run.results.model_params["combine"] == "median"
        assert "equations=" in run.summary()
        assert "BridgeCombination" in run.backtest.models


@pytest.mark.slow
def test_model_building_template_runs(tmp_path, monkeypatch):
    from nowcastbox.pipeline import template_path

    monkeypatch.chdir(tmp_path)
    run = run_pipeline(template_path("model_building"))
    assert run.preselection is not None and run.preselection.n_selected == 10
    assert run.search is not None and run.robustness is not None
    assert run.backtest is not None
    assert not [w for w in run.warnings if "skipped" in w]
    assert run.snapshot is not None and (tmp_path / "snapshots").is_dir()
