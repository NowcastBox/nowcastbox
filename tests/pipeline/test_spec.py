"""Tests of nowcastbox.pipeline.spec (YAML spec parsing and validation)."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest

from nowcastbox.pipeline import (
    BacktestOutput,
    ConnectorSeries,
    DataSpec,
    ModelSpec,
    NowcastSpec,
    OutputsSpec,
    PreprocessingSpec,
    SpecError,
    list_templates,
    load_spec,
    template_path,
)

BASE = {"target": "gdp", "data": {"source": "simulated_dfm"}}


def spec_of(**changes):
    return NowcastSpec.from_dict({**BASE, **changes})


def issues_of(mapping, **kwargs):
    with pytest.raises(SpecError) as info:
        NowcastSpec.from_dict(mapping, **kwargs)
    return dict(info.value.issues), str(info.value)


# ---------------------------------------------------------------- basics
def test_minimal_spec_defaults():
    spec = NowcastSpec.from_dict(BASE)
    assert spec.name == "gdp"
    assert spec.vintage == "today"
    assert spec.model == ModelSpec()
    assert spec.model.method == "em"
    assert spec.preprocessing == PreprocessingSpec()
    assert spec.outputs == OutputsSpec()
    assert spec.outputs.names == ("nowcast",)
    assert spec.snapshot_dir is None
    assert spec.data.kind == "dataset"
    assert spec.target_name == "gdp"
    assert spec.base_dir == Path.cwd()


def test_formula_target_and_name():
    spec = spec_of(target="gdp ~ x01 + x02", name="my-run.v1")
    assert spec.target_name == "gdp"
    assert spec.name == "my-run.v1"


def test_bad_name():
    issues, _ = issues_of({**BASE, "name": "bad name/x"})
    assert "name" in issues


def test_not_a_mapping():
    with pytest.raises(SpecError, match="must be a mapping"):
        NowcastSpec.from_dict(["target"])  # type: ignore[arg-type]


def test_missing_target_and_data():
    issues, text = issues_of({"model": {"type": "MixedFreqDFM"}})
    assert issues["target"] == "is required"
    assert "data" in issues
    assert "2 problems" in text


def test_unknown_top_key_suggestion():
    issues, _ = issues_of({**BASE, "outptus": ["nowcast"]})
    assert "did you mean 'outputs'" in issues["outptus"]


def test_unknown_key_without_close_match_lists_allowed():
    issues, _ = issues_of({**BASE, "zzz": 1})
    assert "allowed:" in issues["zzz"]


def test_error_message_format():
    err = SpecError([("a.b", "bad"), ("", "doc problem")], source="x.yaml")
    text = str(err)
    assert text.splitlines()[0] == "Invalid nowcast spec (x.yaml): 2 problems"
    assert "  - a.b: bad" in text
    assert "  - doc problem" in text
    assert isinstance(err, ValueError)


def test_scalar_type_errors():
    issues, _ = issues_of({**BASE, "target": ["x"], "random_state": "a"})
    assert "must be a string" in issues["target"]
    assert "must be an integer" in issues["random_state"]


def test_empty_target():
    issues, _ = issues_of({**BASE, "target": "  "})
    assert issues["target"] == "must not be empty"


def test_negative_integer():
    issues, _ = issues_of({**BASE, "random_state": -1})
    assert ">= 0" in issues["random_state"]


# ---------------------------------------------------------------- vintage
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("today", "today"),
        ("TODAY", "today"),
        (dt.date(2019, 6, 15), "2019-06-15"),
        (dt.datetime(2019, 6, 15, 10, 0), "2019-06-15"),
        ("2019-06", "2019-06"),
    ],
)
def test_vintage_values(value, expected):
    assert spec_of(vintage=value).vintage == expected


def test_vintage_in_data_section():
    spec = NowcastSpec.from_dict(
        {"target": "gdp", "data": {"source": "simulated_dfm", "vintage": "2019-01-10"}}
    )
    assert spec.vintage == "2019-01-10"


def test_vintage_twice():
    issues, _ = issues_of(
        {
            "target": "gdp",
            "vintage": "2019-01-01",
            "data": {"source": "simulated_dfm", "vintage": "2019-01-01"},
        }
    )
    assert "keep one" in issues["vintage"]


@pytest.mark.parametrize("value", ["yesterday", "2019Q1"])
def test_bad_vintage(value):
    issues, _ = issues_of({**BASE, "vintage": value})
    assert "vintage" in issues


def test_vintage_timestamp():
    assert spec_of(vintage="2019-06-15").vintage_timestamp() == pd.Timestamp("2019-06-15")
    today = pd.Timestamp("2020-01-02 13:45")
    assert spec_of().vintage_timestamp(today) == pd.Timestamp("2020-01-02")
    assert spec_of().vintage_timestamp() == pd.Timestamp.today().normalize()


# ---------------------------------------------------------------- data
def test_dataset_columns_and_bounds():
    spec = NowcastSpec.from_dict(
        {
            "target": "pib",
            "data": {
                "source": "brazil_nowcast",
                "columns": ["ibc_br"],
                "start": dt.date(2010, 1, 1),
                "end": "2019Q4",
                "transform": {"ibc_br": "dlog"},
                "delay": {"ibc_br": 40},
                "blocks": {"ibc_br": ["global"]},
                "categories": {"ibc_br": "hard"},
                "frequency": {"ibc_br": "M"},
            },
        }
    )
    data = spec.data
    assert data.columns == ("ibc_br",)
    assert data.start == "2010-01-01" and data.end == "2019Q4"
    assert data.transform == {"ibc_br": "dlog"}
    assert data.delay == {"ibc_br": 40}
    assert data.blocks == {"ibc_br": ("global",)}
    assert data.categories == {"ibc_br": "hard"}


def test_dataset_unknown_name_suggests():
    issues, _ = issues_of({"target": "gdp", "data": {"source": "simulated"}})
    assert "did you mean 'simulated_dfm'" in issues["data.source"]


@pytest.mark.parametrize("name", ["brazil_calendar", "brazil_vintages"])
def test_dataset_not_a_panel(name):
    issues, _ = issues_of({"target": "pib", "data": {"source": name}})
    assert "not a panel" in issues["data.source"]


def test_dataset_unknown_column_and_target():
    issues, _ = issues_of(
        {"target": "pibb", "data": {"source": "brazil_nowcast", "columns": ["ibc_brr"]}}
    )
    assert "did you mean 'ibc_br'" in issues["data.columns"]
    assert "did you mean 'pib'" in issues["target"]


def test_data_not_mapping_and_missing_source():
    issues, _ = issues_of({"target": "gdp", "data": "simulated_dfm"})
    assert "must be a mapping" in issues["data"]
    issues, _ = issues_of({"target": "gdp", "data": {"vintage": "today"}})
    assert issues["data.source"] == "is required"


def test_data_bad_fields():
    issues, _ = issues_of(
        {
            "target": "gdp",
            "data": {
                "source": "simulated_dfm",
                "start": "notaperiod",
                "end": dt.date(2019, 1, 1),
                "columns": [["x"]],
                "delay": {"x01": "soon"},
                "blocks": {"x01": 3},
                "path": "a.csv",
            },
        }
    )
    assert "is not a period" in issues["data.start"]
    assert "list of names" in issues["data.columns"]
    assert "integer" in issues["data.delay.x01"]
    assert "list of names" in issues["data.blocks.x01"]
    assert "unknown key" in issues["data.path"]


def test_csv_source(tmp_path):
    (tmp_path / "panel.csv").write_text("date,gdp\n2020-01,1\n")
    (tmp_path / "legend.csv").write_text("name,frequency\ngdp,M\n")
    spec = NowcastSpec.from_dict(
        {
            "target": "gdp",
            "data": {
                "source": "csv",
                "path": "panel.csv",
                "index_column": "date",
                "legend": "legend.csv",
            },
        },
        base_dir=tmp_path,
    )
    assert spec.data.kind == "file"
    assert spec.data.path == tmp_path / "panel.csv"
    assert spec.data.legend == tmp_path / "legend.csv"
    assert spec.data.index_column == "date"
    assert spec.data.to_dict()["path"] == str(tmp_path / "panel.csv")


def test_file_source_shortcut(tmp_path):
    file = tmp_path / "p.parquet"
    file.write_bytes(b"")
    spec = NowcastSpec.from_dict({"target": "gdp", "data": {"source": str(file)}})
    assert spec.data.source == "parquet"
    assert spec.data.path == file


def test_csv_missing_file(tmp_path):
    issues, _ = issues_of(
        {"target": "gdp", "data": {"source": "csv", "path": "nope.csv", "legend": "l.csv"}},
        base_dir=tmp_path,
    )
    assert "file not found" in issues["data.path"]
    assert "file not found" in issues["data.legend"]
    issues, _ = issues_of({"target": "gdp", "data": {"source": "csv"}})
    assert issues["data.path"] == "is required"


def test_connectors_source():
    spec = NowcastSpec.from_dict(
        {
            "target": "pib",
            "data": {
                "source": "connectors",
                "series": {
                    "pib": {
                        "source": "IBGE",
                        "table": 1621,
                        "variable": 584,
                        "classifications": {11255: 90707},
                        "frequency": "Q",
                        "transform": "qoq",
                        "delay": 62,
                        "blocks": ["global"],
                        "category": "hard",
                    },
                    "ipca": {"source": "bcb", "code": 433, "options": {"timeout": 5}},
                },
            },
        }
    )
    pib, ipca = spec.data.series
    assert pib == ConnectorSeries(
        name="pib",
        source="ibge",
        table="1621",
        variable="584",
        classifications={"11255": 90707},
        frequency="Q",
        transform="qoq",
        delay=62,
        blocks=("global",),
        category="hard",
    )
    assert ipca.code == "433" and ipca.options == {"timeout": 5}
    assert spec.data.kind == "connectors"
    assert spec.data.to_dict()["series"]["ipca"] == {
        "source": "bcb",
        "code": "433",
        "options": {"timeout": 5},
    }


def test_connectors_errors():
    issues, _ = issues_of(
        {
            "target": "zz",
            "data": {
                "source": "connectors",
                "series": {
                    "a": {"source": "bcbb"},
                    "b": {"source": "ibge"},
                    "c": {"source": "fred"},
                    "d": "bcb",
                    "e": {"code": 1},
                    "f": {"source": "ipea", "code": "X", "colour": 1},
                },
            },
        }
    )
    assert "did you mean 'bcb'" in issues["data.series.a.source"]
    assert "required for source 'ibge'" in issues["data.series.b.table"]
    assert "required for source 'fred'" in issues["data.series.c.code"]
    assert "must be a mapping" in issues["data.series.d"]
    assert issues["data.series.e.source"] == "is required"
    assert "unknown key" in issues["data.series.f.colour"]
    assert "not one of the series" in issues["target"]


def test_connectors_empty_or_bad_series():
    issues, _ = issues_of({"target": "a", "data": {"source": "connectors", "series": {}}})
    assert "at least one series" in issues["data.series"]
    issues, _ = issues_of({"target": "a", "data": {"source": "connectors", "series": [1]}})
    assert "must be a mapping" in issues["data.series"]


# ---------------------------------------------------------------- preprocessing
def test_preprocessing_forms():
    assert spec_of(preprocessing=False).preprocessing.enabled is False
    assert spec_of(preprocessing=True).preprocessing.enabled is True
    prep = spec_of(
        preprocessing={"transform": {"x01": "diff"}, "max_na_prop": 0.5, "replace_outliers": False}
    ).preprocessing
    assert prep.transform == {"x01": "diff"}
    assert prep.options == {"max_na_prop": 0.5, "replace_outliers": False}
    assert prep.to_dict() == {
        "enabled": True,
        "transform": {"x01": "diff"},
        "max_na_prop": 0.5,
        "replace_outliers": False,
    }


def test_preprocessing_errors():
    issues, _ = issues_of(
        {
            **BASE,
            "preprocessing": {"transform": 3, "max_na_prop2": 1, "enabled": "yes", "keep": ["a"]},
        }
    )
    assert "true, false or a mapping" in issues["preprocessing.transform"]
    assert "did you mean 'max_na_prop'" in issues["preprocessing.max_na_prop2"]
    assert "true or false" in issues["preprocessing.enabled"]
    assert "unknown key" in issues["preprocessing.keep"]
    issues, _ = issues_of({**BASE, "preprocessing": "x"})
    assert "must be a mapping" in issues["preprocessing"]


# ---------------------------------------------------------------- model
def test_model_mixed_freq_options():
    spec = spec_of(
        model={
            "type": "em",
            "factors": {"global": 2, "real": 1},
            "factor_lags": 2,
            "horizon": 0,
            "idiosyncratic": "student_t",
            "long_run_mean": "time_varying",
            "kwargs": {"max_iter": 10},
        }
    )
    model = spec.model
    assert model.type == "MixedFreqDFM"
    assert model.n_factors == {"global": 2, "real": 1}
    assert model.blocks == "data"
    assert model.factor_lags == 2 and model.horizon == 0
    assert model.options == {
        "idiosyncratic": "student_t",
        "long_run_mean": "time_varying",
        "max_iter": 10,
    }
    assert model.estimator_class.__name__ == "MixedFreqDFM"
    out = model.to_dict()
    assert out["factors"] == {"global": 2, "real": 1} and out["blocks"] == "data"


def test_model_robust_shorthand():
    model = spec_of(model={"robust": True}).model
    assert model.options == {"idiosyncratic": "student_t", "outliers": "auto"}
    model = spec_of(model={"robust": True, "idiosyncratic": "iid"}).model
    assert model.options["idiosyncratic"] == "iid"
    assert spec_of(model={"robust": False}).model.options == {}


def test_model_two_step():
    model = spec_of(
        model={"type": "TwoStepDFM", "factors": 2, "n_shocks": "auto", "aggregate": "factors"}
    ).model
    assert model.method == "two_step"
    assert model.n_shocks == "auto"
    assert model.estimator_class.__name__ == "TwoStepDFM"
    assert model.to_dict()["n_shocks"] == "auto"
    assert spec_of(model={"type": "two_step", "n_shocks": 1}).model.n_shocks == 1


def test_model_auto_factors():
    model = spec_of(model={"n_factors": "auto", "rmax": 4, "criterion": "IC1"}).model
    assert model.n_factors == "auto"
    assert model.to_dict()["rmax"] == 4 and model.to_dict()["criterion"] == "IC1"


def test_model_errors():
    issues, _ = issues_of({**BASE, "model": {"type": "DFM"}})
    assert "unknown model 'DFM'" in issues["model.type"]
    issues, _ = issues_of(
        {
            **BASE,
            "model": {
                "factors": {"global": 0},
                "n_factors": 1,
                "factor_lags": 0,
                "max_iterr": 3,
                "n_shocks": 2,
                "robust": "yes",
            },
        }
    )
    assert ">= 1" in issues["model.factors.global"]
    assert "not both" in issues["model.n_factors"]
    assert ">= 1" in issues["model.factor_lags"]
    assert "did you mean 'max_iter'" in issues["model.max_iterr"]
    assert "only used by TwoStepDFM" in issues["model.n_shocks"]
    assert "true or false" in issues["model.robust"]


def test_model_more_errors():
    issues, _ = issues_of({**BASE, "model": {"factors": {}}})
    assert "empty" in issues["model.factors"]
    issues, _ = issues_of({**BASE, "model": {"factors": "many"}})
    assert "integer" in issues["model.factors"]
    issues, _ = issues_of({**BASE, "model": {"type": None}})
    assert issues["model.type"] == "is required"
    issues, _ = issues_of({**BASE, "model": 3})
    assert "must be a mapping" in issues["model"]


def test_two_step_rejects_em_options():
    issues, _ = issues_of(
        {
            **BASE,
            "model": {
                "type": "TwoStepDFM",
                "factors": {"global": 1},
                "blocks": "data",
                "idiosyncratic": "student_t",
                "robust": True,
                "n_shocks": 0,
            },
        }
    )
    assert "integer number of factors" in issues["model.factors"]
    assert "only supported by MixedFreqDFM" in issues["model.blocks"]
    assert "only available for MixedFreqDFM" in issues["model.robust"]
    assert ">= 1" in issues["model.n_shocks"]


def test_two_step_variables_has_no_news():
    issues, _ = issues_of(
        {**BASE, "model": {"type": "TwoStepDFM", "aggregate": "variables"}, "outputs": ["news"]}
    )
    assert "state-space model" in issues["outputs.news"]


# ---------------------------------------------------------------- outputs
def test_outputs_list_form():
    outputs = spec_of(
        outputs=["nowcast", "news", "density", "diagnostics", "report", {"density": {"n_boot": 5}}]
    ).outputs
    assert outputs.names == ("nowcast", "news", "density", "diagnostics", "report_html")
    assert outputs.density is not None and outputs.density.n_boot == 5
    assert outputs.news is not None and outputs.news.against == "previous_snapshot"


def test_outputs_mapping_form(tmp_path):
    spec = NowcastSpec.from_dict(
        {
            **BASE,
            "outputs": {
                "nowcast": True,
                "news": {"against": dt.date(2019, 1, 15), "by": "category"},
                "density": False,
                "diagnostics": {"trim": 0.2},
                "html": {"path": "out/r.html", "plotlyjs": "cdn", "title": "T", "n_periods": 4},
                "backtest": {
                    "start": "2018-01-01",
                    "end": "2018-06-01",
                    "benchmarks": ["RandomWalk", {"type": "AR", "p": 2}],
                    "target_offsets": [0],
                    "window": "rolling",
                    "window_length": 60,
                    "refit_every": 3,
                    "model": {"max_iter": 5},
                },
            },
        },
        base_dir=tmp_path,
    )
    outputs = spec.outputs
    assert outputs.density is None
    assert outputs.news is not None and outputs.news.against == "2019-01-15"
    assert outputs.news.by == "category"
    assert outputs.diagnostics is not None and outputs.diagnostics.options == {"trim": 0.2}
    report = outputs.report_html
    assert report is not None and report.path == tmp_path / "out/r.html"
    assert report.plotlyjs == "cdn" and report.title == "T" and report.n_periods == 4
    bt = outputs.backtest
    assert bt == BacktestOutput(
        start="2018-01-01",
        end="2018-06-01",
        benchmarks=(("RandomWalk", {}), ("AR", {"p": 2})),
        target_offsets=(0,),
        window="rolling",
        window_length=60,
        refit_every=3,
        model={"max_iter": 5},
    )
    out = outputs.to_dict()
    assert out["backtest"]["benchmarks"] == [{"type": "RandomWalk"}, {"type": "AR", "p": 2}]
    assert out["diagnostics"] == {"trim": 0.2}
    assert out["report_html"]["path"] == str(tmp_path / "out/r.html")


def test_outputs_single_string_and_absolute_report(tmp_path):
    spec = spec_of(outputs="density")
    assert spec.outputs.names == ("nowcast", "density")
    path = tmp_path / "r.html"
    spec = spec_of(outputs={"report_html": {"path": str(path)}})
    assert spec.outputs.report_html is not None and spec.outputs.report_html.path == path


def test_outputs_errors():
    issues, _ = issues_of(
        {
            **BASE,
            "outputs": {
                "nowcats": True,
                "news": {"against": "whenever", "colour": 1},
                "density": {"n_boot": -1},
                "diagnostics": {"bogus": 1},
                "report_html": {"plotlyjs": "web"},
                "backtest": {
                    "start": "2018-01-01",
                    "target_offsets": "x",
                    "benchmarks": ["ARR", 3],
                },
            },
        }
    )
    assert "did you mean 'nowcast'" in issues["outputs.nowcats"]
    assert "not a date" in issues["outputs.news.against"]
    assert "unknown key" in issues["outputs.news.colour"]
    assert ">= 0" in issues["outputs.density.n_boot"]
    assert "unknown key" in issues["outputs.diagnostics.bogus"]
    assert "'inline' or 'cdn'" in issues["outputs.report_html.plotlyjs"]
    assert "required" in issues["outputs.backtest.end"]
    assert "list of integers" in issues["outputs.backtest.target_offsets"]
    assert "did you mean 'AR'" in issues["outputs.backtest.benchmarks[0]"]
    assert "benchmark name" in issues["outputs.backtest.benchmarks[1]"]


def test_outputs_bad_container():
    issues, _ = issues_of({**BASE, "outputs": 3})
    assert "list of output names" in issues["outputs"]
    issues, _ = issues_of({**BASE, "outputs": [{"density": 3}]})
    assert "must be a mapping" in issues["outputs[0].density"]


def test_backtest_single_benchmark_mapping():
    spec = spec_of(
        outputs={"backtest": {"start": "2018-01", "end": "2018-03", "benchmarks": {"type": "AR"}}}
    )
    assert spec.outputs.backtest is not None
    assert spec.outputs.backtest.benchmarks == (("AR", {}),)


# ---------------------------------------------------------------- yaml / io
def test_from_yaml_and_round_trip(tmp_path):
    path = tmp_path / "my_spec.yaml"
    path.write_text(
        "target: gdp\n"
        "data: {source: simulated_dfm, vintage: 2019-06-15}\n"
        "model: {type: TwoStepDFM, factors: 2}\n"
        "outputs: [nowcast, density]\n"
        "snapshot_dir: snaps\n"
        "description: demo\n"
        "random_state: 3\n"
    )
    spec = NowcastSpec.from_yaml(path)
    assert spec.name == "my_spec"
    assert spec.source == str(path)
    assert spec.snapshot_dir == tmp_path / "snaps"
    assert spec.vintage == "2019-06-15"
    text = spec.to_yaml(tmp_path / "copy.yaml")
    again = NowcastSpec.from_string(text, base_dir=tmp_path)
    assert again.to_dict() == spec.to_dict()
    assert NowcastSpec.from_yaml(tmp_path / "copy.yaml").to_dict() == spec.to_dict()
    assert spec.to_dict()["description"] == "demo"
    assert spec.to_dict()["random_state"] == 3


def test_from_yaml_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        NowcastSpec.from_yaml(tmp_path / "nope.yaml")


def test_yaml_syntax_error():
    with pytest.raises(SpecError, match="invalid YAML") as info:
        NowcastSpec.from_string("target: [gdp\n", source="s.yaml")
    assert info.value.issues[0][0].startswith("line ")


def test_yaml_empty():
    with pytest.raises(SpecError, match="empty"):
        NowcastSpec.from_string("")


def test_replace():
    spec = spec_of()
    assert spec.replace(vintage="2019-01-01").vintage == "2019-01-01"


def test_load_spec_forms(tmp_path):
    path = tmp_path / "s.yaml"
    path.write_text("target: gdp\ndata: {source: simulated_dfm}\n")
    spec = load_spec(path)
    assert load_spec(spec) is spec
    assert load_spec(str(path)).name == "s"
    assert load_spec("target: gdp\ndata: {source: simulated_dfm}\n").target == "gdp"
    assert load_spec(dict(BASE)).target == "gdp"
    with pytest.raises(TypeError):
        load_spec(3)  # type: ignore[arg-type]


def test_templates_parse():
    assert set(list_templates()) == {
        "brazil_pib",
        "connectors",
        "csv",
        "model_building",
        "simulated",
    }
    for name in ("brazil_pib", "connectors", "model_building", "simulated"):
        spec = NowcastSpec.from_yaml(template_path(name))
        assert spec.name == name.replace("connectors", "brazil_live")
    with pytest.raises(SpecError, match="file not found"):
        NowcastSpec.from_yaml(template_path("csv"))


def test_template_paths_resolve_outside_package(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    spec = NowcastSpec.from_yaml(template_path("simulated"))
    assert spec.snapshot_dir == tmp_path / "snapshots"
    assert spec.base_dir == tmp_path


def test_template_unknown():
    with pytest.raises(ValueError, match="Unknown template"):
        template_path("nope")


def test_data_spec_kind():
    assert DataSpec(source="csv").kind == "file"
    assert DataSpec(source="connectors").kind == "connectors"
