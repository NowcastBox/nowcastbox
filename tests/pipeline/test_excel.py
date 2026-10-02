"""Excel workbooks: templates, reading, spec source and result export (plan item 7)."""

from __future__ import annotations

import sys
import types
import warnings

import numpy as np
import pandas as pd
import pytest

from nowcastbox.core.data import MixedFrequencyData
from nowcastbox.datasets import load_simulated_dfm
from nowcastbox.experiment import alternative_models
from nowcastbox.models import TwoStepDFM
from nowcastbox.pipeline import DataSpec, NowcastSpec, SpecError, load_data, run_pipeline
from nowcastbox.pipeline import data as pdata
from nowcastbox.pipeline.data import (
    EXCEL_METADATA_COLUMNS,
    example_workbook_path,
    read_excel_panel,
    write_excel_panel,
    write_run_excel,
)
from nowcastbox.pipeline.spec import ExcelOutput, OutputsSpec

pytest.importorskip("openpyxl")


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def _panel() -> MixedFrequencyData:
    idx = pd.period_range("2019-01", "2021-12", freq="M")
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "ip": rng.normal(size=len(idx)).round(6),
            "pmi": rng.normal(size=len(idx)).round(6),
            "gdp": np.nan,
            "pop": np.nan,
        },
        index=idx,
    )
    frame.loc[idx[idx.month % 3 == 0][:-1], "gdp"] = np.arange(11) / 10.0
    frame.loc[idx[idx.month == 12], "pop"] = [1.5, 2.5, np.nan]
    frame.loc[idx[-1], "pmi"] = np.nan  # ragged edge
    return MixedFrequencyData(
        frame,
        {"ip": "M", "pmi": "M", "gdp": "Q", "pop": "A"},
        transforms={"ip": 5, "pmi": "level", "gdp": 0},
        release_delays={"ip": 40, "gdp": 60, "pop": -10},
        blocks={"ip": ["global", "real"], "pmi": ["global"], "gdp": ["global", "real"]},
        categories={"ip": "hard", "pmi": "soft"},
        descriptions={"ip": "Industrial production"},
        aggregations={"gdp": "mariano_murasawa", "pop": "stock"},
        metadata={"pop": {"units": "millions", "transform_applied": True}},
    )


def _write_book(path, sheets: dict[str, pd.DataFrame]):
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False)
    return path


def _monthly(**columns) -> pd.DataFrame:
    return pd.DataFrame({"date": ["2020-01", "2020-02", "2020-03"], **columns})


# ---------------------------------------------------------------------------- round trip
def test_round_trip_keeps_values_and_metadata(tmp_path):
    panel = _panel()
    file = write_excel_panel(tmp_path / "sub" / "panel.xlsx", panel)
    assert pd.ExcelFile(file).sheet_names == [
        "monthly",
        "quarterly",
        "annual",
        "metadata",
        "readme",
    ]
    back = read_excel_panel(file)
    assert back.equals(panel)
    assert back.metadata == panel.metadata
    assert back.metadata["ip"].transform == 5 and back.metadata["pop"].release_delay == -10


def test_round_trip_simulated_dataset(tmp_path):
    panel = load_simulated_dfm().data
    assert read_excel_panel(write_excel_panel(tmp_path / "sim.xlsx", panel)).equals(panel)


def test_bundled_example_workbook():
    panel = read_excel_panel(example_workbook_path())
    assert panel.columns == ["gdp", *(f"x0{i}" for i in range(1, 9))]
    assert panel.metadata["x07"].category.value == "financial"
    assert panel.metadata["x04"].blocks == ("global", "soft")
    assert str(panel.start) == "2010-01" and str(panel.end) == "2019-12"
    source = load_simulated_dfm().data.truncate("2010-01", None)
    np.testing.assert_allclose(
        panel.values, source.select(panel.columns).values, equal_nan=True, atol=1e-12
    )


def test_empty_template(tmp_path):
    file = write_excel_panel(tmp_path / "t.xlsx")
    book = pd.ExcelFile(file)
    assert book.sheet_names == ["monthly", "quarterly", "metadata", "readme"]
    assert book.parse("metadata").columns.tolist() == list(EXCEL_METADATA_COLUMNS)
    with pytest.raises(SpecError, match="has no series"):
        read_excel_panel(file)


def test_write_rejects_non_monthly_panels(tmp_path):
    idx = pd.period_range("2020-01-06", periods=4, freq="W")
    weekly = MixedFrequencyData(pd.DataFrame({"a": [1.0, 2, 3, 4]}, index=idx), "W")
    with pytest.raises(ValueError, match="monthly panels"):
        write_excel_panel(tmp_path / "w.xlsx", weekly)


# ---------------------------------------------------------------------------- reading
def test_read_dates_aliases_and_sheet_roles(tmp_path):
    monthly = pd.DataFrame(
        {"Date": pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"]), "ip": [1, 2, 3]}
    )
    quarterly = pd.DataFrame({"when": [pd.Timestamp("2020-01-01")], "gdp": [0.5]})
    annual = pd.DataFrame({"year": [2020], "pop": [7.0]})
    meta = pd.DataFrame(
        {
            "name": ["ip", "gdp"],
            "delay": [30, 2.0],
            "blocks": ["global, real", None],
            "transform": [2.0, "dlog"],
            "transform_applied": ["true", None],
            "frequency": ["monthly", "Q"],
        }
    )
    file = _write_book(
        tmp_path / "b.xlsx", {"M": monthly, "trim": quarterly, "annual": annual, "meta": meta}
    )
    panel = read_excel_panel(file, sheets={"monthly": "M", "quarterly": "trim", "metadata": "meta"})
    assert panel.columns == ["ip", "gdp", "pop"]
    assert str(panel.end) == "2020-12"
    assert panel["gdp"].dropna().index.tolist() == [pd.Period("2020-03", "M")]
    assert panel.metadata["ip"].blocks == ("global", "real")
    assert panel.metadata["ip"].transform == 2 and panel.metadata["gdp"].transform == "dlog"
    assert panel.metadata["ip"].transform_applied is True
    assert panel.metadata["gdp"].release_delay == 2
    assert panel.metadata["pop"].frequency.value == "A"
    without = read_excel_panel(file, sheets={"monthly": "M", "annual": None})
    assert without.columns == ["ip"]


@pytest.mark.parametrize(
    ("sheets", "kwargs", "match"),
    [
        ({"monthly": _monthly(ip=[1, 2, 3])}, {"sheets": {"quarterly": "Q"}}, "not found"),
        ({"monthly": _monthly(ip=[1, 2, 3])}, {"sheets": {"weekly": "W"}}, "unknown sheet roles"),
        (
            {"monthly": pd.DataFrame({"date": ["2020-01", "xx"], "ip": [1, 2]})},
            {},
            "cannot parse the dates",
        ),
        (
            {"monthly": pd.DataFrame({"date": ["2020-01", "2020-01"], "ip": [1, 2]})},
            {},
            "duplicated dates",
        ),
        ({"monthly": _monthly(ip=[1, "x", 3])}, {}, "non-numeric"),
        (
            {
                "monthly": _monthly(gdp=[1, 2, 3]),
                "quarterly": pd.DataFrame({"date": ["2020Q1"], "gdp": [1.0]}),
            },
            {},
            "several sheets",
        ),
        (
            {"monthly": _monthly(ip=[1, 2, 3]), "metadata": pd.DataFrame({"freq": ["M"]})},
            {},
            "needs a 'series' column",
        ),
        (
            {
                "monthly": _monthly(ip=[1, 2, 3]),
                "metadata": pd.DataFrame({"series": ["ip"], "colour": ["red"]}),
            },
            {},
            "unknown columns",
        ),
        (
            {
                "monthly": _monthly(ip=[1, 2, 3]),
                "metadata": pd.DataFrame({"series": ["gdp"], "delay_days": [3]}),
            },
            {},
            "not in the data",
        ),
        (
            {
                "monthly": _monthly(ip=[1, 2, 3]),
                "metadata": pd.DataFrame({"series": ["ip"], "frequency": ["Q"]}),
            },
            {},
            "differs from its sheet",
        ),
        (
            {
                "monthly": _monthly(ip=[1, 2, 3]),
                "metadata": pd.DataFrame({"series": ["ip"], "frequency": ["X"]}),
            },
            {},
            "Cannot interpret 'X'",
        ),
        (
            {
                "monthly": _monthly(ip=[1, 2, 3]),
                "metadata": pd.DataFrame({"series": ["ip", None], "units": ["%", "bps"]}),
            },
            {},
            "row 3 has no series name",
        ),
        (
            {
                "monthly": _monthly(ip=[1, 2, 3]),
                "metadata": pd.DataFrame({"series": ["ip"], "transform": [9]}),
            },
            {},
            "Unknown transformation code",
        ),
        (
            {
                "monthly": _monthly(ip=[1, 2, 3]),
                "metadata": pd.DataFrame({"series": ["ip"], "delay_days": [1.5]}),
            },
            {},
            "whole number",
        ),
    ],
)
def test_read_errors(tmp_path, sheets, kwargs, match):
    file = _write_book(tmp_path / "bad.xlsx", sheets)
    with pytest.raises(SpecError, match=match):
        read_excel_panel(file, **kwargs)


def test_empty_data_sheet_and_unnamed_columns(tmp_path):
    monthly = pd.DataFrame({"date": ["2020-01", "2020-02"], "ip": [1.0, 2.0], "": [None, None]})
    file = _write_book(tmp_path / "u.xlsx", {"monthly": monthly, "quarterly": pd.DataFrame()})
    assert read_excel_panel(file).columns == ["ip"]


def test_missing_openpyxl(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "openpyxl", types.ModuleType("blocked"))
    monkeypatch.setitem(sys.modules, "openpyxl", None)
    with pytest.raises(ImportError, match=r"nowcastbox\[excel\]"):
        read_excel_panel(tmp_path / "x.xlsx")
    with pytest.raises(ImportError, match="openpyxl"):
        write_excel_panel(tmp_path / "x.xlsx")


# ---------------------------------------------------------------------------- spec
def test_spec_excel_source(tmp_path):
    file = write_excel_panel(tmp_path / "data" / "panel.xlsx", _panel())
    spec = NowcastSpec.from_dict(
        {
            "target": "gdp",
            "data": {
                "source": "excel",
                "path": "data/panel.xlsx",
                "sheets": {"quarterly": "quarterly", "metadata": "metadata"},
                "columns": ["ip", "pmi"],
                "categories": {"pmi": "financial"},
            },
        },
        base_dir=tmp_path,
    )
    assert spec.data.kind == "excel" and spec.data.path == file
    assert spec.data.sheets == {"quarterly": "quarterly", "metadata": "metadata"}
    panel = load_data(spec)
    assert panel.columns == ["gdp", "ip", "pmi"]
    assert panel.metadata["pmi"].category.value == "financial"
    again = NowcastSpec.from_dict(spec.to_dict(), base_dir=tmp_path)
    assert again.data == spec.data


def test_spec_excel_shortcut_and_validation(tmp_path):
    write_excel_panel(tmp_path / "p.xlsx", _panel())
    spec = NowcastSpec.from_dict({"target": "gdp", "data": {"source": "p.xlsx"}}, base_dir=tmp_path)
    assert spec.data.source == "excel" and spec.data.sheets is None
    with pytest.raises(SpecError) as err:
        NowcastSpec.from_dict(
            {
                "target": "gdp",
                "data": {"source": "excel", "path": "p.xlsx", "sheets": {"montly": "M"}, "x": 1},
            },
            base_dir=tmp_path,
        )
    text = str(err.value)
    assert "data.sheets.montly" in text and "monthly" in text and "data.x" in text


def test_load_excel_needs_a_path():
    with pytest.raises(SpecError, match=r"data\.path"):
        pdata._load_excel(DataSpec(source="excel"))
    assert DataSpec(source="excel").kind == "excel"


def test_pipeline_runs_on_the_example_workbook(tmp_path):
    run = run_pipeline(
        {
            "target": "gdp",
            "data": {"source": str(example_workbook_path())},
            "vintage": "2019-11-15",
            "preprocessing": False,
            "model": {"type": "TwoStepDFM", "factors": 1},
        }
    )
    assert run.data.columns[0] == "gdp"
    assert str(run.data.end) == "2019-11"
    # no look-ahead: x02 (65-day delay) is not released for August at mid-November
    assert np.isnan(run.data["x02"].loc[pd.Period("2019-09", "M")])


# ---------------------------------------------------------------------------- outputs
def test_outputs_excel_spec(tmp_path):
    spec = NowcastSpec.from_dict(
        {
            "target": "gdp",
            "data": {"source": "simulated_dfm"},
            "outputs": ["nowcast", {"excel": {"path": "out/results.xlsx"}}],
        },
        base_dir=tmp_path,
    )
    assert spec.outputs.names == ("nowcast", "excel")
    assert spec.outputs.excel == ExcelOutput(path=tmp_path / "out" / "results.xlsx")
    plain = NowcastSpec.from_dict(
        {"target": "gdp", "data": {"source": "simulated_dfm"}, "outputs": ["xlsx"]}
    )
    assert plain.outputs.excel == ExcelOutput()
    assert OutputsSpec(excel=ExcelOutput()).to_dict() == {"nowcast": True, "excel": True}
    absolute = NowcastSpec.from_dict(
        {
            "target": "gdp",
            "data": {"source": "simulated_dfm"},
            "outputs": {"excel": {"path": str(tmp_path / "r.xlsx")}},
        }
    )
    assert absolute.outputs.to_dict()["excel"] == {"path": str(tmp_path / "r.xlsx")}
    with pytest.raises(SpecError, match=r"outputs\.excel\.sheet"):
        NowcastSpec.from_dict(
            {
                "target": "gdp",
                "data": {"source": "simulated_dfm"},
                "outputs": {"excel": {"sheet": "x"}},
            }
        )


def test_write_run_excel_with_pipeline_run(tmp_path):
    run = run_pipeline(
        {
            "target": "gdp",
            "data": {"source": "simulated_dfm", "columns": ["x01", "x02", "x03"]},
            "vintage": "2019-11-15",
            "preprocessing": False,
            "model": {"type": "MixedFreqDFM", "factors": 1, "max_iter": 10},
            "outputs": ["nowcast", "density", "diagnostics"],
        }
    )
    file = write_run_excel(run, tmp_path / "o" / "run.xlsx")
    book = pd.ExcelFile(file)
    for sheet in ("nowcast", "loadings", "factors", "density", "diagnostics", "data", "info"):
        assert sheet in book.sheet_names
    nowcast = book.parse("nowcast", index_col=0)
    assert nowcast.index[-1] == str(run.results.nowcast.index[-1])
    info = book.parse("info", index_col=0)
    assert info.loc["target", "value"] == "gdp"


class _Table:
    def __init__(self, frame):
        self.frame = frame
        self.by = None

    def to_frame(self, by=None):
        self.by = by
        return self.frame

    def rmsfe_by_horizon(self):
        return self.frame

    def series_overview(self):
        return self.frame


def test_write_run_excel_duck_typed(tmp_path):
    data = load_simulated_dfm().data.select(["gdp", "x01", "x02", "x03", "x04"])
    groups = {"a": ["x01", "x02"], "b": ["x03", "x04"]}
    alt = alternative_models(TwoStepDFM(n_factors=1), data, "gdp", by=groups, refit=False)
    table = _Table(pd.DataFrame({"impact": [0.1]}, index=["x01"]))
    backtest = _Table(pd.DataFrame({"error": [0.2]}))
    spec = types.SimpleNamespace(
        outputs=types.SimpleNamespace(news=types.SimpleNamespace(by="category"))
    )
    run = types.SimpleNamespace(
        results=alt.base, news=table, backtest=backtest, alternatives=alt, spec=spec
    )
    file = write_run_excel(run, tmp_path / "duck.xlsx")
    book = pd.ExcelFile(file)
    assert table.by == "category"
    assert {"news", "backtest", "backtest_rmsfe", "alternatives", "alternatives_range"} <= set(
        book.sheet_names
    )
    assert book.parse("info").empty
    ranges = book.parse("alternatives_range", index_col=0)
    assert ranges.index.tolist() == [str(p) for p in alt.periods]
    alternatives = book.parse("alternatives", index_col=0)
    assert str(alt.periods[0]) in alternatives.columns
    no_spec = types.SimpleNamespace(results=alt.base, news=table)
    write_run_excel(no_spec, tmp_path / "plain.xlsx")
    assert table.by == "series"


def test_private_helpers():
    assert pdata._blank(None) and pdata._blank(float("nan")) and pdata._blank("  ")
    assert not pdata._blank(0)
    frame = pd.DataFrame([[1.0]], columns=pd.period_range("2020Q1", periods=1, freq="Q"))
    assert pdata._excel_ready(frame).columns.tolist() == ["2020Q1"]


def test_metadata_without_frequency_takes_the_sheet_frequency(tmp_path):
    meta = pd.DataFrame({"series": ["ip"], "delay_days": [30], "blocks": ["global, real"]})
    file = _write_book(tmp_path / "m.xlsx", {"monthly": _monthly(ip=[1, 2, 3]), "metadata": meta})
    info = read_excel_panel(file).metadata["ip"]
    assert info.frequency.value == "M" and info.release_delay == 30
    assert info.blocks == ("global", "real")
