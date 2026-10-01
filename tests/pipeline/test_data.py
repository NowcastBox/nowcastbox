"""Tests of nowcastbox.pipeline.data (loading, vintage, preprocessing, hashing)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import nowcastbox.data_sources as data_sources
from nowcastbox.core.data import MixedFrequencyData
from nowcastbox.core.frequency import Frequency
from nowcastbox.pipeline import (
    CONNECTORS,
    NowcastSpec,
    SpecError,
    apply_vintage,
    data_hash,
    frame_hash,
    load_data,
    preprocess,
    read_panel_file,
)
from nowcastbox.pipeline.spec import ConnectorSeries


def make_frame(n=36):
    idx = pd.period_range("2015-01", periods=n, freq="M")
    rng = np.random.default_rng(0)
    ip = 100 + np.cumsum(rng.normal(size=n))
    pmi = 50 + rng.normal(size=n)
    gdp = np.full(n, np.nan)
    gdp[2::3] = 100 + np.cumsum(rng.normal(size=n // 3))
    return pd.DataFrame({"gdp": gdp, "ip": ip, "pmi": pmi}, index=idx)


@pytest.fixture
def csv_file(tmp_path):
    frame = make_frame()
    out = frame.copy()
    out.index = out.index.to_timestamp(how="end").normalize()
    out.index.name = "date"
    path = tmp_path / "panel.csv"
    out.to_csv(path)
    return path


def file_spec(path, **data):
    return NowcastSpec.from_dict(
        {"target": "gdp", "data": {"source": "csv", "path": str(path), **data}}
    )


# ---------------------------------------------------------------- datasets
def test_load_dataset_selection_and_overrides():
    spec = NowcastSpec.from_dict(
        {
            "target": "gdp",
            "data": {
                "source": "simulated_dfm",
                "columns": ["x02", "gdp", "x01"],
                "start": "2010-01",
                "end": "2015-12",
                "transform": {"x01": "diff"},
                "delay": {"x01": 99},
                "blocks": {"x02": ["global", "extra"]},
                "categories": {"x02": "soft"},
            },
        }
    )
    panel = load_data(spec)
    assert panel.columns == ["gdp", "x02", "x01"]
    assert str(panel.start) == "2010-01" and str(panel.end) == "2015-12"
    assert panel.metadata["x01"].transform == "diff"
    assert panel.metadata["x01"].release_delay == 99
    assert panel.metadata["x02"].blocks == ("global", "extra")
    assert panel.metadata["x02"].category is not None
    assert panel.metadata["x02"].category.value == "soft"


def test_load_dataset_all_columns_target_first():
    spec = NowcastSpec.from_dict({"target": "x03", "data": {"source": "simulated_dfm"}})
    panel = load_data(spec)
    assert panel.columns[0] == "x03" and panel.n_series == 21


def test_load_dataset_override_unknown_series():
    spec = NowcastSpec.from_dict(
        {"target": "gdp", "data": {"source": "simulated_dfm", "delay": {"zz": 3}}}
    )
    with pytest.raises(SpecError, match=r"data\.delay\.zz"):
        load_data(spec)


def test_load_dataset_not_panel(monkeypatch):
    import nowcastbox.datasets as datasets

    spec = NowcastSpec.from_dict({"target": "gdp", "data": {"source": "simulated_dfm"}})
    monkeypatch.setattr(datasets, "load_dataset", lambda name: object())
    with pytest.raises(SpecError, match="not a panel"):
        load_data(spec)


def test_load_missing_target_or_columns():
    spec = NowcastSpec.from_dict({"target": "gdp", "data": {"source": "simulated_dfm"}})
    with pytest.raises(SpecError, match="not a series of the data"):
        load_data(spec.replace(target="nope"))
    data = spec.data.__class__(source="simulated_dfm", columns=("x01", "zz"))
    with pytest.raises(SpecError, match="series not in the data"):
        load_data(spec.replace(data=data))


# ---------------------------------------------------------------- files
def test_read_panel_file_csv(csv_file):
    frame = read_panel_file(csv_file, "date")
    assert isinstance(frame.index, pd.PeriodIndex) and frame.index.freqstr == "M"
    assert list(frame.columns) == ["gdp", "ip", "pmi"]
    assert frame["gdp"].notna().sum() == 12


def test_read_panel_file_quarter_labels(tmp_path):
    path = tmp_path / "q.csv"
    path.write_text("period,gdp\n2020Q1,1.0\n2020Q2,2.0\n")
    frame = read_panel_file(path)
    assert frame.index.astype(str).tolist() == ["2020-03", "2020-06"]


def test_read_panel_file_errors(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("date,x\nnot-a-date,1\n")
    with pytest.raises(SpecError, match="cannot parse the dates"):
        read_panel_file(path)
    with pytest.raises(SpecError, match="not found in the file"):
        read_panel_file(path, "when")


def test_read_panel_file_parquet(tmp_path):
    pytest.importorskip("pyarrow")
    frame = make_frame()
    by_index = tmp_path / "a.parquet"
    frame.to_parquet(by_index)
    out = read_panel_file(by_index)
    pd.testing.assert_frame_equal(out, frame.astype(float), check_names=False)
    by_column = tmp_path / "b.parquet"
    flat = frame.copy()
    flat.index = flat.index.to_timestamp()
    flat = flat.reset_index(names="when")
    flat.to_parquet(by_column)
    assert read_panel_file(by_column, "when").index.equals(frame.index)
    assert read_panel_file(by_column).index.equals(frame.index)
    named = tmp_path / "c.parquet"
    frame.rename_axis("when").to_parquet(named)
    assert read_panel_file(named, "when").index.equals(frame.index)


def test_load_csv_with_metadata(csv_file):
    spec = file_spec(
        csv_file,
        index_column="date",
        frequency={"gdp": "Q"},
        transform={"gdp": "qoq", "ip": "dlog"},
        delay={"gdp": 60, "ip": 30, "pmi": 1},
        blocks={"pmi": ["global", "soft"]},
        categories={"pmi": "soft"},
        columns=["ip"],
    )
    panel = load_data(spec)
    assert panel.columns == ["gdp", "ip"]
    assert panel.metadata["gdp"].frequency == Frequency.QUARTERLY
    assert panel.metadata["gdp"].release_delay == 60
    assert panel.metadata["ip"].transform == "dlog"


def test_load_csv_with_legend(tmp_path, csv_file):
    legend = tmp_path / "legend.csv"
    legend.write_text(
        "name,frequency,transform,delay_days,blocks,category\n"
        "gdp,Q,qoq,60,global;real,hard\n"
        "ip,M,dlog,30,global;real,hard\n"
        "pmi,M,level,1,global,soft\n"
    )
    spec = file_spec(csv_file, legend=str(legend), delay={"pmi": 2})
    panel = load_data(spec)
    assert panel.metadata["gdp"].blocks == ("global", "real")
    assert panel.metadata["pmi"].release_delay == 2  # spec overrides the legend
    assert panel.metadata["gdp"].transform == "qoq"


def test_load_csv_legend_errors(tmp_path, csv_file):
    legend = tmp_path / "legend.csv"
    legend.write_text("series,frequency\ngdp,Q\n")
    with pytest.raises(SpecError, match="'name' column"):
        load_data(file_spec(csv_file, legend=str(legend)))
    legend.write_text("name,frequency\nzz,Q\n")
    with pytest.raises(SpecError, match="not in the file"):
        load_data(file_spec(csv_file, legend=str(legend)))


def test_load_csv_duplicated_dates(tmp_path):
    path = tmp_path / "dup.csv"
    path.write_text("date,gdp\n2020-01-31,1\n2020-01-15,2\n")
    with pytest.raises(SpecError, match="duplicated dates"):
        load_data(file_spec(path))


def test_load_csv_gaps_reindexed(tmp_path):
    path = tmp_path / "gap.csv"
    path.write_text("date,gdp,ip\n2020-01,1,1\n2020-03,2,3\n2020-04,3,4\n2020-05,4,5\n")
    panel = load_data(file_spec(path, frequency={"gdp": "M"}))
    assert panel.n_periods == 5
    assert np.isnan(panel.to_frame()["gdp"].iloc[1])


# ---------------------------------------------------------------- connectors
def connector_spec():
    return NowcastSpec.from_dict(
        {
            "target": "gdp",
            "data": {
                "source": "connectors",
                "start": "2015-01",
                "series": {
                    "gdp": {
                        "source": "ibge",
                        "table": 1621,
                        "variable": 584,
                        "classifications": {11255: 90707},
                        "frequency": "Q",
                        "transform": "qoq",
                        "delay": 60,
                        "blocks": ["global"],
                        "category": "hard",
                    },
                    "ip": {"source": "bcb", "code": 21859, "frequency": "M", "delay": 40},
                    "ipea_x": {"source": "ipea", "code": "ABC", "frequency": "M"},
                    "fred_x": {"source": "fred", "code": "INDPRO", "frequency": "M"},
                },
            },
        }
    )


def test_load_connectors_real_fetchers(monkeypatch):
    frame = make_frame()
    calls = {}

    def fake(column):
        def fetch(*args, **kwargs):
            calls[column] = (args, kwargs)
            name = kwargs.get("name") or next(iter(args[0]))
            return frame[[column]].rename(columns={column: name})

        return fetch

    monkeypatch.setattr(data_sources, "fetch_sidra", fake("gdp"))
    monkeypatch.setattr(data_sources, "fetch_sgs", fake("ip"))
    monkeypatch.setattr(data_sources, "fetch_ipeadata", fake("pmi"))
    monkeypatch.setattr(data_sources, "fetch_fred", fake("ip"))
    panel = load_data(connector_spec())
    assert panel.columns == ["gdp", "ip", "ipea_x", "fred_x"]
    assert panel.metadata["gdp"].frequency == Frequency.QUARTERLY
    assert panel.metadata["gdp"].release_delay == 60
    assert panel.metadata["gdp"].blocks == ("global",)
    args, kwargs = calls["gdp"]
    assert args[0] == "1621" and kwargs["base_frequency"] == "M" and kwargs["start"] == "2015-01"
    assert calls["ip"][0][0] == {"fred_x": "INDPRO"}


def test_load_connectors_registry(monkeypatch):
    frame = make_frame()

    def fetch(series: ConnectorSeries, start, end):
        column = {"gdp": "gdp"}.get(series.name, "ip")
        out = frame[[column]]
        # unnamed column: renamed by the loader
        return (
            out.rename(columns={column: "value"})
            if series.name == "fred_x"
            else out.rename(columns={column: series.name})
        )

    for key in ("bcb", "ibge", "ipea", "fred"):
        monkeypatch.setitem(CONNECTORS, key, fetch)
    panel = load_data(connector_spec())
    assert "fred_x" in panel.columns


# ---------------------------------------------------------------- vintage
def simulated():
    spec = NowcastSpec.from_dict({"target": "gdp", "data": {"source": "simulated_dfm"}})
    return load_data(spec)


def test_apply_vintage_truncates_and_releases():
    panel = simulated()
    out = apply_vintage(panel, pd.Timestamp("2015-06-15"))
    assert str(out.end) == "2015-06"
    assert out.n_observations()["x02"] < panel.truncate(end="2015-06").n_observations()["x02"]


def test_apply_vintage_after_end_keeps_grid():
    panel = simulated()
    out = apply_vintage(panel, pd.Timestamp("2030-01-01"))
    assert out.end == panel.end


def test_apply_vintage_errors():
    panel = simulated()
    with pytest.raises(SpecError, match="before the data start"):
        apply_vintage(panel, pd.Timestamp("1990-01-01"))
    frame = make_frame()
    no_delay = MixedFrequencyData(frame, {"gdp": "Q", "ip": "M", "pmi": "M"})
    with pytest.raises(SpecError, match="publication delays"):
        apply_vintage(no_delay, pd.Timestamp("2016-01-01"))
    assert apply_vintage(no_delay, pd.Timestamp("2016-01-01"), explicit=False) is no_delay


# ---------------------------------------------------------------- preprocessing
def test_preprocess_modes(csv_file):
    spec = file_spec(
        csv_file,
        frequency={"gdp": "Q"},
        transform={"gdp": "qoq", "ip": "dlog"},
    )
    raw = load_data(spec)
    out = preprocess(raw, spec)  # metadata transforms
    assert out.metadata["ip"].transform_applied
    np.testing.assert_allclose(
        out.to_frame()["ip"].iloc[1],
        np.log(raw.to_frame()["ip"].iloc[1] / raw.to_frame()["ip"].iloc[0]),
        rtol=1e-6,
    )
    none = preprocess(
        raw, spec.replace(preprocessing=spec.preprocessing.__class__(transform=False))
    )
    np.testing.assert_allclose(
        none.to_frame()["ip"].dropna().iloc[:3], raw.to_frame()["ip"].iloc[:3]
    )
    override = spec.preprocessing.__class__(transform={"ip": "diff"}, options={"replace_na": False})
    out = preprocess(raw, spec.replace(preprocessing=override))
    np.testing.assert_allclose(
        out.to_frame()["ip"].iloc[1], raw.to_frame()["ip"].iloc[1] - raw.to_frame()["ip"].iloc[0]
    )
    bad = spec.preprocessing.__class__(transform={"zz": "diff"})
    with pytest.raises(SpecError, match=r"preprocessing\.transform"):
        preprocess(raw, spec.replace(preprocessing=bad))


# ---------------------------------------------------------------- hashing
def test_hashes():
    panel = simulated()
    assert data_hash(panel) == data_hash(panel.copy())
    assert data_hash(panel) != data_hash(panel.with_metadata("x01", release_delay=1))
    assert data_hash(panel) != data_hash(panel.truncate(end="2018-12"))
    frame = pd.DataFrame({"a": [0.1, 0.2]})
    assert len(frame_hash(frame)) == 64
    assert frame_hash(frame) != frame_hash(frame + 1e-15)


def test_file_spec_without_path():
    from nowcastbox.pipeline import DataSpec

    spec = NowcastSpec.from_dict({"target": "gdp", "data": {"source": "simulated_dfm"}})
    with pytest.raises(SpecError, match="is required"):
        load_data(spec.replace(data=DataSpec(source="csv")))
