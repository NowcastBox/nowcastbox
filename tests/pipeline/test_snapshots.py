"""Tests of nowcastbox.pipeline.snapshots (versioned nowcast snapshots)."""

from __future__ import annotations

import enum
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nowcastbox.pipeline import MANIFEST, Snapshot, SnapshotStore, headline_period, jsonable

IDX = pd.period_range("2020Q1", periods=3, freq="Q")


def table(nowcast=1.0, last_in=0.5):
    return pd.DataFrame(
        {
            "observed": [0.4, 0.6, np.nan],
            "in_sample": [0.45, last_in, np.nan],
            "out_of_sample": [np.nan, np.nan, nowcast],
            "std": [np.nan, np.nan, 0.2],
        },
        index=IDX,
    )


def panel(values=(1.0, 2.0, 3.0)):
    idx = pd.period_range("2020-01", periods=3, freq="M")
    return pd.DataFrame({"x": list(values), "y": [np.nan, 1.0, np.nan]}, index=idx)


@pytest.fixture
def store(tmp_path):
    return SnapshotStore(tmp_path / "snaps")


def write(store, value=1.0, name="demo", **kwargs):
    return store.write(name=name, target="gdp", nowcast=table(value), **kwargs)


def test_write_full_snapshot(store):
    snap = store.write(
        name="demo",
        target="gdp",
        nowcast=table(),
        vintage="2020-08-01",
        data_hash="ab" * 32,
        model="MixedFreqDFM",
        spec={"name": "demo", "target": "gdp", "model": {"type": "MixedFreqDFM"}},
        panel=panel(),
        params={"model_params": {"n_factors": 1}, "array": np.eye(2), "nan": np.nan},
        news=pd.DataFrame({"news": [0.1]}, index=["x"]),
        news_info={"against": "prev", "old_nowcast": 0.9, "new_nowcast": 1.0},
        tables={"density": pd.DataFrame({"mean": [1.0]}, index=IDX[-1:])},
        texts={"report.html": "<html></html>", "notes.txt": "hello"},
        metadata={"extra": 1},
        warnings=["w1"],
    )
    assert isinstance(snap, Snapshot)
    assert snap.path.parent == store.root
    assert snap.name == "demo" and snap.target == "gdp"
    assert snap.vintage == "2020-08-01" and snap.model == "MixedFreqDFM"
    assert snap.data_hash == "ab" * 32
    assert snap.headline_period == "2020Q3" and snap.headline == 1.0
    assert snap.previous is None
    assert snap.manifest["extra"] == 1
    assert snap.manifest["files"] == sorted(snap.files)
    assert snap.report_path == snap.path / "report.html"
    pd.testing.assert_frame_equal(snap.nowcast(), table(), check_names=False, check_freq=False)
    assert snap.nowcast().index.freqstr == "Q-DEC"
    stored_panel = snap.panel()
    assert stored_panel is not None
    pd.testing.assert_frame_equal(stored_panel, panel(), check_names=False)
    assert snap.params()["array"] == [[1.0, 0.0], [0.0, 1.0]]
    assert snap.params()["nan"] is None
    assert snap.spec()["model"]["type"] == "MixedFreqDFM"
    news = snap.news()
    assert news is not None and news.loc["x", "news"] == 0.1
    assert snap.table("density") is not None
    assert snap.table("nope") is None
    assert snap.file("notes.txt") is not None
    assert snap.estimate() == 1.0
    assert snap.estimate("2020Q2") == 0.5
    text = snap.summary()
    assert "news       : 0.9 -> 1 (against prev)" in text
    assert "warning    : w1" in text


def test_minimal_snapshot_defaults(store):
    snap = write(store)
    assert snap.panel() is None
    assert snap.params() == {}
    assert snap.spec() == {}
    assert snap.news() is None
    assert snap.report_path is None
    assert snap.manifest["base_frequency"] is None


def test_headline_period_rules():
    assert str(headline_period(table())) == "2020Q3"
    t = table()
    t["out_of_sample"] = np.nan
    assert headline_period(t) is None
    t = table()
    t["observed"] = np.nan
    t.loc[IDX[0], "out_of_sample"] = 0.3
    assert str(headline_period(t)) == "2020Q1"


def test_no_headline(store):
    t = table()
    t["out_of_sample"] = np.nan
    snap = store.write(name="demo", target="gdp", nowcast=t)
    assert snap.headline_period is None
    assert np.isnan(snap.headline)
    assert np.isnan(snap.estimate())
    assert "headline   : None = n/a" in snap.summary()
    other = write(store, 2.0)
    assert np.isnan(store.diff(other, snap).headline_change)


def test_explicit_headline_period_outside_table(store):
    snap = write(store, headline_period="2030Q1")
    assert snap.headline_period == "2030Q1"
    assert np.isnan(snap.headline)
    assert np.isnan(snap.estimate("2030Q1"))


def test_previous_and_listing(store):
    a = write(store, 1.0)
    b = write(store, 1.5)
    c = write(store, 9.0, name="other")
    assert b.previous == a.id
    assert c.previous is None
    assert len(store) == 3
    assert store.ids("demo") == [a.id, b.id]
    assert [s.id for s in store] == [a.id, b.id, c.id]
    assert a.id in store and "nope" not in store and 3 not in store
    latest = store.latest("demo")
    assert latest is not None and latest.id == b.id
    frame = store.to_frame("demo")
    assert list(frame.index) == [a.id, b.id]
    assert frame["headline"].tolist() == [1.0, 1.5]
    assert not frame["report"].any()
    assert "SnapshotStore(" in repr(store)


def test_empty_store(tmp_path):
    store = SnapshotStore(tmp_path / "none")
    assert len(store) == 0
    assert store.latest() is None
    assert store.to_frame().empty
    assert store.history().empty


def test_load_references(store):
    a = write(store, 1.0)
    b = write(store, 2.0)
    assert store.load(a.id).id == a.id
    assert store.load("latest").id == b.id
    assert store.load("latest~1").id == a.id
    assert store.load(b.id[:20]).id in (a.id, b.id)  # same second: prefix may be ambiguous
    with pytest.raises(KeyError, match="latest~5"):
        store.load("latest~5")
    with pytest.raises(KeyError, match="No snapshot matches"):
        store.load("2099")
    with pytest.raises(KeyError, match="Ambiguous"):
        store.load(a.id[:4])


def test_load_prefix_unique(store):
    a = store.write(name="d", target="gdp", nowcast=table(), created=pd.Timestamp("2021-01-01"))
    b = store.write(name="d", target="gdp", nowcast=table(), created=pd.Timestamp("2022-01-01"))
    assert store.load("2021").id == a.id
    assert store.load("2022", name="d").id == b.id


def test_created_timezones_and_collisions(store):
    stamp = pd.Timestamp("2021-05-01 12:00", tz="America/Sao_Paulo")
    a = store.write(name="d", target="gdp", nowcast=table(), created=stamp)
    b = store.write(name="d", target="gdp", nowcast=table(), created=stamp)
    assert a.created == stamp.tz_convert("UTC")
    assert a.id.startswith("20210501-150000-000000_d")
    assert b.id != a.id and b.id.startswith("20210501-150000-000001_d")


def test_history(store):
    write(store, 1.0, vintage="2020-08-01")
    t = table(2.0, last_in=0.55)
    t.loc[IDX[-1], "observed"] = 1.9
    store.write(
        name="demo", target="gdp", nowcast=t, vintage="2020-09-01", headline_period="2020Q3"
    )
    hist = store.history("demo")
    assert hist["value"].tolist() == [1.0, 2.0]
    assert hist["period"].tolist() == ["2020Q3", "2020Q3"]
    assert np.isnan(hist["observed"].iloc[0]) and hist["observed"].iloc[1] == 1.9
    hist = store.history(period="2020Q2")
    assert hist["value"].tolist() == [0.5, 0.55]
    hist = store.history(period="2031Q1")
    assert np.isnan(hist["value"]).all()


def test_history_without_headline(store):
    t = table()
    t["out_of_sample"] = np.nan
    store.write(name="demo", target="gdp", nowcast=t)
    hist = store.history()
    assert hist["period"].iloc[0] is None and np.isnan(hist["value"].iloc[0])


def test_diff(store):
    a = write(
        store,
        1.0,
        vintage="2020-08-01",
        data_hash="a",
        panel=panel(),
        spec={"vintage": "2020-08-01", "model": {"type": "MixedFreqDFM"}},
        params={"model_params": {"n_factors": 1, "tol": 1e-4}},
    )
    new_panel = panel((1.0, 2.5, 3.0))
    new_panel.loc[new_panel.index[0], "y"] = 7.0
    new_panel.loc[new_panel.index[1], "y"] = np.nan
    b = write(
        store,
        1.4,
        vintage="2020-09-01",
        data_hash="b",
        panel=new_panel,
        spec={"vintage": "2020-09-01", "model": {"type": "MixedFreqDFM"}},
        params={"model_params": {"n_factors": 2, "tol": 1e-4}},
    )
    diff = store.diff(a.id, "latest")
    assert diff.data_changed
    assert diff.data_revisions == {"new": 1, "revised": 1, "removed": 1}
    assert diff.spec_changes == {"vintage": ("2020-08-01", "2020-09-01")}
    assert diff.param_changes == {"n_factors": (1, 2)}
    assert diff.headline_change == pytest.approx(0.4)
    assert diff.nowcast["change"].tolist()[:2] == [0.0, 0.0]
    text = diff.summary()
    assert "data cells       : new 1, revised 1, removed 1" in text
    assert "spec vintage" in text and "params n_factors" in text
    out = diff.to_dict()
    assert out["old"] == a.id and out["new"] == b.id
    assert out["nowcast"]["2020Q3"]["new"] == 1.4
    json.dumps(out)


def test_diff_without_panels_and_frequency_mismatch(store):
    a = write(store)
    b = write(store)
    assert store.diff(a, b).data_revisions == {}
    assert not store.diff(a, b).data_changed
    monthly = pd.DataFrame(
        {"observed": [np.nan], "in_sample": [np.nan], "out_of_sample": [1.0]},
        index=pd.period_range("2020-01", periods=1, freq="M"),
    )
    c = store.write(name="demo", target="gdp", nowcast=monthly)
    with pytest.raises(ValueError, match="different frequencies"):
        store.diff(a, c)


def test_panel_revisions_frequency_mismatch(store):
    a = write(store, panel=panel())
    quarterly = pd.DataFrame({"x": [1.0]}, index=pd.period_range("2020Q1", periods=1, freq="Q"))
    b = write(store, panel=quarterly)
    assert store.diff(a, b).data_revisions == {}


def test_delete(store):
    a = write(store)
    store.delete(a.id)
    assert len(store) == 0
    with pytest.raises(KeyError):
        store.delete("latest")


def test_write_validation(store):
    bad = table().reset_index(drop=True)
    with pytest.raises(ValueError, match="PeriodIndex"):
        store.write(name="d", target="gdp", nowcast=bad)
    with pytest.raises(ValueError, match="lacks the columns"):
        store.write(name="d", target="gdp", nowcast=table().drop(columns="observed"))
    with pytest.raises(ValueError, match="Snapshot name"):
        store.write(name="bad name", target="gdp", nowcast=table())
    with pytest.raises(ValueError, match="Invalid snapshot file name"):
        write(store, texts={"../evil.txt": "x"})
    with pytest.raises(ValueError, match="Invalid snapshot file name"):
        write(store, texts={MANIFEST: "x"})
    # failed writes leave nothing behind
    assert len(store) == 0
    assert not [p for p in store.root.iterdir() if p.name.startswith(".tmp-")]


def test_unreadable_and_foreign_folders(store):
    a = write(store)
    (store.root / "notes").mkdir()
    broken = store.root / "20200101-000000-000000_broken"
    broken.mkdir()
    (broken / MANIFEST).write_text("{not json")
    (store.root / ".tmp-x").mkdir()
    (store.root / "file.txt").write_text("x")
    assert store.ids() == [a.id]


def test_jsonable_conversions():
    class Color(enum.Enum):
        RED = "red"

    value = {
        1: np.int64(3),
        "f": np.float32(1.5),
        "inf": float("inf"),
        "b": True,
        "s": {1, 2} and [1, 2],
        "set": {3},
        "e": Color.RED,
        "arr": np.zeros(3),
        "big": np.zeros(20_001),
        "ser": pd.Series([1.0, np.nan], index=["a", "b"]),
        "df": pd.DataFrame({"a": [1.0]}, index=["r"]),
        "bigdf": pd.DataFrame(np.zeros((101, 100))),
        "ts": pd.Timestamp("2020-01-01"),
        "path": Path("a/b"),
        "obj": object,
    }
    out = jsonable(value)
    assert out["1"] == 3 and out["f"] == 1.5 and out["inf"] is None and out["b"] is True
    assert out["set"] == [3] and out["e"] == "red" and out["arr"] == [0.0, 0.0, 0.0]
    assert out["big"].startswith("<array shape=")
    assert out["ser"] == {"a": 1.0, "b": None}
    assert out["df"] == {"a": {"r": 1.0}}
    assert out["bigdf"].startswith("<DataFrame shape=")
    assert out["ts"] == "2020-01-01 00:00:00" and out["path"] == str(Path("a/b"))
    assert "object" in out["obj"]
    json.dumps(out)
