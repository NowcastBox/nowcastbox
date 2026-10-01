"""Small helpers of nowcastbox.pipeline (formatting, serialisation)."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

import nowcastbox.pipeline.runner as runner
import nowcastbox.pipeline.snapshots as snapshots
from nowcastbox.pipeline import NowcastSpec, SnapshotStore


def test_format_helpers():
    assert runner._fmt("abc") == "abc"
    assert runner._fmt(None) == "None"
    assert runner._fmt(np.nan) == "n/a"
    assert runner._fmt(1.23456) == "1.235"
    assert snapshots._fmt("abc") == "abc"
    assert snapshots._fmt(None) == "n/a"


def test_read_indexed_csv_without_frequency(tmp_path):
    path = tmp_path / "t.csv"
    pd.DataFrame({"a": [1, 2]}, index=["r1", "r2"]).to_csv(path)
    frame = snapshots._read_indexed_csv(path, None)
    assert frame.index.tolist() == ["r1", "r2"] and frame["a"].dtype == float


def test_headline_change_period_missing(tmp_path):
    store = SnapshotStore(tmp_path)
    idx = pd.period_range("2020Q1", periods=1, freq="Q")
    t = pd.DataFrame({"observed": [np.nan], "in_sample": [np.nan], "out_of_sample": [1.0]}, idx)
    a = store.write(name="a", target="y", nowcast=t)
    b = store.write(name="a", target="y", nowcast=t, headline_period="2021Q1")
    assert np.isnan(store.diff(a, b).headline_change)


def test_spec_scalar_list_and_dates():
    spec = NowcastSpec.from_dict(
        {
            "target": "gdp",
            "data": {"source": "simulated_dfm", "columns": "x01"},
            "model": {"exclude_periods": [dt.date(2020, 3, 1)]},
        }
    )
    assert spec.data.columns == ("x01",)
    assert spec.to_dict()["model"]["exclude_periods"] == ["2020-03-01"]
