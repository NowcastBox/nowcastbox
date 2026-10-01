"""Tests of the data-quality report."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nowcastbox.core.data import MixedFrequencyData
from nowcastbox.core.exceptions import DataQualityWarning
from nowcastbox.diagnostics import DataQualityReport, data_quality_report


@pytest.fixture
def panel():
    idx = pd.period_range("2020-01", periods=12, freq="M")
    a = np.r_[np.nan, np.nan, np.arange(5.0), np.nan, 3.0, 4.0, np.nan, np.nan]
    b = np.r_[np.arange(10.0) % 3, 100.0, 1.0]
    q = np.full(12, np.nan)
    q[[2, 5]] = [1.0, 2.0]
    empty = np.full(12, np.nan)
    df = pd.DataFrame({"a": a, "b": b, "q": q, "empty": empty}, index=idx)
    with pytest.warns(DataQualityWarning, match="without any observation"):
        return MixedFrequencyData(
            df,
            {"a": "M", "b": "M", "q": "Q", "empty": "M"},
            release_delays={"a": 30, "b": 30, "q": 60},
            categories={"a": "hard", "b": "soft"},
        )


def test_counts_are_exact(panel):
    rep = data_quality_report(panel)
    t = rep.table
    assert t.loc["a", "n_slots"] == 12 and t.loc["a", "n_obs"] == 7
    assert t.loc["a", "leading_missing"] == 2
    assert t.loc["a", "interior_missing"] == 1
    assert t.loc["a", "ragged_edge"] == 2
    assert t.loc["a", "na_share"] == pytest.approx(5 / 12)
    assert t.loc["a", "first_observed"] == "2020-03"
    assert t.loc["a", "last_observed"] == "2020-10"
    assert t.loc["q", "n_slots"] == 4 and t.loc["q", "ragged_edge"] == 2
    assert t.loc["q", "last_observed"] == "2020Q2"
    assert t.loc["empty", "ragged_edge"] == 12 and t.loc["empty", "leading_missing"] == 0
    assert pd.isna(t.loc["empty", "first_observed"])
    assert t.loc["a", "category"] == "hard" and pd.isna(t.loc["q", "category"])
    assert t.loc["a", "release_delay"] == 30 and pd.isna(t.loc["empty", "release_delay"])


def test_outliers_and_flags(panel):
    rep = data_quality_report(panel, max_na_share=0.4)
    assert rep.table.loc["b", "n_outliers"] == 1
    out = rep.outliers
    assert out.to_dict("records") == [{"series": "b", "period": "2020-11", "value": 100.0}]
    assert rep.table.loc["b", "flags"] == "outliers(1)"
    assert rep.table.loc["empty", "flags"] == "no_observations"
    assert rep.table.loc["a", "flags"] == "na_share>0.4"
    assert rep.flagged == ["a", "b", "q", "empty"]
    assert rep.table["max_abs_loading"].isna().all()


def test_loadings_flags(panel):
    loadings = pd.DataFrame(
        {"f1": [0.9, 0.05, -0.02], "f2": [0.1, 0.02, 0.5]}, index=["a", "b", "q"]
    )
    rep = data_quality_report(panel, loadings=loadings, loading_tol=0.1)
    t = rep.table
    assert t.loc["a", "max_abs_loading"] == pytest.approx(0.9)
    assert bool(t.loc["b", "near_zero_loading"])
    assert not bool(t.loc["q", "near_zero_loading"])
    assert np.isnan(t.loc["empty", "max_abs_loading"])
    assert "near_zero_loading" in t.loc["b", "flags"]
    assert "b" in rep.summary()


def test_publication_summary(panel):
    pub = data_quality_report(panel).publication
    assert pub.columns.tolist() == [
        "frequency",
        "release_delay",
        "n_series",
        "mean_ragged_edge",
        "max_ragged_edge",
        "series",
    ]
    m30 = pub[(pub.frequency == "M") & (pub.release_delay == 30)].iloc[0]
    assert m30["n_series"] == 2 and m30["series"] == "a, b"
    assert m30["mean_ragged_edge"] == pytest.approx(1.0)
    unknown = pub[pub["release_delay"].isna()].iloc[0]
    assert unknown["series"] == "empty"


def test_warning_and_summary(panel):
    with pytest.warns(DataQualityWarning, match="Data-quality flags in"):
        rep = data_quality_report(panel, warn=True)
    assert isinstance(rep, DataQualityReport)
    text = rep.summary()
    assert text.startswith("Data quality") and "! empty" in text and "2020-12" in text
    frame = rep.to_frame()
    frame.loc["a", "n_obs"] = -1
    assert rep.table.loc["a", "n_obs"] == 7


def test_dataframe_input_and_clean_panel(rng):
    idx = pd.period_range("2020-01", periods=24, freq="M")
    df = pd.DataFrame(rng.standard_normal((24, 2)), index=idx, columns=["x", "y"])
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rep = data_quality_report(df, frequency="M", warn=True)
    assert rep.flagged == []
    assert rep.outliers.empty and list(rep.outliers.columns) == ["series", "period", "value"]
    assert rep.thresholds == {"outlier_threshold": 4.0, "loading_tol": 0.1, "max_na_share": 0.5}


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"outlier_threshold": 0}, "outlier_threshold"),
        ({"loading_tol": -1.0}, "loading_tol"),
        ({"max_na_share": 1.5}, "max_na_share"),
        ({"max_na_share": "x"}, "max_na_share"),
        ({"loading_tol": np.inf}, "loading_tol"),
    ],
)
def test_invalid_thresholds(panel, kwargs, match):
    with pytest.raises(ValueError, match=match):
        data_quality_report(panel, **kwargs)
