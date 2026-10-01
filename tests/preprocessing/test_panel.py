"""Tests for nowcastbox.preprocessing.panel."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from nowcastbox.core.data import MixedFrequencyData, StandardizationStats
from nowcastbox.core.exceptions import DataQualityWarning, NowcastDataError
from nowcastbox.preprocessing.aggregation import mariano_murasawa_weights, rolling_aggregate
from nowcastbox.preprocessing.panel import (
    PanelReport,
    destandardize,
    prepare_panel,
    standardize,
)
from nowcastbox.preprocessing.transforms import apply_transforms
from tests.preprocessing.conftest import FREQS

SPECS = {"ip": "dlog", "sales": 2, "gdp": 7}


def test_prepare_panel_dataframe_basic(ragged_levels):
    out = prepare_panel(ragged_levels, SPECS, frequency=FREQS)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["ip", "sales", "gdp"]
    assert str(out.index[0]) == "2015-02"  # first month lost to the first difference
    expected_ip = np.log(ragged_levels["ip"]).diff().iloc[1:]
    np.testing.assert_allclose(out["ip"], expected_ip, equal_nan=True)
    # ragged edge preserved
    assert np.isnan(out["ip"].iloc[-1])
    assert out["sales"].iloc[-2:].isna().all()
    assert np.isnan(out["gdp"].iloc[-1])


def test_prepare_panel_mfd_uses_metadata(panel):
    out, report = prepare_panel(panel, return_report=True)
    assert isinstance(out, MixedFrequencyData)
    assert isinstance(report, PanelReport)
    assert report.transforms == {
        "ip": "log|diff(1)",
        "sales": "diff(1)",
        "gdp": "pct_change(quarter)",
    }
    assert out.metadata["gdp"].release_delay == 60
    assert out.metadata["ip"].transform == "dlog"
    pd.testing.assert_frame_equal(
        out.ragged_edge_mask(), panel.truncate(start=out.start).ragged_edge_mask()
    )
    frame = report.to_frame()
    assert list(frame.columns) == [
        "transform",
        "missing_proportion",
        "dropped",
        "n_outliers",
        "n_filled",
    ]
    assert not frame["dropped"].any()
    assert report.start == out.start


def test_prepare_panel_outliers_and_na(panel):
    frame = panel.to_frame()
    frame.iloc[20, 0] = np.nan  # interior gap in the level -> 2 NaNs in dlog
    frame.iloc[30, 0] = frame.iloc[30, 0] * 5  # spike -> 2 outliers in dlog
    mfd = MixedFrequencyData(frame, FREQS, transforms=SPECS)
    out, report = prepare_panel(mfd, return_report=True)
    assert report.n_outliers["ip"] == 2
    assert report.n_filled["ip"] == 2
    assert out.to_frame()["ip"].iloc[:-1].notna().all()
    # options off: nothing replaced or filled
    raw, report_raw = prepare_panel(
        mfd, replace_outliers=False, replace_na=False, return_report=True
    )
    assert report_raw.n_outliers.sum() == 0
    assert report_raw.n_filled.sum() == 0
    assert raw.to_frame()["ip"].isna().sum() == 3


def test_prepare_panel_fill_ragged_edge(panel):
    out = prepare_panel(panel, fill_ragged_edge=True, edge_method="last")
    frame = out.to_frame()
    assert frame[["ip", "sales"]].notna().all().all()
    assert frame["sales"].iloc[-1] == frame["sales"].iloc[-3]
    assert not np.isnan(frame["gdp"].iloc[-1])


def test_prepare_panel_drops_sparse_series(ragged_levels):
    df = ragged_levels.copy()
    df["sparse"] = np.nan
    df.iloc[-10:, -1] = np.arange(1.0, 11.0) ** 2
    freqs = {**FREQS, "sparse": "M"}
    with pytest.warns(DataQualityWarning, match="Dropped 1 series"):
        out, report = prepare_panel(df, {**SPECS, "sparse": 0}, frequency=freqs, return_report=True)
    assert "sparse" not in out.columns
    assert report.dropped == ["sparse"]
    assert report.missing_proportion["sparse"] > 1 / 3
    assert bool(report.to_frame().loc["sparse", "dropped"])
    # keep protects a series; max_na_prop=None disables dropping
    with warnings.catch_warnings():
        warnings.simplefilter("error", DataQualityWarning)
        kept = prepare_panel(df, {**SPECS, "sparse": 0}, frequency=freqs, keep="sparse")
        assert "sparse" in kept.columns
        kept2 = prepare_panel(df, {**SPECS, "sparse": 0}, frequency=freqs, max_na_prop=None)
        assert "sparse" in kept2.columns


def test_prepare_panel_all_dropped_raises(ragged_levels):
    with pytest.raises(NowcastDataError, match="nothing is left"):
        prepare_panel(ragged_levels, SPECS, frequency=FREQS, max_na_prop=0.0)


def test_prepare_panel_empty_after_transform():
    idx = pd.period_range("2020-01", periods=3, freq="M")
    df = pd.DataFrame({"a": [1.0, np.nan, np.nan]}, index=idx)
    with pytest.raises(NowcastDataError, match="no observation"):
        prepare_panel(df, "diff", frequency="M")


def test_prepare_panel_aggregate(panel):
    out = prepare_panel(panel, aggregate=True, replace_outliers=False, replace_na=False)
    growth = apply_transforms(panel).truncate(start="2015-02")
    expected = rolling_aggregate(growth["ip"], mariano_murasawa_weights(normalize=True))
    np.testing.assert_allclose(out["ip"], expected, equal_nan=True)
    np.testing.assert_allclose(out["gdp"], growth["gdp"], equal_nan=True)
    out_avg, report = prepare_panel(
        panel, aggregate=True, aggregation="average", return_report=True
    )
    assert report.aggregated
    assert out_avg["ip"].iloc[:2].isna().all()


def test_prepare_panel_without_dropping_leading(panel):
    out = prepare_panel(panel, drop_leading_empty=False)
    assert out.start == panel.start
    assert out.to_frame().iloc[0].isna().all()


def test_prepare_panel_validation(panel):
    for bad in [-0.1, 1.5, "a", True]:
        with pytest.raises(ValueError, match="max_na_prop"):
            prepare_panel(panel, max_na_prop=bad)
    with pytest.raises(NowcastDataError, match="keep"):
        prepare_panel(panel, keep=["nope"])
    with pytest.raises(ValueError):
        prepare_panel(panel, transform="nonsense")
    with pytest.raises(NowcastDataError, match="frequency cannot"):
        prepare_panel(panel, frequency=FREQS)


def test_prepare_panel_does_not_mutate_input(ragged_levels):
    before = ragged_levels.copy()
    prepare_panel(ragged_levels, SPECS, frequency=FREQS, fill_ragged_edge=True)
    pd.testing.assert_frame_equal(ragged_levels, before)


def test_prepare_panel_legend_codes_sequence(ragged_levels):
    a = prepare_panel(ragged_levels, ["dlog", 2, 7], frequency=[12, 12, 4])
    b = prepare_panel(ragged_levels, SPECS, frequency=FREQS)
    pd.testing.assert_frame_equal(a, b)


# ---------------------------------------------------------------------- standardize


def test_standardize_dataframe_round_trip(levels):
    z, stats = standardize(levels)
    assert isinstance(stats, StandardizationStats)
    np.testing.assert_allclose(z.mean(), 0.0, atol=1e-12)
    np.testing.assert_allclose(z.std(), 1.0)
    back = destandardize(z, stats)
    pd.testing.assert_frame_equal(back, levels)
    # reuse statistics
    z2, stats2 = standardize(levels * 2, stats)
    assert stats2 is stats
    np.testing.assert_allclose(destandardize(z2, stats), levels * 2)


def test_standardize_mfd_matches_core(panel):
    z, stats = standardize(panel, ddof=0)
    z_core, stats_core = panel.standardize(ddof=0)
    assert z.equals(z_core)
    pd.testing.assert_series_equal(stats.std, stats_core.std)
    assert destandardize(z, stats).equals(panel, atol=1e-9)


def test_standardize_errors(levels):
    bad = levels.copy()
    bad["const"] = 1.0
    with pytest.raises(NowcastDataError, match="zero variance"):
        standardize(bad)
    with pytest.raises(NowcastDataError, match="DataFrame"):
        standardize(levels.to_numpy())  # type: ignore[arg-type]
    _, stats = standardize(levels)
    with pytest.raises(NowcastDataError, match="DataFrame"):
        destandardize(levels.to_numpy(), stats)  # type: ignore[arg-type]
