"""Tests for nowcastbox.preprocessing.missing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.interpolate import CubicSpline

from nowcastbox.core.data import MixedFrequencyData
from nowcastbox.core.exceptions import NowcastDataError
from nowcastbox.preprocessing.missing import (
    fill_missing,
    interior_missing_mask,
    leading_missing_mask,
    missing_proportion,
    ragged_edge_mask,
)
from tests.preprocessing.conftest import FREQS

IDX = pd.period_range("2020-01", periods=10, freq="M")


@pytest.fixture
def gappy() -> pd.Series:
    return pd.Series(
        [np.nan, 1.0, 2.0, np.nan, 4.0, np.nan, np.nan, 7.0, np.nan, np.nan], index=IDX
    )


def test_masks_partition_missing(gappy):
    lead = leading_missing_mask(gappy, frequency="M")
    inner = interior_missing_mask(gappy, frequency="M")
    ragged = ragged_edge_mask(gappy, frequency="M")
    assert np.flatnonzero(lead).tolist() == [0]
    assert np.flatnonzero(inner).tolist() == [3, 5, 6]
    assert np.flatnonzero(ragged).tolist() == [8, 9]
    total = lead.astype(int) + inner.astype(int) + ragged.astype(int)
    assert (total == gappy.isna().astype(int)).all()


def test_all_missing_series_is_ragged():
    x = pd.Series(np.nan, index=IDX)
    assert ragged_edge_mask(x, frequency="M").all()
    assert not interior_missing_mask(x, frequency="M").any()
    assert not leading_missing_mask(x, frequency="M").any()
    assert missing_proportion(x, frequency="M") == 1.0


def test_ragged_mask_matches_core(panel):
    pd.testing.assert_frame_equal(ragged_edge_mask(panel), panel.ragged_edge_mask())
    pd.testing.assert_frame_equal(
        ragged_edge_mask(panel.to_frame(), frequency=FREQS), panel.ragged_edge_mask()
    )


def test_quarterly_masks_ignore_structural_nans(panel):
    inner = interior_missing_mask(panel)
    assert not inner["gdp"].any()
    lead = leading_missing_mask(panel)
    assert not lead.to_numpy().any()


def test_missing_proportion(panel):
    prop = missing_proportion(panel)
    n_q = int(panel.slot_mask()["gdp"].sum())
    assert prop["gdp"] == pytest.approx(1 / n_q)
    assert prop["sales"] == pytest.approx(2 / panel.n_periods)
    prop_df = missing_proportion(panel.to_frame(), frequency=FREQS)
    pd.testing.assert_series_equal(prop, prop_df)
    assert missing_proportion(pd.Series([1.0, np.nan], index=IDX[:2]), frequency="M") == 0.5


def test_fill_linear_and_edges(gappy):
    out = fill_missing(gappy, "linear", frequency="M")
    np.testing.assert_allclose(out.iloc[1:8], np.arange(1.0, 8.0))
    assert out.iloc[[0, 8, 9]].isna().all()


def test_fill_spline_matches_scipy(rng):
    t = np.arange(30.0)
    x = np.sin(t / 4)
    s = pd.Series(x.copy(), index=pd.period_range("2000-01", periods=30, freq="M"))
    gaps = [5, 6, 17, 22]
    s.iloc[gaps] = np.nan
    out = fill_missing(s, "spline", frequency="M")
    obs = s.notna().to_numpy()
    ref = CubicSpline(t[obs], x[obs])(t[gaps])
    np.testing.assert_allclose(out.iloc[gaps], ref)
    np.testing.assert_allclose(out.iloc[gaps], x[gaps], atol=5e-3)
    # observed values untouched
    pd.testing.assert_series_equal(out[obs], s[obs])


def test_fill_spline_exact_on_cubic():
    t = np.arange(12.0)
    x = 0.1 * t**3 - t**2 + 2
    s = pd.Series(x.copy(), index=pd.period_range("2000-01", periods=12, freq="M"))
    s.iloc[[3, 8]] = np.nan
    np.testing.assert_allclose(fill_missing(s, frequency="M"), x, atol=1e-9)


def test_fill_moving_median(gappy):
    out = fill_missing(gappy, "moving_median", window=3, frequency="M")
    assert out.iloc[3] == pytest.approx(np.median([2.0, 4.0]))
    assert out.iloc[5] == pytest.approx(4.0)  # neighbours: 4 and NaN
    assert out.iloc[6] == pytest.approx(7.0)
    assert out.iloc[[0, 8, 9]].isna().all()


def test_fill_moving_median_widens_window():
    x = pd.Series(
        [1.0] + [np.nan] * 7 + [3.0], index=pd.period_range("2000-01", periods=9, freq="M")
    )
    out = fill_missing(x, "moving_median", window=1, frequency="M")
    assert out.notna().all()
    assert out.iloc[4] == pytest.approx(2.0)  # both ends reached at the same width


@pytest.mark.parametrize(
    ("edge_method", "ragged_value", "leading_value"),
    [("last", 7.0, 1.0), ("mean", 3.5, 3.5), ("median", 4.0, 2.0)],
)
def test_fill_edges(gappy, edge_method, ragged_value, leading_value):
    out = fill_missing(
        gappy,
        "linear",
        fill_ragged_edge=True,
        fill_leading=True,
        edge_method=edge_method,
        window=3,
        frequency="M",
    )
    assert out.notna().all()
    assert out.iloc[-1] == pytest.approx(ragged_value)
    assert out.iloc[0] == pytest.approx(leading_value)


def test_fill_edges_on_empty_series():
    x = pd.Series(np.nan, index=IDX)
    out = fill_missing(x, fill_ragged_edge=True, fill_leading=True, frequency="M")
    assert out.isna().all()


def test_fill_quarterly_on_native_grid():
    idx = pd.period_range("2020-01", periods=15, freq="M")
    q = pd.Series(np.nan, index=idx)
    q.iloc[[2, 5, 11, 14]] = [1.0, 2.0, 4.0, 5.0]  # 2020Q3 (month 8) missing
    out = fill_missing(q, "linear", frequency="Q")
    assert out.iloc[8] == pytest.approx(3.0)
    assert out.notna().sum() == 5
    assert out.iloc[[0, 1, 3, 4, 6, 7, 9, 10]].isna().all()


def test_fill_panel(panel):
    frame = panel.to_frame()
    frame.iloc[10, 0] = np.nan
    frame.iloc[11, 2] = np.nan  # a quarter (month 12) in the middle
    mfd = MixedFrequencyData(frame, FREQS)
    out = fill_missing(mfd)
    assert isinstance(out, MixedFrequencyData)
    assert not np.isnan(out.to_frame().iloc[10, 0])
    assert not np.isnan(out.to_frame().iloc[11, 2])
    # ragged edge preserved
    pd.testing.assert_frame_equal(out.ragged_edge_mask(), mfd.ragged_edge_mask())
    out_df = fill_missing(frame, frequency=FREQS, columns=["ip"])
    assert isinstance(out_df, pd.DataFrame)
    assert not np.isnan(out_df.iloc[10, 0])
    assert np.isnan(out_df.iloc[11, 2])
    with pytest.raises(NowcastDataError, match="Unknown"):
        fill_missing(frame, frequency=FREQS, columns=["zz"])


def test_fill_ragged_edge_panel(panel):
    out = fill_missing(panel, fill_ragged_edge=True)
    frame = out.to_frame()
    assert frame.iloc[-1][["ip", "sales"]].notna().all()
    assert not np.isnan(frame["gdp"].iloc[-1])
    assert np.isnan(frame["gdp"].iloc[-2])  # structural NaN never filled


@pytest.mark.parametrize(
    "kwargs",
    [{"method": "cubic"}, {"edge_method": "zero"}, {"window": 0}, {"window": True}],
)
def test_fill_validation(gappy, kwargs):
    with pytest.raises(ValueError):
        fill_missing(gappy, frequency="M", **kwargs)
