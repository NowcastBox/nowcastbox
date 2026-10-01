"""Tests for nowcastbox.preprocessing.outliers."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from nowcastbox.core.data import MixedFrequencyData
from nowcastbox.core.exceptions import DataQualityWarning, NowcastDataError
from nowcastbox.preprocessing.outliers import (
    detect_outliers,
    iqr_outlier_mask,
    moving_median,
    replace_outliers,
)

IDX = pd.period_range("2010-01", periods=120, freq="M")


@pytest.fixture
def noisy(rng) -> pd.Series:
    x = pd.Series(rng.normal(size=120), index=IDX, name="x")
    x.iloc[[10, 60]] = [25.0, -30.0]
    return x


def test_iqr_mask_rule_definition(rng):
    x = rng.normal(size=200)
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    x[[3, 7]] = [med + 4.01 * (q3 - q1) + 100, med - 50]
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    expected = np.abs(x - med) > 4 * (q3 - q1)
    np.testing.assert_array_equal(iqr_outlier_mask(x, 4.0), expected)
    assert iqr_outlier_mask(x, 4.0)[[3, 7]].all()


def test_iqr_threshold_monotone(rng):
    x = rng.standard_t(2, size=300)
    n = [iqr_outlier_mask(x, k).sum() for k in (1.0, 2.0, 4.0, 8.0)]
    assert n == sorted(n, reverse=True)


def test_iqr_mask_edge_cases():
    assert not iqr_outlier_mask(np.array([1.0, 2.0, 100.0])).any()  # < 4 obs
    assert not iqr_outlier_mask(np.array([1.0, 1.0, 1.0, 1.0, 50.0])).any()  # IQR 0
    x = np.array([np.nan, 0.0, 1.0, 2.0, 3.0, 4.0, 100.0])
    mask = iqr_outlier_mask(x)
    assert not mask[0]
    assert mask[-1]


@pytest.mark.parametrize("bad", [0, -1, np.inf, "a", True, [1.0]])
def test_threshold_validation(bad):
    with pytest.raises(ValueError):
        iqr_outlier_mask(np.arange(10.0), bad)


@pytest.mark.parametrize("bad", [0, -1, 2.5, True])
def test_window_validation(bad):
    with pytest.raises(ValueError):
        moving_median(np.arange(10.0), bad)


def test_moving_median_centered():
    x = np.array([1.0, 5.0, 2.0, 8.0, 3.0])
    np.testing.assert_allclose(moving_median(x, 3), [3.0, 2.0, 5.0, 3.0, 5.5])
    np.testing.assert_allclose(moving_median(x, 1), x)
    assert np.isnan(moving_median(np.array([np.nan, np.nan]), 1)).all()


def test_detect_outliers_series(noisy):
    mask = detect_outliers(noisy)
    assert mask.dtype == bool
    assert mask.index.equals(noisy.index)
    assert set(np.flatnonzero(mask)) == {10, 60}


def test_replace_outliers_moving_median(noisy):
    out, mask = replace_outliers(noisy, return_mask=True)
    assert set(np.flatnonzero(mask)) == {10, 60}
    for i in (10, 60):
        expected = np.median([noisy.iloc[i - 1], noisy.iloc[i + 1]])
        assert out.iloc[i] == pytest.approx(expected)
    others = ~mask
    pd.testing.assert_series_equal(out[others], noisy[others])


def test_replace_outliers_other_replacements(noisy):
    out_nan = replace_outliers(noisy, replacement="nan")
    assert out_nan.isna().sum() == 2
    out_med = replace_outliers(noisy, replacement="median")
    clean_median = noisy.drop(noisy.index[[10, 60]]).median()
    assert out_med.iloc[10] == pytest.approx(clean_median)
    with pytest.raises(ValueError, match="replacement"):
        replace_outliers(noisy, replacement="mean")  # type: ignore[arg-type]


def test_replace_outliers_window_only_outliers_falls_back_to_median():
    x = pd.Series(
        [0.0, 1.0, -1.0, 0.5, 40.0, 50.0, 45.0, -0.5, 0.2, 0.1, 0.3, -0.2], index=IDX[:12]
    )
    out, mask = replace_outliers(x, threshold=2.0, window=1, return_mask=True)
    assert mask.iloc[4:7].all()
    assert np.isfinite(out).all()
    assert out.iloc[5] == pytest.approx(x[~mask].median())


def test_no_outliers_returns_identical(rng):
    x = pd.Series(rng.normal(size=50), index=IDX[:50])
    out, mask = replace_outliers(x, threshold=50, return_mask=True)
    assert not mask.any()
    pd.testing.assert_series_equal(out, x)


def test_missing_values_untouched(noisy):
    x = noisy.copy()
    x.iloc[[0, 5, -1]] = np.nan
    out = replace_outliers(x)
    assert out.isna().sum() == 3


def test_zero_iqr_warns():
    x = pd.Series([1.0] * 10 + [5.0], index=IDX[:11])
    with pytest.warns(DataQualityWarning, match="inter-quartile"):
        out = replace_outliers(x)
    pd.testing.assert_series_equal(out, x)


def test_quarterly_series_uses_native_neighbours(rng):
    idx = pd.period_range("2000-01", periods=120, freq="M")
    gdp = pd.Series(np.nan, index=idx)
    q = rng.normal(size=40)
    q[20] = 50.0
    gdp.iloc[2::3] = q
    out = replace_outliers(gdp, frequency="Q")
    pos = 2 + 3 * 20
    assert out.iloc[pos] == pytest.approx(np.median([q[19], q[21]]))
    assert out.iloc[[0, 1, 3, 4]].isna().all()  # structural NaNs preserved


def test_panel_input(rng, ragged_levels):
    df = ragged_levels.copy()
    d = df.diff()
    d["gdp"] = df["gdp"].dropna().diff().reindex(df.index)
    d.iloc[30, 0] = 100.0
    freqs = {"ip": "M", "sales": "M", "gdp": "Q"}
    out, mask = replace_outliers(d, frequency=freqs, return_mask=True)
    assert isinstance(out, pd.DataFrame)
    assert mask.loc[d.index[30], "ip"]
    assert mask.to_numpy().sum() >= 1
    assert out.iloc[30, 0] != 100.0
    detected = detect_outliers(d, frequency=freqs)
    pd.testing.assert_frame_equal(detected, mask)
    # panel in, panel out
    mfd = MixedFrequencyData(d, freqs)
    out_mfd = replace_outliers(mfd)
    assert isinstance(out_mfd, MixedFrequencyData)
    np.testing.assert_allclose(out_mfd.to_frame(), out, equal_nan=True)
    # restrict to some columns
    only_sales = replace_outliers(d, frequency=freqs, columns=["sales"])
    assert only_sales.iloc[30, 0] == 100.0
    with pytest.raises(NowcastDataError, match="Unknown"):
        replace_outliers(d, frequency=freqs, columns=["nope"])


def test_mask_without_outliers_in_panel(levels):
    freqs = {"ip": "M", "sales": "M", "gdp": "Q"}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out, mask = replace_outliers(levels, threshold=100, frequency=freqs, return_mask=True)
    assert not mask.to_numpy().any()
    pd.testing.assert_frame_equal(out, levels, check_names=False)
