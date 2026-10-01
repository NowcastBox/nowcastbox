"""Tests of the internal helpers of nowcastbox.diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st
from statsmodels.stats.multitest import multipletests

from nowcastbox.core.data import MixedFrequencyData
from nowcastbox.core.exceptions import NowcastDataError
from nowcastbox.diagnostics import adjust_pvalues
from nowcastbox.diagnostics._common import (
    aggregate_factors,
    align_factors,
    check_alpha,
    check_correction,
    check_int,
    factor_block_map,
    pca_factors,
    series_design,
    series_weights,
    standardized,
    to_native,
)


@pytest.mark.reference_validation
@pytest.mark.parametrize("method", ["holm", "bonferroni", "fdr_bh"])
def test_adjust_pvalues_matches_statsmodels(method, rng):
    p = rng.uniform(0, 0.2, 25)
    expected = multipletests(p, method=method)[1]
    np.testing.assert_allclose(adjust_pvalues(p, method), expected, atol=1e-12)


def test_adjust_pvalues_nan_and_none():
    p = np.array([0.01, np.nan, 0.04])
    out = adjust_pvalues(p, "bonferroni")
    np.testing.assert_allclose(out, [0.02, np.nan, 0.08])
    np.testing.assert_array_equal(adjust_pvalues(p, "none"), p)
    assert np.isnan(adjust_pvalues([np.nan], "holm")).all()
    with pytest.raises(ValueError, match="correction"):
        adjust_pvalues(p, "sidak")


@pytest.mark.property
@given(
    st.lists(st.floats(0.0, 1.0, allow_nan=False), min_size=1, max_size=30),
    st.sampled_from(["holm", "bonferroni", "fdr_bh"]),
)
def test_adjust_pvalues_properties(p, method):
    raw = np.asarray(p)
    adj = adjust_pvalues(raw, method)
    assert np.all(adj >= raw - 1e-15)
    assert np.all(adj <= 1.0)
    order = np.argsort(raw, kind="stable")
    assert np.all(np.diff(adj[order]) >= -1e-15)


def test_checks():
    assert check_alpha(0.1) == 0.1
    for bad in (0, 1.0, True, "0.1"):
        with pytest.raises(ValueError, match="alpha"):
            check_alpha(bad)
    assert check_int(np.int64(3), "k", 0) == 3
    for bad in (-1, 1.5, True):
        with pytest.raises(ValueError, match="k must"):
            check_int(bad, "k", 0)
    assert check_correction("none") == "none"


def _panel():
    idx = pd.period_range("2000-01", periods=12, freq="M")
    q = np.full(12, np.nan)
    q[2::3] = [1.0, 2.0, 3.0, 4.0]
    df = pd.DataFrame({"m": np.arange(12.0), "q": q, "qs": q}, index=idx)
    return MixedFrequencyData(df, {"m": "M", "q": "Q", "qs": "Q"}, aggregations={"qs": "stock"})


def test_series_weights_defaults_and_overrides():
    panel = _panel()
    w = series_weights(panel, ["m", "q", "qs"])
    np.testing.assert_allclose(w["m"], [1.0])
    np.testing.assert_allclose(w["q"], np.array([1, 2, 3, 2, 1]) / 3)
    np.testing.assert_allclose(w["qs"], [1.0, 0.0, 0.0])
    w2 = series_weights(panel, ["q"], {"q": [1, 1, 1]})
    np.testing.assert_allclose(w2["q"], [1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="weights of 'q'"):
        series_weights(panel, ["q"], {"q": []})
    with pytest.raises(ValueError, match="weights of 'q'"):
        series_weights(panel, ["q"], {"q": [np.nan]})


def test_aggregate_factors_exact():
    f = np.arange(1.0, 7.0)[:, None]
    out = aggregate_factors(f, np.array([1.0, 2.0]))
    assert np.isnan(out[0, 0])
    np.testing.assert_allclose(out[1:, 0], f[1:, 0] + 2 * f[:-1, 0])


def test_align_factors_errors():
    idx = pd.period_range("2000-01", periods=4, freq="M")
    f = pd.DataFrame({"f1": [1.0, 2, 3, 4]}, index=idx)
    out = align_factors(f, pd.period_range("2000-03", periods=4, freq="M"))
    assert out["f1"].isna().sum() == 2
    with pytest.raises(NowcastDataError, match="non-empty"):
        align_factors(f.iloc[:, :0], idx)
    with pytest.raises(NowcastDataError, match="non-empty"):
        align_factors(f["f1"], idx)  # type: ignore[arg-type]
    with pytest.raises(NowcastDataError, match="PeriodIndex"):
        align_factors(f.reset_index(drop=True), idx)
    with pytest.raises(NowcastDataError, match="frequency"):
        align_factors(f, pd.period_range("2000Q1", periods=4, freq="Q"))
    with pytest.raises(NowcastDataError, match="overlap"):
        align_factors(f, pd.period_range("2010-01", periods=4, freq="M"))


def test_standardized():
    z = standardized(np.array([1.0, 2.0, np.nan, 3.0]))
    np.testing.assert_allclose(z[[0, 1, 3]], [-1.0, 0.0, 1.0])
    np.testing.assert_allclose(standardized(np.array([2.0, 2.0])), [0.0, 0.0])
    np.testing.assert_array_equal(standardized(np.array([5.0, np.nan])), [5.0, np.nan])


def test_series_design_and_to_native():
    panel = _panel()
    f = pd.DataFrame({"f1": np.arange(12.0)}, index=panel.index)
    idx, y, X = series_design(panel, f, "q", np.array([1.0, 1.0, 1.0]))
    assert idx.astype(str).tolist() == ["2000-03", "2000-06", "2000-09", "2000-12"]
    np.testing.assert_allclose(X[:, 0], [3.0, 12.0, 21.0, 30.0])
    assert y.mean() == pytest.approx(0.0)
    s = to_native(pd.Series(np.arange(12.0), index=panel.index, name="q"), panel)
    assert str(s.index.freqstr) == "Q-DEC"
    assert s.tolist() == [2.0, 5.0, 8.0, 11.0]


def test_pca_factors(rng):
    n = 80
    f = rng.standard_normal(n)
    x = np.outer(f, [1.0, 0.9, 1.1, 0.8]) + 0.1 * rng.standard_normal((n, 4))
    idx = pd.period_range("2000-01", periods=n, freq="M")
    df = pd.DataFrame(x, index=idx, columns=list("abcd"))
    df.iloc[0, 0] = np.nan
    pf = pca_factors(MixedFrequencyData(df, "M"), 1)
    assert np.isnan(pf.iloc[0, 0])
    assert abs(np.corrcoef(pf.iloc[1:, 0], f[1:])[0, 1]) > 0.99
    assert np.corrcoef(pf.iloc[1:, 0], f[1:])[0, 1] > 0  # sign normalisation
    with pytest.raises(NowcastDataError, match="Principal components"):
        pca_factors(MixedFrequencyData(df, "M"), 5)


def test_factor_block_map():
    out = factor_block_map(["global_f1", "real_f1", "real_f2", "other"], ["global", "real"])
    assert out == {"global": ["global_f1"], "real": ["real_f1", "real_f2"], "all": ["other"]}
    assert factor_block_map(["f1", "f2"], []) == {"all": ["f1", "f2"]}
