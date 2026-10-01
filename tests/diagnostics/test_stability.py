"""Tests of the Breitung-Eickmeier loading-stability tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.stats
from hypothesis import given, settings
from hypothesis import strategies as st

from nowcastbox.core.data import MixedFrequencyData
from nowcastbox.core.exceptions import NowcastDataError
from nowcastbox.diagnostics import (
    LoadingStabilityResult,
    andrews_critical_values,
    loading_stability_test,
    sup_break_pvalue,
)
from nowcastbox.diagnostics.stability import break_ssr
from tests.diagnostics.conftest import simulate_factor_panel


# ---------------------------------------------------------------- regressions
def _brute_force_ssr(y, W, Fb, k):
    d = (np.arange(y.size) >= k).astype(float)[:, None]
    X = np.column_stack([W, Fb * d])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    return float(e @ e)


@pytest.mark.parametrize("r", [1, 2, 3])
def test_break_ssr_matches_brute_force(rng, r):
    n = 60
    F = rng.standard_normal((n, r))
    W = np.column_stack([np.ones(n), F, rng.standard_normal(n)])
    y = W @ rng.standard_normal(W.shape[1]) + rng.standard_normal(n)
    starts = np.arange(10, 50, 7)
    ssr_r, ssr_u = break_ssr(y, W, F, starts)
    beta, *_ = np.linalg.lstsq(W, y, rcond=None)
    assert ssr_r == pytest.approx(float(((y - W @ beta) ** 2).sum()))
    for k, s in zip(starts, ssr_u, strict=True):
        assert s == pytest.approx(_brute_force_ssr(y, W, F, k), rel=1e-9)


def test_lm_equals_n_r2_of_auxiliary_regression(rng):
    """LM = n R^2 of the regression of restricted residuals on (1, F, F d)."""
    n = 120
    f = rng.standard_normal(n)
    idx = pd.period_range("2000-01", periods=n, freq="M")
    x = 0.8 * f + rng.standard_normal(n)
    x[60:] += 0.4 * f[60:]
    data = pd.DataFrame({"x": x}, index=idx)
    factors = pd.DataFrame({"f1": f}, index=idx)
    res = loading_stability_test(data, factors, break_date="2005-01", frequency="M")
    z = (x - x.mean()) / x.std(ddof=1)
    W = np.column_stack([np.ones(n), f])
    e = z - W @ np.linalg.lstsq(W, z, rcond=None)[0]
    X = np.column_stack([W, f * (np.arange(n) >= 60)])
    fit = X @ np.linalg.lstsq(X, e, rcond=None)[0]
    r2 = 1 - ((e - fit) ** 2).sum() / ((e - e.mean()) ** 2).sum()
    assert res.table.loc["x", "lm_stat"] == pytest.approx(n * r2, rel=1e-8)
    assert res.table.loc["x", "lm_pvalue"] == pytest.approx(
        scipy.stats.chi2.sf(n * r2, 1), rel=1e-6
    )
    assert res.table.loc["x", "break_period"] == "2005-01"


@pytest.mark.property
@settings(max_examples=30, deadline=None)
@given(st.integers(0, 10_000), st.integers(1, 2), st.integers(0, 2))
def test_wald_lr_lm_ordering(seed, r, ar_lags):
    data, f = simulate_factor_panel(seed, n_periods=60, n_series=3, n_factors=r, n_break=1)
    factors = pd.DataFrame(f, index=data.index, columns=[f"f{k + 1}" for k in range(r)])
    for date in ("2002-06", None):
        t = loading_stability_test(
            data, factors, break_date=date, ar_lags=ar_lags, frequency="M"
        ).table
        assert np.all(t["wald_stat"] >= t["lr_stat"] - 1e-9)
        assert np.all(t["lr_stat"] >= t["lm_stat"] - 1e-9)
        assert np.all(t["lm_stat"] >= -1e-12)


# ---------------------------------------------------------------- null distribution
@pytest.mark.reference_validation
@pytest.mark.parametrize(
    ("k", "table"),
    [(1, (7.17, 8.68, 12.16)), (2, (10.01, 11.72, 15.56))],
)
def test_andrews_critical_values_match_table(k, table):
    """Andrews (1993, table 1), pi0 = 0.15 (discretisation lowers them slightly)."""
    cv = andrews_critical_values(k, 0.15)
    np.testing.assert_allclose(cv.to_numpy(), table, atol=0.35)
    assert cv.index.tolist() == [0.10, 0.05, 0.01]


def test_sup_pvalue_properties():
    p = sup_break_pvalue([0.0, 5.0, 8.5, 20.0, 1e6, np.nan], 1)
    assert p[0] == pytest.approx(1.0)
    assert np.all(np.diff(p[:5]) <= 0)
    assert p[2] == pytest.approx(0.05, abs=0.01)
    assert p[4] == pytest.approx(1 / (1 + 20_000))
    assert np.isnan(p[5])
    cv = andrews_critical_values(1, levels=[0.05])
    assert float(sup_break_pvalue(cv.iloc[0], 1)) == pytest.approx(0.05, abs=0.002)
    # less trimming -> larger critical values
    assert andrews_critical_values(1, 0.05).iloc[1] > andrews_critical_values(1, 0.15).iloc[1]


def test_null_distribution_arguments():
    for bad in (0.0, 0.5, True, "x"):
        with pytest.raises(ValueError, match="trim"):
            sup_break_pvalue(1.0, 1, bad)
    with pytest.raises(ValueError, match="n_breaking"):
        andrews_critical_values(0)
    with pytest.raises(TypeError, match="Unknown"):
        andrews_critical_values(1, foo=1)
    small = andrews_critical_values(1, n_sim=500, n_grid=100, random_state=1)
    assert small.size == 3


# ---------------------------------------------------------------- size and power
@pytest.mark.parametrize("statistic", ["lm", "wald", "lr"])
def test_size_known_break(statistic):
    rates = []
    for seed in range(40):
        data, _ = simulate_factor_panel(seed, n_series=20)
        res = loading_stability_test(data, n_factors=1, break_date="2008-05", frequency="M")
        rates.append(float((res.table[f"{statistic}_pvalue"] < 0.05).mean()))
    assert 0.025 <= np.mean(rates) <= 0.085


def test_size_sup_break():
    rates = {"lm": [], "wald": [], "lr": []}
    for seed in range(25):
        data, _ = simulate_factor_panel(500 + seed, n_series=20)
        t = loading_stability_test(data, n_factors=1, frequency="M").table
        for s in rates:
            rates[s].append(float((t[f"{s}_pvalue"] < 0.05).mean()))
    assert 0.02 <= np.mean(rates["lm"]) <= 0.085
    assert 0.02 <= np.mean(rates["lr"]) <= 0.10
    assert 0.02 <= np.mean(rates["wald"]) <= 0.12


@pytest.mark.parametrize("break_date", ["2008-05", None])
def test_power_against_loading_breaks(break_date):
    power, others = [], []
    for seed in range(10):
        data, _ = simulate_factor_panel(300 + seed, n_break=4, delta=0.75)
        res = loading_stability_test(data, n_factors=1, break_date=break_date, frequency="M")
        p = res.table["lm_pvalue"].to_numpy()
        power.append(float((p[:4] < 0.05).mean()))
        others.append(float((p[4:] < 0.05).mean()))
    assert np.mean(power) >= 0.9
    assert np.mean(others) <= 0.12


def test_sup_test_locates_break():
    data, _ = simulate_factor_panel(7, n_break=3, delta=1.5, break_at=120)
    res = loading_stability_test(data, n_factors=1, frequency="M")
    dates = pd.PeriodIndex(res.table["break_period"].iloc[:3].tolist(), freq="M")
    true = data.index[120]
    gaps = sorted(abs((d - true).n) for d in dates)
    assert gaps[1] <= 6  # median distance
    assert gaps[-1] <= 18
    assert set(res.rejected) >= {"x0", "x1", "x2"}


def test_ar_lags_correct_size_with_autocorrelated_errors():
    rates = {0: [], 1: []}
    for seed in range(15):
        data, _ = simulate_factor_panel(200 + seed, rho_e=0.6)
        for lags in rates:
            res = loading_stability_test(
                data, n_factors=1, break_date="2008-05", ar_lags=lags, frequency="M"
            )
            rates[lags].append(float((res.table["lm_pvalue"] < 0.05).mean()))
    assert np.mean(rates[0]) > 0.1
    assert np.mean(rates[1]) <= 0.085


def test_quarterly_series_with_aggregated_factors(rng):
    n = 240
    f = np.zeros(n)
    for t in range(1, n):
        f[t] = 0.7 * f[t - 1] + rng.standard_normal()
    idx = pd.period_range("2000-01", periods=n, freq="M")
    agg = np.convolve(f, [1, 2, 3, 2, 1])[:n] / 3
    lam = np.where(np.arange(n) >= 120, 2.0, 0.5)
    q = lam * agg + 0.3 * rng.standard_normal(n)
    q[np.arange(n) % 3 != 2] = np.nan
    qs = 0.5 * agg + 0.3 * rng.standard_normal(n)
    qs[np.arange(n) % 3 != 2] = np.nan
    data = MixedFrequencyData(pd.DataFrame({"q": q, "stable": qs}, index=idx), "Q")
    factors = pd.DataFrame({"f1": f}, index=idx)
    res = loading_stability_test(data, factors, break_date="2010-01", alpha=0.01)
    assert res.table.loc["q", "frequency"] == "Q"
    assert res.table.loc["q", "n_obs"] == 79  # first quarter lacks the lagged factors
    assert res.table.loc["q", "break_period"] == "2010-03"
    assert bool(res.table.loc["q", "reject"])
    assert not bool(res.table.loc["stable", "reject"])


# ---------------------------------------------------------------- result object
def test_result_tables_and_summary():
    data, _ = simulate_factor_panel(11, n_series=12, n_break=2, delta=1.5)
    res = loading_stability_test(data, n_factors=1, frequency="M", correction="fdr_bh")
    assert isinstance(res, LoadingStabilityResult)
    assert res.table.columns.tolist() == [
        "frequency",
        "n_obs",
        "break_period",
        "lm_stat",
        "lm_pvalue",
        "wald_stat",
        "wald_pvalue",
        "lr_stat",
        "lr_pvalue",
        "df",
        "pvalue_adj",
        "reject",
        "note",
    ]
    mt = res.multiple_testing()
    assert mt.index.tolist() == ["lm", "wald", "lr"]
    assert mt.loc["lm", "n_tests"] == 12
    assert mt.loc["lm", "expected_reject"] == pytest.approx(0.6)
    k = int(mt.loc["lm", "n_reject"])
    assert mt.loc["lm", "binomial_pvalue"] == pytest.approx(scipy.stats.binom.sf(k - 1, 12, 0.05))
    assert mt.loc["lm", "n_reject_adjusted"] <= k
    text = res.summary()
    assert "Breitung & Eickmeier" in text and "x0" in text and "unknown break date" in text
    frame = res.to_frame()
    frame.iloc[0, 0] = "changed"
    assert res.table.iloc[0, 0] == "M"


def test_summary_truncates_long_rejection_lists():
    data, _ = simulate_factor_panel(3, n_series=30, n_break=14, delta=2.0)
    res = loading_stability_test(
        data, n_factors=1, break_date="2008-05", frequency="M", correction="none"
    )
    assert len(res.rejected) > 10
    assert "..." in res.summary()
    assert "known break date 2008-05" in res.summary()


def test_statistic_choice_and_correction_none():
    data, _ = simulate_factor_panel(4, n_series=10)
    res = loading_stability_test(
        data, n_factors=1, frequency="M", statistic="wald", correction="none"
    )
    np.testing.assert_allclose(res.table["pvalue_adj"], res.table["wald_pvalue"])


def test_break_date_formats():
    data, _ = simulate_factor_panel(1, n_series=3)
    for value in ("2008Q2", pd.Period("2008-04", freq="M"), pd.Timestamp("2008-04-15")):
        res = loading_stability_test(data, n_factors=1, break_date=value, frequency="M")
        assert res.break_date == pd.Period("2008-04", freq="M")
    for bad in ("1990-01", "2000-01", "2030-01"):
        with pytest.raises(ValueError, match="inside the sample"):
            loading_stability_test(data, n_factors=1, break_date=bad, frequency="M")


def test_notes_for_degenerate_series():
    idx = pd.period_range("2000-01", periods=40, freq="M")
    rng = np.random.default_rng(0)
    f = rng.standard_normal(40)
    short = np.full(40, np.nan)
    short[-4:] = rng.standard_normal(4)
    late = np.full(40, np.nan)
    late[:30] = f[:30] + rng.standard_normal(30)
    df = pd.DataFrame(
        {"short": short, "exact": 2.0 * f + 1.0, "late": late, "ok": f + rng.standard_normal(40)},
        index=idx,
    )
    factors = pd.DataFrame({"f1": f}, index=idx)
    known = loading_stability_test(df, factors, break_date="2002-06", frequency="M").table
    assert known.loc["short", "note"] == "too few observations"
    assert known.loc["exact", "note"] == "perfect fit"
    assert known.loc["late", "note"] == "break date too close to sample ends"
    assert np.isnan(known.loc["late", "lm_pvalue"])
    assert known.loc["ok", "note"] == ""
    sup = loading_stability_test(df, factors, trim=0.49, frequency="M").table
    assert sup.loc["ok", "note"] == ""
    tiny = loading_stability_test(df[["late"]].iloc[:9], factors, trim=0.45, frequency="M")
    assert tiny.table.loc["late", "note"] == "too few observations for trimming"
    lagged = loading_stability_test(df, factors, ar_lags=4, frequency="M").table
    assert lagged.loc["short", "n_obs"] == 0


def test_input_validation():
    data, _ = simulate_factor_panel(1, n_series=3)
    factors = pd.DataFrame({"f1": np.ones(len(data))}, index=data.index)
    with pytest.raises(ValueError, match="either factors or n_factors"):
        loading_stability_test(data, frequency="M")
    with pytest.raises(ValueError, match="not both"):
        loading_stability_test(data, factors, n_factors=1, frequency="M")
    with pytest.raises(ValueError, match="statistic"):
        loading_stability_test(data, n_factors=1, statistic="f", frequency="M")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="ar_lags"):
        loading_stability_test(data, n_factors=1, ar_lags=-1, frequency="M")
    with pytest.raises(ValueError, match="correction"):
        loading_stability_test(data, n_factors=1, correction="x", frequency="M")
    with pytest.raises(ValueError, match="alpha"):
        loading_stability_test(data, n_factors=1, alpha=2, frequency="M")
    with pytest.raises(NowcastDataError, match="Unknown series"):
        loading_stability_test(data, n_factors=1, series=["zz"], frequency="M")
    res = loading_stability_test(data, n_factors=1, series=["x1"], frequency="M")
    assert res.table.index.tolist() == ["x1"]
