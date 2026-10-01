"""Tests of the residual diagnostics (Ljung-Box, Jarque-Bera)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.stats
from statsmodels.stats.diagnostic import acorr_ljungbox

from nowcastbox.diagnostics import (
    ResidualDiagnostics,
    jarque_bera,
    ljung_box,
    residual_diagnostics,
)
from nowcastbox.diagnostics.residuals import autocorrelation


def _ar1(rng, n, phi):
    e = np.zeros(n)
    for t in range(1, n):
        e[t] = phi * e[t - 1] + rng.standard_normal()
    return e


@pytest.mark.reference_validation
@pytest.mark.parametrize(("lags", "model_df"), [(1, 0), (10, 0), (12, 2)])
def test_ljung_box_matches_statsmodels(rng, lags, model_df):
    x = _ar1(rng, 150, 0.3)
    stat, p, h = ljung_box(x, lags, model_df=model_df)
    ref = acorr_ljungbox(x, lags=[lags], model_df=model_df)
    assert h == lags
    assert stat == pytest.approx(float(ref["lb_stat"].iloc[0]), rel=1e-10)
    assert p == pytest.approx(float(ref["lb_pvalue"].iloc[0]), rel=1e-8)


@pytest.mark.reference_validation
def test_jarque_bera_matches_scipy(rng):
    x = rng.standard_t(5, 300)
    stat, p, skew, kurt = jarque_bera(x)
    ref = scipy.stats.jarque_bera(x)
    assert stat == pytest.approx(ref.statistic, rel=1e-10)
    assert p == pytest.approx(ref.pvalue, rel=1e-8)
    assert skew == pytest.approx(scipy.stats.skew(x), rel=1e-10)
    assert kurt == pytest.approx(scipy.stats.kurtosis(x, fisher=False), rel=1e-10)


def test_missing_values_are_handled_pairwise(rng):
    x = rng.standard_normal(100)
    x_nan = x.copy()
    x_nan[[10, 50]] = np.nan
    rho = autocorrelation(x_nan, 2)
    d = np.where(np.isnan(x_nan), 0.0, x_nan - np.nanmean(x_nan))
    assert rho[0] == pytest.approx(float(d[1:] @ d[:-1] / (d @ d)))
    stat, _, h = ljung_box(x_nan)
    assert h == 10 and np.isfinite(stat)
    assert jarque_bera(x_nan)[0] == pytest.approx(scipy.stats.jarque_bera(x[~np.isnan(x_nan)])[0])
    assert np.isnan(autocorrelation(np.ones(5), 1)[0])


def test_default_lags_and_errors():
    assert ljung_box(np.random.default_rng(0).standard_normal(20))[2] == 4
    assert ljung_box(np.random.default_rng(0).standard_normal(6))[2] == 1
    with pytest.raises(ValueError, match="exceed model_df"):
        ljung_box(np.arange(50.0), 2, model_df=2)
    with pytest.raises(ValueError, match="observations"):
        ljung_box(np.arange(5.0), 4)
    with pytest.raises(ValueError, match="infinite"):
        ljung_box(np.array([1.0, np.inf, 2.0]))
    with pytest.raises(ValueError, match="at least 3"):
        jarque_bera([1.0, 2.0])
    with pytest.raises(ValueError, match="constant"):
        jarque_bera(np.ones(10))
    with pytest.raises(ValueError, match="lags"):
        ljung_box(np.arange(50.0), 0)


def test_ljung_box_size_and_power():
    size = [
        ljung_box(np.random.default_rng(s).standard_normal(200), 10)[1] < 0.05 for s in range(300)
    ]
    assert 0.02 <= np.mean(size) <= 0.085
    power = [ljung_box(_ar1(np.random.default_rng(s), 200, 0.4), 10)[1] < 0.05 for s in range(50)]
    assert np.mean(power) >= 0.95


def test_jarque_bera_size_and_power():
    size = [
        jarque_bera(np.random.default_rng(s).standard_normal(500))[1] < 0.05 for s in range(300)
    ]
    assert 0.02 <= np.mean(size) <= 0.085
    power = [jarque_bera(np.random.default_rng(s).standard_t(4, 500))[1] < 0.05 for s in range(50)]
    assert np.mean(power) >= 0.9


def test_residual_diagnostics_inputs(rng):
    e = pd.DataFrame({"wn": rng.standard_normal(200), "ar": _ar1(rng, 200, 0.8)})
    rd = residual_diagnostics(e, lags=8, kind={"ar": "bridge"})
    assert isinstance(rd, ResidualDiagnostics)
    assert rd.table.columns.tolist() == [
        "kind",
        "n_obs",
        "mean",
        "std",
        "lb_lags",
        "lb_stat",
        "lb_pvalue",
        "lb_reject",
        "skewness",
        "kurtosis",
        "jb_stat",
        "jb_pvalue",
        "jb_reject",
        "note",
    ]
    assert rd.table.loc["wn", "kind"] == "residual" and rd.table.loc["ar", "kind"] == "bridge"
    assert bool(rd.table.loc["ar", "lb_reject"]) and rd.table.loc["ar", "lb_lags"] == 8
    s = residual_diagnostics(pd.Series(rng.standard_normal(50)))
    assert s.table.index.tolist() == ["residuals"]
    m = residual_diagnostics({"a": rng.standard_normal(40)}, kind="idiosyncratic")
    assert m.table.loc["a", "kind"] == "idiosyncratic"
    text = rd.summary()
    assert "Residual diagnostics" in text and "autocorrelated: ar" in text
    copy = rd.to_frame()
    copy.loc["wn", "n_obs"] = 0
    assert rd.table.loc["wn", "n_obs"] == 200


def test_residual_diagnostics_degenerate_and_errors(rng):
    rd = residual_diagnostics({"short": [1.0, 2.0], "const": np.ones(30), "empty": [np.nan]})
    t = rd.table
    assert "Ljung-Box" in t.loc["short", "note"] and "Jarque-Bera" in t.loc["short", "note"]
    assert np.isnan(t.loc["const", "jb_pvalue"]) and "constant" in t.loc["const", "note"]
    assert not bool(t.loc["const", "lb_reject"])
    assert np.isnan(t.loc["empty", "mean"])
    assert not np.isnan(t.loc["short", "std"])
    many = residual_diagnostics(
        {f"s{i}": _ar1(np.random.default_rng(i), 200, 0.9) for i in range(12)}
    )
    assert "..." in many.summary()
    with pytest.raises(TypeError, match="residuals must be"):
        residual_diagnostics([1.0, 2.0])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="No residual"):
        residual_diagnostics({})
    with pytest.raises(ValueError, match="lags"):
        residual_diagnostics({"a": rng.standard_normal(30)}, lags=0)
    with pytest.raises(ValueError, match="alpha"):
        residual_diagnostics({"a": rng.standard_normal(30)}, alpha=0)
