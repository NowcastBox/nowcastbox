"""Tests of the factor-contribution diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from nowcastbox.core.data import MixedFrequencyData
from nowcastbox.core.exceptions import NowcastDataError
from nowcastbox.diagnostics import (
    FactorContribution,
    factor_contributions,
    projection_residuals,
)


def _orthogonal_panel(rng, n=4000):
    f = rng.standard_normal((n, 2))
    idx = pd.period_range("1700-01", periods=n, freq="M")
    x = pd.DataFrame(
        {
            "a": 2.0 * f[:, 0] + 1.0 * f[:, 1] + rng.standard_normal(n),
            "b": 0.5 * f[:, 1] + rng.standard_normal(n),
            "noise": rng.standard_normal(n),
        },
        index=idx,
    )
    return x, pd.DataFrame(f, index=idx, columns=["f1", "f2"])


def test_variance_shares_analytical(rng):
    x, f = _orthogonal_panel(rng)
    fc = factor_contributions(x, f, frequency="M")
    shares = fc.share_matrix()
    # Var(a) = 4 + 1 + 1 = 6: shares 4/6 and 1/6; Var(b) = 1.25: shares 0, 0.25/1.25
    np.testing.assert_allclose(shares.loc["a"], [4 / 6, 1 / 6], atol=0.03)
    np.testing.assert_allclose(shares.loc["b"], [0.0, 0.2], atol=0.03)
    np.testing.assert_allclose(shares.loc["noise"], [0.0, 0.0], atol=0.01)
    np.testing.assert_allclose(shares.sum(axis=1), fc.r2["r2"], atol=1e-10)
    assert fc.r2.loc["a", "r2"] == pytest.approx(5 / 6, abs=0.02)


@pytest.mark.reference_validation
def test_r2_matches_statsmodels_ols(rng):
    n = 200
    f = rng.standard_normal((n, 2))
    f[:, 1] += 0.6 * f[:, 0]  # correlated factors
    idx = pd.period_range("2000-01", periods=n, freq="M")
    y = f @ [1.0, -0.5] + rng.standard_normal(n)
    fc = factor_contributions(
        pd.DataFrame({"y": y}, index=idx), pd.DataFrame(f, index=idx), frequency="M"
    )
    ols = sm.OLS(y, sm.add_constant(f)).fit()
    assert fc.r2.loc["y", "r2"] == pytest.approx(ols.rsquared, rel=1e-10)
    assert fc.r2.loc["y", "adj_r2"] == pytest.approx(ols.rsquared_adj, rel=1e-10)
    loadings = fc.shares.set_index("factor")["loading"]
    sd_y = y.std(ddof=1)
    np.testing.assert_allclose(loadings.to_numpy(), ols.params[1:] / sd_y, rtol=1e-10)
    assert fc.share_matrix().sum(axis=1).iloc[0] == pytest.approx(ols.rsquared)


def test_blocks_from_model(em_block_results, block_data):
    fc = factor_contributions(block_data, em_block_results.factors)
    assert fc.factor_blocks == {
        "global": ["global_f1"],
        "real": ["real_f1"],
        "nominal": ["nominal_f1"],
    }
    br = fc.block_r2
    assert set(br["block"]) == {"global", "real", "nominal"}
    assert bool(br.loc[(br.series == "r0") & (br.block == "nominal"), "member"].iloc[0]) is False
    by_block = fc.by_block()
    assert by_block.loc["nominal", "n_series"] == 4
    assert by_block.loc["global", "n_series"] == 9
    assert (br["r2"] <= fc.r2.loc[br["series"], "r2"].to_numpy() + 1e-12).all()
    by_factor = fc.by_factor()
    assert by_factor.loc["real_f1", "block"] == "real"
    assert "Block nominal" in fc.summary()


def test_explicit_factor_blocks_and_validation(rng):
    x, f = _orthogonal_panel(rng, n=300)
    fc = factor_contributions(x, f, factor_blocks={"one": ["f1"], "two": ["f2"]}, frequency="M")
    assert fc.block_r2["member"].all()
    a = fc.block_r2.set_index(["series", "block"])["r2"]
    assert a[("a", "one")] > a[("a", "two")]
    for bad in ({"one": ["zz"]}, {}, {"one": []}):
        with pytest.raises(ValueError, match="factor_blocks"):
            factor_contributions(x, f, factor_blocks=bad, frequency="M")
    with pytest.raises(NowcastDataError, match="Unknown series"):
        factor_contributions(x, f, series=["zz"], frequency="M")
    only = factor_contributions(x, f, series=["b"], frequency="M")
    assert only.r2.index.tolist() == ["b"]
    assert not np.isnan(only.by_block().loc["all", "mean_r2"])


def test_by_block_without_members(rng):
    x, f = _orthogonal_panel(rng, n=100)
    panel = MixedFrequencyData(x, "M", blocks={"a": "real", "b": "real", "noise": "real"})
    fc = factor_contributions(panel, f.rename(columns={"f1": "real_f1", "f2": "nom_f1"}))
    assert fc.factor_blocks == {"real": ["real_f1"], "all": ["nom_f1"]}
    out = factor_contributions(panel, f, factor_blocks={"real": ["f1"], "ghost": ["f2"]}).by_block()
    assert out.loc["ghost", "n_series"] == 3
    fc2 = factor_contributions(
        panel.select(["a"]).with_metadata("a", blocks=("other",)),
        f,
        factor_blocks={"real": ["f1"], "other": ["f2"]},
    )
    bb = fc2.by_block()
    assert bb.loc["real", "n_series"] == 1  # block missing from metadata -> everyone
    assert bb.loc["other", "n_series"] == 1


def test_empty_block_membership(rng):
    x, f = _orthogonal_panel(rng, n=100)
    panel = MixedFrequencyData(x, "M", blocks={"a": "real", "b": "nominal", "noise": "real"})
    fc = factor_contributions(panel, f, factor_blocks={"nominal": ["f1"], "real": ["f2"]})
    sub = FactorContribution(
        shares=fc.shares,
        r2=fc.r2,
        block_r2=fc.block_r2.assign(member=False),
        factor_blocks=fc.factor_blocks,
    )
    bb = sub.by_block()
    assert (bb["n_series"] == 0).all() and bb["mean_r2"].isna().all()


def test_quarterly_and_degenerate_series(example_data, two_step_results):
    fc = factor_contributions(example_data, two_step_results.factors)
    assert fc.r2.loc["gdp", "frequency"] == "Q"
    assert fc.r2.loc["gdp", "r2"] > 0.8
    idx = example_data.index
    df = pd.DataFrame({"const": 1.0, "few": np.nan}, index=idx)
    df.iloc[:2, 1] = [1.0, 2.0]
    fc2 = factor_contributions(df, two_step_results.factors, frequency="M")
    assert np.isnan(fc2.r2.loc["const", "r2"])
    assert np.isnan(fc2.r2.loc["few", "r2"])
    assert fc2.shares["share"].isna().all()


def test_summary_and_projection_residuals(example_data, two_step_results):
    fc = factor_contributions(example_data, two_step_results.factors)
    text = fc.summary(n_series=2)
    assert "Mean R2" in text and "Share f1" in text and "Block" not in text
    res = projection_residuals(example_data, two_step_results.factors)
    assert set(res) == set(example_data.columns)
    gdp = res["gdp"]
    assert str(gdp.index.freqstr) == "Q-DEC"
    assert gdp.mean() == pytest.approx(0.0, abs=1e-10)
    # residuals are orthogonal to the regressors of the projection
    x1 = res["x1"]
    f = two_step_results.factors.reindex(x1.index.asfreq("M"))
    np.testing.assert_allclose(f.to_numpy().T @ x1.to_numpy(), 0.0, atol=1e-8)
