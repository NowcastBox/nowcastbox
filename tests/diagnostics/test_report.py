"""Tests of run_diagnostics / DiagnosticsReport."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from nowcastbox.core.exceptions import ConvergenceWarning, DataQualityWarning, NowcastDataError
from nowcastbox.core.results import NowcastResults, build_nowcast_frame
from nowcastbox.diagnostics import (
    COMPONENTS,
    ConvergenceDiagnostics,
    DataQualityReport,
    DiagnosticsReport,
    FactorContribution,
    LoadingStabilityResult,
    ResidualDiagnostics,
    effective_loadings,
    idiosyncratic_residuals,
    run_diagnostics,
)
from nowcastbox.models import BridgeEquation, TwoStepDFM

FRAMES_EM = {
    "data_quality",
    "outliers",
    "publication_delays",
    "convergence_path",
    "convergence",
    "stability",
    "stability_multiple_testing",
    "contribution_shares",
    "contribution_r2",
    "contribution_block_r2",
    "contribution_by_factor",
    "contribution_by_block",
    "residuals",
    "bridge_residuals",
    "series_overview",
}


def _quiet(func, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return func(*args, **kwargs)


def test_em_report_components(em_results, example_data):
    rep = _quiet(run_diagnostics, em_results)
    assert isinstance(rep, DiagnosticsReport)
    assert isinstance(rep.data_quality, DataQualityReport)
    assert isinstance(rep.convergence, ConvergenceDiagnostics)
    assert isinstance(rep.stability, LoadingStabilityResult)
    assert isinstance(rep.contribution, FactorContribution)
    assert isinstance(rep.residuals, ResidualDiagnostics)
    assert rep.model_name == "MixedFreqDFM" and rep.target == "gdp"
    assert rep.notes == []
    assert set(rep.stability.table.index) == set(example_data.columns)
    assert set(rep.residuals.table["kind"]) == {"idiosyncratic"}
    frames = rep.to_frames()
    assert set(frames) == FRAMES_EM
    assert frames["bridge_residuals"].empty
    ov = rep.series_overview()
    assert ov.index.name == "series" and len(ov) == example_data.n_series
    assert {"na_share", "r2", "unstable_loadings", "lb_pvalue", "flags"} <= set(ov.columns)
    text = rep.summary()
    for header in ("Data quality", "EM convergence", "Loading stability", "Factor contribution"):
        assert header in text
    assert str(rep) == text


def test_em_quarterly_effective_loadings(em_results):
    eff = effective_loadings(em_results)
    ratio = eff.loc["gdp"] / em_results.loadings.loc["gdp"]
    np.testing.assert_allclose(ratio.to_numpy(), 9.0)
    np.testing.assert_allclose(eff.loc["x1"], em_results.loadings.loc["x1"])
    rep = _quiet(run_diagnostics, em_results, components=["data_quality"])
    assert not bool(rep.data_quality.table.loc["gdp", "near_zero_loading"])
    assert effective_loadings(em_results.replace(loadings=None)) is None
    params = dict(em_results.params)
    params["aggregation_weights"] = {"x1": [0.0, 1.0], "zz": [1.0, 2.0]}
    same = effective_loadings(em_results.replace(params=params))
    pd.testing.assert_frame_equal(same, em_results.loadings)


def test_two_step_report(two_step_results):
    rep = run_diagnostics(two_step_results, break_date="2008-01", correction="bonferroni")
    assert rep.convergence is None
    assert any("no log-likelihood path" in n for n in rep.notes)
    assert rep.stability.test == "known" and rep.stability.correction == "bonferroni"
    kinds = rep.residuals.table["kind"]
    assert kinds["gdp"] == "projection"
    assert kinds["x1"] == "idiosyncratic"
    assert kinds["gdp (bridge)"] == "bridge"
    assert len(rep.to_frames()["bridge_residuals"]) == 1
    assert "gdp (bridge)" not in rep.series_overview().index


def test_two_step_variables_note(example_data):
    res = TwoStepDFM(n_factors=1, aggregate="variables").fit(example_data, "gdp")
    rep = run_diagnostics(res, components=["stability", "residuals"])
    assert any("aggregate='variables'" in n for n in rep.notes)
    assert rep.contribution is None and rep.data_quality is None
    kinds = rep.residuals.table["kind"]
    assert set(kinds) == {"idiosyncratic", "projection", "bridge"}
    predictors = [c for c in res.model_data.columns if c != "gdp"]
    assert (kinds.reindex(predictors) == "idiosyncratic").all()
    resid, source = idiosyncratic_residuals(res)
    filtered = res.filter_panel(example_data).to_frame()
    name = predictors[0]
    slot = resid[name].index[0]
    expected = filtered.loc[slot, name] - res.x_forecast.loc[slot, name]
    assert resid[name].iloc[0] == pytest.approx(expected)
    assert source[name] == "model"


def test_block_model_contributions(em_block_results):
    rep = _quiet(run_diagnostics, em_block_results, components=["contribution"])
    assert rep.contribution.factor_blocks == {
        "global": ["global_f1"],
        "real": ["real_f1"],
        "nominal": ["nominal_f1"],
    }
    assert rep.contribution.by_block().loc["global", "n_series"] == 9
    assert set(rep.to_frames()) == {
        "contribution_shares",
        "contribution_r2",
        "contribution_block_r2",
        "contribution_by_factor",
        "contribution_by_block",
        "series_overview",
    }


def test_bridge_equation_results_without_factors():
    rng = np.random.default_rng(0)
    idx = pd.period_range("2010-01", periods=120, freq="M")
    x = rng.standard_normal(120)
    y = np.convolve(x, np.ones(3) / 3)[:120] + 0.2 * rng.standard_normal(120)
    y[np.arange(120) % 3 != 2] = np.nan
    df = pd.DataFrame({"x": x, "y": y}, index=idx)
    res = BridgeEquation(aggregation="average").fit(df, "y", frequency={"x": "M", "y": "Q"})
    rep = run_diagnostics(res)
    assert rep.stability is None and rep.contribution is None
    assert rep.residuals is not None
    assert rep.residuals.table.index.tolist() == ["y (bridge)"]
    assert any("only the bridge residuals" in n for n in rep.notes)
    assert "Notes:" in rep.summary()


def test_results_without_factors_or_bridge(example_data):
    idx = pd.period_range("2000Q1", periods=4, freq="Q")
    frame = build_nowcast_frame(pd.Series([1.0, 2, 3, np.nan], index=idx), pd.Series(1.0, idx))
    res = NowcastResults(target="gdp", nowcast=frame, data=example_data)
    rep = run_diagnostics(res)
    assert rep.residuals is None and rep.stability is None
    assert any(n.endswith("contain no factors.") for n in rep.notes)
    empty = run_diagnostics(res, components=["stability"])
    assert "contribution_r2" not in run_diagnostics(res, components=["residuals"]).to_frames()
    assert empty.series_overview().empty
    assert empty.flagged_series == []
    with pytest.raises(NowcastDataError, match="no factors"):
        idiosyncratic_residuals(res)


def test_warnings_and_flags(em_results):
    path = np.array([-1200.0, -1100.0, -1150.0])
    fake = em_results.replace(loglikelihood_path=path, converged=False)
    with pytest.warns(ConvergenceWarning):
        rep = run_diagnostics(fake, components=["convergence"])
    assert rep.convergence.n_decreases == 1
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        run_diagnostics(fake, components=["convergence"], warn=False)
    loadings = em_results.loadings.copy()
    loadings.loc["x3"] = 0.0
    weak = em_results.replace(loadings=loadings)
    with pytest.warns(DataQualityWarning, match="x3"):
        rep2 = run_diagnostics(weak, components=["data_quality"])
    assert "x3" in rep2.flagged_series
    assert "near_zero_loading" in rep2.series_overview().loc["x3", "flags"]


def test_flags_combine_components(two_step_results, example_data):
    df = example_data.to_frame()
    rng = np.random.default_rng(3)
    e = np.zeros(len(df))
    for t in range(1, len(df)):
        e[t] = 0.95 * e[t - 1] + rng.standard_normal()
    df["x2"] = df["x2"] + 3.0 * e  # strongly autocorrelated idiosyncratic component
    data = example_data.with_data(df)
    rep = run_diagnostics(two_step_results, data, alpha=0.05, correction="none")
    ov = rep.series_overview()
    assert "autocorrelated_residuals" in ov.loc["x2", "flags"]
    assert "x2" in rep.flagged_series
    if bool(ov.loc["x2", "unstable_loadings"]):
        assert "unstable_loadings" in ov.loc["x2", "flags"]
    many = rep.flagged_series
    assert "Flagged series" in rep.summary() and str(len(many)) in rep.summary()


def test_unstable_loadings_flag(two_step_results, example_data):
    df = example_data.to_frame()
    f = two_step_results.factors.reindex(df.index).to_numpy()
    df.loc[df.index[90:], "x4"] = df["x4"].iloc[90:] + 4.0 * f[90:, 0]
    data = example_data.with_data(df)
    rep = run_diagnostics(two_step_results, data, break_date=data.index[90], warn=False)
    assert bool(rep.stability.table.loc["x4", "reject"])
    assert "unstable_loadings" in rep.series_overview().loc["x4", "flags"]


def test_data_argument_and_errors(two_step_results, example_data):
    rep = run_diagnostics(two_step_results, example_data.to_frame(), components=["data_quality"])
    assert rep.data_quality.table.loc["gdp", "frequency"] == "Q"
    with pytest.raises(TypeError, match="NowcastResults"):
        run_diagnostics("not results")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown components"):
        run_diagnostics(two_step_results, components=["foo"])
    no_data = two_step_results.replace(data=None)
    with pytest.raises(NowcastDataError, match="Pass data"):
        run_diagnostics(no_data)
    rep2 = run_diagnostics(no_data, example_data.select(["x1", "x2", "gdp"]))
    assert rep2.stability.table.index.tolist() == ["x1", "x2", "gdp"]
    assert set(COMPONENTS) == {
        "data_quality",
        "convergence",
        "stability",
        "contribution",
        "residuals",
    }


def test_idiosyncratic_residuals_sources(em_results, two_step_results, example_data):
    resid, source = idiosyncratic_residuals(em_results)
    assert set(source.values()) == {"model"}
    common = em_results.common_component.reindex(example_data.index)
    expected = (example_data.to_frame()["x1"] - common["x1"]).dropna()
    np.testing.assert_allclose(resid["x1"].to_numpy(), expected.to_numpy())
    assert resid["x1"].index.equals(expected.index)
    resid2, source2 = idiosyncratic_residuals(two_step_results, example_data.to_frame())
    assert source2["gdp"] == "projection" and source2["x1"] == "model"
    assert list(resid2) == list(example_data.columns)


def test_filtered_residuals_fall_back_when_filter_fails(example_data, monkeypatch):
    res = TwoStepDFM(n_factors=1, aggregate="variables").fit(example_data, "gdp")

    def broken(self, data, **kwargs):
        raise NowcastDataError("no filter")

    monkeypatch.setattr(type(res), "filter_panel", broken)
    _, source = idiosyncratic_residuals(res)
    assert set(source.values()) == {"projection"}
