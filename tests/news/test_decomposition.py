"""Tests of :func:`nowcastbox.news.news_decomposition`."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from nowcastbox.core.data import MixedFrequencyData
from nowcastbox.core.exceptions import DataQualityWarning, NowcastDataError
from nowcastbox.core.results import NowcastResults, build_nowcast_frame
from nowcastbox.models import MixedFreqDFM, TwoStepDFM
from nowcastbox.news import NewsResults, news_decomposition, news_weights
from nowcastbox.news._model import linear_model, to_frame, to_period
from nowcastbox.news.decomposition import _default_period
from tests.news.conftest import simulate_panel, vintage_pair

ATOL = 1e-8


# ====================================================================== brute force
def joint_moments(lin, n_periods):
    """Mean and covariance of (all observations, all states) by direct recursion."""
    ssm = lin.state_space
    m = ssm.n_states
    t_mat, z_mat = ssm.T, ssm.Z
    rqr = ssm.R @ ssm.Q @ ssm.R.T
    means = np.zeros((n_periods, m))
    covs = np.zeros((n_periods, m, m))
    means[0], covs[0] = ssm.a0, ssm.P0
    for t in range(1, n_periods):
        means[t] = t_mat @ means[t - 1] + ssm.c
        covs[t] = t_mat @ covs[t - 1] @ t_mat.T + rqr
    big = np.zeros((n_periods * m, n_periods * m))
    for s in range(n_periods):
        block = covs[s]
        for t in range(s, n_periods):
            big[t * m : (t + 1) * m, s * m : (s + 1) * m] = block
            big[s * m : (s + 1) * m, t * m : (t + 1) * m] = block.T
            block = t_mat @ block
    z_big = np.kron(np.eye(n_periods), z_mat)
    obs_mean = (means @ z_mat.T + ssm.d).ravel()
    h = np.diag(ssm.H) if ssm.H.ndim == 2 else ssm.H
    obs_cov = z_big @ big @ z_big.T + np.diag(np.tile(h, n_periods))
    state_obs = big @ z_big.T
    return means.ravel(), obs_mean, obs_cov, state_obs


def brute_force(lin, old_values, new_values, position):
    n_periods = old_values.shape[0]
    m = lin.state_space.n_states
    state_mean, obs_mean, obs_cov, state_obs = joint_moments(lin, n_periods)
    g = np.zeros(n_periods * m)
    g[position * m : (position + 1) * m] = lin.gain
    s_mean = g @ state_mean + lin.intercept
    s_obs = g @ state_obs
    old_mask = ~np.isnan(old_values.ravel())
    new_mask = ~np.isnan(new_values.ravel())
    rel = new_mask & ~old_mask
    o = np.flatnonzero(old_mask)
    j = np.flatnonzero(rel)
    y = new_values.ravel()
    soo = obs_cov[np.ix_(o, o)]
    sol = np.linalg.solve(soo, np.column_stack([y[o] - obs_mean[o], obs_cov[np.ix_(o, j)]]))
    old_nowcast = s_mean + s_obs[o] @ sol[:, 0]
    expected = obs_mean[j] + obs_cov[np.ix_(j, o)] @ sol[:, 0]
    cov_si = s_obs[j] - s_obs[o] @ sol[:, 1:]
    var_i = obs_cov[np.ix_(j, j)] - obs_cov[np.ix_(j, o)] @ sol[:, 1:]
    b = np.linalg.solve(var_i, cov_si)
    return old_nowcast, expected, b, j


@pytest.fixture(scope="module")
def small_em():
    panel = simulate_panel(n_periods=45, seed=5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = MixedFreqDFM(n_factors=1, max_iter=5).fit(panel, target="gdp")
    return panel, res


class TestBanburaModugnoFormula:
    def test_weights_equal_projection_coefficients(self, small_em):
        panel, res = small_em
        old, new = vintage_pair(panel, n_back=3)
        news = news_decomposition(res, old, new, "2008Q3")
        lin = linear_model(res)
        grid = news.info["grid"]
        position = grid.get_loc(pd.Period("2008-09", freq="M"))
        old_v = lin.standardize(to_frame(old), grid)
        new_v = lin.standardize(to_frame(new), grid)
        old_now, expected, b, cells = brute_force(lin, old_v, new_v, position)
        assert news.old_nowcast == pytest.approx(lin.to_original(old_now), abs=1e-8)
        n = len(lin.series)
        order = np.lexsort((cells // n, cells % n))
        rel = news.releases
        series_idx = rel["series"].map(lin.series.index).to_numpy()
        std = lin.std[series_idx]
        b_std = rel["weight"].to_numpy() * std / lin.scale
        exp_std = (rel["expected"].to_numpy() - lin.mean[series_idx]) / std
        assert len(rel) == len(cells)
        np.testing.assert_allclose(b_std, b[order], atol=1e-7)
        np.testing.assert_allclose(exp_std, expected[order], atol=1e-7)

    def test_news_identity_matches_projection(self, small_em):
        panel, res = small_em
        old, new = vintage_pair(panel, n_back=3)
        news = news_decomposition(res, old, new, "2008Q3")
        assert news.check_identity(ATOL)
        assert abs(news.residual) < 1e-10


# ====================================================================== identities
class TestIdentity:
    @pytest.mark.parametrize("fixture", ["em_results", "ts_results", "em_blocks_results"])
    def test_news_only(self, request, panel, fixture):
        res = request.getfixturevalue(fixture)
        old, new = vintage_pair(panel)
        news = news_decomposition(res, old, new, "2014Q4")
        assert news.n_releases > 0
        assert news.revisions_effect == 0.0
        total = news.old_nowcast + news.revisions_effect + news.news_effect
        assert total + news.reestimation_effect == pytest.approx(news.new_nowcast, abs=ATOL)
        assert news.check_identity(ATOL)

    def test_old_and_new_match_model_predictions(self, em_results, panel):
        old, new = vintage_pair(panel)
        news = news_decomposition(em_results, old, new, "2014Q4")
        slot = pd.Period("2014-12", freq="M")
        assert news.old_nowcast == pytest.approx(em_results.predict(old)["gdp"][slot], abs=1e-10)
        assert news.new_nowcast == pytest.approx(em_results.predict(new)["gdp"][slot], abs=1e-10)

    def test_two_step_matches_update(self, ts_model, panel):
        old, new = vintage_pair(panel)
        news = news_decomposition(ts_model.results_, old, new, "2014Q4")
        upd_old = ts_model.update(old).nowcast["out_of_sample"]["2014Q4"]
        upd_new = ts_model.update(new).nowcast["out_of_sample"]["2014Q4"]
        assert news.old_nowcast == pytest.approx(upd_old, abs=1e-10)
        assert news.new_nowcast == pytest.approx(upd_new, abs=1e-10)

    def test_with_revisions(self, em_results, panel):
        old, new = vintage_pair(panel)
        frame = new.data
        frame.loc[frame.index[-40], "ip"] += 1.5
        frame.loc[frame.index[-20], "pmi"] -= 0.7
        frame.loc[pd.Period("2013-12", freq="M"), "gdp"] += 0.3
        new = new.with_data(frame)
        news = news_decomposition(em_results, old, new, "2014Q4")
        assert news.info["n_revised"] == 3
        assert set(news.revisions.index) == {"ip", "pmi", "gdp"}
        assert news.revisions["impact"].sum() == pytest.approx(news.revisions_effect, abs=1e-10)
        assert news.removal_effect == pytest.approx(0.0, abs=1e-12)
        assert news.check_identity(ATOL)
        by_series = news.to_frame("series")
        assert by_series["revisions"].sum() == pytest.approx(news.revisions_effect, abs=1e-10)
        assert "revisions" in news.waterfall().index

    def test_revision_only_moves_nowcast_like_projection(self, em_results, panel):
        """Revisions with the same pattern equal the change of the nowcast."""
        frame = panel.data
        frame.loc[frame.index[-30], "sales"] += 2.0
        new = panel.with_data(frame)
        news = news_decomposition(em_results, panel, new, "2014Q4")
        assert news.n_releases == 0
        assert news.news_effect == 0.0
        assert news.new_nowcast - news.old_nowcast == pytest.approx(
            news.revisions_effect, abs=1e-12
        )

    def test_removed_values_warn_and_close(self, em_results, panel):
        old, new = vintage_pair(panel)
        frame = new.data
        frame.loc[frame.index[-50], "conf"] = np.nan
        new = new.with_data(frame)
        with pytest.warns(DataQualityWarning, match="missing in the new one"):
            news = news_decomposition(em_results, old, new, "2014Q4")
        assert news.info["n_removed"] == 1
        assert news.removal_effect != 0.0
        assert news.check_identity(ATOL)
        assert "revisions" in news.waterfall().index

    def test_reestimation_effect(self, panel):
        old, new = vintage_pair(panel)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res_old = MixedFreqDFM(n_factors=1, max_iter=5).fit(old, target="gdp")
            res_new = MixedFreqDFM(n_factors=1, max_iter=5).fit(new, target="gdp")
        news = news_decomposition(res_old, old, new, "2014Q4", new_results=res_new)
        assert news.info["reestimated"]
        assert news.reestimation_effect != 0.0
        slot = pd.Period("2014-12", freq="M")
        assert news.new_nowcast == pytest.approx(res_new.predict(new)["gdp"][slot], abs=1e-10)
        assert news.check_identity(ATOL)
        assert "re-estimation" in news.waterfall().index

    def test_same_parameters_no_reestimation(self, em_results, panel):
        old, new = vintage_pair(panel)
        news = news_decomposition(em_results, old, new, "2014Q4", new_results=em_results)
        assert not news.info["reestimated"]
        assert news.reestimation_effect == 0.0

    def test_new_results_other_target(self, em_results, panel):
        old, new = vintage_pair(panel)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            other = MixedFreqDFM(n_factors=1, max_iter=2).fit(panel, target="ip")
        with pytest.raises(ValueError, match="target"):
            news_decomposition(em_results, old, new, "2014Q4", new_results=other)

    def test_new_vintage_extends_grid(self, em_results, panel):
        """A new vintage longer than the model grid is handled (grid extended)."""
        longer = panel.extend(6)
        frame = longer.data
        frame.iloc[-6:, : len(frame.columns) - 1] = 1.0
        new = longer.with_data(frame)
        news = news_decomposition(em_results, panel, new, "2015Q2")
        assert news.check_identity(ATOL)
        assert news.info["grid"][-1] >= pd.Period("2015-06", freq="M")


@settings(max_examples=15, deadline=None)
@given(shift=st.lists(st.floats(-5, 5), min_size=6, max_size=6))
def test_weights_do_not_depend_on_values(shift):
    """Weights depend on the pattern only; the identity holds for any released values."""
    panel = _PANEL
    old, new = vintage_pair(panel)
    frame = new.data
    for k, col in enumerate(["ip", "sales", "pmi", "conf", "spread", "stocks"]):
        frame.loc[frame.index[-2], col] = (
            frame.loc[frame.index[-2], col] + shift[k]
            if not np.isnan(frame.loc[frame.index[-2], col])
            else np.nan
        )
    news = news_decomposition(_RES, old, new.with_data(frame), "2014Q4")
    base = _BASE
    np.testing.assert_allclose(news.releases["weight"], base.releases["weight"], atol=1e-12)
    assert news.check_identity(ATOL)


_PANEL = simulate_panel()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    _RES = MixedFreqDFM(n_factors=1, max_iter=5).fit(_PANEL, target="gdp")
_BASE = news_decomposition(_RES, *vintage_pair(_PANEL), "2014Q4")


def test_impact_linear_in_news(em_results, panel):
    old, new = vintage_pair(panel)
    news = news_decomposition(em_results, old, new, "2014Q4")
    rel = news.releases
    np.testing.assert_allclose(rel["impact"], rel["weight"] * rel["news"], atol=1e-12)
    np.testing.assert_allclose(rel["news"], rel["actual"] - rel["expected"], atol=1e-12)


# ====================================================================== tables
class TestTables:
    @pytest.fixture(scope="class")
    def news(self, em_blocks_results, panel):
        old, new = vintage_pair(panel)
        frame = new.data
        frame.loc[frame.index[-40], "ip"] += 1.0
        return news_decomposition(em_blocks_results, old, new.with_data(frame), "2014Q4")

    def test_release_columns(self, news):
        assert list(news.releases.columns) == [
            "series",
            "reference_period",
            "slot",
            "actual",
            "expected",
            "news",
            "weight",
            "impact",
            "category",
            "block",
        ]
        assert set(news.releases["category"]) <= {"hard", "soft", "financial"}
        pmi = news.releases[news.releases["series"] == "pmi"]
        assert set(pmi["block"]) == {"global"}
        assert news.revisions.loc["ip", "block"] == "global+real"
        assert "global+real" in news.to_frame("block").index

    @pytest.mark.parametrize("by", ["series", "block", "category"])
    def test_groupings_add_up(self, news, by):
        frame = news.to_frame(by)
        assert frame.index.name == by
        assert frame["news"].sum() == pytest.approx(news.news_effect, abs=1e-12)
        assert frame["revisions"].sum() == pytest.approx(news.revisions_effect, abs=1e-10)
        np.testing.assert_allclose(frame["total"], frame["news"] + frame["revisions"])

    def test_category_order(self, news):
        assert list(news.to_frame("category").index) == ["hard", "soft", "financial"]

    def test_release_frame(self, news):
        out = news.to_frame("release")
        pd.testing.assert_frame_equal(out, news.releases)

    def test_bad_grouping(self, news):
        with pytest.raises(ValueError, match="by must be"):
            news.to_frame("nope")
        with pytest.raises(ValueError, match="by must be"):
            news.waterfall("nope")

    def test_waterfall_closes(self, news):
        steps = news.waterfall("block")
        assert steps.index[0] == "old nowcast" and steps.index[-1] == "new nowcast"
        increments = steps.iloc[1:-1].sum()
        assert steps.iloc[0] + increments == pytest.approx(steps.iloc[-1], abs=ATOL)

    def test_top_releases(self, news):
        top = news.top_releases(3)
        assert len(top) == 3
        assert top["impact"].abs().is_monotonic_decreasing

    def test_summary_and_repr(self, news):
        text = news.summary()
        assert "News decomposition: gdp 2014Q4" in text
        assert "Data revisions" in text and "hard" in text
        assert "NewsResults(target='gdp'" in repr(news)
        assert news.total_change == pytest.approx(news.new_nowcast - news.old_nowcast)

    def test_plot_waterfall(self, news):
        fig = news.plot("waterfall", by="series")
        assert fig.axes
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_unknown(self, news):
        with pytest.raises(NotImplementedError, match="No plot"):
            news.plot("nope")

    def test_series_order_unknown_labels(self, news):
        """Labels not in the stored order are appended alphabetically."""
        assert news._order("series", ["zzz", "ip"]) == ["ip", "zzz"]


def test_empty_decomposition(em_results, panel):
    news = news_decomposition(em_results, panel, panel, "2014Q4")
    assert news.n_releases == 0
    assert news.old_nowcast == news.new_nowcast
    assert news.to_frame("category").empty
    assert list(news.waterfall().index) == ["old nowcast", "new nowcast"]
    assert "release" not in news.summary().split("Releases")[1].split("\n")[3]


def test_categories_override(em_results, panel):
    old, new = vintage_pair(panel)
    news = news_decomposition(em_results, old, new, "2014Q4", categories={"pmi": "survey"})
    assert "survey" in news.to_frame("category").index


def test_default_target_period(em_results, panel):
    old, new = vintage_pair(panel)
    news = news_decomposition(em_results, old, new)
    assert news.target_period == pd.Period("2014Q4", freq="Q")


def test_default_period_without_target_column(em_results, panel):
    lin = linear_model(em_results)
    frame = panel.data.drop(columns="gdp")
    assert _default_period(lin, frame) == pd.Period("2014Q4", freq="Q")
    empty = panel.data.assign(gdp=np.nan)
    assert _default_period(lin, empty) == pd.Period("2014Q4", freq="Q")


def test_dataframe_inputs(em_results, panel):
    old, new = vintage_pair(panel)
    news = news_decomposition(em_results, old.data, new.data, "2014Q4")
    assert news.check_identity(ATOL)
    stamped = new.data.copy()
    stamped.index = stamped.index.to_timestamp()
    news2 = news_decomposition(em_results, old, stamped, "2014Q4")
    assert news2.new_nowcast == pytest.approx(news.new_nowcast, abs=1e-12)


# ====================================================================== errors
class TestErrors:
    def test_unsupported_results(self, panel):
        idx = pd.period_range("2020Q1", periods=1, freq="Q")
        frame = build_nowcast_frame(pd.Series([1.0], index=idx), pd.Series([1.0], index=idx))
        with pytest.raises(TypeError, match="MixedFreqDFMResults or TwoStepResults"):
            news_decomposition(NowcastResults(target="y", nowcast=frame), panel, panel)

    def test_two_step_variables_without_model_data(self, panel):
        res = TwoStepDFM(n_factors=1, aggregate="variables").fit(panel, target="gdp")
        with pytest.raises(NotImplementedError, match="model_data"):
            news_decomposition(res.replace(model_data=None), panel, panel, "2014Q4")

    def test_two_step_without_state_space(self, ts_results, panel):
        broken = ts_results.replace(state_space=None)
        with pytest.raises(ValueError, match="state-space"):
            news_decomposition(broken, panel, panel, "2014Q4")

    def test_two_step_no_constant(self, ts_results):
        bridge = ts_results.bridge
        alt = type(bridge)(
            target=bridge.target,
            params=bridge.params.iloc[1:],
            ols=bridge.ols,
            exog_names=bridge.exog_names,
            add_constant=False,
            sample=bridge.sample,
        )
        lin = linear_model(ts_results.replace(bridge=alt))
        assert lin.offset == 0.0

    def test_missing_series(self, em_results, panel):
        with pytest.raises(NowcastDataError, match="model series"):
            news_decomposition(em_results, panel.drop(["ip"]), panel, "2014Q4")

    def test_wrong_frequency_index(self, em_results, panel):
        frame = panel.data
        quarterly = frame.groupby(frame.index.asfreq("Q")).last()
        with pytest.raises(NowcastDataError, match="PeriodIndex"):
            news_decomposition(em_results, quarterly, panel, "2014Q4")

    def test_not_a_panel(self, em_results, panel):
        with pytest.raises(TypeError, match="panel"):
            news_decomposition(em_results, [1, 2], panel, "2014Q4")

    def test_period_too_early(self, ts_results, panel):
        with pytest.raises(ValueError, match="too early"):
            news_decomposition(ts_results, panel, panel, "2005Q1")

    def test_bad_period(self, em_results, panel):
        with pytest.raises(ValueError, match="Invalid target_period"):
            news_decomposition(em_results, panel, panel, "not-a-period")


# ====================================================================== helpers
def test_to_period_from_period():
    from nowcastbox.core.frequency import Frequency

    assert to_period(pd.Period("2020-05", freq="M"), Frequency.QUARTERLY) == pd.Period(
        "2020Q2", freq="Q"
    )


def test_same_parameters(em_results, ts_results):
    lin = linear_model(em_results)
    assert lin.same_parameters(lin)
    assert not lin.same_parameters(linear_model(ts_results))
    other = linear_model(em_results)
    object.__setattr__(other, "offset", other.offset + 1.0)
    assert not other.same_parameters(lin)


def test_metadata_without_stored_data(em_results):
    lin = linear_model(em_results.replace(data=None))
    assert set(lin.categories.values()) == {"uncategorized"}


def test_metadata_partial(ts_results, panel):
    lin = linear_model(ts_results.replace(data=panel.select(["ip", "gdp"])))
    assert lin.categories["ip"] == "hard"
    assert lin.categories["pmi"] == "uncategorized"


class TestNewsWeights:
    def test_default_cells(self, em_results, panel):
        _, new = vintage_pair(panel)
        w = news_weights(em_results, new, "2014Q4")
        assert {"series", "slot", "weight"} == set(w.columns)
        assert w["slot"].min() >= pd.Period("2014-01", freq="M")

    def test_cells_match_news_weights(self, em_results, panel):
        old, new = vintage_pair(panel)
        news = news_decomposition(em_results, old, new, "2014Q4")
        cells = list(zip(news.releases["series"], news.releases["slot"], strict=True))
        w = news_weights(em_results, new, "2014Q4", cells=cells)
        np.testing.assert_allclose(w["weight"], news.releases["weight"], atol=1e-12)

    def test_unobserved_cell(self, em_results, panel):
        with pytest.raises(NowcastDataError, match="not observed"):
            news_weights(em_results, panel, "2014Q4", cells=[("ip", "2014-12")])


def test_news_results_is_frozen(em_results, panel):
    news = news_decomposition(em_results, panel, panel, "2014Q4")
    assert isinstance(news, NewsResults)
    with pytest.raises(AttributeError):
        news.old_nowcast = 0.0  # type: ignore[misc]


def test_mixed_frequency_data_category_none(em_results):
    """Series without category are grouped as 'uncategorized'."""
    panel = simulate_panel()
    plain = MixedFrequencyData(panel.data, panel.frequencies)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = MixedFreqDFM(n_factors=1, max_iter=3).fit(plain, target="gdp")
    old, new = vintage_pair(plain)
    news = news_decomposition(res, old, new, "2014Q4")
    assert list(news.to_frame("category").index) == ["uncategorized"]
    assert set(news.releases["block"]) == {"global"}


# ====================================================================== two-step "variables"
class TestTwoStepVariables:
    @pytest.fixture(scope="class")
    def model(self, panel):
        estimator = TwoStepDFM(n_factors=1, aggregate="variables")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            estimator.fit(panel, target="gdp")
        return estimator

    def test_identity_and_update(self, model, panel):
        old, new = vintage_pair(panel)
        news = news_decomposition(model.results_, old, new, "2014Q4")
        assert news.n_releases > 0
        assert news.check_identity(ATOL)
        upd_old = model.update(old).nowcast["out_of_sample"]["2014Q4"]
        upd_new = model.update(new).nowcast["out_of_sample"]["2014Q4"]
        assert news.old_nowcast == pytest.approx(upd_old, abs=1e-10)
        assert news.new_nowcast == pytest.approx(upd_new, abs=1e-10)

    def test_releases_are_filtered_observations(self, model, panel):
        old, new = vintage_pair(panel)
        news = news_decomposition(model.results_, old, new, "2014Q4")
        filtered = model.results_.filter_panel(new).data
        row = news.releases.iloc[0]
        assert row["actual"] == pytest.approx(filtered.loc[row["slot"], row["series"]])

    def test_raw_revision_is_a_revision_effect(self, model, panel):
        old, new = vintage_pair(panel)
        frame = new.data
        frame.loc[frame.index[-4], "ip"] += 2.0
        news = news_decomposition(model.results_, old, new.with_data(frame), "2014Q4")
        assert news.check_identity(ATOL)
        assert news.revisions.loc["ip", "n_revised"] > 1  # one raw value, several windows
        assert abs(news.revisions_effect) > 1e-6

    def test_tracker_and_contributions(self, model, panel):
        contributions = model.results_.level_contributions(panel, "2014Q4")
        assert contributions.nowcast == pytest.approx(
            model.results_.nowcast["out_of_sample"]["2014Q4"], abs=1e-8
        )


def test_two_step_variables_prefilter_missing_series(panel):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = TwoStepDFM(n_factors=1, aggregate="variables").fit(panel, target="gdp")
    lin = linear_model(res)
    with pytest.raises(NowcastDataError, match="model series"):
        lin.prefilter(panel.data.drop(columns="gdp"))
