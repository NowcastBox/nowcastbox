"""Tests of :func:`nowcastbox.news.level_contributions`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nowcastbox.news import LevelContributions, level_contributions
from tests.news.conftest import vintage_pair


@pytest.mark.parametrize("fixture", ["em_results", "ts_results", "em_blocks_results"])
def test_identity(request, fixture):
    res = request.getfixturevalue(fixture)
    lc = level_contributions(res, target_period="2014Q4")
    assert isinstance(lc, LevelContributions)
    assert lc.check_identity(1e-8)
    total = lc.baseline + lc.contributions["contribution"].sum()
    assert total == pytest.approx(lc.nowcast, abs=1e-8)


def test_nowcast_matches_model(em_results):
    lc = level_contributions(em_results, target_period="2014Q4")
    expected = em_results.nowcast.loc["2014Q4", "out_of_sample"]
    assert lc.nowcast == pytest.approx(expected, abs=1e-9)


def test_baseline_is_target_mean(em_results):
    """Stationary zero-mean state: baseline = sample mean of the target."""
    lc = level_contributions(em_results, target_period="2014Q4")
    assert lc.baseline == pytest.approx(float(em_results.standardization.mean["gdp"]), abs=1e-9)


def test_baseline_two_step_is_intercept(ts_results):
    lc = level_contributions(ts_results, target_period="2014Q4")
    assert lc.baseline == pytest.approx(float(ts_results.bridge.params.iloc[0]), abs=1e-9)
    assert lc.contributions.loc["gdp", "n_obs"] == 0 if "gdp" in lc.contributions.index else True


def test_no_data_equals_baseline(em_results, panel):
    empty = panel.with_data(panel.data * np.nan)
    lc = level_contributions(em_results, empty, "2014Q4")
    assert (lc.contributions["contribution"] == 0).all()
    assert lc.nowcast == pytest.approx(lc.baseline, abs=1e-12)


def test_change_equals_news_plus_revisions(em_results, panel):
    """Difference of level contributions between vintages = nowcast revision."""
    old, new = vintage_pair(panel)
    a = level_contributions(em_results, old, "2014Q4")
    b = level_contributions(em_results, new, "2014Q4")
    diff = (b.contributions["contribution"] - a.contributions["contribution"]).sum()
    assert diff == pytest.approx(b.nowcast - a.nowcast, abs=1e-8)


@pytest.mark.parametrize("by", ["series", "block", "category"])
def test_to_frame(em_blocks_results, by):
    lc = level_contributions(em_blocks_results, target_period="2014Q4")
    frame = lc.to_frame(by)
    assert frame.index.name == by
    assert "baseline" in frame.index
    assert frame["contribution"].sum() == pytest.approx(lc.nowcast, abs=1e-8)


def test_bad_grouping(em_results):
    lc = level_contributions(em_results, target_period="2014Q4")
    with pytest.raises(ValueError, match="by must be"):
        lc.to_frame("x")


def test_default_period(em_results, panel):
    lc = level_contributions(em_results)
    assert lc.target_period == pd.Period("2014Q4", freq="Q")
    no_target = panel.data.drop(columns="gdp")
    with pytest.raises(Exception, match="model series"):
        level_contributions(em_results, no_target)
    empty_target = panel.data.assign(gdp=np.nan)
    lc2 = level_contributions(em_results, empty_target)
    assert lc2.target_period == pd.Period("2014Q4", freq="Q")


def test_default_period_frame_without_target(ts_results, panel):
    lc = level_contributions(ts_results, panel.data.drop(columns="gdp"))
    assert lc.target_period == pd.Period("2014Q4", freq="Q")


def test_no_stored_data(em_results):
    with pytest.raises(ValueError, match="No data"):
        level_contributions(em_results.replace(data=None))


def test_summary_and_plot(em_results):
    import matplotlib.pyplot as plt

    lc = level_contributions(em_results, target_period="2014Q4", categories={"ip": "labour"})
    assert "Level contributions: gdp 2014Q4" in lc.summary("category")
    assert "labour" in lc.to_frame("category").index
    fig = lc.plot(by="category")
    assert fig.axes
    plt.close(fig)
