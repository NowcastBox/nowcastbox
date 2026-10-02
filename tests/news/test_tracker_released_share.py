"""The ``released_share`` column of the nowcast tracker (ECB parity item 5)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from nowcastbox.news import nowcast_tracker
from nowcastbox.news.tracker import _model_predictors, _released_share
from nowcastbox.vintages import VintageStore, pseudo_real_time

START, END = "2014-10-01", "2015-02-28"


@pytest.fixture(scope="module")
def tracker(em_results, panel):
    return nowcast_tracker(em_results, panel, None, "2014Q4", START, END)


def test_column_present_and_bounded(tracker):
    share = tracker.path["released_share"]
    assert list(tracker.path.columns)[-1] == "released_share"
    assert share.between(0.0, 1.0).all()
    assert (np.diff(share.to_numpy()) >= -1e-12).all()  # no revisions: never decreases
    assert share.iloc[0] < share.iloc[-1]
    # final panel: ip and sales miss 2 months, conf and stocks 1 -> 12 of 18 observations
    assert share.iloc[-1] == pytest.approx(12 / 18)


def test_matches_manual_count(tracker, panel):
    predictors = [c for c in panel.columns if c != "gdp"]
    period = pd.Period("2014Q4", freq="Q")
    for date, value in tracker.path["released_share"].items():
        vintage = pseudo_real_time(panel, vintage=date)
        observed = vintage.to_frame().loc["2014-10":"2014-12", predictors].notna()
        assert value == pytest.approx(observed.to_numpy().mean())
        table = vintage.released_share(period, series=predictors)
        assert value == pytest.approx(table.loc["total", "share"])


def test_first_vintage_has_nothing_of_the_quarter(tracker):
    # 2014-10-01: no October data can be out yet (smallest delay is 0 days after month end)
    assert tracker.path["released_share"].iloc[0] == 0.0


def test_summary_shows_share(tracker):
    assert "released" in tracker.summary()


def test_vintage_store_source(em_results, panel, tracker):
    store = VintageStore.from_calendar(panel)
    tr = nowcast_tracker(em_results, store, None, "2014Q4", START, END)
    np.testing.assert_allclose(tr.path["released_share"], tracker.path["released_share"])


def test_target_only_panel_is_nan(panel):
    assert np.isnan(_released_share(panel.select(["gdp"]), pd.Period("2014Q4", "Q"), "gdp"))


def test_only_model_predictors_are_counted(panel):
    # a formula fit on two predictors: unused panel series must not dilute the share
    from nowcastbox.models import TwoStepDFM

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = TwoStepDFM(n_factors=1).fit(panel, "gdp ~ ip + conf")
        tr = nowcast_tracker(res, panel, None, "2014Q4", START, END)
    for date, value in tr.path["released_share"].items():
        vintage = pseudo_real_time(panel, vintage=date)
        table = vintage.released_share(pd.Period("2014Q4", freq="Q"), series=["ip", "conf"])
        assert value == pytest.approx(table.loc["total", "share"])
    # final panel: ip misses 2 months, conf 1 -> 3 of 6 observations
    assert tr.path["released_share"].iloc[-1] == pytest.approx(3 / 6)


def test_model_predictors_helper(panel):
    assert _model_predictors(object(), "gdp") is None
    assert _released_share(panel, pd.Period("2014Q4", "Q"), "gdp", ["ip"]) == pytest.approx(1 / 3)
