"""Tests of :func:`nowcastbox.news.nowcast_tracker`."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from nowcastbox.models import MixedFreqDFM
from nowcastbox.news import NowcastTracker, nowcast_tracker
from nowcastbox.vintages import ReleaseCalendar, VintageStore, pseudo_real_time
from tests.news.conftest import DELAYS

START, END = "2014-10-01", "2015-02-28"


@pytest.fixture(scope="module")
def tracker(em_results, panel):
    return nowcast_tracker(em_results, panel, None, "2014Q4", START, END)


class TestTracker:
    def test_identity(self, tracker):
        assert isinstance(tracker, NowcastTracker)
        assert tracker.check_identity(1e-8)
        np.testing.assert_allclose(
            tracker.contributions.sum(axis=1), tracker.path["change"], atol=1e-8
        )

    def test_path_matches_direct_nowcasts(self, tracker, em_results, panel):
        slot = pd.Period("2014-12", freq="M")
        for date, value in tracker.nowcasts.items():
            vintage = pseudo_real_time(panel, vintage=date)
            assert value == pytest.approx(em_results.predict(vintage)["gdp"][slot], abs=1e-9)

    def test_dates_are_release_dates(self, tracker, panel):
        cal = ReleaseCalendar.from_data(panel)
        expected = cal.release_dates(START, END, panel)
        assert tracker.path.index[0] == pd.Timestamp(START)
        assert list(tracker.path.index[1:]) == list(expected)
        assert tracker.path["n_releases"].iloc[1:].min() >= 1
        assert tracker.path["n_releases"].sum() == len(tracker.releases())

    def test_contribution_columns(self, tracker):
        cols = list(tracker.contributions.columns)
        assert cols[-2:] == ["revisions", "re-estimation"]
        assert set(cols[:-2]) <= {"hard", "soft", "financial"}
        assert (tracker.contributions.iloc[0] == 0).all()
        cum = tracker.cumulative_contributions()
        assert cum.iloc[-1].sum() == pytest.approx(
            tracker.path["nowcast"].iloc[-1] - tracker.path["nowcast"].iloc[0], abs=1e-8
        )

    def test_frames_and_text(self, tracker):
        frame = tracker.to_frame()
        assert frame.columns[0] == "nowcast"
        assert any(c.startswith("contrib:") for c in frame.columns)
        assert "vintage" in tracker.releases().columns
        text = tracker.summary()
        assert "Nowcast tracker: gdp 2014Q4" in text
        assert "2014-10-01" in text

    def test_plot(self, tracker):
        import matplotlib.pyplot as plt

        fig = tracker.plot()
        assert fig.axes
        plt.close(fig)


def test_by_series_with_calendar_object(em_results, panel):
    cal = ReleaseCalendar(DELAYS, frequencies=dict.fromkeys(DELAYS, "M") | {"gdp": "Q"})
    tr = nowcast_tracker(em_results, panel, cal, "2014Q4", START, END, by="series")
    assert tr.check_identity()
    assert "pmi" in tr.contributions.columns


def test_calendar_int_and_mapping(em_results, panel):
    tr_int = nowcast_tracker(em_results, panel, 20, "2014Q4", START, END)
    assert tr_int.check_identity()
    tr_map = nowcast_tracker(em_results, panel, dict(DELAYS), "2014Q4", START, END)
    assert tr_map.check_identity()
    with pytest.raises(ValueError, match="calendar must be"):
        nowcast_tracker(em_results, panel, "bad", "2014Q4", START, END)


def test_default_window(em_results, panel):
    tr = nowcast_tracker(em_results, panel, None, "2014Q4")
    assert tr.path.index[0] == pd.Timestamp("2014-10-01")
    assert tr.path.index[-1] <= pd.Timestamp("2014-12-31")


def test_explicit_dates(em_results, panel):
    dates = ["2014-11-15", "2014-10-01", "2014-12-15", "2014-12-15"]
    tr = nowcast_tracker(em_results, panel, None, "2014Q4", dates=dates)
    assert list(tr.path.index.strftime("%Y-%m-%d")) == ["2014-10-01", "2014-11-15", "2014-12-15"]
    assert tr.check_identity()


def test_vintage_store(em_results, panel):
    store = VintageStore.from_calendar(panel)
    tr = nowcast_tracker(em_results, store, None, "2014Q4", START, END)
    ref = nowcast_tracker(em_results, panel, None, "2014Q4", START, END)
    np.testing.assert_allclose(tr.path["nowcast"], ref.path["nowcast"], atol=1e-10)


def test_vintage_store_unknown_target(panel):
    store = VintageStore.from_calendar(panel.drop(["gdp"]))
    with pytest.raises(ValueError, match="no series"):
        nowcast_tracker(MixedFreqDFM(), store, None, "2014Q4", START, END, target="gdp")


def test_fitted_estimator(em_model, panel, tracker):
    tr = nowcast_tracker(em_model, panel, None, "2014Q4", START, END)
    np.testing.assert_allclose(tr.path["nowcast"], tracker.path["nowcast"], atol=1e-12)


def test_unfitted_estimator_and_refit(panel):
    model = MixedFreqDFM(n_factors=1, max_iter=3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tr = nowcast_tracker(
            model, panel, None, "2014Q4", START, "2014-11-30", target="gdp", refit=True
        )
    assert tr.check_identity(1e-8)
    assert (tr.path["reestimation"].iloc[1:] != 0).any()
    assert not model.is_fitted


def test_unfitted_estimator_fixed(panel):
    model = MixedFreqDFM(n_factors=1, max_iter=3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tr = nowcast_tracker(model, panel, None, "2014Q4", START, "2014-11-30", target="gdp")
    assert (tr.path["reestimation"] == 0).all()


def test_store_unfitted(panel):
    store = VintageStore.from_calendar(panel)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tr = nowcast_tracker(
            MixedFreqDFM(n_factors=1, max_iter=3),
            store,
            None,
            "2014Q4",
            START,
            "2014-11-30",
            target="gdp",
        )
    assert tr.check_identity()


class TestErrors:
    def test_bad_by(self, em_results, panel):
        with pytest.raises(ValueError, match="by must be"):
            nowcast_tracker(em_results, panel, None, "2014Q4", by="x")

    def test_missing_period(self, em_results, panel):
        with pytest.raises(ValueError, match="target_period"):
            nowcast_tracker(em_results, panel)

    def test_bad_model(self, panel):
        with pytest.raises(TypeError, match="model_or_results"):
            nowcast_tracker(object(), panel, None, "2014Q4")

    def test_refit_needs_estimator(self, em_results, panel):
        with pytest.raises(TypeError, match="refit"):
            nowcast_tracker(em_results, panel, None, "2014Q4", refit=True)

    def test_unfitted_needs_target(self, panel):
        with pytest.raises(ValueError, match="target is required"):
            nowcast_tracker(MixedFreqDFM(), panel, None, "2014Q4")

    def test_unknown_target(self, panel):
        with pytest.raises(ValueError, match="no series"):
            nowcast_tracker(MixedFreqDFM(), panel, None, "2014Q4", target="zzz")

    def test_window(self, em_results, panel):
        with pytest.raises(ValueError, match="before end"):
            nowcast_tracker(em_results, panel, None, "2014Q4", "2014-12-01", "2014-11-01")

    def test_too_few_dates(self, em_results, panel):
        with pytest.raises(ValueError, match="two vintage"):
            nowcast_tracker(em_results, panel, None, "2014Q4", dates=["2014-10-01"])
