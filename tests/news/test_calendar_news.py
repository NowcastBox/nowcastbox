"""News, tracker and level contributions on a weekly base grid (calendar aggregation, I1)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from nowcastbox.core.frequency import is_period_end, period_to_base
from nowcastbox.models import MixedFreqDFM
from nowcastbox.news import news_decomposition
from nowcastbox.news._model import linear_model
from tests.models.test_frequencies_em import simulate_weekly_dfm


@pytest.fixture(scope="module")
def weekly():
    sim = simulate_weekly_dfm(n_weeks=260, seed=3)
    data = sim.data.copy()
    quarter_end = data.index[is_period_end(data.index, "Q")]
    data.loc[quarter_end[-2:], "gdp"] = np.nan
    final = sim.panel(data)
    old = data.copy()
    old.iloc[-3:, :] = np.nan  # three weeks earlier, nothing of them released yet
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = MixedFreqDFM(idiosyncratic="iid", max_iter=30).fit(sim.panel(old), "gdp")
    return res, sim.panel(old), final, quarter_end[-1]


def test_linear_model_is_time_varying(weekly):
    res, _, _, _ = weekly
    lin = linear_model(res)
    assert lin.model_builder is not None and lin.target_row is not None
    model = lin.model_for(len(res.grid) + 7)
    assert model.is_time_varying and model.n_periods == len(res.grid) + 7
    assert lin.model_for(len(res.grid) + 7) is model  # cached


def test_news_identity_and_predictions(weekly):
    res, old, new, slot = weekly
    period = slot.asfreq("Q")
    news = news_decomposition(res, old, new, period)
    assert news.n_releases > 0
    assert news.check_identity(1e-8)
    assert period_to_base(period, "W") == slot
    assert news.old_nowcast == pytest.approx(res.predict(old)["gdp"][slot], abs=1e-8)
    assert news.new_nowcast == pytest.approx(res.predict(new)["gdp"][slot], abs=1e-8)
    weekly_rows = news.releases[news.releases["series"].str.startswith("w")]
    assert len(weekly_rows) > 0
    assert isinstance(weekly_rows["slot"].iloc[0], pd.Period)


def test_level_contributions_and_tracker(weekly):
    res, old, new, slot = weekly
    period = slot.asfreq("Q")
    contributions = res.level_contributions(new, period)
    assert contributions.nowcast == pytest.approx(res.predict(new)["gdp"][slot], abs=1e-8)
    assert contributions.check_identity(1e-8)
    delays = dict.fromkeys(new.columns, 3)
    tracker = res.nowcast_tracker(
        new,
        calendar=delays,
        target_period=period,
        start=old.index[-4].end_time.normalize(),
        end=new.index[-1].end_time.normalize() + pd.Timedelta(days=3),
    )
    assert len(tracker.steps) >= 1
    assert all(step.check_identity(1e-8) for step in tracker.steps)
