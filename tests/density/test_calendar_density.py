"""Density nowcasts on a weekly base grid (calendar aggregation, innovation I1)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from nowcastbox.core.exceptions import NowcastDataError
from nowcastbox.core.frequency import is_period_end
from nowcastbox.density import bootstrap_nowcasts, nowcast_distribution, simulate_from_results
from nowcastbox.models import MixedFreqDFM
from tests.models.test_frequencies_em import simulate_weekly_dfm


@pytest.fixture(scope="module")
def weekly_results():
    sim = simulate_weekly_dfm(n_weeks=208, seed=4)
    data = sim.data.copy()
    quarter_end = data.index[is_period_end(data.index, "Q")]
    data.loc[quarter_end[-1:], "gdp"] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return MixedFreqDFM(idiosyncratic="iid", max_iter=20).fit(sim.panel(data), "gdp")


def test_simulation_keeps_pattern(weekly_results):
    sim = simulate_from_results(weekly_results, random_state=0)
    assert sim.observation_mask().equals(weekly_results.data.observation_mask())


def test_parametric_bootstrap(weekly_results):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        boot = bootstrap_nowcasts(weekly_results, 3, random_state=0, refit_params={"max_iter": 3})
    assert boot.method == "parametric" and boot.n_success == 3
    dist = nowcast_distribution(weekly_results)
    assert np.all(np.isfinite(dist.mean)) and np.all(dist.std > 0)


def test_block_bootstrap_rejected(weekly_results):
    with pytest.raises(NowcastDataError, match="parametric"):
        bootstrap_nowcasts(weekly_results, 2, method="block")
