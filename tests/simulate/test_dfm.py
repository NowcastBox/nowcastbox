"""Tests of :mod:`nowcastbox.simulate` (DFMs with known parameters)."""

from __future__ import annotations

import doctest
import importlib
import warnings

import pandas as pd
import pytest

from nowcastbox.core.data import MixedFrequencyData
from nowcastbox.core.frequency import Frequency
from nowcastbox.datasets import load_simulated_dfm
from nowcastbox.models import MixedFreqDFM
from nowcastbox.simulate import SimulatedDFM, dfm, weekly_dfm


def test_doctests() -> None:
    module = importlib.import_module("nowcastbox.simulate.dfm")  # the function shadows it
    result = doctest.testmod(module, optionflags=doctest.ELLIPSIS)
    assert result.failed == 0 and result.attempted > 0


def test_dfm_panel_and_truth() -> None:
    sim = dfm(n_series=6, n_factors=2, n_periods=72, random_state=3)
    assert isinstance(sim, SimulatedDFM) and isinstance(sim.data, MixedFrequencyData)
    data, truth = sim
    assert data.frequencies["gdp"] is Frequency.QUARTERLY
    assert data.release_delays["gdp"] == 45
    assert truth["loadings"].shape == (7, 2)
    assert truth["factors"].index.equals(data.index)


def test_dfm_matches_dataset_generator() -> None:
    ds = load_simulated_dfm()
    sim = dfm()
    pd.testing.assert_frame_equal(sim.data.data, ds.data.data, check_names=False)


def test_dfm_without_ragged_edge() -> None:
    data = dfm(n_series=3, n_factors=1, n_periods=36, ragged_edge=False).data
    assert data.data.iloc[-1, :3].notna().all()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_weeks": 10}, "n_weeks"),
        ({"n_weekly": 0}, "n_weekly"),
        ({"n_monthly": True}, "n_monthly"),
        ({"phi": 1.0}, "phi"),
        ({"noise": -1.0}, "noise"),
    ],
)
def test_weekly_invalid(kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        weekly_dfm(**kwargs)


def test_weekly_dfm_structure() -> None:
    sim = weekly_dfm(n_weeks=156, n_weekly=2, n_monthly=2, random_state=2)
    data = sim.data
    assert data.base_frequency is Frequency.WEEKLY
    assert data.frequencies["m0"] is Frequency.MONTHLY
    assert data.metadata["m0"].aggregation.value == "average"
    # the last quarter is not observed, the monthly series miss the last two weeks
    assert data.data[["m0", "m1"]].iloc[-2:].isna().all().all()
    common = sim.truth["common"]
    observed = data.data["w0"]
    assert observed.corr(common["w0"]) > 0.7


def test_weekly_dfm_full_and_fit() -> None:
    sim = weekly_dfm(n_weeks=208, ragged_edge=False, random_state=4)
    assert sim.data.data["gdp"].notna().sum() >= 13
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = MixedFreqDFM(idiosyncratic="iid", max_iter=30).fit(sim.data, "gdp")
    factor = res.factors.iloc[:, 0]
    assert abs(factor.corr(sim.truth["factor"].reindex(factor.index))) > 0.9
