"""Storage-slot conventions of the backtest on weekly grids (innovation I1).

A lower-frequency value lives in the last base period that *ends* in its period (for a
weekly grid: the last week ending in the quarter), not in the base period containing
the last day of the period.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import nowcastbox as nb
from nowcastbox.core.frequency import Frequency
from nowcastbox.evaluation import PseudoRealTimeBacktest
from nowcastbox.evaluation.backtest import _estimates_from_frame
from nowcastbox.models import MixedFreqDFM


def test_estimates_from_weekly_frame_use_the_storage_slots() -> None:
    weeks = pd.period_range("2019-12-01", periods=30, freq="W")
    frame = pd.DataFrame({"gdp": np.arange(30.0)}, index=weeks)
    out = _estimates_from_frame(frame, "gdp", Frequency.QUARTERLY)
    # 2019Q4 -> week ending 2019-12-29 (position 4); 2020Q1 -> week ending 2020-03-29
    assert out.index.tolist() == [pd.Period("2019Q4"), pd.Period("2020Q1")]
    assert out.tolist() == [4.0, 17.0]


def test_estimates_from_monthly_frame() -> None:
    months = pd.period_range("2020-01", periods=7, freq="M")
    frame = pd.DataFrame({"gdp": np.arange(7.0)}, index=months)
    out = _estimates_from_frame(frame, "gdp", Frequency.QUARTERLY)
    assert out.to_dict() == {pd.Period("2020Q1"): 2.0, pd.Period("2020Q2"): 5.0}


def test_weekly_backtest_without_refit_keeps_forecasting() -> None:
    sim = nb.simulate.weekly_dfm(n_weeks=300, n_weekly=3, n_monthly=2, random_state=2)
    data = sim.data
    delay = {c: 7 if data.metadata[c].frequency is Frequency.WEEKLY else 30 for c in data}
    backtest = PseudoRealTimeBacktest(
        model=MixedFreqDFM(n_factors=1, max_iter=5),
        data=data,
        target="gdp",
        delay=delay,
        start="2015-03-01",
        end="2015-06-30",
        step="M",
        refit_every=3,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frame = backtest.run().forecasts
    assert frame["forecast"].notna().all()


def test_vintage_store_weekly_base_uses_storage_slots() -> None:
    from nowcastbox.vintages import VintageStore

    records = pd.DataFrame(
        {
            "series": ["gdp", "gdp", "x", "x"],
            "reference_period": ["2020Q1", "2020Q2", "2020-03-29", "2020-04-05"],
            "vintage_date": ["2020-05-01", "2020-08-01", "2020-04-01", "2020-04-10"],
            "value": [1.0, 2.0, 3.0, 4.0],
        }
    )
    store = VintageStore(records, frequencies={"gdp": "Q", "x": "W"}, base_frequency="W")
    frame = store.as_of("2020-09-01")
    assert isinstance(frame, pd.DataFrame)
    gdp = frame["gdp"].dropna()
    # 2020Q1 lives in the last week *ending* in March (2020-03-23/2020-03-29)
    assert [str(p) for p in gdp.index] == ["2020-03-23/2020-03-29", "2020-06-22/2020-06-28"]
    panel = store.as_of("2020-09-01", as_mixed=True)
    assert panel.to_native("gdp", dropna=True).to_dict() == {
        pd.Period("2020Q1"): 1.0,
        pd.Period("2020Q2"): 2.0,
    }
    first = store.nth_release(0, series="gdp")
    assert isinstance(first, pd.DataFrame)
    assert first["gdp"].dropna().index.equals(gdp.index)


def test_block_selection_estimates_use_the_storage_slots() -> None:
    from nowcastbox.selection.blocks import _native_estimates

    weeks = pd.period_range("2019-12-01", periods=30, freq="W")
    frame = pd.DataFrame({"gdp": np.arange(30.0)}, index=weeks)
    out = _native_estimates(frame, "gdp", Frequency.QUARTERLY)
    assert out.to_dict() == {pd.Period("2019Q4"): 4.0, pd.Period("2020Q1"): 17.0}
