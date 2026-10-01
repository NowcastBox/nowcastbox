"""Property-based tests (hypothesis) for nowcastbox.vintages."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from nowcastbox.core.data import MixedFrequencyData
from nowcastbox.vintages import ReleaseCalendar, VintageStore, pseudo_real_time
from tests.vintages.conftest import FREQS, make_final_frame

pytestmark = pytest.mark.property

PANEL = MixedFrequencyData(make_final_frame(n_months=30, seed=3), FREQS)

delays_st = st.fixed_dictionaries(
    {
        "ip": st.integers(0, 120),
        "pmi": st.integers(0, 120),
        "gdp": st.integers(0, 200),
    }
)
dates_st = st.dates(min_value=dt.date(2017, 11, 1), max_value=dt.date(2021, 6, 30))


@given(delays=delays_st, vintage=dates_st)
def test_pseudo_real_time_equals_core_as_of(delays: dict[str, int], vintage: dt.date) -> None:
    out = pseudo_real_time(PANEL, delays, vintage)
    assert out.equals(PANEL.as_of(pd.Timestamp(vintage), release_delays=delays))


@given(delays=delays_st, d1=dates_st, d2=dates_st)
def test_information_sets_are_nested(delays: dict[str, int], d1: dt.date, d2: dt.date) -> None:
    old, new = sorted([d1, d2])
    a = pseudo_real_time(PANEL, delays, old).observation_mask()
    b = pseudo_real_time(PANEL, delays, new).observation_mask()
    assert (a <= b).all().all()


@given(delays=delays_st, d1=dates_st, d2=dates_st)
def test_releases_between_is_mask_difference(
    delays: dict[str, int], d1: dt.date, d2: dt.date
) -> None:
    old, new = sorted([d1, d2])
    cal = ReleaseCalendar.from_data(PANEL, delays)
    rel = cal.releases_between(old, new, PANEL)
    diff = cal.release_mask(PANEL, new) & ~cal.release_mask(PANEL, old)
    diff &= PANEL.observation_mask()
    expected = {(s, p) for s in diff.columns for p in diff.index[diff[s].to_numpy()]}
    assert set(zip(rel["series"], rel["slot"], strict=True)) == expected
    # and the releases without data agree on the dates
    free = cal.releases_between(old, new)
    inside = {(s, str(p)) for s, p in zip(free["series"], free["reference_period"], strict=True)}
    assert {
        (s, str(p)) for s, p in zip(rel["series"], rel["reference_period"], strict=True)
    } <= inside


@given(delay=st.integers(0, 400), freq=st.sampled_from(["M", "Q", "A", "W", "D"]), date=dates_st)
def test_available_at_is_last_released(delay: int, freq: str, date: dt.date) -> None:
    cal = ReleaseCalendar({"x": delay}, frequencies={"x": freq})
    last = cal.available_at(date)["x"]
    assert cal.release_date("x", last) <= pd.Timestamp(date)
    assert cal.release_date("x", last + 1) > pd.Timestamp(date)


@given(
    values=st.lists(
        st.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False), min_size=1, max_size=6
    ),
    gaps=st.lists(st.integers(1, 60), min_size=6, max_size=6),
    query=st.integers(-5, 400),
)
def test_store_as_of_returns_latest_known(values: list[float], gaps: list[int], query: int) -> None:
    start = pd.Timestamp("2020-02-01")
    dates = [start + pd.Timedelta(days=int(sum(gaps[: i + 1]))) for i in range(len(values))]
    records = pd.DataFrame(
        {
            "series": "a",
            "reference_period": "2020-01",
            "vintage_date": dates,
            "value": values,
        }
    )
    store = VintageStore(records)
    when = start + pd.Timedelta(days=query)
    known = [v for d, v in zip(dates, values, strict=True) if d <= when]
    got = store.as_of(when)["a"].iloc[0]
    if known:
        assert got == known[-1]
    else:
        assert np.isnan(got)
    # revisions telescope: first release + sum of revisions = latest value
    rev = store.revisions("a")
    total = rev["value"].iloc[0] + rev["revision"].iloc[1:].sum()
    assert total == pytest.approx(values[-1], rel=1e-9, abs=1e-6)
