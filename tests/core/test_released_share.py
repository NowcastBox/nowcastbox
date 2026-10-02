"""Tests of :meth:`MixedFrequencyData.released_share` (ECB parity item 5)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nowcastbox.core.data import MixedFrequencyData, _series_groups
from nowcastbox.core.exceptions import NowcastDataError


@pytest.fixture
def ragged() -> MixedFrequencyData:
    """Known ragged edge in 2020Q2: ip 1/3, sales 2/3, pmi 3/3, gdp 0/1, cpi 3/3."""
    idx = pd.period_range("2020-01", periods=6, freq="M")
    nan = np.nan
    df = pd.DataFrame(
        {
            "ip": [1.0, 1.0, 1.0, 1.0, nan, nan],
            "sales": [1.0, 1.0, 1.0, 1.0, 1.0, nan],
            "pmi": [1.0] * 6,
            "gdp": [nan, nan, 1.0, nan, nan, nan],
            "cpi": [1.0] * 6,
        },
        index=idx,
    )
    return MixedFrequencyData(
        df,
        {"ip": "M", "sales": "M", "pmi": "M", "gdp": "Q", "cpi": "M"},
        categories={"ip": "hard", "sales": "hard", "pmi": "soft", "gdp": "hard"},
        blocks={"ip": ["global", "real"], "sales": ["real"], "pmi": ["global"], "gdp": ["global"]},
        release_delays={"ip": 40, "sales": 30, "pmi": 1, "gdp": 60, "cpi": 10},
    )


def test_manual_counts_by_series(ragged):
    table = ragged.released_share("2020Q2")
    assert table.index.name == "series"
    assert table["released"].to_dict() == {
        "ip": 1,
        "sales": 2,
        "pmi": 3,
        "gdp": 0,
        "cpi": 3,
        "total": 9,
    }
    assert table["expected"].to_dict() == {
        "ip": 3,
        "sales": 3,
        "pmi": 3,
        "gdp": 1,
        "cpi": 3,
        "total": 13,
    }
    assert table.loc["sales", "share"] == pytest.approx(2 / 3)
    assert table.loc["total", "share"] == pytest.approx(9 / 13)
    assert list(table.columns) == ["released", "expected", "weight", "share"]
    assert table["released"].dtype == int


def test_first_quarter_complete(ragged):
    table = ragged.released_share(pd.Period("2020Q1", freq="Q"), by="series")
    assert table.loc["total", "share"] == pytest.approx(1.0)


def test_by_category(ragged):
    table = ragged.released_share("2020Q2", by="category")
    assert table.index.name == "group"
    assert list(table.index) == ["hard", "soft", "uncategorized", "total"]
    assert table.loc["hard", "released"] == 3
    assert table.loc["hard", "expected"] == 7
    assert table.loc["hard", "share"] == pytest.approx(3 / 7)
    assert table.loc["soft", "share"] == pytest.approx(1.0)


def test_by_block_counts_series_in_every_block(ragged):
    table = ragged.released_share("2020Q2", by="block")
    assert list(table.index) == ["global", "real", "unassigned", "total"]
    assert table.loc["global", "expected"] == 3 + 3 + 1  # ip, pmi, gdp
    assert table.loc["real", "released"] == 1 + 2
    assert table.loc["total", "expected"] == 13  # each series once


def test_by_frequency_and_mapping(ragged):
    freq = ragged.released_share("2020Q2", by="frequency")
    assert freq.loc["quarterly", "share"] == 0.0
    assert freq.loc["monthly", "share"] == pytest.approx(9 / 12)
    mapping = ragged.released_share("2020Q2", by={"ip": ["a", "b"], "pmi": "b"})
    assert list(mapping.index) == ["a", "b", "unassigned", "total"]
    assert mapping.loc["b", "released"] == 4
    with pytest.raises(ValueError, match="unknown series"):
        ragged.released_share("2020Q2", by={"nope": "a"})
    with pytest.raises(ValueError, match="by must be"):
        ragged.released_share("2020Q2", by="colour")


def test_weights(ragged):
    weights = {"ip": 3.0, "pmi": 1.0}
    table = ragged.released_share("2020Q2", weights=weights)
    expected = (3.0 * (1 / 3) + 1.0 * 1.0) / 4.0
    assert table.loc["total", "share"] == pytest.approx(expected)
    assert table.loc["total", "weight"] == pytest.approx(4.0)
    assert np.isnan(table.loc["sales", "share"])  # zero weight
    as_series = ragged.released_share("2020Q2", weights=pd.Series(weights), by="category")
    assert as_series.loc["soft", "share"] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="non-negative"):
        ragged.released_share("2020Q2", weights={"ip": -1.0})
    with pytest.raises(ValueError, match="non-negative"):
        ragged.released_share("2020Q2", weights={"ip": np.inf})
    with pytest.raises(ValueError, match="unknown series"):
        ragged.released_share("2020Q2", weights={"zz": 1.0})


def test_series_subset_and_as_of(ragged):
    sub = ragged.released_share("2020Q2", series=["ip", "pmi"])
    assert list(sub.index) == ["ip", "pmi", "total"]
    # On 2020-05-15 pmi has April (delay 1); ip April needs 40 days (June 9).
    vintage = ragged.released_share("2020Q2", series=["ip", "pmi"], as_of="2020-05-15")
    assert vintage.loc["ip", "released"] == 0
    assert vintage.loc["pmi", "released"] == 1
    with pytest.raises(NowcastDataError):
        MixedFrequencyData(ragged.to_frame()[["pmi"]], "M").released_share("2020Q2", as_of="2020")


def test_period_beyond_grid_counts_as_pending(ragged):
    table = ragged.released_share("2020Q3", series=["pmi", "gdp"])
    assert table.loc["pmi", "expected"] == 3
    assert table.loc["pmi", "released"] == 0
    assert table.loc["total", "share"] == 0.0


def test_monthly_period_and_errors(ragged):
    table = ragged.released_share("2020-06")
    assert table.loc["gdp", "expected"] == 1
    assert table.loc["ip", "expected"] == 1
    assert np.isnan(ragged.released_share("2020-05").loc["gdp", "share"])
    with pytest.raises(NowcastDataError, match="higher frequency"):
        ragged.released_share(pd.Period("2020-05-04", freq="D"))


def test_annual_series_without_slots_in_quarter():
    idx = pd.period_range("2020-01", periods=12, freq="M")
    df = pd.DataFrame({"a": [np.nan] * 11 + [1.0], "m": np.ones(12)}, index=idx)
    mfd = MixedFrequencyData(df, {"a": "A", "m": "M"})
    q2 = mfd.released_share("2020Q2")
    assert q2.loc["a", "expected"] == 0
    assert np.isnan(q2.loc["a", "share"])
    assert q2.loc["total", "share"] == 1.0
    assert mfd.released_share("2020Q4").loc["a", "share"] == 1.0
    zero = mfd.released_share("2020Q2", weights={"a": 1.0})
    assert np.isnan(zero.loc["total", "share"])


def test_weekly_grid_calendar_convention():
    idx = pd.period_range("2020-01-06", periods=10, freq="W")
    df = pd.DataFrame({"w": np.r_[np.ones(7), [np.nan] * 3]}, index=idx)
    mfd = MixedFrequencyData(df, "W")
    table = mfd.released_share("2020-02")
    # weeks ending in February 2020: Feb 2, 9, 16, 23 (Mar 1 belongs to March)
    assert table.loc["w", "expected"] == 4
    assert table.loc["w", "released"] == 4


def test_total_name_clash():
    idx = pd.period_range("2020-01", periods=3, freq="M")
    mfd = MixedFrequencyData(pd.DataFrame({"total": [1.0, 2.0, 3.0]}, index=idx), "M")
    with pytest.raises(ValueError, match="clashes"):
        mfd.released_share("2020Q1")


def test_series_groups_helper(ragged):
    groups = _series_groups(ragged.metadata, None)
    assert groups["ip"] == ["ip"]
    assert _series_groups(ragged.metadata, "series") == groups


def test_existing_behaviour_unchanged(ragged):
    before = ragged.to_frame()
    ragged.released_share("2020Q2", as_of="2020-05-15")
    pd.testing.assert_frame_equal(ragged.to_frame(), before)
