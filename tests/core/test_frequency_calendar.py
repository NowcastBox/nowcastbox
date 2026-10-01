"""Calendar-aware frequency helpers (innovation I1): weeks and days within months etc."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from nowcastbox.core.frequency import (
    AggregationType,
    Frequency,
    aggregation_ratio,
    base_to_native,
    calendar_position,
    infer_frequency,
    is_fixed_ratio,
    is_period_end,
    max_periods_per,
    native_period_bounds,
    native_to_base,
    period_to_base,
)

CALENDAR_PAIRS = [("D", "M"), ("D", "Q"), ("D", "A"), ("W", "M"), ("W", "Q"), ("W", "A")]


class TestRatios:
    def test_daily_weekly_ratio_is_fixed(self):
        assert aggregation_ratio("D", "W") == 7
        assert is_fixed_ratio("D", "W")

    @pytest.mark.parametrize(("high", "low"), [("M", "Q"), ("M", "A"), ("Q", "A"), ("W", "W")])
    def test_fixed_pairs(self, high, low):
        assert is_fixed_ratio(high, low)
        assert max_periods_per(high, low) == aggregation_ratio(high, low)

    @pytest.mark.parametrize(("high", "low"), CALENDAR_PAIRS)
    def test_calendar_pairs_are_not_fixed(self, high, low):
        assert not is_fixed_ratio(high, low)
        with pytest.raises(ValueError, match="not fixed"):
            aggregation_ratio(high, low)

    @pytest.mark.parametrize(
        ("high", "low", "expected"),
        [("D", "M", 31), ("D", "Q", 92), ("D", "A", 366), ("W", "M", 5), ("W", "Q", 14)],
    )
    def test_max_periods(self, high, low, expected):
        assert max_periods_per(high, low) == expected

    def test_max_periods_bounds_the_calendar(self):
        weeks = pd.period_range("1990-01-07", periods=52 * 60, freq="W")
        for low in ("M", "Q", "A"):
            _, length = calendar_position(weeks, low)
            assert length.max() == max_periods_per("W", low)
        days = pd.period_range("1996-01-01", periods=366 * 8, freq="D")
        for low in ("M", "Q", "A"):
            _, length = calendar_position(days, low)
            assert length.max() == max_periods_per("D", low)

    def test_swapped_arguments(self):
        with pytest.raises(ValueError, match="swap"):
            is_fixed_ratio("Q", "W")
        with pytest.raises(ValueError, match="swap"):
            max_periods_per("Q", "D")


class TestNativeBounds:
    def test_week_belongs_to_period_of_its_last_day(self):
        # 2020-01-27/2020-02-02 ends in February: first week of February
        first, last = native_period_bounds(pd.PeriodIndex(["2020-01", "2020-02"], freq="M"), "W")
        assert first.astype(str).tolist() == ["2019-12-30/2020-01-05", "2020-01-27/2020-02-02"]
        assert last.astype(str).tolist() == ["2020-01-20/2020-01-26", "2020-02-17/2020-02-23"]

    def test_daily_bounds(self):
        first, last = native_period_bounds(pd.PeriodIndex(["2020Q1"], freq="Q"), "D")
        assert (str(first[0]), str(last[0])) == ("2020-01-01", "2020-03-31")

    def test_monthly_bounds_unchanged(self):
        first, last = native_period_bounds(pd.period_range("2020Q1", periods=2, freq="Q"), "M")
        assert first.astype(str).tolist() == ["2020-01", "2020-04"]
        assert last.astype(str).tolist() == ["2020-03", "2020-06"]

    def test_lower_base_rejected(self):
        with pytest.raises(ValueError, match="Cannot map"):
            native_period_bounds(pd.period_range("2020-01", periods=2, freq="M"), "Q")

    def test_period_to_base_weekly(self):
        assert str(period_to_base("2020Q1", "W")) == "2020-03-23/2020-03-29"
        assert str(period_to_base("2020Q1", "W", how="start")) == "2019-12-30/2020-01-05"
        assert period_to_base("2020-05-04", "D") == pd.Period("2020-05-04", "D")

    def test_native_to_base_weekly(self):
        months = pd.period_range("2020-01", periods=3, freq="M")
        assert native_to_base(months, "W").astype(str).tolist() == [
            "2020-01-20/2020-01-26",
            "2020-02-17/2020-02-23",
            "2020-03-23/2020-03-29",
        ]


class TestCalendarPosition:
    def test_weeks_in_months(self):
        idx = pd.period_range("2020-01-27", periods=6, freq="W")
        pos, length = calendar_position(idx, "M")
        assert pos.tolist() == [0, 1, 2, 3, 0, 1]
        assert length.tolist() == [4, 4, 4, 4, 5, 5]

    @pytest.mark.parametrize(("high", "low"), [*CALENDAR_PAIRS, ("M", "Q"), ("D", "W")])
    def test_slots_partition_the_grid(self, high, low):
        start = "2019-06-03" if high in ("D", "W") else "2019-06"
        idx = pd.period_range(start, periods=900 if high == "D" else 200, freq=high)
        pos, length = calendar_position(idx, low)
        slots = is_period_end(idx, low)
        assert np.array_equal(slots, pos == length - 1)
        native = base_to_native(idx, low)
        # every complete native period has exactly one slot, its last base period
        assert bool(np.all(native[slots][1:] != native[slots][:-1]))
        assert bool(np.all(pos >= 0)) and bool(np.all(pos < length))

    def test_is_period_end_weekly_months(self):
        idx = pd.period_range("2020-01-05", periods=9, freq="W")
        assert idx[is_period_end(idx, "M")].astype(str).tolist() == [
            "2020-01-20/2020-01-26",
            "2020-02-17/2020-02-23",
        ]


class TestInferCalendarFrequency:
    def test_monthly_and_quarterly_on_weekly_grid(self):
        idx = pd.period_range("2018-01-07", periods=160, freq="W")
        m = pd.Series(np.where(is_period_end(idx, "M"), 1.0, np.nan), index=idx)
        q = pd.Series(np.where(is_period_end(idx, "Q"), 1.0, np.nan), index=idx)
        w = pd.Series(np.arange(160.0), index=idx)
        assert infer_frequency(m) is Frequency.MONTHLY
        assert infer_frequency(q) is Frequency.QUARTERLY
        assert infer_frequency(w) is Frequency.WEEKLY

    def test_weekly_on_daily_grid(self):
        idx = pd.period_range("2020-01-01", periods=70, freq="D")
        s = pd.Series(np.where(is_period_end(idx, "W"), 1.0, np.nan), index=idx)
        assert infer_frequency(s) is Frequency.WEEKLY


class TestCalendarWeights:
    @pytest.mark.parametrize("kind", list(AggregationType))
    @pytest.mark.parametrize("k", [1, 3, 4, 5, 13])
    def test_equal_lengths_match_fixed_weights(self, kind, k):
        expected = kind.weights(k, normalize=True)
        assert np.allclose(kind.calendar_weights(k, k), expected)
        assert np.allclose(kind.calendar_weights(k), expected)

    def test_flow_average_stock(self):
        assert AggregationType.FLOW.calendar_weights(5, 4).tolist() == [1.0] * 5
        assert np.allclose(AggregationType.AVERAGE.calendar_weights(4, 5), 0.25)
        assert AggregationType.STOCK.calendar_weights(4).tolist() == [1.0, 0.0, 0.0, 0.0]

    def test_growth_rate_unequal_periods(self):
        w = AggregationType.GROWTH_RATE.calendar_weights(5, 4)
        assert w.size == 5 + 4 - 1
        assert np.allclose(w[:5], np.arange(1, 6) / 5)
        assert np.allclose(w[5:], np.array([3, 2, 1]) / 4)

    @given(n=st.integers(1, 20), n_prev=st.integers(1, 20), seed=st.integers(0, 1000))
    def test_growth_rate_is_exact(self, n, n_prev, seed):
        """Weights map high-frequency growth rates to the growth of the period average."""
        rng = np.random.default_rng(seed)
        level = np.cumsum(rng.standard_normal(n_prev + n + 1))
        growth = np.diff(level)  # x_s = level_s - level_{s-1}
        current, previous = level[-n:], level[-n - n_prev : -n]
        target = current.mean() - previous.mean()
        w = AggregationType.GROWTH_RATE.calendar_weights(n, n_prev)
        assert float(w @ growth[::-1][: w.size]) == pytest.approx(target, abs=1e-10)

    @pytest.mark.parametrize("bad", [0, -1, 2.5, True])
    def test_invalid_lengths(self, bad):
        with pytest.raises(ValueError, match="positive integer"):
            AggregationType.FLOW.calendar_weights(bad)
        with pytest.raises(ValueError, match="positive integer"):
            AggregationType.GROWTH_RATE.calendar_weights(3, bad)
