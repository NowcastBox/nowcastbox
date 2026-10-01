"""Calendar-aware aggregation weights for variable-ratio frequency pairs (innovation I1)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nowcastbox.core.data import MixedFrequencyData
from nowcastbox.core.frequency import (
    AggregationType,
    base_to_native,
    calendar_position,
    is_period_end,
)
from nowcastbox.preprocessing.aggregation import (
    CalendarAggregation,
    aggregate_panel,
    aggregation_weights,
    calendar_aggregation,
    calendar_weight_matrix,
    to_higher_frequency,
    to_lower_frequency,
)

PAIRS = [
    ("D", "M", "2020-01-01", 800),
    ("W", "M", "2019-12-01", 200),
    ("W", "Q", "2019-12-01", 200),
    ("D", "Q", "2020-01-01", 800),
    ("M", "Q", "2019-01", 60),
    ("Q", "A", "2015Q1", 40),
]


def grid(high: str, start: str, n: int) -> pd.PeriodIndex:
    return pd.period_range(start, periods=n, freq=high)


@pytest.mark.parametrize(("high", "low", "start", "n"), PAIRS)
class TestWeightMatrixPerPair:
    def test_flow_sums_every_period_of_the_low_period(self, high, low, start, n):
        idx = grid(high, start, n)
        W = calendar_weight_matrix(idx, low, "flow")
        pos, length = calendar_position(idx, low)
        slots = is_period_end(idx, low)
        assert np.allclose(W.sum(axis=1), pos + 1)
        assert np.allclose(W[slots].sum(axis=1), length[slots])
        assert W.shape[1] == calendar_aggregation(high, low, "flow").max_periods

    def test_average_weights_sum_to_one(self, high, low, start, n):
        W = calendar_weight_matrix(grid(high, start, n), low, "average")
        assert np.allclose(W.sum(axis=1), 1.0)

    def test_stock_is_last_value(self, high, low, start, n):
        W = calendar_weight_matrix(grid(high, start, n), low, "stock")
        assert np.allclose(W[:, 0], 1.0)
        assert np.allclose(W[:, 1:], 0.0)

    def test_growth_rate_is_exact_growth_of_the_average(self, high, low, start, n):
        """``sum_l w_l x_{t-l}`` equals the change of the period mean of the log level."""
        idx = grid(high, start, n)
        rng = np.random.default_rng(1)
        level = pd.Series(np.cumsum(rng.standard_normal(n)), index=idx)
        growth = level.diff()
        agg = calendar_aggregation(high, low)
        out = agg.apply(growth)
        slots = is_period_end(idx, low)
        native = base_to_native(idx, low)
        means = level.groupby(native).mean()
        counts = level.groupby(native).size()
        _, length = calendar_position(idx, low)
        full = pd.Series(length, index=native).groupby(level=0).first()
        complete = counts == full
        expected = means.diff().where(complete & complete.shift(1, fill_value=False))
        got = pd.Series(out.to_numpy()[slots], index=native[slots])
        both = expected.reindex(got.index).notna() & got.notna()
        assert int(both.sum()) >= 3
        assert np.allclose(got[both], expected.reindex(got.index)[both], atol=1e-10)


@pytest.mark.parametrize(("high", "low", "start", "n"), [p for p in PAIRS if p[0] in "MQ"])
def test_matches_fixed_weights_for_fixed_pairs(high, low, start, n):
    idx = grid(high, start, n)
    agg = calendar_aggregation(high, low)
    assert agg.is_fixed
    slots = is_period_end(idx, low)
    W = agg.weight_matrix(idx)
    assert np.allclose(W[slots][1:], aggregation_weights(high, low, normalize=True))


class TestCalendarAggregation:
    def test_properties(self):
        agg = calendar_aggregation("W", "Q")
        assert isinstance(agg, CalendarAggregation)
        assert (agg.max_periods, agg.n_weights) == (14, 27)
        assert not agg.is_fixed
        assert agg.reference_weights.size == 27
        assert calendar_aggregation("W", "Q", "flow").n_weights == 14
        assert calendar_aggregation("M", "Q").is_fixed

    def test_partial_periods_use_running_aggregate(self):
        idx = pd.period_range("2020-01-06", periods=8, freq="W")
        W = calendar_aggregation("W", "M", "average").weight_matrix(idx)
        assert np.allclose(W[0, :2], [0.5, 0.5])  # 2nd week of January (4 weeks)
        assert np.allclose(W[2, :4], 0.25)  # last week of January

    def test_growth_rate_weights_use_previous_length(self):
        # 2020-02 has 4 weeks, 2020-03 has 5: last week of March
        idx = pd.period_range("2020-01-27", periods=9, freq="W")
        W = calendar_aggregation("W", "M").weight_matrix(idx)
        slot = int(np.flatnonzero(is_period_end(idx, "M"))[-1])
        expected = AggregationType.GROWTH_RATE.calendar_weights(5, 4)
        assert np.allclose(W[slot, : expected.size], expected)
        assert np.allclose(W[slot, expected.size :], 0.0)

    def test_wrong_index_frequency(self):
        with pytest.raises(ValueError, match="weekly"):
            calendar_aggregation("W", "M").weight_matrix(
                pd.period_range("2020-01", periods=3, freq="M")
            )

    def test_empty_index(self):
        empty = pd.PeriodIndex([], freq="W")
        assert calendar_aggregation("W", "M").weight_matrix(empty).shape == (0, 9)

    def test_invalid_pair(self):
        with pytest.raises(ValueError, match="swap"):
            calendar_aggregation("Q", "W")

    def test_apply_dataframe_and_missing(self):
        idx = pd.period_range("2021-02-01", periods=31, freq="D")
        frame = pd.DataFrame({"a": np.ones(31), "b": np.ones(31)}, index=idx)
        frame.iloc[3, 1] = np.nan
        out = calendar_aggregation("D", "M", "flow").apply(frame)
        assert float(out.loc[pd.Period("2021-02-28", "D"), "a"]) == 28.0
        assert np.isnan(out.loc[pd.Period("2021-02-28", "D"), "b"])
        assert float(out.loc[pd.Period("2021-03-03", "D"), "b"]) == 3.0

    def test_apply_requires_period_index(self):
        from nowcastbox.core.exceptions import NowcastDataError

        with pytest.raises(NowcastDataError):
            calendar_aggregation("D", "M").apply(pd.Series([1.0, 2.0]))


class TestConversions:
    def test_weekly_to_monthly_last_and_mean(self):
        idx = pd.period_range("2020-01-05", periods=13, freq="W")
        s = pd.Series(np.arange(13.0), index=idx)
        last = to_lower_frequency(s, "M", "last")
        assert last.loc["2020-01"] == 3.0  # week ending 2020-01-26
        assert last.loc["2020-02"] == 7.0  # week ending 2020-02-23
        mean = to_lower_frequency(s, "M", "mean")
        assert mean.loc["2020-02"] == pytest.approx(np.mean([4.0, 5, 6, 7]))

    def test_weekly_to_quarterly_mariano_murasawa(self):
        idx = pd.period_range("2019-01-06", periods=80, freq="W")
        s = pd.Series(np.ones(80), index=idx)
        out = to_lower_frequency(s, "Q", "mariano_murasawa")
        # constant growth: growth of the average = sum of the weights (= (n + n') / 2)
        n = calendar_position(idx, "Q")[1]
        assert np.isfinite(out.loc["2019Q3"])
        slots = is_period_end(idx, "Q")
        assert out.loc["2019Q3"] == pytest.approx((n[slots][2] + n[slots][1]) / 2)

    def test_monthly_to_weekly_end(self):
        m = pd.Series([1.0, 2.0], index=pd.period_range("2020-01", periods=2, freq="M"))
        out = to_higher_frequency(m, "W", "end")
        assert str(out.index[0]) == "2019-12-30/2020-01-05"
        assert str(out.index[-1]) == "2020-02-17/2020-02-23"
        assert out.dropna().index.astype(str).tolist() == [
            "2020-01-20/2020-01-26",
            "2020-02-17/2020-02-23",
        ]

    def test_aggregate_panel_on_weekly_grid(self):
        idx = pd.period_range("2019-01-06", periods=60, freq="W")
        frame = pd.DataFrame(
            {
                "w": np.ones(60),
                "q": np.where(is_period_end(idx, "Q"), 1.0, np.nan),
            },
            index=idx,
        )
        panel = MixedFrequencyData(frame, {"w": "W", "q": "Q"})
        out = aggregate_panel(panel, "flow")
        slots = is_period_end(idx, "Q")
        _, length = calendar_position(idx, "Q")
        values = out["w"].to_numpy()
        assert np.allclose(values[slots], length[slots])
