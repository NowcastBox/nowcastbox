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
    infer_frequency,
    is_period_end,
    native_to_base,
    period_to_base,
)


class TestFrequency:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("M", Frequency.MONTHLY),
            ("monthly", Frequency.MONTHLY),
            (" Month ", Frequency.MONTHLY),
            ("ME", Frequency.MONTHLY),
            ("q", Frequency.QUARTERLY),
            ("Q-DEC", Frequency.QUARTERLY),
            ("QE-DEC", Frequency.QUARTERLY),
            ("quarterly", Frequency.QUARTERLY),
            ("A", Frequency.ANNUAL),
            ("Y", Frequency.ANNUAL),
            ("A-DEC", Frequency.ANNUAL),
            ("YE-DEC", Frequency.ANNUAL),
            ("annual", Frequency.ANNUAL),
            ("W", Frequency.WEEKLY),
            ("W-SUN", Frequency.WEEKLY),
            ("D", Frequency.DAILY),
            ("B", Frequency.DAILY),
            (12, Frequency.MONTHLY),
            (4, Frequency.QUARTERLY),
            (1, Frequency.ANNUAL),
            (52, Frequency.WEEKLY),
            (365, Frequency.DAILY),
            (252, Frequency.DAILY),
            (12.0, Frequency.MONTHLY),
            (np.int64(4), Frequency.QUARTERLY),
            (Frequency.MONTHLY, Frequency.MONTHLY),
        ],
    )
    def test_from_value(self, value, expected):
        assert Frequency.from_value(value) is expected

    @pytest.mark.parametrize("value", ["", "hourly", "X-DEC", 7, 12.5, True, None, 3.0, [12]])
    def test_from_value_invalid(self, value):
        with pytest.raises(ValueError):
            Frequency.from_value(value)

    def test_properties(self):
        assert [f.periods_per_year for f in Frequency] == [365, 52, 12, 4, 1]
        assert Frequency.QUARTERLY.pandas_freq == "Q"
        assert Frequency.ANNUAL.pandas_freq == "Y"
        assert Frequency.MONTHLY.label == "monthly"
        assert Frequency.MONTHLY == "M"

    def test_pandas_aliases_are_valid_period_freqs(self):
        for f in Frequency:
            p = pd.Period("2020-06-15", freq=f.pandas_freq)
            assert Frequency.from_value(p.freqstr) is f

    def test_ordering(self):
        assert Frequency.QUARTERLY.is_lower_than("M")
        assert not Frequency.MONTHLY.is_lower_than("M")
        assert Frequency.DAILY.is_higher_than(Frequency.WEEKLY)
        assert not Frequency.ANNUAL.is_higher_than("Q")

    def test_from_index(self):
        assert Frequency.from_index(pd.period_range("2020", periods=2, freq="Y")) is (
            Frequency.ANNUAL
        )
        with pytest.raises(TypeError):
            Frequency.from_index(pd.date_range("2020", periods=2))  # type: ignore[arg-type]


class TestRatiosAndMapping:
    @pytest.mark.parametrize(
        ("high", "low", "ratio"),
        [("M", "Q", 3), ("M", "A", 12), ("Q", "A", 4), ("M", "M", 1), ("D", "D", 1)],
    )
    def test_aggregation_ratio(self, high, low, ratio):
        assert aggregation_ratio(high, low) == ratio

    def test_aggregation_ratio_errors(self):
        with pytest.raises(ValueError, match="swap"):
            aggregation_ratio("Q", "M")
        with pytest.raises(ValueError, match="not fixed"):
            aggregation_ratio("D", "M")
        with pytest.raises(ValueError, match="not fixed"):
            aggregation_ratio("W", "Q")

    def test_period_to_base(self):
        assert period_to_base("2020Q1", "M") == pd.Period("2020-03", "M")
        assert period_to_base("2020Q1", "M", how="start") == pd.Period("2020-01", "M")
        assert period_to_base(pd.Period("2021", "Y"), "Q") == pd.Period("2021Q4", "Q")
        assert period_to_base("2020-05", "M") == pd.Period("2020-05", "M")

    def test_period_to_base_errors(self):
        with pytest.raises(ValueError, match="how"):
            period_to_base("2020Q1", "M", how="middle")
        with pytest.raises(ValueError, match="base grid"):
            period_to_base("2020-01", "Q")

    def test_base_to_native_and_back(self):
        idx = pd.period_range("2019-01", periods=24, freq="M")
        q = base_to_native(idx, "Q")
        assert q.freqstr.startswith("Q")
        assert (q[2::3] == pd.period_range("2019Q1", periods=8, freq="Q")).all()
        with pytest.raises(ValueError, match="higher frequency"):
            base_to_native(idx, "D")

    def test_native_to_base(self):
        q = pd.period_range("2020Q1", periods=3, freq="Q")
        assert native_to_base(q, "M").astype(str).tolist() == ["2020-03", "2020-06", "2020-09"]
        with pytest.raises(ValueError, match="Cannot map"):
            native_to_base(pd.period_range("2020-01", periods=2, freq="M"), "Q")

    def test_is_period_end(self):
        idx = pd.period_range("2020-01", periods=24, freq="M")
        assert is_period_end(idx, "M").all()
        assert is_period_end(idx, "Q").sum() == 8
        annual = is_period_end(idx, "A")
        assert annual.sum() == 2
        assert idx[annual].month.tolist() == [12, 12]

    def test_is_period_end_quarterly_base(self):
        idx = pd.period_range("2020Q1", periods=8, freq="Q")
        assert is_period_end(idx, "A").tolist() == [False, False, False, True] * 2


class TestInferFrequency:
    def test_infers_quarterly_and_annual(self):
        idx = pd.period_range("2015-01", periods=48, freq="M")
        q = pd.Series(np.where(idx.month % 3 == 0, 1.0, np.nan), index=idx)
        a = pd.Series(np.where(idx.month == 12, 1.0, np.nan), index=idx)
        m = pd.Series(np.arange(48.0), index=idx)
        assert infer_frequency(q) is Frequency.QUARTERLY
        assert infer_frequency(a) is Frequency.ANNUAL
        assert infer_frequency(m) is Frequency.MONTHLY

    def test_few_observations_default_to_base(self):
        idx = pd.period_range("2015-01", periods=12, freq="M")
        s = pd.Series(np.nan, index=idx)
        assert infer_frequency(s) is Frequency.MONTHLY
        s.iloc[2] = 1.0
        assert infer_frequency(s) is Frequency.MONTHLY
        assert infer_frequency(s, min_observations=1) is Frequency.QUARTERLY

    def test_requires_period_index(self):
        with pytest.raises(TypeError):
            infer_frequency(pd.Series([1.0, 2.0]))

    def test_daily_base_has_no_fixed_lower_candidate(self):
        idx = pd.period_range("2020-01-01", periods=10, freq="D")
        assert infer_frequency(pd.Series(1.0, index=idx)) is Frequency.DAILY


class TestAggregationType:
    @pytest.mark.parametrize(
        ("alias", "member"),
        [
            ("flow", AggregationType.FLOW),
            ("SUM", AggregationType.FLOW),
            ("stock", AggregationType.STOCK),
            ("end", AggregationType.STOCK),
            ("mean", AggregationType.AVERAGE),
            ("mariano_murasawa", AggregationType.GROWTH_RATE),
            ("Mariano-Murasawa", AggregationType.GROWTH_RATE),
            ("mm", AggregationType.GROWTH_RATE),
            ("growth rate", AggregationType.GROWTH_RATE),
            (AggregationType.STOCK, AggregationType.STOCK),
        ],
    )
    def test_from_value(self, alias, member):
        assert AggregationType.from_value(alias) is member

    @pytest.mark.parametrize("value", ["median", 3, None])
    def test_from_value_invalid(self, value):
        with pytest.raises(ValueError, match="Unknown aggregation"):
            AggregationType.from_value(value)

    def test_weights(self):
        np.testing.assert_array_equal(AggregationType.GROWTH_RATE.weights(3), [1, 2, 3, 2, 1])
        np.testing.assert_allclose(
            AggregationType.GROWTH_RATE.weights(3, normalize=True), np.array([1, 2, 3, 2, 1]) / 3
        )
        np.testing.assert_array_equal(AggregationType.FLOW.weights(3), [1, 1, 1])
        np.testing.assert_allclose(AggregationType.AVERAGE.weights(3), [1 / 3] * 3)
        np.testing.assert_array_equal(AggregationType.STOCK.weights(3), [1, 0, 0])
        assert AggregationType.GROWTH_RATE.n_lags(3) == 4
        assert AggregationType.FLOW.n_lags(12) == 11

    @pytest.mark.parametrize("ratio", [0, -1, 2.5, True])
    def test_weights_invalid_ratio(self, ratio):
        with pytest.raises(ValueError, match="positive integer"):
            AggregationType.FLOW.weights(ratio)

    def test_mariano_murasawa_is_growth_of_flow(self):
        """Exact identity: the k-period difference of a k-period sum of levels equals the
        triangular-weighted sum of one-period differences (log-linear MM approximation)."""
        rng = np.random.default_rng(1)
        k = 3
        x = np.cumsum(rng.normal(size=40))  # log levels of the monthly series
        dx = np.diff(x)
        flow = np.convolve(x, np.ones(k), mode="valid")  # rolling k-sum, aligned at the end
        lhs = flow[k:] - flow[:-k]  # change between consecutive (overlapping) periods
        w = AggregationType.GROWTH_RATE.weights(k)
        rhs = np.convolve(dx, w, mode="valid")
        np.testing.assert_allclose(lhs, rhs)

    @given(st.integers(min_value=1, max_value=24))
    def test_weight_properties(self, k):
        tri = AggregationType.GROWTH_RATE.weights(k)
        assert len(tri) == 2 * k - 1
        np.testing.assert_allclose(tri, tri[::-1])
        assert tri.sum() == pytest.approx(k * k)
        assert AggregationType.AVERAGE.weights(k).sum() == pytest.approx(1.0)
        assert AggregationType.FLOW.weights(k).sum() == pytest.approx(k)
        assert AggregationType.STOCK.weights(k).sum() == pytest.approx(1.0)


@given(
    year=st.integers(min_value=1950, max_value=2100),
    quarter=st.integers(min_value=1, max_value=4),
)
def test_quarter_roundtrip_through_base(year, quarter):
    q = pd.Period(year=year, quarter=quarter, freq="Q")
    m = period_to_base(q, "M")
    assert m.month == 3 * quarter
    assert m.asfreq("Q") == q
    start = period_to_base(q, "M", how="start")
    assert (m - start).n == 2
