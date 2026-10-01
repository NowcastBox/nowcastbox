from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from nowcastbox.core.data import MixedFrequencyData
from nowcastbox.core.exceptions import DataQualityWarning, NowcastDataError
from nowcastbox.core.frequency import Frequency
from nowcastbox.data_sources._parsing import (
    build_series,
    check_range,
    combine_series,
    filter_period_range,
    infer_native_frequency,
    normalize_codes,
    resolve_native_frequency,
    to_timestamp,
)


class TestToTimestamp:
    @pytest.mark.parametrize(
        ("value", "end", "expected"),
        [
            ("2020-03-15", False, "2020-03-15"),
            ("2020-03", False, "2020-03-01"),
            ("2020-03", True, "2020-03-31"),
            ("2020Q2", False, "2020-04-01"),
            ("2020Q2", True, "2020-06-30"),
            ("2021", True, "2021-12-31"),
            (pd.Period("2020-02", "M"), True, "2020-02-29"),
            (dt.date(2020, 1, 5), False, "2020-01-05"),
            (dt.datetime(2020, 1, 5, 13, 30), False, "2020-01-05"),
            (pd.Timestamp("2020-01-05 10:00"), True, "2020-01-05"),
        ],
    )
    def test_conversions(self, value, end, expected):
        assert to_timestamp(value, end=end) == pd.Timestamp(expected)

    def test_none(self):
        assert to_timestamp(None) is None

    @pytest.mark.parametrize("value", ["yesterday-ish", 3.5, "2020-13"])
    def test_invalid(self, value):
        with pytest.raises(ValueError):
            to_timestamp(value)

    def test_check_range(self):
        check_range(None, pd.Timestamp("2020-01-01"))
        with pytest.raises(ValueError, match="after"):
            check_range(pd.Timestamp("2021-01-01"), pd.Timestamp("2020-01-01"))


class TestInferFrequency:
    @pytest.mark.parametrize(
        ("dates", "expected"),
        [
            (pd.date_range("2020-01-01", periods=5, freq="MS"), Frequency.MONTHLY),
            (pd.date_range("2020-01-31", periods=5, freq="ME"), Frequency.MONTHLY),
            (pd.date_range("2020-01-01", periods=5, freq="QS"), Frequency.QUARTERLY),
            (pd.date_range("2020-04-01", periods=5, freq="QS"), Frequency.QUARTERLY),
            (pd.date_range("2020-03-31", periods=5, freq="QE"), Frequency.QUARTERLY),
            (pd.date_range("2000-01-01", periods=5, freq="YS"), Frequency.ANNUAL),
            (pd.date_range("2020-01-01", periods=30, freq="B"), Frequency.DAILY),
            (pd.date_range("2020-01-01", periods=30, freq="D"), Frequency.DAILY),
            (pd.date_range("2020-01-03", periods=10, freq="W-FRI"), Frequency.WEEKLY),
        ],
    )
    def test_regular(self, dates, expected):
        assert infer_native_frequency(dates) == expected

    def test_gaps_use_gcd(self):
        dates = pd.to_datetime(["2020-01-01", "2020-02-01", "2020-05-01", "2020-06-01"])
        assert infer_native_frequency(dates) == Frequency.MONTHLY
        dates = pd.to_datetime(["2020-01-01", "2020-07-01", "2021-04-01"])
        assert infer_native_frequency(dates) == Frequency.QUARTERLY

    def test_unsorted_and_duplicated_dates(self):
        dates = pd.to_datetime(["2020-03-01", "2020-01-01", "2020-02-01", "2020-02-01"])
        assert infer_native_frequency(dates) == Frequency.MONTHLY

    def test_semiannual_unsupported(self):
        with pytest.raises(NowcastDataError, match="6 months"):
            infer_native_frequency(pd.date_range("2020-01-01", periods=4, freq="6MS"))

    def test_too_few(self):
        with pytest.raises(NowcastDataError, match="two observations"):
            infer_native_frequency(pd.to_datetime(["2020-01-01"]))

    @given(
        freq=st.sampled_from(
            [("MS", Frequency.MONTHLY), ("QS", Frequency.QUARTERLY), ("YS", Frequency.ANNUAL)]
        ),
        start=st.dates(min_value=dt.date(1950, 1, 1), max_value=dt.date(2040, 1, 1)),
        n=st.integers(min_value=2, max_value=60),
        data=st.data(),
    )
    def test_property_regular_with_holes(self, freq, start, n, data):
        alias, expected = freq
        dates = pd.date_range(start, periods=n, freq=alias)
        # keep a random subset that still contains two consecutive dates
        keep = data.draw(st.lists(st.booleans(), min_size=n, max_size=n))
        i = data.draw(st.integers(0, n - 2))
        keep[i] = keep[i + 1] = True
        assert infer_native_frequency(dates[np.array(keep)]) == expected


class TestBuildSeries:
    def test_basic(self):
        dates = pd.to_datetime(["2020-02-01", "2020-01-01", "2020-03-01"])
        s = build_series(dates, ["2", "1", ""], "x")
        assert s.index.freqstr == "M"
        assert s.name == "x"
        assert s.dtype == np.float64
        assert s.index.is_monotonic_increasing
        assert s.iloc[:2].tolist() == [1.0, 2.0]
        assert np.isnan(s.iloc[2])

    def test_explicit_frequency(self):
        s = build_series(pd.to_datetime(["2020-04-01"]), [5], "q", native_frequency="Q")
        assert s.index[0] == pd.Period("2020Q2", "Q")

    def test_non_numeric_warns(self):
        dates = pd.date_range("2020-01-01", periods=3, freq="MS")
        with pytest.warns(DataQualityWarning, match="non-numeric"):
            s = build_series(dates, ["1", "abc", None], "x")
        assert s.isna().tolist() == [False, True, True]

    def test_custom_missing_tokens(self):
        dates = pd.date_range("2020-01-01", periods=2, freq="MS")
        s = build_series(dates, [".", "1.5"], "x", missing_tokens=(".",))
        assert np.isnan(s.iloc[0])

    def test_length_mismatch(self):
        with pytest.raises(NowcastDataError, match="2 dates but 1 values"):
            build_series(pd.date_range("2020-01-01", periods=2, freq="MS"), [1], "x")

    def test_empty(self):
        s = build_series([], [], "x", native_frequency="Q")
        assert len(s) == 0 and s.index.freqstr.startswith("Q")
        assert build_series([], [], "x").index.freqstr == "M"

    def test_duplicates_consistent_are_merged(self):
        dates = pd.to_datetime(["2020-01-01", "2020-01-01", "2020-02-01"])
        s = build_series(dates, ["1", "", "2"], "x", native_frequency="M")
        assert s.tolist() == [1.0, 2.0]

    def test_duplicates_conflicting_raise(self):
        dates = pd.to_datetime(["2020-01-01", "2020-01-15", "2020-02-01"])
        with pytest.raises(NowcastDataError, match="several different values"):
            build_series(dates, [1, 2, 3], "x", native_frequency="M")


class TestFilter:
    def test_filter(self):
        idx = pd.period_range("2020Q1", periods=4, freq="Q")
        s = pd.Series([1.0, 2, 3, 4], index=idx)
        out = filter_period_range(s, pd.Timestamp("2020-04-01"), pd.Timestamp("2020-09-30"))
        assert out.index.astype(str).tolist() == ["2020Q2", "2020Q3"]
        assert filter_period_range(s, None, None).equals(s)
        empty = s.iloc[:0]
        assert filter_period_range(empty, pd.Timestamp("2020-01-01"), None) is empty


class TestCombine:
    def test_same_frequency_contiguous(self):
        a = pd.Series([1.0, 3.0], index=pd.PeriodIndex(["2020-01", "2020-03"], freq="M"))
        b = pd.Series([2.0], index=pd.PeriodIndex(["2020-02"], freq="M"))
        df = combine_series({"b": b, "a": a})
        assert df.columns.tolist() == ["b", "a"]
        assert df.index.astype(str).tolist() == ["2020-01", "2020-02", "2020-03"]
        assert df.index.name == "period"
        assert df.loc[pd.Period("2020-02", "M"), "b"] == 2.0

    def test_daily_not_completed(self):
        idx = pd.PeriodIndex(["2020-01-03", "2020-01-06"], freq="D")
        df = combine_series({"x": pd.Series([1.0, 2.0], index=idx)})
        assert len(df) == 2

    def test_mixed_without_base_raises(self):
        m = pd.Series([1.0], index=pd.PeriodIndex(["2020-01"], freq="M"))
        q = pd.Series([1.0], index=pd.PeriodIndex(["2020Q1"], freq="Q"))
        with pytest.raises(NowcastDataError, match="base_frequency"):
            combine_series({"m": m, "q": q})

    def test_mixed_on_monthly_grid_is_mfd_compatible(self):
        m = pd.Series(np.arange(6.0), index=pd.period_range("2020-01", periods=6, freq="M"))
        q = pd.Series([10.0, 20.0], index=pd.period_range("2020Q1", periods=2, freq="Q"))
        a = pd.Series([5.0], index=pd.PeriodIndex(["2020"], freq="Y"))
        df = combine_series({"m": m, "q": q, "a": a}, base_frequency="M")
        assert df.index.freqstr == "M"
        assert df["q"].dropna().index.astype(str).tolist() == ["2020-03", "2020-06"]
        assert df["a"].dropna().index.astype(str).tolist() == ["2020-12"]
        mfd = MixedFrequencyData(df, {"m": "M", "q": "Q", "a": "A"})
        assert mfd.to_native("q").dropna().tolist() == [10.0, 20.0]

    def test_higher_than_base_raises(self):
        d = pd.Series([1.0], index=pd.PeriodIndex(["2020-01-01"], freq="D"))
        with pytest.raises(NowcastDataError, match="aggregate"):
            combine_series({"d": d}, base_frequency="M")

    def test_empty(self):
        with pytest.raises(NowcastDataError):
            combine_series({})


class TestNormalizeCodes:
    def test_forms(self):
        name = lambda c: f"s_{c}"  # noqa: E731
        assert normalize_codes(433, default_name=name) == {"s_433": "433"}
        assert normalize_codes("ABC", default_name=name) == {"s_ABC": "ABC"}
        assert normalize_codes([1, 2], default_name=name) == {"s_1": "1", "s_2": "2"}
        assert normalize_codes({"ipca": 433}, default_name=name) == {"ipca": "433"}

    def test_errors(self):
        name = str
        with pytest.raises(ValueError, match="At least one"):
            normalize_codes([], default_name=name)
        with pytest.raises(ValueError, match="Duplicated"):
            normalize_codes([1, 1], default_name=name)
        with pytest.raises(TypeError):
            normalize_codes({"": 1}, default_name=name)
        with pytest.raises(TypeError):
            normalize_codes({3: 1}, default_name=name)
        with pytest.raises(ValueError, match="Invalid series code"):
            normalize_codes({"x": None}, default_name=name)
        with pytest.raises(ValueError, match="Invalid series code"):
            normalize_codes({"x": True}, default_name=name)

    def test_resolve_native_frequency(self):
        assert resolve_native_frequency(None, "x") is None
        assert resolve_native_frequency("Q", "x") == Frequency.QUARTERLY
        assert resolve_native_frequency({"x": 12}, "x") == Frequency.MONTHLY
        assert resolve_native_frequency({"y": 12}, "x") is None


def test_non_period_index_rejected():
    s = pd.Series([1.0], index=pd.DatetimeIndex(["2020-01-01"]))
    with pytest.raises(TypeError, match="PeriodIndex"):
        combine_series({"x": s})
    with pytest.raises(TypeError, match="PeriodIndex"):
        filter_period_range(s, None, None)
