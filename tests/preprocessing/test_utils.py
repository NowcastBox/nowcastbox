"""Tests for the private helpers of nowcastbox.preprocessing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nowcastbox.core.exceptions import NowcastDataError
from nowcastbox.core.frequency import Frequency
from nowcastbox.preprocessing._utils import NativeView, per_series, series_frequency


def test_native_view_requires_series():
    with pytest.raises(NowcastDataError, match="pandas Series"):
        NativeView(np.arange(3.0))  # type: ignore[arg-type]


def test_native_view_empty_series():
    s = pd.Series([], index=pd.PeriodIndex([], freq="M"), dtype=float)
    view = NativeView(s, "M")
    assert len(view.native) == 0
    assert len(view.to_original(np.array([]))) == 0


def test_native_view_without_slots():
    s = pd.Series([np.nan], index=pd.period_range("2020-01", periods=1, freq="M"))
    view = NativeView(s, "Q")
    assert len(view.native) == 0
    assert view.frequency is Frequency.QUARTERLY
    out = view.to_original(np.array([]))
    assert out.isna().all()


def test_native_view_round_trip_quarterly():
    idx = pd.period_range("2020-02", periods=8, freq="M")
    s = pd.Series(np.nan, index=idx, name="q")
    s.iloc[[1, 4, 7]] = [1.0, 2.0, 3.0]
    view = NativeView(s, "Q")
    assert view.native.index.freqstr.startswith("Q")
    assert view.native.tolist() == [1.0, 2.0, 3.0]
    pd.testing.assert_series_equal(view.to_original(view.native), s)


def test_series_frequency():
    s = pd.Series([1.0, 2.0], index=pd.period_range("2020Q1", periods=2, freq="Q"))
    assert series_frequency(s, None) is Frequency.QUARTERLY
    assert series_frequency(s, "A") is Frequency.ANNUAL
    with pytest.raises(NowcastDataError, match="higher"):
        series_frequency(s, "M")


def test_per_series():
    cols = ["a", "b"]
    assert per_series(None, cols, "x", default=0) == {"a": 0, "b": 0}
    assert per_series(5, cols, "x") == {"a": 5, "b": 5}
    assert per_series({"a": 1}, cols, "x") == {"a": 1, "b": None}
    assert per_series(pd.Series({"b": 2}), cols, "x") == {"a": None, "b": 2}
    assert per_series((1, 2), cols, "x") == {"a": 1, "b": 2}
    assert per_series(np.array([1, 2]), cols, "x") == {"a": 1, "b": 2}
    with pytest.raises(NowcastDataError, match="unknown"):
        per_series({"c": 1}, cols, "x")
    with pytest.raises(NowcastDataError, match="elements"):
        per_series([1], cols, "x")
