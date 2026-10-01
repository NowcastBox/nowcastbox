"""Regression tests from the final software-quality audit (vintages)."""

from __future__ import annotations

import pandas as pd
import pytest

from nowcastbox.vintages import generate_vintages, pseudo_real_time


@pytest.fixture
def frame() -> pd.DataFrame:
    idx = pd.period_range("2020-01", periods=4, freq="M")
    return pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]}, index=idx)


def test_generate_vintages_accepts_delay_alias(frame: pd.DataFrame) -> None:
    # same keyword as pseudo_real_time(data, delay=..., vintage=...)
    by_delay = list(generate_vintages(frame, start="2020-02-01", end="2020-05-01", delay=5))
    by_calendar = list(generate_vintages(frame, 5, "2020-02-01", "2020-05-01"))
    assert [v.date for v in by_delay] == [v.date for v in by_calendar]
    assert all(a.data.equals(b.data) for a, b in zip(by_delay, by_calendar, strict=True))
    single = pseudo_real_time(frame, delay=5, vintage="2020-03-01")
    assert by_delay[1].data.equals(single)


def test_generate_vintages_rejects_delay_and_calendar(frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="delay"):
        list(generate_vintages(frame, 5, "2020-02-01", "2020-05-01", delay=5))
