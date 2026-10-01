"""MixedFrequencyData on weekly and daily base grids (innovation I1)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nowcastbox.core.data import MixedFrequencyData
from nowcastbox.core.exceptions import NowcastDataError
from nowcastbox.core.frequency import Frequency, is_period_end


def weekly_frame(n_weeks: int = 80, start: str = "2019-01-06") -> pd.DataFrame:
    idx = pd.period_range(start, periods=n_weeks, freq="W")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "claims": rng.standard_normal(n_weeks),
            "ip": np.where(is_period_end(idx, "M"), rng.standard_normal(n_weeks), np.nan),
            "gdp": np.where(is_period_end(idx, "Q"), rng.standard_normal(n_weeks), np.nan),
        },
        index=idx,
    )


class TestWeeklyBase:
    def test_frequencies_are_inferred(self):
        mfd = MixedFrequencyData(weekly_frame())
        assert mfd.base_frequency is Frequency.WEEKLY
        assert mfd.frequencies.to_dict() == {
            "claims": Frequency.WEEKLY,
            "ip": Frequency.MONTHLY,
            "gdp": Frequency.QUARTERLY,
        }

    def test_value_outside_slot_is_rejected(self):
        frame = weekly_frame()
        # the week 2019-01-28/2019-02-03 ends in February: not a January slot
        frame.loc[pd.Period("2019-01-28", "W"), "ip"] = 1.0
        with pytest.raises(NowcastDataError, match="outside the last weekly period"):
            MixedFrequencyData(frame, {"claims": "W", "ip": "M", "gdp": "Q"})

    def test_to_native(self):
        mfd = MixedFrequencyData(weekly_frame(), {"claims": "W", "ip": "M", "gdp": "Q"})
        gdp = mfd.to_native("gdp")
        assert gdp.index.freqstr.startswith("Q")
        assert gdp.index[0] == pd.Period("2019Q1", "Q")
        assert mfd.to_native("ip").index[0] == pd.Period("2019-01", "M")

    def test_truncate_with_quarters(self):
        mfd = MixedFrequencyData(weekly_frame(), {"claims": "W", "ip": "M", "gdp": "Q"})
        head = mfd.truncate(end="2019Q1")
        assert str(head.end) == "2019-03-25/2019-03-31"
        assert str(mfd.truncate(start="2019Q2").start) == "2019-04-01/2019-04-07"

    def test_as_of_uses_native_period_end(self):
        mfd = MixedFrequencyData(
            weekly_frame(),
            {"claims": "W", "ip": "M", "gdp": "Q"},
            release_delays={"claims": 4, "ip": 15, "gdp": 30},
        )
        vintage = mfd.as_of("2019-05-01")
        assert vintage.to_native("gdp", dropna=True).index.astype(str).tolist() == ["2019Q1"]
        assert vintage.to_native("ip", dropna=True).index[-1] == pd.Period("2019-03", "M")
        claims = vintage["claims"].dropna()
        assert claims.index[-1] == pd.Period("2019-04-21", "W")  # ends 04-21, + 4 days

    def test_frequency_ratio_is_not_fixed(self):
        mfd = MixedFrequencyData(weekly_frame(), {"claims": "W", "ip": "M", "gdp": "Q"})
        assert mfd.frequency_ratio("claims") == 1
        with pytest.raises(ValueError, match="not fixed"):
            mfd.frequency_ratio("ip")

    def test_standardize_and_extend(self):
        mfd = MixedFrequencyData(weekly_frame(), {"claims": "W", "ip": "M", "gdp": "Q"})
        z, stats = mfd.standardize()
        assert np.allclose(z.data.mean(), 0.0, atol=1e-12)
        longer = mfd.extend(6)
        assert longer.n_periods == mfd.n_periods + 6
        assert stats.columns == ["claims", "ip", "gdp"]


class TestDailyBase:
    def test_weekly_and_monthly_series_on_daily_grid(self):
        idx = pd.period_range("2021-01-01", periods=120, freq="D")
        frame = pd.DataFrame(
            {
                "fx": np.arange(120.0),
                "w": np.where(is_period_end(idx, "W"), 1.0, np.nan),
                "m": np.where(is_period_end(idx, "M"), 2.0, np.nan),
            },
            index=idx,
        )
        mfd = MixedFrequencyData(frame)
        assert mfd.frequencies.to_dict() == {
            "fx": Frequency.DAILY,
            "w": Frequency.WEEKLY,
            "m": Frequency.MONTHLY,
        }
        assert mfd.frequency_ratio("w") == 7
        assert mfd.to_native("m", dropna=True).index.astype(str).tolist()[:2] == [
            "2021-01",
            "2021-02",
        ]
