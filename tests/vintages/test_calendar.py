"""Tests for nowcastbox.vintages.calendar."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nowcastbox.core.data import MixedFrequencyData
from nowcastbox.core.exceptions import DataQualityWarning, NowcastDataError
from nowcastbox.core.frequency import Frequency
from nowcastbox.vintages import RELEASE_COLUMNS, ReleaseCalendar
from tests.vintages.conftest import DELAYS, FREQS

# --------------------------------------------------------------------- construction


class TestConstruction:
    def test_delays_and_frequencies(self) -> None:
        cal = ReleaseCalendar(DELAYS, frequencies=FREQS)
        assert cal.series == ["ip", "pmi", "gdp"]
        assert cal.delays.to_dict() == DELAYS
        assert cal.delays.dtype == "Int64"
        assert cal.frequencies["gdp"] is Frequency.QUARTERLY
        assert len(cal) == 3
        assert "gdp" in cal
        assert "x" not in cal
        assert "n_series=3" in repr(cal)

    def test_delays_from_series_skips_missing(self) -> None:
        cal = ReleaseCalendar(pd.Series({"a": 10, "b": np.nan, "c": 3.0}))
        assert cal.series == ["a", "c"]
        assert cal.delays.to_dict() == {"a": 10, "c": 3}

    @pytest.mark.parametrize("bad", [-366, -400, 1.5, "10", True])
    def test_invalid_delay(self, bad: object) -> None:
        with pytest.raises(ValueError, match="integer"):
            ReleaseCalendar({"a": bad})

    def test_delays_wrong_type(self) -> None:
        with pytest.raises(ValueError, match="mapping"):
            ReleaseCalendar([1, 2])  # type: ignore[arg-type]

    def test_empty_calendar(self) -> None:
        with pytest.raises(ValueError, match="at least one series"):
            ReleaseCalendar()
        with pytest.raises(ValueError, match="at least one series"):
            ReleaseCalendar({"a": None})

    def test_bad_frequencies_type(self) -> None:
        with pytest.raises(ValueError, match="frequencies"):
            ReleaseCalendar({"a": 1}, frequencies=["M"])  # type: ignore[arg-type]

    def test_explicit_dates_mapping(self) -> None:
        cal = ReleaseCalendar(
            release_dates={"gdp": {"2020Q1": "2020-05-29", "2019Q4": "2020-03-04"}}
        )
        assert cal.frequencies["gdp"] is Frequency.QUARTERLY
        table = cal.explicit_dates("gdp")
        assert list(table.index.astype(str)) == ["2019Q4", "2020Q1"]  # sorted
        assert cal.has_explicit_dates("gdp")
        assert cal.delays.isna().all()

    def test_explicit_dates_series_and_frame(self) -> None:
        dates = pd.Series(
            ["2020-02-10", "2020-03-10"], index=pd.period_range("2020-01", periods=2, freq="M")
        )
        cal = ReleaseCalendar(release_dates={"ip": dates})
        assert cal.release_date("ip", "2020-02") == pd.Timestamp("2020-03-10")
        frame = pd.DataFrame(
            {
                "series": ["ip", "ip", "gdp"],
                "reference_period": ["2020-01", "2020-02", "2020Q1"],
                "release_date": ["2020-02-10", "2020-03-10", "2020-05-29"],
            }
        )
        cal2 = ReleaseCalendar(release_dates=frame)
        assert cal2.series == ["ip", "gdp"]
        assert cal2.release_date("gdp", "2020Q1") == pd.Timestamp("2020-05-29")

    def test_explicit_dates_with_frequency_conversion(self) -> None:
        # a monthly slot label for a quarterly series identifies the quarter
        cal = ReleaseCalendar(
            release_dates={"gdp": {"2020-03": "2020-05-29"}}, frequencies={"gdp": "Q"}
        )
        assert list(cal.explicit_dates("gdp").index.astype(str)) == ["2020Q1"]

    def test_explicit_dates_errors(self) -> None:
        with pytest.raises(NowcastDataError, match="lacks columns"):
            ReleaseCalendar(release_dates=pd.DataFrame({"series": ["a"]}))
        with pytest.raises(NowcastDataError, match="mapping or a long DataFrame"):
            ReleaseCalendar(release_dates=[1])  # type: ignore[arg-type]
        with pytest.raises(NowcastDataError, match="mapping or Series"):
            ReleaseCalendar(release_dates={"a": 3})
        with pytest.raises(NowcastDataError, match="No release dates"):
            ReleaseCalendar(release_dates={"a": {}})
        with pytest.raises(NowcastDataError, match="mix frequencies"):
            ReleaseCalendar(release_dates={"a": {"2020-01": "2020-02-01", "2020Q1": "2020-05-01"}})
        with pytest.raises(NowcastDataError, match="Duplicated"):
            ReleaseCalendar(
                release_dates={"a": {"2020-01": "2020-02-01", "2020-01-15": "2020-02-03"}},
                frequencies={"a": "M"},
            )
        with pytest.raises(NowcastDataError, match="lower than the series frequency"):
            ReleaseCalendar(release_dates={"a": {"2020Q1": "2020-05-01"}}, frequencies={"a": "M"})
        with pytest.raises(NowcastDataError, match="Cannot parse"):
            ReleaseCalendar(release_dates={"a": {"not a period": "2020-05-01"}})
        with pytest.raises(ValueError, match="Cannot parse"):
            ReleaseCalendar(release_dates={"a": {"2020-01": "garbage"}})
        with pytest.raises(ValueError, match="must be a date"):
            ReleaseCalendar(release_dates={"a": {"2020-01": None}})
        with pytest.raises(ValueError, match="missing value"):
            ReleaseCalendar(release_dates={"a": {"2020-01": np.nan}})
        with pytest.raises(NowcastDataError, match="missing"):
            ReleaseCalendar(release_dates={"a": {None: "2020-02-01"}})

    def test_explicit_date_before_period_end_warns(self) -> None:
        with pytest.warns(DataQualityWarning, match="precede the end"):
            ReleaseCalendar(release_dates={"a": {"2020-01": "2020-01-15"}})

    def test_date_keys_need_frequency(self) -> None:
        with pytest.raises(NowcastDataError, match="frequency is required"):
            ReleaseCalendar(release_dates={"a": {pd.Timestamp("2020-01-31"): "2020-02-10"}})
        cal = ReleaseCalendar(
            release_dates={"a": {pd.Timestamp("2020-01-31"): "2020-02-10"}},
            frequencies={"a": "M"},
        )
        assert cal.release_date("a", pd.Period("2020-01", "M")) == pd.Timestamp("2020-02-10")

    def test_from_frame(self) -> None:
        table = pd.DataFrame(
            {"s": ["gdp"], "p": [pd.Period("2020Q1", "Q")], "d": [pd.Timestamp("2020-05-29")]}
        )
        cal = ReleaseCalendar.from_frame(
            table, series_column="s", period_column="p", date_column="d", delays={"gdp": 60}
        )
        assert cal.release_date("gdp", "2020Q1") == pd.Timestamp("2020-05-29")
        assert cal.release_date("gdp", "2020Q2") == pd.Timestamp("2020-08-29")
        with pytest.raises(NowcastDataError, match="lacks columns"):
            ReleaseCalendar.from_frame(table)

    def test_from_data(self, final_panel: MixedFrequencyData) -> None:
        cal = ReleaseCalendar.from_data(final_panel)
        assert cal.delays.to_dict() == DELAYS
        assert cal.frequencies.to_dict() == {
            "ip": Frequency.MONTHLY,
            "pmi": Frequency.MONTHLY,
            "gdp": Frequency.QUARTERLY,
        }
        cal2 = ReleaseCalendar.from_data(final_panel, {"gdp": 90})
        assert cal2.delays["gdp"] == 90
        with pytest.raises(NowcastDataError, match="unknown series"):
            ReleaseCalendar.from_data(final_panel, {"zzz": 1})

    def test_from_data_frame_and_missing_delay(self, final_frame: pd.DataFrame) -> None:
        with pytest.raises(NowcastDataError, match="No release delay"):
            ReleaseCalendar.from_data(final_frame)
        cal = ReleaseCalendar.from_data(final_frame, DELAYS, frequency=FREQS)
        assert cal.frequencies["gdp"] is Frequency.QUARTERLY

    def test_to_frame(self) -> None:
        cal = ReleaseCalendar(
            {"ip": 30}, release_dates={"gdp": {"2020Q1": "2020-05-29", "2020Q2": "2020-09-01"}}
        )
        frame = cal.to_frame()
        assert list(frame.index) == ["ip", "gdp"]
        assert frame.loc["gdp", "n_explicit_dates"] == 2
        assert str(frame.loc["gdp", "last_explicit_period"]) == "2020Q2"
        assert pd.isna(frame.loc["ip", "frequency"])
        assert frame.loc["gdp", "frequency"] == "Q"
        assert pd.isna(frame.loc["gdp", "release_delay"])

    def test_with_delays_and_frequencies(self) -> None:
        cal = ReleaseCalendar({"a": 5})
        cal2 = cal.with_delays({"a": 7, "b": 3}).with_frequencies({"a": "M", "b": "Q"})
        assert cal.delays.to_dict() == {"a": 5}  # immutable
        assert cal2.delays.to_dict() == {"a": 7, "b": 3}
        assert cal2.frequencies["b"] is Frequency.QUARTERLY
        with pytest.raises(NowcastDataError, match="not in the release calendar"):
            cal.with_frequencies({"zzz": "M"})
        explicit = ReleaseCalendar(release_dates={"g": {"2020Q1": "2020-05-29"}})
        with pytest.raises(NowcastDataError, match="conflicts"):
            explicit.with_frequencies({"g": "M"})
        assert explicit.with_frequencies({"g": "Q"}).series == ["g"]

    def test_explicit_dates_of_delay_series_is_empty(self) -> None:
        cal = ReleaseCalendar({"a": 5})
        assert cal.explicit_dates("a").empty
        with pytest.raises(NowcastDataError, match="not in the release calendar"):
            cal.explicit_dates("b")
        with pytest.raises(NowcastDataError, match="not in the release calendar"):
            cal.has_explicit_dates("b")


# --------------------------------------------------------------------- release dates


class TestReleaseDate:
    @pytest.mark.parametrize(
        ("freq", "delay", "period", "expected"),
        [
            ("M", 0, "2020-01", "2020-01-31"),  # same-day release at month end
            ("M", 1, "2020-01", "2020-02-01"),
            ("M", 30, "2020-01", "2020-03-01"),  # crosses short (leap) February
            ("M", 30, "2021-01", "2021-03-02"),  # non-leap
            ("M", 45, "2020-01", "2020-03-16"),  # delay > 1 period
            ("M", 75, "2020-11", "2021-02-13"),  # year boundary, ~2.5 months
            ("Q", 60, "2020Q1", "2020-05-30"),
            ("Q", 60, "2020Q4", "2021-03-01"),
            ("Q", 120, "2020Q3", "2021-01-28"),  # > 1 quarter
            ("A", 90, "2019", "2020-03-30"),
            ("W", 3, "2020-01-06", "2020-01-15"),  # week Mon 6 .. Sun 12
            ("D", 2, "2020-02-28", "2020-03-01"),
        ],
    )
    def test_delay_rule(self, freq: str, delay: int, period: str, expected: str) -> None:
        cal = ReleaseCalendar({"x": delay}, frequencies={"x": freq})
        assert cal.release_date("x", period) == pd.Timestamp(expected)

    def test_explicit_overrides_delay_and_fallback(self) -> None:
        cal = ReleaseCalendar(
            {"gdp": 60},
            release_dates={"gdp": {"2020Q1": "2020-05-29"}},
            frequencies={"gdp": "Q"},
        )
        assert cal.release_date("gdp", "2020Q1") == pd.Timestamp("2020-05-29")
        assert cal.release_date("gdp", "2020Q2") == pd.Timestamp("2020-08-29")
        dates = cal.release_dates_for("gdp", ["2020Q1", "2020Q2", pd.Period("2020Q3", "Q")])
        assert dates.dt.strftime("%Y-%m-%d").tolist() == ["2020-05-29", "2020-08-29", "2020-11-29"]

    def test_unknown_release(self) -> None:
        cal = ReleaseCalendar(release_dates={"gdp": {"2020Q1": "2020-05-29"}})
        with pytest.raises(NowcastDataError, match="No release date"):
            cal.release_date("gdp", "2020Q2")
        assert cal.release_dates_for("gdp", ["2020Q2"]).isna().all()

    def test_unknown_series_or_frequency(self) -> None:
        cal = ReleaseCalendar({"a": 5})
        with pytest.raises(NowcastDataError, match="not in the release calendar"):
            cal.release_date("b", "2020-01")
        with pytest.raises(NowcastDataError, match="frequency of 'a' is unknown"):
            cal.release_date("a", "2020-01")


# --------------------------------------------------------------------- masks


class TestReleaseMask:
    def test_matches_core_as_of(self, final_panel: MixedFrequencyData) -> None:
        cal = ReleaseCalendar.from_data(final_panel)
        for vintage in ["2018-01-15", "2018-05-30", "2018-05-31", "2019-03-01", "2021-06-30"]:
            mask = cal.release_mask(final_panel, vintage)
            expected = final_panel.as_of(vintage).observation_mask()
            observed = final_panel.observation_mask()
            assert ((mask & observed) == expected).all().all(), vintage

    def test_quarterly_boundaries(self, final_panel: MixedFrequencyData) -> None:
        cal = ReleaseCalendar.from_data(final_panel)
        # 2018Q1 (end 31 Mar) + 60 days = 30 May 2018
        before = cal.release_mask(final_panel, "2018-05-29")
        on = cal.release_mask(final_panel, "2018-05-30")
        march = pd.Period("2018-03", "M")
        assert not before.loc[march, "gdp"]
        assert on.loc[march, "gdp"]
        # months 1-2 of the quarter carry the flag of their quarter
        assert on.loc[pd.Period("2018-01", "M"), "gdp"]

    def test_series_not_in_calendar(self, final_frame: pd.DataFrame) -> None:
        cal = ReleaseCalendar({"ip": 1}, frequencies={"ip": "M"})
        with pytest.raises(NowcastDataError, match="not in the release calendar"):
            cal.release_mask(final_frame, "2020-01-01")

    def test_frequency_conflict(self, final_panel: MixedFrequencyData) -> None:
        cal = ReleaseCalendar(DELAYS, frequencies={"ip": "M", "pmi": "M", "gdp": "M"})
        with pytest.raises(NowcastDataError, match="quarterly in the data"):
            cal.release_mask(final_panel, "2020-01-01")

    def test_observed_value_without_release_date(self, final_panel: MixedFrequencyData) -> None:
        cal = ReleaseCalendar({"ip": 1, "pmi": 1}, release_dates={"gdp": {"2018Q1": "2018-05-01"}})
        with pytest.raises(NowcastDataError, match="No release date for observed values"):
            cal.release_mask(final_panel, "2020-01-01")

    def test_unknown_dates_of_missing_values_are_fine(self) -> None:
        idx = pd.period_range("2020-01", periods=6, freq="M")
        frame = pd.DataFrame({"gdp": [np.nan, np.nan, 1.0, np.nan, np.nan, np.nan]}, index=idx)
        cal = ReleaseCalendar(release_dates={"gdp": {"2020Q1": "2020-05-29"}})
        mask = cal.release_mask(MixedFrequencyData(frame, {"gdp": "Q"}), "2021-01-01")
        assert mask["gdp"].tolist() == [True, True, True, False, False, False]

    def test_invalid_vintage(self, final_panel: MixedFrequencyData) -> None:
        cal = ReleaseCalendar.from_data(final_panel)
        for bad in [None, "not a date", pd.Period("2020-01", "M"), True, pd.NaT]:
            with pytest.raises(ValueError):
                cal.release_mask(final_panel, bad)  # type: ignore[arg-type]

    def test_timezone_and_time_dropped(self, final_panel: MixedFrequencyData) -> None:
        cal = ReleaseCalendar.from_data(final_panel)
        a = cal.release_mask(final_panel, pd.Timestamp("2018-05-30 23:00", tz="UTC"))
        b = cal.release_mask(final_panel, "2018-05-30")
        assert a.equals(b)


# --------------------------------------------------------------------- available_at


class TestAvailableAt:
    @pytest.mark.parametrize(
        ("date", "ip", "gdp"),
        [
            ("2020-03-15", "2019-12", "2019Q4"),  # ip Jan released 16 Mar
            ("2020-03-16", "2020-01", "2019Q4"),
            ("2020-03-17", "2020-01", "2019Q4"),
            ("2020-05-29", "2020-03", "2019Q4"),
            ("2020-05-30", "2020-03", "2020Q1"),
        ],
    )
    def test_delay_rule(self, date: str, ip: str, gdp: str) -> None:
        cal = ReleaseCalendar({"ip": 45, "gdp": 60}, frequencies={"ip": "M", "gdp": "Q"})
        out = cal.available_at(date)
        assert str(out["ip"]) == ip
        assert str(out["gdp"]) == gdp

    def test_month_end_boundary(self) -> None:
        cal = ReleaseCalendar({"x": 0}, frequencies={"x": "M"})
        assert str(cal.available_at("2020-02-29")["x"]) == "2020-02"
        assert str(cal.available_at("2020-02-28")["x"]) == "2020-01"

    def test_explicit_and_mixed(self) -> None:
        cal = ReleaseCalendar(
            {"g": 60, "h": 10},
            release_dates={"g": {"2020Q1": "2020-07-15"}, "e": {"2020-01": "2020-02-03"}},
            frequencies={"g": "Q", "h": "M"},
        )
        out = cal.available_at("2020-06-01")
        # 2020Q1 has a late explicit date, so the delay rule only covers 2019Q4
        assert str(out["g"]) == "2019Q4"
        assert out["e"] is None or str(out["e"]) == "2020-01"
        assert str(cal.available_at("2020-07-15")["g"]) == "2020Q1"
        assert cal.available_at("2020-01-01", series=["e"])["e"] is None
        assert str(cal.available_at("2020-02-03", series=["e"])["e"]) == "2020-01"

    def test_consistent_with_mask(self, final_panel: MixedFrequencyData) -> None:
        cal = ReleaseCalendar.from_data(final_panel)
        for date in pd.date_range("2018-01-01", "2020-12-31", freq="17D"):
            last = cal.available_at(date)
            mask = cal.release_mask(final_panel, date)
            for name in final_panel.columns:
                freq = final_panel.metadata[name].frequency
                native = final_panel.index.asfreq(freq.pandas_freq)
                released = native[mask[name].to_numpy()]
                if len(released):
                    assert released.max() == last[name]

    def test_unknown_series(self) -> None:
        cal = ReleaseCalendar({"a": 1}, frequencies={"a": "M"})
        with pytest.raises(NowcastDataError):
            cal.available_at("2020-01-01", series=["b"])


# --------------------------------------------------------------------- releases_between


class TestReleasesBetween:
    def test_without_data(self) -> None:
        cal = ReleaseCalendar({"ip": 40, "gdp": 60}, frequencies={"ip": "M", "gdp": "Q"})
        rel = cal.releases_between("2020-04-01", "2020-06-30")
        assert list(rel.columns) == list(RELEASE_COLUMNS)
        got = [(d.strftime("%Y-%m-%d"), s, str(p)) for d, s, p in rel.itertuples(index=False)]
        assert got == [
            ("2020-04-09", "ip", "2020-02"),  # leap February
            ("2020-05-10", "ip", "2020-03"),
            ("2020-05-30", "gdp", "2020Q1"),
            ("2020-06-09", "ip", "2020-04"),
        ]

    def test_half_open_interval(self) -> None:
        cal = ReleaseCalendar({"ip": 10}, frequencies={"ip": "M"})
        assert len(cal.releases_between("2020-02-10", "2020-03-09")) == 0
        assert len(cal.releases_between("2020-02-09", "2020-02-10")) == 1
        assert len(cal.releases_between("2020-02-10", "2020-02-10")) == 0

    def test_start_after_end(self) -> None:
        cal = ReleaseCalendar({"ip": 10}, frequencies={"ip": "M"})
        with pytest.raises(ValueError, match="must not be after"):
            cal.releases_between("2020-03-01", "2020-02-01")

    def test_explicit_dates_without_data(self) -> None:
        cal = ReleaseCalendar(
            {"g": 60},
            release_dates={"g": {"2020Q1": "2020-07-15"}},
            frequencies={"g": "Q"},
        )
        rel = cal.releases_between("2020-05-01", "2020-12-31")
        assert [str(p) for p in rel["reference_period"]] == ["2020Q1", "2020Q2", "2020Q3"]
        assert rel["release_date"].dt.strftime("%m-%d").tolist() == ["07-15", "08-29", "11-29"]

    def test_empty(self) -> None:
        cal = ReleaseCalendar({"g": 60}, frequencies={"g": "Q"})
        rel = cal.releases_between("2020-06-01", "2020-06-02")
        assert rel.empty
        assert list(rel.columns) == list(RELEASE_COLUMNS)
        assert rel["release_date"].dtype == "datetime64[ns]"

    def test_with_data_matches_mask_difference(self, final_panel: MixedFrequencyData) -> None:
        cal = ReleaseCalendar.from_data(final_panel)
        old, new = "2019-02-20", "2019-06-05"
        rel = cal.releases_between(old, new, final_panel)
        assert list(rel.columns) == [*RELEASE_COLUMNS, "slot", "value"]
        diff = cal.release_mask(final_panel, new) & ~cal.release_mask(final_panel, old)
        diff &= final_panel.observation_mask()
        expected = {(s, p) for s in diff.columns for p in diff.index[diff[s].to_numpy()]}
        assert set(zip(rel["series"], rel["slot"], strict=True)) == expected
        assert rel["release_date"].is_monotonic_increasing
        for row in rel.itertuples(index=False):
            assert final_panel[row.series][row.slot] == row.value

    def test_with_data_series_and_unobserved(self, final_frame: pd.DataFrame) -> None:
        frame = final_frame.copy()
        frame.iloc[-1, 0] = np.nan
        cal = ReleaseCalendar(DELAYS)
        rel = cal.releases_between("2020-01-01", "2021-12-31", frame, series=["ip"])
        rel_all = cal.releases_between(
            "2020-01-01", "2021-12-31", frame, series=["ip"], observed_only=False
        )
        assert set(rel["series"]) == {"ip"}
        assert len(rel_all) == len(rel) + 1
        assert rel_all["value"].isna().sum() == 1

    def test_without_data_matches_with_data(self, final_panel: MixedFrequencyData) -> None:
        cal = ReleaseCalendar(DELAYS, frequencies=FREQS)
        a = cal.releases_between("2018-06-01", "2019-12-31")
        b = cal.releases_between("2018-06-01", "2019-12-31", final_panel)
        pd.testing.assert_frame_equal(a, b[list(RELEASE_COLUMNS)])

    def test_release_dates(self) -> None:
        cal = ReleaseCalendar(
            {"a": 10, "b": 10, "g": 60}, frequencies={"a": "M", "b": "M", "g": "Q"}
        )
        dates = cal.release_dates("2020-01-01", "2020-06-30")
        assert dates.strftime("%m-%d").tolist() == [
            "01-10",
            "02-10",
            "02-29",  # 2019Q4 + 60 days
            "03-10",
            "04-10",
            "05-10",
            "05-30",
            "06-10",
        ]
        assert dates.name == "release_date"


class TestSchedule:
    def test_schedule_columns_and_order(self, final_panel: MixedFrequencyData) -> None:
        cal = ReleaseCalendar.from_data(final_panel)
        table = cal.schedule(final_panel)
        n_quarters = final_panel.n_periods // 3
        assert len(table) == 2 * final_panel.n_periods + n_quarters
        assert table["release_date"].is_monotonic_increasing
        gdp = table[table["series"] == "gdp"]
        assert all(str(s).endswith(("-03", "-06", "-09", "-12")) for s in gdp["slot"])
        assert all(isinstance(p, pd.Period) for p in table["reference_period"])

    def test_schedule_skips_unknown_dates_of_missing(self) -> None:
        idx = pd.period_range("2020-01", periods=6, freq="M")
        frame = pd.DataFrame({"gdp": [np.nan, np.nan, 1.0, np.nan, np.nan, np.nan]}, index=idx)
        cal = ReleaseCalendar(release_dates={"gdp": {"2020Q1": "2020-05-29"}})
        assert len(cal.schedule(MixedFrequencyData(frame, {"gdp": "Q"}))) == 1

    def test_schedule_raises_on_unknown_observed(self) -> None:
        idx = pd.period_range("2020-01", periods=6, freq="M")
        frame = pd.DataFrame({"gdp": [np.nan, np.nan, 1.0, np.nan, np.nan, 2.0]}, index=idx)
        cal = ReleaseCalendar(release_dates={"gdp": {"2020Q1": "2020-05-29"}})
        with pytest.raises(NowcastDataError, match="No release date"):
            cal.schedule(frame)


def test_negative_delay_released_before_period_end() -> None:
    cal = ReleaseCalendar({"survey": -14}, frequencies={"survey": "M"})
    assert cal.release_date("survey", "2020-03") == pd.Timestamp("2020-03-17")
    idx = pd.period_range("2020-01", periods=3, freq="M")
    frame = pd.DataFrame({"survey": [1.0, 2.0, 3.0]}, index=idx)
    assert cal.release_mask(frame, "2020-03-17")["survey"].tolist() == [True, True, True]
    assert cal.release_mask(frame, "2020-03-16")["survey"].tolist() == [True, True, False]


def test_release_mask_with_explicit_dates_and_no_rows() -> None:
    cal = ReleaseCalendar({"a": 10}, {"a": {"2020-02": "2020-03-01"}}, frequencies={"a": "M"})
    idx = pd.period_range("2020-01", periods=3, freq="M")
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0]}, index=idx)
    assert cal.release_mask(frame, "2020-03-05")["a"].tolist() == [True, True, False]
    assert len(cal._dates_for("a", idx[:0])) == 0
