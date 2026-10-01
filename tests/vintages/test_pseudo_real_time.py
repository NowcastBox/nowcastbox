"""Tests for nowcastbox.vintages.pseudo_real_time."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from nowcastbox.core.data import MixedFrequencyData
from nowcastbox.core.exceptions import NowcastDataError
from nowcastbox.vintages import (
    ReleaseCalendar,
    Vintage,
    generate_vintages,
    pseudo_real_time,
    vintage_dates,
)
from tests.vintages.conftest import DELAYS, FREQS


class TestPseudoRealTime:
    def test_frame_in_frame_out(self, final_frame: pd.DataFrame) -> None:
        out = pseudo_real_time(final_frame, DELAYS, "2019-06-15", frequency=FREQS)
        assert isinstance(out, pd.DataFrame)
        assert out.index.equals(final_frame.index)
        assert list(out.columns) == list(final_frame.columns)
        # ip: delay 45 -> April 2019 released 14 Jun (30 Apr + 45)
        assert out["ip"].last_valid_index() == pd.Period("2019-04", "M")
        # pmi: delay 1 -> May released 1 Jun
        assert out["pmi"].last_valid_index() == pd.Period("2019-05", "M")
        # gdp: 2019Q1 released 30 May
        assert out["gdp"].last_valid_index() == pd.Period("2019-03", "M")
        # values that are kept are unchanged
        kept = out.notna().to_numpy()
        assert np.allclose(out.to_numpy()[kept], final_frame.to_numpy()[kept])

    def test_mixed_in_mixed_out_and_metadata_default(self, final_panel: MixedFrequencyData) -> None:
        out = pseudo_real_time(final_panel, vintage="2019-06-15")
        assert isinstance(out, MixedFrequencyData)
        assert out.metadata == final_panel.metadata
        assert out.equals(final_panel.as_of("2019-06-15"))

    @pytest.mark.parametrize(
        "vintage",
        ["2017-12-31", "2018-01-01", "2018-02-15", "2018-05-30", "2019-12-31", "2023-01-01"],
    )
    def test_equals_core_as_of(self, final_panel: MixedFrequencyData, vintage: str) -> None:
        delays = {"ip": 75, "pmi": 0, "gdp": 120}  # delays longer than one period
        out = pseudo_real_time(final_panel, delays, vintage)
        assert out.equals(final_panel.as_of(vintage, release_delays=delays))

    def test_delay_specifications_agree(self, final_frame: pd.DataFrame) -> None:
        expected = pseudo_real_time(final_frame, DELAYS, "2019-06-15", frequency=FREQS)
        variants = [
            pd.Series(DELAYS),
            pd.Series({**DELAYS, "dropped_series": 30}),  # named superset: by name
            pd.Series([45, 1, 60]),  # positional (R legend style)
            [45, 1, 60],
            np.array([45, 1, 60]),
            ReleaseCalendar(DELAYS),
        ]
        for spec in variants:
            got = pseudo_real_time(final_frame, spec, "2019-06-15", frequency=FREQS)
            pd.testing.assert_frame_equal(got, expected)
        got = pseudo_real_time(
            final_frame, vintage="2019-06-15", calendar=ReleaseCalendar(DELAYS), frequency=FREQS
        )
        pd.testing.assert_frame_equal(got, expected)

    def test_scalar_delay(self, final_frame: pd.DataFrame) -> None:
        out = pseudo_real_time(final_frame, 0, "2019-03-31", frequency=FREQS)
        assert out["ip"].last_valid_index() == pd.Period("2019-03", "M")
        assert out["gdp"].last_valid_index() == pd.Period("2019-03", "M")
        out = pseudo_real_time(final_frame, 0, "2019-03-30", frequency=FREQS)
        assert out["ip"].last_valid_index() == pd.Period("2019-02", "M")
        assert out["gdp"].last_valid_index() == pd.Period("2018-12", "M")

    def test_vintage_before_any_release(self, final_frame: pd.DataFrame) -> None:
        out = pseudo_real_time(final_frame, DELAYS, "2000-01-01", frequency=FREQS)
        assert out.isna().all().all()

    def test_explicit_calendar(self, final_frame: pd.DataFrame) -> None:
        cal = ReleaseCalendar(
            {"ip": 45, "pmi": 1, "gdp": 60},
            release_dates={"gdp": {"2019Q1": "2019-06-20"}},  # late release
        )
        out = pseudo_real_time(final_frame, cal, "2019-06-15", frequency=FREQS)
        assert out["gdp"].last_valid_index() == pd.Period("2018-12", "M")
        out = pseudo_real_time(final_frame, cal, "2019-06-20", frequency=FREQS)
        assert out["gdp"].last_valid_index() == pd.Period("2019-03", "M")

    def test_errors(self, final_frame: pd.DataFrame, final_panel: MixedFrequencyData) -> None:
        with pytest.raises(ValueError, match="vintage"):
            pseudo_real_time(final_frame, DELAYS)
        with pytest.raises(ValueError, match="either delay or calendar"):
            pseudo_real_time(final_frame, DELAYS, "2019-01-01", calendar=DELAYS)
        with pytest.raises(ValueError, match="frequency can only"):
            pseudo_real_time(final_panel, DELAYS, "2019-01-01", frequency=FREQS)
        with pytest.raises(ValueError, match="3 series"):
            pseudo_real_time(final_frame, [1, 2], "2019-01-01")
        with pytest.raises(ValueError, match="delay must be"):
            pseudo_real_time(final_frame, "45", "2019-01-01")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="delay must be"):
            pseudo_real_time(final_frame, 1.5, "2019-01-01")  # type: ignore[arg-type]
        with pytest.raises(NowcastDataError, match="No release delay"):
            pseudo_real_time(final_frame, {"ip": 1}, "2019-01-01")
        with pytest.raises(NowcastDataError, match="unknown series"):
            pseudo_real_time(final_frame, {"zzz": 1, **DELAYS}, "2019-01-01")
        with pytest.raises(ValueError, match="2 entries"):
            pseudo_real_time(final_frame, pd.Series([1, 2]), "2019-01-01")

    def test_input_not_modified(self, final_frame: pd.DataFrame) -> None:
        before = final_frame.copy()
        pseudo_real_time(final_frame, DELAYS, "2018-06-01", frequency=FREQS)
        pd.testing.assert_frame_equal(final_frame, before)


class TestVintageDates:
    def test_monthly_anchored_at_month_end(self) -> None:
        dates = vintage_dates("2020-01-31", "2020-06-30", "M")
        assert dates.strftime("%Y-%m-%d").tolist() == [
            "2020-01-31",
            "2020-02-29",
            "2020-03-31",
            "2020-04-30",
            "2020-05-31",
            "2020-06-30",
        ]
        assert dates.name == "vintage"

    @pytest.mark.parametrize(
        ("step", "expected"),
        [
            ("D", 10),
            ("2D", 5),
            (3, 4),
            (pd.Timedelta(days=4), 3),
            ("W", 2),
            ("1w", 2),
        ],
    )
    def test_fixed_steps(self, step: object, expected: int) -> None:
        assert len(vintage_dates("2020-01-01", "2020-01-10", step)) == expected  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        ("step", "expected"),
        [("M", 13), ("ME", 13), ("MS", 13), ("2M", 7), ("Q", 5), ("QE", 5), ("Y", 2), ("A", 2)],
    )
    def test_calendar_steps(self, step: str, expected: int) -> None:
        assert len(vintage_dates("2020-01-15", "2021-01-15", step)) == expected

    def test_date_offset(self) -> None:
        dates = vintage_dates("2020-01-31", "2020-03-31", pd.DateOffset(months=1))
        assert dates.strftime("%m-%d").tolist() == ["01-31", "02-29", "03-31"]

    def test_business_days(self) -> None:
        dates = vintage_dates("2020-01-03", "2020-01-10", "B")  # Friday .. Friday
        assert dates.strftime("%a").tolist() == ["Fri", "Mon", "Tue", "Wed", "Thu", "Fri"]
        weekend = vintage_dates("2020-01-04", "2020-01-08", "B")
        assert weekend.strftime("%m-%d").tolist() == ["01-04", "01-06", "01-07", "01-08"]

    def test_single_date(self) -> None:
        assert len(vintage_dates("2020-01-01", "2020-01-01", "M")) == 1

    @pytest.mark.parametrize("step", ["X", "0M", 0, -2, True, pd.Timedelta(0), "M2", 1.5])
    def test_invalid_step(self, step: object) -> None:
        with pytest.raises(ValueError):
            vintage_dates("2020-01-01", "2020-03-01", step)  # type: ignore[arg-type]

    def test_non_advancing_offset(self) -> None:
        with pytest.raises(ValueError, match="does not advance"):
            vintage_dates("2020-01-01", "2020-03-01", pd.DateOffset(months=0))

    def test_start_after_end(self) -> None:
        with pytest.raises(ValueError, match="must not be after"):
            vintage_dates("2020-03-01", "2020-01-01")

    def test_release_step(self) -> None:
        cal = ReleaseCalendar({"a": 10, "g": 60}, frequencies={"a": "M", "g": "Q"})
        dates = vintage_dates("2020-01-01", "2020-03-31", "release", calendar=cal)
        assert dates.strftime("%m-%d").tolist() == ["01-01", "01-10", "02-10", "02-29", "03-10"]
        with pytest.raises(ValueError, match="requires a calendar"):
            vintage_dates("2020-01-01", "2020-03-31", "release")


class TestGenerateVintages:
    def test_monthly_sequence(self, final_frame: pd.DataFrame) -> None:
        vintages = list(
            generate_vintages(final_frame, DELAYS, "2018-03-01", "2019-03-01", "M", frequency=FREQS)
        )
        assert len(vintages) == 13
        assert all(isinstance(v, Vintage) for v in vintages)
        assert all(isinstance(v.data, pd.DataFrame) for v in vintages)
        for v in vintages:
            expected = pseudo_real_time(final_frame, DELAYS, v.date, frequency=FREQS)
            pd.testing.assert_frame_equal(v.data, expected)

    def test_information_is_nested(self, final_panel: MixedFrequencyData) -> None:
        previous: pd.DataFrame | None = None
        for date, data in generate_vintages(final_panel, None, "2018-01-01", "2020-12-31", "2W"):
            assert isinstance(data, MixedFrequencyData)
            obs = data.observation_mask()
            if previous is not None:
                assert (previous <= obs).all().all(), date
            previous = obs

    def test_release_step_changes_every_time(self, final_panel: MixedFrequencyData) -> None:
        vintages = list(generate_vintages(final_panel, None, "2018-06-01", "2018-12-31", "release"))
        counts = [int(v.data.n_observations().sum()) for v in vintages]
        assert all(b > a for a, b in itertools.pairwise(counts))
        # start + pmi (1d: Jul..Dec) + ip (45d: Apr..Oct) + gdp (60d: Q2, Q3)
        assert len(vintages) == 1 + 6 + 7 + 2

    def test_skip_unchanged(self, final_frame: pd.DataFrame) -> None:
        all_days = list(
            generate_vintages(final_frame, DELAYS, "2018-06-01", "2018-07-31", "D", frequency=FREQS)
        )
        changed = list(
            generate_vintages(
                final_frame,
                DELAYS,
                "2018-06-01",
                "2018-07-31",
                "D",
                frequency=FREQS,
                skip_unchanged=True,
            )
        )
        assert len(all_days) == 61
        # first day + releases: pmi 1 Jun is the first day itself, ip 14 Jun & 15 Jul,
        # pmi 1 Jul, no gdp release (2018Q2 -> 29 Aug)
        assert [v.date.strftime("%m-%d") for v in changed] == ["06-01", "06-14", "07-01", "07-15"]

    def test_errors(self, final_frame: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="start and end"):
            next(generate_vintages(final_frame, DELAYS, frequency=FREQS))
        with pytest.raises(ValueError, match="Invalid step"):
            next(
                generate_vintages(
                    final_frame, DELAYS, "2018-01-01", "2018-03-01", "fortnight", frequency=FREQS
                )
            )
