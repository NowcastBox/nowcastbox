"""Tests for nowcastbox.vintages.vintage_store."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nowcastbox.core.data import MixedFrequencyData
from nowcastbox.core.exceptions import DataQualityWarning, NowcastDataError
from nowcastbox.core.frequency import Frequency
from nowcastbox.vintages import (
    RECORD_COLUMNS,
    ReleaseCalendar,
    VintageStore,
    generate_vintages,
    pseudo_real_time,
)
from tests.vintages.conftest import DELAYS, FREQS

pytestmark = pytest.mark.filterwarnings("ignore:Could not infer format")


@pytest.fixture
def store(revision_records: pd.DataFrame) -> VintageStore:
    return VintageStore(revision_records)


class TestConstruction:
    def test_basic_properties(self, store: VintageStore) -> None:
        assert store.series == ["gdp", "ip"]
        assert store.frequencies.to_dict() == {"gdp": Frequency.QUARTERLY, "ip": Frequency.MONTHLY}
        assert store.base_frequency is Frequency.MONTHLY
        assert len(store) == store.n_records == 10
        assert store.first_vintage == pd.Timestamp("2020-03-10")
        assert store.last_vintage == pd.Timestamp("2020-12-03")
        assert "gdp" in store
        assert "n_series=2" in repr(store)
        records = store.records
        assert list(records.columns) == list(RECORD_COLUMNS)
        assert all(isinstance(p, pd.Period) for p in records["reference_period"])
        assert store.vintage_dates().size == 6
        assert store.vintage_dates("ip").strftime("%m-%d").tolist() == ["03-10", "04-08", "05-12"]

    def test_period_and_date_inputs(self) -> None:
        records = pd.DataFrame(
            {
                "series": ["a", "a"],
                "reference_period": [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-02-01")],
                "vintage_date": [pd.Timestamp("2020-02-15 10:00", tz="UTC")] * 2,
                "value": [1, 2],
            }
        )
        with pytest.raises(NowcastDataError, match="frequency is required"):
            VintageStore(records)
        store = VintageStore(records, frequencies={"a": "M"})
        assert store.as_of("2020-02-15")["a"].tolist() == [1.0, 2.0]

    def test_frequency_column(self) -> None:
        records = pd.DataFrame(
            {
                "series": ["g"],
                "frequency": ["Q"],
                "reference_period": ["2020-03"],
                "vintage_date": ["2020-05-29"],
                "value": [1.0],
            }
        )
        store = VintageStore(records)
        assert store.frequencies["g"] is Frequency.QUARTERLY
        assert str(store.records["reference_period"].iloc[0]) == "2020Q1"
        bad = pd.concat([records, records.assign(frequency="M", vintage_date="2020-06-01")])
        with pytest.raises(NowcastDataError, match="several frequencies"):
            VintageStore(bad)
        # explicit frequencies win over the column
        assert VintageStore(bad, frequencies={"g": "Q"}).n_records == 2

    @pytest.mark.parametrize(
        ("change", "match"),
        [
            (lambda r: r.drop(columns="value"), "lack columns"),
            (lambda r: r.assign(extra=1), "unknown columns"),
            (lambda r: r.iloc[:0], "empty"),
            (lambda r: r.assign(series=[None, *list(r["series"][1:])]), "non-empty"),
            (lambda r: r.assign(series=[" ", *list(r["series"][1:])]), "non-empty"),
            (lambda r: r.assign(value=["x", *list(r["value"][1:])]), "numeric"),
            (lambda r: r.assign(value=[np.inf, *list(r["value"][1:])]), "infinite"),
            (lambda r: r.assign(vintage_date=["bad", *list(r["vintage_date"][1:])]), "parsed"),
            (lambda r: r.assign(vintage_date=[None, *list(r["vintage_date"][1:])]), "missing"),
            (lambda r: pd.concat([r, r.iloc[:1]]), "Duplicated record"),
            (
                lambda r: r.assign(reference_period=["2020-01", *list(r["reference_period"][1:])]),
                "mix frequencies",
            ),
        ],
    )
    def test_invalid_records(
        self, revision_records: pd.DataFrame, change: object, match: str
    ) -> None:
        with pytest.raises(NowcastDataError, match=match):
            VintageStore(change(revision_records))  # type: ignore[operator]

    def test_invalid_types_and_frequencies(self, revision_records: pd.DataFrame) -> None:
        with pytest.raises(NowcastDataError, match="DataFrame"):
            VintageStore([1, 2])  # type: ignore[arg-type]
        with pytest.raises(NowcastDataError, match="unknown series"):
            VintageStore(revision_records, frequencies={"zzz": "M"})
        with pytest.raises(NowcastDataError, match="higher than the base"):
            VintageStore(revision_records, base_frequency="Q")

    def test_quarterly_base_grid(self, revision_records: pd.DataFrame) -> None:
        store = VintageStore(
            revision_records[revision_records["series"] == "gdp"], base_frequency="Q"
        )
        frame = store.latest()
        assert frame.index.freqstr.startswith("Q")
        assert frame["gdp"].tolist() == [-2.4, -9.6, 7.7]


class TestAsOf:
    def test_as_of_values(self, store: VintageStore) -> None:
        frame = store.as_of("2020-06-30")
        assert isinstance(frame, pd.DataFrame)
        assert list(frame.columns) == ["gdp", "ip"]
        assert frame.index.equals(pd.period_range("2020-01", "2020-09", freq="M"))
        assert frame["gdp"].dropna().to_dict() == {pd.Period("2020-03", "M"): -1.5}
        assert frame["ip"].dropna().tolist() == [1.1, -0.8, -9.0]

    def test_vintage_dates_are_inclusive(self, store: VintageStore) -> None:
        assert store.as_of("2020-04-07")["ip"].dropna().tolist() == [0.9]
        assert store.as_of("2020-04-08")["ip"].dropna().tolist() == [1.1, -0.8]

    def test_before_first_vintage(self, store: VintageStore) -> None:
        frame = store.as_of("2019-01-01")
        assert frame.isna().all().all()
        assert frame.shape == (9, 2)

    def test_quarterly_slots(self, store: VintageStore) -> None:
        frame = store.latest()
        gdp = frame["gdp"]
        assert gdp.dropna().index.month.tolist() == [3, 6, 9]
        assert gdp.dropna().tolist() == [-2.4, -9.6, 7.7]

    def test_selection_and_bounds(self, store: VintageStore) -> None:
        frame = store.as_of("2020-12-31", series="ip", start="2019-11", end="2020Q2")
        assert list(frame.columns) == ["ip"]
        assert frame.index[0] == pd.Period("2019-11", "M")
        assert frame.index[-1] == pd.Period("2020-06", "M")
        frame = store.as_of("2020-12-31", start="2020Q2", end=pd.Period("2020-07", "M"))
        assert frame.index[0] == pd.Period("2020-04", "M")
        assert frame.index[-1] == pd.Period("2020-07", "M")
        assert np.isnan(frame.loc[pd.Period("2020-06", "M"), "ip"])
        with pytest.raises(ValueError, match="must not be after"):
            store.as_of("2020-12-31", start="2021-01", end="2020-01")
        with pytest.raises(NowcastDataError, match="grid boundary"):
            store.as_of("2020-12-31", start="garbage")

    def test_as_mixed(self, store: VintageStore) -> None:
        mfd = store.as_of("2020-12-31", as_mixed=True, release_delays={"gdp": 60, "ip": 40})
        assert isinstance(mfd, MixedFrequencyData)
        assert mfd.quarterly_columns == ["gdp"]
        assert mfd.release_delays.to_dict() == {"gdp": 60, "ip": 40}
        with pytest.raises(ValueError, match="as_mixed=True"):
            store.as_of("2020-12-31", release_delays={"gdp": 60})

    def test_unknown_series(self, store: VintageStore) -> None:
        with pytest.raises(NowcastDataError, match="not in the vintage store"):
            store.as_of("2020-12-31", series=["zzz"])

    def test_withdrawn_value(self) -> None:
        records = pd.DataFrame(
            {
                "series": ["a", "a", "a"],
                "reference_period": ["2020-01", "2020-01", "2020-02"],
                "vintage_date": ["2020-02-10", "2020-03-10", "2020-03-10"],
                "value": [1.0, np.nan, 2.0],
            }
        )
        store = VintageStore(records)
        assert store.as_of("2020-02-28")["a"].tolist()[0] == 1.0
        assert np.isnan(store.as_of("2020-03-31")["a"].tolist()[0])


class TestRevisions:
    def test_revisions_table(self, store: VintageStore) -> None:
        rev = store.revisions("gdp")
        assert list(rev.columns) == [
            "reference_period",
            "vintage_date",
            "release_number",
            "value",
            "previous_value",
            "revision",
        ]
        assert [str(p) for p in rev["reference_period"]] == ["2020Q1"] * 3 + ["2020Q2"] * 2 + [
            "2020Q3"
        ]
        assert rev["release_number"].tolist() == [0, 1, 2, 0, 1, 0]
        np.testing.assert_allclose(
            rev["revision"].to_numpy(), [np.nan, -1.0, 0.1, np.nan, 0.1, np.nan], atol=1e-12
        )
        assert rev["previous_value"].iloc[1] == -1.5

    def test_unchanged_records(self) -> None:
        records = pd.DataFrame(
            {
                "series": ["a"] * 3,
                "reference_period": ["2020-01"] * 3,
                "vintage_date": ["2020-02-10", "2020-03-10", "2020-04-10"],
                "value": [1.0, 1.0, 1.5],
            }
        )
        store = VintageStore(records)
        assert store.revisions("a")["release_number"].tolist() == [0, 1]
        full = store.revisions("a", include_unchanged=True)
        assert full["release_number"].tolist() == [0, 0, 1]
        assert full["revision"].iloc[1] != full["revision"].iloc[1]  # NaN: still release 0
        assert full["revision"].iloc[2] == pytest.approx(0.5)

    def test_vintage_matrix(self, store: VintageStore) -> None:
        matrix = store.vintage_matrix("gdp")
        assert [str(p) for p in matrix.index] == ["2020Q1", "2020Q2", "2020Q3"]
        assert matrix.columns.strftime("%m-%d").tolist() == ["05-29", "09-01", "12-03"]
        np.testing.assert_allclose(
            matrix.to_numpy(),
            [[-1.5, -2.5, -2.4], [np.nan, -9.7, -9.6], [np.nan, np.nan, 7.7]],
        )
        # each column equals the as_of panel at that vintage
        for vintage in matrix.columns:
            known = store.as_of(vintage)["gdp"].dropna().to_numpy()
            np.testing.assert_allclose(matrix[vintage].dropna().to_numpy(), known)

    def test_vintage_matrix_withdrawal(self) -> None:
        records = pd.DataFrame(
            {
                "series": ["a", "a", "a"],
                "reference_period": ["2020-01", "2020-01", "2020-02"],
                "vintage_date": ["2020-02-10", "2020-03-10", "2020-04-10"],
                "value": [1.0, np.nan, 2.0],
            }
        )
        matrix = VintageStore(records).vintage_matrix("a")
        np.testing.assert_allclose(
            matrix.to_numpy(), [[1.0, np.nan, np.nan], [np.nan, np.nan, 2.0]]
        )

    def test_nth_release(self, store: VintageStore) -> None:
        first = store.nth_release(0)
        assert first["gdp"].dropna().tolist() == [-1.5, -9.7, 7.7]
        assert first["ip"].dropna().tolist() == [0.9, -0.8, -9.0]
        second = store.nth_release(1, series="gdp")
        assert second["gdp"].dropna().tolist() == [-2.5, -9.6]
        latest = store.nth_release(-1)
        pd.testing.assert_frame_equal(latest, store.latest())
        second_last = store.nth_release(-2, series=["gdp"])
        assert second_last["gdp"].dropna().tolist() == [-2.5, -9.7]
        with pytest.raises(ValueError, match="integer"):
            store.nth_release(1.0)  # type: ignore[arg-type]
        assert store.nth_release(5).isna().all().all()

    def test_revision_summary(self, store: VintageStore) -> None:
        summary = store.revision_summary()
        assert list(summary.index) == ["gdp", "ip"]
        gdp = summary.loc["gdp"]
        total = np.array([-0.9, 0.1, 0.0])
        latest = np.array([-2.4, -9.6, 7.7])
        assert gdp["n_periods"] == 3
        assert gdp["n_revised"] == 2
        assert gdp["mean_n_revisions"] == pytest.approx(1.0)
        assert gdp["mean_revision"] == pytest.approx(total.mean())
        assert gdp["mean_abs_revision"] == pytest.approx(np.abs(total).mean())
        assert gdp["rms_revision"] == pytest.approx(np.sqrt((total**2).mean()))
        assert gdp["std_revision"] == pytest.approx(total.std(ddof=1))
        assert gdp["max_abs_revision"] == pytest.approx(0.9)
        assert gdp["noise_to_signal"] == pytest.approx(total.std(ddof=1) / latest.std(ddof=1))
        ip = store.revision_summary("ip").loc["ip"]
        assert ip["n_revised"] == 1
        assert ip["mean_revision"] == pytest.approx(0.2 / 3)

    def test_revision_summary_degenerate(self) -> None:
        records = pd.DataFrame(
            {
                "series": ["a"],
                "reference_period": ["2020-01"],
                "vintage_date": ["2020-02-10"],
                "value": [1.0],
            }
        )
        row = VintageStore(records).revision_summary().loc["a"]
        assert row["mean_revision"] == 0.0
        assert np.isnan(row["std_revision"])
        assert np.isnan(row["noise_to_signal"])
        withdrawn = pd.concat(
            [records, records.assign(vintage_date="2020-03-10", value=np.nan)], ignore_index=True
        )
        row = VintageStore(withdrawn).revision_summary().loc["a"]
        assert np.isnan(row["mean_revision"])
        assert np.isnan(row["max_abs_revision"])


class TestDerivation:
    def test_to_frame_and_select(self, store: VintageStore) -> None:
        frame = store.to_frame()
        assert list(frame.columns) == [
            "series",
            "frequency",
            "reference_period",
            "vintage_date",
            "value",
        ]
        assert set(frame["frequency"]) == {"Q", "M"}
        sub = store.select(["ip"])
        assert sub.series == ["ip"]
        assert len(sub) == 4
        with pytest.raises(NowcastDataError):
            store.select("zzz")

    def test_add(self, store: VintageStore, revision_records: pd.DataFrame) -> None:
        extra = pd.DataFrame(
            {
                "series": ["gdp"],
                "reference_period": ["2020Q3"],
                "vintage_date": ["2021-03-01"],
                "value": [7.5],
            }
        )
        bigger = store.add(extra)
        assert len(bigger) == len(store) + 1
        assert bigger.latest()["gdp"].dropna().tolist()[-1] == 7.5
        assert len(store) == 10  # immutable
        with pytest.raises(NowcastDataError, match="Duplicated"):
            store.add(revision_records.iloc[:1])
        monthly_gdp = VintageStore(
            pd.DataFrame(
                {
                    "series": ["gdp"],
                    "reference_period": ["2021-01"],
                    "vintage_date": ["2021-03-01"],
                    "value": [1.0],
                }
            )
        )
        with pytest.raises(NowcastDataError, match="Frequency of 'gdp' differs"):
            store.add(monthly_gdp)
        assert store.add(store.select("ip").add(extra).select("gdp")).n_records == 11

    def test_equals(self, store: VintageStore, revision_records: pd.DataFrame) -> None:
        shuffled = revision_records.sample(frac=1.0, random_state=1)
        assert store.equals(VintageStore(shuffled))
        assert not store.equals(store.select("gdp"))
        assert not store.equals("store")
        gdp = revision_records[revision_records["series"] == "gdp"]
        assert not VintageStore(gdp).equals(VintageStore(gdp, base_frequency="Q"))


class TestConstructors:
    def test_from_vintages_roundtrip(self, final_frame: pd.DataFrame) -> None:
        vintages = {
            v.date: v.data
            for v in generate_vintages(
                final_frame, DELAYS, "2018-02-01", "2019-12-01", "M", frequency=FREQS
            )
        }
        # simulate revisions: later vintages revise ip by +0.1 for observations > 2 months old
        revised = {}
        for i, (date, frame) in enumerate(vintages.items()):
            frame = frame.copy()
            if i % 3 == 2:
                frame.iloc[:-3, 0] += 0.1
            revised[date] = frame
        store = VintageStore.from_vintages(revised, frequencies=FREQS)
        for date, frame in revised.items():
            got = store.as_of(date, start=frame.index[0], end=frame.index[-1])
            pd.testing.assert_frame_equal(got, frame, check_names=False, check_freq=False)
        assert store.revisions("ip")["release_number"].max() >= 1
        full = VintageStore.from_vintages(revised, frequencies=FREQS, drop_unchanged=False)
        assert len(full) > len(store)
        assert full.latest().equals(store.latest())

    def test_from_vintages_mixed_and_withdrawal(self) -> None:
        idx = pd.period_range("2020-01", periods=3, freq="M")
        v1 = MixedFrequencyData(pd.DataFrame({"a": [1.0, 2.0, np.nan]}, index=idx), "M")
        v2 = MixedFrequencyData(pd.DataFrame({"a": [np.nan, 2.0, 3.0]}, index=idx), "M")
        store = VintageStore.from_vintages({"2020-03-01": v1, "2020-04-01": v2})
        assert store.as_of("2020-03-15")["a"].tolist() == [1.0, 2.0, np.nan] or np.allclose(
            store.as_of("2020-03-15")["a"].to_numpy(), [1.0, 2.0, np.nan], equal_nan=True
        )
        np.testing.assert_allclose(store.latest()["a"].to_numpy(), [np.nan, 2.0, 3.0])
        assert len(store) == 4  # 2 first releases, 1 withdrawal, 1 new

    def test_from_vintages_errors(self) -> None:
        idx = pd.period_range("2020-01", periods=6, freq="M")
        monthly = pd.DataFrame({"a": np.arange(6.0)}, index=idx)
        quarterly = pd.DataFrame({"a": [np.nan, np.nan, 1.0, np.nan, np.nan, 2.0]}, index=idx)
        with pytest.raises(NowcastDataError, match="empty"):
            VintageStore.from_vintages({})
        with pytest.raises(NowcastDataError, match="changes frequency"):
            VintageStore.from_vintages(
                {"2020-07-01": monthly, "2020-08-01": quarterly}, frequencies=None
            )
        with pytest.raises(NowcastDataError, match="Duplicated vintage dates"):
            VintageStore.from_vintages({"2020-07-01": monthly, pd.Timestamp("2020-07-01"): monthly})
        empty = pd.DataFrame({"a": [np.nan] * 6}, index=idx)
        with pytest.raises(NowcastDataError, match="no observations"):
            VintageStore.from_vintages({"2020-07-01": empty}, frequencies="M")
        q_idx = pd.period_range("2020Q1", periods=2, freq="Q")
        q_base = MixedFrequencyData(pd.DataFrame({"a": [1.0, 2.0]}, index=q_idx), "Q")
        m_base = MixedFrequencyData(monthly, "M")
        with pytest.raises(NowcastDataError, match="different base frequencies"):
            VintageStore.from_vintages({"2020-07-01": m_base, "2020-08-01": q_base})

    @pytest.mark.parametrize(
        "calendar",
        [None, DELAYS, pd.Series(DELAYS), ReleaseCalendar(DELAYS)],
    )
    def test_from_calendar_matches_pseudo_real_time(
        self, final_panel: MixedFrequencyData, calendar: object
    ) -> None:
        store = VintageStore.from_calendar(final_panel, calendar)  # type: ignore[arg-type]
        assert store.frequencies.to_dict() == {
            n: m.frequency for n, m in final_panel.metadata.items()
        }
        for vintage in pd.date_range("2018-01-01", "2021-03-31", freq="23D"):
            expected = pseudo_real_time(final_panel, None, vintage).data
            got = store.as_of(vintage, start=final_panel.start, end=final_panel.end)
            pd.testing.assert_frame_equal(got, expected, check_names=False, check_freq=False)
        assert store.revisions("ip")["release_number"].max() == 0

    def test_from_calendar_empty(self) -> None:
        idx = pd.period_range("2020-01", periods=3, freq="M")
        frame = pd.DataFrame({"a": [np.nan] * 3}, index=idx)
        with (
            pytest.warns(DataQualityWarning),
            pytest.raises(NowcastDataError, match="no observations"),
        ):
            VintageStore.from_calendar(frame, {"a": 1}, frequency="M")


class TestIO:
    def test_csv_roundtrip(self, store: VintageStore, tmp_path: Path) -> None:
        path = tmp_path / "vintages.csv"
        store.to_csv(path, index=True)
        loaded = VintageStore.from_csv(path)
        assert loaded.equals(store)
        pd.testing.assert_frame_equal(loaded.as_of("2020-10-01"), store.as_of("2020-10-01"))

    def test_parquet_roundtrip(self, store: VintageStore, tmp_path: Path) -> None:
        pytest.importorskip("pyarrow")
        path = tmp_path / "vintages.parquet"
        store.to_parquet(path, index=True)
        loaded = VintageStore.from_parquet(path)
        assert loaded.equals(store)
        assert loaded.frequencies.to_dict() == store.frequencies.to_dict()

    @pytest.mark.parametrize("freq", ["D", "W", "A"])
    def test_roundtrip_other_frequencies(self, tmp_path: Path, freq: str) -> None:
        periods = pd.period_range("2020-01-06", periods=3, freq={"A": "Y"}.get(freq, freq))
        records = pd.DataFrame(
            {
                "series": ["x"] * 3,
                "reference_period": list(periods),
                "vintage_date": ["2025-01-01"] * 3,
                "value": [1.0, 2.0, 3.0],
            }
        )
        store = VintageStore(records, base_frequency=freq)
        path = tmp_path / "x.csv"
        store.to_csv(path)
        loaded = VintageStore.from_csv(path, base_frequency=freq)
        assert loaded.equals(store)
        assert loaded.latest()["x"].tolist() == [1.0, 2.0, 3.0]
