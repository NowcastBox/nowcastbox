from __future__ import annotations

import pickle
import warnings

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from nowcastbox.core.data import (
    MixedFrequencyData,
    SeriesCategory,
    SeriesMetadata,
    StandardizationStats,
    as_mixed_frequency_data,
)
from nowcastbox.core.exceptions import DataQualityWarning, NowcastDataError
from nowcastbox.core.frequency import AggregationType, Frequency
from tests.core.conftest import make_panel_frame

# --------------------------------------------------------------------------- metadata


class TestSeriesCategory:
    def test_parse(self):
        assert SeriesCategory.from_value(" Financial ") is SeriesCategory.FINANCIAL
        assert SeriesCategory.from_value(SeriesCategory.HARD) is SeriesCategory.HARD

    @pytest.mark.parametrize("value", ["medium", 1, None])
    def test_invalid(self, value):
        with pytest.raises(ValueError, match="Unknown series category"):
            SeriesCategory.from_value(value)


class TestSeriesMetadata:
    def test_coercion(self):
        m = SeriesMetadata(
            "gdp",
            "quarterly",
            transform=np.nan,
            release_delay=60.0,
            blocks="global",
            category="hard",
            description=None,  # type: ignore[arg-type]
            aggregation="mm",
        )
        assert m.frequency is Frequency.QUARTERLY
        assert m.transform is None
        assert m.release_delay == 60
        assert m.blocks == ("global",)
        assert m.category is SeriesCategory.HARD
        assert m.description == ""
        assert m.aggregation is AggregationType.GROWTH_RATE

    def test_missing_values_become_none(self):
        m = SeriesMetadata("x", "M", release_delay=pd.NA, blocks=np.nan, category=np.nan)
        assert m.release_delay is None
        assert m.blocks == ()
        assert m.category is None

    @pytest.mark.parametrize("delay", [-366, 1.5, "10", True])
    def test_invalid_delay(self, delay):
        with pytest.raises(ValueError, match="release_delay"):
            SeriesMetadata("x", "M", release_delay=delay)

    def test_invalid_blocks(self):
        with pytest.raises(ValueError, match="Duplicated block"):
            SeriesMetadata("x", "M", blocks=["a", "a"])
        with pytest.raises(ValueError, match="blocks must be"):
            SeriesMetadata("x", "M", blocks=3)  # type: ignore[arg-type]

    def test_invalid_name(self):
        with pytest.raises(ValueError, match="non-empty string"):
            SeriesMetadata("", "M")

    def test_replace_and_dict(self):
        m = SeriesMetadata("x", "M", blocks=("a", "b"))
        m2 = m.replace(release_delay=5, category="soft")
        assert m2.release_delay == 5 and m.release_delay is None
        d = m2.to_dict()
        assert d == {
            "name": "x",
            "frequency": "M",
            "transform": None,
            "release_delay": 5,
            "blocks": ["a", "b"],
            "category": "soft",
            "description": "",
            "aggregation": None,
            "units": "",
            "transform_applied": False,
        }
        assert SeriesMetadata(**d) == m2

    def test_frozen(self):
        m = SeriesMetadata("x", "M")
        with pytest.raises(AttributeError):
            m.release_delay = 3  # type: ignore[misc]


# --------------------------------------------------------------------------- stats


class TestStandardizationStats:
    def test_validation(self):
        with pytest.raises(ValueError, match="same series"):
            StandardizationStats(pd.Series({"a": 0.0}), pd.Series({"b": 1.0}))
        with pytest.raises(ValueError, match="strictly positive"):
            StandardizationStats(pd.Series({"a": 0.0}), pd.Series({"a": 0.0}))
        with pytest.raises(ValueError, match="strictly positive"):
            StandardizationStats(pd.Series({"a": 0.0}), pd.Series({"a": np.nan}))

    def test_transform_roundtrip(self):
        s = StandardizationStats(pd.Series({"a": 1.0, "b": -2.0}), pd.Series({"a": 2.0, "b": 4.0}))
        df = pd.DataFrame({"b": [2.0, -2.0], "a": [3.0, 1.0]})
        z = s.transform(df)
        assert z["a"].tolist() == [1.0, 0.0]
        assert z["b"].tolist() == [1.0, 0.0]
        pd.testing.assert_frame_equal(s.inverse_transform(z), df)
        assert s.columns == ["a", "b"]

    def test_unknown_columns(self):
        s = StandardizationStats(pd.Series({"a": 1.0}), pd.Series({"a": 2.0}))
        with pytest.raises(NowcastDataError, match="No standardisation"):
            s.transform(pd.DataFrame({"z": [1.0]}))
        with pytest.raises(NowcastDataError):
            s.inverse_series(1.0, "z")

    def test_inverse_series_types(self):
        s = StandardizationStats(pd.Series({"a": 1.0}), pd.Series({"a": 2.0}))
        np.testing.assert_allclose(s.inverse_series(np.array([0.0, 1.0]), "a"), [1.0, 3.0])
        out = s.inverse_series(pd.Series([1.0]), "a", scale_only=True)
        assert out.tolist() == [2.0]


# --------------------------------------------------------------------------- construction


class TestConstruction:
    def test_basic(self, panel, panel_frame):
        assert panel.n_series == 3
        assert panel.n_periods == 24
        assert panel.shape == (24, 3)
        assert len(panel) == 24
        assert panel.base_frequency is Frequency.MONTHLY
        assert panel.columns == ["ip", "pmi", "gdp"]
        assert list(panel) == panel.columns
        assert "gdp" in panel and "foo" not in panel
        assert panel.monthly_columns == ["ip", "pmi"]
        assert panel.quarterly_columns == ["gdp"]
        assert panel.is_mixed_frequency
        assert str(panel.start) == "2018-01" and str(panel.end) == "2019-12"
        pd.testing.assert_frame_equal(
            panel.to_frame(), panel_frame.rename_axis("period"), check_freq=False
        )
        assert panel.index.name == "period"

    def test_defensive_copies(self, panel):
        df = panel.to_frame()
        df.iloc[0, 0] = 999.0
        assert panel["ip"].iloc[0] != 999.0
        values = panel.values
        values[0, 0] = 999.0
        assert panel.values[0, 0] != 999.0
        s = panel["ip"]
        s.iloc[0] = 999.0
        assert panel.data["ip"].iloc[0] != 999.0

    def test_input_frame_not_modified(self, panel_frame):
        original = panel_frame.copy()
        MixedFrequencyData(panel_frame, {"ip": "M", "pmi": "M", "gdp": "Q"})
        pd.testing.assert_frame_equal(panel_frame, original)

    def test_metadata_properties(self, panel):
        assert panel.frequencies.to_dict() == {
            "ip": Frequency.MONTHLY,
            "pmi": Frequency.MONTHLY,
            "gdp": Frequency.QUARTERLY,
        }
        assert panel.release_delays.tolist() == [40, 1, 60]
        assert panel.transforms.tolist() == [2, 0, 2]
        assert panel.categories["pmi"] is SeriesCategory.SOFT
        assert panel.block_names == ["global", "real", "soft"]
        blocks = panel.blocks
        assert blocks.loc["pmi"].tolist() == [True, False, True]
        assert blocks.index.name == "series"
        assert panel.metadata["gdp"].description == "Real GDP growth"
        mf = panel.metadata_frame()
        assert mf.loc["gdp", "frequency"] == "Q"
        assert mf.loc["ip", "category"] == "hard"
        assert "name" not in mf.columns

    def test_frequency_inference(self, panel_frame):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            mfd = MixedFrequencyData(panel_frame)
        assert mfd.frequencies["gdp"] is Frequency.QUARTERLY
        assert mfd.frequencies["ip"] is Frequency.MONTHLY

    def test_partial_frequencies_inferred(self, panel_frame):
        mfd = MixedFrequencyData(panel_frame, {"ip": "M"})
        assert mfd.frequencies["gdp"] is Frequency.QUARTERLY

    def test_frequency_specs(self, panel_frame):
        a = MixedFrequencyData(panel_frame, pd.Series({"ip": 12, "pmi": 12, "gdp": 4}))
        b = MixedFrequencyData(panel_frame, ["M", "M", "Q"])
        c = MixedFrequencyData(panel_frame, pd.Series([12, 12, 4]))
        d = MixedFrequencyData(panel_frame[["ip", "pmi"]], "M")
        assert a.equals(b) and b.equals(c)
        assert d.frequencies.tolist() == [Frequency.MONTHLY] * 2

    def test_frequency_spec_length_mismatch(self, panel_frame):
        with pytest.raises(NowcastDataError, match="aligned"):
            MixedFrequencyData(panel_frame, ["M", "Q"])

    def test_metadata_objects_and_overrides(self, panel_frame):
        meta = {
            "gdp": SeriesMetadata("ignored-name", "Q", release_delay=90),
            "ip": {"frequency": "M", "category": "hard"},
        }
        mfd = MixedFrequencyData(panel_frame, metadata=meta, release_delays={"gdp": 60})
        assert mfd.metadata["gdp"].release_delay == 60  # keyword overrides metadata
        assert mfd.metadata["gdp"].name == "gdp"
        assert mfd.metadata["ip"].category is SeriesCategory.HARD
        assert mfd.metadata["pmi"].frequency is Frequency.MONTHLY  # inferred

    def test_blocks_dataframe(self, panel_frame):
        blocks = pd.DataFrame(
            {"global": [1, 1, 1], "soft": [0, 1, 0], "real": [1, 0, np.nan]},
            index=["ip", "pmi", "gdp"],
        )
        mfd = MixedFrequencyData(panel_frame, {"gdp": "Q"}, blocks=blocks)
        assert mfd.block_names == ["global", "soft", "real"]
        assert mfd.metadata["pmi"].blocks == ("global", "soft")
        assert mfd.metadata["gdp"].blocks == ("global",)

    def test_datetime_index(self, panel_frame):
        df = panel_frame.copy()
        df.index = df.index.to_timestamp(how="start")
        mfd = MixedFrequencyData(df, {"gdp": "Q"})
        assert mfd.base_frequency is Frequency.MONTHLY
        assert isinstance(mfd.index, pd.PeriodIndex)
        df2 = df.copy()
        df2.index = df.index.to_period("M").to_timestamp(how="end")
        assert MixedFrequencyData(df2, {"gdp": "Q"}, base_frequency="M").n_periods == 24

    def test_datetime_index_without_inferable_frequency(self):
        idx = pd.DatetimeIndex(["2020-01-01", "2020-01-05", "2020-03-01"])
        with pytest.raises(NowcastDataError, match="base_frequency"):
            MixedFrequencyData(pd.DataFrame({"a": [1.0, 2.0, 3.0]}, index=idx))

    def test_base_frequency_conflict(self, panel_frame):
        with pytest.raises(NowcastDataError, match="disagrees"):
            MixedFrequencyData(panel_frame, base_frequency="Q")

    def test_quarterly_base_grid(self):
        idx = pd.period_range("2015Q1", periods=8, freq="Q")
        df = pd.DataFrame(
            {"q": np.arange(8.0), "a": [np.nan, np.nan, np.nan, 1.0, np.nan, np.nan, np.nan, 2.0]},
            index=idx,
        )
        mfd = MixedFrequencyData(df)
        assert mfd.base_frequency is Frequency.QUARTERLY
        assert mfd.frequencies["a"] is Frequency.ANNUAL
        assert mfd.frequency_ratio("a") == 4

    @pytest.mark.parametrize(
        ("frame", "match"),
        [
            (pd.DataFrame(), "at least one row"),
            (
                pd.DataFrame({"a": [1.0]}, index=pd.RangeIndex(1)),
                "PeriodIndex or DatetimeIndex",
            ),
            (
                pd.DataFrame(
                    [[1.0, 2.0]], columns=["a", "a"], index=pd.period_range("2020-01", periods=1)
                ),
                "Duplicated column",
            ),
            (
                pd.DataFrame({0: [1.0]}, index=pd.period_range("2020-01", periods=1, freq="M")),
                "non-empty strings",
            ),
            (
                pd.DataFrame({"a": ["x"]}, index=pd.period_range("2020-01", periods=1, freq="M")),
                "numeric",
            ),
            (
                pd.DataFrame(
                    {"a": [np.inf]}, index=pd.period_range("2020-01", periods=1, freq="M")
                ),
                "Infinite",
            ),
            (
                pd.DataFrame(
                    {"a": [1.0, 2.0]},
                    index=pd.PeriodIndex(["2020-01", "2020-01"], freq="M"),
                ),
                "Duplicated periods",
            ),
        ],
    )
    def test_invalid_frames(self, frame, match):
        with pytest.raises(NowcastDataError, match=match):
            MixedFrequencyData(frame)

    def test_not_a_dataframe(self):
        with pytest.raises(NowcastDataError, match="DataFrame"):
            MixedFrequencyData(np.zeros((3, 2)))  # type: ignore[arg-type]

    def test_value_off_slot(self, panel_frame):
        df = panel_frame.copy()
        df.iloc[0, 2] = 1.0  # January value for a quarterly series
        with pytest.raises(NowcastDataError, match="outside the last monthly period"):
            MixedFrequencyData(df, {"gdp": "Q"})

    def test_frequency_higher_than_base(self, panel_frame):
        with pytest.raises(NowcastDataError, match="higher than the monthly base"):
            MixedFrequencyData(panel_frame, {"ip": "D"})

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"frequencies": {"zzz": "M"}},
            {"metadata": {"zzz": {"frequency": "M"}}},
            {"release_delays": {"zzz": 1}},
            {"blocks": {"zzz": ["g"]}},
            {"categories": {"zzz": "hard"}},
        ],
    )
    def test_unknown_series_in_metadata(self, panel_frame, kwargs):
        with pytest.raises(NowcastDataError, match="unknown series"):
            MixedFrequencyData(panel_frame, **kwargs)

    def test_invalid_metadata_value(self, panel_frame):
        with pytest.raises(NowcastDataError, match="Invalid metadata"):
            MixedFrequencyData(panel_frame, categories={"ip": "medium"})
        with pytest.raises(NowcastDataError, match="Invalid metadata"):
            MixedFrequencyData(panel_frame, metadata={"ip": {"colour": "red"}})

    def test_unsorted_index_is_sorted(self, panel_frame):
        shuffled = panel_frame.iloc[::-1]
        mfd = MixedFrequencyData(shuffled, {"gdp": "Q"})
        assert mfd.index.is_monotonic_increasing

    def test_gaps_are_filled_with_warning(self, panel_frame):
        df = panel_frame.drop(panel_frame.index[[4, 5]])
        with pytest.warns(DataQualityWarning, match="2 missing monthly periods"):
            mfd = MixedFrequencyData(df, {"gdp": "Q"})
        assert mfd.n_periods == 24
        assert mfd["ip"].iloc[4:6].isna().all()

    def test_empty_series_warns(self, panel_frame):
        df = panel_frame.copy()
        df["empty"] = np.nan
        with pytest.warns(DataQualityWarning, match="without any observation"):
            MixedFrequencyData(df, {"gdp": "Q", "empty": "M"})

    def test_integer_data_is_coerced(self):
        idx = pd.period_range("2020-01", periods=3, freq="M")
        mfd = MixedFrequencyData(pd.DataFrame({"a": [1, 2, 3]}, index=idx), "M")
        assert mfd.to_frame()["a"].dtype == np.float64

    def test_from_series(self):
        gdp = pd.Series([1.0, 2.0, 3.0], index=pd.period_range("2020Q1", periods=3, freq="Q"))
        ip = pd.Series(np.arange(10.0), index=pd.period_range("2019-12", periods=10, freq="M"))
        mfd = MixedFrequencyData.from_series({"gdp": gdp, "ip": ip}, release_delays={"gdp": 60})
        assert str(mfd.start) == "2019-12" and str(mfd.end) == "2020-09"
        assert mfd.frequencies["gdp"] is Frequency.QUARTERLY
        assert mfd.to_native("gdp", dropna=True).tolist() == [1.0, 2.0, 3.0]
        assert mfd.release_delays["gdp"] == 60

    def test_from_series_errors(self):
        with pytest.raises(NowcastDataError, match="at least one"):
            MixedFrequencyData.from_series({})
        with pytest.raises(NowcastDataError, match="PeriodIndex"):
            MixedFrequencyData.from_series({"a": pd.Series([1.0])})
        d = pd.Series([1.0], index=pd.period_range("2020-01-01", periods=1, freq="D"))
        with pytest.raises(NowcastDataError, match="Cannot map"):
            MixedFrequencyData.from_series({"d": d})
        q = pd.Series([1.0], index=pd.period_range("2020Q1", periods=1, freq="Q"))
        with pytest.raises(TypeError, match="frequencies"):
            MixedFrequencyData.from_series({"q": q}, frequencies={"q": "Q"})


# --------------------------------------------------------------------------- queries


class TestQueries:
    def test_series_by_frequency(self, panel):
        groups = panel.series_by_frequency()
        assert list(groups) == [Frequency.MONTHLY, Frequency.QUARTERLY]
        assert groups[Frequency.MONTHLY] == ["ip", "pmi"]

    def test_frequency_ratio(self, panel):
        assert panel.frequency_ratio("gdp") == 3
        assert panel.frequency_ratio("ip") == 1
        with pytest.raises(KeyError, match="Unknown series"):
            panel.frequency_ratio("zzz")

    def test_masks(self, panel):
        obs = panel.observation_mask()
        slots = panel.slot_mask()
        missing = panel.missing_mask()
        assert obs.shape == slots.shape == missing.shape == panel.shape
        assert slots["ip"].all()
        assert slots["gdp"].sum() == 8
        assert not (obs & ~slots).to_numpy().any()
        assert missing["gdp"].sum() == 1  # last quarter not released
        assert missing["ip"].sum() == 1

    def test_ragged_edge(self, panel):
        mask = panel.ragged_edge_mask()
        assert mask["ip"].tolist() == [False] * 23 + [True]
        assert mask["pmi"].sum() == 0
        assert mask["gdp"].sum() == 1
        assert mask["gdp"].iloc[-1]

    def test_ragged_edge_ignores_interior_gaps(self, panel_frame):
        df = panel_frame.copy()
        df.iloc[3, 1] = np.nan
        mfd = MixedFrequencyData(df, {"gdp": "Q"})
        assert not mfd.ragged_edge_mask()["pmi"].any()
        assert mfd.missing_mask()["pmi"].sum() == 1

    def test_last_observed_and_counts(self, panel):
        last = panel.last_observed()
        assert str(last["ip"]) == "2019-11"
        assert str(last["gdp"]) == "2019-09"
        assert panel.n_observations().tolist() == [23, 24, 7]

    def test_last_observed_empty_series(self, panel_frame):
        df = panel_frame.copy()
        df["e"] = np.nan
        with pytest.warns(DataQualityWarning):
            mfd = MixedFrequencyData(df, {"gdp": "Q", "e": "M"})
        assert pd.isna(mfd.last_observed()["e"])
        assert mfd.ragged_edge_mask()["e"].all()

    def test_to_native(self, panel):
        native = panel.to_native("gdp")
        assert len(native) == 8
        assert native.index.freqstr.startswith("Q")
        assert native.isna().sum() == 1
        assert len(panel.to_native("gdp", dropna=True)) == 7
        assert len(panel.to_native("ip")) == 24

    def test_getitem_unknown(self, panel):
        with pytest.raises(KeyError):
            panel["zzz"]

    def test_repr(self, panel):
        text = repr(panel)
        assert "n_series=3" in text and "monthly=2" in text and "quarterly=1" in text


# --------------------------------------------------------------------------- derived


class TestDerived:
    def test_copy_equals(self, panel):
        other = panel.copy()
        assert other.equals(panel)
        assert other is not panel
        assert not panel.equals("not a panel")

    def test_equals_detects_differences(self, panel):
        df = panel.to_frame()
        df.iloc[0, 0] += 1.0
        assert not panel.equals(panel.with_data(df))
        assert not panel.equals(panel.with_metadata("ip", release_delay=3))
        assert not panel.equals(panel.select(["ip", "gdp"]))

    def test_select_and_drop(self, panel):
        sub = panel.select(["gdp", "ip"])
        assert sub.columns == ["gdp", "ip"]
        assert sub.block_names == ["global", "real"]
        assert sub.metadata["gdp"].release_delay == 60
        assert panel.select("pmi").columns == ["pmi"]
        assert panel.drop("pmi").columns == ["ip", "gdp"]
        assert panel.drop(["pmi", "ip"]).columns == ["gdp"]

    def test_select_errors(self, panel):
        with pytest.raises(NowcastDataError, match="at least one"):
            panel.select([])
        with pytest.raises(NowcastDataError, match="Duplicated"):
            panel.select(["ip", "ip"])
        with pytest.raises(KeyError):
            panel.select(["zzz"])
        with pytest.raises(KeyError):
            panel.drop(["zzz"])

    def test_truncate(self, panel):
        t = panel.truncate(start="2018Q2", end="2019Q1")
        assert str(t.start) == "2018-04" and str(t.end) == "2019-03"
        t2 = panel.truncate(start=pd.Period("2018-05", "M"))
        assert str(t2.start) == "2018-05" and t2.end == panel.end
        t3 = panel.truncate(end="2018")
        assert str(t3.end) == "2018-12"
        assert panel.truncate().equals(panel)
        with pytest.raises(NowcastDataError, match="No periods"):
            panel.truncate(start="2030-01")

    def test_extend(self, panel):
        ext = panel.extend(6)
        assert ext.n_periods == 30
        assert ext.to_frame().iloc[-6:].isna().all().all()
        assert ext.metadata == panel.metadata
        assert panel.extend(0).equals(panel)
        with pytest.raises(ValueError, match="non-negative"):
            panel.extend(-1)

    def test_with_data(self, panel):
        doubled = panel.with_data(panel.to_frame() * 2)
        np.testing.assert_allclose(doubled.values, panel.values * 2)
        assert doubled.metadata == panel.metadata
        with pytest.raises(NowcastDataError, match="unknown series"):
            panel.with_data(pd.DataFrame({"zzz": [1.0]}, index=panel.index[:1]))
        bad = panel.to_frame()
        bad.iloc[0, 2] = 1.0
        with pytest.raises(NowcastDataError, match="outside"):
            panel.with_data(bad)

    def test_negative_delay_allowed(self):
        assert SeriesMetadata("x", "M", release_delay=-14).release_delay == -14

    def test_with_metadata(self, panel):
        new = panel.with_metadata("pmi", category="financial", blocks=("global",))
        assert new.metadata["pmi"].category is SeriesCategory.FINANCIAL
        assert new.block_names == ["global", "real"]  # unused blocks are dropped
        assert new.blocks.loc["pmi"].tolist() == [True, False]
        assert panel.metadata["pmi"].category is SeriesCategory.SOFT
        with pytest.raises(NowcastDataError, match="Renaming"):
            panel.with_metadata("pmi", name="x")
        with pytest.raises(NowcastDataError, match="Invalid metadata"):
            panel.with_metadata("pmi", release_delay=-400)
        with pytest.raises(NowcastDataError, match="outside"):
            panel.with_metadata("ip", frequency="Q")
        with pytest.raises(KeyError):
            panel.with_metadata("zzz", release_delay=1)

    def test_pickle_roundtrip(self, panel):
        clone = pickle.loads(pickle.dumps(panel))
        assert clone.equals(panel)
        assert clone.block_names == panel.block_names


# --------------------------------------------------------------------------- as_of


class TestAsOf:
    def test_release_rules(self, panel):
        # ip: 40 days, pmi: 1 day, gdp: 60 days
        v = panel.as_of("2019-06-01")
        last = v.last_observed()
        assert str(last["pmi"]) == "2019-05"  # May released on 1 June
        assert str(last["ip"]) == "2019-03"  # April is released on 9 June
        assert str(last["gdp"]) == "2019-03"  # Q1 released 30 May

    def test_exact_release_day_inclusive(self, panel):
        assert str(panel.as_of("2019-06-09").last_observed()["ip"]) == "2019-04"
        assert str(panel.as_of("2019-06-08").last_observed()["ip"]) == "2019-03"

    def test_grid_unchanged_and_monotone(self, panel):
        v1 = panel.as_of("2019-03-15")
        v2 = panel.as_of("2019-09-15")
        assert v1.index.equals(panel.index)
        assert v1.n_observations().le(v2.n_observations()).all()
        assert v2.n_observations().le(panel.n_observations()).all()

    def test_override_and_errors(self, panel_frame, panel):
        mfd = MixedFrequencyData(panel_frame, {"gdp": "Q"})
        with pytest.raises(NowcastDataError, match="No release delay"):
            mfd.as_of("2019-01-01")
        v = mfd.as_of("2019-01-01", release_delays={"ip": 0, "pmi": 0, "gdp": 0})
        assert str(v.last_observed()["gdp"]) == "2018-12"
        with pytest.raises(NowcastDataError, match="unknown series"):
            panel.as_of("2019-01-01", release_delays={"zzz": 1})


# --------------------------------------------------------------------------- standardisation


class TestStandardize:
    def test_moments(self, panel):
        z, stats = panel.standardize()
        frame = z.to_frame()
        np.testing.assert_allclose(frame.mean(), 0.0, atol=1e-12)
        np.testing.assert_allclose(frame.std(ddof=1), 1.0)
        assert stats.ddof == 1
        assert z.metadata == panel.metadata

    def test_roundtrip(self, panel):
        z, stats = panel.standardize(ddof=0)
        np.testing.assert_allclose(z.to_frame().std(ddof=0), 1.0)
        assert z.destandardize(stats).equals(panel, atol=1e-10)

    def test_external_stats(self, panel):
        stats = panel.truncate(end="2018-12").standardization_stats()
        z, used = panel.standardize(stats)
        assert used is stats
        expected = (panel["ip"] - stats.mean["ip"]) / stats.std["ip"]
        np.testing.assert_allclose(z["ip"], expected)

    def test_constant_series_rejected(self, panel_frame):
        df = panel_frame.copy()
        df["const"] = 1.0
        mfd = MixedFrequencyData(df, {"gdp": "Q"})
        with pytest.raises(NowcastDataError, match="const"):
            mfd.standardize()

    def test_stats_missing_series(self, panel):
        stats = panel.select(["ip"]).standardization_stats()
        with pytest.raises(NowcastDataError, match="No standardisation"):
            panel.standardize(stats)


@given(
    hnp.arrays(
        np.float64,
        st.tuples(st.integers(3, 30), st.integers(1, 4)),
        elements=st.floats(-1e3, 1e3, allow_nan=False, allow_subnormal=False),
    )
)
def test_standardize_roundtrip_property(values):
    idx = pd.period_range("2000-01", periods=values.shape[0], freq="M")
    frame = pd.DataFrame(values, index=idx, columns=[f"x{i}" for i in range(values.shape[1])])
    if (frame.std() < 1e-6).any():
        return
    mfd = MixedFrequencyData(frame, "M")
    z, stats = mfd.standardize()
    back = z.destandardize(stats)
    np.testing.assert_allclose(back.values, mfd.values, atol=1e-8 * max(1.0, np.abs(values).max()))


# --------------------------------------------------------------------------- coercion


class TestAsMixedFrequencyData:
    def test_passthrough(self, panel):
        assert as_mixed_frequency_data(panel) is panel

    def test_from_frame(self):
        frame = make_panel_frame()
        mfd = as_mixed_frequency_data(frame, {"gdp": "Q"}, release_delays={"gdp": 1})
        assert mfd.release_delays["gdp"] == 1

    def test_errors(self, panel):
        with pytest.raises(NowcastDataError, match="already"):
            as_mixed_frequency_data(panel, {"gdp": "Q"})
        with pytest.raises(NowcastDataError, match="already"):
            as_mixed_frequency_data(panel, release_delays={"gdp": 1})
        with pytest.raises(NowcastDataError, match="must be"):
            as_mixed_frequency_data([1, 2, 3])  # type: ignore[arg-type]


def test_with_metadata_new_block_is_appended(panel):
    new = panel.with_metadata("pmi", blocks=("global", "labour"))
    assert new.block_names == ["global", "real", "labour"]


def test_as_of_emptying_series_does_not_warn(panel):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        early = panel.as_of("2018-02-15")
    assert early.n_observations()["gdp"] == 0


class TestTransformApplied:
    def test_default_false_and_coercion(self):
        assert SeriesMetadata("a", "M").transform_applied is False
        assert SeriesMetadata("a", "M", transform_applied=np.bool_(True)).transform_applied
        assert SeriesMetadata("a", "M", transform_applied=1).transform_applied is True
        assert SeriesMetadata("a", "M", transform_applied=float("nan")).transform_applied is False

    def test_rejects_strings(self):
        with pytest.raises(ValueError, match="transform_applied"):
            SeriesMetadata("a", "M", transform_applied="yes")

    def test_round_trips_through_dict_and_panel(self):
        idx = pd.period_range("2020-01", periods=3, freq="M")
        meta = SeriesMetadata("a", "M", transform="diff", transform_applied=True)
        mfd = MixedFrequencyData(
            pd.DataFrame({"a": [1.0, 2.0, 3.0]}, index=idx), metadata={"a": meta}
        )
        assert mfd.metadata["a"].transform_applied is True
        assert mfd.metadata_frame().loc["a", "transform_applied"]
        assert SeriesMetadata(**meta.to_dict()).transform_applied is True
