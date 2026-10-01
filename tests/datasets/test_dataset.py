"""Tests of the :class:`nowcastbox.datasets.Dataset` container."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nowcastbox.core.data import MixedFrequencyData, SeriesCategory
from nowcastbox.core.exceptions import NowcastDataError
from nowcastbox.core.frequency import Frequency
from nowcastbox.datasets import LEGEND_COLUMNS, Dataset
from nowcastbox.datasets.dataset import split_blocks
from nowcastbox.vintages import ReleaseCalendar


def build(small_panel: tuple[pd.DataFrame, pd.DataFrame], **kwargs: object) -> Dataset:
    data, legend = small_panel
    defaults: dict[str, object] = {
        "title": "Small",
        "description": "A small panel",
        "target": "gdp",
        "metadata": {"license": "MIT", "citation": "Me", "url": "u", "notes": "n"},
    }
    defaults.update(kwargs)
    return Dataset("small", data, legend, **defaults)  # type: ignore[arg-type]


class TestSplitBlocks:
    def test_string(self) -> None:
        assert split_blocks(" global ;real;; ") == ("global", "real")

    def test_sequence(self) -> None:
        assert split_blocks(["global", "soft"]) == ("global", "soft")

    @pytest.mark.parametrize("value", [None, float("nan"), ""])
    def test_missing(self, value: object) -> None:
        assert split_blocks(value) == ()

    def test_invalid(self) -> None:
        with pytest.raises(ValueError, match="blocks"):
            split_blocks(3)


class TestConstruction:
    def test_basic_properties(self, small_panel: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ds = build(small_panel)
        assert ds.name == "small"
        assert ds.title == "Small"
        assert ds.description == "A small panel"
        assert ds.target == "gdp"
        assert ds.columns == ["a", "b", "gdp"]
        assert ds.n_series == 3
        assert (ds.license, ds.citation, ds.url, ds.notes) == ("MIT", "Me", "u", "n")
        assert isinstance(ds.data, MixedFrequencyData)
        assert ds.data.quarterly_columns == ["gdp"]
        assert list(ds.legend.columns[: len(LEGEND_COLUMNS)]) == list(LEGEND_COLUMNS)

    def test_metadata_propagates_to_panel(
        self, small_panel: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        meta = build(small_panel).data.metadata
        assert meta["gdp"].frequency is Frequency.QUARTERLY
        assert meta["a"].transform == "dlog"
        assert meta["a"].release_delay == 30
        assert meta["b"].blocks == ("global", "nominal")
        assert meta["b"].category is SeriesCategory.FINANCIAL
        assert meta["a"].units == "index"
        assert meta["a"].description == "series a"
        assert meta["a"].transform_applied is False

    def test_legend_views(self, small_panel: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ds = build(small_panel)
        assert ds.frequency.to_dict() == {"a": "M", "b": "M", "gdp": "Q"}
        assert ds.transform.tolist() == ["dlog"] * 3
        assert ds.delay.dtype == "Int64"
        assert ds.categories.tolist() == ["hard", "financial", "hard"]
        assert ds.block_names == ["global", "real", "nominal"]
        blocks = ds.blocks
        assert blocks.loc["b"].tolist() == [1, 0, 1]
        assert blocks.dtypes.unique().tolist() == [np.dtype(int)]

    def test_views_are_copies(self, small_panel: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ds = build(small_panel)
        legend = ds.legend
        legend.loc["a", "description"] = "changed"
        freq = ds.frequency
        freq.loc["a"] = "Q"
        meta = ds.metadata
        meta["license"] = "GPL"
        frame = ds.to_frame()
        frame.iloc[0, 0] = -999.0
        assert ds.legend.loc["a", "description"] == "series a"
        assert ds.frequency["a"] == "M"
        assert ds.license == "MIT"
        assert ds.to_frame().iloc[0, 0] != -999.0

    def test_input_frames_are_not_aliased(
        self, small_panel: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        data, legend = small_panel
        ds = Dataset("x", data, legend)
        data.iloc[0, 0] = -1.0
        legend.loc[0, "units"] = "zzz"
        assert ds.to_frame().iloc[0, 0] != -1.0
        assert ds.legend.loc["a", "units"] == "index"

    def test_optional_fields(self, small_panel: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        data, legend = small_panel
        legend = legend.copy()
        legend["delay_days"] = [None, 5, None]
        legend["category"] = [None, "soft", ""]
        legend["transform"] = [None, "diff", np.nan]
        legend["blocks"] = [None, "", "global"]
        ds = Dataset("x", data, legend)
        assert ds.target is None and ds.license == ""
        meta = ds.data.metadata
        assert meta["a"].release_delay is None
        assert meta["a"].category is None and meta["gdp"].category is None
        assert meta["a"].transform is None and meta["gdp"].transform is None
        assert ds.block_names == ["global"]
        assert ds.calendar().series == ["b"]

    def test_extra_legend_columns_are_kept(
        self, small_panel: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        data, legend = small_panel
        legend = legend.assign(extra=[1, 2, 3])
        ds = Dataset("x", data, legend)
        assert ds.legend.columns[-1] == "extra"
        assert ds.legend["extra"].tolist() == [1, 2, 3]

    def test_legend_order_follows_data(
        self, small_panel: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        data, legend = small_panel
        ds = Dataset("x", data, legend.iloc[::-1])
        assert list(ds.legend.index) == ["a", "b", "gdp"]


class TestValidation:
    def test_data_must_be_a_frame(self, small_panel: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        with pytest.raises(NowcastDataError, match="DataFrame"):
            Dataset("x", small_panel[0].to_numpy(), small_panel[1])  # type: ignore[arg-type]

    def test_missing_legend_columns(self, small_panel: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        data, legend = small_panel
        with pytest.raises(NowcastDataError, match="missing the columns"):
            Dataset("x", data, legend.drop(columns=["units", "blocks"]))

    def test_duplicated_names(self, small_panel: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        data, legend = small_panel
        legend = legend.copy()
        legend.loc[1, "name"] = "a"
        with pytest.raises(NowcastDataError, match="Duplicated"):
            Dataset("x", data, legend)

    def test_legend_data_mismatch(self, small_panel: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        data, legend = small_panel
        with pytest.raises(NowcastDataError, match="disagree"):
            Dataset("x", data.drop(columns="b"), legend)
        with pytest.raises(NowcastDataError, match="disagree"):
            Dataset("x", data, legend.iloc[:2])

    def test_unknown_target(self, small_panel: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        with pytest.raises(NowcastDataError, match="Target"):
            build(small_panel, target="nope")

    def test_non_numeric_delay(self, small_panel: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        data, legend = small_panel
        legend = legend.copy()
        legend["delay_days"] = ["soon", 1, 2]
        with pytest.raises(ValueError):
            Dataset("x", data, legend)

    def test_slot_violation_is_reported(
        self, small_panel: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        data, legend = small_panel
        data = data.copy()
        data.iloc[0, 2] = 1.0  # quarterly value in January
        with pytest.raises(NowcastDataError):
            Dataset("x", data, legend)


class TestDerived:
    def test_calendar(self, small_panel: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        cal = build(small_panel).calendar()
        assert isinstance(cal, ReleaseCalendar)
        assert cal.release_date("gdp", "2020Q1") == pd.Timestamp("2020-04-30")

    def test_select(self, small_panel: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ds = build(small_panel)
        sub = ds.select(["gdp", "a"])
        assert sub.columns == ["gdp", "a"]
        assert sub.target == "gdp"
        assert sub.license == "MIT"
        assert ds.select(["a"]).target is None
        assert ds.select(["b"]).block_names == ["global", "nominal"]

    @pytest.mark.parametrize(
        ("cols", "match"), [([], "at least one"), (["zz"], "Unknown"), (["a", "a"], "Dupl")]
    )
    def test_select_errors(
        self, small_panel: tuple[pd.DataFrame, pd.DataFrame], cols: list[str], match: str
    ) -> None:
        with pytest.raises(NowcastDataError, match=match):
            build(small_panel).select(cols)

    def test_truncate(self, small_panel: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ds = build(small_panel).truncate("2020Q2", "2020-09")
        assert str(ds.data.start) == "2020-04" and str(ds.data.end) == "2020-09"
        assert ds.to_frame()["gdp"].dropna().tolist() == [101.0, 102.5]

    def test_summary_and_repr(self, small_panel: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ds = build(small_panel, metadata={"license": "MIT", "sources": ["here", "there"]})
        text = ds.summary()
        assert "Dataset 'small': Small" in text
        assert "M: 2, Q: 1" in text
        assert "Sources       : here; there" in text
        assert "Blocks        : global, real, nominal" in text
        assert repr(ds).startswith("Dataset(name='small', n_series=3")

    def test_summary_without_optional_fields(
        self, small_panel: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        data, legend = small_panel
        legend = legend.assign(blocks="", category="")
        text = Dataset("x", data, legend).summary()
        assert "Blocks        : none" in text
        assert "Target        : none" in text
        assert "unset: 3" in text
        assert "License       : n/a" in text
        assert "Sources" not in text
