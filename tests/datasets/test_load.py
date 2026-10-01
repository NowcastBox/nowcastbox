"""Tests of the dataset loaders (offline, against the shipped files)."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import nowcastbox.datasets as nbd
from nowcastbox.core.data import SeriesCategory
from nowcastbox.core.exceptions import NowcastDataError
from nowcastbox.core.frequency import is_period_end
from nowcastbox.datasets import Dataset, _io
from nowcastbox.preprocessing import apply_transforms
from nowcastbox.preprocessing.transforms import TRANSFORM_CODES, get_transform
from nowcastbox.vintages import ReleaseCalendar, VintageStore, pseudo_real_time

DATASET_LOADERS = [
    nbd.load_brazil_nowcast,
    nbd.load_us_fred_md,
    nbd.load_nyfed,
    nbd.load_us_grs_like,
    nbd.load_simulated_dfm,
]
CATEGORIES = {c.value for c in SeriesCategory}


@pytest.fixture(scope="module", params=DATASET_LOADERS, ids=lambda f: f.__name__)
def any_dataset(request: pytest.FixtureRequest) -> Dataset:
    return request.param()


# ---------------------------------------------------------------------- generic contract
class TestEveryDataset:
    def test_is_dataset_with_target(self, any_dataset: Dataset) -> None:
        assert isinstance(any_dataset, Dataset)
        assert any_dataset.target in any_dataset.columns
        assert any_dataset.title and any_dataset.description
        assert any_dataset.license and any_dataset.citation

    def test_legend_is_complete(self, any_dataset: Dataset) -> None:
        legend = any_dataset.legend
        assert list(legend.index) == any_dataset.columns
        assert set(legend["frequency"]) <= {"M", "Q"}
        assert set(legend["category"]) <= CATEGORIES
        assert legend["delay_days"].notna().all()
        assert (legend["delay_days"] >= 0).all()
        assert (legend["description"].str.len() > 0).all()
        assert all("global" in b.split(";") for b in legend["blocks"])

    def test_transforms_parse_and_legacy_codes_agree(self, any_dataset: Dataset) -> None:
        for spec, legacy in zip(
            any_dataset.legend["transform"], any_dataset.legend["legacy_code"], strict=True
        ):
            transform = get_transform(spec)
            if legacy != "":
                assert transform == get_transform(TRANSFORM_CODES[int(legacy)])

    def test_quarterly_values_sit_in_slots(self, any_dataset: Dataset) -> None:
        frame = any_dataset.to_frame()
        slots = is_period_end(frame.index, "Q")
        for col in any_dataset.data.quarterly_columns:
            assert frame.loc[~slots, col].isna().all()
            assert frame.loc[slots, col].notna().sum() > 8

    def test_transforms_apply_without_warnings(self, any_dataset: Dataset) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = apply_transforms(any_dataset.data)
        values = out.values
        assert not np.isinf(values).any()
        assert all(meta.transform_applied for meta in out.metadata.values())
        assert (out.n_observations() > 10).all()

    def test_metadata_has_no_legend_or_checksums(self, any_dataset: Dataset) -> None:
        meta = any_dataset.metadata
        assert "series" not in meta and "files" not in meta


# ---------------------------------------------------------------------- catalogue
def test_list_datasets() -> None:
    cat = nbd.list_datasets()
    assert set(cat.index) == set(_io.available_names())
    assert cat.loc["us_grs_like", "start"] == "1982-01"
    assert cat.loc["us_grs_like", "end"] == "2004-12"
    assert cat.loc["nyfed", "n_series"] == 29
    for loader in cat["loader"]:
        assert callable(getattr(nbd, loader))


def test_dataset_info_and_dispatch() -> None:
    info = nbd.dataset_info("brazil_nowcast")
    assert info["loader"] == "load_brazil_nowcast"
    assert len(info["files"]["data"]["sha256"]) == 64
    for name in nbd.list_datasets().index:
        assert nbd.load_dataset(name) is not None
    assert nbd.load_dataset("nyfed", model_only=False).n_series == 29
    with pytest.raises(ValueError, match="Unknown dataset"):
        nbd.load_dataset("nope")


def test_legend_matches_data_file_columns(isolated_store: Path) -> None:
    meta_path = isolated_store / "metadata" / "nyfed.yaml"
    text = meta_path.read_text(encoding="utf-8").replace("name: PAYEMS", "name: PAYEMS_X", 1)
    meta_path.write_text(text, encoding="utf-8")
    with pytest.raises(NowcastDataError, match="different series"):
        nbd.load_nyfed()


def test_dataset_without_legend(isolated_store: Path) -> None:
    path = isolated_store / "metadata" / "nyfed.yaml"
    meta = _io.read_metadata("nyfed")
    meta["series"] = []
    _io.write_metadata(meta, path)
    _io.clear_cache()
    with pytest.raises(NowcastDataError, match="no series legend"):
        nbd.load_nyfed()


def test_data_file_without_period_column(isolated_store: Path) -> None:
    path = _io.data_path("nyfed")
    table = pd.read_csv(path).rename(columns={"period": "date"})
    meta = _io.read_metadata("nyfed")
    meta["files"]["data"]["sha256"] = _io.write_csv_gz(table, path)
    _io.write_metadata(meta, isolated_store / "metadata" / "nyfed.yaml")
    _io.clear_cache()
    with pytest.raises(NowcastDataError, match="period"):
        nbd.load_nyfed()


def test_legend_without_optional_columns(isolated_store: Path) -> None:
    meta = _io.read_metadata("nyfed")
    for row in meta["series"]:
        row.pop("units")
    _io.write_metadata(meta, isolated_store / "metadata" / "nyfed.yaml")
    _io.clear_cache()
    ds = nbd.load_nyfed()
    assert set(ds.legend["units"]) == {""}


# ---------------------------------------------------------------------- Brazil
class TestBrazilNowcast:
    def test_content(self) -> None:
        ds = nbd.load_brazil_nowcast()
        assert 60 <= ds.n_series <= 120
        assert str(ds.data.start) == "2003-01"
        assert ds.data.end >= pd.Period("2026-06", "M")
        assert ds.target == "pib" and ds.transform["pib"] == "qoq"
        assert sorted(ds.block_names) == ["financial", "global", "nominal", "real", "soft"]
        assert set(ds.categories) == {"hard", "soft", "financial"}
        sources = {s.split(" ")[0] for s in ds.legend["source"]}
        assert sources == {"BCB/SGS", "IBGE/SIDRA", "IPEADATA", "BCB/Focus"}
        for name in ("ibc_br", "pim_geral", "pmc_varejo", "pms_volume", "ipca", "selic"):
            assert name in ds.columns

    def test_target_matches_published_index(self) -> None:
        gdp = nbd.load_brazil_nowcast().data.to_native("pib", dropna=True)
        assert gdp.index.freqstr.startswith("Q")
        growth = 100 * gdp.pct_change().dropna()
        assert growth.loc["2020Q2"] < -8  # pandemic contraction
        assert growth.loc["2020Q3"] > 6
        assert abs(growth.loc["2010Q1":"2019Q4"].mean()) < 1.5

    def test_licensing_is_recorded(self) -> None:
        info = nbd.dataset_info("brazil_nowcast")
        excluded = " ".join(e["series"] for e in info["excluded_for_license"])
        for third_party in ("FGV", "CNI", "ANFAVEA"):
            assert third_party in excluded
        assert "ODbL" in info["license"]
        assert isinstance(info["dropped"], list)

    def test_selection_and_sample(self) -> None:
        ds = nbd.load_brazil_nowcast(columns=["pib", "ibc_br"], start="2010-01", end="2019Q4")
        assert ds.columns == ["pib", "ibc_br"]
        assert (str(ds.data.start), str(ds.data.end)) == ("2010-01", "2019-12")

    def test_unknown_column(self) -> None:
        with pytest.raises(NowcastDataError, match="Unknown series"):
            nbd.load_brazil_nowcast(columns=["nope"])


class TestBrazilCalendar:
    def test_calendar(self) -> None:
        cal = nbd.load_brazil_calendar()
        assert isinstance(cal, ReleaseCalendar)
        panel = nbd.load_brazil_nowcast()
        assert set(cal.series) == set(panel.columns)
        assert cal.release_date("pib", "2024Q1") == pd.Timestamp("2024-06-04")
        assert cal.release_date("fbcf", "2021Q2") == pd.Timestamp("2021-09-01")
        # quarters before 2010 fall back on the 62-day delay
        assert cal.release_date("pib", "2005Q1") == pd.Timestamp("2005-03-31") + pd.Timedelta(
            days=62
        )
        assert int(cal.delays["ipca"]) == 10

    def test_frame(self) -> None:
        table = nbd.load_brazil_calendar(as_frame=True)
        assert table.index.name == "series"
        assert table.loc["pib", "explicit_dates"]
        assert not table.loc["ipca", "explicit_dates"]
        assert table["delay_days"].dtype == "Int64"

    def test_pseudo_real_time_uses_explicit_gdp_dates(self) -> None:
        ds = nbd.load_brazil_nowcast(columns=["pib", "ibc_br"])
        cal = nbd.load_brazil_calendar()
        before = pseudo_real_time(ds.data, calendar=cal, vintage="2026-08-31")
        after = pseudo_real_time(ds.data, calendar=cal, vintage="2026-09-01")
        assert str(before.last_observed()["pib"]) == "2026-03"
        assert str(after.last_observed()["pib"]) == "2026-06"

    def test_gdp_releases(self) -> None:
        rel = nbd.load_brazil_gdp_releases()
        assert list(rel.columns[:3]) == ["reference_quarter", "release_date", "date_source"]
        assert rel["reference_quarter"].is_monotonic_increasing
        lags = (
            rel["release_date"]
            - pd.PeriodIndex(rel["reference_quarter"]).to_timestamp(how="end").normalize()
        ).dt.days
        assert lags.between(50, 80).all()
        assert rel["release_date"].is_monotonic_increasing


class TestBrazilVintages:
    def test_store(self) -> None:
        store = nbd.load_brazil_vintages()
        assert isinstance(store, VintageStore)
        assert "pib" in store and "fbcf" in store
        assert store.first_vintage == pd.Timestamp("2010-06-04")
        assert len(store.vintage_dates("pib")) >= 60

    def test_real_time_values(self) -> None:
        store = nbd.load_brazil_vintages(series="pib")
        first = store.nth_release(0)["pib"]
        latest = store.nth_release(-1)["pib"]
        assert float(first.loc["2020-06"]) == pytest.approx(150.3)
        assert float(latest.loc["2020-06"]) != float(first.loc["2020-06"])
        known = store.as_of("2020-08-31")["pib"].dropna()
        assert str(known.index[-1]) == "2020-03"

    def test_latest_vintage_agrees_with_panel(self) -> None:
        store = nbd.load_brazil_vintages(series=["pib"])
        latest = store.as_of(pd.Timestamp("2100-01-01"))["pib"].dropna()
        panel = nbd.load_brazil_nowcast().data.to_native("pib", dropna=True)
        recent = panel.loc["2016Q1":].to_numpy()
        stored = latest.loc["2016-03":].to_numpy()
        assert len(stored) == len(recent)
        np.testing.assert_allclose(stored, recent, atol=0.051)  # PDF has 1 decimal

    def test_frame_and_errors(self) -> None:
        table = nbd.load_brazil_vintages(series="pib", as_frame=True)
        assert set(table["series"]) == {"pib"}
        assert pd.api.types.is_datetime64_any_dtype(table["vintage_date"])
        with pytest.raises(NowcastDataError, match="Unknown series"):
            nbd.load_brazil_vintages(series=["pib", "nope"])


# ---------------------------------------------------------------------- United States
class TestFredMd:
    def test_content(self) -> None:
        ds = nbd.load_us_fred_md()
        assert 100 <= ds.n_series <= 130
        assert str(ds.data.start) == "1959-01"
        assert ds.data.quarterly_columns == ["GDPC1"]
        mapping = {1: "level", 2: "diff", 3: "diff|diff", 4: "log", 5: "dlog"}
        mapping |= {6: "log|diff|diff", 7: "pct_change|diff"}
        for name, row in ds.legend.iterrows():
            assert row["transform"] == mapping[int(row["fred_md_tcode"])], name

    def test_restricted_series_are_not_shipped(self) -> None:
        cols = set(nbd.load_us_fred_md().columns)
        for name in ("S&P 500", "AAA", "BAA", "UMCSENTx", "VIXCLSx"):
            assert name not in cols

    def test_tcode6_is_second_difference_of_logs(self) -> None:
        ds = nbd.load_us_fred_md(start="2000-01", end="2005-12")
        col = next(c for c in ds.columns if ds.legend.loc[c, "fred_md_tcode"] == 6)
        out = apply_transforms(ds.data.select([col])).data[col]
        x = np.log(ds.to_frame()[col])
        np.testing.assert_allclose(out.dropna(), x.diff().diff().dropna(), rtol=1e-10)

    def test_without_gdp(self) -> None:
        ds = nbd.load_us_fred_md(include_gdp=False)
        assert "GDPC1" not in ds.columns and ds.target is None


class TestNyFed:
    def test_model_series(self) -> None:
        ds = nbd.load_nyfed()
        assert ds.n_series == 25
        assert (ds.legend["nyfed_model"] == 1).all()
        assert ds.block_names == ["global", "labor", "real", "soft"]
        assert set(ds.data.quarterly_columns) == {"GDPC1", "ULCNFB"}
        assert ds.blocks["global"].all()
        assert set(ds.legend.loc[ds.blocks["soft"] == 1, "category"]) == {"soft"}
        assert (str(ds.data.start), str(ds.data.end)) == ("1985-01", "2017-01")

    def test_all_series(self) -> None:
        ds = nbd.load_nyfed(model_only=False)
        assert ds.n_series == 29
        assert set(ds.legend["nyfed_transformation"]) == {"lin", "chg", "pch", "pca"}

    def test_pch_is_percent_change(self) -> None:
        ds = nbd.load_nyfed().select(["INDPRO"])
        out = apply_transforms(ds.data).data["INDPRO"]
        x = ds.to_frame()["INDPRO"]
        expected = 100 * (x / x.shift(1) - 1)
        np.testing.assert_allclose(out.dropna(), expected.dropna(), rtol=1e-10)

    def test_pca_approximates_annualised_rate(self) -> None:
        ds = nbd.load_nyfed().select(["GDPC1"])
        out = ds.data.with_data(apply_transforms(ds.data).data).to_native("GDPC1", dropna=True)
        level = ds.data.to_native("GDPC1", dropna=True)
        exact = 100 * ((level / level.shift(1)) ** 4 - 1)
        diff = (out - exact).dropna()
        # log approximation: |error| grows with the rate (0.3 pp at 8 % annualised growth)
        assert (diff.abs() <= 0.05 * exact.loc[diff.index].abs() + 0.05).all()


class TestGrsLike:
    def test_default_sample(self) -> None:
        ds = nbd.load_us_grs_like()
        assert (str(ds.data.start), str(ds.data.end)) == ("1982-01", "2004-12")
        assert "NOT the original panel" in ds.notes
        assert ds.metadata["default_sample"] == {"start": "1982-01", "end": "2004-12"}

    def test_custom_sample(self) -> None:
        full = nbd.load_us_grs_like(start=None, end=None)
        assert str(full.data.start) == "1959-01"
        assert full.n_series == nbd.load_us_fred_md().n_series
        part = nbd.load_us_grs_like(start="1990-01", end=None)
        assert str(part.data.start) == "1990-01"


# ---------------------------------------------------------------------- simulated
class TestSimulated:
    def test_deterministic(self) -> None:
        a = nbd.load_simulated_dfm()
        b = nbd.load_simulated_dfm()
        pd.testing.assert_frame_equal(a.to_frame(), b.to_frame())
        assert a.data.shape == (240, 21)
        c = nbd.load_simulated_dfm(random_state=1)
        assert not a.to_frame().equals(c.to_frame())

    def test_ragged_edge_matches_delays(self) -> None:
        ds = nbd.load_simulated_dfm()
        end = ds.data.end.to_timestamp(how="end").normalize()
        vintage = ds.data.as_of(end + pd.Timedelta(days=5))
        pd.testing.assert_frame_equal(vintage.data.isna(), ds.to_frame().isna())
        lags = ds.metadata["true_params"]["publication_lags"]
        last = ds.data.last_observed()
        for name in ds.columns[:-1]:
            assert last[name] == ds.data.end - int(lags[name])
        assert last["gdp"] == ds.data.end - 3

    def test_without_ragged_edge(self) -> None:
        ds = nbd.load_simulated_dfm(ragged_edge=False, n_monthly=4, n_periods=60)
        assert ds.data.last_observed().eq(ds.data.end).all()
        assert ds.metadata["simulation"]["ragged_edge"] is False

    def test_target_is_mariano_murasawa_aggregate(self) -> None:
        ds = nbd.load_simulated_dfm(ragged_edge=False)
        latent = ds.metadata["true_params"]["gdp_monthly"].to_numpy()
        gdp = ds.to_frame()["gdp"].to_numpy()
        w = np.array([1, 2, 3, 2, 1]) / 3
        for t in np.flatnonzero(~np.isnan(gdp)):
            assert gdp[t] == pytest.approx(w @ latent[t - np.arange(5)])

    def test_estimated_factor_tracks_truth(self) -> None:
        ds = nbd.load_simulated_dfm(n_factors=1, n_monthly=30)
        x = ds.to_frame().drop(columns="gdp").dropna()
        x = (x - x.mean()) / x.std()
        _, _, vt = np.linalg.svd(x.to_numpy(), full_matrices=False)
        pc = x.to_numpy() @ vt[0]
        truth = ds.metadata["true_params"]["factors"].loc[x.index, "f1"].to_numpy()
        assert abs(np.corrcoef(pc, truth)[0, 1]) > 0.9
