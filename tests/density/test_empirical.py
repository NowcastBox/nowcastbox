"""Tests of the empirical error bands (Reifschneider-Tulip / ECB style)."""

from __future__ import annotations

import doctest
import math
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import scipy.stats

import nowcastbox.density.empirical as empirical_module
from nowcastbox.core.data import MixedFrequencyData
from nowcastbox.core.exceptions import DataQualityWarning, NowcastDataError
from nowcastbox.core.results import NowcastResults, build_nowcast_frame
from nowcastbox.density import (
    EMPIRICAL_LEVELS,
    EMPIRICAL_METHODS,
    MAE_TO_SIGMA,
    EmpiricalGaussianDistribution,
    EmpiricalQuantileDistribution,
    NowcastDistribution,
    available_errors,
    backtest_empirical_bands,
    empirical_bands,
    empirical_error_scales,
)
from nowcastbox.evaluation.backtest import BacktestResults
from nowcastbox.evaluation.scoring import crps_sample

SIGMAS = {0: 0.5, 1: 1.0, 2: 1.5}


# ---------------------------------------------------------------------- helpers
def make_table(
    start: str = "1990-01",
    end: str = "2019-12",
    *,
    seed: int = 0,
    release_days: float | None = 100.0,
    sigmas: dict[int, float] | None = None,
) -> pd.DataFrame:
    """Monthly vintages (15th), nowcast of the current quarter, Gaussian errors."""
    sig = SIGMAS if sigmas is None else sigmas
    rng = np.random.default_rng(seed)
    months = pd.period_range(start, end, freq="M")
    rows = []
    for m in months:
        vintage = m.start_time + pd.Timedelta(days=14)
        period = m.asfreq("Q")
        h = int(period.asfreq("M", how="E").ordinal - m.ordinal)
        forecast = float(rng.normal())
        error = float(rng.normal(scale=sig[h]))
        days = (
            (period.end_time.normalize() - vintage).days + release_days if release_days else np.nan
        )
        rows.append(
            {
                "vintage": vintage,
                "target_period": period,
                "offset": 0,
                "kind": "nowcast",
                "months_to_end": h,
                "days_to_end": int((period.end_time.normalize() - vintage).days),
                "days_to_release": days,
                "model": "DFM",
                "forecast": forecast,
                "actual": forecast + error,
                "error": error,
            }
        )
    frame = pd.DataFrame(rows)
    frame["target_period"] = pd.PeriodIndex(frame["target_period"], freq="Q")
    return frame


def make_results(periods=("2020Q1", "2020Q2"), values=(0.4, 0.6), *, info=None, data=None):
    idx = pd.PeriodIndex(list(periods), freq="Q")
    nowcast = build_nowcast_frame(
        pd.Series(np.nan, index=idx), pd.Series(list(values), index=idx, dtype=float)
    )
    return NowcastResults(target="gdp", nowcast=nowcast, info=info or {}, data=data)


@pytest.fixture(scope="module")
def table() -> pd.DataFrame:
    return make_table()


@pytest.fixture(scope="module")
def backtest(table: pd.DataFrame) -> BacktestResults:
    other = table.assign(model="AR", forecast=0.0, error=table["actual"])
    frame = pd.concat([table, other], ignore_index=True)
    return BacktestResults(frame, ["DFM", "AR"], "gdp", reference="AR")


VINTAGE = pd.Timestamp("2020-01-20")  # 2020Q1 at horizon 2, 2020Q2 at horizon 5


# ---------------------------------------------------------------------- doctests
def test_docstring_examples() -> None:
    flags = doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = doctest.testmod(empirical_module, optionflags=flags, verbose=False)
    plt.close("all")
    assert result.failed == 0
    assert result.attempted > 20


def test_constants() -> None:
    assert pytest.approx(math.sqrt(math.pi / 2)) == MAE_TO_SIGMA
    assert EMPIRICAL_LEVELS == (0.575, 0.68, 0.9)
    assert EMPIRICAL_METHODS == ("mae", "rmse", "quantile")
    # +-1 MAE covers ~57.5 % under normality (the ECB convention).
    assert 2 * scipy.stats.norm.cdf(1 / MAE_TO_SIGMA) - 1 == pytest.approx(0.575, abs=5e-4)


# ---------------------------------------------------------------------- available errors
class TestAvailableErrors:
    def test_strictly_before_vintage_and_released(self, table: pd.DataFrame) -> None:
        out = available_errors(table, VINTAGE)
        assert (out["vintage"] < VINTAGE).all()
        release = out["vintage"] + pd.to_timedelta(out["days_to_release"], unit="D")
        assert (release <= VINTAGE).all()
        assert (out["vintage"] >= VINTAGE - pd.DateOffset(years=10)).all()
        # Under "vintage" availability, unreleased outcomes are also used.
        loose = available_errors(table, VINTAGE, availability="vintage")
        assert len(loose) > len(out)
        assert (loose["vintage"] < VINTAGE).all()

    def test_vintage_itself_excluded(self, table: pd.DataFrame) -> None:
        date = table["vintage"].iloc[200]
        out = available_errors(table, date, availability="vintage", window=None)
        assert date not in set(out["vintage"])
        assert len(out) == 200

    def test_release_fallback_without_release_dates(self) -> None:
        table = make_table(release_days=None)
        out = available_errors(table, VINTAGE, window=None)
        ends = [p.end_time.normalize() for p in out["target_period"]]
        assert all(e < VINTAGE for e in ends)
        no_col = table.drop(columns="days_to_release")
        pd.testing.assert_frame_equal(
            available_errors(no_col, VINTAGE, window=None), out.drop(columns="days_to_release")
        )

    @pytest.mark.parametrize(
        ("window", "start"),
        [
            ("36M", pd.DateOffset(months=36)),
            ("520W", pd.DateOffset(weeks=520)),
            ("3650D", pd.DateOffset(days=3650)),
            ("2y", pd.DateOffset(years=2)),
            (pd.DateOffset(years=3), pd.DateOffset(years=3)),
            (pd.Timedelta(days=400), pd.Timedelta(days=400)),
        ],
    )
    def test_windows(self, table: pd.DataFrame, window, start) -> None:
        out = available_errors(table, VINTAGE, window=window)
        assert (out["vintage"] >= VINTAGE - start).all()
        assert out["vintage"].min() < VINTAGE - start + pd.Timedelta(days=40)

    @pytest.mark.parametrize("window", [None, "expanding", "EXPANDING"])
    def test_expanding(self, table: pd.DataFrame, window) -> None:
        out = available_errors(table, VINTAGE, window=window)
        assert out["vintage"].min() == table["vintage"].min()

    @pytest.mark.parametrize("window", ["10X", "0Y", "Y", 5])
    def test_bad_window(self, table: pd.DataFrame, window) -> None:
        with pytest.raises(ValueError, match="window"):
            available_errors(table, VINTAGE, window=window)

    def test_backtest_object_and_model(self, backtest: BacktestResults) -> None:
        main = available_errors(backtest, VINTAGE)
        assert set(main["model"]) == {"DFM"}
        ar = available_errors(backtest, VINTAGE, model="AR")
        assert set(ar["model"]) == {"AR"}
        with pytest.raises(ValueError, match="not in the backtest"):
            available_errors(backtest, VINTAGE, model="XX")

    def test_frame_with_model_column_defaults_to_first(self, backtest: BacktestResults) -> None:
        out = available_errors(backtest.to_frame(), VINTAGE)
        assert set(out["model"]) == {"DFM"}

    def test_bad_inputs(self, table: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="lacks the columns"):
            available_errors(table.drop(columns="error"), VINTAGE)
        with pytest.raises(ValueError, match="availability"):
            available_errors(table, VINTAGE, availability="now")  # type: ignore[arg-type]


# ---------------------------------------------------------------------- error scales
class TestErrorScales:
    def test_statistics(self, table: pd.DataFrame) -> None:
        out = empirical_error_scales(table, VINTAGE, window=None)
        assert out.index.tolist() == [0, 1, 2]
        assert out.index.name == "months_to_end"
        sub = available_errors(table, VINTAGE, window=None)
        e = sub.loc[sub["months_to_end"] == 1, "error"].to_numpy()
        row = out.loc[1]
        assert row["n"] == e.size
        assert row["mae"] == pytest.approx(np.abs(e).mean())
        assert row["rmse"] == pytest.approx(np.sqrt((e**2).mean()))
        assert row["bias"] == pytest.approx(e.mean())
        assert row["sigma_mae"] == pytest.approx(np.abs(e).mean() * math.sqrt(math.pi / 2))
        assert out["n"].dtype.kind == "i"

    def test_mae_scale_recovers_sigma(self) -> None:
        big = make_table("1700-01", "2019-12", seed=3)
        out = empirical_error_scales(big, VINTAGE, window=None)
        for h, sigma in SIGMAS.items():
            assert out.loc[h, "sigma_mae"] == pytest.approx(sigma, rel=0.06)
            assert out.loc[h, "rmse"] == pytest.approx(sigma, rel=0.06)

    def test_insufficient_errors_nan(self, table: pd.DataFrame) -> None:
        out = empirical_error_scales(table, "1990-04-01", min_errors=2)
        assert out["n"].tolist() == [0, 1, 0] or out["mae"].isna().all()
        assert out[["bias", "mae", "rmse", "sigma_mae"]].isna().all().all()

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"min_errors": 0}, "min_errors"),
            ({"min_errors": True}, "min_errors"),
            ({"min_errors": 2.5}, "min_errors"),
            ({"outliers": "trim"}, "outliers"),
            ({"outlier_threshold": 0.0}, "outlier_threshold"),
            ({"availability": "x"}, "availability"),
        ],
    )
    def test_bad_options(self, table: pd.DataFrame, kwargs, match) -> None:
        with pytest.raises(ValueError, match=match):
            empirical_error_scales(table, VINTAGE, **kwargs)


# ---------------------------------------------------------------------- outliers
class TestOutliers:
    def test_winsorize_and_exclude(self, table: pd.DataFrame) -> None:
        dirty = table.copy()
        old = dirty["vintage"] < VINTAGE - pd.DateOffset(years=1)
        pos = dirty.index[old & (dirty["months_to_end"] == 2)][-5:]
        dirty.loc[pos, "error"] = 50.0
        raw = empirical_error_scales(dirty, VINTAGE).loc[2]
        win = empirical_error_scales(dirty, VINTAGE, outliers="winsorize").loc[2]
        exc = empirical_error_scales(dirty, VINTAGE, outliers="exclude").loc[2]
        clean = empirical_error_scales(table, VINTAGE).loc[2]
        assert raw["mae"] > 2 * clean["mae"]
        assert win["n"] == raw["n"]
        assert exc["n"] == raw["n"] - 5
        assert win["mae"] < 1.5 * clean["mae"]
        assert exc["mae"] < 1.5 * clean["mae"]

    def test_adjustment_helpers(self) -> None:
        adjust = empirical_module._adjust_outliers
        e = np.array([0.0, 0.0, 0.0, 0.0, 5.0])  # MAD = 0 -> unchanged
        assert adjust(e, "exclude", 3.0).tolist() == e.tolist()
        two = np.array([0.0, 100.0])  # too few errors -> unchanged
        assert adjust(two, "winsorize", 3.0).tolist() == two.tolist()
        x = np.array([-1.0, -0.5, 0.0, 0.5, 1.0, 30.0])
        assert adjust(x, None, 3.0).tolist() == x.tolist()
        assert adjust(x, "winsorize", 3.0).max() < 30.0
        assert 30.0 not in adjust(x, "exclude", 3.0)


# ---------------------------------------------------------------------- bands for results
class TestEmpiricalBands:
    def test_mae_gaussian_bands(self, table: pd.DataFrame) -> None:
        res = make_results()
        with pytest.warns(DataQualityWarning, match="2020Q2"):
            dist = empirical_bands(res, table, vintage=VINTAGE)
        assert isinstance(dist, NowcastDistribution)
        assert dist.is_gaussian
        assert dist.index.astype(str).tolist() == ["2020Q1"]
        scales = empirical_error_scales(table, VINTAGE)
        assert dist.scales[0, 0] == pytest.approx(scales.loc[2, "sigma_mae"])
        assert dist.point.tolist() == [0.4]
        # The 57.5 % band is (almost exactly) point +- MAE.
        band = dist.interval(0.575)
        assert band["upper"].iloc[0] - 0.4 == pytest.approx(scales.loc[2, "mae"], rel=1e-3)
        info = dist.info
        assert info["method"] == "mae"
        assert info["horizons"] == {"2020Q1": 2}
        assert info["n_errors"] == {"2020Q1": int(scales.loc[2, "n"])}
        assert info["skipped"] == {"2020Q2": 5}
        assert info["levels"] == EMPIRICAL_LEVELS
        assert info["vintage"] == VINTAGE
        assert info["model"] == "DFM"
        no_model = empirical_bands(
            res, table.drop(columns="model"), vintage=VINTAGE, periods="2020Q1"
        )
        assert no_model.info["model"] is None

    def test_rmse(self, backtest: BacktestResults) -> None:
        res = make_results(["2020Q1"], [0.0])
        dist = empirical_bands(res, backtest, vintage=VINTAGE, method="rmse", levels=0.9)
        scales = empirical_error_scales(backtest, VINTAGE)
        assert dist.scales[0, 0] == pytest.approx(scales.loc[2, "rmse"])
        assert dist.info["model"] == "DFM"
        assert dist.info["levels"] == (0.9,)

    def test_quantile(self, table: pd.DataFrame) -> None:
        res = make_results(["2020Q1"], [1.0])
        dist = empirical_bands(res, table, vintage=VINTAGE, method="quantile")
        assert isinstance(dist, EmpiricalQuantileDistribution)
        sub = available_errors(table, VINTAGE)
        e = sub.loc[sub["months_to_end"] == 2, "error"].to_numpy()
        expected = 1.0 + np.quantile(e, [0.05, 0.95])
        assert dist.interval(0.9).to_numpy()[0] == pytest.approx(expected)
        assert dist.levels == EMPIRICAL_LEVELS
        assert dist.target == "gdp"

    def test_no_look_ahead(self, table: pd.DataFrame) -> None:
        res = make_results(["2020Q1"], [0.0])
        base = empirical_bands(res, table, vintage=VINTAGE)
        future = table.copy()
        # Errors of forecasts at or after the vintage, and errors whose outcome is
        # released after the vintage, are unknown at the vintage: changing them must
        # not move the bands.
        release = future["vintage"] + pd.to_timedelta(future["days_to_release"], unit="D")
        unknown = (future["vintage"] >= VINTAGE) | (release > VINTAGE)
        assert unknown.any() and (future["vintage"] < VINTAGE)[unknown].any()
        future.loc[unknown, "error"] = 1e3
        moved = empirical_bands(res, future, vintage=VINTAGE)
        assert moved.scales[0, 0] == base.scales[0, 0]
        # ... while with the loose "vintage" rule the unreleased ones do matter.
        loose = empirical_bands(res, future, vintage=VINTAGE, availability="vintage")
        assert loose.scales[0, 0] > 10 * base.scales[0, 0]

    def test_rolling_window(self, table: pd.DataFrame) -> None:
        res = make_results(["2020Q1"], [0.0])
        base = empirical_bands(res, table, vintage=VINTAGE)
        old = table.copy()
        old.loc[old["vintage"] < VINTAGE - pd.DateOffset(years=10), "error"] = 1e3
        assert empirical_bands(res, old, vintage=VINTAGE).scales[0, 0] == base.scales[0, 0]
        assert empirical_bands(res, old, vintage=VINTAGE, window=None).scales[0, 0] > 10

    def test_insufficient_everywhere(self, table: pd.DataFrame) -> None:
        res = make_results()
        with (
            pytest.warns(DataQualityWarning),
            pytest.raises(NowcastDataError, match="past errors"),
        ):
            empirical_bands(res, table, vintage="1990-03-01")

    def test_zero_scale_skipped(self, table: pd.DataFrame) -> None:
        zero = table.assign(error=0.0)
        res = make_results(["2020Q1"], [0.0])
        with pytest.warns(DataQualityWarning), pytest.raises(NowcastDataError):
            empirical_bands(res, zero, vintage=VINTAGE)
        dist = empirical_bands(res, zero, vintage=VINTAGE, method="quantile")
        assert dist.interval(0.9).to_numpy().tolist() == [[0.0, 0.0]]

    def test_periods(self, table: pd.DataFrame) -> None:
        idx = pd.PeriodIndex(["2019Q4", "2020Q1"], freq="Q")
        nowcast = build_nowcast_frame(
            pd.Series([0.3, np.nan], index=idx), pd.Series([0.2, 0.5], index=idx)
        )
        res = NowcastResults(target="gdp", nowcast=nowcast)
        dist = empirical_bands(res, table, vintage=VINTAGE, periods="2020Q1")
        assert dist.point.tolist() == [0.5]
        dist = empirical_bands(res, table, vintage=VINTAGE, periods=[pd.Period("2020-03", "M")])
        assert dist.index.astype(str).tolist() == ["2020Q1"]
        # In-sample periods may be requested too (horizon -1 has no errors here).
        with pytest.warns(DataQualityWarning), pytest.raises(NowcastDataError):
            empirical_bands(res, table, vintage=VINTAGE, periods=["2019Q4"])
        with pytest.raises(NowcastDataError, match="No estimate"):
            empirical_bands(res, table, vintage=VINTAGE, periods=["2021Q1"])

    def test_no_out_of_sample(self, table: pd.DataFrame) -> None:
        idx = pd.PeriodIndex(["2019Q4"], freq="Q")
        nowcast = build_nowcast_frame(pd.Series([0.3], index=idx), pd.Series([0.2], index=idx))
        res = NowcastResults(target="gdp", nowcast=nowcast)
        with pytest.raises(NowcastDataError, match="out-of-sample"):
            empirical_bands(res, table, vintage=VINTAGE)

    def test_vintage_from_info(self, table: pd.DataFrame) -> None:
        res = make_results(["2020Q1"], [0.0], info={"vintage": "2020-01-20"})
        assert empirical_bands(res, table).info["vintage"] == VINTAGE

    def test_vintage_from_data_edge(self, table: pd.DataFrame) -> None:
        index = pd.period_range("2019-01", "2020-03", freq="M")
        values = np.arange(len(index), dtype=float)
        values[-2:] = np.nan  # last data point: 2020-01 -> vintage 2020-02-01
        data = MixedFrequencyData(pd.DataFrame({"x": values}, index=index), {"x": "M"})
        res = make_results(["2020Q1"], [0.0], data=data)
        dist = empirical_bands(res, table)
        assert dist.info["vintage"] == pd.Timestamp("2020-02-01")
        assert dist.info["horizons"] == {"2020Q1": 1}

    def test_vintage_errors(self, table: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="vintage"):
            empirical_bands(make_results(), table)
        index = pd.period_range("2019-01", periods=3, freq="M")
        data = MixedFrequencyData(pd.DataFrame({"x": [np.nan] * 3}, index=index), {"x": "M"})
        with pytest.raises(ValueError, match="empty"):
            empirical_bands(make_results(data=data), table)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"method": "sd"}, "method"),
            ({"levels": []}, "level"),
            ({"levels": [1.2]}, "level"),
        ],
    )
    def test_bad_options(self, table: pd.DataFrame, kwargs, match) -> None:
        with pytest.raises(ValueError, match=match):
            empirical_bands(make_results(), table, vintage=VINTAGE, **kwargs)

    def test_scores_and_plot(self, table: pd.DataFrame) -> None:
        from nowcastbox.evaluation.scoring import crps

        res = make_results(["2020Q1"], [0.0])
        dist = empirical_bands(res, table, vintage=VINTAGE)
        assert np.isfinite(crps(dist, [0.3])).all()
        ax = dist.plot(levels=dist.info["levels"])
        assert len(ax.collections) == 3
        plt.close("all")


# ---------------------------------------------------------------------- coverage in real time
class TestBacktestBands:
    @pytest.mark.parametrize("method", ["mae", "rmse"])
    def test_nominal_coverage(self, method: str) -> None:
        sim = make_table("1960-01", "2019-12", seed=11)
        out = backtest_empirical_bands(sim, method=method, levels=[0.575, 0.9])
        ok = out[out["n_errors"] >= 30]
        assert len(ok) > 600
        for level, label in [(0.575, "57.5"), (0.9, "90")]:
            hits = (ok["actual"] >= ok[f"lower_{label}"]) & (ok["actual"] <= ok[f"upper_{label}"])
            assert hits.mean() == pytest.approx(level, abs=0.04)

    @pytest.mark.slow
    def test_nominal_coverage_quantile(self) -> None:
        # Empirical quantiles of n errors under-cover by O(1/n) (about 83 % for a 90 %
        # band with n = 40): check consistency with long expanding windows.
        sim = make_table("1880-01", "2019-12", seed=5)
        out = backtest_empirical_bands(sim, method="quantile", levels=[0.575, 0.9], window=None)
        ok = out[out["n_errors"] >= 250]
        assert len(ok) > 500
        for level, label in [(0.575, "57.5"), (0.9, "90")]:
            hits = (ok["actual"] >= ok[f"lower_{label}"]) & (ok["actual"] <= ok[f"upper_{label}"])
            assert hits.mean() == pytest.approx(level, abs=0.04)

    def test_columns_and_nan(self, table: pd.DataFrame) -> None:
        out = backtest_empirical_bands(table, levels=0.9, min_errors=8)
        assert list(out.columns) == [
            "vintage",
            "target_period",
            "months_to_end",
            "forecast",
            "actual",
            "n_errors",
            "sigma",
            "lower_90",
            "upper_90",
        ]
        assert len(out) == len(table)
        early = out.iloc[:10]
        assert early["sigma"].isna().all() and early["lower_90"].isna().all()
        assert (out["n_errors"].iloc[:10] < 8).all()

    def test_matches_empirical_bands(self, table: pd.DataFrame) -> None:
        out = backtest_empirical_bands(table, levels=[0.9])
        row = out.iloc[-1]
        res = make_results([str(row["target_period"])], [row["forecast"]])
        dist = empirical_bands(res, table, vintage=row["vintage"], levels=[0.9])
        band = dist.interval(0.9).iloc[0]
        assert band["lower"] == pytest.approx(row["lower_90"])
        assert band["upper"] == pytest.approx(row["upper_90"])
        assert dist.scales[0, 0] == pytest.approx(row["sigma"])

    def test_quantile_sigma_nan_and_model(self, backtest: BacktestResults) -> None:
        out = backtest_empirical_bands(backtest, model="AR", method="quantile", levels=0.5)
        assert out["sigma"].isna().all()
        assert (out["forecast"] == 0.0).all()
        assert out["lower_50"].notna().any()


# ---------------------------------------------------------------------- quantile object
class TestEmpiricalQuantileDistribution:
    @pytest.fixture
    def dist(self) -> EmpiricalQuantileDistribution:
        return EmpiricalQuantileDistribution(
            ["2020Q1", "2020Q2"],
            [1.0, 2.0],
            [[0.5, -0.5, 0.0, 1.0, -1.0], [2.0, -2.0]],
            target="gdp",
            info={"k": 1},
        )

    def test_attributes(self, dist: EmpiricalQuantileDistribution) -> None:
        assert len(dist) == dist.n_periods == 2
        assert not dist.is_gaussian
        assert dist.index.name == "period"
        assert dist.info == {"k": 1}
        assert dist.n_errors.tolist() == [5, 2]
        assert dist.errors[0].tolist() == [-1.0, -0.5, 0.0, 0.5, 1.0]
        assert dist.mean.tolist() == [1.0, 2.0]
        assert dist.std.tolist() == pytest.approx([np.std([0.5, -0.5, 0, 1, -1]), 2.0])
        assert dist.median.tolist() == [1.0, 2.0]
        assert repr(dist) == "EmpiricalQuantileDistribution('gdp', 2020Q1..2020Q2, 7 errors)"
        anon = EmpiricalQuantileDistribution(pd.PeriodIndex(["2020Q1"], freq="Q"), [0], [[1]])
        assert repr(anon) == "EmpiricalQuantileDistribution(2020Q1, 1 errors)"

    def test_selection(self, dist: EmpiricalQuantileDistribution) -> None:
        assert dist["2020Q2"].point.tolist() == [2.0]
        assert dist[-1].point.tolist() == [2.0]
        assert dist[pd.Period("2020-02", "M")].point.tolist() == [1.0]
        assert dist.select([0, 1]).levels == dist.levels
        with pytest.raises(KeyError):
            dist["2021Q1"]
        with pytest.raises(KeyError):
            dist[5]

    def test_quantiles(self, dist: EmpiricalQuantileDistribution) -> None:
        q = dist.quantiles([0.25, 0.75])
        assert q.to_numpy().tolist() == [[0.5, 1.5], [1.0, 3.0]]
        assert dist.ppf(0.5).tolist() == [[1.0], [2.0]]
        assert dist.interval(0.5).to_numpy().tolist() == [[0.5, 1.5], [1.0, 3.0]]
        for bad in ([], [0.0], [[0.5]]):
            with pytest.raises(ValueError, match="Probabilities"):
                dist.quantiles(bad)
        with pytest.raises(ValueError, match="level"):
            dist.interval(1.0)

    def test_cdf_and_crps(self, dist: EmpiricalQuantileDistribution) -> None:
        assert dist.cdf([1.0, 0.0]).tolist() == [0.6, 0.5]
        obs = np.array([1.3, -0.4])
        expected = [
            crps_sample([obs[0]], [1.0 + dist.errors[0]])[0],
            crps_sample([obs[1]], [2.0 + dist.errors[1]])[0],
        ]
        assert dist.crps(obs) == pytest.approx(expected)

    def test_sample(self, dist: EmpiricalQuantileDistribution) -> None:
        draws = dist.sample(500, random_state=0)
        assert draws.shape == (500, 2)
        assert set(np.round(draws.iloc[:, 1], 8)) <= {0.0, 4.0}
        for bad in (0, True, 1.5):
            with pytest.raises(ValueError, match="positive integer"):
                dist.sample(bad)  # type: ignore[arg-type]

    def test_frames(self, dist: EmpiricalQuantileDistribution) -> None:
        frame = dist.to_frame()
        assert list(frame.columns) == [
            "point",
            "mean",
            "std",
            "median",
            "lower_57.5",
            "upper_57.5",
            "lower_68",
            "upper_68",
            "lower_90",
            "upper_90",
            "n_errors",
        ]
        assert list(dist.to_frame([0.5]).columns)[4:6] == ["lower_50", "upper_50"]
        fan = dist.fan_chart_frame([0.5, 0.9])
        assert fan["level"].tolist() == [0.9, 0.9, 0.5, 0.5]
        assert list(fan.columns) == ["period", "level", "lower", "upper", "median"]

    def test_plot(self, dist: EmpiricalQuantileDistribution) -> None:
        history = pd.Series(
            [0.5, np.nan, 1.5], index=pd.period_range("2019Q2", periods=3, freq="Q")
        )
        ax = dist.plot(history=history, levels=[0.9], title="T", color="C1")
        assert ax.get_title() == "T"
        assert len(ax.lines) == 2
        _, ax2 = plt.subplots()
        assert dist.plot(ax=ax2, history=pd.Series([1.0])) is ax2
        assert ax2.get_title() == "Empirical error bands: gdp"
        plt.close("all")

    def test_fan_chart_registry_duck_typing(self, dist: EmpiricalQuantileDistribution) -> None:
        from nowcastbox.visualization import plot_fan_chart

        fig = plot_fan_chart(dist, backend="matplotlib")
        assert fig is not None
        plt.close("all")

    @pytest.mark.parametrize(
        ("args", "match"),
        [
            ((["2020Q1", "2020Q1"], [0, 0], [[1], [1]]), "duplicated"),
            (([], [], []), "at least one"),
            ((["2020Q1"], [np.nan], [[1]]), "point"),
            ((["2020Q1"], [0, 1], [[1]]), "point"),
            ((["2020Q1"], [0], [[1], [2]]), "one array per period"),
            ((["2020Q1"], [0], [[]]), "at least one error"),
            ((["2020Q1"], [0], [[np.inf]]), "finite"),
        ],
    )
    def test_validation(self, args, match) -> None:
        with pytest.raises(ValueError, match=match):
            EmpiricalQuantileDistribution(*args)

    def test_levels_validation(self) -> None:
        with pytest.raises(ValueError, match="level"):
            EmpiricalQuantileDistribution(["2020Q1"], [0], [[1]], levels=[0.0])


# ---------------------------------------------------------------------- gaussian bands object
class TestEmpiricalGaussianDistribution:
    def test_default_levels_follow_the_bands(self, table: pd.DataFrame) -> None:
        res = make_results(["2020Q1"], [0.4])
        dist = empirical_bands(res, table, vintage=VINTAGE)
        assert isinstance(dist, EmpiricalGaussianDistribution)
        assert isinstance(dist, NowcastDistribution)
        assert dist.levels == EMPIRICAL_LEVELS
        frame = dist.to_frame()
        assert "upper_57.5" in frame.columns and "upper_50" not in frame.columns
        assert frame["upper_57.5"].iloc[0] == pytest.approx(dist.interval(0.575)["upper"].iloc[0])
        assert dist.fan_chart_frame()["level"].tolist() == [0.9, 0.68, 0.575]
        assert list(dist.to_frame(0.5).columns)[-1] == "upper_50"
        assert dist.fan_chart_frame([0.5])["level"].tolist() == [0.5]
        custom = empirical_bands(res, table, vintage=VINTAGE, method="rmse", levels=[0.8])
        assert list(custom.to_frame().columns)[-2:] == ["lower_80", "upper_80"]

    def test_selection_and_repr(self) -> None:
        d = EmpiricalGaussianDistribution(
            ["2020Q1", "2020Q2"],
            [0.0, 1.0],
            [1.0, 2.0],
            target="gdp",
            levels=[0.9],
            info={"method": "mae"},
        )
        one = d["2020Q2"]
        assert isinstance(one, EmpiricalGaussianDistribution)
        assert one.levels == (0.9,)
        assert one.point.tolist() == [1.0] and one.std.tolist() == [2.0]
        assert one.info == {"method": "mae", "levels": (0.9,)}
        assert repr(d).startswith("EmpiricalGaussianDistribution('gdp', 2020Q1..2020Q2")
        with pytest.raises(ValueError, match="level"):
            EmpiricalGaussianDistribution(["2020Q1"], [0.0], [1.0], levels=[1.5])

    def test_plot(self) -> None:
        d = EmpiricalGaussianDistribution(["2020Q1", "2020Q2"], [0.0, 1.0], [1.0, 2.0])
        ax = d.plot()
        assert len(ax.collections) == 3
        assert ax.get_title() == "Empirical error bands:"
        ax = d.plot(levels=[0.9], title="x")
        assert len(ax.collections) == 1 and ax.get_title() == "x"
        plt.close("all")
