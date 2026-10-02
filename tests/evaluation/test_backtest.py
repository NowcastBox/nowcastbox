"""Tests of PseudoRealTimeBacktest and BacktestResults."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from nowcastbox.benchmarks import AR, UMIDAS, BridgeBenchmark, RandomWalk
from nowcastbox.core.base import BaseNowcaster
from nowcastbox.core.data import MixedFrequencyData
from nowcastbox.core.exceptions import NowcastBoxWarning, NowcastDataError
from nowcastbox.core.results import NowcastResults, build_nowcast_frame
from nowcastbox.evaluation import (
    FORECAST_COLUMNS,
    BacktestResults,
    ModelConfidenceSetResult,
    PseudoRealTimeBacktest,
    clark_west,
    diebold_mariano,
)
from nowcastbox.models import TwoStepDFM
from nowcastbox.vintages import ReleaseCalendar, VintageStore, pseudo_real_time, vintage_dates

FREQ = {"x": "M", "gdp": "Q"}
DELAY = {"x": 20, "gdp": 45}


def make_frame(n: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.7 * x[t - 1] + rng.standard_normal()
    idx = pd.period_range("2000-01", periods=n, freq="M")
    gdp = 0.5 + pd.Series(x, index=idx).rolling(3).mean() + 0.3 * rng.standard_normal(n)
    return pd.DataFrame({"x": x, "gdp": gdp.where(idx.month % 3 == 0)}, index=idx)


@pytest.fixture
def frame() -> pd.DataFrame:
    return make_frame()


@pytest.fixture
def data(frame) -> MixedFrequencyData:
    return MixedFrequencyData(frame, FREQ, release_delays=DELAY)


def backtest(data: Any, **kwargs: Any) -> PseudoRealTimeBacktest:
    options: dict[str, Any] = {
        "start": "2008-01-15",
        "end": "2008-12-15",
        "benchmarks": [AR(), BridgeBenchmark()],
    }
    options.update(kwargs)
    target = options.pop("target", "gdp")
    return PseudoRealTimeBacktest(data=data, target=target, **options)


# ---------------------------------------------------------------------- helper forecasters
class Spy:
    """Protocol forecaster recording the panels it is fitted on."""

    panels: ClassVar[list[MixedFrequencyData]] = []

    def fit(self, data: MixedFrequencyData, target: str) -> Spy:
        Spy.panels.append(data)
        self.last = float(data.to_native(target).dropna().iloc[-1])
        return self

    def predict(self, periods: pd.PeriodIndex) -> pd.Series:
        return pd.Series(self.last, index=periods)


class Failing:
    def fit(self, data: MixedFrequencyData, target: str) -> Failing:
        raise NowcastDataError("boom")

    def predict(self, periods: pd.PeriodIndex) -> pd.Series:  # pragma: no cover
        return pd.Series(0.0, index=periods)


def mean_results(data: MixedFrequencyData, target: str, value: float) -> NowcastResults:
    observed = data.to_native(target)
    estimate = pd.Series(value, index=observed.index)
    return NowcastResults(target=target, nowcast=build_nowcast_frame(observed, estimate))


class Counting(BaseNowcaster):
    """Nowcaster predicting the sample mean; counts fits and updates."""

    counts: ClassVar[dict[str, int]] = {"fit": 0, "update": 0}

    def __init__(self, offset: float = 0.0) -> None:
        self.offset = offset

    def _fit(self, data: MixedFrequencyData, target: str, **kw: Any) -> NowcastResults:
        Counting.counts["fit"] += 1
        self.mean_ = float(data.to_native(target).mean()) + self.offset
        return mean_results(data, target, self.mean_)

    def update(self, data: MixedFrequencyData) -> NowcastResults:
        Counting.counts["update"] += 1
        return mean_results(data, self.results_.target, -100.0)


class NoUpdate(BaseNowcaster):
    def __init__(self) -> None:
        pass

    def _fit(self, data: MixedFrequencyData, target: str, **kw: Any) -> NowcastResults:
        return mean_results(data, target, float(len(data.index)))


class BadUpdate(NoUpdate):
    def update(self, data: MixedFrequencyData) -> None:
        return None


@dataclass(frozen=True, kw_only=True, eq=False, repr=False)
class PredictingResults(NowcastResults):
    def predict(self, data: MixedFrequencyData) -> pd.DataFrame:
        return data.data.assign(**{self.target: 7.0})


@dataclass(frozen=True, kw_only=True, eq=False, repr=False)
class BadPredictResults(NowcastResults):
    def predict(self, data: MixedFrequencyData) -> pd.DataFrame:
        return data.data.drop(columns=self.target)


class BadPredict(BaseNowcaster):
    def __init__(self) -> None:
        pass

    def _fit(self, data: MixedFrequencyData, target: str, **kw: Any) -> NowcastResults:
        base = mean_results(data, target, float(len(data.index)))
        return BadPredictResults(target=target, nowcast=base.nowcast)


class EMLike(BaseNowcaster):
    def __init__(self) -> None:
        pass

    def _fit(self, data: MixedFrequencyData, target: str, **kw: Any) -> NowcastResults:
        base = mean_results(data, target, 1.0)
        return PredictingResults(target=target, nowcast=base.nowcast)


# ---------------------------------------------------------------------- running
class TestRun:
    def test_forecast_table(self, data) -> None:
        res = backtest(data).run()
        frame = res.forecasts
        assert list(frame.columns) == list(FORECAST_COLUMNS)
        assert res.models == ["AR", "BridgeBenchmark"]
        assert res.reference == "AR"
        assert res.target == "gdp"
        assert res.info["n_vintages"] == 12
        first = frame[frame["vintage"] == pd.Timestamp("2008-01-15")]
        assert first["target_period"].astype(str).unique().tolist() == [
            "2007Q4",
            "2008Q1",
            "2008Q2",
        ]
        assert first["kind"].unique().tolist() == ["backcast", "nowcast", "forecast"]
        assert first["months_to_end"].unique().tolist() == [-1, 2, 5]
        assert first["days_to_end"].unique().tolist() == [-15, 76, 167]
        # 2007Q4 is released on 2007-12-31 + 45 days = 2008-02-14
        assert first["days_to_release"].iloc[0] == 30.0
        later = frame[frame["vintage"] == pd.Timestamp("2008-02-15")]
        assert "2007Q4" not in later["target_period"].astype(str).tolist()
        actual = data.to_native("gdp")
        row = frame.iloc[0]
        assert row["actual"] == actual[row["target_period"]]
        assert row["error"] == pytest.approx(row["actual"] - row["forecast"])

    def test_benchmark_forecasts_match_direct_fit(self, data, frame) -> None:
        res = backtest(data, benchmarks=[UMIDAS()]).run()
        vintage = pd.Timestamp("2008-05-15")
        panel = pseudo_real_time(data, vintage=vintage).truncate(end="2008-09")
        direct = UMIDAS().fit(panel, "gdp").predict(["2008Q2", "2008Q3"])  # 2008Q1 released
        got = res.forecasts[res.forecasts["vintage"] == vintage]["forecast"].to_numpy()
        np.testing.assert_allclose(got, direct.to_numpy())

    def test_no_look_ahead(self, data) -> None:
        Spy.panels.clear()
        backtest(data, benchmarks=[Spy()]).run()
        dates = vintage_dates("2008-01-15", "2008-12-15", "M")
        assert len(Spy.panels) == 12
        for date, panel in zip(dates, Spy.panels, strict=True):
            expected = pseudo_real_time(data, vintage=date).data.reindex(panel.index)
            pd.testing.assert_frame_equal(panel.data, expected, check_freq=False)
            assert panel.end == pd.Period(date, "M").asfreq("Q").asfreq("M", how="E") + 3

    def test_rolling_window(self, data) -> None:
        Spy.panels.clear()
        backtest(data, benchmarks=[Spy()], window="rolling", window_length=24).run()
        assert Spy.panels[0].start == pd.Period("2006-02", "M")
        assert Spy.panels[-1].start == pd.Period("2007-01", "M")

    def test_offsets_and_include_released(self, data) -> None:
        res = backtest(data, target_offsets=(0,), include_released=True).run()
        assert set(res.forecasts["offset"]) == {0}
        assert res.forecasts.groupby("model").size().tolist() == [12, 12]
        res = backtest(data, target_offsets=(-1,)).run()
        assert set(res.forecasts["months_to_end"]) == {-1}

    def test_forecast_beyond_the_data(self, data) -> None:
        res = backtest(data, start="2009-11-15", end="2009-12-15", target_offsets=(0, 2)).run()
        future = res.forecasts[res.forecasts["target_period"] == pd.Period("2010Q2", "Q")]
        assert np.isfinite(future["forecast"]).all()
        assert future["actual"].isna().all()

    def test_protocol_model_is_refitted(self, data) -> None:
        Spy.panels.clear()
        backtest(data, model=Spy(), refit_every=3, benchmarks=None).run()
        assert len(Spy.panels) == 12

    @pytest.mark.filterwarnings("ignore::nowcastbox.core.exceptions.DataQualityWarning")
    def test_model_with_formula_and_dfm(self, data) -> None:
        res = backtest(data, model=TwoStepDFM(n_factors=1), target="gdp ~ x").run()
        assert res.models == ["TwoStepDFM", "AR", "BridgeBenchmark"]
        assert res.reference == "AR"
        assert res.forecasts["forecast"].notna().all()

    def test_model_name(self, data) -> None:
        res = backtest(data, model=Spy(), model_name="DFM-a", refit_every=6).run()
        assert res.models[0] == "DFM-a"

    def test_refit_every_uses_update(self, data) -> None:
        Counting.counts.update(fit=0, update=0)
        res = backtest(data, model=Counting(), refit_every=4, benchmarks=None).run()
        assert Counting.counts == {"fit": 3, "update": 9}
        values = res.forecasts.groupby("vintage")["forecast"].first()
        assert (values.iloc[[1, 2, 3, 5]] == -100.0).all()
        assert values.iloc[4] != -100.0
        assert res.info["refit_every"] == 4

    def test_refit_without_update(self, data) -> None:
        for model in (NoUpdate(), BadUpdate(), BadPredict()):
            res = backtest(data, model=model, refit_every=3, benchmarks=None).run()
            sizes = res.forecasts.groupby("vintage")["forecast"].first()
            assert sizes.nunique() > 1  # refitted on growing windows

    def test_results_predict_path(self, data) -> None:
        res = backtest(data, model=EMLike(), refit_every=12, benchmarks=None).run()
        values = res.forecasts.groupby("vintage")["forecast"].first()
        assert values.iloc[0] == 1.0
        assert (values.iloc[1:] == 7.0).all()

    def test_sklearn_and_named_benchmarks(self, data) -> None:
        res = backtest(data, benchmarks={"ridge": Ridge(), "rw": RandomWalk()}).run()
        assert res.models == ["ridge", "rw"]
        res = backtest(data, benchmarks=[AR(), AR(p=2), Ridge()]).run()
        assert res.models == ["AR", "AR_2", "Sklearn(Ridge)"]

    def test_invalid_forecaster(self, data) -> None:
        with pytest.raises(TypeError, match="neither"):
            backtest(data, benchmarks=[object()]).run()

    def test_errors_policy(self, data) -> None:
        with pytest.raises(NowcastDataError, match="boom"):
            backtest(data, benchmarks=[AR(), Failing()]).run()
        with pytest.warns(NowcastBoxWarning, match="12 forecaster fits failed"):
            res = backtest(data, benchmarks=[AR(), Failing()], errors="warn").run()
        failing = res.forecasts[res.forecasts["model"] == "Failing"]
        assert failing["forecast"].isna().all()
        assert len(res.info["failures"]) == 12

    def test_parallel_matches_sequential(self, data) -> None:
        sequential = backtest(data).run()
        parallel = backtest(data, n_jobs=2).run()
        pd.testing.assert_frame_equal(sequential.forecasts, parallel.forecasts)

    def test_parallel_without_joblib(self, data, monkeypatch) -> None:
        monkeypatch.setitem(sys.modules, "joblib", None)
        res = backtest(data).run(n_jobs=2)
        assert len(res.forecasts) > 0


class TestReleaseRules:
    def test_delay_specifications(self, frame, data) -> None:
        ref = backtest(data).run().forecasts
        variants = [
            {"delay": DELAY},
            {"delay": pd.Series(DELAY)},
            {"delay": [20, 45]},
            {"calendar": ReleaseCalendar(DELAY, frequencies=FREQ)},
        ]
        for kwargs in variants:
            res = backtest(frame, frequency=FREQ, **kwargs).run()
            pd.testing.assert_frame_equal(res.forecasts, ref)

    def test_integer_delay(self, frame) -> None:
        res = backtest(frame, frequency=FREQ, delay=30).run()
        assert res.forecasts["days_to_release"].iloc[0] == 30 + 31 - 15 - 31

    def test_invalid_delays(self, frame) -> None:
        with pytest.raises(ValueError, match="either delay or calendar"):
            backtest(frame, frequency=FREQ, delay=DELAY, calendar=DELAY).run()
        with pytest.raises(ValueError, match="one value per series"):
            backtest(frame, frequency=FREQ, delay=[1, 2, 3]).run()
        with pytest.raises(ValueError, match="Invalid delay"):
            backtest(frame, frequency=FREQ, delay=1.5).run()

    def test_unknown_release_dates_give_nan(self, data) -> None:
        explicit = {
            "gdp": {str(p): p.end_time.normalize() + pd.Timedelta(days=45)
                    for p in data.to_native("gdp").dropna().index}
        }  # fmt: skip
        calendar = ReleaseCalendar({"x": 20}, explicit, frequencies=FREQ)
        res = backtest(data, calendar=calendar, start="2009-11-15", end="2009-12-15").run()
        future = res.forecasts[res.forecasts["target_period"] > pd.Period("2009Q4", "Q")]
        assert future["days_to_release"].isna().all()

    def test_release_step(self, data) -> None:
        res = backtest(data, step="release", end="2008-03-31").run()
        assert res.info["n_vintages"] == 5


class TestRealTime:
    @pytest.fixture
    def store(self, data) -> VintageStore:
        store = VintageStore.from_calendar(data)
        gdp = data.to_native("gdp").dropna()
        revised = pd.DataFrame(
            {
                "series": "gdp",
                "reference_period": [str(p) for p in gdp.index],
                "vintage_date": [p.end_time.normalize() + pd.Timedelta(days=90) for p in gdp.index],
                "value": gdp.to_numpy() + 1.0,
            }
        )
        return store.add(revised)

    def test_actual_releases(self, store) -> None:
        latest = backtest(store).run()
        first = backtest(store, actual="first").run()
        second = backtest(store, actual=1).run()
        diff = latest.forecasts["actual"] - first.forecasts["actual"]
        np.testing.assert_allclose(diff.dropna(), 1.0)
        pd.testing.assert_series_equal(second.forecasts["actual"], latest.forecasts["actual"])
        assert latest.info["real_time"]
        assert "real-time vintages" in latest.summary()

    def test_store_uses_revised_data(self, store, data) -> None:
        Spy.panels.clear()
        backtest(store, benchmarks=[Spy()], start="2008-07-15", end="2008-07-15").run()
        panel = Spy.panels[0]
        assert panel["gdp"].loc["2008-03"] == pytest.approx(data["gdp"].loc["2008-03"] + 1.0)
        assert np.isnan(panel["gdp"].loc["2008-06"])

    def test_release_step_and_days_to_release(self, store) -> None:
        res = backtest(store, step="release", end="2008-03-31").run()
        assert res.info["n_vintages"] == 6
        frame = res.forecasts
        row = frame[frame["target_period"] == pd.Period("2007Q4", "Q")].iloc[0]
        assert row["days_to_release"] == 30.0

    def test_metadata_passed_to_store(self, store) -> None:
        res = backtest(store, metadata={"release_delays": DELAY}).run()
        assert len(res.forecasts) > 0

    def test_store_validation(self, store) -> None:
        with pytest.raises(ValueError, match="VintageStore"):
            backtest(store, delay=DELAY).run()
        with pytest.raises(ValueError, match="frequency"):
            backtest(store, frequency=FREQ).run()
        with pytest.raises(ValueError, match="Invalid actual"):
            backtest(store, actual="median").run()


class TestValidation:
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"window": "sliding"}, "window"),
            ({"window": "rolling"}, "window_length"),
            ({"errors": "ignore"}, "errors"),
            ({"refit_every": 0}, "refit_every"),
            ({"target_offsets": ()}, "target_offsets"),
            ({"target_offsets": (0.5,)}, "target_offsets"),
            ({"benchmarks": None}, "model and/or benchmarks"),
            ({"start": None}, "start and end"),
            ({"metadata": {"blocks": None}}, "VintageStore"),
            ({"actual": 1}, "pseudo real-time"),
            ({"model_name": ""}, "model_name"),
        ],
    )
    def test_invalid(self, data, kwargs, match) -> None:
        with pytest.raises(ValueError, match=match):
            backtest(data, **kwargs).run()

    def test_missing_data(self) -> None:
        with pytest.raises(ValueError, match="data and target"):
            PseudoRealTimeBacktest(start="2008-01-01", end="2008-02-01").run()


# ---------------------------------------------------------------------- results
@pytest.fixture(scope="module")
def results() -> BacktestResults:
    data = MixedFrequencyData(make_frame(180, seed=3), FREQ, release_delays=DELAY)
    return PseudoRealTimeBacktest(
        data=data,
        target="gdp",
        start="2006-01-15",
        end="2014-11-15",
        benchmarks=[AR(), BridgeBenchmark(), RandomWalk()],
    ).run()


class TestResults:
    def test_actual_series(self, data) -> None:
        actual = data.to_native("gdp").dropna() * 0 + 1.0
        actual.index = actual.index.astype(str)
        res = backtest(data, actual=actual).run()
        assert (res.forecasts["actual"].dropna() == 1.0).all()

    def test_frames(self, results) -> None:
        wide = results.to_frame(wide=True)
        assert wide.columns.tolist() == [
            "AR",
            "BridgeBenchmark",
            "RandomWalk",
            "actual",
            "previous_actual",
        ]
        assert len(wide) == len(results.forecasts) // 3
        assert results.to_frame().equals(results.forecasts)
        assert "n_forecasts" in repr(results)

    def test_parquet(self, results, tmp_path) -> None:
        pytest.importorskip("pyarrow")
        path = results.to_parquet(tmp_path / "bt.parquet")
        back = pd.read_parquet(path)
        assert back.shape == results.forecasts.shape

    def test_evaluable(self, results) -> None:
        common = results.evaluable()
        assert common.groupby(["vintage", "target_period"])["model"].nunique().eq(3).all()
        assert results.evaluable(["AR"], common_sample=False)["model"].unique().tolist() == ["AR"]
        with pytest.raises(ValueError, match="Unknown models"):
            results.evaluable(["XYZ"])

    def test_accuracy(self, results) -> None:
        rmsfe = results.rmsfe_by_horizon()
        assert rmsfe.index.tolist() == [-1, 0, 1, 2, 3, 4, 5]
        assert (rmsfe["BridgeBenchmark"].loc[[-1, 0]] < rmsfe["AR"].loc[[-1, 0]]).all()
        table = results.metrics(horizon="kind")
        assert table.index.tolist() == ["backcast", "forecast", "nowcast"]
        assert table[("n", "AR")].sum() == len(results.evaluable()) / 3
        pooled = results.rmsfe_by_horizon(None, models=["AR"])
        assert pooled.columns.tolist() == ["AR"]

    def test_relative(self, results) -> None:
        rel = results.relative_to()
        assert (rel["AR"] == 1.0).all()
        rel = results.relative_to("RandomWalk", horizon=None, models=["AR"], metric="mae")
        assert rel.columns.tolist() == ["AR", "RandomWalk"]
        no_ref = BacktestResults(results.forecasts, results.models, "gdp")
        with pytest.raises(ValueError, match="No reference"):
            no_ref.relative_to()

    def test_diebold_mariano(self, results) -> None:
        dm = results.diebold_mariano()
        assert dm.index.names == ["model", "horizon"]
        assert set(dm.index.get_level_values("model")) == {"BridgeBenchmark", "RandomWalk"}
        assert dm.loc[("BridgeBenchmark", 0), "statistic"] < 0
        pooled = results.diebold_mariano("RandomWalk", horizon=None, alternative="less")
        assert pooled.loc[("BridgeBenchmark", "all"), "pvalue"] < 0.05

    def test_dm_aggregate_by_target_period(self, results) -> None:
        pooled = results.diebold_mariano("RandomWalk", horizon="kind")
        agg = results.diebold_mariano("RandomWalk", horizon="kind", aggregate="target_period")
        frame = results.evaluable(["BridgeBenchmark", "RandomWalk"])
        for kind in ("backcast", "nowcast"):
            key = ("BridgeBenchmark", kind)
            n_periods = frame.loc[frame["kind"] == kind, "target_period"].nunique()
            assert agg.loc[key, "n_obs"] == n_periods <= pooled.loc[key, "n_obs"]
        key = ("BridgeBenchmark", "nowcast")
        assert agg.loc[key, "n_obs"] < pooled.loc[key, "n_obs"]
        # manual: average the squared-loss differential within each target period
        sub = frame[frame["kind"] == "nowcast"].pivot_table(
            index=["target_period", "vintage"], columns="model", values="error"
        )
        d = (sub["BridgeBenchmark"] ** 2 - sub["RandomWalk"] ** 2).groupby(level=0).mean()
        expected = diebold_mariano(d.to_numpy(), np.zeros(d.size), loss=lambda e: e, h=2)
        got = results.diebold_mariano("RandomWalk", horizon="kind", aggregate="target_period", h=2)
        assert got.loc[("BridgeBenchmark", "nowcast"), "statistic"] == pytest.approx(
            expected.statistic
        )
        with pytest.raises(ValueError, match="aggregate"):
            results.diebold_mariano(aggregate="vintage")

    def test_clark_west(self, results) -> None:
        cw = results.clark_west("RandomWalk", horizon=None)
        dm = results.diebold_mariano("RandomWalk", horizon=None)
        key = ("BridgeBenchmark", "all")
        assert cw.loc[key, "statistic"] < dm.loc[key, "statistic"]
        assert cw.loc[key, "pvalue"] < 0.05
        agg = results.clark_west("RandomWalk", horizon="kind", aggregate="target_period")
        assert agg["n_obs"].gt(0).all()
        frame = results.evaluable(["AR", "RandomWalk"])
        sub = frame.pivot_table(index=["target_period", "vintage"], columns="model", values="error")
        manual = clark_west(sub["AR"].to_numpy(), sub["RandomWalk"].to_numpy())
        got = results.clark_west("RandomWalk", horizon=None).loc[("AR", "all")]
        assert got["statistic"] == pytest.approx(manual.statistic)

    def test_gw_and_mcs_aggregate(self, results) -> None:
        gw = results.giacomini_white(horizon=None, aggregate="target_period")
        n_periods = results.evaluable()["target_period"].nunique()
        assert gw["n_obs"].max() <= n_periods
        pooled = results.mcs(n_bootstrap=100, aggregate="target_period")
        assert isinstance(pooled, ModelConfidenceSetResult)
        by = results.mcs(
            horizon="kind",
            n_bootstrap=50,
            models=["AR", "BridgeBenchmark"],
            aggregate="target_period",
        )
        assert isinstance(by, dict) and sorted(by) == ["backcast", "forecast", "nowcast"]
        with pytest.raises(ValueError, match="aggregate"):
            results.mcs(aggregate="month")

    def test_dm_with_too_few_pairs(self, data) -> None:
        res = backtest(data, start="2008-01-15", end="2008-02-15").run()
        dm = res.diebold_mariano()
        assert dm["statistic"].isna().all()

    def test_giacomini_white(self, results) -> None:
        gw = results.giacomini_white(horizon=None)
        assert gw.loc[("BridgeBenchmark", "all"), "pvalue"] < 0.05
        assert gw["n_obs"].gt(0).all()

    def test_mcs(self, results) -> None:
        pooled = results.mcs(alpha=0.1, n_bootstrap=200)
        assert isinstance(pooled, ModelConfidenceSetResult)
        assert "BridgeBenchmark" in pooled.included
        assert "RandomWalk" not in pooled.included
        by = results.mcs(horizon="kind", n_bootstrap=100, models=["AR", "BridgeBenchmark"])
        assert isinstance(by, dict)
        assert sorted(by) == ["backcast", "forecast", "nowcast"]
        assert by["nowcast"].included == ["BridgeBenchmark"]
        by_month = results.mcs(
            horizon="months_to_end", n_bootstrap=50, models=["AR", "BridgeBenchmark"]
        )
        assert isinstance(by_month, dict) and len(by_month) >= 1

    def test_loss_table(self, results) -> None:
        losses = results.loss_table(loss="absolute", models=["AR", "RandomWalk"])
        assert losses.columns.tolist() == ["AR", "RandomWalk"]
        assert (losses >= 0).all().all()

    def test_summary(self, results) -> None:
        text = results.summary()
        assert "Pseudo real-time backtest" in text
        assert "rel_rmsfe" in text
        assert "RMSFE by months_to_end" in text
        short = results.summary(horizon=None)
        assert "RMSFE by" not in short
        single = BacktestResults(
            results.forecasts[results.forecasts["model"] == "AR"], ["AR"], "gdp", reference="AR"
        )
        assert "rel_rmsfe" not in single.summary()


# ---------------------------------------------------------------------- per-vintage preprocessing
def test_preprocess_is_applied_to_each_vintage_without_look_ahead(data):
    seen: list[tuple[pd.Period, pd.Period]] = []
    final_last = data.to_frame()["x"].last_valid_index()

    def preprocess(vintage: MixedFrequencyData) -> MixedFrequencyData:
        frame = vintage.to_frame()
        seen.append((frame["x"].last_valid_index(), final_last))
        cleaned = frame.copy()
        cleaned["x"] = cleaned["x"] - cleaned["x"].mean()  # stand-in for a cleaning step
        return MixedFrequencyData(cleaned, FREQ, release_delays=DELAY)

    Spy.panels = []
    bt = backtest(data, model=None, benchmarks={"spy": Spy()}, preprocess=preprocess)
    bt.run()
    assert seen, "preprocess was never called"
    # Every vintage handed to preprocess ends before the final sample does.
    assert all(last < final for last, final in seen)
    # The forecasters only ever receive the preprocessed vintages.
    assert all(abs(float(p.to_frame()["x"].mean())) < 1e-12 for p in Spy.panels)


def test_preprocess_none_keeps_raw_vintages(data):
    Spy.panels = []
    backtest(data, model=None, benchmarks={"spy": Spy()}).run()
    assert any(abs(float(p.to_frame()["x"].mean())) > 1e-6 for p in Spy.panels)
