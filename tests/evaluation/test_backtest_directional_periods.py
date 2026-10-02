"""Tests of directional accuracy and sub-period evaluation in BacktestResults."""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np
import pandas as pd
import pytest

from nowcastbox.benchmarks import AR, BridgeBenchmark
from nowcastbox.core.data import MixedFrequencyData
from nowcastbox.core.exceptions import DataQualityWarning
from nowcastbox.evaluation import (
    FORECAST_COLUMNS,
    BacktestResults,
    ModelConfidenceSetResult,
    PseudoRealTimeBacktest,
    covid_periods,
    directional_accuracy,
    pesaran_timmermann,
)
from nowcastbox.vintages import VintageStore, pseudo_real_time

FREQ = {"x": "M", "gdp": "Q"}
DELAY = {"x": 20, "gdp": 45}
PERIODS = {
    "pre-Covid": ("2017Q1", "2019Q4"),
    "Covid": ("2020Q1", "2021Q4"),
    "post-Covid": ("2022Q1", None),
}


def make_frame(n: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.7 * x[t - 1] + rng.standard_normal()
    idx = pd.period_range("2000-01", periods=n, freq="M")
    gdp = 0.5 + pd.Series(x, index=idx).rolling(3).mean() + 0.3 * rng.standard_normal(n)
    return pd.DataFrame({"x": x, "gdp": gdp.where(idx.month % 3 == 0)}, index=idx)


@pytest.fixture(scope="module")
def data() -> MixedFrequencyData:
    return MixedFrequencyData(make_frame(), FREQ, release_delays=DELAY)


@pytest.fixture(scope="module")
def run(data) -> BacktestResults:
    return PseudoRealTimeBacktest(
        data=data,
        target="gdp",
        start="2008-01-15",
        end="2009-06-15",
        benchmarks=[AR(), BridgeBenchmark()],
    ).run()


def synthetic(seed: int = 0) -> BacktestResults:
    """Three monthly nowcasts per quarter (2017Q1-2023Q4) of two models."""
    rng = np.random.default_rng(seed)
    quarters = pd.period_range("2017Q1", "2023Q4", freq="Q")
    actual = pd.Series(rng.standard_normal(len(quarters)), index=quarters)
    rows = []
    for q in quarters:
        for k, month in enumerate(q.asfreq("M", how="S") + np.arange(3)):
            vintage = month.to_timestamp() + pd.Timedelta(days=14)
            prev = actual.get(q - 1, np.nan)
            for model, scale in (("A", 0.3), ("B", 1.0)):
                forecast = actual[q] + scale * rng.standard_normal()
                rows.append(
                    {
                        "vintage": vintage,
                        "target_period": q,
                        "offset": 0,
                        "kind": "nowcast",
                        "months_to_end": 2 - k,
                        "days_to_end": 0,
                        "days_to_release": 60.0,
                        "model": model,
                        "forecast": forecast,
                        "actual": actual[q],
                        "error": actual[q] - forecast,
                        "previous_actual": prev,
                    }
                )
    frame = pd.DataFrame(rows, columns=list(FORECAST_COLUMNS))
    frame["target_period"] = pd.PeriodIndex(frame["target_period"], freq="Q")
    return BacktestResults(frame, ["A", "B"], "gdp", reference="B")


@pytest.fixture(scope="module")
def res() -> BacktestResults:
    return synthetic()


def manual(res: BacktestResults, first: str | None, last: str | None) -> BacktestResults:
    """The filter applied by hand (what the papers' scripts do)."""
    periods = res.forecasts["target_period"]
    keep = np.ones(len(periods), dtype=bool)
    if first is not None:
        keep &= periods >= pd.Period(first, "Q")
    if last is not None:
        keep &= periods <= pd.Period(last, "Q")
    return dataclasses.replace(res, forecasts=res.forecasts[keep])


# ---------------------------------------------------------------------- previous value
class TestPreviousActual:
    def test_column_and_no_look_ahead(self, run, data) -> None:
        frame = run.forecasts
        assert list(frame.columns) == list(FORECAST_COLUMNS)
        for (vintage, period), group in frame.groupby(["vintage", "target_period"]):
            observed = pseudo_real_time(data, vintage=vintage).to_native("gdp", dropna=True)
            known = observed[observed.index <= period - 1]
            expected = known.iloc[-1] if len(known) else np.nan
            np.testing.assert_allclose(group["previous_actual"], expected)

    def test_unreleased_previous_period_uses_last_release(self, run, data) -> None:
        final = data.to_native("gdp")
        frame = run.forecasts
        # 2008-01-15: 2007Q4 is released on 2008-02-14, so the 2008Q1 nowcast compares
        # with 2007Q3 (last value known); from 2008-02-15 on, with 2007Q4.
        early = frame[(frame["vintage"] == "2008-01-15") & (frame["offset"] == 0)]
        assert early["previous_actual"].iloc[0] == pytest.approx(final[pd.Period("2007Q3")])
        later = frame[(frame["vintage"] == "2008-02-15") & (frame["offset"] == 0)]
        assert later["previous_actual"].iloc[0] == pytest.approx(final[pd.Period("2007Q4")])

    def test_preprocess_does_not_change_previous(self, data) -> None:
        def scale(vintage: MixedFrequencyData) -> MixedFrequencyData:
            return vintage.with_data(vintage.to_frame() * 10.0)

        options: dict[str, Any] = {
            "data": data,
            "target": "gdp",
            "start": "2008-01-15",
            "end": "2008-04-15",
            "benchmarks": [AR()],
        }
        raw = PseudoRealTimeBacktest(**options).run()
        scaled = PseudoRealTimeBacktest(**options, preprocess=scale).run()
        pd.testing.assert_series_equal(
            raw.forecasts["previous_actual"], scaled.forecasts["previous_actual"]
        )

    def test_real_time_store_uses_vintage_value(self, data) -> None:
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
        res = PseudoRealTimeBacktest(
            data=store.add(revised),
            target="gdp",
            start="2008-03-15",
            end="2008-03-15",
            benchmarks=[AR()],
        ).run()
        row = res.forecasts[res.forecasts["offset"] == 0].iloc[0]
        # 2008-03-15: 2007Q4 published on 2008-02-14, revised only on 2008-03-30
        assert row["previous_actual"] == pytest.approx(gdp[pd.Period("2007Q4")])
        final = res.directional_accuracy(horizon=None, previous="final", test=False)
        assert final["n_obs"].gt(0).all()

    def test_final_previous(self, run) -> None:
        frame = run.evaluable()
        actual = run.forecasts.groupby("target_period")["actual"].first()
        fda_final = run.directional_accuracy(horizon=None, previous="final", test=False)
        for model in run.models:
            rows = frame[frame["model"] == model]
            prev = actual.reindex(pd.PeriodIndex(rows["target_period"]) - 1).to_numpy()
            expected = directional_accuracy(rows["actual"], rows["forecast"], prev)
            assert fda_final.loc[(model, "all"), "fda"] == pytest.approx(expected)

    def test_parquet_round_trip(self, run, tmp_path) -> None:
        pytest.importorskip("pyarrow")
        back = pd.read_parquet(run.to_parquet(tmp_path / "bt.parquet"))
        assert list(back.columns) == list(FORECAST_COLUMNS)
        reread = BacktestResults(back, run.models, "gdp", reference="AR")
        pd.testing.assert_frame_equal(
            reread.metrics("kind", ("rmsfe", "fda")),
            run.metrics("kind", ("rmsfe", "fda")),
        )
        np.testing.assert_allclose(
            reread.directional_accuracy(previous="final")["fda"],
            run.directional_accuracy(previous="final")["fda"],
        )


# ---------------------------------------------------------------------- directional table
class TestDirectionalTable:
    def test_matches_functions(self, run) -> None:
        table = run.directional_accuracy("kind")
        assert table.index.names == ["model", "horizon"]
        assert table.columns.tolist() == [
            "fda",
            "n_obs",
            "statistic",
            "pvalue",
            "expected_hit_rate",
        ]
        frame = run.evaluable()
        rows = frame[(frame["model"] == "AR") & (frame["kind"] == "backcast")]
        args = (rows["actual"], rows["forecast"], rows["previous_actual"])
        assert table.loc[("AR", "backcast"), "fda"] == pytest.approx(directional_accuracy(*args))
        pt = pesaran_timmermann(*args)
        assert table.loc[("AR", "backcast"), "pvalue"] == pytest.approx(pt.pvalue)
        assert table.loc[("AR", "backcast"), "n_obs"] == len(rows)

    def test_metrics_with_fda(self, run) -> None:
        table = run.metrics("kind", ("rmsfe", "fda", "n"))
        directional = run.directional_accuracy("kind", test=False)
        for kind in ("backcast", "nowcast"):
            assert table.loc[kind, ("fda", "AR")] == pytest.approx(
                directional.loc[("AR", kind), "fda"]
            )
        assert run.rmsfe_by_horizon().equals(run.metrics(metrics=("rmsfe",))["rmsfe"])
        rel = run.relative_to("AR", horizon=None, metric="fda")
        assert rel["AR"].iloc[0] == 1.0

    def test_vintage_and_final_differ(self, run) -> None:
        vintage = run.directional_accuracy("months_to_end", test=False)
        final = run.directional_accuracy("months_to_end", test=False, previous="final")
        assert not np.allclose(vintage["fda"], final["fda"], equal_nan=True)

    def test_small_groups_give_nan_test(self, run) -> None:
        sub = dataclasses.replace(run, forecasts=run.forecasts.iloc[:6])
        table = sub.directional_accuracy(horizon="kind")
        assert table["statistic"].isna().all()
        assert table["fda"].notna().any()

    def test_degenerate_directions_are_silent(self, res) -> None:
        frame = res.forecasts.assign(forecast=res.forecasts["previous_actual"] + 1.0)
        flat = dataclasses.replace(res, forecasts=frame)
        table = flat.directional_accuracy(horizon=None)
        assert table["statistic"].isna().all()

    def test_older_tables_without_previous(self, res) -> None:
        old = dataclasses.replace(res, forecasts=res.forecasts.drop(columns="previous_actual"))
        with pytest.raises(ValueError, match="previous='final'"):
            old.metrics(metrics=("fda",))
        with pytest.raises(ValueError, match="previous='final'"):
            old.directional_accuracy()
        assert old.metrics(metrics=("rmsfe",)).equals(res.metrics(metrics=("rmsfe",)))
        assert old.to_frame(wide=True).columns.tolist() == ["A", "B", "actual"]
        # the previous value of the synthetic table *is* the final one
        pd.testing.assert_frame_equal(
            old.directional_accuracy(previous="final"), res.directional_accuracy()
        )

    def test_no_evaluable_rows(self, res) -> None:
        empty = dataclasses.replace(
            res, forecasts=res.forecasts.assign(actual=np.nan, error=np.nan)
        )
        assert empty.directional_accuracy(previous="final").empty

    def test_invalid_options(self, res) -> None:
        with pytest.raises(ValueError, match="previous"):
            res.metrics(metrics=("fda",), previous="first")
        with pytest.raises(ValueError, match="alternative"):
            res.directional_accuracy(alternative="up")
        with pytest.raises(ValueError, match="Unknown models"):
            res.directional_accuracy(models=["C"])

    def test_missing_horizon_values_are_kept(self, res) -> None:
        # like metrics(), rows with a NaN horizon (e.g. unknown release date) form a group
        unknown = res.forecasts["target_period"] < pd.Period("2020Q1", "Q")
        frame = res.forecasts.assign(days_to_release=np.where(unknown, np.nan, 60.0))
        sub = dataclasses.replace(res, forecasts=frame)
        table = sub.directional_accuracy("days_to_release", test=False)
        metrics = sub.metrics("days_to_release", ("fda",))
        assert len(table) == 4
        for model in sub.models:
            np.testing.assert_allclose(
                table.loc[model, "fda"].to_numpy(), metrics[("fda", model)].to_numpy()
            )

    def test_two_sided_alternative(self, res) -> None:
        greater = res.directional_accuracy(horizon=None)
        two = res.directional_accuracy(horizon=None, alternative="two-sided")
        np.testing.assert_allclose(two["pvalue"], np.minimum(2 * greater["pvalue"], 1.0))


# ---------------------------------------------------------------------- sub-periods
class TestPeriods:
    def test_metrics_equal_manual_filter(self, res) -> None:
        table = res.metrics("months_to_end", ("rmsfe", "mae", "fda", "n"), periods=PERIODS)
        assert table.index.names == ["period", "months_to_end"]
        assert table.index.get_level_values("period").unique().tolist() == list(PERIODS)
        for label, (first, last) in PERIODS.items():
            expected = manual(res, first, last).metrics(
                "months_to_end", ("rmsfe", "mae", "fda", "n")
            )
            got = table.xs(label, level="period")
            pd.testing.assert_frame_equal(got, expected, check_names=False)

    @pytest.mark.parametrize(
        ("method", "kwargs"),
        [
            ("rmsfe_by_horizon", {}),
            ("relative_to", {"horizon": "kind"}),
            ("diebold_mariano", {"horizon": None}),
            ("diebold_mariano", {"horizon": "kind", "aggregate": "target_period", "h": 2}),
            ("clark_west", {"horizon": "kind", "aggregate": "target_period"}),
            ("giacomini_white", {"horizon": None}),
            ("directional_accuracy", {"horizon": "kind"}),
        ],
    )
    def test_tables_equal_manual_filter(self, res, method, kwargs) -> None:
        table = getattr(res, method)(periods=PERIODS, **kwargs)
        assert table.index.names[0] == "period"
        for label, (first, last) in PERIODS.items():
            expected = getattr(manual(res, first, last), method)(**kwargs)
            got = table.xs(label, level="period")
            pd.testing.assert_frame_equal(got, expected, check_names=False)

    def test_mcs(self, res) -> None:
        out = res.mcs(n_bootstrap=50, periods=PERIODS, aggregate="target_period")
        assert isinstance(out, dict) and list(out) == list(PERIODS)
        assert all(isinstance(v, ModelConfidenceSetResult) for v in out.values())
        expected = manual(res, "2020Q1", "2021Q4").mcs(n_bootstrap=50, aggregate="target_period")
        assert isinstance(expected, ModelConfidenceSetResult)
        assert out["Covid"].included == expected.included
        pd.testing.assert_series_equal(out["Covid"].pvalues, expected.pvalues)
        by = res.mcs(horizon="kind", n_bootstrap=20, periods="ex-covid")
        assert list(by) == ["ex-Covid"] and isinstance(by["ex-Covid"], dict)

    def test_covid_shortcut(self, res) -> None:
        table = res.metrics(horizon=None, metrics=("n",), periods="covid")
        labels = table.index.get_level_values("period").tolist()
        assert labels == ["pre-Covid", "Covid", "post-Covid", "ex-Covid"]
        counts = table[("n", "A")].droplevel("all")
        assert counts["Covid"] == 8 * 3
        assert counts["ex-Covid"] == counts["pre-Covid"] + counts["post-Covid"] == 20 * 3
        ex = res.rmsfe_by_horizon(None, periods="EX-Covid")
        assert ex.index.get_level_values("period").unique().tolist() == ["ex-Covid"]

    def test_union_of_intervals(self, res) -> None:
        union = res.metrics(
            horizon=None, metrics=("n",), periods={"edges": [(None, "2017Q4"), ("2023Q1", None)]}
        )
        assert union.iloc[0, 0] == 8 * 3

    def test_flexible_bounds(self, res) -> None:
        by_month = res.metrics(horizon=None, metrics=("n",), periods={"c": ["2020-03", "2021-12"]})
        by_period = res.metrics(
            horizon=None,
            metrics=("n",),
            periods={"c": (pd.Period("2020-03", "M"), pd.Period("2021Q4", "Q"))},
        )
        assert by_month.iloc[0, 0] == by_period.iloc[0, 0] == 8 * 3

    def test_final_previous_crosses_the_boundary(self, res) -> None:
        sub = res.split_periods({"covid": ("2020Q1", "2021Q4")})["covid"]
        old = dataclasses.replace(sub, forecasts=sub.forecasts.drop(columns="previous_actual"))
        table = old.directional_accuracy(horizon=None, previous="final")
        assert table.loc[("A", "all"), "n_obs"] == 8 * 3  # 2020Q1 uses 2019Q4

    def test_empty_period_is_skipped(self, res) -> None:
        with pytest.warns(DataQualityWarning, match="'future'"):
            table = res.metrics(periods={"future": ("2030Q1", None), "all": (None, None)})
        assert table.index.get_level_values("period").unique().tolist() == ["all"]
        with (
            pytest.warns(DataQualityWarning),
            pytest.raises(ValueError, match="any of the sub-periods"),
        ):
            res.metrics(periods={"future": ("2030Q1", None)})

    @pytest.mark.parametrize(
        ("periods", "match"),
        [
            ("pandemic", "shortcut"),
            (["2020Q1", "2021Q4"], "mapping"),
            ({}, "at least one"),
            ({"x": "2020Q1"}, "pair"),
            ({"x": ("2020Q1", "2021Q4", "2022Q1")}, "pair"),
            ({"x": []}, "pair"),
            ({"x": ("2021Q4", "2020Q1")}, "ends before"),
        ],
    )
    def test_invalid_specifications(self, res, periods, match) -> None:
        with pytest.raises(ValueError, match=match):
            res.metrics(periods=periods)

    def test_string_target_periods(self, res) -> None:
        stored = res.forecasts.assign(target_period=res.forecasts["target_period"].astype(str))
        as_strings = dataclasses.replace(res, forecasts=stored)
        pd.testing.assert_frame_equal(
            as_strings.metrics("kind", ("rmsfe",), periods="covid"),
            res.metrics("kind", ("rmsfe",), periods="covid"),
        )


class TestCovidPeriods:
    def test_quarterly_and_monthly(self) -> None:
        quarterly = covid_periods()
        assert quarterly == {
            "pre-Covid": [(None, "2019Q4")],
            "Covid": [("2020Q1", "2021Q4")],
            "post-Covid": [("2022Q1", None)],
            "ex-Covid": [(None, "2019Q4"), ("2022Q1", None)],
        }
        assert covid_periods("M")["Covid"] == [("2020-03", "2021-12")]
        assert covid_periods("Y")["Covid"] == [("2020", "2021")]

    def test_custom_window(self) -> None:
        out = covid_periods("Q", window=("2020-04", "2020-09"))
        assert out["Covid"] == [("2020Q2", "2020Q3")]
        with pytest.raises(ValueError, match="ends before"):
            covid_periods("Q", window=("2021-01", "2020-01"))


def test_summary_unchanged_for_new_column(run) -> None:
    text = run.summary()
    assert "previous_actual" not in text
    assert not math.isnan(run.metrics(horizon=None).iloc[0, 0])
