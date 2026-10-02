"""Tests of the specification search (ECB toolbox parity, item 9)."""

from __future__ import annotations

import dataclasses
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd
import pytest

import nowcastbox as nb
from nowcastbox.benchmarks import AR
from nowcastbox.core.base import ParamsMixin
from nowcastbox.core.exceptions import DataQualityWarning
from nowcastbox.core.results import NowcastResults, build_nowcast_frame
from nowcastbox.models import MixedFreqDFM, TwoStepDFM
from nowcastbox.selection import (
    ParameterSpace,
    SearchResults,
    SpecificationSearch,
    SpecifiedModel,
    TreatmentPlan,
    preselect,
    weighted_score,
)

BACKTEST = {"start": "2018-10-15", "end": "2019-03-15", "benchmarks": [AR(p=1)]}
SPACE = {"n_factors": [1, 2], "n_series": (3, 10), "start": ["2000-01", "2006-01"]}
SCORE = {"rmsfe": 0.7, "fda": 0.3}
HORIZONS = {"nowcast": 0.5, "backcast": 0.25, "forecast": 0.25}


@pytest.fixture(scope="module")
def data():
    return nb.load_simulated_dfm().data


def make_search(data, **kwargs):
    options = {
        "model": TwoStepDFM,
        "data": data,
        "target": "gdp",
        "space": SPACE,
        "n_draws": 4,
        "ranking": "preselect",
        "backtest": BACKTEST,
        "score": SCORE,
        "horizon_weights": HORIZONS,
        "random_state": 0,
    }
    options.update(kwargs)
    return SpecificationSearch(**options)


@pytest.fixture(scope="module")
def result(data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return make_search(data).run()


# ====================================================================== run
class TestRun:
    def test_table(self, result):
        table = result.table()
        assert len(table) == 4
        assert table["rank"].tolist() == [1, 2, 3, 4]
        assert table.columns[:6].tolist() == [
            "rank",
            "draw",
            "score",
            "n_factors",
            "n_series",
            "start",
        ]
        assert bool(table["score"].is_monotonic_increasing)
        assert (table["status"] == "ok").all()
        assert (table["coverage"] == 1.0).all()
        assert set(table["draw"]) == set(result.specs)
        for column in ("rmsfe|nowcast", "fda|backcast", "mae|forecast", "ref|rmsfe|nowcast"):
            assert column in table

    def test_specs_respect_the_space(self, result):
        for spec in result.specs.values():
            assert spec["n_factors"] in (1, 2)
            assert 3 <= spec["n_series"] <= 10
            assert spec["start"] in ("2000-01", "2006-01")

    def test_score_is_the_weighted_score_of_the_criteria(self, result):
        expected = weighted_score(result.evaluations, SCORE, HORIZONS)
        pd.testing.assert_series_equal(result.scores(), expected)

    def test_criteria_are_the_backtest_metrics(self, result):
        draw = next(iter(result.specs))
        backtest = result.backtests[draw]
        assert backtest.models == ["specification", "AR"]
        table = backtest.metrics("kind", ("rmsfe", "fda"))
        row = result.evaluations.loc[draw]
        assert row["rmsfe|nowcast"] == pytest.approx(
            table.loc["nowcast", ("rmsfe", "specification")]
        )
        assert row["fda|backcast"] == pytest.approx(table.loc["backcast", ("fda", "specification")])
        assert row["ref|rmsfe|nowcast"] == pytest.approx(table.loc["nowcast", ("rmsfe", "AR")])

    def test_rescoring(self, result):
        by_fda = result.table({"fda": 1.0}, {"nowcast": 1.0})
        expected = weighted_score(result.evaluations, "fda", {"nowcast": 1.0})
        np.testing.assert_allclose(
            by_fda.set_index("draw")["score"].sort_index(), expected.sort_index()
        )
        relative = result.table(normalize="relative")
        assert (relative["score"] > 0).all()
        assert len(result.table(top=2)) == 2

    def test_best(self, result, data):
        table = result.table()
        best = result.best_spec()
        assert best == result.specs[int(table["draw"].iloc[0])]
        assert result.best_spec(2) == result.specs[int(table["draw"].iloc[1])]
        model = result.best_model()
        assert isinstance(model, SpecifiedModel)
        assert model.n_series == best["n_series"]
        assert model.start == best["start"]
        assert model.params == {"n_factors": best["n_factors"]}
        assert model.ranking == {}
        model.fit(data, "gdp")
        assert len(model.columns_) == best["n_series"]

    @pytest.mark.parametrize("rank", [0, 5])
    def test_best_out_of_range(self, result, rank):
        with pytest.raises(ValueError):
            result.best_spec(rank)

    def test_info_and_summary(self, result):
        info = result.info
        assert info["n_draws"] == 4
        assert info["n_unique"] + info["n_duplicates"] == 4
        assert info["n_evaluated"] == info["n_unique"]
        assert info["entropy"] == 0
        assert not info["grid"]
        text = result.summary()
        assert text.startswith("Specification search for gdp")
        assert "normalize: rank" in text
        assert str(result) == text

    def test_without_context(self, result):
        bare = dataclasses.replace(result, context=None)
        with pytest.raises(ValueError, match="no search context"):
            bare.best_model()
        with pytest.raises(ValueError, match="no search context"):
            bare.covid_robustness()

    def test_reproducible(self, data, result):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            again = make_search(data, n_draws=2).run()
        for draw, spec in again.specs.items():
            assert result.specs[draw] == spec
        columns = ["rmsfe|nowcast", "fda|nowcast", "coverage"]
        pd.testing.assert_frame_equal(
            again.evaluations[columns], result.evaluations.loc[list(again.specs), columns]
        )
        other = make_search(data, random_state=1).run()
        assert other.specs != result.specs

    def test_parallel_equals_sequential(self, data, result):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parallel = make_search(data, n_jobs=2).run()
        columns = ["rmsfe|nowcast", "fda|forecast", "status"]
        pd.testing.assert_frame_equal(parallel.evaluations[columns], result.evaluations[columns])


class TestDesigns:
    def test_exhaustive_grid(self, data):
        out = make_search(
            data,
            space={"n_factors": [1, 2], "factor_lags": 1},
            n_draws=None,
            ranking=None,
            backtest={"start": "2019-01-15", "end": "2019-03-15"},
            score="rmsfe",
            horizon_weights=None,
        ).run()
        assert out.info["grid"]
        assert out.info["entropy"] is None
        assert sorted(s["n_factors"] for s in out.specs.values()) == [1, 2]
        assert len(out.evaluations) == 2

    def test_duplicates_are_evaluated_once(self, data):
        out = make_search(
            data,
            space={"n_factors": [1]},
            n_draws=5,
            ranking=None,
            backtest={"start": "2019-01-15", "end": "2019-03-15"},
            score="rmsfe",
            horizon_weights=None,
        ).run()
        assert list(out.specs) == [0]
        assert out.info["n_duplicates"] == 4

    def test_failed_specification(self, data):
        out = make_search(
            data,
            space={"n_factors": [1, 4], "n_series": [2]},
            n_draws=None,
            ranking=["x01", "x02", "x03"],
            backtest={"start": "2019-01-15", "end": "2019-03-15"},
            score="rmsfe",
            horizon_weights=None,
        ).run()
        table = out.table()
        assert table["status"].tolist() == ["ok", "failed"]
        assert np.isinf(table["score"].iloc[1])
        assert table["n_failed_fits"].iloc[1] > 0
        assert table["error"].iloc[1]
        with pytest.raises(ValueError, match="No valid specification at rank 2"):
            out.best_spec(2)

    def test_incomplete_specification_is_invalid(self, data, result):
        evaluations = result.evaluations.copy()
        draw = evaluations.index[0]
        evaluations.loc[draw, "coverage"] = 0.5
        degraded = dataclasses.replace(result, evaluations=evaluations)
        assert np.isinf(degraded.scores().loc[draw])
        assert np.isfinite(degraded.scores().drop(draw)).all()

    def test_formula_target_and_static_ranking(self, data):
        with pytest.warns(DataQualityWarning, match="looks ahead"):
            out = make_search(
                data,
                target="gdp ~ x01 + x02 + x03 + x04",
                space={"n_series": (1, 4)},
                n_draws=2,
                ranking=preselect(data, "gdp ~ x01 + x02 + x03 + x04", methods=("sis",)),
                backtest={"start": "2019-01-15", "end": "2019-03-15"},
                score="rmsfe",
                horizon_weights=None,
            ).run()
        assert out.context.ranking[0] in ("x01", "x02", "x03", "x04")
        assert len(out.context.ranking) == 4

    def test_default_backtest_window(self, data):
        search = make_search(
            data,
            space={"n_factors": [1]},
            n_draws=None,
            ranking=None,
            backtest={"n_vintages": 2},
            score="rmsfe",
            horizon_weights=None,
        )
        out = search.run()
        kwargs = out.context.backtest
        assert kwargs["end"] == pd.Timestamp("2019-09-30")  # last observed quarter
        assert kwargs["start"] == pd.Timestamp("2019-08-31")
        assert kwargs["errors"] == "warn"

    def test_month_horizons_and_mapping_periods(self, data):
        out = make_search(
            data,
            space={"n_factors": [1]},
            n_draws=None,
            ranking=None,
            backtest={"start": "2018-10-15", "end": "2019-03-15", "target_offsets": (0,)},
            score="rmsfe",
            horizon="months_to_end",
            horizon_weights={0: 1.0, 1: 1.0, 2: 1.0},
            periods={"late": ("2019Q1", None)},
        ).run()
        row = out.evaluations.iloc[0]
        assert {"rmsfe|0", "rmsfe|1", "rmsfe|2"} <= set(out.evaluations.columns)
        assert row["n_forecasts"] == 3  # only the 2019Q1 nowcasts
        assert np.isfinite(out.scores().iloc[0])

    @pytest.mark.parametrize("periods", [("2019Q1", "2019Q1"), [("2019Q1", None)], "ex-covid"])
    def test_periods(self, data, periods):
        out = make_search(
            data,
            space={"n_factors": [1]},
            n_draws=None,
            ranking=None,
            backtest={"start": "2018-12-15", "end": "2019-03-15"},
            score="rmsfe",
            horizon_weights={"nowcast": 1},
            periods=periods,
        ).run()
        assert out.evaluations["status"].iloc[0] == "ok"

    def test_empty_evaluation_window(self, data):
        out = make_search(
            data,
            space={"n_factors": [1]},
            n_draws=None,
            ranking=None,
            backtest={"start": "2019-01-15", "end": "2019-03-15"},
            score="rmsfe",
            horizon_weights=None,
            periods=("2025Q1", None),
        ).run()
        row = out.evaluations.iloc[0]
        assert row["status"] == "failed"
        assert "No forecast" in row["error"]


class _Shrunk(ParamsMixin):
    """A model without ``_validate_params`` (forecasts the scaled target mean)."""

    def __init__(self, scale=1.0):
        self.scale = scale

    def fit(self, data, target):
        y = data.to_native(target)
        estimate = pd.Series(self.scale * y.mean(), index=y.index)
        return NowcastResults(target=target, nowcast=build_nowcast_frame(y, estimate))


class TestOtherModelsAndPreprocessing:
    def test_model_without_validation_hook(self, data):
        out = make_search(
            data,
            model=_Shrunk(),
            space={"scale": [0.5, 1.0], "start": [None, "2010-01"]},
            n_draws=None,
            ranking=None,
            backtest={"start": "2019-01-15", "end": "2019-03-15"},
            score="rmsfe",
            horizon_weights=None,
        ).run()
        assert len(out.evaluations) == 4
        assert (out.evaluations["status"] == "ok").all()

    def test_user_preprocess_is_composed_with_the_treatment(self, data):
        calls = []

        def spy(panel):
            calls.append(panel.n_periods)
            return panel

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = make_search(
                data,
                space={"n_factors": [1]},
                n_draws=None,
                ranking=None,
                backtest={"start": "2019-02-15", "end": "2019-03-15", "preprocess": spy},
                score="rmsfe",
                horizon_weights=None,
            ).run()
            n_search = len(calls)
            rob = out.covid_robustness(top=1, treatments=("outliers",))
        assert n_search == 2
        assert len(calls) == 4
        model = rob.best_model()
        assert "OutlierCorrection" in repr(model.preprocess)
        assert model.preprocess.first is spy
        assert out.best_model().preprocess is spy
        model.fit(data, "gdp")
        assert len(calls) == 5

    def test_criterion_missing_for_every_valid_row_is_dropped(self, result):
        evaluations = result.evaluations.assign(**{"rmsfe|later": np.nan})
        other = dataclasses.replace(
            result, evaluations=evaluations, score_weights={"rmsfe": 1.0}, horizon_weights=None
        )
        scores = other.scores()
        assert np.isfinite(scores).all()
        expected = dataclasses.replace(other, evaluations=result.evaluations).scores()
        pd.testing.assert_series_equal(scores, expected)

    def test_empty_evaluations(self, result):
        empty = dataclasses.replace(result, evaluations=result.evaluations.iloc[:0])
        assert empty.scores().empty
        assert empty.table().empty


# ====================================================================== look-ahead
class TestNoLookAhead:
    def test_future_data_do_not_change_the_evaluation(self, data):
        """Scrambling every predictor value released after the last vintage changes nothing."""
        frame = data.to_frame()
        rng = np.random.default_rng(0)
        future = frame.index > pd.Period("2019-03", freq="M")
        predictors = [c for c in frame.columns if c != "gdp"]
        frame.loc[future, predictors] = rng.normal(size=(int(future.sum()), len(predictors)))
        scrambled = data.with_data(frame)
        options = {
            "n_draws": 2,
            "ranking": {"methods": ("sis", "lars")},
            "backtest": {"start": "2019-01-15", "end": "2019-03-15"},
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            base = make_search(data, **options).run()
            other = make_search(scrambled, **options).run()
        columns = [c for c in base.evaluations.columns if "|" in c]
        pd.testing.assert_frame_equal(base.evaluations[columns], other.evaluations[columns])

    def test_per_vintage_ranking_uses_only_the_vintage(self, data, monkeypatch):
        """Every per-vintage pre-selection sees exactly one pseudo real-time information set."""
        from nowcastbox.selection import _specified as specified_module
        from nowcastbox.vintages import pseudo_real_time

        seen = []
        original = specified_module.preselect

        def spy(panel, target, **options):
            seen.append(panel)
            return original(panel, target, **options)

        monkeypatch.setattr(specified_module, "preselect", spy)
        monkeypatch.setattr(specified_module, "_preselect_cache", OrderedDict())
        dates = pd.date_range("2019-01-15", "2019-03-15", freq=pd.DateOffset(months=1))
        options = {
            "space": {"n_series": [4]},
            "n_draws": None,
            "ranking": {"methods": ("sis",)},
            "backtest": {"start": dates[0], "end": dates[-1]},
            "score": "rmsfe",
            "horizon_weights": None,
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            make_search(data, **options).run()
        assert len(seen) == len(dates)
        for panel, date in zip(seen, dates, strict=True):
            vintage = pseudo_real_time(data, vintage=date).to_frame()
            frame = panel.to_frame()
            common = frame.index.intersection(vintage.index)
            pd.testing.assert_frame_equal(
                frame.loc[common], vintage.loc[common, frame.columns], check_freq=False
            )

    def test_fixed_ranking_after_the_first_vintage_warns(self, data):
        late = preselect(data, "gdp", methods=("sis",), as_of="2019-06-30")
        early = preselect(data, "gdp", methods=("sis",), as_of="2018-06-30")
        options = {"space": {"n_series": [3]}, "n_draws": None, "horizon_weights": None}
        with pytest.warns(DataQualityWarning, match="looks ahead"):
            make_search(data, ranking=late, **options)._setup()
        with warnings.catch_warnings():
            warnings.simplefilter("error", DataQualityWarning)
            make_search(data, ranking=early, **options)._setup()


# ====================================================================== checkpoint
class TestCheckpoint:
    def test_resume(self, data, tmp_path):
        path = tmp_path / "search.parquet"
        options = {
            "space": {"n_factors": [1, 2, 3], "factor_lags": [1, 2]},
            "ranking": None,
            "backtest": {"start": "2019-01-15", "end": "2019-03-15"},
            "score": "rmsfe",
            "horizon_weights": None,
            "checkpoint": path,
        }
        first = make_search(data, n_draws=2, **options).run()
        assert path.exists()
        assert first.info["n_resumed"] == 0
        second = make_search(data, n_draws=4, **options).run()
        assert second.info["n_resumed"] == first.info["n_unique"]
        assert second.info["n_evaluated"] == second.info["n_unique"] - first.info["n_unique"]
        assert set(second.backtests).isdisjoint(first.specs)
        columns = ["rmsfe|nowcast", "status", "coverage"]
        pd.testing.assert_frame_equal(
            second.evaluations.loc[list(first.specs), columns],
            first.evaluations[columns],
            check_dtype=False,
        )
        third = make_search(data, n_draws=4, **options).run()
        assert third.info["n_evaluated"] == 0
        assert third.backtests == {}
        pd.testing.assert_frame_equal(
            third.table().drop(columns="seconds"),
            second.table().drop(columns="seconds"),
            check_dtype=False,
        )

    def test_other_settings_refused(self, data, tmp_path):
        path = tmp_path / "search.csv"
        options = {
            "space": {"n_factors": [1]},
            "n_draws": None,
            "ranking": None,
            "score": "rmsfe",
            "horizon_weights": None,
            "checkpoint": path,
        }
        make_search(data, backtest={"start": "2019-02-15", "end": "2019-03-15"}, **options).run()
        changed = make_search(
            data, backtest={"start": "2019-01-15", "end": "2019-03-15"}, **options
        )
        with pytest.raises(ValueError, match="different settings"):
            changed.run()

    def test_metadata_enters_the_fingerprint(self, data):
        options = {"space": {"n_factors": [1]}, "n_draws": None, "ranking": None}
        base = make_search(data, **options)._setup().info["fingerprint"]
        delayed = data.with_metadata("x01", release_delay=90)
        other = make_search(delayed, **options)._setup().info["fingerprint"]
        assert base != other
        assert make_search(data, **options)._setup().info["fingerprint"] == base

    def test_seed_sequence_draws_are_reproducible(self, data):
        seed = np.random.SeedSequence(3)
        search = make_search(data, random_state=seed, n_draws=6)
        first = [t.spec for t in search._setup().tasks]
        assert [t.spec for t in search._setup().tasks] == first


# ====================================================================== validation
class TestValidation:
    @pytest.mark.parametrize(
        ("kwargs", "error", "message"),
        [
            ({"space": {"n_factorz": [1]}}, ValueError, "neither parameters"),
            ({"space": {"n_factors": [0, 1]}}, ValueError, "Invalid value 0 for 'n_factors'"),
            ({"space": {"n_factors": []}}, ValueError, "has no values"),
            ({"space": {"n_factors": (3, 1)}}, ValueError, "low > high"),
            ({"space": {"n_series": (1, 3)}, "ranking": None}, ValueError, "needs a ranking"),
            ({"space": {"n_series": (1, 30)}}, ValueError, r"must lie in \[1, 20\]"),
            ({"space": {"n_series": (0, 3)}}, ValueError, "must lie in"),
            ({"space": {"n_series": [1.5]}}, ValueError, "must be integers"),
            ({"space": {"start": ["2005-13"]}}, ValueError, "Invalid sample start"),
            ({"space": {"start": ["2030-01"]}}, ValueError, "after the end"),
            ({"n_draws": 0}, ValueError, "n_draws must be an integer"),
            ({"normalize": "z"}, ValueError, "normalize must be one of"),
            ({"max_missing": 1.0}, ValueError, "max_missing must lie"),
            ({"horizon": ""}, ValueError, "horizon must be a column"),
            ({"score": {"rmse": 1}}, ValueError, "Unknown score metric"),
            ({"horizon_weights": {}}, ValueError, "non-empty mapping"),
            ({"metrics": ("crps",)}, ValueError, "Unknown metrics"),
            ({"backtest": {"model": AR()}}, ValueError, "cannot set"),
            ({"backtest": {"strat": "2019-01-01"}}, ValueError, "Invalid backtest settings"),
            ({"ranking": "lasso"}, ValueError, "must be 'preselect'"),
            ({"ranking": {"top": 3}}, ValueError, "Invalid preselect options"),
            ({"ranking": ["x01", "zz"]}, ValueError, "not in the data"),
            ({"ranking": [1, 2]}, TypeError, "sequence of series names"),
            ({"model": 3}, TypeError, "must be a nowcaster"),
            ({"periods": "covid"}, ValueError, "periods must be"),
            (
                {"periods": {"a": ("2019Q1", None), "b": ("2018Q1", None)}},
                ValueError,
                "exactly one",
            ),
        ],
    )
    def test_invalid(self, data, kwargs, error, message):
        with pytest.raises(error, match=message):
            make_search(data, **kwargs).run()

    def test_grid_too_large(self, data):
        search = make_search(data, space={"n_factors": (1, 1000), "factor_lags": (1, 1000)})
        search.n_draws = None
        with pytest.raises(ValueError, match="use n_draws"):
            search.run()

    def test_parameter_space_instance(self, data):
        space = ParameterSpace({"n_factors": [1, 2]})
        setup = make_search(data, space=space, n_draws=None, ranking=None)._setup()
        assert setup.space.names == ["n_factors"]
        assert len(setup.tasks) == 2


# ====================================================================== robustness
class TestCovidRobustness:
    def test_two_step_dfm(self, result, data):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rob = result.covid_robustness(top=1, evaluate_from="2019Q1")
        best_draw = int(result.table()["draw"].iloc[0])
        table = rob.table()
        assert len(table) == 4
        assert set(table["draw"]) == {best_draw}
        status = dict(zip(table["treatment"], table["status"], strict=True))
        assert status == {"none": "ok", "outliers": "ok", "dummy": "skipped", "mask": "skipped"}
        assert np.isinf(table.loc[table["status"] == "skipped", "score"]).all()
        assert rob.treatments == ("none", "dummy", "mask", "outliers")
        pivot = rob.pivot()
        assert pivot.columns.tolist() == ["none", "dummy", "mask", "outliers"]
        assert pivot[["dummy", "mask"]].isna().all().all()
        best = rob.best()
        assert best["treatment"] in ("none", "outliers")
        model = rob.best_model()
        assert isinstance(model, SpecifiedModel)
        assert model.n_series == result.specs[best_draw]["n_series"]
        assert rob.periods == {"from 2019Q1": ("2019Q1", None)}
        backtest = rob.backtests[(best_draw, "none")]
        assert set(backtest.forecasts["target_period"].astype(str)) >= {"2019Q1"}
        assert (table.loc[table["status"] == "ok", "n_forecasts"] < 18).all()
        text = rob.summary()
        assert "has no 'covid' option" in text
        assert str(rob) == text
        assert rob.scores({"rmsfe": 1}, {"nowcast": 1}).notna().all()

    def test_custom_treatment_and_overrides(self, result):
        def nothing(model, target):
            return TreatmentPlan(note="custom")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rob = result.covid_robustness(
                top=2,
                treatments={"plain": "none", "mine": nothing},
                backtest={"start": "2019-01-15"},
                score="rmsfe",
                horizon_weights={"nowcast": 1},
                normalize="none",
                n_jobs=1,
            )
        assert len(rob.evaluations) == 4
        assert rob.normalize == "none"
        pivot = rob.pivot()
        np.testing.assert_allclose(pivot["plain"], pivot["mine"])
        assert rob.table()["score"].iloc[0] == pytest.approx(rob.evaluations["rmsfe|nowcast"].min())

    def test_errors(self, result):
        with pytest.raises(ValueError, match="top must be"):
            result.covid_robustness(top=0)
        with pytest.raises(ValueError, match="could be evaluated"):
            result.covid_robustness(top=1, treatments=("none",), evaluate_from="2030Q1")
        failed = result.evaluations.assign(status="failed")
        with pytest.raises(ValueError, match="finite score"):
            dataclasses.replace(result, evaluations=failed).covid_robustness()

    def test_best_without_valid_pair(self, result):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rob = result.covid_robustness(top=1, treatments=("none", "dummy"))
        broken = dataclasses.replace(rob, evaluations=rob.evaluations.assign(status="failed"))
        with pytest.raises(ValueError, match="finite score"):
            broken.best()

    def test_mixed_freq_dfm_treatments(self, data):
        panel = data.select(["gdp", "x10", "x19", "x12", "x04"])
        model = MixedFreqDFM(max_iter=10, covid_window=("2018-01", "2018-06"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = SpecificationSearch(
                model,
                panel,
                "gdp",
                {"n_factors": [1]},
                n_draws=None,
                backtest={"start": "2018-11-15", "end": "2019-03-15", "refit_every": 2},
                score="rmsfe",
            ).run()
            rob = out.covid_robustness(top=1)
        table = rob.table()
        assert (table["status"] == "ok").all()
        assert len(table) == 4
        assert rob.best_model().params["covid"] in ("none", "dummy", "mask")
        dummy = rob.models[(0, "dummy")]
        assert dummy.params == {"n_factors": 1, "covid": "dummy", "outliers": "none"}
        assert rob.models[(0, "outliers")].params["outliers"] == "auto"
        assert isinstance(out, SearchResults)


@pytest.mark.slow
def test_end_to_end_mixed_freq_dfm(data, tmp_path):
    """The ECB workflow: pre-selection funnel, random search, Covid robustness."""
    model = MixedFreqDFM(max_iter=30, covid_window=("2017-01", "2017-06"))
    search = SpecificationSearch(
        model=model,
        data=data,
        target="gdp",
        space={
            "n_factors": [1, 2],
            "factor_lags": [1, 2],
            "n_series": (4, 10),
            "start": ["2000-01", "2004-01"],
        },
        n_draws=6,
        ranking={"methods": ("tstat", "sis", "lars")},
        backtest={
            "start": "2016-01-15",
            "end": "2019-06-15",
            "refit_every": 3,
            "benchmarks": [AR(p=1)],
        },
        score=SCORE,
        horizon_weights=HORIZONS,
        random_state=42,
        n_jobs=2,
        checkpoint=tmp_path / "search.parquet",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = search.run()
        rob = out.covid_robustness(top=2, evaluate_from="2017Q3")
    assert (out.table()["status"] == "ok").all()
    assert np.isfinite(out.table()["score"]).all()
    assert len(rob.table()) == 8
    assert (rob.table()["status"] == "ok").all()
    fitted = rob.best_model().fit(data, "gdp")
    assert np.isfinite(fitted.get_nowcast())
    resumed = search.run()
    assert resumed.info["n_evaluated"] == 0
