"""Tests for nowcastbox.selection.blocks (block/variable selection, innovation I7)."""

from __future__ import annotations

import warnings
from typing import Any, ClassVar

import matplotlib
import numpy as np
import pandas as pd
import pytest

from nowcastbox.core.base import BaseNowcaster
from nowcastbox.core.data import MixedFrequencyData
from nowcastbox.core.results import NowcastResults, build_nowcast_frame
from nowcastbox.models import MixedFreqDFM, TwoStepDFM
from nowcastbox.models.two_step import simulate_two_step_example
from nowcastbox.selection import (
    SCORINGS,
    SelectionPath,
    ValidationSettings,
    select_blocks,
    select_variables,
)
from nowcastbox.selection import blocks as blocks_module

matplotlib.use("Agg")

REAL = [f"x{i}" for i in range(1, 5)]
SOFT = [f"x{i}" for i in range(5, 9)]
NOISE = ["noise1", "noise2", "noise3"]
GROUPS = {"real": REAL, "soft": SOFT, "noise": NOISE}
FAST = {"n_vintages": 6, "refit_every": 3}


def make_panel(n_periods: int = 120, seed: int = 0, noise_scale: float = 3.0) -> MixedFrequencyData:
    data = simulate_two_step_example(n_periods=n_periods, n_series=8, random_state=seed)
    frame = data.to_frame()
    frame[NOISE] = noise_scale * np.random.default_rng(seed + 1).normal(size=(n_periods, 3))
    freqs = {**data.frequencies.to_dict(), **dict.fromkeys(NOISE, "M")}
    delays = dict.fromkeys(frame.columns, 20) | {"gdp": 45}
    blocks = {**dict.fromkeys(REAL, ("global", "real")), **dict.fromkeys(SOFT, ("global", "soft"))}
    blocks |= dict.fromkeys(NOISE, ("global", "noise"))
    return MixedFrequencyData(frame, freqs, release_delays=delays, blocks=blocks)


@pytest.fixture(scope="module")
def panel() -> MixedFrequencyData:
    return make_panel()


def quiet(func, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return func(*args, **kwargs)


@pytest.fixture(scope="module")
def forward(panel) -> SelectionPath:
    return quiet(select_blocks, TwoStepDFM(n_factors=1), panel, "gdp", GROUPS, validation=FAST)


# ---------------------------------------------------------------------- RMSFE search
class TestForwardRMSFE:
    def test_noise_block_excluded(self, forward):
        assert forward.chosen == ("real", "soft")
        assert forward.selected_series == REAL + SOFT
        assert forward.kind == "blocks" and forward.direction == "forward"
        assert forward.scoring == "rmsfe"

    def test_path_structure(self, forward):
        path = forward.path
        assert path["action"].tolist() == ["start", "add", "add"]
        assert np.isinf(path["score"].iloc[0])
        assert path["score"].iloc[-1] == pytest.approx(forward.score)
        assert np.isnan(path["improvement"].iloc[1])  # from an empty start
        assert path["improvement"].iloc[2] > 0
        assert forward.scores.index.tolist() == [0, 1, 2]
        assert forward.to_frame().equals(path)

    def test_evaluations_and_cache(self, forward):
        ev = forward.evaluations
        # start (empty, not fitted) + 3 singles + 2 pairs + 1 triple
        assert len(ev) == 7
        assert forward.info["n_evaluations"] == 6
        assert (ev.loc[ev["step"] > 0, "coverage"] == 1.0).all()
        full = ev[ev["n_groups"] == 3]
        assert full["score"].iloc[0] > forward.score
        noise_alone = ev[(ev["step"] == 1) & (ev["group"] == "noise")]["score"].iloc[0]
        assert noise_alone > 3 * forward.score
        assert forward.info["vintage_range"][0] < forward.info["vintage_range"][1]

    def test_scores_match_direct_backtest(self, panel, forward):
        """The score of a set equals the RMSFE of the backtest of the restricted model."""
        from nowcastbox.evaluation import PseudoRealTimeBacktest

        settings = ValidationSettings(**FAST)
        start, end = settings.dates(panel, "gdp")

        class Restricted(TwoStepDFM):
            def fit(self, data, target, **kw):
                return super().fit(data.select(["gdp", *REAL]), target, **kw)

        bt = PseudoRealTimeBacktest(
            Restricted(n_factors=1),
            panel,
            "gdp",
            start=start,
            end=end,
            refit_every=3,
            target_offsets=(-1, 0),
        )
        res = quiet(bt.run)
        rmsfe = float(res.metrics(horizon=None, metrics=("rmsfe",)).iloc[0, 0])
        ev = forward.evaluations
        real_score = ev[(ev["step"] == 1) & (ev["group"] == "real")]["score"].iloc[0]
        assert real_score == pytest.approx(rmsfe, rel=1e-10)

    def test_summary_and_plot(self, forward):
        text = forward.summary()
        assert "Blocks selection (forward, rmsfe)" in text
        assert "Selected (2): real, soft" in text
        assert str(forward) == text
        ax = forward.plot()
        assert ax.get_xlabel() == "step"
        fig = forward.plot(backend="plotly")
        assert fig.layout.yaxis.title.text == "rmsfe"

    def test_select(self, panel, forward):
        sub = forward.select(panel)
        assert sub.columns == ["gdp", *REAL, *SOFT]


def test_backward_removes_noise(panel):
    path = quiet(
        select_blocks,
        TwoStepDFM(n_factors=1),
        panel,
        "gdp",
        GROUPS,
        direction="backward",
        validation=FAST,
    )
    assert "noise" not in path.chosen
    assert path.path["action"].iloc[1] == "remove"
    assert path.path["group"].iloc[1] == "noise"
    assert path.path["score"].is_monotonic_decreasing


def test_mae_scoring_and_min_improvement(panel):
    path = quiet(
        select_blocks,
        TwoStepDFM(n_factors=1),
        panel,
        "gdp",
        GROUPS,
        scoring="mae",
        min_improvement=10.0,
        validation=FAST,
    )
    # the first finite score is always accepted, then no step improves by 10
    assert len(path.chosen) == 1
    assert path.scoring == "mae"


def test_max_blocks_and_always_include(panel):
    path = quiet(
        select_blocks,
        TwoStepDFM(n_factors=1),
        panel,
        "gdp",
        GROUPS,
        always_include=["noise"],
        max_blocks=2,
        validation=FAST,
    )
    assert "noise" in path.chosen and len(path.chosen) == 2
    assert path.always_include == ("noise",)
    assert path.path["action"].iloc[0] == "start" and path.path["n_groups"].iloc[0] == 1


def test_blocks_from_metadata(panel):
    path = quiet(
        select_blocks,
        TwoStepDFM(n_factors=1),
        panel,
        "gdp",
        ["real", "soft", "noise"],
        validation=FAST,
    )
    assert set(path.groups) == {"real", "soft", "noise"}
    assert path.chosen == ("real", "soft")
    everything = quiet(
        select_blocks, TwoStepDFM(n_factors=1), panel, "gdp", scoring="bic", max_blocks=1
    )
    assert set(everything.groups) == {"global", "real", "soft", "noise"}


def test_blocks_from_dataframe_and_fixed_series(panel):
    frame = pd.DataFrame(
        {"soft": [s in SOFT for s in panel.columns], "noise": [s in NOISE for s in panel.columns]},
        index=panel.columns,
    )
    path = quiet(
        select_blocks,
        TwoStepDFM(n_factors=1),
        panel,
        "gdp",
        frame,
        scoring="bic",
    )
    assert path.fixed == tuple(REAL)
    assert path.selected_series[:4] == REAL
    assert "Always included series: x1, x2, x3, x4" in path.summary()


def test_failing_candidates_score_inf(panel):
    """A group too small for two factors fails at every vintage -> inf, never chosen."""
    groups = {"tiny": ["x1"], "real": REAL[1:]}
    path = quiet(
        select_blocks,
        TwoStepDFM(n_factors=2),
        panel.select(["gdp", *REAL]),
        "gdp",
        groups,
        validation=FAST,
    )
    ev = path.evaluations
    tiny = ev[(ev["step"] == 1) & (ev["group"] == "tiny")]
    assert np.isinf(tiny["score"].iloc[0]) and tiny["coverage"].iloc[0] == 0.0
    assert path.path["group"].iloc[1] == "real"
    assert path.info["n_failures"] > 0


def test_formula_restricts_candidates(panel):
    path = quiet(
        select_variables,
        TwoStepDFM(n_factors=1),
        panel,
        "gdp ~ x1 + x2 + x3 + noise1",
        scoring="bic",
    )
    assert set(path.groups) == {"x1", "x2", "x3", "noise1"}
    assert path.series_order == ("x1", "x2", "x3", "noise1")


def test_dataframe_input(panel):
    frame = panel.to_frame()[["gdp", *REAL]]
    path = quiet(
        select_variables,
        TwoStepDFM(n_factors=1),
        frame,
        "gdp",
        scoring="aic",
        frequency={**dict.fromkeys(REAL, "M"), "gdp": "Q"},
    )
    assert path.target == "gdp" and len(path.chosen) >= 1


# ---------------------------------------------------------------------- variables / IC
class TestVariables:
    @pytest.mark.parametrize("scoring", ["aic", "bic", "hq"])
    def test_information_criteria_drop_noise(self, panel, scoring):
        path = quiet(
            select_variables,
            TwoStepDFM(n_factors=1),
            panel,
            "gdp",
            scoring=scoring,
        )
        assert path.kind == "variables" and path.scoring == scoring
        assert not set(path.chosen) & set(NOISE)
        ev = path.evaluations
        assert (ev.loc[ev["step"] > 0, "coverage"] == 1.0).all()

    def test_candidate_list_and_groups(self, panel):
        listed = quiet(
            select_variables,
            TwoStepDFM(n_factors=1),
            panel,
            "gdp",
            ["noise1", "noise2"],
            scoring="bic",
        )
        assert set(listed.groups) == {"noise1", "noise2"}
        assert set(listed.fixed) == set(REAL + SOFT + ["noise3"])
        grouped = quiet(
            select_variables,
            TwoStepDFM(n_factors=1),
            panel,
            "gdp",
            {"a": REAL, "b": "x5"},
            scoring="bic",
            always_include=["b"],
            max_variables=1,
        )
        assert grouped.groups["b"] == ("x5",)
        assert grouped.chosen == ("b",)

    def test_backward_ic_on_two_groups(self, panel):
        path = quiet(
            select_variables,
            TwoStepDFM(n_factors=1),
            panel,
            "gdp",
            {"good": REAL + SOFT, "bad": NOISE},
            scoring="bic",
            direction="backward",
        )
        assert path.chosen == ("good",)

    def test_information_criterion_helper(self):
        observed = pd.Series([1.0, 2.0, 3.0, 4.0])
        assert np.isinf(blocks_module._information_criterion(observed, None, 1, "bic"))
        short = pd.Series([1.0, 2.0], index=[0, 1])
        assert np.isinf(blocks_module._information_criterion(observed, short, 1, "bic"))
        fitted = observed + np.array([0.1, -0.1, 0.1, -0.1])
        aic = blocks_module._information_criterion(observed, fitted, 2, "aic")
        assert aic == pytest.approx(4 * np.log(0.01) + 4)
        exact = blocks_module._information_criterion(observed, observed, 0, "hq")
        assert np.isfinite(exact)


# ---------------------------------------------------------------------- other models
class _MeanModel(BaseNowcaster):
    """Toy nowcaster: target = mean of the predictors' quarterly averages (+ init log)."""

    calls: ClassVar[list[Any]] = []

    def __init__(self, init: Any = None, fail_warm: bool = False) -> None:
        self.init = init
        self.fail_warm = fail_warm

    def _validate_params(self) -> None:
        return None

    def _fit(self, data, target, **fit_kwargs) -> NowcastResults:
        _MeanModel.calls.append(self.init is not None)
        if self.fail_warm and self.init is not None:
            raise RuntimeError("warm start failed")
        frame = data.to_frame()
        preds = frame.drop(columns=target).mean(axis=1).ffill()
        q = preds.groupby(preds.index.asfreq("Q")).mean()
        y = data.to_native(target)
        observed = y.reindex(q.index)
        return NowcastResults(
            target=target,
            nowcast=build_nowcast_frame(observed, q),
            model_name="Mean",
            model_params=self.get_params(),
            data=data,
            info={"fit_kwargs": fit_kwargs},
        )


class TestWarmStarts:
    def test_warm_start_used_between_refits(self, panel):
        _MeanModel.calls = []
        path = quiet(
            select_blocks,
            _MeanModel(),
            panel,
            "gdp",
            {"real": REAL},
            validation={"n_vintages": 4, "refit_every": 2},
        )
        # no update/predict -> every vintage is a full fit; refits after the first warm
        assert _MeanModel.calls[0] is False
        assert any(_MeanModel.calls)
        assert path.chosen == ("real",)

    def test_failed_warm_start_falls_back(self, panel):
        _MeanModel.calls = []
        path = quiet(
            select_blocks,
            _MeanModel(fail_warm=True),
            panel,
            "gdp",
            {"real": REAL},
            validation={"n_vintages": 3, "refit_every": 1, "fit_kwargs": {}},
        )
        assert np.isfinite(path.score)
        assert True in _MeanModel.calls

    def test_warm_start_disabled(self, panel):
        _MeanModel.calls = []
        quiet(
            select_blocks,
            _MeanModel(),
            panel,
            "gdp",
            {"real": REAL},
            validation={"n_vintages": 3, "refit_every": 1, "warm_start": False},
        )
        assert not any(_MeanModel.calls)

    def test_em_model_predict_update_path(self):
        data = make_panel(n_periods=96, seed=2)
        sub = data.select(["gdp", *REAL, "noise1"])
        path = quiet(
            select_variables,
            MixedFreqDFM(max_iter=5),
            sub,
            "gdp",
            {"real": REAL, "noise": ["noise1"]},
            validation={"n_vintages": 3, "refit_every": 2},
        )
        assert path.chosen[0] == "real"
        assert np.isfinite(path.score)

    def test_em_information_criterion_uses_predict(self):
        data = make_panel(n_periods=96, seed=2).select(["gdp", *REAL])
        path = quiet(
            select_variables,
            MixedFreqDFM(max_iter=5),
            data,
            "gdp",
            {"a": REAL[:2], "b": REAL[2:]},
            scoring="bic",
        )
        assert np.isfinite(path.score)


def test_benchmark_protocol_model(panel):
    from nowcastbox.benchmarks import BridgeBenchmark

    path = quiet(
        select_variables,
        BridgeBenchmark(),
        panel.select(["gdp", "x1", "noise1"]),
        "gdp",
        validation=FAST,
    )
    assert np.isfinite(path.score) and path.chosen
    assert (path.evaluations.loc[path.evaluations["step"] > 0, "coverage"] == 1.0).all()


# ---------------------------------------------------------------------- settings
class TestValidationSettings:
    def test_default_dates(self, panel):
        start, end = ValidationSettings(n_vintages=3).dates(panel, "gdp")
        last = panel.to_native("gdp", dropna=True).index[-1]
        assert end == last.end_time.normalize()
        assert pd.Period(start, "M") == pd.Period(end, "M") - 2

    def test_explicit_dates(self, panel):
        start, end = ValidationSettings(start="2005-01-15", end="2006-01-15").dates(panel, "gdp")
        assert (start, end) == (pd.Timestamp("2005-01-15"), pd.Timestamp("2006-01-15"))

    def test_target_without_observations(self, panel):
        frame = panel.to_frame()
        frame["gdp"] = np.nan
        empty = panel.with_data(frame)
        with pytest.raises(ValueError, match="no observations"):
            ValidationSettings().dates(empty, "gdp")

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"n_vintages": 0}, "n_vintages"),
            ({"refit_every": True}, "refit_every"),
            ({"max_missing": 1.0}, "max_missing"),
        ],
    )
    def test_invalid(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            ValidationSettings(**kwargs).validate()

    def test_no_evaluable_forecast(self, panel):
        settings = {"start": "2010-03-15", "end": "2010-03-15", "target_offsets": (5,)}
        with pytest.raises(ValueError, match="No evaluable forecast"):
            quiet(
                select_blocks,
                TwoStepDFM(n_factors=1),
                panel.truncate(None, "2009-12"),
                "gdp",
                GROUPS,
                validation=settings,
            )


# ---------------------------------------------------------------------- errors
class TestErrors:
    @pytest.mark.parametrize(
        ("kwargs", "error", "match"),
        [
            ({"direction": "both"}, ValueError, "direction"),
            ({"scoring": "r2"}, ValueError, "scoring"),
            ({"min_improvement": -1.0}, ValueError, "min_improvement"),
            ({"always_include": ["bogus"]}, ValueError, "unknown groups"),
            ({"max_blocks": 0}, ValueError, "max_groups"),
            ({"validation": 3}, TypeError, "validation"),
            ({"validation": {"refit_every": 0}}, ValueError, "refit_every"),
        ],
    )
    def test_options(self, panel, kwargs, error, match):
        with pytest.raises(error, match=match):
            select_blocks(TwoStepDFM(n_factors=1), panel, "gdp", GROUPS, **kwargs)

    def test_model_without_fit(self, panel):
        with pytest.raises(TypeError, match="fit"):
            select_blocks(object(), panel, "gdp", GROUPS)

    def test_bad_blocks(self, panel):
        with pytest.raises(TypeError, match="blocks must be"):
            select_blocks(TwoStepDFM(), panel, "gdp", 3)
        with pytest.raises(ValueError, match="Unknown blocks"):
            select_blocks(TwoStepDFM(), panel, "gdp", ["real", "bogus"])
        with pytest.raises(ValueError, match="Unknown predictors"):
            select_blocks(TwoStepDFM(), panel, "gdp", {"a": ["zzz"]})
        with pytest.raises(ValueError, match="No non-empty"):
            select_blocks(TwoStepDFM(), panel, "gdp", {"a": []})

    def test_no_block_metadata(self):
        data = simulate_two_step_example(random_state=0)
        with pytest.raises(ValueError, match="no block metadata"):
            select_blocks(TwoStepDFM(), data, "gdp")

    def test_bad_candidates(self, panel):
        with pytest.raises(TypeError, match="candidates must be"):
            select_variables(TwoStepDFM(), panel, "gdp", "x1")
        with pytest.raises(TypeError, match="candidates must be"):
            select_variables(TwoStepDFM(), panel, "gdp", 5)


def test_scorings_constant():
    assert SCORINGS == ("rmsfe", "mae", "aic", "bic", "hq")


def test_native_estimates_helper():
    idx = pd.period_range("2020-01", periods=6, freq="M")
    frame = pd.DataFrame({"y": np.arange(6.0)}, index=idx)
    from nowcastbox.core.frequency import Frequency

    out = blocks_module._native_estimates(frame, "y", Frequency.QUARTERLY)
    assert out.tolist() == [2.0, 5.0]
    assert str(out.index[0]) == "2020Q1"


def test_settings_instance_and_ic_failures(panel):
    sub = panel.select(["gdp", "x1", "x2"])
    path = quiet(
        select_variables,
        TwoStepDFM(n_factors=2),
        sub,
        "gdp",
        scoring="bic",
        validation=ValidationSettings(),
    )
    # single series cannot carry two factors: every first step fails -> nothing chosen
    assert path.chosen == ()
    assert np.isinf(path.score)
    assert path.info["n_failures"] == 2


def test_information_update_fallbacks():
    class NoResults:
        def update(self, panel):
            return "not results"

    class FakeResults:
        target = "y"

        def predict(self, panel):
            return "not a frame"

    assert blocks_module._information_update(NoResults(), None, None) is None
    assert blocks_module._information_update(object(), FakeResults(), None) is None
