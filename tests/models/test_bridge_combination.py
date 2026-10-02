"""Tests of BridgeCombination (combination of bridge equations, Bańbura et al. 2023)."""

from __future__ import annotations

import doctest
import time
import warnings
from collections.abc import Sequence
from math import comb

import numpy as np
import pandas as pd
import pytest

import nowcastbox.models.bridge_combination as bc_module
from nowcastbox.core.data import MixedFrequencyData
from nowcastbox.core.exceptions import DataQualityWarning, ModelNotFittedError, NowcastDataError
from nowcastbox.core.results import build_nowcast_frame
from nowcastbox.models import (
    BridgeCombination,
    BridgeCombinationResults,
    BridgeEquation,
    bridge_equation_count,
)
from nowcastbox.models.extrapolation import native_until


def make_panel(
    n_months: int = 180,
    n_monthly: int = 3,
    n_quarterly: int = 1,
    seed: int = 0,
    ragged: bool = True,
) -> MixedFrequencyData:
    """Monthly AR(1) indicators m0.., quarterly indicators q0.., target y driven by m0, m1."""
    rng = np.random.default_rng(seed)
    idx = pd.period_range("2000-01", periods=n_months, freq="M")
    x = np.zeros((n_months, n_monthly))
    for t in range(1, n_months):
        x[t] = 0.5 * x[t - 1] + rng.standard_normal(n_monthly)
    frame = pd.DataFrame(x, index=idx, columns=[f"m{i}" for i in range(n_monthly)])
    quarter_end = idx.month % 3 == 0
    for j in range(n_quarterly):
        frame[f"q{j}"] = np.where(quarter_end, rng.standard_normal(n_months), np.nan)
    signal = frame.iloc[:, : min(2, n_monthly)].sum(axis=1).rolling(3).mean()
    frame["y"] = (0.5 + signal + 0.3 * rng.standard_normal(n_months)).where(quarter_end)
    freq = {c: "M" for c in frame.columns if c.startswith("m")}
    freq |= {c: "Q" for c in frame.columns if c.startswith("q")} | {"y": "Q"}
    if ragged:
        frame.iloc[-3:, frame.columns.get_loc("y")] = np.nan
        frame.iloc[-1:, 0] = np.nan
        if n_monthly > 1:
            frame.iloc[-2:, 1] = np.nan
        for j in range(n_quarterly):
            frame.iloc[-3:, frame.columns.get_loc(f"q{j}")] = np.nan
    delays = dict.fromkeys(freq, 30) | {"y": 60}
    return MixedFrequencyData(frame, freq, release_delays=delays)


@pytest.fixture
def panel() -> MixedFrequencyData:
    return make_panel()


# ---------------------------------------------------------------------- counting
class TestCount:
    @pytest.mark.parametrize(("n_m", "n_q"), [(1, 0), (3, 1), (4, 2), (5, 0)])
    def test_number_of_equations(self, n_m: int, n_q: int) -> None:
        data = make_panel(n_monthly=n_m, n_quarterly=n_q, ragged=False)
        res = BridgeCombination().fit(data, "y")
        expected = (comb(n_m, 1) + comb(n_m, 2)) * (1 + n_q)
        assert res.info["n_equations"] == expected == bridge_equation_count(n_m, n_q)
        assert len(res.equations()) == expected
        assert res.equations()["spec"].is_unique

    def test_count_helper(self) -> None:
        assert bridge_equation_count(50) == 1275
        assert bridge_equation_count(3, 2, max_monthly=1, max_quarterly=2) == 3 * 4
        assert bridge_equation_count(2, 1, min_monthly=0) == 4 * 2 - 1
        assert bridge_equation_count(2, 1, min_monthly=0, include_empty=True) == 8

    def test_min_monthly_zero_with_target_lags_includes_empty(self, panel) -> None:
        res = BridgeCombination(min_monthly=0, target_lags=1).fit(panel, "y")
        specs = res.equations()["spec"].tolist()
        assert "y ~ 1" in specs
        assert "y ~ q0" in specs
        expected = bridge_equation_count(3, 1, min_monthly=0, include_empty=True)
        assert len(specs) == expected

    def test_min_monthly_zero_without_lags_skips_empty(self, panel) -> None:
        res = BridgeCombination(min_monthly=0, max_monthly=1).fit(panel, "y")
        assert "y ~ 1" not in res.equations()["spec"].tolist()

    def test_max_monthly_larger_than_panel(self) -> None:
        data = make_panel(n_monthly=2, n_quarterly=0)
        res = BridgeCombination(max_monthly=5).fit(data, "y")
        assert res.info["n_equations"] == 3


# ---------------------------------------------------------------------- equivalence
class TestEquivalence:
    @pytest.mark.parametrize(
        ("target_lags", "regressor_lags", "aggregation", "formula"),
        [
            (0, 0, "average", "y ~ m0"),
            (1, 0, "average", "y ~ m0 + q0"),
            (2, 1, "mariano_murasawa", "y ~ m0 + q0"),
            (0, 0, "flow", "y ~ m1"),
        ],
    )
    def test_single_equation_matches_bridge_equation(
        self, panel, target_lags, regressor_lags, aggregation, formula
    ) -> None:
        kwargs = {
            "target_lags": target_lags,
            "regressor_lags": regressor_lags,
            "aggregation": aggregation,
            "ar_lags": 2,
            "horizon": 2,
        }
        bridge = BridgeEquation(**kwargs).fit(panel, formula)
        combo = BridgeCombination(max_monthly=1, **kwargs).fit(panel, formula)
        table = combo.equations()
        eq = table.index[table["spec"] == formula][0]
        np.testing.assert_allclose(
            combo.equation_estimates[eq].to_numpy(), bridge.estimate.to_numpy(), atol=1e-10
        )
        coefs = combo.params["coefficients"].loc[eq]
        np.testing.assert_allclose(
            coefs.to_numpy(), bridge.bridge.params.loc[coefs.index].to_numpy(), atol=1e-10
        )
        assert table.loc[eq, "n_obs"] == bridge.bridge.n_obs
        assert table.loc[eq, "rsquared"] == pytest.approx(bridge.bridge.rsquared, abs=1e-10)
        assert table.loc[eq, "sigma"] == pytest.approx(bridge.bridge.sigma, abs=1e-10)

    def test_one_equation_combination_is_the_equation(self) -> None:
        data = make_panel(n_monthly=1, n_quarterly=0)
        bridge = BridgeEquation().fit(data, "y")
        combo = BridgeCombination().fit(data, "y")
        assert combo.info["n_equations"] == 1
        np.testing.assert_allclose(combo.estimate.to_numpy(), bridge.estimate.to_numpy())
        assert combo.get_nowcast() == pytest.approx(bridge.get_nowcast())

    def test_no_constant(self, panel) -> None:
        bridge = BridgeEquation(add_constant=False).fit(panel, "y ~ m0")
        combo = BridgeCombination(add_constant=False).fit(panel, "y ~ m0")
        assert combo.get_nowcast() == pytest.approx(bridge.get_nowcast())
        assert "const" not in combo.params["coefficients"].index.get_level_values("term")


# ---------------------------------------------------------------------- combination
class TestCombination:
    def test_mean(self, panel) -> None:
        res = BridgeCombination().fit(panel, "y")
        est = res.equation_estimates
        np.testing.assert_allclose(res.estimate.to_numpy(), est.mean(axis=1).to_numpy())
        assert res.weights.sum() == pytest.approx(1.0)
        assert np.allclose(res.weights, 1 / len(est.columns))

    def test_median(self, panel) -> None:
        res = BridgeCombination(combine="median").fit(panel, "y")
        est = res.equation_estimates
        np.testing.assert_allclose(res.estimate.to_numpy(), est.median(axis=1).to_numpy())
        assert res.weights.isna().all()

    def test_inverse_mse_weights(self, panel) -> None:
        res = BridgeCombination(combine="inverse_mse").fit(panel, "y")
        table = res.equations()
        assert res.weights.sum() == pytest.approx(1.0)
        inv = 1.0 / table["mse"]
        np.testing.assert_allclose(res.weights.to_numpy(), (inv / inv.sum()).to_numpy())
        manual = (res.equation_estimates * res.weights).sum(axis=1)
        np.testing.assert_allclose(res.estimate.to_numpy(), manual.to_numpy())
        # the true model (m0 + m1) gets the largest weight
        assert table.loc[res.weights.idxmax(), "spec"].startswith("y ~ m0 + m1")

    def test_in_sample_mse_is_mean_squared_residual(self, panel) -> None:
        res = BridgeCombination().fit(panel, "y")
        table = res.equations()
        dof = table["n_obs"] - table["n_coefficients"]
        expected = table["sigma"] ** 2 * dof / table["n_obs"]
        np.testing.assert_allclose(table["mse"].to_numpy(), expected.to_numpy())

    def test_mse_window_and_discount(self, panel) -> None:
        window, delta = 6, 0.8
        res = BridgeCombination(mse_window=window, discount=delta).fit(panel, "y ~ m0")
        bridge = BridgeEquation().fit(panel, "y ~ m0").bridge
        resid = bridge.residuals.to_numpy()[-window:]
        w = delta ** np.arange(window - 1, -1, -1.0)
        expected = float((w * resid**2).sum() / w.sum())
        assert res.equations()["mse"].iloc[0] == pytest.approx(expected)
        assert len(res.info["mse_periods"]) == window

    def test_out_of_sample_mse(self, panel) -> None:
        res = BridgeCombination(mse="out_of_sample", min_train=10, mse_window=12).fit(
            panel, "y ~ m0"
        )
        reg = res.indicators["m0"]
        y = res.observed
        positions = np.flatnonzero(y.notna().to_numpy())[-12:]
        errors = []
        for s in positions:
            train = pd.concat([y.iloc[:s], reg.iloc[:s]], axis=1).dropna()
            if len(train) < 10:
                continue
            X = np.column_stack([np.ones(len(train)), train.iloc[:, 1]])
            beta, *_ = np.linalg.lstsq(X, train.iloc[:, 0].to_numpy(), rcond=None)
            errors.append(y.iloc[s] - (beta[0] + beta[1] * reg.iloc[s]))
        assert res.equations()["mse"].iloc[0] == pytest.approx(np.mean(np.square(errors)))

    def test_out_of_sample_mse_too_short(self) -> None:
        data = make_panel(n_months=36, ragged=False)
        with pytest.raises(NowcastDataError, match="No bridge equation"):
            BridgeCombination(combine="inverse_mse", mse="out_of_sample", min_train=50).fit(
                data, "y"
            )
        # equal weights do not need the MSE
        res = BridgeCombination(mse="out_of_sample", min_train=50).fit(data, "y")
        assert res.equations()["mse"].isna().all()

    def test_trim_drops_worst(self, panel) -> None:
        full = BridgeCombination().fit(panel, "y")
        res = BridgeCombination(trim=0.5).fit(panel, "y")
        table = res.equations()
        n = len(table)
        assert int(table["included"].sum()) == n - n // 2
        kept_max = table.loc[table["included"], "mse"].max()
        dropped_min = table.loc[~table["included"], "mse"].min()
        assert kept_max <= dropped_min
        assert (table.loc[~table["included"], "weight"] == 0).all()
        assert not np.isclose(res.get_nowcast(), full.get_nowcast())
        assert len(res.equations(included_only=True)) == n - n // 2

    def test_trim_keeps_at_least_one(self) -> None:
        data = make_panel(n_monthly=1, n_quarterly=1)
        res = BridgeCombination(trim=0.99).fit(data, "y")
        assert int(res.equations()["included"].sum()) == 1

    def test_inverse_mse_with_perfect_fit(self) -> None:
        idx = pd.period_range("2000-01", periods=60, freq="M")
        x = np.sin(np.arange(60.0))
        z = np.cos(np.arange(60.0) / 3)
        y = pd.Series(x, index=idx).rolling(3).mean().where(idx.month % 3 == 0)
        df = pd.DataFrame({"x": x, "z": z, "y": y}, index=idx)
        res = BridgeCombination(combine="inverse_mse", max_monthly=1).fit(
            df, "y", frequency={"x": "M", "z": "M", "y": "Q"}
        )
        weights = res.weights
        assert np.isfinite(weights).all()
        assert weights.sum() == pytest.approx(1.0)
        assert weights.iloc[0] > 0.999

    def test_nowcast_frame_extras(self, panel) -> None:
        res = BridgeCombination().fit(panel, "y")
        frame = res.nowcast
        for col in ("observed", "in_sample", "out_of_sample", "equation_std", "n_equations"):
            assert col in frame
        est = res.equation_estimates
        np.testing.assert_allclose(frame["equation_min"], est.min(axis=1))
        np.testing.assert_allclose(frame["equation_max"], est.max(axis=1))
        np.testing.assert_allclose(frame["equation_std"], est.std(axis=1, ddof=0))
        assert frame["n_equations"].iloc[-1] == len(est.columns)
        last_obs = res.observed.last_valid_index()
        assert np.isnan(frame.loc[last_obs, "out_of_sample"])
        assert np.isfinite(frame["out_of_sample"].iloc[-2:]).all()


# ---------------------------------------------------------------------- extrapolation
class TestExtrapolation:
    def test_indicators_extrapolated_once_and_shared(self, panel) -> None:
        res = BridgeCombination().fit(panel, "y")
        assert set(res.extrapolated) == {"m0", "m1", "m2", "q0"}
        last = res.nowcast.index[-1]
        for name, series in res.extrapolated.items():
            freq = panel.metadata[name].frequency.pandas_freq
            assert series.index[-1] == last.asfreq(freq, how="E")
            assert series.notna().iloc[-6:].all()
        assert list(res.indicators.columns) == ["m0", "m1", "m2", "q0"]

    def test_no_extrapolation(self, panel) -> None:
        res = BridgeCombination(extrapolation=None).fit(panel, "y")
        current = res.nowcast.index[-2]
        table = res.equations(current)
        # only the equations whose indicators are complete for the quarter (m2 alone)
        finite = table.loc[table["nowcast"].notna(), "spec"].tolist()
        assert finite == ["y ~ m2"]
        assert res.get_nowcast(current) == pytest.approx(table["nowcast"].dropna().iloc[0])
        assert res.nowcast.loc[current, "n_equations"] == 1
        assert np.isnan(res.get_nowcast(res.nowcast.index[-1]))
        assert np.isnan(res.extrapolated["m0"].iloc[-1])
        assert np.isfinite(res.estimate.iloc[:-3].dropna()).all()

    def test_callable_extrapolator(self, panel) -> None:
        calls: list[Sequence[str]] = []

        def zeros(data, columns, end):
            calls.append(list(columns))
            return {c: native_until(data, c, end).fillna(0.0) for c in columns}

        res = BridgeCombination(extrapolation=zeros).fit(panel, "y")
        assert calls == [["m0", "m1", "m2", "q0"]]
        assert res.extrapolated["m0"].iloc[-1] == 0.0

    def test_extrapolator_missing_columns(self, panel) -> None:
        def partial(data, columns, end):
            return {columns[0]: native_until(data, columns[0], end)}

        with pytest.raises(NowcastDataError, match="returned no values"):
            BridgeCombination(extrapolation=partial).fit(panel, "y")

    def test_extrapolation_options(self, panel) -> None:
        a = BridgeCombination(ar_lags=3).fit(panel, "y")
        b = BridgeCombination(extrapolation_options={"ar_lags": 3}).fit(panel, "y")
        np.testing.assert_allclose(a.estimate, b.estimate)
        with pytest.raises(ValueError, match="Unknown extrapolation"):
            BridgeCombination(extrapolation="nope").fit(panel, "y")


# ---------------------------------------------------------------------- pseudo real time
class TestNoLookAhead:
    def test_as_of_equals_vintage(self, panel) -> None:
        vintage = "2012-05-15"
        model = BridgeCombination(combine="inverse_mse", mse="out_of_sample", trim=0.2)
        a = model.fit(panel, "y", as_of=vintage)
        b = BridgeCombination(combine="inverse_mse", mse="out_of_sample", trim=0.2).fit(
            panel.as_of(vintage), "y"
        )
        pd.testing.assert_frame_equal(a.nowcast, b.nowcast)
        pd.testing.assert_series_equal(a.weights, b.weights)

    def test_weights_ignore_future_data(self, panel) -> None:
        vintage = "2010-02-20"
        frame = panel.to_frame()
        after = frame.index > pd.Period("2010-02", "M")
        shocked = frame.copy()
        shocked.loc[after] = shocked.loc[after] * 3.0 + 1.0
        changed = panel.with_data(shocked)
        kwargs = {"combine": "inverse_mse", "mse": "out_of_sample", "trim": 0.3}
        a = BridgeCombination(**kwargs).fit(panel, "y", as_of=vintage)
        b = BridgeCombination(**kwargs).fit(changed, "y", as_of=vintage)
        pd.testing.assert_series_equal(a.weights, b.weights)
        assert a.get_nowcast() == pytest.approx(b.get_nowcast())
        assert not np.allclose(
            BridgeCombination(**kwargs).fit(panel, "y").weights,
            BridgeCombination(**kwargs).fit(changed, "y").weights,
        )

    def test_backtest_runs(self, panel) -> None:
        from nowcastbox.evaluation import PseudoRealTimeBacktest

        bt = PseudoRealTimeBacktest(
            BridgeCombination(max_monthly=1),
            data=panel,
            target="y",
            start="2012-01-15",
            end="2012-12-15",
            step="M",
        )
        res = bt.run()
        frame = res.to_frame()
        assert "BridgeCombination" in res.models
        assert np.isfinite(frame["forecast"]).all()


# ---------------------------------------------------------------------- errors
class TestErrors:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"max_monthly": 0},
            {"min_monthly": 3, "max_monthly": 2},
            {"max_quarterly": -1},
            {"target_lags": 1.5},
            {"regressor_lags": True},
            {"ar_lags": -1},
            {"horizon": -1},
            {"min_train": 0},
            {"mse_window": 0},
            {"combine": "mode"},
            {"mse": "aic"},
            {"discount": 0.0},
            {"discount": 1.5},
            {"trim": 1.0},
            {"trim": -0.1},
            {"add_constant": 1},
            {"extrapolation_options": [("ar_lags", 2)]},
        ],
    )
    def test_invalid_params(self, panel, kwargs) -> None:
        with pytest.raises(ValueError):
            BridgeCombination(**kwargs).fit(panel, "y")

    def test_unknown_fit_option(self, panel) -> None:
        with pytest.raises(TypeError, match="Unexpected fit options"):
            BridgeCombination().fit(panel, "y", nope=1)

    def test_not_fitted(self) -> None:
        with pytest.raises(ModelNotFittedError):
            _ = BridgeCombination().results_

    def test_no_high_frequency_indicator(self, panel) -> None:
        with pytest.raises(NowcastDataError, match="No bridge equation can be formed"):
            BridgeCombination().fit(panel, "y ~ q0")

    def test_lower_frequency_predictor(self) -> None:
        idx = pd.period_range("2000-01", periods=36, freq="M")
        q = np.where(idx.month % 3 == 0, 1.0, np.nan) * np.arange(36)
        df = pd.DataFrame({"y": np.arange(36.0), "q": q}, index=idx)
        with pytest.raises(NowcastDataError, match="lower frequency"):
            BridgeCombination().fit(df, "y", frequency={"y": "M", "q": "Q"})

    def test_predictor_without_observations_is_dropped(self, panel) -> None:
        frame = panel.to_frame()
        frame["m2"] = np.nan
        with pytest.warns(DataQualityWarning, match=r"\['m2'\].*no observations"):
            res = BridgeCombination().fit(panel.with_data(frame), "y")
        expected = BridgeCombination().fit(panel.select(["m0", "m1", "q0", "y"]), "y")
        assert res.info["n_equations"] == bridge_equation_count(2, 1)
        np.testing.assert_allclose(res.estimate, expected.estimate)

    def test_only_empty_predictors(self, panel) -> None:
        frame = panel.to_frame()[["m0", "y"]].assign(m0=np.nan)
        with pytest.warns(DataQualityWarning, match="without any observation"):
            data = MixedFrequencyData(frame, {"m0": "M", "y": "Q"})
        with (
            pytest.warns(DataQualityWarning, match="no observations"),
            pytest.raises(NowcastDataError, match="No bridge equation can be formed"),
        ):
            BridgeCombination().fit(data, "y")

    def test_short_predictor_is_dropped_by_the_extrapolator(self, panel) -> None:
        """A series released only recently (early vintage) must not stop the fit."""
        frame = panel.to_frame()
        frame.iloc[:-3, frame.columns.get_loc("m2")] = np.nan  # 3 observations only
        with pytest.warns(DataQualityWarning, match=r"\['m2'\].*cannot be extrapolated"):
            res = BridgeCombination().fit(panel.with_data(frame), "y")
        assert "m2" not in res.extrapolated
        assert not any("m2" in spec for spec in res.equations()["spec"])
        expected = BridgeCombination().fit(panel.select(["m0", "m1", "q0", "y"]), "y")
        np.testing.assert_allclose(res.estimate, expected.estimate)

    def test_extrapolation_failure_of_every_series_raises(self, panel) -> None:
        frame = panel.to_frame()[["m0", "y"]]
        frame.iloc[:-3, 0] = np.nan
        data = MixedFrequencyData(frame, {"m0": "M", "y": "Q"})
        with pytest.raises(NowcastDataError, match="Too few complete observations"):
            BridgeCombination().fit(data, "y")

    def test_joint_extrapolation_failure_falls_back_to_single_series(self, panel) -> None:
        def joint_only_small(data, columns, end):
            if len(columns) > 1:
                raise NowcastDataError("joint model failed")
            return {c: native_until(data, c, end).fillna(0.0) for c in columns}

        with warnings.catch_warnings():
            warnings.simplefilter("error", DataQualityWarning)
            res = BridgeCombination(extrapolation=joint_only_small).fit(panel, "y")
        assert sorted(res.extrapolated) == ["m0", "m1", "m2", "q0"]

    def test_target_without_observations(self, panel) -> None:
        frame = panel.to_frame()
        frame["y"] = np.nan
        with pytest.raises(NowcastDataError, match="has no observations"):
            BridgeCombination().fit(panel.with_data(frame), "y")

    def test_collinear_equations_warn(self, panel) -> None:
        frame = panel.to_frame()
        frame["m2"] = 2.0 * frame["m0"]
        with pytest.warns(DataQualityWarning, match="could not be estimated"):
            res = BridgeCombination(max_quarterly=0).fit(panel.with_data(frame), "y")
        table = res.equations()
        bad = table.loc[table["spec"] == "y ~ m0 + m2"]
        assert not bad["valid"].iloc[0]
        assert not bad["included"].iloc[0]
        assert np.isnan(bad["nowcast"].iloc[0])
        assert res.weights.sum() == pytest.approx(1.0)

    def test_all_equations_invalid(self) -> None:
        data = make_panel(n_months=6, n_quarterly=0, ragged=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DataQualityWarning)
            with pytest.raises(NowcastDataError, match="No bridge equation could be estimated"):
                BridgeCombination(extrapolation=None).fit(data, "y")


# ---------------------------------------------------------------------- results API
class TestResults:
    def test_equations_table(self, panel) -> None:
        res = BridgeCombination().fit(panel, "y")
        table = res.equations()
        assert list(table.columns) == [
            "spec",
            "monthly",
            "quarterly",
            "n_coefficients",
            "n_obs",
            "rsquared",
            "sigma",
            "mse",
            "valid",
            "included",
            "weight",
            "nowcast",
        ]
        assert table.index[0] == "eq0001"
        np.testing.assert_allclose(
            table["nowcast"].to_numpy(),
            res.equation_estimates.loc[pd.Period("2014Q4", "Q")].to_numpy(),
        )
        assert table.loc["eq0001", "monthly"] == ("m0",)
        assert table.loc["eq0001", "quarterly"] == ()
        first = res.equations("2001Q1")["nowcast"]
        assert first.notna().all()
        with pytest.raises(KeyError):
            res.equations("1990Q1")

    def test_no_period_after_last_observation(self) -> None:
        data = make_panel(ragged=False)
        res = BridgeCombination(horizon=0).fit(data, "y")
        with pytest.raises(KeyError, match="No period after"):
            res.equations()

    def test_tables_required(self) -> None:
        idx = pd.period_range("2020Q1", periods=2, freq="Q")
        frame = build_nowcast_frame(
            pd.Series([1.0, np.nan], index=idx), pd.Series([1.0, 2.0], index=idx)
        )
        res = BridgeCombinationResults(target="y", nowcast=frame)
        for call in (lambda: res.weights, lambda: res.equation_estimates, res.equations):
            with pytest.raises(ValueError, match="no equation table"):
                call()
        assert "Bridge combination" not in res.summary()

    def test_summary(self, panel) -> None:
        text = BridgeCombination().fit(panel, "y").summary()
        assert "Bridge combination" in text
        assert "Equations" in text and "mse=" in text

    def test_coefficients(self, panel) -> None:
        res = BridgeCombination(target_lags=1).fit(panel, "y")
        coefs = res.params["coefficients"]
        assert coefs.index.names == ["equation", "term"]
        assert list(coefs.loc["eq0001"].index) == ["const", "m0", "y_lag1"]

    def test_nowcast_change_by_equation(self, panel) -> None:
        old = BridgeCombination(combine="inverse_mse").fit(panel, "y", as_of="2014-11-20")
        new = BridgeCombination(combine="inverse_mse").fit(panel, "y")
        period = new.nowcast.index[-2]
        change = new.nowcast_change(old, period)
        assert list(change.columns) == [
            "previous",
            "current",
            "previous_weight",
            "current_weight",
            "contribution",
        ]
        total = new.get_nowcast(period) - old.get_nowcast(period)
        assert change["contribution"].sum() == pytest.approx(total)
        assert change["current_weight"].sum() == pytest.approx(1.0)

    def test_nowcast_change_by_indicator(self, panel) -> None:
        old = BridgeCombination(min_monthly=0, target_lags=1).fit(panel, "y", as_of="2014-10-05")
        new = BridgeCombination(min_monthly=0, target_lags=1).fit(panel, "y")
        period = new.nowcast.index[-2]
        change = new.nowcast_change(old, period, by="indicator")
        assert set(change.index) == {"m0", "m1", "m2", "q0", "(target lags)"}
        total = new.get_nowcast(period) - old.get_nowcast(period)
        assert change["contribution"].sum() == pytest.approx(total)

    def test_nowcast_change_median_and_errors(self, panel) -> None:
        old = BridgeCombination(combine="median").fit(panel, "y", as_of="2014-11-20")
        new = BridgeCombination(combine="median").fit(panel, "y")
        assert new.nowcast_change(old)["contribution"].isna().all()
        with pytest.raises(ValueError, match="by must be"):
            new.nowcast_change(old, by="series")

    def test_nowcast_change_with_equations_of_previous_vintage_only(self, panel) -> None:
        frame = panel.to_frame()
        frame["m3"] = frame["m2"].shift(1)
        freq = {c: m.frequency for c, m in panel.metadata.items()} | {"m3": "M"}
        wider = MixedFrequencyData(frame, freq)
        old = BridgeCombination(max_quarterly=0).fit(wider, "y")
        new = BridgeCombination(max_quarterly=0).fit(panel, "y")
        change = new.nowcast_change(old, by="indicator")
        assert "m3" in change.index
        assert "(other)" not in change.index
        total = new.get_nowcast() - old.get_nowcast()
        assert change["contribution"].sum() == pytest.approx(total)

    def test_by_indicator_unknown_spec(self, panel) -> None:
        res = BridgeCombination().fit(panel, "y")
        out = res._by_indicator(pd.Series({"y ~ other": 1.0}))
        assert out.loc["(other)", "contribution"] == 1.0


# ---------------------------------------------------------------------- performance
@pytest.mark.slow
def test_fifty_monthly_indicators_fit_quickly() -> None:
    data = make_panel(n_months=240, n_monthly=50, n_quarterly=1, seed=4)
    started = time.perf_counter()
    res = BridgeCombination(combine="inverse_mse", mse="out_of_sample").fit(data, "y")
    elapsed = time.perf_counter() - started
    assert res.info["n_equations"] == 1275 * 2
    assert elapsed < 60.0
    started = time.perf_counter()
    BridgeCombination().fit(data, "y")
    assert time.perf_counter() - started < 20.0


def test_chunks_split_large_groups(monkeypatch, panel) -> None:
    reference = BridgeCombination().fit(panel, "y")
    monkeypatch.setattr(bc_module, "_CHUNK", 2)
    chunked = BridgeCombination().fit(panel, "y")
    pd.testing.assert_frame_equal(reference.equation_estimates, chunked.equation_estimates)
    assert list(bc_module._chunks([[0], [1], [2], [0, 1]])) == [(0, 2), (2, 3), (3, 4)]


def test_discounted_mse_without_positions() -> None:
    out = bc_module._discounted_mse(np.empty((2, 0)), np.empty(0, dtype=int), 1.0)
    assert np.isnan(out).all()


def test_doctests() -> None:
    result = doctest.testmod(bc_module, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)
    assert result.attempted > 0
    assert result.failed == 0
