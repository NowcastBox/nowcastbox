"""Tests of the MIDAS benchmarks (U-MIDAS, exponential Almon / Beta MIDAS)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from nowcastbox.benchmarks import MIDAS, UMIDAS, beta_weights, exp_almon_weights
from nowcastbox.core.data import MixedFrequencyData
from nowcastbox.core.exceptions import ConvergenceWarning, DataQualityWarning, NowcastDataError
from nowcastbox.models.bridge import ar_extend
from tests.benchmarks.conftest import FREQ, simulate_panel


def manual_design(
    frame: pd.DataFrame, shifts: dict[str, int], n_lags: int, quarters: pd.PeriodIndex
) -> np.ndarray:
    """Regressors [1, x lags, z lags] at the quarter-end months, built with pandas shifts."""
    months = quarters.asfreq("M", how="E")
    cols = [np.ones(len(quarters))]
    for name, shift in shifts.items():
        for j in range(n_lags):
            cols.append(frame[name].shift(shift + j).reindex(months).to_numpy())
    return np.column_stack(cols)


class TestWeights:
    @given(st.floats(-3, 3), st.floats(-0.5, 0.5), st.integers(1, 24))
    def test_exp_almon_sums_to_one(self, t1: float, t2: float, k: int) -> None:
        w = exp_almon_weights(t1, t2, k)
        assert w.shape == (k,)
        assert w.sum() == pytest.approx(1.0)
        assert (w >= 0).all()

    @given(st.floats(0.1, 20), st.floats(0.1, 20), st.integers(1, 24))
    def test_beta_sums_to_one(self, t1: float, t2: float, k: int) -> None:
        w = beta_weights(t1, t2, k)
        assert w.sum() == pytest.approx(1.0)
        assert np.isfinite(w).all()

    def test_shapes(self) -> None:
        np.testing.assert_allclose(beta_weights(1.0, 1.0, 5), np.full(5, 0.2))
        assert beta_weights(3.0, 2.0, 1).tolist() == [1.0]
        assert (np.diff(exp_almon_weights(0.5, -0.3, 8)[2:]) < 0).all()
        hump = beta_weights(2.0, 4.0, 12)
        assert 0 < int(np.argmax(hump)) < 11

    def test_large_parameters_are_stable(self) -> None:
        assert np.isfinite(exp_almon_weights(10.0, 2.0, 30)).all()

    @pytest.mark.parametrize(
        ("func", "args"),
        [
            (exp_almon_weights, (0.0, 0.0, 0)),
            (beta_weights, (1.0, 1.0, 0)),
            (beta_weights, (0.0, 1.0, 3)),
            (beta_weights, (1.0, -1.0, 3)),
        ],
    )
    def test_invalid(self, func, args) -> None:
        with pytest.raises(ValueError, match="must be"):
            func(*args)


class TestUMIDAS:
    def test_exact_recovery(self) -> None:
        frame = simulate_panel(noise=0.0)
        frame.iloc[-1, 2] = np.nan
        bench = UMIDAS(predictors=["x"]).fit(MixedFrequencyData(frame, FREQ), "y")
        assert bench.shifts_.to_dict() == {"x": 0}
        np.testing.assert_allclose(bench.coef_.to_numpy(), [0.5, 1 / 3, 1 / 3, 1 / 3], atol=1e-10)
        expected = 0.5 + frame["x"].iloc[-3:].mean()
        assert bench.predict(["2012Q2"]).iloc[0] == pytest.approx(expected)

    def test_realignment_matches_manual_ols(self, ragged) -> None:
        bench = UMIDAS().fit(ragged, "y")
        assert bench.shifts_.to_dict() == {"x": 1, "z": 2}
        frame = ragged.data
        quarters = pd.period_range("2000Q1", "2012Q1", freq="Q")
        X = manual_design(frame, {"x": 1, "z": 2}, 3, quarters)
        y = frame["y"].reindex(quarters.asfreq("M", how="E")).to_numpy()
        ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
        coef, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
        np.testing.assert_allclose(bench.coef_.to_numpy(), coef, atol=1e-10)
        assert list(bench.coef_.index) == [
            "const", "x_lag0", "x_lag1", "x_lag2", "z_lag0", "z_lag1", "z_lag2",
        ]  # fmt: skip
        target = pd.PeriodIndex(["2012Q2"], freq="Q")
        x_new = manual_design(frame, {"x": 1, "z": 2}, 3, target)
        assert bench.predict(target).iloc[0] == pytest.approx(float(x_new[0] @ coef))

    def test_next_quarter_uses_larger_shift(self, ragged) -> None:
        bench = UMIDAS().fit(ragged, "y")
        frame = ragged.data
        quarters = pd.period_range("2000Q1", "2012Q1", freq="Q")
        X = manual_design(frame, {"x": 4, "z": 5}, 3, quarters)
        y = frame["y"].reindex(quarters.asfreq("M", how="E")).to_numpy()
        ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
        coef, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
        target = pd.PeriodIndex(["2012Q3"], freq="Q")
        x_new = manual_design(frame.reindex(pd.period_range("2000-01", "2012-09", freq="M")),
                              {"x": 4, "z": 5}, 3, target)  # fmt: skip
        out = bench.predict(["2012Q2", "2012Q3", "2011Q4", "1999Q4"])
        assert out.iloc[1] == pytest.approx(float(x_new[0] @ coef))
        assert np.isfinite(out.iloc[2])
        assert np.isnan(out.iloc[3])

    def test_ar_completion(self, ragged) -> None:
        bench = UMIDAS(ragged_edge="ar", ar_lags=2).fit(ragged, "y")
        assert bench.shifts_.to_dict() == {"x": 0, "z": 0}
        frame = ragged.data
        x = ar_extend(frame["x"].to_numpy(), 0, 2)
        x = ar_extend(x, 1, 2)
        z = ar_extend(frame["z"].dropna().to_numpy(), 2, 2)
        regressors = np.concatenate([[1.0], x[-1:-4:-1], z[-1:-4:-1]])
        assert bench.predict(["2012Q2"]).iloc[0] == pytest.approx(regressors @ bench.coef_)
        again = bench.predict(["2012Q1", "2012Q2"])  # cached AR extension is reused
        assert again.iloc[1] == pytest.approx(regressors @ bench.coef_)

    def test_direct_target_lags(self, ragged) -> None:
        bench = UMIDAS(predictors=["x"], target_lags=2).fit(ragged, "y")
        assert list(bench.coef_.index)[-2:] == ["y_L1", "y_L2"]
        frame = ragged.data
        quarters = pd.period_range("2000Q1", "2012Q1", freq="Q")
        y_q = frame["y"].reindex(quarters.asfreq("M", how="E")).to_numpy()
        X = manual_design(frame, {"x": 4}, 3, quarters)
        lags = np.column_stack([np.roll(y_q, 2), np.roll(y_q, 3)])
        lags[:3] = np.nan
        X = np.column_stack([X, lags])
        ok = np.isfinite(X).all(axis=1) & np.isfinite(y_q)
        coef, *_ = np.linalg.lstsq(X[ok], y_q[ok], rcond=None)
        x_new = manual_design(
            frame.reindex(pd.period_range("2000-01", "2012-09", freq="M")),
            {"x": 4},
            3,
            pd.PeriodIndex(["2012Q3"], freq="Q"),
        )
        row = np.concatenate([x_new[0], [y_q[-1], y_q[-2]]])
        assert bench.predict(["2012Q3"]).iloc[0] == pytest.approx(float(row @ coef))

    def test_same_frequency_predictor(self) -> None:
        idx = pd.period_range("2000-01", periods=90, freq="M")
        rng = np.random.default_rng(2)
        q = pd.Series(rng.standard_normal(90), index=idx).where(idx.month % 3 == 0)
        y = 1.0 + 2.0 * q
        y.iloc[-1] = np.nan
        data = MixedFrequencyData(pd.DataFrame({"q": q, "y": y}), {"q": "Q", "y": "Q"})
        bench = UMIDAS().fit(data, "y")
        np.testing.assert_allclose(bench.coef_.to_numpy(), [1.0, 2.0], atol=1e-10)

    def test_too_few_observations(self) -> None:
        frame = simulate_panel(n_months=12)
        with pytest.raises(NowcastDataError, match="Too few"):
            UMIDAS(n_lags=6).fit(MixedFrequencyData(frame, FREQ), "y")

    @pytest.mark.parametrize(
        "kwargs",
        [{"ragged_edge": "fill"}, {"n_lags": 0}, {"target_lags": -1}, {"ar_lags": -1}],
    )
    def test_invalid_parameters(self, panel, kwargs) -> None:
        with pytest.raises(ValueError, match="must be"):
            UMIDAS(**kwargs).fit(panel, "y")

    def test_invalid_predictors(self, panel) -> None:
        for predictors in (["w"], ["y"], []):
            with pytest.raises(NowcastDataError):
                UMIDAS(predictors=predictors).fit(panel, "y")

    def test_string_predictor_and_lower_frequency(self, panel) -> None:
        assert UMIDAS(predictors="x").fit(panel, "y").shifts_.index.tolist() == ["x"]
        frame = panel.data.assign(a=np.where(panel.index.month == 12, 1.0, np.nan))
        data = MixedFrequencyData(frame, {**FREQ, "a": "A"})
        with pytest.raises(NowcastDataError, match="lower frequency"):
            UMIDAS(predictors=["a"]).fit(data, "y")

    def test_predictor_without_observations(self, panel) -> None:
        with pytest.warns(DataQualityWarning):
            data = MixedFrequencyData(panel.data.assign(e=np.nan), {**FREQ, "e": "M"})
        with pytest.raises(NowcastDataError, match="no observations"):
            UMIDAS(predictors=["e"]).fit(data, "y")

    def test_target_without_observations(self, panel) -> None:
        with pytest.warns(DataQualityWarning):
            data = MixedFrequencyData(panel.data.assign(y=np.nan), FREQ)
        with pytest.raises(NowcastDataError, match="no observations"):
            UMIDAS().fit(data, "y")

    def test_refit_resets_state(self, ragged, panel) -> None:
        bench = UMIDAS().fit(ragged, "y")
        bench.fit(panel, "y")
        assert bench.shifts_.to_dict() == {"x": 3, "z": 3}


def midas_frame(weights: np.ndarray, n: int = 600, seed: int = 0) -> MixedFrequencyData:
    rng = np.random.default_rng(seed)
    idx = pd.period_range("1960-01", periods=n, freq="M")
    x = rng.standard_normal(n)
    y = pd.Series(1.0 + 2.0 * np.convolve(x, weights)[:n] + 0.05 * rng.standard_normal(n), idx)
    y.iloc[-1] = np.nan
    frame = pd.DataFrame({"x": x, "y": y.where(idx.month % 3 == 0)}, index=idx)
    return MixedFrequencyData(frame, {"x": "M", "y": "Q"})


class TestMIDAS:
    @pytest.mark.parametrize(
        ("polynomial", "theta"),
        [("exp_almon", (0.4, -0.15)), ("beta", (1.0, 4.0)), ("beta", (2.0, 3.0))],
    )
    def test_parameter_recovery(self, polynomial: str, theta: tuple[float, float]) -> None:
        func = exp_almon_weights if polynomial == "exp_almon" else beta_weights
        w = func(*theta, 9)
        bench = MIDAS(polynomial=polynomial).fit(midas_frame(w), "y")
        assert list(bench.coef_.index) == ["const", "x"]
        np.testing.assert_allclose(bench.coef_.to_numpy(), [1.0, 2.0], atol=0.03)
        np.testing.assert_allclose(bench.lag_weights_["x"], w, atol=0.03)
        assert bench.theta_.shape == (1, 2)

    def test_prediction_is_weighted_sum(self) -> None:
        w = exp_almon_weights(0.2, -0.05, 6)
        data = midas_frame(w)
        bench = MIDAS(n_lags=6).fit(data, "y")
        x = data["x"].to_numpy()
        expected = bench.coef_["const"] + bench.coef_["x"] * (bench.lag_weights_["x"] @ x[::-1][:6])
        assert bench.predict(["2009Q4"]).iloc[0] == pytest.approx(expected)

    def test_two_indicators_and_target_lags(self, ragged) -> None:
        bench = MIDAS(n_lags=4, target_lags=1, polynomial="beta").fit(ragged, "y")
        assert list(bench.coef_.index) == ["const", "x", "z", "y_L1"]
        assert list(bench.theta_.index) == ["x", "z"]
        assert np.isfinite(bench.predict(["2012Q2", "2012Q3"])).all()

    def test_single_lag(self, ragged) -> None:
        bench = MIDAS(n_lags=1, predictors=["x"]).fit(ragged, "y")
        assert bench.lag_weights_["x"].tolist() == [1.0]

    def test_convergence_warning(self) -> None:
        data = midas_frame(beta_weights(1.0, 4.0, 9), n=240)
        with pytest.warns(ConvergenceWarning, match="did not converge"):
            MIDAS(polynomial="beta", max_iter=1).fit(data, "y")

    @pytest.mark.parametrize("kwargs", [{"polynomial": "pdl"}, {"max_iter": 0}])
    def test_invalid_parameters(self, panel, kwargs) -> None:
        with pytest.raises(ValueError, match="must be"):
            MIDAS(**kwargs).fit(panel, "y")

    def test_default_lags(self, panel) -> None:
        assert MIDAS(predictors=["x"]).fit(panel, "y").lag_weights_["x"].size == 9
        assert UMIDAS(predictors=["x"]).fit(panel, "y").coef_.size == 4
