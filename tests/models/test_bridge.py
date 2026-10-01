"""Tests of nowcastbox.models.bridge (bridge regressions and BridgeEquation)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from hypothesis import given, settings
from hypothesis import strategies as st

from nowcastbox.core.data import MixedFrequencyData
from nowcastbox.core.exceptions import ModelNotFittedError, NowcastDataError
from nowcastbox.core.frequency import AggregationType, Frequency
from nowcastbox.models import BridgeEquation, BridgeResults
from nowcastbox.models.bridge import (
    aggregate_to_target,
    ar_extend,
    fit_bridge_regression,
    resolve_aggregation_weights,
)

M, Q, A = Frequency.MONTHLY, Frequency.QUARTERLY, Frequency.ANNUAL


def _quarterly_panel(
    rng: np.random.Generator,
    n_months: int = 120,
    *,
    noise: float = 0.0,
    drop_last_quarter: bool = True,
    ragged: tuple[int, int] = (1, 2),
) -> tuple[MixedFrequencyData, np.ndarray]:
    """Two AR(1) monthly indicators; quarterly y = 1 + 2 avg(x1) - avg(x2) + noise."""
    x = np.zeros((n_months, 2))
    for t in range(1, n_months):
        x[t] = 0.6 * x[t - 1] + rng.standard_normal(2)
    idx = pd.period_range("2000-01", periods=n_months, freq="M")
    avg = pd.DataFrame(x, index=idx).rolling(3).mean()
    y = 1.0 + 2.0 * avg[0] - avg[1] + noise * rng.standard_normal(n_months)
    y = y.where(idx.month % 3 == 0)
    if drop_last_quarter:
        y.iloc[np.flatnonzero(idx.month % 3 == 0)[-1]] = np.nan
    frame = pd.DataFrame({"x1": x[:, 0], "x2": x[:, 1], "y": y}, index=idx)
    for col, k in zip(("x1", "x2"), ragged, strict=True):
        if k:
            frame.iloc[-k:, frame.columns.get_loc(col)] = np.nan
    return MixedFrequencyData(frame, {"x1": "M", "x2": "M", "y": "Q"}), x


# ---------------------------------------------------------------------- regression
class TestFitBridgeRegression:
    def test_matches_statsmodels(self, rng: np.random.Generator) -> None:
        idx = pd.period_range("2000Q1", periods=50, freq="Q")
        x = pd.DataFrame(rng.standard_normal((50, 2)), index=idx, columns=["a", "b"])
        y = pd.Series(0.3 + x @ [1.0, -0.5] + 0.1 * rng.standard_normal(50), name="y")
        y.iloc[[3, 10]] = np.nan
        x.iloc[20, 1] = np.nan
        reg = fit_bridge_regression(y, x)
        mask = y.notna() & x.notna().all(axis=1)
        ref = sm.OLS(y[mask].to_numpy(), sm.add_constant(x[mask].to_numpy())).fit()
        np.testing.assert_allclose(reg.params.to_numpy(), ref.params, atol=1e-12)
        np.testing.assert_allclose(reg.bse.to_numpy(), ref.bse, atol=1e-12)
        assert reg.n_obs == int(mask.sum()) == 47
        assert reg.rsquared == pytest.approx(ref.rsquared)
        assert reg.sigma == pytest.approx(np.sqrt(ref.scale))
        assert list(reg.params.index) == ["const", "a", "b"]
        np.testing.assert_allclose(reg.residuals.to_numpy(), ref.resid)
        np.testing.assert_allclose(reg.fitted_values.to_numpy(), ref.fittedvalues)
        assert reg.residuals.index.equals(y.index[mask])
        assert isinstance(reg.sample, pd.PeriodIndex)

    def test_without_constant_and_hac(self, rng: np.random.Generator) -> None:
        idx = pd.period_range("2000Q1", periods=40, freq="Q")
        x = pd.DataFrame({"a": rng.standard_normal(40)}, index=idx)
        y = pd.Series(2 * x["a"] + rng.standard_normal(40), name="y")
        reg = fit_bridge_regression(
            y, x, add_constant=False, cov_type="HAC", cov_kwds={"maxlags": 2}
        )
        ref = sm.OLS(y.to_numpy(), x.to_numpy()).fit(cov_type="HAC", cov_kwds={"maxlags": 2})
        assert list(reg.params.index) == ["a"]
        np.testing.assert_allclose(reg.bse.to_numpy(), ref.bse)
        assert reg.cov_type == "HAC"
        np.testing.assert_allclose(reg.predict(x).to_numpy(), ref.fittedvalues)

    def test_predict_and_summary(self) -> None:
        idx = pd.period_range("2000Q1", periods=10, freq="Q")
        x = pd.DataFrame({"a": np.arange(10.0)}, index=idx)
        reg = fit_bridge_regression((1 + 3 * x["a"]).rename(None), x)
        assert reg.target == "y"
        new = pd.DataFrame({"a": [1.0, np.nan], "other": [0.0, 0.0]})
        pred = reg.predict(new)
        assert pred.iloc[0] == pytest.approx(4.0)
        assert np.isnan(pred.iloc[1])
        with pytest.raises(NowcastDataError, match="missing"):
            reg.predict(pd.DataFrame({"b": [1.0]}))
        assert "OLS Regression Results" in reg.summary()
        assert "BridgeRegression" in repr(reg)

    def test_errors(self) -> None:
        idx = pd.period_range("2000Q1", periods=3, freq="Q")
        x = pd.DataFrame({"a": [1.0, 2.0, 3.0]}, index=idx)
        y = pd.Series([1.0, 2.0, np.nan], index=idx)
        with pytest.raises(NowcastDataError, match="at least 3"):
            fit_bridge_regression(y, x)
        with pytest.raises(ValueError, match="cov_type"):
            fit_bridge_regression(y, x, cov_type="robust")
        with pytest.raises(NowcastDataError, match="at least one regressor"):
            fit_bridge_regression(y, x.iloc[:, :0], add_constant=False)
        x2 = pd.DataFrame({"a": np.arange(6.0), "b": 2 * np.arange(6.0)})
        x2.index = pd.period_range("2000Q1", periods=6, freq="Q")
        y2 = pd.Series(np.arange(6.0) ** 2, index=x2.index)
        with pytest.raises(NowcastDataError, match="collinear"):
            fit_bridge_regression(y2, x2)

    def test_non_period_index_is_converted(self) -> None:
        idx = pd.date_range("2000-03-31", periods=8, freq="QE")
        x = pd.DataFrame({"a": np.arange(8.0) ** 1.5}, index=idx)
        reg = fit_bridge_regression(pd.Series(np.arange(8.0), index=idx), x)
        assert isinstance(reg.sample, pd.PeriodIndex)


# ---------------------------------------------------------------------- helpers
class TestAggregationWeights:
    @pytest.mark.parametrize(
        ("spec", "normalize", "expected"),
        [
            (None, "ratio", [1 / 3, 2 / 3, 1.0, 2 / 3, 1 / 3]),
            (None, "sum", [1 / 9, 2 / 9, 3 / 9, 2 / 9, 1 / 9]),
            (None, "none", [1.0, 2.0, 3.0, 2.0, 1.0]),
            ("average", "ratio", [1 / 3] * 3),
            ("flow", "ratio", [1.0] * 3),
            ("flow", "sum", [1 / 3] * 3),
            ("stock", "sum", [1.0, 0.0, 0.0]),
            (AggregationType.AVERAGE, "none", [1 / 3] * 3),
            ([0.5, 0.5], "sum", [0.5, 0.5]),
            (np.array([2.0]), "ratio", [2.0]),
        ],
    )
    def test_weights(self, spec: object, normalize: str, expected: list[float]) -> None:
        w = resolve_aggregation_weights(spec, M, Q, normalize=normalize)  # type: ignore[arg-type]
        np.testing.assert_allclose(w, expected)

    def test_same_frequency(self) -> None:
        np.testing.assert_array_equal(resolve_aggregation_weights("average", M, M), [1.0])

    def test_quarter_to_year(self) -> None:
        assert resolve_aggregation_weights("mm", Q, A).size == 7

    def test_errors(self) -> None:
        with pytest.raises(ValueError, match="normalize"):
            resolve_aggregation_weights(None, M, Q, normalize="max")
        for bad in ([], [np.nan], [0.0, 0.0]):
            with pytest.raises(ValueError, match="Explicit"):
                resolve_aggregation_weights(bad, M, Q)
        with pytest.raises(ValueError):
            resolve_aggregation_weights("median", M, Q)


class TestAggregateToTarget:
    def test_values(self) -> None:
        m = pd.Series(np.arange(1.0, 10.0), index=pd.period_range("2020-01", periods=9, freq="M"))
        q = pd.period_range("2019Q4", periods=5, freq="Q")
        out = aggregate_to_target(m, np.array([1.0, 2.0, 3.0, 2.0, 1.0]), q)
        assert np.isnan(out.iloc[0])  # outside the series
        assert np.isnan(out.iloc[1])  # incomplete window (needs 5 months)
        assert out.iloc[2] == pytest.approx(6 + 2 * 5 + 3 * 4 + 2 * 3 + 2)
        assert np.isnan(out.iloc[4])
        assert out.index.equals(q)

    def test_requires_period_index(self) -> None:
        with pytest.raises(NowcastDataError, match="PeriodIndex"):
            aggregate_to_target(pd.Series([1.0]), np.ones(1), pd.period_range("2020Q1", periods=1))

    @settings(max_examples=50, deadline=None)
    @given(
        st.lists(st.floats(-100, 100), min_size=12, max_size=12),
        st.lists(st.floats(-100, 100), min_size=12, max_size=12),
        st.floats(-5, 5),
    )
    def test_linearity(self, a: list[float], b: list[float], c: float) -> None:
        idx = pd.period_range("2020-01", periods=12, freq="M")
        q = pd.period_range("2020Q1", periods=4, freq="Q")
        w = np.array([1.0, 2.0, 3.0, 2.0, 1.0]) / 9
        sa, sb = pd.Series(a, index=idx), pd.Series(b, index=idx)
        lhs = aggregate_to_target(sa + c * sb, w, q)
        rhs = aggregate_to_target(sa, w, q) + c * aggregate_to_target(sb, w, q)
        np.testing.assert_allclose(lhs.to_numpy(), rhs.to_numpy(), atol=1e-8)


class TestArExtend:
    def test_deterministic_ar1(self) -> None:
        x = 0.5 ** np.arange(10.0)
        out = ar_extend(x, 3, ar_lags=1)
        np.testing.assert_allclose(out[10:], 0.5 ** np.arange(10.0, 13.0), atol=1e-12)
        np.testing.assert_array_equal(out[:10], x)

    def test_matches_ols_forecast(self, rng: np.random.Generator) -> None:
        x = rng.standard_normal(60).cumsum() * 0.1
        out = ar_extend(x, 2, ar_lags=2)
        design = np.column_stack([np.ones(58), x[1:59], x[0:58]])
        coef, *_ = np.linalg.lstsq(design, x[2:], rcond=None)
        f1 = coef[0] + coef[1] * x[-1] + coef[2] * x[-2]
        f2 = coef[0] + coef[1] * f1 + coef[2] * x[-1]
        np.testing.assert_allclose(out[60:], [f1, f2], atol=1e-12)

    def test_trailing_nan_and_mean(self) -> None:
        x = np.array([1.0, 3.0, np.nan, 2.0, np.nan, np.nan])
        out = ar_extend(x, 2, ar_lags=0)
        np.testing.assert_allclose(out, [1.0, 3.0, np.nan, 2.0, 2.0, 2.0])
        np.testing.assert_array_equal(ar_extend(x, 0)[:4], x[:4])
        assert ar_extend(x, 0).size == 4

    def test_interior_gap_in_lags(self) -> None:
        x = np.array([1.0, 2.0, 1.5, 2.5, 1.8, 2.2, 1.9, np.nan, 2.0])
        out = ar_extend(x, 1, ar_lags=2)
        assert np.isfinite(out[-1])

    def test_errors(self) -> None:
        with pytest.raises(NowcastDataError, match="without observations"):
            ar_extend(np.array([np.nan, np.nan]), 1)
        with pytest.raises(NowcastDataError, match="Too few"):
            ar_extend(np.array([1.0, 2.0, 3.0]), 1, ar_lags=2)

    @settings(max_examples=40, deadline=None)
    @given(
        st.lists(st.floats(-50, 50), min_size=8, max_size=30),
        st.integers(0, 6),
        st.integers(0, 2),
    )
    def test_property_keeps_history(self, values: list[float], n_ahead: int, p: int) -> None:
        x = np.asarray(values)
        out = ar_extend(x, n_ahead, ar_lags=p)
        assert out.size == x.size + n_ahead
        np.testing.assert_array_equal(out[: x.size], x)
        assert np.isfinite(out).all()


# ---------------------------------------------------------------------- estimator
class TestBridgeEquation:
    def test_exact_recovery(self, rng: np.random.Generator) -> None:
        data, _ = _quarterly_panel(rng, ragged=(0, 0))
        res = BridgeEquation(horizon=0).fit(data, "y")
        assert isinstance(res, BridgeResults)
        np.testing.assert_allclose(
            res.bridge.params.to_numpy() if res.bridge else [], [1.0, 2.0, -1.0], atol=1e-10
        )
        np.testing.assert_allclose(res.coefficients.to_numpy(), [1.0, 2.0, -1.0], atol=1e-10)
        # complete last quarter: nowcast equals the true value
        frame = data.to_frame()
        truth = 1 + 2 * frame["x1"].iloc[-3:].mean() - frame["x2"].iloc[-3:].mean()
        assert res.get_nowcast() == pytest.approx(truth)
        assert str(res.nowcast.index[-1]) == "2009Q4"
        np.testing.assert_allclose(
            res.in_sample.dropna().to_numpy(), res.observed.dropna().to_numpy(), atol=1e-10
        )

    def test_ragged_edge_uses_ar_forecasts(self, rng: np.random.Generator) -> None:
        data, _ = _quarterly_panel(rng, ragged=(1, 2))
        res = BridgeEquation(horizon=1, ar_lags=1).fit(data, "y")
        x1 = ar_extend(data["x1"].to_numpy(), 4, ar_lags=1)
        x2 = ar_extend(data["x2"].to_numpy(), 5, ar_lags=1)
        nowcast = 1 + 2 * x1[117:120].mean() - x2[117:120].mean()
        forecast = 1 + 2 * x1[120:123].mean() - x2[120:123].mean()
        assert res.get_nowcast("2009Q4") == pytest.approx(nowcast, abs=1e-9)
        assert res.get_nowcast("2010Q1") == pytest.approx(forecast, abs=1e-9)
        assert res.regressors is not None
        assert list(res.regressors.columns) == ["x1", "x2"]
        assert res.nowcast.index[-1] == pd.Period("2010Q1", freq="Q")
        assert res.info["n_obs"] == 39

    def test_fill_none(self, rng: np.random.Generator) -> None:
        data, _ = _quarterly_panel(rng)
        res = BridgeEquation(fill_method="none").fit(data, "y")
        assert np.isnan(res.get_nowcast("2009Q4"))
        assert np.isnan(res.get_nowcast("2010Q1"))

    def test_mariano_murasawa_aggregation(self, rng: np.random.Generator) -> None:
        data, _ = _quarterly_panel(rng, noise=0.1)
        res = BridgeEquation(aggregation="mariano_murasawa").fit(data, "y ~ x1")
        assert res.regressors is not None
        frame = data.to_frame()
        w = np.array([1, 2, 3, 2, 1]) / 3
        expected = float(frame["x1"].iloc[109:114][::-1].to_numpy() @ w)
        assert res.regressors.loc["2009Q2", "x1"] == pytest.approx(expected)
        assert list(res.bridge.params.index) == ["const", "x1"]  # type: ignore[union-attr]

    def test_lags(self, rng: np.random.Generator) -> None:
        data, _ = _quarterly_panel(rng, noise=0.3)
        res = BridgeEquation(regressor_lags=1, target_lags=2, horizon=2).fit(data, "y")
        assert res.bridge is not None
        assert list(res.bridge.params.index) == [
            "const",
            "x1",
            "x2",
            "x1_lag1",
            "x2_lag1",
            "y_lag1",
            "y_lag2",
        ]
        # iterated forecast: the 2010Q2 forecast uses the 2010Q1 and 2009Q4 predictions
        assert res.regressors is not None
        row = res.regressors.loc["2010Q2"].copy()
        row["y_lag1"] = res.get_nowcast("2010Q1")
        row["y_lag2"] = res.get_nowcast("2009Q4")
        manual = res.bridge.params["const"] + float(row @ res.bridge.params.iloc[1:])
        assert res.get_nowcast("2010Q2") == pytest.approx(manual)
        # in-sample periods use the observed lag
        row = res.regressors.loc["2005Q1"]
        manual = res.bridge.params["const"] + float(row @ res.bridge.params.iloc[1:])
        assert res.nowcast.loc["2005Q1", "in_sample"] == pytest.approx(manual)

    def test_pure_autoregression(self, rng: np.random.Generator) -> None:
        idx = pd.period_range("2000Q1", periods=40, freq="Q")
        y = np.zeros(40)
        for t in range(1, 40):
            y[t] = 0.5 * y[t - 1] + rng.standard_normal()
        df = pd.DataFrame({"y": y}, index=idx)
        res = BridgeEquation(target_lags=1, horizon=2).fit(df, "y", frequency="Q")
        assert res.bridge is not None
        assert list(res.bridge.params.index) == ["const", "y_lag1"]
        assert res.nowcast.index[-1] == pd.Period("2010Q2", freq="Q")

    def test_monthly_target_and_same_frequency_regressor(self, rng: np.random.Generator) -> None:
        idx = pd.period_range("2000-01", periods=60, freq="M")
        x = rng.standard_normal(60)
        y = 0.5 + 1.5 * x
        y[-1] = np.nan
        x[-1] = np.nan
        df = pd.DataFrame({"x": x, "y": y}, index=idx)
        res = BridgeEquation(horizon=0).fit(df, "y", frequency="M")
        assert res.target_frequency is Frequency.MONTHLY
        x_ext = ar_extend(x, 1, ar_lags=1)
        assert res.get_nowcast() == pytest.approx(0.5 + 1.5 * x_ext[-1])

    def test_quarterly_predictor(self, rng: np.random.Generator) -> None:
        idx = pd.period_range("2000-01", periods=60, freq="M")
        q = pd.Series(rng.standard_normal(60), index=idx).where(idx.month % 3 == 0)
        y = (2.0 * q).copy()
        df = pd.DataFrame({"q": q, "y": y}, index=idx)
        res = BridgeEquation().fit(df, "y", frequency={"q": "Q", "y": "Q"})
        assert res.bridge is not None
        assert res.bridge.params["q"] == pytest.approx(2.0)

    def test_summary_and_params(self, rng: np.random.Generator) -> None:
        data, _ = _quarterly_panel(rng, noise=0.2)
        model = BridgeEquation()
        res = model.fit(data, "y")
        text = res.summary()
        assert "Bridge equation" in text
        assert "R-squared" in text
        assert res.model_name == "BridgeEquation"
        assert res.params["sigma"] == pytest.approx(res.bridge.sigma)  # type: ignore[union-attr]
        assert model.is_fitted
        assert model.clone().get_params() == model.get_params()

    def test_results_without_bridge(self) -> None:
        idx = pd.period_range("2020Q1", periods=1, freq="Q")
        frame = pd.DataFrame(
            {"observed": [1.0], "in_sample": [1.0], "out_of_sample": [np.nan]}, index=idx
        )
        res = BridgeResults(target="y", nowcast=frame)
        with pytest.raises(ValueError, match="no bridge"):
            _ = res.coefficients
        assert "Bridge equation" not in res.summary()

    def test_errors(self, rng: np.random.Generator) -> None:
        data, _ = _quarterly_panel(rng)
        with pytest.raises(NowcastDataError, match="at least one predictor"):
            BridgeEquation().fit(data.select(["y"]), "y")
        empty = data.with_data(data.to_frame().assign(y=np.nan))
        with pytest.raises(NowcastDataError, match="no observations"):
            BridgeEquation().fit(empty, "y")
        no_x = data.with_data(data.to_frame().assign(x2=np.nan))
        with pytest.raises(NowcastDataError, match="no observations"):
            BridgeEquation().fit(no_x, "y")
        with pytest.raises(TypeError, match="Unexpected"):
            BridgeEquation().fit(data, "y", foo=1)
        with pytest.raises(ModelNotFittedError):
            _ = BridgeEquation().results_

    def test_lower_frequency_predictor_error(self, rng: np.random.Generator) -> None:
        idx = pd.period_range("2000-01", periods=48, freq="M")
        a = pd.Series(rng.standard_normal(48), index=idx).where(idx.month == 12)
        y = pd.Series(rng.standard_normal(48), index=idx).where(idx.month % 3 == 0)
        df = pd.DataFrame({"a": a, "y": y})
        with pytest.raises(NowcastDataError, match="lower frequency"):
            BridgeEquation().fit(df, "y", frequency={"a": "A", "y": "Q"})

    @pytest.mark.parametrize(
        ("param", "value"),
        [
            ("regressor_lags", -1),
            ("target_lags", 1.5),
            ("ar_lags", True),
            ("horizon", -2),
            ("fill_method", "spline"),
            ("cov_type", "robust"),
            ("add_constant", 1),
            ("aggregation", 3.0),
        ],
    )
    def test_invalid_params(self, rng: np.random.Generator, param: str, value: object) -> None:
        data, _ = _quarterly_panel(rng)
        with pytest.raises(ValueError):
            BridgeEquation(**{param: value}).fit(data, "y")  # type: ignore[arg-type]

    def test_explicit_weights_and_none_aggregation(self, rng: np.random.Generator) -> None:
        data, _ = _quarterly_panel(rng, ragged=(0, 0), drop_last_quarter=False)
        res = BridgeEquation(aggregation=[0.0, 0.0, 1.0], horizon=0).fit(data, "y")
        assert res.regressors is not None
        assert res.regressors.loc["2000Q2", "x1"] == pytest.approx(data["x1"].iloc[3])
        res2 = BridgeEquation(aggregation=None, horizon=0).fit(data, "y")  # type: ignore[arg-type]
        assert res2.regressors is not None
        assert np.isnan(res2.regressors.loc["2000Q1", "x1"])  # MM needs 5 months
