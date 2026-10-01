"""Tests of nowcastbox.models.two_step (TwoStepDFM, Giannone-Reichlin-Small 2008)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from nowcastbox.core.data import MixedFrequencyData
from nowcastbox.core.exceptions import DataQualityWarning, ModelNotFittedError, NowcastDataError
from nowcastbox.core.results import FactorResults, NowcastResults
from nowcastbox.models import TwoStepDFM, TwoStepResults
from nowcastbox.models.bridge import BridgeRegression
from nowcastbox.models.two_step import AGGREGATE_OPTIONS, simulate_two_step_example
from nowcastbox.preprocessing.aggregation import rolling_aggregate
from nowcastbox.statespace import StateSpace


def _span_correlation(estimated: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Correlation of each true factor with its projection on the estimated factors."""
    ok = np.isfinite(estimated).all(axis=1) & np.isfinite(true).all(axis=1)
    x, y = estimated[ok], true[ok]
    x = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    fitted = x @ beta
    return np.array([np.corrcoef(fitted[:, k], y[:, k])[0, 1] for k in range(y.shape[1])])


@pytest.fixture(scope="module")
def sim1() -> tuple[MixedFrequencyData, pd.DataFrame]:
    out = simulate_two_step_example(random_state=11, n_series=15, return_factors=True)
    assert isinstance(out, tuple)
    return out


@pytest.fixture(scope="module")
def fitted1(sim1: tuple[MixedFrequencyData, pd.DataFrame]) -> TwoStepResults:
    res = TwoStepDFM(n_factors=1, factor_lags=1).fit(sim1[0], "gdp")
    assert isinstance(res, TwoStepResults)
    return res


# ---------------------------------------------------------------------- recovery
class TestFactorRecovery:
    @pytest.mark.parametrize("n_factors", [1, 2, 3])
    def test_factors_span_true_factors(self, n_factors: int) -> None:
        data, true = simulate_two_step_example(
            n_periods=240,
            n_series=30,
            n_factors=n_factors,
            random_state=100 + n_factors,
            return_factors=True,
        )
        res = TwoStepDFM(n_factors=n_factors, factor_lags=1).fit(data, "gdp")
        assert res.factors is not None
        est = res.factors.loc[true.index].to_numpy()
        corr = _span_correlation(est, true.to_numpy())
        assert np.all(corr > 0.9), corr
        # the reverse projection also holds: no spurious estimated factor
        assert np.all(_span_correlation(true.to_numpy(), est) > 0.9)

    def test_ragged_edge_factors_are_accurate(self) -> None:
        data, true = simulate_two_step_example(
            n_periods=240,
            n_series=30,
            ragged_edge=(0, 1, 2, 3, 4, 5),
            random_state=7,
            return_factors=True,
        )
        res = TwoStepDFM(n_factors=1).fit(data, "gdp")
        assert res.factors is not None
        est = res.factors.loc[true.index, "f1"].to_numpy()
        # last 5 months are partially observed; the smoother still tracks the factor
        assert np.corrcoef(est, true["f1"])[0, 1] > 0.95
        err_edge = np.abs(est[-5:] - _scale(est, true["f1"].to_numpy())[-5:])
        assert np.all(np.isfinite(err_edge))

    def test_variables_mode_recovers_aggregated_factors(self) -> None:
        data, true = simulate_two_step_example(
            n_periods=240, n_series=30, n_factors=2, random_state=5, return_factors=True
        )
        res = TwoStepDFM(n_factors=2, aggregate="variables").fit(data, "gdp")
        w = np.array([1.0, 2.0, 3.0, 2.0, 1.0]) / 3
        agg_true = np.column_stack([rolling_aggregate(true[c].to_numpy(), w) for c in true])
        assert res.factors is not None
        est = res.factors.loc[true.index].to_numpy()
        assert np.all(_span_correlation(est, agg_true) > 0.9)

    def test_large_panel_smoothed_close_to_pca(self) -> None:
        data, _ = simulate_two_step_example(
            n_periods=240, n_series=80, ragged_edge=(0,), random_state=3, return_factors=True
        )
        res = TwoStepDFM(n_factors=1, horizon=0).fit(data, "gdp")
        z, _ = data.drop("gdp").standardize()
        x = z.to_frame().to_numpy()
        pca = x @ res.loadings.to_numpy()  # type: ignore[union-attr]
        assert res.factors is not None
        assert np.corrcoef(pca[:, 0], res.factors["f1"].iloc[: len(x)])[0, 1] > 0.99

    def test_nowcast_accuracy(self) -> None:
        data = simulate_two_step_example(n_periods=240, n_series=30, random_state=21)
        assert isinstance(data, MixedFrequencyData)
        res = TwoStepDFM(n_factors=1).fit(data, "gdp")
        assert res.bridge is not None
        assert res.bridge.rsquared > 0.8
        assert res.bridge.params["f1"] != 0


def _scale(est: np.ndarray, true: np.ndarray) -> np.ndarray:
    beta = np.polyfit(true, est, 1)
    return np.polyval(beta, true)


# ---------------------------------------------------------------------- output contract
class TestResultsContract:
    def test_types_and_shapes(
        self, sim1: tuple[MixedFrequencyData, pd.DataFrame], fitted1: TwoStepResults
    ) -> None:
        data = sim1[0]
        res = fitted1
        assert isinstance(res, FactorResults)
        assert isinstance(res, NowcastResults)
        assert res.model_name == "TwoStepDFM"
        assert res.target == "gdp"
        # nowcast frame: quarterly, through the next quarter (horizon=1)
        assert res.nowcast.index.freqstr.startswith("Q")
        assert str(res.nowcast.index[0]) == "2000Q1"
        assert str(res.nowcast.index[-1]) == "2015Q1"
        assert list(res.nowcast.columns) == ["observed", "in_sample", "out_of_sample", "std"]
        # factors on the extended monthly grid
        assert res.factors is not None
        assert list(res.factors.columns) == ["f1"]
        assert isinstance(res.factors.index, pd.PeriodIndex)
        assert str(res.factors.index[0]) == "2000-01"
        assert str(res.factors.index[-1]) == "2015-03"
        assert not res.factors.isna().any().any()
        # loadings and parameters
        predictors = [c for c in data.columns if c != "gdp"]
        assert res.loadings is not None
        assert list(res.loadings.index) == predictors
        assert res.A.shape == (1, 1)
        assert res.B.shape == (1, 1)
        np.testing.assert_allclose(res.BB, res.B @ res.B.T)
        assert list(res.Psi.index) == predictors
        assert (res.Psi > 0).all()
        assert res.eigenvalues is not None
        assert len(res.eigenvalues) == len(predictors)
        assert res.eigenvalues.is_monotonic_decreasing
        assert res.explained_variance_ratio.sum() == pytest.approx(1.0)
        assert res.factor_lags == 1
        assert res.n_shocks == 1
        assert res.aggregate == "factors"
        np.testing.assert_allclose(res.aggregation_weights, np.array([1, 2, 3, 2, 1]) / 9)  # type: ignore[arg-type]
        assert isinstance(res.state_space, StateSpace)
        assert res.state_space.n_states == 5  # factor and 4 lags for the MM weights
        assert isinstance(res.bridge, BridgeRegression)
        assert res.loglikelihood is not None
        assert np.isfinite(res.loglikelihood)
        assert res.standardization is not None
        assert res.data is not None
        assert res.data.equals(data)
        for key in ("A", "B", "Psi", "loadings", "eigenvalues", "residual_cov"):
            assert key in res.params
        assert res.info["n_balanced_periods"] == 178
        assert "fit_time" in res.info
        assert res.transition_matrices()[0].shape == (1, 1)

    def test_exactly_one_estimate_column(self, fitted1: TwoStepResults) -> None:
        nc = fitted1.nowcast.dropna(subset=["in_sample", "out_of_sample"], how="all")
        both = nc["in_sample"].notna() & nc["out_of_sample"].notna()
        assert not both.any()
        assert nc["out_of_sample"].notna().sum() == 2  # nowcast + one-quarter forecast
        assert np.isnan(fitted1.nowcast["in_sample"].iloc[0])  # MM needs 5 months

    def test_in_sample_equals_bridge_fit(self, fitted1: TwoStepResults) -> None:
        assert fitted1.bridge is not None
        fitted = fitted1.bridge.fitted_values
        np.testing.assert_allclose(
            fitted1.nowcast.loc[fitted.index, "in_sample"].to_numpy(), fitted.to_numpy()
        )

    def test_aggregated_factors(self, fitted1: TwoStepResults) -> None:
        agg = fitted1.aggregated_factors
        assert agg is not None
        assert fitted1.factors is not None
        assert agg.index.equals(fitted1.nowcast.index)
        w = np.array([1, 2, 3, 2, 1]) / 9
        manual = rolling_aggregate(fitted1.factors["f1"].to_numpy(), w)
        q_end = fitted1.factors.index.month % 3 == 0
        np.testing.assert_allclose(agg["f1"].to_numpy()[1:], manual[q_end][1:], atol=1e-10)

    def test_std_column(self, fitted1: TwoStepResults) -> None:
        std = fitted1.nowcast["std"]
        assert fitted1.bridge is not None
        sigma = fitted1.bridge.sigma
        assert (std.dropna() >= sigma - 1e-12).all()
        # filtering uncertainty grows with the horizon
        assert std.iloc[-1] > std.iloc[-2] > std.iloc[-3]

    def test_x_forecast_and_filled(
        self, sim1: tuple[MixedFrequencyData, pd.DataFrame], fitted1: TwoStepResults
    ) -> None:
        data = sim1[0]
        x_fc = fitted1.x_forecast
        x_filled = fitted1.x_filled
        assert x_fc is not None
        assert x_filled is not None
        assert x_fc.shape == (183, 15)
        assert not x_filled.isna().any().any()
        observed = data.to_frame().drop(columns="gdp")
        mask = observed.notna()
        pd.testing.assert_frame_equal(
            x_filled.loc[observed.index][mask], observed[mask], check_freq=False
        )
        # ragged edge filled with the model forecast in original units
        assert x_filled.loc["2014-12", "x2"] == pytest.approx(x_fc.loc["2014-12", "x2"])
        # common component explains much of each series (correlation with the data)
        assert np.corrcoef(x_fc["x1"].iloc[:170], observed["x1"].iloc[:170])[0, 1] > 0.6

    def test_summary(self, fitted1: TwoStepResults) -> None:
        text = fitted1.summary()
        for section in ("Principal components", "Bridge equation", "Factor dynamics"):
            assert section in text
        assert "Variance explained" in text

    def test_results_missing_fields(self) -> None:
        idx = pd.period_range("2020Q1", periods=1, freq="Q")
        frame = pd.DataFrame(
            {"observed": [1.0], "in_sample": [1.0], "out_of_sample": [np.nan]}, index=idx
        )
        res = TwoStepResults(target="y", nowcast=frame)
        for attr in ("A", "B", "BB", "Psi", "explained_variance_ratio"):
            with pytest.raises(ValueError):
                getattr(res, attr)
        assert "Bridge equation" not in res.summary()

    def test_save_load(self, fitted1: TwoStepResults, tmp_path: object) -> None:
        import pathlib

        path = fitted1.save(pathlib.Path(str(tmp_path)) / "two_step.pkl")
        loaded = TwoStepResults.load(path)
        pd.testing.assert_frame_equal(loaded.nowcast, fitted1.nowcast)


# ---------------------------------------------------------------------- variants
class TestVariants:
    def test_variables_mode(self, sim1: tuple[MixedFrequencyData, pd.DataFrame]) -> None:
        res = TwoStepDFM(n_factors=1, aggregate="2s").fit(sim1[0], "gdp")
        assert res.aggregate == "variables"
        np.testing.assert_array_equal(res.aggregation_weights, [1.0])  # type: ignore[arg-type]
        assert res.state_space is not None
        assert res.state_space.n_states == 1
        assert res.x_forecast is not None
        # the model panel holds the filtered (MM-aggregated) variables
        x1 = sim1[0]["x1"].to_numpy()
        filt = rolling_aggregate(x1, np.array([1, 2, 3, 2, 1]) / 3)
        assert res.x_filled is not None
        np.testing.assert_allclose(res.x_filled["x1"].to_numpy()[4:180], filt[4:], atol=1e-10)
        assert res.bridge is not None
        assert res.bridge.rsquared > 0.5

    @pytest.mark.parametrize("alias", sorted(AGGREGATE_OPTIONS))
    def test_aliases(self, sim1: tuple[MixedFrequencyData, pd.DataFrame], alias: str) -> None:
        res = TwoStepDFM(n_factors=1, aggregate=alias).fit(sim1[0], "gdp")  # type: ignore[arg-type]
        assert res.aggregate == AGGREGATE_OPTIONS[alias]

    def test_factor_lags_and_shocks(self, sim1: tuple[MixedFrequencyData, pd.DataFrame]) -> None:
        res = TwoStepDFM(n_factors=3, factor_lags=2, n_shocks=1).fit(sim1[0], "gdp")
        assert res.A.shape == (3, 6)
        assert res.B.shape == (3, 1)
        assert np.linalg.matrix_rank(res.BB) == 1
        assert res.n_shocks == 1
        assert len(res.transition_matrices()) == 2
        assert res.state_space is not None
        assert res.state_space.n_states == 15
        assert res.state_space.n_disturbances == 1
        # B'B = leading eigenvalue of the VAR residual covariance
        ev = np.linalg.eigvalsh(res.params["residual_cov"])
        assert float((res.B.T @ res.B)[0, 0]) == pytest.approx(ev.max())

    def test_more_lags_than_weights(self, sim1: tuple[MixedFrequencyData, pd.DataFrame]) -> None:
        res = TwoStepDFM(n_factors=1, factor_lags=6).fit(sim1[0], "gdp")
        assert res.state_space is not None
        assert res.state_space.n_states == 6

    def test_monthly_target(self) -> None:
        rng = np.random.default_rng(9)
        n = 150
        f = np.zeros(n)
        for t in range(1, n):
            f[t] = 0.8 * f[t - 1] + rng.standard_normal()
        x = np.outer(f, rng.uniform(0.5, 1.5, 12)) + 0.5 * rng.standard_normal((n, 12))
        y = 1.0 + 0.5 * f + 0.1 * rng.standard_normal(n)
        y[-2:] = np.nan
        x[-1, :6] = np.nan
        idx = pd.period_range("2005-01", periods=n, freq="M")
        df = pd.DataFrame(x, index=idx, columns=[f"x{i}" for i in range(12)])
        df["ip"] = y
        res = TwoStepDFM(n_factors=1, horizon=2).fit(df, "ip", frequency="M")
        assert res.target_frequency.value == "M"
        assert res.nowcast.index[-1] == pd.Period("2017-08", freq="M")
        assert res.nowcast["out_of_sample"].notna().sum() == 4
        assert res.bridge is not None
        assert res.bridge.rsquared > 0.9
        np.testing.assert_array_equal(res.aggregation_weights, [1.0])  # type: ignore[arg-type]
        assert res.nowcast["in_sample"].notna().sum() == n - 2

    def test_panel_ending_mid_quarter(self, sim1: tuple[MixedFrequencyData, pd.DataFrame]) -> None:
        data = sim1[0].truncate(end="2014-11")
        res = TwoStepDFM(n_factors=1, horizon=0).fit(data, "gdp")
        assert res.factors is not None
        assert str(res.factors.index[-1]) == "2014-12"
        assert str(res.nowcast.index[-1]) == "2014Q4"
        assert np.isfinite(res.get_nowcast())

    def test_quarterly_predictor(self, sim1: tuple[MixedFrequencyData, pd.DataFrame]) -> None:
        data, true = sim1
        frame = data.to_frame()
        rng = np.random.default_rng(4)
        w = np.array([1, 2, 3, 2, 1]) / 3
        q_val = rolling_aggregate(true["f1"].to_numpy(), w) + 0.1 * rng.standard_normal(180)
        frame["qx"] = pd.Series(q_val, index=frame.index).where(frame.index.month % 3 == 0)
        frame.loc["2014-12", "qx"] = np.nan
        panel = MixedFrequencyData(frame, data.frequencies.to_dict() | {"qx": "Q"})
        for mode, n_states in (("factors", 5), ("variables", 1)):
            res = TwoStepDFM(n_factors=1, aggregate=mode).fit(panel, "gdp")  # type: ignore[arg-type]
            assert res.loadings is not None
            assert "qx" in res.loadings.index
            assert res.state_space is not None
            assert res.state_space.n_states == n_states
            assert res.x_filled is not None
            assert np.isfinite(res.x_filled.loc["2014-12", "qx"])
            assert np.isnan(res.x_filled.loc["2014-11", "qx"])  # not a storage slot
        # factors mode: quarterly series loads on MM-aggregated lags of the factor
        res = TwoStepDFM(n_factors=1).fit(panel, "gdp")
        assert res.state_space is not None
        z_row = res.state_space.Z[-1]
        np.testing.assert_allclose(z_row / z_row[2], [1 / 3, 2 / 3, 1, 2 / 3, 1 / 3])

    def test_short_quarterly_predictor_is_excluded(
        self, sim1: tuple[MixedFrequencyData, pd.DataFrame]
    ) -> None:
        data = sim1[0]
        frame = data.to_frame()
        frame["qx"] = np.nan
        frame.loc["2014-09", "qx"] = 1.0
        frame.loc["2014-06", "qx"] = 2.0
        panel = MixedFrequencyData(frame, data.frequencies.to_dict() | {"qx": "Q"})
        with pytest.warns(DataQualityWarning, match="excluded"):
            res = TwoStepDFM(n_factors=1).fit(panel, "gdp")
        assert res.info["excluded_predictors"] == ["qx"]
        assert res.loadings is not None
        assert "qx" not in res.loadings.index

    @pytest.mark.parametrize("method", ["univariate", "multivariate"])
    def test_filter_methods_agree(
        self,
        sim1: tuple[MixedFrequencyData, pd.DataFrame],
        fitted1: TwoStepResults,
        method: str,
    ) -> None:
        res = TwoStepDFM(n_factors=1, filter_method=method).fit(sim1[0], "gdp")  # type: ignore[arg-type]
        np.testing.assert_allclose(
            res.nowcast["out_of_sample"].dropna(),
            fitted1.nowcast["out_of_sample"].dropna(),
            atol=1e-8,
        )
        res_c = TwoStepDFM(n_factors=1, collapse=True).fit(sim1[0], "gdp")
        assert res_c.factors is not None
        assert fitted1.factors is not None
        np.testing.assert_allclose(res_c.factors, fitted1.factors, atol=1e-8)

    def test_aggregation_settings(self, sim1: tuple[MixedFrequencyData, pd.DataFrame]) -> None:
        data = sim1[0]
        res = TwoStepDFM(n_factors=1, aggregation="average").fit(data, "gdp")
        np.testing.assert_allclose(res.aggregation_weights, [1 / 3] * 3)  # type: ignore[arg-type]
        res = TwoStepDFM(n_factors=1, aggregation=[0.0, 0.0, 2.0]).fit(data, "gdp")
        np.testing.assert_allclose(res.aggregation_weights, [0.0, 0.0, 2.0])  # type: ignore[arg-type]
        # target metadata is used when aggregation is None
        meta = data.with_metadata("gdp", aggregation="stock")
        res = TwoStepDFM(n_factors=1).fit(meta, "gdp")
        np.testing.assert_allclose(res.aggregation_weights, [1.0, 0.0, 0.0])  # type: ignore[arg-type]

    def test_formula_target(self, sim1: tuple[MixedFrequencyData, pd.DataFrame]) -> None:
        res = TwoStepDFM(n_factors=1).fit(sim1[0], "gdp ~ x1 + x2 + x3 + x4")
        assert res.loadings is not None
        assert list(res.loadings.index) == ["x1", "x2", "x3", "x4"]

    def test_params_api(self) -> None:
        model = TwoStepDFM(n_factors=3, aggregate="variables")
        params = model.get_params()
        assert params["n_factors"] == 3
        assert model.clone().get_params() == params
        model.set_params(n_factors=2)
        assert model.n_factors == 2


# ---------------------------------------------------------------------- new data
class TestNewData:
    def test_update_same_data_reproduces_fit(
        self, sim1: tuple[MixedFrequencyData, pd.DataFrame]
    ) -> None:
        model = TwoStepDFM(n_factors=2, factor_lags=2)
        res = model.fit(sim1[0], "gdp")
        again = model.update(sim1[0])
        pd.testing.assert_frame_equal(again.nowcast, res.nowcast)
        assert again.factors is not None
        assert res.factors is not None
        pd.testing.assert_frame_equal(again.factors, res.factors)
        assert again.info["updated"] is True
        assert model.results_ is res

    def test_nowcast_reacts_to_new_data(
        self, sim1: tuple[MixedFrequencyData, pd.DataFrame]
    ) -> None:
        data = sim1[0]
        model = TwoStepDFM(n_factors=1)
        res = model.fit(data, "gdp")
        frame = data.to_frame()
        assert np.isnan(frame.loc["2014-12", "x2"])
        # a release far above the expectation of the model moves the nowcast
        x_fc = res.x_forecast
        assert x_fc is not None
        loading = res.loadings.loc["x2", "f1"]  # type: ignore[union-attr]
        surprise = 3.0 * float(data["x2"].std())
        frame.loc["2014-12", "x2"] = x_fc.loc["2014-12", "x2"] + surprise
        new = data.with_data(frame)
        updated = model.update(new)
        beta = res.bridge.params["f1"]  # type: ignore[union-attr]
        delta = updated.get_nowcast("2014Q4") - res.get_nowcast("2014Q4")
        assert abs(delta) > 1e-3
        assert np.sign(delta) == np.sign(beta * loading)

    def test_expected_release_does_not_move_nowcast(
        self, sim1: tuple[MixedFrequencyData, pd.DataFrame]
    ) -> None:
        """Releasing exactly the model expectation leaves the Kalman update unchanged."""
        data = sim1[0]
        model = TwoStepDFM(n_factors=2)
        res = model.fit(data, "gdp")
        frame = data.to_frame()
        x_fc = res.x_forecast
        assert x_fc is not None
        for col in ("x2", "x3"):
            frame.loc["2014-12", col] = x_fc.loc["2014-12", col]
        updated = model.update(data.with_data(frame))
        np.testing.assert_allclose(
            updated.nowcast["out_of_sample"].dropna(),
            res.nowcast["out_of_sample"].dropna(),
            atol=1e-9,
        )
        # ... but the uncertainty shrinks
        assert updated.nowcast.loc["2014Q4", "std"] < res.nowcast.loc["2014Q4", "std"]

    def test_refit_with_more_data_changes_nowcast(
        self, sim1: tuple[MixedFrequencyData, pd.DataFrame]
    ) -> None:
        data = sim1[0]
        early = data.as_of("2014-12-31", release_delays=dict.fromkeys(data.columns, 0))
        res_a = TwoStepDFM(n_factors=1).fit(early.truncate(end="2014-10"), "gdp")
        res_b = TwoStepDFM(n_factors=1).fit(data, "gdp")
        assert res_a.get_nowcast("2014Q4") != pytest.approx(res_b.get_nowcast("2014Q4"))

    def test_update_new_vintage_with_longer_grid(
        self, sim1: tuple[MixedFrequencyData, pd.DataFrame]
    ) -> None:
        data = sim1[0]
        model = TwoStepDFM(n_factors=1, horizon=0)
        model.fit(data.truncate(end="2014-09"), "gdp")
        out = model.update(data)
        assert str(out.nowcast.index[-1]) == "2014Q4"
        assert np.isfinite(out.get_nowcast("2014Q4"))

    def test_update_errors(self, sim1: tuple[MixedFrequencyData, pd.DataFrame]) -> None:
        data = sim1[0]
        model = TwoStepDFM(n_factors=1)
        with pytest.raises(ModelNotFittedError):
            model.update(data)
        model.fit(data, "gdp")
        with pytest.raises(NowcastDataError, match="lack"):
            model.update(data.drop("x1"))
        q_frame = data.to_frame()
        q_frame["x1"] = q_frame["x1"].where(q_frame.index.month % 3 == 0)
        bad = MixedFrequencyData(q_frame, data.frequencies.to_dict() | {"x1": "Q"})
        with pytest.raises(NowcastDataError, match="frequency"):
            model.update(bad)
        quarterly = MixedFrequencyData(
            pd.DataFrame(
                {c: np.arange(8.0) + i for i, c in enumerate(data.columns)},
                index=pd.period_range("2000Q1", periods=8, freq="Q"),
            ),
            "Q",
        )
        with pytest.raises(NowcastDataError, match="base frequency"):
            model.update(quarterly)


# ---------------------------------------------------------------------- warnings / errors
class TestDataProblems:
    def test_interior_missing_warns(self, sim1: tuple[MixedFrequencyData, pd.DataFrame]) -> None:
        frame = sim1[0].to_frame()
        frame.loc["2005-06", "x1"] = np.nan
        panel = sim1[0].with_data(frame)
        with pytest.warns(DataQualityWarning, match="excluded from the principal"):
            res = TwoStepDFM(n_factors=1).fit(panel, "gdp")
        assert res.info["n_balanced_periods"] == 177
        assert res.x_filled is not None
        assert np.isfinite(res.x_filled.loc["2005-06", "x1"])

    def test_psi_floor_warns(self, sim1: tuple[MixedFrequencyData, pd.DataFrame]) -> None:
        with pytest.warns(DataQualityWarning, match="floor"):
            res = TwoStepDFM(n_factors=1, idio_variance_floor=5.0).fit(sim1[0], "gdp")
        assert (res.Psi == 5.0).all()

    def test_nonstationary_var_warns(self) -> None:
        rng = np.random.default_rng(2)
        n = 120
        f = np.zeros(n)
        for t in range(1, n):
            f[t] = 1.04 * f[t - 1] + rng.standard_normal()
        x = np.outer(f, rng.uniform(0.8, 1.2, 8)) + 0.1 * rng.standard_normal((n, 8))
        idx = pd.period_range("2000-01", periods=n, freq="M")
        df = pd.DataFrame(x, index=idx, columns=[f"x{i}" for i in range(8)])
        df["y"] = pd.Series(f, index=idx).where(idx.month % 3 == 0)
        freqs = dict.fromkeys(df.columns[:-1], "M") | {"y": "Q"}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = TwoStepDFM(n_factors=1, idio_variance_floor=1e-12).fit(df, "y", frequency=freqs)
        messages = [str(w.message) for w in caught if issubclass(w.category, DataQualityWarning)]
        assert any("not stationary" in m for m in messages)
        assert res.state_space is not None
        assert not res.state_space.is_stationary
        assert np.isfinite(res.get_nowcast())

    def test_too_few_predictors(self, sim1: tuple[MixedFrequencyData, pd.DataFrame]) -> None:
        with pytest.raises(NowcastDataError, match="at least n_factors=3"):
            TwoStepDFM(n_factors=3).fit(sim1[0], "gdp ~ x1 + x2")

    def test_target_without_observations(
        self, sim1: tuple[MixedFrequencyData, pd.DataFrame]
    ) -> None:
        panel = sim1[0].with_data(sim1[0].to_frame().assign(gdp=np.nan))
        with pytest.raises(NowcastDataError, match="no observations"):
            TwoStepDFM(n_factors=1).fit(panel, "gdp")

    def test_short_balanced_panel(self, sim1: tuple[MixedFrequencyData, pd.DataFrame]) -> None:
        frame = sim1[0].to_frame()
        frame.loc[:"2014-09", "x1"] = np.nan
        panel = sim1[0].with_data(frame)
        with pytest.raises(NowcastDataError, match="balanced part"):
            TwoStepDFM(n_factors=1).fit(panel, "gdp")

    def test_unexpected_fit_option(self, sim1: tuple[MixedFrequencyData, pd.DataFrame]) -> None:
        with pytest.raises(TypeError, match="Unexpected"):
            TwoStepDFM(n_factors=1).fit(sim1[0], "gdp", tol=1e-3)

    @pytest.mark.parametrize(
        ("params", "match"),
        [
            ({"n_factors": 0}, "n_factors"),
            ({"n_factors": 1.5}, "n_factors"),
            ({"factor_lags": 0}, "factor_lags"),
            ({"horizon": -1}, "horizon"),
            ({"ddof": True}, "ddof"),
            ({"n_factors": 2, "n_shocks": 3}, "n_shocks"),
            ({"n_shocks": 0}, "n_shocks"),
            ({"aggregate": "both"}, "aggregate"),
            ({"filter_method": "fast"}, "filter_method"),
            ({"collapse": 1}, "collapse"),
            ({"idio_variance_floor": 0.0}, "idio_variance_floor"),
            ({"idio_variance_floor": "a"}, "idio_variance_floor"),
            ({"aggregation": "median"}, None),
            ({"aggregation": [0.0, 0.0]}, "Explicit"),
            ({"aggregation": [[1.0]]}, "Explicit"),
            ({"aggregation": object()}, "Invalid aggregation"),
        ],
    )
    def test_invalid_params(
        self, sim1: tuple[MixedFrequencyData, pd.DataFrame], params: dict, match: str | None
    ) -> None:
        with pytest.raises(ValueError, match=match):
            TwoStepDFM(**params).fit(sim1[0], "gdp")

    def test_enum_aggregation_is_valid(self, sim1: tuple[MixedFrequencyData, pd.DataFrame]) -> None:
        from nowcastbox.core.frequency import AggregationType

        res = TwoStepDFM(n_factors=1, aggregation=AggregationType.FLOW).fit(sim1[0], "gdp")
        np.testing.assert_allclose(res.aggregation_weights, [1 / 3] * 3)  # type: ignore[arg-type]


# ---------------------------------------------------------------------- properties
class TestProperties:
    @settings(
        max_examples=10,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    @given(
        st.permutations(list(range(15))), st.lists(st.floats(0.1, 10.0), min_size=15, max_size=15)
    )
    def test_invariance(
        self,
        sim1: tuple[MixedFrequencyData, pd.DataFrame],
        fitted1: TwoStepResults,
        order: list[int],
        scales: list[float],
    ) -> None:
        """Nowcasts do not depend on the order or the units of the predictors."""
        data = sim1[0]
        frame = data.to_frame()
        cols = [f"x{i + 1}" for i in order]
        frame[[f"x{i + 1}" for i in range(15)]] *= np.asarray(scales)
        frame = frame[[*cols, "gdp"]]
        panel = MixedFrequencyData(frame, dict.fromkeys(cols, "M") | {"gdp": "Q"})
        res = TwoStepDFM(n_factors=1).fit(panel, "gdp")
        np.testing.assert_allclose(
            res.nowcast[["in_sample", "out_of_sample"]].to_numpy(),
            fitted1.nowcast[["in_sample", "out_of_sample"]].to_numpy(),
            atol=1e-7,
        )


def test_simulation_helper_defaults() -> None:
    data = simulate_two_step_example(random_state=0)
    assert isinstance(data, MixedFrequencyData)
    assert data.n_series == 11
    last = data.last_observed()
    assert str(last["x1"]) == "2014-12"
    assert str(last["x2"]) == "2014-11"
    assert str(last["x3"]) == "2014-10"
    assert str(last["gdp"]) == "2014-09"


@pytest.mark.parametrize("aggregate", ["factors", "variables"])
def test_no_backcast_before_first_target_observation(aggregate) -> None:
    data = simulate_two_step_example(random_state=0)
    frame = data.data
    frame.loc[frame.index[:60], "gdp"] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DataQualityWarning)
        res = TwoStepDFM(n_factors=1, aggregate=aggregate).fit(data.with_data(frame), "gdp")
    nowcast = res.nowcast
    first = nowcast["observed"].first_valid_index()
    early = nowcast.loc[nowcast.index < first]
    assert len(early) > 0
    assert early["out_of_sample"].isna().all() and early["in_sample"].isna().all()
    assert np.isfinite(res.get_nowcast())
    assert bool(nowcast["out_of_sample"].notna().iloc[-1])
