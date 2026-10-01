"""Tests of BridgeBenchmark and the scikit-learn adapter (I11)."""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge

from nowcastbox.benchmarks import BridgeBenchmark, SklearnBenchmark, clone_estimator
from nowcastbox.core.data import MixedFrequencyData
from nowcastbox.core.exceptions import ModelNotFittedError, NowcastDataError
from nowcastbox.models import BridgeEquation
from tests.benchmarks.conftest import FREQ, simulate_panel

PERIODS = ["2011Q4", "2012Q1", "2012Q2", "2012Q3", "2013Q1"]


class TestBridgeBenchmark:
    def test_matches_bridge_equation(self, ragged) -> None:
        bench = BridgeBenchmark(ar_lags=2).fit(ragged, "y")
        ref = BridgeEquation(ar_lags=2, horizon=3).fit(ragged, "y ~ x + z")
        out = bench.predict(PERIODS)
        expected = ref.estimate.reindex(pd.PeriodIndex(PERIODS, freq="Q"))
        np.testing.assert_allclose(out.to_numpy(), expected.to_numpy(), atol=1e-12)
        pd.testing.assert_series_equal(bench.coef_, ref.coefficients, check_names=False)
        assert bench.results_.nowcast.index[-1] == pd.Period("2013Q1", "Q")

    def test_horizon_cache_and_predictors(self, ragged) -> None:
        bench = BridgeBenchmark(predictors=["x"], target_lags=1).fit(ragged, "y")
        first = bench.results_
        bench.predict(["2012Q2"])
        assert bench.results_ is first
        bench.predict(["2012Q4"])
        assert bench.results_ is not first
        assert list(bench.coef_.index) == ["const", "x", "y_lag1"]
        assert np.isnan(bench.predict(["1990Q1"]).iloc[0])

    def test_validation(self, panel) -> None:
        with pytest.raises(ValueError, match="ar_lags"):
            BridgeBenchmark(ar_lags=-1).fit(panel, "y")
        with pytest.raises(NowcastDataError):
            BridgeBenchmark(predictors=["nope"]).fit(panel, "y")
        with pytest.raises(ModelNotFittedError):
            _ = BridgeBenchmark().results_

    def test_refit_resets(self, ragged, panel) -> None:
        bench = BridgeBenchmark().fit(ragged, "y")
        bench.fit(panel, "y")
        assert bench.results_.data is not None
        assert bench.results_.data.end == panel.end


class TestSklearnBenchmark:
    def test_linear_regression_equals_bridge(self, ragged) -> None:
        bench = SklearnBenchmark(LinearRegression()).fit(ragged, "y")
        bridge = BridgeBenchmark().fit(ragged, "y")
        np.testing.assert_allclose(bench.predict(PERIODS), bridge.predict(PERIODS), atol=1e-10)
        assert bench.feature_names_ == ["x", "z"]
        assert bench.n_obs_ == 49

    def test_target_and_regressor_lags_equal_bridge(self, ragged) -> None:
        kwargs = {"regressor_lags": 1, "target_lags": 1}
        bench = SklearnBenchmark(LinearRegression(), **kwargs).fit(ragged, "y")
        bridge = BridgeBenchmark(**kwargs).fit(ragged, "y")
        np.testing.assert_allclose(bench.predict(PERIODS), bridge.predict(PERIODS), atol=1e-10)
        assert bench.feature_names_ == ["x", "z", "x_lag1", "z_lag1", "y_lag1"]

    def test_nonlinear_estimator_and_clone(self, ragged) -> None:
        forest = RandomForestRegressor(n_estimators=10, random_state=0)
        bench = SklearnBenchmark(forest, predictors=["x"], label="RF").fit(ragged, "y")
        assert bench.name == "RF"
        assert bench.estimator_ is not forest
        assert not hasattr(forest, "estimators_")
        assert np.isfinite(bench.predict(["2012Q2"])).all()

    def test_custom_estimator_without_sklearn_api(self, ragged) -> None:
        class MeanModel:
            def fit(self, X, y):
                self.mean = float(np.mean(y))
                return self

            def predict(self, X):
                return np.full(len(X), self.mean)

        bench = SklearnBenchmark(MeanModel()).fit(ragged, "y")
        assert bench.name == "Sklearn(MeanModel)"
        expected = ragged.to_native("y").dropna().mean()
        assert bench.predict(["2012Q2"]).iloc[0] == pytest.approx(expected)

    def test_invalid_estimator(self, panel) -> None:
        with pytest.raises(TypeError, match="predict"):
            SklearnBenchmark(object()).fit(panel, "y")
        with pytest.raises(ValueError, match="target_lags"):
            SklearnBenchmark(Ridge(), target_lags=-1).fit(panel, "y")

    def test_too_few_rows(self) -> None:
        frame = simulate_panel(n_months=6)
        with pytest.raises(NowcastDataError, match="Too few"):
            SklearnBenchmark(Ridge(), target_lags=1).fit(MixedFrequencyData(frame, FREQ), "y")

    def test_not_fitted(self) -> None:
        bench = SklearnBenchmark(Ridge())
        for attr in ("estimator_", "feature_names_", "n_obs_"):
            with pytest.raises(ModelNotFittedError):
                getattr(bench, attr)

    def test_all_nan_features_give_nan(self, ragged) -> None:
        bench = SklearnBenchmark(Ridge(), regressor_lags=2).fit(ragged, "y")
        out = bench.predict(["2000Q1", "2000Q2"])
        assert np.isnan(out).all()
        empty = pd.DataFrame({"a": [np.nan, np.nan]})
        assert np.isnan(bench._predict_frame(empty)).all()


class TestCloneEstimator:
    def test_sklearn_clone(self) -> None:
        est = Ridge(alpha=3.0).fit(np.ones((3, 1)), np.arange(3.0))
        copy = clone_estimator(est)
        assert copy.alpha == 3.0
        assert not hasattr(copy, "coef_")

    def test_fallback_without_sklearn(self, monkeypatch) -> None:
        monkeypatch.setitem(sys.modules, "sklearn.base", None)
        est = Ridge(alpha=2.0).fit(np.ones((3, 1)), np.arange(3.0))
        copy = clone_estimator(est)
        assert copy is not est
        assert hasattr(copy, "coef_")

    def test_fallback_for_non_sklearn_objects(self) -> None:
        obj = {"a": [1]}
        copy = clone_estimator(obj)
        assert copy == obj
        assert copy is not obj
