"""Tests of the univariate benchmarks: AR, RandomWalk, HistoricalMean."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st
from statsmodels.tsa.ar_model import AutoReg

from nowcastbox.benchmarks import AR, HistoricalMean, RandomWalk
from nowcastbox.core.base import BenchmarkForecaster
from nowcastbox.core.exceptions import DataQualityWarning, ModelNotFittedError, NowcastDataError
from tests.benchmarks.conftest import quarterly


def simulate_ar(phi: list[float], n: int, seed: int = 0, const: float = 0.3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y = np.zeros(n + 100)
    for t in range(len(phi), y.size):
        y[t] = const + sum(p * y[t - j - 1] for j, p in enumerate(phi)) + rng.standard_normal()
    return y[100:]


class TestAR:
    @pytest.mark.reference_validation
    @pytest.mark.parametrize(("p", "trend"), [(1, "c"), (2, "c"), (3, "n")])
    def test_matches_statsmodels_autoreg(self, p: int, trend: str) -> None:
        y = simulate_ar([0.5, 0.2], 120)
        bench = AR(p=p, trend=trend).fit(quarterly(y), "y")
        ref = AutoReg(y, lags=p, trend=trend).fit()
        np.testing.assert_allclose(bench.coef_.to_numpy(), ref.params, atol=1e-10)
        periods = pd.period_range("2020Q1", periods=4, freq="Q")
        np.testing.assert_allclose(bench.predict(periods).to_numpy(), ref.forecast(4), atol=1e-10)
        in_sample = pd.period_range("1990Q1", periods=120, freq="Q")
        fitted = bench.predict(in_sample).to_numpy()
        assert np.isnan(fitted[:p]).all()
        np.testing.assert_allclose(fitted[p:], ref.fittedvalues, atol=1e-10)
        assert bench.n_obs_ == 120 - p
        assert bench.sigma2_ == pytest.approx(ref.sigma2 * (120 - p) / (120 - p - len(ref.params)))

    @pytest.mark.parametrize("criterion", ["aic", "bic"])
    def test_order_selection(self, criterion: str) -> None:
        y = simulate_ar([0.3, 0.5], 600, seed=3)
        bench = AR(p=criterion, max_p=5).fit(quarterly(y), "y")
        assert bench.order_ == 2
        assert list(bench.coef_.index) == ["const", "ar.L1", "ar.L2"]

    def test_order_zero_is_mean(self) -> None:
        y = np.array([1.0, 2.0, 3.0, 6.0])
        bench = AR(p=0).fit(quarterly(y), "y")
        assert bench.coef_.to_dict() == {"const": pytest.approx(3.0)}
        np.testing.assert_allclose(bench.predict(["1991Q2", "1990Q2"]), [3.0, 3.0])

    def test_order_zero_without_constant_predicts_zero(self) -> None:
        bench = AR(p=0, trend="n").fit(quarterly([1.0, 2.0]), "y")
        assert bench.coef_.empty
        assert bench.predict(["1991Q1"]).tolist() == [0.0]

    def test_selection_skips_unidentified_orders(self) -> None:
        bench = AR(p="aic", max_p=3, trend="n").fit(quarterly(simulate_ar([0.5], 40)), "y")
        assert 1 <= bench.order_ <= 3

    def test_selection_with_short_sample_falls_back_to_mean(self) -> None:
        bench = AR(p="bic", max_p=10).fit(quarterly([1.0, 2.0, 4.0]), "y")
        assert bench.order_ == 0

    def test_gap_at_origin_uses_mean(self) -> None:
        y = simulate_ar([0.5, 0.2], 60)
        y[-2] = np.nan
        bench = AR(p=2).fit(quarterly(y), "y")
        forecast = bench.predict(["2005Q1"]).iloc[0]
        const, phi1, phi2 = bench.coef_
        assert forecast == pytest.approx(const + phi1 * y[-1] + phi2 * np.nanmean(y))

    def test_monthly_target(self) -> None:
        y = simulate_ar([0.7], 80)
        idx = pd.period_range("2000-01", periods=80, freq="M")
        bench = AR().fit(pd.DataFrame({"y": y}, index=idx), "y", frequency="M")
        assert bench.target_frequency_.value == "M"
        assert np.isfinite(bench.predict(["2006-09", "2007-01"])).all()

    def test_ragged_target_and_periods_before_start(self, ragged) -> None:
        bench = AR().fit(ragged, "y")
        out = bench.predict(["1999Q4", "2000Q1", "2012Q2"])
        assert np.isnan(out.iloc[:2]).all()
        assert np.isfinite(out.iloc[2])
        assert str(bench.history_.index[-1]) == "2012Q1"

    @pytest.mark.parametrize(
        "kwargs",
        [{"trend": "ct"}, {"p": -1}, {"p": "hqic"}, {"max_p": -1}, {"p": True}, {"p": 1.5}],
    )
    def test_invalid_parameters(self, kwargs) -> None:
        with pytest.raises(ValueError, match="must be"):
            AR(**kwargs).fit(quarterly(np.arange(10.0)), "y")

    def test_too_few_observations(self) -> None:
        with pytest.raises(NowcastDataError, match="Too few"):
            AR(p=3).fit(quarterly([1.0, 2.0, 3.0, 4.0]), "y")

    def test_no_observation(self) -> None:
        with (
            pytest.warns(DataQualityWarning),
            pytest.raises(NowcastDataError, match="no observations"),
        ):
            AR().fit(quarterly([np.nan, np.nan]), "y")

    def test_not_fitted(self) -> None:
        bench = AR()
        assert not bench.is_fitted
        with pytest.raises(ModelNotFittedError):
            bench.predict(["2020Q1"])
        for attr in ("order_", "coef_", "sigma2_", "n_obs_", "history_"):
            with pytest.raises(ModelNotFittedError):
                getattr(bench, attr)

    def test_protocol_and_params(self) -> None:
        bench = AR(p=2)
        assert isinstance(bench, BenchmarkForecaster)
        assert bench.get_params() == {"p": 2, "max_p": 4, "trend": "c"}
        assert bench.name == "AR"
        assert repr(bench.clone()) == "AR(p=2, max_p=4, trend='c')"


class TestRandomWalk:
    def test_forecasts_and_in_sample(self) -> None:
        data = quarterly([1.0, 2.0, np.nan, 5.0])
        bench = RandomWalk().fit(data, "y")
        out = bench.predict(["1990Q1", "1990Q2", "1990Q4", "1991Q3"])
        assert np.isnan(out.iloc[0])
        assert out.iloc[1:].tolist()[0] == 1.0
        assert np.isnan(out.iloc[2])
        assert out.iloc[3] == 5.0
        assert bench.drift_ == 0.0

    def test_drift(self) -> None:
        data = quarterly([1.0, 2.0, 3.0, 4.0])
        bench = RandomWalk(drift=True).fit(data, "y")
        assert bench.drift_ == pytest.approx(1.0)
        assert bench.predict(["1991Q1", "1991Q3"]).tolist() == [5.0, 7.0]
        assert bench.predict(["1990Q3"]).tolist() == [3.0]

    def test_drift_needs_consecutive_values(self) -> None:
        with pytest.raises(NowcastDataError, match="consecutive"):
            RandomWalk(drift=True).fit(quarterly([1.0, np.nan, 3.0]), "y")

    def test_invalid_drift(self) -> None:
        with pytest.raises(ValueError, match="bool"):
            RandomWalk(drift="yes").fit(quarterly([1.0, 2.0]), "y")  # type: ignore[arg-type]

    @given(st.lists(st.floats(-1e6, 1e6), min_size=1, max_size=30), st.integers(1, 12))
    def test_no_change_property(self, values: list[float], h: int) -> None:
        bench = RandomWalk().fit(quarterly(values), "y")
        last = pd.Period("1990Q1", "Q") + (len(values) - 1)
        assert bench.predict([last + h]).iloc[0] == values[-1]


class TestHistoricalMean:
    def test_mean_and_window(self) -> None:
        data = quarterly([1.0, np.nan, 3.0, 8.0])
        assert HistoricalMean().fit(data, "y").mean_ == 4.0
        assert HistoricalMean(window=2).fit(data, "y").mean_ == 5.5
        out = HistoricalMean().fit(data, "y").predict(["1980Q1", "2000Q1"])
        assert out.tolist() == [4.0, 4.0]

    def test_invalid_window(self) -> None:
        with pytest.raises(ValueError, match="window"):
            HistoricalMean(window=0).fit(quarterly([1.0]), "y")

    def test_not_fitted(self) -> None:
        with pytest.raises(ModelNotFittedError):
            _ = HistoricalMean().mean_

    @given(st.lists(st.floats(-1e6, 1e6), min_size=1, max_size=40))
    def test_equals_numpy_mean(self, values: list[float]) -> None:
        bench = HistoricalMean().fit(quarterly(values), "y")
        assert bench.mean_ == pytest.approx(float(np.mean(values)), rel=1e-9, abs=1e-6)
