"""Tests of BridgeCombinationBenchmark."""

from __future__ import annotations

import doctest

import numpy as np
import pytest

import nowcastbox.benchmarks.bridge_combination as module
from nowcastbox.benchmarks import AR, BridgeCombinationBenchmark
from nowcastbox.core.data import MixedFrequencyData
from nowcastbox.core.exceptions import ModelNotFittedError, NowcastDataError
from nowcastbox.models import BridgeCombination
from tests.benchmarks.conftest import FREQ, simulate_panel


def test_predict_matches_model(ragged) -> None:
    bench = BridgeCombinationBenchmark(combine="inverse_mse").fit(ragged, "y")
    model = BridgeCombination(combine="inverse_mse", horizon=1).fit(ragged, "y")
    periods = model.nowcast.index[-2:]
    pred = bench.predict(periods)
    np.testing.assert_allclose(pred.to_numpy(), model.estimate.loc[periods].to_numpy())
    assert bench.results_.info["n_equations"] == 3
    assert bench.weights_.sum() == pytest.approx(1.0)


def test_horizon_extension_and_cache(ragged) -> None:
    bench = BridgeCombinationBenchmark().fit(ragged, "y")
    first = bench.results_
    assert first.nowcast.index[-1] == ragged.end.asfreq("Q")
    bench.predict(["2012Q2"])
    assert bench.results_ is first  # no refit for a period already covered
    out = bench.predict(["2012Q3", "2013Q1"])
    assert np.isfinite(out).all()
    assert bench.results_.nowcast.index[-1] == out.index[-1]
    back = bench.predict(["2001Q1"])
    assert np.isfinite(back).all()


def test_predictors_subset(ragged) -> None:
    bench = BridgeCombinationBenchmark(predictors=["x"], target_lags=1).fit(ragged, "y")
    assert bench.results_.equations()["spec"].tolist() == ["y ~ x"]


def test_errors(panel) -> None:
    with pytest.raises(ModelNotFittedError):
        _ = BridgeCombinationBenchmark().results_
    with pytest.raises(ValueError):
        BridgeCombinationBenchmark(ar_lags=-1).fit(panel, "y")
    with pytest.raises(ValueError):
        BridgeCombinationBenchmark(combine="mode").fit(panel, "y")
    with pytest.raises(NowcastDataError):
        BridgeCombinationBenchmark(predictors=["nope"]).fit(panel, "y")


def test_refit_resets_state(panel, ragged) -> None:
    bench = BridgeCombinationBenchmark()
    bench.fit(panel, "y")
    first = bench.results_
    bench.fit(ragged, "y")
    assert bench.results_ is not first
    assert bench.results_.data is not None
    assert bench.results_.data.end == ragged.end


def test_clone_keeps_params() -> None:
    bench = BridgeCombinationBenchmark(max_monthly=1, trim=0.2, extrapolation_options={"a": 1})
    params = bench.clone().get_params()
    assert params["max_monthly"] == 1 and params["trim"] == 0.2


def test_in_pseudo_real_time_backtest() -> None:
    from nowcastbox.evaluation import PseudoRealTimeBacktest

    frame = simulate_panel(n_months=150, seed=3)
    data = MixedFrequencyData(frame, FREQ, release_delays={"x": 20, "z": 10, "y": 45})
    bt = PseudoRealTimeBacktest(
        data=data,
        target="y",
        start="2009-01-15",
        end="2010-12-15",
        step="M",
        benchmarks={
            "AR": AR(p=1),
            "combo": BridgeCombinationBenchmark(combine="inverse_mse", mse="out_of_sample"),
        },
    )
    res = bt.run()
    assert res.models == ["AR", "combo"]
    frame_out = res.to_frame()
    assert np.isfinite(frame_out["forecast"]).all()
    relative = res.relative_to("AR")
    assert (relative.loc[[-1, 0], "combo"] < 1).all()


def test_doctests() -> None:
    result = doctest.testmod(module, optionflags=doctest.ELLIPSIS)
    assert result.attempted > 0
    assert result.failed == 0
