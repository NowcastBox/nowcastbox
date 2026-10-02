"""Tests of the ``"bvar"`` indicator extrapolator (``models/bvar_extrapolation.py``)."""

from __future__ import annotations

import doctest
import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest

import nowcastbox.models.bvar_extrapolation as bx
from nowcastbox.benchmarks import BridgeCombinationBenchmark
from nowcastbox.core.data import MixedFrequencyData
from nowcastbox.core.exceptions import ConvergenceWarning, DataQualityWarning, NowcastDataError
from nowcastbox.models import (
    ARExtrapolator,
    BridgeCombination,
    BVARExtrapolator,
    MonthlyBVAR,
    fit_blocked_bvar,
    fit_monthly_bvar,
)
from nowcastbox.models.bridge import ar_extend
from nowcastbox.models.bvar import BlockedBVAR
from nowcastbox.models.extrapolation import make_extrapolator, native_until
from tests.models.bvar_helpers import (
    analytic_conditional,
    random_covariance,
    simulate_var,
    stable_coefficients,
)

FIXED = {"lambda": 0.5}


def monthly_panel(
    n_months: int = 240, n: int = 3, lags: int = 2, seed: int = 0, quarterly: bool = False
) -> MixedFrequencyData:
    """Monthly VAR(``lags``) indicators x1..xn (ragged edge 0/1/2 months), quarterly y."""
    rng = np.random.default_rng(seed)
    A = stable_coefficients(n, lags, rng)
    values = simulate_var(A, 0.3 * rng.standard_normal(n), random_covariance(n, rng), n_months, rng)
    idx = pd.period_range("2000-01", periods=n_months, freq="M")
    names = [f"x{i + 1}" for i in range(n)]
    frame = pd.DataFrame(values, index=idx, columns=names)
    for i in range(1, n):
        frame.iloc[-i:, i] = np.nan
    quarter_end = idx.month % 3 == 0
    signal = frame["x1"].rolling(3).mean() + 0.5 * frame["x2"].rolling(3).mean()
    y = (1.0 + signal + 0.3 * rng.standard_normal(n_months)).where(quarter_end)
    y.iloc[-3:] = np.nan
    frame["y"] = y
    freq = dict.fromkeys(names, "M") | {"y": "Q"}
    if quarterly:
        frame["q1"] = np.where(quarter_end, rng.standard_normal(n_months), np.nan)
        frame.loc[frame.index[-3:], "q1"] = np.nan
        freq["q1"] = "Q"
    return MixedFrequencyData(frame, freq)


@pytest.fixture(scope="module")
def panel() -> MixedFrequencyData:
    return monthly_panel()


COLUMNS = ["x1", "x2", "x3"]


# ====================================================================== equality
@pytest.mark.parametrize("standardize", [False, True])
@pytest.mark.parametrize("lags", [1, 2])
def test_monthly_fill_is_the_conditional_expectation(
    panel: MixedFrequencyData, standardize: bool, lags: int
) -> None:
    ext = BVARExtrapolator(lags=lags, prior=FIXED, standardize=standardize)
    assert ext.mode(panel, COLUMNS) == "monthly"
    end = pd.Period(panel.end, "M").asfreq("Q") + 1
    out = ext(panel, COLUMNS, end)
    engine = ext.engine(panel, COLUMNS)
    assert isinstance(engine, MonthlyBVAR)
    par = engine.parameters
    # independent computation: joint Gaussian of the edge conditioned by direct inversion
    std = engine.standardization.transform(panel.data[COLUMNS]).to_numpy()
    n_rows = len(std) + 3
    std = np.vstack([std, np.full((3, 3), np.nan)])
    last = int(np.flatnonzero(np.isfinite(std).all(axis=1))[-1])
    presample = std[last - lags + 1 : last + 1]
    future = std[last + 1 : n_rows]
    mean, var = analytic_conditional(par.coefficients, par.intercept, par.sigma, presample, future)
    index = pd.period_range(panel.index[last + 1], periods=len(future), freq="M")
    expected = engine.standardization.inverse_transform(
        pd.DataFrame(mean, index=index, columns=COLUMNS)
    )
    for col in COLUMNS:
        series = out[col]
        assert series.index[-1] == end.asfreq("M", how="E")
        observed = panel.data[col].dropna()
        np.testing.assert_allclose(series.loc[observed.index], observed.to_numpy())
        filled = series.loc[observed.index[-1] + 1 :]
        np.testing.assert_allclose(filled, expected.loc[filled.index, col], atol=1e-9)
    _, _, edge_var = engine.edge(engine.standardized(panel.extend(3))[1])
    np.testing.assert_allclose(edge_var[last + 1 :], var, atol=1e-9)


def test_blocked_mode_equals_blocked_engine() -> None:
    data = monthly_panel(quarterly=True)
    cols = [*COLUMNS, "q1"]
    ext = BVARExtrapolator(prior=FIXED)
    assert ext.mode(data, cols) == "blocked"
    end = pd.Period(data.end, "M").asfreq("Q")
    out = ext(data, cols, end)
    engine = ext.engine(data, cols)
    assert isinstance(engine, BlockedBVAR) and engine.lags == 1
    direct = fit_blocked_bvar(data.select(cols), prior=FIXED).predict(data.select(cols))
    for col in COLUMNS:
        tail = out[col].loc[data.data[col].dropna().index[-1] + 1 :]
        np.testing.assert_allclose(tail, direct.loc[tail.index, col], rtol=1e-12)
    assert str(out["q1"].index[-1]) == str(end)
    assert out["q1"].iloc[-1] == pytest.approx(direct["q1"].dropna().iloc[-1], rel=1e-12)
    forced = BVARExtrapolator(prior=FIXED, blocking=True)
    assert forced.mode(data, COLUMNS) == "blocked"
    assert isinstance(forced.engine(data, COLUMNS), BlockedBVAR)


def test_monthly_glp_default(panel: MixedFrequencyData) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        ext = make_extrapolator("bvar")
        out = ext(panel, COLUMNS, pd.Period(panel.end, "M").asfreq("Q"))
    assert isinstance(ext, BVARExtrapolator)
    engine = ext.engine(panel, COLUMNS)  # cached
    assert engine.lags == 3 and engine.selection is not None
    assert 0 < engine.hyperparameters.lambda_ < 10
    assert all(out[c].notna().all() for c in COLUMNS)


# ====================================================================== cache
def test_engine_estimated_once_and_cached(
    panel: MixedFrequencyData, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []
    original = bx._fit_monthly

    def counting(*args: Any, **kwargs: Any) -> MonthlyBVAR:
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(bx, "_fit_monthly", counting)
    ext = BVARExtrapolator(prior=FIXED)
    end = pd.Period(panel.end, "M").asfreq("Q")
    first = ext(panel, COLUMNS, end)
    longer = ext(panel, COLUMNS, end + 2)
    assert len(calls) == 1
    pd.testing.assert_series_equal(longer["x3"].loc[first["x3"].index], first["x3"])
    # another vintage or another set of columns: new estimation, bounded cache
    for k in range(1, 6):
        ext(panel.truncate(end=panel.end - k), COLUMNS, end)
    assert len(calls) == 6
    assert len(ext._engines) == bx._CACHE_SIZE
    ext(panel, ["x1", "x2"], end)
    assert len(calls) == 7


def test_bridge_combination_extrapolates_once_per_fit(
    panel: MixedFrequencyData, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []
    original = BVARExtrapolator.__call__

    def counting(self: BVARExtrapolator, data: Any, columns: Any, end: Any) -> Any:
        calls.append(list(columns))
        return original(self, data, columns, end)

    monkeypatch.setattr(BVARExtrapolator, "__call__", counting)
    model = BridgeCombination(extrapolation="bvar", extrapolation_options={"prior": FIXED})
    res = model.fit(panel, "y")
    assert calls == [COLUMNS]
    ext = BVARExtrapolator(prior=FIXED)
    direct = ext(panel, COLUMNS, res.nowcast.index[-1])
    for col in COLUMNS:
        pd.testing.assert_series_equal(res.extrapolated[col], direct[col])


# ====================================================================== bridge combination
def test_bridge_combination_end_to_end_with_bvar(panel: MixedFrequencyData) -> None:
    res = BridgeCombination(
        max_monthly=2, extrapolation="bvar", extrapolation_options={"prior": FIXED}
    ).fit(panel, "y")
    assert res.info["n_equations"] == 6
    nowcast = res.nowcast
    assert np.isfinite(nowcast["out_of_sample"].iloc[-2:]).all()
    # with the same extrapolated indicators a callable gives the same combination
    completed = res.extrapolated

    def frozen(data: Any, columns: Any, end: Any) -> dict[str, pd.Series]:
        return {c: completed[c] for c in columns}

    same = BridgeCombination(max_monthly=2, extrapolation=frozen).fit(panel, "y")
    np.testing.assert_allclose(same.estimate, res.estimate)
    ar = BridgeCombination(max_monthly=2).fit(panel, "y")
    assert not np.allclose(ar.estimate.iloc[-2:], res.estimate.iloc[-2:])
    # quarterly indicators: the blocked VAR completes everything jointly
    mixed = monthly_panel(quarterly=True)
    out = BridgeCombination(extrapolation="bvar", extrapolation_options={"prior": FIXED}).fit(
        mixed, "y"
    )
    assert set(out.extrapolated) == {*COLUMNS, "q1"}
    assert np.isfinite(out.get_nowcast())


def test_bvar_failure_falls_back_series_by_series(panel: MixedFrequencyData) -> None:
    frame = panel.data.copy()
    frame["x4"] = np.nan
    frame.loc[frame.index[-4:], "x4"] = [0.1, 0.2, 0.3, 0.4]  # too short for a joint VAR
    data = MixedFrequencyData(frame, dict(panel.frequencies) | {"x4": "M"})
    model = BridgeCombination(
        max_monthly=1, extrapolation="bvar", extrapolation_options={"prior": FIXED}
    )
    with pytest.warns(DataQualityWarning, match="x4"):
        res = model.fit(data, "y")
    assert set(res.extrapolated) == set(COLUMNS)


def test_ar_path_unchanged(panel: MixedFrequencyData) -> None:
    default = BridgeCombination().fit(panel, "y")
    explicit = BridgeCombination(extrapolation=ARExtrapolator(ar_lags=1)).fit(panel, "y")
    np.testing.assert_array_equal(default.estimate, explicit.estimate)
    end = default.nowcast.index[-1]
    for col in COLUMNS:
        native = native_until(panel, col, end).to_numpy()
        last = int(np.flatnonzero(np.isfinite(native))[-1])
        manual = ar_extend(native, len(native) - last - 1, 1)
        np.testing.assert_array_equal(default.extrapolated[col].to_numpy(), manual)


def test_benchmark_reuses_the_extrapolator(
    panel: MixedFrequencyData, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []
    original = bx._fit_monthly

    def counting(*args: Any, **kwargs: Any) -> MonthlyBVAR:
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(bx, "_fit_monthly", counting)
    bench = BridgeCombinationBenchmark(
        max_monthly=1, extrapolation="bvar", extrapolation_options={"prior": FIXED}
    )
    bench.fit(panel, "y")
    current = pd.Period(panel.end, "M").asfreq("Q")
    pred = bench.predict([current, current + 1, current + 2])
    assert np.isfinite(pred).all()
    assert len(calls) == 1  # horizon 0 and the refit for horizon 2 share the VAR
    model = BridgeCombination(
        max_monthly=1, horizon=2, extrapolation="bvar", extrapolation_options={"prior": FIXED}
    )
    np.testing.assert_allclose(
        pred.to_numpy(), model.fit(panel, "y").estimate.loc[pred.index].to_numpy()
    )
    ar = BridgeCombinationBenchmark(max_monthly=1)
    ar.fit(panel, "y")
    assert isinstance(ar.results_.model_params["extrapolation"], ARExtrapolator)
    custom = BridgeCombinationBenchmark(max_monthly=1, extrapolation=ARExtrapolator(2))
    custom.fit(panel, "y")
    assert custom.results_.model_params["extrapolation"] == ARExtrapolator(2)


# ====================================================================== errors
@pytest.mark.parametrize(
    "kwargs",
    [{"lags": 0}, {"blocking": "yes"}, {"prior": "flat"}, {"standardize": 1}],
)
def test_invalid_options(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        BVARExtrapolator(**kwargs)


def test_data_errors(panel: MixedFrequencyData) -> None:
    end = pd.Period(panel.end, "M").asfreq("Q")
    frame = panel.data.copy()
    frame["x4"] = np.nan
    with pytest.warns(DataQualityWarning, match="x4"):
        empty = MixedFrequencyData(frame, dict(panel.frequencies) | {"x4": "M"})
    ext = BVARExtrapolator(prior=FIXED)
    with pytest.raises(NowcastDataError, match="x4"):
        ext(empty, ["x1", "x4"], end)
    with pytest.raises(NowcastDataError, match="x4"):
        fit_monthly_bvar(empty.select(["x1", "x4"]), prior=FIXED)
    with pytest.raises(NowcastDataError, match="base frequency"):
        BVARExtrapolator(prior=FIXED, blocking=False)(panel, ["x1", "y"], end)
    with pytest.raises(NowcastDataError, match="consecutive complete"):
        ext(panel.truncate(end="2000-04"), COLUMNS, end)
    with pytest.raises(ValueError, match="lags"):
        fit_monthly_bvar(panel.select(COLUMNS), lags=0)


def test_interior_gap_warns(panel: MixedFrequencyData) -> None:
    frame = panel.data.copy()
    frame.loc[frame.index[100], "x2"] = np.nan
    gappy = panel.with_data(frame)
    with pytest.warns(DataQualityWarning, match=r"earlier periods.*\(x2\)"):
        eng = fit_monthly_bvar(gappy.select(COLUMNS), prior=FIXED)
    assert eng.sample[0] == 101


def test_engine_on_frames(panel: MixedFrequencyData) -> None:
    eng = fit_monthly_bvar(panel.select(COLUMNS), prior=FIXED, standardize=False, lags=1)
    np.testing.assert_allclose(eng.standardization.std, 1.0)
    pred = eng.predict(panel.data[COLUMNS])
    assert pred.notna().all().all()
    shorter = eng.standardized(panel.data[COLUMNS].iloc[:-5])
    assert shorter[1].shape == eng.values.shape
    with pytest.raises(NowcastDataError, match="x3"):
        eng.standardized(panel.data[["x1", "x2"]])
    end, mean, var = eng.edge(eng.values[:-2])  # complete window at the end: nothing to fill
    assert end == len(eng.values) - 3
    np.testing.assert_array_equal(mean, eng.values[:-2])
    assert not var.any()


def test_doctests() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        result = doctest.testmod(bx, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)
    assert result.failed == 0 and result.attempted > 0
