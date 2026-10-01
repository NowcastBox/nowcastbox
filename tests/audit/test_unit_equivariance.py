"""Equivariance of nowcasts to the units of the data.

Changing the units of the target (``y -> a y + b``) must change every point nowcast to
``a * nowcast + b`` (and standard deviations to ``|a| * std``); rescaling, shifting or
flipping the sign of a predictor must not change the nowcast. Violations reveal silent
destandardisation, sign/rotation or optimiser-tolerance errors.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest

from nowcastbox.benchmarks import AR, MIDAS, UMIDAS, exp_almon_weights
from nowcastbox.models import BridgeEquation, MixedFreqDFM, TwoStepDFM

FREQ = {"gdp": "Q"}


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    rng = np.random.default_rng(4)
    n = 150
    f = np.zeros(n)
    for t in range(1, n):
        f[t] = 0.7 * f[t - 1] + rng.standard_normal()
    x = np.outer(f, [1.0, 0.8, 0.6, 0.9]) + 0.5 * rng.standard_normal((n, 4))
    gdp = np.convolve(f, [1, 2, 3, 2, 1])[:n] / 3 + 0.3 * rng.standard_normal(n) + 2.0
    gdp[np.arange(n) % 3 != 2] = np.nan
    idx = pd.period_range("2008-01", periods=n, freq="M")
    df = pd.DataFrame(x, index=idx, columns=list("abcd")).assign(gdp=gdp)
    df.iloc[-4:, 0] = np.nan
    df.iloc[-2:, 1:4] = np.nan
    df.iloc[-3:, 4] = np.nan
    return df


def _target_affine(df: pd.DataFrame, a: float, b: float) -> pd.DataFrame:
    out = df.copy()
    out["gdp"] = a * out["gdp"] + b
    return out


def _predictor_change(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["a"] = 7.0 * out["a"] + 3.0
    out["b"] = -out["b"]
    return out


def _estimate(nowcast: pd.DataFrame) -> pd.Series:
    return nowcast["in_sample"].fillna(nowcast["out_of_sample"])


MODELS: dict[str, Callable[[], Any]] = {
    "em_ar1": lambda: MixedFreqDFM(n_factors=1, max_iter=15, tol=0),
    "em_iid": lambda: MixedFreqDFM(n_factors=1, idiosyncratic="iid", max_iter=15, tol=0),
    "em_student_t": lambda: MixedFreqDFM(n_factors=1, idiosyncratic="student_t", max_iter=5, tol=0),
    "em_long_run": lambda: MixedFreqDFM(
        n_factors=1, idiosyncratic="iid", long_run_mean="time_varying", max_iter=5, tol=0
    ),
    "two_step": lambda: TwoStepDFM(n_factors=1),
    "two_step_variables": lambda: TwoStepDFM(n_factors=1, aggregate="variables"),
    "bridge": lambda: BridgeEquation(),
}


@pytest.mark.parametrize("name", list(MODELS))
def test_models_are_equivariant_to_units(panel: pd.DataFrame, name: str) -> None:
    make = MODELS[name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = make().fit(panel, "gdp", frequency=FREQ).nowcast
        scaled = make().fit(_target_affine(panel, 10.0, -5.0), "gdp", frequency=FREQ).nowcast
        moved = make().fit(_predictor_change(panel), "gdp", frequency=FREQ).nowcast
    est = _estimate(base)
    np.testing.assert_allclose(_estimate(scaled), 10.0 * est - 5.0, atol=1e-8)
    np.testing.assert_allclose(_estimate(moved), est, atol=1e-8)
    if "std" in base:
        np.testing.assert_allclose(scaled["std"], 10.0 * base["std"], atol=1e-8)


BENCHMARKS: dict[str, Callable[[], Any]] = {
    "ar": lambda: AR(p=1),
    "umidas": lambda: UMIDAS(),
    "midas": lambda: MIDAS(),
}


@pytest.mark.parametrize("name", list(BENCHMARKS))
def test_benchmarks_are_equivariant_to_units(panel: pd.DataFrame, name: str) -> None:
    make = BENCHMARKS[name]
    periods = pd.period_range("2019Q2", periods=3, freq="Q")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = make().fit(panel, "gdp", frequency=FREQ).predict(periods)
        scaled = make().fit(_target_affine(panel, 10.0, -5.0), "gdp", frequency=FREQ)
        moved = make().fit(_predictor_change(panel), "gdp", frequency=FREQ)
    tol = 1e-4 if name == "midas" else 1e-8  # NLS: solver precision, not units
    np.testing.assert_allclose(scaled.predict(periods), 10.0 * base - 5.0, rtol=tol, atol=tol)
    np.testing.assert_allclose(moved.predict(periods), base, rtol=tol, atol=tol)


def test_midas_lag_polynomial_does_not_depend_on_target_units() -> None:
    """L-BFGS-B stopping rules are absolute: the objective must be scale free."""
    rng = np.random.default_rng(0)
    n = 600
    idx = pd.period_range("1960-01", periods=n, freq="M")
    x = rng.standard_normal(n)
    w = exp_almon_weights(0.3, -0.1, 6)
    y = pd.Series(1.0 + 2.0 * np.convolve(x, w)[:n] + 0.05 * rng.standard_normal(n), index=idx)
    y.iloc[-1] = np.nan  # last quarter unobserved: the nowcast equation has no shift
    thetas = []
    for scale in (1.0, 1e-3):
        frame = pd.DataFrame({"x": x, "y": scale * y.where(idx.month % 3 == 0)}, index=idx)
        bench = MIDAS(n_lags=6).fit(frame, "y", frequency={"x": "M", "y": "Q"})
        thetas.append(bench.theta_.to_numpy())
        assert np.abs(bench.lag_weights_["x"] - w).max() < 5e-3
    np.testing.assert_allclose(thetas[1], thetas[0], atol=1e-4)
