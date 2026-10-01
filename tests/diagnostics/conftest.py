"""Fixtures and simulators for the diagnostics tests."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from nowcastbox.core.data import MixedFrequencyData
from nowcastbox.models import MixedFreqDFM, TwoStepDFM
from nowcastbox.models.two_step import simulate_two_step_example


def simulate_factor_panel(
    seed: int,
    *,
    n_periods: int = 200,
    n_series: int = 40,
    n_factors: int = 1,
    n_break: int = 0,
    delta: float = 1.0,
    break_at: int | None = None,
    rho_e: float = 0.0,
    start: str = "2000-01",
) -> tuple[pd.DataFrame, np.ndarray]:
    """Monthly static factor panel; loadings of the first ``n_break`` series shift by
    ``delta`` from row ``break_at`` (default: middle) on."""
    rng = np.random.default_rng(seed)
    r = n_factors
    f = np.zeros((n_periods, r))
    for t in range(1, n_periods):
        f[t] = 0.5 * f[t - 1] + rng.standard_normal(r)
    lam = rng.normal(1.0, 0.5, (n_series, r))
    u = rng.standard_normal((n_periods, n_series))
    e = np.zeros_like(u)
    for t in range(n_periods):
        e[t] = (rho_e * e[t - 1] if t else 0.0) + u[t]
    x = f @ lam.T + e
    if n_break:
        k = n_periods // 2 if break_at is None else break_at
        post = np.arange(n_periods) >= k
        x[post, :n_break] += f[post] @ (delta * np.ones((n_break, r))).T
    idx = pd.period_range(start, periods=n_periods, freq="M")
    cols = [f"x{i}" for i in range(n_series)]
    return pd.DataFrame(x, index=idx, columns=cols), f


@pytest.fixture(scope="session")
def example_data() -> MixedFrequencyData:
    return simulate_two_step_example(random_state=0, n_factors=2)


@pytest.fixture(scope="session")
def em_results(example_data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return MixedFreqDFM(n_factors=2, max_iter=100).fit(example_data, "gdp")


@pytest.fixture(scope="session")
def two_step_results(example_data):
    return TwoStepDFM(n_factors=2).fit(example_data, "gdp")


@pytest.fixture(scope="session")
def block_data() -> MixedFrequencyData:
    rng = np.random.default_rng(5)
    n = 150
    f = np.zeros((n, 2))
    for t in range(1, n):
        f[t] = 0.6 * f[t - 1] + rng.standard_normal(2)
    x = np.column_stack(
        [f[:, 0] + 0.5 * rng.standard_normal(n) for _ in range(4)]
        + [f[:, 1] + 0.5 * rng.standard_normal(n) for _ in range(4)]
    )
    idx = pd.period_range("2005-01", periods=n, freq="M")
    cols = [f"r{i}" for i in range(4)] + [f"n{i}" for i in range(4)]
    df = pd.DataFrame(x, index=idx, columns=cols)
    gdp = np.convolve(f[:, 0], [1, 2, 3, 2, 1])[:n] / 3 + 0.3 * rng.standard_normal(n)
    gdp[np.arange(n) % 3 != 2] = np.nan
    df["gdp"] = gdp
    df.iloc[-2:, :] = np.nan
    blocks = {c: ("global", "real") if c.startswith("r") else ("global", "nominal") for c in cols}
    blocks["gdp"] = ("global", "real")
    return MixedFrequencyData(
        df,
        {**dict.fromkeys(cols, "M"), "gdp": "Q"},
        blocks=blocks,
        release_delays={**dict.fromkeys(cols, 15), "gdp": 60},
    )


@pytest.fixture(scope="session")
def em_block_results(block_data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return MixedFreqDFM(
            n_factors={"global": 1, "real": 1, "nominal": 1}, blocks="data", max_iter=30
        ).fit(block_data, "gdp")
