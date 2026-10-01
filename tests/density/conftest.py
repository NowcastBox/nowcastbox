"""Shared fixtures of the density tests (fitted models on small simulated panels)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from nowcastbox.core.data import MixedFrequencyData
from nowcastbox.core.frequency import AggregationType
from nowcastbox.models import MixedFreqDFM, TwoStepDFM
from nowcastbox.models.two_step import simulate_two_step_example
from nowcastbox.preprocessing.aggregation import rolling_aggregate


def simulate_with_truth(
    seed: int, n_periods: int = 150, n_series: int = 8
) -> tuple[MixedFrequencyData, float, pd.Period]:
    """Monthly factor panel + quarterly target whose last quarter is hidden.

    Returns the panel, the hidden true value of the target and its period.
    """
    rng = np.random.default_rng(seed)
    burn = 50
    f = np.zeros(n_periods + burn)
    for t in range(1, f.size):
        f[t] = 0.7 * f[t - 1] + rng.standard_normal()
    f = f[burn:]
    lam = rng.uniform(0.5, 1.5, size=n_series)
    x = np.outer(f, lam) + 0.7 * rng.standard_normal((n_periods, n_series))
    x[-1, ::2] = np.nan
    w = AggregationType.GROWTH_RATE.weights(3) / 9.0
    y = 0.5 + rolling_aggregate(f, w) + 0.2 * rng.standard_normal(n_periods)
    index = pd.period_range("2000-01", periods=n_periods, freq="M")
    target = pd.Series(y, index=index).where(index.month % 3 == 0)
    last = int(np.flatnonzero(index.month % 3 == 0)[-1])
    truth = float(target.iloc[last])
    target.iloc[last] = np.nan
    frame = pd.DataFrame(x, index=index, columns=[f"x{i + 1}" for i in range(n_series)])
    frame["gdp"] = target
    data = MixedFrequencyData(frame, dict.fromkeys(frame.columns[:-1], "M") | {"gdp": "Q"})
    return data, truth, index[last].asfreq("Q")


@pytest.fixture(scope="session")
def panel() -> MixedFrequencyData:
    """Small simulated panel (10 monthly predictors, quarterly gdp)."""
    return simulate_two_step_example(random_state=0)


@pytest.fixture(scope="session")
def two_step_results(panel: MixedFrequencyData):
    """TwoStepDFM(aggregate='factors') results."""
    return TwoStepDFM(n_factors=1).fit(panel, "gdp")


@pytest.fixture(scope="session")
def two_step_variables_results(panel: MixedFrequencyData):
    """TwoStepDFM(aggregate='variables') results."""
    return TwoStepDFM(n_factors=1, aggregate="variables").fit(panel, "gdp")


@pytest.fixture(scope="session")
def em_results(panel: MixedFrequencyData):
    """MixedFreqDFM results."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return MixedFreqDFM(n_factors=1, max_iter=100).fit(panel, "gdp")
