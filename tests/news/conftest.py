"""Fixtures of the news tests: simulated mixed-frequency panels and fitted models."""

from __future__ import annotations

import warnings

import matplotlib
import numpy as np
import pandas as pd
import pytest

from nowcastbox.core.data import MixedFrequencyData
from nowcastbox.models import MixedFreqDFM, TwoStepDFM

matplotlib.use("Agg")

MONTHLY = ["ip", "sales", "pmi", "conf", "spread", "stocks"]
CATEGORIES = {
    "ip": "hard",
    "sales": "hard",
    "pmi": "soft",
    "conf": "soft",
    "spread": "financial",
    "stocks": "financial",
    "gdp": "hard",
}
BLOCKS = {
    "ip": ["global", "real"],
    "sales": ["global", "real"],
    "pmi": ["global"],
    "conf": ["global"],
    "spread": ["global"],
    "stocks": ["global"],
    "gdp": ["global", "real"],
}
DELAYS = {"ip": 40, "sales": 35, "pmi": 1, "conf": 5, "spread": 0, "stocks": 0, "gdp": 60}


def simulate_panel(n_periods: int = 120, seed: int = 3, ragged: bool = True) -> MixedFrequencyData:
    """Monthly factor model with a quarterly (Mariano-Murasawa) target ``gdp``."""
    rng = np.random.default_rng(seed)
    f = np.zeros(n_periods)
    for t in range(1, n_periods):
        f[t] = 0.7 * f[t - 1] + rng.standard_normal()
    loads = np.array([1.0, 0.8, 0.9, 0.6, -0.5, 0.4])
    x = np.outer(f, loads) + 0.6 * rng.standard_normal((n_periods, len(loads)))
    gdp = (
        0.5 + np.convolve(f, [1, 2, 3, 2, 1])[:n_periods] / 9 + 0.2 * rng.standard_normal(n_periods)
    )
    gdp[np.arange(n_periods) % 3 != 2] = np.nan
    idx = pd.period_range("2005-01", periods=n_periods, freq="M")
    frame = pd.DataFrame(x + 2.0, index=idx, columns=MONTHLY).assign(gdp=gdp)
    if ragged:
        for col, lag in zip(MONTHLY, [2, 2, 0, 1, 0, 1], strict=True):
            if lag:
                frame.iloc[-lag:, frame.columns.get_loc(col)] = np.nan
        frame.iloc[-3:, frame.columns.get_loc("gdp")] = np.nan
    return MixedFrequencyData(
        frame,
        {**dict.fromkeys(MONTHLY, "M"), "gdp": "Q"},
        categories=CATEGORIES,
        release_delays=DELAYS,
        blocks=BLOCKS,
    )


@pytest.fixture(scope="session")
def panel() -> MixedFrequencyData:
    return simulate_panel()


@pytest.fixture(scope="session")
def em_model(panel):
    model = MixedFreqDFM(n_factors=1, max_iter=25, tol=1e-6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(panel, target="gdp")
    return model


@pytest.fixture(scope="session")
def em_results(em_model):
    return em_model.results_


@pytest.fixture(scope="session")
def em_blocks_results(panel):
    model = MixedFreqDFM(n_factors=1, blocks="data", max_iter=10, idiosyncratic="iid")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.fit(panel, target="gdp")


@pytest.fixture(scope="session")
def ts_model(panel):
    model = TwoStepDFM(n_factors=1, factor_lags=1)
    model.fit(panel, target="gdp")
    return model


@pytest.fixture(scope="session")
def ts_results(ts_model):
    return ts_model.results_


def vintage_pair(panel: MixedFrequencyData, n_back: int = 2):
    """(old, new): old drops the last ``n_back`` months of every series."""
    old = panel.data
    old.iloc[-n_back:, :] = np.nan
    return panel.with_data(old), panel
