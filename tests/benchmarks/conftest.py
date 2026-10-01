"""Fixtures of the benchmark tests: small simulated mixed-frequency panels."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nowcastbox.core.data import MixedFrequencyData

FREQ = {"x": "M", "z": "M", "y": "Q"}


def simulate_panel(n_months: int = 150, seed: int = 0, noise: float = 0.2) -> pd.DataFrame:
    """Monthly AR(1) indicators x, z and a quarterly target driven by x."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n_months)
    for t in range(1, n_months):
        x[t] = 0.6 * x[t - 1] + rng.standard_normal()
    z = rng.standard_normal(n_months)
    idx = pd.period_range("2000-01", periods=n_months, freq="M")
    y = 0.5 + pd.Series(x, index=idx).rolling(3).mean() + noise * rng.standard_normal(n_months)
    y = y.where(idx.month % 3 == 0)
    return pd.DataFrame({"x": x, "z": z, "y": y}, index=idx)


@pytest.fixture
def panel() -> MixedFrequencyData:
    """Complete panel (150 months, last quarter observed)."""
    return MixedFrequencyData(simulate_panel(), FREQ)


@pytest.fixture
def ragged() -> MixedFrequencyData:
    """Panel with a ragged edge: last quarter of y missing, x one month, z two months."""
    frame = simulate_panel()
    frame.iloc[-3:, 2] = np.nan
    frame.iloc[-1:, 0] = np.nan
    frame.iloc[-2:, 1] = np.nan
    return MixedFrequencyData(frame, FREQ)


def quarterly(values: np.ndarray, start: str = "1990Q1") -> MixedFrequencyData:
    """Quarterly-only panel with one series ``y``."""
    idx = pd.period_range(start, periods=len(values), freq="Q")
    return MixedFrequencyData(pd.DataFrame({"y": np.asarray(values, dtype=float)}, index=idx), "Q")
