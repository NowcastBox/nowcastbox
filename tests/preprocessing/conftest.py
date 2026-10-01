"""Fixtures for the preprocessing tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nowcastbox.core.data import MixedFrequencyData


def make_levels(n_months: int = 60, start: str = "2015-01", seed: int = 0) -> pd.DataFrame:
    """Positive monthly + quarterly level panel on a monthly grid (storage convention)."""
    rng = np.random.default_rng(seed)
    idx = pd.period_range(start, periods=n_months, freq="M")
    ip = 100 * np.exp(np.cumsum(rng.normal(0.002, 0.01, n_months)))
    sales = 50 * np.exp(np.cumsum(rng.normal(0.001, 0.02, n_months)))
    gdp = np.full(n_months, np.nan)
    q_end = np.asarray(idx.month % 3 == 0)
    gdp[q_end] = 1000 * np.exp(np.cumsum(rng.normal(0.005, 0.01, int(q_end.sum()))))
    return pd.DataFrame({"ip": ip, "sales": sales, "gdp": gdp}, index=idx)


FREQS = {"ip": "M", "sales": "M", "gdp": "Q"}


@pytest.fixture
def levels() -> pd.DataFrame:
    return make_levels()


@pytest.fixture
def ragged_levels(levels: pd.DataFrame) -> pd.DataFrame:
    df = levels.copy()
    df.iloc[-1, 0] = np.nan  # ip: one month of delay
    df.iloc[-2:, 1] = np.nan  # sales: two months of delay
    df.iloc[-1, 2] = np.nan  # gdp: last quarter not released
    return df


@pytest.fixture
def panel(ragged_levels: pd.DataFrame) -> MixedFrequencyData:
    return MixedFrequencyData(
        ragged_levels,
        FREQS,
        transforms={"ip": "dlog", "sales": 2, "gdp": 7},
        release_delays={"ip": 40, "sales": 30, "gdp": 60},
    )


@pytest.fixture
def monthly_series(levels: pd.DataFrame) -> pd.Series:
    return levels["ip"]


@pytest.fixture
def quarterly_native(levels: pd.DataFrame) -> pd.Series:
    s = levels["gdp"].dropna()
    s.index = s.index.asfreq("Q")
    return s
