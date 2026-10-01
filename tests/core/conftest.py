"""Fixtures for the core tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nowcastbox.core.data import MixedFrequencyData


def make_panel_frame(
    n_months: int = 24, start: str = "2018-01", seed: int = 0, ragged: bool = True
) -> pd.DataFrame:
    """Monthly + quarterly panel frame following the storage convention."""
    rng = np.random.default_rng(seed)
    idx = pd.period_range(start, periods=n_months, freq="M")
    ip = rng.normal(size=n_months)
    pmi = rng.normal(size=n_months)
    gdp = np.full(n_months, np.nan)
    q_end = np.asarray(idx.month % 3 == 0)
    gdp[q_end] = rng.normal(size=int(q_end.sum()))
    df = pd.DataFrame({"ip": ip, "pmi": pmi, "gdp": gdp}, index=idx)
    if ragged:
        df.iloc[-1, 0] = np.nan  # ip released with 1 month delay
        last_q = np.flatnonzero(q_end)[-1]
        df.iloc[last_q, 2] = np.nan  # last quarter of GDP not released
    return df


@pytest.fixture
def panel_frame() -> pd.DataFrame:
    return make_panel_frame()


@pytest.fixture
def panel(panel_frame: pd.DataFrame) -> MixedFrequencyData:
    return MixedFrequencyData(
        panel_frame,
        {"ip": "M", "pmi": "M", "gdp": "Q"},
        release_delays={"ip": 40, "pmi": 1, "gdp": 60},
        blocks={"ip": ["global", "real"], "pmi": ["global", "soft"], "gdp": ["global", "real"]},
        categories={"ip": "hard", "pmi": "soft", "gdp": "hard"},
        transforms={"ip": 2, "pmi": 0, "gdp": 2},
        descriptions={"gdp": "Real GDP growth"},
    )
