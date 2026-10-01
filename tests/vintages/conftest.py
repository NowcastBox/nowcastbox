"""Fixtures for the vintages tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nowcastbox.core.data import MixedFrequencyData

DELAYS = {"ip": 45, "pmi": 1, "gdp": 60}
FREQS = {"ip": "M", "pmi": "M", "gdp": "Q"}


def make_final_frame(n_months: int = 36, start: str = "2018-01", seed: int = 0) -> pd.DataFrame:
    """Complete (final) monthly + quarterly panel, quarterly values in the 3rd month."""
    rng = np.random.default_rng(seed)
    idx = pd.period_range(start, periods=n_months, freq="M")
    gdp = np.full(n_months, np.nan)
    q_end = np.asarray(idx.month % 3 == 0)
    gdp[q_end] = rng.normal(size=int(q_end.sum()))
    return pd.DataFrame(
        {"ip": rng.normal(size=n_months), "pmi": rng.normal(size=n_months), "gdp": gdp},
        index=idx,
    )


@pytest.fixture
def final_frame() -> pd.DataFrame:
    """Final dataset as a DataFrame."""
    return make_final_frame()


@pytest.fixture
def final_panel(final_frame: pd.DataFrame) -> MixedFrequencyData:
    """Final dataset with release-delay metadata."""
    return MixedFrequencyData(final_frame, FREQS, release_delays=DELAYS)


@pytest.fixture
def revision_records() -> pd.DataFrame:
    """Small real-time data set with revisions (quarterly GDP + monthly IP)."""
    return pd.DataFrame(
        {
            "series": ["gdp"] * 6 + ["ip"] * 4,
            "reference_period": [
                "2020Q1",
                "2020Q1",
                "2020Q1",
                "2020Q2",
                "2020Q2",
                "2020Q3",
                "2020-01",
                "2020-01",
                "2020-02",
                "2020-03",
            ],
            "vintage_date": [
                "2020-05-29",
                "2020-09-01",
                "2020-12-03",
                "2020-09-01",
                "2020-12-03",
                "2020-12-03",
                "2020-03-10",
                "2020-04-08",
                "2020-04-08",
                "2020-05-12",
            ],
            "value": [-1.5, -2.5, -2.4, -9.7, -9.6, 7.7, 0.9, 1.1, -0.8, -9.0],
        }
    )
