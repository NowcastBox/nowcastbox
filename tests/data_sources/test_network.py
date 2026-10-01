"""Tests against the real public APIs (run with ``--run-network``)."""

from __future__ import annotations

import os

import pandas as pd
import pytest

from nowcastbox.core.data import MixedFrequencyData
from nowcastbox.data_sources import (
    DataSourceError,
    HTTPStatusError,
    fetch_fred,
    fetch_ipeadata,
    fetch_sgs,
    fetch_sidra,
)

pytestmark = pytest.mark.network


def _skip_if_unreachable(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except HTTPStatusError:
        raise
    except DataSourceError as err:  # connection errors / timeouts after retries
        pytest.skip(f"service unreachable: {err}")


def test_bcb_monthly_and_quarterly_on_monthly_grid():
    df = _skip_if_unreachable(
        fetch_sgs,
        {"ipca": 433, "gdp_index": 22099},
        start="2019-01",
        end="2020-12",
        base_frequency="M",
        cache=False,
        backoff_factor=2.0,
    )
    assert df.index.freqstr == "M" and len(df) == 24
    assert df.loc[pd.Period("2020-01", "M"), "ipca"] == pytest.approx(0.21)
    assert df["gdp_index"].notna().sum() == 8
    mfd = MixedFrequencyData(df, {"ipca": "M", "gdp_index": "Q"})
    assert mfd.quarterly_columns == ["gdp_index"]


def test_bcb_daily_chunked_over_ten_years():
    df = _skip_if_unreachable(
        fetch_sgs, 1, start="2009-12-01", end="2020-01-31", cache=False, backoff_factor=2.0
    )
    assert df.index.freqstr == "D"
    assert df.index[0] >= pd.Period("2009-12-01", "D")
    assert df.index[-1] <= pd.Period("2020-01-31", "D")
    assert len(df) > 2400  # ~ 250 business days a year


def test_sidra_pim():
    df = _skip_if_unreachable(
        fetch_sidra, 8888, 12606, {"c544": 129314}, periods="202301-202312", name="pim", cache=False
    )
    assert df.columns.tolist() == ["pim"]
    assert df.index.astype(str).tolist()[0] == "2023-01" and len(df) == 12


def test_ipeadata_selic():
    df = _skip_if_unreachable(
        fetch_ipeadata,
        {"selic": "BM12_TJOVER12"},
        start="2020-01",
        end="2020-12",
        cache=False,
        timeout=30.0,
        max_retries=1,
    )
    assert df.index.freqstr == "M" and len(df) == 12


@pytest.mark.skipif(not os.environ.get("FRED_API_KEY"), reason="FRED_API_KEY not set")
def test_fred_gdp():
    df = _skip_if_unreachable(
        fetch_fred, "GDPC1", start="2015-01-01", end="2019-12-31", cache=False
    )
    assert df.index.freqstr.startswith("Q") and len(df) == 20
