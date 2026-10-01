"""Network checks that the primary-source codes of the shipped panels still resolve.

Skipped unless ``--run-network`` (or ``NOWCASTBOX_RUN_NETWORK=1``) is given.
"""

from __future__ import annotations

import pandas as pd
import pytest

import nowcastbox.datasets as nbd
from nowcastbox.data_sources import fetch_sgs_series, fetch_sidra

pytestmark = pytest.mark.network


def test_sgs_codes_of_the_brazilian_panel_resolve() -> None:
    legend = nbd.load_brazil_nowcast().legend
    sgs = legend[legend["source"] == "BCB/SGS"]
    for code in sgs["source_code"].iloc[:3]:
        series = fetch_sgs_series(int(code), start="2024-01-01", cache=False)
        assert len(series) > 0


def test_gdp_target_matches_sidra() -> None:
    panel = nbd.load_brazil_nowcast().data.to_native("pib", dropna=True)
    live = fetch_sidra(1621, 584, {"c11255": 90707}, name="pib", cache=False)["pib"]
    common = panel.index.intersection(live.index)
    assert len(common) > 80
    # the latest SIDRA vintage may revise the seasonal adjustment slightly
    assert (panel.loc[common] - live.loc[common]).abs().max() < 2.0
    assert isinstance(live.index, pd.PeriodIndex)
