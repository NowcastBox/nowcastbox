"""Tests of ``NowcastResults.distribution(method="empirical")`` (empirical error bands)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nowcastbox.core.results import NowcastResults, build_nowcast_frame
from nowcastbox.density import (
    EmpiricalQuantileDistribution,
    NowcastDistribution,
    empirical_bands,
)


def _table() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    months = pd.period_range("2005-01", "2019-12", freq="M")
    vintages = [m.start_time + pd.Timedelta(days=14) for m in months]
    periods = pd.PeriodIndex([m.asfreq("Q") for m in months], freq="Q")
    horizon = [
        int(p.asfreq("M", how="E").ordinal - m.ordinal)
        for p, m in zip(periods, months, strict=True)
    ]
    error = rng.normal(size=len(months))
    return pd.DataFrame(
        {
            "vintage": vintages,
            "target_period": periods,
            "months_to_end": horizon,
            "days_to_release": 150.0,
            "forecast": 0.0,
            "actual": error,
            "error": error,
        }
    )


@pytest.fixture
def results() -> NowcastResults:
    idx = pd.period_range("2019Q3", periods=3, freq="Q")
    observed = pd.Series([0.2, np.nan, np.nan], index=idx)
    estimate = pd.Series([0.1, 0.4, 0.5], index=idx)
    nowcast = build_nowcast_frame(observed, estimate, extra={"std": pd.Series(0.3, index=idx)})
    return NowcastResults(target="gdp", nowcast=nowcast, info={"vintage": "2020-01-20"})


def test_empirical_shortcut_matches_function(results: NowcastResults) -> None:
    table = _table()
    via_method = results.distribution(method="empirical", backtest=table, periods="2020Q1")
    direct = empirical_bands(results, table, periods="2020Q1")
    assert isinstance(via_method, NowcastDistribution)
    assert via_method.scales.tolist() == direct.scales.tolist()
    assert via_method.info["method"] == "mae"


def test_empirical_shortcut_options(results: NowcastResults) -> None:
    dist = results.distribution(
        method="empirical",
        backtest=_table(),
        empirical_method="quantile",
        periods=["2020Q1"],
        window=None,
        levels=(0.9,),
    )
    assert isinstance(dist, EmpiricalQuantileDistribution)
    assert dist.levels == (0.9,)


def test_empirical_shortcut_requires_backtest(results: NowcastResults) -> None:
    with pytest.raises(ValueError, match="backtest"):
        results.distribution(method="empirical")


def test_default_behaviour_unchanged(results: NowcastResults) -> None:
    dist = results.distribution()
    assert isinstance(dist, NowcastDistribution)
    assert dist.scales[:, 0].tolist() == [0.3, 0.3]
