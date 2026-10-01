"""Bai-Ng criteria vs the R package ``nowcasting`` 1.1.2 (``ICfactors``, ``ICshocks``).

Fixtures: ``scripts/reference_fixtures/r_selection.R``. Panels: balanced parts of
``Bpanel(USGDP)`` (189 x 306), ``Bpanel(BRGDP)`` (86 x 215) and of the simulated
predictors (24 x 238); the R-processed panels are rebuilt with the validated emulation of
``Bpanel`` (``_bpanel.py``).

Findings (``docs/validation/selection.md``):

* ``ICfactors`` = :func:`~nowcastbox.selection.select_factors` (standardised panel,
  ``IC_{p1..p3}``): identical ``r*`` and criteria to ~1e-15 (plan: exact / 1e-8);
* ``ICshocks`` uses the statistic ``D1`` on the eigenvalues of the VAR residual
  covariance - identical to nowcastbox's default - but its bound is
  ``m / min(N, T)^(1 / (2 - delta))`` instead of Bai & Ng's (2007)
  ``m / min(N^(1/2 - delta), T^(1/2 - delta))`` (recovered to 1e-10 from the values of
  ``m`` at which ``q*`` changes). R's bound is smaller, so R selects more shocks (often
  ``q* = r``); with R's bound nowcastbox reproduces every ``q*``.
"""

from __future__ import annotations

import warnings
from functools import cache

import numpy as np
import pandas as pd
import pytest

from nowcastbox.selection import select_factors, select_shocks, shock_bound
from tests.reference_validation._bpanel import emulate_bpanel
from tests.reference_validation._helpers import (
    TOL_IC,
    codes,
    max_abs,
    read_json,
    read_periods,
)

pytestmark = pytest.mark.reference_validation


@cache
def _panel(name: str) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if name == "usgdp":
            raw = read_periods("inputs/usgdp_base.csv.gz").drop(columns="RGDPGR")
            spec = {k: v for k, v in codes("usgdp").items() if k != "RGDPGR"}
            return emulate_bpanel(raw, spec).dropna()
        if name == "brgdp":
            return emulate_bpanel(
                read_periods("inputs/brgdp_base.csv.gz"), codes("brgdp"), h=0
            ).dropna()
        return read_periods("inputs/simulated.csv").drop(columns="gdp").dropna()


def r_bound(n_series: int, n_periods: int, delta: float, m: float) -> float:
    """Bound used by ``ICshocks`` (identified from the black-box outputs)."""
    return m / min(n_series, n_periods) ** (1.0 / (2.0 - delta))


def _q_with_bound(statistics: pd.Series, bound: float, r: int) -> int:
    below = np.flatnonzero(statistics.to_numpy() < bound)
    return int(statistics.index[below[0]]) if below.size else r


def _ic_cases() -> list[str]:
    try:
        return sorted(read_json("r/icfactors.json"))
    except pytest.skip.Exception:  # pragma: no cover - fixtures missing
        return []


@pytest.mark.parametrize("key", _ic_cases())
def test_icfactors(key: str) -> None:
    case = read_json("r/icfactors.json")[key]
    x = _panel(case["panel"])
    assert x.shape == (case["n_rows"], case["n_cols"])
    res = select_factors(x, rmax=case["rmax"], criterion=f"IC{case['type']}")
    assert res.r_star == case["r_star"]
    ours = res.criteria[f"IC{case['type']}"].to_numpy()[1:]  # R reports r = 1..rmax
    assert max_abs(ours, case["IC"]) < TOL_IC


def test_icshocks_statistic_and_bound() -> None:
    """The thresholds in m equal D1_k * min(N, T)^(1/(2-delta)) with nowcastbox's D1."""
    errors = []
    for case in read_json("r/icshocks_thresholds.json"):
        x = _panel(case["panel"])
        res = select_shocks(x, n_factors=case["r"], factor_lags=case["p"])
        d1 = float(res.statistics["D1"].loc[case["target"]])
        n_obs, n_series = x.shape
        predicted = d1 * min(n_series, n_obs) ** (1.0 / (2.0 - case["delta"]))
        errors.append(abs(predicted / case["m"] - 1.0))
    assert max(errors) < 1e-8


def test_icshocks_q_star_with_r_bound() -> None:
    mismatches = []
    for case in read_json("r/icshocks_grid.json"):
        x = _panel(case["panel"])
        res = select_shocks(x, n_factors=case["r"], factor_lags=case["p"])
        n_obs, n_series = x.shape
        bound = r_bound(n_series, n_obs, case["delta"], case["m"])
        q = _q_with_bound(res.statistics["D1"], bound, case["r"])
        if q != case["q_star"]:
            mismatches.append(case)
    assert not mismatches


@pytest.mark.reference_divergence
def test_icshocks_bound_differs_from_paper() -> None:
    """nowcastbox follows Bai & Ng (2007); R's bound is smaller, so R selects more shocks."""
    grid = read_json("r/icshocks_grid.json")
    n_diff = 0
    for case in grid:
        x = _panel(case["panel"])
        res = select_shocks(
            x, n_factors=case["r"], factor_lags=case["p"], delta=case["delta"], m=case["m"]
        )
        n_obs, n_series = x.shape
        paper = shock_bound(n_series, n_obs, delta=case["delta"], m=case["m"])
        assert paper == pytest.approx(
            case["m"] / min(n_series ** (0.5 - case["delta"]), n_obs ** (0.5 - case["delta"]))
        )
        assert r_bound(n_series, n_obs, case["delta"], case["m"]) < paper
        n_diff += res.q_star != case["q_star"]
        assert res.q_star <= case["q_star"]
    assert 0 < n_diff < len(grid)  # documented: the selections differ in part of the grid
