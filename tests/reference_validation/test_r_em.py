"""EM (Bańbura & Modugno, 2014) vs ``nowcast(method = "EM")`` of the R package ``nowcasting``.

Fixtures: ``scripts/reference_fixtures/r_em.R`` - NYFED (the package example: 4 blocks
with one factor each, VAR(1), AR(1) idiosyncratic components, Mariano-Murasawa
restriction for GDP and unit labour costs, measurement noise 1e-4) and the simulated
panel (one block, two factors).

Findings (``docs/validation/em.md``):

* R's final model (``Res$A, C, Q, R, Z_0, V_0``; ``Z_0``/``V_0`` are the moments of the
  state at ``t = 0``) run through nowcastbox's Kalman smoother reproduces R's smoothed
  factors and smoothed data to ~1e-13;
* the log-likelihood R prints is the Gaussian log-likelihood **without** the constant
  ``-n/2 log(2 pi)``; nowcastbox's value at R's final parameters lies just above R's last
  printed value, as expected for the remaining iterations;
* the estimators diverge (``reference_divergence``): different starting values and EM
  stopping points; with the sequential block start of Bańbura & Modugno
  (``init="pca_given"``, used by these comparisons) nowcastbox reaches a **higher**
  likelihood than R on both panels (in 20 iterations on NYFED, ~13x faster). The default
  multi-start (``init="pca"``, best block order) ends 382.6 points above R on NYFED, at a
  different local maximum (R's "Real" factor lies outside its factor space; nowcasts
  within 0.14 sd of R's);
* R's ``yfcst`` for the EM is not the smoothed signal (or its common component) of the
  returned model and could not be reproduced from ``Res``; it is only compared loosely.
"""

from __future__ import annotations

import warnings
from functools import cache
from typing import Any

import numpy as np
import pandas as pd
import pytest

from nowcastbox.models import MixedFreqDFM
from nowcastbox.statespace import StateSpace, kalman_smoother, loglikelihood
from tests.reference_validation._helpers import (
    abs_correlations,
    legend,
    max_abs,
    pad,
    r_estimate,
    r_standardize,
    read_json,
    read_periods,
    symmetric,
)

pytestmark = pytest.mark.reference_validation

TOL_SMOOTHER = 1e-8
CASES = {"nyfed_em": "GDPC1", "sim_em": "gdp"}


@cache
def _panel(prefix: str) -> pd.DataFrame:
    if prefix == "nyfed_em":
        return read_periods("r/nyfed_em_panel.csv.gz")
    return pad(read_periods("inputs/simulated.csv"))


def r_model_t0(spec: dict[str, Any]) -> StateSpace:
    """R's EM model; ``Z_0``/``V_0`` describe the state at t = 0 (one step before the data)."""
    a = np.asarray(spec["A"], dtype=float)
    q = symmetric(np.asarray(spec["Q"], dtype=float))
    z0 = np.asarray(spec["Z_0"], dtype=float)
    v0 = symmetric(np.asarray(spec["V_0"], dtype=float))
    return StateSpace(
        a,
        np.asarray(spec["C"], dtype=float),
        q,
        np.asarray(spec["R"], dtype=float),
        initial_state=a @ z0,
        initial_state_cov=symmetric(a @ v0 @ a.T + q),
    )


def _standardized(prefix: str) -> tuple[dict[str, Any], np.ndarray]:
    spec = read_json(f"r/{prefix}.json")
    return spec, r_standardize(_panel(prefix), spec, "Mx", "Wx")


def _constant(z: np.ndarray) -> float:
    return 0.5 * float(np.isfinite(z).sum()) * np.log(2.0 * np.pi)


def _horizon(prefix: str, res: Any) -> tuple[pd.Series, pd.Series, float]:
    """Our and R's estimates after the last target observation (nowcast + forecasts)."""
    yfcst = read_periods(f"r/{prefix}_yfcst.csv", "Q")
    last = yfcst["y"].last_valid_index()
    ref = yfcst["out"].loc[yfcst.index > last]
    ours = res.nowcast["out_of_sample"].reindex(ref.index)
    return ours, ref, float(yfcst["y"].std())


@cache
@cache
def _fit(prefix: str, init: str = "pca_given") -> Any:
    """Fit with the sequential block start of Bańbura & Modugno (as R) by default."""
    panel = _panel(prefix)
    target = CASES[prefix]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if prefix == "nyfed_em":
            leg = legend("nyfed")
            freq = {
                n: ("Q" if f == 4 else "M")
                for n, f in zip(leg["name"], leg["frequency"], strict=True)
            }
            blocks = leg.set_index("name")[["Global", "Soft", "Real", "Labor"]]
            model = MixedFreqDFM(n_factors=1, factor_lags=1, blocks=blocks, init=init)
            return model.fit(panel, target, frequency=freq)
        return MixedFreqDFM(n_factors=2, factor_lags=1).fit(panel, target, frequency={target: "Q"})


# ---------------------------------------------------------------------- exact layer
@pytest.mark.parametrize("prefix", sorted(CASES))
def test_standardisation_matches(prefix: str) -> None:
    spec = read_json(f"r/{prefix}.json")
    panel = _panel(prefix)[list(spec["columns"])]
    assert max_abs(panel.mean().to_numpy(), spec["Mx"]) < 1e-10
    assert max_abs(panel.std().to_numpy(), spec["Wx"]) < 1e-10


@pytest.mark.parametrize("prefix", sorted(CASES))
def test_smoother_with_r_parameters(prefix: str) -> None:
    spec, z = _standardized(prefix)
    model = r_model_t0(spec)
    smoothed = kalman_smoother(model, z)
    x_sm = read_periods(f"r/{prefix}_x_sm.csv.gz")[list(spec["columns"])]
    assert max_abs(smoothed.smoothed_signal(), x_sm.to_numpy()) < TOL_SMOOTHER
    factors = read_periods(f"r/{prefix}_factors.csv")
    n_blocks = np.asarray(spec["blocks"]).shape[1]
    r = int(spec["r"])
    # R's state: per block r factors x 5 lags; the current factors open each block
    cols = [b * 5 * r + k for b in range(n_blocks) for k in range(r)]
    assert max_abs(smoothed.smoothed_state[:, cols], factors.to_numpy()) < TOL_SMOOTHER


@pytest.mark.reference_divergence
@pytest.mark.parametrize("prefix", sorted(CASES))
def test_initial_state_convention(prefix: str) -> None:
    """Using ``Z_0``/``V_0`` as the moments of the *first* state does not reproduce R."""
    spec, z = _standardized(prefix)
    naive = StateSpace(
        np.asarray(spec["A"]),
        np.asarray(spec["C"]),
        symmetric(np.asarray(spec["Q"])),
        np.asarray(spec["R"]),
        initial_state=np.asarray(spec["Z_0"]),
        initial_state_cov=symmetric(np.asarray(spec["V_0"])),
    )
    x_sm = read_periods(f"r/{prefix}_x_sm.csv.gz")[list(spec["columns"])].to_numpy()
    assert max_abs(kalman_smoother(naive, z).smoothed_signal(), x_sm) > 1e-3


@pytest.mark.parametrize("prefix", sorted(CASES))
def test_loglikelihood_definition(prefix: str) -> None:
    """R prints the log-likelihood without the Gaussian constant (EM path every 5 iterations)."""
    spec, z = _standardized(prefix)
    path = spec["loglik_path"]
    last = float(np.atleast_1d(path["to"])[-1])
    ours = loglikelihood(r_model_t0(spec), z) + _constant(z)
    # the final parameters come a few iterations after the last printed value
    assert last <= ours <= last + 2.0
    increments = np.diff(np.atleast_1d(path["to"]))
    assert bool((increments > 0).all())


# ---------------------------------------------------------------------- estimators
@pytest.mark.reference_divergence
@pytest.mark.parametrize("prefix", sorted(CASES))
def test_nowcastbox_optimum_not_worse(prefix: str) -> None:
    """Different starting values / stopping points; our optimum has a higher likelihood."""
    spec, z = _standardized(prefix)
    at_r = loglikelihood(r_model_t0(spec), z)
    res = _fit(prefix)
    assert res.converged
    assert res.loglikelihood > at_r
    assert res.loglikelihood - at_r < 0.01 * abs(at_r)  # documented: +20.6 (NYFED), +1.6 (sim)


@pytest.mark.reference_divergence
def test_nyfed_multistart_beats_sequential_start() -> None:
    """Default ``init="pca"`` (best block order) ends far above R and the sequential start."""
    spec, z = _standardized("nyfed_em")
    at_r = loglikelihood(r_model_t0(spec), z)
    sequential = _fit("nyfed_em")
    best = _fit("nyfed_em", "pca")
    assert best.info["initialization"]["block_order"] == "reversed"
    assert best.loglikelihood > sequential.loglikelihood > at_r  # documented: +382.6 vs R
    ours, ref, scale = _horizon("nyfed_em", best)
    assert max_abs(ours, ref) < 0.2 * scale  # documented: 0.136 sd


@pytest.mark.reference_divergence
def test_nyfed_nowcast_path() -> None:
    res = _fit("nyfed_em")
    ours, ref, scale = _horizon("nyfed_em", res)
    assert len(ref) >= 4
    assert not bool(ours.isna().any())
    assert max_abs(ours, ref) < 0.2 * scale  # documented: 0.067 sd
    factors = read_periods("r/nyfed_em_factors.csv")
    ours = res.factors.reindex(factors.index).to_numpy()
    corr = abs_correlations(ours, factors.to_numpy())
    assert corr.min() > 0.75  # documented: Global 0.98, Soft 0.82, Real 0.97, Labor 0.99


@pytest.mark.reference_divergence
def test_simulated_nowcast_path() -> None:
    res = _fit("sim_em")
    ours, ref, scale = _horizon("sim_em", res)
    assert not bool(ours.isna().any())
    assert max_abs(ours, ref) < 0.05 * scale  # documented: 0.013 sd
    factors = read_periods("r/sim_em_factors.csv")
    ours = res.factors.reindex(factors.index).to_numpy()
    design = np.column_stack([ours, np.ones(len(ours))])
    coef, *_ = np.linalg.lstsq(design, factors.to_numpy(), rcond=None)
    resid = factors.to_numpy() - design @ coef
    r2 = 1.0 - (resid**2).sum(axis=0) / ((factors - factors.mean()) ** 2).sum(axis=0).to_numpy()
    assert r2.min() > 0.999  # same factor space up to rotation


@pytest.mark.reference_divergence
@pytest.mark.parametrize("prefix", sorted(CASES))
def test_r_yfcst_is_not_the_returned_signal(prefix: str) -> None:
    """R's EM ``yfcst`` differs from the smoothed target signal of its own ``Res``."""
    spec = read_json(f"r/{prefix}.json")
    target = CASES[prefix]
    i = list(spec["columns"]).index(target)
    x_sm = read_periods(f"r/{prefix}_x_sm.csv.gz")[target]
    level = x_sm * spec["Wx"][i] + spec["Mx"][i]
    quarterly = level[level.index.month % 3 == 0]  # type: ignore[attr-defined]
    quarterly.index = quarterly.index.asfreq("Q")  # type: ignore[attr-defined]
    reference = r_estimate(read_periods(f"r/{prefix}_yfcst.csv", "Q"))
    gap = max_abs(quarterly.reindex(reference.index), reference)
    assert gap > 1e-3 * float(reference.std())
