"""Simulation of mixed-frequency DFMs with known parameters (helpers for the EM tests).

The helpers are imported by the other ``test_em*`` modules as
``tests.models.test_em_simulation``; the tests in this file check the simulator itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MM_WEIGHTS = np.array([1.0, 2.0, 3.0, 2.0, 1.0])


@dataclass
class SimulatedPanel:
    data: pd.DataFrame  # base grid, quarterly series stored in the 3rd month
    frequencies: dict[str, str]
    factors: dict[str, np.ndarray]  # block -> (T, r_b)
    common: pd.DataFrame  # true common component (monthly grid, quarterly: aggregated)
    truth: pd.DataFrame  # values without missing data (quarterly: aggregated latent)
    blocks: pd.DataFrame  # series x blocks membership
    transition: dict[str, np.ndarray]
    loadings: pd.DataFrame  # lag-0 loadings, series x factors
    idio_ar: pd.Series


def _var1(rng: np.random.Generator, n: int, A: np.ndarray) -> np.ndarray:
    r = A.shape[0]
    shocks = rng.standard_normal((n + 50, r))
    f = np.zeros((n + 50, r))
    for t in range(1, n + 50):
        f[t] = A @ f[t - 1] + shocks[t]
    return f[50:]


def _ar1(rng: np.random.Generator, n: int, rho: float, sigma: float) -> np.ndarray:
    e = np.zeros(n + 50)
    shocks = sigma * rng.standard_normal(n + 50)
    for t in range(1, n + 50):
        e[t] = rho * e[t - 1] + shocks[t]
    return e[50:]


def _aggregate(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    out = np.full_like(x, np.nan)
    L = weights.size
    for t in range(L - 1, x.size):
        out[t] = weights @ x[t - np.arange(L)]
    return out


def simulate_mixed_dfm(
    *,
    n_periods: int = 240,
    n_monthly: int = 20,
    n_quarterly: int = 2,
    blocks: dict[str, int] | None = None,
    block_share: float = 0.5,
    ar_coef: float = 0.7,
    idio_ar: float = 0.4,
    idio_scale: float = 0.6,
    quarterly_idio_scale: float = 0.15,
    ragged: int = 2,
    seed: int = 0,
    start: str = "2000-01",
    quarterly_weights: np.ndarray = MM_WEIGHTS,
) -> SimulatedPanel:
    """Simulate a monthly-quarterly DFM.

    Block "global" loads on every series; further blocks load on the first
    ``block_share`` fraction of the monthly series and on every quarterly series.
    The quarterly series are aggregates (``quarterly_weights``) of latent monthly series
    observed in the 3rd month of the quarter. The last ``ragged`` months of the monthly
    series alternate missing values (ragged edge); the quarterly series miss their last
    quarter.
    """
    rng = np.random.default_rng(seed)
    blocks = {"global": 1} if blocks is None else dict(blocks)
    idx = pd.period_range(start, periods=n_periods, freq="M")
    names_m = [f"m{i}" for i in range(n_monthly)]
    names_q = [f"q{j}" for j in range(n_quarterly)]
    names = names_m + names_q
    membership = pd.DataFrame(False, index=names, columns=list(blocks))
    membership["global"] = True
    for b in list(blocks)[1:]:
        k = max(1, int(block_share * n_monthly))
        membership.loc[names_m[:k], b] = True
        membership.loc[names_q, b] = True
    factors, transitions = {}, {}
    for b, r in blocks.items():
        A = ar_coef * np.eye(r) + 0.05 * rng.standard_normal((r, r))
        transitions[b] = A
        factors[b] = _var1(rng, n_periods, A)
    factor_names = [f"{b}_f{k + 1}" for b, r in blocks.items() for k in range(r)]
    loadings = pd.DataFrame(0.0, index=names, columns=factor_names)
    for b, r in blocks.items():
        cols = [f"{b}_f{k + 1}" for k in range(r)]
        members = membership.index[membership[b]]
        loadings.loc[members, cols] = rng.uniform(0.5, 1.5, (len(members), r)) * rng.choice(
            [-1.0, 1.0], (len(members), r)
        )
    F = np.hstack([factors[b] for b in blocks])
    common_latent = F @ loadings.to_numpy().T
    rhos = pd.Series(idio_ar, index=names)
    data, common, truth = {}, {}, {}
    for i, name in enumerate(names):
        quarterly = name in names_q
        scale = quarterly_idio_scale if quarterly else idio_scale
        e = _ar1(rng, n_periods, idio_ar, scale)
        x = common_latent[:, i] + e
        if quarterly:
            agg_x = _aggregate(x, quarterly_weights)
            agg_c = _aggregate(common_latent[:, i], quarterly_weights)
            truth[name], common[name] = agg_x, agg_c
            obs = agg_x.copy()
            obs[(idx.month % 3) != 0] = np.nan
            data[name] = obs
        else:
            truth[name], common[name] = x, common_latent[:, i]
            data[name] = x.copy()
    frame = pd.DataFrame(data, index=idx)
    for k in range(ragged):
        cols = names_m if k == 0 else names_m[k % 2 :: 2]
        frame.iloc[n_periods - 1 - k, [names.index(c) for c in cols]] = np.nan
    if n_quarterly:
        last_slot = np.flatnonzero(idx.month % 3 == 0)[-1]
        frame.iloc[last_slot, [names.index(c) for c in names_q]] = np.nan
    freqs = {n: ("Q" if n in names_q else "M") for n in names}
    return SimulatedPanel(
        data=frame,
        frequencies=freqs,
        factors=factors,
        common=pd.DataFrame(common, index=idx),
        truth=pd.DataFrame(truth, index=idx),
        blocks=membership,
        transition=transitions,
        loadings=loadings,
        idio_ar=rhos,
    )


def test_simulator_shapes_and_slots():
    sim = simulate_mixed_dfm(n_periods=60, n_monthly=6, n_quarterly=2, seed=1)
    assert sim.data.shape == (60, 8)
    q = sim.data["q0"]
    months = sim.data.index.month
    assert q[months % 3 != 0].isna().all()
    assert q[months % 3 == 0].iloc[2:-1].notna().all()
    assert np.isnan(sim.data.iloc[-1, :6]).all()


def test_simulator_aggregation_identity():
    sim = simulate_mixed_dfm(n_periods=48, n_monthly=4, n_quarterly=1, seed=2)
    q = sim.truth["q0"].to_numpy()
    c = sim.common["q0"].to_numpy()
    assert np.isfinite(q[4:]).all()
    assert np.isfinite(c[4:]).all()


def test_simulator_blocks():
    sim = simulate_mixed_dfm(
        n_periods=48, n_monthly=6, n_quarterly=1, blocks={"global": 1, "real": 1}, seed=3
    )
    assert sim.blocks["global"].all()
    assert sim.blocks["real"].sum() == 4
    assert set(sim.factors) == {"global", "real"}
