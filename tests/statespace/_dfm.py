"""Builders of dynamic-factor state-space models used by the structured-smoother tests.

The layout follows Bańbura & Modugno (2014): factor blocks in companion form (VAR(p)
with enough lags for the Mariano-Murasawa aggregation), one AR(1) idiosyncratic chain
per series (five states for quarterly series), small measurement noise.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg

from nowcastbox.statespace import StateSpace, companion_matrix, stationary_initial_cov

MM_WEIGHTS = np.array([1.0, 2.0, 3.0, 2.0, 1.0]) / 3.0


def dfm_state_space(
    n_monthly: int,
    n_quarterly: int,
    *,
    rng: np.random.Generator,
    n_factors: int = 1,
    factor_lags: int = 1,
    n_blocks: int = 1,
    idiosyncratic: str = "ar1",
    obs_var: float = 1e-4,
    intercepts: bool = False,
    initial_mean: bool = False,
    n_patterns: int = 0,
    n_periods: int | None = None,
) -> StateSpace:
    """Mixed-frequency DFM: ``n_blocks`` blocks of ``n_factors`` VAR(``factor_lags``) factors.

    Series ``i`` loads on block 0 and on block ``i % n_blocks``. With ``n_patterns > 0`` a
    time-varying observation equation with that many randomly perturbed loading/noise
    systems is built for ``n_periods`` periods.
    """
    n = n_monthly + n_quarterly
    r = n_factors
    s = max(factor_lags, 5 if n_quarterly else 1)
    blocks_t, blocks_q, blocks_p0 = [], [], []
    for _ in range(n_blocks):
        coefs = [np.diag(rng.uniform(0.2, 0.6, r)) / factor_lags for _ in range(factor_lags)]
        comp = companion_matrix(coefs, n_lags=s)
        a_q = rng.normal(size=(r, r))
        q = a_q @ a_q.T / r + 0.5 * np.eye(r)
        full = np.zeros_like(comp)
        full[:r, :r] = q
        blocks_t.append(comp)
        blocks_q.append(q)
        blocks_p0.append(stationary_initial_cov(comp, full))
    n_fs = n_blocks * r * s
    lengths = [1] * n_monthly + [5] * n_quarterly
    m = n_fs + (sum(lengths) if idiosyncratic == "ar1" else 0)
    g = n_blocks * r + (n if idiosyncratic == "ar1" else 0)
    t_mat = np.zeros((m, m))
    sel = np.zeros((m, g))
    q_mat = np.zeros((g, g))
    p0 = np.zeros((m, m))
    t_mat[:n_fs, :n_fs] = scipy.linalg.block_diag(*blocks_t)
    p0[:n_fs, :n_fs] = scipy.linalg.block_diag(*blocks_p0)
    for b in range(n_blocks):
        start = b * r * s
        sel[start : start + r, b * r : (b + 1) * r] = np.eye(r)
        q_mat[b * r : (b + 1) * r, b * r : (b + 1) * r] = blocks_q[b]
    z = np.zeros((n, m))
    offset = n_fs
    for i in range(n):
        blocks = sorted({0, i % n_blocks})
        lam = rng.uniform(0.3, 1.0, size=(len(blocks), r))
        weights = np.ones(1) if lengths[i] == 1 else MM_WEIGHTS
        for bi, b in enumerate(blocks):
            for lag, w in enumerate(weights):
                start = b * r * s + lag * r
                z[i, start : start + r] = w * lam[bi]
        if idiosyncratic == "ar1":
            rho = rng.uniform(-0.5, 0.8)
            var = rng.uniform(0.2, 1.0)
            idx = np.arange(offset, offset + lengths[i])
            t_mat[idx[0], idx[0]] = rho
            t_mat[idx[1:], idx[:-1]] = 1.0
            z[i, idx] = weights
            sel[idx[0], n_blocks * r + i] = 1.0
            q_mat[n_blocks * r + i, n_blocks * r + i] = var
            lags = np.abs(np.subtract.outer(np.arange(idx.size), np.arange(idx.size)))
            p0[np.ix_(idx, idx)] = var / (1.0 - rho**2) * rho**lags
            offset += lengths[i]
    h = np.full(n, obs_var) if idiosyncratic == "ar1" else rng.uniform(0.2, 1.0, n)
    kwargs: dict = {
        "selection": sel,
        "initial_state_cov": p0,
        "initial_state": rng.normal(size=m) * 0.3 if initial_mean else np.zeros(m),
    }
    if intercepts:
        c = np.zeros(m)
        c[[b * r * s for b in range(n_blocks)]] = rng.normal(size=n_blocks) * 0.2
        kwargs["state_intercept"] = c
        kwargs["obs_intercept"] = rng.normal(size=n) * 0.5
    if n_patterns:
        assert n_periods is not None
        zs = np.stack([z * (z != 0) * rng.uniform(0.8, 1.2, z.shape) for _ in range(n_patterns)])
        hs = np.stack([h * rng.uniform(0.5, 2.0, n) for _ in range(n_patterns)])
        d = kwargs.pop("obs_intercept", np.zeros(n))
        kwargs["obs_intercept"] = np.stack([d + 0.1 * k for k in range(n_patterns)])
        kwargs["obs_index"] = rng.integers(0, n_patterns, n_periods)
        return StateSpace(t_mat, zs, q_mat, hs, **kwargs)
    return StateSpace(t_mat, z, q_mat, h, **kwargs)


def ragged_panel(
    model: StateSpace, n_periods: int, rng: np.random.Generator, n_quarterly: int = 0
) -> np.ndarray:
    """Simulate from ``model`` and blank out a nowcasting-style missing pattern.

    The last ``n_quarterly`` series are observed only in the third month of a quarter.
    """
    from nowcastbox.statespace import simulate_state_space

    y, _ = simulate_state_space(model, n_periods, random_state=rng)
    n = model.n_obs
    if n_quarterly:
        y[np.arange(n_periods) % 3 != 2, n - n_quarterly :] = np.nan
    y[: n_periods // 6, : max(1, n // 5)] = np.nan  # late starters
    y[-1, : n // 2] = np.nan  # ragged edge
    y[-2:, : n // 4] = np.nan
    y[rng.random(y.shape) < 0.03] = np.nan  # scattered gaps
    return y
