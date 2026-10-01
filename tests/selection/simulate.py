"""Simulated factor models with known numbers of factors and shocks (test helpers)."""

from __future__ import annotations

import numpy as np


def static_factor_panel(
    n_periods: int, n_series: int, n_factors: int, *, seed: int, noise: float = 1.0
) -> np.ndarray:
    """``X = F Lambda' + e`` with iid N(0, 1) factors, loadings and noise."""
    rng = np.random.default_rng(seed)
    f = rng.normal(size=(n_periods, n_factors))
    lam = rng.normal(size=(n_series, n_factors))
    return f @ lam.T + noise * rng.normal(size=(n_periods, n_series))


def dynamic_factor_panel(
    n_periods: int,
    n_series: int,
    n_shocks: int,
    n_stacked_lags: int,
    *,
    seed: int,
    phi: float = 0.6,
    noise: float = 1.0,
) -> tuple[np.ndarray, int]:
    """Panel whose ``r = q (s + 1)`` static factors are ``(f_t, ..., f_{t-s})``.

    ``f_t = phi f_{t-1} + eps_t`` is a ``q``-variate AR(1) with iid shocks, so the
    static factors follow a VAR(1) whose innovation covariance has rank ``q``.

    Returns
    -------
    x : numpy.ndarray
        ``(n_periods, n_series)`` panel.
    n_factors : int
        Number of static factors ``r``.
    """
    rng = np.random.default_rng(seed)
    burn = 100
    total = n_periods + n_stacked_lags + burn
    f = np.zeros((total, n_shocks))
    eps = rng.normal(size=(total, n_shocks))
    for t in range(1, total):
        f[t] = phi * f[t - 1] + eps[t]
    f = f[burn:]
    stacked = np.hstack(
        [f[n_stacked_lags - j : n_stacked_lags - j + n_periods] for j in range(n_stacked_lags + 1)]
    )
    n_factors = stacked.shape[1]
    lam = rng.normal(size=(n_series, n_factors))
    x = stacked @ lam.T + noise * rng.normal(size=(n_periods, n_series))
    return x, n_factors


def reduced_rank_var_panel(
    n_periods: int,
    n_series: int,
    n_factors: int,
    n_shocks: int,
    rho: float,
    *,
    seed: int,
) -> np.ndarray:
    """DGP 3 of Bai & Ng (2007): ``F_t = rho F_{t-1} + R eps_t`` with ``rank(R) = q``.

    ``x_it = lambda_i' F_t + e_it`` with iid N(0, 1) loadings, shocks and noise (100
    burn-in periods). Used with the reference numbers produced by the R package
    ``nowcasting`` (called as a black box) in ``test_bai_ng_reference.py``.
    """
    rng = np.random.default_rng(seed)
    burn = 100
    r_mat = rng.normal(size=(n_factors, n_shocks))
    f = np.zeros((n_periods + burn, n_factors))
    for t in range(1, n_periods + burn):
        f[t] = rho * f[t - 1] + r_mat @ rng.normal(size=n_shocks)
    f = f[burn:]
    lam = rng.normal(size=(n_series, n_factors))
    return f @ lam.T + rng.normal(size=(n_periods, n_series))
