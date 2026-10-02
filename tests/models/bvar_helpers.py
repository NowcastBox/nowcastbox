"""Helpers of the large-BVAR tests: blocked VAR simulation and analytic conditioning."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nowcastbox.core.data import MixedFrequencyData


def stable_coefficients(
    n: int, lags: int, rng: np.random.Generator, radius: float = 0.6
) -> np.ndarray:
    """Random VAR coefficients ``(p, n, n)`` with companion spectral radius ``radius``."""
    A = rng.standard_normal((lags, n, n)) / np.sqrt(n * lags)
    comp = np.zeros((n * lags, n * lags))
    comp[:n] = np.hstack(list(A))
    if lags > 1:
        comp[n:, :-n] = np.eye(n * (lags - 1))
    rho = float(np.max(np.abs(np.linalg.eigvals(comp))))
    scale = radius / rho
    return np.stack([A[l] * scale ** (l + 1) for l in range(lags)])


def random_covariance(n: int, rng: np.random.Generator) -> np.ndarray:
    """Well-conditioned random covariance matrix."""
    L = rng.standard_normal((n, n)) / np.sqrt(n)
    return L @ L.T + 0.5 * np.eye(n)


def simulate_var(
    A: np.ndarray,
    c: np.ndarray,
    sigma: np.ndarray,
    n_obs: int,
    rng: np.random.Generator,
    burn: int = 200,
) -> np.ndarray:
    """Simulate ``Y_t = c + sum_l A_l Y_{t-l} + e_t`` (``n_obs`` rows after a burn-in)."""
    p, n, _ = A.shape
    G = np.linalg.cholesky(sigma)
    y = np.zeros((n_obs + burn + p, n))
    for t in range(p, len(y)):
        y[t] = c + sum(A[l] @ y[t - 1 - l] for l in range(p)) + G @ rng.standard_normal(n)
    return y[-n_obs:]


def blocked_panel(
    values: np.ndarray, n_monthly: int, n_quarterly: int, start: str = "1990Q1"
) -> MixedFrequencyData:
    """Monthly panel whose blocked vector is ``values``.

    Column order of ``values``: ``x1[m1], x1[m2], x1[m3], x2[m1], ..., q1, q2, ...``
    (the layout order of a panel with columns ``x1..xM, q1..qQ``).
    """
    n_q = values.shape[0]
    quarters = pd.period_range(start, periods=n_q, freq="Q")
    months = pd.period_range(quarters[0].asfreq("M", how="S"), periods=3 * n_q, freq="M")
    data: dict[str, np.ndarray] = {}
    for i in range(n_monthly):
        data[f"x{i + 1}"] = values[:, 3 * i : 3 * i + 3].reshape(-1)
    for j in range(n_quarterly):
        col = np.full(3 * n_q, np.nan)
        col[2::3] = values[:, 3 * n_monthly + j]
        data[f"q{j + 1}"] = col
    frame = pd.DataFrame(data, index=months)
    freqs = {**{f"x{i + 1}": "M" for i in range(n_monthly)}}
    freqs.update({f"q{j + 1}": "Q" for j in range(n_quarterly)})
    return MixedFrequencyData(frame, freqs)


def analytic_conditional(
    A: np.ndarray,
    c: np.ndarray,
    sigma: np.ndarray,
    presample: np.ndarray,
    future: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Conditional mean and variances of the edge from the companion-form moments.

    Independent of the library: the state ``alpha_i = (Y_i, ..., Y_{i-p+1})`` has
    ``E[alpha_i] = T E[alpha_{i-1}] + c*`` and ``Cov(alpha_i, alpha_j) = T^{i-j} V_j``
    with ``V_j = T V_{j-1} T' + R Sigma R'`` (``V_0 = 0``, known pre-sample); the joint
    Gaussian of the edge is conditioned on its observed entries by direct inversion.
    """
    p, n, _ = A.shape
    h = future.shape[0]
    m = n * p
    T = np.zeros((m, m))
    T[:n] = np.hstack(list(A))
    if p > 1:
        T[n:, :-n] = np.eye(m - n)
    cs = np.zeros(m)
    cs[:n] = c
    rqr = np.zeros((m, m))
    rqr[:n, :n] = sigma
    alpha = np.concatenate([presample[::-1][l] for l in range(p)])
    means, covs = [], []
    V = np.zeros((m, m))
    for _ in range(h):
        alpha = T @ alpha + cs
        V = T @ V @ T.T + rqr
        means.append(alpha[:n].copy())
        covs.append(V.copy())
    mu = np.concatenate(means)
    C = np.zeros((h * n, h * n))
    for i in range(h):
        for j in range(i + 1):
            block = (np.linalg.matrix_power(T, i - j) @ covs[j])[:n, :n]
            C[i * n : (i + 1) * n, j * n : (j + 1) * n] = block
            C[j * n : (j + 1) * n, i * n : (i + 1) * n] = block.T
    y = future.ravel()
    o = np.isfinite(y)
    u = ~o
    K = C[np.ix_(u, o)] @ np.linalg.inv(C[np.ix_(o, o)])
    mean = y.copy()
    mean[u] = mu[u] + K @ (y[o] - mu[o])
    var = np.zeros(h * n)
    var[u] = np.diag(C[np.ix_(u, u)] - K @ C[np.ix_(o, u)])
    return mean.reshape(h, n), var.reshape(h, n)
