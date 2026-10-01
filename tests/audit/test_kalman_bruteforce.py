"""Independent brute-force checks of the Kalman filter and smoother.

The log-likelihood and the smoothed moments are recomputed by stacking the whole state
path :math:`(\\alpha_1, \\dots, \\alpha_n)` and the observed entries of :math:`y` into one
joint Gaussian vector and conditioning directly (no recursion). Every filtering path
(univariate, multivariate, structured) must agree with it.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy import stats

from nowcastbox.models._em_steps import EMParameters, StateLayout, build_state_space
from nowcastbox.statespace import (
    StateSpace,
    kalman_filter,
    kalman_smoother,
    smoothed_moments,
)


def _joint_moments(model: StateSpace, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean and covariance of the stacked state path (time-invariant ``T, c, R, Q``)."""
    m = model.n_states
    T, R, Q = model.T, model.R, model.Q
    c = np.zeros(m) if model.c is None else np.asarray(model.c, dtype=float).reshape(-1)[:m]
    mean = np.zeros(n * m)
    cov = np.zeros((n * m, n * m))
    a, P = np.asarray(model.a0, float), np.asarray(model.P0, float)
    RQR = R @ Q @ R.T
    for t in range(n):
        mean[t * m : (t + 1) * m] = a
        cov[t * m : (t + 1) * m, t * m : (t + 1) * m] = P
        # covariances with earlier states: Cov(a_t, a_s) = T Cov(a_{t-1}, a_s)
        for s in range(t):
            prev = cov[(t - 1) * m : t * m, s * m : (s + 1) * m]
            block = T @ prev
            cov[t * m : (t + 1) * m, s * m : (s + 1) * m] = block
            cov[s * m : (s + 1) * m, t * m : (t + 1) * m] = block.T
        a = T @ a + c
        P = T @ P @ T.T + RQR
    return mean, cov


def _brute_force(model: StateSpace, y: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    n, p = y.shape
    m = model.n_states
    mean, cov = _joint_moments(model, n)
    Zs = model.designs(n)
    d = model.obs_intercept_store[model.period_index(n)]
    Hs = model.obs_cov_store[model.period_index(n)]
    if Hs.ndim == 2:  # diagonal store, one row per period
        Hs = np.stack([np.diag(h) for h in Hs])
    rows, means, sel = [], [], []
    for t in range(n):
        for i in range(p):
            if np.isnan(y[t, i]):
                continue
            row = np.zeros(n * m)
            row[t * m : (t + 1) * m] = Zs[t, i]
            rows.append(row)
            means.append(row @ mean + d[t, i])
            sel.append((t, i))
    L = np.array(rows)
    noise = np.zeros((len(sel), len(sel)))
    for a_, (t1, i1) in enumerate(sel):
        for b_, (t2, i2) in enumerate(sel):
            if t1 == t2:
                noise[a_, b_] = Hs[t1, i1, i2]
    S = L @ cov @ L.T + noise
    obs = np.array([y[t, i] for t, i in sel])
    ll = float(stats.multivariate_normal(np.array(means), S).logpdf(obs))
    K = cov @ L.T @ np.linalg.inv(S)
    post_mean = mean + K @ (obs - np.array(means))
    post_cov = cov - K @ L @ cov
    return ll, post_mean.reshape(n, m), post_cov


def _check(model: StateSpace, y: np.ndarray, methods: tuple[str, ...]) -> None:
    n, m = y.shape[0], model.n_states
    ll, a_s, V = _brute_force(model, y)
    for method in methods:
        assert kalman_filter(model, y, method=method).loglikelihood == pytest.approx(
            ll, rel=1e-9, abs=1e-8
        )
        sm = kalman_smoother(model, y, method=method)
        np.testing.assert_allclose(sm.smoothed_state, a_s, atol=1e-8)
        for t in range(n):
            blk = V[t * m : (t + 1) * m, t * m : (t + 1) * m]
            np.testing.assert_allclose(sm.smoothed_state_cov[t], blk, atol=1e-8)
        for t in range(n - 1):
            # convention: autocov[t] = Cov(alpha_{t+1}, alpha_t | Y)
            blk = V[(t + 1) * m : (t + 2) * m, t * m : (t + 1) * m]
            np.testing.assert_allclose(sm.smoothed_state_autocov[t], blk, atol=1e-8)


def test_random_model_with_missing_data_matches_joint_gaussian() -> None:
    rng = np.random.default_rng(11)
    m, p, n = 3, 4, 9
    T = 0.4 * rng.standard_normal((m, m))
    Z = rng.standard_normal((p, m))
    A = rng.standard_normal((2, 2))
    R = rng.standard_normal((m, 2))
    model = StateSpace(
        T,
        Z,
        A @ A.T + 0.1 * np.eye(2),
        rng.uniform(0.2, 1.0, p),
        selection=R,
        initial_state=rng.standard_normal(m),
        initial_state_cov=np.eye(m) * 1.5,
    )
    y = rng.standard_normal((n, p))
    y[rng.random((n, p)) < 0.35] = np.nan
    y[4] = np.nan  # a fully missing period
    _check(model, y, ("univariate", "multivariate"))


def test_full_measurement_covariance_matches_joint_gaussian() -> None:
    rng = np.random.default_rng(5)
    m, p, n = 2, 3, 7
    B = rng.standard_normal((p, p))
    model = StateSpace(
        np.diag([0.8, -0.3]),
        rng.standard_normal((p, m)),
        np.eye(m),
        B @ B.T + 0.2 * np.eye(p),
    )
    y = rng.standard_normal((n, p))
    y[rng.random((n, p)) < 0.3] = np.nan
    _check(model, y, ("multivariate",))


def _mixed_frequency_model() -> tuple[StateSpace, np.ndarray]:
    """Mariano-Murasawa DFM (AR(1) idiosyncratic) with quarterly series in the state."""
    layout = StateLayout(
        ["a", "b", "c", "gdp"],
        ["g"],
        [1],
        1,
        np.ones((4, 1), bool),
        [[1.0], [1.0], [1.0], [1, 2, 3, 2, 1]],
        "ar1",
    )
    loadings = np.zeros((4, layout.n_factor_states))
    loadings[:3, 0] = [0.9, -0.6, 0.4]
    loadings[3, :5] = 0.3 * np.array([1, 2, 3, 2, 1])
    params = EMParameters(
        (np.array([[0.6]]),),
        (np.array([[0.8]]),),
        loadings,
        np.array([0.3, -0.2, 0.5, 0.1]),
        np.array([0.4, 0.3, 0.5, 0.2]),
        np.full(4, 0.05),
    )
    model = build_state_space(params, layout)
    rng = np.random.default_rng(2)
    n = 10
    y = rng.standard_normal((n, 4))
    y[np.arange(n) % 3 != 2, 3] = np.nan
    y[-2:, :2] = np.nan
    return model, y


def test_mixed_frequency_dfm_all_paths_match_joint_gaussian() -> None:
    model, y = _mixed_frequency_model()
    _check(model, y, ("univariate", "multivariate"))
    ll, a_s, _ = _brute_force(model, y)
    mom = smoothed_moments(model, y, method="structured")
    assert mom.smoother.loglikelihood == pytest.approx(ll, rel=1e-9)
    np.testing.assert_allclose(mom.smoother.smoothed_state, a_s, atol=1e-8)


def test_calendar_design_on_weekly_grid_matches_joint_gaussian() -> None:
    """Time-varying observation equation (weekly grid, monthly + quarterly series)."""
    import nowcastbox as nb
    from nowcastbox.models import MixedFreqDFM

    sim = nb.simulate.weekly_dfm(n_weeks=120, n_weekly=2, n_monthly=1, random_state=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # two EM iterations only: not converged
        res = MixedFreqDFM(n_factors=1, idiosyncratic="iid", max_iter=2, tol=0).fit(sim.data, "gdp")
    layout, params = res.state_layout, res.em_parameters
    assert layout is not None and params is not None and layout.is_time_varying
    n = 40
    y = sim.data.standardize()[0].values[:n]
    model = build_state_space(params, layout, n)
    _check(model, y, ("univariate", "multivariate"))
