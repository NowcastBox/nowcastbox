"""Reference implementations used by the statespace tests (statsmodels + naive NumPy)."""

from __future__ import annotations

import numpy as np
from statsmodels.tsa.statespace.kalman_smoother import KalmanSmoother

from nowcastbox.statespace import StateSpace


def statsmodels_smoother(model: StateSpace, y: np.ndarray):
    """Run statsmodels' Kalman smoother (known initialization) on ``model``."""
    ks = KalmanSmoother(model.n_obs, model.n_states, model.n_disturbances)
    ks.bind(np.asarray(y, dtype=float).copy())
    ks["design"] = model.Z
    ks["obs_intercept"] = model.d
    ks["obs_cov"] = model.obs_cov_matrix
    ks["transition"] = model.T
    ks["state_intercept"] = model.c
    ks["selection"] = model.R
    ks["state_cov"] = model.Q
    ks.initialize_known(np.array(model.a0), np.array(model.P0))
    return ks.smooth()


def naive_filter(model: StateSpace, y: np.ndarray):
    """Textbook multivariate filter with explicit inverses (row selection)."""
    n_periods = y.shape[0]
    m = model.n_states
    a = model.a0.copy()
    p = model.P0.copy()
    h = model.obs_cov_matrix
    rqr = model.state_disturbance_cov
    a_pred = [a.copy()]
    p_pred = [p.copy()]
    a_filt, p_filt, ll = [], [], 0.0
    for t in range(n_periods):
        obs = ~np.isnan(y[t])
        if obs.any():
            z = model.Z[obs]
            f = z @ p @ z.T + h[np.ix_(obs, obs)]
            v = y[t, obs] - model.d[obs] - z @ a
            finv = np.linalg.inv(f)
            k = p @ z.T @ finv
            a = a + k @ v
            p = p - k @ z @ p
            ll += -0.5 * (obs.sum() * np.log(2 * np.pi) + np.linalg.slogdet(f)[1] + v @ finv @ v)
        a_filt.append(a.copy())
        p_filt.append(p.copy())
        a = model.T @ a + model.c
        p = model.T @ p @ model.T.T + rqr
        a_pred.append(a.copy())
        p_pred.append(p.copy())
    assert len(a_filt) == n_periods and m == a.size
    return (np.array(a_pred), np.array(p_pred), np.array(a_filt), np.array(p_filt), ll)


def naive_rts(model: StateSpace, y: np.ndarray):
    """Rauch-Tung-Striebel smoother with explicit inverses + lag-one covariances."""
    a_pred, p_pred, a_filt, p_filt, _ = naive_filter(model, y)
    n_periods = y.shape[0]
    a_s = np.empty_like(a_filt)
    v_s = np.empty_like(p_filt)
    autocov = np.empty_like(p_filt)
    a_s[-1] = a_filt[-1]
    v_s[-1] = p_filt[-1]
    autocov[-1] = model.T @ p_filt[-1]
    for t in range(n_periods - 2, -1, -1):
        j = p_filt[t] @ model.T.T @ np.linalg.inv(p_pred[t + 1])
        a_s[t] = a_filt[t] + j @ (a_s[t + 1] - a_pred[t + 1])
        v_s[t] = p_filt[t] + j @ (v_s[t + 1] - p_pred[t + 1]) @ j.T
        autocov[t] = v_s[t + 1] @ j.T
    return a_s, v_s, autocov
