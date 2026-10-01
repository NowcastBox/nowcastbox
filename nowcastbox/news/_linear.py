r"""Batched linear smoothing of one state functional for many data sets.

For a fixed missing-data pattern the Kalman filter covariances, gains and the matrices
:math:`L_t` do not depend on the observed values, so the smoothed state is an affine
function of the data. This module evaluates

.. math:: g'\,\mathbb{E}[\alpha_\tau \mid y^{(k)}] + h, \qquad k = 1, \dots, K,

for :math:`K` data sets sharing one pattern in a single pass: the covariance recursions
run once and the mean recursions are vectorised over the data sets. It uses the
prediction form of the Kalman filter and the backward smoothing recursion of Durbin &
Koopman (2012, eqs. 4.24 and 4.39-4.44):

.. math::

    v_t = y_t - Z_t a_t - d_t,\quad F_t = Z_t P_t Z_t' + H_t,\quad
    K_t = T P_t Z_t' F_t^{-1},\quad L_t = T - K_t Z_t,

    a_{t+1} = T a_t + c + K_t v_t,\quad P_{t+1} = T P_t L_t' + R Q R',

    r_{t-1} = Z_t' F_t^{-1} v_t + L_t' r_t,\quad r_n = 0,\qquad
    \hat\alpha_\tau = a_\tau + P_\tau r_{\tau-1}.

Only the quantities for :math:`t \ge \tau` are stored. This is what makes the news
weights (one unit data set per release) and the level contributions (one data set per
series) cheap (Koopman & Harvey, 2003, observation weights).

References
----------
Durbin, J., & Koopman, S. J. (2012). *Time Series Analysis by State Space Methods*
(2nd ed.), sections 4.3 and 4.4.

Koopman, S. J., & Harvey, A. (2003). Computing observation weights for signal
extraction and filtering. *Journal of Economic Dynamics and Control*, 27(7),
1317-1333.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from nowcastbox.statespace import StateSpace

__all__ = ["batched_functional"]

FloatArray = NDArray[np.float64]


def _solve(f_mat: FloatArray, rhs: FloatArray) -> FloatArray:
    """``F^{-1} rhs`` (pseudo-inverse when ``F`` is singular)."""
    try:
        return np.asarray(np.linalg.solve(f_mat, rhs), dtype=np.float64)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(f_mat) @ rhs


def _observation_system(
    model: StateSpace, n_periods: int
) -> tuple[FloatArray, FloatArray, FloatArray, NDArray[np.int64]]:
    """Stores of ``Z``, ``d``, full ``H`` and the per-period index into them."""
    return (
        np.asarray(model.design_store, dtype=np.float64),
        np.asarray(model.obs_intercept_store, dtype=np.float64),
        np.asarray(model.obs_cov_matrix_store, dtype=np.float64),
        model.period_index(n_periods),
    )


def batched_functional(
    model: StateSpace,
    gain: FloatArray,
    intercept: float,
    pattern: NDArray[np.bool_],
    data: FloatArray,
    position: int,
) -> FloatArray:
    """Smoothed linear functional of the state at one period for many data sets.

    Parameters
    ----------
    model : StateSpace
        State-space model; a time-varying observation equation
        (``StateSpace(obs_index=...)``, e.g. calendar aggregation on weekly grids) must
        cover exactly ``n_periods`` periods.
    gain : numpy.ndarray, shape (n_states,)
        Vector ``g``.
    intercept : float
        ``h``.
    pattern : numpy.ndarray of bool, shape (n_periods, n_obs)
        Observed cells (common to every data set).
    data : numpy.ndarray, shape (n_periods, n_obs, K)
        Data sets; only the cells in ``pattern`` are read.
    position : int
        Period ``tau`` (0-based) of the functional.

    Returns
    -------
    numpy.ndarray, shape (K,)
        ``g' E[alpha_tau | data_k] + h`` for every data set.

    Raises
    ------
    ValueError
        On inconsistent shapes, a position outside the sample or a time-varying model
        that does not cover ``n_periods`` periods.

    Examples
    --------
    >>> import numpy as np
    >>> from nowcastbox.statespace import StateSpace, kalman_smoother
    >>> ssm = StateSpace([[0.5]], [[1.0]], [[1.0]], [1.0])
    >>> y = np.array([[1.0], [np.nan], [0.5]])
    >>> out = batched_functional(ssm, np.ones(1), 0.0, ~np.isnan(y), y[:, :, None], 1)
    >>> ref = kalman_smoother(ssm, y).smoothed_state[1, 0]
    >>> bool(np.isclose(out[0], ref))
    True
    """
    n_periods, n_obs, n_sets = data.shape
    if pattern.shape != (n_periods, n_obs) or n_obs != model.n_obs:
        raise ValueError("pattern/data shapes do not match the model.")
    if not 0 <= position < n_periods:
        raise ValueError(f"position {position} outside the sample of {n_periods} periods.")
    designs, intercepts, covs, index = _observation_system(model, n_periods)
    t_mat, c_vec = model.T, model.c
    rqr = model.R @ model.Q @ model.R.T
    a = np.repeat(model.a0[:, None], n_sets, axis=1)
    p_mat = np.array(model.P0, dtype=np.float64)
    stored: list[tuple[FloatArray, FloatArray]] = []
    a_tau, p_tau = a, p_mat
    for t in range(n_periods):
        if t == position:
            a_tau, p_tau = a.copy(), p_mat.copy()
        obs = pattern[t]
        if obs.any():
            k = index[t]
            zt = designs[k][obs]
            v = data[t, obs, :] - zt @ a - intercepts[k][obs, None]
            m_mat = p_mat @ zt.T
            f_mat = zt @ m_mat + covs[k][np.ix_(obs, obs)]
            sol = _solve(f_mat, np.hstack([v, zt]))
            finv_v, finv_z = sol[:, :n_sets], sol[:, n_sets:]
            l_mat = t_mat - t_mat @ m_mat @ finv_z
            a = t_mat @ (a + m_mat @ finv_v) + c_vec[:, None]
            u = zt.T @ finv_v
        else:
            l_mat = t_mat
            a = t_mat @ a + c_vec[:, None]
            u = np.zeros_like(a)
        if t >= position:
            stored.append((u, l_mat))
        p_mat = t_mat @ p_mat @ l_mat.T + rqr
        p_mat = 0.5 * (p_mat + p_mat.T)
    r = np.zeros_like(a)
    for u, l_mat in reversed(stored):
        r = u + l_mat.T @ r
    smoothed = a_tau + p_tau @ r
    return gain @ smoothed + intercept
