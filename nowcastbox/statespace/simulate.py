"""Simulation helpers for state-space models (tests, tutorials, Monte Carlo).

* :func:`simulate_state_space` draws states and observations from a :class:`StateSpace`.
* :func:`random_state_space` builds a random stationary model with known dimensions.
* :func:`random_missing` blanks out observations at random (arbitrary missing patterns).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from nowcastbox.statespace.representation import StateSpace

__all__ = ["random_missing", "random_state_space", "simulate_state_space"]

FloatArray = NDArray[np.float64]
RandomState = int | np.random.Generator | None


def _psd_sqrt(matrix: FloatArray) -> FloatArray:
    """Symmetric square root of a PSD matrix (works for singular matrices)."""
    eigval, eigvec = np.linalg.eigh(0.5 * (matrix + matrix.T))
    return eigvec * np.sqrt(np.clip(eigval, 0.0, None))


def simulate_state_space(
    model: StateSpace,
    n_periods: int,
    *,
    initial_state: ArrayLike | None = None,
    random_state: RandomState = None,
) -> tuple[FloatArray, FloatArray]:
    """Simulate observations and states from a state-space model.

    Parameters
    ----------
    model : StateSpace
        Model to simulate from.
    n_periods : int
        Number of periods (``>= 1``).
    initial_state : array_like, shape (n_states,), optional
        Fixed value of ``alpha_1``. Default: drawn from ``N(a0, P0)``.
    random_state : int, numpy.random.Generator or None
        Seed or generator.

    Returns
    -------
    observations : numpy.ndarray, shape (n_periods, n_obs)
        Simulated ``y_t`` (complete).
    states : numpy.ndarray, shape (n_periods, n_states)
        Simulated ``alpha_t``.

    Raises
    ------
    ValueError
        If ``n_periods < 1``, ``initial_state`` has the wrong shape or the model has a
        time-varying observation equation for a different number of periods.

    Examples
    --------
    >>> from nowcastbox.statespace import StateSpace, simulate_state_space
    >>> ssm = StateSpace([[0.5]], [[1.0], [2.0]], [[1.0]], [0.1, 0.1])
    >>> y, alpha = simulate_state_space(ssm, 100, random_state=0)
    >>> y.shape, alpha.shape
    ((100, 2), (100, 1))
    """
    if int(n_periods) != n_periods or n_periods < 1:
        msg = f"n_periods must be a positive integer, got {n_periods}"
        raise ValueError(msg)
    n_periods = int(n_periods)
    rng = np.random.default_rng(random_state)
    m, n, g = model.n_states, model.n_obs, model.n_disturbances
    if initial_state is None:
        alpha = model.a0 + _psd_sqrt(model.P0) @ rng.standard_normal(m)
    else:
        alpha = np.asarray(initial_state, dtype=np.float64).copy()
        if alpha.shape != (m,):
            msg = f"initial_state must have shape ({m},), got {alpha.shape}"
            raise ValueError(msg)
    slots = model.period_index(n_periods)
    q_sqrt = model.R @ _psd_sqrt(model.Q)
    if model.obs_cov_is_diagonal:
        h_sd = np.sqrt(model.obs_cov_diagonal_store[slots])
        eps = rng.standard_normal((n_periods, n)) * h_sd
    else:
        roots = np.stack([_psd_sqrt(h) for h in model.obs_cov_matrix_store])
        eps = np.einsum("tij,tj->ti", roots[slots], rng.standard_normal((n_periods, n)))
    eta = rng.standard_normal((n_periods, g)) @ q_sqrt.T
    states = np.empty((n_periods, m))
    for t in range(n_periods):
        states[t] = alpha
        alpha = model.T @ alpha + model.c + eta[t]
    signal = np.einsum("tij,tj->ti", model.designs(n_periods), states)
    observations = signal + model.obs_intercept_store[slots] + eps
    return observations, states


def random_state_space(
    n_obs: int,
    n_states: int,
    *,
    n_disturbances: int | None = None,
    diagonal_obs_cov: bool = True,
    intercepts: bool = False,
    spectral_radius: float = 0.9,
    random_state: RandomState = None,
) -> StateSpace:
    """Random stationary state-space model (for tests and Monte Carlo).

    Parameters
    ----------
    n_obs : int
        Number of observed series ``n >= 1``.
    n_states : int
        Number of states ``m >= 1``.
    n_disturbances : int, optional
        Number of state disturbances ``g`` (``1 <= g <= m``); default ``m``.
    diagonal_obs_cov : bool, default True
        Diagonal ``H`` (vector) if True, otherwise a random full PD matrix.
    intercepts : bool, default False
        Draw non-zero intercepts ``c`` and ``d``.
    spectral_radius : float, default 0.9
        Spectral radius of ``T`` in ``(0, 1)``.
    random_state : int, numpy.random.Generator or None
        Seed or generator.

    Returns
    -------
    StateSpace
        Model initialized at its stationary distribution.

    Raises
    ------
    ValueError
        On invalid dimensions or ``spectral_radius`` outside ``(0, 1)``.

    Examples
    --------
    >>> from nowcastbox.statespace import random_state_space
    >>> ssm = random_state_space(5, 2, random_state=1)
    >>> ssm.n_obs, ssm.n_states, ssm.is_stationary
    (5, 2, True)
    """
    if n_obs < 1 or n_states < 1:
        msg = f"n_obs and n_states must be >= 1, got {n_obs} and {n_states}"
        raise ValueError(msg)
    g = n_states if n_disturbances is None else int(n_disturbances)
    if not 1 <= g <= n_states:
        msg = f"n_disturbances must be in [1, n_states], got {g}"
        raise ValueError(msg)
    if not 0.0 < spectral_radius < 1.0:
        msg = f"spectral_radius must be in (0, 1), got {spectral_radius}"
        raise ValueError(msg)
    rng = np.random.default_rng(random_state)
    t_mat = rng.standard_normal((n_states, n_states))
    t_mat *= spectral_radius / float(np.max(np.abs(np.linalg.eigvals(t_mat))))
    z_mat = rng.standard_normal((n_obs, n_states))
    r_mat = rng.standard_normal((n_states, g)) if g < n_states else np.eye(n_states)
    a_q = rng.standard_normal((g, g))
    q_mat = a_q @ a_q.T / g + 0.5 * np.eye(g)
    if diagonal_obs_cov:
        h_mat: FloatArray = rng.uniform(0.2, 1.5, size=n_obs)
    else:
        a_h = rng.standard_normal((n_obs, n_obs))
        h_mat = a_h @ a_h.T / n_obs + 0.3 * np.eye(n_obs)
    c_vec = rng.standard_normal(n_states) * 0.5 if intercepts else None
    d_vec = rng.standard_normal(n_obs) if intercepts else None
    return StateSpace(
        t_mat,
        z_mat,
        q_mat,
        h_mat,
        selection=r_mat,
        state_intercept=c_vec,
        obs_intercept=d_vec,
    )


def random_missing(
    observations: ArrayLike,
    fraction: float,
    *,
    random_state: RandomState = None,
) -> FloatArray:
    """Return a copy of ``observations`` with a random fraction of entries set to ``NaN``.

    Parameters
    ----------
    observations : array_like
        Array of observations.
    fraction : float
        Probability that each entry is blanked, in ``[0, 1]``.
    random_state : int, numpy.random.Generator or None
        Seed or generator.

    Returns
    -------
    numpy.ndarray
        Float copy with missing values.

    Raises
    ------
    ValueError
        If ``fraction`` is outside ``[0, 1]``.

    Examples
    --------
    >>> import numpy as np
    >>> from nowcastbox.statespace import random_missing
    >>> y = random_missing(np.ones((50, 4)), 0.5, random_state=0)
    >>> 0 < int(np.isnan(y).sum()) < 200
    True
    """
    if not 0.0 <= fraction <= 1.0:
        msg = f"fraction must be in [0, 1], got {fraction}"
        raise ValueError(msg)
    rng = np.random.default_rng(random_state)
    y = np.array(observations, dtype=np.float64, copy=True)
    y[rng.random(y.shape) < fraction] = np.nan
    return y
