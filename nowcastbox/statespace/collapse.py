"""Collapsed observation vector for large cross-sections (Jungbacker & Koopman, 2015).

When the number of series ``n`` is much larger than the rank ``k`` of the design matrix
(the typical dynamic-factor setting, ``k = r`` factors), the observation equation can be
transformed, period by period, into an equivalent ``k``-dimensional one without losing
information about the state. With ``L_t L_t' = H_t`` (Cholesky factor of the observed
block of ``H``) and the thin SVD ``L_t^{-1} Z_t = U S V'``:

* ``y*_t = U_k' L_t^{-1} (y_t - d_t) = (S_k V_k') alpha_t + e*_t``, ``e*_t ~ N(0, I_k)``;
* the orthogonal complement ``y^+_t = U_perp' L_t^{-1} (y_t - d_t) ~ N(0, I_{p_t - k})`` is
  independent of the state.

Hence the filtered and smoothed states computed from ``y*`` are *exactly* those of the
original model and the log-likelihood differs only by a known term
``-1/2 [(p_t - k) log 2 pi + log|H_t| + ||y^+_t||^2]`` (Jungbacker & Koopman, 2015,
sec. 2). The transformed noise is white with unit variance, so the univariate filter
(Koopman & Durbin, 2000) applies even when ``H`` is a full matrix.

Missing values are handled per *pattern* of observed series: the transformation is
computed once for every distinct pattern (and observation system, for a time-varying
observation equation), which keeps the cost low for nowcasting panels (balanced body
plus a few ragged-edge patterns).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg
from numpy.typing import ArrayLike, NDArray

from nowcastbox.statespace.representation import StateSpace

__all__ = ["CollapsedObservations", "collapse_observations"]

FloatArray = NDArray[np.float64]
_LOG_2PI = float(np.log(2.0 * np.pi))


@dataclass(frozen=True)
class CollapsedObservations:
    """Collapsed (low-dimensional) observation equation.

    The collapsed model is ``y*_t = Z*_{j(t)} alpha_t + e*_t`` with ``e*_t ~ N(0, I)``,
    where ``j(t) = pattern_index[t]`` indexes the distinct missing-data patterns.

    Attributes
    ----------
    observations : numpy.ndarray, shape (n_periods, k_max)
        Collapsed observations ``y*_t``; unused trailing entries are ``NaN``.
    design : numpy.ndarray, shape (n_patterns, k_max, n_states)
        Collapsed design matrices ``Z*_j = S_k V_k'`` (zero-padded).
    pattern_index : numpy.ndarray of int, shape (n_periods,)
        Pattern used in each period.
    n_collapsed : numpy.ndarray of int, shape (n_periods,)
        Dimension ``k_t`` of the collapsed observation in each period.
    loglikelihood_adjustment : numpy.ndarray, shape (n_periods,)
        Term added to the collapsed log-likelihood of each period to obtain the
        log-likelihood of the original observations.

    Examples
    --------
    >>> import numpy as np
    >>> from nowcastbox.statespace import StateSpace, collapse_observations
    >>> ssm = StateSpace([[0.5]], np.ones((30, 1)), [[1.0]], np.ones(30))
    >>> col = collapse_observations(ssm, np.zeros((4, 30)))
    >>> col.observations.shape, col.n_patterns
    ((4, 1), 1)
    """

    observations: FloatArray
    design: FloatArray
    pattern_index: NDArray[np.int64]
    n_collapsed: NDArray[np.int64]
    loglikelihood_adjustment: FloatArray

    @property
    def n_patterns(self) -> int:
        """Number of distinct missing-data patterns."""
        return int(self.design.shape[0])

    @property
    def n_periods(self) -> int:
        """Number of periods."""
        return int(self.observations.shape[0])

    @property
    def max_dimension(self) -> int:
        """Largest collapsed dimension ``k_max``."""
        return int(self.observations.shape[1])


def _whiten(
    model: StateSpace,
    slot: int,
    idx: NDArray[np.int64],
    z_obs: FloatArray,
    y_obs: FloatArray,
) -> tuple[FloatArray, FloatArray, float]:
    """Return ``L^{-1} Z``, ``L^{-1} y`` and ``log|H|`` for the observed block ``idx``."""
    if model.obs_cov_is_diagonal:
        h_obs = model.obs_cov_diagonal_store[slot][idx]
        if np.any(h_obs <= 0.0):
            msg = (
                "collapse requires a positive definite observation covariance for the "
                f"observed series {idx.tolist()}"
            )
            raise ValueError(msg)
        scale = 1.0 / np.sqrt(h_obs)
        return z_obs * scale[:, None], y_obs * scale[:, None], float(np.sum(np.log(h_obs)))
    h_mat = model.obs_cov_matrix_store[slot][np.ix_(idx, idx)]
    try:
        chol = np.linalg.cholesky(h_mat)
    except np.linalg.LinAlgError as err:
        msg = (
            "collapse requires a positive definite observation covariance for the "
            f"observed series {idx.tolist()}"
        )
        raise ValueError(msg) from err
    z_tilde = scipy.linalg.solve_triangular(chol, z_obs, lower=True)
    y_tilde = scipy.linalg.solve_triangular(chol, y_obs, lower=True)
    return z_tilde, y_tilde, 2.0 * float(np.sum(np.log(np.diag(chol))))


def collapse_observations(
    model: StateSpace,
    observations: ArrayLike,
    *,
    rank_tol: float | None = None,
) -> CollapsedObservations:
    """Collapse the observation vector of ``model`` (Jungbacker & Koopman, 2015).

    Parameters
    ----------
    model : StateSpace
        State-space model. The observed block of ``H`` must be positive definite in
        every period (series with zero noise variance cannot be collapsed).
    observations : array_like, shape (n_periods, n_obs)
        Observations with ``NaN`` for missing values.
    rank_tol : float, optional
        Singular values of ``L_t^{-1} Z_t`` below this threshold are treated as zero.
        Default: ``max(S) * max(p_t, m) * eps``.

    Returns
    -------
    CollapsedObservations
        Collapsed observation equation and log-likelihood adjustments.

    Raises
    ------
    ValueError
        If the observed block of ``H`` is not positive definite or ``rank_tol < 0``.
    nowcastbox.core.exceptions.NowcastDataError
        If the observations are malformed (wrong shape, infinite values).

    Examples
    --------
    >>> import numpy as np
    >>> from nowcastbox.statespace import StateSpace, collapse_observations
    >>> rng = np.random.default_rng(0)
    >>> ssm = StateSpace([[0.9]], rng.normal(size=(50, 1)), [[1.0]], np.ones(50))
    >>> y = rng.normal(size=(10, 50))
    >>> y[-1, 25:] = np.nan
    >>> col = collapse_observations(ssm, y)
    >>> col.n_patterns, int(col.n_collapsed.max())
    (2, 1)
    """
    from nowcastbox.statespace.kalman import prepare_observations

    if rank_tol is not None and rank_tol < 0:
        msg = f"rank_tol must be non-negative, got {rank_tol}"
        raise ValueError(msg)
    y_raw = prepare_observations(model, observations)
    n_periods = y_raw.shape[0]
    slots = model.period_index(n_periods)
    y = y_raw - model.obs_intercept_store[slots]
    m = model.n_states
    mask = ~np.isnan(y)
    keys = np.column_stack([slots, mask.astype(np.int64)])
    unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1).astype(np.int64)
    patterns = unique_keys[:, 1:].astype(bool)
    pattern_slots = unique_keys[:, 0]

    pieces: list[tuple[FloatArray, FloatArray]] = []
    n_collapsed = np.zeros(n_periods, dtype=np.int64)
    adjustment = np.zeros(n_periods)
    for j, pattern in enumerate(patterns):
        idx = np.flatnonzero(pattern)
        rows = np.flatnonzero(inverse == j)
        if idx.size == 0:
            pieces.append((np.zeros((0, m)), np.full((rows.size, 0), np.nan)))
            continue
        slot = int(pattern_slots[j])
        z_obs = model.design_store[slot][idx]
        z_tilde, y_tilde, logdet = _whiten(model, slot, idx, z_obs, y[np.ix_(rows, idx)].T)
        u_mat, sing, vt_mat = np.linalg.svd(z_tilde, full_matrices=False)
        tol = (
            rank_tol
            if rank_tol is not None
            else float(sing.max()) * max(z_tilde.shape) * np.finfo(float).eps
        )
        k = int(np.sum(sing > tol))
        u_k = u_mat[:, :k]
        y_star = u_k.T @ y_tilde
        resid = y_tilde - u_k @ y_star
        adjustment[rows] = -0.5 * (
            (idx.size - k) * _LOG_2PI + logdet + np.sum(resid * resid, axis=0)
        )
        n_collapsed[rows] = k
        pieces.append((sing[:k, None] * vt_mat[:k], y_star.T))

    k_max = max(1, max(piece[0].shape[0] for piece in pieces))
    design = np.zeros((len(pieces), k_max, m))
    y_out = np.full((n_periods, k_max), np.nan)
    for j, (z_star, y_star_t) in enumerate(pieces):
        k = z_star.shape[0]
        design[j, :k] = z_star
        y_out[np.flatnonzero(inverse == j), :k] = y_star_t
    return CollapsedObservations(
        observations=y_out,
        design=design,
        pattern_index=inverse,
        n_collapsed=n_collapsed,
        loglikelihood_adjustment=adjustment,
    )
