"""Numerical state-space core.

Kalman filter with arbitrary missing observations (univariate treatment of Koopman &
Durbin, 2000, compiled with Numba; multivariate fallback for full ``H``), collapsed
observation vectors for large panels (Jungbacker & Koopman, 2015), fixed-interval
smoother with lag-one covariances (de Jong, 1989; Durbin & Koopman, 2012), Gaussian
log-likelihood, the :class:`StateSpace` container and simulation helpers. No
statsmodels on the critical path (plan section 5.1).

Large dynamic factor models (innovation I2): :func:`detect_structure` recognizes
companion blocks and idiosyncratic chains loaded by a single series, and
:func:`structured_smoother` computes the exact smoother on the sparse precision of the
whole state path (cost ``O(N n^2)`` instead of ``O(n N^3)``); :func:`smoothed_moments`
is the EM E-step that picks the fastest exact algorithm. The observation equation may be
time varying (``StateSpace(..., obs_index=...)``) and
:func:`approximate_diffuse_initial_cov` provides the approximate diffuse initialization.

Notation (Durbin & Koopman, 2012)::

    y_t       = Z alpha_t + d + eps_t,    eps_t ~ N(0, H)
    alpha_t+1 = T alpha_t + c + R eta_t,  eta_t ~ N(0, Q),   alpha_1 ~ N(a0, P0)

Examples
--------
>>> import numpy as np
>>> from nowcastbox.statespace import StateSpace, kalman_smoother
>>> ssm = StateSpace([[0.8]], [[1.0], [0.5]], [[1.0]], [0.5, 0.2])
>>> y = np.array([[0.3, np.nan], [np.nan, 0.4]])
>>> kalman_smoother(ssm, y).smoothed_state.shape
(2, 1)
"""

from nowcastbox.statespace.collapse import CollapsedObservations, collapse_observations
from nowcastbox.statespace.kalman import (
    FilterMethod,
    FilterResult,
    kalman_filter,
    loglikelihood,
    prepare_observations,
)
from nowcastbox.statespace.representation import (
    StateSpace,
    approximate_diffuse_initial_cov,
    companion_matrix,
    stationary_initial_cov,
)
from nowcastbox.statespace.simulate import (
    random_missing,
    random_state_space,
    simulate_state_space,
)
from nowcastbox.statespace.smoother import SmootherResult, kalman_smoother, smooth
from nowcastbox.statespace.structure import (
    CompanionBlock,
    StateGroup,
    StateStructure,
    detect_structure,
)
from nowcastbox.statespace.structured import (
    SmoothedMoments,
    StructuredCovariance,
    StructuredSmootherResult,
    smoothed_moments,
    structured_smoother,
)

__all__ = [
    "CollapsedObservations",
    "CompanionBlock",
    "FilterMethod",
    "FilterResult",
    "SmoothedMoments",
    "SmootherResult",
    "StateGroup",
    "StateSpace",
    "StateStructure",
    "StructuredCovariance",
    "StructuredSmootherResult",
    "approximate_diffuse_initial_cov",
    "collapse_observations",
    "companion_matrix",
    "detect_structure",
    "kalman_filter",
    "kalman_smoother",
    "loglikelihood",
    "prepare_observations",
    "random_missing",
    "random_state_space",
    "simulate_state_space",
    "smooth",
    "smoothed_moments",
    "stationary_initial_cov",
    "structured_smoother",
]
