"""Linear Gaussian state-space representation.

The notation follows Durbin & Koopman (2012, ch. 3 and 4)::

    y_t       = Z alpha_t + d + eps_t,        eps_t ~ N(0, H)
    alpha_t+1 = T alpha_t + c + R eta_t,      eta_t ~ N(0, Q)
    alpha_1   ~ N(a0, P0)

``y_t`` is ``(n_obs,)``, ``alpha_t`` is ``(n_states,)`` and ``eta_t`` is
``(n_disturbances,)``. **Timing convention:** ``a0`` and ``P0`` are the mean and covariance
of the *first* state ``alpha_1`` (the predicted state of the first period), exactly as
``a_1``/``P_1`` in Durbin & Koopman and the ``"known"`` initialization of
``statsmodels``. Algorithms that start from a pre-sample state ``alpha_0`` (e.g. the EM
algorithm of Shumway & Stoffer, 1982) can prepend one period of missing observations to
``y``: the first state of the augmented sample then plays the role of ``alpha_0``.

The state equation is time invariant. The observation equation is time invariant by
default or, with ``obs_index``, time varying through a store of observation systems
indexed per period (``Z_t = design[obs_index[t]]``, likewise ``H_t`` and ``d_t``). ``H``
can be passed as a vector (its diagonal, the case handled by the univariate filter of
Koopman & Durbin, 2000) or as a full covariance matrix.

:func:`approximate_diffuse_initial_cov` builds the approximate diffuse initialization
(``P0 = kappa I`` on non-stationary states; Durbin & Koopman, 2012, sec. 5.1).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import scipy.linalg
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "StateSpace",
    "approximate_diffuse_initial_cov",
    "companion_matrix",
    "stationary_initial_cov",
]

FloatArray = NDArray[np.float64]

_SYMMETRY_RTOL = 1e-8
_PSD_TOL = 1e-8


def _as_float_array(value: ArrayLike, name: str, ndim: int | tuple[int, ...]) -> FloatArray:
    """Convert ``value`` to a finite, C-contiguous float64 array of the expected rank."""
    try:
        array = np.array(value, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as err:
        msg = f"{name} must be a numeric array, got {type(value).__name__}"
        raise TypeError(msg) from err
    allowed = (ndim,) if isinstance(ndim, int) else ndim
    if array.ndim not in allowed:
        msg = f"{name} must have {' or '.join(map(str, allowed))} dimension(s), got {array.ndim}"
        raise ValueError(msg)
    if not np.all(np.isfinite(array)):
        msg = f"{name} contains NaN or infinite values"
        raise ValueError(msg)
    return np.ascontiguousarray(array)


def _check_shape(array: NDArray[Any], name: str, shape: tuple[int, ...]) -> None:
    if array.shape != shape:
        msg = f"{name} must have shape {shape}, got {array.shape}"
        raise ValueError(msg)


def _check_covariance(matrix: FloatArray, name: str) -> None:
    """Raise ``ValueError`` if ``matrix`` is not symmetric positive semi-definite."""
    scale = max(1.0, float(np.max(np.abs(matrix))))
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=_SYMMETRY_RTOL * scale):
        msg = f"{name} must be symmetric"
        raise ValueError(msg)
    min_eig = float(np.linalg.eigvalsh(0.5 * (matrix + matrix.T)).min())
    if min_eig < -_PSD_TOL * scale:
        msg = f"{name} must be positive semi-definite (smallest eigenvalue {min_eig:.3g})"
        raise ValueError(msg)


def _readonly(array: NDArray[Any]) -> FloatArray:
    array.setflags(write=False)
    return array


def stationary_initial_cov(transition: ArrayLike, state_disturbance_cov: ArrayLike) -> FloatArray:
    """Unconditional covariance of a stationary state vector.

    Solves the discrete Lyapunov equation ``P = T P T' + R Q R'`` (Durbin & Koopman,
    2012, sec. 5.6.2).

    Parameters
    ----------
    transition : array_like, shape (n_states, n_states)
        Transition matrix ``T``. All eigenvalues must lie strictly inside the unit circle.
    state_disturbance_cov : array_like, shape (n_states, n_states)
        Covariance of the state innovation ``R Q R'``.

    Returns
    -------
    numpy.ndarray, shape (n_states, n_states)
        Symmetric unconditional covariance ``P``.

    Raises
    ------
    ValueError
        If ``T`` has an eigenvalue with modulus ``>= 1`` or the shapes are inconsistent.

    Examples
    --------
    >>> import numpy as np
    >>> from nowcastbox.statespace import stationary_initial_cov
    >>> float(stationary_initial_cov([[0.5]], [[0.75]])[0, 0])
    1.0
    """
    t_mat = _as_float_array(transition, "transition", 2)
    rqr = _as_float_array(state_disturbance_cov, "state_disturbance_cov", 2)
    m = t_mat.shape[0]
    _check_shape(t_mat, "transition", (m, m))
    _check_shape(rqr, "state_disturbance_cov", (m, m))
    radius = _spectral_radius(t_mat)
    if radius >= 1.0 - 1e-10:
        msg = (
            f"transition matrix is not stationary (spectral radius {radius:.6g} >= 1); "
            "provide initial_state_cov explicitly"
        )
        raise ValueError(msg)
    cov = scipy.linalg.solve_discrete_lyapunov(t_mat, rqr)
    return np.ascontiguousarray(0.5 * (cov + cov.T))


def companion_matrix(coefficients: Sequence[ArrayLike], n_lags: int | None = None) -> FloatArray:
    """Companion (first-order) form of a VAR(p) transition.

    For ``f_t = A_1 f_{t-1} + ... + A_p f_{t-p} + u_t`` with ``f_t`` of dimension ``r``,
    returns the ``(r s) x (r s)`` matrix of the stacked state
    ``(f_t', f_{t-1}', ..., f_{t-s+1}')'`` with ``s = max(p, n_lags)``. Extra lags beyond
    ``p`` (needed e.g. by the Mariano-Murasawa aggregation, which loads on five monthly
    lags) get zero coefficients.

    Parameters
    ----------
    coefficients : sequence of array_like, each (r, r)
        Autoregressive matrices ``[A_1, ..., A_p]`` (``p >= 1``).
    n_lags : int, optional
        Number of lag blocks ``s`` in the state. Defaults to ``p``; must be ``>= p``.

    Returns
    -------
    numpy.ndarray, shape (r * s, r * s)
        Companion transition matrix.

    Raises
    ------
    ValueError
        If no coefficient matrix is given, shapes differ or ``n_lags < p``.

    Examples
    --------
    >>> from nowcastbox.statespace import companion_matrix
    >>> companion_matrix([[[0.5]], [[0.2]]], n_lags=3)
    array([[0.5, 0.2, 0. ],
           [1. , 0. , 0. ],
           [0. , 1. , 0. ]])
    """
    if len(coefficients) == 0:
        msg = "coefficients must contain at least one matrix"
        raise ValueError(msg)
    mats = [_as_float_array(a, f"coefficients[{i}]", 2) for i, a in enumerate(coefficients)]
    r = mats[0].shape[0]
    for i, mat in enumerate(mats):
        _check_shape(mat, f"coefficients[{i}]", (r, r))
    p = len(mats)
    s = p if n_lags is None else int(n_lags)
    if s < p:
        msg = f"n_lags ({s}) must be >= the number of coefficient matrices ({p})"
        raise ValueError(msg)
    out = np.zeros((r * s, r * s))
    out[:r, : r * p] = np.hstack(mats)
    if s > 1:
        out[r:, : r * (s - 1)] = np.eye(r * (s - 1))
    return out


def _connected_blocks(*matrices: FloatArray) -> list[NDArray[np.int64]]:
    """Index sets of the connected components of the union of the non-zero patterns."""
    from scipy.sparse.csgraph import connected_components

    pattern = np.zeros(matrices[0].shape, dtype=bool)
    for matrix in matrices:
        pattern |= matrix != 0.0
    pattern |= pattern.T
    n_comp, labels = connected_components(pattern, directed=False)
    return [np.flatnonzero(labels == k).astype(np.int64) for k in range(n_comp)]


def approximate_diffuse_initial_cov(
    transition: ArrayLike,
    state_disturbance_cov: ArrayLike,
    *,
    kappa: float = 1e6,
    states: str | ArrayLike = "nonstationary",
) -> FloatArray:
    """Approximate diffuse initial covariance ``P0 = kappa I`` on (some of) the states.

    The exact diffuse initialization ``P_1 = kappa P_inf + P_*`` with ``kappa -> inf``
    (Durbin & Koopman, 2012, ch. 5; Harvey & Phillips, 1979) is approximated by a large
    finite ``kappa`` (Durbin & Koopman, 2012, sec. 5.1). Diffuse states get variance
    ``kappa`` and no correlation with the others; the remaining states get the stationary
    covariance of their own sub-system (Lyapunov equation).

    Parameters
    ----------
    transition : array_like, shape (n_states, n_states)
        Transition matrix ``T``.
    state_disturbance_cov : array_like, shape (n_states, n_states)
        ``R Q R'``.
    kappa : float, default 1e6
        Variance of the diffuse states (``> 0``).
    states : {"nonstationary", "all"} or array_like of int or bool, default "nonstationary"
        Which states are diffuse. ``"nonstationary"`` splits the state vector into
        independent blocks (connected components of the non-zero patterns of ``T`` and
        ``R Q R'``) and makes diffuse every block whose transition has an eigenvalue of
        modulus ``>= 1``; ``"all"`` gives ``kappa I``; an index array or boolean mask
        selects the diffuse states explicitly (the others must form a stationary
        sub-system).

    Returns
    -------
    numpy.ndarray, shape (n_states, n_states)
        Initial covariance, usable as ``initial_state_cov`` of :class:`StateSpace`.

    Raises
    ------
    ValueError
        If ``kappa <= 0``, ``states`` is invalid or the non-diffuse states are not
        stationary.

    Notes
    -----
    The log-likelihood of the first periods is dominated by ``kappa``; as in Durbin &
    Koopman (2012, sec. 7.2.2) drop the first ``d`` contributions
    (``FilterResult.loglikelihood_obs[d:]``) when comparing models, ``d`` being the
    number of diffuse states.

    Examples
    --------
    A random walk plus a stationary AR(1):

    >>> from nowcastbox.statespace import approximate_diffuse_initial_cov
    >>> approximate_diffuse_initial_cov([[1.0, 0.0], [0.0, 0.5]], [[1.0, 0.0], [0.0, 0.75]])
    array([[1.e+06, 0.e+00],
           [0.e+00, 1.e+00]])
    """
    t_mat = _as_float_array(transition, "transition", 2)
    rqr = _as_float_array(state_disturbance_cov, "state_disturbance_cov", 2)
    m = t_mat.shape[0]
    _check_shape(t_mat, "transition", (m, m))
    _check_shape(rqr, "state_disturbance_cov", (m, m))
    if not (np.isfinite(kappa) and kappa > 0.0):
        msg = f"kappa must be a positive finite number, got {kappa}"
        raise ValueError(msg)
    diffuse = _diffuse_mask(t_mat, rqr, states)
    p0 = np.zeros((m, m))
    p0[diffuse, diffuse] = float(kappa)
    rest = np.flatnonzero(~diffuse)
    if rest.size:
        sub = np.ix_(rest, rest)
        try:
            p0[sub] = stationary_initial_cov(t_mat[sub], rqr[sub])
        except ValueError as err:
            msg = "the non-diffuse states do not form a stationary sub-system; " + str(err)
            raise ValueError(msg) from err
    return p0


def _diffuse_mask(t_mat: FloatArray, rqr: FloatArray, states: str | ArrayLike) -> NDArray[np.bool_]:
    m = t_mat.shape[0]
    if isinstance(states, str):
        if states == "all":
            return np.ones(m, dtype=bool)
        if states != "nonstationary":
            msg = f"states must be 'nonstationary', 'all' or an index array, got {states!r}"
            raise ValueError(msg)
        mask = np.zeros(m, dtype=bool)
        for block in _connected_blocks(t_mat, rqr):
            if _spectral_radius(t_mat[np.ix_(block, block)]) >= 1.0 - 1e-10:
                mask[block] = True
        return mask
    sel = np.asarray(states)
    if sel.dtype == bool:
        if sel.shape != (m,):
            msg = f"a boolean states mask must have shape ({m},), got {sel.shape}"
            raise ValueError(msg)
        return sel.copy()
    if sel.ndim != 1 or not np.issubdtype(sel.dtype, np.integer):
        msg = "states must be 'nonstationary', 'all', a boolean mask or integer indices"
        raise ValueError(msg)
    if sel.size and (sel.min() < -m or sel.max() >= m):
        msg = f"state indices out of range for {m} states"
        raise ValueError(msg)
    mask = np.zeros(m, dtype=bool)
    mask[sel] = True
    return mask


class StateSpace:
    """Linear Gaussian state-space model.

    ``y_t = Z_t alpha_t + d_t + eps_t`` with ``eps_t ~ N(0, H_t)`` and
    ``alpha_{t+1} = T alpha_t + c + R eta_t`` with ``eta_t ~ N(0, Q)``;
    ``alpha_1 ~ N(a0, P0)`` (Durbin & Koopman, 2012, notation).

    The state equation is time invariant. The observation equation is time invariant
    by default; a **time-varying** observation equation is given as a *store* of
    ``k`` observation systems plus a per-period index ``obs_index``: in period ``t``
    the model uses ``Z_t = design[obs_index[t]]``, ``H_t = obs_cov[obs_index[t]]`` and
    ``d_t = obs_intercept[obs_index[t]]`` (e.g. calendar-dependent aggregation weights,
    or a few regimes of the measurement noise). The number of periods is then fixed
    to ``len(obs_index)``.

    Instances are immutable: arrays are copied on construction and exposed read-only.
    Use :meth:`replace` to derive a modified model.

    Parameters
    ----------
    transition : array_like, shape (n_states, n_states)
        Transition matrix ``T``.
    design : array_like, shape (n_obs, n_states) or (k, n_obs, n_states)
        Design (loadings) matrix ``Z``; a 3-D store when ``obs_index`` is given.
    state_cov : array_like, shape (n_disturbances, n_disturbances)
        Covariance ``Q`` of the state disturbances (symmetric PSD).
    obs_cov : array_like
        Observation-noise covariance ``H``. Time invariant: ``(n_obs,)`` (diagonal,
        non-negative) or ``(n_obs, n_obs)`` (symmetric PSD). With ``obs_index``:
        ``(k, n_obs)`` (diagonals) or ``(k, n_obs, n_obs)``.
    selection : array_like, shape (n_states, n_disturbances), optional
        Selection matrix ``R``. Defaults to the identity (then ``Q`` is ``m x m``).
    state_intercept : array_like, shape (n_states,), optional
        ``c``; defaults to zeros.
    obs_intercept : array_like, shape (n_obs,) or (k, n_obs), optional
        ``d``; defaults to zeros.
    initial_state : array_like, shape (n_states,), optional
        ``a0 = E[alpha_1]``. Defaults to the unconditional mean ``(I - T)^{-1} c`` when
        ``T`` is stationary, otherwise zeros.
    initial_state_cov : array_like, shape (n_states, n_states), optional
        ``P0 = Var[alpha_1]`` (symmetric PSD). Defaults to the stationary covariance
        (see :func:`stationary_initial_cov`); required when ``T`` is not stationary
        (see :func:`approximate_diffuse_initial_cov`).
    obs_index : array_like of int, shape (n_periods,), optional
        Index into the observation store for every period (time-varying observation
        equation). Every value must be in ``[0, k)``.

    Attributes
    ----------
    T, Z, R, Q, H, c, d, a0, P0 : numpy.ndarray
        Read-only system matrices (``H`` keeps the shape it was given with). ``Z``,
        ``H`` and ``d`` raise ``ValueError`` for a time-varying observation equation;
        use ``design_store``, ``obs_cov_store`` and ``obs_intercept_store`` then.
    n_states, n_obs, n_disturbances : int
        Dimensions ``m``, ``n`` and ``g``.

    Raises
    ------
    ValueError
        On inconsistent shapes, non-finite values or non-PSD covariance matrices.

    Examples
    --------
    AR(1) signal observed with noise:

    >>> from nowcastbox.statespace import StateSpace
    >>> ssm = StateSpace(
    ...     transition=[[0.8]], design=[[1.0], [0.5]], state_cov=[[1.0]], obs_cov=[0.5, 0.2]
    ... )
    >>> ssm.n_states, ssm.n_obs, ssm.obs_cov_is_diagonal
    (1, 2, True)
    >>> round(float(ssm.P0[0, 0]), 4)
    2.7778

    A time-varying loading (``Z_t = 1`` in odd periods, ``2`` in even ones):

    >>> tv = StateSpace([[0.8]], [[[1.0]], [[2.0]]], [[1.0]], [[0.5], [0.5]], obs_index=[0, 1, 0])
    >>> tv.is_time_varying, tv.n_periods, tv.design_at(1)
    (True, 3, array([[2.]]))
    """

    def __init__(
        self,
        transition: ArrayLike,
        design: ArrayLike,
        state_cov: ArrayLike,
        obs_cov: ArrayLike,
        *,
        selection: ArrayLike | None = None,
        state_intercept: ArrayLike | None = None,
        obs_intercept: ArrayLike | None = None,
        initial_state: ArrayLike | None = None,
        initial_state_cov: ArrayLike | None = None,
        obs_index: ArrayLike | None = None,
    ) -> None:
        t_mat = _as_float_array(transition, "transition", 2)
        m = t_mat.shape[0]
        if m < 1:
            msg = "the model needs at least one state"
            raise ValueError(msg)
        _check_shape(t_mat, "transition", (m, m))
        time_varying = obs_index is not None
        z_store, h_store, d_store, index = self._observation_stores(
            m, design, obs_cov, obs_intercept, obs_index
        )

        r_mat = np.eye(m) if selection is None else _as_float_array(selection, "selection", 2)
        if r_mat.shape[0] != m or r_mat.shape[1] < 1:
            msg = f"selection must have shape ({m}, g) with g >= 1, got {r_mat.shape}"
            raise ValueError(msg)
        g = r_mat.shape[1]
        q_mat = _as_float_array(state_cov, "state_cov", 2)
        _check_shape(q_mat, "state_cov", (g, g))
        _check_covariance(q_mat, "state_cov")

        c_vec = (
            np.zeros(m)
            if state_intercept is None
            else _as_float_array(state_intercept, "state_intercept", 1)
        )
        _check_shape(c_vec, "state_intercept", (m,))

        rqr = r_mat @ q_mat @ r_mat.T
        rqr = 0.5 * (rqr + rqr.T)
        stationary = _spectral_radius(t_mat) < 1.0 - 1e-10

        if initial_state is None:
            a0 = np.linalg.solve(np.eye(m) - t_mat, c_vec) if stationary else np.zeros(m)
        else:
            a0 = _as_float_array(initial_state, "initial_state", 1)
        _check_shape(a0, "initial_state", (m,))

        if initial_state_cov is None:
            if not stationary:
                msg = (
                    "transition matrix is not stationary; initial_state_cov must be given "
                    "explicitly (e.g. approximate_diffuse_initial_cov(T, RQR'))"
                )
                raise ValueError(msg)
            p0 = stationary_initial_cov(t_mat, rqr)
        else:
            p0 = _as_float_array(initial_state_cov, "initial_state_cov", 2)
            _check_shape(p0, "initial_state_cov", (m, m))
            _check_covariance(p0, "initial_state_cov")
            p0 = 0.5 * (p0 + p0.T)

        self._time_varying = time_varying
        self._T = _readonly(np.ascontiguousarray(t_mat))
        self._Zs = _readonly(np.ascontiguousarray(z_store))
        self._Hs = _readonly(np.ascontiguousarray(h_store))
        self._ds = _readonly(np.ascontiguousarray(d_store))
        self._obs_index = None if index is None else _readonly_int(index)
        self._R = _readonly(np.ascontiguousarray(r_mat))
        self._Q = _readonly(np.ascontiguousarray(0.5 * (q_mat + q_mat.T)))
        self._c = _readonly(np.ascontiguousarray(c_vec))
        self._a0 = _readonly(np.ascontiguousarray(a0))
        self._P0 = _readonly(np.ascontiguousarray(p0))
        self._RQR = _readonly(np.ascontiguousarray(rqr))

    @staticmethod
    def _observation_stores(
        m: int,
        design: ArrayLike,
        obs_cov: ArrayLike,
        obs_intercept: ArrayLike | None,
        obs_index: ArrayLike | None,
    ) -> tuple[FloatArray, FloatArray, FloatArray, NDArray[np.int64] | None]:
        """Validate the observation equation and return ``(Z, H, d)`` stores and the index."""
        lead = 1 if obs_index is not None else 0
        z = _as_float_array(design, "design", 2 + lead)
        z_store = z if lead else z[None]
        k, n = z_store.shape[0], z_store.shape[1]
        if n < 1:
            msg = "the model needs at least one observed series"
            raise ValueError(msg)
        if k < 1:
            msg = "the observation store must contain at least one design matrix"
            raise ValueError(msg)
        _check_shape(z_store, "design", (k, n, m))

        h = _as_float_array(obs_cov, "obs_cov", (1 + lead, 2 + lead))
        h_store = h if lead else h[None]
        if h_store.ndim == 2:
            _check_shape(h_store, "obs_cov", (k, n) if lead else (1, n))
            if np.any(h_store < 0):
                msg = "obs_cov (diagonal) must be non-negative"
                raise ValueError(msg)
        else:
            _check_shape(h_store, "obs_cov", (k, n, n) if lead else (1, n, n))
            for j in range(h_store.shape[0]):
                _check_covariance(h_store[j], "obs_cov")
            h_store = 0.5 * (h_store + np.swapaxes(h_store, 1, 2))

        if obs_intercept is None:
            d_store = np.zeros((k, n))
        else:
            d = _as_float_array(obs_intercept, "obs_intercept", 1 + lead)
            d_store = d if lead else d[None]
            _check_shape(d_store, "obs_intercept", (k, n) if lead else (1, n))

        index: NDArray[np.int64] | None = None
        if obs_index is not None:
            raw = np.asarray(obs_index)
            if raw.ndim != 1 or raw.size == 0 or not np.issubdtype(raw.dtype, np.integer):
                msg = "obs_index must be a non-empty 1-D array of integers"
                raise ValueError(msg)
            if raw.min() < 0 or raw.max() >= k:
                msg = f"obs_index values must be in [0, {k})"
                raise ValueError(msg)
            index = np.ascontiguousarray(raw, dtype=np.int64)
        return z_store, h_store, d_store, index

    # ------------------------------------------------------------------ matrices
    def _invariant(self, name: str) -> None:
        if self._time_varying:
            msg = (
                f"{name} is time varying in this model; use design_store / obs_cov_store / "
                "obs_intercept_store (or design_at(t)) instead"
            )
            raise ValueError(msg)

    @property
    def T(self) -> FloatArray:  # noqa: N802 - math notation
        """Transition matrix ``T``, shape ``(n_states, n_states)``."""
        return self._T

    @property
    def Z(self) -> FloatArray:  # noqa: N802 - math notation
        """Design matrix ``Z``, shape ``(n_obs, n_states)`` (time-invariant models only)."""
        self._invariant("Z")
        return self._Zs[0]

    @property
    def R(self) -> FloatArray:  # noqa: N802 - math notation
        """Selection matrix ``R``, shape ``(n_states, n_disturbances)``."""
        return self._R

    @property
    def Q(self) -> FloatArray:  # noqa: N802 - math notation
        """State-disturbance covariance ``Q``, shape ``(n_disturbances, n_disturbances)``."""
        return self._Q

    @property
    def H(self) -> FloatArray:  # noqa: N802 - math notation
        """Observation covariance ``H`` as given: ``(n_obs,)`` diagonal or ``(n_obs, n_obs)``."""
        self._invariant("H")
        return self._Hs[0]

    @property
    def c(self) -> FloatArray:
        """State intercept ``c``, shape ``(n_states,)``."""
        return self._c

    @property
    def d(self) -> FloatArray:
        """Observation intercept ``d``, shape ``(n_obs,)`` (time-invariant models only)."""
        self._invariant("d")
        return self._ds[0]

    @property
    def a0(self) -> FloatArray:
        """Mean of the first state ``alpha_1``, shape ``(n_states,)``."""
        return self._a0

    @property
    def P0(self) -> FloatArray:  # noqa: N802 - math notation
        """Covariance of the first state ``alpha_1``, shape ``(n_states, n_states)``."""
        return self._P0

    @property
    def state_disturbance_cov(self) -> FloatArray:
        """``R Q R'``, the covariance of the state innovation, shape ``(n_states, n_states)``."""
        return self._RQR

    # ------------------------------------------------------- observation stores
    @property
    def is_time_varying(self) -> bool:
        """Whether the observation equation is time varying (``obs_index`` given)."""
        return self._time_varying

    @property
    def obs_index(self) -> NDArray[np.int64] | None:
        """Per-period index into the observation store (``None`` if time invariant)."""
        return self._obs_index

    @property
    def n_periods(self) -> int | None:
        """Number of periods fixed by ``obs_index`` (``None`` if time invariant)."""
        return None if self._obs_index is None else int(self._obs_index.size)

    @property
    def design_store(self) -> FloatArray:
        """Design matrices, shape ``(k, n_obs, n_states)`` (``k = 1`` if time invariant)."""
        return self._Zs

    @property
    def obs_cov_store(self) -> FloatArray:
        """Observation covariances: ``(k, n_obs)`` diagonals or ``(k, n_obs, n_obs)``."""
        return self._Hs

    @property
    def obs_intercept_store(self) -> FloatArray:
        """Observation intercepts, shape ``(k, n_obs)``."""
        return self._ds

    @property
    def obs_cov_diagonal_store(self) -> FloatArray:
        """Diagonals of the stored observation covariances, shape ``(k, n_obs)``."""
        if self._Hs.ndim == 2:
            return self._Hs.copy()
        return np.ascontiguousarray(np.diagonal(self._Hs, axis1=1, axis2=2))

    @property
    def obs_cov_matrix_store(self) -> FloatArray:
        """Stored observation covariances as full matrices, shape ``(k, n_obs, n_obs)``."""
        if self._Hs.ndim == 3:
            return self._Hs.copy()
        k, n = self._Hs.shape
        out = np.zeros((k, n, n))
        out[:, np.arange(n), np.arange(n)] = self._Hs
        return out

    def period_index(self, n_periods: int) -> NDArray[np.int64]:
        """Store index of every period for a sample of ``n_periods`` periods.

        Parameters
        ----------
        n_periods : int
            Number of periods of the observations.

        Returns
        -------
        numpy.ndarray of int, shape (n_periods,)
            ``obs_index`` (time-varying model) or zeros.

        Raises
        ------
        ValueError
            If the model is time varying and ``n_periods != len(obs_index)``.

        Examples
        --------
        >>> from nowcastbox.statespace import StateSpace
        >>> StateSpace([[0.5]], [[1.0]], [[1.0]], [1.0]).period_index(3)
        array([0, 0, 0])
        """
        if self._obs_index is None:
            return np.zeros(int(n_periods), dtype=np.int64)
        if int(n_periods) != self._obs_index.size:
            msg = (
                f"the model has a time-varying observation equation for "
                f"{self._obs_index.size} periods, got {n_periods} periods"
            )
            raise ValueError(msg)
        return np.ascontiguousarray(self._obs_index)

    def design_at(self, period: int) -> FloatArray:
        """Design matrix ``Z_t`` of a 0-based period.

        Parameters
        ----------
        period : int
            0-based period (ignored for time-invariant models).

        Returns
        -------
        numpy.ndarray, shape (n_obs, n_states)
            Read-only design matrix.

        Raises
        ------
        IndexError
            If ``period`` is out of range of ``obs_index``.

        Examples
        --------
        >>> from nowcastbox.statespace import StateSpace
        >>> StateSpace([[0.5]], [[1.0]], [[1.0]], [1.0]).design_at(10)
        array([[1.]])
        """
        return self._Zs[self._slot(period)]

    def obs_cov_at(self, period: int) -> FloatArray:
        """Observation covariance ``H_t`` of a 0-based period, as a full matrix.

        Parameters
        ----------
        period : int
            0-based period (ignored for time-invariant models).

        Returns
        -------
        numpy.ndarray, shape (n_obs, n_obs)
            Observation-noise covariance.

        Raises
        ------
        IndexError
            If ``period`` is out of range of ``obs_index``.

        Examples
        --------
        >>> from nowcastbox.statespace import StateSpace
        >>> StateSpace([[0.5]], [[1.0], [1.0]], [[1.0]], [1.0, 2.0]).obs_cov_at(0)
        array([[1., 0.],
               [0., 2.]])
        """
        h = self._Hs[self._slot(period)]
        return np.diag(h) if h.ndim == 1 else h.copy()

    def obs_intercept_at(self, period: int) -> FloatArray:
        """Observation intercept ``d_t`` of a 0-based period.

        Parameters
        ----------
        period : int
            0-based period (ignored for time-invariant models).

        Returns
        -------
        numpy.ndarray, shape (n_obs,)
            Read-only intercept.

        Raises
        ------
        IndexError
            If ``period`` is out of range of ``obs_index``.

        Examples
        --------
        >>> from nowcastbox.statespace import StateSpace
        >>> StateSpace([[0.5]], [[1.0]], [[1.0]], [1.0], obs_intercept=[2.0]).obs_intercept_at(0)
        array([2.])
        """
        return self._ds[self._slot(period)]

    def _slot(self, period: int) -> int:
        if self._obs_index is None:
            return 0
        n_periods = self._obs_index.size
        if not -n_periods <= period < n_periods:
            msg = f"period {period} out of range for {n_periods} periods"
            raise IndexError(msg)
        return int(self._obs_index[period])

    def designs(self, n_periods: int) -> FloatArray:
        """Design matrices of every period, shape ``(n_periods, n_obs, n_states)``.

        Parameters
        ----------
        n_periods : int
            Number of periods (must equal ``len(obs_index)`` for time-varying models).

        Returns
        -------
        numpy.ndarray
            ``Z_t`` stacked over periods (a broadcast view for invariant models).

        Examples
        --------
        >>> from nowcastbox.statespace import StateSpace
        >>> StateSpace([[0.5]], [[1.0]], [[1.0]], [1.0]).designs(4).shape
        (4, 1, 1)
        """
        index = self.period_index(n_periods)
        if self._obs_index is None:
            return np.broadcast_to(self._Zs[0], (int(n_periods), *self._Zs.shape[1:]))
        return self._Zs[index]

    # --------------------------------------------------------------- dimensions
    @property
    def n_states(self) -> int:
        """Number of states ``m``."""
        return int(self._T.shape[0])

    @property
    def n_obs(self) -> int:
        """Number of observed series ``n``."""
        return int(self._Zs.shape[1])

    @property
    def n_disturbances(self) -> int:
        """Number of state disturbances ``g`` (columns of ``R``)."""
        return int(self._R.shape[1])

    @property
    def obs_cov_is_diagonal(self) -> bool:
        """Whether ``H`` (every ``H_t``) is diagonal (vector or zero off-diagonals)."""
        if self._Hs.ndim == 2:
            return True
        off = self._Hs.copy()
        idx = np.arange(self.n_obs)
        off[:, idx, idx] = 0.0
        return bool(np.count_nonzero(off) == 0)

    @property
    def obs_cov_diagonal(self) -> FloatArray:
        """Diagonal of ``H``, shape ``(n_obs,)`` (time-invariant models only)."""
        self._invariant("H")
        return self.obs_cov_diagonal_store[0]

    @property
    def obs_cov_matrix(self) -> FloatArray:
        """``H`` as a full ``(n_obs, n_obs)`` matrix (time-invariant models only)."""
        self._invariant("H")
        return self.obs_cov_matrix_store[0]

    @property
    def is_stationary(self) -> bool:
        """Whether all eigenvalues of ``T`` lie strictly inside the unit circle."""
        return _spectral_radius(self._T) < 1.0 - 1e-10

    # ------------------------------------------------------------------ helpers
    def to_dict(self) -> dict[str, Any]:
        """Return copies of the system matrices keyed by constructor argument name.

        Returns
        -------
        dict of str to numpy.ndarray
            Keys ``transition, design, state_cov, obs_cov, selection, state_intercept,
            obs_intercept, initial_state, initial_state_cov`` (plus ``obs_index`` for a
            time-varying observation equation, whose observation entries are stores).

        Examples
        --------
        >>> from nowcastbox.statespace import StateSpace
        >>> ssm = StateSpace([[0.5]], [[1.0]], [[1.0]], [1.0])
        >>> sorted(ssm.to_dict())[:3]
        ['design', 'initial_state', 'initial_state_cov']
        """
        tv = self._time_varying
        out: dict[str, Any] = {
            "transition": self._T.copy(),
            "design": self._Zs.copy() if tv else self._Zs[0].copy(),
            "state_cov": self._Q.copy(),
            "obs_cov": self._Hs.copy() if tv else self._Hs[0].copy(),
            "selection": self._R.copy(),
            "state_intercept": self._c.copy(),
            "obs_intercept": self._ds.copy() if tv else self._ds[0].copy(),
            "initial_state": self._a0.copy(),
            "initial_state_cov": self._P0.copy(),
        }
        if self._obs_index is not None:
            out["obs_index"] = self._obs_index.copy()
        return out

    def replace(self, **changes: Any) -> StateSpace:
        """Return a new model with some matrices replaced.

        Parameters
        ----------
        **changes
            Constructor arguments to override (e.g. ``design=...``). Passing
            ``initial_state=None`` / ``initial_state_cov=None`` recomputes the stationary
            defaults; ``obs_index`` switches to (or changes) a time-varying observation
            equation, in which case the observation entries must be stores.

        Returns
        -------
        StateSpace
            New validated model.

        Raises
        ------
        TypeError
            If an unknown argument name is given.

        Examples
        --------
        >>> from nowcastbox.statespace import StateSpace
        >>> ssm = StateSpace([[0.5]], [[1.0]], [[1.0]], [1.0])
        >>> ssm.replace(obs_cov=[2.0]).H
        array([2.])
        """
        params = self.to_dict()
        unknown = set(changes) - set(params) - {"obs_index"}
        if unknown:
            msg = f"unknown StateSpace argument(s): {sorted(unknown)}"
            raise TypeError(msg)
        params.update(changes)
        return StateSpace(
            params.pop("transition"),
            params.pop("design"),
            params.pop("state_cov"),
            params.pop("obs_cov"),
            **params,
        )

    def with_stationary_initialization(self) -> StateSpace:
        """Return a copy initialized at the unconditional mean and covariance.

        Returns
        -------
        StateSpace
            Model with ``a0 = (I - T)^{-1} c`` and ``P0`` solving the Lyapunov equation.

        Raises
        ------
        ValueError
            If ``T`` is not stationary.

        Examples
        --------
        >>> from nowcastbox.statespace import StateSpace
        >>> ssm = StateSpace([[0.5]], [[1.0]], [[0.75]], [1.0], initial_state_cov=[[9.0]])
        >>> float(ssm.with_stationary_initialization().P0[0, 0])
        1.0
        """
        if not self.is_stationary:
            msg = "transition matrix is not stationary; no stationary initialization exists"
            raise ValueError(msg)
        return self.replace(initial_state=None, initial_state_cov=None)

    def with_approximate_diffuse(
        self, kappa: float = 1e6, *, states: str | ArrayLike = "nonstationary"
    ) -> StateSpace:
        """Return a copy with an approximate diffuse initialization.

        Diffuse states get mean zero and variance ``kappa``; the others the stationary
        distribution of their sub-system (see :func:`approximate_diffuse_initial_cov`).

        Parameters
        ----------
        kappa : float, default 1e6
            Variance of the diffuse states.
        states : {"nonstationary", "all"} or array_like, default "nonstationary"
            Which states are diffuse (see :func:`approximate_diffuse_initial_cov`).

        Returns
        -------
        StateSpace
            Re-initialized model.

        Raises
        ------
        ValueError
            On invalid ``kappa``/``states``.

        Examples
        --------
        >>> from nowcastbox.statespace import StateSpace
        >>> rw = StateSpace([[1.0]], [[1.0]], [[1.0]], [1.0], initial_state_cov=[[1.0]])
        >>> float(rw.with_approximate_diffuse(1e7).P0[0, 0])
        10000000.0
        """
        p0 = approximate_diffuse_initial_cov(self._T, self._RQR, kappa=kappa, states=states)
        diffuse = _diffuse_mask(self._T, self._RQR, states)
        a0 = self._a0.copy()
        a0[diffuse] = 0.0
        return self.replace(initial_state=a0, initial_state_cov=p0)

    def filter(self, observations: ArrayLike, **kwargs: Any) -> Any:
        """Run the Kalman filter; shortcut for :func:`~nowcastbox.statespace.kalman_filter`.

        Parameters
        ----------
        observations : array_like, shape (n_periods, n_obs)
            Observations with ``NaN`` for missing values.
        **kwargs
            Passed to :func:`~nowcastbox.statespace.kalman_filter`.

        Returns
        -------
        FilterResult
            Filter output.

        Examples
        --------
        >>> import numpy as np
        >>> from nowcastbox.statespace import StateSpace
        >>> ssm = StateSpace([[0.5]], [[1.0]], [[1.0]], [1.0])
        >>> res = ssm.filter(np.array([[0.1], [np.nan], [0.3]]))
        >>> res.filtered_state.shape
        (3, 1)
        """
        from nowcastbox.statespace.kalman import kalman_filter

        return kalman_filter(self, observations, **kwargs)

    def smooth(self, observations: ArrayLike, **kwargs: Any) -> Any:
        """Run filter and smoother; shortcut for :func:`~nowcastbox.statespace.kalman_smoother`.

        Parameters
        ----------
        observations : array_like, shape (n_periods, n_obs)
            Observations with ``NaN`` for missing values.
        **kwargs
            Passed to :func:`~nowcastbox.statespace.kalman_smoother`.

        Returns
        -------
        SmootherResult
            Smoother output.

        Examples
        --------
        >>> import numpy as np
        >>> from nowcastbox.statespace import StateSpace
        >>> ssm = StateSpace([[0.5]], [[1.0]], [[1.0]], [1.0])
        >>> ssm.smooth(np.array([[0.1], [np.nan], [0.3]])).smoothed_state.shape
        (3, 1)
        """
        from nowcastbox.statespace.smoother import kalman_smoother

        return kalman_smoother(self, observations, **kwargs)

    def __repr__(self) -> str:
        h_kind = "diagonal" if self.obs_cov_is_diagonal else "full"
        tv = f", time_varying={self._Zs.shape[0]}" if self._time_varying else ""
        return (
            f"StateSpace(n_obs={self.n_obs}, n_states={self.n_states}, "
            f"n_disturbances={self.n_disturbances}, obs_cov={h_kind}{tv})"
        )


def _readonly_int(array: NDArray[np.int64]) -> NDArray[np.int64]:
    array = np.array(array, dtype=np.int64, copy=True)
    array.setflags(write=False)
    return array


def _spectral_radius(matrix: FloatArray) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(matrix))))
