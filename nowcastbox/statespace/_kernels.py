"""Numba kernels of the Kalman filter and the fixed-interval smoother.

Private module: the public entry points are :func:`nowcastbox.statespace.kalman_filter`
and :func:`nowcastbox.statespace.kalman_smoother`, which validate inputs.

Time-varying observation equations (needed by the collapsed filter and by missing-data
patterns with full ``H``) are supported through *pattern stores*: ``design`` has shape
``(n_patterns, n, m)`` and ``pattern_index[t]`` selects the slice used in period ``t``.
A time-invariant model is the special case ``n_patterns = 1``.

Besides the usual filter output the kernels accumulate, per period, the quantities
needed by the smoother (Durbin & Koopman, 2012, sec. 4.4 and 6.4):

* ``score[t]  = Z_t' F_t^{-1} v_t``      (m,)
* ``info[t]   = Z_t' F_t^{-1} Z_t``      (m, m)

restricted to the observed rows. With them the backward recursions
``r_{t-1} = u_t + G_t' T' r_t`` and ``N_{t-1} = M_t + G_t' T' N_t T G_t``,
``G_t = I - M_t P_t``, need no matrix inversion, which keeps the smoother valid for
singular predicted covariances (e.g. lagged factors in the state vector).

Performance: all work arrays are allocated once per call (the per-period loops do not
allocate); products with ``T`` use a CSR representation when ``T`` is sparse (companion
and idiosyncratic blocks make it so in dynamic factor models), and BLAS (``np.dot`` with
an output buffer) otherwise; dense ``m x m`` products use BLAS.
"""

from __future__ import annotations

import numpy as np
from numba import njit

_LOG_2PI = float(np.log(2.0 * np.pi))
# Prediction-error variances below this threshold are treated as zero: the observation
# is perfectly predicted, carries no information and is skipped (Durbin & Koopman, 2012,
# sec. 6.4.2).
F_TOL = 1e-14


# Re-association lets LLVM vectorize reductions; no NaN/inf semantics are relaxed.
_FAST = {"reassoc", "contract"}


@njit(cache=True)
def _symmetrize_inplace(matrix: np.ndarray) -> None:  # pragma: no cover - numba
    m = matrix.shape[0]
    for i in range(m):
        for j in range(i):
            v = 0.5 * (matrix[i, j] + matrix[j, i])
            matrix[i, j] = v
            matrix[j, i] = v


@njit(cache=True)
def _nonzero_pattern(design: np.ndarray) -> tuple:  # pragma: no cover - numba
    """Column indices of the non-zero loadings of every design row (sparse ``z_i``)."""
    n_pat, n_obs, m = design.shape
    idx = np.zeros((n_pat, n_obs, m), dtype=np.int64)
    cnt = np.zeros((n_pat, n_obs), dtype=np.int64)
    for k in range(n_pat):
        for i in range(n_obs):
            c = 0
            for j in range(m):
                if design[k, i, j] != 0.0:
                    idx[k, i, c] = j
                    c += 1
            cnt[k, i] = c
    return idx, cnt


@njit(cache=True, fastmath=_FAST)
def _left_multiply(  # pragma: no cover - numba
    t_ptr: np.ndarray,
    t_idx: np.ndarray,
    t_val: np.ndarray,
    transition: np.ndarray,
    sparse: bool,
    x: np.ndarray,
    out: np.ndarray,
) -> None:
    """``out = T x`` for an ``m x m`` matrix ``x``."""
    if not sparse:
        np.dot(transition, x, out)
        return
    m = x.shape[0]
    ncol = x.shape[1]
    for i in range(m):
        for c in range(ncol):
            out[i, c] = 0.0
        for q in range(t_ptr[i], t_ptr[i + 1]):
            v = t_val[q]
            k = t_idx[q]
            for c in range(ncol):
                out[i, c] += v * x[k, c]


@njit(cache=True, fastmath=_FAST)
def _right_multiply_t(  # pragma: no cover - numba
    t_ptr: np.ndarray,
    t_idx: np.ndarray,
    t_val: np.ndarray,
    transition_t: np.ndarray,
    sparse: bool,
    x: np.ndarray,
    out: np.ndarray,
) -> None:
    """``out = x T'`` (``transition_t`` is ``T'`` stored contiguously)."""
    if not sparse:
        np.dot(x, transition_t, out)
        return
    nrow = x.shape[0]
    m = transition_t.shape[0]
    for i in range(nrow):
        for j in range(m):
            s = 0.0
            for q in range(t_ptr[j], t_ptr[j + 1]):
                s += t_val[q] * x[i, t_idx[q]]
            out[i, j] = s


@njit(cache=True, fastmath=_FAST)
def _predict(  # pragma: no cover - numba
    a: np.ndarray,
    p: np.ndarray,
    transition: np.ndarray,
    transition_t: np.ndarray,
    state_intercept: np.ndarray,
    rqr: np.ndarray,
    t_ptr: np.ndarray,
    t_idx: np.ndarray,
    t_val: np.ndarray,
    sparse: bool,
    a_out: np.ndarray,
    p_out: np.ndarray,
    work: np.ndarray,
) -> None:
    """``a_out = T a + c`` and ``p_out = T p T' + R Q R'`` (symmetrized), no allocation."""
    m = a.size
    if sparse:
        for i in range(m):
            s = state_intercept[i]
            for q in range(t_ptr[i], t_ptr[i + 1]):
                s += t_val[q] * a[t_idx[q]]
            a_out[i] = s
    else:
        for i in range(m):
            s = state_intercept[i]
            for k in range(m):
                s += transition[i, k] * a[k]
            a_out[i] = s
    _left_multiply(t_ptr, t_idx, t_val, transition, sparse, p, work)
    _right_multiply_t(t_ptr, t_idx, t_val, transition_t, sparse, work, p_out)
    for i in range(m):
        for j in range(m):
            p_out[i, j] += rqr[i, j]
    _symmetrize_inplace(p_out)


@njit(cache=True, fastmath=_FAST)
def univariate_filter(  # noqa: C901  # pragma: no cover - numba
    y: np.ndarray,
    design: np.ndarray,
    obs_intercept: np.ndarray,
    obs_var: np.ndarray,
    pattern_index: np.ndarray,
    transition: np.ndarray,
    state_intercept: np.ndarray,
    rqr: np.ndarray,
    a1: np.ndarray,
    p1: np.ndarray,
    store_smoother: bool,
    t_ptr: np.ndarray,
    t_idx: np.ndarray,
    t_val: np.ndarray,
    sparse: bool,
) -> tuple:
    """Univariate treatment of multivariate observations (Koopman & Durbin, 2000).

    The per-observation updates are written as explicit loops that only visit the
    non-zero loadings of ``z_i`` (``O(m * nnz(z_i) + m^2)`` per observation, no
    allocation inside the period loop). ``t_ptr``/``t_idx``/``t_val`` is the CSR form of
    ``T`` (used when ``sparse``). ``M_t`` is accumulated in its lower triangle only.
    """
    n_periods, n_obs = y.shape
    m = transition.shape[0]
    a_pred = np.empty((n_periods + 1, m))
    p_pred = np.empty((n_periods + 1, m, m))
    a_filt = np.empty((n_periods, m))
    p_filt = np.empty((n_periods, m, m))
    ll_obs = np.zeros(n_periods)
    n_used = np.zeros(n_periods, dtype=np.int64)
    n_store = n_periods if store_smoother else 0
    score = np.zeros((n_store, m))
    info = np.zeros((n_store, m, m))
    t_mat_t = np.ascontiguousarray(transition.T)
    nz_idx, nz_cnt = _nonzero_pattern(design)

    a = a1.copy()
    p = p1.copy()
    _symmetrize_inplace(p)
    a_pred[0] = a
    p_pred[0] = p
    pz = np.empty(m)
    w = np.empty(m)
    g = np.empty((m, m))
    u = np.empty(m)
    mm = np.empty((m, m))
    work = np.empty((m, m))
    for t in range(n_periods):
        k = pattern_index[t]
        if store_smoother:
            g[:, :] = 0.0
            for j in range(m):
                g[j, j] = 1.0
            u[:] = 0.0
            mm[:, :] = 0.0
        ll = 0.0
        for i in range(n_obs):
            yi = y[t, i]
            if np.isnan(yi):
                continue
            cnt = nz_cnt[k, i]
            for j in range(m):
                pz[j] = 0.0
            for q in range(cnt):
                col = nz_idx[k, i, q]
                zc = design[k, i, col]
                for j in range(m):
                    pz[j] += p[col, j] * zc  # p symmetric: row access is contiguous
            f = obs_var[k, i]
            za = 0.0
            for q in range(cnt):
                col = nz_idx[k, i, q]
                f += design[k, i, col] * pz[col]
                za += design[k, i, col] * a[col]
            if f <= F_TOL:
                continue
            v = yi - obs_intercept[k, i] - za
            vf = v / f
            finv = 1.0 / f
            if store_smoother:
                for j in range(m):
                    w[j] = 0.0
                for q in range(cnt):
                    col = nz_idx[k, i, q]
                    zc = design[k, i, col]
                    for j in range(m):
                        w[j] += zc * g[col, j]
                for j in range(m):
                    u[j] += w[j] * vf
                    wj = w[j] * finv
                    kj = pz[j] * finv
                    for l in range(j + 1):
                        mm[j, l] += wj * w[l]
                    for l in range(m):
                        g[j, l] -= kj * w[l]
            for j in range(m):
                a[j] += pz[j] * vf
                kj = pz[j] * finv
                for l in range(m):
                    p[j, l] -= kj * pz[l]
            ll += -0.5 * (_LOG_2PI + np.log(f) + v * vf)
            n_used[t] += 1
        _symmetrize_inplace(p)
        a_filt[t] = a
        p_filt[t] = p
        ll_obs[t] = ll
        if store_smoother:
            score[t] = u
            for j in range(m):
                for l in range(j):
                    mm[l, j] = mm[j, l]
            info[t] = mm
        _predict(
            a, p, transition, t_mat_t, state_intercept, rqr, t_ptr, t_idx, t_val, sparse,
            a_pred[t + 1], p_pred[t + 1], work,
        )  # fmt: skip
        a[:] = a_pred[t + 1]
        p[:, :] = p_pred[t + 1]
    return a_pred, p_pred, a_filt, p_filt, ll_obs, n_used, score, info


@njit(cache=True)
def multivariate_filter(  # pragma: no cover - numba
    y: np.ndarray,
    design: np.ndarray,
    obs_intercept: np.ndarray,
    obs_cov: np.ndarray,
    pattern_index: np.ndarray,
    transition: np.ndarray,
    state_intercept: np.ndarray,
    rqr: np.ndarray,
    a1: np.ndarray,
    p1: np.ndarray,
    store_smoother: bool,
    t_ptr: np.ndarray,
    t_idx: np.ndarray,
    t_val: np.ndarray,
    sparse: bool,
) -> tuple:
    """Standard Kalman filter selecting the observed rows of each period."""
    n_periods = y.shape[0]
    m = transition.shape[0]
    a_pred = np.empty((n_periods + 1, m))
    p_pred = np.empty((n_periods + 1, m, m))
    a_filt = np.empty((n_periods, m))
    p_filt = np.empty((n_periods, m, m))
    ll_obs = np.zeros(n_periods)
    n_used = np.zeros(n_periods, dtype=np.int64)
    n_store = n_periods if store_smoother else 0
    score = np.zeros((n_store, m))
    info = np.zeros((n_store, m, m))
    t_mat_t = np.ascontiguousarray(transition.T)
    work = np.empty((m, m))

    a = a1.copy()
    p = p1.copy()
    _symmetrize_inplace(p)
    a_pred[0] = a
    p_pred[0] = p
    for t in range(n_periods):
        k = pattern_index[t]
        idx = np.flatnonzero(~np.isnan(y[t]))
        n_t = idx.size
        if n_t > 0:
            z = np.ascontiguousarray(design[k][idx, :])
            h = np.empty((n_t, n_t))
            for r in range(n_t):
                for s in range(n_t):
                    h[r, s] = obs_cov[k, idx[r], idx[s]]
            v = y[t, idx] - obs_intercept[k, idx] - z @ a
            pzt = p @ np.ascontiguousarray(z.T)
            f = z @ pzt + h
            _symmetrize_inplace(f)
            chol = np.linalg.cholesky(f)
            rhs = np.empty((n_t, m + 1))
            rhs[:, 0] = v
            rhs[:, 1:] = z
            sol = np.linalg.solve(f, rhs)
            finv_v = np.ascontiguousarray(sol[:, 0])
            finv_z = np.ascontiguousarray(sol[:, 1:])
            zt = np.ascontiguousarray(z.T)
            a = a + pzt @ finv_v
            p = p - pzt @ finv_z @ p
            _symmetrize_inplace(p)
            logdet = 2.0 * np.sum(np.log(np.diag(chol)))
            ll_obs[t] = -0.5 * (n_t * _LOG_2PI + logdet + v @ finv_v)
            n_used[t] = n_t
            if store_smoother:
                score[t] = zt @ finv_v
                mm = zt @ finv_z
                _symmetrize_inplace(mm)
                info[t] = mm
        a_filt[t] = a
        p_filt[t] = p
        _predict(
            a, p, transition, t_mat_t, state_intercept, rqr, t_ptr, t_idx, t_val, sparse,
            a_pred[t + 1], p_pred[t + 1], work,
        )  # fmt: skip
        a = a_pred[t + 1].copy()
        p = p_pred[t + 1].copy()
    return a_pred, p_pred, a_filt, p_filt, ll_obs, n_used, score, info


@njit(cache=True, fastmath=_FAST)
def _subtract(out: np.ndarray, a: np.ndarray, b: np.ndarray) -> None:  # pragma: no cover
    """``out = a - b`` elementwise (2-D)."""
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = a[i, j] - b[i, j]


@njit(cache=True, fastmath=_FAST)
def _add(out: np.ndarray, a: np.ndarray, b: np.ndarray) -> None:  # pragma: no cover
    """``out = a + b`` elementwise (2-D)."""
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = a[i, j] + b[i, j]


@njit(cache=True, fastmath=_FAST)
def _matvec_add(  # pragma: no cover - numba
    out: np.ndarray, base: np.ndarray, mat: np.ndarray, x: np.ndarray
) -> None:
    """``out = base + mat @ x`` (no allocation)."""
    for i in range(out.size):
        s = base[i]
        for j in range(x.size):
            s += mat[i, j] * x[j]
        out[i] = s


@njit(cache=True, fastmath=_FAST)
def fixed_interval_smoother(  # pragma: no cover - numba
    a_pred: np.ndarray,
    p_pred: np.ndarray,
    p_filt: np.ndarray,
    score: np.ndarray,
    info: np.ndarray,
    transition: np.ndarray,
    t_ptr: np.ndarray,
    t_idx: np.ndarray,
    t_val: np.ndarray,
    sparse: bool,
) -> tuple:
    """Backward state smoothing recursions with lag-one covariances.

    Returns smoothed states ``E[alpha_t | Y_n]``, covariances ``Var[alpha_t | Y_n]`` and
    ``Cov(alpha_{t+1}, alpha_t | Y_n)`` for ``t = 1..n`` (the last one involving the
    out-of-sample state ``alpha_{n+1}``), see Durbin & Koopman (2012, eqs. 4.44, 4.69).
    Seven dense ``m x m`` products per period (BLAS); products with ``T`` are sparse.
    """
    n_periods = p_filt.shape[0]
    m = transition.shape[0]
    a_s = np.empty((n_periods, m))
    v_s = np.empty((n_periods, m, m))
    autocov = np.empty((n_periods, m, m))
    r = np.zeros(m)
    r_new = np.empty(m)
    n_mat = np.zeros((m, m))
    w1 = np.empty((m, m))
    w2 = np.empty((m, m))
    w3 = np.empty((m, m))
    lt = np.empty((m, m))
    lt_t = np.empty((m, m))
    eye = np.eye(m)
    t_mat_t = np.ascontiguousarray(transition.T)
    for t in range(n_periods - 1, -1, -1):
        # here r, n_mat hold r_t, N_t (information from periods t+1, ..., n)
        # autocov_t = (I - P_{t+1} N_t) T P_{t|t}
        _left_multiply(t_ptr, t_idx, t_val, transition, sparse, p_filt[t], w1)
        np.dot(n_mat, w1, w2)
        np.dot(p_pred[t + 1], w2, w3)
        _subtract(autocov[t], w1, w3)
        pt = p_pred[t]
        # L_t' = (I - M_t P_t) T'
        np.dot(info[t], pt, w2)
        _subtract(w1, eye, w2)
        _right_multiply_t(t_ptr, t_idx, t_val, t_mat_t, sparse, w1, lt)
        _matvec_add(r_new, score[t], lt, r)
        r[:] = r_new
        lt_t[:, :] = lt.T
        np.dot(lt, n_mat, w2)
        np.dot(w2, lt_t, w3)
        _add(n_mat, info[t], w3)
        _symmetrize_inplace(n_mat)
        _matvec_add(a_s[t], a_pred[t], pt, r)
        np.dot(pt, n_mat, w2)
        np.dot(w2, pt, w3)
        _subtract(v_s[t], pt, w3)
        _symmetrize_inplace(v_s[t])
    return a_s, v_s, autocov
