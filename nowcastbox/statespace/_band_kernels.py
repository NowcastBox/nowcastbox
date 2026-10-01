"""Numba kernels of the structured (precision-based) smoother.

Private module used by :mod:`nowcastbox.statespace.structured`. Symmetric banded
matrices are stored in LAPACK lower-band layout: ``ab[d, j] = A[j + d, j]`` for
``d = 0..bw``. Dense right-hand sides are row-major ``(n, k)`` arrays of which only the
first ``ncols`` columns are used, so a single workspace can be reused for every group
(no allocation inside the kernels).

Algorithms: banded Cholesky factorization and solves (Golub & Van Loan, 2013, sec.
4.3); selected inversion of a banded precision matrix inside its band (Takahashi,
Fagan & Chin, 1973; Rue & Martino, 2007); block elimination of conditionally
independent groups (Schur complement; Rue & Held, 2005, ch. 2).
"""

from __future__ import annotations

import numpy as np
from numba import njit

# Re-association lets LLVM vectorize the dot products; no NaN/inf semantics are relaxed.
_FAST = {"reassoc", "contract"}


@njit(cache=True)
def band_add_template(  # pragma: no cover - numba
    ab: np.ndarray,
    rhs: np.ndarray,
    mat: np.ndarray,
    lin: np.ndarray,
    offsets: np.ndarray,
    step: int,
    p_start: int,
    p_stop: int,
) -> None:
    """Add ``sum_p x_p' mat x_p - 2 lin' x_p`` (``x_p = x[offsets + p step]``) to a band."""
    k = offsets.size
    for p in range(p_start, p_stop):
        base = p * step
        for a in range(k):
            va = base + offsets[a]
            rhs[va] += lin[a]
            for b in range(k):
                vb = base + offsets[b]
                if va >= vb:
                    ab[va - vb, vb] += mat[a, b]


@njit(cache=True)
def dense_add_template(  # pragma: no cover - numba
    dense: np.ndarray,
    rhs: np.ndarray,
    mat: np.ndarray,
    lin: np.ndarray,
    offsets: np.ndarray,
    step: int,
    p_start: int,
    p_stop: int,
) -> None:
    """Dense counterpart of :func:`band_add_template` (full symmetric storage)."""
    k = offsets.size
    for p in range(p_start, p_stop):
        base = p * step
        for a in range(k):
            va = base + offsets[a]
            rhs[va] += lin[a]
            for b in range(k):
                dense[va, base + offsets[b]] += mat[a, b]


@njit(cache=True)
def dense_add_observations(  # pragma: no cover - numba
    dense: np.ndarray,
    rhs: np.ndarray,
    f_idx: np.ndarray,
    f_val: np.ndarray,
    y: np.ndarray,
    h: np.ndarray,
) -> None:
    """Add ``sum_o f_o f_o' / h_o`` and ``sum_o f_o y_o / h_o`` (sparse ``f_o``)."""
    n_obs, kf = f_idx.shape
    for o in range(n_obs):
        hinv = 1.0 / h[o]
        for a in range(kf):
            va = f_idx[o, a]
            fa = f_val[o, a] * hinv
            rhs[va] += fa * y[o]
            for b in range(kf):
                dense[va, f_idx[o, b]] += fa * f_val[o, b]


@njit(cache=True)
def band_cholesky(ab: np.ndarray) -> bool:  # pragma: no cover - numba
    """In-place lower Cholesky factor of a symmetric positive definite band matrix."""
    bw = ab.shape[0] - 1
    n = ab.shape[1]
    for j in range(n):
        s = ab[0, j]
        for k in range(max(0, j - bw), j):
            lk = ab[j - k, k]
            s -= lk * lk
        if not s > 0.0:
            return False
        ljj = np.sqrt(s)
        ab[0, j] = ljj
        for i in range(j + 1, min(n, j + bw + 1)):
            s = ab[i - j, j]
            for k in range(max(0, i - bw), j):
                s -= ab[i - k, k] * ab[j - k, k]
            ab[i - j, j] = s / ljj
    return True


@njit(cache=True, fastmath=_FAST)
def band_solve(ab: np.ndarray, x: np.ndarray, ncols: int) -> None:  # pragma: no cover - numba
    """Solve ``L L' X = B`` in place for the first ``ncols`` columns of ``x`` (rows = n)."""
    bw = ab.shape[0] - 1
    n = ab.shape[1]
    for i in range(n):
        for k in range(max(0, i - bw), i):
            lik = ab[i - k, k]
            for c in range(ncols):
                x[i, c] -= lik * x[k, c]
        inv = 1.0 / ab[0, i]
        for c in range(ncols):
            x[i, c] *= inv
    for i in range(n - 1, -1, -1):
        for k in range(i + 1, min(n, i + bw + 1)):
            lki = ab[k - i, i]
            for c in range(ncols):
                x[i, c] -= lki * x[k, c]
        inv = 1.0 / ab[0, i]
        for c in range(ncols):
            x[i, c] *= inv


@njit(cache=True)
def band_solve_vector(ab: np.ndarray, x: np.ndarray) -> None:  # pragma: no cover - numba
    """Solve ``L L' x = b`` in place for a vector."""
    bw = ab.shape[0] - 1
    n = ab.shape[1]
    for i in range(n):
        s = x[i]
        for k in range(max(0, i - bw), i):
            s -= ab[i - k, k] * x[k]
        x[i] = s / ab[0, i]
    for i in range(n - 1, -1, -1):
        s = x[i]
        for k in range(i + 1, min(n, i + bw + 1)):
            s -= ab[k - i, i] * x[k]
        x[i] = s / ab[0, i]


@njit(cache=True, inline="always")
def _band_get(band: np.ndarray, i: int, j: int) -> float:  # pragma: no cover - numba
    if i >= j:
        return band[i - j, j]
    return band[j - i, i]


@njit(cache=True)
def band_selected_inverse(ab: np.ndarray, out: np.ndarray) -> None:  # pragma: no cover - numba
    """Entries of ``(L L')^{-1}`` inside the band (Takahashi recursions).

    ``out`` has the shape of ``ab`` and receives ``out[d, j] = Sigma[j + d, j]``.
    """
    bw = ab.shape[0] - 1
    n = ab.shape[1]
    for j in range(n - 1, -1, -1):
        ljj = ab[0, j]
        kmax = min(n - 1, j + bw)
        for i in range(kmax, j - 1, -1):
            s = 0.0
            for k in range(j + 1, kmax + 1):
                s += ab[k - j, j] * _band_get(out, i, k)
            delta = 1.0 / ljj if i == j else 0.0
            out[i - j, j] = (delta - s) / ljj


@njit(cache=True)
def _fill_design_rhs(  # pragma: no cover - numba
    target: np.ndarray, g_idx: np.ndarray, g_val: np.ndarray, n_obs: int
) -> None:
    """``target[:, :n_obs] = Gmat'`` (column ``o`` holds the sparse vector ``g_o``)."""
    n = target.shape[0]
    for i in range(n):
        for o in range(n_obs):
            target[i, o] = 0.0
    kg = g_idx.shape[1]
    for o in range(n_obs):
        for a in range(kg):
            target[g_idx[o, a], o] += g_val[o, a]


@njit(cache=True, fastmath=_FAST)
def _add_group_observations(  # pragma: no cover - numba
    ab: np.ndarray,
    rhs_g: np.ndarray,
    g_idx: np.ndarray,
    g_val: np.ndarray,
    y: np.ndarray,
    h: np.ndarray,
) -> None:
    """Add ``sum_o g_o g_o' / h_o`` to the band and ``sum_o g_o y_o / h_o`` to ``rhs_g``."""
    kg = g_idx.shape[1]
    for o in range(y.size):
        hinv = 1.0 / h[o]
        for a in range(kg):
            va = g_idx[o, a]
            ga = g_val[o, a] * hinv
            rhs_g[va] += ga * y[o]
            for b in range(kg):
                vb = g_idx[o, b]
                if va >= vb:
                    ab[va - vb, vb] += ga * g_val[o, b]


@njit(cache=True, fastmath=_FAST, inline="always")
def _scatter_outer(  # pragma: no cover - numba
    schur: np.ndarray, f_idx: np.ndarray, f_val: np.ndarray, o: int, o2: int, coef: float
) -> None:
    """``schur += coef (f_o f_o2' + f_o2 f_o')`` (once when ``o == o2``)."""
    kf = f_idx.shape[1]
    for a in range(kf):
        ca = coef * f_val[o, a]
        va = f_idx[o, a]
        for b in range(kf):
            vb = f_idx[o2, b]
            val = ca * f_val[o2, b]
            schur[va, vb] += val
            if o2 != o:
                schur[vb, va] += val


@njit(cache=True, fastmath=_FAST)
def _schur_update(  # pragma: no cover - numba
    g_idx: np.ndarray,
    g_val: np.ndarray,
    f_idx: np.ndarray,
    f_val: np.ndarray,
    y: np.ndarray,
    h: np.ndarray,
    x_g: np.ndarray,
    work_y: np.ndarray,
    work_row: np.ndarray,
    schur: np.ndarray,
    rhs_f: np.ndarray,
) -> None:
    """Add ``F' (D - D B D) F`` (``B = G Omega^{-1} G'`` from ``work_y``) and its linear term."""
    n_obs = y.size
    kg = g_idx.shape[1]
    kf = f_idx.shape[1]
    for o in range(n_obs):
        ho = 1.0 / h[o]
        # linear term: f_o (y_o - g_o' x_g) / h_o
        gx = 0.0
        for a in range(kg):
            gx += g_val[o, a] * x_g[g_idx[o, a]]
        r = (y[o] - gx) * ho
        for a in range(kf):
            rhs_f[f_idx[o, a]] += f_val[o, a] * r
        # row o of B (symmetric: columns o2 >= o only)
        for o2 in range(o, n_obs):
            work_row[o2] = 0.0
        for a in range(kg):
            ga = g_val[o, a]
            row = g_idx[o, a]
            for o2 in range(o, n_obs):
                work_row[o2] += ga * work_y[row, o2]
        for o2 in range(o, n_obs):
            coef = -work_row[o2] * ho / h[o2]
            if o == o2:
                coef += ho
            _scatter_outer(schur, f_idx, f_val, o, o2, coef)


@njit(cache=True)
def group_eliminate(  # pragma: no cover - numba
    ab: np.ndarray,
    rhs_g: np.ndarray,
    g_idx: np.ndarray,
    g_val: np.ndarray,
    f_idx: np.ndarray,
    f_val: np.ndarray,
    y: np.ndarray,
    h: np.ndarray,
    schur: np.ndarray,
    rhs_f: np.ndarray,
    work_y: np.ndarray,
    work_row: np.ndarray,
    x_g: np.ndarray,
) -> float:
    """Eliminate one conditionally independent group from the joint precision.

    On entry ``ab``/``rhs_g`` hold the prior part of the group precision and linear
    term; the group's observations (sparse rows ``g_o`` on the group, ``f_o`` on the
    common variables, value ``y_o``, variance ``h_o``) are added, the band is factorized
    in place and the Schur complement ``F' (D - D G Omega^{-1} G' D) F`` and the matching
    linear term are added to ``schur``/``rhs_f``. ``x_g`` receives ``Omega^{-1} b_g``.
    ``work_y`` (``n x n_obs``) and ``work_row`` (``n_obs``) are workspaces.

    Returns the log-determinant of the group precision (``nan`` if it is not positive
    definite).
    """
    _add_group_observations(ab, rhs_g, g_idx, g_val, y, h)
    if not band_cholesky(ab):
        return np.nan
    n = ab.shape[1]
    logdet = 0.0
    for j in range(n):
        logdet += 2.0 * np.log(ab[0, j])
        x_g[j] = rhs_g[j]
    band_solve_vector(ab, x_g)
    if y.size == 0 or f_idx.shape[1] == 0:
        return logdet
    _fill_design_rhs(work_y, g_idx, g_val, y.size)
    band_solve(ab, work_y, y.size)
    _schur_update(g_idx, g_val, f_idx, f_val, y, h, x_g, work_y, work_row, schur, rhs_f)
    return logdet


@njit(cache=True, fastmath=_FAST)
def _correct_mean(  # pragma: no cover - numba
    f_idx: np.ndarray,
    f_val: np.ndarray,
    h: np.ndarray,
    mu_f: np.ndarray,
    work_y: np.ndarray,
    work_row: np.ndarray,
    mu_g: np.ndarray,
) -> None:
    """``mu_g -= Y D F mu_f`` (``Y = Omega^{-1} G'`` in ``work_y``)."""
    n_obs = h.size
    kf = f_idx.shape[1]
    for o in range(n_obs):
        fm = 0.0
        for a in range(kf):
            fm += f_val[o, a] * mu_f[f_idx[o, a]]
        work_row[o] = fm / h[o]
    for j in range(mu_g.size):
        s = 0.0
        for o in range(n_obs):
            s += work_y[j, o] * work_row[o]
        mu_g[j] -= s


@njit(cache=True, fastmath=_FAST)
def _cross_rhs(  # pragma: no cover - numba
    g_idx: np.ndarray,
    g_val: np.ndarray,
    f_idx: np.ndarray,
    f_val: np.ndarray,
    h: np.ndarray,
    sigma_ff: np.ndarray,
    work_phi: np.ndarray,
    cross: np.ndarray,
) -> None:
    """``cross = -G' D F Sigma_ff`` (right-hand side of the cross-covariance solve)."""
    n_obs = h.size
    kg = g_idx.shape[1]
    kf = f_idx.shape[1]
    n_f = sigma_ff.shape[0]
    for o in range(n_obs):
        ho = 1.0 / h[o]
        for c in range(n_f):
            work_phi[o, c] = 0.0
        for a in range(kf):
            fa = f_val[o, a] * ho
            row = f_idx[o, a]
            for c in range(n_f):
                work_phi[o, c] += fa * sigma_ff[row, c]
    for j in range(cross.shape[0]):
        for c in range(n_f):
            cross[j, c] = 0.0
    for o in range(n_obs):
        for a in range(kg):
            ga = g_val[o, a]
            row = g_idx[o, a]
            for c in range(n_f):
                cross[row, c] -= ga * work_phi[o, c]


@njit(cache=True, fastmath=_FAST)
def _band_correction(  # pragma: no cover - numba
    f_idx: np.ndarray,
    f_val: np.ndarray,
    h: np.ndarray,
    cross: np.ndarray,
    work_y: np.ndarray,
    work_psi: np.ndarray,
    band_cov: np.ndarray,
) -> None:
    """``band_cov -= band(cross F' D Y')`` (``Y = Omega^{-1} G'`` in ``work_y``)."""
    n = band_cov.shape[1]
    n_obs = h.size
    kf = f_idx.shape[1]
    for j in range(n):
        for o in range(n_obs):
            s = 0.0
            for a in range(kf):
                s += cross[j, f_idx[o, a]] * f_val[o, a]
            work_psi[j, o] = s / h[o]
    bw = band_cov.shape[0] - 1
    for j in range(n):
        for d in range(min(bw, n - 1 - j) + 1):
            j1 = j + d
            s = 0.0
            for o in range(n_obs):
                s += work_psi[j1, o] * work_y[j, o]
            band_cov[d, j] -= s


@njit(cache=True)
def group_backsubstitute(  # pragma: no cover - numba
    ab: np.ndarray,
    g_idx: np.ndarray,
    g_val: np.ndarray,
    f_idx: np.ndarray,
    f_val: np.ndarray,
    h: np.ndarray,
    mu_f: np.ndarray,
    sigma_ff: np.ndarray,
    x_g: np.ndarray,
    work_y: np.ndarray,
    cross: np.ndarray,
    work_phi: np.ndarray,
    work_psi: np.ndarray,
    work_row: np.ndarray,
    band_cov: np.ndarray,
    mu_g: np.ndarray,
) -> None:
    """Posterior moments of an eliminated group given those of the common variables.

    With the factorized group precision ``Omega`` (``ab``), the group's observation
    rows and the common posterior ``N(mu_f, Sigma_ff)``:

    * ``mu_g = Omega^{-1} (b_g - Omega_gf mu_f)``;
    * ``cross = Cov(g, f) = -Omega^{-1} Omega_gf Sigma_ff`` (rows: group variables);
    * ``band_cov`` = band of ``Var(g) = Omega^{-1} - Cov(g, f) Omega_fg Omega^{-1}``.
    """
    n = ab.shape[1]
    n_f = mu_f.size
    mu_g[:] = x_g
    band_selected_inverse(ab, band_cov)
    if h.size == 0 or f_idx.shape[1] == 0 or n_f == 0:
        cross[:n, :] = 0.0
        return
    _fill_design_rhs(work_y, g_idx, g_val, h.size)
    band_solve(ab, work_y, h.size)
    _correct_mean(f_idx, f_val, h, mu_f, work_y, work_row, mu_g)
    _cross_rhs(g_idx, g_val, f_idx, f_val, h, sigma_ff, work_phi, cross)
    band_solve(ab, cross, n_f)
    _band_correction(f_idx, f_val, h, cross, work_y, work_psi, band_cov)
