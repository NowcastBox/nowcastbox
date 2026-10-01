r"""Structured (precision-based) smoother for dynamic-factor state-space models.

The dense Kalman smoother costs ``O(n m^3)`` for ``n`` periods and ``m`` states. In the
mixed-frequency dynamic factor model with AR(1) idiosyncratic components (Bańbura &
Modugno, 2014) ``m`` grows with the number of series ``N`` (one idiosyncratic chain per
series, five lags for quarterly series), so the E-step of the EM algorithm costs
``O(n N^3)``. This module exploits the structure detected by
:func:`~nowcastbox.statespace.detect_structure`:

* every state block is a companion (VAR) chain, so the whole state path is described by
  the current values ``x_t`` of every chain;
* idiosyncratic blocks are loaded by one series only and ``H`` is diagonal, so given the
  common (factor) path the private paths of different series are independent.

The joint posterior of all state paths is Gaussian with a sparse *arrowhead* precision
matrix (Rue & Held, 2005; Chan & Jeliazkov, 2009): one banded block per private group
plus the dense coupling with the common path. Each private group is eliminated with a
banded Cholesky factorization (Schur complement), the common path is solved densely,
and the required covariances of every private group are recovered from the band of its
precision (Takahashi, Fagan & Chin, 1973) plus a low-cost correction. The cost is
``O(N n^2 w)`` for a common path of width ``w`` instead of ``O(n N^3)``; results are
**exact** (identical to the Kalman smoother up to rounding).

What is computed is everything the EM algorithm needs: smoothed states, the
log-likelihood (Gaussian identity ``p(y) = p(y|x) p(x) / p(x|y)``), and the smoothed
covariances and lag-one covariances inside the common block, inside each private group
and between each private group and the common block. Covariances between two different
private groups are not formed (they are never needed by the M-step); accessing them
raises ``ValueError``.

Use :func:`smoothed_moments` for an EM E-step that picks the fastest exact algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import scipy.linalg
from numpy.typing import ArrayLike, NDArray

from nowcastbox._logging import get_logger
from nowcastbox.statespace import _band_kernels as _bk
from nowcastbox.statespace.kalman import prepare_observations
from nowcastbox.statespace.representation import StateSpace
from nowcastbox.statespace.smoother import SmootherResult, kalman_smoother
from nowcastbox.statespace.structure import StateStructure, detect_structure

__all__ = [
    "SmoothedMoments",
    "StructuredCovariance",
    "StructuredSmootherResult",
    "smoothed_moments",
    "structured_smoother",
]

logger = get_logger(__name__)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
MomentsMethod = Literal["auto", "structured", "univariate", "multivariate"]
_LOG_2PI = float(np.log(2.0 * np.pi))


# ====================================================================== layout
@dataclass(frozen=True)
class _GroupLayout:
    """Path layout of one group: ``var(p, j) = (p - lag_j + s - 1) w + off_j``."""

    width: int
    n_lags: int
    n_time: int
    bandwidth: int
    states: IntArray

    @property
    def size(self) -> int:
        return self.n_time * self.width


@dataclass(frozen=True)
class _Template:
    """Quadratic form ``x' M x - 2 l' x + const`` repeated over periods."""

    mat: FloatArray
    lin: FloatArray
    const: float
    offsets: IntArray
    p_start: int
    p_stop: int


def _spd_inverse_logdet(matrix: FloatArray) -> tuple[FloatArray, float]:
    """Inverse and log-determinant of a small symmetric positive definite matrix."""
    if matrix.shape[0] == 1:
        x = float(matrix[0, 0])
        return np.array([[1.0 / x]]), float(np.log(x))
    chol = np.linalg.cholesky(matrix)
    inv_chol = scipy.linalg.solve_triangular(chol, np.eye(matrix.shape[0]), lower=True)
    return inv_chol.T @ inv_chol, 2.0 * float(np.sum(np.log(np.diag(chol))))


def _var(
    structure: StateStructure, layout: _GroupLayout, periods: IntArray, states: IntArray
) -> IntArray:
    """Path-variable index of ``states`` at 0-based ``periods`` (broadcasting)."""
    lag = structure.state_lag[states]
    off = structure.state_offset[states]
    return (periods - lag + layout.n_lags - 1) * layout.width + off


def _layouts(structure: StateStructure, n_periods: int) -> list[_GroupLayout]:
    out = []
    for grp in structure.groups:
        w, s = grp.width, max(grp.n_lags, 1)
        out.append(
            _GroupLayout(
                width=w,
                n_lags=s,
                n_time=n_periods + s,
                bandwidth=max((s + 1) * w - 1, 0),
                states=grp.states,
            )
        )
    return out


def _templates(
    model: StateSpace, structure: StateStructure, layouts: list[_GroupLayout], n_periods: int
) -> tuple[list[list[_Template]], list[IntArray], float]:
    """Prior quadratic forms of every group, padding variables and ``log|Omega_0|`` terms.

    Returns the templates per group, the padding variables per group and
    ``sum_b (log|P0_b| + n log|Q_b|)``.
    """
    per_group: list[list[_Template]] = [[] for _ in structure.groups]
    padding: list[list[int]] = [[] for _ in structure.groups]
    logdet = 0.0
    t_mat, c = model.T, model.c
    for blk in structure.blocks:
        lay = layouts[blk.group]
        states = blk.states
        lags = structure.state_lag[states]
        offs = structure.state_offset[states]
        base0 = (lay.n_lags - 1 - lags) * lay.width + offs  # vars of alpha_1 at p = 0
        p0_inv, p0_logdet = _spd_inverse_logdet(blk.initial_cov)
        a0 = model.a0[states]
        per_group[blk.group].append(
            _Template(p0_inv, p0_inv @ a0, float(a0 @ p0_inv @ a0), base0, 0, 1)
        )
        leaders = blk.leaders
        r = leaders.size
        lead_off = lay.n_lags * lay.width + structure.state_offset[leaders]
        d_mat = np.hstack([np.eye(r), -t_mat[np.ix_(leaders, states)]])
        q_inv, q_logdet = _spd_inverse_logdet(blk.disturbance_cov)
        c_lead = c[leaders]
        mat = d_mat.T @ q_inv @ d_mat
        per_group[blk.group].append(
            _Template(
                0.5 * (mat + mat.T),
                d_mat.T @ q_inv @ c_lead,
                float(c_lead @ q_inv @ c_lead),
                np.concatenate([lead_off, base0]).astype(np.int64),
                0,
                n_periods,
            )
        )
        logdet += p0_logdet + n_periods * q_logdet
        for ld, length in zip(leaders, blk.chain_lengths, strict=True):
            off = int(structure.state_offset[ld])
            for tau in range(lay.n_lags - int(length)):
                padding[blk.group].append(tau * lay.width + off)
    return per_group, [np.asarray(p, dtype=np.int64) for p in padding], logdet


@dataclass
class _SeriesObs:
    """Observation rows of one series split into common (``f``) and group (``g``) parts."""

    periods: IntArray
    f_idx: IntArray
    f_val: FloatArray
    g_idx: IntArray
    g_val: FloatArray
    y: FloatArray
    h: FloatArray


def _series_observations(
    model: StateSpace,
    structure: StateStructure,
    layouts: list[_GroupLayout],
    y: FloatArray,
    index: IntArray,
) -> list[_SeriesObs | None]:
    zs, ds, hs = model.design_store, model.obs_intercept_store, model.obs_cov_diagonal_store
    out: list[_SeriesObs | None] = []
    common = layouts[0]
    for i in range(model.n_obs):
        periods = np.flatnonzero(~np.isnan(y[:, i])).astype(np.int64)
        if periods.size == 0:
            out.append(None)
            continue
        slots = index[periods]
        rows = zs[slots, i, :]
        loaded = np.flatnonzero(np.any(rows != 0.0, axis=0)).astype(np.int64)
        groups = structure.state_group[loaded]
        f_states = loaded[groups == 0]
        g_states = loaded[groups != 0]
        g = int(structure.series_group[i])
        if g_states.size:
            g_layout = layouts[g]
            g_idx = _var(structure, g_layout, periods[:, None], g_states[None, :])
        else:
            g_idx = np.zeros((periods.size, 0), dtype=np.int64)
        f_idx = (
            _var(structure, common, periods[:, None], f_states[None, :])
            if f_states.size
            else np.zeros((periods.size, 0), dtype=np.int64)
        )
        out.append(
            _SeriesObs(
                periods=periods,
                f_idx=np.ascontiguousarray(f_idx, dtype=np.int64),
                f_val=np.ascontiguousarray(rows[:, f_states]),
                g_idx=np.ascontiguousarray(g_idx, dtype=np.int64),
                g_val=np.ascontiguousarray(rows[:, g_states]),
                y=np.ascontiguousarray(y[periods, i] - ds[slots, i]),
                h=np.ascontiguousarray(hs[slots, i]),
            )
        )
    return out


def _prior_quadratic(templates: list[_Template], mean: FloatArray, width: int) -> float:
    total = 0.0
    for tpl in templates:
        p = np.arange(tpl.p_start, tpl.p_stop)
        x = mean[tpl.offsets[None, :] + p[:, None] * width]
        total += float(np.einsum("pi,ij,pj->", x, tpl.mat, x))
        total += -2.0 * float(np.sum(x @ tpl.lin)) + tpl.const * p.size
    return total


# ====================================================================== result
@dataclass(frozen=True, eq=False)
class _GroupMoments:
    states: IntArray
    cov: FloatArray
    autocov: FloatArray
    cross_cov: FloatArray  # (n, m_c, m_g): Cov(alpha_t^c, alpha_t^g)
    cross_autocov_cg: FloatArray  # (n, m_c, m_g): Cov(alpha_{t+1}^c, alpha_t^g)
    cross_autocov_gc: FloatArray  # (n, m_g, m_c): Cov(alpha_{t+1}^g, alpha_t^c)


class StructuredCovariance:
    """Read-only view of the smoothed (auto)covariances of a structured smoother.

    Behaves like the ``(n_periods, n_states, n_states)`` arrays of
    :class:`~nowcastbox.statespace.SmootherResult` for every entry that the structured
    smoother computes (inside the common block, inside a private group, and between a
    private group and the common block). Indexing is **orthogonal**: ``view[t, rows,
    cols]`` with integers, slices or 1-D integer arrays (``numpy.ix_`` meshes are
    accepted too) selects the outer product of the three index sets.

    Parameters
    ----------
    result : StructuredSmootherResult
        Owning result.
    lag : {0, 1}
        ``0`` for ``Var(alpha_t | Y)``, ``1`` for ``Cov(alpha_{t+1}, alpha_t | Y)``.

    Raises
    ------
    ValueError
        (On indexing) if the selection mixes states of two different private groups.

    Examples
    --------
    >>> import numpy as np
    >>> from nowcastbox.statespace import StateSpace, structured_smoother
    >>> T = np.diag([0.7, 0.5, 0.3])
    >>> Z = np.array([[1.0, 1.0, 0.0], [0.5, 0.0, 1.0]])
    >>> ssm = StateSpace(T, Z, np.eye(3), [0.1, 0.1])
    >>> res = structured_smoother(ssm, np.ones((4, 2)))
    >>> res.smoothed_state_cov[np.ix_([0, 1], [0, 1], [0, 1])].shape
    (2, 2, 2)
    """

    def __init__(self, result: StructuredSmootherResult, lag: int) -> None:
        self._result = result
        self._lag = int(lag)

    @property
    def shape(self) -> tuple[int, int, int]:
        """``(n_periods, n_states, n_states)``."""
        n, m = self._result.smoothed_state.shape
        return (n, m, m)

    @property
    def ndim(self) -> int:
        """Always 3."""
        return 3

    def __len__(self) -> int:
        return self.shape[0]

    @staticmethod
    def _normalize(item: Any, size: int) -> tuple[IntArray, bool]:
        if isinstance(item, (int, np.integer)):
            k = int(item)
            if not -size <= k < size:
                msg = f"index {k} out of range for size {size}"
                raise IndexError(msg)
            return np.array([k % size], dtype=np.int64), True
        if isinstance(item, slice):
            return np.arange(size, dtype=np.int64)[item], False
        arr = np.asarray(item)
        if arr.dtype == bool:
            return np.flatnonzero(arr.ravel()).astype(np.int64), False
        arr = arr.ravel().astype(np.int64)
        if arr.size and (arr.min() < -size or arr.max() >= size):
            msg = f"index out of range for size {size}"
            raise IndexError(msg)
        return arr % size if arr.size else arr, False

    def __getitem__(self, key: Any) -> FloatArray:
        n, m, _ = self.shape
        if not isinstance(key, tuple):
            key = (key,)
        if len(key) > 3:
            msg = "too many indices for a 3-D covariance view"
            raise IndexError(msg)
        key = key + (slice(None),) * (3 - len(key))
        periods, sq0 = self._normalize(key[0], n)
        rows, sq1 = self._normalize(key[1], m)
        cols, sq2 = self._normalize(key[2], m)
        out = self._result.covariance(periods, rows, cols, self._lag)
        squeeze = tuple(ax for ax, flag in enumerate((sq0, sq1, sq2)) if flag)
        return np.squeeze(out, axis=squeeze) if squeeze else out

    def to_array(self, fill_value: float = np.nan) -> FloatArray:
        """Dense ``(n_periods, n_states, n_states)`` array.

        Parameters
        ----------
        fill_value : float, default nan
            Value of the entries that are not computed (between two private groups).

        Returns
        -------
        numpy.ndarray
            Covariances (``O(n m^2)`` memory).

        Examples
        --------
        >>> import numpy as np
        >>> from nowcastbox.statespace import StateSpace, structured_smoother
        >>> ssm = StateSpace(np.diag([0.7, 0.5, 0.3]), [[1, 1, 0], [1, 0, 1]], np.eye(3), [1, 1])
        >>> full = structured_smoother(ssm, np.ones((3, 2))).smoothed_state_cov.to_array()
        >>> bool(np.isnan(full[0, 1, 2])), bool(np.isfinite(full[0, 0, 1]))
        (True, True)
        """
        return self._result.dense_covariance(self._lag, float(fill_value))


@dataclass(frozen=True, eq=False)
class StructuredSmootherResult:
    """Output of :func:`structured_smoother`.

    Attributes
    ----------
    model : StateSpace
        The model.
    structure : StateStructure
        Block structure used.
    observations : numpy.ndarray, shape (n_periods, n_obs)
        Observations (``NaN`` = missing).
    smoothed_state : numpy.ndarray, shape (n_periods, n_states)
        ``E[alpha_t | Y_n]``.
    loglikelihood : float
        Gaussian log-likelihood ``log p(y_1, ..., y_n)``.
    common_cov, common_autocov : numpy.ndarray, shape (n_periods, m_c, m_c)
        ``Var(alpha^c_t | Y)`` and ``Cov(alpha^c_{t+1}, alpha^c_t | Y)`` of the common
        states (``structure.common_states``).

    Examples
    --------
    >>> import numpy as np
    >>> from nowcastbox.statespace import StateSpace, kalman_smoother, structured_smoother
    >>> T = np.diag([0.7, 0.5, 0.3])
    >>> Z = np.array([[1.0, 1.0, 0.0], [0.5, 0.0, 1.0]])
    >>> ssm = StateSpace(T, Z, np.eye(3), [0.1, 0.2])
    >>> y = np.array([[1.0, np.nan], [0.3, -0.2], [np.nan, 0.5]])
    >>> fast, ref = structured_smoother(ssm, y), kalman_smoother(ssm, y)
    >>> bool(np.allclose(fast.smoothed_state, ref.smoothed_state))
    True
    >>> bool(abs(fast.loglikelihood - ref.loglikelihood) < 1e-10)
    True
    """

    model: StateSpace
    structure: StateStructure = field(repr=False)
    observations: FloatArray = field(repr=False)
    smoothed_state: FloatArray = field(repr=False)
    loglikelihood: float
    common_cov: FloatArray = field(repr=False)
    common_autocov: FloatArray = field(repr=False)
    _groups: tuple[_GroupMoments | None, ...] = field(repr=False)

    # ------------------------------------------------------------------ basics
    @property
    def n_periods(self) -> int:
        """Number of periods."""
        return int(self.smoothed_state.shape[0])

    @property
    def smoothed_state_cov(self) -> StructuredCovariance:
        """``Var(alpha_t | Y)`` as an orthogonally indexable view (see StructuredCovariance)."""
        return StructuredCovariance(self, 0)

    @property
    def smoothed_state_autocov(self) -> StructuredCovariance:
        """``Cov(alpha_{t+1}, alpha_t | Y)`` (row ``n-1`` involves ``alpha_{n+1}``)."""
        return StructuredCovariance(self, 1)

    def group_moments(self, group: int) -> tuple[IntArray, FloatArray, FloatArray]:
        """States, covariances and lag-one covariances of one private group.

        Parameters
        ----------
        group : int
            Group index (``1 .. structure.n_private_groups``).

        Returns
        -------
        states : numpy.ndarray of int
            States of the group.
        cov : numpy.ndarray, shape (n_periods, m_g, m_g)
            ``Var(alpha^g_t | Y)``.
        autocov : numpy.ndarray, shape (n_periods, m_g, m_g)
            ``Cov(alpha^g_{t+1}, alpha^g_t | Y)``.

        Raises
        ------
        IndexError
            If ``group`` is not a private group.

        Examples
        --------
        >>> import numpy as np
        >>> from nowcastbox.statespace import StateSpace, structured_smoother
        >>> ssm = StateSpace(np.diag([0.7, 0.5, 0.3]), [[1, 1, 0], [1, 0, 1]], np.eye(3), [1, 1])
        >>> states, cov, _ = structured_smoother(ssm, np.ones((3, 2))).group_moments(1)
        >>> states.tolist(), cov.shape
        ([1], (3, 1, 1))
        """
        if not 1 <= group < len(self._groups):
            msg = f"group must be in [1, {len(self._groups) - 1}], got {group}"
            raise IndexError(msg)
        gm = self._groups[group]
        assert gm is not None  # noqa: S101 - groups >= 1 are always filled
        return gm.states, gm.cov, gm.autocov

    def smoothed_signal(self) -> FloatArray:
        """Smoothed observations ``E[Z_t alpha_t + d_t | Y]`` for every series.

        Returns
        -------
        numpy.ndarray, shape (n_periods, n_obs)
            Smoothed signal.

        Examples
        --------
        >>> import numpy as np
        >>> from nowcastbox.statespace import StateSpace, structured_smoother
        >>> ssm = StateSpace(np.diag([0.7, 0.5, 0.3]), [[1, 1, 0], [1, 0, 1]], np.eye(3), [1, 1])
        >>> structured_smoother(ssm, np.ones((3, 2))).smoothed_signal().shape
        (3, 2)
        """
        n = self.n_periods
        zs = self.model.designs(n)
        index = self.model.period_index(n)
        return (
            np.einsum("tij,tj->ti", zs, self.smoothed_state) + self.model.obs_intercept_store[index]
        )

    def smoothed_signal_variance(self) -> FloatArray:
        """Variance of the smoothed signal ``Var(Z_t alpha_t | Y)`` of every series.

        Returns
        -------
        numpy.ndarray, shape (n_periods, n_obs)
            ``z_{t,i}' V_t z_{t,i}`` (a series loads only on common states and its own
            private group, so every term is available).

        Examples
        --------
        >>> import numpy as np
        >>> from nowcastbox.statespace import StateSpace, kalman_smoother, structured_smoother
        >>> ssm = StateSpace(np.diag([0.7, 0.5, 0.3]), [[1, 1, 0], [1, 0, 1]], np.eye(3), [1, 1])
        >>> y = np.ones((3, 2))
        >>> ref = kalman_smoother(ssm, y).smoothed_signal_cov(1).diagonal()
        >>> bool(np.allclose(structured_smoother(ssm, y).smoothed_signal_variance()[1], ref))
        True
        """
        n = self.n_periods
        index = self.model.period_index(n)
        out = np.zeros((n, self.model.n_obs))
        for i in range(self.model.n_obs):
            rows = self.model.design_store[:, i, :]
            states = np.flatnonzero(np.any(rows != 0.0, axis=0))
            if states.size == 0:
                continue
            cov = self.covariance(np.arange(n), states, states, 0)
            z = rows[index][:, states]
            out[:, i] = np.einsum("ti,tij,tj->t", z, cov, z)
        return out

    # ------------------------------------------------------------------ internals
    def _position(self, states: IntArray) -> tuple[IntArray, IntArray]:
        """Group and position inside the group's stored arrays of every state."""
        st = self.structure
        groups = st.state_group[states]
        pos = np.empty(states.size, dtype=np.int64)
        for g in np.unique(groups):
            sel = groups == g
            pos[sel] = np.searchsorted(st.groups[int(g)].states, states[sel])
        return groups, pos

    def _pair(
        self, gr: int, gc: int, periods: IntArray, pr: IntArray, pc: IntArray, lag: int
    ) -> FloatArray:
        if gr == 0 and gc == 0:
            src = self.common_cov if lag == 0 else self.common_autocov
            return src[np.ix_(periods, pr, pc)]
        if gr != 0 and gc != 0:
            if gr != gc:
                msg = (
                    "covariances between two different private groups are not computed by "
                    "the structured smoother"
                )
                raise ValueError(msg)
            gm = self._groups[gr]
            assert gm is not None  # noqa: S101
            src = gm.cov if lag == 0 else gm.autocov
            return src[np.ix_(periods, pr, pc)]
        if gr == 0:  # common rows, private columns
            gm = self._groups[gc]
            assert gm is not None  # noqa: S101
            src = gm.cross_cov if lag == 0 else gm.cross_autocov_cg
            return src[np.ix_(periods, pr, pc)]
        gm = self._groups[gr]  # private rows, common columns
        assert gm is not None  # noqa: S101
        if lag == 0:
            return np.swapaxes(gm.cross_cov[np.ix_(periods, pc, pr)], 1, 2)
        return gm.cross_autocov_gc[np.ix_(periods, pr, pc)]

    def covariance(
        self, periods: ArrayLike, rows: ArrayLike, cols: ArrayLike, lag: int = 0
    ) -> FloatArray:
        """Smoothed (lag-one) covariances on an orthogonal selection of periods and states.

        Parameters
        ----------
        periods : array_like of int
            0-based periods.
        rows, cols : array_like of int
            State indices.
        lag : {0, 1}, default 0
            ``0``: ``Var(alpha_t | Y)``; ``1``: ``Cov(alpha_{t+1}, alpha_t | Y)``.

        Returns
        -------
        numpy.ndarray, shape (len(periods), len(rows), len(cols))
            Requested entries.

        Raises
        ------
        ValueError
            If the selection pairs states of two different private groups, or ``lag`` is
            not 0 or 1.

        Examples
        --------
        >>> import numpy as np
        >>> from nowcastbox.statespace import StateSpace, structured_smoother
        >>> ssm = StateSpace(np.diag([0.7, 0.5, 0.3]), [[1, 1, 0], [1, 0, 1]], np.eye(3), [1, 1])
        >>> structured_smoother(ssm, np.ones((3, 2))).covariance([0, 2], [0, 1], [0, 1]).shape
        (2, 2, 2)
        """
        if lag not in (0, 1):
            msg = f"lag must be 0 or 1, got {lag}"
            raise ValueError(msg)
        periods = np.asarray(periods, dtype=np.int64).ravel()
        rows = np.asarray(rows, dtype=np.int64).ravel()
        cols = np.asarray(cols, dtype=np.int64).ravel()
        out = np.empty((periods.size, rows.size, cols.size))
        g_rows, p_rows = self._position(rows)
        g_cols, p_cols = self._position(cols)
        for gr in np.unique(g_rows):
            ri = np.flatnonzero(g_rows == gr)
            for gc in np.unique(g_cols):
                ci = np.flatnonzero(g_cols == gc)
                block = self._pair(int(gr), int(gc), periods, p_rows[ri], p_cols[ci], lag)
                out[np.ix_(np.arange(periods.size), ri, ci)] = block
        return out

    def moment_sums(self) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Smoothed second-moment sums of the EM E-step.

        Returns
        -------
        s11, s00, s10 : numpy.ndarray, shape (n_states, n_states)
            ``sum_{t=2}^n E[alpha_t alpha_t' | Y]``, ``sum_{t=2}^n E[alpha_{t-1}
            alpha_{t-1}' | Y]`` and ``sum_{t=2}^n E[alpha_t alpha_{t-1}' | Y]``; entries
            between two different private groups are ``NaN`` (not computed).

        Examples
        --------
        >>> import numpy as np
        >>> from nowcastbox.statespace import StateSpace, structured_smoother
        >>> ssm = StateSpace(np.diag([0.7, 0.5, 0.3]), [[1, 1, 0], [1, 0, 1]], np.eye(3), [1, 1])
        >>> s11, s00, s10 = structured_smoother(ssm, np.ones((4, 2))).moment_sums()
        >>> bool(np.isnan(s11[1, 2])), bool(np.isfinite(s11[0, 2]))
        (True, True)
        """
        a = self.smoothed_state
        m = a.shape[1]
        cov = np.full((3, m, m), np.nan)
        cs = self.structure.groups[0].states
        blocks: list[tuple[IntArray, IntArray, FloatArray, FloatArray, FloatArray]] = []
        if cs.size:
            blocks.append(
                (cs, cs, self.common_cov[1:].sum(0), self.common_cov[:-1].sum(0),
                 self.common_autocov[:-1].sum(0))
            )  # fmt: skip
        for gm in self._groups[1:]:
            assert gm is not None  # noqa: S101
            gs = gm.states
            blocks.append((gs, gs, gm.cov[1:].sum(0), gm.cov[:-1].sum(0), gm.autocov[:-1].sum(0)))
            if cs.size:
                v1 = gm.cross_cov[1:].sum(0)
                v0 = gm.cross_cov[:-1].sum(0)
                blocks.append((cs, gs, v1, v0, gm.cross_autocov_cg[:-1].sum(0)))
                blocks.append((gs, cs, v1.T, v0.T, gm.cross_autocov_gc[:-1].sum(0)))
        for rows, cols, v1, v0, c10 in blocks:
            ix = np.ix_(rows, cols)
            cov[0][ix] = v1
            cov[1][ix] = v0
            cov[2][ix] = c10
        s11 = a[1:].T @ a[1:] + cov[0]
        s00 = a[:-1].T @ a[:-1] + cov[1]
        s10 = a[1:].T @ a[:-1] + cov[2]
        return s11, s00, s10

    def dense_covariance(self, lag: int = 0, fill_value: float = np.nan) -> FloatArray:
        """All smoothed (lag-one) covariances as a dense ``(n, m, m)`` array.

        Parameters
        ----------
        lag : {0, 1}, default 0
            ``0``: ``Var(alpha_t | Y)``; ``1``: ``Cov(alpha_{t+1}, alpha_t | Y)``.
        fill_value : float, default nan
            Value of the entries between two different private groups (not computed).

        Returns
        -------
        numpy.ndarray, shape (n_periods, n_states, n_states)
            Covariances.

        Examples
        --------
        >>> import numpy as np
        >>> from nowcastbox.statespace import StateSpace, structured_smoother
        >>> ssm = StateSpace(np.diag([0.7, 0.5, 0.3]), [[1, 1, 0], [1, 0, 1]], np.eye(3), [1, 1])
        >>> structured_smoother(ssm, np.ones((3, 2))).dense_covariance(1).shape
        (3, 3, 3)
        """
        n, m = self.smoothed_state.shape
        out = np.full((n, m, m), float(fill_value))
        periods = np.arange(n)
        groups = self.structure.groups
        common = groups[0].states
        for g, grp in enumerate(groups):
            if grp.states.size == 0:
                continue
            out[np.ix_(periods, grp.states, grp.states)] = self.covariance(
                periods, grp.states, grp.states, lag
            )
            if g > 0 and common.size:
                out[np.ix_(periods, grp.states, common)] = self.covariance(
                    periods, grp.states, common, lag
                )
                out[np.ix_(periods, common, grp.states)] = self.covariance(
                    periods, common, grp.states, lag
                )
        return out


# ====================================================================== algorithm
def _resolve_structure(model: StateSpace, structure: StateStructure | None) -> StateStructure:
    if structure is None:
        return detect_structure(model)
    if structure.state_group.size != model.n_states:
        msg = "structure does not match the number of states of the model"
        raise ValueError(msg)
    return structure


def _common_system(
    layout: _GroupLayout,
    templates: list[_Template],
    padding: IntArray,
    obs: list[_SeriesObs | None],
    series_group: IntArray,
) -> tuple[FloatArray, FloatArray]:
    size = layout.size
    schur = np.zeros((size, size))
    rhs = np.zeros(size)
    for tpl in templates:
        _bk.dense_add_template(
            schur, rhs, tpl.mat, tpl.lin, tpl.offsets, layout.width, tpl.p_start, tpl.p_stop
        )
    if padding.size:
        schur[padding, padding] += 1.0
    for i, ob in enumerate(obs):
        if ob is None or series_group[i] > 0 or ob.f_idx.shape[1] == 0:
            continue
        _bk.dense_add_observations(schur, rhs, ob.f_idx, ob.f_val, ob.y, ob.h)
    return schur, rhs


def _group_band(
    layout: _GroupLayout, templates: list[_Template], padding: IntArray
) -> tuple[FloatArray, FloatArray]:
    ab = np.zeros((layout.bandwidth + 1, layout.size))
    rhs = np.zeros(layout.size)
    for tpl in templates:
        _bk.band_add_template(
            ab, rhs, tpl.mat, tpl.lin, tpl.offsets, layout.width, tpl.p_start, tpl.p_stop
        )
    if padding.size:
        ab[0, padding] += 1.0
    return ab, rhs


@dataclass
class _Workspace:
    """Reusable buffers of the group kernels (allocated once per smoother call)."""

    y: FloatArray
    cross: FloatArray
    phi: FloatArray
    psi: FloatArray
    row: FloatArray

    @classmethod
    def allocate(cls, max_size: int, max_obs: int, n_f: int) -> _Workspace:
        k = max(max_obs, 1)
        return cls(
            y=np.empty((max_size, k)),
            cross=np.empty((max_size, n_f)),
            phi=np.empty((k, n_f)),
            psi=np.empty((max_size, k)),
            row=np.empty(k),
        )


_EMPTY_IDX = np.zeros((0, 0), dtype=np.int64)
_EMPTY_VAL = np.zeros((0, 0))
_EMPTY_VEC = np.zeros(0)


def _group_obs(
    ob: _SeriesObs | None,
) -> tuple[IntArray, FloatArray, IntArray, FloatArray, FloatArray, FloatArray]:
    if ob is None:
        return _EMPTY_IDX, _EMPTY_VAL, _EMPTY_IDX, _EMPTY_VAL, _EMPTY_VEC, _EMPTY_VEC
    return ob.g_idx, ob.g_val, ob.f_idx, ob.f_val, ob.y, ob.h


def structured_smoother(
    model: StateSpace,
    observations: ArrayLike,
    *,
    structure: StateStructure | None = None,
) -> StructuredSmootherResult:
    """Exact smoother exploiting the companion/idiosyncratic block structure.

    Parameters
    ----------
    model : StateSpace
        Model with the structure described in :func:`~nowcastbox.statespace.detect_structure`
        (diagonal positive ``H``, companion blocks, idiosyncratic blocks loaded by one
        series). Time-varying observation equations are supported.
    observations : array_like, shape (n_periods, n_obs)
        Observations with ``NaN`` for missing values.
    structure : StateStructure, optional
        Pre-computed structure (default: detected from ``model``).

    Returns
    -------
    StructuredSmootherResult
        Smoothed states, log-likelihood and the (lag-one) covariances inside the common
        block, inside each private group and between them.

    Raises
    ------
    ValueError
        If the model does not have the required structure.
    numpy.linalg.LinAlgError
        If the posterior precision is not positive definite (numerically singular model).
    nowcastbox.core.exceptions.NowcastDataError
        On malformed observations.

    Examples
    --------
    One factor, two series with AR(1) idiosyncratic components:

    >>> import numpy as np
    >>> from nowcastbox.statespace import StateSpace, kalman_smoother, structured_smoother
    >>> T = np.diag([0.8, 0.4, -0.2])
    >>> Z = np.array([[1.0, 1.0, 0.0], [0.7, 0.0, 1.0]])
    >>> ssm = StateSpace(T, Z, np.diag([1.0, 0.3, 0.3]), [1e-4, 1e-4])
    >>> y = np.random.default_rng(0).standard_normal((50, 2))
    >>> y[-3:, 0] = np.nan
    >>> fast, ref = structured_smoother(ssm, y), kalman_smoother(ssm, y)
    >>> bool(np.allclose(fast.smoothed_state, ref.smoothed_state, atol=1e-8))
    True
    >>> bool(np.allclose(fast.smoothed_state_cov[:, 0, 0], ref.smoothed_state_cov[:, 0, 0]))
    True
    """
    y = prepare_observations(model, observations)
    n = y.shape[0]
    st = _resolve_structure(model, structure)
    index = model.period_index(n)
    layouts = _layouts(st, n)
    templates, padding, prior_logdet = _templates(model, st, layouts, n)
    obs = _series_observations(model, st, layouts, y, index)

    common = layouts[0]
    n_f = common.size
    schur, rhs_f = _common_system(common, templates[0], padding[0], obs, st.series_group)
    owners = {grp_id: grp.owner for grp_id, grp in enumerate(st.groups)}
    max_size = max((lay.size for lay in layouts[1:]), default=0)
    max_obs = max((ob.y.size for ob in obs if ob is not None), default=0)
    ws = _Workspace.allocate(max_size, max_obs, n_f)

    logdet_post = 0.0
    factors: list[tuple[FloatArray, FloatArray] | None] = [None]
    for g in range(1, len(layouts)):
        owner = owners[g]
        ob = obs[owner] if owner >= 0 else None
        ab, rhs_g = _group_band(layouts[g], templates[g], padding[g])
        x_g = np.empty(layouts[g].size)
        g_idx, g_val, f_idx, f_val, yv, hv = _group_obs(ob)
        ld = _bk.group_eliminate(
            ab, rhs_g, g_idx, g_val, f_idx, f_val, yv, hv, schur, rhs_f, ws.y, ws.row, x_g
        )
        if not np.isfinite(ld):
            msg = f"the posterior precision of state group {g} is not positive definite"
            raise np.linalg.LinAlgError(msg)
        logdet_post += ld
        factors.append((ab, x_g))

    if n_f:
        try:
            chol = scipy.linalg.cho_factor(schur, lower=True, check_finite=False)
        except np.linalg.LinAlgError as err:
            msg = "the posterior precision of the common states is not positive definite"
            raise np.linalg.LinAlgError(msg) from err
        logdet_post += 2.0 * float(np.sum(np.log(np.diag(chol[0]))))
        mu_f = scipy.linalg.cho_solve(chol, rhs_f, check_finite=False)
        sigma_ff = scipy.linalg.cho_solve(chol, np.eye(n_f), check_finite=False)
        sigma_ff = np.ascontiguousarray(0.5 * (sigma_ff + sigma_ff.T))
    else:
        mu_f = np.zeros(0)
        sigma_ff = np.zeros((0, 0))

    m = model.n_states
    periods = np.arange(n)
    mean = np.zeros((n, m))
    cs = common.states
    idx_f = _var(st, common, periods[:, None], cs[None, :])
    idx_f1 = _var(st, common, periods[:, None] + 1, cs[None, :])
    if cs.size:
        mean[:, cs] = mu_f[idx_f]
    common_cov = sigma_ff[idx_f[:, :, None], idx_f[:, None, :]]
    common_autocov = sigma_ff[idx_f1[:, :, None], idx_f[:, None, :]]
    prior_q = _prior_quadratic(templates[0], mu_f, common.width) if n_f else 0.0

    groups: list[_GroupMoments | None] = [None]
    for g in range(1, len(layouts)):
        lay = layouts[g]
        owner = owners[g]
        ob = obs[owner] if owner >= 0 else None
        fac = factors[g]
        assert fac is not None  # noqa: S101
        ab, x_g = fac
        g_idx, g_val, f_idx, f_val, _yv, hv = _group_obs(ob)
        band_cov = np.empty_like(ab)
        mu_g = np.empty(lay.size)
        cross = ws.cross[: lay.size]
        _bk.group_backsubstitute(
            ab, g_idx, g_val, f_idx, f_val, hv, mu_f, sigma_ff, x_g,
            ws.y, cross, ws.phi, ws.psi, ws.row, band_cov, mu_g,
        )  # fmt: skip
        gs = lay.states
        idx_g = _var(st, lay, periods[:, None], gs[None, :])
        idx_g1 = _var(st, lay, periods[:, None] + 1, gs[None, :])
        mean[:, gs] = mu_g[idx_g]
        groups.append(
            _GroupMoments(
                states=gs,
                cov=_band_lookup(band_cov, idx_g[:, :, None], idx_g[:, None, :]),
                autocov=_band_lookup(band_cov, idx_g1[:, :, None], idx_g[:, None, :]),
                cross_cov=cross[idx_g[:, None, :], idx_f[:, :, None]],
                cross_autocov_cg=cross[idx_g[:, None, :], idx_f1[:, :, None]],
                cross_autocov_gc=cross[idx_g1[:, :, None], idx_f[:, None, :]],
            )
        )
        prior_q += _prior_quadratic(templates[g], mu_g, lay.width)

    loglik = _loglikelihood(model, y, index, mean, obs, prior_q, prior_logdet, logdet_post)
    return StructuredSmootherResult(
        model=model,
        structure=st,
        observations=y,
        smoothed_state=mean,
        loglikelihood=loglik,
        common_cov=np.ascontiguousarray(common_cov),
        common_autocov=np.ascontiguousarray(common_autocov),
        _groups=tuple(groups),
    )


def _band_lookup(band: FloatArray, rows: IntArray, cols: IntArray) -> FloatArray:
    lo = np.maximum(rows, cols)
    hi = np.minimum(rows, cols)
    return band[lo - hi, hi]


def _loglikelihood(
    model: StateSpace,
    y: FloatArray,
    index: IntArray,
    mean: FloatArray,
    obs: list[_SeriesObs | None],
    prior_q: float,
    prior_logdet: float,
    logdet_post: float,
) -> float:
    """``log p(y) = log p(y | x) + log p(x) - log p(x | y)`` at the posterior mean."""
    fitted = np.einsum("tij,tj->ti", model.designs(y.shape[0]), mean)
    resid = y - fitted - model.obs_intercept_store[index]
    h = model.obs_cov_diagonal_store[index]
    observed = ~np.isnan(y)
    n_obs = int(observed.sum())
    quad = float(np.sum(resid[observed] ** 2 / h[observed]))
    log_h = float(np.sum(np.log(h[observed])))
    _ = obs
    return -0.5 * (n_obs * _LOG_2PI + log_h + quad + prior_q + prior_logdet + logdet_post)


# ====================================================================== EM moments
@dataclass(frozen=True, eq=False)
class SmoothedMoments:
    """Smoother output and the smoothed second-moment sums of the EM E-step.

    Attribute names match ``nowcastbox.models._em_steps.SufficientStatistics`` so the
    object can be used by the M-step unchanged.

    Attributes
    ----------
    smoother : SmootherResult or StructuredSmootherResult
        Smoother output (``smoothed_state``, ``smoothed_state_cov``, ...).
    s11 : numpy.ndarray, shape (m, m)
        ``sum_{t=2}^n E[alpha_t alpha_t' | Y]``.
    s00 : numpy.ndarray, shape (m, m)
        ``sum_{t=2}^n E[alpha_{t-1} alpha_{t-1}' | Y]``.
    s10 : numpy.ndarray, shape (m, m)
        ``sum_{t=2}^n E[alpha_t alpha_{t-1}' | Y]``.
    n_pairs : int
        ``n - 1``.
    method : str
        ``"structured"``, ``"univariate"`` or ``"multivariate"``.

    Notes
    -----
    With the structured smoother, entries between two different private groups are
    ``NaN`` (not computed; never used by the M-step).

    Examples
    --------
    >>> import numpy as np
    >>> from nowcastbox.statespace import StateSpace, smoothed_moments
    >>> ssm = StateSpace([[0.5]], [[1.0]], [[1.0]], [1.0])
    >>> mom = smoothed_moments(ssm, np.array([[1.0], [np.nan], [0.2]]))
    >>> mom.n_pairs, mom.s11.shape
    (2, (1, 1))
    """

    smoother: SmootherResult | StructuredSmootherResult = field(repr=False)
    s11: FloatArray = field(repr=False)
    s00: FloatArray = field(repr=False)
    s10: FloatArray = field(repr=False)
    n_pairs: int
    method: str

    @property
    def loglikelihood(self) -> float:
        """Log-likelihood of the data under the model."""
        return float(self.smoother.loglikelihood)


def _dense_moment_sums(sm: SmootherResult) -> tuple[FloatArray, FloatArray, FloatArray]:
    a, p, c = sm.smoothed_state, sm.smoothed_state_cov, sm.smoothed_state_autocov
    s11 = a[1:].T @ a[1:] + p[1:].sum(axis=0)
    s00 = a[:-1].T @ a[:-1] + p[:-1].sum(axis=0)
    s10 = a[1:].T @ a[:-1] + c[:-1].sum(axis=0)
    return s11, s00, s10


_DENSE_PERIOD_OVERHEAD = 1.0e5
"""Fixed cost of one period of the dense filter/smoother (call overhead)."""

_STRUCTURED_SCALE = 3.4
"""Relative cost of one flop of the structured smoother versus the dense one."""

_STRUCTURED_GROUP_OVERHEAD = 3.9e4
"""Fixed cost of one group-period of the structured smoother (per-group band setup)."""

_STRUCTURED_CALL_OVERHEAD = 1.0e7
"""Fixed cost of one structured smoother call (structure analysis, factor block)."""


def _dense_cost(model: StateSpace, y: FloatArray) -> float:
    """Estimated cost of the dense smoother (flops plus per-period overhead).

    The constants of :func:`_dense_cost` and :func:`_structured_cost` (units: one dense
    flop, about 7e-11 s) were calibrated on E-step timings of the models of
    the software paper (``nowcastbox-papers/01_software_jss``; one factor, AR(1) idiosyncratic components, N = 10-200,
    T = 120-600, one BLAS thread): the crossover lies between N = 35 (dense faster) and
    N = 50 (structured faster) at T = 300, and tiny models use the dense smoother.
    """
    m = model.n_states
    n_obs = float(np.sum(~np.isnan(y))) / max(y.shape[0], 1)
    return y.shape[0] * (7.0 * m**3 + 4.0 * n_obs * m**2 + _DENSE_PERIOD_OVERHEAD)


def _structured_cost(structure: StateStructure, y: FloatArray) -> float:
    """Estimated cost of the structured smoother (see :func:`_dense_cost`)."""
    n = y.shape[0]
    lays = _layouts(structure, n)
    n_f = lays[0].size
    cost = 1.5 * float(n_f) ** 3
    for g, lay in enumerate(lays[1:], start=1):
        owner = structure.groups[g].owner
        n_obs = float(np.sum(~np.isnan(y[:, owner]))) if owner >= 0 else 0.0
        cost += lay.size * (n_obs + n_f) * (lay.bandwidth + 2) * 4.0
        cost += lay.size * (2 * lay.bandwidth + 1) * n_obs
    overhead = _STRUCTURED_GROUP_OVERHEAD * len(lays) * n + _STRUCTURED_CALL_OVERHEAD
    return _STRUCTURED_SCALE * cost + overhead


def smoothed_moments(
    model: StateSpace,
    observations: ArrayLike,
    *,
    method: MomentsMethod = "auto",
) -> SmoothedMoments:
    """E-step of the EM algorithm: smoother plus smoothed second-moment sums.

    Drop-in fast path for the E-step of :class:`~nowcastbox.models.MixedFreqDFM`
    (Shumway & Stoffer, 1982; Bańbura & Modugno, 2014): returns the same quantities as
    ``nowcastbox.models._em_steps.e_step`` but, with ``method="auto"``, uses the exact
    :func:`structured_smoother` whenever the model has the required structure and it is
    estimated to be cheaper than the dense Kalman smoother.

    Parameters
    ----------
    model : StateSpace
        Model at the current parameters.
    observations : array_like, shape (n_periods, n_obs)
        Observations (``NaN`` = missing), at least two periods.
    method : {"auto", "structured", "univariate", "multivariate"}, default "auto"
        ``"structured"`` forces the structured smoother (``ValueError`` if not
        applicable); ``"univariate"``/``"multivariate"`` force the dense Kalman smoother
        with that filter.

    Returns
    -------
    SmoothedMoments
        Smoother output, ``s11``, ``s00``, ``s10`` and ``n_pairs``.

    Raises
    ------
    ValueError
        If there are fewer than two periods, ``method`` is unknown or the structured
        smoother is forced on a model without the structure.

    Examples
    --------
    >>> import numpy as np
    >>> from nowcastbox.statespace import StateSpace, smoothed_moments
    >>> T = np.diag([0.7, 0.5, 0.3])
    >>> Z = np.array([[1.0, 1.0, 0.0], [0.5, 0.0, 1.0]])
    >>> ssm = StateSpace(T, Z, np.eye(3), [1e-4, 1e-4])
    >>> y = np.random.default_rng(1).standard_normal((30, 2))
    >>> fast = smoothed_moments(ssm, y, method="structured")
    >>> ref = smoothed_moments(ssm, y, method="univariate")
    >>> (
    ...     bool(np.allclose(fast.s10[0, :], ref.s10[0, :])),
    ...     bool(abs(fast.loglikelihood - ref.loglikelihood) < 1e-8),
    ... )
    (True, True)
    """
    if method not in ("auto", "structured", "univariate", "multivariate"):
        msg = f"method must be 'auto', 'structured', 'univariate' or 'multivariate', got {method!r}"
        raise ValueError(msg)
    y = prepare_observations(model, observations)
    if y.shape[0] < 2:
        msg = "the E-step needs at least two periods"
        raise ValueError(msg)
    chosen: str = method
    structure: StateStructure | None = None
    if method in ("auto", "structured"):
        try:
            structure = detect_structure(model)
        except ValueError:
            if method == "structured":
                raise
            structure = None
        if method == "auto":
            use = structure is not None and _structured_cost(structure, y) < _dense_cost(model, y)
            chosen = (
                "structured"
                if use
                else "univariate"
                if model.obs_cov_is_diagonal
                else "multivariate"
            )
    if chosen == "structured":
        res = structured_smoother(model, y, structure=structure)
        s11, s00, s10 = res.moment_sums()
        logger.debug("E-step: structured smoother (%d groups)", len(res.structure.groups))
        return SmoothedMoments(res, s11, s00, s10, y.shape[0] - 1, "structured")
    sm = kalman_smoother(model, y, method=chosen)  # type: ignore[arg-type]
    s11, s00, s10 = _dense_moment_sums(sm)
    return SmoothedMoments(sm, s11, s00, s10, y.shape[0] - 1, chosen)
