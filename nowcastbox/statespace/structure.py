"""Detection of the block structure of dynamic-factor state-space models.

Many state-space forms used for nowcasting (Bańbura & Modugno, 2014; Mariano &
Murasawa, 2003) have a state vector that is a direct sum of independent *companion
blocks*: a VAR(s) process ``x_{t+1} = A_1 x_t + ... + A_s x_{t-s+1} + c + u_t`` stored
as ``(x_t, x_{t-1}, ..., x_{t-s+1})``, where the lag states are exact copies (rows of
``T`` that are unit shifts, no disturbance). Factor blocks, idiosyncratic AR(1)
components and the lag chains required by the Mariano-Murasawa aggregation are all of
this form. When, in addition, ``H`` is diagonal and every idiosyncratic block is loaded
by a single series, the states of different series are conditionally independent given
the common blocks, which :func:`~nowcastbox.statespace.structured_smoother` exploits.

:func:`detect_structure` recovers that decomposition from the system matrices alone:

* *chains*: every state is either a **leader** (non-shift row of ``T`` or non-zero
  disturbance) or a lag copy of another state; following the copies gives each state
  its chain and lag;
* *blocks*: connected components of the non-zero patterns of ``T``, ``R Q R'`` and
  ``P0`` (entries of ``P0`` smaller than ``1e-14 sqrt(P0_ii P0_jj)`` count as zeros, so
  rounding noise of a Lyapunov solver does not merge independent blocks);
* *groups*: blocks loaded by two or more series form the **common** group; the blocks
  loaded only by series ``i`` form the **private** group of ``i``; blocks loaded by
  nobody form groups without owner.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from nowcastbox.statespace.representation import StateSpace

__all__ = ["CompanionBlock", "StateGroup", "StateStructure", "detect_structure"]

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

_P0_ZERO_TOL = 1e-14
"""Entries of ``P0`` below this fraction of ``sqrt(P0_ii P0_jj)`` are rounding noise."""


@dataclass(frozen=True, eq=False)
class CompanionBlock:
    """One companion block of the state vector.

    Attributes
    ----------
    states : numpy.ndarray of int
        State indices of the block (model order).
    leaders : numpy.ndarray of int
        States carrying the current value ``x_t`` of each chain (lag 0).
    chain_lengths : numpy.ndarray of int
        Number of states (lags) of every chain, aligned with ``leaders``.
    disturbance_cov : numpy.ndarray
        ``(R Q R')`` restricted to the leaders (positive definite).
    initial_cov : numpy.ndarray
        ``P0`` restricted to ``states`` (positive definite).
    group : int
        Group the block belongs to (0 = common).

    Examples
    --------
    >>> from nowcastbox.statespace import StateSpace, detect_structure
    >>> ssm = StateSpace([[0.5]], [[1.0], [2.0]], [[1.0]], [1.0, 1.0])
    >>> detect_structure(ssm).blocks[0].chain_lengths.tolist()
    [1]
    """

    states: IntArray
    leaders: IntArray
    chain_lengths: IntArray
    disturbance_cov: FloatArray = field(repr=False)
    initial_cov: FloatArray = field(repr=False)
    group: int


@dataclass(frozen=True, eq=False)
class StateGroup:
    """A set of blocks sharing the same path layout in the structured smoother.

    Attributes
    ----------
    owner : int
        Series loading the group (``-1`` for the common group and for unobserved
        blocks).
    blocks : tuple of int
        Indices into :attr:`StateStructure.blocks`.
    states : numpy.ndarray of int
        States of the group (model order).
    width : int
        Number of chains ``w`` (variables per time point).
    n_lags : int
        Longest chain ``s``.

    Examples
    --------
    >>> from nowcastbox.statespace import StateSpace, detect_structure
    >>> ssm = StateSpace([[0.5]], [[1.0], [2.0]], [[1.0]], [1.0, 1.0])
    >>> detect_structure(ssm).groups[0].width
    1
    """

    owner: int
    blocks: tuple[int, ...]
    states: IntArray
    width: int
    n_lags: int


@dataclass(frozen=True, eq=False)
class StateStructure:
    """Companion-block decomposition of a :class:`StateSpace` (see :func:`detect_structure`).

    Attributes
    ----------
    blocks : tuple of CompanionBlock
        Independent companion blocks.
    groups : tuple of StateGroup
        ``groups[0]`` is the common group (possibly empty); the others are private or
        unobserved groups.
    state_group : numpy.ndarray of int, shape (n_states,)
        Group of every state.
    state_lag : numpy.ndarray of int, shape (n_states,)
        Lag of every state within its chain (0 for leaders).
    state_offset : numpy.ndarray of int, shape (n_states,)
        Position of the state's chain within its group (``0 .. width - 1``).
    series_group : numpy.ndarray of int, shape (n_obs,)
        Private group of every series (``-1`` when the series has none).

    Examples
    --------
    >>> import numpy as np
    >>> from nowcastbox.statespace import StateSpace, detect_structure
    >>> T = np.diag([0.7, 0.5, 0.3])  # factor + one AR(1) idiosyncratic per series
    >>> Z = np.array([[1.0, 1.0, 0.0], [0.5, 0.0, 1.0]])
    >>> s = detect_structure(StateSpace(T, Z, np.eye(3), [1e-4, 1e-4]))
    >>> s.n_common_states, s.series_group.tolist()
    (1, [1, 2])
    """

    blocks: tuple[CompanionBlock, ...]
    groups: tuple[StateGroup, ...]
    state_group: IntArray = field(repr=False)
    state_lag: IntArray = field(repr=False)
    state_offset: IntArray = field(repr=False)
    series_group: IntArray = field(repr=False)

    @property
    def common_states(self) -> IntArray:
        """States of the common group (model order)."""
        return self.groups[0].states

    @property
    def n_common_states(self) -> int:
        """Number of common states."""
        return int(self.groups[0].states.size)

    @property
    def n_private_groups(self) -> int:
        """Number of non-common groups."""
        return len(self.groups) - 1


def _chains(model: StateSpace) -> tuple[IntArray, IntArray, IntArray]:
    """Leader (chain head) and lag of every state; raises ``ValueError`` if inconsistent."""
    t_mat, rqr, c = model.T, model.state_disturbance_cov, model.c
    m = model.n_states
    parent = np.full(m, -1, dtype=np.int64)
    nz_rows = t_mat != 0.0
    noisy = np.any(rqr != 0.0, axis=1)
    for j in range(m):
        cols = np.flatnonzero(nz_rows[j])
        if (
            not noisy[j]
            and c[j] == 0.0
            and cols.size == 1
            and cols[0] != j
            and t_mat[j, cols[0]] == 1.0
        ):
            parent[j] = cols[0]
    head = np.full(m, -1, dtype=np.int64)
    lag = np.zeros(m, dtype=np.int64)
    for j in range(m):
        k, depth = j, 0
        while parent[k] >= 0:
            k = int(parent[k])
            depth += 1
            if depth > m:
                msg = "the shift rows of T form a cycle (no companion structure)"
                raise ValueError(msg)
        head[j], lag[j] = k, depth
    # every (chain, lag) must be unique: a state has at most one lag copy
    child_count = np.bincount(parent[parent >= 0], minlength=m)
    if np.any(child_count > 1):
        msg = "a state has more than one lag copy (no companion structure)"
        raise ValueError(msg)
    return head, lag, parent


def _blocks(model: StateSpace) -> list[IntArray]:
    from scipy.sparse.csgraph import connected_components

    p0 = model.P0
    scale = np.sqrt(np.outer(np.abs(np.diag(p0)), np.abs(np.diag(p0))))
    p0_pattern = np.abs(p0) > _P0_ZERO_TOL * scale
    pattern = (model.T != 0.0) | (model.state_disturbance_cov != 0.0) | p0_pattern
    pattern |= pattern.T
    n_comp, labels = connected_components(pattern, directed=False)
    return [np.flatnonzero(labels == k).astype(np.int64) for k in range(n_comp)]


def _positive_definite(matrix: FloatArray) -> bool:
    if matrix.shape == (1, 1):
        return bool(matrix[0, 0] > 0.0)
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        return False
    return True


def _check_observation_equation(model: StateSpace) -> None:
    if not model.obs_cov_is_diagonal:
        msg = "the structured smoother requires a diagonal observation covariance H"
        raise ValueError(msg)
    h = model.obs_cov_diagonal_store
    if np.any(h <= 0.0):
        msg = "the structured smoother requires strictly positive observation variances"
        raise ValueError(msg)


def _make_block(
    model: StateSpace, states: IntArray, head: IntArray, lag: IntArray
) -> tuple[IntArray, IntArray, FloatArray, FloatArray]:
    leaders = states[head[states] == states]
    lengths = np.array([int(np.sum(head[states] == ld)) for ld in leaders], dtype=np.int64)
    # the lags of each chain are 0..len-1: guaranteed by the unique-copy check in _chains
    rqr = model.state_disturbance_cov
    q_lead = rqr[np.ix_(leaders, leaders)]
    if not _positive_definite(q_lead):
        msg = (
            "a companion block has a singular disturbance covariance on its leading "
            "states (deterministic states are not supported by the structured smoother)"
        )
        raise ValueError(msg)
    p0 = model.P0[np.ix_(states, states)]
    if not _positive_definite(p0):
        msg = "the initial covariance P0 of a companion block is singular"
        raise ValueError(msg)
    return leaders, lengths, q_lead.copy(), p0.copy()


def detect_structure(model: StateSpace) -> StateStructure:
    """Decompose the state vector into companion blocks and observation groups.

    Parameters
    ----------
    model : StateSpace
        State-space model (time-invariant or time-varying observation equation).

    Returns
    -------
    StateStructure
        Blocks, groups and the position of every state.

    Raises
    ------
    ValueError
        If the model does not have the required structure: non-diagonal or singular
        ``H``, lag rows of ``T`` that are not unit shifts, deterministic leading states
        (singular ``R Q R'`` block) or a singular ``P0`` block.

    Examples
    --------
    One factor with an AR(1) and a quarterly series loading on five lags
    (Mariano-Murasawa weights) with its own idiosyncratic chain:

    >>> import numpy as np
    >>> from nowcastbox.statespace import StateSpace, companion_matrix, detect_structure
    >>> m = 5 + 5
    >>> T = np.zeros((m, m))
    >>> T[:5, :5] = companion_matrix([[[0.7]]], n_lags=5)
    >>> T[5:, 5:] = companion_matrix([[[0.3]]], n_lags=5)
    >>> w = np.array([1, 2, 3, 2, 1]) / 3
    >>> Z = np.zeros((2, m))
    >>> Z[0, 0] = 1.0
    >>> Z[1, :5] = 0.8 * w
    >>> Z[1, 5:] = w
    >>> R = np.zeros((m, 2))
    >>> R[0, 0] = R[5, 1] = 1.0
    >>> s = detect_structure(StateSpace(T, Z, np.eye(2), [0.5, 1e-4], selection=R))
    >>> [b.chain_lengths.tolist() for b in s.blocks], s.series_group.tolist()
    ([[5], [5]], [-1, 1])
    """
    _check_observation_equation(model)
    head, lag, _parent = _chains(model)
    m, n = model.n_states, model.n_obs
    used = np.unique(model.period_index(model.n_periods or 1))
    loaded = np.any(model.design_store[used] != 0.0, axis=0)  # (n, m)

    raw_blocks = _blocks(model)
    owners: list[int] = []
    block_data = []
    for states in raw_blocks:
        block_data.append(_make_block(model, states, head, lag))
        series = np.flatnonzero(loaded[:, states].any(axis=1))
        owners.append(int(series[0]) if series.size == 1 else (-2 if series.size > 1 else -1))

    # groups: 0 = common, then one per owning series, then one per unobserved block
    group_members: list[list[int]] = [[b for b, o in enumerate(owners) if o == -2]]
    group_owner = [-1]
    series_group = np.full(n, -1, dtype=np.int64)
    for i in sorted({o for o in owners if o >= 0}):
        series_group[i] = len(group_members)
        group_members.append([b for b, o in enumerate(owners) if o == i])
        group_owner.append(i)
    for b, o in enumerate(owners):
        if o == -1:
            group_members.append([b])
            group_owner.append(-1)

    state_group = np.zeros(m, dtype=np.int64)
    state_offset = np.zeros(m, dtype=np.int64)
    blocks: list[CompanionBlock] = []
    groups: list[StateGroup] = []
    block_group = np.zeros(len(raw_blocks), dtype=np.int64)
    for g, members in enumerate(group_members):
        block_group[members] = g
    for b, states in enumerate(raw_blocks):
        leaders, lengths, q_lead, p0 = block_data[b]
        blocks.append(CompanionBlock(states, leaders, lengths, q_lead, p0, int(block_group[b])))
    for g, members in enumerate(group_members):
        width, n_lags, states_g = 0, 0, []
        for b in members:
            blk = blocks[b]
            for ld, length in zip(blk.leaders, blk.chain_lengths, strict=True):
                chain = blk.states[head[blk.states] == ld]
                state_offset[chain] = width
                width += 1
                n_lags = max(n_lags, int(length))
            states_g.append(blk.states)
        states_arr = np.sort(np.concatenate(states_g)) if states_g else np.zeros(0, dtype=np.int64)
        state_group[states_arr] = g
        groups.append(StateGroup(group_owner[g], tuple(members), states_arr, width, n_lags))
    return StateStructure(
        blocks=tuple(blocks),
        groups=tuple(groups),
        state_group=state_group,
        state_lag=lag,
        state_offset=state_offset,
        series_group=series_group,
    )
