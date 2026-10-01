"""Detection of the companion / idiosyncratic block structure."""

from __future__ import annotations

import numpy as np
import pytest

from nowcastbox.statespace import (
    CompanionBlock,
    StateGroup,
    StateSpace,
    StateStructure,
    companion_matrix,
    detect_structure,
)

from ._dfm import dfm_state_space


def test_mixed_frequency_layout(rng):
    ssm = dfm_state_space(3, 2, rng=rng, n_factors=2, factor_lags=2)
    st = detect_structure(ssm)
    assert isinstance(st, StateStructure)
    # one factor block (2 chains of 5 lags) + 5 idiosyncratic chains
    assert len(st.blocks) == 6
    factor = st.blocks[0]
    assert isinstance(factor, CompanionBlock)
    assert factor.group == 0
    assert factor.chain_lengths.tolist() == [5, 5]
    assert factor.disturbance_cov.shape == (2, 2)
    assert factor.initial_cov.shape == (10, 10)
    assert st.n_common_states == 10
    np.testing.assert_array_equal(st.common_states, np.arange(10))
    assert st.n_private_groups == 5
    assert st.series_group.tolist() == [1, 2, 3, 4, 5]
    lengths = [b.chain_lengths.tolist() for b in st.blocks[1:]]
    assert lengths == [[1], [1], [1], [5], [5]]
    grp = st.groups[4]
    assert isinstance(grp, StateGroup)
    assert (grp.owner, grp.width, grp.n_lags) == (3, 1, 5)
    # lags of the factor chains: (f1_t, f2_t, f1_{t-1}, ...)
    assert st.state_lag[:10].tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    assert st.state_offset[:10].tolist() == [0, 1] * 5


def test_two_common_blocks_share_the_common_group(rng):
    ssm = dfm_state_space(4, 0, rng=rng, n_blocks=2, idiosyncratic="iid")
    st = detect_structure(ssm)
    assert len(st.groups) == 1
    assert st.groups[0].blocks == (0, 1)
    assert st.groups[0].width == 2
    assert (st.series_group == -1).all()


def test_unloaded_block_has_its_own_group():
    ssm = StateSpace(np.diag([0.5, 0.4]), [[1.0, 0.0], [2.0, 0.0]], np.eye(2), [1.0, 1.0])
    st = detect_structure(ssm)
    assert [g.owner for g in st.groups] == [-1, -1]
    assert st.groups[1].states.tolist() == [1]


def test_time_varying_loadings_use_union_of_patterns():
    zs = np.array([[[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]], [[1.0, 0.0, 0.0], [1.0, 0.0, 1.0]]])
    ssm = StateSpace(np.diag([0.5, 0.4, 0.3]), zs, np.eye(3), [[1.0, 1.0]] * 2, obs_index=[0, 1])
    st = detect_structure(ssm)
    assert st.series_group.tolist() == [1, 2]


def test_rounding_noise_in_p0_does_not_merge_blocks():
    p0 = np.diag([2.0, 1.5])
    p0[0, 1] = p0[1, 0] = 1e-18
    ssm = StateSpace(
        np.diag([0.5, 0.4]), [[1.0, 1.0], [1.0, 0.0]], np.eye(2), [1.0, 1.0], initial_state_cov=p0
    )
    assert len(detect_structure(ssm).blocks) == 2


class TestRejected:
    def test_full_obs_cov(self):
        ssm = StateSpace([[0.5]], [[1.0], [1.0]], [[1.0]], [[1.0, 0.5], [0.5, 1.0]])
        with pytest.raises(ValueError, match="diagonal"):
            detect_structure(ssm)

    def test_zero_obs_variance(self):
        ssm = StateSpace([[0.5]], [[1.0], [1.0]], [[1.0]], [1.0, 0.0])
        with pytest.raises(ValueError, match="positive"):
            detect_structure(ssm)

    def test_shift_cycle(self):
        # two states copying each other, the second with a separate noisy state
        t_mat = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.5]])
        sel = np.array([[0.0], [0.0], [1.0]])
        p0 = np.eye(3)
        ssm = StateSpace(t_mat, np.eye(3), [[1.0]], np.ones(3), selection=sel, initial_state_cov=p0)
        with pytest.raises(ValueError, match="cycle"):
            detect_structure(ssm)

    def test_two_copies_of_one_state(self):
        t_mat = np.array([[0.5, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        sel = np.array([[1.0], [0.0], [0.0]])
        ssm = StateSpace(
            t_mat, np.eye(3), [[1.0]], np.ones(3), selection=sel, initial_state_cov=np.eye(3)
        )
        with pytest.raises(ValueError, match="more than one lag copy"):
            detect_structure(ssm)

    def test_deterministic_leader(self):
        # a constant state (T = 1, no noise) is not a stochastic companion chain
        t_mat = np.diag([0.5, 1.0])
        sel = np.array([[1.0], [0.0]])
        ssm = StateSpace(
            t_mat, [[1.0, 1.0]], [[1.0]], [1.0], selection=sel, initial_state_cov=np.eye(2)
        )
        with pytest.raises(ValueError, match="singular disturbance"):
            detect_structure(ssm)

    def test_singular_initial_cov(self):
        t_mat = companion_matrix([[[0.5]]], n_lags=2)
        sel = np.array([[1.0], [0.0]])
        ssm = StateSpace(
            t_mat, [[1.0, 0.5]], [[1.0]], [1.0], selection=sel, initial_state_cov=np.ones((2, 2))
        )
        with pytest.raises(ValueError, match="singular"):
            detect_structure(ssm)

    def test_singular_multivariate_initial_cov(self):
        ssm = StateSpace(
            np.array([[0.5, 0.1], [0.0, 0.4]]), [[1.0, 1.0]], np.eye(2), [1.0],
            initial_state_cov=np.ones((2, 2)),
        )  # fmt: skip
        with pytest.raises(ValueError, match="singular"):
            detect_structure(ssm)
