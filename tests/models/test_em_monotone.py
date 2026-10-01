"""Exact M-step with the initial-state term: monotone EM (STATUS gap 8)."""

from __future__ import annotations

import numpy as np
import pytest

from nowcastbox.core.data import MixedFrequencyData
from nowcastbox.models._em_steps import (
    EMParameters,
    StateLayout,
    _ar1_objectives,
    _ar1_segment_cov,
    _factor_objective,
    _first_period_cov,
    _gaussian_term,
    _safeguarded_step,
    build_state_space,
    e_step,
    m_step,
)
from nowcastbox.models._init_conditions import pca_initial_parameters
from nowcastbox.models.em import build_layout
from tests.models.test_em_simulation import simulate_mixed_dfm


def _setup(seed: int, idiosyncratic: str):
    sim = simulate_mixed_dfm(n_periods=72, n_monthly=6, n_quarterly=1, seed=seed)
    panel, _ = MixedFrequencyData(sim.data, sim.frequencies).standardize()
    layout = build_layout(panel, 1, 1, None, idiosyncratic)
    return panel.values, layout, pca_initial_parameters(panel, layout)


def _path(y, layout, params, n_iter, *, initial_term, method="univariate"):
    lls = []
    for _ in range(n_iter):
        stats = e_step(build_state_space(params, layout), y, method=method)
        lls.append(stats.loglikelihood)
        params = m_step(stats, layout, y, params, initial_term=initial_term)
    return np.asarray(lls)


@pytest.mark.parametrize(("seed", "idiosyncratic"), [(4120, "iid"), (1464, "ar1")])
def test_initial_term_restores_monotonicity(seed, idiosyncratic):
    """The classical update decreases the likelihood on these seeds; the exact one does not."""
    y, layout, params = _setup(seed, idiosyncratic)
    classical = _path(y, layout, params, 16, initial_term=False)
    exact = _path(y, layout, params, 16, initial_term=True)
    assert np.min(np.diff(classical) / np.abs(classical[1:])) < -1e-8
    assert np.all(np.diff(exact) >= -1e-10 * np.abs(exact[1:]))


@pytest.mark.parametrize("method", ["univariate", "structured"])
def test_same_safeguard_with_both_e_steps(method):
    y, layout, params = _setup(1464, "ar1")
    reference = _path(y, layout, params, 6, initial_term=True)
    other = _path(y, layout, params, 6, initial_term=True, method=method)
    assert np.allclose(reference, other, rtol=0, atol=1e-7)


def test_first_period_cov_structured_matches_dense():
    y, layout, params = _setup(7, "ar1")
    model = build_state_space(params, layout)
    dense = _first_period_cov(e_step(model, y, method="univariate"))
    fast = _first_period_cov(e_step(model, y, method="structured"))
    known = np.isfinite(fast)
    assert known.sum() > layout.n_factor_states**2
    assert np.allclose(fast[known], dense[known], atol=1e-9)


def test_ar1_objective_matches_gaussian_density():
    rng = np.random.default_rng(0)
    for length in (1, 3, 5):
        X = rng.standard_normal((length, length + 2))
        M = X @ X.T / 2.0
        rho, var, n_pairs = 0.6, 1.7, 20
        transitions = (5.0, 2.0, 4.0)
        initial = (
            M[0, 0],
            np.diag(M)[1:].sum(),
            np.trace(M, offset=-1),
            np.diag(M)[:-1].sum(),
            float(length),
        )
        value = _ar1_objectives(
            np.array(rho),
            np.array(var),
            tuple(np.array(v) for v in transitions),  # type: ignore[arg-type]
            tuple(np.array(v) for v in initial),  # type: ignore[arg-type]
            n_pairs,
        )
        ssr = transitions[0] - 2 * rho * transitions[1] + rho**2 * transitions[2]
        expected = -0.5 * (n_pairs * np.log(var) + ssr / var)
        expected += _gaussian_term(_ar1_segment_cov(rho, var, length), M)
        assert float(value) == pytest.approx(expected, rel=1e-12)


def test_ar1_objective_rejects_invalid_values():
    ones = tuple(np.ones(1) for _ in range(3))
    initial = tuple(np.ones(1) for _ in range(5))
    unit = np.array([1.0])
    out = _ar1_objectives(unit, unit, ones, initial, 10)  # type: ignore[arg-type]
    assert out[0] == -np.inf


def test_gaussian_term_of_singular_matrix_is_minus_infinity():
    assert _gaussian_term(np.zeros((2, 2)), np.eye(2)) == -np.inf


def test_safeguarded_step():
    assert _safeguarded_step(lambda s: s) == 1.0  # increasing: full step
    assert _safeguarded_step(lambda s: -((s - 0.3) ** 2)) == 0.5  # first improving halving
    assert _safeguarded_step(lambda s: -s) == 0.0  # never improves: keep the old value


def test_factor_objective_is_maximised_near_the_closed_form():
    y, layout, params = _setup(11, "iid")
    stats = e_step(build_state_space(params, layout), y)
    new = m_step(stats, layout, y, params)
    M = np.outer(stats.smoother.smoothed_state[0, :5], stats.smoother.smoothed_state[0, :5])
    M = M + stats.smoother.smoothed_state_cov[0][:5, :5]
    best = _factor_objective(new.transition[0], new.factor_cov[0], 0, stats, layout, M)
    old = _factor_objective(params.transition[0], params.factor_cov[0], 0, stats, layout, M)
    assert best >= old
    for scale in (0.9, 1.1):
        moved = _factor_objective(new.transition[0] * scale, new.factor_cov[0], 0, stats, layout, M)
        assert moved <= best


def test_m_step_is_idempotent_for_fixed_moments():
    """Re-running the M-step from its own output on the same moments changes nothing."""
    lay = StateLayout(["a", "b"], ["g"], [1], 1, np.ones((2, 1), bool), [[1.0], [1.0]], "ar1")
    p = EMParameters(
        (np.array([[0.5]]),),
        (np.eye(1),),
        np.ones((2, 1)),
        np.array([0.3, 0.2]),
        np.ones(2),
        np.full(2, 1e-4),
    )
    y = np.random.default_rng(1).standard_normal((40, 2))
    stats = e_step(build_state_space(p, lay), y)
    new = m_step(stats, lay, y, p)
    again = m_step(stats, lay, y, new)  # same moments: the old update cannot be improved on
    assert np.allclose(again.transition[0], new.transition[0])
    assert np.allclose(again.idio_ar, new.idio_ar)
    assert np.allclose(again.idio_var, new.idio_var)
