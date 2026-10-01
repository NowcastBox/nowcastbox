"""Bai & Ng (2002, 2007) defaults checked against the papers and the R package.

Reference numbers: the R package ``nowcasting`` (GPL-3) was **called as a black box**
(its source was not read) on the panels below, regenerated here with the same seeds::

    library(nowcasting)
    ICfactors(x, rmax = 8, type = 2)$r_star
    ICshocks(x, r = r, p = 1)$q_star     # defaults delta = 0.1, m = 1
    ICshocks(x, r = r, p = 2)$q_star

The panels were written to CSV with ``numpy.savetxt`` (full precision) from
:func:`tests.selection.simulate.reduced_rank_var_panel` (cases 0-5, DGP 3 of Bai & Ng,
2007) and :func:`tests.selection.simulate.dynamic_factor_panel` (cases 6-8).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from nowcastbox.core.exceptions import DataQualityWarning
from nowcastbox.selection import (
    DEFAULT_M,
    MIN_RELIABLE_DIM,
    select_factors,
    select_shocks,
    shock_bound,
    shock_statistics,
)
from nowcastbox.selection._panel import fit_var_ols, prepare_panel, principal_components
from tests.selection.simulate import dynamic_factor_panel, reduced_rank_var_panel

# case: (generator args, r used in ICshocks, true q, R r_star (IC2), R q_star p=1, p=2)
DGP3 = {
    0: ((200, 100, 5, 3, 0.5), 5, 3, 5, 3, 3),
    1: ((150, 80, 4, 2, 0.7), 4, 2, 2, 2, 2),
    2: ((300, 120, 3, 1, 0.5), 3, 1, 1, 1, 1),
    3: ((100, 60, 4, 4, 0.5), 4, 4, 4, 3, 3),
    4: ((200, 50, 5, 2, 0.8), 5, 2, 8, 2, 2),
    5: ((240, 100, 2, 1, 0.9), 2, 1, 1, 2, 2),
}
DYNAMIC = {
    6: ((1, 1, 1), 2, 1, 2, 1, 1),
    7: ((2, 2, 1), 4, 2, 4, 2, 2),
    8: ((3, 1, 2), 3, 1, 3, 1, 1),
}
# Cases where our q differs from the R black box (see test_shocks_reference_divergence).
DIVERGENT = {4, 5}


def _panel(case: int) -> tuple[np.ndarray, int, int, int, int, int]:
    if case in DGP3:
        (t, n, r, q, rho), r_used, q_true, r_ic2, q1, q2 = DGP3[case]
        x = reduced_rank_var_panel(t, n, r, q, rho, seed=100 + case)
    else:
        (seed, q, s), r_used, q_true, r_ic2, q1, q2 = DYNAMIC[case]
        x, _ = dynamic_factor_panel(200, 80, q, s, seed=seed)
    return x, r_used, q_true, r_ic2, q1, q2


ALL_CASES = sorted([*DGP3, *DYNAMIC])


@pytest.mark.reference_validation
@pytest.mark.parametrize("case", ALL_CASES)
def test_factors_match_r_ic2(case):
    x, _, _, r_ic2, _, _ = _panel(case)
    assert select_factors(x, rmax=8, criterion="IC2").r_star == r_ic2


@pytest.mark.reference_validation
@pytest.mark.parametrize("case", [c for c in ALL_CASES if c not in DIVERGENT])
@pytest.mark.parametrize("lags", [1, 2])
def test_shocks_match_r(case, lags):
    x, r_used, _, _, q1, q2 = _panel(case)
    res = select_shocks(x, n_factors=r_used, factor_lags=lags)
    assert res.q_star == (q1 if lags == 1 else q2)


@pytest.mark.reference_divergence
@pytest.mark.parametrize("case", sorted(DIVERGENT))
def test_shocks_reference_divergence(case):
    """R selects q = 2 where we select 1; in case 5 the true q is 1 (ours is right).

    In both cases ``D1_1`` is far below the bound ``M_NT`` under the paper's
    normalisation, so the divergence is not a borderline rounding effect.
    """
    x, r_used, q_true, _, q1, _ = _panel(case)
    res = select_shocks(x, n_factors=r_used, factor_lags=1)
    assert res.q_star == 1 != q1
    assert res.statistics.loc[1, "D1"] < 0.8 * res.bound
    if case == 5:
        assert res.q_star == q_true


@pytest.mark.parametrize("case", ALL_CASES)
def test_shocks_close_to_truth_with_paper_defaults(case):
    """Paper defaults (VAR(2), delta = 0.1, m = 1, covariance, D1) are within one of q."""
    x, r_used, q_true, _, _, _ = _panel(case)
    assert abs(select_shocks(x, n_factors=r_used).q_star - q_true) <= 1


def test_paper_normalisation_beats_unit_variance_factors():
    """With F'F/T = I (unit-variance factors) the covariance test over-selects q = r."""
    x, r_used, q_true, _, _, _ = _panel(0)
    panel = prepare_panel(x)
    f, _, _ = principal_components(panel.values, r_used)
    _, u = fit_var_ols(f, 1)
    stats = shock_statistics(np.linalg.eigvalsh(u.T @ u / len(u)))
    bound = shock_bound(panel.n_series, panel.n_periods)
    unit_q = int(stats.index[np.flatnonzero(stats["D1"].to_numpy() < bound)[0]])
    assert unit_q == r_used
    assert select_shocks(x, n_factors=r_used, factor_lags=1).q_star == q_true


class TestDefaults:
    def test_default_m_table(self):
        assert DEFAULT_M == {
            ("covariance", "D1"): 1.0,
            ("covariance", "D2"): 1.0,
            ("correlation", "D1"): 1.25,
            ("correlation", "D2"): 2.25,
        }

    def test_correlation_uses_statistic_specific_m(self):
        x, _ = dynamic_factor_panel(200, 80, 1, 1, seed=1)
        res = select_shocks(x, n_factors=2, matrix="correlation", statistic="D2")
        assert res.m_by_statistic == {"D1": 1.25, "D2": 2.25}
        assert res.m == 2.25
        expected = 2.25 / min(80**0.4, 200**0.4)
        assert res.bound == pytest.approx(expected)
        assert res.bound_by_statistic["D1"] == pytest.approx(1.25 / min(80**0.4, 200**0.4))
        assert "D2: m=2.25" in res.summary()

    def test_explicit_m_applies_to_both(self):
        x, _ = dynamic_factor_panel(200, 80, 1, 1, seed=1)
        res = select_shocks(x, n_factors=2, matrix="correlation", m=0.5)
        assert res.m_by_statistic == {"D1": 0.5, "D2": 0.5}
        assert res.bound_by_statistic["D1"] == res.bound_by_statistic["D2"] == res.bound

    def test_default_factor_lags_is_two(self):
        x, _ = dynamic_factor_panel(200, 80, 1, 1, seed=1)
        assert select_shocks(x, n_factors=2).factor_lags == 2

    def test_invalid_m(self):
        x, _ = dynamic_factor_panel(200, 80, 1, 1, seed=1)
        with pytest.raises(ValueError, match="m must be positive"):
            select_shocks(x, n_factors=2, m=0.0)

    def test_correlation_invariant_to_factor_scaling(self):
        """The correlation variant does not depend on the PC normalisation."""
        x, r = dynamic_factor_panel(200, 60, 2, 1, seed=4)
        res = select_shocks(x, n_factors=r, factor_lags=1, matrix="correlation")
        panel = prepare_panel(x)
        f, _, _ = principal_components(panel.values, r)
        _, u = fit_var_ols(f, 1)
        s = u.T @ u / len(u)
        d = np.sqrt(np.diag(s))
        corr = s / np.outer(d, d)
        np.testing.assert_allclose(
            res.eigenvalues.to_numpy(), np.sort(np.linalg.eigvalsh(corr))[::-1], atol=1e-10
        )


class TestSmallSampleWarning:
    def test_warns_below_threshold(self, rng):
        x = rng.normal(size=(100, MIN_RELIABLE_DIM - 1))
        with pytest.warns(DataQualityWarning, match=r"min\(N, T\) < 20"):
            select_factors(x, rmax=3)

    def test_warns_small_t(self, rng):
        x = rng.normal(size=(15, 50))
        with pytest.warns(DataQualityWarning, match="T=15"):
            select_factors(x, rmax=3)

    def test_no_warning_at_threshold(self, rng):
        x = rng.normal(size=(MIN_RELIABLE_DIM, MIN_RELIABLE_DIM))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            select_factors(x, rmax=3)

    def test_can_be_disabled(self, rng):
        x = rng.normal(size=(100, 8))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            select_factors(x, rmax=3, warn_small_sample=False)
