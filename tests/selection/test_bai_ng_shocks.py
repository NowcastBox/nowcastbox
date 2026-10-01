"""Tests for nowcastbox.selection.bai_ng_shocks (Bai & Ng, 2007)."""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from nowcastbox.core.exceptions import NowcastDataError
from nowcastbox.selection import (
    ShockSelectionResult,
    select_shocks,
    shock_bound,
    shock_statistics,
)
from nowcastbox.selection._panel import fit_var_ols, prepare_panel, principal_components
from tests.selection.simulate import dynamic_factor_panel, static_factor_panel

matplotlib.use("Agg")


class TestBound:
    @pytest.mark.parametrize(("n", "t", "delta", "m"), [(100, 200, 0.1, 1.0), (300, 50, 0.2, 1.5)])
    def test_formula(self, n, t, delta, m):
        expected = m / min(n ** (0.5 - delta), t ** (0.5 - delta))
        assert shock_bound(n, t, delta, m) == pytest.approx(expected)

    @pytest.mark.parametrize(
        ("delta", "m"),
        [(0.0, 1.0), (0.5, 1.0), (0.1, 0.0), (0.1, -1.0), (np.nan, 1.0), (0.1, np.inf)],
    )
    def test_invalid(self, delta, m):
        with pytest.raises(ValueError, match=r"delta|m must"):
            shock_bound(10, 10, delta, m)

    def test_invalid_dims(self):
        with pytest.raises(ValueError, match="positive"):
            shock_bound(0, 10)


class TestStatistics:
    def test_manual(self):
        c = np.array([4.0, 2.0, 1.0])
        stats = shock_statistics(c[::-1])  # order does not matter
        total = 16 + 4 + 1
        np.testing.assert_allclose(stats["D1"], np.sqrt([4 / total, 1 / total, 0.0]))
        np.testing.assert_allclose(stats["D2"], np.sqrt([5 / total, 1 / total, 0.0]))
        assert list(stats.index) == [1, 2, 3]

    def test_negative_eigenvalues_clipped(self):
        stats = shock_statistics(np.array([1.0, -1e-12]))
        assert stats.loc[1, "D1"] == 0.0

    def test_zero_matrix(self):
        with pytest.raises(NowcastDataError, match="zero"):
            shock_statistics(np.zeros(3))


class TestRecovery:
    @pytest.mark.parametrize("seed", range(6))
    @pytest.mark.parametrize(("q", "lags"), [(1, 1), (2, 1), (1, 2), (3, 0)])
    def test_recovers_q(self, seed, q, lags):
        x, r = dynamic_factor_panel(300, 150, q, lags, seed=seed)
        res = select_shocks(x, n_factors=r, factor_lags=1)
        assert res.q_by_statistic == {"D1": q, "D2": q}
        assert res.q_star == q

    @pytest.mark.parametrize("seed", range(3))
    def test_correlation_matrix_variant(self, seed):
        x, r = dynamic_factor_panel(300, 150, 2, 1, seed=seed)
        res = select_shocks(x, n_factors=r, matrix="correlation", statistic="D2")
        assert res.q_star == 2 and res.matrix == "correlation"
        np.testing.assert_allclose(np.diag(res.residual_covariance), 1.0)

    def test_q_equals_r_when_shocks_full_rank(self):
        # static factors iid -> VAR residual covariance has full rank r
        x = static_factor_panel(300, 150, 3, seed=1)
        res = select_shocks(x, n_factors=3, factor_lags=1)
        assert res.q_star == 3

    def test_n_factors_selected_automatically(self):
        x, r = dynamic_factor_panel(300, 150, 1, 1, seed=0)
        res = select_shocks(x)
        assert res.n_factors == r == 2
        assert res.q_star == 1

    def test_no_factor_found_raises(self):
        x = np.random.default_rng(5).normal(size=(200, 100))
        with pytest.raises(NowcastDataError, match="no factor"):
            select_shocks(x)


class TestInternals:
    def test_residual_covariance_matches_statsmodels_var(self):
        x, r = dynamic_factor_panel(200, 60, 1, 1, seed=3)
        res = select_shocks(x, n_factors=r, factor_lags=2)
        panel = prepare_panel(x)
        f, _, sv = principal_components(panel.values, r)
        # Bai-Ng (2007) normalisation Lambda'Lambda/N = I: F = X V / sqrt(N)
        _, _, vt = np.linalg.svd(panel.values, full_matrices=False)
        f_bn = panel.values @ vt[:r].T / np.sqrt(panel.n_series)
        np.testing.assert_allclose(np.abs(f * sv[:r] / np.sqrt(x.size)), np.abs(f_bn))
        f = f * sv[:r] / np.sqrt(x.size)
        sm_res = sm.tsa.VAR(f).fit(2, trend="c")
        sigma_ml = sm_res.resid.T @ sm_res.resid / sm_res.resid.shape[0]
        np.testing.assert_allclose(res.residual_covariance, sigma_ml, atol=1e-10)
        np.testing.assert_allclose(
            res.eigenvalues.to_numpy(), np.sort(np.linalg.eigvalsh(sigma_ml))[::-1], atol=1e-10
        )

    def test_fit_var_ols_against_statsmodels(self, rng):
        y = rng.normal(size=(80, 3)).cumsum(axis=0) * 0.1 + rng.normal(size=(80, 3))
        coef, resid = fit_var_ols(y, 2)
        sm_res = sm.tsa.VAR(y).fit(2, trend="c")
        np.testing.assert_allclose(coef, sm_res.params.T, atol=1e-10)
        np.testing.assert_allclose(resid, sm_res.resid, atol=1e-10)
        coef_nt, _ = fit_var_ols(y, 1, trend=False)
        assert coef_nt.shape == (3, 3)

    def test_fit_var_too_short(self):
        with pytest.raises(NowcastDataError, match="periods"):
            fit_var_ols(np.ones((4, 2)), 2)


class TestResult:
    @pytest.fixture
    def res(self) -> ShockSelectionResult:
        x, r = dynamic_factor_panel(200, 80, 1, 1, seed=2)
        cols = [f"s{i}" for i in range(80)]
        return select_shocks(pd.DataFrame(x, columns=cols), n_factors=r, factor_lags=1)

    def test_fields(self, res):
        assert res.n_factors == 2 and res.factor_lags == 1
        assert res.delta == 0.1 and res.m == 1.0
        assert res.n_series == 80 and res.n_periods == 200
        assert res.columns[-1] == "s79"
        assert res.bound == pytest.approx(shock_bound(80, 200))
        assert (res.statistics["D1"] <= res.statistics["D2"] + 1e-15).all()
        assert res.residual_covariance.shape == (2, 2)

    def test_summary(self, res):
        text = res.summary()
        assert "q* = 1" in text and "M_NT=" in text
        assert str(res) == text

    def test_plot(self, res):
        import matplotlib.pyplot as plt

        ax = res.plot()
        assert ax.get_ylabel() == "D statistic"
        plt.close("all")
        fig = res.plot(backend="plotly")
        assert fig.layout.xaxis.title.text == "number of shocks k"

    def test_larger_m_selects_fewer_shocks(self):
        x, r = dynamic_factor_panel(300, 150, 2, 1, seed=0)
        loose = select_shocks(x, n_factors=r, m=10.0)
        assert loose.q_star <= 2
        assert loose.q_star == 1 or loose.bound > loose.statistics.loc[1, "D1"]


class TestValidation:
    @pytest.fixture
    def x(self):
        return dynamic_factor_panel(100, 40, 1, 1, seed=0)[0]

    def test_invalid_statistic(self, x):
        with pytest.raises(ValueError, match="statistic"):
            select_shocks(x, 2, statistic="D3")

    def test_invalid_matrix(self, x):
        with pytest.raises(ValueError, match="matrix"):
            select_shocks(x, 2, matrix="precision")

    @pytest.mark.parametrize("lags", [0, -1, 1.5, True])
    def test_invalid_factor_lags(self, x, lags):
        with pytest.raises(ValueError, match="factor_lags"):
            select_shocks(x, 2, factor_lags=lags)

    @pytest.mark.parametrize("n_factors", [0, 2.0, False])
    def test_invalid_n_factors(self, x, n_factors):
        with pytest.raises(ValueError, match="n_factors"):
            select_shocks(x, n_factors)

    def test_n_factors_too_large(self, x):
        with pytest.raises(ValueError, match="exceeds"):
            select_shocks(x, 41)

    def test_invalid_delta(self, x):
        with pytest.raises(ValueError, match="delta"):
            select_shocks(x, 2, delta=0.7)

    def test_var_too_short(self):
        x = np.random.default_rng(0).normal(size=(8, 10))
        with pytest.raises(NowcastDataError, match="VAR"):
            select_shocks(x, n_factors=3, factor_lags=3)

    def test_correlation_with_degenerate_residual(self):
        # Every series is a linear trend: the single PCA factor is exactly fitted by a
        # VAR(1) with intercept, so its residual has (numerically) zero variance.
        trend = np.arange(60, dtype=float)
        x = np.column_stack([a * trend + b for a, b in [(1, 0), (2, 1), (-1, 3), (0.5, 2)]])
        with pytest.raises(NowcastDataError, match="zero variance"):
            select_shocks(x, n_factors=1, matrix="correlation")
