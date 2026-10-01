"""Tests of the first-step building blocks (principal components, factor VAR, B)."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from nowcastbox.core.exceptions import NowcastDataError
from nowcastbox.models._pca import (
    FactorVAR,
    fit_factor_var,
    principal_components,
    shock_loadings,
)


# ---------------------------------------------------------------------- principal components
class TestPrincipalComponents:
    def test_matches_eigendecomposition(self, rng: np.random.Generator) -> None:
        x = rng.standard_normal((80, 6)) @ rng.standard_normal((6, 6))
        pc = principal_components(x, 3)
        cov = x.T @ x / 80
        eigval = np.sort(np.linalg.eigvalsh(cov))[::-1]
        np.testing.assert_allclose(pc.eigenvalues, eigval, rtol=1e-10, atol=1e-12)
        # loadings are eigenvectors of the covariance
        np.testing.assert_allclose(cov @ pc.loadings, pc.loadings * eigval[:3], atol=1e-9)

    def test_normalisation(self, rng: np.random.Generator) -> None:
        x = rng.standard_normal((120, 8))
        pc = principal_components(x, 4)
        np.testing.assert_allclose(pc.loadings.T @ pc.loadings, np.eye(4), atol=1e-12)
        np.testing.assert_allclose(
            pc.factors.T @ pc.factors / 120, np.diag(pc.eigenvalues[:4]), atol=1e-10
        )
        np.testing.assert_allclose(pc.factors, x @ pc.loadings)
        np.testing.assert_allclose(pc.residuals, x - pc.factors @ pc.loadings.T)
        # residuals orthogonal to the factors
        np.testing.assert_allclose(pc.factors.T @ pc.residuals, 0.0, atol=1e-9)

    def test_matches_svd(self, rng: np.random.Generator) -> None:
        x = rng.standard_normal((60, 5))
        pc = principal_components(x, 2)
        _, s, vt = np.linalg.svd(x, full_matrices=False)
        np.testing.assert_allclose(pc.eigenvalues, s**2 / 60, rtol=1e-10)
        for k in range(2):
            assert abs(abs(vt[k] @ pc.loadings[:, k]) - 1.0) < 1e-10

    def test_sign_convention(self, rng: np.random.Generator) -> None:
        x = rng.standard_normal((50, 7))
        pc = principal_components(x, 3)
        assert np.all(pc.loadings.sum(axis=0) >= 0)
        pc_neg = principal_components(-x, 3)
        np.testing.assert_allclose(pc_neg.loadings, pc.loadings, atol=1e-10)
        np.testing.assert_allclose(pc_neg.factors, -pc.factors, atol=1e-10)

    def test_explained_variance(self, rng: np.random.Generator) -> None:
        pc = principal_components(rng.standard_normal((40, 5)), 2)
        assert pc.n_factors == 2
        assert pc.explained_variance_ratio.sum() == pytest.approx(1.0)
        np.testing.assert_allclose(pc.idiosyncratic_variance, (pc.residuals**2).mean(axis=0))

    def test_zero_matrix(self) -> None:
        pc = principal_components(np.zeros((10, 3)), 1)
        np.testing.assert_array_equal(pc.explained_variance_ratio, np.zeros(3))

    def test_recovers_single_factor(self, rng: np.random.Generator) -> None:
        f = rng.standard_normal(500)
        lam = rng.uniform(0.5, 1.5, 30)
        x = np.outer(f, lam) + 0.3 * rng.standard_normal((500, 30))
        pc = principal_components(x, 1)
        assert abs(np.corrcoef(pc.factors[:, 0], f)[0, 1]) > 0.99

    @pytest.mark.parametrize(
        ("x", "match"),
        [
            (np.ones(5), "2-D"),
            (np.empty((0, 3)), "2-D"),
            (np.array([[1.0, np.nan], [2.0, 3.0]]), "missing"),
            (np.array([[1.0, np.inf], [2.0, 3.0]]), "missing"),
        ],
    )
    def test_bad_data(self, x: np.ndarray, match: str) -> None:
        with pytest.raises(NowcastDataError, match=match):
            principal_components(x, 1)

    @pytest.mark.parametrize("n_factors", [0, 4, 1.5, True])
    def test_bad_n_factors(self, n_factors: object) -> None:
        with pytest.raises(ValueError, match="n_factors"):
            principal_components(np.ones((3, 5)), n_factors)  # type: ignore[arg-type]

    @settings(max_examples=40, deadline=None)
    @given(
        arrays(
            np.float64,
            st.tuples(st.integers(5, 30), st.integers(2, 8)),
            elements=st.floats(-10, 10, allow_nan=False, allow_infinity=False),
        ),
        st.integers(1, 2),
    )
    def test_property_reconstruction(self, x: np.ndarray, r: int) -> None:
        pc = principal_components(x, r)
        np.testing.assert_allclose(pc.loadings.T @ pc.loadings, np.eye(r), atol=1e-8)
        np.testing.assert_allclose(pc.factors @ pc.loadings.T + pc.residuals, x, atol=1e-8)
        assert np.all(np.diff(pc.eigenvalues) <= 1e-9)
        # PCA minimises the residual sum of squares: it equals the sum of dropped eigenvalues
        ssr = float((pc.residuals**2).sum()) / x.shape[0]
        assert ssr == pytest.approx(float(pc.eigenvalues[r:].sum()), abs=1e-6 * (1 + ssr))


# ---------------------------------------------------------------------- factor VAR
def _simulate_var(rng: np.random.Generator, coefs: list[np.ndarray], n: int) -> np.ndarray:
    r = coefs[0].shape[0]
    p = len(coefs)
    f = np.zeros((n + 100, r))
    for t in range(p, n + 100):
        f[t] = sum(a @ f[t - i - 1] for i, a in enumerate(coefs)) + rng.standard_normal(r)
    return f[100:]


class TestFactorVAR:
    def test_matches_statsmodels(self, rng: np.random.Generator) -> None:
        from statsmodels.tsa.api import VAR

        a1 = np.array([[0.5, 0.1], [0.0, 0.3]])
        a2 = np.array([[0.1, 0.0], [0.05, 0.2]])
        f = _simulate_var(rng, [a1, a2], 300)
        var = fit_factor_var(f, 2)
        sm_res = VAR(f).fit(2, trend="n")
        np.testing.assert_allclose(var.coefficients, np.hstack(sm_res.coefs), atol=1e-10)
        np.testing.assert_allclose(var.residuals, sm_res.resid, atol=1e-10)
        np.testing.assert_allclose(var.residual_cov, sm_res.sigma_u_mle, atol=1e-10)
        mats = var.matrices()
        assert len(mats) == 2
        np.testing.assert_allclose(mats[1], sm_res.coefs[1], atol=1e-10)

    def test_recovers_coefficients(self, rng: np.random.Generator) -> None:
        a1 = np.array([[0.6, 0.2], [-0.1, 0.4]])
        f = _simulate_var(rng, [a1], 20000)
        var = fit_factor_var(f, 1)
        np.testing.assert_allclose(var.coefficients, a1, atol=0.03)
        np.testing.assert_allclose(var.residual_cov, np.eye(2), atol=0.05)

    def test_one_dimensional_input(self, rng: np.random.Generator) -> None:
        f = rng.standard_normal(50)
        var = fit_factor_var(f, 1)
        assert var.coefficients.shape == (1, 1)
        assert isinstance(var, FactorVAR)

    def test_nan_rows_are_skipped(self, rng: np.random.Generator) -> None:
        f = _simulate_var(rng, [0.5 * np.eye(2)], 200)
        g = f.copy()
        g[100] = np.nan
        var = fit_factor_var(g, 1)
        # rows t=100 (dependent) and t=101 (lag) are dropped
        y = np.vstack([f[1:100], f[102:]])
        x = np.vstack([f[0:99], f[101:-1]])
        coef, *_ = np.linalg.lstsq(x, y, rcond=None)
        np.testing.assert_allclose(var.coefficients, coef.T, atol=1e-12)
        assert var.residuals.shape == (197, 2)

    def test_too_few_periods(self) -> None:
        with pytest.raises(NowcastDataError, match="usable periods"):
            fit_factor_var(np.ones((3, 2)), 2)
        with pytest.raises(NowcastDataError, match="usable periods"):
            fit_factor_var(np.ones((2, 2)), 3)

    def test_bad_inputs(self) -> None:
        with pytest.raises(NowcastDataError, match="infinite"):
            fit_factor_var(np.array([[1.0], [np.inf], [2.0]]), 1)
        with pytest.raises(NowcastDataError, match="2-D"):
            fit_factor_var(np.ones((2, 2, 2)), 1)
        with pytest.raises(ValueError, match="integer"):
            fit_factor_var(np.ones((10, 1)), 1.0)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match=">= 1"):
            fit_factor_var(np.ones((10, 1)), 0)


# ---------------------------------------------------------------------- shock loadings
class TestShockLoadings:
    def test_full_rank_reproduces_covariance(self, rng: np.random.Generator) -> None:
        m = rng.standard_normal((3, 3))
        cov = m @ m.T
        b, ev = shock_loadings(cov, 3)
        np.testing.assert_allclose(b @ b.T, cov, atol=1e-10)
        np.testing.assert_allclose(ev, np.sort(np.linalg.eigvalsh(cov))[::-1], atol=1e-10)

    def test_reduced_rank(self, rng: np.random.Generator) -> None:
        m = rng.standard_normal((4, 4))
        cov = m @ m.T
        b, ev = shock_loadings(cov, 2)
        assert b.shape == (4, 2)
        assert np.linalg.matrix_rank(b @ b.T) == 2
        # best rank-2 approximation: error norm equals the dropped eigenvalues
        resid = np.linalg.eigvalsh(cov - b @ b.T)
        np.testing.assert_allclose(np.sort(resid)[::-1][:2], ev[2:], atol=1e-9)
        np.testing.assert_allclose(b.T @ b, np.diag(ev[:2]), atol=1e-9)
        assert np.all(b.sum(axis=0) >= 0)

    def test_errors(self) -> None:
        with pytest.raises(ValueError, match="square"):
            shock_loadings(np.ones((2, 3)), 1)
        with pytest.raises(ValueError, match="between"):
            shock_loadings(np.eye(2), 3)
        with pytest.raises(ValueError, match="between"):
            shock_loadings(np.eye(2), 0)
        with pytest.raises(ValueError, match="integer"):
            shock_loadings(np.eye(2), 1.0)  # type: ignore[arg-type]
