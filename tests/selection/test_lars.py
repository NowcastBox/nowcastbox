"""Tests for nowcastbox.selection._lars (least angle regression, Efron et al. 2004)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from nowcastbox.core.exceptions import ConvergenceWarning, NowcastDataError
from nowcastbox.selection import LarsPath, lars_path
from nowcastbox.selection._lars import _drop_step, _one_step, _State


def _standardized(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = X - X.mean(axis=0)
    return X / X.std(axis=0), y - y.mean()


def _sparse(seed: int, n: int = 60, p: int = 10) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    X[:, 3] += 0.8 * X[:, 1]
    y = X[:, :4] @ np.array([1.0, -2.0, 0.5, 1.5]) + rng.normal(size=n)
    return _standardized(X, y)


def _factor_design(seed: int, n: int = 40, p: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Strongly correlated predictors: the lasso path drops variables for some seeds."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3)) @ rng.normal(size=(3, p)) + 0.5 * rng.normal(size=(n, p))
    y = X @ rng.normal(size=p) + rng.normal(size=n)
    return _standardized(X, y)


def _reentry_design() -> tuple[np.ndarray, np.ndarray]:
    """Design (n=11, p=8) whose lasso path drops predictor 5 and then re-adds it."""
    rng = np.random.default_rng(112)
    n, p = int(rng.integers(5, 40)), int(rng.integers(2, 60))
    X = rng.normal(size=(n, p))
    X, _ = _standardized(X, np.zeros(n))
    y = X[:, :3].sum(axis=1) + rng.normal(size=n)
    return X, y - y.mean()


# ---------------------------------------------------------------------------
# Reference: scikit-learn (numerical reference only, never imported by the package)
# ---------------------------------------------------------------------------
@pytest.mark.reference_validation
class TestAgainstSklearn:
    @pytest.mark.parametrize("seed", range(5))
    @pytest.mark.parametrize("method", ["lar", "lasso"])
    def test_path_matches(self, seed, method):
        linear_model = pytest.importorskip("sklearn.linear_model")
        X, y = _sparse(seed)
        alphas, active, coefs = linear_model.lars_path(X, y, method=method)
        path = lars_path(X, y, method=method)
        assert path.active == [int(j) for j in active]
        assert path.entry_order == [int(j) for j in active]
        np.testing.assert_allclose(path.alphas, alphas, atol=1e-10)
        np.testing.assert_allclose(path.coefs, coefs, atol=1e-10)

    @pytest.mark.parametrize("seed", [4, 5, 6, 9, 13])
    def test_lasso_with_drops_matches(self, seed):
        linear_model = pytest.importorskip("sklearn.linear_model")
        X, y = _factor_design(seed)
        path = lars_path(X, y, method="lasso")
        assert path.drops, "design chosen so that the lasso modification drops a variable"
        alphas, active, coefs = linear_model.lars_path(X, y, method="lasso")
        assert path.active == [int(j) for j in active]
        np.testing.assert_allclose(path.alphas, alphas, atol=1e-10)
        np.testing.assert_allclose(path.coefs, coefs, atol=1e-10)

    def test_more_predictors_than_observations(self):
        linear_model = pytest.importorskip("sklearn.linear_model")
        rng = np.random.default_rng(1)
        X, y = _standardized(rng.normal(size=(15, 30)), rng.normal(size=15))
        path = lars_path(X, y)
        assert len(path.active) == 14  # rank of the centred design is n - 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, active, coefs = linear_model.lars_path(X, y, method="lar")
        k = len(path.active)
        assert path.entry_order == [int(j) for j in active[:k]]
        np.testing.assert_allclose(path.coefs[:, :k], coefs[:, :k], atol=1e-8)

    def test_lasso_reentry_after_drop_matches(self):
        """A predictor dropped by the lasso modification re-enters with the opposite sign.

        Regression test: the zero-length crossing of the just-dropped predictor must not
        hide its later re-entry through the other boundary.
        """
        linear_model = pytest.importorskip("sklearn.linear_model")
        X, y = _reentry_design()
        path = lars_path(X, y, method="lasso")
        assert 5 in path.drops and 5 in path.active
        alphas, active, coefs = linear_model.lars_path(X, y, method="lasso")
        assert path.active == [int(j) for j in active]
        np.testing.assert_allclose(path.alphas, alphas, atol=1e-10)
        np.testing.assert_allclose(path.coefs, coefs, atol=1e-10)


# ---------------------------------------------------------------------------
# Properties from the paper
# ---------------------------------------------------------------------------
class TestProperties:
    @pytest.mark.parametrize("method", ["lar", "lasso"])
    def test_largest_correlation_equals_penalty_at_every_knot(self, method):
        """KKT check over many random designs (n < p and n > p, correlated columns)."""
        for seed in range(150):
            rng = np.random.default_rng(seed)
            n, p = int(rng.integers(5, 40)), int(rng.integers(2, 60))
            X = rng.normal(size=(n, p))
            if seed % 3 == 0:
                X[:, 1:] += 0.9 * X[:, [0]]
            X, _ = _standardized(X, np.zeros(n))
            y = X[:, : min(3, p)].sum(axis=1) + rng.normal(size=n)
            y = y - y.mean()
            path = lars_path(X, y, method=method)
            corr = np.abs(X.T @ (y[:, None] - X @ path.coefs)).max(axis=0) / n
            np.testing.assert_allclose(corr, path.alphas, atol=1e-8, err_msg=f"seed {seed}")

    @pytest.mark.parametrize("perturbation", [2, 13, 17])
    def test_lasso_kkt_after_rank_saturation(self, perturbation):
        """Regression: centred n = p design (rank n - 1) perturbed at 1e-15.

        The active set reaches rank(X) before the end of the path; the lasso used to keep
        dropping and re-adding predictors there and violated the optimality conditions on
        some BLAS builds. The path now ends with a step to the least-squares fit.
        """
        rng = np.random.default_rng(7)
        n, p = int(rng.integers(5, 40)), int(rng.integers(2, 60))
        X, _ = _standardized(rng.normal(size=(n, p)), np.zeros(n))
        y = X[:, :3].sum(axis=1) + rng.normal(size=n)
        y = y - y.mean()
        X = X + np.random.default_rng(1000 + perturbation).normal(scale=1e-15, size=X.shape)
        path = lars_path(X, y, method="lasso")
        corr = np.abs(X.T @ (y[:, None] - X @ path.coefs)).max(axis=0) / n
        np.testing.assert_allclose(corr, path.alphas, atol=1e-8)
        assert (path.coefs != 0).sum(axis=0).max() <= np.linalg.matrix_rank(X)
        assert path.alphas[-1] == pytest.approx(0.0, abs=1e-12)

    def test_ends_at_least_squares(self):
        X, y = _sparse(0)
        path = lars_path(X, y)
        ols = np.linalg.lstsq(X, y, rcond=None)[0]
        np.testing.assert_allclose(path.coefs[:, -1], ols, atol=1e-10)
        assert path.alphas[-1] == pytest.approx(0.0, abs=1e-12)
        assert path.n_steps == X.shape[1]

    def test_equal_correlations_of_active_set(self):
        """Every active predictor has the same absolute correlation (eq. 2.15)."""
        X, y = _sparse(2)
        path = lars_path(X, y, max_steps=4)
        for step in range(1, path.n_steps):
            c = X.T @ (y - X @ path.coefs[:, step])
            active = path.entry_order[: step + 1]
            np.testing.assert_allclose(np.abs(c[active]), path.alphas[step] * len(y))

    def test_alphas_decrease(self):
        X, y = _sparse(3)
        assert np.all(np.diff(lars_path(X, y).alphas) < 0)

    def test_lasso_coefficients_never_cross_zero(self):
        X, y = _factor_design(4)
        path = lars_path(X, y, method="lasso")
        signs = np.sign(path.coefs)
        for j in range(X.shape[1]):
            nonzero = signs[j][signs[j] != 0]
            assert np.all(nonzero == nonzero[0]) if nonzero.size else True
        dropped = path.drops[0]
        assert dropped in path.entry_order

    def test_lar_and_lasso_agree_without_drops(self):
        X, y = _sparse(1)
        a, b = lars_path(X, y), lars_path(X, y, method="lasso")
        assert not b.drops
        np.testing.assert_allclose(a.coefs, b.coefs)

    def test_max_steps_and_entry_step(self):
        X, y = _sparse(0)
        path = lars_path(X, y, max_steps=2)
        assert path.n_steps == 2
        steps = path.entry_step
        assert np.isnan(steps).sum() == X.shape[1] - 3
        assert steps[path.entry_order[0]] == 1.0
        assert isinstance(path, LarsPath)

    def test_zero_response(self):
        path = lars_path(np.eye(3), np.zeros(3))
        assert path.n_steps == 0 and path.entry_order == []
        assert np.isnan(path.entry_step).all()

    def test_tolerance_stops_exact_fit(self):
        X = np.eye(3)
        path = lars_path(X, np.array([3.0, 0.0, 0.0]))
        assert path.n_steps == 1
        np.testing.assert_allclose(path.coefs[:, -1], [3.0, 0.0, 0.0])

    def test_duplicate_column_never_enters(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(30, 1))
        X = np.hstack([x, x])
        path = lars_path(X, 2 * x[:, 0])
        assert len(path.active) == 1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
class TestEdgeCases:
    def test_collinear_active_set_warns(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(20, 1))
        X = np.hstack([x, x])
        state = _State(beta=np.zeros(2), active=[0, 1], entry_order=[0, 1])
        state.alphas.append(1.0)
        with pytest.warns(ConvergenceWarning, match="collinear"):
            assert _one_step(X, x[:, 0], state, "lar", 2) is False

    def test_drop_step_without_candidates(self):
        assert _drop_step(np.zeros(2), np.ones(2), [0, 1]) == (np.inf, None)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"method": "ridge"}, "method"),
            ({"max_steps": 0}, "max_steps"),
            ({"max_steps": True}, "max_steps"),
            ({"tol": 0.0}, "tol"),
            ({"tol": np.nan}, "tol"),
        ],
    )
    def test_invalid_arguments(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            lars_path(np.eye(3), np.ones(3), **kwargs)

    def test_invalid_data(self):
        with pytest.raises(NowcastDataError, match="2-D"):
            lars_path(np.ones(3), np.ones(3))
        with pytest.raises(NowcastDataError, match="2-D"):
            lars_path(np.ones((3, 2)), np.ones(4))
        with pytest.raises(NowcastDataError, match="finite"):
            lars_path(np.array([[1.0], [np.nan]]), np.ones(2))
