"""Tests of the batched OLS core of BridgeCombination."""

from __future__ import annotations

import doctest

import numpy as np
import pytest

import nowcastbox.models._bridge_batch as batch_module
from nowcastbox.models._bridge_batch import (
    batched_ols,
    design_tensor,
    one_step_errors,
    recursive_predict,
    training_mask,
)


@pytest.fixture
def problem(rng):
    n_eq, n_per, k = 7, 40, 3
    X = rng.standard_normal((n_eq, n_per, k))
    X[:, :, 0] = 1.0
    y = rng.standard_normal(n_per)
    y[[3, 10]] = np.nan
    X[2, :5, 1] = np.nan
    return X, y


def test_batched_ols_matches_lstsq(problem) -> None:
    X, y = problem
    mask = training_mask(X, y)
    fit = batched_ols(X, y, mask)
    assert fit.valid.all()
    for e in range(X.shape[0]):
        rows = mask[e]
        beta, *_ = np.linalg.lstsq(X[e, rows], y[rows], rcond=None)
        np.testing.assert_allclose(fit.beta[e], beta, atol=1e-10)
        resid = y[rows] - X[e, rows] @ beta
        assert fit.ssr[e] == pytest.approx(resid @ resid)
        assert fit.n_obs[e] == rows.sum()
        np.testing.assert_allclose(fit.residuals[e, rows], resid, atol=1e-10)
        assert np.isnan(fit.residuals[e, ~rows]).all()
        centred = y[rows] - y[rows].mean()
        assert fit.tss[e] == pytest.approx(centred @ centred)


def test_training_mask_end(problem) -> None:
    X, y = problem
    mask = training_mask(X, y, end=20)
    assert not mask[:, 20:].any()
    assert mask[0, :20].sum() == 18  # y missing at 3 and 10


def test_invalid_equations() -> None:
    X = np.ones((2, 3, 2))
    X[1, :, 1] = [0.0, 1.0, 2.0]
    fit = batched_ols(X, np.array([1.0, 2.0, 3.0]), np.ones((2, 3), dtype=bool))
    # the first design is collinear; with two periods (< k + 1) nothing is estimable
    assert fit.valid.tolist() == [False, True]
    assert np.isnan(fit.beta[0]).all()
    assert np.isnan(fit.ssr[0])
    short = batched_ols(X[:, :2], np.array([1.0, 2.0]), np.ones((2, 2), dtype=bool))
    assert not short.valid.any()


def test_more_coefficients_than_periods() -> None:
    fit = batched_ols(np.ones((1, 2, 3)), np.ones(2), np.ones((1, 2), dtype=bool))
    assert not fit.valid.any()
    assert np.isnan(fit.beta).all()


def test_design_tensor() -> None:
    features = np.arange(12.0).reshape(4, 3)
    X = design_tensor(features, np.array([[0, 2], [1, 1]]))
    assert X.shape == (2, 4, 2)
    np.testing.assert_array_equal(X[0, :, 1], features[:, 2])


def test_recursive_predict_without_lags(problem) -> None:
    X, y = problem
    beta = np.ones((X.shape[0], X.shape[2]))
    out = recursive_predict(X, beta, y, 0)
    np.testing.assert_allclose(out, X.sum(axis=2))


def test_recursive_predict_iterates() -> None:
    X = np.ones((1, 5, 3))  # const, regressor (=1), one target lag
    observed = np.array([1.0, 2.0, np.nan, np.nan, np.nan])
    beta = np.array([[0.5, 0.0, 0.5]])
    out = recursive_predict(X, beta, observed, 1)[0]
    # 0.5 + 0.5 * y_{t-1}, with predictions replacing the missing lags
    assert np.isnan(out[0])
    assert out[1] == pytest.approx(1.0)
    assert out[2] == pytest.approx(1.5)
    assert out[3] == pytest.approx(1.25)
    assert out[4] == pytest.approx(1.125)


def test_one_step_errors_match_loop(problem) -> None:
    X, y = problem
    positions = np.flatnonzero(np.isfinite(y))[-5:]
    errors = one_step_errors(X, y, positions, min_train=8)
    for e in range(X.shape[0]):
        for i, s in enumerate(positions):
            rows = training_mask(X[e : e + 1], y, end=int(s))[0]
            beta, *_ = np.linalg.lstsq(X[e, rows], y[rows], rcond=None)
            assert errors[e, i] == pytest.approx(y[s] - X[e, s] @ beta)


def test_one_step_errors_min_train(problem) -> None:
    X, y = problem
    errors = one_step_errors(X, y, np.array([5, 39]), min_train=20)
    assert np.isnan(errors[:, 0]).all()
    assert np.isfinite(errors[:, 1]).all()


def test_doctests() -> None:
    result = doctest.testmod(batch_module, optionflags=doctest.ELLIPSIS)
    assert result.attempted > 0
    assert result.failed == 0
