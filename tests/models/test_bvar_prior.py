"""Tests of the NIW prior machinery of the large BVAR (``models/_bvar_prior.py``).

Validation (plan, item 11): (a) closed-form marginal likelihood vs brute-force numerical
integration and vs the matrix-variate t density; (b) posterior vs the conjugate formulas
from Bayes' rule; (c) hierarchical lambda on simulated VARs and Monte Carlo moments of
posterior draws; (d) dummy observations reproduce the stated prior moments.
"""

from __future__ import annotations

import doctest
import warnings
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
import scipy.integrate
import scipy.stats

import nowcastbox.models._bvar_prior as bp
from nowcastbox.core.exceptions import ConvergenceWarning, NowcastDataError
from nowcastbox.models._bvar_prior import (
    BVARHyperparameters,
    GammaHyperprior,
    GLPHyperpriors,
    HyperGradient,
    InverseGammaHyperprior,
    NIWPrior,
    PriorSettings,
    VARSystem,
    dummy_initial_observation_dummies,
    implied_prior_from_dummies,
    log_hyperprior,
    log_hyperprior_gradient,
    log_marginal_likelihood,
    log_marginal_likelihood_gradient,
    minnesota_dummies,
    minnesota_prior,
    niw_posterior,
    numerical_hessian,
    posterior,
    prior_dummies,
    select_hyperparameters,
    sum_of_coefficients_dummies,
    var_design,
)

FLAT = GLPHyperpriors(lambda_=None, mu=None, delta=None, psi=None)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def simulate_var(rng: np.random.Generator, n: int, T: int, rho: float = 0.5) -> np.ndarray:
    """Diagonal VAR(1) with unit innovations (after a burn-in)."""
    y = np.zeros((T + 50, n))
    for t in range(1, T + 50):
        y[t] = rho * y[t - 1] + rng.standard_normal(n)
    return y[50:]


def simulate_factor(rng: np.random.Generator, n: int, T: int) -> np.ndarray:
    """One AR(1) factor plus idiosyncratic noise: strongly collinear large systems."""
    f = np.zeros(T + 50)
    for t in range(1, T + 50):
        f[t] = 0.8 * f[t - 1] + rng.standard_normal()
    loadings = rng.uniform(0.5, 1.5, n)
    return (np.outer(f, loadings) + rng.standard_normal((T + 50, n)))[50:]


def matrix_t_log_density(
    Y: np.ndarray,
    X: np.ndarray,
    b: np.ndarray,
    omega_full: np.ndarray,
    scale: np.ndarray,
    dof: float,
) -> float:
    """log p(Y) from the matrix-variate t marginal, computed with T x T matrices."""
    T, n = Y.shape
    V = np.eye(T) + X @ omega_full @ X.T
    R = Y - X @ b
    inner = scale + R.T @ np.linalg.solve(V, R)
    return float(
        -n * T / 2 * np.log(np.pi)
        + bp._multigammaln((dof + T) / 2, n)  # checked against scipy below
        - bp._multigammaln(dof / 2, n)
        - n / 2 * np.linalg.slogdet(V)[1]
        + dof / 2 * np.linalg.slogdet(scale)[1]
        - (dof + T) / 2 * np.linalg.slogdet(inner)[1]
    )


def explicit_posterior(Y: np.ndarray, X: np.ndarray, prior: NIWPrior):
    """Conjugate update written literally (explicit inverses) from Bayes' rule."""
    omega_inv = np.diag(1.0 / prior.omega)
    omega_bar = np.linalg.inv(X.T @ X + omega_inv)
    b_bar = omega_bar @ (X.T @ Y + omega_inv @ prior.mean)
    scale_bar = (
        prior.scale
        + Y.T @ Y
        + prior.mean.T @ omega_inv @ prior.mean
        - b_bar.T @ np.linalg.inv(omega_bar) @ b_bar
    )
    return b_bar, omega_bar, scale_bar, prior.dof + Y.shape[0]


@pytest.fixture
def system() -> VARSystem:
    rng = np.random.default_rng(11)
    return VARSystem.from_array(np.cumsum(rng.standard_normal((50, 3)), axis=0), lags=2)


@pytest.fixture
def hyper(system: VARSystem) -> BVARHyperparameters:
    return BVARHyperparameters(0.3, system.ar_residual_variances(), mu=0.8, delta=1.5)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def test_var_design_layout() -> None:
    y = np.arange(12.0).reshape(6, 2)
    Y, X = var_design(y, 2)
    assert Y.shape == (4, 2) and X.shape == (4, 5)
    np.testing.assert_array_equal(Y, y[2:])
    np.testing.assert_array_equal(X[:, 0], 1.0)
    np.testing.assert_array_equal(X[:, 1:3], y[1:5])  # lag 1
    np.testing.assert_array_equal(X[:, 3:5], y[0:4])  # lag 2
    _, X0 = var_design(y, 2, constant=False)
    np.testing.assert_array_equal(X0, X[:, 1:])


@pytest.mark.parametrize(
    ("y", "lags", "exc"),
    [
        (np.ones((5, 2)), 0, ValueError),
        (np.ones(5), 1, NowcastDataError),
        (np.array([[1.0], [np.nan], [2.0]]), 1, NowcastDataError),
        (np.ones((2, 2)), 2, NowcastDataError),
    ],
)
def test_var_design_errors(y: np.ndarray, lags: int, exc: type[Exception]) -> None:
    with pytest.raises(exc):
        var_design(y, lags)


def test_system_properties(system: VARSystem) -> None:
    assert (system.n, system.n_obs, system.k, system.lags) == (3, 48, 7, 2)
    assert system.lag_column(1, 0) == 1 and system.lag_column(2, 2) == 6
    no_const = VARSystem.from_array(np.ones((5, 2)), lags=1, constant=False)
    assert no_const.lag_column(1, 1) == 1


def test_ar_residual_variances_match_ols(system: VARSystem) -> None:
    y = system.Y[:, 1]
    X = np.column_stack([np.ones(system.n_obs), system.X[:, system.lag_column(1, 1)]])
    resid = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    expected = resid @ resid / (system.n_obs - 2)
    assert system.ar_residual_variances(1)[1] == pytest.approx(expected)
    assert system.ar_residual_variances().shape == (3,)
    with pytest.raises(ValueError, match="lags must be in"):
        system.ar_residual_variances(3)


def test_ar_residual_variances_floor_and_no_constant() -> None:
    flat = VARSystem.from_array(np.full((10, 1), 2.0), lags=1)
    assert flat.ar_residual_variances()[0] > 0
    rng = np.random.default_rng(0)
    nc = VARSystem.from_array(rng.standard_normal((80, 2)), lags=1, constant=False)
    assert np.all(nc.ar_residual_variances() > 0)


# ---------------------------------------------------------------------------
# prior settings and hyperparameters
# ---------------------------------------------------------------------------
def test_prior_settings() -> None:
    assert PriorSettings(prior_mean=0.5).own_lag_mean(2).tolist() == [0.5, 0.5]
    assert PriorSettings(prior_mean=(1.0, 0.0)).own_lag_mean(2).tolist() == [1.0, 0.0]
    with pytest.raises(ValueError, match="variables"):
        PriorSettings(prior_mean=[1.0, 0.0]).own_lag_mean(3)
    with pytest.raises(ValueError, match="prior_mean"):
        PriorSettings(prior_mean="ar")
    with pytest.raises(ValueError, match="positive"):
        PriorSettings(lag_decay=0.0)
    with pytest.raises(ValueError, match="positive"):
        PriorSettings(intercept_variance=-1.0)
    with pytest.raises(ValueError, match="dof"):
        PriorSettings(dof=3.0).degrees_of_freedom(2)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lambda_": 0.0, "psi": [1.0]},
        {"lambda_": 0.2, "psi": [1.0, -1.0]},
        {"lambda_": 0.2, "psi": [1.0], "mu": 0.0},
        {"lambda_": 0.2, "psi": [1.0], "delta": np.nan},
    ],
)
def test_hyperparameters_validation(kwargs: dict) -> None:
    with pytest.raises(ValueError, match="positive"):
        BVARHyperparameters(**kwargs)


def test_niw_prior_sigma_mean_infinite() -> None:
    prior = NIWPrior(np.zeros((1, 2)), np.ones(1), np.eye(2), 3.0)
    assert np.all(np.isinf(prior.sigma_mean))


# ---------------------------------------------------------------------------
# (d) prior construction and dummy observations
# ---------------------------------------------------------------------------
def test_minnesota_prior_moments(system: VARSystem) -> None:
    psi = np.array([1.0, 2.0, 4.0])
    settings = PriorSettings(prior_mean=[1.0, 0.0, 0.5], lag_decay=1.5, dof=8.0)
    prior = minnesota_prior(system, BVARHyperparameters(0.3, psi), settings)
    n, d = 3, 8.0
    assert prior.dof == d and prior.omega[0] == settings.intercept_variance
    for lag in (1, 2):
        for j in range(n):
            expected = 0.3**2 * (d - n - 1) / (lag**1.5 * psi[j])
            assert prior.omega[system.lag_column(lag, j)] == pytest.approx(expected)
    # Var(B_{l,ji} | Sigma) = lambda^2 Sigma_ii / (l^kappa E[Sigma_jj])
    sigma_mean = prior.sigma_mean
    for j in range(n):
        var_coef = prior.omega[system.lag_column(2, j)] * sigma_mean[0, 0]
        target = 0.3**2 * sigma_mean[0, 0] / (2**1.5 * sigma_mean[j, j])
        assert var_coef == pytest.approx(target)
    expected_mean = np.zeros((system.k, n))
    expected_mean[[1, 2, 3], [0, 1, 2]] = [1.0, 0.0, 0.5]
    np.testing.assert_array_equal(prior.mean, expected_mean)
    np.testing.assert_array_equal(prior.scale, np.diag(psi))
    with pytest.raises(ValueError, match="psi has"):
        minnesota_prior(system, BVARHyperparameters(0.3, np.array([1.0])))


@pytest.mark.parametrize("constant", [True, False])
@pytest.mark.parametrize("dof", [None, 9.0])
def test_minnesota_dummies_reproduce_prior(constant: bool, dof: float | None) -> None:
    rng = np.random.default_rng(1)
    s = VARSystem.from_array(rng.standard_normal((30, 3)), lags=3, constant=constant)
    h = BVARHyperparameters(0.25, np.array([0.5, 1.0, 3.0]))
    settings = PriorSettings(prior_mean=[1.0, 0.2, 0.0], dof=dof)
    prior = minnesota_prior(s, h, settings)
    mean, omega, scale = implied_prior_from_dummies(*minnesota_dummies(s, h, settings))
    np.testing.assert_allclose(mean, prior.mean, atol=1e-10)
    np.testing.assert_allclose(omega, np.diag(prior.omega), rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(scale, prior.scale, atol=1e-10)


def test_soc_and_dio_rows(system: VARSystem) -> None:
    Yd, Xd = sum_of_coefficients_dummies(system, 0.5)
    np.testing.assert_allclose(Yd, np.diag(system.y0_mean) / 0.5)
    np.testing.assert_array_equal(Xd[:, 0], 0.0)
    np.testing.assert_allclose(Xd[:, 1:4], Yd)
    np.testing.assert_allclose(Xd[:, 4:7], Yd)
    yd, xd = dummy_initial_observation_dummies(system, 2.0)
    assert xd[0, 0] == 0.5
    np.testing.assert_allclose(xd[0, 1:4], system.y0_mean / 2.0)
    np.testing.assert_allclose(yd[0], system.y0_mean / 2.0)
    nc = VARSystem.from_array(np.arange(8.0).reshape(4, 2), lags=1, constant=False)
    assert sum_of_coefficients_dummies(nc, 1.0)[1].shape == (2, 2)
    assert dummy_initial_observation_dummies(nc, 1.0)[1].shape == (1, 2)


def test_prior_dummies_combinations(system: VARSystem) -> None:
    base = BVARHyperparameters(0.2, np.ones(3))
    assert prior_dummies(system, base)[0].shape == (0, 3)
    assert prior_dummies(system, replace(base, mu=1.0))[0].shape == (3, 3)
    assert prior_dummies(system, replace(base, delta=1.0))[1].shape == (1, 7)
    assert prior_dummies(system, replace(base, mu=1.0, delta=1.0))[0].shape == (4, 3)


def test_soc_dummy_encodes_its_linear_restriction(system: VARSystem) -> None:
    """Each SoC row adds (ybar_i/mu)^2 (sum over lags) to the precision of B."""
    h = BVARHyperparameters(0.3, system.ar_residual_variances(), mu=0.7)
    post = posterior(system, h)
    prior = minnesota_prior(system, h)
    _, Xd = sum_of_coefficients_dummies(system, 0.7)
    precision = np.diag(1.0 / prior.omega) + system.X.T @ system.X + Xd.T @ Xd
    np.testing.assert_allclose(
        np.linalg.inv(post.omega), precision, rtol=1e-6, atol=1e-6 * np.abs(precision).max()
    )


def test_tight_soc_and_dio_impose_unit_roots() -> None:
    rng = np.random.default_rng(5)
    s = VARSystem.from_array(simulate_var(rng, 3, 80, rho=0.3) + 5.0, lags=2)
    psi = s.ar_residual_variances()
    post = posterior(s, BVARHyperparameters(10.0, psi, mu=1e-5))
    B = post.mean
    lag_sum = B[1:4] + B[4:7]  # sum of lag matrices (rows = regressors)
    np.testing.assert_allclose(lag_sum, np.eye(3), atol=1e-3)
    post = posterior(s, BVARHyperparameters(10.0, psi, delta=1e-5))
    B = post.mean
    implied = B[0] + s.y0_mean @ (B[1:4] + B[4:7])
    np.testing.assert_allclose(implied, s.y0_mean, rtol=1e-4)


# ---------------------------------------------------------------------------
# (b) posterior vs Bayes' rule
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n_obs", [60, 6])  # primal (T >= k) and push-through (T < k)
def test_posterior_matches_explicit_formulas(n_obs: int) -> None:
    rng = np.random.default_rng(2)
    s = VARSystem.from_array(np.cumsum(rng.standard_normal((n_obs + 2, 3)), 0), lags=2)
    h = BVARHyperparameters(0.4, s.ar_residual_variances() + 0.1)
    prior = minnesota_prior(s, h, PriorSettings(intercept_variance=10.0))
    post = niw_posterior(s.Y, s.X, prior)
    b_bar, omega_bar, scale_bar, dof = explicit_posterior(s.Y, s.X, prior)
    np.testing.assert_allclose(post.mean, b_bar, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(post.omega, omega_bar, rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(post.scale, scale_bar, rtol=1e-8)
    assert post.dof == dof
    np.testing.assert_allclose(post.sigma_mean, scale_bar / (dof - 4))


def test_posterior_with_dummies_is_stacked_update(
    system: VARSystem, hyper: BVARHyperparameters
) -> None:
    post = posterior(system, hyper)
    Yd, Xd = prior_dummies(system, hyper)
    prior = minnesota_prior(system, hyper)
    stacked = niw_posterior(np.vstack([Yd, system.Y]), np.vstack([Xd, system.X]), prior)
    np.testing.assert_allclose(post.mean, stacked.mean)
    assert post.dof == prior.dof + system.n_obs + 4


@pytest.mark.slow
def test_posterior_mean_by_numerical_integration() -> None:
    """n = 1, k = 1: E[beta | Y] and E[sigma^2 | Y] by quadrature of prior x likelihood."""
    rng = np.random.default_rng(4)
    y = np.zeros(12)
    for t in range(1, 12):
        y[t] = 0.7 * y[t - 1] + rng.standard_normal()
    s = VARSystem.from_array(y[:, None], 1, constant=False)
    h = BVARHyperparameters(0.4, np.array([0.8]))
    prior = minnesota_prior(s, h)
    Y, X = s.Y[:, 0], s.X[:, 0]
    b0, om, psi, d = prior.mean[0, 0], prior.omega[0], prior.scale[0, 0], prior.dof

    def kernel(beta: float, sig2: float) -> float:
        log_lik = scipy.stats.norm.logpdf(Y, X * beta, np.sqrt(sig2)).sum()
        log_prior = scipy.stats.norm.logpdf(beta, b0, np.sqrt(sig2 * om))
        log_prior += scipy.stats.invgamma.logpdf(sig2, d / 2, scale=psi / 2)
        return float(np.exp(log_lik + log_prior + 15.0))

    opts = {"epsabs": 1e-12, "epsrel": 1e-9}
    z = scipy.integrate.dblquad(kernel, 1e-6, 40, -4, 6, **opts)[0]
    m_beta = scipy.integrate.dblquad(lambda b, v: b * kernel(b, v), 1e-6, 40, -4, 6, **opts)[0]
    m_sig = scipy.integrate.dblquad(lambda b, v: v * kernel(b, v), 1e-6, 40, -4, 6, **opts)[0]
    post = posterior(s, h)
    assert post.mean[0, 0] == pytest.approx(m_beta / z, rel=1e-6)
    assert post.sigma_mean[0, 0] == pytest.approx(m_sig / z, rel=1e-5)


# ---------------------------------------------------------------------------
# (a) marginal likelihood
# ---------------------------------------------------------------------------
def test_marginal_likelihood_brute_force_integration() -> None:
    rng = np.random.default_rng(3)
    y = np.zeros(9)
    for t in range(1, 9):
        y[t] = 0.6 * y[t - 1] + rng.standard_normal()
    s = VARSystem.from_array(y[:, None], 1, constant=False)
    h = BVARHyperparameters(0.5, np.array([1.3]))
    prior = minnesota_prior(s, h)
    Y, X = s.Y[:, 0], s.X[:, 0]
    b0, om, psi, d = prior.mean[0, 0], prior.omega[0], prior.scale[0, 0], prior.dof

    def joint(beta: float, sig2: float) -> float:
        value = scipy.stats.norm.logpdf(Y, X * beta, np.sqrt(sig2)).sum()
        value += scipy.stats.norm.logpdf(beta, b0, np.sqrt(sig2 * om))
        value += scipy.stats.invgamma.logpdf(sig2, d / 2, scale=psi / 2)
        return float(np.exp(value))

    integral = scipy.integrate.dblquad(joint, 1e-6, 50, -6, 8, epsabs=1e-14, epsrel=1e-10)[0]
    assert log_marginal_likelihood(s, h) == pytest.approx(np.log(integral), abs=1e-6)


@pytest.mark.parametrize(("n_obs", "constant"), [(40, True), (5, True), (30, False)])
def test_marginal_likelihood_matches_matrix_t(n_obs: int, constant: bool) -> None:
    rng = np.random.default_rng(7)
    s = VARSystem.from_array(
        np.cumsum(rng.standard_normal((n_obs + 2, 3)), 0), lags=2, constant=constant
    )
    h = BVARHyperparameters(0.3, s.ar_residual_variances())
    prior = minnesota_prior(s, h, PriorSettings(intercept_variance=100.0, dof=7.0))
    expected = matrix_t_log_density(
        s.Y, s.X, prior.mean, np.diag(prior.omega), prior.scale, prior.dof
    )
    got = log_marginal_likelihood(s, h, PriorSettings(intercept_variance=100.0, dof=7.0))
    assert got == pytest.approx(expected, rel=1e-10)


def test_dummy_ratio_equals_marginal_under_updated_prior(
    system: VARSystem, hyper: BVARHyperparameters
) -> None:
    """p(Y | Y+) is the matrix-t density of Y under the NIW prior updated by the dummies."""
    prior = minnesota_prior(system, hyper)
    Yd, Xd = prior_dummies(system, hyper)
    updated = niw_posterior(Yd, Xd, prior)
    expected = matrix_t_log_density(
        system.Y, system.X, updated.mean, updated.omega, updated.scale, updated.dof
    )
    assert log_marginal_likelihood(system, hyper) == pytest.approx(expected, rel=1e-8)


def _levels_system(seed: int, n: int, n_obs: int, lags: int) -> VARSystem:
    """Data in 100 x log levels (around 460) driven by a common trend: the realistic case."""
    rng = np.random.default_rng(seed)
    trend = np.cumsum(rng.standard_normal(n_obs + lags))
    noise = 0.3 * np.cumsum(rng.standard_normal((n_obs + lags, n)), 0)
    return VARSystem.from_array(460.0 + np.outer(trend, rng.standard_normal(n)) + noise, lags)


def _mp_matrix_t(Y: np.ndarray, X: np.ndarray, prior: NIWPrior) -> float:
    """Matrix-t log density in 60-digit arithmetic (reference without rounding issues)."""
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = 60
    T, n = Y.shape
    Xm, Ym, bm = mp.matrix(X.tolist()), mp.matrix(Y.tolist()), mp.matrix(prior.mean.tolist())
    V = mp.eye(T) + Xm * mp.diag([mp.mpf(v) for v in prior.omega]) * Xm.T
    R = Ym - Xm * bm
    scale = mp.matrix(prior.scale.tolist())
    inner = scale + R.T * mp.inverse(V) * R
    d = mp.mpf(prior.dof)

    def lgam(a: Any) -> Any:
        return sum(mp.loggamma(a - mp.mpf(j) / 2) for j in range(n))

    value = (
        -mp.mpf(n * T) / 2 * mp.log(mp.pi)
        + lgam((d + T) / 2)
        - lgam(d / 2)
        - mp.mpf(n) / 2 * mp.log(mp.det(V))
        + d / 2 * mp.log(mp.det(scale))
        - (d + T) / 2 * mp.log(mp.det(inner))
    )
    return float(value)


@pytest.mark.parametrize(("lambda_", "mu"), [(1.0, 1e-4), (10.0, 1e-2), (0.2, 1.0)])
def test_marginal_likelihood_accurate_for_data_in_levels(lambda_: float, mu: float) -> None:
    """Large dummy rows (levels ~ 460, tight SoC/DIO) must not destroy the accuracy.

    Forming X'X here squares a condition number of ~1e9 and the log ML is off by hundreds;
    the reference is the matrix-t density evaluated in 60-digit arithmetic.
    """
    s = _levels_system(7, 3, 25, 2)
    h = BVARHyperparameters(lambda_, s.ar_residual_variances(1), mu=mu, delta=mu)
    prior = minnesota_prior(s, h)
    Yd, Xd = prior_dummies(s, h)
    expected = _mp_matrix_t(np.vstack([Yd, s.Y]), np.vstack([Xd, s.X]), prior) - _mp_matrix_t(
        Yd, Xd, prior
    )
    assert log_marginal_likelihood(s, h) == pytest.approx(expected, rel=1e-10)


@pytest.mark.parametrize(("lambda_", "mu"), [(1e-3, 1e-4), (1.0, 1e-3), (10.0, 1.0)])
def test_gradient_accurate_for_large_system_in_levels(lambda_: float, mu: float) -> None:
    """k > T, levels data, extreme hyperparameters: analytic gradient = finite differences."""
    s = _levels_system(5, 20, 40, 4)
    h = BVARHyperparameters(lambda_, s.ar_residual_variances(1), mu=mu, delta=mu)

    def f(x: BVARHyperparameters) -> float:
        return log_marginal_likelihood(s, x)

    _, grad = log_marginal_likelihood_gradient(s, h)
    assert grad.lambda_ == pytest.approx(_fd(f, h, "lambda_"), rel=1e-5, abs=1e-4)
    assert grad.mu == pytest.approx(_fd(f, h, "mu"), rel=1e-5, abs=1e-4)
    assert grad.delta == pytest.approx(_fd(f, h, "delta"), rel=1e-5, abs=1e-4)
    assert grad.psi[3] == pytest.approx(_fd(f, h, "psi", 3), rel=1e-5, abs=1e-4)


def test_non_positive_definite_scale_raises() -> None:
    prior = NIWPrior(np.zeros((1, 2)), np.ones(1), -np.eye(2), 5.0)
    with pytest.raises(NowcastDataError, match="positive definite"):
        niw_posterior(np.ones((3, 2)), np.ones((3, 1)), prior)


# ---------------------------------------------------------------------------
# gradient
# ---------------------------------------------------------------------------
def _fd(
    f: Callable[[BVARHyperparameters], float],
    h: BVARHyperparameters,
    field: str,
    j: int | None = None,
    eps: float = 1e-5,
):
    def shifted(sign: float) -> float:
        if field == "psi":
            psi = h.psi.copy()
            psi[j] *= np.exp(sign * eps)
            return f(replace(h, psi=psi))
        return f(replace(h, **{field: getattr(h, field) * np.exp(sign * eps)}))

    return (shifted(1.0) - shifted(-1.0)) / (2 * eps)


@pytest.mark.parametrize(
    ("n_obs", "n", "lags", "constant"), [(60, 3, 2, True), (40, 2, 2, False), (25, 4, 3, True)]
)
@pytest.mark.parametrize("dummies", [(None, None), (0.7, None), (None, 1.4), (0.7, 1.4)])
def test_gradient_matches_finite_differences(
    n_obs: int, n: int, lags: int, constant: bool, dummies: tuple[float | None, float | None]
) -> None:
    rng = np.random.default_rng(n_obs)
    s = VARSystem.from_array(
        np.cumsum(rng.standard_normal((n_obs, n)), 0), lags=lags, constant=constant
    )
    mu, delta = dummies
    h = BVARHyperparameters(0.3, s.ar_residual_variances() * rng.uniform(0.5, 2, n), mu, delta)
    settings = PriorSettings(prior_mean=0.9, intercept_variance=100.0)

    def f(x: BVARHyperparameters) -> float:
        return log_marginal_likelihood(s, x, settings)

    value, grad = log_marginal_likelihood_gradient(s, h, settings)
    assert value == pytest.approx(f(h))
    assert grad.lambda_ == pytest.approx(_fd(f, h, "lambda_"), rel=1e-5, abs=1e-5)
    for j in range(n):
        assert grad.psi[j] == pytest.approx(_fd(f, h, "psi", j), rel=1e-5, abs=1e-5)
    for name in ("mu", "delta"):
        if getattr(h, name) is None:
            assert getattr(grad, name) is None
        else:
            assert getattr(grad, name) == pytest.approx(_fd(f, h, name), rel=1e-5, abs=1e-5)


def test_hyper_gradient_addition() -> None:
    a = HyperGradient(1.0, np.array([1.0, 2.0]), mu=None, delta=3.0)
    b = HyperGradient(0.5, np.array([1.0, 0.0]), mu=2.0, delta=None)
    c = a + b
    assert (c.lambda_, c.mu, c.delta) == (1.5, 2.0, 3.0)
    assert c.psi.tolist() == [2.0, 2.0]
    assert (a + a).mu is None


# ---------------------------------------------------------------------------
# hyperpriors
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(("mode", "sd"), [(0.2, 0.4), (1.0, 1.0), (0.0, 0.5)])
def test_gamma_hyperprior_mode_and_sd(mode: float, sd: float) -> None:
    g = GammaHyperprior(mode, sd)
    dist: Any = scipy.stats.gamma(g.shape, scale=g.scale)
    assert dist.std() == pytest.approx(sd)
    assert (g.shape - 1) * g.scale == pytest.approx(mode)
    for x in (0.05, 0.3, 2.0):
        assert g.logpdf(x) == pytest.approx(dist.logpdf(x))
        eps = 1e-6
        fd = (g.logpdf(x * np.exp(eps)) - g.logpdf(x * np.exp(-eps))) / (2 * eps)
        assert g.dlogpdf_dlog(x) == pytest.approx(fd, rel=1e-6, abs=1e-8)


def test_inverse_gamma_hyperprior() -> None:
    ig = InverseGammaHyperprior(0.02**2, 0.02**2)
    for x in (0.01, 1.0, 5.0):
        assert ig.logpdf(x) == pytest.approx(scipy.stats.invgamma.logpdf(x, 0.0004, scale=0.0004))
        eps = 1e-6
        fd = (ig.logpdf(x * np.exp(eps)) - ig.logpdf(x * np.exp(-eps))) / (2 * eps)
        assert ig.dlogpdf_dlog(x) == pytest.approx(fd, rel=1e-6)


def test_log_hyperprior_and_gradient() -> None:
    h = BVARHyperparameters(0.3, np.array([1.0, 2.0]), mu=0.5, delta=2.0)
    hp = GLPHyperpriors()
    assert hp.lambda_ is not None and hp.mu is not None and hp.delta is not None
    assert hp.psi is not None
    expected = (
        hp.lambda_.logpdf(0.3)
        + hp.mu.logpdf(0.5)
        + hp.delta.logpdf(2.0)
        + hp.psi.logpdf(1.0)
        + hp.psi.logpdf(2.0)
    )
    assert log_hyperprior(h) == pytest.approx(expected)
    assert log_hyperprior(h, FLAT) == 0.0
    grad = log_hyperprior_gradient(h)
    assert grad.mu == pytest.approx(hp.mu.dlogpdf_dlog(0.5))
    assert grad.psi[1] == pytest.approx(hp.psi.dlogpdf_dlog(2.0))
    flat_grad = log_hyperprior_gradient(h, FLAT)
    assert (flat_grad.lambda_, flat_grad.mu, flat_grad.delta) == (0.0, 0.0, 0.0)
    assert flat_grad.psi.tolist() == [0.0, 0.0]
    off = log_hyperprior_gradient(BVARHyperparameters(0.3, np.array([1.0])))
    assert off.mu is None and off.delta is None


# ---------------------------------------------------------------------------
# (c) hyperparameter selection
# ---------------------------------------------------------------------------
def test_selection_matches_grid_search() -> None:
    rng = np.random.default_rng(8)
    s = VARSystem.from_array(simulate_var(rng, 4, 120), lags=2)
    settings = PriorSettings("white_noise")
    fixed = BVARHyperparameters(0.2, s.ar_residual_variances(1))
    sel = select_hyperparameters(s, settings, estimate=["lambda"], start=fixed)
    grid = np.exp(np.linspace(np.log(0.02), np.log(3.0), 400))
    objective = [
        log_marginal_likelihood(s, replace(fixed, lambda_=g), settings)
        + log_hyperprior(replace(fixed, lambda_=g))
        for g in grid
    ]
    best = grid[int(np.argmax(objective))]
    assert sel.success
    assert sel.hyperparameters.lambda_ == pytest.approx(best, rel=0.02)
    assert sel.log_posterior == pytest.approx(max(objective), abs=1e-3)
    assert sel.hyperparameters.psi.tolist() == fixed.psi.tolist()
    # inverse Hessian equals 1 / curvature of the negative objective in log lambda
    z, eps = np.log(sel.hyperparameters.lambda_), 1e-3

    def neg(zz: float) -> float:
        hz = replace(fixed, lambda_=float(np.exp(zz)))
        return -(log_marginal_likelihood(s, hz, settings) + log_hyperprior(hz))

    curvature = (neg(z + eps) - 2 * neg(z) + neg(z - eps)) / eps**2
    assert sel.inverse_hessian is not None
    assert sel.inverse_hessian[0, 0] == pytest.approx(1 / curvature, rel=1e-3)
    se = sel.standard_errors
    assert se is not None and se[0] == pytest.approx(np.sqrt(1 / curvature), rel=1e-3)


def test_larger_systems_get_tighter_lambda() -> None:
    """GLP (2015) / Bańbura et al. (2010): more (collinear) variables -> more shrinkage."""
    small, large = [], []
    for seed in range(3):
        rng = np.random.default_rng(seed)
        data = simulate_factor(rng, 20, 200)
        for n, out in ((3, small), (20, large)):
            s = VARSystem.from_array(data[:, :n], lags=2)
            sel = select_hyperparameters(
                s, PriorSettings("white_noise"), estimate=("lambda",), hessian=False,
                sum_of_coefficients=False, initial_observation=False,
            )  # fmt: skip
            out.append(sel.hyperparameters.lambda_)
    assert np.mean(large) < 0.7 * np.mean(small)


def test_noisier_data_get_tighter_lambda() -> None:
    rng = np.random.default_rng(0)
    signal = simulate_var(rng, 5, 150, rho=0.8)
    lambdas = []
    for noise in (0.0, 1.0, 3.0):
        y = signal + noise * np.random.default_rng(1).standard_normal(signal.shape)
        s = VARSystem.from_array(y, lags=2)
        sel = select_hyperparameters(
            s, PriorSettings("white_noise"), hessian=False,
            sum_of_coefficients=False, initial_observation=False,
        )  # fmt: skip
        lambdas.append(sel.hyperparameters.lambda_)
    assert lambdas[0] > lambdas[1] > lambdas[2]


def test_full_glp_selection_with_dummies() -> None:
    rng = np.random.default_rng(9)
    y = np.cumsum(simulate_var(rng, 3, 100, rho=0.3), axis=0)  # I(1) data
    s = VARSystem.from_array(y, lags=2)
    sel = select_hyperparameters(s)
    assert sel.names == ("lambda", "mu", "delta", "psi[0]", "psi[1]", "psi[2]")
    assert sel.success and sel.x.shape == (6,)
    assert sel.inverse_hessian is not None and sel.inverse_hessian.shape == (6, 6)
    assert np.all(np.linalg.eigvalsh(sel.inverse_hessian) > 0)
    h = sel.hyperparameters
    assert 0.01 < h.lambda_ < 5
    assert sel.log_ml == pytest.approx(log_marginal_likelihood(s, h))
    assert sel.log_posterior == pytest.approx(sel.log_ml + log_hyperprior(h), abs=1e-8)
    assert sel.n_evaluations > 1
    # local optimality: no coordinate move increases the log posterior
    for i in range(6):
        for sign in (-1, 1):
            z = sel.x.copy()
            z[i] += sign * 0.05
            hz = bp._Packer(h, ("lambda", "mu", "delta", "psi")).unpack(z)
            assert log_marginal_likelihood(s, hz) + log_hyperprior(hz) <= sel.log_posterior


def test_selection_argument_errors(system: VARSystem) -> None:
    with pytest.raises(ValueError, match="unknown"):
        select_hyperparameters(system, estimate=["tau"])
    with pytest.raises(ValueError, match="at least one"):
        select_hyperparameters(system, estimate=[])
    with pytest.raises(ValueError, match="switched off"):
        select_hyperparameters(system, estimate=["mu"], sum_of_coefficients=False)
    single = select_hyperparameters(system, estimate="lambda", hessian=False)  # bare string
    assert single.names == ("lambda",)


def test_selection_warns_without_convergence(system: VARSystem) -> None:
    with pytest.warns(ConvergenceWarning):
        sel = select_hyperparameters(system, max_iter=1, hessian=False)
    assert not sel.success
    assert sel.standard_errors is None


def test_objective_penalises_failures(monkeypatch: pytest.MonkeyPatch, system: VARSystem) -> None:
    start = BVARHyperparameters(0.2, np.ones(3))
    packer = bp._Packer(start, ("lambda",))
    fun = bp._objective(system, packer, PriorSettings(), FLAT, [0])

    def broken(*args: object, **kwargs: object) -> None:
        raise NowcastDataError("boom")

    monkeypatch.setattr(bp, "log_marginal_likelihood_gradient", broken)
    value, grad = fun(np.zeros(1))
    assert value == bp._PENALTY and grad.tolist() == [0.0]
    monkeypatch.setattr(
        bp,
        "log_marginal_likelihood_gradient",
        lambda *a, **k: (np.nan, HyperGradient(0.0, start.psi)),
    )
    assert fun(np.zeros(1))[0] == bp._PENALTY


def test_numerical_hessian_and_safe_inverse() -> None:
    A = np.array([[4.0, 1.0], [1.0, 3.0]])
    H = numerical_hessian(lambda x: A @ x, np.array([0.3, -0.2]))
    np.testing.assert_allclose(H, A, atol=1e-8)
    np.testing.assert_allclose(bp._safe_inverse(A), np.linalg.inv(A))
    indefinite = np.array([[1.0, 0.0], [0.0, -1.0]])
    inv = bp._safe_inverse(indefinite)
    assert np.all(np.linalg.eigvalsh(inv) > 0)


# ---------------------------------------------------------------------------
# (c) posterior simulation
# ---------------------------------------------------------------------------
def test_posterior_draws_monte_carlo_moments() -> None:
    rng = np.random.default_rng(10)
    s = VARSystem.from_array(simulate_var(rng, 2, 30), lags=1)
    post = posterior(s, BVARHyperparameters(0.3, s.ar_residual_variances(), mu=1.0))
    m = 100_000
    B, S = post.draw(m, rng=123)
    n = post.n
    nu = post.dof
    sigma_mean = post.scale / (nu - n - 1)
    np.testing.assert_allclose(S.mean(axis=0), sigma_mean, rtol=0.02, atol=0.002)
    var_s11 = 2 * post.scale[0, 0] ** 2 / ((nu - n - 1) ** 2 * (nu - n - 3))
    assert S[:, 0, 0].var() == pytest.approx(var_s11, rel=0.05)
    sd_b = np.sqrt(np.outer(np.diag(post.omega), np.diag(sigma_mean)))
    np.testing.assert_allclose((B.mean(axis=0) - post.mean) / sd_b, 0.0, atol=0.02)
    # Cov(vec B) = E[Sigma] (x) Omega_bar: check one within- and one cross-equation entry
    cov = np.cov(B[:, 1, 0], B[:, 2, 1])
    assert cov[0, 0] == pytest.approx(sigma_mean[0, 0] * post.omega[1, 1], rel=0.03)
    expected_cross = sigma_mean[0, 1] * post.omega[1, 2]
    assert cov[0, 1] == pytest.approx(expected_cross, abs=0.03 * np.sqrt(cov[0, 0] * cov[1, 1]))
    assert np.all(np.linalg.eigvalsh(S) > 0)


def test_posterior_draws_match_scipy_inverse_wishart() -> None:
    rng = np.random.default_rng(12)
    s = VARSystem.from_array(rng.standard_normal((40, 3)), lags=1)
    post = posterior(s, BVARHyperparameters(0.2, np.ones(3)))
    _, S = post.draw(20_000, rng=1)
    ref = scipy.stats.invwishart(df=post.dof, scale=post.scale).rvs(20_000, random_state=2)
    np.testing.assert_allclose(S.mean(axis=0), ref.mean(axis=0), rtol=0.03, atol=0.003)
    np.testing.assert_allclose(S.var(axis=0), ref.var(axis=0), rtol=0.1, atol=1e-4)


def test_draw_shapes_and_errors(system: VARSystem, hyper: BVARHyperparameters) -> None:
    post = posterior(system, hyper)
    B, S = post.draw(5, rng=np.random.default_rng(0))
    assert B.shape == (5, system.k, 3) and S.shape == (5, 3, 3)
    B2, _ = post.draw(5, rng=np.random.default_rng(0))
    np.testing.assert_array_equal(B, B2)
    with pytest.raises(ValueError, match="n_draws"):
        post.draw(0)


def test_doctests() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        result = doctest.testmod(bp, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)
    assert result.failed == 0


def test_multigammaln_matches_scipy() -> None:
    from scipy import special

    ref = special.multigammaln  # pyright: ignore[reportAttributeAccessIssue]
    for a, n in ((3.7, 4), (10.0, 1), (25.5, 12)):
        assert bp._multigammaln(a, n) == pytest.approx(ref(a, n))
