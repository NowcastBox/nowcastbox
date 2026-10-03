"""Blocked random-walk prior of the large BVAR (``prior_mean="blocked_random_walk"``).

Validation: (a) the dummy observations encode the stated (non-diagonal) prior moments;
(b) the closed-form marginal likelihood matches an independent computation with a full
prior-mean matrix and grouped dummy priors, and its analytic gradient matches finite
differences; (c) the grouped sum-of-coefficients / initial-observation rows are satisfied
exactly by the blocked random walk; (d) on a simulated monthly random walk the posterior
is centred on the last month and GLP selects a much tighter lambda than with the own-lag
random walk; (e) with a very tight prior the conditional nowcast is the last observed
month.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from scipy.special import gammaln

from nowcastbox.core.data import MixedFrequencyData
from nowcastbox.models import LargeBVAR
from nowcastbox.models._bvar_prior import (
    BVARHyperparameters,
    PriorSettings,
    VARSystem,
    dummy_initial_observation_dummies,
    implied_prior_from_dummies,
    log_marginal_likelihood,
    log_marginal_likelihood_gradient,
    minnesota_dummies,
    minnesota_prior,
    posterior,
    prior_dummies,
    sum_of_coefficients_dummies,
)
from nowcastbox.models.bvar import (
    _prior_settings,  # pyright: ignore[reportPrivateUsage]
    fit_blocked_bvar,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
BLOCKED_A1 = np.array(  # one monthly series (3 blocks) + one quarterly series
    [[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float
)
GROUPS = ["x", "x", "x", "q"]


def _blocked_rw(rng: np.random.Generator, n_quarters: int, n_monthly: int) -> np.ndarray:
    """Blocked vector of ``n_monthly`` monthly random walks + one quarterly random walk."""
    months = np.cumsum(rng.standard_normal((3 * n_quarters, n_monthly)), axis=0)
    blocks = months.reshape(n_quarters, 3, n_monthly).transpose(0, 2, 1).reshape(n_quarters, -1)
    quarterly = np.cumsum(rng.standard_normal(n_quarters))
    return np.column_stack([blocks, quarterly])


def _system(seed: int = 0, n_quarters: int = 40, lags: int = 2) -> VARSystem:
    return VARSystem.from_array(_blocked_rw(np.random.default_rng(seed), n_quarters, 1), lags)


def _settings(**kwargs: object) -> PriorSettings:
    return PriorSettings(prior_mean=BLOCKED_A1, unit_root_groups=GROUPS, **kwargs)  # type: ignore[arg-type]


def multigammaln(a: float, n: int) -> float:
    """Log multivariate Gamma function."""
    return n * (n - 1) / 4 * np.log(np.pi) + float(np.sum(gammaln(a - np.arange(n) / 2)))


def _direct_log_ml(
    Y: np.ndarray, X: np.ndarray, b: np.ndarray, omega: np.ndarray, psi: np.ndarray, d: float
) -> float:
    """GLP (2015, appendix) formula with explicit cross-products (independent of the QR)."""
    T, n = Y.shape
    Om_inv = np.diag(1.0 / omega)
    P = X.T @ X + Om_inv
    B = np.linalg.solve(P, X.T @ Y + Om_inv @ b)
    E = Y - X @ B
    S = psi + E.T @ E + (B - b).T @ Om_inv @ (B - b)
    logdet = lambda A: np.linalg.slogdet(A)[1]  # noqa: E731
    return float(
        -0.5 * n * T * np.log(np.pi)
        + multigammaln((d + T) / 2, n)
        - multigammaln(d / 2, n)
        - 0.5 * n * np.sum(np.log(omega))
        + 0.5 * d * logdet(psi)
        - 0.5 * n * logdet(P)
        - 0.5 * (d + T) * logdet(S)
    )


def _fd(
    f: Callable[[BVARHyperparameters], float], h: BVARHyperparameters, field: str, eps: float = 1e-5
) -> float:
    def shifted(sign: float) -> float:
        return f(replace(h, **{field: getattr(h, field) * np.exp(sign * eps)}))

    return (shifted(1.0) - shifted(-1.0)) / (2 * eps)


# ---------------------------------------------------------------------------
# PriorSettings with a full mean matrix and unit-root groups
# ---------------------------------------------------------------------------
def test_settings_matrix_and_groups() -> None:
    s = _settings()
    np.testing.assert_array_equal(s.first_lag_mean(4), BLOCKED_A1)
    np.testing.assert_array_equal(s.own_lag_mean(4), [0, 0, 1, 1])
    np.testing.assert_array_equal(s.group_index(4), [0, 0, 0, 1])
    assert hash(s) == hash(_settings()) and s == _settings()  # stored as tuples
    np.testing.assert_array_equal(PriorSettings([1.0, 0.0]).first_lag_mean(2), np.diag([1, 0]))
    with pytest.raises(ValueError, match="square"):
        PriorSettings(prior_mean=np.ones((2, 3)))
    with pytest.raises(ValueError, match="square and finite"):
        PriorSettings(prior_mean=[[np.nan]])
    with pytest.raises(ValueError, match="shape"):
        s.first_lag_mean(3)
    with pytest.raises(ValueError, match="labels"):
        s.group_index(3)


def test_minnesota_prior_places_full_mean() -> None:
    system = _system()
    hyper = BVARHyperparameters(0.3, system.ar_residual_variances(1))
    prior = minnesota_prior(system, hyper, _settings())
    first = [system.lag_column(1, j) for j in range(4)]
    np.testing.assert_array_equal(prior.mean[first], BLOCKED_A1.T)
    assert np.count_nonzero(prior.mean) == 4  # nothing on the constant or lag 2
    # variances unchanged: quarterly lag decay, common to all equations
    plain = minnesota_prior(system, hyper, PriorSettings())
    np.testing.assert_allclose(prior.omega, plain.omega)


def test_minnesota_dummies_encode_full_mean() -> None:
    system = _system()
    hyper = BVARHyperparameters(0.3, system.ar_residual_variances(1))
    settings = _settings()
    prior = minnesota_prior(system, hyper, settings)
    mean, omega, scale = implied_prior_from_dummies(*minnesota_dummies(system, hyper, settings))
    np.testing.assert_allclose(mean, prior.mean, atol=1e-10)
    np.testing.assert_allclose(np.diag(omega), prior.omega, rtol=1e-8)
    np.testing.assert_allclose(scale, prior.scale, rtol=1e-10, atol=1e-12)


# ---------------------------------------------------------------------------
# grouped dummy priors
# ---------------------------------------------------------------------------
def test_grouped_dummy_rows() -> None:
    system = _system(lags=1)
    y0 = system.y0_mean
    level = y0[:3].mean()
    Yd, Xd = sum_of_coefficients_dummies(system, 0.5, [0, 0, 0, 1])
    np.testing.assert_allclose(Yd, [[level, level, level, 0], [0, 0, 0, y0[3]]] / np.array(0.5))
    np.testing.assert_allclose(Xd[:, 1:], Yd)
    np.testing.assert_array_equal(Xd[:, 0], 0.0)
    yd, xd = dummy_initial_observation_dummies(system, 2.0, [0, 0, 0, 1])
    np.testing.assert_allclose(yd[0], np.r_[level, level, level, y0[3]] / 2.0)
    np.testing.assert_allclose(xd[0], np.r_[0.5, yd[0]])
    # singleton groups (default) = the usual priors
    for fn, tight in ((sum_of_coefficients_dummies, 0.7), (dummy_initial_observation_dummies, 3)):
        for a, b in zip(fn(system, tight), fn(system, tight, [0, 1, 2, 3]), strict=True):
            np.testing.assert_allclose(a, b)
    with pytest.raises(ValueError, match="labels"):
        sum_of_coefficients_dummies(system, 1.0, [0, 1])
    hyper = BVARHyperparameters(0.2, np.ones(4), mu=1.0, delta=1.0)
    assert prior_dummies(system, hyper, _settings())[0].shape == (3, 4)
    assert prior_dummies(system, hyper, PriorSettings())[0].shape == (5, 4)


@pytest.mark.parametrize("lags", [1, 3])
def test_blocked_random_walk_satisfies_grouped_dummies_exactly(lags: int) -> None:
    """At the blocked-RW mean (constant 0) every grouped dummy row has zero residual.

    SoC: a series whose three blocks sit at a common level in all lags stays there; the
    other variables do not move. With per-block rows (the usual prior) it does not hold.
    """
    system = _system(seed=4, lags=lags)
    hyper = BVARHyperparameters(0.2, np.ones(4), mu=0.3, delta=0.3)
    b = minnesota_prior(system, hyper, _settings()).mean
    Yd, Xd = prior_dummies(system, hyper, _settings())
    np.testing.assert_allclose(Yd - Xd @ b, 0.0, atol=1e-12)
    Yd_old, Xd_old = prior_dummies(system, hyper, PriorSettings())
    assert np.max(np.abs(Yd_old - Xd_old @ b)) > 0.1


def test_tight_grouped_soc_imposes_one_unit_root_per_series() -> None:
    system = _system(seed=2, n_quarters=60)
    hyper = BVARHyperparameters(5.0, system.ar_residual_variances(1), mu=1e-5)
    B = posterior(system, hyper, _settings()).mean
    total = B[1:5] + B[5:9]  # sum over lags; rows = regressors, columns = equations
    np.testing.assert_allclose(total[:3].sum(axis=0), [1, 1, 1, 0], atol=1e-6)
    np.testing.assert_allclose(total[3], [0, 0, 0, 1], atol=1e-6)


# ---------------------------------------------------------------------------
# marginal likelihood and gradient with a non-diagonal prior mean
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dummies", [False, True])
def test_marginal_likelihood_matches_direct_formula(dummies: bool) -> None:
    system = _system(seed=1, n_quarters=30)
    settings = _settings(intercept_variance=100.0)
    hyper = BVARHyperparameters(
        0.4, system.ar_residual_variances(1), mu=2.0 if dummies else None, delta=dummies or None
    )
    prior = minnesota_prior(system, hyper, settings)
    psi, d = prior.scale, prior.dof
    Yd, Xd = prior_dummies(system, hyper, settings)
    Y, X = np.vstack([Yd, system.Y]), np.vstack([Xd, system.X])
    expected = _direct_log_ml(Y, X, prior.mean, prior.omega, psi, d)
    if dummies:
        expected -= _direct_log_ml(Yd, Xd, prior.mean, prior.omega, psi, d)
    assert log_marginal_likelihood(system, hyper, settings) == pytest.approx(expected, rel=1e-9)
    # and it really depends on the mean matrix
    assert log_marginal_likelihood(system, hyper, PriorSettings()) != pytest.approx(expected)


@pytest.mark.parametrize("delta", [None, 0.7])
def test_gradient_matches_finite_differences(delta: float | None) -> None:
    system = _system(seed=3, n_quarters=35)
    settings = _settings()
    hyper = BVARHyperparameters(0.3, system.ar_residual_variances(1), mu=0.8, delta=delta)

    def f(x: BVARHyperparameters) -> float:
        return log_marginal_likelihood(system, x, settings)

    value, grad = log_marginal_likelihood_gradient(system, hyper, settings)
    assert value == pytest.approx(f(hyper))
    assert grad.lambda_ == pytest.approx(_fd(f, hyper, "lambda_"), rel=1e-5, abs=1e-5)
    assert grad.mu == pytest.approx(_fd(f, hyper, "mu"), rel=1e-5, abs=1e-5)
    if delta is not None:
        assert grad.delta == pytest.approx(_fd(f, hyper, "delta"), rel=1e-5, abs=1e-5)
    for j in range(4):
        psi = hyper.psi.copy()
        eps = 1e-5
        up, down = psi * 1.0, psi * 1.0
        up[j] *= np.exp(eps)
        down[j] *= np.exp(-eps)
        fd = (f(replace(hyper, psi=up)) - f(replace(hyper, psi=down))) / (2 * eps)
        assert grad.psi[j] == pytest.approx(fd, rel=1e-5, abs=1e-5)


# ---------------------------------------------------------------------------
# LargeBVAR options
# ---------------------------------------------------------------------------
def test_prior_settings_from_options() -> None:
    names = ["a", "a", "a", "g", "b", "b", "b"]
    months = [1, 2, 3, 3, 1, 2, 3]
    s = _prior_settings(names, "blocked_random_walk", 2.0, months)
    A1 = s.first_lag_mean(7)
    np.testing.assert_array_equal(A1[:3, 2], 1.0)
    np.testing.assert_array_equal(A1[4:, 6], 1.0)
    assert A1[3, 3] == 1.0 and A1.sum() == 7
    np.testing.assert_array_equal(s.group_index(7), [0, 0, 0, 1, 2, 2, 2])
    mixed = _prior_settings(names, {"a": "blocked_random_walk", "b": 0.5}, 2.0, months)
    A1 = mixed.first_lag_mean(7)
    np.testing.assert_array_equal(A1[:3, 2], 1.0)
    np.testing.assert_array_equal(np.diag(A1)[3:], [0.0, 0.5, 0.5, 0.5])
    # the last month is found by month number, not by column order
    swapped = _prior_settings(["a", "a"], "blocked_random_walk", 2.0, [3, 1])
    np.testing.assert_array_equal(swapped.first_lag_mean(2), [[1, 0], [1, 0]])
    # monthly engine (no months): every variable is its own last month = random walk
    flat = _prior_settings(["a", "b"], "blocked_random_walk", 2.0)
    np.testing.assert_array_equal(flat.first_lag_mean(2), np.eye(2))
    # the other options are unchanged
    assert _prior_settings(names, "random_walk", 2.0, months) == PriorSettings("random_walk")
    assert _prior_settings(names, {"a": 1}, 2.0, months).unit_root_groups is None


def _monthly_rw_panel(seed: int, n_quarters: int, n_series: int) -> MixedFrequencyData:
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.standard_normal((3 * n_quarters, n_series)), axis=0)
    idx = pd.period_range("1990-01", periods=3 * n_quarters, freq="M")
    frame = pd.DataFrame(x, index=idx, columns=[f"x{i}" for i in range(n_series)])
    return MixedFrequencyData(frame, dict.fromkeys(frame.columns, "M"))


def test_glp_lambda_much_tighter_on_monthly_random_walk() -> None:
    panel = _monthly_rw_panel(0, 80, 5)
    fits = {
        pm: fit_blocked_bvar(panel, lags=2, prior_mean=pm)
        for pm in ("random_walk", "blocked_random_walk")
    }
    old, new = fits["random_walk"], fits["blocked_random_walk"]
    assert new.hyperparameters.lambda_ < 0.1 * old.hyperparameters.lambda_
    assert new.selection is not None and old.selection is not None
    assert new.selection.log_ml > old.selection.log_ml + 100
    B = new.posterior.mean  # rows: 1 + lag-1 blocks of x0..x4, then lag 2
    for s in range(5):
        cols = slice(3 * s, 3 * s + 3)
        np.testing.assert_allclose(B[1 + 3 * s + 2, cols], 1.0, atol=0.05)
        np.testing.assert_allclose(B[1 + 3 * s, cols], 0.0, atol=0.05)


def test_glp_with_dummy_priors_on_monthly_random_walk() -> None:
    panel = _monthly_rw_panel(1, 80, 3)
    kw = {"lags": 2, "sum_of_coefficients": True, "initial_observation": True}
    old = fit_blocked_bvar(panel, prior_mean="random_walk", **kw)
    new = fit_blocked_bvar(panel, prior_mean="blocked_random_walk", **kw)
    assert new.hyperparameters.lambda_ < 0.1 * old.hyperparameters.lambda_
    assert new.selection is not None and old.selection is not None
    assert new.selection.log_ml > old.selection.log_ml


def _tight_fits(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    idx = pd.period_range("1990-01", periods=len(x), freq="M")
    frame = pd.DataFrame(x, index=idx, columns=[f"x{i}" for i in range(x.shape[1])])
    panel = MixedFrequencyData(frame, dict.fromkeys(frame.columns, "M"))
    kw = {"lags": 1, "prior": {"lambda": 1e-5}, "standardize": False}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        new = fit_blocked_bvar(panel, prior_mean="blocked_random_walk", **kw)
        old = fit_blocked_bvar(panel, prior_mean="random_walk", **kw)
    return new.edge(new.values)[1], old.edge(old.values)[1]


def test_tight_prior_nowcast_is_last_observed_month() -> None:
    """lambda -> 0: every unreleased month equals the last released month of its series."""
    n_q, n_s = 200, 3
    rng = np.random.default_rng(5)
    inc = rng.standard_normal((3 * n_q, n_s))
    est = np.arange(3, 3 * (n_q - 1))  # increments entering the estimation (lags=1)
    for m in range(3):  # zero mean by month of quarter: the free constant is then zero
        rows = est[est % 3 == m]
        inc[rows] -= inc[rows].mean(axis=0)
    x = np.cumsum(inc, axis=0)
    x[-3:] = np.nan  # nothing released in the last quarter
    mean, mean_old = _tight_fits(x)
    for s in range(n_s):
        np.testing.assert_allclose(mean[-1, 3 * s : 3 * s + 3], x[-4, s], atol=1e-4)
        # the own-lag random walk repeats the same month of the previous quarter instead
        # (up to the free constant: three-month increments are not demeaned here)
        np.testing.assert_allclose(mean_old[-1, 3 * s : 3 * s + 3], x[-6:-3, s], atol=0.1)
    # first month of x0 released: its later months are centred on it (monthly random walk;
    # up to the sample covariance of the innovations, which sets E[e2 | e1])
    x[-3, 0] = x[-4, 0] + 1.5
    mean, _ = _tight_fits(x)
    np.testing.assert_allclose(mean[-1, 1:3] - x[-4, 0], 1.5, rtol=0.15)


def test_large_bvar_levels_end_to_end() -> None:
    panel = _monthly_rw_panel(2, 60, 3)
    frame = panel.data.copy()
    gdp = frame.rolling(3).mean().sum(axis=1)
    frame["gdp"] = np.where(frame.index.month % 3 == 0, gdp, np.nan)
    frame.iloc[-2:, :2] = np.nan
    frame.iloc[-3:, 3] = np.nan
    data = MixedFrequencyData(frame, {**dict.fromkeys(panel.data.columns, "M"), "gdp": "Q"})
    res = LargeBVAR(
        lags=2,
        prior_mean={"x0": "blocked_random_walk", "x1": "blocked_random_walk", "gdp": 1.0},
        sum_of_coefficients=True,
        initial_observation=True,
    ).fit(data, "gdp")
    assert res.bvar is not None
    assert res.bvar.settings.unit_root_groups is not None
    assert np.isfinite(res.get_nowcast())
