"""Simulation validation of the large BVAR (plan, item 11).

* parameter recovery on data simulated from a known blocked VAR;
* posterior draws centred on the posterior mean;
* calibration of the posterior predictive intervals (Monte Carlo, slow);
* nowcast accuracy on a simulated mixed-frequency DFM panel, compared with
  :class:`~nowcastbox.models.MixedFreqDFM` (slow);
* runtime with N = 50 monthly series (slow, generous bound).
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pytest
import scipy.stats

import nowcastbox as nb
from nowcastbox.core.data import MixedFrequencyData
from nowcastbox.core.exceptions import ConvergenceWarning
from nowcastbox.density import NowcastDistribution
from nowcastbox.models import LargeBVAR, MixedFreqDFM
from tests.models.bvar_helpers import (
    blocked_panel,
    random_covariance,
    simulate_var,
    stable_coefficients,
)


@pytest.mark.parametrize("prior", [{"lambda": 10.0}, "glp"])
def test_parameter_recovery_on_simulated_blocked_var(prior: object) -> None:
    rng = np.random.default_rng(11)
    n, p = 7, 2
    A = stable_coefficients(n, p, rng)
    c = 0.2 * rng.standard_normal(n)
    sigma = random_covariance(n, rng)
    values = simulate_var(A, c, sigma, 600, rng)
    panel = blocked_panel(values, n_monthly=2, n_quarterly=1)
    res = LargeBVAR(lags=p, prior=prior, standardize=False).fit(panel, "q1")
    eng = res.bvar
    assert eng is not None
    est = eng.parameters
    assert np.abs(est.coefficients - A).max() < 0.15
    assert np.sqrt(np.mean((est.coefficients - A) ** 2)) < 0.05
    np.testing.assert_allclose(est.intercept, c, atol=0.15)
    np.testing.assert_allclose(est.sigma, sigma, atol=0.2 * np.abs(sigma).max())
    if prior == "glp":
        assert res.bvar.selection is not None and res.bvar.selection.success  # type: ignore[union-attr]


def test_posterior_draws_centred_on_posterior_mean() -> None:
    rng = np.random.default_rng(5)
    A = stable_coefficients(4, 1, rng)
    values = simulate_var(A, np.zeros(4), np.eye(4), 80, rng)
    panel = blocked_panel(values, n_monthly=1, n_quarterly=1)
    res = LargeBVAR(prior={"lambda": 0.5}).fit(panel, "q1")
    post = res.bvar.posterior  # type: ignore[union-attr]
    B, S = post.draw(4000, 0)
    se = B.std(axis=0) / np.sqrt(len(B))
    assert np.all(np.abs(B.mean(axis=0) - post.mean) < 5 * se + 1e-12)
    np.testing.assert_allclose(S.mean(axis=0), post.sigma_mean, atol=0.05)


@pytest.mark.slow
def test_posterior_intervals_are_calibrated() -> None:
    """Coverage of the 90% interval and uniform PIT of the nowcast (300 replications)."""
    rng = np.random.default_rng(0)
    A = stable_coefficients(7, 1, rng)
    sigma = random_covariance(7, rng)
    hits, pits = [], []
    for rep in range(300):
        values = simulate_var(A, np.zeros(7), sigma, 121, rng)
        truth = values[-1, 6]
        values[-1, 2:] = np.nan  # only the first two months of x1 released
        panel = blocked_panel(values, n_monthly=2, n_quarterly=1)
        res = LargeBVAR(n_draws=200, random_state=rep).fit(panel, "q1")
        dist = res.distribution()
        assert isinstance(dist, NowcastDistribution)
        lower, upper = dist.interval(0.9).iloc[0]
        hits.append(lower <= truth <= upper)
        pits.append(float(np.ravel(dist.cdf(truth))[0]))
    coverage = float(np.mean(hits))
    assert 0.85 <= coverage <= 0.95
    assert scipy.stats.kstest(pits, "uniform").pvalue > 0.01


def _masked_dfm_panel(seed: int) -> tuple[MixedFrequencyData, float]:
    """Simulated DFM panel with the last GDP hidden and a ragged edge of 0-2 months."""
    sim = nb.simulate.dfm(
        n_series=10, n_factors=1, n_periods=360, ragged_edge=False, random_state=seed
    )
    frame = sim.data.data.copy()
    truth = float(frame["gdp"].iloc[-1])
    frame.iloc[-1, frame.columns.get_loc("gdp")] = np.nan
    gen = np.random.default_rng(seed)
    for col in frame.columns.drop("gdp"):
        lag = int(gen.integers(0, 3))
        if lag:
            frame.iloc[-lag:, frame.columns.get_loc(col)] = np.nan
    return sim.data.with_data(frame), truth


@pytest.mark.slow
def test_nowcast_accuracy_comparable_to_dfm() -> None:
    """On DFM-generated data (the DFM is the true model) the BVAR stays close to it.

    Twenty panels of 10 monthly indicators and quarterly GDP (30 years); the last GDP
    is nowcast with a ragged edge of 0-2 months. Observed RMSEs: BVAR(1) 1.27,
    BVAR(2) 1.29, MixedFreqDFM 1.00, historical mean 2.78.
    """
    errors: dict[str, list[float]] = {"bvar": [], "dfm": [], "mean": []}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for seed in range(20):
            data, truth = _masked_dfm_panel(seed)
            errors["bvar"].append(LargeBVAR(lags=1).fit(data, "gdp").get_nowcast() - truth)
            errors["dfm"].append(MixedFreqDFM(n_factors=1).fit(data, "gdp").get_nowcast() - truth)
            errors["mean"].append(float(data.to_native("gdp").mean()) - truth)
    rmse = {k: float(np.sqrt(np.mean(np.square(v)))) for k, v in errors.items()}
    assert rmse["bvar"] < 1.6 * rmse["dfm"]
    assert rmse["bvar"] < 0.6 * rmse["mean"]


@pytest.mark.slow
@pytest.mark.benchmark
def test_runtime_with_50_monthly_series() -> None:
    """GLP fit, 200 posterior draws and a news decomposition with N = 50 (n = 151).

    Observed on one BLAS thread: fit 0.4-0.5 s, 200 draws 0.6-1.0 s, news 2-10 s
    (lags 1-2).
    """
    sim = nb.simulate.dfm(n_series=50, n_factors=2, n_periods=240, random_state=0)
    started = time.perf_counter()
    res = LargeBVAR(lags=2).fit(sim.data, "gdp")
    fit_time = time.perf_counter() - started
    assert res.n_variables == 151
    started = time.perf_counter()
    dist = res.distribution(n_draws=200, random_state=0)
    assert isinstance(dist, NowcastDistribution)
    draw_time = time.perf_counter() - started
    assert dist.n_components == 200
    old = sim.data.truncate(end=sim.data.index[-2])
    started = time.perf_counter()
    news = res.news(old, sim.data)
    news_time = time.perf_counter() - started
    assert news.check_identity()
    assert fit_time < 60.0
    assert draw_time < 60.0
    assert news_time < 120.0
