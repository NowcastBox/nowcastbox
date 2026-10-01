"""Comparison of MixedFreqDFM with statsmodels' DynamicFactorMQ (numerical reference only).

Both implement the Banbura-Modugno model with one factor, VAR(1) dynamics, AR(1)
idiosyncratic components for monthly series and a monthly AR(1) idiosyncratic component
aggregated with the Mariano-Murasawa weights for quarterly series. With
``obs_noise_var=0`` the two specifications coincide, so the log-likelihood of the same
parameters must agree to machine precision, and our EM optimum must be at least as good
as the statsmodels EM optimum.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from nowcastbox.models import EMParameters, MixedFreqDFM
from nowcastbox.models._em_steps import build_state_space
from nowcastbox.statespace import loglikelihood
from tests.models.test_em_simulation import simulate_mixed_dfm

DynamicFactorMQ = pytest.importorskip(
    "statsmodels.tsa.statespace.dynamic_factor_mq"
).DynamicFactorMQ

pytestmark = pytest.mark.reference_validation


@pytest.fixture(scope="module")
def setup():
    sim = simulate_mixed_dfm(n_periods=150, n_monthly=8, n_quarterly=1, seed=7, ragged=2)
    data = sim.data
    data = (data - data.mean()) / data.std()  # standardised once, for both libraries
    monthly = data[[c for c in data.columns if c.startswith("m")]]
    quarterly = data["q0"].dropna()
    quarterly.index = quarterly.index.asfreq("Q")
    sm_model = DynamicFactorMQ(
        monthly,
        endog_quarterly=quarterly.to_frame(),
        factors=1,
        factor_orders=1,
        idiosyncratic_ar1=True,
        standardize=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sm_res = sm_model.fit_em(maxiter=300, tolerance=1e-8, disp=False)
    ours = MixedFreqDFM(n_factors=1, factor_lags=1, obs_noise_var=0.0, tol=1e-8, max_iter=2000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = ours.fit(data, target="q0", frequency=sim.frequencies)
    return data, sm_model, sm_res, res


def _map_statsmodels_params(sm_params: pd.Series, layout) -> EMParameters:
    loadings = np.zeros((layout.n_series, layout.n_factor_states))
    rho, var = np.zeros(layout.n_series), np.zeros(layout.n_series)
    for i, name in enumerate(layout.series):
        w = layout.weights[i]
        loadings[i, layout.loading_index(i)] = sm_params[f"loading.0->{name}"] * w
        kind = "eps_Q" if w.size > 1 else "eps_M"
        rho[i] = sm_params[f"L1.{kind}.{name}"]
        var[i] = sm_params[f"sigma2.{name}"]
    A = np.array([[sm_params["L1.0->0"]]])
    Q = np.array([[sm_params["fb(0).cov.chol[1,1]"] ** 2]])
    return EMParameters((A,), (Q,), loadings, rho, var, np.zeros(layout.n_series))


def test_same_parameters_same_loglikelihood(setup):
    data, sm_model, sm_res, res = setup
    params = _map_statsmodels_params(sm_res.params, res.state_layout)
    ours = loglikelihood(build_state_space(params, res.state_layout), data.to_numpy())
    reference = sm_model.loglike(sm_res.params)
    assert ours == pytest.approx(reference, rel=1e-9)


def test_em_optimum_at_least_as_good(setup):
    _, sm_model, sm_res, res = setup
    reference = sm_model.loglike(sm_res.params)
    assert res.loglikelihood >= reference - 1e-6 * abs(reference)
    assert res.loglikelihood == pytest.approx(reference, rel=2e-3)


def test_our_optimum_evaluated_by_statsmodels(setup):
    """Our estimate, written in statsmodels' parametrisation, gives our log-likelihood."""
    _, sm_model, sm_res, res = setup
    lay, p = res.state_layout, res.em_parameters
    params = sm_res.params.copy()
    for i, name in enumerate(lay.series):
        params[f"loading.0->{name}"] = p.loadings[i, lay.loading_index(i)[0]]
        kind = "eps_Q" if lay.weights[i].size > 1 else "eps_M"
        params[f"L1.{kind}.{name}"] = p.idio_ar[i]
        params[f"sigma2.{name}"] = p.idio_var[i]
    params["L1.0->0"] = p.transition[0][0, 0]
    params["fb(0).cov.chol[1,1]"] = np.sqrt(p.factor_cov[0][0, 0])
    assert sm_model.loglike(params) == pytest.approx(res.loglikelihood, rel=1e-9)


def test_smoothed_factor_matches_reference(setup):
    data, _, sm_res, res = setup
    sm_factor = pd.Series(
        np.asarray(sm_res.factors.smoothed).ravel()[: len(data)], index=data.index
    )
    ours = res.factors["f1"].reindex(data.index)
    assert abs(np.corrcoef(sm_factor, ours)[0, 1]) > 0.995


@pytest.mark.reference_divergence
def test_reported_llf_of_fit_em_differs_from_loglike(setup):
    """Documented divergence: ``fit_em(...).llf`` of statsmodels 0.14 is not
    ``loglike(params)``; we always compare against ``loglike``."""
    _, sm_model, sm_res, _ = setup
    reported, evaluated = sm_res.llf, sm_model.loglike(sm_res.params)
    assert np.isfinite(reported)
    assert np.isfinite(evaluated)
    assert abs(reported - evaluated) > 1e-8 * abs(evaluated)
