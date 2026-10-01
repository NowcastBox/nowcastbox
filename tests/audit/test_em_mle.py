"""The EM fixed point is the maximum-likelihood estimate (independent check).

The exact Gaussian log-likelihood (Kalman filter) is maximised directly with L-BFGS
over the free parameters of a one-factor mixed-frequency model with Mariano-Murasawa
restrictions, starting at the converged EM estimate. A wrong M-step (e.g. the
restricted least squares of the quarterly loadings or the idiosyncratic variances)
leaves an ascent direction and the optimiser gains likelihood; a correct one does
not (apart from the tiny effect of the stationary initial-state term on the factor
VAR, which the closed-form update of Bańbura & Modugno, 2014, ignores).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize

from nowcastbox.core.data import as_mixed_frequency_data
from nowcastbox.models import MixedFreqDFM
from nowcastbox.models._em_steps import EMParameters, build_state_space
from nowcastbox.statespace import kalman_filter

MM = np.array([1.0, 2.0, 3.0, 2.0, 1.0])


def _panel() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    n = 180
    f = np.zeros(n)
    for t in range(1, n):
        f[t] = 0.7 * f[t - 1] + rng.standard_normal()
    x = np.outer(f, [1.0, 0.8, 0.6, 0.9, -0.5]) + 0.6 * rng.standard_normal((n, 5))
    gdp = np.convolve(f, MM)[:n] / 3 + 0.5 * rng.standard_normal(n)
    gdp[np.arange(n) % 3 != 2] = np.nan
    idx = pd.period_range("2005-01", periods=n, freq="M")
    df = pd.DataFrame(x, index=idx, columns=list("abcde")).assign(gdp=gdp)
    df.iloc[-2:, :3] = np.nan
    return df


@pytest.mark.slow
def test_em_iid_fixed_point_is_the_likelihood_maximum() -> None:
    df = _panel()
    model = MixedFreqDFM(
        n_factors=1, idiosyncratic="iid", max_iter=3000, tol=1e-11, filter_method="univariate"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = model.fit(df, target="gdp", frequency={"gdp": "Q"})
    params, layout = res.em_parameters, res.state_layout
    assert params is not None and layout is not None
    y = as_mixed_frequency_data(df, {"gdp": "Q"}).standardize()[0].values

    def unpack(theta: np.ndarray) -> EMParameters:
        out = params.copy()
        out.loadings[:5, 0] = theta[:5]
        out.loadings[5, :5] = theta[5] * MM  # Mariano-Murasawa restriction
        out.transition[0][0, 0] = theta[6]
        out.factor_cov[0][0, 0] = np.exp(theta[7])
        out.obs_var[:] = np.exp(theta[8:14])
        return out

    def loglik(theta: np.ndarray) -> float:
        return kalman_filter(build_state_space(unpack(theta), layout), y).loglikelihood

    theta0 = np.r_[
        params.loadings[:5, 0],
        params.loadings[5, 0],
        params.transition[0][0, 0],
        np.log(params.factor_cov[0][0, 0]),
        np.log(params.obs_var),
    ]
    # the restriction holds exactly in the estimate
    np.testing.assert_allclose(params.loadings[5, :5], params.loadings[5, 0] * MM, rtol=1e-10)
    base = loglik(theta0)
    assert base == pytest.approx(res.loglikelihood, abs=1e-8)
    opt = minimize(lambda t: -loglik(t), theta0, method="L-BFGS-B")
    assert -opt.fun - base < 0.02  # (observed: 0.004, the initial-state term)
    # loadings and measurement variances: the M-step is exact -> zero score
    np.testing.assert_allclose(opt.x[:6], theta0[:6], atol=2e-3)
    np.testing.assert_allclose(opt.x[8:], theta0[8:], atol=2e-3)
