"""MixedFreqDFM vs statsmodels' ``DynamicFactorMQ`` on a NY Fed-like specification.

Fixture: ``scripts/reference_fixtures/statsmodels_dfmq.py`` (NYFED panel transformed by
``Bpanel``, standardised once; four blocks with one factor each, VAR(1), AR(1)
idiosyncratic components, Mariano-Murasawa quarterly restriction, no measurement noise:
``MixedFreqDFM(..., obs_noise_var=0)``). statsmodels is a numerical reference only.

Findings (``docs/validation/statsmodels.md``):

* the two log-likelihood functions coincide: statsmodels' optimum evaluated by nowcastbox
  gives ``loglike(params)`` to ~1e-12;
* warm-started at statsmodels' optimum, nowcastbox's EM keeps increasing the likelihood
  (monotone, tiny steps) and reproduces statsmodels' smoothed factors and nowcasts;
* from the sequential block principal components of Bańbura & Modugno
  (``init="pca_given"``) nowcastbox's EM converges to a local maximum about 418
  log-likelihood points **below** statsmodels' (start -9994 vs -9721). Since the wave-3
  integration ``init="pca"`` tries every block order and starts from the most likely
  one (here the reversed order, start -9717): the gap falls to 46 points
  (``reference_divergence``: different local maxima of the same likelihood).
"""

from __future__ import annotations

import warnings
from functools import cache
from typing import Any

import numpy as np
import pandas as pd
import pytest

from nowcastbox.models import EMParameters, MixedFreqDFM
from nowcastbox.models._em_steps import build_state_space
from nowcastbox.statespace import kalman_smoother, loglikelihood
from tests.reference_validation._helpers import legend, max_abs, read_json, read_periods

pytestmark = pytest.mark.reference_validation

BLOCKS = ["Global", "Soft", "Real", "Labor"]


@cache
def _setup() -> tuple[dict[str, Any], pd.DataFrame, dict[str, str], pd.DataFrame]:
    ref = read_json("statsmodels/nyfed_dfmq.json")
    panel = read_periods("r/nyfed_em_panel.csv.gz").loc[ref["sample"][0] : ref["sample"][1]]
    z = (panel - panel.mean()) / panel.std()
    leg = legend("nyfed")
    freq = {n: ("Q" if f == 4 else "M") for n, f in zip(leg["name"], leg["frequency"], strict=True)}
    blocks = leg.set_index("name")[BLOCKS]
    return ref, z, freq, blocks


def _model(**kwargs: Any) -> MixedFreqDFM:
    _, _, _, blocks = _setup()
    return MixedFreqDFM(n_factors=1, factor_lags=1, blocks=blocks, obs_noise_var=0.0, **kwargs)


@cache
def _layout_and_data() -> tuple[Any, np.ndarray]:
    _, z, freq, _ = _setup()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = _model(max_iter=0).fit(z, "GDPC1", frequency=freq)
    layout = res.state_layout
    assert res.data is not None
    values = res.data.standardize()[0].to_frame()[list(layout.series)].to_numpy()
    return layout, values


def statsmodels_parameters(
    params: dict[str, float], factor_names: list[str], layout: Any
) -> EMParameters:
    """statsmodels' parametrisation written as nowcastbox's EMParameters."""
    n = layout.n_series
    loadings = np.zeros((n, layout.n_factor_states))
    rho, var = np.zeros(n), np.zeros(n)
    for i, name in enumerate(layout.series):
        w = layout.weights[i]
        for b, block in enumerate(layout.block_names):
            if layout.membership[i, b]:
                for j, wj in enumerate(w):
                    loadings[i, layout.block_offsets[b] + j] = (
                        params[f"loading.{block}->{name}"] * wj
                    )
        kind = "eps_Q" if w.size > 1 else "eps_M"
        rho[i] = params[f"L1.{kind}.{name}"]
        var[i] = params[f"sigma2.{name}"]
    transition = tuple(np.array([[params[f"L1.{b}->{b}"]]]) for b in layout.block_names)
    cov = tuple(
        np.array([[params[f"fb({factor_names.index(b)}).cov.chol[1,1]"] ** 2]])
        for b in layout.block_names
    )
    return EMParameters(transition, cov, loadings, rho, var, np.zeros(n))


@cache
def _sm_params() -> EMParameters:
    ref = _setup()[0]
    layout, _ = _layout_and_data()
    return statsmodels_parameters(ref["params"], list(ref["factor_names"]), layout)


def test_same_likelihood_function() -> None:
    ref = _setup()[0]
    layout, values = _layout_and_data()
    ours = loglikelihood(build_state_space(_sm_params(), layout), values)
    assert ours == pytest.approx(ref["loglike"], rel=1e-9)


def test_em_from_statsmodels_optimum_is_monotone() -> None:
    ref, z, freq, _ = _setup()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = _model(max_iter=5, tol=0.0, init=_sm_params()).fit(z, "GDPC1", frequency=freq)
    path = np.asarray(res.loglikelihood_path)
    assert path[0] == pytest.approx(ref["loglike"], rel=1e-9)
    assert bool((np.diff(path) >= -1e-9 * abs(path[0])).all())
    assert path[-1] - path[0] < 1e-4 * abs(path[0])  # statsmodels' point is (nearly) stationary


def test_smoothed_quantities_at_statsmodels_optimum() -> None:
    ref, z, freq, _ = _setup()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = _model(max_iter=0, init=_sm_params()).fit(z, "GDPC1", frequency=freq)
    periods = pd.PeriodIndex(ref["periods"], freq="M")
    for block in BLOCKS:
        ours = res.factors[f"{block}_f1"].reindex(periods).to_numpy()
        assert max_abs(ours, ref["smoothed_factors"][block]) < 1e-6
    # smoothed (standardised) GDP signal of every month, from the estimated model
    model, layout, grid = res.state_space_model()
    values = z.reindex(grid)[list(layout.series)].to_numpy()
    signal = kalman_smoother(model, values).smoothed_signal()
    gdp = pd.Series(signal[:, list(layout.series).index("GDPC1")], index=grid).reindex(periods)
    assert max_abs(gdp, ref["smoothed_quarterly_signal"]["GDPC1"]) < 1e-6


@pytest.mark.reference_divergence
def test_default_initialisation_reaches_lower_optimum() -> None:
    ref, z, freq, _ = _setup()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = _model(tol=1e-6, max_iter=2000).fit(z, "GDPC1", frequency=freq)
    assert res.converged
    init = res.info["initialization"]
    assert init["block_order"] == "reversed"  # multi-start: best starting log-likelihood
    assert init["loglikelihood"]["reversed"] > -9721.11  # above statsmodels' start
    gap = ref["loglike"] - res.loglikelihood
    assert 0.0 < gap < 50.0  # documented: 46.0 points (417.6 with the sequential start)


@pytest.mark.reference_divergence
def test_sequential_initialisation_reaches_poorer_optimum() -> None:
    ref, z, freq, _ = _setup()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = _model(tol=1e-6, max_iter=2000, init="pca_given").fit(z, "GDPC1", frequency=freq)
    assert res.info["initialization"] == {"method": "pca", "block_order": "given"}
    gap = ref["loglike"] - res.loglikelihood
    assert 400.0 < gap < 0.1 * abs(ref["loglike"])  # documented: 417.6 points
