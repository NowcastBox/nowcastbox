"""Property-based and performance tests of MixedFreqDFM."""

from __future__ import annotations

import time
import warnings

import numpy as np
import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from nowcastbox.core.exceptions import ConvergenceWarning
from nowcastbox.models import MixedFreqDFM
from tests.models.test_em_simulation import simulate_mixed_dfm


@pytest.mark.property
@settings(max_examples=8, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    seed=st.integers(0, 10_000),
    idiosyncratic=st.sampled_from(["ar1", "iid"]),
    factor_lags=st.integers(1, 2),
    missing=st.floats(0.0, 0.3),
)
@example(seed=4120, idiosyncratic="iid", factor_lags=1, missing=0.0)
@example(seed=1464, idiosyncratic="ar1", factor_lags=1, missing=0.0)
def test_loglikelihood_never_decreases(seed, idiosyncratic, factor_lags, missing):
    """EM is monotone for arbitrary missing-data patterns (up to numerical noise).

    The transition updates are safeguarded with the initial-state term of the expected
    complete-data log-likelihood (generalized EM, ``m_step(initial_term=True)``); the
    seeds 4120 ("iid") and 1464 ("ar1") lowered the log-likelihood (2.2e-7 and 3.0e-5
    relative) when that term was ignored, identically with both E-steps.
    """
    sim = simulate_mixed_dfm(n_periods=72, n_monthly=6, n_quarterly=1, seed=seed)
    data = sim.data.copy()
    rng = np.random.default_rng(seed)
    monthly = [c for c in data.columns if c.startswith("m")]
    mask = rng.random((len(data), len(monthly))) < missing
    data[monthly] = data[monthly].mask(mask)
    model = MixedFreqDFM(factor_lags=factor_lags, idiosyncratic=idiosyncratic, max_iter=15, tol=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        res = model.fit(data, "q0", frequency=sim.frequencies)
    path = res.loglikelihood_path
    assert np.all(np.isfinite(path))
    assert np.all(np.diff(path) >= -1e-8 * np.abs(path[1:]))
    assert res.info["n_loglikelihood_decreases"] == 0


@pytest.mark.property
@settings(max_examples=5, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10_000), scale=st.floats(0.1, 100.0), shift=st.floats(-50, 50))
def test_affine_invariance_of_nowcast(seed, scale, shift):
    """Rescaling the data rescales the nowcast (standardisation is internal)."""
    sim = simulate_mixed_dfm(n_periods=60, n_monthly=5, n_quarterly=1, seed=seed)
    model = MixedFreqDFM(max_iter=5, tol=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        base = model.fit(sim.data, "q0", frequency=sim.frequencies)
        moved = model.fit(sim.data * scale + shift, "q0", frequency=sim.frequencies)
    assert moved.get_nowcast() == pytest.approx(
        base.get_nowcast() * scale + shift, rel=1e-6, abs=1e-6
    )
    path_shift = moved.loglikelihood_path - base.loglikelihood_path
    assert np.allclose(path_shift, path_shift[0], atol=1e-6)


@pytest.mark.slow
def test_performance_smoke_n100_t240(timing_reliable):
    """N=100, T=240 with two blocks: a few EM iterations in reasonable time."""
    threadpoolctl = pytest.importorskip("threadpoolctl")
    sim = simulate_mixed_dfm(
        n_periods=240, n_monthly=95, n_quarterly=5, blocks={"global": 1, "real": 1}, seed=0
    )
    model = MixedFreqDFM(n_factors=1, factor_lags=2, blocks=sim.blocks, max_iter=5, tol=0.0)
    # single-threaded BLAS: avoids oversubscription when the suite runs in parallel
    with threadpoolctl.threadpool_limits(1), warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        start = time.perf_counter()
        res = model.fit(sim.data, "q0", frequency=sim.frequencies)
        elapsed = time.perf_counter() - start
    assert res.state_layout.n_states == 130
    assert res.n_iter == 5
    assert np.all(np.diff(res.loglikelihood_path) > -1e-7 * np.abs(res.loglikelihood_path[1:]))
    if timing_reliable:
        assert elapsed < 120.0, elapsed
