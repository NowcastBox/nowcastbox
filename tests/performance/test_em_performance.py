"""Performance regression tests of the EM E-step (innovation I2).

Marked ``slow`` and ``benchmark``; run with ``pytest tests/performance -m slow``. The
thresholds are deliberately loose (several times the timings measured on the
development machine, single BLAS thread) so that only real regressions fail; the
precise comparison with statsmodels lives in ``benchmarks/bench_em.py``.

Reference timings (WSL2, 1 BLAS thread, N = 200, T = 300, 285 states): structured
E-step about 0.35 s, dense E-step about 5.5 s, statsmodels ``DynamicFactorMQ`` about
7 s per EM iteration.
"""

from __future__ import annotations

import os
import time
import warnings

import numpy as np
import pytest

from benchmarks.bench_em import simulate_panel, time_statsmodels
from nowcastbox.core.data import MixedFrequencyData
from nowcastbox.models._em_steps import build_state_space, e_step
from nowcastbox.models._init_conditions import pca_initial_parameters
from nowcastbox.models.em import build_layout
from nowcastbox.statespace import kalman_smoother, smoothed_moments, structured_smoother

pytestmark = [pytest.mark.slow, pytest.mark.benchmark]


def _em_model(n_monthly: int, n_quarterly: int, n_periods: int):
    frame, freq = simulate_panel(n_monthly, n_quarterly, n_periods, seed=0)
    standardized, _ = MixedFrequencyData(frame, freq).standardize()
    layout = build_layout(standardized, 1, 1, None, "ar1")
    params = pca_initial_parameters(standardized, layout)
    return build_state_space(params, layout), standardized.values, frame, freq


def _best_of(func, repeat: int = 3) -> float:
    best = np.inf
    for _ in range(repeat):
        start = time.perf_counter()
        func()
        best = min(best, time.perf_counter() - start)
    return best


def test_structured_e_step_large_panel(timing_reliable):
    """N = 200, T = 300: the structured E-step is fast and exact."""
    model, y, _, _ = _em_model(180, 20, 300)
    smoothed_moments(model, y[:12], method="structured")  # JIT warm-up
    seconds = _best_of(lambda: smoothed_moments(model, y, method="structured"), repeat=2)
    if timing_reliable:
        assert seconds < 3.0, f"structured E-step took {seconds:.2f} s (reference ~0.35 s)"
    mom = smoothed_moments(model, y)
    assert mom.method == "structured"
    start = time.perf_counter()
    ref = e_step(model, y)
    dense = time.perf_counter() - start
    if timing_reliable:
        assert dense / seconds > 5.0, f"speed-up {dense / seconds:.1f}x (reference ~13x)"
    assert mom.loglikelihood == pytest.approx(ref.loglikelihood, rel=1e-10)
    known = np.isfinite(mom.s11)
    np.testing.assert_allclose(mom.s11[known], ref.s11[known], rtol=1e-7, atol=1e-7)


def test_structured_faster_than_dense(timing_reliable):
    """N = 100, T = 240: at least 3x faster than the dense smoother (measured ~4.6x).

    The dense smoother costs O(n m^3) and the structured one O(N n^2): the gap widens with
    the number of series (about 13x at N = 200).
    """
    model, y, _, _ = _em_model(90, 10, 240)
    structured_smoother(model, y[:12])
    kalman_smoother(model, y[:12])
    fast = _best_of(lambda: structured_smoother(model, y))
    dense = _best_of(lambda: kalman_smoother(model, y), repeat=1)
    if not timing_reliable:
        pytest.skip("timings are not meaningful under coverage or xdist")
    assert dense / fast > 3.0, f"speed-up {dense / fast:.1f}x (dense {dense:.2f} s)"


def test_dense_smoother_no_regression(timing_reliable):
    """Dense filter + smoother, 145 states, 240 periods (reference ~0.7 s)."""
    model, y, _, _ = _em_model(90, 10, 240)
    kalman_smoother(model, y[:12])
    seconds = _best_of(lambda: kalman_smoother(model, y), repeat=1)
    if not timing_reliable:
        pytest.skip("timings are not meaningful under coverage or xdist")
    assert seconds < 6.0, f"dense smoother took {seconds:.2f} s"


def _coverage_active() -> bool:
    """Whether coverage measurement is running (it slows our Python-level code only)."""
    try:
        import coverage
    except ImportError:  # pragma: no cover - coverage is a test dependency
        return False
    return coverage.Coverage.current() is not None


def test_faster_than_statsmodels_per_iteration():
    """N = 100, T = 240: EM iteration at least 2.5x faster than DynamicFactorMQ.

    Measured 3.7x-5x at this size (the gap grows with N: ~12x at N = 200, see
    ``benchmarks/bench_em.py``); the bound leaves room for timing noise.

    Skipped under coverage measurement: line tracing inflates the cost of our
    Python-level code (structured smoother bookkeeping, M-step) but not of the compiled
    statsmodels filter, so the ratio is not meaningful there. Also skipped inside a
    ``pytest-xdist`` worker, where concurrent tests distort the marginal timing of
    statsmodels (run ``pytest tests/performance`` alone).
    """
    from nowcastbox.models._em_steps import m_step

    if _coverage_active():
        pytest.skip("timing ratio is not meaningful under coverage measurement")
    if os.environ.get("PYTEST_XDIST_WORKER"):
        pytest.skip("timing ratio is not meaningful with tests running in parallel (xdist)")

    model, y, frame, freq = _em_model(90, 10, 240)
    standardized, _ = MixedFrequencyData(frame, freq).standardize()
    layout = build_layout(standardized, 1, 1, None, "ar1")
    params = pca_initial_parameters(standardized, layout)
    smoothed_moments(model, y[:12])

    def iteration() -> None:
        stats = smoothed_moments(build_state_space(params, layout), y)
        m_step(stats, layout, y, params)  # type: ignore[arg-type]

    ours = _best_of(iteration, repeat=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # best of two runs, as for our own timing: one statsmodels run is noisy
        reference = min(
            time_statsmodels(frame, freq, iterations=1)["seconds_per_iteration"] for _ in range(2)
        )
    assert reference / ours > 2.5, f"ratio {reference / ours:.1f}x (ours {ours:.3f} s)"
