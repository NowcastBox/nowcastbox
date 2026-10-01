"""Brute-force re-implementations of the forecast-evaluation statistics.

Each statistic is recomputed from its definition in the original paper with plain
loops / numerical integration and compared with :mod:`nowcastbox.evaluation`.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from itertools import pairwise

import numpy as np
import pytest
from scipy import integrate, stats

from nowcastbox.evaluation import diebold_mariano, giacomini_white, model_confidence_set
from nowcastbox.evaluation.scoring import (
    christoffersen_test,
    crps_gaussian,
    crps_mixture,
    crps_sample,
    pit,
)


# ---------------------------------------------------------------- Diebold-Mariano
@pytest.mark.parametrize("h", [1, 2, 4])
def test_dm_hln_matches_definition(h: int) -> None:
    rng = np.random.default_rng(h)
    e1, e2 = rng.normal(0, 1, 60), rng.normal(0, 1.3, 60)
    d = e1**2 - e2**2
    n = d.size
    dbar = d.mean()
    gamma = [sum((d[t] - dbar) * (d[t - k] - dbar) for t in range(k, n)) / n for k in range(h)]
    lrv = gamma[0] + 2 * sum(gamma[1:])
    dm = dbar / math.sqrt(lrv / n)
    hln = dm * math.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)  # HLN (1997) eq. 9
    res = diebold_mariano(e1, e2, h=h)
    assert res.statistic == pytest.approx(hln, rel=1e-12)
    assert res.pvalue == pytest.approx(2 * stats.t.sf(abs(hln), n - 1), rel=1e-10)
    raw = diebold_mariano(e1, e2, h=h, hln=False)
    assert raw.statistic == pytest.approx(dm, rel=1e-12)
    assert raw.pvalue == pytest.approx(2 * stats.norm.sf(abs(dm)), rel=1e-10)


# ---------------------------------------------------------------- Giacomini-White
def test_gw_one_step_matches_definition() -> None:
    rng = np.random.default_rng(7)
    e1, e2 = rng.normal(0, 1, 80), rng.normal(0, 1.2, 80)
    d = e1**2 - e2**2
    # GW (2006), tau = 1: Z_t = h_{t-1} d_t with h_{t-1} = (1, d_{t-1})
    Z = np.array([[d[t], d[t - 1] * d[t]] for t in range(1, d.size)])
    n = Z.shape[0]
    zbar = Z.mean(axis=0)
    omega = sum(np.outer(z, z) for z in Z) / n
    stat = n * zbar @ np.linalg.inv(omega) @ zbar
    res = giacomini_white(e1, e2)
    assert res.statistic == pytest.approx(stat, rel=1e-10)
    assert res.pvalue == pytest.approx(stats.chi2.sf(stat, 2), rel=1e-10)


# ---------------------------------------------------------------- MCS
def test_mcs_tmax_first_step_matches_definition() -> None:
    """First elimination step of HLN (2011) with an explicit bootstrap loop."""
    rng = np.random.default_rng(3)
    losses = rng.standard_normal((120, 3)) ** 2 * np.array([1.0, 1.1, 1.6])
    block, n_boot = 4, 300
    res = model_confidence_set(losses, n_bootstrap=n_boot, block_length=block, random_state=11)
    # bootstrap indices exactly as the implementation draws them (circular blocks)
    gen = np.random.default_rng(11)
    n, m = losses.shape
    n_blocks = -(-n // block)
    starts = gen.integers(0, n, size=(n_boot, n_blocks))
    lbar = losses.mean(axis=0)
    zeta = np.empty((n_boot, m))
    for b in range(n_boot):
        idx = [(s + j) % n for s in starts[b] for j in range(block)][:n]
        zeta[b] = losses[idx].mean(axis=0) - lbar
    d_i = lbar - lbar.mean()  # \bar d_{i.}
    zd = zeta - zeta.mean(axis=1, keepdims=True)
    var = (zd**2).mean(axis=0)
    t_i = d_i / np.sqrt(var)
    t_boot = (zd / np.sqrt(var)).max(axis=1)
    p_first = float(np.mean(t_boot >= t_i.max()))
    worst = f"model{int(np.argmax(t_i)) + 1}"
    assert res.eliminated[0] == worst
    assert res.pvalues.iloc[0] == pytest.approx(p_first)


# ---------------------------------------------------------------- CRPS
def _crps_integral(cdf: Callable[[float], float], y: float, lo: float, hi: float) -> float:
    left = integrate.quad(lambda x: cdf(x) ** 2, lo, y, limit=200)[0]
    right = integrate.quad(lambda x: (1 - cdf(x)) ** 2, y, hi, limit=200)[0]
    return left + right


@pytest.mark.parametrize("y", [-1.7, 0.0, 0.4, 3.0])
def test_crps_gaussian_and_mixture_match_integral_definition(y: float) -> None:
    mu, sd = 0.3, 1.4
    ref = _crps_integral(lambda x: stats.norm.cdf(x, mu, sd), y, -30, 30)
    assert float(crps_gaussian(y, mu, sd)) == pytest.approx(ref, rel=1e-7)
    locs, scales, w = np.array([-1.0, 0.5, 2.0]), np.array([0.5, 1.0, 0.8]), np.array([1, 2, 1])
    wn = w / w.sum()

    def mix_cdf(x: float) -> float:
        return float(np.sum(wn * stats.norm.cdf(x, locs, scales)))

    ref_mix = _crps_integral(mix_cdf, y, -30, 30)
    assert float(crps_mixture(y, locs, scales, w)) == pytest.approx(ref_mix, rel=1e-7)


def test_crps_sample_matches_double_sum() -> None:
    rng = np.random.default_rng(0)
    x, y = rng.standard_normal(25), 0.37
    m = x.size
    first = np.mean(np.abs(x - y))
    pair = sum(abs(a - b) for a in x for b in x)
    assert float(crps_sample(y, x)) == pytest.approx(first - pair / (2 * m * m), rel=1e-12)
    fair = first - pair / (2 * m * (m - 1))
    assert float(crps_sample(y, x, fair=True)) == pytest.approx(fair, rel=1e-12)
    # empirical-cdf integral form
    ecdf = lambda t: float(np.mean(x <= t))  # noqa: E731
    pts = np.sort(np.append(x, y))
    integral = sum((ecdf(a) - (1.0 if a >= y else 0.0)) ** 2 * (b - a) for a, b in pairwise(pts))
    assert float(crps_sample(y, x)) == pytest.approx(integral, rel=1e-10)


def test_randomized_pit_is_uniform_for_exchangeable_draws() -> None:
    rng = np.random.default_rng(1)
    draws = rng.integers(0, 4, size=(4000, 9)).astype(float)  # many ties
    y = rng.integers(0, 4, size=4000).astype(float)
    u = pit(draws, y, randomize=True, random_state=2)
    assert stats.kstest(u, "uniform").pvalue > 0.01


# ---------------------------------------------------------------- Christoffersen
def test_christoffersen_matches_definition() -> None:
    hits = np.array([1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1])
    p = 0.8
    n1, n0 = hits.sum(), hits.size - hits.sum()
    pi = n1 / hits.size
    lr_uc = -2 * ((n1 * np.log(p) + n0 * np.log(1 - p)) - (n1 * np.log(pi) + n0 * np.log(1 - pi)))
    pairs = list(pairwise(hits))
    n = {(i, j): sum(1 for a, b in pairs if (a, b) == (i, j)) for i in (0, 1) for j in (0, 1)}
    p01 = n[0, 1] / (n[0, 0] + n[0, 1])
    p11 = n[1, 1] / (n[1, 0] + n[1, 1])
    p2 = (n[0, 1] + n[1, 1]) / len(pairs)
    l_markov = (
        n[0, 0] * np.log(1 - p01)
        + n[0, 1] * np.log(p01)
        + n[1, 0] * np.log(1 - p11)
        + n[1, 1] * np.log(p11)
    )
    l_ind = (n[0, 0] + n[1, 0]) * np.log(1 - p2) + (n[0, 1] + n[1, 1]) * np.log(p2)
    res = christoffersen_test(hits, p)
    assert res.lr_uc == pytest.approx(lr_uc, rel=1e-12)
    assert res.lr_ind == pytest.approx(-2 * (l_ind - l_markov), rel=1e-12)
    assert res.pvalue_cc == pytest.approx(stats.chi2.sf(res.lr_uc + res.lr_ind, 2), rel=1e-12)
