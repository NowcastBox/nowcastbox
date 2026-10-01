"""Benchmark of the MixedFreqDFM EM algorithm (innovation I2).

Compares, on a simulated monthly + quarterly panel (one global factor, AR(1) factor,
AR(1) idiosyncratic components, Mariano-Murasawa aggregation of the quarterly series):

* ``statsmodels`` - ``DynamicFactorMQ(...).fit_em`` (same model: 1 factor, factor order
  1, ``idiosyncratic_ar1=True``; identical state dimension);
* ``dense`` - the EM of :class:`nowcastbox.models.MixedFreqDFM` with the dense Kalman
  smoother E-step (``filter_method="univariate"``);
* ``structured`` - the same EM loop with the exact structured smoother
  (:func:`nowcastbox.statespace.smoothed_moments`; ``filter_method="auto"``, the
  default of ``MixedFreqDFM`` since the wave-2 integration).

Per-iteration times are reported (statsmodels: marginal cost of an extra EM iteration,
which excludes its start-up work), plus full ``fit`` times for a fixed number of
iterations. The log-likelihoods of the dense and structured paths are checked to agree.

Usage::

    python benchmarks/bench_em.py                      # N = 200, T = 300
    python benchmarks/bench_em.py --monthly 90 --quarterly 10 --periods 240
    python benchmarks/bench_em.py --no-statsmodels --json results.json

BLAS is limited to ``--threads`` threads (default 1) for every library, so the
comparison measures algorithms, not thread scheduling.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from typing import Any


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--monthly", type=int, default=180, help="monthly series")
    parser.add_argument("--quarterly", type=int, default=20, help="quarterly series")
    parser.add_argument("--periods", type=int, default=300, help="monthly periods")
    parser.add_argument("--iterations", type=int, default=3, help="EM iterations timed")
    parser.add_argument("--threads", type=int, default=1, help="BLAS threads")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-statsmodels", action="store_true", help="skip statsmodels")
    parser.add_argument("--no-dense", action="store_true", help="skip the dense E-step")
    parser.add_argument("--no-fit", action="store_true", help="skip the full fit timings")
    parser.add_argument("--json", default=None, help="write the results to this file")
    return parser.parse_args()


ARGS = _parse() if __name__ == "__main__" else None
if ARGS is not None:  # must precede the numpy import
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = str(ARGS.threads)

import numpy as np
import pandas as pd


def simulate_panel(
    n_monthly: int, n_quarterly: int, n_periods: int, seed: int = 0
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Monthly/quarterly panel driven by one AR(1) factor, with a ragged edge.

    Quarterly series are Mariano-Murasawa aggregates of latent monthly series, stored
    in the third month of each quarter.
    """
    rng = np.random.default_rng(seed)
    n = n_monthly + n_quarterly
    f = np.zeros(n_periods)
    e = np.zeros((n_periods, n))
    for t in range(1, n_periods):
        f[t] = 0.7 * f[t - 1] + rng.standard_normal()
        e[t] = 0.4 * e[t - 1] + 0.8 * rng.standard_normal(n)
    x = f[:, None] * rng.uniform(0.3, 1.0, n) + e
    xq = np.full((n_periods, n_quarterly), np.nan)
    w = np.array([1.0, 2.0, 3.0, 2.0, 1.0]) / 3.0
    for t in range(4, n_periods):
        xq[t] = w @ x[t - np.arange(5), n_monthly:]
    xq[np.arange(n_periods) % 3 != 2] = np.nan
    xq[:12] = np.nan
    xm = x[:, :n_monthly].copy()
    xm[-1, : n_monthly // 2] = np.nan
    xm[-2:, : n_monthly // 4] = np.nan
    xm[:24, : max(1, n_monthly // 18)] = np.nan
    index = pd.period_range("2000-01", periods=n_periods, freq="M")
    columns = [f"m{i}" for i in range(n_monthly)] + [f"q{i}" for i in range(n_quarterly)]
    frame = pd.DataFrame(np.hstack([xm, xq]), index=index, columns=columns)
    return frame, {c: ("Q" if c.startswith("q") else "M") for c in columns}


def _setup(frame: pd.DataFrame, freq: dict[str, str]) -> tuple[Any, Any, Any]:
    from nowcastbox.core.data import MixedFrequencyData
    from nowcastbox.models._init_conditions import pca_initial_parameters
    from nowcastbox.models.em import build_layout

    panel = MixedFrequencyData(frame, freq)
    standardized, _ = panel.standardize()
    layout = build_layout(standardized, 1, 1, None, "ar1")
    params = pca_initial_parameters(standardized, layout)
    return standardized.values, layout, params


def time_em_loop(y: Any, layout: Any, params: Any, iterations: int, fast: bool) -> dict[str, Any]:
    """Time ``iterations`` EM iterations (build model + E-step + M-step)."""
    from nowcastbox.models._em_steps import build_state_space, e_step, m_step

    method = "structured" if fast else "univariate"
    e_step(build_state_space(params, layout), y[:12], method=method)  # JIT warm-up
    lls, times = [], []
    for _ in range(iterations):
        start = time.perf_counter()
        stats = e_step(build_state_space(params, layout), y, method=method)
        params = m_step(stats, layout, y, params)
        times.append(time.perf_counter() - start)
        lls.append(float(stats.loglikelihood))
    return {"seconds_per_iteration": float(np.median(times)), "times": times, "loglik": lls}


def time_statsmodels(frame: pd.DataFrame, freq: dict[str, str], iterations: int) -> dict[str, Any]:
    """Marginal cost of an EM iteration of ``DynamicFactorMQ`` and a full fit."""
    from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ

    quarterly = [c for c, f in freq.items() if f == "Q"]
    monthly = frame.drop(columns=quarterly)
    q = frame[quarterly].copy()
    q.index = q.index.asfreq("Q")
    q = q.dropna(how="all").reindex(pd.period_range(q.index[0], q.index[-1], freq="Q"))
    model = DynamicFactorMQ(
        monthly, endog_quarterly=q, factors=1, factor_orders=1, idiosyncratic_ar1=True
    )
    out: dict[str, Any] = {"k_states": int(model.k_states)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit_em(maxiter=1, disp=False)  # warm-up
        start = time.perf_counter()
        model.fit_em(maxiter=1, disp=False)
        t_one = time.perf_counter() - start
        start = time.perf_counter()
        res = model.fit_em(maxiter=1 + iterations, disp=False)
        t_more = time.perf_counter() - start
    out["fit_seconds_1_iteration"] = t_one
    out[f"fit_seconds_{1 + iterations}_iterations"] = t_more
    out["seconds_per_iteration"] = (t_more - t_one) / iterations
    out["llf"] = float(res.llf)
    return out


def time_fit(
    frame: pd.DataFrame, freq: dict[str, str], iterations: int, filter_method: str
) -> tuple[float, Any]:
    """Wall time of ``MixedFreqDFM(max_iter=iterations, tol=0, filter_method=...).fit``.

    ``filter_method="auto"`` (the default) uses the structured smoother in the E-step
    and in the final smoothing pass; ``"univariate"`` forces the dense Kalman smoother.
    Returns the wall time and the nowcast frame (to check the variants agree).
    """
    from nowcastbox.models import MixedFreqDFM

    target = next(c for c, f in freq.items() if f == "Q")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        start = time.perf_counter()
        res = MixedFreqDFM(
            n_factors=1, max_iter=iterations, tol=0.0, filter_method=filter_method
        ).fit(frame, target=target, frequency=freq)
        return time.perf_counter() - start, res.nowcast


def main(args: argparse.Namespace) -> dict[str, Any]:
    """Run the benchmark and print a summary table."""
    frame, freq = simulate_panel(args.monthly, args.quarterly, args.periods, args.seed)
    y, layout, params = _setup(frame, freq)
    results: dict[str, Any] = {
        "config": {
            "n_monthly": args.monthly,
            "n_quarterly": args.quarterly,
            "n_periods": args.periods,
            "n_states": layout.n_states,
            "iterations": args.iterations,
            "blas_threads": args.threads,
        }
    }
    results["structured"] = time_em_loop(y, layout, params, args.iterations, fast=True)
    if not args.no_dense:
        results["dense"] = time_em_loop(y, layout, params, args.iterations, fast=False)
        diff = np.max(
            np.abs(np.array(results["dense"]["loglik"]) - results["structured"]["loglik"])
        )
        results["max_abs_loglik_difference"] = float(diff)
    if not args.no_statsmodels:
        results["statsmodels"] = time_statsmodels(frame, freq, args.iterations)
    if not args.no_fit:
        n_fit = args.iterations
        sec, now_fast = time_fit(frame, freq, n_fit, "auto")
        results["fit_structured_seconds"] = sec
        if not args.no_dense:
            sec, now_dense = time_fit(frame, freq, n_fit, "univariate")
            results["fit_dense_seconds"] = sec
            num = now_dense.select_dtypes("number")
            results["max_abs_nowcast_difference"] = float(
                np.nanmax(np.abs(now_fast[num.columns].to_numpy() - num.to_numpy()))
            )
    _report(results)
    return results


def _report(results: dict[str, Any]) -> None:
    cfg = results["config"]
    fast = results["structured"]["seconds_per_iteration"]
    lines = [
        f"MixedFreqDFM EM benchmark: N = {cfg['n_monthly']} monthly + {cfg['n_quarterly']} "
        f"quarterly, T = {cfg['n_periods']}, {cfg['n_states']} states, "
        f"BLAS threads = {cfg['blas_threads']}",
        f"{'method':<34}{'s / iteration':>14}{'speed-up':>10}",
        f"{'nowcastbox structured E-step':<34}{fast:>14.3f}{1.0:>10.1f}",
    ]
    labels = {"dense": "nowcastbox dense E-step", "statsmodels": "statsmodels DynamicFactorMQ"}
    for key, label in labels.items():
        if key in results:
            sec = results[key]["seconds_per_iteration"]
            lines.append(f"{label:<34}{sec:>14.3f}{sec / fast:>10.1f}")
    if "max_abs_loglik_difference" in results:
        diff = results["max_abs_loglik_difference"]
        lines.append(f"max |loglik dense - structured| = {diff:.2e}")
    if "fit_structured_seconds" in results:
        n_fit = cfg["iterations"]
        fits = [("MixedFreqDFM.fit (filter_method='auto')", results["fit_structured_seconds"])]
        if "fit_dense_seconds" in results:
            fits.append(("MixedFreqDFM.fit ('univariate', dense)", results["fit_dense_seconds"]))
        if "statsmodels" in results:
            sm_fit = results["statsmodels"][f"fit_seconds_{1 + n_fit}_iterations"]
            fits.append((f"DynamicFactorMQ.fit_em ({1 + n_fit} it.)", sm_fit))
        lines.append(f"full fits ({n_fit} EM iterations unless stated):")
        lines += [f"  {label:<46}{sec:>8.2f} s" for label, sec in fits]
        if "max_abs_nowcast_difference" in results:
            diff = results["max_abs_nowcast_difference"]
            lines.append(f"  max |nowcast difference| vs dense fit = {diff:.2e}")
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    assert ARGS is not None
    output = main(ARGS)
    if ARGS.json:
        with open(ARGS.json, "w", encoding="utf-8") as handle:
            json.dump(output, handle, indent=2)
