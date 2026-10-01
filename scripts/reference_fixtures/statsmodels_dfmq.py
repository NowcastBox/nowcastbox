"""Reference fit of statsmodels' ``DynamicFactorMQ`` on a NY Fed-like specification.

Specification (Bańbura & Modugno, 2014; Bok et al., 2018): the NYFED panel transformed
by ``Bpanel`` (fixture ``r/nyfed_em_panel.csv.gz``, written by ``r_em.R``), standardised
once (mean / standard deviation over the observed values, ``ddof=1``), four blocks
(Global, Soft, Real, Labor) with one factor each, VAR(1) factor dynamics, AR(1)
idiosyncratic components and the Mariano-Murasawa restriction for the quarterly series.
This is the model of ``MixedFreqDFM(n_factors=1, factor_lags=1, blocks=...,
idiosyncratic="ar1", obs_noise_var=0)``.

Writes ``tests/reference_validation/fixtures/statsmodels/nyfed_dfmq.json`` (parameters,
``loglike(params)``, smoothed block factors and the smoothed quarterly signals) and the
fit time. statsmodels is used only as a numerical reference (never copied).

Run from the repository root (about 10-20 minutes, single BLAS thread)::

    OMP_NUM_THREADS=1 python3 scripts/reference_fixtures/statsmodels_dfmq.py
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels
from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ

FIXTURES = Path("tests/reference_validation/fixtures")
OUT = FIXTURES / "statsmodels" / "nyfed_dfmq.json"
BLOCK_NAMES = ("Global", "Soft", "Real", "Labor")
SAMPLE = ("1985-02", "2017-01")  # rows with at least one observation
MAX_ITER = 1000
TOLERANCE = 1e-6


def load_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Standardised NYFED panel (monthly grid) and its legend."""
    frame = pd.read_csv(FIXTURES / "r" / "nyfed_em_panel.csv.gz")
    frame.index = pd.PeriodIndex(frame.pop("period"), freq="M")
    frame = frame.loc[SAMPLE[0] : SAMPLE[1]]
    legend = pd.read_csv(FIXTURES / "inputs" / "nyfed_legend.csv")
    z = (frame - frame.mean()) / frame.std(ddof=1)
    return z, legend


def build_model(z: pd.DataFrame, legend: pd.DataFrame) -> DynamicFactorMQ:
    """DynamicFactorMQ with the NY Fed block structure."""
    quarterly = [n for n, f in zip(legend["name"], legend["frequency"], strict=True) if f == 4]
    monthly = [c for c in z.columns if c not in quarterly]
    blocks = {
        name: [b for b in BLOCK_NAMES if int(row[b]) == 1]
        for name, row in legend.set_index("name").iterrows()
    }
    zq = z[quarterly]
    zq = zq[zq.index.month % 3 == 0]
    zq.index = zq.index.asfreq("Q")
    return DynamicFactorMQ(
        z[monthly],
        endog_quarterly=zq,
        factors=blocks,
        factor_orders=1,
        idiosyncratic_ar1=True,
        standardize=False,
    )


def main() -> None:
    """Fit and write the fixture."""
    z, legend = load_panel()
    model = build_model(z, legend)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        started = time.perf_counter()
        res = model.fit_em(maxiter=MAX_ITER, tolerance=TOLERANCE, disp=False)
        seconds = time.perf_counter() - started
    params = res.params
    # smoothed quantities AT the returned parameters (fit_em's own results hold those of
    # the last E-step, i.e. of the previous parameters)
    at_params = model.smooth(params)
    smoothed = np.asarray(at_params.smoothed_state)  # (k_states, nobs)
    design = np.asarray(model.ssm["design"])
    signal = (design @ smoothed).T  # (nobs, k_endog) in standardised units
    endog_names = list(model.endog_names)
    n_obs = z.shape[0]
    factor_names = [str(f) for f in model.factor_names]
    smoothed_factors = at_params.factors.smoothed  # DataFrame, one column per factor
    factor_cols = {
        name: np.asarray(smoothed_factors[name], dtype=float)[:n_obs].tolist()
        for name in factor_names
    }
    quarterly = [n for n in endog_names if n in set(legend.loc[legend.frequency == 4, "name"])]
    payload = {
        "statsmodels_version": statsmodels.__version__,
        "sample": list(SAMPLE),
        "periods": [str(p) for p in z.index],
        "columns": list(z.columns),
        "endog_names": endog_names,
        "max_iter": MAX_ITER,
        "tolerance": TOLERANCE,
        "n_iter": int(res.mle_retvals.get("iter", -1)) if res.mle_retvals else -1,
        "fit_seconds": seconds,
        "params": {k: float(v) for k, v in params.items()},
        "loglike": float(model.loglike(params)),
        "llf_reported": float(res.llf),
        "factor_names": factor_names,
        "smoothed_factors": factor_cols,
        "smoothed_quarterly_signal": {
            name: signal[:n_obs, endog_names.index(name)].tolist() for name in quarterly
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1))
    print(f"wrote {OUT}: loglike={payload['loglike']:.4f}, {seconds:.1f}s")


if __name__ == "__main__":
    main()
