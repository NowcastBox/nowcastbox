"""Run time of nowcastbox vs the R package ``nowcasting`` 1.1.2 (and statsmodels).

The R timings are read from ``tests/reference_validation/fixtures/r/timings.json``,
written by ``scripts/reference_fixtures/r_timings.R`` (median of repeated calls of the
black-box R functions on an idle machine); ``--rerun-r`` re-runs that script first
(needs R and the package). The nowcastbox equivalents are timed here on the **same
inputs** (the R-processed panels are rebuilt with the validated emulation of ``Bpanel``
from ``tests/reference_validation/_bpanel.py``):

=========================  ==================================  ===========================
task                       R                                   nowcastbox
=========================  ==================================  ===========================
transformations (BRGDP)    ``Bpanel(NA.replace = FALSE)``      ``apply_transforms``
panel cleaning             ``Bpanel`` (defaults)               ``prepare_panel``
two-step (USGDP, sim.)     ``nowcast(method = "2s"/"2s_agg")``  ``TwoStepDFM``
EM (NYFED, sim.)           ``nowcast(method = "EM")``          ``MixedFreqDFM``
Bai-Ng                     ``ICfactors`` / ``ICshocks``        ``select_factors`` / ``select_shocks``
vintages                   ``PRTDB`` (per vintage)             ``pseudo_real_time``
=========================  ==================================  ===========================

The algorithms differ where documented in ``docs/validation`` (e.g. outlier replacement,
EM starting values and number of iterations), so the ratios compare the tasks, not
identical computations. The statsmodels ``DynamicFactorMQ`` fit time of the NY Fed-like
specification (``tests/reference_validation/fixtures/statsmodels/nyfed_dfmq.json``) is
compared with ``MixedFreqDFM(obs_noise_var=0)``.

Usage (from the repository root)::

    python benchmarks/bench_vs_r.py                   # markdown table
    python benchmarks/bench_vs_r.py --repeat 5 --json bench_vs_r.json
    python benchmarks/bench_vs_r.py --rerun-r          # refresh the R timings first

BLAS is limited to ``--threads`` threads (default 1).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--repeat", type=int, default=3, help="repetitions (median)")
    parser.add_argument("--threads", type=int, default=1, help="BLAS threads")
    parser.add_argument("--rerun-r", action="store_true", help="re-run r_timings.R first")
    parser.add_argument("--only", default=None, help="comma-separated task names")
    parser.add_argument("--json", default=None, help="write the results to this file")
    return parser.parse_args()


ARGS = _parse() if __name__ == "__main__" else None
if ARGS is not None:  # must precede the numpy import
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = str(ARGS.threads)

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd

from nowcastbox.models import MixedFreqDFM, TwoStepDFM
from nowcastbox.preprocessing import apply_transforms, prepare_panel
from nowcastbox.selection import select_factors, select_shocks
from nowcastbox.vintages import pseudo_real_time
from tests.reference_validation._bpanel import emulate_bpanel
from tests.reference_validation._helpers import (
    FIXTURES,
    codes,
    legend,
    pad,
    quarter_end_only,
    read_json,
    read_periods,
)

Task = Callable[[], Any]


def _quiet(func: Task) -> Task:
    def run() -> Any:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return func()

    return run


def build_tasks() -> dict[str, tuple[str, Task]]:
    """Task name -> (description, nowcastbox callable); names match the R timing keys."""
    brgdp = read_periods("inputs/brgdp_base.csv.gz")
    br_codes = codes("brgdp")
    us_raw = read_periods("inputs/usgdp_base.csv.gz")
    us_codes = codes("usgdp")
    x_us = us_raw.drop(columns="RGDPGR")
    x_codes = {k: v for k, v in us_codes.items() if k != "RGDPGR"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        us_filtered = emulate_bpanel(x_us, x_codes, aggregate=True)
        us_plain = emulate_bpanel(us_raw, us_codes)
    us_2s = us_filtered.assign(RGDPGR=quarter_end_only(us_raw["RGDPGR"].reindex(us_filtered.index)))
    us_2s_agg = us_plain.assign(RGDPGR=quarter_end_only(us_plain["RGDPGR"]))
    sim = pad(read_periods("inputs/simulated.csv"))
    ny_panel = read_periods("r/nyfed_em_panel.csv.gz")
    ny_leg = legend("nyfed")
    ny_freq = {
        n: ("Q" if f == 4 else "M")
        for n, f in zip(ny_leg["name"], ny_leg["frequency"], strict=True)
    }
    ny_blocks = ny_leg.set_index("name")[["Global", "Soft", "Real", "Labor"]]
    us_mixed = x_us.copy().assign(RGDPGR=quarter_end_only(us_raw["RGDPGR"]))
    us_bal = us_plain.drop(columns="RGDPGR").dropna()  # the panel given to ICfactors
    vintages = pd.date_range("2008-01-15", "2017-12-15", freq="MS") + pd.Timedelta(days=14)
    br_delay = dict(zip(legend("brgdp")["name"], legend("brgdp")["delay"].astype(int), strict=True))

    def prtdb_like() -> None:
        for v in vintages[:24]:
            pseudo_real_time(brgdp, delay=br_delay, vintage=v, frequency="M")

    q_freq = {"RGDPGR": "Q"}
    tasks: dict[str, tuple[str, Task]] = {
        "bpanel_transform_brgdp": (
            "codes 0-7, BRGDP (100 series)",
            lambda: apply_transforms(brgdp, br_codes, frequency="M"),
        ),
        "bpanel_default_brgdp": (
            "panel cleaning, BRGDP",
            lambda: prepare_panel(brgdp, br_codes, frequency="M"),
        ),
        "bpanel_default_usgdp": (
            "panel cleaning, USGDP (192 series)",
            lambda: prepare_panel(x_us, x_codes, frequency="M"),
        ),
        "bpanel_aggregate_usgdp": (
            "cleaning + MM filter, USGDP",
            lambda: prepare_panel(
                us_mixed,
                {**x_codes, "RGDPGR": 0},
                frequency={**dict.fromkeys(x_codes, "M"), "RGDPGR": "Q"},
                aggregate=True,
                keep="RGDPGR",
            ),
        ),
        "usgdp_2s": (
            "2s, USGDP (r=2, p=2, q=2)",
            lambda: TwoStepDFM(2, 2, 2, aggregate="variables", aggregation=[1.0], horizon=0).fit(
                us_2s, "RGDPGR", frequency=q_freq
            ),
        ),
        "usgdp_2s_agg": (
            "2s_agg, USGDP (r=2, p=2, q=2)",
            lambda: TwoStepDFM(2, 2, 2, aggregate="factors", horizon=0).fit(
                us_2s_agg, "RGDPGR", frequency=q_freq
            ),
        ),
        "simulated_2s_agg": (
            "2s_agg, simulated (24 x 252)",
            lambda: TwoStepDFM(2, 1, 2, horizon=0).fit(sim, "gdp", frequency={"gdp": "Q"}),
        ),
        "simulated_em": (
            "EM, simulated (1 block, r=2)",
            lambda: MixedFreqDFM(n_factors=2).fit(sim, "gdp", frequency={"gdp": "Q"}),
        ),
        "nyfed_em": (
            "EM, NYFED (4 blocks, 53 states)",
            lambda: MixedFreqDFM(n_factors=1, blocks=ny_blocks).fit(
                ny_panel, "GDPC1", frequency=ny_freq
            ),
        ),
        "icfactors_usgdp": (
            "Bai-Ng IC, USGDP (rmax=15)",
            lambda: select_factors(us_bal, rmax=15, criterion="IC2"),
        ),
        "icshocks_usgdp": (
            "Bai-Ng shocks, USGDP (r=4, p=2)",
            lambda: select_shocks(us_bal, n_factors=4, factor_lags=2),
        ),
        "prtdb_brgdp_per_vintage": ("pseudo real-time vintage, BRGDP", prtdb_like),
    }
    return {k: (d, _quiet(f)) for k, (d, f) in tasks.items()}


PER_CALL_DIVISOR = {"prtdb_brgdp_per_vintage": 24}


def time_task(func: Task, repeat: int) -> float:
    """Median wall-clock seconds over ``repeat`` runs (after one warm-up run)."""
    func()  # warm-up: imports, Numba compilation, caches
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        func()
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def statsmodels_comparison(repeat: int) -> dict[str, Any] | None:
    """MixedFreqDFM (no measurement noise) vs the stored DynamicFactorMQ fit time."""
    path = FIXTURES / "statsmodels" / "nyfed_dfmq.json"
    if not path.exists():
        return None
    ref = read_json("statsmodels/nyfed_dfmq.json")
    panel = read_periods("r/nyfed_em_panel.csv.gz").loc[ref["sample"][0] : ref["sample"][1]]
    z = (panel - panel.mean()) / panel.std()
    leg = legend("nyfed")
    freq = {n: ("Q" if f == 4 else "M") for n, f in zip(leg["name"], leg["frequency"], strict=True)}
    blocks = leg.set_index("name")[["Global", "Soft", "Real", "Labor"]]
    model = MixedFreqDFM(n_factors=1, blocks=blocks, obs_noise_var=0.0, tol=1e-6, max_iter=2000)
    holder: dict[str, Any] = {}

    def fit() -> None:
        holder["res"] = model.fit(z, "GDPC1", frequency=freq)

    seconds = time_task(_quiet(fit), max(1, repeat // 2))
    res = holder["res"]
    return {
        "statsmodels_seconds": ref["fit_seconds"],
        "statsmodels_iterations": ref["n_iter"],
        "nowcastbox_seconds": seconds,
        "nowcastbox_iterations": res.n_iter,
        "statsmodels_loglike": ref["loglike"],
        "nowcastbox_loglike": res.loglikelihood,
    }


def main(args: argparse.Namespace) -> dict[str, Any]:
    if args.rerun_r:
        rscript = shutil.which("Rscript")
        if rscript is None:
            raise SystemExit("--rerun-r needs Rscript on the PATH")
        script = REPO / "scripts" / "reference_fixtures" / "r_timings.R"
        subprocess.run([rscript, str(script)], check=True, cwd=REPO)  # noqa: S603 - fixed args
    r_times = read_json("r/timings.json")
    tasks = build_tasks()
    selected = list(tasks) if args.only is None else [t.strip() for t in args.only.split(",")]
    rows = []
    for name in selected:
        description, func = tasks[name]
        ours = time_task(func, args.repeat) / PER_CALL_DIVISOR.get(name, 1)
        r_seconds = r_times.get(name)
        ratio = None if r_seconds is None else float(r_seconds) / ours
        rows.append(
            {
                "task": name,
                "description": description,
                "r_seconds": r_seconds,
                "nowcastbox_seconds": ours,
                "speedup": ratio,
            }
        )
    return {
        "threads": args.threads,
        "repeat": args.repeat,
        "r_timed_on": r_times.get("timed_on"),
        "rows": rows,
        "statsmodels": statsmodels_comparison(args.repeat),
        "numpy": np.__version__,
    }


def _fmt(x: float | None, digits: int = 3) -> str:
    return "-" if x is None else f"{x:.{digits}f}"


def report(results: dict[str, Any]) -> str:
    lines = [
        "| Task | R `nowcasting` (s) | nowcastbox (s) | R / nowcastbox |",
        "|---|---:|---:|---:|",
    ]
    for row in results["rows"]:
        lines.append(
            f"| {row['description']} | {_fmt(row['r_seconds'])} | "
            f"{_fmt(row['nowcastbox_seconds'])} | {_fmt(row['speedup'], 1)}x |"
        )
    sm = results["statsmodels"]
    if sm is not None:
        lines += [
            "",
            "| NY Fed-like EM (no measurement noise, tol 1e-6) | seconds | iterations | log-lik |",
            "|---|---:|---:|---:|",
            f"| statsmodels `DynamicFactorMQ.fit_em` | {sm['statsmodels_seconds']:.2f} | "
            f"{sm['statsmodels_iterations']} | {sm['statsmodels_loglike']:.2f} |",
            f"| nowcastbox `MixedFreqDFM` | {sm['nowcastbox_seconds']:.2f} | "
            f"{sm['nowcastbox_iterations']} | {sm['nowcastbox_loglike']:.2f} |",
        ]
    return "\n".join(lines)


if __name__ == "__main__":
    assert ARGS is not None
    out = main(ARGS)
    print(report(out))
    if ARGS.json:
        Path(ARGS.json).write_text(json.dumps(out, indent=2, default=float))
