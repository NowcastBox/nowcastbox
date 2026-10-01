"""Accuracy metrics of nowcastbox vs the reference fixtures, as markdown tables.

Recomputes the numbers quoted in ``docs/validation/*.md`` from the committed fixtures
(no R needed): maximum absolute errors, correlations and log-likelihood gaps of every
comparison layer. Run from the repository root::

    OMP_NUM_THREADS=1 python3 scripts/reference_fixtures/report_metrics.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nowcastbox.preprocessing import detect_outliers, replace_outliers  # noqa: E402
from nowcastbox.selection import select_factors  # noqa: E402
from nowcastbox.statespace import kalman_smoother, loglikelihood  # noqa: E402
from tests.reference_validation import test_r_em as em  # noqa: E402
from tests.reference_validation import test_r_two_step as ts  # noqa: E402
from tests.reference_validation import test_statsmodels_dfmq as smt  # noqa: E402
from tests.reference_validation._bpanel import correct_series, transform_monthly  # noqa: E402
from tests.reference_validation._helpers import (  # noqa: E402
    abs_correlations,
    codes,
    max_abs,
    max_rel,
    nowcast_estimate,
    r_estimate,
    read_json,
    read_periods,
)


def _row(*cells: Any) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def _e(x: float) -> str:
    return f"{x:.1e}"


def preprocessing() -> list[str]:
    """Codes 0-7 and outlier rule (BRGDP)."""
    raw = read_periods("inputs/brgdp_base.csv.gz")
    ours = transform_monthly(raw, codes("brgdp"))
    ref = read_periods("r/bpanel_transform_brgdp.csv.gz")
    ours = ours[ref.columns]
    flagged = detect_outliers(ours, 4.0, frequency="M")
    emulated = pd.DataFrame(
        {c: correct_series(ours[c].to_numpy(), na_replace=False) for c in ref.columns},
        index=ours.index,
    )
    replaced = replace_outliers(ours, 4.0, window=3, frequency="M")
    scaled = ((replaced - ref).abs() / ours.std()).where(flagged).to_numpy()
    return [
        "| Comparison (BRGDP, 99 series x 216 months) | Value |",
        "|---|---:|",
        _row(
            "codes, non-outlying cells: max rel. error",
            _e(max_rel(ours.where(~flagged), ref.where(~flagged))),
        ),
        _row("outliers flagged (nowcastbox = R)", int(flagged.to_numpy().sum())),
        _row("R rule emulated: max rel. error", _e(max_rel(emulated, ref))),
        _row("outlier cells, |nowcastbox - R| / sd: max", f"{np.nanmax(scaled):.2f}"),
        _row("outlier cells, |nowcastbox - R| / sd: median", f"{np.nanmedian(scaled):.2f}"),
    ]


def selection() -> list[str]:
    """ICfactors cases."""
    from tests.reference_validation.test_r_selection import _panel

    worst = 0.0
    same = 0
    cases = read_json("r/icfactors.json")
    for case in cases.values():
        res = select_factors(
            _panel(case["panel"]), rmax=case["rmax"], criterion=f"IC{case['type']}"
        )
        worst = max(worst, max_abs(res.criteria[f"IC{case['type']}"].to_numpy()[1:], case["IC"]))
        same += res.r_star == case["r_star"]
    return [
        "| ICfactors | Value |",
        "|---|---:|",
        _row("cases (3 panels x rmax 8/15 x IC1-3)", len(cases)),
        _row("identical r*", same),
        _row("max abs error of the criteria", _e(worst)),
    ]


def two_step() -> list[str]:
    """Second step with R parameters and default-pipeline divergence."""
    lines = [
        "| Case | 2nd step: factors | nowcasts | defaults: min corr | defaults: max abs nowcast diff / sd(y) |",
        "|---|---:|---:|---:|---:|",
    ]
    for prefix in ts.SECOND_STEP_CASES:
        c = ts.case(prefix)
        factors, _, prediction = ts._second_step(c)
        rf = read_periods(f"r/{prefix}_factors.csv")
        yf = r_estimate(read_periods(f"r/{prefix}_yfcst.csv", "Q"))
        e_f = max_abs(factors.to_numpy(), rf.reindex(factors.index).to_numpy())
        e_y = max_abs(prediction.reindex(yf.index), yf)
        res = ts._ours(c)
        corr = abs_correlations(res.factors.reindex(rf.index).to_numpy(), rf.to_numpy()).min()
        gap = max_abs(nowcast_estimate(res.nowcast).reindex(yf.index), yf) / float(yf.std())
        lines.append(_row(prefix, _e(e_f), _e(e_y), f"{corr:.5f}", f"{gap:.4f}"))
    return lines


def em_metrics() -> list[str]:
    """EM: smoother at R parameters, log-likelihoods, nowcast path."""
    lines = [
        "| Case | smoother at R params | loglik(R params)+const vs last printed | nowcastbox - R loglik | horizon: max abs diff / sd(y) | iterations |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for prefix in sorted(em.CASES):
        spec, z = em._standardized(prefix)
        sm = kalman_smoother(em.r_model_t0(spec), z)
        x_sm = read_periods(f"r/{prefix}_x_sm.csv.gz")[list(spec["columns"])]
        at_r = loglikelihood(em.r_model_t0(spec), z)
        last = float(np.atleast_1d(spec["loglik_path"]["to"])[-1])
        res = em._fit(prefix)
        ours, ref, scale = em._horizon(prefix, res)
        lines.append(
            _row(
                prefix,
                _e(max_abs(sm.smoothed_signal(), x_sm.to_numpy())),
                f"{at_r + em._constant(z) - last:+.3f}",
                f"{res.loglikelihood - at_r:+.2f}",
                f"{max_abs(ours, ref) / scale:.3f}",
                res.n_iter,
            )
        )
    return lines


def statsmodels_metrics() -> list[str]:
    """MixedFreqDFM vs DynamicFactorMQ."""
    ref, z, freq, _ = smt._setup()
    layout, values = smt._layout_and_data()
    from nowcastbox.models._em_steps import build_state_space

    at_sm = loglikelihood(build_state_space(smt._sm_params(), layout), values)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        start = smt._model(max_iter=0).fit(z, "GDPC1", frequency=freq)
        res = smt._model(tol=1e-6, max_iter=2000).fit(z, "GDPC1", frequency=freq)
    import statsmodels_dfmq as script

    zz, leg = script.load_panel()
    model = script.build_model(zz, leg)
    return [
        "| Quantity | statsmodels | nowcastbox |",
        "|---|---:|---:|",
        _row("log-likelihood at statsmodels' optimum", f"{ref['loglike']:.4f}", f"{at_sm:.4f}"),
        _row(
            "log-likelihood at the starting values",
            f"{model.loglike(model.start_params):.2f}",
            f"{start.loglikelihood:.2f}",
        ),
        _row(
            "log-likelihood at convergence (tol 1e-6)",
            f"{ref['loglike']:.2f}",
            f"{res.loglikelihood:.2f}",
        ),
        _row("EM iterations", ref["n_iter"], res.n_iter),
    ]


def main() -> None:
    """Print every table."""
    sys.path.insert(0, str(REPO / "scripts" / "reference_fixtures"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for title, func in [
            ("Pre-processing", preprocessing),
            ("Selection", selection),
            ("Two-step", two_step),
            ("EM", em_metrics),
            ("statsmodels", statsmodels_metrics),
        ]:
            print(f"\n## {title}\n")
            print("\n".join(func()))


if __name__ == "__main__":
    main()
