"""Notebook 15 - nowcastbox vs statsmodels' DynamicFactorMQ and the R package nowcasting."""

# %% [markdown]
# # 15 · Comparison with statsmodels and the R package `nowcasting`
#
# Two established implementations cover part of what `nowcastbox` does:
#
# * **statsmodels** `DynamicFactorMQ` (Python, BSD) — the mixed-frequency DFM of
#   Bańbura & Modugno (2014) estimated by EM;
# * the **R package `nowcasting`** (de Valk, de Mattos & Ferreira, 2019, GPL-3) — two-step
#   DFM (`"2s"`, `"2s_agg"`), EM (`"EM"`), `Bpanel`, `ICfactors`/`ICshocks`.
#
# `nowcastbox` is an independent MIT implementation written from the papers. statsmodels
# is a dependency and is called **live** below. The R package is only ever **called as a
# black box** (its source is never read): its outputs were stored as reference fixtures
# (`tests/reference_validation/fixtures/`, produced by `scripts/reference_fixtures/`)
# and its run times by `examples/R/15_r_nowcasting.R`. This notebook reads those files
# when present and explains what is missing otherwise.

# %%
import json
import sys
import time
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(HERE.parent))

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ
from utils import EXAMPLES_DIR, OUTPUTS_DIR, SEED, display, md, setup

import nowcastbox as nb

setup(blas_threads=1)  # same BLAS threads for both libraries: compare algorithms
warnings.filterwarnings("ignore", module="statsmodels")  # EM stopping messages (see text)
FIXTURES = EXAMPLES_DIR.parent / "tests" / "reference_validation" / "fixtures"

# %% [markdown]
# ## 1. statsmodels: the same model on the NY Fed panel
#
# One global factor, VAR(1), AR(1) idiosyncratic components, Mariano-Murasawa
# restriction for GDP and unit labour costs, no extra measurement noise
# (`obs_noise_var=0`). Both libraries standardise the data and run EM to a relative
# tolerance of 1e-6.

# %%
ds = nb.load_nyfed()
panel = nb.prepare_panel(
    ds.data, ds.transform, replace_outliers=False, replace_na=False, max_na_prop=1.0
)
quarterly = [c for c in panel.columns if c in panel.quarterly_columns]
monthly_df = panel.data.drop(columns=quarterly)
quarterly_df = panel.data[quarterly].copy()
quarterly_df.index = quarterly_df.index.asfreq("Q")
quarterly_df = quarterly_df.groupby(level=0).last()

start = time.perf_counter()
sm_model = DynamicFactorMQ(
    monthly_df, endog_quarterly=quarterly_df, factors=1, factor_orders=1, idiosyncratic_ar1=True
)
sm_res = sm_model.fit_em(maxiter=500, tolerance=1e-6, disp=False)
sm_seconds = time.perf_counter() - start

start = time.perf_counter()
nb_res = nb.MixedFreqDFM(n_factors=1, factor_lags=1, obs_noise_var=0.0, tol=1e-6, max_iter=500).fit(
    panel, target="GDPC1"
)
nb_seconds = time.perf_counter() - start

sm_factor = sm_res.factors.smoothed.iloc[:, 0]
nb_factor = nb_res.factors["f1"].reindex(sm_factor.index)
both = sm_factor.notna() & nb_factor.notna()
comparison = pd.DataFrame(
    {
        "statsmodels DynamicFactorMQ": {
            "EM iterations": sm_res.mle_retvals["iter"],
            "seconds": round(sm_seconds, 2),
            "log-likelihood (own convention)": round(float(sm_res.llf), 2),
            "nowcast 2017Q1 (% SAAR)": round(
                float(sm_res.predict(start="2017-03", end="2017-03")["GDPC1"].iloc[0]), 3
            ),
        },
        "nowcastbox MixedFreqDFM": {
            "EM iterations": nb_res.n_iter,
            "seconds": round(nb_seconds, 2),
            "log-likelihood (own convention)": round(float(nb_res.loglikelihood), 2),
            "nowcast 2017Q1 (% SAAR)": round(float(nb_res.get_nowcast("2017Q1")), 3),
        },
    }
)
display(comparison)
corr = abs(np.corrcoef(sm_factor[both], nb_factor[both])[0, 1])
md(f"|correlation| of the two smoothed global factors: **{corr:.4f}**.")

# %% [markdown]
# **Reading.** The two implementations deliver practically the same factor (correlation
# ≈ 0.99) and nowcasts about a tenth of a percentage point apart (annualised rates, so
# well within the nowcast uncertainty). Two caveats when comparing the numbers:
#
# * **log-likelihood conventions differ** — statsmodels' `fit_em(...).llf` is not
#   identical to `model.loglike(params)` (see `STATUS.md`, known gaps), and the R package
#   prints the likelihood without the $-\tfrac{n}{2}\log 2\pi$ constant. The reference
#   tests therefore compare each library's optimum *evaluated by the other's likelihood
#   function*: at statsmodels' parameters both functions agree to ~1e-12
#   (`tests/reference_validation/test_statsmodels_dfmq.py`);
# * **stopping rules differ** — in this configuration statsmodels stops after very few
#   EM iterations because its reported likelihood decreases, while `nowcastbox` iterates
#   to the tolerance.
#
# On the four-block NY Fed specification the validation suite also documents a
# **divergence**: the EM has several local maxima. From the block-sequential principal
# components of Bańbura & Modugno (`init="pca_given"`) `nowcastbox` converges about 418
# log-likelihood points below statsmodels; the default `init="pca"` tries every block
# order, starts from the most likely one (a better start than statsmodels') and ends 46
# points below it. Warm-started at statsmodels' optimum, `nowcastbox` reproduces its
# factors and nowcasts.

# %% [markdown]
# ## 2. Speed of an EM iteration (innovation I2)
#
# The cost of an EM iteration is dominated by the Kalman smoother. `nowcastbox` uses an
# exact *structured* smoother that exploits the block structure of the idiosyncratic
# states (and univariate filtering with Numba kernels); statsmodels uses its general
# multivariate smoother. We time a fixed number of iterations on a simulated
# monthly/quarterly panel (one factor, AR(1) idiosyncratic terms, T = 240) of growing
# width, up to N = 120 to keep the notebook within a minute. The marginal timing of
# statsmodels is noisy (it subtracts two runs), so read the orders of magnitude, not
# the decimals.


# %%
def simulate(n_monthly: int, n_quarterly: int, n_periods: int = 240) -> tuple[pd.DataFrame, dict]:
    """One-factor monthly/quarterly panel with Mariano-Murasawa quarterly series."""
    rng = np.random.default_rng(SEED)
    n = n_monthly + n_quarterly
    f, e = np.zeros(n_periods), np.zeros((n_periods, n))
    for t in range(1, n_periods):
        f[t] = 0.7 * f[t - 1] + rng.standard_normal()
        e[t] = 0.4 * e[t - 1] + 0.8 * rng.standard_normal(n)
    x = f[:, None] * rng.uniform(0.3, 1.0, n) + e
    xq = np.full((n_periods, n_quarterly), np.nan)
    w = np.array([1.0, 2.0, 3.0, 2.0, 1.0]) / 3.0
    for t in range(4, n_periods):
        xq[t] = w @ x[t - np.arange(5), n_monthly:]
    xq[np.arange(n_periods) % 3 != 2] = np.nan
    idx = pd.period_range("2000-01", periods=n_periods, freq="M")
    cols = [f"m{i}" for i in range(n_monthly)] + [f"q{i}" for i in range(n_quarterly)]
    frame = pd.DataFrame(np.hstack([x[:, :n_monthly], xq]), index=idx, columns=cols)
    return frame, {c: ("Q" if c.startswith("q") else "M") for c in cols}


def time_both(n_monthly: int, n_quarterly: int, iters: int = 3) -> dict:
    """Seconds per EM iteration of each library (marginal cost of extra iterations)."""
    frame, freq = simulate(n_monthly, n_quarterly)
    qcols = [c for c, f in freq.items() if f == "Q"]
    q = frame[qcols].copy()
    q.index = q.index.asfreq("Q")
    q = q.groupby(level=0).last()
    mod = DynamicFactorMQ(
        frame.drop(columns=qcols),
        endog_quarterly=q,
        factors=1,
        factor_orders=1,
        idiosyncratic_ar1=True,
    )
    t0 = time.perf_counter()
    mod.fit_em(maxiter=1, disp=False)
    t1 = time.perf_counter()
    mod.fit_em(maxiter=1 + iters, disp=False)
    t2 = time.perf_counter()
    sm = ((t2 - t1) - (t1 - t0)) / iters

    def nb_fit(k: int) -> float:
        start = time.perf_counter()
        nb.MixedFreqDFM(n_factors=1, max_iter=k, tol=0.0).fit(
            frame, target=qcols[0], frequency=freq
        )
        return time.perf_counter() - start

    nb_fit(1)  # Numba compilation
    ours = (nb_fit(1 + iters) - nb_fit(1)) / iters
    return {"N": n_monthly + n_quarterly, "statsmodels s/iter": sm, "nowcastbox s/iter": ours}


speed = pd.DataFrame([time_both(n - n // 10, n // 10, iters=2) for n in (20, 60, 120)]).set_index(
    "N"
)
speed["speed-up"] = speed["statsmodels s/iter"] / speed["nowcastbox s/iter"]
display(speed.round(3))

# %% [markdown]
# For small panels statsmodels is as fast or faster — its compiled smoother has little
# overhead, while `nowcastbox` pays Python-level bookkeeping per iteration. The advantage
# of the structured smoother grows with the cross-section: the dense smoother's cost grows
# with the cube of the state dimension (one AR(1) idiosyncratic state per series), the
# structured smoother's roughly linearly in N. The crossover is around N ≈ 50–80; at
# N = 200, T = 300 the benchmark script `benchmarks/bench_em.py` measures ≈ 12× per
# iteration (STATUS.md). Large panels are where innovation I2 matters.

# %% [markdown]
# ## 3. The R package `nowcasting` (reference fixtures)
#
# The fixtures were produced by calling `nowcasting` 1.1.2 in R (`Bpanel`, `nowcast`,
# `ICfactors`, `ICshocks`) on its own example data and on a shared simulated panel.

# %%
r_dir = FIXTURES / "r"
HAVE_R = (r_dir / "nyfed_em.json").exists()
if HAVE_R:
    session = json.loads((r_dir / "session.json").read_text())
    md(
        f"Fixtures found: R {session['r_version']}, nowcasting {session['nowcasting_version']}, "
        f"generated {session['generated']}."
    )
else:
    md(
        "**R fixtures not found** (`tests/reference_validation/fixtures/r/`). Generate them with "
        "the scripts in `scripts/reference_fixtures/` (requires R and the `nowcasting` "
        "package); the comparison cells below are skipped."
    )

# %% [markdown]
# ### 3.1 Number of factors: `ICfactors` vs `select_factors`
#
# On the balanced simulated panel, R's Bai-Ng criteria and `nowcastbox`'s are the same
# formulas; the selected r and the criterion values should coincide.

# %%
if HAVE_R and (r_dir / "icfactors.json").exists():
    cases = json.loads((r_dir / "icfactors.json").read_text())
    sim = pd.read_csv(FIXTURES / "inputs" / "simulated.csv", index_col="period")
    x = sim.drop(columns="gdp").dropna()
    rows = []
    for key, case in cases.items():
        if case["panel"] != "simulated":
            continue
        ours = nb.select_factors(x, rmax=case["rmax"], criterion=f"IC{case['type']}")
        values = ours.criteria[f"IC{case['type']}"].to_numpy()[1:]  # R reports r = 1..rmax
        rows.append(
            {
                "case": key,
                "R r*": case["r_star"],
                "nowcastbox r*": ours.r_star,
                "max |IC difference|": float(np.max(np.abs(values - np.asarray(case["IC"])))),
            }
        )
    display(pd.DataFrame(rows).set_index("case"))
else:
    print("skipped: icfactors.json not available")

# %% [markdown]
# ### 3.2 EM on the NY Fed example: same model, same smoother
#
# The R package returns its final EM parameters. Running them through `nowcastbox`'s
# Kalman smoother on R's standardised panel reproduces R's smoothed data exactly — the
# two libraries implement the same state-space model (R's initial state moments refer to
# the period before the sample).

# %%
if HAVE_R:
    spec = json.loads((r_dir / "nyfed_em.json").read_text())
    cols = list(spec["columns"])
    r_panel = pd.read_csv(r_dir / "nyfed_em_panel.csv.gz", index_col="period")[cols]
    z = (r_panel.to_numpy(float) - np.asarray(spec["Mx"])) / np.asarray(spec["Wx"])
    A = np.asarray(spec["A"], float)
    Q = np.asarray(spec["Q"], float)
    Q = 0.5 * (Q + Q.T)
    V0 = np.asarray(spec["V_0"], float)
    model_r = nb.StateSpace(
        A,
        np.asarray(spec["C"], float),
        Q,
        np.asarray(spec["R"], float),
        initial_state=A @ np.asarray(spec["Z_0"], float),
        initial_state_cov=0.5 * ((A @ V0 @ A.T + Q) + (A @ V0 @ A.T + Q).T),
    )
    smoothed = nb.kalman_smoother(model_r, z).smoothed_signal()
    r_x_sm = pd.read_csv(r_dir / "nyfed_em_x_sm.csv.gz", index_col="period")[cols].to_numpy(float)
    ok = np.isfinite(smoothed) & np.isfinite(r_x_sm)
    md(
        "R's EM parameters through nowcastbox's smoother: max |difference| of the smoothed "
        f"(standardised) data = **{np.max(np.abs(smoothed[ok] - r_x_sm[ok])):.1e}**."
    )
else:
    print("skipped: R EM fixtures not available")

# %% [markdown]
# ### 3.3 Estimation: likelihood, nowcasts and run time

# %%
if HAVE_R:
    legend = pd.read_csv(FIXTURES / "inputs" / "nyfed_legend.csv")
    freq = {
        n: ("Q" if f == 4 else "M")
        for n, f in zip(legend["name"], legend["frequency"], strict=True)
    }
    blocks = legend.set_index("name")[["Global", "Soft", "Real", "Labor"]]
    frame = r_panel.copy()
    frame.index = pd.PeriodIndex(frame.index, freq="M")
    start = time.perf_counter()
    ours = nb.MixedFreqDFM(n_factors=1, factor_lags=1, blocks=blocks).fit(
        frame, "GDPC1", frequency=freq
    )
    ours_seconds = time.perf_counter() - start
    loglik_r = spec["loglik_path"]["to"][-1]
    constant = 0.5 * np.isfinite(z).sum() * np.log(2 * np.pi)
    yfcst = pd.read_csv(r_dir / "nyfed_em_yfcst.csv", index_col="period")
    last = yfcst["y"].last_valid_index()
    ref = yfcst["out"].loc[yfcst.index > last]
    ours_now = ours.nowcast["out_of_sample"].reindex(pd.PeriodIndex(ref.index, freq="Q"))
    timings_file = OUTPUTS_DIR / "R" / "r_timings.csv"
    r_time = (
        float(pd.read_csv(timings_file).set_index("case").loc["nyfed_em", "seconds"])
        if timings_file.exists()
        else float("nan")
    )
    display(
        pd.DataFrame(
            {
                "R nowcasting (EM)": {
                    "log-likelihood (+ 2π constant)": round(loglik_r - constant, 1),
                    "EM iterations (R: last printed)": spec["loglik_path"]["iteration"][-1],
                    f"nowcast {ref.index[0]}": round(float(ref.iloc[0]), 4),
                    "seconds": round(r_time, 1),
                },
                "nowcastbox MixedFreqDFM": {
                    "log-likelihood (+ 2π constant)": round(float(ours.loglikelihood), 1),
                    "EM iterations (R: last printed)": ours.n_iter,
                    f"nowcast {ref.index[0]}": round(float(ours_now.iloc[0]), 4),
                    "seconds": round(ours_seconds, 1),
                },
            }
        )
    )
    if not timings_file.exists():
        print("R run time not available: run `Rscript examples/R/15_r_nowcasting.R`.")
else:
    print("skipped: R EM fixtures not available")

# %% [markdown]
# **Reading.** Starting values and stopping rules differ, so the two EM runs do not end
# at the same parameters (`reference_divergence` in the validation suite): on this
# panel `nowcastbox` reaches a *higher* likelihood in fewer iterations, and the
# nowcasts of 2017Q1 (in the units of R's `Bpanel` transformation) differ by a few
# hundredths of a percentage point. R also reports a
# nowcast (`yfcst`) that is not exactly the smoothed signal of its returned model, so
# nowcasts are compared loosely. The run time differs by one to two orders of
# magnitude (R's implementation is pure R with dense matrix algebra).
#
# The full validation tables — `Bpanel` transformations (exact to 1e-10), Bai-Ng
# criteria (1e-15), two-step DFM with R's parameters (1e-13), `ICshocks` bound
# differences, statsmodels likelihoods — are in `docs/validation/` and
# `tests/reference_validation/`.
#
# ### References
#
# * Bańbura, M. & Modugno, M. (2014). *Journal of Applied Econometrics*, 29(1), 133–160.
# * de Valk, S., de Mattos, D. & Ferreira, P. (2019). Nowcasting: an R package for
#   predicting economic variables using dynamic factor models. *The R Journal*, 11(1),
#   230–244.
# * Koopman, S. J. & Durbin, J. (2000). Fast filtering and smoothing for multivariate
#   state space models. *Journal of Time Series Analysis*, 21(3), 281–296.
# * Seabold, S. & Perktold, J. (2010). statsmodels: econometric and statistical modeling
#   with Python. *Proceedings of the 9th Python in Science Conference*.
