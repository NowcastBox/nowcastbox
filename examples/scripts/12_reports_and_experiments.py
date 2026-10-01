"""Notebook 12 - Comparing specifications with NowcastExperiment and writing HTML reports."""

# %% [markdown]
# # 12 · Experiments and reports
#
# Two tools organise the work around the models:
#
# * `NowcastExperiment` fits several specifications on the **same** data, tabulates
#   their nowcasts, runs a pseudo real-time backtest for each and ranks them — the
#   bookkeeping of a model-selection study (modelled on `panelbox.experiment`);
# * `NowcastReport` writes a self-contained **HTML report** for one model: headline
#   nowcast with bands, path, news and nowcast tracker, data flow (ragged edge),
#   backtest accuracy and DFM diagnostics (innovation I9).

# %%
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(HERE.parent))

from utils import OUTPUTS_DIR, brazil_panel, display, md, setup, show

import nowcastbox as nb

setup(blas_threads=1)

# %% [markdown]
# ## 1. Four specifications
#
# On the compact Brazilian panel (GDP + 21 indicators):
#
# | name | model |
# |---|---|
# | `two_step` | two-step DFM (Doz, Giannone & Reichlin, 2011), r = 2, VAR(1), bridge on quarterly factors |
# | `em_global` | EM DFM, one global factor |
# | `em_blocks` | EM DFM, global + real/nominal/financial/soft blocks (NY Fed style) |
# | `em_robust` | `em_blocks` with Student-t idiosyncratic errors (I3) |

# %%
ds, panel = brazil_panel(start="2005-01")
exp = nb.NowcastExperiment(
    panel,
    "pib",
    models={
        "two_step": nb.TwoStepDFM(n_factors=2, factor_lags=1),
        "em_global": nb.MixedFreqDFM(n_factors=1, max_iter=200),
        "em_blocks": nb.MixedFreqDFM(n_factors=1, blocks="data", max_iter=200),
        "em_robust": nb.MixedFreqDFM(
            n_factors=1, blocks="data", idiosyncratic="student_t", max_iter=200
        ),
    },
)
results = exp.fit_all()
print(exp.summary())

# %%
display(exp.compare("2026Q3"))

# %%
display(exp.nowcast_table().tail(6))
fig = exp.plot("nowcasts", backend="matplotlib", n_periods=16)
show(fig, "12_nowcasts")

# %% [markdown]
# The four specifications agree on the broad picture — weak growth in 2026Q3 — but
# differ by a few tenths of a percentage point, the same order as the revision a
# nowcast typically undergoes within a quarter. Which one to trust is an empirical
# question.

# %% [markdown]
# ## 2. Backtesting every specification
#
# `run_backtest()` without a custom hook runs a `PseudoRealTimeBacktest` of each model
# with the given options (here: monthly vintages in 2022–2025 with the actual release
# calendar, parameters re-estimated quarterly).

# %%
backtests = exp.run_backtest(
    calendar=nb.load_brazil_calendar(),
    start="2022-01-15",
    end="2025-12-15",
    step="M",
    refit_every=3,
    n_jobs=4,
)
rmsfe = exp.rmsfe_table()
display((100 * rmsfe).round(3))
fig = exp.plot("rmsfe", backend="matplotlib", relative_to="two_step")
show(fig, "12_rmsfe")

# %%
best = rmsfe.loc[[0, -1]].mean().idxmin()
md(
    f"Averaging the nowcast (`months_to_end = 0`) and backcast (`−1`) horizons, the most "
    f"accurate specification over 2022–2025 is **`{best}`**. With 16 target quarters this "
    "is indicative only: notebook 08 shows how to test the differences "
    "(Diebold-Mariano, Model Confidence Set)."
)

# %% [markdown]
# ## 3. Diagnostics of the chosen model (I9)
#
# `res.diagnostics()` checks the convergence of the EM, the stability of the loadings
# (Breitung & Eickmeier, 2011, sup-LM tests over break dates), the contribution of each
# factor, residual autocorrelation and normality, and data-quality issues.

# %%
res = exp.get_results("em_blocks")
diag = res.diagnostics()
print(diag.summary())

# %%
display(diag.series_overview().head(10))

# %% [markdown]
# **Reading the diagnostics.** The EM converged and the data-quality checks are clean,
# but the loadings of about half the indicators are unstable over 2005–2026 — not
# surprising for an economy that went through the 2014–2016 recession, a change of the
# industrial survey's base year and the pandemic. Many idiosyncratic residuals are
# autocorrelated and fat-tailed even with AR(1) idiosyncratic terms (in part because
# 2020 is in the sample). These flags argue for the robust options of notebook 10 and
# for re-checking the specification periodically; they do not invalidate the nowcast,
# whose out-of-sample accuracy is the ultimate test (section 2).

# %% [markdown]
# ## 4. The HTML report
#
# The report bundles everything a monetary-policy briefing needs. We add the news of the
# last month and the nowcast tracker of the quarter (notebook 07), the predictive
# density (notebook 13) and the backtest of section 2.

# %%
calendar = nb.load_brazil_calendar()
old = nb.pseudo_real_time(panel, calendar=calendar, vintage="2026-09-01")
news = res.news(old, panel, target_period="2026Q3")
tracker = res.nowcast_tracker(panel, calendar, "2026Q3", "2026-07-01", "2026-10-01")
OUTPUTS_DIR.mkdir(exist_ok=True)
path = OUTPUTS_DIR / "12_report_em_blocks.html"
html = exp.report(
    "em_blocks",
    path,
    title="Brazilian GDP nowcast - EM DFM with blocks",
    news=news,
    tracker=tracker,
    quantiles=res.distribution(),
    diagnostics=diag,
    author="nowcastbox example 12",
    plotlyjs="cdn",  # smaller file; "inline" makes it work offline
)
sections = [
    s for s in ("headline", "path", "news", "data_flow", "diagnostics") if f'id="{s}"' in html
]
md(
    f"Report written to `{path.relative_to(OUTPUTS_DIR.parent)}` "
    f"({len(html) / 1024:.0f} KiB) with sections: {', '.join(sections)} (the estimation "
    "and diagnostics card also holds the factor charts and the backtest RMSFE). "
    "Open it in a browser — the charts are interactive Plotly figures."
)

# %% [markdown]
# `NowcastReport` can also be used directly with any results object
# (`nb.NowcastReport(res, news=..., theme="academic").to_html(path)`); `theme` accepts
# the themes of `nb.visualization.list_themes()` and `template_dir` a custom Jinja2
# template.
#
# ### References
#
# * Breitung, J. & Eickmeier, S. (2011). Testing for structural breaks in dynamic factor
#   models. *Journal of Econometrics*, 163(1), 71–84.
# * Doz, C., Giannone, D. & Reichlin, L. (2011). *Journal of Econometrics*, 164(1),
#   188–205.
# * Bańbura, M. & Modugno, M. (2014). *Journal of Applied Econometrics*, 29(1), 133–160.
