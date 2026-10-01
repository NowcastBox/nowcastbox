"""Notebook 11 - Production: YAML spec, CLI, versioned snapshots and the nowcast tracker."""

# %% [markdown]
# # 11 · A production nowcasting pipeline: YAML, CLI and snapshots
#
# Research code answers "what is the nowcast?"; production code must also answer
# "what was the nowcast last week, with which data and which model, and why did it
# change?". Innovation I10 of `nowcastbox` is a **declarative pipeline** (plan §6.2):
#
# * a **YAML spec** describes data, vintage, preprocessing, model and outputs;
# * the **CLI** `nowcastbox run spec.yaml` executes it (e.g. from `cron` after each data
#   release);
# * every run writes a **versioned snapshot** (spec, data hash, nowcast table,
#   parameters, density, news against the previous snapshot, HTML report);
# * the snapshot history is the **nowcast tracker** of the production system (I6).
#
# > **Status.** The `nowcastbox.pipeline` module and the `run`/`snapshots` commands were
# > being finalised in parallel with these notebooks; this notebook follows the spec of
# > plan §6.2. If the module is unavailable in your installation, the cells print an
# > explanation instead of failing.

# %%
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(HERE.parent))

import matplotlib.pyplot as plt
import pandas as pd
from utils import OUTPUTS_DIR, display, md, setup, show

import nowcastbox as nb

setup()
PIPELINE = importlib.util.find_spec("nowcastbox.pipeline.runner") is not None
WORKDIR = OUTPUTS_DIR / "11_pipeline"
shutil.rmtree(WORKDIR, ignore_errors=True)  # start from an empty snapshot directory
WORKDIR.mkdir(parents=True)


def cli(*args: str) -> None:
    """Run ``nowcastbox <args>`` in WORKDIR and print its output."""
    if not PIPELINE:
        print(f"[nowcastbox {' '.join(args)}] pipeline module not available in this install")
        return
    cmd = [sys.executable, "-m", "nowcastbox.cli", *args]
    out = subprocess.run(cmd, cwd=WORKDIR, capture_output=True, text=True, check=False)  # noqa: S603
    text = f"{out.stdout}{out.stderr}".replace(str(WORKDIR), ".")  # relative paths
    print(f"$ nowcastbox {' '.join(args)}\n{text}".rstrip())


md(f"Pipeline module available: **{PIPELINE}**; working directory `{WORKDIR.name}/`.")

# %% [markdown]
# ## 1. The spec
#
# The YAML below is the plan's example (§6.2), adapted to a compact panel so the
# notebook runs in seconds: the target, the data source (a built-in dataset; CSV/Parquet
# files and the BCB/IBGE/IPEA/FRED connectors are also accepted), the information set
# (`vintage`), the model with its blocks and the robust/long-run-mean options (I3, I4),
# the outputs and the snapshot directory.

# %%
SPEC = """\
name: brazil_pib
description: Brazilian GDP (QoQ, SA) - compact panel, blocks, robust EM.
target: pib

data:
  source: brazil_nowcast          # built-in dataset or bcb/ibge/ipea/fred connectors
  columns: [pib, ibc_br, pim_geral, pim_transformacao, pmc_varejo, pmc_ampliado,
            ipca, selic, icbr, credito_saldo_total, focus_pib]
  start: 2005-01
  vintage: today                  # or a date: pseudo real-time information set

preprocessing:
  transform: true                 # legend transformations (qoq, dlog, ...)

model:
  type: MixedFreqDFM
  factors: {global: 1, real: 1, soft: 1}
  idiosyncratic: student_t        # I3
  long_run_mean: time_varying     # I4
  max_iter: 100

outputs: [nowcast, news, density, report_html]
snapshot_dir: ./snapshots
random_state: 0
"""
(WORKDIR / "nowcast_pib.yaml").write_text(SPEC, encoding="utf-8")
cli("validate", "nowcast_pib.yaml")

# %% [markdown]
# `validate` parses the spec, checks every field (unknown keys, invalid model options,
# missing series) and reports errors with their location, without estimating anything.
# `nowcastbox init --list` shows the bundled templates (built-in data, CSV files,
# connectors, simulated data).

# %% [markdown]
# ## 2. Three months of production
#
# In production the spec is run after each relevant release. Here we replay the third
# quarter of 2026 with three runs, overriding the vintage on the command line
# (`--vintage`), as a scheduler would do on 15 July, 15 August and 15 September.

# %%
for vintage in ("2026-07-15", "2026-08-15", "2026-09-15"):
    cli("run", "nowcast_pib.yaml", "--vintage", vintage)
    print()

# %% [markdown]
# Each run prints the headline nowcast with its 68 %/90 % bands, the snapshot written
# and — from the second run on — the **news** against the previous snapshot: the change
# of the nowcast split into the effect of new releases, data revisions and
# re-estimation, with the largest contributions by series (Bańbura & Modugno, 2014).

# %% [markdown]
# ## 3. Snapshots
#
# Snapshots are plain directories (`manifest.json` + CSV files + `report.html`), easy to
# archive, diff and audit.

# %%
cli("snapshots", "list", "snapshots")

# %%
cli("snapshots", "diff", "snapshots")

# %% [markdown]
# ## 4. The same from Python
#
# The CLI is a thin layer over `nowcastbox.pipeline`: `run_pipeline` accepts a spec file,
# a mapping or a `NowcastSpec`, and `SnapshotStore` reads the snapshot directory.

# %%
if PIPELINE:
    from nowcastbox.pipeline import NowcastSpec, SnapshotStore, run_pipeline

    spec = NowcastSpec.from_yaml(WORKDIR / "nowcast_pib.yaml")
    run = run_pipeline(  # the 1 October run, from Python: a fourth snapshot
        spec, snapshot_dir=WORKDIR / "snapshots", vintage="2026-10-01"
    )
    print(run.summary().replace(str(WORKDIR), "."))
else:
    print("nowcastbox.pipeline not available: skipped.")

# %% [markdown]
# ## 5. The production nowcast tracker
#
# The snapshot history is the record of what the system said at each date. For the
# third quarter of 2026:

# %%
if PIPELINE:
    store = SnapshotStore(WORKDIR / "snapshots")
    history = store.history(period="2026Q3")
    display(history[["vintage", "period", "value"]])
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(pd.to_datetime(history["vintage"]), 100 * history["value"], "o-")
    ax.set_ylabel("nowcast of 2026Q3, % QoQ")
    ax.set_title("Production nowcast tracker (one point per snapshot)")
    show(fig, "11_history")

# %% [markdown]
# The estimate for 2026Q3 fell from about +0.7 % in mid-July (when the quarter had just
# started and the model leaned on its unconditional mean and the Focus survey) to
# around zero in mid-September, as the June and July industrial production figures —
# the dominant news items in the run logs above — came in weak.
#
# Between snapshots the nowcast can also be tracked **release by release** with the
# tracker of notebook 07 (`res.nowcast_tracker(...)`), which the report of each run
# includes when the `news` output is requested.

# %% [markdown]
# ## 6. Scheduling
#
# A typical deployment runs the spec every working day at 9 a.m., after the overnight
# releases, and publishes the latest report:
#
# ```cron
# 0 9 * * 1-5  cd /srv/nowcast && nowcastbox run nowcast_pib.yaml --quiet \
#              && cp "$(ls -d snapshots/*/ | tail -1)report.html" /var/www/nowcast/index.html
# ```
#
# With `vintage: today` the information set is whatever the data source returns on that
# day; with connectors (`source: bcb`, `ibge`, `ipea`, `fred`) the data are downloaded
# and cached; snapshots record the data hash, so any published number can be traced
# back to its exact inputs.
#
# ### References
#
# * Bańbura, M. & Modugno, M. (2014). *Journal of Applied Econometrics*, 29(1), 133–160.
# * Bok, B., Caratelli, D., Giannone, D., Sbordone, A. M. & Tambalotti, A. (2018).
#   Macroeconomic nowcasting and forecasting with big data. *Annual Review of
#   Economics*, 10, 615–643 (the NY Fed's weekly production cycle).

# %%
_ = nb.__version__  # the notebook was executed with this nowcastbox version
md(f"Executed with nowcastbox {nb.__version__}.")
