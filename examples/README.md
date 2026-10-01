# NowcastBox examples

Executed notebooks (saved **with outputs**, static Matplotlib charts that render on
GitHub), their script twins and helpers. Everything runs **offline** with the datasets
shipped with `nowcastbox`; cells that download data are opt-in
(`NOWCASTBOX_EXAMPLES_NETWORK=1`) and fall back to the shipped data.

| # | Notebook | What you will learn | Data | Runtime* |
|---|---|---|---|---|
| 01 | [Quickstart](notebooks/01_quickstart.ipynb) | First nowcast in ten lines with `nb.nowcast`; the object-oriented route | `load_brazil_nowcast` | ~40 s |
| 02 | [Data preparation](notebooks/02_data_preparation.ipynb) | `MixedFrequencyData`, transformation codes, outliers, missing values and the ragged edge, `prepare_panel`, Mariano-Murasawa | Brazil | ~10 s |
| 03 | [Replicating Giannone, Reichlin & Small (2008)](notebooks/03_replicate_giannone_2008.ipynb) | Two-step DFM, bridge equation, residual ACF 1985–2004, pseudo real-time evaluation 1995–2004 (**approximation**: FRED-MD-based GRS-like panel) | `load_us_grs_like` | ~50 s |
| 04 | [Selecting factors and shocks](notebooks/04_selecting_factors_shocks.ipynb) | Bai-Ng (2002) criteria, Bai-Ng (2007) shocks, sensitivity to `rmax`, `delta`, `m`; targeted predictors | simulated, US, Brazil | ~15 s |
| 05 | [EM with blocks: NY Fed Staff Nowcast](notebooks/05_em_nyfed_replication.ipynb) | `MixedFreqDFM` with global/real/labor/soft blocks, convergence, loadings, weekly news | `load_nyfed` | ~30 s |
| 06 | [Brazilian GDP vintages](notebooks/06_brazil_gdp_vintages.ipynb) | Real IBGE vintages and revisions, release calendar, real-time store, real-time evaluation (first vs latest release) | Brazil + `load_brazil_vintages` | ~60 s |
| 07 | [News decomposition](notebooks/07_news_decomposition.ipynb) | News by series/category/block, revisions, re-estimation, nowcast tracker, level contributions | Brazil | ~55 s |
| 08 | [Pseudo real-time evaluation 2012–2025](notebooks/08_pseudo_real_time_evaluation.ipynb) | Backtest vs AR, RW, bridge, U-MIDAS, MIDAS, scikit-learn ridge; RMSFE by horizon, Diebold-Mariano, MCS, Parquet export | Brazil | ~60 s |
| 09 | [Live Brazilian nowcast](notebooks/09_brazil_live_nowcast.ipynb) | BCB/IBGE/IPEA connectors (**network, opt-in**; offline fallback), robust EM, density, news, HTML report | Brazil (live or shipped) | ~20 s |
| 10 | [COVID robustness](notebooks/10_covid_robustness.ipynb) | Pre-cleaning vs Student-t, outliers in the EM, COVID dummies/mask; time-varying long-run mean | Brazil | ~40 s |
| 11 | [Production pipeline](notebooks/11_production_pipeline.ipynb) | YAML spec, `nowcastbox validate/run/snapshots`, versioned snapshots, production tracker | Brazil | ~30 s |
| 12 | [Experiments and reports](notebooks/12_reports_and_experiments.ipynb) | `NowcastExperiment`, backtests per specification, diagnostics (I9), `NowcastReport` | Brazil | ~35 s |
| 13 | [Density nowcasts](notebooks/13_density_nowcasts.ipynb) | Filtering vs parameter uncertainty (bootstrap), CRPS, log score, PIT, coverage tests | Brazil | ~65 s |
| 14 | [Weekly and daily data](notebooks/14_weekly_daily_data.ipynb) | Weekly base grid, calendar-dependent aggregation (I1), value of timeliness; daily SGS data (**network, opt-in**) | simulated (+ live) | ~30 s |
| 15 | [Comparison with R and statsmodels](notebooks/15_comparison_r_statsmodels.ipynb) | Same model in `DynamicFactorMQ` (live) and R `nowcasting` (reference fixtures): numbers and run time | NY Fed, simulated | ~60 s |

\* single machine, 4 BLAS threads; see each notebook.

## Layout

```
examples/
├── notebooks/        executed notebooks (generated from scripts/)
├── scripts/          the same examples as percent-format Python scripts (source of truth)
├── utils/            helpers (setup, show/display, Brazilian panels, real-time store,
│                     live download) + build_notebooks.py
├── R/                R scripts calling the GPL package `nowcasting` as a black box
├── cheatsheets/      nowcastbox_cheatsheet.md (one page)
└── outputs/          files written by the examples (reports, snapshots, Parquet, figures)
```

## Running

```bash
pip install -e ".[dev]"                     # nowcastbox + jupyter tooling
jupyter lab examples/notebooks              # browse / re-run interactively

python3 examples/scripts/07_news_decomposition.py      # run one example as a script
python3 examples/utils/build_notebooks.py              # regenerate + execute all notebooks
python3 examples/utils/build_notebooks.py 08 13        # only some of them
pytest --nbmake examples/notebooks                     # CI check: every notebook runs

NOWCASTBOX_EXAMPLES_NETWORK=1 jupyter lab ...          # enable the download cells (09, 14)
Rscript examples/R/15_r_nowcasting.R                   # R run times for notebook 15
```

The scripts are the single source of truth: edit `scripts/NN_*.py` (cells start with
`# %%`, Markdown cells with `# %% [markdown]`) and rebuild the notebook. Seeds are fixed
(`utils.SEED`), so re-executions reproduce the same numbers; timings vary by machine.

## Data and licences

All datasets are shipped with `nowcastbox` from open sources (BCB/SGS, IBGE/SIDRA,
IPEADATA, FRED-MD, the BSD-licensed NY Fed replication files); see
`nb.list_datasets()` and `docs/datasets/`. The GRS (2008) panel is **approximated** with
FRED-MD (`load_us_grs_like`): the original files are not available from a primary
source. The R package `nowcasting` (GPL-3) is only *called* to produce reference
numbers; its code is never read or translated.
