# Pipeline and CLI (innovation I10)

!!! note "New in 0.1.0"
    The declarative pipeline (`nowcastbox.pipeline`, re-exported as `nb.run_pipeline`,
    `nb.NowcastSpec` and `nb.SnapshotStore`) and the `nowcastbox` command.

Production nowcasting is a routine: every week new data arrive, the model is re-run, the
nowcast and its news are published and archived. NowcastBox describes that routine in a
**YAML specification** and runs it with one command, keeping a **versioned snapshot** of
every run.

## The specification

```yaml
# nowcast_pib.yaml  ->  nowcastbox run nowcast_pib.yaml
name: brazil_pib
target: pib
data:
  source: brazil_nowcast        # built-in dataset, csv/parquet/excel file or connectors
  start: 2010-01
  vintage: today                # information set: today or a date (pseudo real time)
preprocessing:
  transform: true               # transformations of the dataset legend
  max_na_prop: 0.5
model:
  type: MixedFreqDFM
  factors: {global: 1, real: 1, soft: 1}
  idiosyncratic: student_t      # I3
  long_run_mean: time_varying   # I4
  max_iter: 200
outputs: [nowcast, news, density, diagnostics, report_html]
snapshot_dir: ./snapshots
random_state: 0
```

| Section | Content |
|---|---|
| `target`, `name`, `description` | what is nowcast and the run's name (used for snapshots) |
| `data` | `source` (dataset name, `csv`, `parquet`, `excel` or connector series), sample window, `vintage`, metadata overrides (frequencies, transforms, delays, blocks, categories) |
| `preprocessing` | `prepare_panel` options (`enabled: false` to skip) |
| `model` | `type` (`MixedFreqDFM`, `TwoStepDFM`, `bridge_combination` or `large_bvar`), `factors` (number or per block; not for the bridge combination or the BVAR), and any estimator option |
| `selection` | optional [model building](#model-building) before the nowcast: `preselect` (pre-selection of the indicators) and `search` (specification search with a Covid robustness step) |
| `outputs` | `nowcast`, `news` (against the previous snapshot or a date), `density` (`n_boot`), `diagnostics`, `report_html`, `backtest` (with `metrics` and sub-`periods`), `empirical_bands`, `heatmap`, `alternatives`, `excel` (workbook with the results) |
| `snapshot_dir`, `random_state` | archive location and seed |

Specs are validated before anything runs; errors point to the location in the file
(`model.factors.global: ...`) and suggest close matches for misspelled keys.

## Excel workbooks

!!! note "New in 0.2.0"
    Excel data source, `nowcastbox init --excel` and the `excel` output. Install the
    optional extra: `pip install "nowcastbox[excel]"` (it adds `openpyxl`); without it
    every Excel function raises an `ImportError` that says so.

Many forecasting units keep their data in spreadsheets. As in the templates of the ECB
Nowcasting Toolbox (Linzenich & Meunier, 2024), a workbook holds one sheet per frequency
and one sheet of metadata:

| Sheet | Content |
|---|---|
| `monthly` | column `date` (`2020-01`, or an Excel date), then one column per monthly series |
| `quarterly` | column `date` (`2020Q1`, or any date inside the quarter), then the quarterly series |
| `annual` | column `date` (`2020`), then the annual series (optional sheet) |
| `metadata` | one row per series: `series`, `frequency`, `transform` (code 0–7 or a name such as `dlog`), `delay_days`, `blocks` (`global;real`), `category` (`hard`, `soft`, `financial`, `other`), `description`, `aggregation`, `units`, `transform_applied` |
| `readme` | instructions (ignored when reading) |

Only `series` is required in the metadata sheet; frequencies come from the sheet of each
series (a `frequency` column must agree with it), transformation codes are checked with
`transform_from_code` (0 level, 1 % change, 2 difference, 3 difference of the year-on-year
rate, 4 difference of the annual difference, 5 annual difference, 6 year-on-year rate,
7 quarter-on-quarter rate) and quarterly/annual values are stored in the last month of
their period, as everywhere in NowcastBox. Empty cells are missing values (the ragged
edge). The row order of the metadata sheet sets the column order of the panel.

```python
from nowcastbox.pipeline import NowcastSpec, run_pipeline
from nowcastbox.pipeline.data import example_workbook_path, read_excel_panel, write_excel_panel

panel = read_excel_panel(example_workbook_path())     # bundled example (simulated data)
panel.metadata_frame()[["frequency", "transform", "release_delay", "category"]]

write_excel_panel("my_data.xlsx", panel)              # any monthly panel -> workbook
read_excel_panel("my_data.xlsx").equals(panel)        # exact round trip
write_excel_panel("empty_template.xlsx")              # headers only, to fill by hand
```

In a spec, `source: excel` (or a path ending in `.xlsx`) reads the workbook; `sheets`
renames the sheets of each role (`null` disables one), and the usual overrides
(`columns`, `start`, `delay`, `categories`, ...) apply on top of the metadata sheet:

```python
spec = NowcastSpec.from_dict(
    {
        "name": "excel_demo",
        "target": "gdp",
        "data": {
            "source": "excel",
            "path": "my_data.xlsx",
            "sheets": {"monthly": "monthly", "quarterly": "quarterly", "metadata": "metadata"},
        },
        "vintage": "2019-11-15",          # pseudo real time from the delay_days column
        "preprocessing": False,
        "model": {"type": "TwoStepDFM", "factors": 1},
        "outputs": ["nowcast", {"excel": {"path": "results.xlsx"}}],
    }
)
spec.data.kind, spec.outputs.names
```

The `excel` output exports the run to a workbook (`nowcast`, `loadings`/`factors`,
`density`, `news`, `diagnostics`, `backtest`/`backtest_rmsfe`/`backtest_metrics`,
`empirical_bands`, `heatmap`, `alternatives`/`alternatives_range`, `data` and an `info`
sheet, as available; with a `snapshot_dir` the snapshot also gets `results.xlsx`); `write_run_excel(run, path)` does the same
for any `PipelineRun`:

```python
from nowcastbox.pipeline.data import write_run_excel

run = run_pipeline(spec)
write_run_excel(run, "results_copy.xlsx")
```

## Evaluation and conjunctural outputs

!!! note "New in 0.2.0"
    The outputs below reproduce the evaluation and conjunctural products of the ECB
    Nowcasting Toolbox (Linzenich & Meunier, 2024).

| Output | Options | What it adds |
|---|---|---|
| `backtest` | `metrics` (`rmsfe`, `mse`, `mae`, `bias`, `fda`, `n`), `periods` (`covid`, `ex-covid` or `{label: [first, last]}`) | accuracy table by horizon and sub-period (`run.backtest_metrics`), including the [forecast directional accuracy](evaluation/metrics.md) |
| `empirical_bands` | `method` (`mae`, `rmse`, `quantile`), `window` (`10Y`, `all`), `levels`, `outliers`, `min_errors`, `availability` | [empirical error bands](density/empirical-bands.md) from the backtest's past errors at the same horizon (needs `backtest`) |
| `heatmap` | `by` (`series`, `category`, `block`, `frequency`), `smooth` (`mm`, `none`), `window`, `last` | [z-scores of the indicators](visualization/indicator-heatmap.md) at the vintage (`run.heatmap`) |
| `alternatives` | `by` (`category`, `block`), `drop` (`1`, `[1, 2]`), `refit` | [nowcasts without one or two groups](evaluation/alternative-models.md) (`run.alternatives`) |

The HTML report gains the empirical bands (headline tiles and fan chart), the range of
the alternative nowcasts, the [share of the nowcast period's data already
released](data/released-share.md), the heatmap and the accuracy table; snapshots and the
Excel workbook store the corresponding tables.

```python
spec = {
    "name": "ecb_outputs",
    "target": "gdp",
    "data": {
        "source": "simulated_dfm",
        "columns": ["x01", "x02", "x03", "x04", "x05", "x06"],
        "start": "2008-01",
        "categories": {"x01": "hard", "x02": "hard", "x03": "soft", "x04": "soft",
                       "x05": "financial", "x06": "financial"},
    },
    "vintage": "2019-11-15",
    "preprocessing": False,
    "model": {"type": "TwoStepDFM", "factors": 1},
    "outputs": {
        "backtest": {
            "start": "2015-01-01",
            "end": "2019-10-01",
            "target_offsets": [0],
            "metrics": ["rmsfe", "fda", "n"],
            "periods": {"2015-2017": ["2015Q1", "2017Q4"], "2018-": ["2018Q1", None]},
        },
        "empirical_bands": {"min_errors": 4},
        "heatmap": {"by": "category"},
        "alternatives": {"drop": 1, "refit": False},
        "report_html": {"plotlyjs": "cdn"},
    },
}
run = run_pipeline(spec)
run.backtest_metrics
run.empirical_bands.to_frame()
run.alternatives.range()
```

## Model building

!!! note "New in 0.2.0"
    The `selection` section and `model: {type: bridge_combination}` reproduce the
    model-building workflow of the ECB Nowcasting Toolbox; see
    [Building a model from scratch](model-building.md).

| Key | Options | What it does |
|---|---|---|
| `selection.preselect` | any argument of [`preselect`](selection/preselection.md) (`methods`, `x_lags`, `weights`, `top`, `horizon`, `aggregation`...), `apply` (default `true`) | ranks the indicators at the run's vintage (`run.preselection`); with `apply: true` the model uses the selected indicators only, and the `backtest` output repeats the pre-selection on every vintage (no look-ahead) |
| `selection.search` | `space` (required), `n_draws` (`null` = grid), `ranking`, `backtest` (`start`, `end`, `step`, `refit_every`, `n_vintages`, `benchmarks`...), `score`, `horizon_weights`, `periods`, `normalize`, `metrics`, `max_missing`, `n_jobs`, `checkpoint`, `covid_robustness` (`top`, `treatments`, `evaluate_from`), `apply` (default `false`) | runs a [specification search](selection/specification-search.md) with the spec's model as template (`run.search`, `run.robustness`); with `n_series` in `space` the funnel uses the `preselect` options, recomputed on every vintage; with `apply: true` the best specification (of the robustness step when it runs) becomes the model of the run |
| `model.type: bridge_combination` | options of [`BridgeCombination`](models/bridge-combination.md) (`max_monthly`, `max_quarterly`, `combine`, `mse`, `trim`, `extrapolation`...) | combination of all small bridge equations; `news` and `density` are not available (use `empirical_bands` with a `backtest`); `extrapolation: bvar` completes the indicators with a Bayesian VAR |
| `model.type: large_bvar` (alias `bvar`; new in 0.2.0) | options of [`LargeBVAR`](models/large-bvar.md) (`lags`, `prior`, `prior_mean`, `sum_of_coefficients`, `initial_observation`, `estimate_psi`, `n_draws`...) and `horizon` | large mixed-frequency Bayesian VAR (Cimadomo et al., 2022); `news` works with a quarterly target; `density` uses the posterior mixture when `n_draws > 0` (`n_boot` is rejected) |

The search runs on the vintage data before preprocessing, with the spec's publication
delays and preprocessing applied to every pseudo real-time vintage. A failure of a
selection stage does not stop the run: it is recorded in `run.warnings` and the spec's
model is used. Snapshots and the Excel workbook store the `preselection`, `search` and
`robustness` tables, and the HTML report lists them in its diagnostics section.
`nowcastbox init --template model_building` writes a commented example.

```python
spec = {
    "name": "model_building",
    "target": "gdp",
    "data": {"source": "simulated_dfm"},
    "vintage": "2019-11-15",
    "preprocessing": False,
    "model": {"type": "TwoStepDFM", "factors": 1},
    "selection": {
        "preselect": {"methods": ["tstat", "sis"], "top": 8},
        "search": {
            "space": {"n_factors": [1, 2], "n_series": [4, 8]},
            "n_draws": None,
            "backtest": {"start": "2018-06-15", "end": "2019-06-15"},
            "covid_robustness": {"top": 2, "treatments": ["none", "outliers"]},
            "apply": True,
        },
    },
}
run = run_pipeline(spec)
run.preselection.selected
run.search.table()
run.robustness.pivot()
```

## Command line

```bash
nowcastbox init nowcast_pib.yaml --template brazil_pib   # commented example spec
nowcastbox init --list                                   # brazil_pib, connectors, csv, model_building, simulated
nowcastbox init data.xlsx --excel                        # empty Excel data template
nowcastbox init data.xlsx --excel --example              # filled example workbook
nowcastbox validate nowcast_pib.yaml --check-data
nowcastbox run nowcast_pib.yaml                          # nowcast + outputs + snapshot
nowcastbox run nowcast_pib.yaml --vintage 2024-11-15 --json
nowcastbox snapshots list nowcast_pib.yaml
nowcastbox snapshots history nowcast_pib.yaml            # nowcast path across runs
nowcastbox snapshots diff nowcast_pib.yaml <id1> <id2>
nowcastbox datasets list
```

Exit codes: `0` success, `1` runtime error, `2` invalid specification or usage.

## From Python

The same pipeline runs from Python; `run_pipeline` accepts a spec object, a mapping, a
YAML file or YAML text:

```python
from pathlib import Path

from nowcastbox.pipeline import NowcastSpec, SnapshotStore, run_pipeline, template_text

Path("simulated.yaml").write_text(template_text("simulated"), encoding="utf-8")
spec = NowcastSpec.from_yaml("simulated.yaml")
spec.model

run = run_pipeline(spec)                      # vintage 2019-11-15 of the simulated data
print(run.summary())
run.headline_period, run.headline
run.nowcast.tail(2)
```

A later run archives a new snapshot; the store keeps the history of the nowcast:

```python
later = run_pipeline(spec, vintage="2019-12-20")
store = SnapshotStore(spec.snapshot_dir)
store.to_frame()[["vintage", "headline_period", "headline"]]
store.history()
```

`PipelineRun` exposes the estimation results (`results`), the panel, the news
decomposition, the predictive distribution, the diagnostics, the HTML report path and
the snapshot written. Failures of optional outputs (news, diagnostics, backtest, empirical
bands, heatmap, alternative models, report, Excel workbook) do not stop the run; they are
listed in `run.warnings`.

The command-line entry point is also callable from Python:

```python
from nowcastbox.cli import main

main(["validate", "simulated.yaml"])
main(["snapshots", "list", "simulated.yaml"])
```

## Equivalent code

Every step of the pipeline is an ordinary NowcastBox call; the `brazil_pib` spec above
corresponds to the following code (here with fixed vintages instead of `today`):

```python
import nowcastbox as nb

ds = nb.load_brazil_nowcast(start="2010-01")
panel = nb.prepare_panel(ds.data, ds.transform, max_na_prop=0.5, keep="pib")
vintage = nb.pseudo_real_time(panel, delay=ds.delay, vintage="2025-06-15")
previous_vintage = nb.pseudo_real_time(panel, delay=ds.delay, vintage="2025-05-15")

# factors: {global: 1, real: 1, soft: 1} keeps only these blocks of the metadata
factors = {"global": 1, "real": 1, "soft": 1}
blocks = {c: [b for b in panel.metadata[c].blocks if b in factors] for c in panel.columns}
res = nb.MixedFreqDFM(n_factors=factors, blocks=blocks, idiosyncratic="student_t",
                      long_run_mean="time_varying", max_iter=200).fit(vintage, "pib")
news = res.news(previous_vintage, vintage)
dist = res.distribution(random_state=0)
nb.NowcastReport(res, news=news, quantiles=dist, diagnostics=True).to_html("report.html")
```
