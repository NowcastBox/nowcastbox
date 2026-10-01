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
  source: brazil_nowcast        # built-in dataset, csv/parquet file or connectors
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
| `data` | `source` (dataset name, `csv`, `parquet` or connector series), sample window, `vintage`, metadata overrides (frequencies, transforms, delays, blocks, categories) |
| `preprocessing` | `prepare_panel` options (`enabled: false` to skip) |
| `model` | `type` (`MixedFreqDFM` or `TwoStepDFM`), `factors` (number or per block), and any estimator option |
| `outputs` | `nowcast`, `news` (against the previous snapshot or a date), `density` (`n_boot`), `diagnostics`, `report_html`, `backtest` |
| `snapshot_dir`, `random_state` | archive location and seed |

Specs are validated before anything runs; errors point to the location in the file
(`model.factors.global: ...`) and suggest close matches for misspelled keys.

## Command line

```bash
nowcastbox init nowcast_pib.yaml --template brazil_pib   # commented example spec
nowcastbox init --list                                   # brazil_pib, connectors, csv, simulated
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
the snapshot written. Failures of optional outputs (news, diagnostics, report) do not
stop the run; they are listed in `run.warnings`.

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
