# Reproducing the reference fixtures

The comparisons in this section run in the normal test suite (`tests/reference_validation/`,
markers `reference_validation` and `reference_divergence`) against **committed fixtures**,
so neither R nor statsmodels' long fits are needed at test time. Every test skips
cleanly when a fixture is missing.

## Clean-room rule

The R package [`nowcasting`](https://github.com/nmecsys/nowcasting) is GPL-3; nowcastbox
is MIT. The package is used **only as a black box**: the scripts call its exported
functions on its bundled datasets and store the inputs and outputs. Its source code was
never read. Where a detail of R's behaviour had to be known (for example how `Bpanel`
replaces outliers), it was taken from the help page and identified by comparing
candidate rules with the black-box outputs. statsmodels is used as a numerical reference
only.

## Scripts (`scripts/reference_fixtures/`)

| Script | Produces | Time |
|---|---|---|
| `export_inputs.R` | `fixtures/inputs/` - `USGDP`, `NYFED`, `BRGDP` as CSV (test inputs only, never shipped) | seconds |
| `make_simulated.py` | `fixtures/inputs/simulated.csv` - shared simulated panel (2 factors, 24 monthly series, quarterly target, ragged edge) | seconds |
| `r_preprocessing.R` | `Bpanel` (codes 0-7, defaults, aggregation), `month2qtr`, `qtr2month` | ~15 s |
| `r_selection.R` | `ICfactors`, `ICshocks` (q* grid and bisection thresholds) | ~2 min |
| `r_two_step.R` | `nowcast(method = "2s" / "2s_agg")` on USGDP and the simulated panel | ~1 min |
| `r_em.R` | `nowcast(method = "EM")` on NYFED and the simulated panel (+ printed log-likelihood path) | ~1-2 min |
| `r_prtdb.R` | `PRTDB` on BRGDP / NYFED, daily and monthly vintages | ~1 min |
| `r_timings.R` | `fixtures/r/timings.json` (run on an idle machine) | ~5 min |
| `statsmodels_dfmq.py` | `fixtures/statsmodels/nyfed_dfmq.json` (`DynamicFactorMQ` on a NY Fed-like spec) | ~10 s |
| `report_metrics.py` | prints the accuracy tables of these pages | ~1 min |

Run them from the repository root, in this order:

```bash
Rscript scripts/reference_fixtures/export_inputs.R
python3 scripts/reference_fixtures/make_simulated.py
for s in r_preprocessing r_selection r_two_step r_em r_prtdb; do
  Rscript scripts/reference_fixtures/$s.R
done
OMP_NUM_THREADS=1 python3 scripts/reference_fixtures/statsmodels_dfmq.py
OMP_NUM_THREADS=1 Rscript scripts/reference_fixtures/r_timings.R
OMP_NUM_THREADS=1 python3 scripts/reference_fixtures/report_metrics.py
python3 benchmarks/bench_vs_r.py
```

Requirements: R >= 4 with `nowcasting` 1.1.2 and `jsonlite`. The fixtures were generated with
R 4.5.2 and statsmodels 0.14.6. The fixtures total about 1.6 MB (the budget is 3 MB). To keep
them small, the large R-processed USGDP panels are stored only as column subsets; the tests
rebuild the full panels with `tests/reference_validation/_bpanel.py`, which is validated
against those subsets and against the full BRGDP outputs to about 1e-14.
