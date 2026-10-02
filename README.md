<div align="center">
  <img src="https://raw.githubusercontent.com/NowcastBox/nowcastbox/main/docs/assets/images/logo.png" alt="NowcastBox logo" width="360">

  <p><strong>Nowcasting with dynamic factor models in Python.</strong></p>
</div>

[![Tests](https://github.com/NowcastBox/nowcastbox/actions/workflows/tests.yml/badge.svg)](https://github.com/NowcastBox/nowcastbox/actions/workflows/tests.yml)
[![Coverage](https://img.shields.io/badge/coverage-99%25-brightgreen)](https://codecov.io/gh/NowcastBox/nowcastbox)
[![PyPI](https://img.shields.io/badge/pypi-not%20released-lightgrey)](https://pypi.org/project/nowcastbox/)
[![Python](https://img.shields.io/badge/python-3.10%E2%80%933.13-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/NowcastBox/nowcastbox/blob/main/LICENSE)
[![Documentation](https://readthedocs.org/projects/nowcastbox/badge/?version=latest)](https://nowcastbox.readthedocs.io/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-261230.svg)](https://github.com/astral-sh/ruff)

NowcastBox estimates the present, the recent past and the near future of low-frequency
macroeconomic variables (GDP) from large mixed-frequency panels with ragged edges. It
covers the whole workflow in one library:

```
data -> transformations -> vintages -> factor selection -> estimation -> nowcast
     -> news decomposition -> density -> real-time evaluation -> diagnostics -> report
```

> **Status: 0.1.0, first public release.** The API is tested on every commit
> (about 3,300 tests, 99 % branch coverage) but names may still change before 1.0.

## Features

| Area | What you get |
|---|---|
| **Models** | Two-step DFM (Giannone, Reichlin & Small, 2008; Doz, Giannone & Reichlin, 2011), aggregating factors or variables; mixed-frequency EM DFM with blocks and AR(1)/iid idiosyncratic components (Bańbura & Modugno, 2014); bridge equations |
| **Data** | `MixedFrequencyData` (daily, weekly, monthly, quarterly and annual series on one grid), named and invertible transformations (codes 0–7 of the R package plus `"dlog"`, `"yoy"`...), outliers, gap filling, flow/stock/average/Mariano-Murasawa aggregation, `prepare_panel` |
| **Selection** | Bai & Ng (2002) factors, Bai & Ng (2007) shocks, targeted predictors (Bai & Ng, 2008), block and variable selection in pseudo real time |
| **Real time** | Release calendars, pseudo real-time vintages, real vintages with revisions (`VintageStore`, Brazilian GDP vintages, ALFRED) |
| **Explainability** | News decomposition by series, block and category, revisions vs. releases, nowcast tracker, contributions to the level |
| **Evaluation** | Pseudo/real-time backtests by horizon, AR, random walk, mean, bridge, U-MIDAS, MIDAS and any scikit-learn regressor as benchmarks, Diebold-Mariano (HLN), Giacomini-White, Model Confidence Set, CRPS, log score, PIT and coverage tests |
| **Production** | YAML pipeline, `nowcastbox` command line, versioned snapshots, HTML reports, `NowcastExperiment` |
| **Data access** | BCB/SGS, IBGE/SIDRA, IPEADATA and FRED/ALFRED connectors with an on-disk cache; seven built-in datasets |

## Innovations

What NowcastBox adds to the R package `nowcasting` and to `statsmodels`' `DynamicFactorMQ`:

| # | Innovation | Status |
|---|---|---|
| I1 | **Arbitrary frequencies**: weekly/daily base grids with calendar-aware aggregation weights (4–5 weeks per month, 12–14 per quarter) | Done for the EM model; two-step/bridge/benchmarks need fixed ratios (M/Q/A) |
| I2 | **Scalable EM**: univariate Kalman filtering, collapsing, an exact structured smoother; ~12× faster per EM iteration than `DynamicFactorMQ` at N = 200 | Done |
| I3 | **Robustness**: Student-t idiosyncratic errors, outlier detection inside the EM, pandemic mask/dummies | Done |
| I4 | **Time-varying long-run mean** of GDP growth (Antolin-Diaz, Drechsel & Petrella, 2017) | Done |
| I5 | **Density nowcasts** with filtering and parameter (bootstrap) uncertainty, scored by CRPS/PIT | Done |
| I6 | **Explainability**: news, tracker, level contributions | Done |
| I7 | **Variable selection**: targeted predictors, block/variable selection validated in pseudo real time | Done |
| I8 | **Real vintages**: `VintageStore`, IBGE GDP vintages, ALFRED | Partial (IBC-Br vintages pending) |
| I9 | **DFM diagnostics**: loading stability (Breitung-Eickmeier), convergence, residuals, data quality | Done |
| I10 | **Production pipeline**: YAML spec, CLI, snapshots, HTML reports | Done |
| I11 | **Interoperability**: scikit-learn benchmarks, Parquet export | Done |

## Installation

```bash
pip install nowcastbox
```

Development version:

```bash
git clone https://github.com/NowcastBox/nowcastbox.git
cd nowcastbox
pip install -e ".[dev]"          # + tests, linters, notebooks tooling
```

Python ≥ 3.10. Core dependencies: numpy, pandas ≥ 2, scipy, numba, statsmodels,
matplotlib, plotly, jinja2, pyyaml, requests, joblib. Extras: `[data]` (Parquet for vintage
stores), `[docs]`, `[test]`. With many CPU cores, small dense matrix work is often faster
with one BLAS thread (`OMP_NUM_THREADS=1`), see the
[performance guide](https://nowcastbox.readthedocs.io/en/latest/user-guide/performance/).

## Quickstart

Nowcast Brazilian GDP growth from a dozen monthly indicators, as of 15 November 2024:

```python
import nowcastbox as nb

COLUMNS = ["pib", "ibc_br", "pim_geral", "pmc_varejo", "pms_volume",
           "energia_consumo_total", "ipca", "selic", "focus_pib"]
ds = nb.load_brazil_nowcast(columns=COLUMNS, start="2012-01", end="2024-12")
panel = nb.prepare_panel(ds.data, ds.transform, keep="pib")   # stationary, cleaned
vintage = nb.pseudo_real_time(panel, delay=ds.delay, vintage="2024-11-15")

res = nb.MixedFreqDFM(n_factors=1).fit(vintage, target="pib")  # Bańbura-Modugno EM
print(res.summary())
res.nowcast.tail(3)          # observed, in_sample, out_of_sample, std, 68/90 % bands

old = nb.pseudo_real_time(panel, delay=ds.delay, vintage="2024-10-15")
news = res.news(old, vintage, target_period="2024Q4")         # what moved the nowcast?
news.to_frame(by="category")
```

One call does preprocessing, the Bai-Ng choice of the number of factors, estimation and
a density nowcast:

```python
quick = nb.nowcast(vintage, target="pib", method="em", density=True)
quick.nowcast.tail(2)[["out_of_sample", "lower_90", "median", "upper_90"]]
```

In production, describe the nowcast in YAML and run it from the command line; every run
writes a versioned snapshot and news against the previous one:

```bash
nowcastbox init nowcast.yaml -t simulated     # example spec (also: brazil_pib, csv, connectors)
nowcastbox run nowcast.yaml                   # nowcast, news, density, report, snapshot
nowcastbox snapshots history ./snapshots      # nowcast path across runs
```

## Datasets

| Loader | Content |
|---|---|
| `load_brazil_nowcast()` | 99 Brazilian series 2003–2026 (BCB, IBGE, IPEA): GDP, IBC-Br, PIM, PMC, PMS, labour, credit, prices, external sector, Focus; transformations, publication delays, blocks, categories |
| `load_brazil_calendar()` | Real release calendar of Brazilian GDP 2010Q1–2026Q2 |
| `load_brazil_vintages()` | 65 real-time vintages of Brazilian quarterly GDP (IBGE) |
| `load_nyfed()` | NY Fed Staff Nowcast replication panel (US) with blocks |
| `load_us_fred_md()` | FRED-MD monthly database (McCracken & Ng, 2016) + FRED-QD real GDP |
| `load_us_grs_like()` | GRS (2008)-like US panel built from FRED-MD |
| `load_simulated_dfm()` | Simulated mixed-frequency DFM with known parameters (also `nb.simulate.dfm`, `nb.simulate.weekly_dfm`) |

## Documentation and examples

- **Documentation** (MkDocs Material): getting started, a user guide with runnable
  examples on every page, theory pages (DFM, Kalman, EM, Mariano-Murasawa, Bai-Ng, news,
  density scoring, forecast evaluation), comparisons with R `nowcasting` and
  `DynamicFactorMQ`, validation reports and the API reference — `mkdocs serve`.
- **15 example notebooks** in [`examples/notebooks`](https://github.com/NowcastBox/nowcastbox/tree/main/examples/notebooks) (saved with
  outputs): quickstart, data preparation, a GRS (2008) approximation, Bai-Ng, the NY Fed
  EM with blocks, Brazilian vintages, news, a 2012–2025 pseudo real-time evaluation,
  live data, COVID robustness, the production pipeline, reports, density nowcasts,
  weekly data and the comparison with R/statsmodels.

## Validation

The classical methods are checked against reference implementations run as black boxes
(`tests/reference_validation/`, [`docs/validation`](https://nowcastbox.readthedocs.io/en/latest/validation/)):

| Component | Reference | Result |
|---|---|---|
| Transformations 0–7, Mariano-Murasawa filter, `month2qtr`/`qtr2month` | R `nowcasting` 1.1.2 | identical to ≤ 1e-10 |
| Bai-Ng (2002) criteria | R `ICfactors` | same r* in 18/18 cases, criteria to 6e-15 |
| Two-step DFM with R's parameters | R `nowcast("2s"/"2s_agg")` | factors, bridge and nowcasts to ~1e-12 |
| EM smoother with R's final parameters | R `nowcast("EM")` | factors and smoothed data to 1e-13 |
| Pseudo real-time vintages | R `PRTDB` | identical in 407 vintages (incl. negative delays) |
| Kalman filter, likelihood, smoothed factors | statsmodels `DynamicFactorMQ` | log-likelihood to 1e-12, factors to < 1e-6 |

Documented divergences (different defaults, starting values or stopping rules, and a few
problems on the R side) are quantified in the validation pages. On the NY Fed EM,
NowcastBox runs about 30× faster than R; an EM iteration at N = 200 is about 12× faster
than `DynamicFactorMQ`.

## Related work

NowcastBox is an independent, MIT-licensed implementation written from the academic
literature. It is *inspired by* the R package
[`nowcasting`](https://github.com/nmecsys/nowcasting) (de Valk, de Mattos & Ferreira,
2019, *The R Journal*), and relates to `statsmodels.tsa.DynamicFactorMQ` and the FRBNY
Staff Nowcast code; these are used only as external benchmarks of results (the GPL code
of the R package was never read).

## Citation

If you use NowcastBox, please cite it (see [`CITATION.cff`](https://github.com/NowcastBox/nowcastbox/blob/main/CITATION.cff)):

```bibtex
@software{nowcastbox,
  author  = {Haase, Gustavo and Sanches, Alexandre Le{\~a}o},
  title   = {{NowcastBox}: Nowcasting with Dynamic Factor Models in Python},
  year    = {2026},
  version = {0.1.0},
  license = {MIT},
  url     = {https://github.com/NowcastBox/nowcastbox}
}
```

## Authors

- **Gustavo Haase** (gustavo.haase@gmail.com)
- **Alexandre Leão Sanches** (a.leaosanches@gmail.com)

Programa de Pós-Graduação em Economia — Universidade Católica de Brasília (UCB).

## License

[MIT](https://github.com/NowcastBox/nowcastbox/blob/main/LICENSE). Bundled datasets keep the licenses of their sources (listed in each
dataset's metadata and documentation page).
