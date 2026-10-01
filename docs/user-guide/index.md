# User guide

The user guide follows the nowcasting workflow. Every page has a runnable example
(they are executed by the test-suite on every commit, `tests/docs/`).

| Step | Pages |
|---|---|
| **Data** | [Mixed-frequency data](data/mixed-frequency-data.md) · [Transformations](data/transformations.md) · [Outliers and missing data](data/outliers-missing.md) · [Aggregation and frequencies](data/aggregation.md) |
| **Vintages** | [Pseudo real-time data](vintages/pseudo-real-time.md) · [Release calendar](vintages/release-calendar.md) · [Real-time vintages](vintages/real-vintages.md) |
| **Models** | [Two-step DFM](models/two-step-dfm.md) · [EM DFM](models/em-dfm.md) · [Blocks](models/blocks.md) · [Robust estimation](models/robust.md) · [Long-run mean](models/long-run-mean.md) · [Bridge equations](models/bridge.md) · [Benchmarks](models/benchmarks.md) |
| **Selection** | [Number of factors](selection/number-of-factors.md) · [Number of shocks](selection/number-of-shocks.md) · [Targeted predictors](selection/targeted-predictors.md) |
| **Explainability** | [News decomposition](news/news-decomposition.md) · [Nowcast tracker](news/tracker.md) · [Level contributions](news/level-contributions.md) |
| **Uncertainty** | [Density nowcasts](density/density-nowcasts.md) · [Scoring densities](density/scoring.md) |
| **Evaluation** | [Backtesting](evaluation/backtesting.md) · [Metrics](evaluation/metrics.md) · [Forecast comparison tests](evaluation/forecast-comparison-tests.md) · [Experiments](evaluation/experiments.md) |
| **Diagnostics** | [DFM diagnostics](diagnostics.md) |
| **Data sources** | [Overview and cache](data-sources/index.md) · [BCB](data-sources/bcb.md) · [IBGE](data-sources/ibge.md) · [IPEADATA](data-sources/ipea.md) · [FRED](data-sources/fred.md) |
| **Output** | [Plots](visualization/plots.md) · [HTML reports](visualization/reports.md) · [Pipeline and CLI](pipeline-cli.md) |
| **Performance** | [Performance](performance.md) |

## Three layers of API

1. **Objects** (scikit-learn/statsmodels style): `Model(**params).fit(data, target)`
   returns an immutable results object.
2. **One call**: `nb.nowcast(data, target, method="em", ...)` chooses the number of
   factors, estimates and optionally adds a density.
3. **Declarative pipeline** (YAML + `nowcastbox run`): see
   [Pipeline and CLI](pipeline-cli.md).

## Conventions used in the examples

- `import nowcastbox as nb`.
- Most examples use the built-in datasets, so they run offline:
  `nb.load_simulated_dfm()` (simulated, known parameters, fast) and
  `nb.load_brazil_nowcast()` (99 Brazilian series). See [Datasets](../datasets/index.md).
- Plots return Plotly figures by default: call `fig.show()` in a notebook or
  `fig.write_html(...)`; pass `backend="matplotlib"` for static figures.
- Warnings are part of the API (`ConvergenceWarning`, `DataQualityWarning`): read them.
