# API reference

The reference is generated from the NumPy docstrings of the code (mkdocstrings), so it
is always in sync with the installed version. Every public name is exported by the
`__init__.py` of its subpackage; the most used ones are also available at the top
level (`import nowcastbox as nb; nb.MixedFreqDFM`).

| Page | Subpackage | Content |
|---|---|---|
| [Top level](toplevel.md) | `nowcastbox`, `nowcastbox.api`, `nowcastbox.utils` | `nb.nowcast`, `nb.utils.month_to_quarter` / `quarter_to_month` |
| [Core](core.md) | `nowcastbox.core` | `MixedFrequencyData`, `Frequency`, `AggregationType`, `NowcastResults`, base classes, formulas, exceptions |
| [Preprocessing](preprocessing.md) | `nowcastbox.preprocessing` | transformations (codes 0–7 and named), outliers, missing values, aggregation, `prepare_panel` |
| [State space](statespace.md) | `nowcastbox.statespace` | `StateSpace`, Kalman filter and smoother, structured smoother (I2) |
| [Models](models.md) | `nowcastbox.models` | `TwoStepDFM`, `MixedFreqDFM`, `BridgeEquation` |
| [Selection](selection.md) | `nowcastbox.selection` | Bai-Ng criteria, targeted predictors |
| [Vintages](vintages.md) | `nowcastbox.vintages` | `pseudo_real_time`, `ReleaseCalendar`, `VintageStore` |
| [News](news.md) | `nowcastbox.news` | news decomposition, tracker, level contributions (I6) |
| [Density](density.md) | `nowcastbox.density` | `NowcastDistribution`, bootstrap (I5) |
| [Evaluation](evaluation.md) | `nowcastbox.evaluation` | backtest, metrics, DM/GW/MCS, density scores |
| [Benchmarks](benchmarks.md) | `nowcastbox.benchmarks` | AR, random walk, mean, bridge, U-MIDAS, MIDAS, scikit-learn (I11) |
| [Diagnostics](diagnostics.md) | `nowcastbox.diagnostics` | loading stability, EM convergence, commonality, residuals (I9) |
| [Data sources](data_sources.md) | `nowcastbox.data_sources` | BCB/SGS, IBGE/SIDRA, IPEADATA, FRED/ALFRED, disk cache |
| [Datasets](datasets.md) | `nowcastbox.datasets` | built-in datasets and `Dataset` |
| [Visualization](visualization.md) | `nowcastbox.visualization` | Plotly/Matplotlib charts and themes |
| [Reports](reports.md) | `nowcastbox.reports` | `NowcastReport` (HTML) |
| [Experiment](experiment.md) | `nowcastbox.experiment` | `NowcastExperiment` |
| [Pipeline, CLI and simulation](pipeline.md) | `nowcastbox.pipeline`, `nowcastbox.cli`, `nowcastbox.simulate` | production pipeline (I10), command line |
