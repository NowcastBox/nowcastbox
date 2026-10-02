# Comparison with the R package `nowcasting`

The R package [`nowcasting`](https://github.com/nmecsys/nowcasting) (de Valk, de Mattos &
Ferreira, 2019, *The R Journal*; FGV/IBRE, GPL-3, last release 1.1.2 in 2019) made the
two-step and EM factor models accessible to applied economists and inspired NowcastBox.

!!! info "Independent implementation"
    NowcastBox is **not a port**. It is an MIT-licensed implementation written from the
    academic literature, with its own architecture and API. No code of the R package
    was read or translated; the package is only *run* as a black box to produce
    reference numbers for the validation suite (see [Validation](../validation/index.md)).

## Concepts side by side

| Task | R `nowcasting` | NowcastBox |
|---|---|---|
| Estimate a DFM and nowcast | `nowcast(formula, data, r, q, p, method = "2s" / "2s_agg" / "EM", blocks, frequency)` | `TwoStepDFM(aggregate="variables" / "factors")`, `MixedFreqDFM(blocks=...)`, or one call `nb.nowcast(data, target, method=...)` |
| Formula interface | `y ~ .` | `target="y ~ ."`, `"y ~ x1 + x2"`, `"y ~ . - x3"` |
| Stationary transformations | numeric codes 0–7 in a balanced-panel helper | named, composable, invertible transforms (`"dlog"`, `"yoy"`, `Diff(1) \| PctChange(12)`); codes 0–7 accepted |
| Outliers and missing values | replacement of outliers and NAs before estimation | `prepare_panel` (IQR rule, moving median, spline) **or** inside the model (Student-t, automatic outliers, pandemic dummies) |
| Mixed frequencies | monthly + quarterly, `ts` objects with frequency 12 / 4 | `MixedFrequencyData`: daily, weekly, monthly, quarterly, annual with calendar-aware weights, per-series metadata, `PeriodIndex` grid |
| Pseudo real-time data | function based on publication delays (days) | `pseudo_real_time`, `generate_vintages`, `ReleaseCalendar` with explicit release dates |
| Real vintages | relied on a remote MySQL database (no longer available) | `VintageStore` (CSV/Parquet, revision triangles, statistics), Brazilian GDP vintages, ALFRED |
| Number of factors | Bai & Ng (2002) criteria | `select_factors` (IC1–IC3, PC1–PC3, all reported) |
| Number of shocks | Bai & Ng (2007) | `select_shocks` (D1/D2, covariance or correlation matrix) |
| Frequency conversion | month ↔ quarter helpers | `month_to_quarter`, `quarter_to_month`, `to_lower_frequency`, `to_higher_frequency` |
| Plots | forecast, factors, eigenvalues, eigenvectors | `results.plot(kind)` with Plotly and Matplotlib, themes, news waterfalls, trackers, fan charts |
| News decomposition | implemented internally, not exported | `res.news(old, new, period)`: by series, block, category; revisions and re-estimation; tracker; level contributions |
| Evaluation | — | `PseudoRealTimeBacktest`, RMSFE by horizon, DM/GW/MCS, density scores |
| Benchmarks | — | AR, random walk, mean, bridge, U-MIDAS, MIDAS, scikit-learn |
| Other nowcasting models | — | combination of bridge equations, large mixed-frequency Bayesian VAR (`LargeBVAR`, GLP prior, conditional forecasts) |
| Data access | SGS download (HTTP) | BCB/SGS, IBGE/SIDRA, IPEADATA, FRED/ALFRED over HTTPS with cache |
| Datasets | US and Brazilian replication data | rebuilt from primary sources: Brazilian panel (99 series), GDP vintages, release calendar, NY Fed, FRED-MD, GRS-like, simulated |

## Differences in method

These differences are documented choices, not bugs; where the numbers differ from the R
package for the same specification, the validation pages quantify the gap.

- **EM**: lag order not limited by the implementation (the R package limits $p$); block
  structure and aggregation restrictions are generated for any number of factors and any
  fixed frequency ratio; the E-step uses the exact structured smoother (I2); convergence
  uses the relative change of the log-likelihood.
- **Initialisation and missing values**: spline-filled PCA starting values as in Bańbura
  & Modugno (2014), with a multi-start over block orders for nested blocks; interior
  gaps and the ragged edge are distinguished explicitly.
- **Errors**: invalid hyper-parameters raise exceptions instead of warnings.
- **Units**: results are reported in the units of the target (after transformations),
  with standard deviations and density bands.
- **Scope**: robust estimation, long-run mean, densities, diagnostics, backtesting,
  reports and the production pipeline have no counterpart in the R package.

## Migrating a workflow

A typical R workflow — transform a panel, choose $r$ with the Bai-Ng criterion, estimate
with `method = "2s_agg"`, plot the forecast — becomes:

```python
import nowcastbox as nb

ds = nb.load_brazil_nowcast(
    columns=["pib", "ibc_br", "pim_geral", "pmc_varejo", "ipca", "selic"],
    start="2012-01",
    end="2024-12",
)
panel = nb.prepare_panel(ds.data, ds.transform, keep="pib")
ic = nb.select_factors(panel.drop(["pib"]), rmax=4, criterion="IC2")
res = nb.TwoStepDFM(n_factors=ic.r_star, factor_lags=1, aggregate="factors").fit(
    panel, target="pib ~ ."
)
res.nowcast.tail(3)
fig = res.plot("forecast")
```
