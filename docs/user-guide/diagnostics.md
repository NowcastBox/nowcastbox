# DFM diagnostics (innovation I9)

A nowcast is only as good as the model behind it. `nowcastbox.diagnostics` checks fitted
factor models along five dimensions and collects them in a `DiagnosticsReport`:

| Component | Question | Method |
|---|---|---|
| `data_quality` | Missing data, ragged edge, outliers, near-zero loadings, publication delays? | `data_quality_report` |
| `convergence` | Did the EM converge? Monotone log-likelihood? | `em_convergence` |
| `stability` | Are the loadings stable over time? | Breitung & Eickmeier (2011) LM/Wald/LR tests, sup over an unknown break date (Andrews, 1993), multiple-testing correction |
| `contribution` | How much of each series is explained by the factors (commonality, $R^2$)? Which factor matters for which block? | `factor_contributions` |
| `residuals` | Are idiosyncratic and bridge residuals white noise and Gaussian? | Ljung-Box, Jarque-Bera |

## Running every diagnostic

```python
import nowcastbox as nb

ds = nb.load_simulated_dfm()
res = nb.MixedFreqDFM(n_factors=2).fit(ds.data, "gdp")

report = res.diagnostics()                       # = nb.run_diagnostics(res)
print(report.summary())
report.flagged_series
report.series_overview().head()                  # one row per series, every check
```

`run_diagnostics(results, data=None, *, break_date=None, trim=0.15, statistic="lm",
alpha=0.05, correction="holm", components=None, ...)` controls each test;
`components=["stability", "contribution"]` runs a subset. Problems found are also
emitted as `ConvergenceWarning` / `DataQualityWarning` (`warn=False` silences them).

## Components

```python
report.convergence.converged, report.convergence.n_iter
report.stability.to_frame().head()               # statistic, p-value, break date per series
report.stability.multiple_testing                # rejections vs. expected under H0
report.contribution.summary()
report.contribution.by_block
report.residuals.to_frame().head()
report.data_quality.to_frame().head()
frames = report.to_frames()                      # every table, for export
```

## Loading stability at a known date

Test a break at a given date (e.g. the start of the pandemic) instead of the sup over
all dates:

```python
from nowcastbox.diagnostics import loading_stability_test

stab = loading_stability_test(ds.data, res.factors, break_date="2010-01", statistic="wald")
stab.summary()
stab.rejected
```

The tests regress each series on the factors and on the factors interacted with a
post-break dummy; the LM statistic is $n R^2$ of the auxiliary regression of the
residuals (Breitung & Eickmeier, 2011). With `ar_lags > 0` lagged residuals absorb
idiosyncratic autocorrelation. p-values of the sup tests are simulated from Andrews'
(1993) limiting distribution (cached after the first call).

## In the HTML report

`NowcastReport(results, diagnostics=True)` adds a diagnostics section (see
[HTML reports](visualization/reports.md)).
