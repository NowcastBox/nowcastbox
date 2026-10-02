# Comparison with `statsmodels` `DynamicFactorMQ`

`statsmodels.tsa.statespace.dynamic_factor_mq.DynamicFactorMQ` (statsmodels ≥ 0.12)
estimates mixed-frequency dynamic factor models by EM following Bańbura & Modugno (2014),
with factor blocks, AR(1) idiosyncratic components and news. It is an excellent
general-purpose implementation and NowcastBox uses it as a **numerical reference** in its
test-suite (filter, smoother and EM steps are compared on identical models) and as a
performance benchmark. No statsmodels code is copied.

## Concepts side by side

| Concept | `DynamicFactorMQ` | NowcastBox |
|---|---|---|
| Data | monthly and quarterly DataFrames passed separately (`endog`, `endog_quarterly`) | one `MixedFrequencyData` with per-series metadata (frequency, delays, blocks, category, aggregation) |
| Factor blocks | `factors={series: [blocks]}`, `factor_orders`, `factor_multiplicities` | `blocks=` (mapping, DataFrame, `"data"`), `n_factors={block: r}`, `factor_lags` |
| Idiosyncratic | `idiosyncratic_ar1=True/False` | `idiosyncratic="ar1" / "iid" / "student_t"` |
| Aggregation of quarterly series | Mariano-Murasawa | Mariano-Murasawa, flow, average, stock; any fixed ratio (M/Q/A) |
| Estimation | `fit()` (EM) | `fit()` (EM, structured E-step, warm starts) |
| Standardisation | `standardize=True` | always; results in the units of the target |
| News | `results.news(comparison, impact_date=...)` | `res.news(old, new, period)`: by series/block/category, revisions vs. new releases, re-estimation; tracker; level contributions |
| Two-step estimator | — | `TwoStepDFM` (Giannone, Reichlin & Small, 2008) |
| Number of factors / shocks | — | Bai & Ng (2002, 2007) |
| Robust estimation, long-run mean | — | Student-t, outliers, pandemic options (I3); random-walk mean (I4) |
| Density nowcasts | Gaussian forecast intervals | Gaussian + bootstrap parameter uncertainty, CRPS / PIT (I5) |
| Vintages, backtesting, benchmarks | — | `pseudo_real_time`, `VintageStore`, `PseudoRealTimeBacktest`, AR/MIDAS/scikit-learn |
| Diagnostics, reports, pipeline | — | loading stability, convergence, HTML reports, YAML pipeline |
| Other nowcasting models | — | combination of bridge equations, large mixed-frequency Bayesian VAR (`LargeBVAR`) |

## Performance

On the same model (180 monthly + 20 quarterly series, 300 months, one factor, AR(1)
idiosyncratic components, 285 states, one BLAS thread), one EM iteration takes about
**0.40 s** in NowcastBox (structured smoother) against **4.8 s** for
`DynamicFactorMQ.fit_em`: about **12× faster**, with identical log-likelihoods for the
dense and structured paths (to $10^{-10}$). See [Performance](../user-guide/performance.md)
and `benchmarks/bench_em.py`.

## The same model in both libraries

```python
import pandas as pd
import statsmodels.api as sm

import nowcastbox as nb

ds = nb.load_simulated_dfm()
frame = ds.data.to_frame()
monthly = frame.drop(columns="gdp")                  # monthly PeriodIndex
quarterly = ds.data.to_native("gdp").to_frame()      # quarterly PeriodIndex

# statsmodels
mod = sm.tsa.DynamicFactorMQ(monthly, endog_quarterly=quarterly, factors=1,
                             factor_orders=1, idiosyncratic_ar1=True)
sm_res = mod.fit(disp=False)

# nowcastbox
nb_res = nb.MixedFreqDFM(n_factors=1, factor_lags=1, idiosyncratic="ar1").fit(ds.data, "gdp")
nb_res.loglikelihood, sm_res.llf
```

Estimates differ slightly because of starting values, the treatment of the initial
state in the M-step and the convergence rule; on identical parameters the filtered and
smoothed quantities coincide.
