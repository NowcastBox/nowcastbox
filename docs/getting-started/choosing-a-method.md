# Choosing a method

| Situation | Recommended | Why |
|---|---|---|
| First look at a large monthly panel, quarterly target | `TwoStepDFM` (`aggregate="factors"`) | Fast (PCA + Kalman), few tuning choices, consistent for large N and T (Doz, Giannone & Reichlin, 2011) |
| Many quarterly series, arbitrary missing patterns, short histories | `MixedFreqDFM` (EM) | Maximum likelihood with any missing pattern; quarterly series load on the monthly factors through aggregation restrictions (Bańbura & Modugno, 2014) |
| Groups of series with their own co-movement (real, nominal, financial, soft) | `MixedFreqDFM(blocks=...)` | Block-specific factors; news and contributions by block |
| Pandemic period or fat-tailed data | `MixedFreqDFM(idiosyncratic="student_t")`, `outliers="auto"`, `covid="dummy"` | Robust estimation without dropping the sample (I3) |
| Drift in trend growth (e.g. Brazil after 2014) | `MixedFreqDFM(long_run_mean="time_varying")` | Avoids a biased nowcast when potential growth changes (I4) |
| A few strong indicators | `BridgeEquation` | Transparent OLS on aggregated indicators |
| Benchmarks for evaluation | `AR`, `RandomWalk`, `HistoricalMean`, `BridgeBenchmark`, `UMIDAS`, `MIDAS`, `SklearnBenchmark` | Required to judge whether the factor model adds value |
| One call with sensible defaults | `nb.nowcast(...)` | Bai-Ng selection of the number of factors, EM or two-step, optional density |

## Two-step versus EM

| | Two-step | EM |
|---|---|---|
| Estimation | PCA on the balanced part, VAR on the factors, one Kalman smoother pass | Iterates Kalman smoothing (E-step) and closed-form updates (M-step) to the ML estimate |
| Lower-frequency series | Bridged through aggregated factors (`"factors"`) or pre-filtered variables (`"variables"`) | In the state space, with exact aggregation restrictions |
| Missing data in estimation | Needs a balanced block for PCA | Arbitrary pattern |
| Blocks | No | Yes |
| Robust options, long-run mean | No | Yes |
| News decomposition | Yes (`"variables"`: on the filtered predictors) | Yes |
| Weekly / daily base grid (I1) | No (fixed-ratio pairs only) | Yes (calendar-aware aggregation) |
| Speed | Fastest | Fast with the structured E-step (I2) |

## Choosing the number of factors

Start from the Bai-Ng (2002) criteria (`select_factors`, IC2 by default) and check the
sensitivity of the nowcast with an [experiment](../user-guide/evaluation/experiments.md).
In nowcasting applications one factor per block is often enough (Bańbura & Modugno,
2014); with a single global block, 1–3 factors are typical.

```python
import nowcastbox as nb

ds = nb.load_simulated_dfm()           # 20 monthly indicators + quarterly GDP, 2 factors
ic = nb.select_factors(ds.data.drop(["gdp"]), rmax=8, criterion="IC2")
print(ic.summary())

exp = nb.NowcastExperiment(ds.data, "gdp")
exp.add_model("two-step", nb.TwoStepDFM(n_factors=ic.r_star))
exp.add_model("em", nb.MixedFreqDFM(n_factors=ic.r_star))
exp.add_model("bridge", nb.BridgeEquation())
exp.fit_all()
print(exp.summary())
```

The final word belongs to an out-of-sample comparison in pseudo real time
([backtesting](../user-guide/evaluation/backtesting.md)), by nowcast horizon and against
simple benchmarks.
