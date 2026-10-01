# Scoring densities

`nowcastbox.evaluation.scoring` (also `nb.scoring`) evaluates density nowcasts with
**proper scoring rules** (Gneiting & Raftery, 2007) and calibration tests. Every score
accepts a `NowcastDistribution` (Gaussian or mixture) or explicit parameters/samples.

| Function | Measures | Better |
|---|---|---|
| `crps(forecast, observed)` | continuous ranked probability score (closed form for Gaussians and mixtures) | lower |
| `log_score(forecast, observed)` | $-\ln p(y)$ | lower |
| `pit(forecast, observed)` | probability integral transform $F(y)$ | uniform |
| `interval_score(observed, lower, upper, level)` | width + penalty for misses | lower |
| `interval_coverage`, `interval_hits` | empirical coverage of a band | = nominal |
| `quantile_score`, `weighted_quantile_score` | pinball loss (uniform/centre/tails weights) | lower |
| `ks_uniformity_test(pit)` | Kolmogorov-Smirnov test of uniform PITs | large p-value |
| `berkowitz_test(pit)` | LR test of $\Phi^{-1}(\text{PIT})$ iid $N(0,1)$ | large p-value |
| `christoffersen_test(hits, coverage)` | unconditional coverage, independence, conditional coverage | large p-values |

Also: `crps_gaussian`, `crps_mixture`, `crps_sample` (energy form, `fair=True`),
`log_score_gaussian`, `log_score_sample` (kernel density).

## Scoring a sequence of density nowcasts

The loop below re-estimates a model at the middle of each quarter in pseudo real time,
keeps the predictive distribution of the current quarter and scores it against the final
value.

```python
import numpy as np
import pandas as pd
import nowcastbox as nb
from nowcastbox.evaluation import scoring

ds = nb.load_simulated_dfm()
final = ds.data.to_native("gdp")

rows = []
for quarter in pd.period_range("2015Q1", "2019Q3", freq="Q"):
    vintage_date = quarter.asfreq("M", how="start") + 1          # middle month of the quarter
    vintage = ds.data.as_of(vintage_date.to_timestamp(how="end"))
    res = nb.TwoStepDFM(n_factors=2).fit(vintage, "gdp")
    dist = res.distribution(periods=[quarter])
    y = final[quarter]
    rows.append(
        {
            "quarter": quarter,
            "mean": float(dist.mean.iloc[0]),
            "std": float(dist.std.iloc[0]),
            "actual": y,
            "crps": float(scoring.crps(dist, [y])[0]),
            "log_score": float(scoring.log_score(dist, [y])[0]),
            "pit": float(scoring.pit(dist, [y])[0]),
        }
    )
table = pd.DataFrame(rows).set_index("quarter")
table.round(3).head()
table[["crps", "log_score"]].mean()
```

## Calibration

A well calibrated density has uniform PITs and bands that contain the outcome at their
nominal rate, without clustering of misses:

```python
scoring.ks_uniformity_test(table["pit"])
scoring.berkowitz_test(table["pit"])

lower = table["mean"] - 1.645 * table["std"]
upper = table["mean"] + 1.645 * table["std"]
hits = scoring.interval_hits(table["actual"], lower, upper)
scoring.interval_coverage(table["actual"], lower, upper)
scoring.christoffersen_test(hits, coverage=0.9)
scoring.interval_score(table["actual"], lower, upper, level=0.9).mean()
```

## Comparing two density forecasts

Scores are losses, so the [forecast comparison tests](../evaluation/forecast-comparison-tests.md)
apply to them directly. For example, compare the model density with a naive Gaussian
centred at the historical mean:

```python
from nowcastbox.evaluation import diebold_mariano

naive_crps = scoring.crps_gaussian(table["actual"], final.mean(), final.std())
# DM on score differences: pass the scores as "errors" with an absolute loss
diebold_mariano(table["crps"].to_numpy(), naive_crps, loss="absolute")
```

Small samples (here 19 quarters) give tests little power; in applications use the full
evaluation period and several nowcast horizons.
