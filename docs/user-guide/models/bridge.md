# Bridge equations

A bridge equation regresses the quarterly target on quarterly aggregates of a few monthly
indicators (Baffigi, Golinelli & Parigi, 2004). Because the indicators of the current
quarter are incomplete at the ragged edge, each indicator is first **extended with AR
forecasts** up to the end of the forecast horizon, then aggregated:

$$
y^Q_\tau = \alpha + \sum_j \sum_{l=0}^{L} \beta_{jl}\, \bar x^Q_{j,\tau-l}
+ \sum_{l=1}^{P} \phi_l\, y^Q_{\tau-l} + e_\tau ,
\qquad \bar x^Q_{j,\tau} = \sum_{s} w_s\, x_{j,3\tau-s}.
$$

## Fitting

```python
import nowcastbox as nb

ds = nb.load_simulated_dfm()
bridge = nb.BridgeEquation(aggregation="average", regressor_lags=0, target_lags=0,
                           ar_lags=1, horizon=1)
res = bridge.fit(ds.data, target="gdp ~ x01 + x04 + x10")      # formula selects predictors
print(res.summary())
res.nowcast.tail(3)
```

| Parameter | Default | Meaning |
|---|---|---|
| `aggregation` | `"average"` | `"average"`, `"flow"`, `"stock"`, `"mariano_murasawa"` or explicit weights (most recent first) |
| `regressor_lags` | `0` | lags of each aggregated predictor (target periods) |
| `target_lags` | `0` | autoregressive lags of the target (iterated forecasts) |
| `fill_method`, `ar_lags` | `"ar"`, `1` | ragged-edge extension of the predictors (`"none"` leaves incomplete quarters without estimate) |
| `horizon` | `1` | target periods forecast after the current one |
| `cov_type` | `"nonrobust"` | statsmodels covariance of the OLS coefficients (e.g. `"HC1"`) |

`res.bridge` is the fitted regression (coefficients, standard errors, R²):

```python
res.bridge.params
res.bridge.rsquared
```

## Bridge equations on factors

The two-step DFM ends with a bridge equation on the aggregated factors
(`TwoStepDFM(aggregate="factors")`); a bridge on a handful of hand-picked indicators is
its transparent, low-dimensional counterpart. Use [targeted
predictors](../selection/targeted-predictors.md) to pick the indicators objectively.

```python
picked = nb.select_targeted_predictors(
    ds.data.drop(["gdp"]), ds.data.to_frame()["gdp"], method="soft", n_predictors=3
)
formula = "gdp ~ " + " + ".join(picked.selected)
nb.BridgeEquation().fit(ds.data, formula).get_nowcast()
```

As a benchmark in backtests use `nb.benchmarks.BridgeBenchmark` (same estimator behind
the benchmark interface; see [Benchmarks](benchmarks.md)). To avoid picking the
indicators at all, [combine all small bridge equations](bridge-combination.md)
(`nb.BridgeCombination`, Bańbura et al., 2023).
