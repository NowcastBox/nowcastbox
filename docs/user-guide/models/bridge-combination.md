# Combination of bridge equations

Instead of choosing one [bridge equation](bridge.md), `BridgeCombination` estimates
**every** small bridge equation that can be built from the candidate indicators — by
default all equations with one or two monthly indicators and zero or one quarterly
indicator — and averages their nowcasts. This is the approach of Bańbura, Belousova,
Bodnár & Tóth (2023, ECB Working Paper 2815) and one of the models of the ECB nowcasting
toolbox (Linzenich & Meunier, 2024). See the [theory page](../../theory/bridge-combination.md)
for the formulas.

## Fitting

```python
import nowcastbox as nb
from nowcastbox.models import BridgeCombination

ds = nb.load_simulated_dfm()                       # 20 monthly indicators + quarterly GDP
model = BridgeCombination(max_monthly=2, max_quarterly=1, target_lags=0,
                          combine="mean", extrapolation="ar", ar_lags=1)
res = model.fit(ds.data, target="gdp")             # or "gdp ~ x01 + x02 + ..." to restrict
res.info["n_equations"]                            # 20 + 190 = 210 equations
res.get_nowcast()
res.nowcast.tail(3)                                # observed / in_sample / out_of_sample + dispersion
```

The `nowcast` frame follows the usual conventions (`observed`, `in_sample`,
`out_of_sample` on the quarterly index) and adds the cross-equation dispersion of each
quarter: `equation_std`, `equation_min`, `equation_max` and `n_equations` (equations
entering that quarter's combination).

| Parameter | Default | Meaning |
|---|---|---|
| `max_monthly`, `min_monthly` | `2`, `1` | number of higher-frequency indicators per equation |
| `max_quarterly` | `1` | number of target-frequency indicators per equation (0 to `max_quarterly`) |
| `target_lags`, `regressor_lags` | `0`, `0` | autoregressive lags of the target (iterated) and lags of each aggregated indicator |
| `aggregation` | `"average"` | monthly-to-quarterly aggregation, as in `BridgeEquation` |
| `combine` | `"mean"` | `"mean"`, `"median"` or `"inverse_mse"` |
| `mse` | `"in_sample"` | accuracy behind `inverse_mse` and `trim`: `"in_sample"` residuals or recursive one-step `"out_of_sample"` errors |
| `mse_window`, `discount`, `min_train` | all, `1.0`, `8` | last quarters used in the MSE, discount factor of older errors (Stock & Watson, 2004), minimum training sample of the out-of-sample errors |
| `trim` | `0.0` | share of the worst equations (by MSE) discarded before combining |
| `extrapolation`, `ar_lags`, `extrapolation_options` | `"ar"`, `1`, — | completion of the indicators at the ragged edge (see below); `None` = no completion |
| `horizon` | `1` | quarters forecast after the current one |

The number of equations grows quickly with the number of indicators —
`nowcastbox.models.bridge_equation_count(50)` is 1,275 — but the indicators are
extrapolated **once per fit** and all equations are estimated by batched least squares,
so 50 monthly indicators fit in well under a second (a few seconds with out-of-sample
MSEs).

## Inspecting the equations

`res.equations()` returns one row per equation with its specification, fit statistics,
weight and nowcast for the current quarter (pass a period to choose another quarter):

```python
table = res.equations()                  # spec, monthly, quarterly, n_obs, rsquared, sigma,
                                         # mse, valid, included, weight, nowcast
table.sort_values("mse").head()
res.equations("2019Q4", included_only=True)
res.equation_estimates                   # quarters x equations
res.params["coefficients"].loc["eq0001"] # coefficients of one equation
res.indicators                           # extrapolated indicators aggregated to quarters
print(res.summary())
```

Equations that cannot be estimated (collinear indicators, too short a sample) are
reported with `valid=False`, excluded from the combination and announced by a
`DataQualityWarning`.

## Weighting and trimming

```python
res = BridgeCombination(
    combine="inverse_mse",           # w_e proportional to 1 / MSE_e
    mse="out_of_sample",             # recursive one-step errors within the vintage
    mse_window=20, discount=0.95,    # last 20 quarters, older errors discounted
    trim=0.25,                       # drop the 25 % worst equations first
).fit(ds.data, "gdp")
res.weights.sum()                    # 1.0
```

Equal weights are the default: they are hard to beat when many similar models are
combined (Timmermann, 2006). The median is a robust alternative (`combine="median"`).

## Ragged edge: pluggable extrapolators

Each indicator is completed up to the end of the forecast horizon before aggregation.
`extrapolation="ar"` uses iterated AR(`ar_lags`) forecasts (the treatment of
`BridgeEquation`); `extrapolation=None` does not complete the indicators, so an
equation contributes to a quarter only when its indicators are complete. Any callable
with the signature `f(data, columns, end) -> {name: series}` can be passed, or
registered under a name — the planned large-BVAR extrapolation will be registered as
`"bvar"` in the same way:

```python
from nowcastbox.models import register_extrapolator
from nowcastbox.models.extrapolation import native_until

class LastValue:
    """Carry the last observation forward."""
    def __call__(self, data, columns, end):
        return {c: native_until(data, c, end).ffill() for c in columns}

register_extrapolator("last_value", LastValue)
BridgeCombination(extrapolation="last_value").fit(ds.data, "gdp").get_nowcast()
```

The completed series are kept in `res.extrapolated`. Indicators without any observation
in the panel, or that the extrapolator rejects (too few observations for the AR model,
typically a series that starts after an early pseudo real-time vintage), are left out
of the combination with a `DataQualityWarning` instead of stopping the fit; a
multivariate extrapolator that fails on the whole panel is retried series by series.

## Pseudo real time, backtests and news

All estimates, MSEs, weights and the trimming use only the information set passed to
`fit`; `fit(data, target, as_of="2019-11-15")` fits on the pseudo real-time vintage
`data.as_of(...)`. The model can be the main model of a backtest, and
`nowcastbox.benchmarks.BridgeCombinationBenchmark` wraps it as a benchmark:

```python
from nowcastbox.benchmarks import AR, BridgeBenchmark, BridgeCombinationBenchmark
from nowcastbox.evaluation import PseudoRealTimeBacktest

bt = PseudoRealTimeBacktest(
    BridgeCombination(combine="inverse_mse", mse="out_of_sample", mse_window=20),
    data=ds.data, target="gdp", start="2016-01-15", end="2019-12-15", step="M",
    benchmarks={"AR": AR(p=1), "bridge (x10)": BridgeBenchmark(predictors=["x10"]),
                "combo (mean)": BridgeCombinationBenchmark()},
)
bt.run().relative_to("AR")
```

The exact news decomposition (`res.news`) needs a state-space model and is not available
for bridge combinations. `nowcast_change` gives an approximation: the revision of the
combined nowcast between two vintages split into equation contributions (or shared
equally by the indicators of each equation). It mixes new releases, revisions and
re-estimation.

```python
old = BridgeCombination().fit(ds.data, "gdp", as_of="2019-11-15")
new = BridgeCombination().fit(ds.data, "gdp", as_of="2019-12-15")
new.nowcast_change(old, "2019Q4")                    # by equation
new.nowcast_change(old, "2019Q4", by="indicator")    # sums to the revision
```
