# Large Bayesian VAR

`LargeBVAR` nowcasts with a large mixed-frequency Bayesian VAR, using the "blocking"
approach of Cimadomo, Giannone, Lenza, Monti & Sokol (2022). Each monthly series becomes
three quarterly variables, one per month of the quarter. Quarterly series enter as they
are. The whole vector follows a quarterly VAR with a Minnesota-type prior whose
tightness is chosen from the data. The ragged edge is handled by conditional forecasts:
the model predicts what is missing from everything already released. See the
[theory page](../../theory/large-bvar.md) for the details and the
[BVAR prior page](bvar-prior.md) for the prior machinery.

## Fitting

```python
import nowcastbox as nb
from nowcastbox.models import LargeBVAR

sim = nb.simulate.dfm(n_series=10, n_factors=1, n_periods=240, random_state=0)
res = LargeBVAR(lags=1).fit(sim.data, "gdp")      # formulas work too: "gdp ~ x01 + x02"
res.get_nowcast()                                  # nowcast of the current quarter
res.nowcast.tail()                                 # observed / in_sample / out_of_sample / std / bands
print(res.summary())                               # lambda, estimation sample, conditioning window
```

| Parameter | Default | Meaning |
|---|---|---|
| `lags` | `1` | quarterly lags of the blocked VAR (1–2 for growth rates; Cimadomo et al. use 5 in log-levels) |
| `prior` | `"glp"` | `"glp"`: hierarchical choice of $\lambda$ (and $\mu$, $\delta$); a mapping such as `{"lambda": 0.2}` fixes them |
| `prior_mean` | `"white_noise"` | own-lag prior mean: 0 for growth rates, `"random_walk"` for levels, a number, or `{series: value}` |
| `sum_of_coefficients` | `False` | sum-of-coefficients prior (for levels) |
| `initial_observation` | `False` | dummy-initial-observation prior (for levels) |
| `estimate_psi` | `False` | also estimate the prior scales $\psi$ (default: AR(1) residual variances) |
| `standardize` | `True` | standardise the series before blocking |
| `n_draws` | `0` | posterior draws stored for the density nowcast |
| `random_state` | `None` | seed of the draws |

`fit(..., horizon=h)` forecasts `h` more target periods. The panel must be on a monthly
grid and contain monthly and quarterly series only. The target may be quarterly or
monthly.

The model is estimated on the balanced part of the blocked panel: the longest run of
complete quarters that ends at the last complete quarter. A series that starts late
therefore shortens the sample for every variable. Drop or backfill such series if the
sample becomes too short; the fit emits a `DataQualityWarning` naming them whenever
quarters with data are left out. Fitting needs at least `lags + 3` complete quarters.

### Data in levels

For (log-)levels, use the random-walk prior mean and the two dummy priors, as in
Cimadomo et al. (2022):

```python
LargeBVAR(lags=5, prior_mean="random_walk",
          sum_of_coefficients=True, initial_observation=True)
```

The nowcast is then in the units of the target (for example a log level). Compute
growth rates from the forecast levels yourself.

## What the results contain

```python
res.blocked_data().head()        # quarterly blocked panel: x01[m1], x01[m2], x01[m3], ..., gdp
res.coefficients()               # posterior mean of B (const and lag rows, one column per equation)
res.params["lambda"], res.params["sigma"]
res.window_end                   # last quarter of the last window of `lags` complete quarters
res.smoothed_data.tail()         # E[x_t | data] for every series, monthly grid, original units
vintage = sim.data.truncate(end="2019-10")
res.predict(vintage)             # the same for another vintage, with the estimated parameters
```

`in_sample` holds the one-quarter-ahead VAR prediction for quarters where the target is
observed. `out_of_sample` holds the conditional forecast for the others (backcast,
nowcast, forecasts).

## Density nowcasts

```python
res = LargeBVAR(lags=1, n_draws=500, random_state=0).fit(sim.data, "gdp")
dist = res.distribution()          # mixture over 500 posterior draws of (B, Sigma)
dist.interval(0.9), dist.to_frame()
dist.plot()                        # fan chart

# without stored draws:
res.distribution(n_draws=500, random_state=0)   # draws computed on demand
res.distribution(method="analytic")             # Gaussian from the `std` column
```

For each posterior draw, the target given the released data is Gaussian with a
closed-form mean and variance. The distribution is the mixture of these Gaussians.
When draws are stored, the `std` column and the 68%/90% bands of `res.nowcast` come from
this mixture. Without draws they come from the Kalman smoother at the posterior mean,
so they include filtering uncertainty only. The mixture works with
`nowcastbox.evaluation.scoring` (CRPS, PIT, coverage) and with the empirical bands
(`method="empirical"`).

## News and contributions

```python
old = sim.data.truncate(end="2019-10")
news = res.news(old, sim.data)       # target must be quarterly
news.summary()
news.releases                        # series, month, actual, expected, weight, impact
res.level_contributions().to_frame("series")
```

With the parameters fixed at the posterior mean the model is linear and Gaussian, so
the news decomposition of Bańbura & Modugno (2014) is exact: `check_identity()` holds.
Releases are reported by original series and month. `new_results=` adds the
re-estimation effect. `nowcast_tracker` is not available for this model; compute the
news between consecutive vintages instead.

## Pseudo real time

```python
from nowcastbox.evaluation import PseudoRealTimeBacktest

bt = PseudoRealTimeBacktest(LargeBVAR(lags=1), sim.data, "gdp",
                            start="2017-01-15", end="2019-12-15", refit_every=3)
bt.run().metrics()
```

Between re-estimations the backtest updates the nowcast with `results.predict(vintage)`
and keeps the parameters fixed. Every fit uses only the vintage's information,
including its own standardisation statistics.

## BVAR extrapolation in bridge equations

Importing `nowcastbox.models` registers the `"bvar"` extrapolator
(`nowcastbox.models.BVARExtrapolator`). It completes all the indicators of a
`BridgeCombination` jointly. When every indicator is monthly it uses a monthly BVAR; when
quarterly indicators are also completed it uses the blocked BVAR described on this page:

```python
from nowcastbox.models import BridgeCombination

BridgeCombination(extrapolation="bvar",
                  extrapolation_options={"prior": {"lambda": 0.2}}).fit(sim.data, "gdp")
```

If the BVAR cannot be estimated, for example because there are too few complete
periods, the extrapolator raises `NowcastDataError`. `BridgeCombination` then
extrapolates series by series. See
[Combination of bridge equations](bridge-combination.md#bvar-extrapolation) for the
options.

## Performance

With 50 monthly series (151 blocked variables), 80 quarters and one BLAS thread:

- a fit with the GLP prior took about 0.5 s;
- 200 posterior draws took about 1 s;
- a news decomposition took 2–10 s, depending on the number of lags.

Multi-threaded OpenBLAS can be slower on these matrix sizes. Set
`OPENBLAS_NUM_THREADS` (or use `threadpoolctl`) when running many fits.
