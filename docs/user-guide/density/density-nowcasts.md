# Density nowcasts (innovation I5)

A point nowcast without uncertainty is hard to use: a 0.3 % nowcast of quarterly growth
means something very different with a standard deviation of 0.2 or of 1.0 points.
NowcastBox produces the **predictive distribution** of the target combining

1. **filtering uncertainty** — the Kalman smoother variance of the target given the data,
   $\operatorname{Var}(y_\tau \mid \Omega, \hat\theta)$ (analytic, Gaussian);
2. **parameter uncertainty** — the variation of the nowcast across bootstrap re-estimations
   of the parameters $\theta^\ast_b$ (Stoffer & Wall, 1991; Pfeffermann & Tiller, 2005).

The result is a Gaussian (filtering only) or a **mixture of Gaussians** over the bootstrap
replications,
$p(y_\tau \mid \Omega) \approx \frac1B \sum_b N\!\left(\mu_b, \sigma_b^2\right)$, whose
variance splits into $\operatorname{E}_b[\sigma_b^2]$ (filtering) and
$\operatorname{Var}_b[\mu_b]$ (parameters) — see
[Density nowcasts and scoring](../../theory/density-scoring.md).

## Analytic (Gaussian) distribution

```python
import nowcastbox as nb

ds = nb.load_simulated_dfm()
vintage = ds.data.as_of("2019-11-15")
res = nb.MixedFreqDFM(n_factors=2).fit(vintage, target="gdp")

dist = res.distribution()                    # = nb.nowcast_distribution(res)
dist.to_frame()                              # point, mean, std, median, 50/68/90 % bands
dist.interval(0.9)
dist.quantiles([0.05, 0.25, 0.5, 0.75, 0.95])
```

## With parameter uncertainty (bootstrap)

```python
boot = res.distribution(n_boot=5, random_state=0)    # use 200+ in applications
boot.n_components
boot.variance_decomposition                           # filtering vs parameter variance
boot.info["method"]
```

| Argument | Default | Meaning |
|---|---|---|
| `n_boot` | `0` | bootstrap replications (`0` = analytic Gaussian) |
| `method` | `"auto"` | `"parametric"` (simulate from the fitted model, re-estimate, re-filter the actual data) or `"block"` (moving-block bootstrap of whole target periods); `"auto"`: parametric when available, block otherwise |
| `periods` | out-of-sample periods | target periods of the distribution |
| `refit_params`, `warm_start` | | estimator options of the refits (e.g. fewer EM iterations) |
| `n_jobs` | `None` | parallel refits (joblib) |
| `center` | `True` | centre the mixture at the point nowcast |

Bootstrap refits multiply the estimation time by `n_boot`: keep `n_boot` small while
exploring and use `n_jobs=-1` and `refit_params={"max_iter": 50}` for production runs.

## Working with the distribution

`NowcastDistribution` behaves like a vector of scipy distributions, one per period:

```python
q4 = boot.select(["2019Q4"])
q4.mean, q4.std, q4.median
q4.cdf([0.0])                  # probability of negative growth
q4.ppf([0.05, 0.95])
draws = q4.sample(1000, random_state=1)
draws.describe().T
```

## Plots and the nowcast frame

```python
ax = boot.plot(history=res.nowcast["observed"].dropna().tail(12))   # Matplotlib fan chart
fig = res.plot("density", n_boot=0)                                 # Plotly fan chart
fig = res.plot("fan")                                               # bands from the std column
```

`nb.nowcast(..., density=True, n_boot=...)` stores the distribution in
`results.info["distribution"]` and overwrites `std`, `median` and the 68 %/90 % bounds of
`results.nowcast` with it:

```python
quick = nb.nowcast(vintage, "gdp", method="two_step", n_factors=2, density=True)
quick.nowcast.tail(2)[["out_of_sample", "lower_90", "median", "upper_90"]]
```

## Two-step results

The two-step model's `std` combines the smoothing variance of the aggregated factors and
the bridge residual variance; its bootstrap re-estimates PCA, VAR and bridge. With
`aggregate="variables"` the parametric bootstrap simulates the *filtered* panel the state
space describes (`results.model_data`) and refits with `fit(..., prefiltered=True)`.
On weekly or daily grids (calendar aggregation, I1) only the parametric bootstrap is
available: whole target periods have a variable number of weeks, so the block bootstrap
raises an error.

```python
two_step = nb.TwoStepDFM(n_factors=2).fit(vintage, "gdp")
two_step.distribution(n_boot=5, random_state=0).variance_decomposition
```

Evaluate densities out of sample with proper scoring rules: see
[Scoring densities](scoring.md).
