# EM mixed-frequency DFM

`MixedFreqDFM` estimates a dynamic factor model by maximum likelihood with the EM
algorithm of Bańbura & Modugno (2014), with:

- an **arbitrary pattern of missing data** (ragged edge, late starts, gaps);
- **lower-frequency series in the state space**: a quarterly series loads on the
  current and lagged monthly factors with Mariano-Murasawa (or flow/average/stock)
  restrictions;
- **AR(1) idiosyncratic components** (or white noise, or Student-t);
- **blocks** of factors (see [Blocks](blocks.md));
- robust and long-run-mean extensions ([Robust estimation](robust.md),
  [Long-run mean](long-run-mean.md)).

See [EM algorithm](../../theory/em-algorithm.md) for the E- and M-steps.

## Fitting

```python
import nowcastbox as nb

ds = nb.load_simulated_dfm()
model = nb.MixedFreqDFM(n_factors=2, factor_lags=1, max_iter=500, tol=1e-4)
res = model.fit(ds.data, target="gdp")
print(res.summary())
res.converged, res.n_iter, res.loglikelihood
```

| Parameter | Default | Meaning |
|---|---|---|
| `n_factors` | `1` | factors per block (int or `{block: r}`) |
| `factor_lags` | `1` | VAR order of each block |
| `blocks` | `None` | block structure (`None` = one global block, `"data"` = metadata) |
| `idiosyncratic` | `"ar1"` | `"ar1"`, `"iid"` or `"student_t"` |
| `max_iter`, `tol` | `500`, `1e-4` | stop when the relative log-likelihood change is below `tol` |
| `init` | `"pca"` | starting values: PCA on a spline-filled panel (with several blocks: the best of four block orders, see [Blocks](blocks.md)), `"pca_<order>"`, or previous results / `EMParameters` (warm start) |
| `obs_noise_var` | `1e-4` | small fixed measurement noise of the `"ar1"` specification |
| `filter_method` | `"auto"` | E-step smoother: exact structured smoother (I2) when cheaper, dense Kalman otherwise |

`fit(data, target, horizon=0)` accepts a `horizon`: target periods forecast beyond the
period containing the end of the sample.

## Results

```python
res.nowcast.tail(3)             # with std and 68 % / 90 % bands out of sample
res.get_nowcast()               # first period after the last target observation
res.factors.tail(3)
res.loadings.head()
res.idiosyncratic_ar.head()     # AR(1) coefficients of the idiosyncratic components
res.loglikelihood_path[:5]      # monotone (up to the initial-state term)
res.predict().tail(2)           # E[x_t | data] for every series, original units
```

`state_space_model()` returns the estimated `StateSpace`, the state layout and the grid
— the entry point used by the news, density and diagnostics modules.

## Convergence

The EM stops when the relative change of the log-likelihood,
$(\ell_k - \ell_{k-1}) / \tfrac12(|\ell_k| + |\ell_{k-1}|)$, falls below `tol`. If
`max_iter` is reached first a `ConvergenceWarning` is raised:

```python
import warnings

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    short = nb.MixedFreqDFM(n_factors=2, max_iter=2, tol=1e-12).fit(ds.data, "gdp")
[w.category.__name__ for w in caught]
short.converged
```

`res.plot("loglikelihood")` and `res.diagnostics().convergence` show the path.

## Warm starts and new vintages

Re-estimating every month from scratch is wasteful: pass the previous results as `init`
(the model structure must be the same), or keep the parameters fixed and only smooth the
new data with `predict`/`smooth`:

```python
old = ds.data.as_of("2019-11-01")
new = ds.data.as_of("2020-01-05")
first = nb.MixedFreqDFM(n_factors=2).fit(old, "gdp")
warm = nb.MixedFreqDFM(n_factors=2, init=first).fit(new, "gdp")
warm.n_iter                                      # few iterations from a good start
first.predict(new)["gdp"].loc["2019-12"]         # fixed parameters, new information
```

## Idiosyncratic specifications

| `idiosyncratic` | State | Use |
|---|---|---|
| `"ar1"` (default) | one AR(1) state per series (lags for lower-frequency series) | persistent idiosyncratic noise, as in Bańbura & Modugno (2014) |
| `"iid"` | none; estimated diagonal $H$ | smaller state, faster, adequate for noisy indicators |
| `"student_t"` | none; scale-mixture weights | fat tails / outliers ([Robust estimation](robust.md)) |

```python
iid = nb.MixedFreqDFM(n_factors=2, idiosyncratic="iid").fit(ds.data, "gdp")
iid.idiosyncratic_variance.head(3)
```

## The one-call interface

`nb.nowcast(data, target, method="em")` chooses the number of factors with the Bai-Ng
criterion when `n_factors` is omitted and forwards the robust, long-run-mean and density
options:

```python
quick = nb.nowcast(ds.data, "gdp", method="em", rmax=6)
quick.n_factors, quick.info["selection"].r_star
```
