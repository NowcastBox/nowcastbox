# Two-step DFM

`TwoStepDFM` implements the estimator of Giannone, Reichlin & Small (2008) and Doz,
Giannone & Reichlin (2011):

1. **PCA** on the balanced part of the standardised monthly panel gives the loadings
   $\hat\Lambda$ and factors $\hat f_t$;
2. a **VAR($p$)** on $\hat f_t$ gives $\hat A_1,\dots,\hat A_p$; the residual covariance
   is reduced to rank $q$ (number of dynamic shocks) to get $\hat B$; $\hat\Psi$ is the
   diagonal of the idiosyncratic variances;
3. the **Kalman smoother** re-estimates the factors on the full, ragged panel;
4. a **bridge equation** links the quarterly target to the factors.

See [Dynamic factor models](../../theory/dfm.md) for the model and its state-space form.

## Fitting

```python
import nowcastbox as nb

ds = nb.load_simulated_dfm()        # 20 monthly indicators + quarterly gdp, 2 true factors
model = nb.TwoStepDFM(n_factors=2, factor_lags=1, n_shocks=2, aggregate="factors")
res = model.fit(ds.data, target="gdp")
print(res.summary())
res.nowcast.tail(3)               # observed / in_sample / out_of_sample / std
```

| Parameter | Default | Meaning |
|---|---|---|
| `n_factors` | `2` | static factors $r$ (see [Number of factors](../selection/number-of-factors.md)) |
| `factor_lags` | `1` | VAR order $p$ |
| `n_shocks` | `None` (= $r$) | dynamic shocks $q \le r$ (see [Number of shocks](../selection/number-of-shocks.md)) |
| `aggregate` | `"factors"` | how the quarterly target is bridged (below) |
| `aggregation` | metadata / Mariano-Murasawa | weights linking months to the target frequency |
| `horizon` | `1` | target periods forecast after the current one |
| `filter_method` | `"auto"` | Kalman filter variant (univariate treatment for large panels) |
| `collapse` | `False` | collapse the observation vector (Jungbacker & Koopman, 2015): identical results, faster for large N |

## `aggregate="factors"` versus `"variables"`

=== "factors (2s_agg)"

    The monthly factors are aggregated to quarterly frequency with the normalised
    Mariano-Murasawa filter $\tfrac19(1,2,3,2,1)$ and the target is regressed on them.
    This is the variant supported by the news decomposition.

    ```python
    res_f = nb.TwoStepDFM(n_factors=2, aggregate="factors").fit(ds.data, "gdp")
    res_f.aggregated_factors.tail(3)
    res_f.bridge.params
    ```

=== "variables (2s)"

    The monthly predictors are first filtered to quarterly quantities
    ($\tfrac13(1,2,3,2,1)$), the factors are extracted from the filtered panel and the
    bridge uses them directly.

    ```python
    res_v = nb.TwoStepDFM(n_factors=2, aggregate="variables").fit(ds.data, "gdp")
    res_v.get_nowcast()
    ```

The aliases `"2s_agg"` and `"2s"` follow the naming of the nowcasting literature.

## Results

```python
res.factors.tail(3)                # smoothed monthly factors f1, f2
res.loadings.head()                # Lambda (series x factors)
res.A                              # [A_1 ... A_p], shape (r, r p)
res.BB                             # B B' (rank q)
res.Psi.head()                     # idiosyncratic variances
res.eigenvalues.head()             # of the panel correlation matrix (scree plot)
res.explained_variance_ratio.head()
res.bridge.summary()               # statsmodels OLS of the bridge
res.x_forecast.tail(2)             # completed monthly panel (smoothed + forecast)
```

The `std` column of `nowcast` is the standard deviation of the bridge prediction,
$\sqrt{\beta' V_t \beta + \sigma_e^2}$, which combines the smoothing uncertainty of the
aggregated factors and the bridge residual variance (parameter uncertainty is added by
[density nowcasts](../density/density-nowcasts.md)).

## Updating with new data

`update` keeps every estimated parameter and only re-runs the Kalman smoother and the
bridge on a new vintage — the information update of a real-time exercise:

```python
old = ds.data.as_of("2019-11-01")
new = ds.data.as_of("2020-01-05")
model = nb.TwoStepDFM(n_factors=2)
first = model.fit(old, "gdp")
updated = model.update(new)
first.get_nowcast("2019Q4"), updated.get_nowcast("2019Q4")
```

## Plots

```python
fig = res.plot("forecast")       # observed vs in-sample vs out-of-sample with bands
fig = res.plot("factors")
fig = res.plot("eigenvalues")    # scree plot
fig = res.plot("loadings", style="heatmap")
```

!!! note "When to prefer the EM model"
    The two-step estimator needs a balanced block for PCA and does not model blocks,
    robust errors or a long-run mean. For many quarterly series or irregular missing
    patterns use the [EM mixed-frequency DFM](em-dfm.md).
