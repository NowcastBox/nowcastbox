# Number of factors

Bai & Ng (2002) choose the number of static factors $r$ by penalising the fit of the
principal components,

$$
IC_{k}(r) = \ln V(r) + r\, g_k(N, T), \qquad
PC_{k}(r) = V(r) + r\, \hat\sigma^2 g_k(N, T),
$$

where $V(r)$ is the average squared residual of the panel after $r$ principal components
and the penalties are

| $k$ | $g_k(N, T)$ |
|---|---|
| 1 | $\frac{N+T}{NT}\ln\frac{NT}{N+T}$ |
| 2 | $\frac{N+T}{NT}\ln C^2_{NT}$ |
| 3 | $\frac{\ln C^2_{NT}}{C^2_{NT}}$ |

with $C^2_{NT} = \min(N, T)$. See [Bai-Ng criteria](../../theory/bai-ng.md).

## Usage

```python
import nowcastbox as nb

ds = nb.load_simulated_dfm()                       # generated with 2 factors
indicators = ds.data.drop(["gdp"])                 # base-frequency predictors only

ic = nb.select_factors(indicators, rmax=8, criterion="IC2")
print(ic.summary())
ic.r_star, ic.r_star_by_criterion
ic.to_frame().round(3)                             # V(r) and the six criteria by r
```

| Argument | Default | Meaning |
|---|---|---|
| `rmax` | `10` | largest $r$ considered ($1 \le r_{\max} < \min(N, T)$) |
| `criterion` | `"IC2"` | criterion defining `r_star` (`"IC1"`–`"IC3"`, `"PC1"`–`"PC3"`); all six are computed |
| `standardize` | `True` | standardise the series (the criteria are not scale invariant) |
| `missing` | `"drop_rows"` | obtain a balanced panel by dropping rows (`"drop_columns"`, `"raise"`) |

With a `MixedFrequencyData` only the base-frequency series are used. Missing values
(ragged edge, late starts) are dropped with a `DataQualityWarning`.

## Plots

```python
fig = ic.plot("criteria")          # IC/PC curves with the minimum marked (Matplotlib)
fig = ic.plot("eigenvalues")       # scree plot
ic.explained_variance_ratio.head()
```

## Practical advice

- **IC2** is the most used criterion in nowcasting; the PC criteria tend to select more
  factors and depend on $r_{\max}$.
- The criteria are unreliable when $\min(N, T) < 20$ (a warning is emitted) and tend to
  over-estimate $r$ with strongly heteroskedastic idiosyncratic noise.
- In nowcasting, accuracy is often flat in $r$ beyond 1–3 factors; confirm the choice
  out of sample (an [experiment](../evaluation/experiments.md) or a
  [backtest](../evaluation/backtesting.md)).
- `nb.nowcast(..., n_factors=None)` runs this selection automatically (`rmax`,
  `criterion`) and stores the result in `results.info["selection"]`.

```python
res = nb.nowcast(ds.data, "gdp", method="two_step", rmax=6, criterion="IC2")
res.n_factors, res.info["selection"].criterion
```
