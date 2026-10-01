# Number of shocks

In the two-step model the $r$ static factors follow a VAR driven by $q \le r$
**primitive (dynamic) shocks**, $f_t = \sum_i A_i f_{t-i} + B u_t$ with $B$ of rank $q$.
Static factors often include lags of a smaller number of dynamic factors, so $q < r$ is
common. Bai & Ng (2007) estimate $q$ from the eigenvalues
$c_1 \ge c_2 \ge \dots \ge c_r$ of the covariance (or correlation) matrix of the VAR
residuals $\hat u_t$:

$$
\hat D_{1,k} = \left(\frac{c_{k+1}^2}{\sum_{j=1}^r c_j^2}\right)^{1/2}, \qquad
\hat D_{2,k} = \left(\frac{\sum_{j=k+1}^r c_j^2}{\sum_{j=1}^r c_j^2}\right)^{1/2},
$$

and $\hat q$ is the smallest $k$ with $\hat D_k < m / \min(N^{1/2-\delta}, T^{1/2-\delta})$
for $0 < \delta < 1/2$ and $m > 0$. See [Bai-Ng criteria](../../theory/bai-ng.md).

## Usage

```python
import nowcastbox as nb

ds = nb.load_simulated_dfm()
indicators = ds.data.drop(["gdp"])

shocks = nb.select_shocks(indicators, n_factors=2, factor_lags=2, delta=0.1)
print(shocks.summary())
shocks.q_star, shocks.q_by_statistic
shocks.statistics.round(4)
shocks.bound
```

| Argument | Default | Meaning |
|---|---|---|
| `n_factors` | `None` | $r$; chosen with `select_factors` (IC2) when omitted |
| `factor_lags` | `2` | VAR order of the factors (Bai & Ng, 2007, use VAR(2)) |
| `delta` | `0.1` | rate parameter of the bound, $0 < \delta < 1/2$ |
| `m` | `None` | scale of the bound (default: 1 with the covariance matrix; 1.25 / 2.25 for $D_1$ / $D_2$ with the correlation matrix) |
| `statistic` | `"D1"` | statistic defining `q_star` (both are computed) |
| `matrix` | `"covariance"` | eigenvalues of the residual covariance or correlation matrix |

## In the two-step model

```python
model = nb.TwoStepDFM(n_factors=2, factor_lags=2, n_shocks=shocks.q_star)
res = model.fit(ds.data, "gdp")
res.BB.round(3)                 # B B' has rank q

auto = nb.nowcast(ds.data, "gdp", method="two_step", n_factors=2, n_shocks="auto")
auto.n_shocks
```

The choice is sensitive to `delta` and `m`; report it with a sensitivity table:

```python
import pandas as pd

pd.DataFrame(
    {
        delta: [nb.select_shocks(indicators, 2, 2, delta=delta, m=m).q_star for m in (0.5, 1.0, 1.5)]
        for delta in (0.1, 0.2, 0.3)
    },
    index=pd.Index([0.5, 1.0, 1.5], name="m"),
)
```
