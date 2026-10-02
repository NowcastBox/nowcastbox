# Share of data already released

How much of the target quarter does the nowcast already "see"? For a target period
$\tau$ and a vintage of the panel, each series $i$ has $E_i$ **expected** observations
inside $\tau$: its storage slots in the period. A monthly series has three in a quarter, a
quarterly series has one, and an annual series has none outside the fourth quarter. The
series also has $R_i \le E_i$ **released** observations, the slots that already hold a
value. The share of a group $G$ is

$$
\text{share}_G = \frac{\sum_{i\in G} w_i\,R_i/E_i}{\sum_{i\in G} w_i},
\qquad w_i = E_i \text{ by default},
$$

so with the default weights it is simply $\sum R_i / \sum E_i$. You can instead pass
weights $w_i \ge 0$, for example the absolute weight of each series in the model's
nowcast. Linzenich & Meunier (2024, ECB WP 3004) report this share next to each nowcast.

```python
import numpy as np, pandas as pd
from nowcastbox import MixedFrequencyData

idx = pd.period_range("2020-01", periods=6, freq="M")
panel = MixedFrequencyData(
    pd.DataFrame({
        "ip":  [1, 1, 1, 1, np.nan, np.nan],
        "pmi": [1, 1, 1, 1, 1, 1],
        "gdp": [np.nan, np.nan, 1, np.nan, np.nan, np.nan],
    }, index=idx, dtype=float),
    {"ip": "M", "pmi": "M", "gdp": "Q"},
    categories={"ip": "hard", "pmi": "soft", "gdp": "hard"},
    release_delays={"ip": 40, "pmi": 1, "gdp": 60},
)

panel.released_share("2020Q2", series=["ip", "pmi"])
#        released  expected  weight     share
# ip            1         3     3.0  0.333333
# pmi           3         3     3.0  1.000000
# total         4         6     6.0  0.666667

panel.released_share("2020Q2", by="category")         # hard / soft / total
panel.released_share("2020Q2", as_of="2020-05-15")    # information set of a vintage
panel.released_share("2020Q2", weights={"ip": 0.7, "pmi": 0.3})
```

- `by`: `None`/`"series"`, `"category"` (`"uncategorized"` when unset), `"block"` (a
  series counts in each of its blocks), `"frequency"` or a mapping
  `{series: group or [groups]}`. The `total` row counts each series once.
- Base periods of $\tau$ that lie beyond the end of the grid count as expected but not
  released.

## In the nowcast tracker

`nowcast_tracker` adds a `released_share` column to `tracker.path`. It holds the share of
the target period's predictor observations available at each vintage. Only the model's
own predictors count (its estimation panel without the target), so a formula fit such as
`"gdp ~ ip + pmi"` is not diluted by series the model does not use. Read the nowcast path
against this flow of information:

```python
import nowcastbox as nb

ds = nb.load_simulated_dfm()
data = ds.data.select(["gdp", "x01", "x02", "x03", "x04"]).truncate(start="2008-01")
res = nb.TwoStepDFM(n_factors=1).fit(data.as_of("2019-11-15"), "gdp")
tracker = res.nowcast_tracker(data, target_period="2019Q4", start="2019-10-01", end="2019-12-31")
tracker.path[["nowcast", "released_share"]]
```

## Charts

```python
from nowcastbox.visualization import plot_released_share, released_share_table

released_share_table(res)                 # results: current nowcast period, target excluded
plot_released_share(panel, "2020Q2", by="category", backend="matplotlib")
res.plot("released_share")                # registered plot kind
```

## See also

- [Nowcast tracker](../news/tracker.md): the `released_share` column follows the share
  through the vintages.
- [Indicator heatmap](../visualization/indicator-heatmap.md) and [HTML
  reports](../visualization/reports.md), which show the share of the nowcast period by
  category.
