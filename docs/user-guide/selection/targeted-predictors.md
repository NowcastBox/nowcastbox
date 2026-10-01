# Targeted predictors and variable selection (innovation I7)

More data is not always better: factors extracted from many series weakly related to the
target can be dominated by irrelevant co-movement (Boivin & Ng, 2006). NowcastBox offers
two complementary approaches:

- **targeted predictors** (Bai & Ng, 2008): screen the indicators by their marginal
  predictive power for the target before extracting factors;
- **block / variable selection validated in pseudo real time**: greedy search over
  groups of series scored by the out-of-sample RMSFE of the nowcasting model itself.

## Hard thresholding

Each predictor is regressed on the target (aggregated to the target frequency, with
optional target lags and a forecast horizon) and kept when its HAC $|t|$-statistic exceeds
a threshold (1.28, 1.65 or 2.58 in Bai & Ng, 2008):

```python
import nowcastbox as nb

ds = nb.load_simulated_dfm()
x = ds.data.drop(["gdp"])                       # monthly indicators
y = ds.data.to_frame()["gdp"]                   # quarterly target on the monthly grid

hard = nb.hard_threshold(x, y, threshold=1.65, cov_type="hac", max_predictors=10)
print(hard.summary())
hard.selected[:5], hard.n_selected
```

## Soft thresholding (elastic net)

The elastic net path ranks predictors by the order in which they enter as the penalty
decreases; the first `n_predictors` are kept (Bai & Ng, 2008, use 30). The elastic net
(Zou & Hastie, 2005) is solved by coordinate descent:

```python
soft = nb.soft_threshold(x, y, n_predictors=8, l1_ratio=0.5)
print(soft.summary())
soft.ranking.head()
```

`select_targeted_predictors(x, y, method="hard" | "soft", **kwargs)` dispatches to
either. The result's `transform` keeps the selected columns of a panel, so a factor model
can be estimated on the targeted set:

```python
list(soft.transform(x).columns)                # only the selected predictors

panel = ds.data.select([*soft.selected, "gdp"])  # targeted predictors + the target
res = nb.TwoStepDFM(n_factors=1).fit(panel, "gdp")
res.get_nowcast()
```

## Block and variable selection in pseudo real time

!!! note "New in 0.1.0"
    `select_blocks` and `select_variables` (innovation I7) are available as
    `nb.select_blocks` / `nb.select_variables` and from `nowcastbox.selection`.

`select_blocks(model, data, target, blocks, ...)` runs a greedy forward (or backward)
search over **groups of predictors**: at each step it adds the group that most reduces
the pseudo real-time RMSFE of the model (or an information criterion), re-estimating the
model over a set of vintages built from the release delays. `select_variables` does the
same over single series.

```python
from nowcastbox.selection import select_blocks

groups = {
    "g1": [f"x{i:02d}" for i in range(1, 8)],
    "g2": [f"x{i:02d}" for i in range(8, 15)],
    "g3": [f"x{i:02d}" for i in range(15, 21)],
}
path = select_blocks(
    nb.TwoStepDFM(n_factors=2),
    ds.data,
    "gdp",
    groups,
    direction="forward",
    scoring="rmsfe",
    validation={"n_vintages": 6, "refit_every": 3},
)
print(path.summary())
path.chosen, path.selected_series[:5]
```

| Argument | Meaning |
|---|---|
| `blocks` | `None` (panel block metadata), block names, `{block: [series]}` or a boolean DataFrame |
| `direction` | `"forward"` or `"backward"` |
| `scoring` | `"rmsfe"`, `"mae"` (pseudo real time) or `"aic"`, `"bic"`, `"hq"` |
| `always_include`, `max_blocks`, `min_improvement` | constraints of the search |
| `validation` | vintage design (`ValidationSettings`: number of vintages, offsets, window, `refit_every`) |

Selection is itself a modelling choice: evaluate the selected specification on a period
not used for the search (e.g. select on 2012–2018, backtest on 2019–2024).
