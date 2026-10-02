# Pre-selection of indicators (t-stat, SIS, LARS)

!!! note "New in 0.3.0"
    `preselect`, `sis`, `lars_select` and `lars_path` are available from
    `nowcastbox.selection` (ECB-toolbox parity, item 8).

The first step of the model-building workflow of the ECB Nowcasting Toolbox (Linzenich &
Meunier, 2024) is to shortlist, among many candidate indicators, those most related to the
target. `preselect` ranks every candidate with three complementary criteria and combines
the rankings into one weighted score:

| method | ranking criterion | reference |
|---|---|---|
| `"tstat"` | HAC $\lvert t\rvert$ of the indicator in a regression of the target on a constant, target lags and the indicator | Bai & Ng (2008) |
| `"sis"` | absolute marginal correlation (sure independence screening) | Fan & Lv (2008) |
| `"lars"` | order of entry on the least angle regression path | Efron et al. (2004) |

The definitions, the treatment of mixed frequencies and the score formula are in
[Theory › Pre-selection](../../theory/preselection.md).

## Basic use

```python
import nowcastbox as nb
from nowcastbox.selection import preselect

data = nb.load_simulated_dfm().data        # 20 monthly indicators + quarterly "gdp"

pre = preselect(
    data,
    target="gdp",
    methods=("tstat", "sis", "lars"),
    x_lags=(0, 1),                          # contemporaneous value and one-quarter lag
    weights=None,                           # equal weights
    top=8,
)
print(pre.summary())
pre.table()            # one row per series, sorted by the aggregated score
pre.selected           # ['x10', 'x19', 'x12', ...]
```

`pre.table()` has one row per series with

- `rank`, `score` (aggregated score in $[0, 1]$), `selected`, `best_lag`;
- `rank_tstat`, `rank_sis`, `rank_lars`: the rank of the series under each method (best
  over its lags) and the corresponding statistic (`abs_t`, `abs_corr`, `lars_entry`);
- `frequency`, `release_delay` (days), `category` and `blocks` from the series metadata,
  so the shortlist can be checked for timeliness and coverage of groups.

`pre.table("candidate")` gives the same information for every (series, lag) pair,
`pre.ranking("sis")` the series ordered by one method, `pre.results["lars"]` the full
candidate-level result of a method, and `pre.plot()` a bar chart of the scores (selected
series highlighted; `backend="plotly"` is also available).

The selected indicators feed directly into a model:

```python
small = pre.transform(data)                 # = data.select(pre.selected + ["gdp"])
res = nb.TwoStepDFM(n_factors=1).fit(small, "gdp")
```

## Mixed frequencies, leads and lags

The rankings are computed at the frequency of the target: monthly indicators are first
aggregated to quarters with their aggregation rule (`SeriesMetadata.aggregation`), or
with the `aggregation` argument when the metadata has none (`"average"` by default; a
mapping sets the rule per series), exactly as in a bridge equation. Quarterly candidates
are used as they are; candidates at a lower frequency are skipped with a warning.

```python
pre = preselect(data, "gdp", aggregation={"x01": "mariano_murasawa"}, top=8)

# Leads and lags in target periods: -1 = next quarter (lead), 2 = two quarters back
pre = preselect(data, "gdp", x_lags=(-1, 0, 1, 2), top=8)
pre.table("candidate").head()    # candidates x10_lead1, x10, x10_lag1, x10_lag2, ...
```

`align_to_target(data, "gdp", x_lags=(0, 1))` returns the aligned candidate frame and the
target, if you want to inspect what is ranked.

## Weights, methods and options

```python
# Give the t-statistic twice the weight of the other two criteria
preselect(data, "gdp", weights={"tstat": 2, "sis": 1, "lars": 1}, top=8)

# Only one or two criteria
preselect(data, "gdp", methods=("sis", "lars"), top=8)

# Forecast horizon, target lags as controls, lasso variant of LARS
preselect(data, "gdp", horizon=1, y_lags=1, lars_method="lasso", top=8)

# Restrict the candidates with a formula or a list
preselect(data, "gdp ~ x01 + x02 + x03 + x04", top=2)
```

By default 30 series are kept (`top=None` means $\min(30, N)$, the size used by Bai & Ng,
2008).

## Pseudo real time (no look-ahead)

In a pseudo real-time exercise the shortlist must be built with the information available
at each vintage. `as_of` masks every value released after the date (using the release
delays of the metadata, or `release_delays=` overrides) before anything is computed:

```python
pre_2015 = preselect(data, "gdp", as_of="2015-03-31", top=8)
pre_2015.n_obs        # quarters of GDP published by 31 March 2015
```

## The individual rankings

The building blocks can be used on their own, with the same input conventions as the
targeted predictors (`x` a frame of candidates, `y` the target aligned with it):

```python
from nowcastbox.selection import align_to_target, lars_path, lars_select, sis

x, y = align_to_target(data, "gdp")
sis(x, y).selected                         # top floor(n / log n) by |corr| (Fan & Lv)
lars_select(x, y, n_predictors=10).selected  # first 10 to enter the LARS path

ok = x.notna().all(axis=1) & y.notna()
z = (x[ok] - x[ok].mean()) / x[ok].std(ddof=0)
path = lars_path(z.to_numpy(), (y[ok] - y[ok].mean()).to_numpy(), method="lasso")
path.entry_order, path.alphas, path.coefs   # full path (Efron et al., 2004)
```

`select_targeted_predictors(x, y, method="sis" | "lars")` dispatches to these as well.
`lars_path` is an in-house implementation of the algorithm of Efron et al. (2004); the
test suite checks it against `sklearn.linear_model.lars_path` up to $\min(n-1, p)$ steps
(scikit-learn is not a dependency of NowcastBox).

## Next step

The ranking feeds the funnel strategy of the
[specification search](specification-search.md), which picks the number of indicators
together with the other settings of the model; the whole workflow is shown in
[Building a model from scratch](../model-building.md).
