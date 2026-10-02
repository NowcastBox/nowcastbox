# Building a model from scratch

!!! note "New in 0.3.0"
    This guide chains the model-building tools added for parity with the ECB Nowcasting
    Toolbox (Linzenich & Meunier, 2024, §2): `nb.preselect`, `nb.SpecificationSearch`
    (with its Covid robustness step) and `nb.BridgeCombination`.

Starting from a large panel of candidate indicators and a target, the ECB toolbox builds a
nowcasting model in three steps:

1. **Pre-selection**: rank the indicators against the target (t-statistic, sure
   independence screening and least angle regression) and keep the best ones.
2. **Specification search**: draw many specifications (number of factors, lags, number
   of indicators taken from the top of the ranking, first period of the sample), evaluate
   each in pseudo real time and rank them by a score weighted by horizon and metric.
3. **Covid robustness**: re-evaluate the best specifications with each treatment of the
   pandemic observations on the period after it, and keep the best pair.

A combination of small bridge equations gives a transparent benchmark for the result.
Every step only uses data available at each vintage, so the evaluation is free of
look-ahead. The pages linked below give the details of each step.

## 1. Pre-selection

```python
import warnings

import nowcastbox as nb

warnings.simplefilter("ignore", nb.DataQualityWarning)
data = nb.load_simulated_dfm().data          # 20 monthly indicators + quarterly "gdp"

pre = nb.preselect(
    data,
    "gdp",
    methods=("tstat", "sis", "lars"),
    x_lags=(0, 1),                           # each indicator also with one quarter of lag
    weights={"tstat": 1, "sis": 1, "lars": 2},
    top=10,
    as_of="2018-12-31",                      # rankings with the data released by then
)
pre.table().head()                           # rank, score, best lag, per-method ranks...
pre.selected                                 # the 10 best indicators
```

See [Pre-selection](selection/preselection.md) for the rankings, the aggregated score
and the treatment of mixed frequencies.

## 2. Specification search

The funnel strategy (`"n_series"` in the space) takes the top indicators of the
pre-selection, which `ranking=` recomputes on the information set of every
re-estimation vintage:

```python
from nowcastbox.benchmarks import AR

search = nb.SpecificationSearch(
    model=nb.TwoStepDFM,
    data=data,
    target="gdp",
    space={"n_factors": [1, 2], "n_series": (4, 10), "start": ["2000-01", "2006-01"]},
    n_draws=6,
    ranking={"methods": ("tstat", "sis", "lars"), "x_lags": (0, 1)},
    backtest={"start": "2018-01-15", "end": "2019-06-15", "benchmarks": [AR(p=1)]},
    score={"rmsfe": 0.7, "fda": 0.3},
    horizon_weights={"backcast": 0.25, "nowcast": 0.5, "forecast": 0.25},
    random_state=0,
)
out = search.run()
out.table(top=3)[["rank", "score", "n_factors", "n_series", "start"]]
```

See [Specification search](selection/specification-search.md) for the space, the score,
checkpoints and parallel runs.

## 3. Covid robustness

`covid_robustness` re-runs the best specifications with each treatment of the pandemic
observations (none, dummies, masking, outlier correction) and scores them on the target
periods from `evaluate_from` (in practice the quarters after the pandemic; the simulated
data end in 2019, so the example uses 2019). Treatments a model cannot apply are skipped
with a note:

```python
rob = out.covid_robustness(top=2, treatments=("none", "outliers"), evaluate_from="2019Q1")
rob.pivot()                                  # score by specification and treatment
model = rob.best_model()                     # unfitted SpecifiedModel
res = model.fit(data, "gdp")
res.get_nowcast()
```

## 4. A benchmark: combining bridge equations

```python
combo = nb.BridgeCombination(max_monthly=2, max_quarterly=0, combine="mean")
bridge = combo.fit(data.select([*pre.selected, "gdp"]), "gdp")
bridge.info["n_equations"], bridge.get_nowcast()
```

See [Combination of bridge equations](models/bridge-combination.md); both models can be
compared in one [backtest](evaluation/backtesting.md), with the combination as a
benchmark (`nb.benchmarks.BridgeCombinationBenchmark`).

## The same workflow in production

The [pipeline](pipeline-cli.md#model-building) runs the three steps from a YAML spec
before the nowcast (`selection.preselect`, `selection.search` with `covid_robustness`
and `apply: true`), stores their tables in the snapshot and the Excel workbook, and lists
them in the HTML report. `nowcastbox init --template model_building` writes an example.

## References

- Linzenich, J. & Meunier, B. (2024). Nowcasting made easier: a toolbox for economists.
  ECB Working Paper No. 3004.
- Bańbura, M., Belousova, I., Bodnár, K. & Tóth, M. B. (2023). Nowcasting employment in
  the euro area. ECB Working Paper No. 2815.
