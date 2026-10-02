# Specification search and Covid robustness

!!! note "New in 0.3.0"
    `SpecificationSearch`, `SearchResults`, `CovidRobustness` and `SpecifiedModel` are
    available from `nowcastbox.selection` (ECB-toolbox parity, item 9).

After the [pre-selection](preselection.md) of the indicators, the second step of the
model-building workflow of the ECB Nowcasting Toolbox (Linzenich & Meunier, 2024, §2.2-2.3)
is to search for a good **specification**: how many factors, how many lags in the factor
VAR, which blocks, how many indicators and which first period of the sample. The toolbox
draws many specifications within bounds, evaluates each one in pseudo real time and ranks
them by a score that weights the accuracy at each horizon and for several metrics. The
best specifications are then re-evaluated with different treatments of the pandemic
observations.

`SpecificationSearch` implements this workflow on top of
[`PseudoRealTimeBacktest`](../evaluation/backtesting.md) and
[`preselect`](preselection.md). The definitions are in
[Theory › Specification search](../../theory/specification-search.md).

## Basic use

```python
import nowcastbox as nb
from nowcastbox.benchmarks import AR
from nowcastbox.selection import SpecificationSearch

data = nb.load_simulated_dfm().data        # 20 monthly indicators + quarterly "gdp"

search = SpecificationSearch(
    model=nb.MixedFreqDFM(max_iter=50),
    data=data,
    target="gdp",
    space={
        "n_factors": [1, 2],               # list: choices
        "factor_lags": [1, 2],
        "n_series": (5, 15),               # (low, high): integers low..high
        "start": ["2000-01", "2005-01"],   # first period of the estimation sample
    },
    n_draws=8,                             # None = exhaustive grid
    ranking="preselect",                   # funnel: top n_series of the pre-selection
    backtest={
        "start": "2017-01-15",
        "end": "2019-06-15",
        "refit_every": 3,
        "benchmarks": [AR(p=1)],
    },
    score={"rmsfe": 0.7, "fda": 0.3},
    horizon_weights={"nowcast": 0.5, "backcast": 0.25, "forecast": 0.25},
    random_state=0,
    n_jobs=-1,
    checkpoint="search.parquet",
)
out = search.run()
print(out.summary(top=5))
out.table()            # one row per specification, best first
out.best_spec()        # {'n_factors': 1, 'factor_lags': 1, 'n_series': 14, 'start': '2000-01'}
model = out.best_model()
res = model.fit(data, "gdp")
```

`out.table()` has one row per specification with

- `rank`, `draw` (number of the random draw) and `score` (lower is better);
- the settings (`n_factors`, `factor_lags`, `n_series`, `start`...);
- `status` (`"ok"` or `"failed"`), `coverage` (share of the evaluated forecasts that
  could be computed) and `n_forecasts`;
- the criteria `"<metric>|<horizon>"`, e.g. `rmsfe|nowcast` and `fda|backcast`, and the
  same criteria for the reference benchmark (`ref|rmsfe|nowcast`...) when the backtest
  has benchmarks;
- `error` (first failure message), `n_failed_fits`, `n_warnings` and `seconds`.

`out.backtests[draw]` holds the full
[`BacktestResults`](../evaluation/backtesting.md) of every specification evaluated in
the run, so any other accuracy table or test is available.

## The search space

Every key of `space` is a parameter of the model (anything accepted by
`model.set_params`) or one of two special settings:

| key | meaning |
|---|---|
| `"start"` | first base period of the estimation sample (`None` = the whole sample) |
| `"n_series"` | number of indicators taken from the top of `ranking` (funnel strategy) |

The values of a setting are given as

- a **list** of choices: `[1, 2, 3]`, `["2005-01", "2010-01"]`, a list of block
  structures;
- a **pair of integers** `(low, high)`: every integer from `low` to `high`;
- a single value: a fixed setting.

A pair of integers is always read as a range; write a list (`[1, 3]`) for two choices.
Unknown keys, invalid values (checked with the model's own validation) and funnel sizes
larger than the ranking raise a `ValueError` before anything is estimated.

**Random draws and grids.** With `n_draws=n` each setting is drawn uniformly over its
values. Draw `i` uses its own generator, the `i`-th child of
`numpy.random.SeedSequence(random_state)`, so draws are reproducible and do not depend on
`n_draws`: increasing `n_draws` keeps the first draws and adds new ones. Duplicate draws
are evaluated once (`out.info["n_duplicates"]`). `n_draws=None` evaluates the exhaustive
grid (`ParameterSpace(space).size` specifications).

## Funnel strategy and look-ahead

With `"n_series"` in the space the model uses the first `n_series` indicators of a
ranking. There are two kinds of ranking:

<!-- skip-test -->
```python
from nowcastbox.selection import preselect

# Recomputed at every re-estimation vintage on the data available at that date
SpecificationSearch(..., ranking="preselect")
SpecificationSearch(..., ranking={"methods": ("tstat", "sis"), "x_lags": (0, 1)})

# Fixed ranking (a PreselectionResult or a list of names)
pre = preselect(data, "gdp", as_of="2016-12-31")
SpecificationSearch(..., ranking=pre)
```

The per-vintage ranking (`"preselect"` or a mapping of `preselect` options) runs
`preselect` on the information set of the vintage inside the backtest, so the indicators
are chosen with the data that were available at that date. Results are cached, so
specifications that share a vintage share the ranking. A fixed `PreselectionResult`
computed without `as_of`, or with an `as_of` after the first vintage, uses data released
later and triggers a `DataQualityWarning`.

## The backtest

`backtest` takes the keyword arguments of
[`PseudoRealTimeBacktest`](../evaluation/backtesting.md): `start`/`end`/`step` of the
vintages, `delay` or `calendar`, `refit_every`, `target_offsets`, `window`,
`benchmarks`, `fit_kwargs`, `preprocess`, `n_jobs`. Without `start`/`end`, the
`n_vintages` (default 12) monthly vintages that end with the last observed target period
are used. Failed fits are recorded as missing forecasts (`errors="warn"`), and a
specification with more than `max_missing` (default 10 %) missing forecasts gets an
infinite score.

Each specification is fitted with `refit_every` as in any backtest: between
re-estimations only the information set is updated. The funnel indicators are chosen at
each re-estimation and kept until the next one.

## The score

The score is a weighted mean of normalised losses over metrics and horizons:

```python
score={"rmsfe": 0.7, "fda": 0.3}                  # metric weights
horizon_weights={"nowcast": 0.5, "backcast": 0.25, "forecast": 0.25}
normalize="rank"                                  # "rank" | "relative" | "none"
```

- **Metrics** (`SCORE_METRICS`): `"rmsfe"`, `"mse"`, `"mae"`, `"bias"` (absolute value)
  and `"fda"`, the forecast directional accuracy (loss `1 - FDA`), with the same
  definitions as [`BacktestResults.metrics`](../evaluation/metrics.md).
- **Horizons** are the values of the `horizon` column of the forecast table. The default,
  `"kind"`, is `backcast`/`nowcast`/`forecast`. `horizon="months_to_end"` with keys
  `0, 1, 2` weights the months before the end of the quarter instead. When
  `horizon_weights` is not given, all evaluated horizons get the same weight.
- **Normalisation.** `"rank"` (default) replaces each loss by its rank among the
  specifications, mapped to $[0, 1]$. This makes RMSFE and FDA comparable.
  `"relative"` divides each loss by that of the reference benchmark (the first entry of
  `backtest["benchmarks"]`). `"none"` uses the raw losses.

`periods` restricts the criteria to some target periods. It takes a
`(first, last)` pair, a list of pairs, `{"label": pair}` or `"ex-covid"`, as in
[`BacktestResults.split_periods`](../evaluation/metrics.md).

The criteria are stored, so the same evaluations can be re-ranked with other weights
without re-estimating anything:

```python
out.table({"fda": 1.0}, {"nowcast": 1.0})              # rank by nowcast FDA only
out.table({"rmsfe": 1.0}, normalize="relative")        # relative RMSFE vs. AR
out.scores()                                           # the default score, by draw
```

`nowcastbox.selection.weighted_score(criteria, score, horizon_weights, normalize=...)`
computes the score of any table of criteria.

## Checkpointing long searches

With `checkpoint="search.parquet"`, the criteria of each specification are written to
disk as soon as it is evaluated. The write is atomic: a temporary file replaces the old
one. Running the same search again reads the file and evaluates only the missing
specifications:

```python
out = search.run()             # interrupted after 120 of 200 draws...
out = search.run()             # ...resumes: evaluates the remaining 80
out.info["n_resumed"], out.info["n_evaluated"]
```

The file records a fingerprint of the settings: model, data, target, backtest, ranking,
metrics, horizon column and evaluation periods. A checkpoint written by a search with
different settings is refused with a `ValueError`. The score weights are not part of the
fingerprint, because the score is recomputed from the stored criteria. Without a Parquet
engine (`pyarrow`, in the `test` extra), the checkpoint is written as CSV next to the
requested path, with a warning.

Resumed specifications have no entry in `out.backtests` (only their criteria are
stored). With `random_state=None` the draws are random and cannot be resumed;
`out.info["entropy"]` gives the seed that was used.

## Covid robustness

The last step of the ECB workflow re-evaluates the best specifications with each
treatment of the pandemic observations, on the target periods after the pandemic (the
simulated data end in 2019, so the example evaluates 2019):

```python
rob = out.covid_robustness(
    top=3,
    treatments=("none", "dummy", "mask", "outliers"),
    evaluate_from="2019Q1",   # with real data: the quarters after the pandemic, e.g. "2022Q1"
)
rob.table()        # one row per (specification, treatment), best first
rob.pivot()        # scores: one row per specification, one column per treatment
rob.best()         # {'draw': 7, 'treatment': 'dummy', 'score': ..., 'note': ...}
model = rob.best_model()
```

The treatments map to existing options:

| treatment | `MixedFreqDFM` | models without these options (e.g. `TwoStepDFM`) |
|---|---|---|
| `"none"` | `covid="none"`, `outliers="none"` | unchanged |
| `"dummy"` | `covid="dummy"`: impulse dummies in the factor VAR over the pandemic window | skipped, with a note |
| `"mask"` | `covid="mask"`: pandemic observations treated as missing | skipped, with a note |
| `"outliers"` | `outliers="auto"`: robust EM that drops flagged observations | IQR correction of the predictors at every vintage (`OutlierCorrection`, never the target) |

The pandemic window is the model's `covid_window` (default March 2020 to December 2021).
Skipped pairs appear in `rob.table()` with `status="skipped"` and the reason in `note`,
and as NaN in `rob.pivot()`. The score uses the weights and normalisation of the search
unless `score=`, `horizon_weights=` or `normalize=` are given. Horizons that do not exist
in the evaluation window are dropped from the weights, with a warning. For example, there
is no backcast of a period before `evaluate_from`. `backtest={"end": ...}` overrides the
backtest settings, for instance to extend the vintages beyond those of the search.

Custom treatments are functions `(model, target) -> TreatmentPlan`:

```python
from nowcastbox.selection import TreatmentPlan

def exclude_2020q2(model, target):
    if "exclude_periods" not in model.get_params():
        return TreatmentPlan.unsupported("no exclude_periods option")
    return TreatmentPlan({"exclude_periods": [("2020-04", "2020-06")]}, note="2020Q2 out")

rob = out.covid_robustness(top=3, treatments={"none": "none", "q2": exclude_2020q2})
```

## Using a specification

`best_model()` returns a `SpecifiedModel`: the estimator with the parameters of the
specification, plus its sample start and funnel. Its `fit(data, target)` keeps the target
and the top `n_series` indicators of the ranking, drops the periods before `start`, and
returns the inner model's results. With `ranking="preselect"` the ranking is recomputed
on the data given to `fit`, so the final model uses the latest pre-selection. A
`SpecifiedModel` can also be passed to `PseudoRealTimeBacktest` (also with
`refit_every > 1`: its `update` keeps the fitted parameters and indicators) or to
`NowcastExperiment`, like any nowcaster:

```python
from nowcastbox.selection import SpecifiedModel

model = SpecifiedModel(nb.MixedFreqDFM, params={"n_factors": 2}, start="2005-01",
                       n_series=12, ranking={"methods": ("sis", "lars")})
res = model.fit(data, "gdp")
model.columns_     # the indicators used
```

!!! warning "Preprocessing travels with the model"
    `best_model()` (and `CovidRobustness.best_model()`) put the `preprocess` of the
    search's `backtest=` inside the returned `SpecifiedModel`. When you backtest that
    model again, do not pass the same `preprocess` to `PseudoRealTimeBacktest`, or the
    vintages are preprocessed twice. The [pipeline](../pipeline-cli.md#model-building)
    takes care of this.

## Cost

Each specification is a full pseudo real-time backtest, so the cost is roughly
`n_draws × n_vintages / refit_every` model estimations. To keep it manageable:

- use `refit_every` of 3 (quarterly re-estimation) or more;
- start with a coarse random search (`n_draws` of 50 to 100) and a short backtest;
- set `n_jobs=-1` to evaluate specifications in parallel;
- use `checkpoint=`, so an interrupted search can be resumed;
- for `MixedFreqDFM`, cap `max_iter` during the search and refit the final model with
  the default.

## References

- Linzenich, J. & Meunier, B. (2024). Nowcasting made easier: a toolbox for economists.
  ECB Working Paper No. 3004.
- Bergstra, J. & Bengio, Y. (2012). Random search for hyper-parameter optimization.
  *Journal of Machine Learning Research*, 13, 281-305.
- Bańbura, M., Giannone, D., Modugno, M. & Reichlin, L. (2013). Now-casting and the
  real-time data flow. In *Handbook of Economic Forecasting*, vol. 2A, 195-237.
