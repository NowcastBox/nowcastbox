# Backtesting

A nowcasting model is evaluated by replaying history: at each vintage date the model is
estimated with the information available then, it produces backcasts, nowcasts and
forecasts, and these are compared with the realised values. `PseudoRealTimeBacktest`
runs that loop for the model and any number of benchmarks and returns a tidy
`BacktestResults`.

## A pseudo real-time backtest

```python
import nowcastbox as nb
from nowcastbox.benchmarks import AR, RandomWalk

ds = nb.load_simulated_dfm()

backtest = nb.PseudoRealTimeBacktest(
    model=nb.MixedFreqDFM(n_factors=2, max_iter=50),
    data=ds.data,                       # final data; vintages are rebuilt from the delays
    target="gdp",
    delay=ds.delay,                     # or calendar=ReleaseCalendar(...)
    start="2018-07-15",
    end="2019-12-15",
    step="M",                           # one vintage per month ("release": every release)
    benchmarks={"AR": AR(p=1), "RW": RandomWalk()},
    refit_every=3,                      # re-estimate quarterly, update the data monthly
)
out = backtest.run()
print(out.summary())
```

At each vintage the backtest evaluates the target periods at `target_offsets=(-1, 0, 1)`
relative to the vintage's quarter: the **backcast** of the previous quarter (while it is
not yet released), the **nowcast** of the current quarter and the **forecast** of the next.

| Argument | Default | Meaning |
|---|---|---|
| `model` | — | main model (cloned at every re-estimation) |
| `data` | — | final panel (pseudo real time) or a `VintageStore` (real time) |
| `calendar` / `delay` | panel metadata | release rule of the vintages |
| `start`, `end`, `step` | — / — / `"M"` | vintage dates (`"M"`, `"2W"`, `"Q"`, days, `"release"`) |
| `benchmarks` | `None` | list or `{name: forecaster}`; scikit-learn regressors are wrapped automatically |
| `window`, `window_length` | `"expanding"` | expanding or rolling estimation window (months) |
| `refit_every` | `1` | re-estimate every k vintages; in between only the information is updated |
| `target_offsets` | `(-1, 0, 1)` | backcast / nowcast / forecast |
| `include_released` | `False` | also evaluate periods already released at the vintage |
| `actual` | `"final"` | realisations: final data, or with a store `"first"`, `"latest"` or the n-th release |
| `n_jobs` | `None` | parallel vintages (joblib) |
| `errors` | `"raise"` | `"warn"` records NaN when a fit fails |

## Results

```python
out.to_frame().head()                  # one row per vintage x target period x model
out.rmsfe_by_horizon()                 # RMSFE by months to the end of the target quarter
out.relative_to("AR")                  # RMSFE ratios (< 1: better than AR)
out.metrics(metrics=("rmsfe", "mae", "bias", "n"))
out.evaluable().head()                 # the common sample used for comparisons
```

The default horizon label is `months_to_end`: months between the vintage and the end of
the target quarter (2 = first month of the quarter, 0 = last month, negative =
backcasts). `days_to_end` and `days_to_release` give finer horizons (use them for weekly
or daily vintages).

Statistical comparisons (Diebold-Mariano, Giacomini-White, Model Confidence Set) are on
[Forecast comparison tests](forecast-comparison-tests.md).

```python
fig = nb.visualization.plot_rmsfe_by_horizon(out.rmsfe_by_horizon())
out.to_parquet("backtest.parquet")     # needs the [data] extra (pyarrow)
```

## Real-time evaluation with real vintages

With a `VintageStore` each vintage uses the values **actually published** at that date,
and `actual` chooses which release is the outcome. Here a store of Brazilian GDP growth
vintages is built from the IBGE level vintages and two univariate benchmarks are compared
against the first release:

```python
import pandas as pd

levels = nb.load_brazil_vintages(series="pib")
growth = {}
for date in levels.vintage_dates("pib"):
    if date >= pd.Timestamp("2014-01-01"):
        panel = levels.as_of(date, series="pib", start="2003-01", end=date.to_period("M"))
        growth[date] = nb.apply_transforms(panel, "qoq", frequency={"pib": "Q"})
store = nb.VintageStore.from_vintages(growth, frequencies={"pib": "Q"})

real_time = nb.PseudoRealTimeBacktest(
    model=AR(p=1),
    data=store,
    target="pib",
    start="2016-01-01",
    end="2024-12-31",
    step="release",                    # one vintage per IBGE release
    benchmarks={"Mean": nb.benchmarks.HistoricalMean()},
    actual="first",                    # score against the first release
)
print(real_time.run().summary())
```

For a factor model in real time, the store must contain every indicator; build it from
connectors with ALFRED-style vintages or combine `VintageStore.from_calendar(panel, delays)`
(pseudo vintages of the indicators) with the real target vintages using `add`.

## Good practice

- Evaluate on a **common sample** of target periods across models (the default).
- Report accuracy **by horizon**: nowcasts improve as the quarter progresses; averages
  over horizons hide that.
- Include simple benchmarks (AR, random walk, mean) and the relevant competitor
  (bridge, MIDAS, survey expectations).
- Keep the evaluation period out of any model-selection step.
- Use `refit_every > 1` and warm starts to keep long backtests fast; check that the
  results do not change materially with `refit_every=1`.
