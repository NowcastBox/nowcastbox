# Aggregation and frequencies

A quarterly series is an **aggregate** of an unobserved monthly series. How the two are
linked depends on what the series measures, and the model uses that link to let the
monthly factors explain the quarterly data (see
[Temporal aggregation](../../theory/mariano-murasawa.md) for the derivations).

## Frequencies

`Frequency` has five members: `DAILY` (`"D"`), `WEEKLY` (`"W"`), `MONTHLY` (`"M"`),
`QUARTERLY` (`"Q"`) and `ANNUAL` (`"A"`). The base grid of a panel is its highest
frequency, and lower frequencies store their values in the last base period of each
native period.

```python
import numpy as np
import pandas as pd
import nowcastbox as nb
from nowcastbox.core import Frequency, aggregation_ratio

Frequency.from_value("quarterly"), Frequency.from_value(12), Frequency.from_value("QE")
aggregation_ratio("M", "Q"), aggregation_ratio("M", "A"), aggregation_ratio("Q", "A")
```

## Aggregation types

Weights apply to $(x_t, x_{t-1}, \dots)$ with $t$ the last high-frequency period of the
low-frequency period and $k$ the frequency ratio:

| `AggregationType` | Value | Weights ($k = 3$) | Use for |
|---|---|---|---|
| `FLOW` | `"flow"` | $(1, 1, 1)$ | levels of flows (sum over the quarter) |
| `AVERAGE` | `"average"` | $\tfrac13(1, 1, 1)$ | levels of averages (rates, indices) |
| `STOCK` | `"stock"` | $(1, 0, 0)$ | end-of-period stocks |
| `GROWTH_RATE` | `"mariano_murasawa"` (`"mm"`) | $(1, 2, 3, 2, 1)$, or $\tfrac13(1,2,3,2,1)$ normalised | growth rates of flows (Mariano & Murasawa, 2003) — the default for quarterly GDP growth |

```python
from nowcastbox.core import AggregationType
from nowcastbox.preprocessing import aggregation_weights, loading_constraints, temporal_aggregation

AggregationType.GROWTH_RATE.weights(3)                     # [1, 2, 3, 2, 1]
AggregationType.GROWTH_RATE.weights(3, normalize=True)     # [1/3, 2/3, 1, 2/3, 1/3]
aggregation_weights("M", "A", "flow")                      # twelve ones
agg = temporal_aggregation("M", "Q")                        # Mariano-Murasawa by default
agg.ratio, agg.n_lags, agg.weights.tolist()
```

The aggregation of each series is part of its metadata (`aggregations={"gdp": "flow"}`
when building the panel); `None` means the model default (Mariano-Murasawa for
`MixedFreqDFM`). In the EM model the weights become linear restrictions on the loadings
of the lower-frequency series, $R\lambda = 0$:

```python
R, q = loading_constraints(aggregation_weights("M", "Q"), n_factors=1)
R
```

## Converting between frequencies

```python
months = pd.period_range("2023-01", periods=12, freq="M")
ip = pd.Series(np.arange(1.0, 13.0), index=months)

nb.month_to_quarter(ip)                    # mean of the three months
nb.month_to_quarter(ip, "last")            # end-of-quarter value (also "first", 1/2/3)
nb.month_to_quarter(ip, "mariano_murasawa", complete=False)

q = nb.month_to_quarter(ip, "sum")
nb.quarter_to_month(q, "end")              # back on the monthly grid (3rd month)
nb.quarter_to_month(q, "divide")           # spread a flow evenly over the months
```

`to_lower_frequency(x, frequency, how)` and `to_higher_frequency(x, frequency, how)`
generalise these to any pair (`"linear"` and `"spline"` interpolation are available for
higher frequencies). `aggregate_panel` filters every base-frequency series of a panel with
the weights of the lowest frequency — the "aggregate the variables" variant of the
two-step model.

## Monthly, quarterly and annual in one model

Any pair with a **fixed ratio** (M/Q, M/A, Q/A) works in `MixedFreqDFM`: an annual series
simply gets 12 (flow/average) or 23 (Mariano-Murasawa) monthly lags in the state.

```python
ds = nb.load_simulated_dfm()
frame = ds.data.to_frame()
gdp_q = ds.data.to_native("gdp")
annual = gdp_q.groupby(gdp_q.index.year).sum()          # an annual flow
frame["gdp_annual"] = np.nan
for year, value in annual.iloc[:-1].items():             # last year not yet published
    frame.loc[pd.Period(f"{year}-12", "M"), "gdp_annual"] = value

freqs = {c: "M" for c in frame.columns if c.startswith("x")}
data = nb.MixedFrequencyData(
    frame,
    frequencies={**freqs, "gdp": "Q", "gdp_annual": "A"},
    aggregations={"gdp_annual": "flow"},
)
res = nb.MixedFreqDFM(n_factors=2).fit(data, target="gdp_annual")
res.nowcast.tail(2)[["observed", "out_of_sample", "std"]]
```

## Weekly and daily data (innovation I1)

!!! note "New in 0.1.0"
    Calendar-aware aggregation is supported by `MixedFreqDFM` and by the tools built on
    it (news, tracker, level contributions, parametric density bootstrap, diagnostics).
    `TwoStepDFM`, `BridgeEquation` and the benchmarks still need fixed-ratio pairs
    (monthly/quarterly/annual).

Between days and months, or weeks and quarters, the number of high-frequency periods per
low-frequency period **varies over the calendar** (28–31 days per month, 4 or 5 weeks).
NowcastBox handles these pairs with calendar-aware weights:

- the **base grid** is the highest frequency of the panel (`"W"` or `"D"`);
- a base period belongs to the native period that contains its **last day** (the pandas
  convention): the week 2020-01-27/2020-02-02 belongs to February 2020;
- a lower-frequency value is stored in the **last base period** of its native period
  (the last week ending in the month);
- aggregation weights are computed per period from the actual number of base periods
  (`AggregationType.calendar_weights`): sums and averages over 4 or 5 weeks, and the
  Mariano-Murasawa weights generalised to periods of unequal length.

```python
from nowcastbox.core.frequency import calendar_position, is_fixed_ratio, max_periods_per

is_fixed_ratio("M", "Q"), is_fixed_ratio("W", "M")
max_periods_per("W", "M"), max_periods_per("D", "M")
AggregationType.AVERAGE.calendar_weights(5)                 # a five-week month
AggregationType.GROWTH_RATE.calendar_weights(4, 5).round(3)  # 4-week month after a 5-week one
try:
    aggregation_ratio("W", "M")
except ValueError as err:
    print(err)               # no fixed ratio: use the calendar-aware helpers
```

### A weekly panel with a monthly target

```python
from nowcastbox.core.frequency import is_period_end

rng = np.random.default_rng(0)
weeks = pd.period_range("2022-01-03", periods=104, freq="W")
cycle = np.cumsum(rng.normal(0, 0.3, len(weeks)))
frame = pd.DataFrame(
    {
        "claims": cycle + rng.normal(0, 0.3, len(weeks)),     # weekly indicators
        "mobility": cycle + rng.normal(0, 0.3, len(weeks)),
    },
    index=weeks,
)
month_end = is_period_end(weeks, "M")                       # last week of each month
monthly_ip = pd.Series(cycle).rolling(4).mean().to_numpy() + rng.normal(0, 0.1, len(weeks))
frame["ip"] = np.where(month_end, monthly_ip, np.nan)       # stored in the month's last week

weekly = nb.MixedFrequencyData(
    frame,
    frequencies={"claims": "W", "mobility": "W", "ip": "M"},
    aggregations={"ip": "average"},
)
weekly.base_frequency, weekly.to_native("ip").tail(3)

res = nb.MixedFreqDFM(n_factors=1, max_iter=20).fit(weekly, target="ip")
res.nowcast.tail(2)[["observed", "out_of_sample", "std"]]
```

Daily panels work the same way (`freq="D"` base grid); the state then carries a month of
daily lags of the factors for each monthly series, so keep daily panels short or aggregate
daily financial data to weekly or monthly averages first:

```python
days = pd.period_range("2024-01-01", "2024-06-30", freq="D")
spread = pd.Series(2 + rng.normal(0, 0.1, len(days)).cumsum() / 10, index=days)
nb.preprocessing.to_lower_frequency(spread, "M", how="mean", complete=False).round(3)
```

For the release calendar, weekly and daily vintages use `days_to_end` /
`days_to_release` as backtest horizons (`months_to_end` counts calendar months).

### News, densities and simulation on weekly grids

The analysis tools rebuild the time-varying observation equation for each vintage, so
the usual calls work unchanged; `nb.simulate.weekly_dfm` draws a weekly + monthly +
quarterly panel from a known model:

```python
sim = nb.simulate.weekly_dfm(n_weeks=156, random_state=0)
old = sim.data.with_data(sim.data.data.iloc[:-2].reindex(sim.data.index))  # two weeks earlier
weekly_res = nb.MixedFreqDFM(idiosyncratic="iid", max_iter=20).fit(old, "gdp")
news = weekly_res.news(old, sim.data)          # releases of the last two weeks
news.check_identity(), weekly_res.distribution().mean.round(2)
```

The block bootstrap of densities needs a fixed number of base periods per target period
and is not available on weekly or daily grids (`method="parametric"` is).
