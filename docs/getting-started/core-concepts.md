# Core concepts

## The mixed-frequency data grid

Every panel lives on a **contiguous base grid**: a `pandas.PeriodIndex` at the highest
frequency of the panel (monthly in most applications). Series of a lower frequency
store each value in the **last base period** of its native period:

| base period | `ip` (monthly) | `gdp` (quarterly) |
|---|---|---|
| 2024-01 | 101.2 | NaN |
| 2024-02 | 101.9 | NaN |
| 2024-03 | 102.4 | **0.8** (2024Q1) |
| 2024-04 | 102.0 | NaN |

The NaN in January and February are *structural*: they are not missing data, there is
simply no quarterly value to store. `MixedFrequencyData` keeps that distinction through
per-series metadata, so masks and models know which cells are storage **slots**.

```python
import numpy as np
import pandas as pd
import nowcastbox as nb

idx = pd.period_range("2019-01", "2024-12", freq="M")      # 72 months
rng = np.random.default_rng(0)
cycle = np.sin(np.arange(72) / 6) + rng.normal(0, 0.3, 72)
gdp = pd.Series(cycle, index=idx).rolling(3).mean()
gdp[idx.month % 3 != 0] = np.nan                            # quarterly slots only
gdp["2024-12"] = np.nan                                     # 2024Q4 not released yet
frame = pd.DataFrame(
    {
        "ip": cycle + rng.normal(0, 0.5, 72),
        "pmi": cycle + rng.normal(0, 0.8, 72),
        "gdp": gdp,
    },
    index=idx,
)
frame.loc["2024-11":, "ip"] = np.nan          # IP is published with a longer delay

data = nb.MixedFrequencyData(
    frame,
    frequencies={"ip": "M", "pmi": "M", "gdp": "Q"},
    release_delays={"ip": 40, "pmi": 1, "gdp": 60},   # days after the period ends
    categories={"ip": "hard", "pmi": "soft", "gdp": "hard"},
)
print(data)
data.slot_mask().tail(3)       # where each series *can* have a value
```

A value of a quarterly series outside its slot (e.g. in February) raises
`NowcastDataError`: the contract is enforced, never guessed silently.

## The ragged edge

Indicators are published with different delays, so at any date the end of the panel is
**ragged**: the PMI is out for last month, industrial production stops two months
earlier and GDP a whole quarter earlier. Nowcasting models exploit the most recent
observations of each series instead of cutting the panel to a balanced block.

```python
data.last_observed()           # last observed period of every series
data.ragged_edge_mask().tail(4)  # slots after the last observation
```

Ragged-edge cells are *not* filled by `prepare_panel` (unless asked): the Kalman filter
handles them as missing observations and that is precisely what produces the nowcast.

## Vintages

A **vintage** is the information set available at a date. With publication delays (in
days after the end of the reference period) the vintage of any date can be rebuilt from
the final data — a *pseudo real-time* vintage:

```python
v = data.as_of("2024-11-15")   # same grid, later values removed
v.last_observed()
```

The rule is: the observation of native period $p$ is published at
$\text{end}(p) + \text{delay}$ and is kept if that date is on or before the vintage
date. Pseudo real-time vintages ignore **revisions**; real vintages (several published
values for the same period) live in a [`VintageStore`](../user-guide/vintages/real-vintages.md).

Two nested vintages $\Omega_v \subseteq \Omega_{v+1}$ are the input of the
[news decomposition](../user-guide/news/news-decomposition.md), and a sequence of
vintages is the input of a [backtest](../user-guide/evaluation/backtesting.md).

## Metadata

Each series carries a `SeriesMetadata`: native frequency, transformation (to apply or
already applied), release delay, blocks, category (`hard`, `soft`, `financial`,
`other`), aggregation type and description. Every operation (`select`, `truncate`,
`as_of`, transformations) preserves it.

```python
data.metadata_frame()[["frequency", "release_delay", "category"]]
```

## Units and standardisation

Models standardise the panel internally (mean zero, unit variance over the observed
values) and report every result — nowcast, standard deviations, news impacts — in the
**units of the target as passed to `fit`** (after transformations). Invert
transformations with `invert_transforms` to go back to levels.

## Results

Every estimator returns an immutable `NowcastResults` whose `nowcast` frame is indexed by
the target's **native** periods (quarters for GDP), with exactly one of
`in_sample` (target observed) or `out_of_sample` (backcast, nowcast, forecast) filled per
period, plus `std` and 68 %/90 % bands when available:

```python
res = nb.TwoStepDFM(n_factors=1).fit(v, target="gdp")
res.nowcast
```

Results expose `summary()`, `to_frame()`, `plot(kind)`, `save()/load()` and the analysis
methods `news`, `nowcast_tracker`, `level_contributions`, `distribution` and
`diagnostics`.

## Formulas

The target can be a formula selecting the predictors:

```python
nb.TwoStepDFM(n_factors=1).fit(v, target="gdp ~ ip + pmi").model_name
```

`"gdp ~ ."` uses every other series and `"gdp ~ . - pmi"` all but `pmi`.
