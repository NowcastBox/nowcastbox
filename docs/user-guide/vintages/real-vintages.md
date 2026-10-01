# Real-time vintages (innovation I8)

Statistical offices **revise** published data. A genuine real-time exercise uses, at
each date, the values that were published then — not today's revised numbers.
`VintageStore` keeps every published value in long format (Croushore & Stark, 2001):

| `series` | `reference_period` | `vintage_date` | `value` |
|---|---|---|---|
| pib | 2023Q4 | 2024-03-01 | 100.0 |
| pib | 2023Q4 | 2024-06-04 | 100.3 |

## Brazilian GDP vintages

`load_brazil_vintages()` holds the real-time vintages of Brazilian quarterly GDP and its
components as published by IBGE (Contas Nacionais Trimestrais, 2010 onwards).

```python
import nowcastbox as nb

store = nb.load_brazil_vintages()
store
store.series
store.vintage_dates("pib")[:5]
```

## The data as of a date

```python
pib_2020 = store.as_of("2020-06-30", series="pib", start="2018-01", end="2020-06")
pib_2020.tail(6)          # 2020Q1 as first published; 2020Q2 not yet out
```

`as_of(..., as_mixed=True)` returns a `MixedFrequencyData`; `latest()` gives the most
recent vintage and `nth_release(n)` the $n$-th published value of every period (`0` =
first release).

```python
first = store.nth_release(0, series="pib")
latest = store.latest(series="pib")
first.dropna().tail(3).join(latest.dropna().tail(3), lsuffix="_first", rsuffix="_latest")
```

## Revisions

```python
store.vintage_matrix("pib").iloc[-5:, -4:]     # periods x vintages ("revision triangle")
store.revisions("pib").tail()                   # every revision with its size
store.revision_summary("pib")                   # mean, mean absolute, noise-to-signal...
```

The summary statistics follow Aruoba (2008): mean revision (bias), mean absolute and RMS
revision, and the noise-to-signal ratio $\sigma_r / \sigma_{x}$.

## Building your own store

```python
import pandas as pd

ds = nb.load_brazil_nowcast(columns=["pib", "ibc_br"], start="2022-01", end="2024-06")
records = pd.DataFrame(
    {
        "series": ["gdp", "gdp", "gdp"],
        "reference_period": ["2024Q1", "2024Q1", "2024Q2"],
        "vintage_date": ["2024-06-04", "2024-09-03", "2024-09-03"],
        "value": [0.8, 1.0, 1.4],
    }
)
mine = nb.VintageStore(records, frequencies={"gdp": "Q"})
mine.as_of("2024-07-01")["gdp"].dropna()

# pseudo real-time vintages of a final panel, stored as a VintageStore
pseudo = nb.VintageStore.from_calendar(ds.data, ds.delay)
pseudo.n_records
```

Stores read and write CSV and Parquet (`to_csv`, `from_csv`, `to_parquet`, `from_parquet`;
Parquet needs the `[data]` extra), and combine with `add`. `VintageStore.from_vintages`
builds a store from a mapping `{date: panel}`, e.g. downloaded ALFRED vintages (see
[FRED / ALFRED](../data-sources/fred.md)).

## Real-time backtests

Pass a `VintageStore` as `data` to `PseudoRealTimeBacktest` to evaluate with the values
actually available at each date, and choose which release counts as the outcome
(`actual="first"`, `"latest"` or the $n$-th release). See
[Backtesting](../evaluation/backtesting.md#real-time-evaluation-with-real-vintages).
