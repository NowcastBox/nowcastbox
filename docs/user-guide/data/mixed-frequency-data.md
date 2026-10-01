# Mixed-frequency data

`MixedFrequencyData` is the container every model, vintage and transformation works
with. It wraps a `DataFrame` on a contiguous base `PeriodIndex` and a
`SeriesMetadata` per series (see [Core concepts](../../getting-started/core-concepts.md)).

## Building a panel

=== "From native series"

    ```python
    import numpy as np
    import pandas as pd
    import nowcastbox as nb

    rng = np.random.default_rng(0)
    months = pd.period_range("2022-01", "2023-12", freq="M")
    ip = pd.Series(100 + rng.normal(0, 1, 24).cumsum(), index=months)
    pmi = pd.Series(50 + rng.normal(0, 2, 23), index=months[:-1])  # one month shorter
    gdp = pd.Series(
        [1.0, 0.8, 0.5, 0.6, 0.9, 1.1, 0.7],
        index=pd.period_range("2022Q1", periods=7, freq="Q"),
    )

    data = nb.MixedFrequencyData.from_series(
        {"gdp": gdp, "ip": ip, "pmi": pmi},
        base_frequency="M",
        release_delays={"gdp": 60, "ip": 40, "pmi": 1},
        categories={"gdp": "hard", "ip": "hard", "pmi": "soft"},
        blocks={"gdp": ["global", "real"], "ip": ["global", "real"], "pmi": ["global", "soft"]},
        descriptions={"gdp": "Real GDP, % q/q", "ip": "Industrial production index"},
    )
    print(data)
    ```

=== "From a monthly DataFrame"

    ```python
    frame = data.to_frame()          # quarterly values in the 3rd month of the quarter
    same = nb.MixedFrequencyData(
        frame,
        frequencies={"gdp": "Q", "ip": "M", "pmi": "M"},   # or a legend column: 4 / 12
        release_delays={"gdp": 60, "ip": 40, "pmi": 1},
    )
    same.frequencies
    ```

Frequencies can be given as `Frequency` members, strings (`"M"`, `"monthly"`, `"Q"`,
`"QE"`, `"A"`, `"Y"`) or periods per year (`12`, `4`, `1` — the convention of R's `ts`).
When omitted they are inferred from the observation pattern (`infer_frequency`), which is
convenient but less safe than stating them.

`as_mixed_frequency_data(df, frequency=...)` is the coercion used by every `fit`, so
plain `DataFrame`s are accepted everywhere with a `frequency=` argument.

## Inspecting

```python
data.metadata_frame()                  # one row per series
data.series_by_frequency()             # {Frequency.MONTHLY: [...], Frequency.QUARTERLY: [...]}
data.to_native("gdp").tail(3)          # quarterly series on its own PeriodIndex
data.n_observations()                  # observed values per series
data.blocks                            # series x blocks (bool)
```

## Masks

| Method | True where |
|---|---|
| `observation_mask()` | a value is observed |
| `slot_mask()` | the cell is a storage slot of the series (every month for monthly, the 3rd month for quarterly) |
| `missing_mask()` | a slot has no value (gaps **and** ragged edge) |
| `ragged_edge_mask()` | a slot comes after the last observation of the series |

```python
data.ragged_edge_mask().tail(3)
data.last_observed()
```

## Deriving new panels

Every method returns a **new**, validated object (the container is immutable):

```python
data.select(["gdp", "ip"])            # subset of series
data.drop(["pmi"])
data.truncate("2023Q1")               # quarter labels: start -> January, end -> March
data.extend(3)                        # three empty months: forecast horizon
data.with_metadata("pmi", category="soft", release_delay=2)
data.as_of("2023-11-15")              # pseudo real-time vintage
```

## Standardisation

```python
z, stats = data.standardize()          # (x - mean) / std over observed values, ddof=1
stats.mean.round(2)
back = z.destandardize(stats)
assert back.equals(data, atol=1e-10)
```

Models standardise internally and report results in the original units, so you rarely
need this directly. `stats.inverse_series(values, column, scale_only=True)` rescales
standard deviations and news impacts (no mean).

## Validation rules

The container refuses ambiguous data with a `NowcastDataError`:

- values outside the slots of a lower-frequency series;
- series of a *higher* frequency than the base grid (use a finer base grid instead);
- infinite values, duplicated or empty names;
- metadata for series that do not exist.

Gaps in the index are filled with NaN and reported by a `DataQualityWarning`.

```python
bad = frame.copy()
bad.loc["2023-02", "gdp"] = 1.0        # a quarterly value in February
try:
    nb.MixedFrequencyData(bad, frequencies={"gdp": "Q", "ip": "M", "pmi": "M"})
except nb.NowcastDataError as err:
    print(err)
```
