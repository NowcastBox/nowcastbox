# Transformations

Dynamic factor models need stationary inputs. NowcastBox transformations are **named,
composable, frequency-aware and invertible**: the same specification transforms a panel
and rebuilds levels from a nowcast.

## Named transformations

| Name | Meaning | Lag unit |
|---|---|---|
| `"level"` (`"none"`, `"identity"`) | $x_t$ | — |
| `"log"` | $\ln x_t$ | — |
| `"diff"` | $x_t - x_{t-1}$ | native period |
| `"pct_change"` (`"growth"`) | $x_t / x_{t-1} - 1$ | native period |
| `"dlog"` (`"log_diff"`) | $\ln x_t - \ln x_{t-1}$ | native period |
| `"mom"` | $x_t / x_{t-1\,\text{month}} - 1$ | calendar month |
| `"qoq"` | $x_t / x_{t-1\,\text{quarter}} - 1$ | calendar quarter |
| `"yoy"` | $x_t / x_{t-1\,\text{year}} - 1$ | calendar year |
| `"annual_diff"` | $x_t - x_{t-1\,\text{year}}$ | calendar year |
| `"dlog_yoy"` | $\ln x_t - \ln x_{t-1\,\text{year}}$ | calendar year |
| `"diff_of_yoy"` | $\Delta_1$ of the year-on-year growth | |
| `"diff_of_annual_diff"` | $\Delta_1 \Delta_{\text{year}} x_t$ | |

Calendar lags adapt to the series' native frequency: `"yoy"` is a 12-period change for a
monthly series and a 4-period change for a quarterly one.

### Legacy codes 0–7

The numeric codes used in the nowcasting literature (and in the R package
`nowcasting`) are accepted as shortcuts:

| Code | Name | Formula |
|---|---|---|
| 0 | `level` | $x_t$ |
| 1 | `pct_change` | $(x_t - x_{t-1})/x_{t-1}$ |
| 2 | `diff` | $x_t - x_{t-1}$ |
| 3 | `diff_of_yoy` | $\Delta_1\left[(x_t - x_{t-12})/x_{t-12}\right]$ |
| 4 | `diff_of_annual_diff` | $\Delta_1 \Delta_{12} x_t$ |
| 5 | `annual_diff` | $x_t - x_{t-12}$ |
| 6 | `yoy` | $(x_t - x_{t-12})/x_{t-12}$ |
| 7 | `qoq` | $(x_t - x_{t-3})/x_{t-3}$ |

## Transform objects

```python
import numpy as np
import pandas as pd
import nowcastbox as nb
from nowcastbox.preprocessing import get_transform, transform_from_code

get_transform("dlog"), transform_from_code(3)

# compose with "|": apply the left transform first
t = nb.Log() | nb.Diff(1)
quarters = pd.period_range("2020Q1", periods=6, freq="Q")
gdp = pd.Series([100.0, 101.0, 99.5, 102.0, 103.1, 104.0], index=quarters)
growth = t.apply(gdp)
growth.round(4).tolist()
```

`Diff(lag)` and `PctChange(lag)` accept integers (native periods) or calendar units
(`"month"`, `"quarter"`, `"year"`); `Scale(factor)` multiplies; `Identity()` does
nothing.

## Inverting: from a growth nowcast back to levels

```python
history = gdp.iloc[:4]                          # known levels
forecast = pd.Series([0.012, 0.008], index=quarters[4:])  # log-growth nowcasts
t.inverse(forecast, history).round(2)
```

## Transforming a panel

`apply_transforms` takes one specification for all series, a mapping by name, or a
sequence aligned with the columns (e.g. a legend column):

```python
months = pd.period_range("2020-01", periods=24, freq="M")
rng = np.random.default_rng(1)
levels = pd.DataFrame(
    {
        "ip": 100 * np.exp(np.cumsum(rng.normal(0.002, 0.01, 24))),
        "cpi": 100 * np.exp(np.cumsum(rng.normal(0.004, 0.002, 24))),
        "gdp": np.where(np.arange(24) % 3 == 2, np.linspace(100, 106, 24), np.nan),
    },
    index=months,
)
data = nb.MixedFrequencyData(levels, frequencies={"ip": "M", "cpi": "M", "gdp": "Q"})

growth = nb.apply_transforms(data, {"ip": "dlog", "cpi": "yoy", "gdp": "qoq"})
growth.metadata["gdp"].transform, growth.metadata["gdp"].transform_applied
```

With a `MixedFrequencyData`, the metadata records what was applied
(`transform_applied=True`), so `invert_transforms` needs no specification and applying
the metadata transforms twice is a no-op:

```python
rebuilt = nb.invert_transforms(growth, data)
np.allclose(rebuilt.to_frame()["ip"], data.to_frame()["ip"])
```

## In `prepare_panel`

`prepare_panel(data, transform, ...)` applies the transformations and then cleans the
panel ([outliers and missing data](outliers-missing.md)). With a built-in dataset the
legend provides the transformations:

```python
ds = nb.load_brazil_nowcast(columns=["pib", "ibc_br", "ipca"], start="2015-01", end="2019-12")
ds.transform                      # qoq / dlog / ... per series
panel = nb.prepare_panel(ds.data, ds.transform)
panel.metadata["ibc_br"].transform
```
