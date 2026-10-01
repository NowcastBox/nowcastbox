# Outliers and missing data

Large macro panels contain outliers (strikes, policy changes, the pandemic) and gaps
(discontinued series, late starts). NowcastBox separates three kinds of missing cells:

| Kind | Example | Default treatment |
|---|---|---|
| **Leading** | a series that starts in 2011 | kept missing (models handle it; `prepare_panel` drops the leading periods where *no* series is observed) |
| **Interior** | a month skipped by the statistical office | filled by a cubic spline (`replace_na=True`) |
| **Ragged edge** | the last months not yet released | **kept missing** — the models need it to nowcast |

## Outliers: the IQR rule

An observation is an outlier when it is far from the median of the series relative to
the interquartile range,

$$
|x_t - \operatorname{median}(x)| > c \cdot \operatorname{IQR}(x), \qquad c = 4 \text{ by default},
$$

and it is replaced by a centred moving median of the neighbouring observations (Stock &
Watson, 2002). The rule is applied to the **transformed** (stationary) series.

```python
import numpy as np
import pandas as pd
import nowcastbox as nb
from nowcastbox.preprocessing import (
    detect_outliers,
    fill_missing,
    interior_missing_mask,
    missing_proportion,
    replace_outliers,
)

rng = np.random.default_rng(0)
idx = pd.period_range("2020-01", periods=36, freq="M")
a = pd.Series(rng.normal(0, 1, 36), index=idx)
a.iloc[10] = 15.0                  # an outlier
a.iloc[[5, 6, 20]] = np.nan        # interior gaps
a.iloc[-2:] = np.nan               # ragged edge
panel = pd.DataFrame({"a": a, "b": rng.normal(0, 1, 36)}, index=idx)

detect_outliers(panel, frequency="M").sum()
cleaned, flags = replace_outliers(panel, threshold=4.0, window=3, frequency="M",
                                  return_mask=True)
cleaned.loc["2020-11", "a"], int(flags["a"].sum())
```

`replacement="median"` uses the full-sample median and `replacement="nan"` turns
outliers into missing values (let the Kalman filter skip them).

## Interior gaps

```python
interior_missing_mask(panel, frequency="M").sum()     # 3 interior gaps in "a"
missing_proportion(panel, frequency="M")

filled = fill_missing(panel, method="spline", frequency="M")
filled["a"].iloc[[5, 6, 20, -2, -1]]                    # ragged edge still NaN
```

Methods: `"spline"` (cubic spline, as in the initialisation of Bańbura & Modugno, 2014),
`"linear"` and `"moving_median"` (`window`). Filling the ragged edge is possible
(`fill_ragged_edge=True` with `edge_method="median" | "last" | "mean"`) but defeats the
purpose of a nowcasting model; use it only for methods that need a balanced panel.

## Everything at once: `prepare_panel`

`prepare_panel` chains transformation → outlier replacement → interior filling →
dropping of sparse series, and can report what it did:

```python
ds = nb.load_brazil_nowcast(start="2012-01", end="2024-12")
prepared, report = nb.prepare_panel(
    ds.data,
    ds.transform,
    outlier_threshold=4.0,
    max_na_prop=1 / 3,          # drop series with > 1/3 of their slots missing
    keep="pib",                 # never drop the target
    return_report=True,
)
report.dropped                  # e.g. series that start too late
report.n_outliers.sort_values(ascending=False).head()
report.n_filled.sum()
```

| Option | Default | Effect |
|---|---|---|
| `replace_outliers` | `True` | IQR rule on the transformed series |
| `outlier_threshold`, `outlier_window` | `4.0`, `3` | $c$ and the moving-median window |
| `replace_na`, `na_method` | `True`, `"spline"` | interior filling |
| `fill_ragged_edge` | `False` | keep the ragged edge |
| `max_na_prop` | `1/3` | drop series with too many missing slots (`None` keeps all) |
| `keep` | `None` | series never dropped |
| `aggregate` | `False` | pre-filter monthly series with the aggregation weights ("2s" variant) |

!!! note "Outliers inside the model"
    For EM estimation, outliers can also be handled *inside* the model
    (`MixedFreqDFM(outliers="auto")`, Student-t errors, pandemic dummies) instead of
    being replaced beforehand. See [Robust estimation](../models/robust.md).
