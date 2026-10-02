# Accuracy metrics

Point-forecast accuracy is measured on the forecast errors $e_t = y_t - \hat y_t$
(Hyndman & Koehler, 2006):

| Function | Formula |
|---|---|
| `rmsfe(errors)` | $\sqrt{\tfrac1n \sum e_t^2}$ |
| `mse(errors)` | $\tfrac1n \sum e_t^2$ |
| `mae(errors)` | $\tfrac1n \sum \lvert e_t \rvert$ |
| `bias(errors)` | $\tfrac1n \sum e_t$ |
| `relative_rmsfe(errors, reference_errors)` | $\text{RMSFE} / \text{RMSFE}_{\text{ref}}$ |
| `loss_values(errors, loss)` | squared (`"squared"`), absolute (`"absolute"`) or a callable |

NaN errors are ignored.

```python
import numpy as np
import pandas as pd
from nowcastbox.evaluation import (
    accuracy_by_horizon,
    bias,
    forecast_errors,
    mae,
    metric_by_horizon,
    relative_rmsfe,
    rmsfe,
)

actual = np.array([0.5, 0.2, -0.1, 0.8, 0.4])
model = np.array([0.4, 0.3, 0.1, 0.6, 0.5])
naive = np.array([0.3, 0.5, 0.2, -0.1, 0.8])

e_model = forecast_errors(actual, model)
e_naive = forecast_errors(actual, naive)
rmsfe(e_model), mae(e_model), bias(e_model), relative_rmsfe(e_model, e_naive)
```

## By horizon

`metric_by_horizon` and `accuracy_by_horizon` work on any long table with a model
column, an error column and a horizon column — the format of
`BacktestResults.to_frame()`:

```python
rng = np.random.default_rng(0)
frame = pd.DataFrame(
    {
        "model": np.repeat(["dfm", "ar"], 60),
        "months_to_end": np.tile(np.repeat([2, 1, 0], 20), 2),
        "error": np.concatenate(
            [rng.normal(0, s, 20) for s in (1.0, 0.8, 0.6)]      # dfm improves in the quarter
            + [rng.normal(0, 1.0, 20) for _ in range(3)]        # ar does not
        ),
    }
)
metric_by_horizon(frame, "rmsfe")
accuracy_by_horizon(frame, metrics=("rmsfe", "mae", "bias", "n"))
```

From a backtest the same tables come from methods of `BacktestResults`:
`rmsfe_by_horizon()`, `metrics()` and `relative_to(reference)`; see
[Backtesting](backtesting.md).

## Directional accuracy

Policy users often care whether the nowcast gets the **direction** right: does growth
pick up or slow down compared with the previous quarter? The forecast directional
accuracy (FDA; Linzenich & Meunier, 2024) is the share of forecasts whose predicted
change has the sign of the actual change, both measured from a previous value
$y^p_t$:

$$
\text{FDA} = \frac1n \sum_t \mathbb 1\big[(y_t - y^p_t)(\hat y_t - y^p_t) > 0\big].
$$

A coin flip scores about 0.5. Whether a hit rate is better than chance is tested with
Pesaran & Timmermann (1992), which compares it with the rate expected if predicted and
actual directions were independent (one-sided by default):

```python
from nowcastbox.evaluation import DIRECTIONAL_METRICS, directional_accuracy, pesaran_timmermann

previous = np.array([0.3, 0.5, 0.2, -0.1, 0.8])      # last observed value of each target
directional_accuracy(actual, model, previous)         # 1.0: all five directions right

change = rng.normal(0, 1, 120)
noisy = change + rng.normal(0, 0.8, 120)
pt = pesaran_timmermann(change, noisy, np.zeros(120))
pt.hit_rate, pt.expected_hit_rate, pt.statistic, pt.pvalue
```

Directional metrics need the actual, forecast and previous values, not only the errors,
so they live in their own registry (`DIRECTIONAL_METRICS`); `metric_by_horizon` and
`accuracy_by_horizon` accept them by name when the table has `actual`, `forecast` and
`previous_actual` columns:

```python
table = pd.DataFrame(
    {
        "model": np.repeat(["dfm", "ar"], 5),
        "months_to_end": 0,
        "actual": np.tile(actual, 2),
        "forecast": np.concatenate([model, naive]),
        "previous_actual": np.tile(previous, 2),
    }
).assign(error=lambda f: f["actual"] - f["forecast"])
accuracy_by_horizon(table, metrics=("rmsfe", "fda", "n"))
```

In a backtest, `previous_actual` is recorded for every forecast, and
`BacktestResults.directional_accuracy()` gives the FDA with the Pesaran-Timmermann
p-value by horizon; see [Backtesting](backtesting.md#directional-accuracy).

Density forecasts are evaluated with proper scores (CRPS, log score) instead: see
[Scoring densities](../density/scoring.md).
