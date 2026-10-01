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

Density forecasts are evaluated with proper scores (CRPS, log score) instead: see
[Scoring densities](../density/scoring.md).
