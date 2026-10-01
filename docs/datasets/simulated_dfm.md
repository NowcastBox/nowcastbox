# `load_simulated_dfm()` — Simulated mixed-frequency dynamic factor model

Monthly indicators driven by AR(1) common factors and a quarterly target linked to its latent monthly counterpart by the Mariano-Murasawa (1, 2, 3, 2, 1)/3 restriction, generated deterministically (seed 0) with known parameters, for tests and tutorials.

## At a glance

| | |
|---|---|
| Loader | `nowcastbox.datasets.load_simulated_dfm` |
| Default target | `gdp` |
| N series | 21 |
| Start | 2000-01 |
| End | 2019-12 |
| Frequencies | M: 20, Q: 1 |

## Usage

```python
import pandas as pd
from nowcastbox.datasets import load_simulated_dfm

ds = load_simulated_dfm(n_monthly=20, n_factors=2)
truth = ds.metadata["true_params"]           # factors, loadings, A, Q, ...
vintage = ds.data.as_of(ds.data.end.to_timestamp(how="end") + pd.Timedelta(days=5))
```

## Sources and license

- nowcastbox.datasets._simulated.simulate_mixed_frequency_dfm (generated on load, no data file)

**License / terms of use.** MIT (generated data, part of nowcastbox)

## Notes

Factors f_t = A f_{t-1} + u_t with A = diag(0.8..0.4) and unit stationary variance; x_it = lambda_i' f_t + e_it with common-variance shares drawn on [0.3, 0.8]; the latent monthly target has share 0.7. Indicators have publication lags of 0-2 months (delays 5/35/65 days) and the last quarter of the target (delay 45 days) is unobserved, so as_of(end of sample + 5 days) reproduces the ragged edge. True parameters are in Dataset.metadata['true_params'].

## Citation

Data-generating process: Giannone, D., Reichlin, L. & Small, D. (2008), Journal of Monetary Economics 55(4); Banbura, M. & Modugno, M. (2014), Journal of Applied Econometrics 29(1); Mariano, R. S. & Murasawa, Y. (2003), Journal of Applied Econometrics 18(4).
