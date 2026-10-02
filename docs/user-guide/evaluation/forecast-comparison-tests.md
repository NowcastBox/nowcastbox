# Forecast comparison tests

Is the lower RMSFE of a factor model significant, or luck? Four tests are available,
both as functions of error series and as methods of `BacktestResults` (by horizon). See
[Forecast evaluation](../../theory/forecast-evaluation.md) for the statistics.

| Test | Null hypothesis | Function / method |
|---|---|---|
| Diebold & Mariano (1995) with the Harvey, Leybourne & Newbold (1997) correction | equal expected loss, $E[d_t] = 0$ | `diebold_mariano` |
| Clark & West (2007), for *nested* models (e.g. an AR inside a DFM or a bridge) | equal MSPE once the larger model's estimation noise is removed | `clark_west` |
| Giacomini & White (2006) | equal *conditional* expected loss, $E[d_{t+h} \mid \mathcal F_t] = 0$ | `giacomini_white` |
| Model Confidence Set, Hansen, Lunde & Nason (2011) | equal expected loss within the set | `model_confidence_set` / `BacktestResults.mcs` |

$d_t = L(e_{1t}) - L(e_{2t})$ is the loss differential (squared or absolute loss).

## Pairwise tests on error series

```python
import numpy as np
import pandas as pd
from nowcastbox.evaluation import diebold_mariano, giacomini_white, model_confidence_set

rng = np.random.default_rng(0)
n = 80
e_dfm = rng.normal(0, 0.8, n)
e_ar = rng.normal(0, 1.0, n)
e_rw = rng.normal(0, 1.3, n)

dm = diebold_mariano(e_dfm, e_ar, h=1, loss="squared", hln=True)
dm.statistic, dm.pvalue, dm.mean_loss_differential      # negative: the first is better

diebold_mariano(e_dfm, e_ar, alternative="less").pvalue  # one-sided: DFM more accurate

gw = giacomini_white(e_dfm, e_ar, h=1)                    # instruments (1, d_{t-h})
gw.statistic, gw.pvalue
```

When the benchmark is nested in the competing model, the DM test is undersized. The
Clark-West test adjusts the larger model's squared errors,
$d_t = e_{1t}^2 - (e_{1t} - e_{2t})^2 - e_{2t}^2$, and is usually one-sided:

```python
from nowcastbox.evaluation import clark_west

cw = clark_west(e_dfm, e_ar)          # first argument: the larger model
cw.statistic, cw.pvalue               # alternative="less": the larger model is better
```

For $h$-step forecasts the long-run variance of $d_t$ uses $h-1$ autocovariances (DM)
or a Newey-West HAC estimator (GW).

## Model Confidence Set

The MCS keeps the set of models that cannot be distinguished from the best at level
$1-\alpha$, eliminating the worst model while the equivalence hypothesis is rejected
(block bootstrap of the loss differentials):

```python
losses = pd.DataFrame({"dfm": e_dfm**2, "ar": e_ar**2, "rw": e_rw**2})
mcs = model_confidence_set(losses, alpha=0.1, statistic="max", n_bootstrap=500)
mcs.included, mcs.pvalues
```

## From a backtest

`BacktestResults` runs the tests by horizon on the common evaluation sample, against a
reference model:

```python
import nowcastbox as nb
from nowcastbox.benchmarks import AR, RandomWalk

ds = nb.load_simulated_dfm()
out = nb.PseudoRealTimeBacktest(
    model=nb.TwoStepDFM(n_factors=2),
    data=ds.data,
    target="gdp",
    delay=ds.delay,
    start="2016-01-15",
    end="2019-12-15",
    benchmarks={"AR": AR(p=1), "RW": RandomWalk()},
).run()

out.diebold_mariano(reference="AR")              # by months_to_end
out.diebold_mariano(reference="AR", horizon="kind", aggregate="target_period", h=2)
out.clark_west(reference="AR")                   # AR nested in the competitors
out.giacomini_white(reference="AR")
out.mcs(alpha=0.1, n_bootstrap=500, aggregate="target_period")
```

!!! warning "Several vintages per target period"
    Grouping by `"kind"` (or `horizon=None`) stacks the monthly nowcasts of the same
    quarter. Their loss differentials are strongly correlated, so a long-run variance
    truncated at lag 0 (`h=1`) overstates significance. Pass
    `aggregate="target_period"` to average the losses within each target period (one
    observation per quarter) and set `h` to the number of quarters ahead of the longest
    forecast in the group (e.g. `h=2` for one-quarter-ahead forecasts). Testing each
    `months_to_end` separately avoids the stacking but keeps the overlap of multi-step
    forecasts: use `h` > 1 there too.

!!! warning "Small samples"
    Nowcast evaluations rarely have more than 40–60 quarters per horizon. The HLN
    correction and Student-t critical values help, but power is low: report relative
    RMSFEs together with p-values, and when pooling horizons average the losses per
    target period (`aggregate="target_period"`) rather than stacking them.
