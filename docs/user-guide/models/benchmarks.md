# Benchmarks

A nowcasting model is only useful if it beats simple alternatives. `nowcastbox.benchmarks`
provides the usual ones, all with the benchmark protocol:

<!-- skip-test -->
```python
bench.fit(data, target)          # uses only the information in this vintage
bench.predict(periods)           # pandas Series indexed by target periods
```

| Benchmark | Model | Reference |
|---|---|---|
| `AR(p=1, max_p=4, trend="c")` | autoregression of the target; `p="aic"`/`"bic"` selects the order | Lütkepohl (2005) |
| `RandomWalk(drift=False)` | last observed value (with optional drift) | |
| `HistoricalMean(window=None)` | expanding or rolling mean | Campbell & Thompson (2008) |
| `BridgeBenchmark(predictors=None, aggregation="average", ...)` | bridge equation with AR-extended indicators | Baffigi et al. (2004) |
| `BridgeCombinationBenchmark(predictors=None, max_monthly=2, combine="mean", ...)` | [combination of all small bridge equations](bridge-combination.md) | Bańbura, Belousova, Bodnár & Tóth (2023) |
| `UMIDAS(predictors=None, n_lags=None, ...)` | unrestricted MIDAS by OLS | Foroni, Marcellino & Schumacher (2015) |
| `MIDAS(polynomial="exp_almon" \| "beta", ...)` | MIDAS with exponential Almon or Beta lag polynomial, NLS | Ghysels, Sinko & Valkanov (2007) |
| `SklearnBenchmark(estimator, ...)` | any scikit-learn regressor on aggregated indicators (I11) | |

The MIDAS benchmarks realign the ragged edge (Marcellino & Schumacher, 2010): each
indicator is shifted so that its last observation becomes the most recent lag.

## Example

```python
import pandas as pd
import nowcastbox as nb
from nowcastbox.benchmarks import AR, UMIDAS, BridgeBenchmark, HistoricalMean, RandomWalk

ds = nb.load_simulated_dfm()
periods = pd.period_range("2019Q4", periods=2, freq="Q")

benchmarks = {
    "AR(bic)": AR(p="bic", max_p=4),
    "RW": RandomWalk(),
    "Mean": HistoricalMean(),
    "Bridge": BridgeBenchmark(predictors=["x01", "x04", "x10"]),
    "U-MIDAS": UMIDAS(predictors=["x01", "x04", "x10"], n_lags=3),
}
pd.DataFrame(
    {name: b.fit(ds.data, "gdp").predict(periods) for name, b in benchmarks.items()}
).round(3)
```

```python
ar = AR(p="bic", max_p=4).fit(ds.data, "gdp")
ar.order_, ar.name
```

MIDAS with a parametric lag polynomial is estimated by non-linear least squares and is
slower; restrict the predictors and lags:

```python
from nowcastbox.benchmarks import MIDAS, beta_weights, exp_almon_weights

exp_almon_weights(0.1, -0.05, 6).round(3)        # normalised lag weights
beta_weights(1.0, 3.0, 6).round(3)
midas = MIDAS(predictors=["x10"], polynomial="exp_almon", n_lags=6).fit(ds.data, "gdp")
midas.predict(periods).round(3)
```

## Any scikit-learn regressor (I11)

`SklearnBenchmark` aggregates the indicators to the target frequency (as a bridge
equation, AR-extended at the ragged edge) and fits the regressor. Inside a backtest,
scikit-learn estimators passed in `benchmarks` are wrapped automatically.

<!-- skip-test -->
```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV

from nowcastbox.benchmarks import SklearnBenchmark

ridge = SklearnBenchmark(RidgeCV(alphas=[0.1, 1.0, 10.0]), label="ridge")
forest = SklearnBenchmark(RandomForestRegressor(n_estimators=200, random_state=0))
ridge.fit(ds.data, "gdp").predict(periods)
```

(`scikit-learn` is not a dependency of NowcastBox; install it to use this adapter.)

## In a backtest

Benchmarks are re-estimated at every vintage alongside the main model, and RMSFE ratios
are reported relative to one of them:

```python
bt = nb.PseudoRealTimeBacktest(
    model=nb.TwoStepDFM(n_factors=2),
    data=ds.data,
    target="gdp",
    delay=ds.delay,
    start="2018-01-01",
    end="2019-06-01",
    step="2M",
    benchmarks={"AR": AR(p=1), "RW": RandomWalk()},
)
out = bt.run()
out.relative_to("AR")
```

See [Backtesting](../evaluation/backtesting.md) and
[Forecast comparison tests](../evaluation/forecast-comparison-tests.md).
