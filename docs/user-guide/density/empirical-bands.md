# Empirical error bands

Model-based densities ([Predictive distributions](density-nowcasts.md)) measure the
uncertainty the model *believes* it has. Central banks often report a simpler and more
robust measure: how wrong the same model has been **in the past, at the same point of
the quarter**. This is the approach of the Federal Reserve projection ranges
(Reifschneider & Tulip, 2019), of the Eurosystem/ECB staff projection ranges (ECB, 2009)
and of the ECB nowcasting toolbox (Linzenich & Meunier, 2024, §2.3).

## The method

For a nowcast of period $\tau$ made at vintage $v$, let $h$ be the number of months from
the vintage month to the end of $\tau$ (the `months_to_end` column of a backtest: 2 at
the start of a quarter, 0 in its last month, negative for backcasts). The bands use the
past errors $e = y - \hat y$ of the same model at the **same horizon** $h$ that were
**known at $v$**:

* forecast made strictly before the vintage, $s < v$, inside a rolling window
  ($s \ge v - 10$ years by default);
* outcome already released at $v$: $s + \text{days\_to\_release} \le v$ (when the
  release date is unknown, the target period must have ended before $v$). Use
  `availability="vintage"` to drop this second rule.

With $n_h$ such errors, three scale estimators are available:

| `method` | Distribution | Scale |
|---|---|---|
| `"mae"` (default) | Gaussian, centred at the nowcast | $\hat\sigma_h = \mathrm{MAE}_h \sqrt{\pi/2}$ |
| `"rmse"` | Gaussian, centred at the nowcast | $\hat\sigma_h = \mathrm{RMSE}_h$ |
| `"quantile"` | empirical distribution of $\hat y + e_i$ | empirical quantiles |

For Gaussian errors $\operatorname{E}|e| = \sigma\sqrt{2/\pi}$, so
$\mathrm{MAE}\sqrt{\pi/2}$ estimates $\sigma$, and the ECB band "nowcast ± 1 MAE" is the
central **57.5 %** interval ($2\Phi(\sqrt{2/\pi}) - 1 \approx 0.575$). Bands of any level
follow from the Gaussian quantiles. The default levels are therefore 57.5 % (ECB
convention), 68 % and 90 %.

Past errors can be adjusted for outliers (e.g. the Covid quarters) before computing the
scale: `outliers="winsorize"` clips errors outside
$\operatorname{median} \pm k \cdot 1.4826\,\mathrm{MAD}$ (default $k = 3$,
`outlier_threshold`) and `outliers="exclude"` drops them.

## Bands around a nowcast

```python
import nowcastbox as nb
from nowcastbox.benchmarks import AR
from nowcastbox.density import empirical_bands, empirical_error_scales

ds = nb.load_simulated_dfm()
backtest = nb.PseudoRealTimeBacktest(
    model=nb.TwoStepDFM(n_factors=2),
    data=ds.data,
    target="gdp",
    delay=ds.delay,
    start="2010-01-15",
    end="2019-10-15",
    benchmarks={"AR": AR(p=1)},
).run()

vintage = "2019-11-15"
res = nb.TwoStepDFM(n_factors=2).fit(ds.data.as_of(vintage), target="gdp")

bands = empirical_bands(res, backtest, vintage=vintage)      # ECB: MAE, 10-year window
bands.to_frame()                         # point, std, lower/upper 57.5, 68, 90
bands.interval(0.575)                    # = nowcast ± MAE
bands.info["horizons"], bands.info["n_errors"]

# Same thing from the results object
res.distribution(method="empirical", backtest=backtest, vintage=vintage)

# Error statistics by horizon known at the vintage
empirical_error_scales(backtest, vintage)    # n, bias, mae, rmse, sigma_mae
```

`empirical_bands` returns an `EmpiricalGaussianDistribution` for `"mae"` and `"rmse"`: a
Gaussian [`NowcastDistribution`](density-nowcasts.md#working-with-the-distribution) whose
`to_frame()`, `fan_chart_frame()` and `plot()` default to the band levels (57.5 %, 68 %,
90 %, or the `levels` you passed) instead of the generic 50/68/90 %. Everything that works with model densities works here too: `interval`,
`quantiles`, `to_frame`, `plot()`, `nowcastbox.evaluation.scoring.crps`, PIT and coverage
tests. With `method="quantile"` it returns an `EmpiricalQuantileDistribution` with the
same `interval` / `quantiles` / `to_frame` / `fan_chart_frame` / `plot` / `sample` API and
an exact ensemble `crps(observed)`.

Options:

* `vintage` — date of the nowcast's information set. Default: `results.info["vintage"]`
  if present, otherwise the first day after the last month with data (pass it
  explicitly whenever possible: the horizon depends on it);
* `model` — backtest model whose errors are used (default: the main model);
* `window` — `"10Y"` (default), `"36M"`, `"520W"`, a `pandas.DateOffset`, or `None` for an
  expanding window;
* `min_errors` (default 8) — a period whose horizon has fewer past errors gets no band:
  it is left out of the distribution with a `DataQualityWarning` and listed in
  `info["skipped"]`; if no period qualifies a `NowcastDataError` is raised.

## Checking the coverage in real time

`backtest_empirical_bands` gives every forecast of a backtest the band it would have had
in real time (using only the errors known at its own vintage), so the coverage of the
bands can be evaluated honestly:

```python
from nowcastbox.density import backtest_empirical_bands
from nowcastbox.evaluation.scoring import christoffersen_test, interval_hits

table = backtest_empirical_bands(backtest, levels=[0.575, 0.9]).dropna()
hits = interval_hits(table["actual"], table["lower_90"], table["upper_90"])
hits.mean()                              # empirical coverage of the 90 % band
christoffersen_test(hits, 0.9)           # unconditional coverage / independence
```

!!! warning "Revised outcomes"
    The errors are those of the backtest table, `actual - forecast`. With a
    `VintageStore` and the default `actual="final"`, the outcomes are the latest
    estimates, which were not known at past vintages even after the first release.
    For a strict real-time information set run the backtest with `actual="first"`
    (first releases).

!!! note "Several vintages per month"
    The horizon is counted in months. With weekly or daily vintages every forecast made
    in the same month enters the error set at that horizon, so `n_errors` (and
    `min_errors`) counts forecasts, not distinct target periods.

!!! note "Small samples"
    With a 10-year window there are about 40 errors per horizon for a quarterly target.
    The Gaussian bands (`"mae"`, `"rmse"`) are close to nominal under normal errors;
    empirical quantiles of $n$ errors under-cover by $O(1/n)$ (a 90 % band from 40
    errors covers about 83 %), so prefer `"quantile"` with long backtests.

## References

* Reifschneider, D. & Tulip, P. (2019). Gauging the uncertainty of the economic outlook
  using historical forecasting errors: the Federal Reserve's approach. *International
  Journal of Forecasting*, 35(4), 1564-1582.
* European Central Bank (2009). *New procedure for constructing Eurosystem and ECB staff
  projection ranges*.
* Linzenich, J. & Meunier, B. (2024). Nowcasting made easier: a toolbox for economists.
  ECB Working Paper No. 3004.

## See also

- [Backtesting](../evaluation/backtesting.md): the errors come from a
  `PseudoRealTimeBacktest` (use `actual="first"` with real vintages).
- [Predictive distributions](density-nowcasts.md) for model-based densities and
  [Scoring densities](scoring.md) for their evaluation.
- [HTML reports](../visualization/reports.md) (`bands=`) and the pipeline output
  `empirical_bands` ([Pipeline and CLI](../pipeline-cli.md)).
