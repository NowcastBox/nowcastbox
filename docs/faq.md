# FAQ

## General

**Is NowcastBox a port of the R package `nowcasting`?**
No. It is an independent, MIT-licensed implementation written from the academic
literature, with its own API. The R package is used only as an external benchmark of
results. See [Comparison](comparison/r-nowcasting.md).

**Two-step or EM?**
Start with the EM model (`MixedFreqDFM`) when you have several quarterly series,
irregular missing data, blocks, or need robust options; use the two-step model for a
quick, transparent first pass on a large monthly panel. Decide with a backtest. See
[Choosing a method](getting-started/choosing-a-method.md).

**How many factors?**
Use `select_factors` (IC2) as a starting point and check the out-of-sample accuracy of
1–3 factors per block. See [Number of factors](user-guide/selection/number-of-factors.md).

## Data

**Where do quarterly values go on the monthly grid?**
In the third month of the quarter (March, June, September, December); the other months
are structural NaN. Annual values go in December.

**My panel has a value of a quarterly series in February. Why the error?**
`MixedFrequencyData` refuses values outside a series' storage slots
(`NowcastDataError`): the frequency is probably wrong, or the data were stored at the
start of the quarter. Shift them with `nb.quarter_to_month(series, "end")`.

**Should I fill the ragged edge before estimating?**
No. The models handle missing observations; the end of the panel is precisely what
produces the nowcast. `prepare_panel` leaves it untouched by default.

**Can I use weekly or daily data?**
Yes: use a weekly or daily base grid; monthly and quarterly series are linked to it with
calendar-aware aggregation weights (I1). Long daily panels make the state large, so
aggregating daily financial data to weekly or monthly averages is often preferable. See
[Aggregation and frequencies](user-guide/data/aggregation.md).

**Can I use my own data instead of the built-in datasets?**
Yes: any `DataFrame` with a monthly `PeriodIndex` (or a `MixedFrequencyData` built with
`from_series`), plus frequencies and, for vintages, publication delays. The
[data connectors](user-guide/data-sources/index.md) download BCB, IBGE, IPEA and FRED
series.

## Results

**In which units is the nowcast?**
In the units of the target as passed to `fit` (after transformations, before the internal
standardisation): e.g. quarter-on-quarter growth if the target was transformed with
`"qoq"`. Use `invert_transforms` to go back to levels.

**Why does the nowcast change when nothing seems to have been released?**
Either the parameters were re-estimated (the news decomposition reports a re-estimation
effect when you pass `new_results`), or data were revised (`revisions_effect`). Use
`res.news(old, new, period)`.

**How do I get uncertainty bands?**
`res.nowcast` has `std` and 68 %/90 % bands for out-of-sample periods; `res.distribution(n_boot=...)`
adds parameter uncertainty. See [Density nowcasts](user-guide/density/density-nowcasts.md).

**Why does the EM emit a `ConvergenceWarning`?**
It reached `max_iter` before the relative change of the log-likelihood fell below
`tol`. Increase `max_iter`, start from previous results (`init=`), or check the data
([diagnostics](user-guide/diagnostics.md)).

## Performance

**The EM is slow on my machine.**
Limit BLAS to one thread (`OMP_NUM_THREADS=1`) — multithreaded BLAS is often much slower
for the matrices of a Kalman filter. Keep `filter_method="auto"`. See
[Performance](user-guide/performance.md).

**How do I speed up a long backtest?**
Use `refit_every=3` (or more), `n_jobs=-1`, and fewer EM iterations for the refits.

## Citing

See `CITATION.cff` in the repository. Please also cite the papers of the methods you use
(see [References](theory/references.md)).
