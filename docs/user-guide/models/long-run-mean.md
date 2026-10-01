# Time-varying long-run mean (innovation I4)

Standardising the panel assumes that the mean of GDP growth is constant. When trend
growth changes — Brazil grew about 4 % a year in 2004–2013 and about 1 % after 2014 — a
constant mean biases the nowcast towards the old average. Following Antolin-Diaz,
Drechsel & Petrella (2017), `long_run_mean="time_varying"` adds a **random-walk long-run
mean** $\mu_t$ to the target:

$$
y_t = \mu_t + \lambda' f_t + \varepsilon_t, \qquad \mu_t = \mu_{t-1} + \eta_t,
\quad \eta_t \sim N(0, \sigma^2_\eta),
$$

one extra state with a unit loading on the target (and, optionally, estimated loadings on
further series, `long_run_series`). The variance $\sigma^2_\eta$ is estimated by EM, or
fixed with `long_run_variance` (a signal-to-noise ratio relative to the target variance in
standardised units), which avoids the well-known pile-up of its ML estimate at zero
(Stock & Watson, 1998).

## Example

```python
import nowcastbox as nb

ds = nb.load_brazil_nowcast(
    columns=["pib", "ibc_br", "pim_geral", "pmc_varejo", "pms_volume", "ipca", "selic"],
    start="2012-01",
    end="2024-12",
)
panel = nb.prepare_panel(ds.data, ds.transform, keep="pib")
vintage = nb.pseudo_real_time(panel, delay=ds.delay, vintage="2024-11-15")

constant = nb.MixedFreqDFM(n_factors=1).fit(vintage, "pib")
varying = nb.MixedFreqDFM(n_factors=1, long_run_mean="time_varying").fit(vintage, "pib")
fixed = nb.MixedFreqDFM(
    n_factors=1, long_run_mean="time_varying", long_run_variance=0.01
).fit(vintage, "pib")

varying.long_run_mean.tail(3)                     # smoothed mu_t (and its std)
fixed.nowcast["long_run_mean"].dropna().iloc[[0, -1]]   # first and last quarter
constant.get_nowcast("2024Q4"), varying.get_nowcast("2024Q4"), fixed.get_nowcast("2024Q4")
```

`results.long_run_mean` holds the smoothed path with its standard deviation on the
monthly grid, and the `nowcast` frame gets a `long_run_mean` column on the target's
quarterly grid, so the nowcast can be read as *trend + cycle*.

Through `nb.nowcast`:

```python
res = nb.nowcast(vintage, "pib", method="em", n_factors=1, long_run_mean="time_varying")
"long_run_mean" in res.nowcast.columns
```

## Practical advice

- With short samples the estimated $\sigma^2_\eta$ often collapses to (almost) zero and
  the model reverts to a constant mean; fix `long_run_variance` (e.g. `0.005`–`0.05`) when
  you believe trend growth moved, and compare out of sample.
- Add `long_run_series=["ibc_br"]` to let a monthly activity index share the trend.
- The long-run mean is a state: it enters the news decomposition, the density and the
  diagnostics like any other state.
