# Experiments

`NowcastExperiment` fits several specifications on the same data and compares them
side by side (the `panelbox.experiment` pattern): nowcasts, log-likelihoods, in-sample
fit, timings, and optionally a pseudo real-time backtest of each.

```python
import nowcastbox as nb

ds = nb.load_simulated_dfm()
exp = nb.NowcastExperiment(ds.data, "gdp")
exp.add_model("2s, r=1", nb.TwoStepDFM(n_factors=1))
exp.add_model("2s, r=2", nb.TwoStepDFM(n_factors=2))
exp.add_model("em, r=2", nb.MixedFreqDFM(n_factors=2))
exp.add_model("em, iid", nb.MixedFreqDFM(n_factors=2, idiosyncratic="iid"))
exp.add_model("bridge", nb.BridgeEquation())
exp.fit_all()

print(exp.summary())
exp.compare()                      # one row per model
exp.nowcast_table().tail(3)        # nowcasts side by side with the observed target
```

Results are kept: `exp.get_results("em, r=2")`, `exp.results`, `exp.list_models()`.
`fit_all(errors="warn")` records failures instead of raising.

## Backtesting every model

`run_backtest()` runs a `PseudoRealTimeBacktest` for each model (keyword arguments go to
the backtest) and `rmsfe_table()` collects the RMSFE by horizon:

```python
exp.run_backtest(
    models=["2s, r=2", "em, r=2", "bridge"],
    delay=ds.delay,
    start="2018-07-15",
    end="2019-12-15",
    refit_every=3,
)
exp.rmsfe_table()
```

## Plots and reports

```python
fig = exp.plot("nowcasts")         # nowcast paths of every model
fig = exp.plot("rmsfe")            # RMSFE by horizon (after run_backtest)
exp.report("em, r=2", "em_report.html")
```
