# NowcastBox cheat sheet

`import nowcastbox as nb` · MIT · docs: `mkdocs serve` · examples: `examples/notebooks`

## Data

| Task | Code |
|---|---|
| Built-in datasets | `nb.list_datasets()` · `nb.load_brazil_nowcast()` · `nb.load_nyfed()` · `nb.load_us_grs_like()` · `nb.load_us_fred_md()` · `nb.load_simulated_dfm()` |
| Dataset parts | `ds.data` (levels, `MixedFrequencyData`) · `ds.legend` · `ds.transform` · `ds.delay` · `ds.blocks` · `ds.calendar()` · `ds.select(cols)` · `ds.truncate(a, b)` |
| Own data | `nb.MixedFrequencyData(df, frequencies={"ip": "M", "gdp": "Q"}, release_delays={...}, blocks=..., categories=...)` (quarterly values in the 3rd month) |
| From native series | `nb.MixedFrequencyData.from_series({"gdp": q_series, "ip": m_series}, base_frequency="M")` (`"W"` for weekly grids, I1) |
| Transform | `nb.apply_transforms(data, {"ip": "dlog", "gdp": "qoq"})` · codes 0–7 · `nb.invert_transforms(t, levels)` |
| Clean | `nb.prepare_panel(data, ds.transform, return_report=True)` (outliers > 4 IQR, interior NAs, drop > 1/3 missing) |
| Ragged edge | `data.last_observed()` · `data.ragged_edge_mask()` · `nb.visualization.plot_data_availability(data, backend="matplotlib")` |
| Vintages | `nb.pseudo_real_time(data, calendar=nb.load_brazil_calendar(), vintage="2026-08-15")` · `data.as_of(date)` · `nb.load_brazil_vintages()` (real IBGE vintages) · `VintageStore.as_of(date, as_mixed=True)` |
| Download | `nb.data_sources.fetch_sgs({"ibc": 24364})` · `fetch_sidra(1621, 584, {"c11255": 90707})` · `fetch_ipeadata({...})` · `fetch_fred(...)` (cached) |

## Models

| Task | Code |
|---|---|
| One call | `nb.nowcast(data, "pib", method="em", preprocess=True, density=True)` |
| How many factors / shocks | `nb.select_factors(x, rmax=10, criterion="IC2")` · `nb.select_shocks(x, n_factors=r, factor_lags=2)` |
| Two-step (GRS 2008) | `nb.TwoStepDFM(n_factors=2, factor_lags=2, n_shocks=2, aggregate="factors").fit(panel, "gdp")` |
| EM (Bańbura-Modugno) | `nb.MixedFreqDFM(n_factors=1, blocks="data", max_iter=500, tol=1e-4).fit(panel, "gdp")` |
| Formula targets | `fit(panel, "gdp ~ ip + pmi")` · `"gdp ~ . - x3"` |
| Robust (I3) | `idiosyncratic="student_t"` · `outliers="auto"` · `covid="dummy"` / `"mask"` · `exclude_periods=[("2020-03", "2021-06")]` |
| Long-run mean (I4) | `long_run_mean="time_varying"` → `res.long_run_mean` |

## Results

| Task | Code |
|---|---|
| Nowcast | `res.summary()` · `res.nowcast` (`observed`, `in_sample`, `out_of_sample`, bands) · `res.get_nowcast("2026Q3")` |
| Model | `res.factors` · `res.loadings` · `res.transition` · `res.idiosyncratic_variance` · `res.bridge.summary()` (two-step) |
| Plots | `res.plot(kind, backend="matplotlib")`, kind ∈ `forecast`, `fan`, `density`, `factors`, `loadings`, `eigenvalues`, `loglikelihood`, `data_availability` |
| Save | `res.save("res.pkl")` · `nb.NowcastResults.load("res.pkl")` |

## Explain, evaluate, report

| Task | Code |
|---|---|
| News (I6) | `news = res.news(old, new, target_period="2026Q3")` · `news.summary()` · `news.top_releases(10)` · `news.plot("waterfall", by="category")` |
| Tracker | `res.nowcast_tracker(panel, calendar, "2026Q3", "2026-07-01", "2026-10-01").plot("path")` |
| Level contributions | `res.level_contributions(panel, "2026Q3").plot("bar")` |
| Density (I5) | `res.distribution(n_boot=50, method="block")` · `.interval(0.9)` · `.variance_decomposition` |
| Scores | `nb.scoring.crps_gaussian(y, mu, sd)` · `log_score_gaussian` · `pit` · `interval_coverage` · `christoffersen_test` · `berkowitz_test` |
| Backtest | `nb.PseudoRealTimeBacktest(model, data, "pib", calendar=cal, start=..., end=..., step="M", benchmarks=[nb.benchmarks.AR(p=1), nb.benchmarks.MIDAS()], n_jobs=4).run()` |
| Backtest results | `out.rmsfe_by_horizon()` · `out.relative_to("AR")` · `out.diebold_mariano(reference="AR")` · `out.mcs(horizon="kind")` · `out.to_parquet(path)` |
| Benchmarks | `AR`, `RandomWalk`, `HistoricalMean`, `BridgeBenchmark`, `UMIDAS`, `MIDAS`, `SklearnBenchmark(RidgeCV())` (I11) |
| Diagnostics (I9) | `res.diagnostics().summary()` · `.series_overview()` |
| Compare specs | `exp = nb.NowcastExperiment(panel, "pib", models={...}); exp.fit_all(); exp.compare(); exp.run_backtest(...)` |
| HTML report | `nb.NowcastReport(res, news=news, tracker=tr, quantiles=dist, diagnostics=True).to_html("report.html")` |

## Production (I10)

```bash
nowcastbox init --list                  # bundled spec templates
nowcastbox validate nowcast.yaml        # check a spec
nowcastbox run nowcast.yaml [--vintage 2026-09-15]   # estimate, news, density, report, snapshot
nowcastbox snapshots list|history|diff snapshots/
```

```yaml
target: pib
data: {source: brazil_nowcast, vintage: today}
model: {type: MixedFreqDFM, factors: {global: 1, real: 1, soft: 1},
        idiosyncratic: student_t, long_run_mean: time_varying}
outputs: [nowcast, news, density, report_html]
snapshot_dir: ./snapshots
```

## Conventions

Monthly base grid (`PeriodIndex`); lower-frequency values in the last month of their
period · results in the target's original (transformed) units · explicit errors
(`NowcastDataError`, `FormulaError`, `ModelNotFittedError`) · warnings
`ConvergenceWarning`, `DataQualityWarning` · every random method takes `random_state`.
