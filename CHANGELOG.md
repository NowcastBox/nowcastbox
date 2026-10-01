# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed (post-audit)

- Look-ahead bias in the pipeline backtest: `outputs.backtest` used the panel already cleaned on
  the full sample (IQR outlier flags, centred moving-median replacement, spline gap filling).
  `PseudoRealTimeBacktest` gained `preprocess=` (a callable applied to every vintage) and the
  pipeline now backtests the uncleaned vintage data, re-running the spec's preprocessing per
  vintage.
- NYFED dataset metadata now points to the shipped BSD-3-Clause license file.
- `python -m nowcastbox` runs the CLI.

### Fixed (final software-quality audit)

- `joblib` is now a declared dependency: on a clean install `n_jobs != 1` raised
  `ImportError` in the density bootstrap and silently ran backtests sequentially.
- Packaging: PEP 639 license metadata (`license = "MIT"`, `license-files`; setuptools
  >= 77) instead of the deprecated TOML table and license classifier; the BSD-3-Clause
  notice of the bundled NY Fed data is shipped in the wheel
  (`nowcastbox/datasets/licenses/`).
- `import nowcastbox` no longer imports `statsmodels` (about 0.3-0.7 s faster); it is
  loaded when a bridge regression is estimated.
- A non-string `target` (e.g. `None`) raises `FormulaError` with a clear message instead
  of `TypeError: argument of type 'NoneType' is not iterable`.
- `generate_vintages(..., delay=...)` accepts the same keyword as `pseudo_real_time`.
- `NowcastSpec.from_yaml` given YAML text raises `FileNotFoundError` pointing to
  `from_string` instead of a raw `OSError`.
- Consistent error message for positive EM parameters (`df`, `outlier_threshold`,
  `long_run_variance`).
- Docs: the statsmodels comparison and the pipeline "equivalent code" examples now run
  (and are executed by the docs tests).

## [0.1.0] - release candidate (draft release notes, not published)

First public release: the complete DFM nowcasting workflow of the plan
(`desenvolvimento/PROJETO_DESENVOLVIMENTO.md`) — data, vintages, selection, estimation, news, density,
evaluation, diagnostics, reports and a production pipeline — with the innovations
I1–I11 (I8 partial: real vintages of Brazilian GDP and ALFRED only).

### Highlights

- **Models**: `TwoStepDFM` (Giannone, Reichlin & Small 2008; Doz, Giannone & Reichlin
  2011) and `MixedFreqDFM` (Bańbura & Modugno 2014) with blocks, Student-t errors,
  in-EM outlier detection, pandemic options and a time-varying long-run mean.
- **Any frequency (I1)**: weekly and daily base grids with calendar-aware aggregation
  in the EM model and in every analysis tool built on it.
- **Speed (I2)**: exact structured smoother, ~12x faster per EM iteration than
  `DynamicFactorMQ` at N = 200; ~35x faster than R `nowcasting` on the NY Fed EM.
- **Production (I10)**: YAML specification, `nowcastbox` command line and versioned
  snapshots.
- **Validation**: 90+ reference tests against R `nowcasting` 1.1.2 (black box) and
  statsmodels; 15 executed example notebooks; a MkDocs site with runnable examples.

### Behaviour changes in this release cycle (wave-3 integration)

- `MixedFreqDFM(init="pca")` (default) is a **multi-start** with several blocks: the
  block principal components are computed in four orders (`BLOCK_ORDERS`: given,
  reversed, smaller blocks first, independent) and the EM starts from the one with the
  highest log-likelihood (`results.info["initialization"]`). Estimates of multi-block
  models change (on the NY Fed specification the final log-likelihood rises by 372
  points); `init="pca_given"` restores the sequential start of Bańbura & Modugno.
- `nb.nowcast(..., preprocess=True)` no longer cleans the **target**: outlier
  replacement and gap filling apply to the predictors only (the 2020 GDP quarters were
  winsorised before); `preprocess={"clean_target": True}` restores the old behaviour.
- `MixedFreqDFM` and `TwoStepDFM` no longer report `out_of_sample` estimates for target
  periods **before the first observation** of the target (the EM keeps them in the
  `common` column).
- `select_shocks` follows Bai & Ng (2007) more closely: factors normalised with
  Λ'Λ/N = I, `factor_lags=2` and `m=None` (1 for the covariance matrix, 1.25/2.25 for
  the correlation matrix) by default, so `q_star` may change (usually lower).
  `select_factors` warns when min(N, T) < 20.
- Release delays may be **negative** (down to -365 days): values published before the
  end of their reference period (e.g. regional business surveys); NYFED vintages now
  match R's `PRTDB` for every series.
- `filter_method="auto"` uses a recalibrated cost model (per-call and per-group
  overheads of the structured smoother): small panels (N up to about 35 at T = 300) use
  the dense smoother, which is faster there; estimates are unchanged.
- Bundled pipeline templates resolve relative paths (`snapshot_dir`, reports) against
  the working directory, never inside the installed package.

### Added

- Project scaffold: `pyproject.toml` (MIT), CI workflows, pre-commit, MkDocs Material
  configuration, contribution guides and citation metadata.
- `nowcastbox.core`: `MixedFrequencyData` (mixed-frequency panel on a base period
  grid with per-series metadata, ragged-edge masks, pseudo real-time `as_of`,
  standardisation), `Frequency` and `AggregationType` (incl. Mariano-Murasawa
  weights), `BaseNowcaster` / `BaseBenchmark`, `NowcastResults` / `FactorResults`,
  formula parsing (`"y ~ ."`), custom exceptions and warnings.
- Package skeleton for all planned subpackages and `nowcastbox._logging`.
- `nowcastbox.preprocessing`: named and invertible transformations (codes 0-7 and
  names such as `"dlog"`, `"pct_change"`; `Transform`, `Diff`, `Log`, `PctChange`,
  `Scale`, `Compose`, `TemporalAggregation`), IQR outlier detection/replacement,
  interior-gap filling (spline/linear/moving median) that preserves the ragged edge,
  temporal aggregation (`month_to_quarter`, `quarter_to_month`, Mariano-Murasawa
  weights, `loading_constraints`) and the one-call `prepare_panel` (with `PanelReport`).
- `nowcastbox.statespace`: `StateSpace` representation, Kalman filter with missing
  data (univariate treatment, Koopman & Durbin 2000, Numba kernels; multivariate
  fallback), fixed-interval smoother, log-likelihood, observation collapsing
  (Jungbacker & Koopman 2015), simulation helpers and stationary initialisation.
- `nowcastbox.selection`: Bai & Ng (2002) IC/PC criteria (`select_factors`), Bai & Ng
  (2007) primitive shocks (`select_shocks`), targeted predictors (hard/soft
  thresholding, elastic net; Bai & Ng 2008).
- `nowcastbox.vintages`: `pseudo_real_time`, `generate_vintages`, `ReleaseCalendar`
  (delays and explicit release dates) and `VintageStore` (real-time vintages with
  revisions, CSV/Parquet I/O).
- `nowcastbox.data_sources`: BCB/SGS, IBGE/SIDRA, IPEADATA and FRED/ALFRED
  connectors with retrying HTTP and an on-disk cache.
- `nowcastbox.models`: `TwoStepDFM` (Giannone, Reichlin & Small 2008; Doz, Giannone
  & Reichlin 2011) with `aggregate="factors"|"variables"` and `BridgeEquation`;
  `MixedFreqDFM` (Bańbura & Modugno 2014 EM with blocks, AR(1) idiosyncratic
  components and Mariano-Murasawa restrictions).
- Top-level API (wave-1 integration): `nowcastbox` re-exports the main classes and
  functions (`nb.prepare_panel`, `nb.select_factors`, `nb.select_shocks`,
  `nb.pseudo_real_time`, `nb.ReleaseCalendar`, `nb.VintageStore`, `nb.TwoStepDFM`,
  `nb.MixedFreqDFM`, `nb.MixedFrequencyData`, transforms, Kalman functions) and the
  subpackages as attributes; `nb.utils.month_to_quarter` / `quarter_to_month`.
- `nowcastbox.nowcast(data, target, method="em"|"two_step", n_factors=None, ...)`
  (`nowcastbox/api.py`): optional preprocessing, Bai-Ng choice of the number of
  factors (floored at one, with a warning when `min(N, T) < 20`), optional
  `n_shocks="auto"` (Bai-Ng 2007) and estimation; the selections are stored in
  `results.info`.
- `SeriesMetadata.transform_applied` (core contract): distinguishes "transformation
  to apply" from "already applied"; set by `apply_transforms`/`prepare_panel`, reset
  by `invert_transforms`.
- End-to-end integration tests (`tests/integration/test_end_to_end.py`): levels ->
  `prepare_panel` -> `select_factors` -> `TwoStepDFM`/`MixedFreqDFM` -> nowcast, and
  pseudo real-time vintages -> refit -> nowcast revisions.
- Root `tests/conftest.py`: `--run-network` option and network skip hook (moved from
  `tests/data_sources`), BLAS limited to one thread per test session
  (`threadpoolctl`; `NOWCASTBOX_TEST_BLAS_THREADS` overrides).
- Wave 2 — `nowcastbox.news` (I6): Bańbura-Modugno news decomposition
  (`news_decomposition` -> `NewsResults`, split into data revisions, news by
  series/block/category/release and re-estimation; exact identity), `nowcast_tracker`
  (`NowcastTracker`), `level_contributions`, `news_weights`, `data_revisions` /
  `compare_vintages`, batched linear smoother.
- Wave 2 — `nowcastbox.density` (I5): `NowcastDistribution` (Gaussian or Gaussian
  mixture: quantiles, intervals, pdf/cdf, sampling, fan chart), parametric and
  moving-block bootstrap with re-estimation (`bootstrap_nowcasts`),
  `nowcast_distribution`; `nowcastbox.evaluation.scoring` (CRPS, log score, PIT,
  Berkowitz/KS uniformity, interval coverage and Christoffersen tests, quantile scores).
- Wave 2 — `nowcastbox.benchmarks` (AR(p), random walk, historical mean, bridge, U-MIDAS,
  MIDAS with exponential Almon/Beta weights, `SklearnBenchmark` for any scikit-learn
  regressor, I11) and `nowcastbox.evaluation` (`PseudoRealTimeBacktest` with pseudo or
  real-time vintages -> `BacktestResults`; RMSFE/MAE/bias by horizon; Diebold-Mariano
  with HLN correction, Giacomini-White, Model Confidence Set).
- Wave 2 — `MixedFreqDFM` robustness (I3: `idiosyncratic="student_t"`, `outliers="auto"`,
  `covid="mask"|"dummy"`, `exclude_periods`) and time-varying long-run mean (I4:
  `long_run_mean="time_varying"`); new results fields (`observation_weights`,
  `student_t_df`, `outlier_flags`, `excluded_observations`, `interventions`,
  `state_offset`, `long_run_mean`).
- Wave 2 — `nowcastbox.diagnostics` (I9): Breitung-Eickmeier loading-stability tests,
  EM convergence, factor contributions/commonality, data-quality report, Ljung-Box and
  Jarque-Bera; `run_diagnostics` -> `DiagnosticsReport`.
- Wave 2 — `nowcastbox.statespace` performance (I2): exact structured smoother
  (`structured_smoother`, `smoothed_moments`, `detect_structure`), public time-varying
  observation equations (`StateSpace(obs_index=...)`), approximate diffuse
  initialisation, `SmootherResult.smoothed_signal_variance()`; allocation-free dense
  kernels with sparse transitions; `benchmarks/bench_em.py`.
- Wave 2 — `nowcastbox.datasets`: `Dataset` and loaders `load_brazil_nowcast`,
  `load_brazil_calendar`, `load_brazil_gdp_releases`, `load_brazil_vintages`,
  `load_us_fred_md`, `load_nyfed`, `load_us_grs_like`, `load_simulated_dfm`
  (checksummed CSV.gz + YAML metadata; build scripts in `scripts/build_datasets/`).
- Wave 2 — `nowcastbox.visualization` (Plotly/Matplotlib, themes; forecast, fan,
  factors, eigenvalues, loadings, data availability, news waterfall, tracker, RMSFE,
  log-likelihood), `nowcastbox.reports.NowcastReport` (HTML) and
  `nowcastbox.experiment.NowcastExperiment`.
- Wave-2 integration: top-level exports (`nb.news`, `nb.density`, `nb.evaluation`,
  `nb.benchmarks`, `nb.diagnostics`, `nb.visualization`, `nb.reports`,
  `nb.experiment`, `nb.news_decomposition`, `nb.nowcast_tracker`,
  `nb.level_contributions`, `nb.PseudoRealTimeBacktest`, `nb.NowcastDistribution`,
  `nb.nowcast_distribution`, `nb.scoring`, `nb.run_diagnostics`, `nb.NowcastExperiment`,
  `nb.NowcastReport`, `nb.load_brazil_nowcast` and the other dataset loaders).
- Native analysis methods on every `NowcastResults`: `news(old, new, target_period)`,
  `nowcast_tracker(...)`, `level_contributions(...)`, `distribution(...)`,
  `diagnostics(...)`; `MixedFreqDFMResults.state_space_model()`.
- `nb.nowcast(..., idiosyncratic=, long_run_mean=, outliers=, density=True, n_boot=,
  random_state=)` and `blocks="auto"`; `nowcastbox.api.add_density`.
- `results.plot("density")`; `NewsResults.plot("waterfall")` and
  `NowcastTracker.plot("path")` accept `backend=`/`theme=`; `plot_fan_chart` accepts a
  `NowcastDistribution`; `NowcastReport(diagnostics=True|DiagnosticsReport)` and real
  `NowcastTracker`/`NowcastDistribution`/`BacktestResults` inputs;
  `NowcastExperiment.run_backtest()` defaults to a pseudo real-time backtest.
- `fetch_sidra`/`fetch_sidra_raw(dash_as=...)`: choose the value of SIDRA's `-` token
  (e.g. NaN for index series that are not available).
- `tests/integration/test_wave2_end_to_end.py`: Brazilian dataset -> robust EM -> news
  and tracker -> density and scores -> backtest vs AR -> diagnostics -> HTML report.
- MkDocs navigation for the datasets pages and API pages of the wave-2 subpackages.
- Wave 3 — **I1 calendar-aware aggregation**: `core.frequency.is_fixed_ratio`,
  `max_periods_per`, `native_period_bounds`, `calendar_position`,
  `AggregationType.calendar_weights`; `preprocessing.CalendarAggregation`,
  `calendar_aggregation`, `calendar_weight_matrix`; `MixedFreqDFM` on weekly/daily grids
  (time-varying design, `StateLayout(calendar=, start=)`,
  `MixedFreqDFMResults.state_space_model(n_periods)`,
  `params["aggregation_weight_paths"]`).
- Wave 3 — news decomposition, nowcast tracker and level contributions for calendar
  (weekly/daily) EM models (the time-varying observation equation is rebuilt per
  vintage) and for `TwoStepDFM(aggregate="variables")` (vintages pass through the fitted
  predictor filters); `batched_functional` supports time-varying observation equations.
- Wave 3 — density: parametric bootstrap for calendar EM models and for
  `TwoStepDFM(aggregate="variables")` (simulates the filtered panel, refits with
  `prefiltered=True`); explicit error for the block bootstrap on calendar grids.
- Wave 3 — diagnostics on calendar grids (per-period aggregation weights in
  `series_weights`/`aggregate_factors`); model-based idiosyncratic residuals for
  `TwoStepDFM(aggregate="variables")` (filtered predictors minus the fitted signal).
- Wave 3 — **monotone EM**: generalized EM step halving of the transition updates
  against the exact expected complete-data log-likelihood including the stationary
  initial-state term (`m_step(initial_term=True)`).
- Wave 3 — **I7** `selection.select_blocks` / `select_variables` (greedy forward or
  backward search scored by pseudo real-time RMSFE/MAE or an information criterion) ->
  `SelectionPath`; `ValidationSettings`.
- Wave 3 — `TwoStepResults.model_data`, `variable_weights`, `is_filtered`,
  `filter_panel()`; `TwoStepDFM.fit/update(..., prefiltered=True)`.
- Wave 3 — **I10** `nowcastbox.pipeline` (`NowcastSpec`, `load_spec`, `run_pipeline` ->
  `PipelineRun`, `SnapshotStore`/`Snapshot`/`SnapshotDiff`, data loaders for datasets,
  CSV/Parquet files and the BCB/IBGE/IPEA/FRED connectors, four templates) and the
  `nowcastbox` command line (`run`, `validate`, `init`, `datasets`, `snapshots`).
- Wave 3 — `nowcastbox.simulate`: `dfm()` (monthly + quarterly) and `weekly_dfm()`
  (weekly + monthly + quarterly) -> `SimulatedDFM(data, truth)`.
- Wave 3 — top-level exports: `nb.run_pipeline`, `nb.NowcastSpec`, `nb.PipelineRun`,
  `nb.SnapshotStore`, `nb.select_blocks`, `nb.select_variables`, `nb.SelectionPath`,
  `nb.ValidationSettings`, `nb.CalendarAggregation`, `nb.calendar_aggregation`,
  `nb.pipeline`, `nb.cli`, `nb.simulate`; `core`/`preprocessing` export the calendar
  helpers.
- Wave 3 — `PseudoRealTimeBacktest(model_name=...)` labels the main model.
- Wave 3 — reference validation (`tests/reference_validation/`, 90+ tests) against R
  `nowcasting` 1.1.2 called as a black box (`scripts/reference_fixtures/`) and
  statsmodels; `docs/validation/`; `benchmarks/bench_vs_r.py`.
- Wave 3 — documentation site (getting started, 40 user-guide pages with tested
  examples, theory, comparison, validation, API), 15 example notebooks executed with
  outputs (`examples/`), `tests/integration/test_notebooks.py` (slow) and a CI job
  that runs them; JSS software paper and applied Brazilian GDP paper drafts (`paper/`).

### Changed

- `pyproject.toml`: the `data` extra now holds `pyarrow` (Parquet I/O of
  `VintageStore`); the unused `requests-cache` and `fredapi` extras were removed;
  `threadpoolctl` added to the `test`/`dev` extras (and `pyarrow` to `dev`).
- `resolve_transforms` gained `include_applied=False`.
- `MixedFreqDFM(filter_method="auto")` is the new default: the EM E-step and the final
  smoothing pass use the exact structured smoother when it is cheaper (same estimates up
  to ~1e-11; about 12x faster per EM iteration than statsmodels `DynamicFactorMQ` at
  N = 200, T = 300); `"structured"` forces it and `"univariate"`/`"multivariate"` force
  the dense Kalman smoother. The robust E-steps and `results.smooth()` stay dense.
- `nowcastbox.news.attach()` adds nothing to `NowcastResults` (the methods are native);
  it remains available for duck-typed results classes. The news default plot functions
  are public (`plot_waterfall_default`, `plot_path_default`).
- `NowcastReport` diagnostics placeholder now explains `diagnostics=True`.
- `pseudo_real_time`/`ReleaseCalendar.release_mask` compute the period ends once per
  frequency (about 18x faster per vintage, 8x faster than R's `PRTDB`);
  `MixedFrequencyData.slot_mask` and the slot validation compute `is_period_end` once
  per frequency.
- `api.select_n_factors` no longer warns twice about small panels.
- CI: the test matrix deselects `slow` tests; a separate job runs the documentation
  examples, the README quickstart and the notebooks; the reference-validation workflow
  regenerates the R/statsmodels fixtures in the documented order.

### Fixed

- `prepare_panel`/`apply_transforms` called twice with `transform=None` on a
  `MixedFrequencyData` no longer transform the data twice (the metadata transform is
  skipped for series flagged `transform_applied`); an explicit re-transformation
  emits a `DataQualityWarning`.
- `NowcastResults.summary()` widens the parameter-name column for names longer than
  22 characters instead of running them into their values.
- Flaky hypothesis health check in the outlier property test (too many inputs
  filtered out).
- `mkdocs build --strict` failed on dataclass docstrings documenting `**kwargs`
  (results classes now describe the inherited fields in *Notes*).
- EM monotonicity property test was flaky: hypothesis-found cases (relative decreases
  2.2e-7 and 3.0e-5 on T = 72, identical with the dense and structured E-steps, due to
  the initial-distribution term ignored by the M-step) are pinned and the tolerance
  (1e-4 relative) documented.
- `BacktestResults.mcs(horizon="months_to_end")` raised `KeyError`.
- `pseudo_real_time` with a named delay Series covering more series than the panel
  (e.g. `Dataset.delay` after `prepare_panel` dropped columns) raised instead of
  selecting by name.
- `fill_missing` on weekly/daily grids wrote low-frequency values into the wrong week
  (`preprocessing._utils` now maps native periods with `native_to_base`).
- `run_pipeline(..., snapshot=False)` skipped the news against the previous snapshot.
- A hypothesis-found subnormal underflow in `test_relative_rmsfe_scale`.
