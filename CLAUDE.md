# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in
this repository.

## Quick Start Commands

### Installation & Setup
```bash
pip install -e ".[dev]"        # editable install with dev tools
# (system Python under PEP 668: pip install --user --break-system-packages -e ".[dev]")
pre-commit install             # git hooks
```

### Development Workflow
```bash
# Lint and format
ruff check nowcastbox/ tests/
ruff format nowcastbox/ tests/

# Type checking
pyright nowcastbox/

# Tests
pytest                                   # all tests
pytest tests/core/                       # one module
pytest -k ragged                         # by name
pytest -m "not slow and not network"     # skip slow / internet tests
pytest tests/core --cov=nowcastbox/core --cov-report=term-missing

# Quality
interrogate nowcastbox -v      # docstring coverage (>= 95%)
radon cc nowcastbox -a -nc     # complexity
bandit -c pyproject.toml -r nowcastbox -ll

# Docs
mkdocs serve                   # http://localhost:8000
mkdocs build --strict
```

### Quality Requirements (enforced in CI)
- Coverage >= 90% overall (>= 95% for `core/` and `statespace/`), branch coverage.
- Docstring coverage >= 95%, NumPy style with Parameters/Returns/Raises/Examples.
- Cyclomatic complexity <= 10 per function (ruff C90).
- ruff clean (line length 100); pyright "standard" mode clean.
- pytest markers: `slow`, `network`, `reference_divergence`, `reference_validation`,
  `benchmark`, `integration`, `property`.

## Licensing rule (MIT clean room)

Implement methods **from the literature** (plan §4, §11). Never read, consult or
translate the source of the GPL R package `nowcasting` (or any copy on disk); never
copy statsmodels code (it may be used as a dependency and as a numerical reference in
tests). Record the papers used per module in `desenvolvimento/FONTES_POR_MODULO.md`.

## Architecture

The plan is `desenvolvimento/PROJETO_DESENVOLVIMENTO.md`; the binding core contracts are
documented in `desenvolvimento/CONTRATOS.md` — read it before touching any module. The
papers live in the private repository `NowcastBox/nowcastbox-papers`.

### Core contracts (`nowcastbox/core/`)
- `data.py` — `MixedFrequencyData`: DataFrame on a contiguous base `PeriodIndex`
  (monthly in v0.1) + per-series `SeriesMetadata` (frequency, transform, release delay
  in days, blocks, category, aggregation). Lower-frequency values live in the **last
  base period** of their period (quarterly → 3rd month). Provides masks (ragged edge,
  slots, missing), `as_of` (pseudo real-time), `standardize`/`destandardize`,
  `select`/`truncate`/`extend`/`with_data`.
- `frequency.py` — `Frequency` enum (D, W, M, Q, A), `AggregationType` (flow, stock,
  average, Mariano-Murasawa) with weights, period-grid helpers.
- `base.py` — `BaseNowcaster` (sklearn-like params, `fit(data, target) -> NowcastResults`,
  formula targets), `BaseBenchmark` + `BenchmarkForecaster` protocol.
- `results.py` — frozen `NowcastResults` (`nowcast` frame with `observed`/`in_sample`/
  `out_of_sample` on the target's native PeriodIndex), `FactorResults`, plot registry
  (`register_plot`), `save`/`load`.
- `formula.py` — `"y ~ ."`, `"y ~ x1 + x2"`, `"y ~ . - x3"`.
- `exceptions.py` — `NowcastDataError`, `ModelNotFittedError`, `FormulaError`,
  `ConvergenceWarning`, `DataQualityWarning`.

### Other subpackages
`preprocessing/` (transforms 0-7, outliers, missing, aggregation, `prepare_panel`),
`statespace/` (Kalman filter/smoother with NaNs, Numba; exact structured smoother used by
the EM, `filter_method="auto"`), `models/` (`TwoStepDFM`, `MixedFreqDFM` with robust and
long-run-mean options, `BridgeEquation`), `selection/` (Bai-Ng), `vintages/`, `news/`
(news decomposition, tracker), `density/` (density nowcasts), `benchmarks/`,
`evaluation/` (backtest, forecast tests, `scoring`), `diagnostics/`, `data_sources/`
(BCB, IBGE, IPEA, FRED), `datasets/`, `simulate/`, `visualization/` (registers plots),
`reports/`, `experiment/`, `pipeline/`, `cli/`. `NowcastResults` exposes `news`,
`nowcast_tracker`, `level_contributions`, `distribution` and `diagnostics` methods
(lazy delegation to those subpackages).

### Data Flow
```
DataFrame (monthly PeriodIndex) ──► MixedFrequencyData ──► preprocessing
     ──► Model(...).fit(data, target="gdp" | "gdp ~ .") ──► NowcastResults
     ──► summary() / nowcast / plot(kind) / news / evaluation
```

## Conventions

- State space notation (Durbin & Koopman): `α_{t+1} = T α_t + R η_t`, `y_t = Z α_t + ε_t`.
  Math-named variables (`Lambda`, `Psi`, `A`, `Q`) are allowed (ruff N803/N806 ignored).
- Each subpackage exports its public API in its own `__init__.py`; the top-level
  `nowcastbox/__init__.py` is edited only in the integration step.
- Tests mirror the source tree: `tests/<subpackage>/`. Shared fixtures: `tests/conftest.py`
  (`rng`); hypothesis profile via `HYPOTHESIS_PROFILE=ci|dev`.
- Logging via `nowcastbox._logging.get_logger(__name__)`; never `print`.

## Key Files
- `pyproject.toml` — metadata, deps, ruff/pytest/coverage/pyright/interrogate config.
- `desenvolvimento/CONTRATOS.md` — core contracts and module ownership map.
- `desenvolvimento/FONTES_POR_MODULO.md` — clean-room source record.
- `nowcastbox/core/data.py` — the central data container.
