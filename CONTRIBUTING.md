# Contributing to NowcastBox

Thank you for your interest in contributing! This guide summarises the workflow and
quality rules (see also `CLAUDE.md` and `desenvolvimento/CONTRATOS.md`).

## Development setup

```bash
git clone https://github.com/NowcastBox/nowcastbox.git
cd nowcastbox
pip install -e ".[dev]"
pre-commit install
```

## Workflow

1. Create a feature branch (`feat/<topic>`, `fix/<topic>`).
2. Write code **and tests** (unit + analytical/simulation checks; `hypothesis` property
   tests where natural).
3. Run the quality gates locally:

   ```bash
   ruff check nowcastbox tests && ruff format nowcastbox tests
   pyright nowcastbox
   pytest --cov=nowcastbox
   interrogate nowcastbox
   ```

4. Update `CHANGELOG.md` (*Unreleased* section) and the documentation.
5. Open a pull request; a review by the co-author is mandatory. Use
   [Conventional Commits](https://www.conventionalcommits.org/) for commit messages.

## Quality requirements

| Aspect | Target |
|---|---|
| Lint/format | ruff, 0 errors (line length 100, NumPy docstrings) |
| Coverage | ≥ 90 % overall, ≥ 95 % for `core/` and `statespace/` |
| Docstrings | ≥ 95 % (interrogate), NumPy style with *Parameters, Returns, Raises, Examples* |
| Complexity | ≤ 10 per function (mccabe) |
| Typing | full type hints; pyright clean |

## Clean-room rule (MIT licence)

NowcastBox is MIT-licensed. Methods are implemented **from the academic literature**.
Do **not** read, consult or translate the source code of GPL-licensed packages (in
particular the R package `nowcasting`), and do not copy code from other projects.
Reference implementations may only be run as black boxes to generate test fixtures.
Record the papers used for each module in `desenvolvimento/FONTES_POR_MODULO.md`.

## Conventions

- Data: `pandas.DataFrame` with a monthly `PeriodIndex` base grid; quarterly values
  in the third month of the quarter; metadata in `MixedFrequencyData`.
- Parameters in descriptive `snake_case` (`n_factors`, `factor_lags`, `n_shocks`,
  `blocks`).
- Explicit errors (`NowcastDataError`, `ValueError`), never silent failures; warnings
  from `nowcastbox.core.exceptions` (`ConvergenceWarning`, `DataQualityWarning`).
- Every source of randomness takes a `random_state`.
