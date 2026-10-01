# Installation

NowcastBox requires **Python 3.10 or later**.

## From PyPI

```bash
pip install nowcastbox            # once released on PyPI
```

## From a clone (development version)

```bash
git clone https://github.com/NowcastBox/nowcastbox.git
cd nowcastbox
pip install -e ".[dev]"
```

On a system Python protected by PEP 668 (Debian/Ubuntu), add
`--user --break-system-packages` or use a virtual environment.

## Dependencies

| Required | Used for |
|---|---|
| `numpy`, `scipy`, `pandas >= 2.0` | arrays, linear algebra, period-indexed panels |
| `numba` | compiled Kalman filter kernels |
| `statsmodels` | OLS of bridge equations and VARs (never on the Kalman/EM critical path) |
| `matplotlib`, `plotly` | publication and interactive charts |
| `jinja2` | HTML reports |
| `pyyaml` | dataset metadata and the YAML pipeline |
| `requests` | BCB, IBGE, IPEA and FRED connectors |
| `joblib` | parallel bootstrap and backtests (`n_jobs`) |

Optional extras:

| Extra | Installs | When you need it |
|---|---|---|
| `[data]` | `pyarrow` | Parquet I/O of `VintageStore` and `BacktestResults.to_parquet` |
| `[docs]` | MkDocs Material, mkdocstrings | building this documentation |
| `[test]` | pytest, hypothesis, threadpoolctl... | running the test-suite |
| `[dev]` | everything above + ruff, pyright, pre-commit | contributing |
| `[r]` | `rpy2` | regenerating the reference fixtures from R |

`scikit-learn` is not a dependency: install it only to use
[`SklearnBenchmark`](../user-guide/models/benchmarks.md#any-scikit-learn-regressor-i11).

## Check the installation

```python
import nowcastbox as nb

print(nb.__version__)
print(nb.list_datasets()[["title"]])
```

!!! tip "Multithreaded BLAS"
    The Kalman smoother and the EM algorithm work with many small and medium-sized
    matrices, for which multithreaded OpenBLAS/MKL is often *slower* than a single
    thread (thread oversubscription, notably under WSL or with many cores). If fits
    look unexpectedly slow, set `OMP_NUM_THREADS=1` (or `OPENBLAS_NUM_THREADS=1`)
    before starting Python. See [Performance](../user-guide/performance.md).

## Building the documentation

```bash
pip install -e ".[docs]"
mkdocs serve            # http://localhost:8000
mkdocs build --strict
```
