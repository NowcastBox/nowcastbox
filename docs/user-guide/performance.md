# Performance

## Limit BLAS threads

The Kalman filter, the smoothers and the EM work with many **small and medium-sized
matrices**. For these, multithreaded OpenBLAS/MKL usually *slows things down*: the cost
of waking and synchronising threads exceeds the work, and with many cores (or under WSL,
or when you also parallelise with `n_jobs`) threads oversubscribe the CPU. Slowdowns of
several times are common.

Set the number of BLAS threads **before** importing NumPy:

```bash
export OMP_NUM_THREADS=1          # or OPENBLAS_NUM_THREADS=1 / MKL_NUM_THREADS=1
python my_nowcast.py
```

or limit them inside a session with `threadpoolctl`:

```python
import nowcastbox as nb

try:
    from threadpoolctl import threadpool_limits
except ImportError:            # threadpoolctl is installed with the [test] extra
    threadpool_limits = None

ds = nb.load_simulated_dfm()
if threadpool_limits is not None:
    with threadpool_limits(limits=1, user_api="blas"):
        res = nb.MixedFreqDFM(n_factors=2).fit(ds.data, "gdp")
else:
    res = nb.MixedFreqDFM(n_factors=2).fit(ds.data, "gdp")
res.converged
```

The test-suite runs with one BLAS thread (`tests/conftest.py`); use
`NOWCASTBOX_TEST_BLAS_THREADS` to change it. Parallelise at a coarser level instead —
vintages of a backtest (`PseudoRealTimeBacktest(n_jobs=-1)`) or bootstrap replications
(`distribution(n_jobs=-1)`).

## The structured EM (innovation I2)

The E-step of the EM needs smoothed moments of a large state: with $N$ series, AR(1)
idiosyncratic components and Mariano-Murasawa lags for quarterly series the state has
hundreds of elements, and the dense Kalman smoother costs $O(T m^3)$ for state dimension
$m$. NowcastBox exploits the structure of the model:

- the **univariate treatment** of the observations (Koopman & Durbin, 2000) avoids
  inverting $N \times N$ matrices;
- **collapsing** the observation vector (Jungbacker & Koopman, 2015) for the two-step
  model (`TwoStepDFM(collapse=True)`);
- the **exact structured smoother**: the posterior of the whole state path has a sparse
  precision matrix — banded blocks for each idiosyncratic chain, coupled only through
  the common factors — so the smoothed moments come from banded Cholesky factorisations
  and a selected inverse at a cost $O(N n^2)$ instead of $O(n N^3)$ (Rue & Held, 2005;
  Chan & Jeliazkov, 2009). It is exact: estimates agree with the dense smoother to
  about $10^{-10}$.

`MixedFreqDFM(filter_method="auto")` (the default) uses the structured smoother whenever
the model has the required structure and a cost heuristic predicts it is cheaper; force
a path with `"structured"`, `"univariate"` or `"multivariate"`:

```python
import time

timings = {}
for method in ("auto", "univariate"):
    start = time.perf_counter()
    fit = nb.MixedFreqDFM(n_factors=2, max_iter=5, tol=0.0, filter_method=method).fit(
        ds.data, "gdp"
    )
    timings[method] = (time.perf_counter() - start, fit.loglikelihood)
timings
```

### Benchmark

`python benchmarks/bench_em.py` compares the EM with `statsmodels`'
`DynamicFactorMQ(...).fit_em` on the **same model** (N = 200: 180 monthly + 20 quarterly
series, T = 300, one factor, AR(1) idiosyncratic components, Mariano-Murasawa
aggregation, 285 states, one BLAS thread):

| | nowcastbox (`"auto"`, structured) | nowcastbox dense (`"univariate"`) | statsmodels `DynamicFactorMQ` |
|---|---:|---:|---:|
| seconds per EM iteration | **0.40** | 4.95 | 4.84 |
| speed-up vs. statsmodels | **≈ 12×** | ≈ 1× | 1× |

Log-likelihoods of the dense and structured paths agree to $4 \times 10^{-11}$.
Timings vary between machines and runs; regenerate them with the script
(`--monthly`, `--quarterly`, `--periods`, `--threads`, `--json`).

### Limits of the structured smoother

- Its cost grows as $O(N T^2 w) + O((T w)^3)$ for a common block of width $w$: for very
  long samples ($T \gtrsim 1000$) or wide common blocks the dense smoother wins, and
  `"auto"` switches to it.
- It has a fixed overhead per series and period: for small panels the dense smoother is
  faster (one factor, $T = 300$: dense up to $N \approx 35$, structured from
  $N \approx 50$). The cost model of `"auto"` was calibrated on such E-step timings.
- It needs a strictly positive diagonal measurement noise (`obs_noise_var > 0`).
- The robust E-steps (Student-t, outliers, pandemic options) and
  `MixedFreqDFMResults.smooth/predict` use the dense smoother.

## Other tips

| Situation | Tip |
|---|---|
| Monthly re-estimation in a backtest | `refit_every=3` (or more) and warm starts: between refits only the information is updated |
| Many specifications | `MixedFreqDFM(init=previous_results)` warm-starts the EM |
| Bootstrap densities | `refit_params={"max_iter": 50}`, `n_jobs=-1` |
| Large two-step panels | `TwoStepDFM(collapse=True)` |
| First call is slow | Numba compiles the Kalman kernels on first use (cached afterwards) |
| MIDAS benchmark is slow | restrict `predictors` and `n_lags`; prefer `UMIDAS` for many indicators |
