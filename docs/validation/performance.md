# Performance vs R `nowcasting` and statsmodels

Measured with `python benchmarks/bench_vs_r.py --repeat 3` on WSL2, one BLAS thread. The
nowcastbox time is the median of 3 runs after a warm-up. The R timings come from
`scripts/reference_fixtures/r_timings.R`, run alone on an idle machine on 2026-10-01: the
median of 3-5 calls, except the NYFED EM (one call). Both libraries get the **same
inputs**.

| Task | R `nowcasting` (s) | nowcastbox (s) | R / nowcastbox |
|---|---:|---:|---:|
| Codes 0-7, BRGDP (100 series) | 0.533 | 0.038 | 13.9x |
| Panel cleaning, BRGDP | 0.428 | 0.221 | 1.9x |
| Panel cleaning, USGDP (192 series) | 1.320 | 0.504 | 2.6x |
| Cleaning + MM filter, USGDP | 1.535 | 0.771 | 2.0x |
| 2s, USGDP (r = 2, p = 2, q = 2) | 1.584 | 0.408 | 3.9x |
| 2s_agg, USGDP (r = 2, p = 2, q = 2) | 1.643 | 0.386 | 4.3x |
| 2s_agg, simulated (24 x 252) | 0.193 | 0.060 | 3.2x |
| EM, simulated (1 block, r = 2) | 2.479 | 0.249 | 10.0x |
| **EM, NYFED (4 blocks, 53 states)** | **56.439** | **1.606** | **35.1x** |
| Bai-Ng IC, USGDP (rmax = 15) | 0.103 | 0.003 | 30.0x |
| Bai-Ng shocks, USGDP (r = 4, p = 2) | 0.019 | 0.009 | 2.1x |
| Pseudo real-time vintage, BRGDP | 0.060 | 0.007 | 8.1x |

| NY Fed-like EM, no measurement noise, tol 1e-6 | Seconds | Iterations | Log-likelihood |
|---|---:|---:|---:|
| statsmodels `DynamicFactorMQ.fit_em` | 7.20 | 71 | −9347.64 |
| nowcastbox `MixedFreqDFM` (`init="pca"`, best block order) | 7.77 | 121 | −9393.68 |
| nowcastbox `MixedFreqDFM` (`init="pca_given"`, sequential start) | 6.24 | 88 | −9765.28 |

Re-measured at the wave-3 integration (2026-10-01, idle machine, `--repeat 3`): the
release mask of `pseudo_real_time` now computes the period ends once per frequency and
`MixedFrequencyData.slot_mask` once per frequency, which made the vintages about 20x
and the panel cleaning about 2x faster than in the first measurement.

Notes:

- Some of these tasks do not run the same algorithm in both libraries:
  - *Panel cleaning*: the outlier and missing-value rules differ (see
    [Pre-processing](preprocessing.md)).
  - *EM, NYFED*: nowcastbox stops after 20 iterations and R after about 43. Per
    iteration, nowcastbox is roughly 15x faster (0.08 s against about 1.3 s).
  - *NY Fed-like EM*: with `obs_noise_var=0` nowcastbox falls back to the dense
    smoother, because the structured smoother (innovation I2) needs a positive
    measurement noise.
- The R timings are sensitive to load: under concurrent CPU use the same NYFED EM took
  124 s. Refresh them on an idle machine with
  `python benchmarks/bench_vs_r.py --rerun-r`.
