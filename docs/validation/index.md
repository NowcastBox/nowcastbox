# Validation against reference implementations

nowcastbox is checked against the R package
[`nowcasting`](https://github.com/nmecsys/nowcasting) 1.1.2 (de Valk et al., 2019) and
against statsmodels' `DynamicFactorMQ` (plan §10.4). The R package is GPL-3 and was used
**only as a black box**: its exported functions were called on its bundled datasets
(`USGDP`, `NYFED`, `BRGDP`) and on a shared simulated panel, and the outputs were stored
as test fixtures. See [Reproducing the fixtures](reproducing.md).

Every comparison is a test in `tests/reference_validation/`. Where the two specifications
are truly equivalent, the plan's tolerances apply (marker `reference_validation`). Where
nowcastbox intentionally does something different, or where R deviates from the
literature, the difference is measured and explained, and the test is marked
`reference_divergence`.

## Summary

| Component | R / statsmodels function | Result | Max abs error (equivalent spec) | Plan tolerance | Page |
|---|---|---|---:|---:|---|
| Transformations, codes 0-7 (monthly; quarterly 6/7) | `Bpanel` | **identical** on every non-outlying cell | 2.5e-16 (rel.) | 1e-10 | [Pre-processing](preprocessing.md) |
| Outlier detection (4 IQR) | `Bpanel` | **identical flags** (138/138 BRGDP cells) | - | - | [Pre-processing](preprocessing.md) |
| Outlier replacement, missing-value filling | `Bpanel` | divergence (moving median vs moving average; no filling of lag-lost values) | max 2.43 sd, median 0.44 sd at outlier cells | - | [Pre-processing](preprocessing.md) |
| Mariano-Murasawa aggregation | `Bpanel(aggregate = TRUE)` | identical with `normalize=False`; R filters before correcting outliers | < 1e-10 | 1e-10 | [Pre-processing](preprocessing.md) |
| Series dropped (> 1/3 missing) | `Bpanel` | identical set | - | - | [Pre-processing](preprocessing.md) |
| Monthly to quarterly (month 1/2/3, mean) | `month2qtr` | identical | 4.3e-14 | 1e-10 | [Pre-processing](preprocessing.md) |
| Quarterly to monthly (placement, linear interpolation) | `qtr2month` | identical; interpolation anchored in month 1/2 not offered | 4.8e-12 | 1e-10 | [Pre-processing](preprocessing.md) |
| Bai-Ng (2002) IC, r* | `ICfactors` | **identical r\*** in 18/18 cases | 6.4e-15 | exact / 1e-8 | [Selection](selection.md) |
| Bai-Ng (2007) primitive shocks, q* | `ICshocks` | same D1 statistic; **R's bound differs from the paper** | thresholds 1e-10 | exact | [Selection](selection.md) |
| Two-step, first step (PCA, VAR, Ψ, BB) | `nowcast("2s"/"2s_agg")` | identical up to documented ddof conventions | 3e-14 | 1e-6 | [Two-step](two_step.md) |
| Two-step, Kalman smoother + bridge + nowcasts | `nowcast("2s"/"2s_agg")` | **identical** with R's parameters | 1.5e-12 | 1e-6 | [Two-step](two_step.md) |
| Two-step, full default pipeline | `nowcast("2s"/"2s_agg")` | divergence (standardisation sample, ddof, initial state); factor corr ≥ 0.996 | ≤ 0.068 sd(y) | - | [Two-step](two_step.md) |
| EM, Kalman smoother at R's estimates | `nowcast("EM")` | **identical** (Z_0/V_0 at t = 0) | 9.4e-14 | 1e-4 | [EM](em.md) |
| EM, log-likelihood | `nowcast("EM")` (printed) | same function minus the Gaussian constant | consistent within the EM path | 1e-4 | [EM](em.md) |
| EM estimates / nowcasts | `nowcast("EM")` | divergence (start, stopping); nowcastbox reaches a **higher** likelihood (+20.6 on NYFED) | ≤ 0.067 sd(y) on the horizon | - | [EM](em.md) |
| EM vs `DynamicFactorMQ`, likelihood and smoother | statsmodels 0.14.6 | **identical** at statsmodels' estimates | ~1e-12 rel. / < 1e-6 | 1e-4 | [statsmodels](statsmodels.md) |
| EM vs `DynamicFactorMQ`, optimum | statsmodels 0.14.6 | different local maximum: **46 points lower** with the multi-start `init="pca"` (418 with the sequential `init="pca_given"`) | - | - | [statsmodels](statsmodels.md) |
| Pseudo real-time vintages | `PRTDB` | **identical** in 191 BRGDP and 216 NYFED vintages, including negative delays | exact | exact | [Vintages](vintages.md) |
| News decomposition | (`News_DFM_ML` is not exported) | not comparable through the black box | - | 1e-5 | - |

Run time: nowcastbox is faster on every estimation task, for example the NYFED EM takes
1.6 s against R's 56 s, and a pseudo real-time vintage takes 0.007 s against 0.060 s.
See [Performance](performance.md).

## Discrepancies found

**Issues found in nowcastbox** (all addressed in the wave-3 integration):

1. *Multi-block EM initialisation* (`models/_init_conditions.py`): the block-sequential
   PCA start led to a local maximum 418 log-likelihood points below statsmodels' optimum
   on the NY Fed specification. The default `init="pca"` is now a multi-start over block
   orders and ends 46 points below it, at another local maximum of the same likelihood
   (still open: multi-block EM has several local maxima). Both EM implementations are
   correct: started from statsmodels' optimum, nowcastbox's EM stays there.
2. *Negative release delays*: surveys published before the end of their reference month
   (Empire State and Philadelphia Fed, delay −14 days in NYFED) are now accepted, and
   every NYFED vintage matches `PRTDB`.
3. *Vintage speed*: `pseudo_real_time` went from 3x slower than `PRTDB` to 8x faster.

**R-side deviations** (nowcastbox follows the literature):

* `ICshocks` uses the bound m / min(N, T)^{1/(2−δ)} rather than Bai & Ng's
  m / min(N^{1/2−δ}, T^{1/2−δ}), so R selects more shocks (often q = r).
* `nowcast("2s"/"2s_agg")` with q < r returns a **non-symmetric** `BB`, so its state
  covariance is invalid.
* `nowcast("2s_agg")` estimates on data standardised over the balanced rows but maps
  `xfcst` back with the full-sample mean and standard deviation.
* `Bpanel` corrects outliers even with `NA.replace = FALSE`.
* `month2qtr(reference_month = "mean")` fails on multivariate series.
* R's EM `yfcst` is not the smoothed signal of the returned model and could not be
  reproduced from `Res`.
* `nowcast()` needs empty rows for the forecast horizon and fails on leading missing
  values in `"2s"`.

## Pages

- [Pre-processing](preprocessing.md) · [Selection](selection.md) · [Two-step](two_step.md) ·
  [EM](em.md) · [statsmodels](statsmodels.md) · [Vintages](vintages.md) ·
  [Performance](performance.md) · [Reproducing the fixtures](reproducing.md)
