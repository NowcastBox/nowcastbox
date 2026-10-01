# Two-step DFM vs `nowcast(method = "2s" / "2s_agg")`

Tests: `tests/reference_validation/test_r_two_step.py`. Fixtures:
`scripts/reference_fixtures/r_two_step.R`.

The cases are:

- the package examples on **USGDP**, the Giannone, Reichlin & Small (2008) panel: 189
  predictors, r = 2, p = 2, q = 2. In `"2s"` the predictors are filtered by
  `Bpanel(aggregate = TRUE)`; in `"2s_agg"` the factors are aggregated;
- the **simulated** panel (24 monthly series, quarterly target, ragged edge): r = 2,
  p = 1;
- a **balanced** version of the simulated panel with no ragged edge, with p = 1 and p = 2.

nowcastbox mapping: `TwoStepDFM(n_factors=r, factor_lags=p, n_shocks=q,
aggregate="factors")` for `"2s_agg"`. For `"2s"`, the test feeds R's filtered panel to
`aggregate="variables", aggregation=[1.0]`.

## Layer 1: first step (PCA, VAR, idiosyncratic variances)

This layer is estimated on the balanced rows, where both libraries standardise on the
same sample.

| Quantity | Agreement | Convention |
|---|---:|---|
| Loadings Λ (up to sign) | ~1e-15 | same |
| VAR matrices A | ~1e-14 | same (OLS, no intercept) |
| Eigenvalues | ~1e-15 after x T/(T−1) | R: `cov` (1/(T−1)); nowcastbox: 1/T |
| Ψ | ~1e-15 after x T/(T−1) | R: 1/(T−1); nowcastbox: 1/T |
| BB (q = r) | ~3e-14 | R: demeaned `cov` (ddof = 1) of the VAR residuals; nowcastbox: e'e/(T−p) |

## Layer 2: Kalman smoother, aggregation, bridge

Here R's own parameters (Λ, A, BB, Ψ, `initx`, `initV`, mean, std) go through
nowcastbox's Kalman smoother, Mariano-Murasawa factor aggregation and bridge OLS:

| Case | Factors max abs | Nowcasts (`yfcst`) max abs | Bridge coefficients |
|---|---:|---:|---:|
| usgdp_2s | 1.5e-12 | 3.0e-14 | < 1e-6 |
| usgdp_2s_agg | 7.6e-13 | 1.1e-13 | < 1e-6 |
| sim_2s | 4.9e-13 | 4.1e-14 | < 1e-6 |
| sim_2s_agg | 2.6e-13 | 7.8e-14 | < 1e-6 |
| simbal_2s_agg | 1.8e-13 | 8.1e-14 | < 1e-6 |
| simbal_2s_agg_p2 | 3.1e-13 | 1.1e-13 | < 1e-6 |

The plan's tolerance is 1e-6. R's bridge uses the unnormalised weights (1, 2, 3, 2, 1),
so its coefficients are 1/9 of nowcastbox's; the predictions are identical. R's `initx` and
`initV` are the moments of the **first** state.

## Layer 3: full default pipelines (`reference_divergence`)

| Case | Min \|corr\| of factors | Max \|nowcast diff\| / sd(y) |
|---|---:|---:|
| usgdp_2s | 0.99999 | 0.0048 |
| usgdp_2s_agg | 0.99618 | 0.0677 |
| sim_2s | 1.00000 | 0.0006 |
| sim_2s_agg | 0.99996 | 0.0023 |
| simbal_2s_agg | 1.00000 | 0.0000 (6e-5 abs) |
| simbal_2s_agg_p2 | 1.00000 | 0.0001 |

The pipelines differ in three ways:

1. **Standardisation sample.** R uses the mean and standard deviation of the balanced
   rows; nowcastbox uses every observation, as its core contract requires. On USGDP the
   2007 ragged-edge observations (for example the jump in the BAA spread) change the
   statistics by up to 6%. This explains the larger `usgdp_2s_agg` gap.
2. **Moment conventions.** These are the 1/(T−1) and ddof differences of layer 1, about
   0.4% on these samples.
3. **Initial state.** R starts from the first PC factor values and their sample
   covariance; nowcastbox starts from the stationary distribution of the VAR.

## R-side issues found

- **q < r gives an invalid state covariance.** With r = 2, q = 1, R returns
  `BB = [[1.19, 1.19], [4.89, 4.89]]`, which is not symmetric and not B·B'. nowcastbox
  returns the rank-q matrix built from the leading eigenvectors of the residual covariance
  (Doz, Giannone & Reichlin, 2011).
- **Inconsistent de-standardisation of `xfcst`.** R estimates with balanced-row
  statistics but maps the smoothed common component back with the full-sample mean and
  standard deviation (identified exactly). nowcastbox uses one set of statistics
  throughout. The gap is 0.012 at the end of the simulated forecast horizon.
- `nowcast()` needs empty rows for the forecast horizon (`Bpanel` adds h = 12) and fails
  in `"2s"` when the filtered predictors have leading missing values.
