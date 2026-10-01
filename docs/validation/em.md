# EM vs `nowcast(method = "EM")`

Tests: `tests/reference_validation/test_r_em.py`. Fixtures:
`scripts/reference_fixtures/r_em.R`.

The cases are:

- **NYFED**, the package example and a replication of the FRBNY Staff Nowcast: 25
  series, 4 blocks (Global, Soft, Real, Labor) with one factor each, VAR(1), AR(1)
  idiosyncratic components, Mariano-Murasawa restriction for GDP and unit labour costs,
  measurement noise 1e-4, 53 states;
- the **simulated** panel: one block, two factors.

nowcastbox: `MixedFreqDFM(n_factors=1, factor_lags=1, blocks=<legend blocks>)` with the
defaults `idiosyncratic="ar1"`, `obs_noise_var=1e-4`, `tol=1e-4`. These are the same model
class and the same stopping rule as R.

## Exact layer

| Check | NYFED | Simulated |
|---|---:|---:|
| Standardisation (mean / sd over observed values, ddof = 1) | < 1e-10 | < 1e-10 |
| nowcastbox smoother at R's estimates (`Res$A, C, Q, R, Z_0, V_0`): smoothed data and factors | 9.4e-14 | 1.8e-14 |
| loglik(R's estimates) + n/2·log(2π) − R's last printed value | +0.762 | +0.122 |

- `Z_0` and `V_0` are the moments of the state at **t = 0**, one step before the first
  observation. Using them for the first state does not reproduce R; this is tested
  explicitly.
- R does not return the log-likelihood; it prints it every 5 iterations, and the fixture
  script captures it. The printed value is the Gaussian log-likelihood **without the
  constant** −n/2·log(2π), where n is the number of observations. nowcastbox's value at
  R's final estimates lies just above the last printed value (iteration 40 on NYFED), as
  expected for the few remaining iterations.

## Estimators (`reference_divergence`)

| Case | nowcastbox − R log-likelihood | Iterations (nowcastbox) | Horizon nowcasts: max abs diff / sd(y) | Factors |
|---|---:|---:|---:|---|
| NYFED (`init="pca_given"`) | **+20.59** | 20 (R: about 43) | 0.067 | corr Global 0.98, Soft 0.82, Real 0.97, Labor 0.99 |
| NYFED (default `init="pca"`) | **+382.6** | 22 | 0.136 | another local maximum: R's Real factor lies outside its factor space |
| Simulated | **+1.57** | 4 | 0.013 | same space (R² of the rotation > 0.999) |

The first row uses the sequential block start of Bańbura & Modugno (`init="pca_given"`),
the scheme closest to R's. The default multi-start (best of four block orders by the
starting log-likelihood, see [statsmodels](statsmodels.md)) reaches a much higher
likelihood on this four-block specification, with different block factors.

Both log-likelihoods are evaluated by nowcastbox's filter on the same standardised data.
The starting values differ (both are principal components of the spline-filled panel, but
different implementations), and EM is stopped when the relative change falls below 1e-4.
EM is slow near an optimum, so the two runs stop at different points. nowcastbox stops at
the higher likelihood. The nowcast comparison covers the periods after the last GDP
observation (nowcast and forecasts).

R's EM `yfcst` and `xfcst` are not the smoothed signal, or its common component, of the
model returned in `Res`, and could not be reproduced from it. `xfcst` does not even equal
the data in-sample. They are compared only loosely, and
`test_r_yfcst_is_not_the_returned_signal` documents this.

For a stricter EM comparison on an identical likelihood, see
[statsmodels](statsmodels.md).
