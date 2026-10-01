# MixedFreqDFM vs statsmodels `DynamicFactorMQ` (NY Fed-like specification)

Tests: `tests/reference_validation/test_statsmodels_dfmq.py` (plus the one-block
comparison in `tests/models/test_em_reference.py`). Fixture:
`scripts/reference_fixtures/statsmodels_dfmq.py` (statsmodels 0.14.6).

**Specification.** The NYFED panel is transformed by `Bpanel`, standardised once over the
observed values (ddof = 1), and restricted to 1985-02 to 2017-01. It has four blocks
(Global, Soft, Real, Labor) with one factor each, block-diagonal VAR(1), AR(1)
idiosyncratic components (monthly and quarterly), the Mariano-Murasawa restriction and
**no measurement noise**:

```python
DynamicFactorMQ(z_monthly, endog_quarterly=z_quarterly, factors=blocks,
                factor_orders=1, idiosyncratic_ar1=True, standardize=False).fit_em(tolerance=1e-6)
MixedFreqDFM(n_factors=1, factor_lags=1, blocks=blocks, obs_noise_var=0.0, tol=1e-6)
```

## Results

| Quantity | statsmodels | nowcastbox |
|---|---:|---:|
| Log-likelihood at statsmodels' estimates | −9347.6357 | **−9347.6357** (rel. diff ~1e-12) |
| Smoothed block factors and GDP signal at those estimates | | max abs diff < 1e-6 |
| EM warm-started at statsmodels' estimates (5 iterations) | | monotone, +0.018 |
| Log-likelihood at the default starting values | −9721.11 | **−9717.41** (`init="pca"`; sequential start: −9994.30) |
| Log-likelihood at convergence | −9347.64 | **−9393.68** (sequential start: −9765.28) |
| EM iterations / fit time (1 BLAS thread) | 71 / 7.2 s | 121 / 7.8 s (sequential: 88 / 6.2 s) |

- **Same model.** The likelihood functions coincide, and so do the smoothed quantities at
  equal parameters. statsmodels' `fit_em(...)` results hold the smoothed states of the
  last E-step, which are those of the *previous* parameters. The fixture therefore uses
  `model.smooth(params)`. Its reported `llf` also differs from `loglike(params)`
  (lacuna 8).
- **Correct EM steps.** Started at statsmodels' optimum, nowcastbox's EM keeps increasing
  the likelihood, by tiny monotone steps.
- **Local optima and the start (`reference_divergence`).** The block-sequential principal
  components of Bańbura & Modugno (2014) (`init="pca_given"`) start 273 log-likelihood
  points below statsmodels', and EM converges to a **different local maximum, 418 points
  lower** (600 iterations without a tolerance stay at −9765.24); R's EM lands in the
  same region (see [EM](em.md)). Since the wave-3 integration the default `init="pca"`
  is a multi-start: it computes the block components in every order of
  `BLOCK_ORDERS` (given, reversed, smaller blocks first, independent), evaluates the
  log-likelihood of each start (one Kalman filter pass) and starts from the best — here
  the reversed order, at −9717.41, slightly *above* statsmodels' start. EM then ends at
  −9393.68, **46 points** below statsmodels' optimum (another local maximum of the same
  likelihood). `results.info["initialization"]` records the candidates.
