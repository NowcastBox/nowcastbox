# BVAR prior and hyperparameter selection

The large Bayesian VAR rests on a Normal-Inverse-Wishart prior (Minnesota,
sum-of-coefficients and dummy-initial-observation components) whose tightness is chosen
by maximising the marginal likelihood plus a hyperprior (Giannone, Lenza & Primiceri,
2015). The building blocks live in `nowcastbox.models._bvar_prior`. They work on complete
numeric arrays, and you can use them directly to fit a single-frequency BVAR. See the
[theory page](../../theory/bvar-prior.md) for the formulas.

## A VAR in stacked form

```python
import numpy as np
from nowcastbox.models._bvar_prior import VARSystem

rng = np.random.default_rng(0)
y = np.cumsum(rng.standard_normal((120, 4)), axis=0)   # 120 periods, 4 variables (levels)
system = VARSystem.from_array(y, lags=4)                # Y (116 x 4), X (116 x 17)
system.ar_residual_variances(1)                         # AR(1) residual variances (psi)
```

The array must have no missing values. The ragged edge is handled by the
mixed-frequency model, not here.

## Choosing the hyperparameters

```python
from nowcastbox.models._bvar_prior import PriorSettings, select_hyperparameters

settings = PriorSettings(prior_mean="random_walk")      # "white_noise" for growth rates
sel = select_hyperparameters(system, settings)          # lambda, mu, delta and psi
sel.hyperparameters.lambda_, sel.hyperparameters.mu, sel.hyperparameters.delta
sel.log_ml, sel.log_posterior
sel.names, sel.standard_errors                          # log-scale, from the inverse Hessian
```

| Argument | Default | Meaning |
|---|---|---|
| `settings.prior_mean` | `"random_walk"` | prior mean of the own first lag: `"random_walk"` (1), `"white_noise"` (0), a number or one value per variable |
| `settings.lag_decay` | `2.0` | exponent $\kappa$ of the lag decay $1/l^\kappa$ |
| `settings.intercept_variance` | `1e6` | prior variance factor of the constant (diffuse) |
| `settings.dof` | $n + 2$ | prior degrees of freedom of $\Sigma$ |
| `estimate` | all hyperparameters in use | subset of `"lambda"`, `"mu"`, `"delta"`, `"psi"` to optimise; the others stay at `start` |
| `start` | $\lambda = 0.2$, $\mu = \delta = 1$, AR(1) $\psi$ | starting point and fixed values |
| `sum_of_coefficients`, `initial_observation` | `True` | include the sum-of-coefficients / dummy-initial-observation priors |
| `hyperpriors` | GLP's | `GLPHyperpriors(lambda_=..., mu=..., delta=..., psi=...)`; `None` entries are flat (all `None` = maximum marginal likelihood) |
| `hessian` | `True` | return the inverse Hessian on the log scale |

The optimiser uses the analytic gradient of the log marginal likelihood. With
$n = 100$ variables and three lags ($k = 301$ coefficients per equation), selecting
$\lambda$, $\mu$, $\delta$ and the 100 scales $\psi_j$ takes about one second with a
few BLAS threads (`OPENBLAS_NUM_THREADS=4`); on many-core machines the default thread
count makes these small factorisations much slower (tens of seconds).
If the optimiser does not converge, a `ConvergenceWarning` is issued.

To fix the scales at the AR residual variances, which is common practice and faster,
estimate only the tightness parameters:

```python
sel = select_hyperparameters(system, settings, estimate=("lambda", "mu", "delta"))
```

## Posterior and draws

```python
from nowcastbox.models._bvar_prior import posterior, log_marginal_likelihood

post = posterior(system, sel.hyperparameters, settings)
post.mean            # posterior mean of B (k x n): constant, lag 1, ..., lag p
post.sigma_mean      # posterior mean of Sigma
post.omega           # posterior Omega-bar (k x k)
B, Sigma = post.draw(2000, rng=1)     # exact NIW draws: (2000, k, n), (2000, n, n)

log_marginal_likelihood(system, sel.hyperparameters, settings)
```

## Lower-level pieces

| Function | Returns |
|---|---|
| `minnesota_prior(system, hyper, settings)` | `NIWPrior` ($b$, diagonal $\Omega$, $\Psi$, $d$) |
| `minnesota_dummies`, `sum_of_coefficients_dummies`, `dummy_initial_observation_dummies`, `prior_dummies` | dummy-observation rows $(Y_d, X_d)$ |
| `implied_prior_from_dummies(Yd, Xd)` | prior mean, $\Omega$ and $\Psi$ encoded by dummy rows |
| `niw_posterior(Y, X, prior)` | `NIWPosterior` and $\log p(Y)$ for any NIW prior with diagonal $\Omega$ |
| `log_marginal_likelihood_gradient(system, hyper, settings)` | value and gradient in $\log\lambda, \log\psi, \log\mu, \log\delta$ |
| `log_hyperprior`, `log_hyperprior_gradient` | hyperprior density and gradient |
