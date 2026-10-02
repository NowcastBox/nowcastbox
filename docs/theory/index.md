# Theory

This section derives the methods implemented in NowcastBox, with the notation used in
the code and in the docstrings. Every implementation was written from the papers listed
in the [references](references.md) (the clean-room record per module is kept in
`desenvolvimento/FONTES_POR_MODULO.md`).

| Page | Content |
|---|---|
| [Dynamic factor models](dfm.md) | approximate DFM, static and dynamic factors, state-space form, two-step estimator, blocks |
| [Kalman filter and smoother](kalman-filter-smoother.md) | filter with missing data, univariate treatment, collapsing, smoother, lag-one covariances, structured smoother |
| [EM algorithm](em-algorithm.md) | E- and M-steps with missing data, AR(1) idiosyncratic components, aggregation restrictions, Student-t, long-run mean |
| [Temporal aggregation](mariano-murasawa.md) | Mariano-Murasawa and general flow / stock / average restrictions |
| [Bai-Ng criteria](bai-ng.md) | number of factors and of primitive shocks |
| [News decomposition](news.md) | revision of a nowcast as weighted news, revisions, re-estimation |
| [Density nowcasts and scoring](density-scoring.md) | predictive distributions, bootstrap, proper scores, calibration tests |
| [Forecast evaluation](forecast-evaluation.md) | pseudo real-time design, RMSFE by horizon, DM, GW, MCS |
| [Pre-selection](preselection.md) | t-stat, SIS and LARS rankings of the indicators, aggregated score, leads and lags |
| [Specification search](specification-search.md) | random and grid search of specifications, weighted score by horizon and metric, Covid robustness |
| [Combination of bridge equations](bridge-combination.md) | thick modelling with small bridge equations, batched OLS, mean/median/inverse-MSE combination, trimming, BVAR extrapolation of the indicators |
| [BVAR priors and hierarchical selection](bvar-prior.md) | Normal-Inverse-Wishart Minnesota prior, sum-of-coefficients and dummy-initial-observation priors, closed-form marginal likelihood, GLP hyperpriors |
| [Large mixed-frequency Bayesian VAR](large-bvar.md) | blocking of monthly series into quarterly variables, conditional forecasts for the ragged edge, posterior-mixture densities, news |

## Notation

| Symbol | Meaning | In the code |
|---|---|---|
| $t = 1, \dots, T$ | base periods (months) | `data.index` |
| $x_t$ ($N \times 1$) | standardised panel | `data.standardize()` |
| $f_t$ ($r \times 1$) | common factors | `results.factors` |
| $\Lambda$ ($N \times r$) | loadings | `results.loadings` |
| $\varepsilon_t$, $\Psi$ | idiosyncratic components and their variances | `results.idiosyncratic_variance` |
| $A_1, \dots, A_p$ | factor VAR matrices | `results.transition`, `transition_matrices()` |
| $q$, $B$ | number of dynamic shocks and their loadings | `results.n_shocks`, `results.shock_loadings` |
| $\alpha_t$ | state vector | `results.smoothed_state` |
| $T, Z, R, Q, H, c, d$ | state-space matrices (Durbin & Koopman, 2012) | `StateSpace(transition, design, state_cov, obs_cov, ...)` |
| $\Omega_v$ | information set of vintage $v$ | a `MixedFrequencyData` vintage |
| $y^Q_\tau$ | quarterly target in quarter $\tau$ | `results.nowcast` |

The state-space form follows Durbin & Koopman (2012):

$$
y_t = Z \alpha_t + d + \varepsilon_t, \quad \varepsilon_t \sim N(0, H), \qquad
\alpha_{t+1} = T \alpha_t + c + R \eta_t, \quad \eta_t \sim N(0, Q),
$$

with $\alpha_1 \sim N(a_1, P_1)$ and missing elements of $y_t$ allowed.
