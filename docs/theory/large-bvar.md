# Large mixed-frequency Bayesian VAR

`LargeBVAR` implements the "blocking" Bayesian VAR (B-BVAR) of Cimadomo, Giannone, Lenza,
Monti & Sokol (2022). It is a quarterly VAR for monthly and quarterly data, estimated
with the conjugate prior of [the BVAR prior page](bvar-prior.md). Its nowcasts are
forecasts conditional on all the data released at the vintage. The model and its
helpers are in `nowcastbox.models.bvar` and `nowcastbox.models._bvar_blocking`.

## Blocking (stacking)

All frequencies are aligned on the lowest one, the quarter. A monthly variable
$x_{t_m}$ becomes three quarterly variables, one for each month of the quarter:

$$
x^q_{t} = \big(x_{3t-2},\; x_{3t-1},\; x_{3t}\big).
$$

In the library these are named `x[m1]`, `x[m2]` and `x[m3]`. A quarterly variable $y_t$
enters as it is. The blocked vector $Y_t = (y_t', x^{q\,\prime}_t)'$ has
$n = Q + 3M$ entries and follows a quarterly VAR($p$):

$$
Y_t = c + A_1 Y_{t-1} + \dots + A_p Y_{t-p} + \varepsilon_t,
\qquad \varepsilon_t \sim N(0, \Sigma).
$$

Because each month is a separate variable, the model does not need an aggregation rule
(flow, stock, average or Mariano-Murasawa). The VAR learns how the months relate to the
quarterly variables, both through the lag coefficients and through the
contemporaneous covariance $\Sigma$. The cost is size: 50 monthly indicators and GDP
give $n = 151$ variables. The Bayesian shrinkage below is what makes such a VAR
estimable on a few decades of quarterly data.

The series are standardised before blocking (default `standardize=True`), like in the
factor models. Results are reported in the original units of the target.

## Prior and estimation sample

The prior is the Normal-Inverse-Wishart Minnesota prior in the parameterisation of
Giannone, Lenza & Primiceri (2015): $\Psi = \operatorname{diag}(\psi)$ and $d = n + 2$
degrees of freedom. The prior variance of lag $l$ of variable $j$ is
$\lambda^2 / (l^2 \psi_j)$, scaled by $\Sigma$. Two dummy-observation priors are optional:

- the sum-of-coefficients prior (Doan, Litterman & Sims, 1984), tightness $\mu$;
- the dummy-initial-observation prior (Sims, 1993; Sims & Zha, 1998), tightness
  $\delta$.

With `prior="glp"`, $\lambda$ (and $\mu$, $\delta$ when those priors are switched on)
is set to the mode of its posterior, under GLP's Gamma hyperpriors. As in Cimadomo et al.
(2022), the scales $\psi_j$ are fixed at the residual variances of univariate AR(1)
regressions; `estimate_psi=True` estimates them too. The prior mean of the own first
lag is 0 by default (`prior_mean="white_noise"`), which suits the stationary
transformations used in nowcasting panels. For (log-)levels, use `"random_walk"`
together with the two dummy priors, as in Cimadomo et al. (2022).

The closed-form posterior needs a complete data matrix. `LargeBVAR` therefore
estimates on the balanced part of the blocked panel: the longest run of complete
quarters that ends at the last complete quarter. A series that starts late shortens
this sample for all variables, and so does an interior gap in any series. When quarters
with data are left out this way, the fit emits a `DataQualityWarning` naming the series
missing in the quarter just before the sample. The point parameters are the posterior means
$\bar B$ and $\operatorname{E}[\Sigma \mid Y] = \bar\Psi / (\bar d - n - 1)$.

## Ragged edge: conditional forecasts

At a given vintage, the last quarters are only partly observed. For example, the
first two months of an indicator may be released while GDP is not. Given the
parameters, the nowcast is the expectation of the missing entries conditional on the
released ones. This is the conditional forecast of Waggoner & Zha (1999), which
Bańbura, Giannone & Lenza (2015) compute with the Kalman smoother.

The VAR is written in companion form, in the notation of Durbin & Koopman (2012):

$$
\alpha_{t+1} = T\alpha_t + c^\ast + R\eta_t,\qquad
Y_t = Z\alpha_t,\qquad
\alpha_t = (Y_t', \dots, Y_{t-p+1}')',
$$

where $T$ is the companion matrix, $R = (I_n, 0)'$, $Z = (I_n, 0)$, $Q = \Sigma$ and
there is no measurement error ($H = 0$). Unreleased entries are `NaN`, which the
univariate Kalman filter of `nowcastbox.statespace` skips. The VAR is Markov of order
$p$, so the forecast after the last window of $p$ complete quarters ($T_0$) depends on
earlier data only through that window. The smoother therefore starts at $T_0 + 1$, with
the exact initial state

$$
\alpha_{T_0+1} \sim N\Big(\big(c + \textstyle\sum_l A_l Y_{T_0+1-l},\,
Y_{T_0}, \dots\big),\ \operatorname{diag}(\Sigma, 0, \dots, 0)\Big).
$$

The smoothed mean of the target is the backcast, nowcast or forecast (`out_of_sample`).
Its smoothed variance gives the `std` column when no posterior draws are requested.
For periods where the target is observed, `in_sample` holds the one-quarter-ahead VAR
prediction.

The same moments can be written in closed form. From the moving-average representation
$Y_{T_0+i} = \mu_i + \sum_{s \le i} \Phi_{i-s}\varepsilon_{T_0+s}$, the edge is a
Gaussian vector $\mu + Mu$ with $u \sim N(0, I)$. Conditioning on its observed entries
$o$ gives

$$
\operatorname{E}[Y_c \mid Y_o] = \mu_c + M_c M_o'(M_o M_o')^{-1}(y_o - \mu_o),\qquad
\operatorname{Var}[Y_c \mid Y_o] = M_c M_c' - M_c M_o'(M_o M_o')^{-1} M_o M_c'.
$$

`conditional_moments` computes this form, and the test suite checks it against the
Kalman smoother and against a third, direct computation.

## Density nowcasts

Parameter uncertainty is integrated by posterior simulation. For each draw
$(B^{(i)}, \Sigma^{(i)})$ of the Normal-Inverse-Wishart posterior, the target is Gaussian
given the released data, with the conditional mean and variance above. The predictive
distribution is therefore the mixture

$$
p(y_\tau \mid \Omega) \approx \frac{1}{N}\sum_{i=1}^{N}
N\big(m_\tau(B^{(i)}, \Sigma^{(i)}),\ s^2_\tau(B^{(i)}, \Sigma^{(i)})\big).
$$

This is the Rao-Blackwellised version of drawing a conditional path for each parameter
draw: the conditional simulation step is integrated analytically. The mixture is a
`NowcastDistribution` with one component per draw, so the CRPS, log score, PIT and fan
charts of `nowcastbox.evaluation.scoring` apply unchanged (see
[Density nowcasts and scoring](density-scoring.md)). The point nowcast stays the
conditional forecast at the posterior mean. The mean of the mixture differs from it
slightly, because the forecast is non-linear in $B$ beyond one step.

## News

With fixed parameters the model is linear and Gaussian, so the news decomposition of
Bańbura & Modugno (2014) applies (see [News decomposition](news.md)). `LargeBVARResults.news` converts both vintages to the
blocked quarterly representation and runs `nowcastbox.news` on the state-space model
above. That model starts $p$ quarters into the estimation sample, so revisions inside
the conditioning window are counted. The releases are then mapped back to the original
series and months: a release of `ip[m2]` in 2024Q2 is reported as `ip`, 2024-05. Level
contributions work the same way. The nowcast tracker is not available, because it
builds monthly-grid vintages internally.

## Validation

- **Blocking:** blocking followed by unblocking is the identity; layouts and errors are
  tested for unsupported frequencies.
- **Conditional forecasts:** the Kalman smoother, the closed form and the fitted
  model's nowcast all match a direct computation from the companion-form moments
  ($\operatorname{Cov}(\alpha_i, \alpha_j) = T^{i-j}V_j$) to $10^{-9}$. The news
  model, which starts at the beginning of the sample, gives the same nowcasts as the
  edge smoother.
- **Parameter recovery:** on 600 quarters simulated from a known blocked VAR(2) with
  $n = 7$, the posterior means recover $A_1, A_2$ (maximum absolute error < 0.15, RMSE
  < 0.05), $c$ and $\Sigma$, with a loose prior and with GLP's choice of $\lambda$.
- **Calibration:** over 300 Monte Carlo replications with $T = 120$ quarters, the 90%
  posterior intervals of the nowcast cover the realised value 89.7% of the time, and the
  PIT values are uniform (Kolmogorov-Smirnov).
- **Accuracy:** on 20 simulated mixed-frequency DFM panels (10 indicators, 30 years,
  ragged edge of 0 to 2 months), the RMSE of the GDP nowcast was 1.27 for `LargeBVAR`
  (1 lag) and 1.29 (2 lags). `MixedFreqDFM`, which is the true model here, gave 1.00,
  and the historical mean gave 2.78.
- **Run time:** with $N = 50$ monthly series ($n = 151$), 80 quarters and one BLAS
  thread, a GLP fit took 0.4–0.5 s and 200 posterior draws took 0.6–1.0 s. A news
  decomposition took 2 s with 1 lag and 10 s with 2 lags.
- **Pseudo real time:** fits at a vintage do not depend on later data, and
  `PseudoRealTimeBacktest` reproduces manual fits on the `as_of` vintages.

## References

- Bańbura, M., Giannone, D., & Lenza, M. (2015). Conditional forecasts and scenario
  analysis with vector autoregressions for large cross-sections. *International Journal
  of Forecasting*, 31(3), 739–756.
- Bańbura, M., Giannone, D., & Reichlin, L. (2010). Large Bayesian vector auto
  regressions. *Journal of Applied Econometrics*, 25(1), 71–92.
- Bańbura, M., & Modugno, M. (2014). Maximum likelihood estimation of factor models on
  datasets with arbitrary pattern of missing data. *Journal of Applied Econometrics*,
  29(1), 133–160.
- Cimadomo, J., Giannone, D., Lenza, M., Monti, F., & Sokol, A. (2022). Nowcasting with
  large Bayesian vector autoregressions. *Journal of Econometrics*, 231(2), 500–519.
- Doan, T., Litterman, R., & Sims, C. (1984). Forecasting and conditional projection
  using realistic prior distributions. *Econometric Reviews*, 3(1), 1–100.
- Durbin, J., & Koopman, S. J. (2012). *Time Series Analysis by State Space Methods*
  (2nd ed.). Oxford University Press.
- Giannone, D., Lenza, M., & Primiceri, G. E. (2015). Prior selection for vector
  autoregressions. *Review of Economics and Statistics*, 97(2), 436–451.
- Sims, C. A., & Zha, T. (1998). Bayesian methods for dynamic multivariate models.
  *International Economic Review*, 39(4), 949–968.
- Waggoner, D. F., & Zha, T. (1999). Conditional forecasts in dynamic multivariate
  models. *Review of Economics and Statistics*, 81(4), 639–651.
