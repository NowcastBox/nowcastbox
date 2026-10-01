# EM algorithm

`MixedFreqDFM` estimates the dynamic factor model by maximum likelihood with the EM
algorithm (Dempster, Laird & Rubin, 1977; Shumway & Stoffer, 1982; Watson & Engle,
1983), in the form of Bańbura & Modugno (2014) for an **arbitrary pattern of missing
data**, blocks of factors, AR(1) idiosyncratic components and aggregation restrictions
for lower-frequency series.

## The idea

With the states known, the parameters $\theta = (\Lambda, A, Q, \Psi, \rho, \dots)$ would
be estimated by least squares. EM iterates

- **E-step**: compute the expected complete-data log-likelihood
  $\mathcal Q(\theta; \theta^{(k)}) = \mathbb E_{\theta^{(k)}}[\log p(Y, \alpha; \theta) \mid \Omega]$,
  which only needs the smoothed moments $\mathbb E[\alpha_t \mid \Omega]$,
  $\mathbb E[\alpha_t \alpha_t' \mid \Omega]$ and $\mathbb E[\alpha_t \alpha_{t-1}' \mid \Omega]$
  from the [Kalman smoother](kalman-filter-smoother.md) (or the exact structured
  smoother, I2);
- **M-step**: maximise $\mathcal Q$ in closed form (regressions on smoothed moments).

Each iteration does not decrease the likelihood (up to the treatment of the initial
state below).

Write $\hat f_t = \mathbb E[f_t \mid \Omega]$ and
$\widehat{f_t f_s'} = \mathbb E[f_t f_s' \mid \Omega] = \hat f_t \hat f_s' + \operatorname{Cov}(f_t, f_s \mid \Omega)$.

## M-step

**Factor dynamics.** For each block with VAR($p$) and stacked lags
$F_{t-1} = (f_{t-1}', \dots, f_{t-p}')'$:

$$
\hat A = \Big(\sum_t \widehat{f_t F_{t-1}'}\Big)\Big(\sum_t \widehat{F_{t-1} F_{t-1}'}\Big)^{-1},
\qquad
\hat Q = \frac1T \Big(\sum_t \widehat{f_t f_t'} - \hat A \sum_t \widehat{F_{t-1} f_t'}\Big).
$$

**Loadings with missing data.** Let $W_t$ be the diagonal selection matrix of the
observed series at $t$. For monthly series with white-noise idiosyncratic components
(Bańbura & Modugno, 2014, eq. 9),

$$
\operatorname{vec}(\hat\Lambda) =
\Big(\sum_t \widehat{f_t f_t'} \otimes W_t\Big)^{-1}
\operatorname{vec}\Big(\sum_t W_t x_t \hat f_t'\Big),
$$

which reduces to series-by-series regressions on the periods where each series is
observed. With AR(1) idiosyncratic states the expected idiosyncratic component is
subtracted: $\sum_t W_t (x_t \hat f_t' - \widehat{e_t f_t'})$.

**Idiosyncratic variances** (white noise):

$$
\hat\Psi = \operatorname{diag}\Big\{ \frac1T \sum_t \Big[
W_t x_t x_t' W_t - W_t x_t \hat f_t' \hat\Lambda' W_t - W_t \hat\Lambda \hat f_t x_t' W_t
+ W_t \hat\Lambda \widehat{f_t f_t'} \hat\Lambda' W_t + (I - W_t)\Psi^{(k)}(I - W_t)
\Big]\Big\},
$$

where the last term keeps the current variance for unobserved entries.

**AR(1) idiosyncratic components** $e_{i,t} = \rho_i e_{i,t-1} + \nu_{i,t}$:

$$
\hat\rho_i = \frac{\sum_t \widehat{e_{i,t} e_{i,t-1}}}{\sum_t \widehat{e_{i,t-1}^2}}, \qquad
\hat\sigma^2_i = \frac1T \Big(\sum_t \widehat{e_{i,t}^2} - \hat\rho_i \sum_t \widehat{e_{i,t-1} e_{i,t}}\Big).
$$

## Aggregation restrictions

A quarterly series loads on $f_t, \dots, f_{t-L}$ with loadings $\lambda_0, \dots, \lambda_L$
proportional to the aggregation weights $w$ (see [Temporal aggregation](mariano-murasawa.md)):
$w_s \lambda_0 - w_0 \lambda_s = 0$ for $s = 1, \dots, L$, i.e. $R \operatorname{vec}(\lambda) = q$
with $q = 0$. The restricted least-squares update is

$$
\operatorname{vec}(\hat\lambda_r) = \operatorname{vec}(\hat\lambda_u)
+ M^{-1} R' \left(R M^{-1} R'\right)^{-1}\big(q - R \operatorname{vec}(\hat\lambda_u)\big),
\qquad M = \sum_t \widehat{\tilde f_t \tilde f_t'} \otimes W_t ,
$$

with $\tilde f_t = (f_t', \dots, f_{t-L}')'$ and $\hat\lambda_u$ the unrestricted
estimate. `preprocessing.loading_constraints` builds $R$ and $q$ for any weights, so the
same step handles quarterly, annual, flow, stock and average series (I1).

## Starting values and convergence

The EM starts from principal components of the panel with missing values filled by
cubic splines (interior) and moving medians (edges), a VAR on those factors, and OLS
loadings (Bańbura & Modugno, 2014). Warm starts from previous results are possible
(`init=results`).

The algorithm stops when the relative change of the log-likelihood

$$
\frac{\ell_k - \ell_{k-1}}{\tfrac12(|\ell_k| + |\ell_{k-1}|)} < \text{tol}
$$

(Doz, Giannone & Reichlin, 2012), or after `max_iter` iterations with a
`ConvergenceWarning`.

!!! note "Initial-state term"
    As in Bańbura & Modugno (2014), the M-step of $A$ and $Q$ ignores the contribution
    of the initial-state distribution (which depends on $A$ and $Q$ through the
    stationary covariance). Monotonicity therefore holds up to that term; in short
    samples tiny decreases (relative $10^{-7}$–$10^{-5}$) can occur and are reported in
    `results.info["n_loglikelihood_decreases"]`.

## Robust extensions (I3)

**Student-t idiosyncratic errors.** $\varepsilon_{i,t} = \lambda_{i,t}^{-1/2} u_{i,t}$ with
$u_{i,t} \sim N(0, \psi_i)$ and $\lambda_{i,t} \sim \text{Gamma}(\nu/2, \nu/2)$ gives a
Student-t with $\nu$ degrees of freedom (Andrews & Mallows, 1974). The ECM of Lange,
Little & Taylor (1989) and Liu & Rubin (1995) alternates

$$
\mathbb E[\lambda_{i,t} \mid \cdot] = \frac{\nu + 1}{\nu + \delta^2_{i,t}}, \qquad
\delta^2_{i,t} = \frac{\mathbb E[(x_{i,t} - \lambda_i' f_t)^2 \mid \Omega]}{\psi_i},
$$

a weighted Kalman smoother / M-step (observation variances $\psi_i / \mathbb E[\lambda_{i,t}]$),
and a one-dimensional maximisation over $\nu$. The variational treatment
$q(\alpha) q(\lambda)$ (Beal & Ghahramani, 2003) makes the reported objective a lower
bound of the Student-t log-likelihood that increases monotonically.

**Automatic outliers.** Observations with standardised one-step-ahead prediction errors
$|v_{t,i}| / \sqrt{F_{t,i}}$ above a threshold (Durbin & Koopman, 2012, §2.12) are set to
missing and the EM re-run until the flags are stable.

**Pandemic dummies.** With `covid="dummy"` the factor VAR gets an impulse dummy for each
period of the window, $f_t = \sum_i A_i f_{t-i} + \gamma_t + B u_t$; equivalently, the
transitions inside the window do not enter the M-step of $A$ and $Q$, which protects
the dynamics from the swing while the factors still track the data.

## Time-varying long-run mean (I4)

Following Antolin-Diaz, Drechsel & Petrella (2017), the target gets a random-walk mean,

$$
x_{\text{target},t} = \mu_t + \lambda' f_t + \varepsilon_t, \qquad \mu_t = \mu_{t-1} + \eta_t,
$$

one extra state with a unit loading (for a quarterly target the loading follows the
aggregation weights). $\sigma^2_\eta$ is updated in the M-step as
$\frac1T \sum_t \mathbb E[(\mu_t - \mu_{t-1})^2 \mid \Omega]$ (Shumway & Stoffer, 1982)
or fixed by the user to avoid the pile-up of its estimate at zero (Stock & Watson, 1998).
