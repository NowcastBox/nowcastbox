# Kalman filter and smoother

The model is

$$
y_t = Z \alpha_t + d + \varepsilon_t, \quad \varepsilon_t \sim N(0, H), \qquad
\alpha_{t+1} = T \alpha_t + c + R \eta_t, \quad \eta_t \sim N(0, Q),
\qquad \alpha_1 \sim N(a_1, P_1).
$$

The notation and recursions follow Durbin & Koopman (2012, ch. 4 and 6).

## Filter with missing observations

Let $W_t$ select the observed elements of $y_t$ (a matrix of rows of the identity). With
$y^\ast_t = W_t y_t$, $Z^\ast_t = W_t Z$, $H^\ast_t = W_t H W_t'$, the filter is

$$
\begin{aligned}
v_t &= y^\ast_t - Z^\ast_t a_t - W_t d, & F_t &= Z^\ast_t P_t Z^{\ast\prime}_t + H^\ast_t, \\
K_t &= T P_t Z^{\ast\prime}_t F_t^{-1}, & L_t &= T - K_t Z^\ast_t, \\
a_{t+1} &= T a_t + c + K_t v_t, & P_{t+1} &= T P_t L_t' + R Q R',
\end{aligned}
$$

where $a_t = \mathbb E[\alpha_t \mid y_1, \dots, y_{t-1}]$ and $P_t$ its variance. When
nothing is observed at $t$, $v_t$ is empty, $K_t = 0$ and the step is a pure prediction.
The Gaussian log-likelihood is the prediction-error decomposition

$$
\log L = -\frac12 \sum_t \left( p_t \log 2\pi + \log |F_t| + v_t' F_t^{-1} v_t \right),
$$

with $p_t$ the number of observations at $t$.

**Initialisation.** For a stationary transition $P_1$ solves the Lyapunov equation
$P_1 = T P_1 T' + R Q R'$ (`stationary_initial_cov`); for non-stationary states (the
random-walk long-run mean) an approximate diffuse prior $P_1 = \kappa I$ with large
$\kappa$ is used (`approximate_diffuse_initial_cov`).

## Univariate treatment

With a diagonal $H = \operatorname{diag}(\sigma^2_1, \dots, \sigma^2_N)$ the observations
of period $t$ can be processed **one at a time** (Koopman & Durbin, 2000). For the
observed series $i = 1, \dots, p_t$ with rows $z_i$ of $Z$:

$$
\begin{aligned}
v_{t,i} &= y_{t,i} - z_i a_{t,i} - d_i, & F_{t,i} &= z_i P_{t,i} z_i' + \sigma^2_i, \\
K_{t,i} &= P_{t,i} z_i' / F_{t,i}, & & \\
a_{t,i+1} &= a_{t,i} + K_{t,i} v_{t,i}, & P_{t,i+1} &= P_{t,i} - K_{t,i} F_{t,i} K_{t,i}',
\end{aligned}
$$

followed by the transition $a_{t+1,1} = T a_{t,p_t+1} + c$,
$P_{t+1,1} = T P_{t,p_t+1} T' + RQR'$. Only scalars are inverted, so the cost grows
linearly in the number of observations and missing values cost nothing. The
log-likelihood is $-\tfrac12 \sum_{t,i} (\log 2\pi + \log F_{t,i} + v_{t,i}^2/F_{t,i})$.
This is `filter_method="univariate"` (Numba kernels); a non-diagonal $H$ uses the
multivariate recursions.

## Collapsing the observations

When $N$ is much larger than the state, the observation vector can be **collapsed**
(Jungbacker & Koopman, 2015). With $Z_t^\ast$ of full column rank and
$A_{L,t} = (Z^{\ast\prime}_t H^{\ast-1}_t Z^\ast_t)^{-1} Z^{\ast\prime}_t H^{\ast-1}_t$, the
transformed observation $y^L_t = A_{L,t} y^\ast_t$ satisfies

$$
y^L_t = \alpha_t + \varepsilon^L_t, \qquad
\varepsilon^L_t \sim N\!\big(0, (Z^{\ast\prime}_t H^{\ast-1}_t Z^\ast_t)^{-1}\big),
$$

and the remaining $p_t - m$ directions do not depend on the state. Filtering $y^L_t$
gives the same smoothed states; the log-likelihood adds a term computed from the
residuals $e_t = y^\ast_t - Z^\ast_t \hat\alpha^{GLS}_t$. Available as
`collapse=True` in `kalman_filter`, `kalman_smoother` and `TwoStepDFM`.

## Fixed-interval smoother

The backward recursions (de Jong, 1989; Durbin & Koopman, 2012, §4.4), for
$t = T, \dots, 1$ with $r_T = 0$, $N_T = 0$:

$$
\begin{aligned}
r_{t-1} &= Z^{\ast\prime}_t F_t^{-1} v_t + L_t' r_t, &
N_{t-1} &= Z^{\ast\prime}_t F_t^{-1} Z^\ast_t + L_t' N_t L_t, \\
\hat\alpha_t &= a_t + P_t r_{t-1}, & V_t &= P_t - P_t N_{t-1} P_t ,
\end{aligned}
$$

give $\hat\alpha_t = \mathbb E[\alpha_t \mid Y_T]$ and $V_t = \operatorname{Var}(\alpha_t \mid Y_T)$
without inverting $P_t$. The EM also needs the **lag-one covariances**

$$
\operatorname{Cov}(\alpha_t, \alpha_{t+1} \mid Y_T) = P_t L_t' (I - N_t P_{t+1}) ,
$$

(Durbin & Koopman, 2012, eq. 4.69). The univariate treatment has the analogous
recursions over the observations of each period.

Smoothed signals and their variances follow: $\hat y_t = Z \hat\alpha_t + d$ and
$\operatorname{Var}(Z\alpha_t \mid Y_T) = Z V_t Z'$; the nowcast of an aggregated target
is the signal of its row of $Z$ in the state.

## The structured smoother (innovation I2)

In a dynamic factor model with AR(1) idiosyncratic components most of the state is
made of **private chains**: the idiosyncratic states of series $i$ appear only in the
observation of series $i$ and in their own AR(1) transition. Stacking the whole path
$\alpha = (\alpha_1', \dots, \alpha_T')'$, the posterior is Gaussian with precision

$$
\Omega = \Omega_{\text{prior}} + \sum_t Z_t' H_t^{-1} Z_t ,
$$

which is sparse: banded within each private chain, with dense coupling only through the
common (factor) block (Rue & Held, 2005; Chan & Jeliazkov, 2009). Ordering the private
groups $G$ before the common block $F$,

$$
\Omega = \begin{bmatrix} \Omega_{GG} & \Omega_{GF} \\ \Omega_{FG} & \Omega_{FF} \end{bmatrix},
\quad
\Sigma_{FF} = (\Omega_{FF} - \Omega_{FG}\Omega_{GG}^{-1}\Omega_{GF})^{-1},
\quad
\Sigma_{GF} = -\Omega_{GG}^{-1} \Omega_{GF} \Sigma_{FF},
$$

where $\Omega_{GG}$ is block diagonal with banded blocks, factored by banded Cholesky.
The required elements of $\Sigma_{GG} = \Omega_{GG}^{-1} - \Sigma_{GF}\Omega_{FG}\Omega_{GG}^{-1}$
— only those inside the band, which is all the EM needs — come from the selected-inverse
recursion of Takahashi, Fagan & Chin (1973). The log-likelihood follows from
$p(y) = p(y \mid \alpha) p(\alpha) / p(\alpha \mid y)$ evaluated at the posterior mean.

The result is **exact** (equal to the Kalman smoother to rounding). The dense smoother
costs $O(T m^3)$ for a state of dimension $m$, and $m$ grows with the number of series
(one or more idiosyncratic states each); the structured smoother's cost grows only
linearly in the number of series, $O(N T^2 w) + O((T w)^3)$ for a common block of width
$w$ — much cheaper for large panels, but not for very long samples.
`filter_method="auto"` picks it when a cost model predicts it is cheaper; see
[Performance](../user-guide/performance.md).
