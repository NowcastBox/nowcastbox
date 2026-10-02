# Density nowcasts and scoring

## Predictive distribution

Given the parameters $\hat\theta$, the Kalman smoother delivers the Gaussian predictive
distribution of the target in period $\tau$,

$$
y_\tau \mid \Omega, \hat\theta \sim N\!\left(\hat y_\tau,\; z_\tau V_\tau z_\tau' + h_\tau\right),
$$

where $z_\tau$ is the row of the observation equation of the (aggregated) target and
$V_\tau$ the smoothed state variance — the **filtering uncertainty**. For the two-step
model the bridge prediction variance $\beta' V_\tau \beta + \sigma^2_e$ plays this role.

**Parameter uncertainty** is added by bootstrap (Stoffer & Wall, 1991; Pfeffermann &
Tiller, 2005; Rodríguez & Ruiz, 2012). For $b = 1, \dots, B$:

1. *parametric*: simulate a panel from the fitted model with the observed missing
   pattern, or *block*: resample whole target periods of the panel with a moving-block
   bootstrap (Künsch, 1989);
2. re-estimate $\theta^\ast_b$ on the bootstrap panel;
3. filter the **original** data with $\theta^\ast_b$ to get $\mu_b$ and $\sigma^2_b$.

The predictive density is the mixture (Hamilton, 1986)

$$
p(y_\tau \mid \Omega) \approx \frac1B \sum_{b=1}^B \phi\!\left(y_\tau; \mu_b, \sigma^2_b\right),
\qquad
\operatorname{Var}(y_\tau \mid \Omega) = \underbrace{\frac1B \sum_b \sigma^2_b}_{\text{filtering}}
+ \underbrace{\frac1B \sum_b (\mu_b - \bar\mu)^2}_{\text{parameters}} ,
$$

optionally re-centred at the point nowcast. Quantiles are obtained by inverting the
mixture CDF numerically.

The [large Bayesian VAR](large-bvar.md#density-nowcasts) uses the same mixture form, with
draws from the Normal-Inverse-Wishart posterior of the VAR coefficients and covariance in
place of the bootstrap replications.

## Proper scoring rules

A scoring rule $S(F, y)$ is *proper* if the expected score is optimised by the true
distribution (Gneiting & Raftery, 2007). Scores are negatively oriented (lower is
better).

**Continuous ranked probability score**

$$
\operatorname{CRPS}(F, y) = \int_{-\infty}^{\infty} \left(F(x) - \mathbb 1\{x \ge y\}\right)^2 dx
= \mathbb E_F|X - y| - \tfrac12 \mathbb E_F|X - X'| .
$$

For $F = N(\mu, \sigma^2)$ and $z = (y - \mu)/\sigma$ (Gneiting et al., 2005):

$$
\operatorname{CRPS} = \sigma \left[ z\,(2\Phi(z) - 1) + 2\phi(z) - \tfrac{1}{\sqrt\pi} \right].
$$

Gaussian mixtures have a closed form too (Grimit et al., 2006), and samples use the
energy form (the "fair" version of Ferro, 2014, is available).

**Log score** $\operatorname{LogS}(F, y) = -\ln p(y)$.

**Interval score** for a central $(1-\alpha)$ interval $[l, u]$:

$$
\operatorname{IS}_\alpha(l, u; y) = (u - l) + \frac{2}{\alpha}(l - y)\mathbb 1\{y < l\}
+ \frac{2}{\alpha}(y - u)\mathbb 1\{y > u\}.
$$

**Quantile (pinball) score** for the $\tau$-quantile $q$:
$\operatorname{QS}_\tau(q, y) = 2\left(\mathbb 1\{y < q\} - \tau\right)(q - y)$; weighted
averages over quantile levels emphasise the centre or the tails (Gneiting & Ranjan,
2011).

## Calibration

**PIT.** If $F_t$ is the true conditional distribution, $u_t = F_t(y_t)$ is iid uniform
(Dawid, 1984; Diebold, Gunther & Tay, 1998). Uniformity is tested with the
Kolmogorov-Smirnov test, and jointly with independence by **Berkowitz (2001)**: with
$z_t = \Phi^{-1}(u_t)$ fit $z_t - \mu = \rho(z_{t-1} - \mu) + \varepsilon_t$,
$\varepsilon_t \sim N(0, \sigma^2)$, and compare with $\mu = 0, \rho = 0, \sigma = 1$:

$$
LR_3 = -2\left[\ell(0, 1, 0) - \ell(\hat\mu, \hat\sigma^2, \hat\rho)\right] \sim \chi^2(3).
$$

**Interval coverage (Christoffersen, 1998).** With hits $I_t = \mathbb 1\{y_t \in [l_t, u_t]\}$
and nominal coverage $p$:

$$
LR_{uc} = -2 \ln \frac{p^{n_1}(1-p)^{n_0}}{\hat\pi^{n_1}(1-\hat\pi)^{n_0}} \sim \chi^2(1),
$$

$LR_{ind}$ tests a first-order Markov chain against independence ($\chi^2(1)$) and
$LR_{cc} = LR_{uc} + LR_{ind} \sim \chi^2(2)$.
