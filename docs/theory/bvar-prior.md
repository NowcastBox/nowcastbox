# Bayesian VAR priors and their hierarchical choice

The large Bayesian VAR used for nowcasting (Cimadomo, Giannone, Lenza, Monti & Sokol,
2022) rests on a conjugate Normal-Inverse-Wishart (NIW) prior whose tightness is chosen
from the data, as in Giannone, Lenza & Primiceri (2015, GLP). This page documents the
prior machinery of `nowcastbox.models._bvar_prior`; the mixed-frequency model built on
it is described separately.

## The VAR in stacked form

A VAR($p$) for the $n$-vector $y_t$,
$y_t = c + B_1 y_{t-1} + \dots + B_p y_{t-p} + \varepsilon_t$,
$\varepsilon_t \sim N(0, \Sigma)$, is written as

$$
Y = X B + E, \qquad \operatorname{vec}(E) \sim N(0, \Sigma \otimes I_T),
$$

with row $t$ of $Y$ ($T \times n$) equal to $y_t'$ and row $t$ of $X$ ($T \times k$,
$k = 1 + np$) equal to $x_t' = (1, y_{t-1}', \dots, y_{t-p}')$; the first $p$
observations are the pre-sample. Column $i$ of $B$ ($k \times n$) holds equation $i$.

## Normal-Inverse-Wishart prior

$$
\Sigma \sim IW(\Psi, d), \qquad
\operatorname{vec}(B) \mid \Sigma \sim N\big(\operatorname{vec}(b),\ \Sigma \otimes \Omega\big).
$$

### Minnesota prior

The Minnesota prior (Litterman, 1986; Doan, Litterman & Sims, 1984) shrinks every
equation towards a random walk (or white noise for stationary series). In GLP's NIW
parameterisation $\Psi = \operatorname{diag}(\psi_1, \dots, \psi_n)$, $d = n + 2$ (the
smallest integer degrees of freedom with a finite $\operatorname{E}[\Sigma] =
\Psi/(d - n - 1)$), and $\Omega$ is diagonal with

$$
\Omega_{(l, j)} = \frac{\lambda^2 (d - n - 1)}{l^{\kappa}\, \psi_j}
\quad\text{(lag } l \text{ of variable } j\text{)}, \qquad
\Omega_{cc} = \text{large (diffuse constant)},
$$

so that

$$
\operatorname{E}[B_{l,ij}] = \begin{cases}\delta_i & i = j,\ l = 1\\ 0 & \text{otherwise}\end{cases},
\qquad
\operatorname{Var}(B_{l,ij} \mid \Sigma) = \frac{\lambda^2}{l^\kappa}
\frac{\Sigma_{ii}}{\operatorname{E}[\Sigma_{jj}]},
$$

where $B_{l,ij}$ is the coefficient of $y_{j,t-l}$ in equation $i$, $\delta_i = 1$
(random walk) or $0$ (white noise), $\kappa = 2$ and $\lambda$ is the **overall
tightness**: $\lambda \to 0$ imposes the prior mean, $\lambda \to \infty$ gives OLS. The
scales $\psi_j$ are customarily set to residual variances of univariate AR regressions
or estimated together with $\lambda$.

The same prior can be written as $np + n + 1$ dummy observations (Bańbura, Giannone &
Reichlin, 2010): with $\sigma_j = \sqrt{\psi_j}$ and $\tilde\lambda =
\lambda\sqrt{d - n - 1}$, one row per lag and variable with $X = l^{\kappa/2}\sigma_j /
\tilde\lambda$ in the column of $y_{j,t-l}$ (and $Y = \delta_j\sigma_j/\tilde\lambda$ in
column $j$ for $l = 1$), $n$ rows $Y = \operatorname{diag}(\sigma)$, $X = 0$, and one row
for the constant. OLS on these rows returns exactly $b$, $\Omega$ and $\Psi$; the tests
check this identity.

### Sum-of-coefficients and dummy-initial-observation priors

Let $\bar y_0$ be the average of the $p$ pre-sample observations. The
**sum-of-coefficients** prior (Doan, Litterman & Sims, 1984) adds $n$ rows

$$
Y^{+} = \operatorname{diag}(\bar y_0)/\mu, \qquad X^{+} = (0, Y^{+}, \dots, Y^{+}),
$$

which say that, when every variable stays at its initial level, it is best predicted by
that level — i.e. $I - \sum_l B_l \approx 0$ variable by variable (a unit root in each
equation, no cointegration) with tightness $\mu$. The **dummy-initial-observation**
prior (Sims, 1993; Sims & Zha, 1998) adds one row

$$
y^{++} = \bar y_0'/\delta, \qquad x^{++} = (1/\delta, y^{++}, \dots, y^{++}),
$$

which says that a no-change forecast is good at the beginning of the sample, allowing
for cointegration (a common stochastic trend). Both priors become dogmatic as $\mu,
\delta \to 0$ and vanish as $\mu, \delta \to \infty$. The dummy rows are prepended to the
data; the prior they encode is the NIW prior updated by them.

### Blocked random walk (mixed-frequency VAR with blocking)

In the blocked VAR of the [large BVAR](large-bvar.md) every monthly series $x$ becomes
three quarterly variables $x^{(1)}_t, x^{(2)}_t, x^{(3)}_t$ (months of quarter $t$).
Cimadomo et al. (2022, §2.2) use for the blocked vector "the same means and variances"
as in their monthly VAR, $\operatorname{E}[A_1] = I_N$ (their eq. 2): every block is
centred on **its own** first lag, i.e. $x^{(1)}_t$ on $x^{(1)}_{t-1}$, three months
earlier. This is `prior_mean="random_walk"`.

If $x$ is a monthly random walk, $x_\tau = x_{\tau-1} + u_\tau$, the blocked
representation is

$$
x^{(1)}_t = x^{(3)}_{t-1} + u_1, \qquad
x^{(2)}_t = x^{(3)}_{t-1} + u_1 + u_2, \qquad
x^{(3)}_t = x^{(3)}_{t-1} + u_1 + u_2 + u_3,
$$

a quarterly VAR(1) whose reduced-form first-lag matrix has, inside the block of $x$,
**ones in the column of the last month** $x^{(3)}_{t-1}$ and zeros elsewhere (and
correlated innovations, which $\Sigma$ captures). `prior_mean="blocked_random_walk"`
centres the Minnesota prior on it: $\operatorname{E}[A_1]_{(x,m),(x,3)} = 1$ for
$m = 1, 2, 3$; quarterly series keep the own-lag random walk; in a mapping, individual
series may be `"blocked_random_walk"`, `"random_walk"`, `"white_noise"` or a number.
Under the own-lag random walk the conditional nowcast of an unreleased month in a tight
prior repeats the same month of the previous quarter; under the blocked random walk it is
the last released month — the no-change forecast of a monthly random walk.

*Prior variances.* The conjugate NIW prior requires $\operatorname{Var}(\operatorname{vec}
B \mid \Sigma) = \Sigma \otimes \Omega$ with one $\Omega$ for all equations, so the
variance of a coefficient may depend on the regressor and the lag but not on the
equation. A "monthly-lag" decay ($x^{(3)}_{t-1}$ is one month before $x^{(1)}_t$ but three
months before $x^{(3)}_t$) is therefore not available; the variances keep the quarterly
lag decay $1/l^\kappa$ and the scales $\psi_j$, exactly as in Cimadomo et al. (eq. 3).

*Sum-of-coefficients and initial observation.* The usual rows impose
$\sum_l A_l = I$, i.e. a unit root **per block**, which contradicts the blocked random
walk (whose $\sum_l A_l$ restricted to the block of $x$ is $\iota e_3'$, not $I_3$). A
monthly I(1) series has a single unit root in the blocked VAR (the blocks cointegrate:
$x^{(2)}_t - x^{(1)}_t$ is stationary), with eigenvector $\iota_x = (1, 1, 1)'$:
$\sum_l A_l\, \iota_x = \iota_x$. With the blocked random walk the three blocks of a
series therefore form one *unit-root group* $g$ with common starting level
$\bar y_{0,g}$ (the mean of $\bar y_0$ over the blocks, i.e. of the $3p$ pre-sample
months) and the sum-of-coefficients prior has **one row per series**,

$$
Y^{+}_{g,i} = \frac{\bar y_{0,g}}{\mu}\,1\{i \in g\}, \qquad X^{+}_g = (0, Y^{+}_g, \dots, Y^{+}_g),
$$

which states: when all blocks of the series sit at a common level in all lags, the
series stays there and the other variables do not move. The dummy-initial-observation row
uses the group levels as well. The blocked random walk (constant 0) satisfies all these
rows exactly; singleton groups give back the usual priors. The marginal likelihood,
posterior and gradient are unchanged in form (they hold for any prior mean $b$; the
$\mu$-derivative sums over the $G$ group rows).

When to use it: blocked VARs in (log-)levels with Cimadomo-style settings
(random-walk centred prior, sum-of-coefficients and initial-observation priors, several
quarterly lags). With the own-lag centring, the data contradict the prior in two of the
three equations of every monthly series and the hierarchical choice compensates with a
loose $\lambda$ (and, with dummies, a very tight $\mu$); on a simulated monthly random
walk the selected $\lambda$ drops by two orders of magnitude with the blocked centring.
This refinement is nowcastbox's derivation, not part of the paper.

## Posterior and marginal likelihood

Conjugacy gives

$$
\Sigma \mid Y \sim IW(\bar\Psi, d + T), \qquad
\operatorname{vec}(B) \mid \Sigma, Y \sim N\big(\operatorname{vec}(\bar B), \Sigma \otimes \bar\Omega\big),
$$

$$
\bar\Omega = (X'X + \Omega^{-1})^{-1}, \quad
\bar B = \bar\Omega (X'Y + \Omega^{-1} b), \quad
\bar\Psi = \Psi + \hat E'\hat E + (\bar B - b)'\Omega^{-1}(\bar B - b),
\quad \hat E = Y - X\bar B,
$$

and the marginal likelihood is available in closed form (GLP, 2015, appendix A):

$$
p(Y \mid \lambda, \psi) = \pi^{-nT/2}\,
\frac{\Gamma_n\!\big(\tfrac{d+T}{2}\big)}{\Gamma_n\!\big(\tfrac d2\big)}\,
|\Omega|^{-n/2}\, |\Psi|^{d/2}\, |X'X + \Omega^{-1}|^{-n/2}\, |\bar\Psi|^{-\frac{d+T}{2}}.
$$

With the dummy observations, the likelihood of the actual data is the ratio
$p(Y \mid Y^{+}) = p(Y, Y^{+}) / p(Y^{+})$ of two such expressions.

**Numerics.** With diagonal $\Omega$ and $X_s = X\Omega^{1/2}$,
$|\Omega|\,|X'X + \Omega^{-1}| = |I_k + X_s'X_s| = |R|^2$, where $R$ is the triangular
factor of the thin QR decomposition of the stacked matrix $(X_s', I_k)'$; $\bar B - b =
\Omega^{1/2} z$ with $z$ the least-squares solution of $(X_s', I_k)' z \approx
((Y - Xb)', 0)'$, $\bar\Psi$ uses the residual form above, and the leverages needed by
the gradient are squared row norms of $Q$. The cross-product $X'X$ is never formed: with
data in levels (say $100\log$ of an index, around 460) and a tight sum-of-coefficients or
initial-observation prior, the dummy rows are of order $10^6$ and squaring the condition
number wipes out the identity part of $I_k + X_s'X_s$ — a Cholesky-based evaluation of
the same formula is then off by hundreds of log points. No matrix is inverted explicitly.
(QR is several times slower than a Cholesky factor in principle; with more than a few
BLAS threads on small matrices both are dominated by threading overhead, so limiting
`OPENBLAS_NUM_THREADS` speeds up the hyperparameter search considerably.)

## Hierarchical choice of the hyperparameters

GLP treat $\theta = (\lambda, \mu, \delta, \psi)$ as parameters with hyperprior
$p(\theta)$ and choose the posterior mode

$$
\hat\theta = \arg\max_\theta\ \log p(Y \mid \theta) + \log p(\theta).
$$

Default hyperpriors (GLP, 2015): Gamma for $\lambda$ with mode 0.2 and standard deviation
0.4; Gamma with mode 1 and standard deviation 1 for $\mu$ and $\delta$; inverse-Gamma
with shape and scale $0.02^2$ for each $\psi_j$ (almost flat). A Gamma with mode $m$ and
standard deviation $s$ has scale $\vartheta = (\sqrt{m^2 + 4s^2} - m)/2$ and shape
$1 + m/\vartheta$.

The optimisation runs on $\log\theta$ with L-BFGS-B and an **analytic gradient**. Since
$\bar\Psi - \Psi$ is the minimum over $B$ of
$S(B) = (Y - XB)'(Y - XB) + (B - b)'\Omega^{-1}(B - b)$, the envelope theorem gives its
derivative as that of $S$ at fixed $\bar B$, and

$$
\frac{\partial\log p}{\partial\log\Omega_{ii}} =
-\frac n2\Big(1 - \frac{\bar\Omega_{ii}}{\Omega_{ii}}\Big)
+ \frac{d+T}{2}\,\frac{(\bar B - b)_{i\cdot}\,\bar\Psi^{-1}(\bar B - b)_{i\cdot}'}{\Omega_{ii}},
\qquad
\frac{\partial\log p}{\partial\log\psi_j}\Big|_{\Psi} = \frac d2 - \frac{d+T}{2}\,\psi_j(\bar\Psi^{-1})_{jj},
$$

while scaling a dummy row $r$ by $s$ gives
$\partial\log p/\partial\log s = -n\,x_r'\bar\Omega x_r - (d+T)\,\hat e_r'\bar\Psi^{-1}\hat e_r$.
The chain rule through $\Omega_{(l,j)} \propto \lambda^2/\psi_j$ and the dummy rows
($\propto 1/\mu$, $1/\delta$) yields the gradient for $\lambda$, $\psi$, $\mu$ and $\delta$
at the cost of one likelihood evaluation, which makes the estimation of all $n$ scales
$\psi_j$ affordable in large systems. The inverse Hessian of the negative objective at
the mode (central differences of the gradient) is returned as the proposal covariance of
a random-walk Metropolis step, should the hyperparameters be integrated out rather than
fixed at the mode.

GLP show that the selected $\lambda$ decreases as the cross-section grows — the more
(collinear) variables, the more shrinkage is needed to avoid over-fitting — echoing
Bańbura, Giannone & Reichlin (2010) and De Mol, Giannone & Reichlin (2008).

## Posterior simulation

Draws of $(B, \Sigma)$ are exact. With $\bar\Psi = CC'$ and $A$ the lower-triangular
Bartlett factor of a $W(I_n, d + T)$ draw ($A_{ii}^2 \sim \chi^2_{d+T-i+1}$, standard
normal entries below the diagonal), $\Sigma = GG'$ with $G = C A^{-\top}$ is an
$IW(\bar\Psi, d+T)$ draw; then

$$
B = \bar B + \Omega^{1/2} R^{-1} Z G', \qquad R'R = I_k + \Omega^{1/2}X'X\Omega^{1/2},
\quad Z_{ij} \sim N(0, 1),
$$

has $\operatorname{vec}(B) \mid \Sigma \sim N(\operatorname{vec}(\bar B), \Sigma\otimes\bar\Omega)$.
Only triangular solves are used.

## Validation

The test-suite checks: the closed-form marginal likelihood against two-dimensional
numerical integration of likelihood × prior (univariate AR(1)) and against the
matrix-variate $t$ density of $Y$ computed with $T \times T$ matrices ($k \le T$ and
$k > T$, with and without constant) and, for data in levels with tight dummy priors,
against the same density evaluated in 60-digit arithmetic; the dummy-observation ratio
against the matrix-$t$ density under the NIW prior updated by the dummies; the posterior
against the conjugate formulas written with explicit inverses and against quadrature of
the posterior mean; the analytic gradient against finite differences; the optimiser
against a grid search, and the inverse Hessian against the numerical curvature; Monte
Carlo moments of the draws ($\operatorname{E}[\Sigma]$, $\operatorname{Var}(\Sigma_{11})$,
$\operatorname{E}[B]$, $\operatorname{Cov}(\operatorname{vec}B)$) and agreement with
`scipy.stats.invwishart`; that the Minnesota dummies reproduce $b$, $\Omega$ and $\Psi$;
and that very tight sum-of-coefficients / initial-observation priors impose
$\sum_l B_l = I$ and $c + \sum_l B_l\bar y_0 = \bar y_0$. For the blocked random walk
(`tests/models/test_bvar_blocked_prior.py`): the Minnesota dummies reproduce the full
prior-mean matrix; the closed-form marginal likelihood with a non-diagonal mean and
grouped dummies matches the GLP formula written with explicit cross-products and its
gradient matches finite differences; the grouped rows have zero residual at the blocked
random walk; a tight grouped sum-of-coefficients prior imposes
$\sum_l A_l\iota_g = \iota_g$; on a simulated monthly random walk the posterior is
centred on $x^{(3)}_{t-1}$ and GLP selects a $\lambda$ more than ten times tighter
than with the own-lag centring; and with $\lambda \to 0$ the conditional nowcast of an
unreleased quarter is the last released month. On simulated data the selected
$\lambda$ falls with the number of (factor-driven) variables and with the amount of
measurement noise.

## References

- Bańbura, M., Giannone, D., & Reichlin, L. (2010). Large Bayesian vector auto regressions. *Journal of Applied Econometrics*, 25(1), 71–92.
- Cimadomo, J., Giannone, D., Lenza, M., Monti, F., & Sokol, A. (2022). Nowcasting with large Bayesian vector autoregressions. *Journal of Econometrics*, 231(2), 500–519.
- De Mol, C., Giannone, D., & Reichlin, L. (2008). Forecasting using a large number of predictors: Is Bayesian shrinkage a valid alternative to principal components? *Journal of Econometrics*, 146(2), 318–328.
- Doan, T., Litterman, R., & Sims, C. (1984). Forecasting and conditional projection using realistic prior distributions. *Econometric Reviews*, 3(1), 1–100.
- Giannone, D., Lenza, M., & Primiceri, G. E. (2015). Prior selection for vector autoregressions. *Review of Economics and Statistics*, 97(2), 436–451.
- Litterman, R. B. (1986). Forecasting with Bayesian vector autoregressions — five years of experience. *Journal of Business & Economic Statistics*, 4(1), 25–38.
- Sims, C. A. (1993). A nine-variable probabilistic macroeconomic forecasting model. In J. H. Stock & M. W. Watson (Eds.), *Business Cycles, Indicators and Forecasting* (pp. 179–212). University of Chicago Press.
- Sims, C. A., & Zha, T. (1998). Bayesian methods for dynamic multivariate models. *International Economic Review*, 39(4), 949–968.
