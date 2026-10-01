# Bai-Ng criteria

## Number of static factors (Bai & Ng, 2002)

Let $X$ be the $T \times N$ standardised balanced panel and, for $r$ factors estimated
by principal components,

$$
V(r) = \min_{\Lambda, F} \frac{1}{NT} \sum_{i=1}^N \sum_{t=1}^T \left(x_{it} - \lambda_i' f_t\right)^2
= \frac{1}{NT} \sum_{j > r} \mu_j ,
$$

the average squared residual ($\mu_j$ are the eigenvalues of $X'X$). With
$C^2_{NT} = \min(N, T)$ the penalties are

$$
g_1(N,T) = \frac{N+T}{NT} \ln\frac{NT}{N+T}, \qquad
g_2(N,T) = \frac{N+T}{NT} \ln C^2_{NT}, \qquad
g_3(N,T) = \frac{\ln C^2_{NT}}{C^2_{NT}},
$$

and the criteria

$$
IC_{p,k}(r) = \ln V(r) + r\, g_k(N, T), \qquad
PC_{p,k}(r) = V(r) + r\, \hat\sigma^2 g_k(N, T),
$$

with $\hat\sigma^2 = V(r_{\max})$. The estimate $\hat r = \arg\min_{0 \le r \le r_{\max}}$
is consistent as $N, T \to \infty$. The $IC$ criteria do not depend on the scale of
$\hat\sigma^2$ and hence on $r_{\max}$; the $PC$ criteria do.

In small panels ($\min(N, T) < 20$) or with strongly heteroskedastic idiosyncratic
noise the criteria tend to over-estimate $r$ (Bai & Ng, 2002, simulations).

## Number of primitive shocks (Bai & Ng, 2007)

When static factors include lags of dynamic factors, the VAR of the static factors

$$
\hat f_t = \sum_{i=1}^p A_i \hat f_{t-i} + \hat u_t
$$

has residuals of reduced rank: $u_t = B \eta_t$ with $B$ of rank $q < r$. Let
$c_1 \ge \dots \ge c_r$ be the eigenvalues of the sample covariance (or correlation)
matrix of $\hat u_t$, and define

$$
\hat D_{1,k} = \left( \frac{c_{k+1}^2}{\sum_{j=1}^r c_j^2} \right)^{1/2}, \qquad
\hat D_{2,k} = \left( \frac{\sum_{j=k+1}^{r} c_j^2}{\sum_{j=1}^r c_j^2} \right)^{1/2}.
$$

With the bound

$$
M_{NT}(\delta) = \frac{m}{\min\left(N^{1/2 - \delta},\, T^{1/2 - \delta}\right)}, \qquad 0 < \delta < \tfrac12,\; m > 0,
$$

the estimate is $\hat q = \min\{k : \hat D_k < M_{NT}(\delta)\}$. Bai & Ng (2007)
recommend $m = 1$ with the covariance matrix and $m = 1.25$ ($D_1$) or $2.25$ ($D_2$)
with the correlation matrix; results depend on $\delta$ and $m$, so a sensitivity table is
good practice.

In the two-step model $q$ sets the rank of $Q = BB'$, with
$\hat B = P_q M_q^{1/2}$ from the leading eigenpairs of $\hat\Sigma_u$.

## Targeted predictors (Bai & Ng, 2008)

Before extracting factors, predictors can be screened for their relevance to the target:

- **hard thresholding**: regress the target $y_{t+h}$ on each predictor $x_{it}$ (with
  optional lags of $y$) and keep the predictors with HAC $|t_i| > c$ (e.g. $c = 1.65$);
- **soft thresholding**: rank predictors by the order in which they enter the elastic-net
  path (Zou & Hastie, 2005),
  $\min_\beta \frac{1}{2n}\|y - X\beta\|^2 + \alpha\left(\rho\|\beta\|_1 + \tfrac{1-\rho}{2}\|\beta\|_2^2\right)$,
  solved by coordinate descent (Friedman, Hastie & Tibshirani, 2010), and keep the first
  $k$.
