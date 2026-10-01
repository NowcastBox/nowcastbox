# Dynamic factor models

## The approximate dynamic factor model

Let $x_t$ be the $N \times 1$ vector of standardised indicators. The dynamic factor model
writes it as the sum of a **common component**, driven by a few factors, and an
**idiosyncratic component**:

$$
x_t = \Lambda f_t + \varepsilon_t, \qquad \varepsilon_t \sim N(0, \Psi),
$$

$$
f_t = \sum_{i=1}^{p} A_i f_{t-i} + B u_t, \qquad u_t \sim N(0, I_q),
$$

with $f_t$ the $r \times 1$ static factors, $\Lambda$ the $N \times r$ loadings, $A_i$ the
VAR matrices of the factors and $B$ an $r \times q$ matrix: the $r$ factors are driven by
$q \le r$ **primitive shocks** (Bai & Ng, 2007). The model is *approximate* (Chamberlain
& Rothschild, 1983; Stock & Watson, 2002): idiosyncratic components may be weakly
cross-correlated and serially correlated; estimation treats $\Psi$ as diagonal, which is
a misspecification that vanishes as $N, T \to \infty$ (Doz, Giannone & Reichlin, 2011,
2012).

**Identification.** $\Lambda f_t = (\Lambda G)(G^{-1} f_t)$ for any invertible $G$, so
factors are identified only up to rotation. Principal components fix the rotation by
$\Lambda'\Lambda / N$ diagonal and $F'F/T = I$; nowcasts, news and common components are
invariant to it.

## State-space form

With $p$ factor lags the state contains the factors and their lags,
$\alpha_t = (f_t', f_{t-1}', \dots, f_{t-p+1}')'$, and

$$
T = \begin{bmatrix} A_1 & A_2 & \cdots & A_p \\ I_r & 0 & \cdots & 0 \\
 & \ddots & & \vdots \\ 0 & \cdots & I_r & 0 \end{bmatrix}, \quad
R = \begin{bmatrix} I_r \\ 0 \end{bmatrix}, \quad Q = B B', \quad
Z = \begin{bmatrix} \Lambda & 0 & \cdots & 0 \end{bmatrix}, \quad H = \Psi .
$$

Missing observations (ragged edge, late starts, quarterly slots) simply remove rows of
$Z$ and $y_t$ at each $t$ (see [Kalman filter](kalman-filter-smoother.md)).

### Lower-frequency series

A quarterly series observed in the third month of each quarter is a weighted sum of an
unobserved monthly counterpart $x^\ast_t = \lambda' f_t + e^\ast_t$:

$$
x^Q_t = \sum_{s=0}^{L} w_s\, x^\ast_{t-s}
= \sum_{s=0}^{L} w_s \lambda' f_{t-s} + \sum_{s=0}^{L} w_s e^\ast_{t-s},
$$

so its row of $Z$ loads on $f_t, \dots, f_{t-L}$ — the state must contain $L+1$ lags of
the factors (5 for Mariano-Murasawa) — with loadings proportional to the weights
$w = (1, 2, 3, 2, 1)$. See [Temporal aggregation](mariano-murasawa.md).

### Idiosyncratic dynamics

Bańbura & Modugno (2014) put AR(1) idiosyncratic components in the state:
$\varepsilon_{i,t} = \rho_i \varepsilon_{i,t-1} + \nu_{i,t}$, with a small fixed
measurement noise left in $H$ (`obs_noise_var`). For a quarterly series the latent
monthly AR(1) is aggregated with the same weights, which requires $L+1$ lags of
$e^\ast_{i,t}$ in the state. `idiosyncratic="iid"` drops these states and estimates a
diagonal $H$ instead.

### Blocks

With blocks $b = 1, \dots, B$ each block has its own factors $f^b_t$ following an
independent VAR, and $\Lambda$ has zeros where a series does not belong to a block
(Bańbura, Giannone & Reichlin, 2011). The transition matrix is block diagonal.

## The two-step estimator

Giannone, Reichlin & Small (2008) and Doz, Giannone & Reichlin (2011):

1. **Principal components.** On the balanced part of the panel, the eigenvectors $V_r$ of
   the $r$ largest eigenvalues of the sample correlation matrix give
   $\hat\Lambda = V_r$ and $\hat f_t = V_r' x_t$ (up to scale).
2. **Factor VAR.** OLS of $\hat f_t$ on $\hat f_{t-1}, \dots, \hat f_{t-p}$ gives
   $\hat A_i$ and the residual covariance $\hat\Sigma_u$. With $q < r$,
   $\hat B = P_q M_q^{1/2}$ from the $q$ largest eigenvalues $M_q$ and eigenvectors $P_q$
   of $\hat\Sigma_u$ (rank-$q$ approximation).
3. **Idiosyncratic variances.** $\hat\Psi = \operatorname{diag}\big(\widehat{\operatorname{Var}}(x_t - \hat\Lambda \hat f_t)\big)$.
4. **Kalman smoother** on the full, unbalanced panel with these parameters gives
   $\hat f_{t \mid T}$, which uses every available observation, including the ragged edge.

Doz, Giannone & Reichlin (2011) show that the smoothed factors are consistent for the
space spanned by the true factors as $N, T \to \infty$, even though the model ignores
idiosyncratic cross- and serial correlation.

### Bridging to the quarterly target

The target is linked to the factors by a bridge regression at the quarterly frequency:

$$
y^Q_\tau = \alpha + \beta' \bar f^Q_\tau + e_\tau ,
$$

where (`aggregate="factors"`) $\bar f^Q_\tau = \tfrac19 (f_t + 2 f_{t-1} + 3 f_{t-2} + 2 f_{t-3} + f_{t-4})$
for the last month $t$ of quarter $\tau$, or (`aggregate="variables"`) the factors are
extracted from monthly series filtered with $\tfrac13(1,2,3,2,1)$ so that they are
already quarterly quantities. Nowcasts use the smoothed (and forecast) factors of the
current and next quarters. The prediction variance
$\beta' V_\tau \beta + \hat\sigma^2_e$ combines the smoothing variance $V_\tau$ of
$\bar f^Q_\tau$ and the bridge residual variance.

## Maximum likelihood (EM)

Doz, Giannone & Reichlin (2012) show that quasi-maximum likelihood estimation of the
same model, treating it as exact, is consistent and more efficient than the two-step
estimator, and Bańbura & Modugno (2014) give an EM algorithm for an arbitrary pattern
of missing data. See [EM algorithm](em-algorithm.md).
