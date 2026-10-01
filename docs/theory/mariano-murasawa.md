# Temporal aggregation

A lower-frequency series is an aggregate of an unobserved higher-frequency series. The
aggregation defines linear restrictions that let a monthly factor model explain
quarterly (or annual) data.

## Mariano & Murasawa (2003)

Let $Y_t$ be the unobserved **monthly level** of a flow (e.g. GDP) and approximate the
quarterly level by the geometric mean of the three months,

$$
Y^Q_t = (Y_t\, Y_{t-1}\, Y_{t-2})^{1/3}, \qquad t = \text{last month of the quarter}.
$$

The quarterly log growth $y^Q_t = \ln Y^Q_t - \ln Y^Q_{t-3}$ is then an exact linear
combination of the monthly log growth rates $y_t = \ln Y_t - \ln Y_{t-1}$:

$$
y^Q_t = \tfrac13\left(\ln Y_t + \ln Y_{t-1} + \ln Y_{t-2}\right)
      - \tfrac13\left(\ln Y_{t-3} + \ln Y_{t-4} + \ln Y_{t-5}\right)
      = \tfrac13\left(y_t + 2y_{t-1} + 3y_{t-2} + 2y_{t-3} + y_{t-4}\right).
$$

The triangular weights $(1, 2, 3, 2, 1)$ follow from telescoping. Using the arithmetic
mean instead of the geometric mean makes the relation an approximation, accurate for
small growth rates.

### General ratio

For $k$ high-frequency periods per low-frequency period the same argument gives
triangular weights of length $2k - 1$:

$$
w = (1, 2, \dots, k-1, k, k-1, \dots, 2, 1), \qquad y^{L}_t = \tfrac1k \sum_{s=0}^{2k-2} w_s\, y_{t-s},
$$

($k = 3$ month → quarter, $k = 12$ month → year, $k = 4$ quarter → year).
`AggregationType.GROWTH_RATE.weights(k, normalize=True)` returns $w/k$.

## Levels: flow, average, stock

For series in levels (or already stationary quantities aggregated linearly):

| Type | Relation ($k$ = ratio) | Weights |
|---|---|---|
| flow | $x^L_t = \sum_{s=0}^{k-1} x_{t-s}$ | $(1, \dots, 1)$ |
| average | $x^L_t = \tfrac1k \sum_{s=0}^{k-1} x_{t-s}$ | $(1, \dots, 1)/k$ |
| stock | $x^L_t = x_t$ | $(1, 0, \dots, 0)$ |

## Restrictions on the loadings

If the latent high-frequency series follows the factor model
$x^\ast_t = \lambda' f_t + e^\ast_t$, the observed aggregate is

$$
x^L_t = \sum_{s} w_s\, x^\ast_{t-s} = \sum_s \lambda_s' f_{t-s} + \sum_s w_s e^\ast_{t-s},
\qquad \lambda_s = w_s \lambda .
$$

In a model with free loadings $\lambda_0, \dots, \lambda_L$ on $f_t, \dots, f_{t-L}$ this is
the set of linear restrictions

$$
w_s \lambda_0 - w_0 \lambda_s = 0, \qquad s = 1, \dots, L ,
$$

written $R \operatorname{vec}(\lambda) = 0$ (Bańbura & Modugno, 2014). For
Mariano-Murasawa weights ($w_0 = 1$):

$$
R = \begin{bmatrix}
2 & -1 & 0 & 0 & 0 \\ 3 & 0 & -1 & 0 & 0 \\ 2 & 0 & 0 & -1 & 0 \\ 1 & 0 & 0 & 0 & -1
\end{bmatrix} \otimes I_r .
$$

The idiosyncratic component of the aggregate is the same weighted sum of the latent
AR(1) idiosyncratic process, which is why the state carries $L$ lags of it for each
lower-frequency series.

## Calendar-aware aggregation (I1)

Between days and months or weeks and months/quarters the number of high-frequency
periods per low-frequency period varies. Let the current low-frequency period $T$
contain $n$ high-frequency periods and the previous one $n'$. With high-frequency log
levels $\tilde x_s$ and growth rates $x_s = \tilde x_s - \tilde x_{s-1}$, the growth of
the (geometric) period average is

$$
y_T = \frac1n \sum_{j \in T} \tilde x_j - \frac1{n'} \sum_{j \in T-1} \tilde x_j ,
$$

which, written in growth rates, puts weight $(l+1)/n$ on lag $l < n$ and
$(n' - v)/n'$ on lag $n - 1 + v$, $v = 1, \dots, n'-1$ (a trapezoid of length
$n + n' - 1$). With $n = n' = k$ it reduces to the normalised Mariano-Murasawa weights.
Flows, averages and stocks use $(1, \dots, 1)$, $(1, \dots, 1)/n$ and
$(1, 0, \dots, 0)$ with the period's own length. Because the weights change from period
to period, the observation equation is time varying ($Z_t$; the public `StateSpace` supports
time-varying $Z_t$, $H_t$, $d_t$ through `obs_index`) and the state carries the maximum number of lags over the
calendar (`max_periods_per`). See [Aggregation and frequencies](../user-guide/data/aggregation.md).
