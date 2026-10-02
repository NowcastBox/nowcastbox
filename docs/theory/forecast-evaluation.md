# Forecast evaluation

## Pseudo real-time design

For a set of vintage dates $v_1 < v_2 < \dots$ the evaluation:

1. builds the information set $\Omega_{v}$ of each date — from publication delays
   (pseudo real time; Giannone, Reichlin & Small, 2008) or from a store of real
   vintages (real time; Croushore, 2011);
2. estimates the model on $\Omega_v$ (expanding or rolling window, every vintage or every
   $k$ vintages);
3. records the backcast, nowcast and forecast of the target periods around $v$;
4. compares them with the realised value (final, latest, first or $n$-th release).

Accuracy is reported **by horizon** — months (or days) between the vintage and the end
of the target quarter — because nowcasts improve as information accumulates through the
quarter (Bańbura, Giannone, Modugno & Reichlin, 2013).

## Metrics

With errors $e_t = y_t - \hat y_t$ on a common sample of $n$ forecasts:
$\text{RMSFE} = \sqrt{n^{-1}\sum e_t^2}$, $\text{MAE} = n^{-1}\sum |e_t|$,
bias $= n^{-1}\sum e_t$, and the relative RMSFE against a benchmark.

## Directional accuracy

The forecast directional accuracy (FDA) of Linzenich & Meunier (2024) counts how often a
forecast predicts correctly whether the target rises or falls with respect to a previous
value $y^p_t$:

$$
\text{FDA} = \frac1n \sum_{t=1}^n I_t, \qquad
I_t = \mathbb 1\big[(y_t - y^p_t)(\hat y_t - y^p_t) > 0\big].
$$

For a growth rate this is whether the economy accelerates or decelerates (Blaskowitz &
Herwartz, 2011, discuss directional errors). In a backtest, $y^p_t$ is the value of the
period before the target period **known at the vintage date** — the last released value
when that period is not yet published — so that the reference uses no information the
forecaster did not have; the final value of the previous period is an option.

The Pesaran & Timmermann (1992) test asks whether the hit rate exceeds what independent
directions would give. With $x_t = \mathbb 1[\hat y_t - y^p_t > 0]$,
$z_t = \mathbb 1[y_t - y^p_t > 0]$, $\hat P = n^{-1}\sum \mathbb 1[x_t = z_t]$,
$\hat P_x = \bar x$, $\hat P_z = \bar z$ and
$\hat P_\ast = \hat P_z \hat P_x + (1-\hat P_z)(1-\hat P_x)$,

$$
S_n = \frac{\hat P - \hat P_\ast}{\sqrt{\hat V(\hat P) - \hat V(\hat P_\ast)}}
\;\xrightarrow{d}\; N(0, 1),
\qquad \hat V(\hat P) = \frac{\hat P_\ast (1 - \hat P_\ast)}{n},
$$

$$
\hat V(\hat P_\ast) = \frac{(2\hat P_z - 1)^2 \hat P_x (1-\hat P_x)}{n}
+ \frac{(2\hat P_x - 1)^2 \hat P_z (1-\hat P_z)}{n}
+ \frac{4 \hat P_z \hat P_x (1-\hat P_z)(1-\hat P_x)}{n^2}.
$$

The test is one-sided ($H_1$: directional predictability). It is undefined when every
predicted or every actual change has the same sign.

## Sub-periods

Accuracy and test statistics can be dominated by a few extreme quarters (the Covid-19
recession and rebound). Evaluating on sub-periods of **target periods** — e.g.
pre-Covid, Covid (2020Q1-2021Q4, the quarters overlapping March 2020-December 2021),
post-Covid and the full sample without the pandemic — shows whether a ranking of models
is driven by them. Each sub-period forms its own common sample before the metrics and
tests are computed.

## Diebold-Mariano test

For loss differentials $d_t = L(e_{1t}) - L(e_{2t})$ of two $h$-step forecasts, the null
of equal predictive accuracy $\mathbb E[d_t] = 0$ is tested with (Diebold & Mariano, 1995)

$$
DM = \frac{\bar d}{\sqrt{\hat V / n}}, \qquad
\hat V = \hat\gamma_0 + 2 \sum_{k=1}^{h-1} \hat\gamma_k ,
$$

where $\hat\gamma_k$ are the autocovariances of $d_t$ (Bartlett weights are used if
$\hat V \le 0$). Harvey, Leybourne & Newbold (1997) correct the small-sample size,

$$
DM^\ast = DM \sqrt{\frac{n + 1 - 2h + h(h-1)/n}{n}}, \qquad DM^\ast \sim t_{n-1}.
$$

When several nowcasts of the same target period are pooled (e.g. the three monthly
nowcasts of a quarter), their differentials are strongly correlated and $\hat V$ with
$h = 1$ understates the variance of $\bar d$. `aggregate="target_period"` averages the
losses within each target period, so that $n$ is the number of periods.

## Clark-West test

When model 1 nests model 2 (e.g. a bridge equation or a DFM that includes the
autoregressive terms of an AR benchmark), the population MSPEs are equal under $H_0$ but
the larger model's sample MSPE is inflated by the noise of estimating parameters that are
zero, and the DM test is undersized. Clark & West (2007) adjust the differential,

$$
d_t = e_{1t}^2 - (\hat y_{2t} - \hat y_{1t})^2 - e_{2t}^2
    = e_{1t}^2 - (e_{1t} - e_{2t})^2 - e_{2t}^2 ,
$$

and compare $\bar d / \sqrt{\hat V / n}$ (same $\hat V$ as above) with standard normal
critical values, usually one-sided ($\mathbb E[d_t] < 0$: the larger model is more
accurate).

## Giacomini-White test

The conditional test of Giacomini & White (2006) asks whether the loss differential is
predictable from information $h_t$ available at the forecast origin:
$H_0: \mathbb E[h_t d_{t+h}] = 0$. With $Z_t = h_t d_{t+h}$ and $h_t = (1, d_t)'$,

$$
GW = n\, \bar Z' \hat\Omega^{-1} \bar Z \sim \chi^2_{q},
$$

with $\hat\Omega$ the sample covariance of $Z_t$ (Newey-West HAC for $h > 1$). It applies
to forecasts from estimated models with finite (rolling) estimation windows.

## Model Confidence Set

Hansen, Lunde & Nason (2011) build the set $\widehat{\mathcal M}^\ast_{1-\alpha}$ of models
that contains the best ones with probability $\ge 1-\alpha$. Starting from all models,
test the equivalence hypothesis $H_{0,\mathcal M}: \mathbb E[d_{ij,t}] = 0$ for all
$i, j \in \mathcal M$ with

$$
T_{\max} = \max_{i \in \mathcal M} \frac{\bar d_{i\cdot}}{\widehat{\operatorname{sd}}(\bar d_{i\cdot})}
\quad\text{or}\quad
T_{R} = \max_{i,j \in \mathcal M} \frac{|\bar d_{ij}|}{\widehat{\operatorname{sd}}(\bar d_{ij})},
$$

whose distribution is obtained by a (circular) moving-block bootstrap. While the
hypothesis is rejected, eliminate the worst model; MCS p-values are the running maximum
of the test p-values along the elimination sequence.
