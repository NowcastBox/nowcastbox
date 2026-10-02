# News decomposition

## Setting

Let $\Omega_v$ and $\Omega_{v+1}$ be two consecutive information sets. Between them,
(a) new observations $I_{v+1} = \{x_j : j \in J\}$ are **released**, and (b) some values
of $\Omega_v$ may be **revised**. The parameters $\theta$ are kept fixed (the effect of
re-estimating them is treated separately below). In a linear Gaussian state-space model
conditional expectations are linear projections, which makes an exact decomposition
possible (Bańbura & Modugno, 2014; Bańbura, Giannone, Modugno & Reichlin, 2013).

## News without revisions

If $\Omega_v \subset \Omega_{v+1}$ (no revisions), by the properties of orthogonal
projections

$$
\mathbb E[y_\tau \mid \Omega_{v+1}]
= \mathbb E[y_\tau \mid \Omega_v] + \mathbb E\big[y_\tau \mid I_{v+1} - \mathbb E[I_{v+1} \mid \Omega_v]\big],
$$

because the **news** $\mathcal N = I_{v+1} - \mathbb E[I_{v+1} \mid \Omega_v]$ is orthogonal
to $\Omega_v$. Hence

$$
\underbrace{\mathbb E[y_\tau \mid \Omega_{v+1}] - \mathbb E[y_\tau \mid \Omega_v]}_{\text{revision}}
= \mathbb E[y_\tau \mathcal N'] \, \mathbb E[\mathcal N \mathcal N']^{-1} \mathcal N
= \sum_{j \in J} b_j \left(x_j - \mathbb E[x_j \mid \Omega_v]\right).
$$

The weight vector $b = \operatorname{Cov}(y_\tau, \mathcal N \mid \Omega_v)\operatorname{Var}(\mathcal N \mid \Omega_v)^{-1}$
depends only on the model and on the *pattern* of available data, not on the values.
The **impact** of release $j$ is $b_j \times$ news$_j$, in target units.

## With data revisions

Introduce the intermediate information set $\Omega^\ast$: the **pattern** of $\Omega_v$
with the **revised values** of $\Omega_{v+1}$. Then

$$
\mathbb E[y_\tau \mid \Omega_{v+1}] - \mathbb E[y_\tau \mid \Omega_v]
= \underbrace{\mathbb E[y_\tau \mid \Omega^\ast] - \mathbb E[y_\tau \mid \Omega_v]}_{\text{revisions effect}}
+ \underbrace{\sum_{j} b_j \left(x_j - \mathbb E[x_j \mid \Omega^\ast]\right)}_{\text{news effect}} ,
$$

and the revisions effect splits by series by linearity. Values present in $\Omega_v$
and absent from $\Omega_{v+1}$ are reported as a removal effect.

## Re-estimation

If the parameters are re-estimated on the new vintage ($\theta_{v+1}$),

$$
\mathbb E_{\theta_{v+1}}[y_\tau \mid \Omega_{v+1}] - \mathbb E_{\theta_v}[y_\tau \mid \Omega_v]
= \text{revisions} + \text{news} + \underbrace{\mathbb E_{\theta_{v+1}}[y_\tau \mid \Omega_{v+1}] - \mathbb E_{\theta_v}[y_\tau \mid \Omega_{v+1}]}_{\text{re-estimation}} .
$$

## Computation

The weights could be computed from the joint Gaussian distribution of $(y_\tau, \mathcal N)$,
which requires the covariances of the smoothed states across periods. NowcastBox uses
instead the **linearity of the Kalman smoother in the data** for a fixed pattern of
missing values (Durbin & Koopman, 2012, §4.4): the smoothed signal is
$\hat y_\tau = \sum_{t,i} \omega_{\tau, (t,i)} x_{t,i} + \text{const}$, and the weights
$\omega$ are obtained by smoothing unit impulses in a batch. The news weights $b_j$, the
expectations $\mathbb E[x_j \mid \Omega^\ast]$ and the level contributions
$\sum_t \omega_{\tau,(t,i)} x_{t,i}$ (Koopman & Harvey, 2003) all follow. The identity
old + revisions + news (+ re-estimation) = new is checked to about $10^{-12}$
(`NewsResults.check_identity`), and the weights were validated against brute-force
Gaussian conditioning.

## Aggregation by group

Impacts add up, so they can be grouped by series, block, category (hard / soft /
financial) or release date — the basis of the [nowcast tracker](../user-guide/news/tracker.md),
which chains decompositions through every release of a quarter.

## Other linear-Gaussian models

The same decomposition applies to any model that is linear and Gaussian given its
parameters. The [large Bayesian VAR](large-bvar.md) exposes its blocked quarterly
state-space form through `linear_nowcast_model()`, so `nowcastbox.news` decomposes its
nowcast revisions exactly and maps the releases back to the original monthly series.
