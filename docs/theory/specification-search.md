# Specification search and Covid robustness

The ECB Nowcasting Toolbox (Linzenich & Meunier, 2024, §2.2-2.3) builds a model in three
steps. It pre-selects the indicators ([Pre-selection](preselection.md)), searches over
specifications of the model scored in pseudo real time, and finally checks how robust the
best specifications are to the treatment of the pandemic observations.
`nowcastbox.selection.SpecificationSearch` implements the last two steps from the
paper's description. This page documents the definitions and the choices made.

## Specifications

A specification $s$ fixes:

- the parameters of the model, e.g. the number of factors $r$, the order $p$ of the
  factor VAR, the block structure, the idiosyncratic dynamics;
- the first period $t_0$ of the estimation sample;
- the number $k$ of indicators. They are taken from the top of a ranking of the
  candidates (the "funnel" strategy), usually the aggregated pre-selection score.

The search space is a product of finite sets, one per setting (an integer range
$\{a, \dots, b\}$ is one such set). The **exhaustive grid** evaluates every element. A
**random search** draws each setting independently and uniformly over its set, which
covers large spaces far more efficiently than a grid when only a few settings matter
(Bergstra & Bengio, 2012). Draw $i$ uses its own generator, seeded by the $i$-th child of
`numpy.random.SeedSequence(random_state)`. The specification of draw $i$ is therefore a
function of the seed and $i$ only: extending a search keeps the earlier draws, and an
interrupted search resumes from its checkpoint with exactly the same draws.

## Pseudo real-time evaluation

Each specification is evaluated with the design of Giannone, Reichlin & Small (2008)
and Bańbura et al. (2013), as implemented by `PseudoRealTimeBacktest`. At each vintage
$v$, the final data are masked with the release calendar. The model is re-estimated every
`refit_every` vintages, and only the information set is updated in between. The nowcasts
of the target periods around $v$ are then recorded. Let $e_{s,j}$ be the errors of
specification $s$ in horizon group $j$, where $j$ is backcast, nowcast or forecast, or the
number of months to the end of the quarter. The criteria are

$$
\mathrm{RMSFE}_{s,j} = \Big(\tfrac{1}{n_j}\textstyle\sum e_{s,j}^2\Big)^{1/2},
\qquad
\mathrm{FDA}_{s,j} = \tfrac{1}{n_j}\textstyle\sum
\mathbb 1\{\operatorname{sign}(\hat y - y^{p}) = \operatorname{sign}(y - y^{p})\},
$$

and MSE, MAE and bias. Here $y^{p}$ is the value of the previous target period known at
the vintage (Linzenich & Meunier, 2024, note 9; see
[Forecast evaluation](forecast-evaluation.md)).

**No look-ahead in the funnel.** If the ranking used to choose the $k$ indicators is
computed on the final data, the evaluation is contaminated: the indicators are chosen
knowing their full-sample correlation with the target, including the periods being
forecast. With a per-vintage ranking, the pre-selection is run on the information set
of each re-estimation vintage, which only contains data released by that date. The
indicators are then kept fixed until the next re-estimation. A fixed ranking is
accepted, but a `DataQualityWarning` is issued when it was computed with data released
after the first vintage.

## Weighted score

Let $m$ index the metrics, with weights $w_m \ge 0$, and $j$ the horizons, with weights
$v_j \ge 0$. The loss $L_{s,m,j}$ is the metric for RMSFE, MSE and MAE, $|\text{bias}|$
for the bias, and $1 - \mathrm{FDA}$ for directional accuracy, so lower is always better.
The score of specification $s$ is

$$
S_s = \frac{\sum_m \sum_j w_m v_j\, \tilde L_{s,m,j}}{\sum_m w_m \sum_j v_j},
$$

where $\tilde L$ is one of three normalisations:

- **rank** (default): $\tilde L_{s,m,j} = (R_{s,m,j} - 1)/(N - 1)$, where $R$ is the rank
  of $L_{s,m,j}$ among the $N$ valid specifications (average ranks for ties). It is 0 for
  the best and 1 for the worst. Being scale free, it can mix RMSFEs, which are measured in
  the units of the target, with directional accuracies, which are proportions. It is the
  same construction as the aggregated score of the pre-selection.
- **relative**: $\tilde L_{s,m,j} = L_{s,m,j} / L_{\text{ref},m,j}$, the ratio to the
  same loss of a reference benchmark (e.g. an AR model) computed on the same vintages.
  For RMSFE this is the usual relative RMSFE; for FDA it is the ratio of the
  directional error rates.
- **none**: the raw losses (useful with a single metric).

A specification is **invalid**, with $S_s = \infty$, when its backtest produced no
evaluable forecast, when more than a share `max_missing` of its forecasts are missing
because of failed fits, or when one of its weighted criteria is missing. Invalid
specifications do not enter the ranks of the valid ones. Horizons that were never
evaluated, such as backcasts in a window that starts after them, are dropped from the
weights, with a warning.

The criteria are stored for every specification, so the score can be recomputed with
other weights without re-estimating anything.

## Covid robustness

The pandemic produced swings in activity and in many indicators that are an order of
magnitude larger than usual. A factor model estimated on them can see its loadings, VAR
coefficients and innovation variances distorted. The toolbox therefore re-evaluates the
best specifications with several treatments and compares them on the target periods
after the pandemic (post-2021, as described in the plan of this item). The treatments
available here are:

| treatment | definition |
|---|---|
| none | the observations are used as they are |
| dummies | an impulse dummy for each period of the pandemic window enters the factor VAR, $f_t = \sum_l A_l f_{t-l} + \delta_t + u_t$, so the swing is absorbed by $\delta_t$ and does not distort $A_l$ and $\operatorname{var}(u_t)$ (`MixedFreqDFM(covid="dummy")`) |
| masking | the observations of the pandemic window are treated as missing in the estimation; the Kalman smoother bridges the gap (`covid="mask"`) |
| outlier correction | the robust EM drops observations whose standardised prediction error exceeds a threshold (`outliers="auto"`); for models without that option, the IQR rule with moving-median replacement is applied to the predictors of every vintage (never to the target) |

Each (specification, treatment) pair is backtested again. The evaluation is restricted to
target periods from `evaluate_from`, and the pairs are ranked with the score above. A
treatment that a model cannot apply, such as dummies in a two-step model, is reported as
skipped rather than silently replaced. Because the outlier correction is applied to each
vintage's information set, there is no look-ahead: a value is flagged using only the
observations available at that vintage.

## Choices and differences

- The toolbox's code was not consulted. The design follows the description in the
  working paper. The normalisation of the score by ranks is a choice of this
  implementation: the paper describes a score weighted by horizon and metric, and ranks
  make metrics with different units comparable without a benchmark. Relative and raw
  losses are available as options.
- Random draws are uniform and independent. Duplicates are evaluated once.
- The funnel uses the first $k$ indicators of the ranking. When the ranking is
  recomputed per vintage, two vintages may use different indicators. This is what a
  forecaster repeating the exercise in real time would do.

## References

Bańbura, M., Giannone, D., Modugno, M. & Reichlin, L. (2013). Now-casting and the real-time
data flow. In *Handbook of Economic Forecasting*, vol. 2A, 195-237.

Bergstra, J. & Bengio, Y. (2012). Random search for hyper-parameter optimization. *Journal
of Machine Learning Research*, 13, 281-305.

Giannone, D., Reichlin, L. & Small, D. (2008). Nowcasting: The real-time informational
content of macroeconomic data. *Journal of Monetary Economics*, 55(4), 665-676.

Linzenich, J. & Meunier, B. (2024). Nowcasting made easier: a toolbox for economists. ECB
Working Paper No. 3004.

Pesaran, M. H. & Timmermann, A. (1992). A simple nonparametric test of predictive
performance. *Journal of Business & Economic Statistics*, 10(4), 461-465.
