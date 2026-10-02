# Pre-selection of indicators: t-stat, SIS and LARS

The ECB Nowcasting Toolbox (Linzenich & Meunier, 2024, §2.1) starts the construction of a
model by ranking the candidate indicators against the target with several criteria and
combining the rankings into one score (an approach applied by Chinn, Meunier & Stumpner,
2023). `nowcastbox.selection.preselect` implements this workflow from the papers listed
below; this page documents the definitions and the choices made.

## Alignment of mixed frequencies

The rankings are computed at the frequency of the target. Let $\tau(t)$ be the last
high-frequency period of target period $t$ (the third month of a quarter). A monthly
candidate $x$ enters as its aggregate

$$
\bar x_t = \sum_{i} w_i\, x_{\tau(t) - i},
$$

with the weights of the series' aggregation rule (`SeriesMetadata.aggregation`), or of
the `aggregation` argument when the metadata has none:

| rule | weights (monthly to quarterly) |
|---|---|
| `"average"` (default) | $\tfrac13(1, 1, 1)$ |
| `"flow"` | $(1, 1, 1)$ |
| `"stock"` | $(1, 0, 0)$ (last month) |
| `"mariano_murasawa"` | $\tfrac13(1, 2, 3, 2, 1)$ |

These are exactly the regressors a bridge equation uses, so the ranking measures the
information the indicator carries *at the frequency where the model is evaluated*. The
average is the default because it is the usual bridge-equation choice for indicators in
growth rates or levels of a stationary transformation; for month-on-month growth rates
of a quarterly growth target the Mariano-Murasawa rule is the consistent one. Candidates
already at the target frequency are used as they are, and candidates at a lower frequency
(or with a non-fixed frequency ratio, e.g. weekly to monthly) are skipped with a warning.
An aggregate is missing when any of its months is missing, so the ragged edge never
produces a partial quarter.

**Leads and lags.** With `x_lags`, each aggregated candidate also enters shifted:
$\bar x_{t-l}$ for $l > 0$ (column `<series>_lag<l>`) and the lead $\bar x_{t+|l|}$ for
$l < 0$ (`<series>_lead<|l|>`). Every (series, lag) pair is ranked as a separate
candidate. At the series level, a series' rank under method $m$ is the rank of its best
lag, and `best_lag` is the lag with the highest aggregated candidate score.

## The three rankings

Rows where the target is missing are removed before the target is shifted by the
forecast horizon $h$ (in target periods), as for the targeted predictors.

**t-statistic** (Bai & Ng, 2008, hard thresholding). For every candidate,

$$
y_{t+h} = \alpha + \sum_{j=1}^{J} \phi_j\, y_{t+1-j} + \beta_i\, \bar x_{it} + \varepsilon_{t+h},
$$

with $J$ = `y_lags`, estimated by OLS on the observations available for that candidate;
candidates are ranked by $|t_{\beta_i}|$ with Newey-West (Bartlett) or classical
standard errors. This reuses `hard_threshold` with threshold 0.

**Sure independence screening** (Fan & Lv, 2008). Candidates are ranked by the absolute
marginal correlation $|\operatorname{corr}(\bar x_{it}, y_{t+h})|$ on the pairwise
complete observations. Fan & Lv show that, under conditions on the design, keeping the
$d = \lfloor n / \log n \rfloor$ largest correlations retains all relevant predictors with
probability tending to one (*sure screening*); `sis` uses this default size.

**Least angle regression** (Efron, Hastie, Johnstone & Tibshirani, 2004). The
candidates are standardised and the target centred; starting from $\hat\mu_0 = 0$, LARS
adds at every step the predictor most correlated with the current residual and moves the
fit along the *equiangular* direction of the active set,

$$
\mathcal G_{\mathcal A} = X_{\mathcal A}'X_{\mathcal A}, \quad
A_{\mathcal A} = (1'\mathcal G_{\mathcal A}^{-1}1)^{-1/2}, \quad
w_{\mathcal A} = A_{\mathcal A}\,\mathcal G_{\mathcal A}^{-1}1, \quad
u_{\mathcal A} = X_{\mathcal A} w_{\mathcal A}
$$

($X_{\mathcal A}$ holds the active columns multiplied by the signs of their correlations),
until another predictor has the same absolute correlation with the residual:

$$
\hat\gamma = {\min_{j \in \mathcal A^c}}^{+}\left\{
\frac{\hat C - \hat c_j}{A_{\mathcal A} - a_j},\
\frac{\hat C + \hat c_j}{A_{\mathcal A} + a_j}\right\},
\qquad a = X' u_{\mathcal A}.
$$

The rank of a candidate is its order of entry. With `lars_method="lasso"` the lasso
modification (Efron et al., 2004, §3.1) removes a predictor whose coefficient would
change sign; the dropped predictor keeps the position of its first entry. Missing
aggregated values (e.g. lags at the start of the sample) are replaced by the mean
(`missing="mean"`, the default of `preselect`) or the incomplete periods are dropped
(`missing="drop"`). The implementation is our own (no scikit-learn dependency) and is
validated against `sklearn.linear_model.lars_path` in the test suite (identical order of
entry, penalties and coefficients up to $\min(n-1, p)$ steps, including lasso paths in
which a predictor is dropped and re-enters). Beyond the point where the centred design
loses rank, the plain-LAR path of scikit-learn becomes numerically degenerate (its knots
violate the Karush-Kuhn-Tucker conditions), whereas the in-house path stops there; the
tests therefore compare the first $\min(n-1, p)$ steps.

## Aggregated score

Let $r_m(i) \in \{1, \dots, N\}$ be the rank of candidate $i$ under method $m$ (1 =
best). Candidates a method cannot rank (too few observations, constant, never entering
the LARS path) share the positions left free and receive their average,
$(n_m + 1 + N)/2$ with $n_m$ the number of ranked candidates. The normalised score and
the aggregated score are

$$
s_m(i) = 1 - \frac{r_m(i) - 1}{N - 1} \in [0, 1], \qquad
S(i) = \frac{\sum_m w_m\, s_m(i)}{\sum_m w_m},
$$

with method weights $w_m \ge 0$ (`weights=`, default 1 each). Series are sorted by $S$,
ties broken by the method ranks in the order of `methods`, and the first `top` series are
selected (default $\min(30, N)$, the size used by Bai & Ng, 2008).

*Hand example.* Three series ranked (tstat, sis, lars) = $a: (1, 2, 3)$,
$b: (2, 1, 1)$, $c: (3, 3, 2)$ have normalised scores $a: (1, 0.5, 0)$,
$b: (0.5, 1, 1)$, $c: (0, 0, 0.5)$, hence $S = (0.5, 0.833, 0.167)$ with equal weights
and $S = (0.625, 0.75, 0.125)$ with $w = (2, 1, 1)$; the order is $b, a, c$ in both cases.

## Real time

Inside a pseudo real-time evaluation the pre-selection must only use data released at
the vintage. `preselect(..., as_of=date)` first replaces the panel by
`MixedFrequencyData.as_of(date)` (values released after `end of period + release delay`
are masked), so that the target, the aggregated indicators and therefore all rankings
use the vintage information set only.

## References

- Linzenich, J., & Meunier, B. (2024). Nowcasting made easier: a toolbox for economists.
  ECB Working Paper No. 3004.
- Chinn, M. D., Meunier, B., & Stumpner, S. (2023). Nowcasting world trade with machine
  learning: a three-step approach. ECB Working Paper No. 2808.
- Bai, J., & Ng, S. (2008). Forecasting economic time series using targeted predictors.
  *Journal of Econometrics*, 146(2), 304–317.
- Fan, J., & Lv, J. (2008). Sure independence screening for ultrahigh dimensional
  feature space. *Journal of the Royal Statistical Society B*, 70(5), 849–911.
- Efron, B., Hastie, T., Johnstone, I., & Tibshirani, R. (2004). Least angle regression.
  *The Annals of Statistics*, 32(2), 407–499.
- Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity
  and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703–708.
