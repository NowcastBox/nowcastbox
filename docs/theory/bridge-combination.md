# Combination of bridge equations

## Thick modelling with small bridge equations

A single bridge equation (Baffigi, Golinelli & Parigi, 2004; Diron, 2008) relies on a
few hand-picked indicators. Bańbura, Belousova, Bodnár & Tóth (2023) instead estimate
**every** small bridge equation that can be built from a set of candidate indicators and
pool their predictions. With $N_m$ higher-frequency (monthly) indicators and $N_q$
indicators of the target frequency (quarterly), equation $e$ uses a subset $S_e$ of
$a \in \{m_0, \dots, m\}$ monthly and $b \in \{0, \dots, q\}$ quarterly indicators:

$$
y_\tau = \alpha_e + \sum_{k \in S_e} \sum_{l=0}^{L} \beta_{e,k,l}\, \bar x^{(k)}_{\tau-l}
       + \sum_{j=1}^{J} \phi_{e,j}\, y_{\tau-j} + u_{e,\tau},
\qquad
\bar x^{(k)}_\tau = \sum_{s} w_s\, x^{(k)}_{3\tau - s},
$$

where $\bar x^{(k)}_\tau$ is the quarterly aggregate of indicator $k$ (average, sum,
end-of-period or Mariano-Murasawa weights $w$) and the target lags are optional. The
number of equations is

$$
E = \Big(\sum_{a=m_0}^{m} \binom{N_m}{a}\Big)\Big(\sum_{b=0}^{q} \binom{N_q}{b}\Big),
$$

e.g. $\binom{50}{1} + \binom{50}{2} = 1\,275$ equations with 50 monthly indicators,
$m_0 = 1$, $m = 2$ and no quarterly indicator; $(N_m + \binom{N_m}{2})(1 + N_q)$ with
one quarterly indicator at most (the default of the toolbox of Linzenich & Meunier,
2024, which follows Bańbura et al., 2023).

## Ragged edge: extrapolate once, share across equations

At the ragged edge the most recent months of some indicators are not released yet. As
in the individual bridge equations, every indicator is completed up to the end of the
forecast horizon by an *extrapolator* — by default iterated AR($p$) forecasts estimated
on the indicator's own history — and only then aggregated. Because the completion of an
indicator does not depend on the equation it enters, it is done **once per vintage** and
the aggregated indicators $\bar x^{(k)}_\tau$ are shared by all $E$ equations. The
extrapolator is pluggable: a multivariate model completes all indicators jointly through
the same interface.

### BVAR extrapolation

`extrapolation="bvar"` completes the indicators with the conditional forecast of one
Bayesian VAR estimated on them. The VAR has the Normal-Inverse-Wishart Minnesota prior,
and its tightness is the posterior mode of the hierarchical model of Giannone, Lenza &
Primiceri (2015) (see [BVAR priors](bvar-prior.md)). Let $X$ hold the released entries of
the indicators at the vintage. With the parameters at their posterior mean
$(\bar B, \bar\Sigma)$, every unreleased entry is replaced by

$$
\hat x_{i,t} = E\big[x_{i,t} \mid X;\ \bar B, \bar\Sigma\big],
$$

which is the conditional forecast of Waggoner & Zha (1999) and Bańbura, Giannone & Lenza
(2015). It is computed with the Kalman smoother on the companion form of the VAR, with
missing values for the unreleased entries, starting from the last window of $p$
complete periods. A released value of one indicator therefore moves the completion of
the others through the estimated covariance $\bar\Sigma$ and the lag coefficients. The
AR extrapolator ignores these links.

Two VARs are available:

- **Monthly VAR** (default when every indicator is monthly). This is a VAR($p$) on the
  monthly indicators, the large monthly BVAR of Bańbura, Giannone & Reichlin (2010), with
  $p = 3$ by default. Extrapolating monthly indicators is a monthly forecasting problem.
  The monthly VAR has $N$ variables and three times as many observations as a blocked
  VAR, and each newly complete month enlarges the conditioning window.
- **Blocked quarterly VAR** (default when quarterly indicators are also completed). This
  is the "blocking" model of Cimadomo, Giannone, Lenza, Monti & Sokol (2022), used by
  [`LargeBVAR`](large-bvar.md): each monthly series becomes three quarterly variables,
  so monthly and quarterly indicators share one VAR. It has $3N$ variables (150 for 50
  indicators) and only quarterly observations, with $p = 1$ by default, which covers the
  same one-quarter memory as the monthly default.

The estimation sample is the longest run of complete periods that ends at the last
complete one. The VAR is estimated once per vintage, and the extrapolator instance caches
it, keyed by a hash of the data. Calls for a longer horizon on the same vintage reuse it,
for example the refits that `BridgeCombinationBenchmark.predict` makes. When the joint VAR
cannot be estimated (too few complete periods), the combination retries series by series
and leaves out the series that still fail.

Only the point extrapolation enters the bridge equations. Like the AR extrapolation, it
ignores the uncertainty of the completed indicators.

## Estimation in batches

Each equation is estimated by OLS on the quarters where the target and all its
regressors are observed. Equations with the same number of coefficients $k$ are stacked
into a tensor $X \in \mathbb R^{E \times T \times k}$; the rows an equation cannot use
are set to zero in both $X_e$ and $y$, which leaves its OLS solution unchanged, and all
problems are solved at once by a batched QR decomposition,

$$
X_e = Q_e R_e, \qquad \hat\beta_e = R_e^{-1} Q_e' y .
$$

An equation is flagged as not estimable (and excluded from the combination) when $R_e$
is numerically singular — collinear indicators — or when fewer than $k + 1$ quarters
are available. Predictions for quarters with an unobserved target lag iterate the
equation's own forecasts, as in a single bridge equation.

## Combination

With $\hat y_{e,\tau}$ the prediction of equation $e$ for quarter $\tau$, the combined
nowcast is

$$
\hat y_\tau = \sum_{e \in \mathcal E_\tau} \tilde w_{e}\, \hat y_{e,\tau},
\qquad \tilde w_e = \frac{w_e}{\sum_{e' \in \mathcal E_\tau} w_{e'}},
$$

where $\mathcal E_\tau$ is the set of included equations with a prediction for $\tau$
(without extrapolation, an equation whose indicators are incomplete for $\tau$ drops out
for that quarter only). The weights are:

| `combine` | Weights | Reference |
|---|---|---|
| `"mean"` | $w_e = 1$ | Bańbura et al. (2023); Timmermann (2006) on the robustness of equal weights |
| `"median"` | cross-equation median of $\hat y_{e,\tau}$ | robust to outlying equations |
| `"inverse_mse"` | $w_e = 1 / \mathrm{MSE}_e$ | Stock & Watson (2004) |

### Accuracy measure

Stock & Watson (2004) weight forecasts by their discounted past mean squared forecast
error. With evaluation quarters $\tau_1 < \dots < \tau_n$ (the last `mse_window` quarters
with an observed target) and a discount factor $\delta \in (0, 1]$,

$$
\mathrm{MSE}_e = \frac{\sum_{i} \delta^{\tau_n - \tau_i}\, e_{e,\tau_i}^2}
                      {\sum_{i} \delta^{\tau_n - \tau_i}} ,
$$

where $e_{e,\tau}$ is either the in-sample residual (`mse="in_sample"`) or the one-step
**pseudo out-of-sample** error (`mse="out_of_sample"`): for each evaluation quarter
$\tau_i$ the equation is re-estimated on the quarters before $\tau_i$ only (at least
`min_train` of them) and predicts $y_{\tau_i}$. The out-of-sample errors use the
indicators as they are in the current vintage (past quarters are complete), so they
measure the fit of the equation rather than its real-time accuracy, but they never use
information that is not in the vintage.

### Trimming

Timmermann (2006) discusses trimming: discarding the worst-performing models before
combining. `trim` $= \pi$ drops the $\lfloor \pi E^\ast \rfloor$ estimable equations with
the largest MSE ($E^\ast$ estimable equations; at least one is always kept).

## Pseudo real time and news

Everything — extrapolation, coefficients, MSEs, weights and trimming — is computed from
the information set of the vintage passed to `fit` (or `data.as_of(vintage)` with the
`as_of=` option), so the combination can be evaluated in a pseudo real-time backtest
without look-ahead.

Bridge combinations have no state-space representation, so the exact news decomposition
of the dynamic factor models (Bańbura & Modugno, 2014) does not apply. The revision
between two vintages is split into equation contributions
$\tilde w^{new}_e \hat y^{new}_{e,\tau} - \tilde w^{old}_e \hat y^{old}_{e,\tau}$, which
sum to the revision of the combined nowcast (mean and inverse-MSE combinations); a
contribution mixes new releases, data revisions and re-estimation, and can be shared
equally by the indicators of the equation for an indicator-level view.

## References

- Baffigi, A., Golinelli, R., & Parigi, G. (2004). Bridge models to forecast the euro area GDP. *International Journal of Forecasting*, 20(3), 447–460.
- Bańbura, M., Belousova, I., Bodnár, K., & Tóth, M. B. (2023). Nowcasting employment in the euro area. ECB Working Paper No. 2815.
- Bańbura, M., Giannone, D., & Lenza, M. (2015). Conditional forecasts and scenario analysis with vector autoregressions for large cross-sections. *International Journal of Forecasting*, 31(3), 739–756.
- Bańbura, M., Giannone, D., & Reichlin, L. (2010). Large Bayesian vector auto regressions. *Journal of Applied Econometrics*, 25(1), 71–92.
- Bańbura, M., & Modugno, M. (2014). Maximum likelihood estimation of factor models on datasets with arbitrary pattern of missing data. *Journal of Applied Econometrics*, 29(1), 133–160.
- Cimadomo, J., Giannone, D., Lenza, M., Monti, F., & Sokol, A. (2022). Nowcasting with large Bayesian vector autoregressions. *Journal of Econometrics*, 231(2), 500–519.
- Diron, M. (2008). Short-term forecasts of euro area real GDP growth: an assessment of real-time performance based on vintage data. *Journal of Forecasting*, 27(5), 371–390.
- Giannone, D., Lenza, M., & Primiceri, G. E. (2015). Prior selection for vector autoregressions. *Review of Economics and Statistics*, 97(2), 436–451.
- Linzenich, J., & Meunier, B. (2024). Nowcasting made easier: a toolbox for economists. ECB Working Paper No. 3004.
- Stock, J. H., & Watson, M. W. (2004). Combination forecasts of output growth in a seven-country data set. *Journal of Forecasting*, 23(6), 405–430.
- Timmermann, A. (2006). Forecast combinations. In G. Elliott, C. W. J. Granger, & A. Timmermann (Eds.), *Handbook of Economic Forecasting* (Vol. 1, pp. 135–196). Elsevier.
- Waggoner, D. F., & Zha, T. (1999). Conditional forecasts in dynamic multivariate models. *Review of Economics and Statistics*, 81(4), 639–651.
