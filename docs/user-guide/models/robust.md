# Robust estimation (innovation I3)

Extreme observations distort the EM estimates: in 2020 a single quarter of Brazilian GDP
fell by almost 10 % and monthly indicators by 20–30 %. Dropping 2020–2021 throws away
information; ignoring the swing inflates factor variances and biases the loadings.
`MixedFreqDFM` offers four complementary treatments (Antolin-Diaz, Drechsel & Petrella,
2017, 2024), all inside the likelihood.

| Option | What it does | Results |
|---|---|---|
| `idiosyncratic="student_t"` | idiosyncratic errors $\varepsilon_{it} \sim t_\nu$ (scale mixture of normals), estimated by ECM; `df=None` estimates $\nu$ | `student_t_df`, `observation_weights` |
| `outliers="auto"` | after convergence, observations whose standardised one-step-ahead prediction error exceeds `outlier_threshold` are treated as missing and the EM is re-run (≤ 3 passes) | `outlier_flags` |
| `covid="mask"` | observations in `covid_window` are treated as missing | — |
| `covid="dummy"` | impulse dummies in the factor VAR for each period of `covid_window` | `interventions` |
| `exclude_periods=[...]` | any periods treated as missing in the estimation | `excluded_observations` |

The states are always smoothed through the treated periods, and `nowcast["observed"]`
keeps the data.

## Example: the pandemic in Brazil

```python
import nowcastbox as nb

COLUMNS = ["pib", "ibc_br", "pim_geral", "pmc_varejo", "pms_volume", "energia_consumo_total",
           "anp_diesel", "secex_exportacoes", "ipca", "selic", "credito_concessoes_sa",
           "icbr", "focus_pib"]
ds = nb.load_brazil_nowcast(columns=COLUMNS, start="2012-01", end="2024-12")
# keep the outliers in the data: the model treats them
panel = nb.prepare_panel(ds.data, ds.transform, keep="pib", replace_outliers=False)
vintage = nb.pseudo_real_time(panel, delay=ds.delay, vintage="2024-11-15")

gaussian = nb.MixedFreqDFM(n_factors=1).fit(vintage, "pib")
student = nb.MixedFreqDFM(n_factors=1, idiosyncratic="student_t").fit(vintage, "pib")
student.student_t_df                                   # estimated degrees of freedom
student.observation_weights.loc["2020-03":"2020-06", ["ibc_br", "pim_geral"]].round(2)
```

Weights below one down-weight an observation in the M-step: April 2020 industrial
production gets a small weight instead of dragging the loadings.

```python
auto = nb.MixedFreqDFM(n_factors=1, outliers="auto", outlier_threshold=4.0).fit(vintage, "pib")
flags = auto.outlier_flags
flags.sum()[flags.sum() > 0]                           # outliers found per series

dummy = nb.MixedFreqDFM(n_factors=1, covid="dummy",
                        covid_window=("2020-03", "2021-12")).fit(vintage, "pib")
dummy.interventions.head(3)                            # impulse effects on the factor

masked = nb.MixedFreqDFM(n_factors=1, exclude_periods=[("2020-03", "2020-12")]).fit(vintage, "pib")

{name: round(r.get_nowcast("2024Q4"), 4)
 for name, r in {"gaussian": gaussian, "student_t": student, "auto": auto,
                 "dummy": dummy, "exclude": masked}.items()}
```

The same options are available through the one-call interface:

```python
robust = nb.nowcast(vintage, "pib", method="em", n_factors=1,
                    idiosyncratic="student_t", outliers="auto")
robust.model_params["idiosyncratic"], robust.model_params["outliers"]
```

## Which one?

- **Student-t** is the most general: no window to choose, the data decide which
  observations are extreme. It requires the white-noise idiosyncratic layout and the
  reported log-likelihood is the variational lower bound maximised by the ECM.
- **`outliers="auto"`** is simple and transparent (flags you can inspect), and works
  with AR(1) idiosyncratic components.
- **COVID dummies** keep the pandemic observations informative for the factors while
  protecting the VAR dynamics — the preferred option when the swing is common to all
  series.
- **Masking / excluding** is the bluntest: it removes information but is easy to explain.

Compare them on a backtest that includes 2020–2022 (see
[Backtesting](../evaluation/backtesting.md)) and inspect the
[diagnostics](../diagnostics.md).
