# News decomposition (innovation I6)

When new data arrive the nowcast moves. Bańbura & Modugno (2014) showed that, in a linear
Gaussian state-space model, the revision is an exact weighted sum of the **news** — the
surprise of each release relative to what the model expected:

$$
\underbrace{\mathbb{E}[y_\tau \mid \Omega_{v+1}] - \mathbb{E}[y_\tau \mid \Omega_v]}_{\text{nowcast revision}}
= \underbrace{\text{revisions effect}}_{\text{revised old data}}
+ \sum_{j \in \text{releases}} \underbrace{b_j}_{\text{weight}}
\underbrace{\left(x_j - \mathbb{E}[x_j \mid \Omega^\ast]\right)}_{\text{news}}
\;(+\ \text{re-estimation}),
$$

where $\Omega^\ast$ is the old information set with revised values. The decomposition is
exact (see [News decomposition](../../theory/news.md) for the derivation); NowcastBox
checks the identity to ~1e-12.

## Decomposing a revision

```python
import nowcastbox as nb

COLUMNS = ["pib", "ibc_br", "pim_geral", "pmc_varejo", "pms_volume", "energia_consumo_total",
           "anp_diesel", "secex_exportacoes", "ipca", "selic", "credito_concessoes_sa",
           "icbr", "focus_pib"]
ds = nb.load_brazil_nowcast(columns=COLUMNS, start="2012-01", end="2024-12")
panel = nb.prepare_panel(ds.data, ds.transform, keep="pib")

old = nb.pseudo_real_time(panel, delay=ds.delay, vintage="2024-10-15")
new = nb.pseudo_real_time(panel, delay=ds.delay, vintage="2024-11-15")

res = nb.MixedFreqDFM(n_factors=1, blocks="data").fit(new, target="pib")
news = res.news(old, new, target_period="2024Q4")
print(news.summary())
news.check_identity()
```

`res.news(old, new, period)` is the method form of
`nb.news_decomposition(res, old, new, period)`. Parameters are those of `res` for both
vintages, so only information changes. Supported models: `MixedFreqDFM` (also on weekly
or daily grids with calendar aggregation: the time-varying observation equation is
rebuilt for each vintage) and `TwoStepDFM`. With `aggregate="variables"` every vintage is
passed through the fitted predictor filters (`results.filter_panel`): a raw release is
attributed to the filtered observation it completes, and the revision of a raw value
(which changes several filtered values) is counted with the data revisions.

## Tables

```python
news.old_nowcast, news.news_effect, news.revisions_effect, news.new_nowcast

news.top_releases(5)[["series", "reference_period", "actual", "expected", "weight", "impact"]]
news.to_frame(by="series").head()
news.to_frame(by="category")      # hard / soft / financial
news.to_frame(by="block")
news.waterfall(by="category")     # old nowcast -> contributions -> new nowcast
```

Each release row reports the **actual** value, the model **expectation** given the old
information, the **news** (`actual - expected`, units of the series), the **weight**
(impact per unit of news) and the **impact** on the target (`weight × news`, target
units).

## Waterfall chart

```python
fig = news.plot("waterfall")                          # Matplotlib (default of nowcastbox.news)
fig = news.plot("waterfall", backend="plotly")        # interactive
```

## Revisions and re-estimation

Real vintages also **revise** previously published values. The decomposition separates
the effect of revised data (`revisions_effect`, per series in `news.revisions`) from the
news of new releases. To illustrate, revise the August industrial production in the
new vintage:

```python
frame = new.to_frame()
frame.loc["2024-08", "pim_geral"] += 0.01             # a revision of an old release
revised = new.with_data(frame)
with_revision = res.news(old, revised, "2024Q4")
with_revision.revisions_effect, with_revision.revisions
```

Passing `new_results` (the model re-estimated on the new vintage) adds the
**re-estimation** effect — the change due to new parameters:

```python
res_old = nb.MixedFreqDFM(n_factors=1, blocks="data").fit(old, "pib")
full = nb.news_decomposition(res_old, old, new, "2024Q4", new_results=res)
full.reestimation_effect, full.check_identity()
```

## Which cells changed?

```python
nb.news.data_revisions(old, revised).head()   # released / revised / removed cells
```

## Reading the results

- Weights depend on the **timing**: early in the quarter soft data and financial
  indicators get large weights; once hard data for the quarter arrive their weights
  dominate (Bańbura & Rünstler, 2011).
- A large impact needs both a large surprise and a large weight.
- Grouping by category answers the classic question "hard or soft data?"; grouping by
  block shows which part of the economy moved the nowcast.
- Follow the decomposition through every release of a quarter with the
  [nowcast tracker](tracker.md).
