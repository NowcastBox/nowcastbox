# HTML reports

`NowcastReport` renders a self-contained HTML report of a nowcast (Jinja2 + Plotly,
following the `panelbox.report` pattern), ready to e-mail or publish:

| Section | Content | Needs |
|---|---|---|
| headline | latest nowcast, previous value, change, band, empirical bands | results (+ `bands`) |
| path | observed vs. in-sample vs. out-of-sample, fan chart, empirical bands, alternative models | results (+ `quantiles`, `bands`, `alternatives`) |
| news | waterfall and table of releases | `news` |
| tracker | nowcast through the vintages with contributions | `tracker` |
| data flow | ragged-edge heatmap, share of the nowcast period's data released, release table, indicator z-score heatmap | results with data (+ `heatmap`) |
| backtest | RMSFE by horizon, accuracy table (sub-periods, FDA) | `backtest`, `backtest_metrics` |
| diagnostics | DFM diagnostics (I9) | `diagnostics=True` or a report |

## Example

```python
import nowcastbox as nb

ds = nb.load_brazil_nowcast(
    columns=["pib", "ibc_br", "pim_geral", "pmc_varejo", "pms_volume", "ipca", "selic",
             "credito_concessoes_sa", "focus_pib"],
    start="2012-01",
    end="2024-12",
)
panel = nb.prepare_panel(ds.data, ds.transform, keep="pib")
old = nb.pseudo_real_time(panel, delay=ds.delay, vintage="2024-10-15")
new = nb.pseudo_real_time(panel, delay=ds.delay, vintage="2024-11-15")

res = nb.MixedFreqDFM(n_factors=1, blocks="data").fit(new, "pib")
news = res.news(old, new, "2024Q4")
dist = res.distribution()

html_report = nb.NowcastReport(
    res,
    title="Brazil GDP nowcast — 15 November 2024",
    news=news,
    news_group_by="category",
    quantiles=dist,
    diagnostics=True,
    author="Research department",
    notes="Pseudo real-time vintage; quarter-on-quarter growth, seasonally adjusted.",
    plotlyjs="cdn",           # "inline" (default) embeds plotly.js: larger, works offline
)
html = html_report.to_html("nowcast_report.html")
len(html) > 10_000
```

`to_html(path=None)` returns the HTML (and writes it when a path is given); `render()`
returns it without writing; `context()` exposes the template variables.

## Conjunctural sections (ECB toolbox parity)

!!! note "New in 0.2.0"

The report can also show the outputs of the ECB Nowcasting Toolbox (Linzenich & Meunier,
2024): `bands=` takes the [empirical error bands](../density/empirical-bands.md) (tiles at
57.5/68/90 % and a fan chart), `alternatives=` the [nowcasts of alternative
models](../evaluation/alternative-models.md) (chart and min/median/max table),
`heatmap=True` (or an `IndicatorZScores`) adds the [indicator
heatmap](indicator-heatmap.md) and `backtest_metrics=` an accuracy table such as
`bt.metrics(metrics=("rmsfe", "fda", "n"), periods="covid")`. The [share of the
nowcast period's predictor data already released](../data/released-share.md), by
category, is shown by default (`released_share=False` hides it).

```python
groups = {"activity": ["ibc_br", "pim_geral", "pmc_varejo", "pms_volume"],
          "prices_rates": ["ipca", "selic"],
          "credit_surveys": ["credito_concessoes_sa", "focus_pib"]}
alt = nb.alternative_models(res, new, "pib", by=groups, drop=1, refit=False)
conjuncture = nb.NowcastReport(res, alternatives=alt, heatmap=True, plotlyjs="cdn").render()
"Nowcasts of alternative models" in conjuncture and "Indicator z-scores" in conjuncture
```

## Custom templates

Copy `nowcastbox/reports/templates/nowcast_report.html`, edit it and pass
`template_dir=` — it receives the same context:

```python
from pathlib import Path

custom = Path("my_templates")
custom.mkdir(exist_ok=True)
(custom / "nowcast_report.html").write_text(
    "<html><body><h1>{{ title }}</h1></body></html>",
    encoding="utf-8",
)
minimal = nb.NowcastReport(res, title="Minimal", template_dir=custom, plotlyjs="cdn").render()
"<h1>Minimal</h1>" in minimal
```

## From an experiment

`NowcastExperiment.report(name, path)` writes the report of one of its models (see
[Experiments](../evaluation/experiments.md)).
