# HTML reports

`NowcastReport` renders a self-contained HTML report of a nowcast (Jinja2 + Plotly,
following the `panelbox.report` pattern), ready to e-mail or publish:

| Section | Content | Needs |
|---|---|---|
| headline | latest nowcast, previous value, change, band | results |
| path | observed vs. in-sample vs. out-of-sample, fan chart | results (+ `quantiles`) |
| news | waterfall and table of releases | `news` |
| tracker | nowcast through the vintages with contributions | `tracker` |
| data flow | ragged-edge heatmap and release table | results with data |
| backtest | RMSFE by horizon | `backtest` |
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
