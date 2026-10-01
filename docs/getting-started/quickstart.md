# Quickstart

This page nowcasts Brazilian quarterly GDP growth from a dozen monthly indicators,
explains how the nowcast moved when new data arrived and puts a band around it. Every
step is detailed in the [user guide](../user-guide/index.md).

## 1. Load a dataset

Built-in datasets ship the panel **in levels** with a legend (frequency, suggested
transformation, publication delay in days, blocks and category of each series).

```python
import nowcastbox as nb

COLUMNS = [
    "pib",                    # quarterly GDP (target)
    "ibc_br", "pim_geral", "pmc_varejo", "pms_volume",   # hard data
    "energia_consumo_total", "anp_diesel", "secex_exportacoes",
    "ipca", "selic", "credito_concessoes_sa", "icbr",    # prices, financial
    "focus_pib",              # market expectations (soft)
]
ds = nb.load_brazil_nowcast(columns=COLUMNS, start="2012-01", end="2024-12")
print(ds.summary())
ds.legend[["frequency", "transform", "delay_days", "blocks", "category"]]
```

## 2. Prepare the panel

`prepare_panel` applies the stationary transformations of the legend (log-differences,
quarter-on-quarter growth...), replaces outliers by a moving median and fills interior
gaps, **without** touching the ragged edge that the models exploit.

```python
panel = nb.prepare_panel(ds.data, ds.transform, keep="pib")
panel
```

## 3. Take a vintage

`pseudo_real_time` removes every observation that had not been published by a date,
using the publication delays. On 15 November 2024 the third-quarter GDP was not yet out
and most monthly indicators stopped in September or October.

```python
vintage = nb.pseudo_real_time(panel, delay=ds.delay, vintage="2024-11-15")
vintage.last_observed()
```

## 4. Estimate and nowcast

```python
res = nb.MixedFreqDFM(n_factors=1).fit(vintage, target="pib")
print(res.summary())

res.nowcast.tail(4)          # observed, in_sample, out_of_sample, std, bands
print("2024Q3 nowcast:", res.get_nowcast("2024Q3"))
```

The same in one call, with the number of factors chosen by the Bai-Ng criterion and a
density nowcast:

```python
quick = nb.nowcast(vintage, target="pib", method="em", density=True)
quick.nowcast.tail(2)[["out_of_sample", "lower_90", "median", "upper_90"]]
```

## 5. Explain the revision

Between mid-October and mid-November new releases moved the 2024Q4 nowcast. The news
decomposition attributes the change to each release (surprise × weight):

```python
old = nb.pseudo_real_time(panel, delay=ds.delay, vintage="2024-10-15")
news = res.news(old, vintage, target_period="2024Q4")
print(news.summary())
news.to_frame(by="category")
```

## 6. Plot and report

```python
fig = res.plot("forecast")                 # Plotly figure (fig.show() in a notebook)
fig_mpl = res.plot("forecast", backend="matplotlib")
waterfall = news.plot("waterfall")

nb.NowcastReport(res, news=news, title="Brazil GDP nowcast").to_html("report.html")
```

## Where next

- [Core concepts](core-concepts.md): the base grid, the ragged edge and vintages.
- [Choosing a method](choosing-a-method.md): two-step, EM, bridge or a benchmark?
- [Backtesting](../user-guide/evaluation/backtesting.md): how accurate is the nowcast
  at each point of the quarter?
