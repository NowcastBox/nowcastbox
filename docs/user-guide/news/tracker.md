# Nowcast tracker

The **nowcast tracker** follows the nowcast of one target period through a sequence of
vintages — typically every release date of the quarter — and attributes each change to
news by category, block or series, as in the charts published by the New York Fed Staff
Nowcast (Bok et al., 2018).

```python
import nowcastbox as nb

COLUMNS = ["pib", "ibc_br", "pim_geral", "pmc_varejo", "pms_volume", "energia_consumo_total",
           "anp_diesel", "secex_exportacoes", "ipca", "selic", "credito_concessoes_sa",
           "icbr", "focus_pib"]
ds = nb.load_brazil_nowcast(columns=COLUMNS, start="2012-01", end="2024-12")
panel = nb.prepare_panel(ds.data, ds.transform, keep="pib")
calendar = ds.calendar()                       # release dates of every series

vintage = nb.pseudo_real_time(panel, calendar=calendar, vintage="2024-10-01")
res = nb.MixedFreqDFM(n_factors=1).fit(vintage, target="pib")

tracker = res.nowcast_tracker(panel, calendar, "2024Q4", start="2024-10-01", end="2024-12-31")
print(tracker.summary())
tracker.check_identity()
```

Between consecutive vintages the change of the nowcast is decomposed into news,
revisions and (with `refit=True`) re-estimation, so the cumulative contributions add up
to the path.

## Tables and chart

```python
tracker.nowcasts.tail()                       # nowcast at each vintage date
tracker.to_frame().tail(3)                     # change, news, revisions, contributions
tracker.cumulative_contributions().tail(3)     # by group (default: category)
tracker.releases().head()                      # every release with its impact
fig = tracker.plot("path")                     # line + stacked contributions (Matplotlib)
fig = tracker.plot("path", backend="plotly")
```

## Options

| Argument | Default | Meaning |
|---|---|---|
| `calendar` | release delays of the data | `ReleaseCalendar` or delays defining the vintages |
| `start`, `end` | quarter of the target period | first and last vintage date |
| `dates` | release dates in `[start, end]` | explicit vintage dates instead |
| `by` | `"category"` | grouping of contributions: `"category"`, `"block"`, `"series"` |
| `refit` | `False` | re-estimate the parameters at every vintage (adds a re-estimation effect; slower) |

```python
monthly = res.nowcast_tracker(
    panel, calendar, "2024Q4",
    dates=["2024-10-01", "2024-11-01", "2024-12-01", "2025-01-01"],
    by="block",
)
monthly.to_frame()[["nowcast", "change", "news"]]
```

The functional form `nb.nowcast_tracker(model_or_results, data, calendar, period, ...)`
also accepts an **unfitted model** (with `target=`), which is estimated on the first
vintage.
