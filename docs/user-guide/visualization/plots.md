# Plots

Every chart is available with two backends:

- `backend="plotly"` (default): interactive figures for notebooks and HTML reports
  (`fig.show()`, `fig.write_html("chart.html")`);
- `backend="matplotlib"`: static figures for papers (`fig.savefig("chart.pdf")`).

and takes a `theme=` (`"default"`, `"academic"`, `"presentation"` or your own `Theme`).

## Results plots

Fitted results draw themselves with `results.plot(kind)`:

| `kind` | Results | Shows |
|---|---|---|
| `"forecast"` | all | observed target, in-sample fit, out-of-sample nowcasts with 68/90 % bands |
| `"fan"` | with a `std` column | fan chart from Gaussian quantiles |
| `"density"` | with a `std` column | fan chart of `results.distribution(...)` (accepts `n_boot`; with `method="empirical", backtest=bt` the [empirical error bands](../density/empirical-bands.md) at 57.5/68/90 %) |
| `"data_availability"` / `"ragged_edge"` | with data | heatmap of observed / missing / ragged-edge cells by series |
| `"released_share"` | with data | [share of a target period's data already released](../data/released-share.md) |
| `"indicator_heatmap"` | with data | [z-score heatmap of the indicators](indicator-heatmap.md) |
| `"factors"` | factor models | smoothed factors (forecast part shaded) |
| `"eigenvalues"` | factor models | scree plot of the panel correlation matrix |
| `"loadings"` | factor models | loadings heatmap (`style="heatmap"`) or bars (`style="bar"`) |
| `"loglikelihood"` | `MixedFreqDFM` | EM log-likelihood path |

```python
import nowcastbox as nb

ds = nb.load_simulated_dfm()
res = nb.MixedFreqDFM(n_factors=2).fit(ds.data, "gdp")

nb.core.available_plots(type(res))
fig = res.plot("forecast")
fig = res.plot("forecast", backend="matplotlib", theme="academic")
fig = res.plot("data_availability")
fig = res.plot("loadings", style="bar")
fig = res.plot("loglikelihood")
```

## Analysis plots

News, trackers, level contributions, factor selection and backtests have their own
`plot` methods or functions in `nowcastbox.visualization`:

```python
ic = nb.select_factors(ds.data.drop(["gdp"]), rmax=8)
fig = nb.visualization.plot_factor_selection(ic)           # Plotly IC curves
fig = ic.plot("criteria")                                  # Matplotlib version

old = ds.data.as_of("2019-11-01")
new = ds.data.as_of("2020-01-10")
news = res.news(old, new, "2019Q4")
fig = news.plot("waterfall")                               # Matplotlib (default)
fig = nb.visualization.plot_news_waterfall(news, group_by="category", backend="plotly")

table = nb.visualization.release_table(res)                # last release of each series
table.head()
```

| Function | Input |
|---|---|
| `plot_forecast`, `plot_fan_chart` | results, a nowcast frame, quantiles or a `NowcastDistribution` |
| `plot_empirical_bands` | output of `nowcastbox.density.empirical_bands` (fan at the band levels) |
| `plot_indicator_heatmap`, `heatmap_table` | `IndicatorZScores`, `MixedFrequencyData` or results |
| `plot_released_share`, `released_share_table` | `MixedFrequencyData` or results and a target period |
| `plot_factors`, `plot_loadings`, `plot_eigenvalues` | factor results or frames |
| `plot_data_availability`, `release_table` | `MixedFrequencyData` or results |
| `plot_news_waterfall`, `plot_nowcast_tracker` | `NewsResults` / `NowcastTracker` or tidy tables |
| `plot_factor_selection` | `FactorSelectionResult` |
| `plot_rmsfe_by_horizon` | `BacktestResults` or an RMSFE table |
| `plot_loglikelihood` | `MixedFreqDFMResults` |

## Themes

```python
from dataclasses import replace

nb.visualization.list_themes()
base = nb.visualization.get_theme("default")
brand = replace(base, name="brand", out_of_sample_color="#b5179e", font_size=13)
nb.visualization.register_theme(brand, overwrite=True)
fig = res.plot("forecast", theme="brand")
```

`set_default_theme("academic")` changes the default of every chart. The default palette
has a fixed order validated for colour-vision deficiencies; observed / in-sample /
out-of-sample roles keep the same colours across all charts.

## Custom plots

Register your own kind for a results class:

```python
import matplotlib.pyplot as plt
from nowcastbox.core.results import NowcastResults, register_plot


@register_plot("residual_hist")
def plot_residual_hist(results, backend="matplotlib", **kwargs):
    frame = results.nowcast.dropna(subset=["observed", "in_sample"])
    fig, ax = plt.subplots()
    ax.hist(frame["observed"] - frame["in_sample"], bins=15)
    ax.set_title(f"In-sample errors: {results.target}")
    return fig


fig = res.plot("residual_hist")
```
