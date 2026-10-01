"""Notebook 02 - Data preparation: mixed frequencies, transformations, outliers, ragged edges."""

# %% [markdown]
# # 02 · Data preparation
#
# A nowcasting panel is messy by construction:
#
# * **mixed frequencies** — GDP is quarterly, most indicators are monthly;
# * **non-stationarity** — indices in levels must be turned into growth rates or
#   differences before a factor model can be estimated (Stock & Watson, 2002);
# * **outliers** — strikes, droughts, tax changes and, above all, the 2020 pandemic;
# * **missing values** — series that start late, interior gaps and the **ragged edge**:
#   at any date, the most recent observations of each series are missing because of
#   different publication delays (Giannone, Reichlin & Small, 2008).
#
# This notebook walks through the tools that handle each problem: the
# `MixedFrequencyData` container, the transformation codes, the outlier and missing-value
# tools of `nowcastbox.preprocessing`, and `prepare_panel`, which chains them.

# %%
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(HERE.parent))

import numpy as np
import pandas as pd
from utils import SEED, display, md, setup, show

import nowcastbox as nb
from nowcastbox import preprocessing as pp

setup()

# %% [markdown]
# ## 1. The `MixedFrequencyData` contract
#
# Every estimator consumes a `MixedFrequencyData`: a `DataFrame` on a contiguous
# **monthly** `PeriodIndex` plus per-series metadata. A quarterly value lives in the
# **last month** of its quarter (March, June, September, December); the other two
# months are structural `NaN`. Building one from scratch takes a DataFrame and the
# native frequencies; release delays (in days after the end of the reference period)
# make pseudo real-time vintages possible.

# %%
rng = np.random.default_rng(SEED)
idx = pd.period_range("2024-01", periods=9, freq="M")
toy = pd.DataFrame(
    {
        "ip": 100 + rng.standard_normal(9).cumsum(),  # monthly industrial production
        "gdp": [np.nan, np.nan, 1.0, np.nan, np.nan, 1.2, np.nan, np.nan, np.nan],
    },
    index=idx,
)
toy_data = nb.MixedFrequencyData(
    toy,
    frequencies={"ip": "M", "gdp": "Q"},
    release_delays={"ip": 40, "gdp": 60},
)
display(toy_data.metadata_frame()[["frequency", "release_delay"]])
display(toy_data.to_native("gdp"))  # the quarterly series on its own quarterly index

# %% [markdown]
# A value in a month that is not a quarter end, or an `inf`, is rejected with an
# explicit `NowcastDataError` rather than silently accepted:

# %%
bad = toy.copy()
bad.loc[pd.Period("2024-02", "M"), "gdp"] = 0.7  # February is not a quarter end
try:
    nb.MixedFrequencyData(bad, frequencies={"ip": "M", "gdp": "Q"})
except nb.NowcastDataError as err:
    print(f"NowcastDataError: {err}")

# %% [markdown]
# ## 2. A real panel: the shipped Brazilian dataset
#
# `load_brazil_nowcast()` returns a `Dataset`: values **in levels** as published, plus a
# legend with source codes, frequencies, the stationarity transformation, the typical
# publication delay, factor blocks and category (hard/soft/financial). We keep eight
# representative series for the illustrations.

# %%
cols = [
    "pib",  # GDP volume index, SA (quarterly) - the target
    "ibc_br",  # BCB economic activity index (monthly GDP proxy)
    "pim_geral",  # industrial production
    "pmc_varejo",  # retail sales
    "ipca",  # consumer prices
    "selic",  # policy rate
    "energia_consumo_total",  # electricity consumption (not seasonally adjusted)
    "focus_pib",  # Focus survey: expected GDP growth
]
ds = nb.load_brazil_nowcast().select(cols).truncate("2010-01", None)
print(ds.summary())
display(ds.legend[["description", "frequency", "transform", "delay_days", "category"]])

# %% [markdown]
# ## 3. Transformations
#
# Each legend row carries a **named** transformation; the numeric codes 0–7 used by the
# R package `nowcasting` (`Bpanel`) are also accepted. Every transformation is an
# invertible object, so nowcasts of growth rates can be mapped back to levels.

# %%
codes = pd.DataFrame(
    {
        "code": list(pp.TRANSFORM_CODES),
        "transformation": [str(pp.transform_from_code(c)) for c in pp.TRANSFORM_CODES],
    }
).set_index("code")
named = pd.Series({k: str(pp.get_transform(k)) for k in ["dlog", "qoq", "yoy", "dlog_yoy"]})
display(codes, named.rename("named transformation").to_frame())

# %% [markdown]
# Seasonally adjusted activity indices enter as monthly log-differences (`dlog`), GDP as
# quarter-on-quarter growth (`qoq`), the non-seasonally-adjusted electricity consumption
# as a 12-month log-difference (`dlog_yoy`, which removes seasonality), rates in first
# differences. `apply_transforms` applies them and records the fact in the metadata
# (`transform_applied=True`), so applying them twice is a no-op.

# %%
stationary = nb.apply_transforms(ds.data, ds.transform)
display(stationary.data.tail(4))
print(stationary.metadata["ibc_br"].transform, stationary.metadata["ibc_br"].transform_applied)

# %%
back = nb.invert_transforms(stationary, ds.data)  # needs the history in levels
err = (back.data["ibc_br"] - ds.data.data["ibc_br"]).abs().max()
print(f"max |levels - inverted(transformed)| for ibc_br: {err:.2e}")

# %% [markdown]
# ## 4. Outliers
#
# `detect_outliers` flags observations farther than `threshold` inter-quartile ranges
# from the median (the rule used by Giannone et al. and the R package, applied here
# from the literature). On the Brazilian panel the flags concentrate in the pandemic
# (March-September 2020), with an earlier episode in May-June 2018: the nationwide
# truckers' strike that paralysed industry for ten days and the rebound that followed.

# %%
flags = pp.detect_outliers(stationary, threshold=4.0)
flagged = flags.stack()  # noqa: PD013
flagged = flagged[flagged]
display(flagged.reset_index().groupby("period")["level_1"].apply(list).rename("series"))

# %%
import matplotlib.pyplot as plt

clean = pp.replace_outliers(stationary, threshold=4.0, window=3, replacement="moving_median")
fig, ax = plt.subplots()
raw = stationary.data["pim_geral"].loc["2018-01":"2022-12"]
fixed = clean.data["pim_geral"].loc["2018-01":"2022-12"]
ax.plot(raw.index.to_timestamp(), raw.to_numpy(), label="dlog(industrial production)")
ax.plot(fixed.index.to_timestamp(), fixed.to_numpy(), ls="--", label="outliers replaced")
ax.set_title("Outlier replacement by a centred moving median (April-May 2020)")
ax.legend()
show(fig, "02_outliers")

# %% [markdown]
# **Should the pandemic be "cleaned"?** Replacing the April 2020 collapse by a moving
# median protects the factor loadings from a handful of extreme points, but it also
# throws away the largest common shock in the sample. For the *estimation* of a DFM this
# is usually a good trade-off; for nowcasting *during* such an episode it is not.
# Notebook 10 compares this pre-cleaning with the robust alternatives built into the EM
# (Student-t errors, outlier detection inside the EM, pandemic dummies; innovation I3).

# %% [markdown]
# ## 5. Missing values and the ragged edge
#
# Three kinds of missing data matter: **leading** (a series starts late), **interior**
# (gaps), and the **ragged edge** at the end of the sample. The ragged edge is
# information, not a problem to be fixed: it is what makes nowcasting a real-time
# exercise. The Kalman filter handles it exactly; preprocessing should only fill
# interior gaps.

# %%
display(stationary.last_observed().rename("last observation").to_frame())

# %%
fig = nb.visualization.plot_data_availability(stationary, n_periods=18, backend="matplotlib")
show(fig, "02_ragged_edge")

# %% [markdown]
# On 1 October 2026 the policy rate and the Focus survey are known for September,
# consumer prices for August, the activity indicators for July and GDP only for the
# second quarter. The *same* panel seen on an earlier date is obtained with `as_of`,
# which keeps an observation only if its release date (end of the reference period +
# delay) is on or before the vintage date:

# %%
earlier = stationary.as_of("2026-06-15")
display(
    pd.DataFrame(
        {"2026-10-01 (today)": stationary.last_observed(), "2026-06-15": earlier.last_observed()}
    )
)

# %% [markdown]
# ## 6. `prepare_panel`: everything at once
#
# `prepare_panel` applies the transformations, replaces outliers, fills **interior**
# gaps by splines (never the ragged edge, unless asked), drops series with more than a
# third of missing values and returns a report. It mirrors the role of `Bpanel` in the R
# package `nowcasting` (de Valk et al., 2019), with named transformations and an
# explicit audit trail.

# %%
full = nb.load_brazil_nowcast()
panel, report = nb.prepare_panel(full.data, full.transform, return_report=True)
md(
    f"`prepare_panel` kept **{panel.n_series}** of {full.n_series} series and dropped "
    f"{len(report.dropped)} with more than 1/3 missing values: "
    + ", ".join(f"`{c}`" for c in report.dropped)
)
summary = report.to_frame()
display(summary.sort_values("n_outliers", ascending=False).head(8))

# %% [markdown]
# The dropped series (unemployment from the continuous PNAD, CAGED, credit concessions,
# ...) start in 2012 or later. They are informative about the recent past, so a modeller
# may prefer to keep them with `max_na_prop=1.0` — the EM estimator handles arbitrary
# missing patterns (Bańbura & Modugno, 2014), while the two-step estimator needs a
# balanced block for its principal components.

# %% [markdown]
# ## 7. Temporal aggregation (Mariano & Murasawa, 2003)
#
# Quarterly GDP growth is not observed monthly, but it can be written as a weighted sum
# of an unobserved monthly growth rate. With $y_t$ the monthly growth of the latent
# monthly GDP (in logs), the quarterly growth is approximately
#
# $$y^Q_t \approx \tfrac{1}{3}\left(y_t + 2y_{t-1} + 3y_{t-2} + 2y_{t-3} + y_{t-4}\right).$$
#
# These triangular weights enter the measurement equation of the EM model as linear
# restrictions on the loadings of quarterly series.

# %%
print("Mariano-Murasawa weights:", pp.mariano_murasawa_weights(3, normalize=True))
monthly_to_q = nb.month_to_quarter(ds.data.data["ibc_br"], how="mean")
display(monthly_to_q.tail(4).rename("ibc_br, quarterly average").to_frame())

# %% [markdown]
# ### References
#
# * Bańbura, M. & Modugno, M. (2014). *Journal of Applied Econometrics*, 29(1), 133–160.
# * de Valk, S., de Mattos, D. & Ferreira, P. (2019). Nowcasting: an R package for
#   predicting economic variables using dynamic factor models. *The R Journal*, 11(1).
# * Giannone, D., Reichlin, L. & Small, D. (2008). *Journal of Monetary Economics*,
#   55(4), 665–676.
# * Mariano, R. S. & Murasawa, Y. (2003). A new coincident index of business cycles
#   based on monthly and quarterly series. *Journal of Applied Econometrics*, 18(4),
#   427–443.
# * Stock, J. H. & Watson, M. W. (2002). Forecasting using principal components from a
#   large number of predictors. *JASA*, 97(460), 1167–1179.
