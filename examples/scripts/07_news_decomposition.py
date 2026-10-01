"""Notebook 07 - News decomposition: why did the nowcast change?"""

# %% [markdown]
# # 07 · News decomposition: why did the nowcast change?
#
# A nowcast that moves without explanation is of little use to a policy committee.
# Bańbura & Modugno (2014), building on Bańbura, Giannone & Reichlin (2011), showed that
# in a linear Gaussian state-space model the revision of a nowcast between two data
# vintages $\Omega_v \subset \Omega_{v+1}$ is an exact weighted sum of **news**:
#
# $$
# \mathbb{E}[y_t \mid \Omega_{v+1}] - \mathbb{E}[y_t \mid \Omega_v]
# = \sum_{j \in I_{v+1}} b_j \left(x_j - \mathbb{E}[x_j \mid \Omega_v]\right),
# $$
#
# where the sum runs over the newly released observations $x_j$, the term in brackets
# is the **surprise** (actual minus what the model expected) and $b_j$ is a **weight**
# reflecting how informative the series is about the target given everything already
# known. When old observations are also revised, the change splits into a *revisions*
# effect and a *news* effect; when the parameters are re-estimated, a third
# *re-estimation* term appears. `nowcastbox` computes all three exactly (innovation I6)
# and aggregates the news by series, block or category (hard / soft / financial).

# %%
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(HERE.parent))

from utils import (
    brazil_panel,
    brazil_realtime_store,
    display,
    md,
    pct,
    setup,
    show,
    store_metadata,
)

import nowcastbox as nb

setup()

# %% [markdown]
# ## 1. Model and vintages
#
# The compact Brazilian panel (GDP + 21 monthly indicators) with **blocks**: every
# series loads on a global factor, and on a real, nominal, financial or soft (Focus
# survey) factor, according to the dataset metadata. Pseudo real-time vintages use the
# actual GDP release dates and typical delays of the indicators
# (`load_brazil_calendar`).

# %%
ds, panel = brazil_panel(start="2005-01")
calendar = nb.load_brazil_calendar()
display(panel.blocks.astype(int).sum().rename("series").to_frame().T)

old = nb.pseudo_real_time(panel, calendar=calendar, vintage="2026-08-15")
new = nb.pseudo_real_time(panel, calendar=calendar, vintage="2026-09-15")
model = nb.MixedFreqDFM(n_factors=1, factor_lags=1, blocks="data")
res = model.fit(new, target="pib")
print(res.summary(n_periods=4))

# %% [markdown]
# ## 2. One month of news: 15 August → 15 September 2026
#
# The target is GDP growth in 2026Q3. `res.news(old, new, period)` uses the parameters
# of `res` to compute the expectations under both information sets.

# %%
news = res.news(old, new, target_period="2026Q3")
print(news.summary())
assert news.check_identity()  # old + revisions + news + re-estimation == new

# %%
top = news.top_releases(8)
display(top[["series", "reference_period", "actual", "expected", "weight", "impact"]])

# %%
fig = news.plot("waterfall", by="category")
show(fig, "07_waterfall_category")

# %% [markdown]
# **Reading the table.** The `impact` column is `weight × (actual − expected)`. July
# manufacturing output came in well above what the model expected and carries a large
# weight, so it is the main upward driver of the 2026Q3 nowcast. August inflation was
# much lower than expected; its weight is negative, so a negative price surprise
# *raises* the GDP nowcast — in this reduced-form model lower inflation goes together
# with stronger real activity (real income gains in a disinflation driven by food and
# fuel prices). The Focus survey's expected GDP growth was revised down, pulling the
# nowcast down slightly: the *soft* contribution is negative. The weights are not
# causal effects; they are the regression coefficients of the target on the surprises.

# %%
fig = news.plot("waterfall", by="block")
show(fig, "07_waterfall_block")
display(news.to_frame("block"))

# %% [markdown]
# ## 3. News *and* revisions: a GDP release day
#
# On 1 September 2026 IBGE published 2026Q2 GDP and, as always, revised the previous
# quarters. With the real-time store (pseudo real-time indicators + real GDP vintages,
# notebook 06), the vintages before and after that release differ in two ways: a new
# observation (2026Q2) and revised old ones. The decomposition separates the two.

# %%
store = brazil_realtime_store(panel, calendar)
meta = store_metadata(panel)  # release delays, blocks and categories of the panel
before = store.as_of("2026-08-31", as_mixed=True, **meta)
after = store.as_of("2026-09-02", as_mixed=True, **meta)
res_rt = model.fit(after, target="pib")
release_day = res_rt.news(before, after, target_period="2026Q3")
print(release_day.summary())

# %%
revised = (after.to_native("pib") - before.to_native("pib")).dropna()
revised = revised[revised.abs() > 1e-12]
md(
    f"IBGE's release revised the growth of {len(revised)} past quarters "
    f"(largest change {100 * revised.abs().max():.2f} pp, in {revised.abs().idxmax()}); "
    f"the **revisions effect** on the 2026Q3 nowcast is "
    f"**{100 * release_day.revisions_effect:+.4f} pp** and the **news** "
    f"**{100 * release_day.news_effect:+.4f} pp**."
)

# %% [markdown]
# Both effects are tiny, and that is informative. The 2026Q2 GDP figure was close to
# what the model already expected from the monthly data of April–June, so its surprise
# is small even though its weight is large. Revisions of *past* quarters barely matter
# for the current quarter: in a factor model the factors of past quarters are pinned
# down by the monthly indicators, so a revision of old GDP growth mostly changes the
# idiosyncratic component of those quarters, which has little persistence. Revisions
# of the *monthly* indicators would matter more, but real vintages of the indicators
# are not shipped (the indicators are pseudo real time here; see the STATUS notes on
# IBC-Br vintages).

# %% [markdown]
# ## 4. Re-estimating the parameters
#
# In production the model is usually re-estimated when new data arrive. Passing the
# re-estimated results as `new_results` adds a re-estimation term, so that the
# decomposition still adds up exactly.

# %%
res_old = model.fit(old, target="pib")
with_refit = res_old.news(old, new, target_period="2026Q3", new_results=res)
display(with_refit.waterfall("category").rename("step").to_frame())

# %% [markdown]
# Re-estimating on one more month of data barely moves the parameters, so the
# re-estimation effect is small relative to the news: the information content of the
# releases, not parameter instability, drives the month-to-month revisions.

# %% [markdown]
# ## 5. The nowcast tracker: the whole quarter release by release
#
# `res.nowcast_tracker` replays the release calendar between two dates, creating a
# vintage at every release day and decomposing each step. Here: the 2026Q3 nowcast from
# the start of June (before the quarter began) to 1 October 2026.

# %%
tracker = res.nowcast_tracker(panel, calendar, "2026Q3", "2026-06-01", "2026-10-01")
assert tracker.check_identity()
fig = tracker.plot("path")
show(fig, "07_tracker")

# %%
path = tracker.to_frame()
big = path.loc[path["change"].abs().nlargest(4).index].sort_index()
display(big[["nowcast", "change", "n_releases"]])
releases = tracker.releases()
for date in big.index:
    day = releases[releases["vintage"] == date]
    md(f"**{date.date()}**: " + ", ".join(sorted(set(day["series"]))))

# %% [markdown]
# Three of the four largest moves happen on the release days of IBGE's industrial
# production survey (PIM-PF, early each month): hard data on activity in the quarter's
# own months, with large weights. In particular the June figure (released on 4 August)
# cut the nowcast by about half a percentage point and the July figure (4 September)
# recovered part of it. Financial prices and Focus expectations, released daily or
# weekly, cause frequent but small revisions.

# %% [markdown]
# ## 6. Contributions to the *level* of the nowcast
#
# News explains *changes*. The level of the nowcast can also be written as a baseline
# (the unconditional mean) plus a contribution of every series, since the smoothed
# target is linear in the data (innovation I6).

# %%
level = res.level_contributions(new, "2026Q3")
assert level.check_identity()
contrib = level.to_frame()
display(contrib.sort_values("contribution", key=abs, ascending=False).head(10))
fig = level.plot("bar")
show(fig, "07_level")
md(
    f"Nowcast {pct(level.nowcast)} = baseline {pct(level.baseline)} + the sum of the "
    "series contributions."
)

# %% [markdown]
# ### References
#
# * Bańbura, M., Giannone, D. & Reichlin, L. (2011). Nowcasting. *Oxford Handbook of
#   Economic Forecasting*, 193–224.
# * Bańbura, M. & Modugno, M. (2014). Maximum likelihood estimation of factor models on
#   datasets with arbitrary pattern of missing data. *Journal of Applied Econometrics*,
#   29(1), 133–160.
# * Bok, B., Caratelli, D., Giannone, D., Sbordone, A. M. & Tambalotti, A. (2018).
#   Macroeconomic nowcasting and forecasting with big data. *Annual Review of
#   Economics*, 10, 615–643.
