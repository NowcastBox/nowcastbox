r"""Contribution of each series to the *level* of the nowcast (innovation I6).

For a given information set the Kalman smoother is linear in the (standardised)
observations, so the nowcast of the target is

.. math::

    \hat y_\tau = o + s\Bigl(c + \sum_{i} \sum_{t \in \mathcal{O}_i} w_{i,t}\, z_{i,t}\Bigr),

with :math:`z_{i,t}` the standardised observations of series :math:`i`,
:math:`w_{i,t}` the smoother weights (Koopman & Harvey, 2003; Bańbura & Modugno,
2014), :math:`c` the contribution of the initial state and :math:`(o, s)` the map to
original units. The **contribution of series** :math:`i` is
:math:`s\sum_t w_{i,t} z_{i,t}` and the **baseline** is :math:`o + s c` (the
unconditional mean of the target for :class:`~nowcastbox.models.MixedFreqDFM`, the bridge
intercept for :class:`~nowcastbox.models.TwoStepDFM`). Contributions are computed
exactly, one smoother pass per series, by keeping the observation pattern fixed and
zeroing the other series.

References
----------
Koopman, S. J., & Harvey, A. (2003). Computing observation weights for signal
extraction and filtering. *Journal of Economic Dynamics and Control*, 27(7),
1317-1333.

Bańbura, M., & Modugno, M. (2014). *Journal of Applied Econometrics*, 29(1), 133-160.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from nowcastbox.core.results import NowcastResults
from nowcastbox.news._model import linear_model, position, to_frame, to_period
from nowcastbox.news.plotting import bar_chart, plot_news_object, register_news_plot

__all__ = ["LevelContributions", "level_contributions"]

_GROUPINGS = ("series", "block", "category")


@dataclass(frozen=True, kw_only=True, eq=False)
class LevelContributions:
    """Decomposition of the level of a nowcast into series contributions.

    Parameters
    ----------
    target : str
        Target series.
    target_period : pandas.Period
        Target period.
    nowcast : float
        Nowcast (original units).
    baseline : float
        Part of the nowcast not due to any observation (mean / intercept and initial
        state).
    contributions : pandas.DataFrame
        Index = model series; columns ``contribution`` (target units), ``n_obs``
        (observations used), ``category`` and ``block``. ``baseline +
        contributions["contribution"].sum() == nowcast``.

    Examples
    --------
    >>> lc = level_contributions(res, data, "2020Q2")  # doctest: +SKIP
    >>> lc.to_frame("category")  # doctest: +SKIP
    """

    target: str
    target_period: pd.Period
    nowcast: float
    baseline: float
    contributions: pd.DataFrame

    def to_frame(self, by: str = "series") -> pd.DataFrame:
        """Contributions aggregated by series, block or category.

        Parameters
        ----------
        by : {"series", "block", "category"}, default "series"
            Grouping.

        Returns
        -------
        pandas.DataFrame
            Column ``contribution`` indexed by group, plus a ``"baseline"`` row; the
            column sums to :attr:`nowcast`.

        Raises
        ------
        ValueError
            On an unknown grouping.

        Examples
        --------
        >>> lc.to_frame("block")  # doctest: +SKIP
        """
        if by not in _GROUPINGS:
            raise ValueError(f"by must be one of {_GROUPINGS}, got {by!r}.")
        frame = self.contributions
        if by == "series":
            grouped = frame["contribution"]
        else:
            grouped = frame.groupby(frame[by], sort=False)["contribution"].sum()
        out = grouped.to_frame("contribution")
        out.loc["baseline"] = self.baseline
        out.index.name = by
        return out

    def check_identity(self, atol: float = 1e-8) -> bool:
        """Whether baseline plus contributions reproduce the nowcast.

        Parameters
        ----------
        atol : float, default 1e-8
            Absolute tolerance.

        Returns
        -------
        bool
            True when the decomposition closes.

        Examples
        --------
        >>> lc.check_identity()  # doctest: +SKIP
        True
        """
        total = self.baseline + float(self.contributions["contribution"].sum())
        return bool(abs(total - self.nowcast) <= atol)

    def summary(self, by: str = "series") -> str:
        """Formatted text summary.

        Parameters
        ----------
        by : {"series", "block", "category"}, default "series"
            Grouping.

        Returns
        -------
        str
            Multi-line summary.

        Examples
        --------
        >>> print(lc.summary())  # doctest: +SKIP
        """
        width = 60
        lines = [
            "=" * width,
            f"Level contributions: {self.target} {self.target_period}",
            "=" * width,
            f"  {'Nowcast':<30}{self.nowcast:>14.6f}",
            "-" * width,
        ]
        for label, value in self.to_frame(by)["contribution"].items():
            lines.append(f"  {str(label)[:29]:<30}{value:>14.6f}")
        lines.append("=" * width)
        return "\n".join(lines)

    def plot(self, kind: str = "bar", **kwargs: Any) -> Any:
        """Plot the contributions through the news plot registry.

        Parameters
        ----------
        kind : str, default "bar"
            Plot kind.
        **kwargs
            Passed to the plotting function (``by``, ``ax``, ``title``...).

        Returns
        -------
        object
            Figure.

        Examples
        --------
        >>> lc.plot("bar", by="category")  # doctest: +SKIP
        """
        return plot_news_object(self, kind, **kwargs)


@register_news_plot("bar", LevelContributions)
def _plot_bar(lc: LevelContributions, by: str = "series", **kwargs: Any) -> Any:
    """Default matplotlib bar chart of the level contributions."""
    kwargs.setdefault("title", f"{lc.target} {lc.target_period}: contributions to the level")
    return bar_chart(lc.to_frame(by)["contribution"], **kwargs)


def level_contributions(
    results: NowcastResults,
    data: object = None,
    target_period: object = None,
    *,
    categories: Mapping[str, object] | None = None,
) -> LevelContributions:
    """Contribution of each series to the level of the nowcast.

    Parameters
    ----------
    results : NowcastResults
        :class:`~nowcastbox.models.MixedFreqDFM` or :class:`~nowcastbox.models.TwoStepDFM`
        results (contributions of the filtered predictors for ``aggregate="variables"``).
    data : MixedFrequencyData or pandas.DataFrame, optional
        Information set (default: the estimation panel stored in ``results``).
    target_period : period-like, optional
        Target period (default: the first period after the last target observation in
        ``data``).
    categories : mapping of str to category, optional
        Override of the series categories.

    Returns
    -------
    LevelContributions
        Baseline and per-series contributions (exact: they add up to the nowcast).

    Raises
    ------
    ValueError
        If no data are given and none are stored in ``results``.

    Examples
    --------
    >>> import nowcastbox as nb
    >>> from nowcastbox.news import level_contributions
    >>> from nowcastbox.models.two_step import simulate_two_step_example
    >>> data = simulate_two_step_example(random_state=0)
    >>> res = nb.MixedFreqDFM(n_factors=1, max_iter=20).fit(data, "gdp")
    >>> lc = level_contributions(res)
    >>> lc.check_identity()
    True
    """
    lin = linear_model(results, categories)
    panel = results.data if data is None else data
    if panel is None:
        raise ValueError("No data given and none stored in the results.")
    frame = to_frame(panel)
    if target_period is None:
        freq = lin.target_frequency.pandas_freq
        observed = frame[lin.target].dropna() if lin.target in frame.columns else frame.iloc[:0]
        last = observed.index[-1] if len(observed) else frame.index[-1]
        period = last.asfreq(freq) + (1 if len(observed) else 0)
    else:
        period = to_period(target_period, lin.target_frequency)
    slot = lin.target_slot(period)
    grid = lin.grid(slot, frame)
    pos = position(grid, slot)
    values = lin.standardize(frame, grid)
    pattern = ~np.isnan(values)
    filled = np.where(pattern, values, 0.0)
    n = len(lin.series)
    data = np.zeros((*pattern.shape, n + 2))
    for i in range(n):
        data[:, i, i + 1] = filled[:, i]
    data[:, :, n + 1] = filled
    out = lin.evaluate_many(pattern, data, pos)
    base = float(out[0])
    rows = [
        {
            "series": name,
            "contribution": lin.scale * float(out[i + 1] - base),
            "n_obs": int(pattern[:, i].sum()),
            "category": lin.categories[name],
            "block": lin.blocks[name],
        }
        for i, name in enumerate(lin.series)
    ]
    contributions = pd.DataFrame(rows).set_index("series")
    return LevelContributions(
        target=lin.target,
        target_period=period,
        nowcast=lin.to_original(float(out[n + 1])),
        baseline=lin.to_original(base),
        contributions=contributions,
    )
