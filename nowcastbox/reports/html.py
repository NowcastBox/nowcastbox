"""Self-contained HTML nowcast report (Jinja2 template + embedded Plotly figures).

:class:`NowcastReport` turns a fitted :class:`~nowcastbox.core.results.NowcastResults`
(plus optional news decomposition, nowcast tracker, density quantiles and backtest
results, all as plain pandas objects) into one HTML file with the sections

``headline``
    The current nowcast, its period, prediction intervals and empirical error bands.
``path``
    Observed target, in-sample fit and out-of-sample estimates (chart + table), the
    fan chart when quantiles are available, the empirical error bands and the range of
    the nowcasts of alternative models (without one or two groups of indicators).
``news``
    News waterfall and table, nowcast tracker (placeholders when not supplied).
``data_flow``
    Ragged-edge heatmap, share of the nowcast period's data already released,
    per-series release status and the z-score heatmap of the indicators.
``diagnostics``
    Estimation summary, factor charts, EM convergence, backtest RMSFE and accuracy
    table (by horizon and sub-period, directional accuracy) and a placeholder for
    further model diagnostics.

By default plotly.js is embedded inline, so the file works offline (about 4.5 MB);
``plotlyjs="cdn"`` gives a small file that loads plotly.js from the CDN.
"""

from __future__ import annotations

import datetime as _dt
import math
from collections.abc import Sequence
from importlib import resources
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape
from markupsafe import Markup

from nowcastbox.__version__ import __version__
from nowcastbox._logging import get_logger
from nowcastbox.core.results import FactorResults, NowcastResults
from nowcastbox.visualization import (
    interval_levels,
    news_waterfall_table,
    plot_data_availability,
    plot_eigenvalues,
    plot_empirical_bands,
    plot_factors,
    plot_fan_chart,
    plot_forecast,
    plot_indicator_heatmap,
    plot_loadings,
    plot_loglikelihood,
    plot_news_waterfall,
    plot_nowcast_tracker,
    plot_rmsfe_by_horizon,
    release_table,
    released_share_table,
    rmsfe_frame,
)
from nowcastbox.visualization.themes import Theme, get_theme

__all__ = ["REPORT_SECTIONS", "NowcastReport", "nowcast_period"]

logger = get_logger(__name__)

REPORT_SECTIONS: tuple[tuple[str, str], ...] = (
    ("headline", "Headline nowcast"),
    ("path", "Nowcast path"),
    ("news", "News"),
    ("data_flow", "Data flow"),
    ("diagnostics", "Diagnostics"),
)
"""``(id, title)`` of the report sections, in order."""

DEFAULT_TEMPLATE = "nowcast_report.html"


def nowcast_period(results: NowcastResults) -> pd.Period | None:
    """First target period after the last observation (the current nowcast).

    Parameters
    ----------
    results : NowcastResults
        Fitted results.

    Returns
    -------
    pandas.Period or None
        The period, or ``None`` if the target is observed until the end.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from nowcastbox.core.results import NowcastResults, build_nowcast_frame
    >>> idx = pd.period_range("2020Q1", periods=3, freq="Q")
    >>> f = build_nowcast_frame(
    ...     pd.Series([1.0, np.nan, np.nan], index=idx), pd.Series([1.0, 2.0, 3.0], index=idx)
    ... )
    >>> str(nowcast_period(NowcastResults(target="y", nowcast=f)))
    '2020Q2'
    """
    observed = results.nowcast["observed"]
    last = observed.last_valid_index()
    index = results.nowcast.index
    later = index if last is None else index[index > last]
    return None if len(later) == 0 else later[0]


def _fmt(value: Any, digits: int = 3) -> str:
    """Format a table cell."""
    if value is None:
        return "-"
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return "-" if math.isnan(float(value)) else f"{float(value):,.{digits}f}"
    if value is pd.NA or value is pd.NaT:
        return "-"
    return str(value)


def _table(frame: pd.DataFrame, caption: str, *, digits: int = 3) -> dict[str, Any]:
    """Template representation of a DataFrame (cells are formatted strings)."""
    rows = [
        (_fmt(label), [_fmt(v, digits) for v in row])
        for label, row in zip(frame.index, frame.itertuples(index=False), strict=True)
    ]
    return {
        "caption": caption,
        "index_name": str(frame.index.name or ""),
        "columns": [str(c) for c in frame.columns],
        "rows": rows,
    }


def _str_index(frame: pd.DataFrame) -> pd.DataFrame:
    """Copy with the (period) index as text."""
    out = frame.copy()
    out.index = pd.Index([str(p) for p in out.index], name=out.index.name or "period")
    return out


def _flat_index(frame: pd.DataFrame) -> pd.DataFrame:
    """Copy with MultiIndex rows/columns joined into text labels (``"Covid | 1"``)."""
    out = frame.copy()
    if isinstance(out.index, pd.MultiIndex):
        name = " | ".join(str(n) for n in out.index.names if n is not None)
        labels = [" | ".join(str(v) for v in key) for key in out.index]
        out.index = pd.Index(labels, name=name)
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = pd.Index([" | ".join(str(v) for v in key) for key in out.columns])
    return out


class NowcastReport:
    """HTML report of a nowcast.

    Parameters
    ----------
    results : NowcastResults
        Fitted results (any model; factor charts appear for factor models).
    title : str, optional
        Report title (default ``"Nowcast report: <target>"``).
    news : pandas.DataFrame, pandas.Series or news-results object, optional
        News decomposition (see :mod:`nowcastbox.visualization.news` for the
        columns): drawn as a waterfall and listed in a table.
    old_nowcast, new_nowcast : float, optional
        Nowcast before/after the news (override attributes of a news object).
    news_group_by : str, optional
        Group the waterfall by this column (``"category"``, ``"block"``...).
    tracker : pandas.DataFrame or NowcastTracker, optional
        Nowcast tracker (:class:`~nowcastbox.news.NowcastTracker` or a wide/long table,
        see :func:`~nowcastbox.visualization.plot_nowcast_tracker`).
    quantiles : pandas.DataFrame or NowcastDistribution, optional
        Predictive quantiles for a fan chart, or a
        :class:`~nowcastbox.density.NowcastDistribution` (see
        :func:`~nowcastbox.visualization.plot_fan_chart`).
    backtest : pandas.DataFrame or BacktestResults, optional
        RMSFE by horizon and model (see
        :func:`~nowcastbox.visualization.rmsfe_frame`).
    diagnostics : DiagnosticsReport or bool, optional
        DFM diagnostics (innovation I9) shown in the diagnostics section: a
        :class:`~nowcastbox.diagnostics.DiagnosticsReport`, or ``True`` to run
        :func:`~nowcastbox.diagnostics.run_diagnostics` on ``results`` (failures are
        reported as a note).
    backtest_reference : str, optional
        Model used for relative RMSFE in the backtest chart.
    backtest_metrics : pandas.DataFrame, optional
        Accuracy table to show in the diagnostics section, e.g.
        ``backtest.metrics(metrics=("rmsfe", "fda", "n"), periods="covid")``.
    bands : EmpiricalGaussianDistribution or EmpiricalQuantileDistribution, optional
        Empirical error bands (:func:`~nowcastbox.density.empirical_bands`): headline
        tiles, a fan chart at the band levels and a table.
    alternatives : AlternativeNowcasts, optional
        Nowcasts of alternative models
        (:func:`~nowcastbox.experiment.alternative_models`): chart and range table.
    heatmap : IndicatorZScores or bool, optional
        Z-score heatmap of the indicators in the data-flow section: an
        :class:`~nowcastbox.diagnostics.IndicatorZScores`, or ``True`` to compute it from
        the results' data (target excluded).
    heatmap_last : int, default 24
        Number of most recent periods in the heatmap.
    released_share : bool, default True
        Show the share of the nowcast period's predictor data already released (by
        category) in the data-flow section.
    author : str, optional
        Shown in the header.
    notes : str, optional
        Free text shown under the headline.
    theme : Theme or str, optional
        Visual theme of the figures.
    n_periods : int, default 12
        Number of most recent target periods in the nowcast table.
    data_flow_periods : int, default 24
        Number of base periods in the data-availability heatmap.
    plotlyjs : {"inline", "cdn"}, default "inline"
        Embed plotly.js (offline, self-contained) or load it from the CDN.
    template_dir : str or pathlib.Path, optional
        Directory with a custom ``nowcast_report.html`` (receives the same context,
        see :meth:`context`).

    Raises
    ------
    TypeError
        If ``results`` is not a :class:`NowcastResults`.
    ValueError
        Invalid ``plotlyjs`` or ``n_periods``.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from nowcastbox.core.results import NowcastResults, build_nowcast_frame
    >>> idx = pd.period_range("2020Q1", periods=4, freq="Q")
    >>> f = build_nowcast_frame(
    ...     pd.Series([1.0, 2.0, 1.5, np.nan], index=idx),
    ...     pd.Series([1.1, 1.8, 1.6, 1.2], index=idx),
    ... )
    >>> report = NowcastReport(NowcastResults(target="gdp", nowcast=f), plotlyjs="cdn")
    >>> html = report.to_html()
    >>> 'id="headline"' in html and 'id="diagnostics"' in html
    True
    """

    def __init__(
        self,
        results: NowcastResults,
        *,
        title: str | None = None,
        news: Any = None,
        old_nowcast: float | None = None,
        new_nowcast: float | None = None,
        news_group_by: str | None = None,
        tracker: Any = None,
        quantiles: Any = None,
        backtest: Any = None,
        backtest_reference: str | None = None,
        backtest_metrics: pd.DataFrame | None = None,
        bands: Any = None,
        alternatives: Any = None,
        heatmap: Any = None,
        heatmap_last: int = 24,
        released_share: bool = True,
        diagnostics: Any = None,
        author: str | None = None,
        notes: str | None = None,
        theme: Theme | str | None = None,
        n_periods: int = 12,
        data_flow_periods: int = 24,
        plotlyjs: Literal["inline", "cdn"] = "inline",
        template_dir: str | Path | None = None,
    ) -> None:
        if not isinstance(results, NowcastResults):
            raise TypeError(f"results must be NowcastResults; got {type(results).__name__}.")
        if plotlyjs not in ("inline", "cdn"):
            raise ValueError(f"plotlyjs must be 'inline' or 'cdn'; got {plotlyjs!r}.")
        if n_periods < 1 or data_flow_periods < 1 or heatmap_last < 1:
            raise ValueError("n_periods, data_flow_periods and heatmap_last must be >= 1.")
        self.results = results
        self.title = title or f"Nowcast report: {results.target}"
        self.news = news
        self.old_nowcast = old_nowcast
        self.new_nowcast = new_nowcast
        self.news_group_by = news_group_by
        self.tracker = tracker
        self.quantiles = quantiles
        self.backtest = backtest
        self.backtest_reference = backtest_reference
        self.backtest_metrics = backtest_metrics
        self.bands = bands
        self.alternatives = alternatives
        self.heatmap = heatmap
        self.heatmap_last = heatmap_last
        self.released_share = released_share
        self.diagnostics = diagnostics
        self.author = author
        self.notes = notes
        self.theme = get_theme(theme)
        self.n_periods = n_periods
        self.data_flow_periods = data_flow_periods
        self.plotlyjs = plotlyjs
        self.template_dir = None if template_dir is None else Path(template_dir)
        self._js_included = False

    # ------------------------------------------------------------------ figures
    def _figure_html(self, fig: Any) -> Markup:
        """Embed a Plotly figure; plotly.js goes with the first figure only."""
        if self._js_included:
            include: bool | str = False
        else:
            include = True if self.plotlyjs == "inline" else "cdn"
            self._js_included = True
        html = fig.to_html(
            full_html=False,
            include_plotlyjs=include,
            config={"displaylogo": False, "responsive": True},
            default_width="100%",
        )
        return Markup(html)  # noqa: S704  # nosec B704 - HTML generated by plotly, not user input

    def _try_figure(self, func: Any, *args: Any, **kwargs: Any) -> Markup | str:
        """Build a figure; on a data problem return the reason as a placeholder text."""
        try:
            fig = func(*args, backend="plotly", theme=self.theme, **kwargs)
        except (ValueError, TypeError) as exc:
            logger.warning("Report figure %s skipped: %s", getattr(func, "__name__", func), exc)
            return f"Chart not available: {exc}"
        return self._figure_html(fig)

    @staticmethod
    def _collect(items: Sequence[Markup | str], section: dict[str, Any]) -> None:
        for item in items:
            if isinstance(item, Markup):
                section["figures"].append(item)
            else:
                section["placeholders"].append(item)

    # ------------------------------------------------------------------ sections
    @staticmethod
    def _section(sid: str) -> dict[str, Any]:
        title = dict(REPORT_SECTIONS)[sid]
        return {
            "id": sid,
            "title": title,
            "tiles": [],
            "text": "",
            "figures": [],
            "tables": [],
            "pre": "",
            "placeholders": [],
        }

    def _headline(self) -> dict[str, Any]:
        res = self.results
        sec = self._section("headline")
        period = nowcast_period(res)
        frame = res.nowcast
        if period is None:
            sec["placeholders"].append(
                "The target is observed until the end of the sample: no nowcast period."
            )
        else:
            row = frame.iloc[int(np.flatnonzero(frame.index == period)[0])]
            sec["tiles"].append(
                {"label": f"Nowcast {period}", "value": _fmt(res.get_nowcast(period)), "hero": True}
            )
            for lvl in interval_levels(frame):
                low, high = row[f"lower_{lvl}"], row[f"upper_{lvl}"]
                if pd.notna(low) and pd.notna(high):
                    sec["tiles"].append(
                        {
                            "label": f"{lvl}% interval",
                            "value": f"[{_fmt(low)}, {_fmt(high)}]",
                            "hero": False,
                        }
                    )
            if "std" in frame.columns and pd.notna(row["std"]):
                sec["tiles"].append(
                    {"label": "Std. deviation", "value": _fmt(row["std"]), "hero": False}
                )
            sec["tiles"] += self._band_tiles(period)
        last = frame["observed"].last_valid_index()
        if last is not None:
            sec["tiles"].append(
                {
                    "label": f"Last observed ({last})",
                    "value": _fmt(frame["observed"].dropna().iloc[-1]),
                    "hero": False,
                }
            )
        old = self._news_level("old_nowcast", self.old_nowcast)
        if old is not None and period is not None:
            change = res.get_nowcast(period) - float(old)
            sec["tiles"].append(
                {"label": "Change vs previous", "value": f"{change:+.3f}", "hero": False}
            )
        sec["text"] = self.notes or ""
        return sec

    def _band_tiles(self, period: pd.Period) -> list[dict[str, Any]]:
        """Headline tiles of the empirical error bands at the nowcast period."""
        if self.bands is None:
            return []
        tiles = []
        for level in self.bands.levels:
            band = self.bands.interval(level)
            labels = [str(p) for p in band.index]
            if str(period) not in labels:
                return []
            row = band.iloc[labels.index(str(period))]
            tiles.append(
                {
                    "label": f"{100 * level:g}% empirical band",
                    "value": f"[{_fmt(row['lower'])}, {_fmt(row['upper'])}]",
                    "hero": False,
                }
            )
        return tiles

    def _news_level(self, name: str, explicit: float | None) -> float | None:
        if explicit is not None:
            return explicit
        value = getattr(self.news, name, None) if self.news is not None else None
        return None if value is None else float(value)

    def _path(self) -> dict[str, Any]:
        sec = self._section("path")
        items = [self._try_figure(plot_forecast, self.results)]
        if self.quantiles is not None:
            items.append(
                self._try_figure(
                    plot_fan_chart, self.quantiles, observed=self.results.observed.dropna()
                )
            )
        if self.bands is not None:
            items.append(
                self._try_figure(
                    plot_empirical_bands, self.bands, observed=self.results.observed.dropna()
                )
            )
        if self.alternatives is not None:
            items.append(self._try_figure(self.alternatives.plot))
        self._collect(items, sec)
        cols = [c for c in self.results.nowcast.columns if c != "common"]
        table = self.results.nowcast[cols].iloc[-self.n_periods :].copy()
        table.index = pd.Index([str(p) for p in table.index], name="period")
        sec["tables"].append(_table(table, f"Last {len(table)} periods"))
        if self.bands is not None:
            sec["tables"].append(_table(_str_index(self.bands.to_frame()), "Empirical error bands"))
        if self.alternatives is not None:
            rng = self.alternatives.range().drop(columns=["min_model", "max_model"])
            sec["tables"].append(
                _table(_str_index(rng), "Nowcasts of alternative models (without 1-2 groups)")
            )
        return sec

    def _news_section(self) -> dict[str, Any]:
        sec = self._section("news")
        if self.news is None and self.tracker is None:
            sec["placeholders"].append(
                "No news decomposition supplied. Pass news= (impacts by release) and/or "
                "tracker= (nowcast by vintage) to NowcastReport to fill this section."
            )
            return sec
        if self.news is not None:
            kwargs = {
                "group_by": self.news_group_by,
                "old_nowcast": self.old_nowcast,
                "new_nowcast": self.new_nowcast,
            }
            self._collect([self._try_figure(plot_news_waterfall, self.news, **kwargs)], sec)
            try:
                bars = news_waterfall_table(self.news, top_n=None, **kwargs)
            except (ValueError, TypeError) as exc:
                sec["placeholders"].append(f"News table not available: {exc}")
            else:
                shown = bars.set_index("label")[["value", "start", "end"]]
                shown.index.name = "release"
                sec["tables"].append(_table(shown, "Impacts on the nowcast"))
        if self.tracker is not None:
            self._collect([self._try_figure(plot_nowcast_tracker, self.tracker)], sec)
        return sec

    def _data_flow(self) -> dict[str, Any]:
        sec = self._section("data_flow")
        if self.results.data is None:
            sec["placeholders"].append("The results do not carry their estimation data.")
            return sec
        self._collect(
            [
                self._try_figure(
                    plot_data_availability, self.results, n_periods=self.data_flow_periods
                )
            ],
            sec,
        )
        self._released_share(sec)
        sec["tables"].append(_table(release_table(self.results), "Release status by series"))
        self._heatmap(sec)
        return sec

    def _released_share(self, sec: dict[str, Any]) -> None:
        """Share of the nowcast period's predictor data already released, by category."""
        period = nowcast_period(self.results)
        data = self.results.data
        if not self.released_share or period is None or data is None:
            return
        predictors = [c for c in data.columns if c != self.results.target]
        try:
            share = released_share_table(data, period, by="category", series=predictors)
        except (ValueError, TypeError) as exc:
            sec["placeholders"].append(f"Released-data share not available: {exc}")
            return
        sec["tables"].append(_table(share, f"Share of the {period} data already released"))

    def _heatmap(self, sec: dict[str, Any]) -> None:
        """Z-score heatmap of the indicators (requested with ``heatmap=``)."""
        if self.heatmap is None or self.heatmap is False:
            return
        source: Any = self.heatmap
        kwargs: dict[str, Any] = {"last": self.heatmap_last}
        if source is True:
            data = self.results.data
            source = data
            kwargs["series"] = [c for c in data.columns if c != self.results.target]  # type: ignore[union-attr]
        self._collect([self._try_figure(plot_indicator_heatmap, source, **kwargs)], sec)

    def _diagnostics(self) -> dict[str, Any]:
        res = self.results
        sec = self._section("diagnostics")
        info = pd.DataFrame(
            {
                "value": [
                    res.model_name or type(res).__name__,
                    res.n_factors,
                    res.loglikelihood,
                    res.n_iter,
                    res.converged,
                ]
            },
            index=pd.Index(
                ["Model", "Factors", "Log-likelihood", "Iterations", "Converged"], name="statistic"
            ),
        )
        sec["tables"].append(_table(info, "Estimation"))
        if res.model_params:
            params = pd.DataFrame(
                {"value": [str(v) for v in res.model_params.values()]},
                index=pd.Index([str(k) for k in res.model_params], name="parameter"),
            )
            sec["tables"].append(_table(params, "Model parameters"))
        items: list[Markup | str] = []
        if isinstance(res, FactorResults) and res.factors is not None:
            items.append(self._try_figure(plot_factors, res))
        if isinstance(res, FactorResults) and res.loadings is not None:
            items.append(self._try_figure(plot_loadings, res))
        if isinstance(res, FactorResults):
            items.append(self._try_figure(plot_eigenvalues, res))
        if getattr(res, "loglikelihood_path", None) is not None:
            items.append(self._try_figure(plot_loglikelihood, res))
        if self.backtest is not None:
            items.append(
                self._try_figure(
                    plot_rmsfe_by_horizon, self.backtest, relative_to=self.backtest_reference
                )
            )
            try:
                sec["tables"].append(_table(rmsfe_frame(self.backtest), "RMSFE by horizon"))
            except (ValueError, TypeError) as exc:
                sec["placeholders"].append(f"Backtest table not available: {exc}")
        if self.backtest_metrics is not None:
            sec["tables"].append(_table(_flat_index(self.backtest_metrics), "Backtest accuracy"))
        self._collect(items, sec)
        sec["pre"] = res.summary()
        self._dfm_diagnostics(sec)
        return sec

    def _dfm_diagnostics(self, sec: dict[str, Any]) -> None:
        """Loading stability, contributions and data-quality alerts (I9)."""
        diag = self.diagnostics
        if diag is None or diag is False:
            sec["placeholders"].append(
                "Pass diagnostics=True (or a DiagnosticsReport from "
                "nowcastbox.diagnostics.run_diagnostics) to add loading stability, factor "
                "contributions and data-quality alerts."
            )
            return
        if diag is True:
            from nowcastbox.diagnostics import run_diagnostics

            try:
                diag = run_diagnostics(self.results, warn=False)
            except (ValueError, TypeError) as exc:  # NowcastDataError is a ValueError
                sec["placeholders"].append(f"Diagnostics not available: {exc}")
                return
        overview = diag.series_overview()
        if not overview.empty:
            sec["tables"].append(_table(overview, "DFM diagnostics by series"))
        sec["pre"] = f"{sec['pre']}\n\n{diag.summary()}"

    # ------------------------------------------------------------------ output
    def context(self) -> dict[str, Any]:
        """Template context: ``title``, ``meta``, ``theme`` and ``sections``.

        Each section is a dict with ``id``, ``title``, ``tiles`` (``label``/``value``/
        ``hero``), ``text``, ``figures`` (HTML), ``tables`` (``caption``,
        ``index_name``, ``columns``, ``rows``), ``pre`` and ``placeholders``.

        Returns
        -------
        dict
            Context passed to the Jinja2 template.

        Examples
        --------
        >>> import pandas as pd
        >>> from nowcastbox.core.results import NowcastResults, build_nowcast_frame
        >>> idx = pd.period_range("2020Q1", periods=2, freq="Q")
        >>> f = build_nowcast_frame(
        ...     pd.Series([1.0, None], index=idx), pd.Series([1.0, 2.0], index=idx)
        ... )
        >>> ctx = NowcastReport(NowcastResults(target="y", nowcast=f), plotlyjs="cdn").context()
        >>> [s["id"] for s in ctx["sections"]]
        ['headline', 'path', 'news', 'data_flow', 'diagnostics']
        """
        self._js_included = False
        res = self.results
        data = res.data
        sections = [
            self._headline(),
            self._path(),
            self._news_section(),
            self._data_flow(),
            self._diagnostics(),
        ]
        meta = {
            "version": __version__,
            "model_name": res.model_name or type(res).__name__,
            "target": res.target,
            "sample": f"{data.start} - {data.end}" if data is not None else "",
            "generated": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "author": self.author or "",
        }
        return {
            "title": self.title,
            "meta": meta,
            "theme": self.theme,
            "sections": sections,
            "plotly_head": "",
        }

    def _environment(self) -> Environment:
        if self.template_dir is not None:
            directory = self.template_dir
        else:
            directory = Path(str(resources.files("nowcastbox.reports") / "templates"))
        return Environment(
            loader=FileSystemLoader(str(directory)),
            autoescape=select_autoescape(default=True, default_for_string=True),
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def render(self) -> str:
        """Render the report to an HTML string.

        Returns
        -------
        str
            Complete HTML document.

        Examples
        --------
        >>> import pandas as pd
        >>> from nowcastbox.core.results import NowcastResults, build_nowcast_frame
        >>> idx = pd.period_range("2020Q1", periods=2, freq="Q")
        >>> f = build_nowcast_frame(
        ...     pd.Series([1.0, None], index=idx), pd.Series([1.0, 2.0], index=idx)
        ... )
        >>> NowcastReport(NowcastResults(target="y", nowcast=f), plotlyjs="cdn").render()[:15]
        '<!doctype html>'
        """
        template = self._environment().get_template(DEFAULT_TEMPLATE)
        return template.render(**self.context())

    def to_html(self, path: str | Path | None = None) -> str:
        """Render the report and optionally write it to ``path``.

        Parameters
        ----------
        path : str or pathlib.Path, optional
            Output file (parent directories are created).

        Returns
        -------
        str
            The HTML document.

        Examples
        --------
        >>> report.to_html("nowcast_report.html")  # doctest: +SKIP
        """
        html = self.render()
        if path is not None:
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(html, encoding="utf-8")
            logger.info("Nowcast report written to %s", target)
        return html

    def __repr__(self) -> str:
        return f"NowcastReport(target={self.results.target!r}, title={self.title!r})"
