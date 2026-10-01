"""HTML reports (Jinja2 + Plotly), following the ``panelbox.report`` pattern.

:class:`NowcastReport` renders a self-contained HTML report of a nowcast with the
sections headline, nowcast path, news, data flow and diagnostics::

    from nowcastbox.reports import NowcastReport

    NowcastReport(results, news=news_table).to_html("report.html")
"""

from nowcastbox.reports.html import REPORT_SECTIONS, NowcastReport, nowcast_period

__all__ = ["REPORT_SECTIONS", "NowcastReport", "nowcast_period"]
