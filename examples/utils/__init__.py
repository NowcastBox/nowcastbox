"""Shared helpers for the nowcastbox example notebooks and scripts.

Every notebook (``examples/notebooks/*.ipynb``) and its script twin
(``examples/scripts/*.py``) starts with::

    import sys
    from pathlib import Path

    HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    sys.path.insert(0, str(HERE.parent))  # the examples/ folder

    from utils import setup

    setup()

The helpers keep the examples short and deterministic:

* :func:`setup` fixes the seeds, limits BLAS threads, silences the expected
  convergence warnings of short demo fits and applies a static Matplotlib style (static
  images render on GitHub; Plotly is used sparingly);
* :func:`show` displays a Matplotlib figure inline (notebook) or saves it under
  ``examples/outputs/figures`` (script) and closes it;
* :func:`display` / :func:`md` render tables and Markdown in both contexts;
* :func:`brazil_panel` and :func:`brazil_core_columns` build the reduced Brazilian
  panels used by several notebooks.

Examples
--------
>>> from utils import pct
>>> pct(0.0123)
'1.23%'
"""

from __future__ import annotations

import os
import random
import sys
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "BRAZIL_CORE",
    "EXAMPLES_DIR",
    "OUTPUTS_DIR",
    "SEED",
    "brazil_core_columns",
    "brazil_panel",
    "brazil_realtime_store",
    "display",
    "fetch_live_panel",
    "gdp_growth_vintages",
    "in_notebook",
    "md",
    "network_enabled",
    "pct",
    "setup",
    "show",
    "store_metadata",
]

EXAMPLES_DIR = Path(__file__).resolve().parent.parent
"""The ``examples/`` folder."""

OUTPUTS_DIR = EXAMPLES_DIR / "outputs"
"""Where the examples write their files (reports, snapshots, figures, CSVs)."""

SEED = 20260101
"""Seed used by every example (``random``, NumPy and the ``random_state`` arguments)."""

_FIGURE_COUNTER = {"n": 0}


def in_notebook() -> bool:
    """Whether the code runs inside a Jupyter kernel.

    Returns
    -------
    bool
        ``True`` inside an IPython kernel (notebook), ``False`` in a plain script.

    Examples
    --------
    >>> in_notebook()
    False
    """
    try:
        from IPython import get_ipython
    except ImportError:  # pragma: no cover - IPython ships with Jupyter
        return False
    shell = get_ipython()
    return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"


def network_enabled() -> bool:
    """Whether the examples may access the internet.

    Network cells run only when the environment variable ``NOWCASTBOX_EXAMPLES_NETWORK``
    is ``1`` (the same opt-in philosophy as the ``network`` pytest marker); otherwise
    they fall back to the data shipped with the package, so every notebook runs offline
    and in CI.

    Returns
    -------
    bool
        ``True`` if ``NOWCASTBOX_EXAMPLES_NETWORK=1``.

    Examples
    --------
    >>> import os
    >>> os.environ.pop("NOWCASTBOX_EXAMPLES_NETWORK", None) is None or True
    True
    >>> network_enabled()
    False
    """
    return os.environ.get("NOWCASTBOX_EXAMPLES_NETWORK", "0").strip() == "1"


def setup(*, blas_threads: int | None = 4, seed: int = SEED, quiet: bool = True) -> None:
    """Configure a reproducible, quiet environment for the examples.

    Parameters
    ----------
    blas_threads : int or None, default 4
        Maximum number of BLAS/OpenMP threads (``threadpoolctl``); ``None`` leaves the
        library defaults. Limiting threads makes timings reproducible and avoids the
        oversubscription seen on WSL with many cores.
    seed : int, default :data:`SEED`
        Seed of :mod:`random` and of NumPy's legacy global generator.
    quiet : bool, default True
        Ignore :class:`~nowcastbox.core.exceptions.ConvergenceWarning` and
        :class:`~nowcastbox.core.exceptions.DataQualityWarning` (the demo fits use
        small ``max_iter`` on purpose; the notebooks discuss the warnings where
        relevant) and the ``FutureWarning``/``RuntimeWarning`` noise of third-party
        libraries.

    Examples
    --------
    >>> setup(blas_threads=None)
    """
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - legacy global seed for third-party code
    if blas_threads is not None:
        try:
            from threadpoolctl import threadpool_limits

            threadpool_limits(blas_threads)
        except ImportError:  # pragma: no cover - threadpoolctl comes with scikit-learn
            pass
    if quiet:
        from nowcastbox.core.exceptions import ConvergenceWarning, DataQualityWarning

        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=DataQualityWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
    pd.set_option("display.width", 120)
    pd.set_option("display.max_columns", 12)
    pd.set_option("display.precision", 4)
    import matplotlib

    if not in_notebook():
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.figsize": (9.0, 4.2),
            "figure.dpi": 90,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.bbox": "tight",
        }
    )


def show(fig: Any, name: str | None = None) -> None:
    """Show a figure inline (notebook) or save it (script), then close it.

    Parameters
    ----------
    fig : matplotlib.figure.Figure or plotly.graph_objects.Figure
        Figure returned by a ``plot`` method.
    name : str, optional
        File stem used when running as a script
        (``examples/outputs/figures/<name>.png``). Defaults to a running counter.

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> show(fig, name="doctest_figure")
    """
    is_plotly = fig.__class__.__module__.startswith("plotly")
    if not is_plotly and not hasattr(fig, "savefig") and hasattr(fig, "figure"):
        fig = fig.figure  # some plot methods return the Matplotlib Axes
    if in_notebook():
        if is_plotly:
            fig.show()
        else:
            from IPython.display import display as ipy_display

            ipy_display(fig)
    else:
        _FIGURE_COUNTER["n"] += 1
        stem = name or f"figure_{_FIGURE_COUNTER['n']:02d}"
        folder = OUTPUTS_DIR / "figures"
        folder.mkdir(parents=True, exist_ok=True)
        if is_plotly:
            fig.write_html(folder / f"{stem}.html", include_plotlyjs="cdn")
        else:
            fig.savefig(folder / f"{stem}.png")
    if not is_plotly:
        import matplotlib.pyplot as plt

        plt.close(fig)


def display(*objects: Any) -> None:
    """Display objects with rich output in a notebook, ``print`` them in a script.

    Parameters
    ----------
    *objects
        Objects to display (DataFrames are rendered as HTML tables in notebooks).

    Examples
    --------
    >>> display(1, "two")
    1
    two
    """
    if in_notebook():
        from IPython.display import display as ipy_display

        ipy_display(*objects)
    else:
        for obj in objects:
            print(obj)


def md(text: str) -> None:
    """Render Markdown in a notebook (plain text in a script).

    Parameters
    ----------
    text : str
        Markdown text (usually an f-string with computed numbers).

    Examples
    --------
    >>> md("**bold**")
    **bold**
    """
    if in_notebook():
        from IPython.display import Markdown
        from IPython.display import display as ipy_display

        ipy_display(Markdown(text))
    else:
        print(text)


def pct(value: float, digits: int = 2) -> str:
    """Format a growth rate stored as a fraction (``0.0123``) as a percentage.

    Parameters
    ----------
    value : float
        Fraction (the Brazilian GDP target is a quarter-on-quarter growth rate).
    digits : int, default 2
        Decimal places.

    Returns
    -------
    str
        ``"1.23%"``; ``"n/a"`` for NaN.

    Examples
    --------
    >>> pct(-0.004, 1)
    '-0.4%'
    """
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{100 * value:.{digits}f}%"


BRAZIL_CORE: tuple[str, ...] = (
    "pib",  # target: GDP volume index, SA (quarterly; QoQ growth after transformation)
    # hard data: activity
    "ibc_br",
    "pim_geral",
    "pim_transformacao",
    "pim_bens_capital",
    "pim_bens_consumo_duraveis",
    "pmc_varejo",
    "pmc_ampliado",
    "ipea_fbcf",
    "ipea_consumo_aparente_industria",
    "energia_consumo_industria",
    "anp_diesel",
    "arrecadacao_federal",
    "secex_exportacoes",
    "secex_importacoes_bk",
    "ipca",
    # financial
    "selic",
    "cambio_ptax",
    "icbr",
    "credito_saldo_total",
    # soft: Focus survey expectations
    "focus_pib",
    "focus_ipca_12m",
)
"""Compact Brazilian panel used by several notebooks: GDP + 21 monthly indicators
covering industry, retail, investment, energy, fuel, taxes, trade, prices, financial
conditions and survey expectations, all observed since 2003."""


def brazil_core_columns() -> list[str]:
    """The compact Brazilian panel used by the examples (:data:`BRAZIL_CORE`).

    Returns
    -------
    list of str
        Column names of :func:`nowcastbox.load_brazil_nowcast`, target first.

    Examples
    --------
    >>> cols = brazil_core_columns()
    >>> cols[0], len(cols)
    ('pib', 22)
    """
    return list(BRAZIL_CORE)


def brazil_panel(
    columns: Sequence[str] | None = None,
    start: str = "2005-01",
    end: str | None = None,
    *,
    prepare: bool = True,
    **prepare_kwargs: Any,
) -> tuple[Any, Any]:
    """Load (a subset of) the shipped Brazilian panel and make it stationary.

    Parameters
    ----------
    columns : sequence of str, optional
        Series to keep (default :data:`BRAZIL_CORE`).
    start, end : str, optional
        Sample bounds (monthly periods).
    prepare : bool, default True
        Apply :func:`nowcastbox.prepare_panel` (legend transformations, outliers,
        interior gaps).
    **prepare_kwargs
        Extra options for :func:`nowcastbox.prepare_panel`.

    Returns
    -------
    dataset : nowcastbox.Dataset
        The selected dataset (levels, legend, calendar).
    panel : nowcastbox.MixedFrequencyData
        Stationary panel (or the levels when ``prepare=False``).

    Examples
    --------
    >>> ds, panel = brazil_panel(["pib", "ibc_br", "pim_geral"], start="2015-01")
    >>> panel.columns
    ['pib', 'ibc_br', 'pim_geral']
    """
    import nowcastbox as nb

    full = nb.load_brazil_nowcast()
    cols = list(columns) if columns is not None else brazil_core_columns()
    ds = full.select(cols).truncate(start, end)
    if not prepare:
        return ds, ds.data
    return ds, nb.prepare_panel(ds.data, ds.transform, **prepare_kwargs)


def gdp_growth_vintages(series: str = "pib") -> pd.DataFrame:
    """Real-time vintages of quarter-on-quarter growth from the IBGE index vintages.

    :func:`nowcastbox.load_brazil_vintages` stores each release of the GDP volume index
    (levels, rounded to one decimal as published). The growth rate of a quarter in a
    given vintage is computed from that vintage's own index, so the records capture
    the revisions of the growth rates.

    Parameters
    ----------
    series : str, default "pib"
        Series of the vintage store.

    Returns
    -------
    pandas.DataFrame
        Records with columns ``series``, ``reference_period`` (quarterly periods),
        ``vintage_date`` and ``value`` (growth as a fraction), ready for
        :class:`nowcastbox.VintageStore`.

    Examples
    --------
    >>> rec = gdp_growth_vintages()
    >>> list(rec.columns)
    ['series', 'reference_period', 'vintage_date', 'value']
    """
    import nowcastbox as nb

    matrix = nb.load_brazil_vintages().vintage_matrix(series)
    growth = matrix.pct_change(fill_method=None)
    records = growth.stack().rename("value").reset_index().dropna()  # noqa: PD013
    records.insert(0, "series", series)
    return records[["series", "reference_period", "vintage_date", "value"]]


def brazil_realtime_store(panel: Any, calendar: Any = None) -> Any:
    """Vintage store mixing pseudo real-time indicators and **real** GDP vintages.

    The monthly indicators are cut by the release calendar (pseudo real time: their
    revisions are not available), while the target ``pib`` takes the growth rates of
    the actual IBGE releases (:func:`gdp_growth_vintages`, from June 2010).

    Parameters
    ----------
    panel : nowcastbox.MixedFrequencyData
        Stationary panel containing ``pib`` (QoQ growth).
    calendar : nowcastbox.ReleaseCalendar, optional
        Default :func:`nowcastbox.load_brazil_calendar` (actual GDP release dates).

    Returns
    -------
    nowcastbox.VintageStore
        The combined store.

    Examples
    --------
    >>> _, panel = brazil_panel(["pib", "ibc_br"], start="2010-01")
    >>> store = brazil_realtime_store(panel)
    >>> sorted(store.series)
    ['ibc_br', 'pib']
    """
    import nowcastbox as nb

    calendar = nb.load_brazil_calendar() if calendar is None else calendar
    pseudo = nb.VintageStore.from_calendar(panel, calendar)
    others = pseudo.records[pseudo.records["series"] != "pib"]
    records = pd.concat([others, gdp_growth_vintages("pib")], ignore_index=True)
    return nb.VintageStore(records, frequencies=panel.frequencies)


def store_metadata(panel: Any) -> dict[str, Any]:
    """Metadata keyword arguments that rebuild a panel's metadata from a vintage store.

    :meth:`nowcastbox.VintageStore.as_of` returns values only; pass these keywords
    (``store.as_of(date, as_mixed=True, **store_metadata(panel))``, or
    ``PseudoRealTimeBacktest(metadata=store_metadata(panel))``) to keep the release
    delays, blocks and categories of ``panel``.

    Parameters
    ----------
    panel : nowcastbox.MixedFrequencyData
        Panel whose metadata to copy.

    Returns
    -------
    dict
        ``release_delays``, ``blocks`` and ``categories``.

    Examples
    --------
    >>> _, panel = brazil_panel(["pib", "ibc_br"], start="2015-01")
    >>> sorted(store_metadata(panel))
    ['blocks', 'categories', 'release_delays']
    """
    return {
        "release_delays": {k: int(v) for k, v in panel.release_delays.dropna().items()},
        "blocks": panel.blocks,
        "categories": {k: v.value for k, v in panel.categories.dropna().items()},
    }


def _parse_sidra_code(code: str) -> dict[str, Any]:
    """``"t8888/v12607/c544=129314"`` -> keyword arguments of ``fetch_sidra``."""
    out: dict[str, Any] = {"classifications": {}}
    for token in code.split("/"):
        if token.startswith("t"):
            out["table"] = int(token[1:])
        elif token.startswith("v"):
            out["variable"] = int(token[1:])
        elif token.startswith("c") and "=" in token:
            key, value = token.split("=", 1)
            out["classifications"][key] = int(value)
    return out


def _fetch_one(name: str, source: str, code: str, start: str, timeout: float) -> pd.Series:
    """Download one legend series in levels on the monthly grid (raises on failure)."""
    from nowcastbox import data_sources as dsrc

    common = {"start": start, "base_frequency": "M", "timeout": timeout, "max_retries": 2}
    if source == "BCB/SGS":
        return dsrc.fetch_sgs({name: int(code)}, **common)[name]
    if source == "IBGE/SIDRA":
        frame = dsrc.fetch_sidra(**_parse_sidra_code(code), name=name, dash_as=np.nan, **common)
        return frame.iloc[:, 0].rename(name)
    if source == "IPEADATA":
        return dsrc.fetch_ipeadata({name: code}, **common)[name]
    raise NotImplementedError(f"no connector for source {source!r}")


def fetch_live_panel(
    dataset: Any, *, start: str = "2003-01-01", timeout: float = 30.0
) -> tuple[Any, pd.DataFrame]:
    """Download the latest data of a Brazilian dataset, falling back to shipped values.

    Each series of ``dataset`` is downloaded from its primary source (BCB/SGS,
    IBGE/SIDRA or IPEADATA, using the codes of the dataset legend). Series whose
    download fails, or whose source has no connector (the Focus survey, served by the
    BCB Olinda API), keep the values shipped with the package.

    Parameters
    ----------
    dataset : nowcastbox.Dataset
        A (subset of) :func:`nowcastbox.load_brazil_nowcast`.
    start : str, default "2003-01-01"
        First date requested.
    timeout : float, default 30.0
        Per-request timeout in seconds.

    Returns
    -------
    panel : nowcastbox.MixedFrequencyData
        Levels on a monthly grid extended to the latest observation, with the
        dataset's metadata.
    status : pandas.DataFrame
        Per series: ``source``, ``origin`` (``"live"`` or ``"shipped"``), the reason
        of a fallback and the last observation.

    Examples
    --------
    >>> import nowcastbox as nb
    >>> ds = nb.load_brazil_nowcast().select(["pib", "ibc_br"])
    >>> panel, status = fetch_live_panel(ds)  # doctest: +SKIP
    """
    import nowcastbox as nb

    legend = dataset.legend
    shipped = dataset.data.data
    columns: dict[str, pd.Series] = {}
    rows = {}
    for name, row in legend.iterrows():
        try:
            series = _fetch_one(str(name), row["source"], row["source_code"], start, timeout)
            origin, reason = "live", ""
        except Exception as err:  # network, API changes, missing connector
            series, origin, reason = shipped[name], "shipped", f"{type(err).__name__}: {err}"
        columns[str(name)] = series.dropna()
        rows[name] = {"source": row["source"], "origin": origin, "fallback reason": reason[:80]}
    end = max([s.index.max() for s in columns.values() if len(s)] + [shipped.index[-1]])
    index = pd.period_range(shipped.index[0], end, freq="M")
    frame = pd.DataFrame({k: v.reindex(index) for k, v in columns.items()}, index=index)
    panel = nb.MixedFrequencyData(frame, metadata=dataset.data.metadata)
    status = pd.DataFrame(rows).T
    status["last observation"] = panel.last_observed()
    return panel, status


if __name__ == "__main__":  # pragma: no cover
    sys.exit(0)
