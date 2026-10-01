"""Loaders of the built-in datasets (plan section 7).

All loaders read files shipped with the package (no network access), verify their
SHA-256 digest against the metadata YAML the first time they are read in a process and
cache the parsed tables. The data are rebuilt from the primary sources by the scripts in
``scripts/build_datasets/`` of the source repository.

========================  ===============================================================
Loader                    Content
========================  ===============================================================
load_brazil_nowcast       ~100 Brazilian monthly/quarterly series (BCB, IBGE, IPEA), 2003-
load_brazil_calendar      Release delays of that panel + actual GDP release dates
load_brazil_vintages      Real-time vintages of GDP and components (IBGE releases, 2010-)
load_us_fred_md           FRED-MD (McCracken & Ng, 2016) + FRED-QD real GDP
load_nyfed                FRBNY Staff Nowcast replication panel (Bok et al., 2018)
load_us_grs_like          FRED-MD-based approximation of the Giannone et al. (2008) panel
load_simulated_dfm        Deterministic simulated mixed-frequency DFM with known parameters
========================  ===============================================================
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal, overload

import pandas as pd

from nowcastbox._logging import get_logger
from nowcastbox.core.exceptions import NowcastDataError
from nowcastbox.datasets._io import available_names, read_metadata, read_table
from nowcastbox.datasets._simulated import simulate_mixed_frequency_dfm
from nowcastbox.datasets.dataset import LEGEND_COLUMNS, Dataset
from nowcastbox.vintages.calendar import ReleaseCalendar
from nowcastbox.vintages.vintage_store import VintageStore

__all__ = [
    "dataset_info",
    "list_datasets",
    "load_brazil_calendar",
    "load_brazil_gdp_releases",
    "load_brazil_nowcast",
    "load_brazil_vintages",
    "load_dataset",
    "load_nyfed",
    "load_simulated_dfm",
    "load_us_fred_md",
    "load_us_grs_like",
]

logger = get_logger(__name__)

_PROVENANCE_KEYS = ("title", "description", "target", "series", "files", "summary")


# ---------------------------------------------------------------------- helpers
def _legend(meta: dict[str, Any]) -> pd.DataFrame:
    rows = meta.get("series") or []
    if not rows:
        raise NowcastDataError(f"Dataset {meta.get('name')!r} has no series legend.")
    legend = pd.DataFrame(rows)
    for col in LEGEND_COLUMNS:
        if col not in legend.columns:
            legend[col] = None
    return legend


def _wide(name: str, verify: bool) -> pd.DataFrame:
    table = read_table(name, "data", verify=verify)
    if "period" not in table.columns:
        raise NowcastDataError(f"Data file of {name!r} has no 'period' column.")
    index = pd.PeriodIndex(table.pop("period").astype(str), freq="M", name="period")
    table.index = index
    return table.astype(float)


def _provenance(meta: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in meta.items() if k not in _PROVENANCE_KEYS}


def _dataset(
    name: str,
    *,
    verify: bool,
    data_name: str | None = None,
    columns: Sequence[str] | None = None,
    start: object = None,
    end: object = None,
) -> Dataset:
    meta = read_metadata(name)
    source_meta = meta if data_name is None else read_metadata(data_name)
    legend = _legend(source_meta)
    frame = _wide(data_name or name, verify)
    if list(frame.columns) != list(legend["name"]):
        raise NowcastDataError(f"Data file and legend of {name!r} list different series.")
    ds = Dataset(
        name,
        frame,
        legend,
        title=str(meta.get("title", "")),
        description=str(meta.get("description", "")),
        target=meta.get("target"),
        metadata=_provenance(meta),
    )
    if columns is not None:
        ds = ds.select(columns)
    if start is not None or end is not None:
        ds = ds.truncate(start, end)
    return ds


# ---------------------------------------------------------------------- catalogue
def list_datasets() -> pd.DataFrame:
    """Catalogue of the built-in datasets.

    Returns
    -------
    pandas.DataFrame
        One row per dataset (index ``name``) with the columns ``title``, ``loader``,
        ``n_series``, ``start``, ``end``, ``target`` and ``license``.

    See Also
    --------
    dataset_info : Full metadata of one dataset.

    Examples
    --------
    >>> from nowcastbox.datasets import list_datasets
    >>> cat = list_datasets()
    >>> "brazil_nowcast" in cat.index and "load_nyfed" in set(cat["loader"])
    True
    """
    rows = []
    for name in available_names():
        meta = read_metadata(name)
        summary = dict(meta.get("summary") or {})
        summary.update(meta.get("default_sample") or {})
        rows.append(
            {
                "name": name,
                "title": meta.get("title", ""),
                "loader": meta.get("loader", ""),
                "n_series": summary.get("n_series", len(meta.get("series") or [])),
                "start": summary.get("start"),
                "end": summary.get("end"),
                "target": meta.get("target"),
                "license": meta.get("license", ""),
            }
        )
    return pd.DataFrame(rows).set_index("name")


def dataset_info(name: str) -> dict[str, Any]:
    """Full metadata of a dataset (provenance, license, citation, checksums, legend).

    Parameters
    ----------
    name : str
        Dataset name, as in :func:`list_datasets`.

    Returns
    -------
    dict
        Parsed metadata YAML (a copy).

    Raises
    ------
    ValueError
        If the dataset does not exist.

    Examples
    --------
    >>> from nowcastbox.datasets import dataset_info
    >>> dataset_info("nyfed")["files"]["data"]["path"]
    'nyfed.csv.gz'
    """
    return read_metadata(name)


def load_dataset(name: str, **kwargs: Any) -> Any:
    """Load a dataset by name (dispatches to its ``load_*`` function).

    Parameters
    ----------
    name : str
        Dataset name, as in :func:`list_datasets`.
    **kwargs
        Passed to the loader.

    Returns
    -------
    Dataset, ReleaseCalendar or VintageStore
        Whatever the loader returns.

    Raises
    ------
    ValueError
        If the dataset does not exist.

    Examples
    --------
    >>> from nowcastbox.datasets import load_dataset
    >>> load_dataset("nyfed").target
    'GDPC1'
    """
    loaders: dict[str, Callable[..., Any]] = {
        "brazil_nowcast": load_brazil_nowcast,
        "brazil_calendar": load_brazil_calendar,
        "brazil_vintages": load_brazil_vintages,
        "us_fred_md": load_us_fred_md,
        "nyfed": load_nyfed,
        "us_grs_like": load_us_grs_like,
        "simulated_dfm": load_simulated_dfm,
    }
    if name not in loaders:
        raise ValueError(f"Unknown dataset {name!r}; available datasets: {sorted(loaders)}.")
    return loaders[name](**kwargs)


# ---------------------------------------------------------------------- Brazil
def load_brazil_nowcast(
    *,
    columns: Sequence[str] | None = None,
    start: object = None,
    end: object = None,
    verify: bool = True,
) -> Dataset:
    """Brazilian nowcasting panel: ~100 monthly/quarterly series, 2003 to the latest release.

    National accounts (target ``"pib"``: seasonally adjusted GDP volume index, to be
    used as QoQ growth via its ``"qoq"`` transform), IBC-Br, industrial production
    (PIM-PF), retail (PMC) and services (PMS) surveys, labour market (PNAD Contínua,
    Caged), external trade, prices (IPCA, INPC, IPCA-15, IPP), money, credit, fiscal and
    financial variables, and BCB Focus survey expectations. Only open-license primary
    sources (BCB/SGS and Focus, IBGE/SIDRA, IPEADATA); values are in levels.

    Parameters
    ----------
    columns : sequence of str, optional
        Subset of series (default: all).
    start, end : period-like, optional
        Sample bounds (e.g. ``"2010-01"``, ``"2019Q4"``).
    verify : bool, default True
        Verify the SHA-256 digest of the data file on first load.

    Returns
    -------
    Dataset
        Panel, legend (description, source, code, frequency, transform, legacy code,
        delay in days, blocks ``global``/``real``/``nominal``/``financial``/``soft``,
        category, units) and provenance.

    Raises
    ------
    NowcastDataError
        Corrupted data file, unknown columns or an empty sample.

    See Also
    --------
    load_brazil_calendar : Release delays and GDP release dates.
    load_brazil_vintages : Real-time GDP vintages.

    Notes
    -----
    Delays are stylised typical publication lags. Seasonally adjusted series are used
    whenever the source publishes them; unadjusted ones get 12-month transformations.
    The series reflect the latest vintage at build time (see ``metadata["built"]``).

    Examples
    --------
    >>> from nowcastbox.datasets import load_brazil_nowcast
    >>> ds = load_brazil_nowcast()
    >>> ds.target, ds.frequency["pib"], ds.transform["pib"]
    ('pib', 'Q', 'qoq')
    >>> sorted(ds.block_names)
    ['financial', 'global', 'nominal', 'real', 'soft']
    """
    return _dataset("brazil_nowcast", verify=verify, columns=columns, start=start, end=end)


def load_brazil_gdp_releases(*, verify: bool = True) -> pd.DataFrame:
    """Release dates of the IBGE quarterly national accounts (GDP) since 2010Q1.

    Parameters
    ----------
    verify : bool, default True
        Verify the SHA-256 digest of the file on first load.

    Returns
    -------
    pandas.DataFrame
        Columns ``reference_quarter`` (Period), ``release_date`` (Timestamp),
        ``date_source`` (provenance of the date) and ``file`` (publication URL).

    Raises
    ------
    NowcastDataError
        Corrupted data file.

    Examples
    --------
    >>> from nowcastbox.datasets import load_brazil_gdp_releases
    >>> rel = load_brazil_gdp_releases()
    >>> str(rel["reference_quarter"].iloc[0])
    '2010Q1'
    """
    table = read_table("brazil_vintages", "releases", verify=verify)
    table["reference_quarter"] = pd.PeriodIndex(table["reference_quarter"], freq="Q")
    table["release_date"] = pd.to_datetime(table["release_date"])
    return table


@overload
def load_brazil_calendar(
    *, as_frame: Literal[False] = ..., verify: bool = ...
) -> ReleaseCalendar: ...


@overload
def load_brazil_calendar(*, as_frame: Literal[True], verify: bool = ...) -> pd.DataFrame: ...


def load_brazil_calendar(
    *, as_frame: bool = False, verify: bool = True
) -> ReleaseCalendar | pd.DataFrame:
    """Release calendar of :func:`load_brazil_nowcast`.

    Every series gets its typical publication delay (days after the end of the
    reference period); GDP and its components additionally get the **actual** IBGE
    release dates since 2010Q1 (explicit dates override the delay rule).

    Parameters
    ----------
    as_frame : bool, default False
        Return the delay table instead of a :class:`ReleaseCalendar`.
    verify : bool, default True
        Verify the SHA-256 digest of the release-date file on first load.

    Returns
    -------
    ReleaseCalendar or pandas.DataFrame
        Calendar, or a table indexed by series with columns ``frequency``,
        ``delay_days``, ``source``, ``explicit_dates`` (bool) and ``description``.

    Raises
    ------
    NowcastDataError
        Corrupted data file.

    See Also
    --------
    nowcastbox.vintages.pseudo_real_time : Apply the calendar to the panel.

    Examples
    --------
    >>> from nowcastbox.datasets import load_brazil_calendar
    >>> cal = load_brazil_calendar()
    >>> cal.release_date("pib", "2024Q1")
    Timestamp('2024-06-04 00:00:00')
    >>> int(cal.delays["ipca"])
    10
    """
    meta = read_metadata("brazil_calendar")
    legend = _legend(read_metadata("brazil_nowcast")).set_index("name")
    explicit = [s for s in meta.get("explicit_dates_for", []) if s in legend.index]
    if as_frame:
        out = legend[["frequency", "delay_days", "source", "description"]].copy()
        out["delay_days"] = pd.array(out["delay_days"], dtype="Int64")
        out.insert(3, "explicit_dates", out.index.isin(explicit))
        out.index.name = "series"
        return out
    releases = load_brazil_gdp_releases(verify=verify)
    dates = pd.DataFrame(
        {
            "series": [s for s in explicit for _ in range(len(releases))],
            "reference_period": list(releases["reference_quarter"]) * len(explicit),
            "release_date": list(releases["release_date"]) * len(explicit),
        }
    )
    delays = legend["delay_days"].dropna().astype(int)
    return ReleaseCalendar(
        delays, dates if explicit else None, frequencies=legend.loc[delays.index, "frequency"]
    )


@overload
def load_brazil_vintages(
    *, series: Sequence[str] | str | None = ..., as_frame: Literal[False] = ..., verify: bool = ...
) -> VintageStore: ...


@overload
def load_brazil_vintages(
    *, series: Sequence[str] | str | None = ..., as_frame: Literal[True], verify: bool = ...
) -> pd.DataFrame: ...


def load_brazil_vintages(
    *,
    series: Sequence[str] | str | None = None,
    as_frame: bool = False,
    verify: bool = True,
) -> VintageStore | pd.DataFrame:
    """Real-time vintages of Brazilian quarterly GDP and components (IBGE releases).

    Seasonally adjusted chained volume indices (1995 = 100) as printed in each IBGE
    quarterly national accounts release since 2010Q1: ``pib`` (GDP), ``pib_agropecuaria``,
    ``pib_industria``, ``pib_servicos``, ``va_pb``, ``consumo_familias``,
    ``consumo_governo``, ``fbcf``, ``exportacoes_cn`` and ``importacoes_cn`` (names match
    :func:`load_brazil_nowcast`). The vintage date of each value is the release date.

    Parameters
    ----------
    series : str or sequence of str, optional
        Subset of series (default: all).
    as_frame : bool, default False
        Return the long table instead of a :class:`VintageStore`.
    verify : bool, default True
        Verify the SHA-256 digest of the data file on first load.

    Returns
    -------
    VintageStore or pandas.DataFrame
        Store (``as_of``, ``revisions``, ``nth_release`` ...) or long table with
        columns ``series``, ``frequency``, ``reference_period``, ``vintage_date``,
        ``value`` (a row only when a value is first published or revised).

    Raises
    ------
    NowcastDataError
        Corrupted data file or unknown series.

    Notes
    -----
    Each publication prints a window of the history (13-15 years in recent releases), so
    values of older quarters keep their last printed vintage. The 2021Q2 publication is
    missing because its table omits the reference quarter; 2006Q4-2009Q4 releases are
    Word files whose tables could not be extracted.

    Examples
    --------
    >>> from nowcastbox.datasets import load_brazil_vintages
    >>> store = load_brazil_vintages(series="pib")
    >>> first = store.nth_release(0)["pib"]  # first release of each quarter
    >>> float(first.loc["2020-06"])  # 2020Q2, published 2020-09-01
    150.3
    """
    table = read_table("brazil_vintages", "data", verify=verify)
    if series is not None:
        wanted = [series] if isinstance(series, str) else list(series)
        unknown = sorted(set(wanted) - set(table["series"]))
        if unknown:
            raise NowcastDataError(f"Unknown series {unknown} in brazil_vintages.")
        table = table[table["series"].isin(wanted)].reset_index(drop=True)
    if as_frame:
        table["vintage_date"] = pd.to_datetime(table["vintage_date"])
        return table
    return VintageStore(table)


# ---------------------------------------------------------------------- United States
def load_us_fred_md(
    *,
    include_gdp: bool = True,
    start: object = None,
    end: object = None,
    verify: bool = True,
) -> Dataset:
    """FRED-MD monthly US database (McCracken & Ng, 2016) plus quarterly real GDP.

    About 120 monthly series from 1959 in levels with the official FRED-MD
    transformation codes mapped to named transforms (``fred_md_tcode`` legend column:
    1 ``level``, 2 ``diff``, 3 ``diff|diff``, 4 ``log``, 5 ``dlog``, 6
    ``log|diff|diff``, 7 ``pct_change|diff``), the FRED-MD group of each series as a
    block (plus ``global``) and real GDP (``GDPC1``) from FRED-QD in the third month of
    each quarter. Series under third-party copyright (S&P, Moody's, Michigan sentiment,
    VIX) are not shipped.

    Parameters
    ----------
    include_gdp : bool, default True
        Keep the quarterly ``GDPC1`` column (the default target).
    start, end : period-like, optional
        Sample bounds.
    verify : bool, default True
        Verify the SHA-256 digest of the data file on first load.

    Returns
    -------
    Dataset
        Panel, legend and provenance (``metadata["vintage"]`` gives the FRED-MD vintage).

    Raises
    ------
    NowcastDataError
        Corrupted data file or empty sample.

    Examples
    --------
    >>> from nowcastbox.datasets import load_us_fred_md
    >>> ds = load_us_fred_md()
    >>> ds.legend.loc["INDPRO", "transform"], int(ds.legend.loc["INDPRO", "fred_md_tcode"])
    ('dlog', 5)
    >>> ds.frequency["GDPC1"]
    'Q'
    """
    ds = _dataset("us_fred_md", verify=verify, start=start, end=end)
    if not include_gdp:
        ds = ds.select([c for c in ds.columns if c != "GDPC1"])
    return ds


def load_nyfed(*, model_only: bool = True, verify: bool = True) -> Dataset:
    """FRBNY Staff Nowcast replication panel (Bok et al., 2018).

    The public example data of the NY Fed nowcasting repository (vintage 2017-01-27,
    1985-01 onwards): monthly and quarterly US series with the NY Fed blocks
    (``global``, ``soft``, ``real``, ``labor``), transformations and categories.
    Target: real GDP (``GDPC1``).

    Parameters
    ----------
    model_only : bool, default True
        Keep only the series of the example model (``nyfed_model == 1``; 25 series) or
        all 29 series of the specification file.
    verify : bool, default True
        Verify the SHA-256 digest of the data file on first load.

    Returns
    -------
    Dataset
        Panel, legend (with ``nyfed_transformation``, ``nyfed_category`` and
        ``nyfed_model`` columns) and provenance.

    Raises
    ------
    NowcastDataError
        Corrupted data file.

    Notes
    -----
    NY Fed transformation codes are mapped to named transforms: ``lin`` -> ``level``,
    ``chg`` -> ``diff``, ``pch`` -> ``pct_change(1)|scale(100)`` and ``pca``
    (compounded annual rate) -> ``log|diff(1)|scale(400)``, its first-order
    approximation.

    Examples
    --------
    >>> from nowcastbox.datasets import load_nyfed
    >>> ds = load_nyfed()
    >>> ds.n_series, ds.target, ds.block_names
    (25, 'GDPC1', ['global', 'labor', 'real', 'soft'])
    """
    ds = _dataset("nyfed", verify=verify)
    if model_only:
        flags = ds.legend["nyfed_model"].astype(int)
        ds = ds.select([c for c in ds.columns if flags[c] == 1])
    return ds


_GRS_DEFAULT = object()


def load_us_grs_like(
    *,
    start: object = _GRS_DEFAULT,
    end: object = _GRS_DEFAULT,
    verify: bool = True,
) -> Dataset:
    """FRED-MD-based approximation of the Giannone, Reichlin & Small (2008) US panel.

    The original GRS (2008) replication files (about 200 monthly series and GDP,
    1982-2004) are not available from a primary source, so this loader provides a
    documented substitute: the FRED-MD series with their transformations, quarterly
    real GDP and stylised release delays, restricted by default to the GRS sample
    1982-01 to 2004-12. It is **not** the original panel (fewer series, current
    vintages, FRED-MD definitions).

    Parameters
    ----------
    start, end : period-like or None, optional
        Sample bounds; default 1982-01 and 2004-12. Pass ``None`` for no bound.
    verify : bool, default True
        Verify the SHA-256 digest of the data file on first load.

    Returns
    -------
    Dataset
        Panel, legend and provenance (``metadata["notes"]`` explains the approximation).

    Raises
    ------
    NowcastDataError
        Corrupted data file or empty sample.

    See Also
    --------
    load_us_fred_md : The full FRED-MD panel.

    Examples
    --------
    >>> from nowcastbox.datasets import load_us_grs_like
    >>> ds = load_us_grs_like()
    >>> str(ds.data.start), str(ds.data.end), ds.target
    ('1982-01', '2004-12', 'GDPC1')
    """
    sample = read_metadata("us_grs_like").get("default_sample") or {}
    first = sample.get("start") if start is _GRS_DEFAULT else start
    last = sample.get("end") if end is _GRS_DEFAULT else end
    return _dataset("us_grs_like", verify=verify, data_name="us_fred_md", start=first, end=last)


# ---------------------------------------------------------------------- simulated
def load_simulated_dfm(
    *,
    n_monthly: int = 20,
    n_factors: int = 2,
    n_periods: int = 240,
    start: str = "2000-01",
    ragged_edge: bool = True,
    random_state: int | None = 0,
) -> Dataset:
    """Deterministic simulated mixed-frequency DFM with known parameters.

    ``n_monthly`` monthly indicators ``x01``, ``x02``, ... load on ``n_factors``
    AR(1) factors; the quarterly target ``gdp`` aggregates its latent monthly
    counterpart with the Mariano-Murasawa weights (1, 2, 3, 2, 1)/3. With the default
    seed the data are identical on every platform. True parameters and latent paths are
    in ``ds.metadata["true_params"]``.

    Parameters
    ----------
    n_monthly : int, default 20
        Number of monthly indicators.
    n_factors : int, default 2
        Number of factors.
    n_periods : int, default 240
        Number of months (>= 24).
    start : str, default "2000-01"
        First month.
    ragged_edge : bool, default True
        Publication lags of 0-2 months for the indicators and an unobserved last
        quarter for the target. The legend delays reproduce the ragged edge at the
        information date ``end of sample + 5 days``.
    random_state : int or None, default 0
        Seed (``None``: non-deterministic).

    Returns
    -------
    Dataset
        Stationary simulated panel (transform ``level``), legend and true parameters.

    Raises
    ------
    ValueError
        On invalid sizes.

    Examples
    --------
    >>> from nowcastbox.datasets import load_simulated_dfm
    >>> ds = load_simulated_dfm(n_monthly=10, n_factors=1, n_periods=120)
    >>> ds.n_series, ds.target, ds.data.quarterly_columns
    (11, 'gdp', ['gdp'])
    >>> ds.metadata["true_params"]["loadings"].shape
    (11, 1)
    """
    data, truth = simulate_mixed_frequency_dfm(
        n_monthly,
        n_factors,
        n_periods,
        start=start,
        ragged_edge=ragged_edge,
        random_state=random_state,
    )
    meta = read_metadata("simulated_dfm")
    names = list(data.columns)
    legend = pd.DataFrame(
        {
            "name": names,
            "description": [
                f"Simulated monthly indicator {n}"
                if n != "gdp"
                else "Simulated quarterly target (Mariano-Murasawa aggregate of a latent "
                "monthly series)"
                for n in names
            ],
            "source": "nowcastbox simulation",
            "source_code": "",
            "frequency": ["Q" if n == "gdp" else "M" for n in names],
            "transform": "level",
            "legacy_code": "0",
            "delay_days": [int(truth["delay_days"][n]) for n in names],
            "blocks": "global",
            "category": "hard",
            "units": "standardised units",
        }
    )
    provenance = _provenance(meta)
    provenance["true_params"] = truth
    provenance["simulation"] = {
        "n_monthly": n_monthly,
        "n_factors": n_factors,
        "n_periods": n_periods,
        "start": start,
        "ragged_edge": ragged_edge,
        "random_state": random_state,
    }
    return Dataset(
        "simulated_dfm",
        data,
        legend,
        title=str(meta.get("title", "")),
        description=str(meta.get("description", "")),
        target="gdp",
        metadata=provenance,
    )
