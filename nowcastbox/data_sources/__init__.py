"""Connectors to public data APIs.

BCB/SGS, IBGE/SIDRA, IPEADATA and (optional) FRED/ALFRED, with a local on-disk cache.
Every ``fetch_*`` function returns ``float64`` pandas objects indexed by a
:class:`pandas.PeriodIndex` at the series' native frequency, or — with
``base_frequency="M"`` — on a monthly grid where lower-frequency values sit in the
last month of their period (the :class:`~nowcastbox.core.data.MixedFrequencyData`
convention).

Examples
--------
>>> from nowcastbox.data_sources import fetch_sgs
>>> panel = fetch_sgs(
...     {"ipca": 433, "gdp": 22099}, start="2010-01", base_frequency="M"
... )  # doctest: +SKIP
"""

from nowcastbox.data_sources.bcb import (
    SGS_URL,
    SGS_WINDOW_YEARS,
    fetch_sgs,
    fetch_sgs_series,
    sgs_windows,
)
from nowcastbox.data_sources.cache import (
    DEFAULT_TTL,
    DiskCache,
    default_cache_dir,
    get_default_cache,
    resolve_cache,
    set_default_cache,
)
from nowcastbox.data_sources.exceptions import DataSourceError, HTTPStatusError, MissingAPIKeyError
from nowcastbox.data_sources.fred import (
    FRED_API_KEY_ENV,
    FRED_URL,
    fetch_fred,
    fetch_fred_series,
    get_fred_api_key,
)
from nowcastbox.data_sources.ibge import (
    SIDRA_URL,
    fetch_sidra,
    fetch_sidra_raw,
    parse_sidra_period,
    sidra_path,
)
from nowcastbox.data_sources.ipea import (
    IPEADATA_URL,
    fetch_ipeadata,
    fetch_ipeadata_metadata,
    fetch_ipeadata_series,
)

__all__ = [
    "DEFAULT_TTL",
    "FRED_API_KEY_ENV",
    "FRED_URL",
    "IPEADATA_URL",
    "SGS_URL",
    "SGS_WINDOW_YEARS",
    "SIDRA_URL",
    "DataSourceError",
    "DiskCache",
    "HTTPStatusError",
    "MissingAPIKeyError",
    "default_cache_dir",
    "fetch_fred",
    "fetch_fred_series",
    "fetch_ipeadata",
    "fetch_ipeadata_metadata",
    "fetch_ipeadata_series",
    "fetch_sgs",
    "fetch_sgs_series",
    "fetch_sidra",
    "fetch_sidra_raw",
    "get_default_cache",
    "get_fred_api_key",
    "parse_sidra_period",
    "resolve_cache",
    "set_default_cache",
    "sgs_windows",
    "sidra_path",
]
