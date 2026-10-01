"""Built-in datasets (plan section 7).

Replication datasets rebuilt from primary sources and new Brazilian datasets, returned
as typed :class:`Dataset` objects (panel in levels + legend + provenance), plus
real-time GDP vintages (:class:`~nowcastbox.vintages.VintageStore`) and release
calendars (:class:`~nowcastbox.vintages.ReleaseCalendar`). Every shipped file is
verified against the SHA-256 digest recorded in its metadata YAML on first load; the
build scripts live in ``scripts/build_datasets/`` of the source repository.

Examples
--------
>>> import nowcastbox.datasets as nbd
>>> ds = nbd.load_brazil_nowcast()
>>> ds.target
'pib'
>>> list(nbd.list_datasets().index)  # doctest: +NORMALIZE_WHITESPACE
['brazil_calendar', 'brazil_nowcast', 'brazil_vintages', 'nyfed', 'simulated_dfm',
 'us_fred_md', 'us_grs_like']
"""

from nowcastbox.datasets._io import clear_cache, verify_file
from nowcastbox.datasets._simulated import simulate_mixed_frequency_dfm
from nowcastbox.datasets.dataset import LEGEND_COLUMNS, Dataset
from nowcastbox.datasets.load import (
    dataset_info,
    list_datasets,
    load_brazil_calendar,
    load_brazil_gdp_releases,
    load_brazil_nowcast,
    load_brazil_vintages,
    load_dataset,
    load_nyfed,
    load_simulated_dfm,
    load_us_fred_md,
    load_us_grs_like,
)

__all__ = [
    "LEGEND_COLUMNS",
    "Dataset",
    "clear_cache",
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
    "simulate_mixed_frequency_dfm",
    "verify_file",
]
