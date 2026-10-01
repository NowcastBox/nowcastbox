"""Data vintages.

* :class:`ReleaseCalendar` - when each observation is published (delays in days after
  the end of the reference period, or explicit release dates).
* :func:`pseudo_real_time` / :func:`generate_vintages` - the final dataset as it would
  have been observed at past dates (pseudo real-time, no revisions).
* :class:`VintageStore` - genuine real-time vintages with revisions (long format),
  ``as_of(date)`` panels, revision triangles and statistics, CSV/Parquet I/O.
"""

from nowcastbox.vintages.calendar import RELEASE_COLUMNS, ReleaseCalendar
from nowcastbox.vintages.pseudo_real_time import (
    Vintage,
    generate_vintages,
    pseudo_real_time,
    vintage_dates,
)
from nowcastbox.vintages.vintage_store import RECORD_COLUMNS, VintageStore

__all__ = [
    "RECORD_COLUMNS",
    "RELEASE_COLUMNS",
    "ReleaseCalendar",
    "Vintage",
    "VintageStore",
    "generate_vintages",
    "pseudo_real_time",
    "vintage_dates",
]
