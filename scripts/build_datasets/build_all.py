"""Rebuild every shipped dataset from its primary source (needs network access).

Order matters: the calendar uses the panel legend and the GDP release dates.

Usage::

    python3 scripts/build_datasets/build_all.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = [
    "build_brazil_nowcast.py",
    "build_brazil_vintages.py",
    "build_brazil_calendar.py",
    "build_us_fred_md.py",
    "build_nyfed.py",
]


def main() -> int:
    """Run the build scripts in order; stop at the first failure."""
    for script in SCRIPTS:
        code = subprocess.call([sys.executable, str(HERE / script)])
        if code != 0:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
