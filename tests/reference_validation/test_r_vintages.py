"""Pseudo real-time vintages vs ``PRTDB`` of the R package ``nowcasting`` 1.1.2.

Fixtures: ``scripts/reference_fixtures/r_prtdb.R`` - BRGDP (100 series, delays 0-84
days) and NYFED (25 series, monthly and quarterly, delays -14 to 63 days) on daily
vintages around month ends plus monthly vintages over a decade. ``PRTDB`` keeps the
values and blanks the unreleased ones, so a vintage is summarised per series by the
number of values kept and the position of the last one.

Findings (``docs/validation/vintages.md``): with the release date "end of the reference
period + delay days" (inclusive), :func:`nowcastbox.vintages.pseudo_real_time` reproduces
every R vintage. ``PRTDB`` also truncates the rows after the vintage month (nowcastbox
keeps the grid, the extra rows are empty). Negative delays (the two regional Fed surveys
of NYFED, released before the end of their reference month) are accepted since the
wave-3 integration and reproduce R as well.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nowcastbox.vintages import pseudo_real_time
from tests.reference_validation._helpers import fixture_path, legend, max_abs, read_periods

pytestmark = pytest.mark.reference_validation


def _frequencies(name: str) -> dict[str, str] | str:
    leg = legend(name)
    if "frequency" not in leg:
        return "M"
    return {n: ("Q" if f == 4 else "M") for n, f in zip(leg["name"], leg["frequency"], strict=True)}


def _compare(name: str, columns: list[str]) -> list[tuple[str, str]]:
    raw = read_periods(f"inputs/{name}_base.csv.gz")[columns]
    leg = legend(name).set_index("name")
    delay = {c: int(leg.loc[c, "delay"]) for c in columns}
    freq = _frequencies(name)
    if isinstance(freq, dict):
        freq = {c: freq[c] for c in columns}
    last = pd.read_csv(fixture_path(f"r/prtdb_{name}_last.csv.gz"))
    n_obs = pd.read_csv(fixture_path(f"r/prtdb_{name}_n_obs.csv.gz"))
    mismatches = []
    for k, vintage in enumerate(last["vintage"]):
        out = pseudo_real_time(raw, delay=delay, vintage=vintage, frequency=freq)
        kept = out.notna()
        assert max_abs(out.where(kept), raw.where(kept)) == 0.0  # values never change
        our_last = np.array(
            [np.flatnonzero(kept[c].to_numpy())[-1] if kept[c].any() else -1 for c in columns]
        )
        bad = (our_last != last.loc[k, columns].to_numpy(dtype=int)) | (
            kept.sum().to_numpy() != n_obs.loc[k, columns].to_numpy(dtype=int)
        )
        mismatches += [(str(vintage), c) for c, b in zip(columns, bad, strict=True) if b]
    return mismatches


def test_brgdp_vintages() -> None:
    columns = list(legend("brgdp")["name"])
    assert _compare("brgdp", columns) == []


def test_nyfed_vintages_non_negative_delays() -> None:
    leg = legend("nyfed")
    columns = [n for n, d in zip(leg["name"], leg["delay"], strict=True) if d >= 0]
    assert len(columns) == 23
    assert _compare("nyfed", columns) == []


def test_daily_vintages_cover_month_ends() -> None:
    vintages = pd.to_datetime(pd.read_csv(fixture_path("r/prtdb_brgdp_last.csv.gz"))["vintage"])
    assert bool(vintages.dt.is_month_end.any())
    assert bool((vintages.dt.day == 1).any())


def test_nyfed_vintages_with_negative_delays() -> None:
    """R accepts delay = -14 (survey released mid-month); so does nowcastbox."""
    leg = legend("nyfed")
    negative = [n for n, d in zip(leg["name"], leg["delay"], strict=True) if d < 0]
    assert negative == ["GACDISA066MSFRBNY", "GACDFSA066MSFRBPHI"]
    assert _compare("nyfed", list(leg["name"])) == []
