"""Generate ``docs/datasets/*.md`` (one page per dataset) from the metadata YAML files.

Run after any rebuild so the variable dictionaries match the shipped data::

    python3 scripts/build_datasets/make_docs.py
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from _common import REPO_ROOT

from nowcastbox.datasets._io import available_names, read_metadata

DOCS = REPO_ROOT / "docs" / "datasets"

USAGE = {
    "brazil_nowcast": """```python
import nowcastbox as nb
from nowcastbox.datasets import load_brazil_nowcast, load_brazil_calendar

ds = load_brazil_nowcast()
print(ds.summary())
panel = nb.prepare_panel(ds.data)            # applies the legend transforms
res = nb.MixedFreqDFM(n_factors=1, blocks=ds.blocks).fit(panel, target="pib")

# pseudo real-time vintage with the actual GDP release dates
vintage = nb.pseudo_real_time(ds.data, calendar=load_brazil_calendar(), vintage="2024-05-15")
```""",
    "brazil_calendar": """```python
from nowcastbox.datasets import (
    load_brazil_calendar, load_brazil_gdp_releases, load_brazil_nowcast,
)

cal = load_brazil_calendar()                 # ReleaseCalendar
cal.release_date("pib", "2024Q1")            # Timestamp('2024-06-04')
ds = load_brazil_nowcast()
cal.releases_between("2024-05-01", "2024-06-30", ds.data)  # what was published
load_brazil_calendar(as_frame=True)          # delay table
load_brazil_gdp_releases()                   # GDP release dates and their provenance
```""",
    "brazil_vintages": """```python
from nowcastbox.datasets import load_brazil_vintages

store = load_brazil_vintages()               # VintageStore
store.as_of("2020-08-31")                    # GDP as known on that date
store.nth_release(0)["pib"]                  # first releases
store.revision_summary()                     # Aruoba (2008) statistics
```""",
    "us_fred_md": """```python
from nowcastbox.datasets import load_us_fred_md

ds = load_us_fred_md(start="1985-01")
ds.legend[["transform", "fred_md_tcode", "fred_md_group"]]
```""",
    "nyfed": """```python
import nowcastbox as nb
from nowcastbox.datasets import load_nyfed

ds = load_nyfed()
x = nb.prepare_panel(ds.data)
res = nb.MixedFreqDFM(n_factors=1, factor_lags=1, blocks=ds.blocks).fit(x, target="GDPC1")
```""",
    "us_grs_like": """```python
import nowcastbox as nb
from nowcastbox.datasets import load_us_grs_like

ds = load_us_grs_like()                      # 1982-01 .. 2004-12
res = nb.TwoStepDFM(n_factors=2, factor_lags=2).fit(nb.prepare_panel(ds.data), target="GDPC1")
```""",
    "simulated_dfm": """```python
import pandas as pd
from nowcastbox.datasets import load_simulated_dfm

ds = load_simulated_dfm(n_monthly=20, n_factors=2)
truth = ds.metadata["true_params"]           # factors, loadings, A, Q, ...
vintage = ds.data.as_of(ds.data.end.to_timestamp(how="end") + pd.Timedelta(days=5))
```""",
}


def _cell(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def _dictionary(series: list[dict[str, Any]]) -> list[str]:
    cols = ["name", "description", "source", "source_code", "frequency", "transform",
            "delay_days", "blocks", "category", "units"]  # fmt: skip
    present = [c for c in cols if any(c in s for s in series)]
    lines = ["| " + " | ".join(present) + " |", "|" + "---|" * len(present)]
    for s in series:
        lines.append("| " + " | ".join(_cell(s.get(c, "")) for c in present) + " |")
    return lines


def _glance(name: str, meta: dict[str, Any]) -> list[str]:
    loader = meta.get("loader", "")
    summary = {**(meta.get("summary") or {})}
    out = [
        "## At a glance",
        "",
        "| | |",
        "|---|---|",
        f"| Loader | `nowcastbox.datasets.{loader}` |",
    ]
    if meta.get("target"):
        out.append(f"| Default target | `{meta['target']}` |")
    out += [
        f"| {key.replace('_', ' ').capitalize()} | {summary[key]} |"
        for key in ("n_series", "start", "end", "n_vintages")
        if key in summary
    ]
    if "frequencies" in summary:
        freqs = ", ".join(f"{k}: {v}" for k, v in summary["frequencies"].items())
        out.append(f"| Frequencies | {freqs} |")
    if meta.get("vintage"):
        out.append(f"| Vintage | {meta['vintage']} |")
    built = meta.get("built") or {}
    if built.get("date"):
        out.append(f"| Built | {built['date']} (`{built.get('script', '')}`) |")
    return [*out, "", "## Usage", "", USAGE.get(name, ""), ""]


def _legend_section(name: str, meta: dict[str, Any]) -> list[str]:
    if name == "us_grs_like":
        return ["The variable dictionary is that of [`load_us_fred_md`](us_fred_md.md).", ""]
    series = meta.get("series") or []
    if not series or "description" not in series[0]:
        return []
    out: list[str] = []
    blocks = Counter(b for s in series for b in str(s.get("blocks", "")).split(";") if b)
    cats = Counter(str(s.get("category", "")) for s in series if s.get("category"))
    if blocks:
        out += ["**Blocks:** " + ", ".join(f"`{b}` ({n})" for b, n in blocks.items()), ""]
    if cats:
        out += ["**Categories:** " + ", ".join(f"`{c}` ({n})" for c, n in cats.items()), ""]
    return [*out, "## Variable dictionary", "", *_dictionary(series), ""]


def _bullets(title: str, items: list[str]) -> list[str]:
    return [f"**{title}**", "", *[f"- {i}" for i in items], ""] if items else []


def _provenance_section(meta: dict[str, Any]) -> list[str]:
    out: list[str] = []
    if meta.get("programmes"):
        out += ["## Release programmes", "", "| Programme | Typical delay (days) | Rule |",
                "|---|---|---|"]  # fmt: skip
        out += [
            f"| {_cell(p['programme'])} | {p['typical_delay_days']} | {_cell(p['rule'])} |"
            for p in meta["programmes"]
        ]
        out.append("")
    out += ["## Sources and license", "", *[f"- {_cell(s)}" for s in meta.get("sources") or []]]
    out += ["", f"**License / terms of use.** {meta.get('license', '')}", ""]
    if meta.get("license_notice"):
        out += [f"> {meta['license_notice']}", ""]
    out += _bullets(
        "Excluded for licensing reasons:",
        [
            f"{_cell(e['series'])}: {_cell(e['reason'])}"
            for e in meta.get("excluded_for_license") or []
        ],
    )
    out += _bullets(
        "Series dropped at build time (failed verification):",
        [
            f"`{d['name']}` ({d['source']} {d['code']}): {_cell(d['reason'])}"
            for d in meta.get("dropped") or []
        ],
    )
    out += _bullets(
        "Releases without a parsed vintage:",
        [f"{s['reference_quarter']}: {_cell(s['reason'])}" for s in meta.get("skipped") or []],
    )
    return out


def _files_section(meta: dict[str, Any]) -> list[str]:
    files = meta.get("files") or {}
    if not files:
        return []
    out = ["## Files and integrity", "", "| key | file | SHA-256 |", "|---|---|---|"]
    out += [f"| {k} | `{v['path']}` | `{v['sha256']}` |" for k, v in files.items()]
    return [*out, "", "Digests are verified on first load (`verify=True`); rebuild with "
            "`python3 scripts/build_datasets/build_all.py`.", ""]  # fmt: skip


def page(name: str) -> str:
    """Markdown page of one dataset."""
    meta = read_metadata(name)
    out = [f"# `{meta.get('loader', '')}()` — {meta.get('title', name)}", "",
           str(meta.get("description", "")), ""]  # fmt: skip
    out += _glance(name, meta)
    out += _legend_section(name, meta)
    out += _provenance_section(meta)
    out += ["## Notes", "", str(meta.get("notes", "")), ""]
    out += ["## Citation", "", str(meta.get("citation", "")), ""]
    out += _files_section(meta)
    return "\n".join(out)


def index() -> str:
    """Overview page."""
    lines = [
        "# Datasets",
        "",
        "Built-in datasets are rebuilt from their **primary sources** by the scripts in "
        "`scripts/build_datasets/` and shipped as compressed CSV with a metadata YAML "
        "(description, sources, license, citation, SHA-256 digests and the series legend). "
        "Loaders work offline and verify the digests on first load.",
        "",
        "Every `Dataset` holds the panel **in levels** (`ds.data`, a `MixedFrequencyData` "
        "with quarterly values in the third month of the quarter) and a legend with the "
        "columns `name`, `description`, `source`, `source_code`, `frequency`, `transform` "
        "(named `nowcastbox.preprocessing` transformation), `legacy_code` (equivalent code "
        "0-7, if any), `delay_days` (typical publication lag after the end of the period), "
        "`blocks`, `category` (`hard`/`soft`/`financial`) and `units`.",
        "",
        "| Dataset | Loader | Series | Sample | Target |",
        "|---|---|---|---|---|",
    ]
    for name in available_names():
        meta = read_metadata(name)
        s = {**(meta.get("summary") or {}), **(meta.get("default_sample") or {})}
        lines.append(
            f"| [{meta.get('title', name)}]({name}.md) | `{meta.get('loader')}()` | "
            f"{s.get('n_series', '')} | {s.get('start', '')} – {s.get('end', '')} | "
            f"{meta.get('target') or ''} |"
        )
    lines += [
        "",
        "```python",
        "import nowcastbox.datasets as nbd",
        "nbd.list_datasets()",
        "```",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    """Write every page."""
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "index.md").write_text(index(), encoding="utf-8")
    for name in available_names():
        (DOCS / f"{name}.md").write_text(page(name), encoding="utf-8")


if __name__ == "__main__":
    main()
