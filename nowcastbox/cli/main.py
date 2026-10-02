"""The ``nowcastbox`` command-line interface (innovation I10, plan §6.2).

Subcommands::

    nowcastbox run SPEC.yaml [--snapshot-dir DIR] [--no-snapshot] [--vintage DATE] [--json]
    nowcastbox validate SPEC.yaml [--check-data] [--show]
    nowcastbox init [PATH] [--template NAME] [--force] [--list] [--excel [--example]]
    nowcastbox datasets list | info NAME
    nowcastbox snapshots list|show|diff|history LOCATION ...
    nowcastbox --version

``LOCATION`` is a snapshot directory or a spec file (its ``snapshot_dir`` and ``name``
are used). Exit codes: ``0`` success, ``1`` runtime error, ``2`` invalid spec or usage.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, TextIO

import pandas as pd

__all__ = ["EXIT_ERROR", "EXIT_OK", "EXIT_USAGE", "build_parser", "main"]

EXIT_OK = 0
"""Exit code of a successful command."""
EXIT_ERROR = 1
"""Exit code of a runtime failure (data, model, missing snapshot ...)."""
EXIT_USAGE = 2
"""Exit code of an invalid spec or command line."""


class _CommandError(Exception):
    """Command failure carrying an exit code."""

    def __init__(self, message: str, code: int = EXIT_ERROR) -> None:
        super().__init__(message)
        self.code = code


def _out(text: str, stream: TextIO | None = None) -> None:
    target = sys.stdout if stream is None else stream
    target.write(text if text.endswith("\n") else text + "\n")


def _json(value: Any) -> None:
    from nowcastbox.pipeline import jsonable

    _out(json.dumps(jsonable(value), indent=2))


def _frame_text(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "(none)"
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        return frame.to_string()


# ---------------------------------------------------------------------------- run
def _read_spec(path: str) -> Any:
    """Parse a spec file (a missing file is a usage error)."""
    from nowcastbox.pipeline import NowcastSpec

    try:
        return NowcastSpec.from_yaml(path)
    except FileNotFoundError as err:
        raise _CommandError(str(err), EXIT_USAGE) from err


def _cmd_run(args: argparse.Namespace) -> int:
    from nowcastbox.pipeline import run_pipeline

    spec = _read_spec(args.spec)
    run = run_pipeline(
        spec,
        snapshot=not args.no_snapshot,
        snapshot_dir=args.snapshot_dir,
        vintage=args.vintage,
    )
    if args.json:
        _json(run.to_dict())
    else:
        _out(run.summary())
    return EXIT_OK


def _cmd_validate(args: argparse.Namespace) -> int:
    from nowcastbox.pipeline import apply_vintage, load_data, preprocess

    spec = _read_spec(args.spec)
    lines = [
        f"OK: {args.spec}",
        f"  name      : {spec.name}",
        f"  target    : {spec.target}",
        f"  data      : {spec.data.source} ({spec.data.kind})",
        f"  vintage   : {spec.vintage}",
        f"  model     : {spec.model.type} (factors={spec.model.n_factors})",
        f"  outputs   : {', '.join(spec.outputs.names)}",
        f"  snapshots : {spec.snapshot_dir or '(none)'}",
    ]
    if args.check_data:
        raw = load_data(spec)
        data = apply_vintage(raw, spec.vintage_timestamp(), explicit=spec.vintage != "today")
        panel = preprocess(data, spec)
        lines.append(
            f"  data check: {data.n_series} series {data.start} to {data.end}; "
            f"{panel.n_series} after preprocessing"
        )
    _out("\n".join(lines))
    if args.show:
        _out(spec.to_yaml())
    return EXIT_OK


# ---------------------------------------------------------------------------- init
def _cmd_init(args: argparse.Namespace) -> int:
    from nowcastbox.pipeline import list_templates, template_text

    if args.list:
        _out("\n".join(list_templates()))
        return EXIT_OK
    if args.excel:
        return _init_excel(args)
    if args.example:
        raise _CommandError("--example is only valid with --excel.", EXIT_USAGE)
    try:
        text = template_text(args.template)
    except ValueError as err:
        raise _CommandError(str(err), EXIT_USAGE) from err
    path = Path(args.path or "nowcast.yaml")
    _check_new(path, args.force)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    _out(f"Wrote example spec {path} (template {args.template!r}).")
    _out(f"Next: nowcastbox validate {path} && nowcastbox run {path}")
    return EXIT_OK


def _check_new(path: Path, force: bool) -> None:
    if path.exists() and not force:
        raise _CommandError(f"{path} already exists (use --force to overwrite).", EXIT_USAGE)


def _init_excel(args: argparse.Namespace) -> int:
    """Write an Excel data template (empty, or the bundled example with ``--example``)."""
    import shutil

    from nowcastbox.pipeline.data import example_workbook_path, write_excel_panel

    path = Path(args.path or "nowcast.xlsx")
    if path.suffix.lower() not in (".xlsx", ".xlsm"):
        raise _CommandError(f"{path} is not an .xlsx file.", EXIT_USAGE)
    _check_new(path, args.force)
    if args.example:
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(example_workbook_path(), path)
        _out(f"Wrote example Excel workbook {path}.")
    else:
        try:
            write_excel_panel(path)
        except ImportError as err:
            raise _CommandError(str(err)) from err
        _out(f"Wrote empty Excel data template {path} (sheets monthly, quarterly, metadata).")
    _out(f"Use it in a spec: data: {{source: excel, path: {path.name}}}")
    return EXIT_OK


# ---------------------------------------------------------------------------- datasets
def _cmd_datasets(args: argparse.Namespace) -> int:
    from nowcastbox.datasets import dataset_info, list_datasets

    if args.datasets_command == "list":
        table = list_datasets()[["title", "n_series", "start", "end", "target"]]
        if args.json:
            _json(table.reset_index().to_dict(orient="records"))
        else:
            _out(_frame_text(table))
        return EXIT_OK
    try:
        info = dataset_info(args.name)
    except ValueError as err:
        raise _CommandError(str(err)) from err
    if args.json:
        _json(info)
    else:
        _out(_dataset_text(args.name, info))
    return EXIT_OK


def _dataset_text(name: str, info: dict[str, Any]) -> str:
    lines = [f"{name}: {info.get('title', '')}"]
    for key in ("loader", "target", "license", "url", "citation"):
        if info.get(key):
            lines.append(f"  {key:<8}: {str(info[key]).strip()}")
    if info.get("description"):
        lines += ["", str(info["description"]).strip()]
    series = info.get("series") or []
    if series:
        cols = ["name", "frequency", "transform", "delay_days", "blocks", "category"]
        table = pd.DataFrame(series)
        table = table[[c for c in cols if c in table.columns]]
        lines += ["", f"{len(series)} series:", _frame_text(table.set_index("name"))]
    return "\n".join(lines)


# ---------------------------------------------------------------------------- snapshots
def _store(location: str) -> tuple[Any, str | None]:
    """``(SnapshotStore, name filter)`` from a directory or a spec file."""
    from nowcastbox.pipeline import NowcastSpec, SnapshotStore

    path = Path(location)
    if path.is_file() and path.suffix in (".yaml", ".yml"):
        spec = NowcastSpec.from_yaml(path)
        if spec.snapshot_dir is None:
            raise _CommandError(f"{path} has no snapshot_dir.", EXIT_USAGE)
        return SnapshotStore(spec.snapshot_dir), spec.name
    if not path.is_dir():
        raise _CommandError(f"{path} is neither a snapshot directory nor a spec file.")
    return SnapshotStore(path), None


def _cmd_snapshots(args: argparse.Namespace) -> int:
    store, spec_name = _store(args.location)
    name = args.name if args.name is not None else spec_name
    handlers: dict[str, Callable[[Any, str | None, argparse.Namespace], None]] = {
        "list": _snap_list,
        "show": _snap_show,
        "diff": _snap_diff,
        "history": _snap_history,
    }
    try:
        handlers[args.snapshots_command](store, name, args)
    except KeyError as err:
        raise _CommandError(str(err.args[0]) if err.args else str(err)) from err
    return EXIT_OK


def _snap_list(store: Any, name: str | None, args: argparse.Namespace) -> None:
    table = store.to_frame(name)
    if args.json:
        _json(table.reset_index().to_dict(orient="records"))
    else:
        _out(_frame_text(table))


def _snap_show(store: Any, name: str | None, args: argparse.Namespace) -> None:
    snap = store.load(args.ref, name=name)
    if args.json:
        _json(snap.manifest)
        return
    _out(snap.summary())
    _out("")
    _out(_frame_text(snap.nowcast().tail(args.periods)))
    news = snap.news()
    if news is not None:
        _out("")
        _out("News against " + str((snap.manifest.get("news") or {}).get("against")) + ":")
        _out(_frame_text(news.head(args.periods)))


def _snap_diff(store: Any, name: str | None, args: argparse.Namespace) -> None:
    diff = store.diff(store.load(args.old, name=name), store.load(args.new, name=name))
    if args.json:
        _json(diff.to_dict())
    else:
        _out(diff.summary(n_periods=args.periods))


def _snap_history(store: Any, name: str | None, args: argparse.Namespace) -> None:
    table = store.history(name, period=args.period)
    if args.json:
        _json(table.reset_index().to_dict(orient="records"))
    else:
        _out(_frame_text(table))


# ---------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser of the ``nowcastbox`` command.

    Returns
    -------
    argparse.ArgumentParser
        Parser with the ``run``, ``validate``, ``init``, ``datasets`` and
        ``snapshots`` subcommands.

    Examples
    --------
    >>> args = build_parser().parse_args(["run", "spec.yaml", "--no-snapshot"])
    >>> args.command, args.spec, args.no_snapshot
    ('run', 'spec.yaml', True)
    """
    from nowcastbox.__version__ import __version__

    parser = argparse.ArgumentParser(
        prog="nowcastbox",
        description="Nowcasting with dynamic factor models: declarative production pipeline.",
    )
    parser.add_argument("-V", "--version", action="version", version=f"nowcastbox {__version__}")
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument("-v", "--verbose", action="store_true", help="log progress (INFO)")
    verbosity.add_argument("-q", "--quiet", action="store_true", help="log errors only")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True
    _add_run(sub)
    _add_validate(sub)
    _add_init(sub)
    _add_datasets(sub)
    _add_snapshots(sub)
    return parser


def _add_run(sub: Any) -> None:
    p = sub.add_parser("run", help="run a nowcast spec (YAML)")
    p.add_argument("spec", help="spec file")
    p.add_argument("--snapshot-dir", help="override the spec's snapshot_dir")
    p.add_argument("--no-snapshot", action="store_true", help="do not write a snapshot")
    p.add_argument("--vintage", help="override the spec's vintage ('today' or YYYY-MM-DD)")
    p.add_argument("--json", action="store_true", help="print a JSON summary")
    p.set_defaults(handler=_cmd_run)


def _add_validate(sub: Any) -> None:
    p = sub.add_parser("validate", help="check a spec without running it")
    p.add_argument("spec", help="spec file")
    p.add_argument("--check-data", action="store_true", help="also load and preprocess the data")
    p.add_argument("--show", action="store_true", help="print the normalised spec")
    p.set_defaults(handler=_cmd_validate)


def _add_init(sub: Any) -> None:
    from nowcastbox.pipeline import DEFAULT_TEMPLATE

    p = sub.add_parser("init", help="write an example spec (or an Excel data template)")
    p.add_argument(
        "path",
        nargs="?",
        default=None,
        help="output file (nowcast.yaml; nowcast.xlsx with --excel)",
    )
    p.add_argument("-t", "--template", default=DEFAULT_TEMPLATE, help="template name")
    p.add_argument("-f", "--force", action="store_true", help="overwrite an existing file")
    p.add_argument("--list", action="store_true", help="list the templates")
    p.add_argument(
        "--excel", action="store_true", help="write an empty Excel data template (.xlsx)"
    )
    p.add_argument(
        "--example", action="store_true", help="with --excel: the filled example workbook"
    )
    p.set_defaults(handler=_cmd_init)


def _add_datasets(sub: Any) -> None:
    p = sub.add_parser("datasets", help="built-in datasets")
    dsub = p.add_subparsers(dest="datasets_command", metavar="ACTION")
    dsub.required = True
    lst = dsub.add_parser("list", help="list the built-in datasets")
    lst.add_argument("--json", action="store_true", help="JSON output")
    info = dsub.add_parser("info", help="metadata of a dataset")
    info.add_argument("name", help="dataset name")
    info.add_argument("--json", action="store_true", help="JSON output")
    p.set_defaults(handler=_cmd_datasets)


def _add_snapshots(sub: Any) -> None:
    p = sub.add_parser("snapshots", help="versioned nowcast snapshots")
    ssub = p.add_subparsers(dest="snapshots_command", metavar="ACTION")
    ssub.required = True

    def common(q: argparse.ArgumentParser) -> None:
        q.add_argument("location", help="snapshot directory or spec file")
        q.add_argument("--name", help="only snapshots of this spec name")
        q.add_argument("--json", action="store_true", help="JSON output")

    common(ssub.add_parser("list", help="list the snapshots"))
    show = ssub.add_parser("show", help="show one snapshot")
    common(show)
    show.add_argument("ref", nargs="?", default="latest", help="id, prefix or latest[~k]")
    show.add_argument("--periods", type=int, default=8, help="rows shown")
    diff = ssub.add_parser("diff", help="compare two snapshots")
    common(diff)
    diff.add_argument("old", nargs="?", default="latest~1", help="old snapshot (latest~1)")
    diff.add_argument("new", nargs="?", default="latest", help="new snapshot (latest)")
    diff.add_argument("--periods", type=int, default=6, help="target periods shown")
    hist = ssub.add_parser("history", help="nowcast history across snapshots")
    common(hist)
    hist.add_argument("--period", help="target period (default: each headline period)")
    p.set_defaults(handler=_cmd_snapshots)


# ---------------------------------------------------------------------------- main
def _configure_logging(args: argparse.Namespace) -> None:
    from nowcastbox._logging import set_log_level

    if args.verbose:
        set_log_level(logging.INFO)
    elif args.quiet:
        set_log_level(logging.ERROR)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ``nowcastbox`` command-line interface.

    Parameters
    ----------
    argv : sequence of str, optional
        Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns
    -------
    int
        Exit code: ``0`` success, ``1`` runtime error, ``2`` invalid spec or usage.

    Examples
    --------
    >>> from nowcastbox.cli import main
    >>> main(["init", "--list"])  # doctest: +NORMALIZE_WHITESPACE
    brazil_pib
    connectors
    csv
    simulated
    0
    """
    from nowcastbox.core.exceptions import NowcastBoxError
    from nowcastbox.pipeline import SpecError

    parser = build_parser()
    try:
        args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    except SystemExit as exc:  # --help, --version and usage errors
        return exc.code if isinstance(exc.code, int) else EXIT_USAGE
    _configure_logging(args)
    try:
        return int(args.handler(args))
    except SpecError as err:
        _out(f"nowcastbox: {err}", sys.stderr)
        return EXIT_USAGE
    except _CommandError as err:
        _out(f"nowcastbox: error: {err}", sys.stderr)
        return err.code
    except (NowcastBoxError, OSError, ValueError, KeyError) as err:
        _out(f"nowcastbox: error: {type(err).__name__}: {err}", sys.stderr)
        return EXIT_ERROR
