"""Build (and execute) the example notebooks from their script twins.

The single source of truth of every example is a *percent-format* Python script in
``examples/scripts/`` (the format used by Jupytext, VS Code and Spyder): code cells start
with ``# %%`` and Markdown cells with ``# %% [markdown]`` (each Markdown line prefixed by
``# ``). This tool converts the scripts into ``examples/notebooks/*.ipynb`` and executes
them in place with :mod:`nbclient`, so the notebooks are saved **with outputs** and
render on GitHub.

Usage (from the repository root)::

    python3 examples/utils/build_notebooks.py               # build + execute all
    python3 examples/utils/build_notebooks.py 01 07         # only notebooks 01 and 07
    python3 examples/utils/build_notebooks.py --no-execute  # convert only

The executed notebooks are checked in CI with ``pytest --nbmake examples/notebooks``.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

EXAMPLES = Path(__file__).resolve().parent.parent
SCRIPTS = EXAMPLES / "scripts"
NOTEBOOKS = EXAMPLES / "notebooks"
_MARKER = re.compile(r"^# %%(?P<md>\s*\[markdown\])?\s*$")


def script_to_cells(text: str) -> list[nbformat.NotebookNode]:
    """Split a percent-format script into notebook cells.

    Lines before the first ``# %%`` marker (a module docstring) are dropped.

    Parameters
    ----------
    text : str
        Script source.

    Returns
    -------
    list of nbformat.NotebookNode
        Markdown and code cells in order.

    Examples
    --------
    >>> cells = script_to_cells("# %% [markdown]\\n# # Title\\n# %%\\nx = 1\\n")
    >>> [c.cell_type for c in cells], cells[0].source, cells[1].source
    (['markdown', 'code'], '# Title', 'x = 1')
    """
    cells: list[nbformat.NotebookNode] = []
    kind: str | None = None
    lines: list[str] = []

    def flush() -> None:
        if kind is None:
            return
        while lines and not lines[-1].strip():
            lines.pop()
        while lines and not lines[0].strip():
            lines.pop(0)
        if not lines:
            return
        if kind == "markdown":
            body = [ln[2:] if ln.startswith("# ") else ln.lstrip("#") for ln in lines]
            cells.append(nbformat.v4.new_markdown_cell("\n".join(body)))
        else:
            cells.append(nbformat.v4.new_code_cell("\n".join(lines)))

    for line in text.splitlines():
        match = _MARKER.match(line)
        if match:
            flush()
            kind = "markdown" if match.group("md") else "code"
            lines = []
        elif kind is not None:
            lines.append(line)
    flush()
    return cells


def build(script: Path, execute: bool = True, timeout: int = 600) -> tuple[Path, float]:
    """Convert one script into a notebook and (optionally) execute it in place.

    Parameters
    ----------
    script : pathlib.Path
        ``examples/scripts/NN_name.py``.
    execute : bool, default True
        Run the notebook with :class:`nbclient.NotebookClient` (cwd =
        ``examples/notebooks``) and save the outputs.
    timeout : int, default 600
        Per-cell timeout in seconds.

    Returns
    -------
    path : pathlib.Path
        Notebook written.
    seconds : float
        Execution wall time (0 when not executed).
    """
    nb = nbformat.v4.new_notebook()
    nb.cells = script_to_cells(script.read_text(encoding="utf-8"))
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python"}
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    out = NOTEBOOKS / f"{script.stem}.ipynb"
    seconds = 0.0
    if execute:
        start = time.perf_counter()
        client = NotebookClient(
            nb,
            timeout=timeout,
            kernel_name="python3",
            resources={"metadata": {"path": str(NOTEBOOKS)}},
        )
        client.execute()
        seconds = time.perf_counter() - start
    nbformat.write(nb, out)
    return out, seconds


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Build the nowcastbox example notebooks.")
    parser.add_argument("only", nargs="*", help="notebook numbers or stems (default: all)")
    parser.add_argument("--no-execute", action="store_true", help="convert without running")
    args = parser.parse_args(argv)
    scripts = sorted(SCRIPTS.glob("[0-9][0-9]_*.py"))
    if args.only:
        scripts = [s for s in scripts if any(s.stem.startswith(k) for k in args.only)]
    for script in scripts:
        path, seconds = build(script, execute=not args.no_execute)
        print(f"{path.relative_to(EXAMPLES)}  {seconds:6.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
