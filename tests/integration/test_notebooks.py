"""Execute every example notebook end to end against the installed API (plan §9).

The notebooks in ``examples/notebooks`` are built from ``examples/scripts`` by
``examples/utils/build_notebooks.py`` and saved with outputs. This test re-executes each
one in memory with :mod:`nbclient` (kernel cwd = ``examples/notebooks``, as when they
were built) and fails on the first cell that raises. Network cells are skipped by the
notebooks themselves unless ``NOWCASTBOX_EXAMPLES_NETWORK=1``.

The whole set takes several minutes, so the tests are marked ``slow`` (deselect with
``-m "not slow"``); run a subset with ``-k 14_weekly``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")
nbclient = pytest.importorskip("nbclient")

NOTEBOOKS = Path(__file__).resolve().parents[2] / "examples" / "notebooks"
PATHS = sorted(NOTEBOOKS.glob("[0-9][0-9]_*.ipynb"))

pytestmark = [pytest.mark.slow, pytest.mark.integration]


def test_every_script_has_a_notebook() -> None:
    scripts = sorted((NOTEBOOKS.parent / "scripts").glob("[0-9][0-9]_*.py"))
    assert len(scripts) == 15
    assert [p.stem for p in PATHS] == [s.stem for s in scripts]


@pytest.mark.parametrize("path", PATHS, ids=[p.stem for p in PATHS])
def test_notebook_executes(path: Path) -> None:
    notebook = nbformat.read(path, as_version=4)
    env_timeout = int(os.environ.get("NOWCASTBOX_NOTEBOOK_TIMEOUT", "900"))
    client = nbclient.NotebookClient(
        notebook,
        timeout=env_timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOKS)}},
    )
    client.execute()
    errors = [
        output
        for cell in notebook.cells
        if cell.cell_type == "code"
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    assert not errors
