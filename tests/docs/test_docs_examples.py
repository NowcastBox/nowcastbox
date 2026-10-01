"""Execute the Python examples of the documentation and check the MkDocs navigation.

Every page of ``docs/`` (except the API reference and the pages owned by the datasets
and validation work packages) is scanned for fenced ``python`` code blocks. The blocks
of a page run in order, in one namespace, inside a temporary working directory, so a
page is a runnable script. A block preceded by the HTML comment ``<!-- skip-test -->``
is not executed (network access, files that do not exist, planned APIs).

The navigation test checks that every page referenced by ``mkdocs.yml`` exists (the
datasets and validation pages are written by other work packages and are only
required once present) and that every public subpackage has an API reference page.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
SKIP_MARKER = "<!-- skip-test -->"
EXCLUDED_DIRS = {"api", "datasets", "validation"}
EXTERNAL_PAGES = ("datasets/", "validation/")
FENCE = re.compile(r"^(?P<indent>[ \t]*)(?P<fence>```+|~~~+)\s*(?P<info>[^\n`]*)$")


def _pages() -> list[Path]:
    pages = []
    for path in sorted(DOCS.rglob("*.md")):
        relative = path.relative_to(DOCS)
        if relative.parts[0] in EXCLUDED_DIRS:
            continue
        pages.append(path)
    return pages


def extract_python_blocks(text: str) -> list[str]:
    """Return the runnable ``python`` blocks of a Markdown page, in order.

    Parameters
    ----------
    text : str
        Markdown source.

    Returns
    -------
    list of str
        Dedented code of every fenced block whose info string starts with ``python``
        (or ``py``) and that is not preceded by ``<!-- skip-test -->``.
    """
    lines = text.splitlines()
    blocks: list[str] = []
    i = 0
    previous = ""
    while i < len(lines):
        match = FENCE.match(lines[i])
        if match is None:
            if lines[i].strip():
                previous = lines[i].strip()
            i += 1
            continue
        indent, fence = match.group("indent"), match.group("fence")
        language = match.group("info").strip().split(" ")[0].lower() if match.group("info") else ""
        body: list[str] = []
        i += 1
        while i < len(lines) and not lines[i].strip().startswith(fence):
            line = lines[i]
            body.append(line[len(indent) :] if line.startswith(indent) else line.lstrip())
            i += 1
        i += 1  # closing fence
        if language in {"python", "py"} and previous != SKIP_MARKER:
            blocks.append("\n".join(body))
        previous = ""
    return blocks


def test_extract_python_blocks() -> None:
    text = "\n".join(
        [
            "Intro",
            "```python",
            "x = 1",
            "```",
            SKIP_MARKER,
            "```python",
            "raise RuntimeError",
            "```",
            '=== "Tab"',
            "",
            '    ```python title="indented"',
            "    y = x + 1",
            "    ```",
            "```bash",
            "echo no",
            "```",
        ]
    )
    assert extract_python_blocks(text) == ["x = 1", "y = x + 1"]


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("page", _pages(), ids=lambda p: str(p.relative_to(DOCS)))
def test_page_examples_run(page: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    blocks = extract_python_blocks(page.read_text(encoding="utf-8"))
    if not blocks:
        pytest.skip("no runnable Python example on this page")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MPLBACKEND", "Agg")
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    from nowcastbox.core import results as core_results
    from nowcastbox.visualization import themes

    # examples may register plots and themes: restore the global registries afterwards
    saved_plots = dict(core_results._PLOT_REGISTRY)
    saved_themes = dict(themes._THEMES)
    saved_default = list(themes._DEFAULT)
    namespace: dict[str, Any] = {"__name__": "__docs__"}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for number, code in enumerate(blocks, start=1):
                compiled = compile(code, f"{page.name}[block {number}]", "exec")
                exec(compiled, namespace)  # noqa: S102 - documentation examples
    finally:
        plt.close("all")
        core_results._PLOT_REGISTRY.clear()
        core_results._PLOT_REGISTRY.update(saved_plots)
        themes._THEMES.clear()
        themes._THEMES.update(saved_themes)
        themes._DEFAULT[:] = saved_default


def test_user_guide_pages_have_examples() -> None:
    guide = DOCS / "user-guide"
    without = [
        path.relative_to(DOCS).as_posix()
        for path in sorted(guide.rglob("*.md"))
        if path.name != "index.md" or path.parent.name == "data-sources"
        if not extract_python_blocks(path.read_text(encoding="utf-8"))
    ]
    assert without == []


def _nav_pages(node: Any) -> list[str]:
    if isinstance(node, str):
        return [node] if node.endswith(".md") else []
    if isinstance(node, list):
        return [page for item in node for page in _nav_pages(item)]
    if isinstance(node, dict):
        return [page for value in node.values() for page in _nav_pages(value)]
    return []


def _load_mkdocs_config() -> dict[str, Any]:
    import yaml

    class _Loader(yaml.SafeLoader):
        """Safe loader that ignores the ``!!python/name`` tags of mkdocs.yml."""

    def _ignore(loader: yaml.SafeLoader, suffix: str, node: yaml.Node) -> None:
        return None

    _Loader.add_multi_constructor("tag:yaml.org,2002:python/", _ignore)
    with open(ROOT / "mkdocs.yml", encoding="utf-8") as stream:
        return yaml.load(stream, Loader=_Loader)  # noqa: S506 - SafeLoader subclass


def test_nav_pages_exist() -> None:
    config = _load_mkdocs_config()
    pages = _nav_pages(config["nav"])
    assert "index.md" in pages
    assert "datasets/index.md" in pages
    assert "validation/index.md" in pages
    missing = [
        page for page in pages if not (DOCS / page).exists() and not page.startswith(EXTERNAL_PAGES)
    ]
    assert missing == []
    assert len(pages) == len(set(pages)), "a page appears twice in the navigation"


def test_every_page_is_in_nav() -> None:
    pages = set(_nav_pages(_load_mkdocs_config()["nav"]))
    # mkdocs navigation uses "/" on every platform (Windows paths use "\\")
    orphans = [
        path.relative_to(DOCS).as_posix()
        for path in _pages()
        if path.relative_to(DOCS).as_posix() not in pages
    ]
    assert orphans == []


def test_every_subpackage_has_api_page() -> None:
    import nowcastbox

    package_dir = Path(nowcastbox.__file__).parent
    subpackages = sorted(p.name for p in package_dir.iterdir() if (p / "__init__.py").exists())
    api_text = "\n".join(p.read_text(encoding="utf-8") for p in (DOCS / "api").glob("*.md"))
    documented = set(re.findall(r"^::: nowcastbox\.(\w+)", api_text, flags=re.MULTILINE))
    assert [name for name in subpackages if name not in documented] == []


def test_examples_do_not_write_into_the_repository() -> None:
    # the examples run in tmp_path: files they write must not appear in the repository
    for name in ("report.html", "nowcast_report.html", "backtest.parquet", "simulated.yaml"):
        assert not (ROOT / name).exists()
        assert not (DOCS / name).exists()
    assert not (ROOT / "nowcastbox" / "pipeline" / "templates" / "snapshots").exists()
