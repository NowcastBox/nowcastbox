"""Example spec files shipped with the package (``nowcastbox init``).

The templates live in ``nowcastbox/pipeline/templates/*.yaml``:

- ``brazil_pib`` — Brazilian GDP from the built-in BCB/IBGE/IPEA panel with blocks,
  Student-t errors and a time-varying long-run mean (plan §6.2);
- ``simulated`` — small and fast example on the simulated dataset (pseudo real-time
  vintage, news against an earlier vintage, density, diagnostics, report);
- ``csv`` — your own data from a CSV file with metadata in the spec;
- ``connectors`` — live data from the BCB/SGS and IBGE/SIDRA APIs;
- ``model_building`` — the ECB toolbox workflow on the simulated dataset:
  pre-selection, specification search, Covid robustness, nowcast with the best
  specification (``selection`` section).
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path

__all__ = ["DEFAULT_TEMPLATE", "list_templates", "template_path", "template_text"]

DEFAULT_TEMPLATE = "brazil_pib"
"""Template written by ``nowcastbox init`` when none is named."""


def _folder() -> Path:
    return Path(str(resources.files("nowcastbox.pipeline") / "templates"))


def list_templates() -> list[str]:
    """Names of the bundled example specs.

    Returns
    -------
    list of str
        Template names (file stems), sorted.

    Examples
    --------
    >>> list_templates()
    ['brazil_pib', 'connectors', 'csv', 'model_building', 'simulated']
    """
    return sorted(p.stem for p in _folder().glob("*.yaml"))


def template_path(name: str = DEFAULT_TEMPLATE) -> Path:
    """Path of a bundled example spec.

    Parameters
    ----------
    name : str, default "brazil_pib"
        Template name (see :func:`list_templates`).

    Returns
    -------
    pathlib.Path
        YAML file.

    Raises
    ------
    ValueError
        If the template does not exist.

    Examples
    --------
    >>> template_path("simulated").name
    'simulated.yaml'
    """
    path = _folder() / f"{name}.yaml"
    if not path.is_file():
        raise ValueError(f"Unknown template {name!r}; available: {list_templates()}.")
    return path


def template_text(name: str = DEFAULT_TEMPLATE) -> str:
    """Text of a bundled example spec.

    Parameters
    ----------
    name : str, default "brazil_pib"
        Template name.

    Returns
    -------
    str
        YAML text (with comments).

    Raises
    ------
    ValueError
        If the template does not exist.

    Examples
    --------
    >>> "target: pib" in template_text("brazil_pib")
    True
    """
    return template_path(name).read_text(encoding="utf-8")
