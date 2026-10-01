"""Command-line interface: ``nowcastbox run spec.yaml`` (innovation I10, plan §6.2).

The console script ``nowcastbox`` (``pyproject.toml``: ``nowcastbox.cli:main``) exposes
the declarative pipeline of :mod:`nowcastbox.pipeline`: ``run``, ``validate``, ``init``,
``datasets list|info`` and ``snapshots list|show|diff|history``. ``python -m
nowcastbox.cli`` is equivalent.

Examples
--------
>>> from nowcastbox.cli import main
>>> main(["--version"])  # doctest: +SKIP
0
"""

from nowcastbox.cli.main import EXIT_ERROR, EXIT_OK, EXIT_USAGE, build_parser, main

__all__ = ["EXIT_ERROR", "EXIT_OK", "EXIT_USAGE", "build_parser", "main"]
