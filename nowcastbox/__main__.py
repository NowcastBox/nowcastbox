"""``python -m nowcastbox``: run the command-line interface."""

import sys

from nowcastbox.cli.main import main

if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
