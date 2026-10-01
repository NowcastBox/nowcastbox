--8<-- "CONTRIBUTING.md"

## Documentation

The documentation is built with MkDocs Material and mkdocstrings:

```bash
pip install -e ".[docs]"
mkdocs serve              # live preview on http://localhost:8000
mkdocs build --strict     # what CI runs
```

- Pages are written in English; theory pages use MathJax (`$...$`, `$$...$$`) with the
  notation of [Theory](theory/index.md).
- Every user-guide page has a runnable example. `tests/docs/test_docs_examples.py`
  executes the Python blocks of each page in order (one namespace per page, in a
  temporary directory). Mark a block that must not run (network access, planned APIs)
  with an HTML comment `<!-- skip-test -->` on the line before the fence.
- The API reference is generated from the docstrings: document new public names in
  their subpackage's `__init__.py` (`__all__`) and with NumPy docstrings including
  *Examples*.
- New pages must be added to the `nav` of `mkdocs.yml` (a test checks it).
