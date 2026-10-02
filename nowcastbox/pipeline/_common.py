"""Helpers shared by the pipeline modules."""

from __future__ import annotations

import datetime as _dt
from collections.abc import Mapping
from pathlib import Path
from typing import Any

__all__ = ["plain"]


def plain(value: Any) -> Any:
    """YAML-safe copy of ``value`` (dates to ISO strings, tuples to lists).

    Parameters
    ----------
    value : Any
        Mapping, sequence or scalar.

    Returns
    -------
    Any
        The same structure with plain YAML types.

    Examples
    --------
    >>> import datetime
    >>> plain({"a": (1, datetime.date(2020, 1, 31))})
    {'a': [1, '2020-01-31']}
    """
    if isinstance(value, Mapping):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [plain(v) for v in value]
    if isinstance(value, _dt.date):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return value
