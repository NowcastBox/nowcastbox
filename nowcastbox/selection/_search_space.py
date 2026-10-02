"""Parameter spaces of the specification search: random draws and exhaustive grids.

A space maps each searched setting to its admissible values:

* a **list** (or any sequence other than a pair of integers, or a ``range``) gives the
  discrete choices, e.g. ``{"n_factors": [1, 2, 3]}`` or
  ``{"start": ["2005-01", "2010-01"]}``;
* a **pair of integers** ``(low, high)`` is the inclusive integer range
  ``low, low + 1, ..., high`` (``{"n_series": (20, 80)}``);
* any other value (a scalar, a string or a mapping such as a block structure) is a
  fixed setting (one choice).

Random draws are reproducible: draw :math:`i` uses its own generator
``numpy.random.default_rng(SeedSequence(random_state).spawn(n)[i])``, whose state does
not depend on the number of draws, so a longer search repeats the draws of a shorter
one with the same seed (and a checkpointed search resumes exactly).
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

__all__ = ["ParameterSpace", "canonical", "spec_key"]

RandomState = int | np.random.SeedSequence | None


def _is_int(value: object) -> bool:
    return isinstance(value, int | np.integer) and not isinstance(value, bool)


def _seed_sequence(random_state: RandomState) -> np.random.SeedSequence:
    """A fresh seed sequence (a given one is copied, so spawning never mutates it)."""
    if isinstance(random_state, np.random.SeedSequence):
        return np.random.SeedSequence(
            random_state.entropy,
            spawn_key=random_state.spawn_key,
            pool_size=random_state.pool_size,
        )
    return np.random.SeedSequence(random_state)


def canonical(value: Any) -> Any:
    """Hashable, session-independent representation of a setting.

    Parameters
    ----------
    value : object
        A setting (scalar, period, timestamp, sequence or mapping).

    Returns
    -------
    object
        NumPy scalars become Python scalars, periods and timestamps their string, lists
        tuples and mappings sorted tuples of ``(key, value)`` pairs.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> canonical({"b": [1, 2], "a": np.int64(3)})
    (('a', 3), ('b', (1, 2)))
    >>> canonical(pd.Period("2005-01", freq="M"))
    '2005-01'
    """
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Period | pd.Timestamp):
        return str(value)
    if isinstance(value, Mapping):
        return tuple(sorted((str(k), canonical(v)) for k, v in value.items()))
    if isinstance(value, list | tuple | np.ndarray):
        return tuple(canonical(v) for v in value)
    return value


def spec_key(spec: Mapping[str, Any]) -> str:
    """Stable text key of a specification (used to deduplicate and to resume).

    Parameters
    ----------
    spec : mapping
        Setting name to value, in the order of the space.

    Returns
    -------
    str
        ``repr`` of the canonical ``(name, value)`` pairs.

    Examples
    --------
    >>> spec_key({"n_factors": 2, "start": "2005-01"})
    "(('n_factors', 2), ('start', '2005-01'))"
    """
    return repr(tuple((str(k), canonical(v)) for k, v in spec.items()))


@dataclass(frozen=True)
class _Dimension:
    """One searched setting: discrete choices or an inclusive integer range."""

    name: str
    choices: tuple[Any, ...] = ()
    low: int | None = None
    high: int | None = None

    @property
    def is_range(self) -> bool:
        return self.low is not None

    @property
    def size(self) -> int:
        if self.low is not None and self.high is not None:
            return self.high - self.low + 1
        return len(self.choices)

    def values(self) -> tuple[Any, ...]:
        if self.low is not None and self.high is not None:
            return tuple(range(self.low, self.high + 1))
        return self.choices

    def endpoints(self) -> tuple[Any, ...]:
        """Values worth validating (the bounds of a range, every choice otherwise)."""
        if self.low is not None and self.high is not None:
            return (self.low, self.high)
        return self.choices

    def draw(self, rng: np.random.Generator) -> Any:
        if self.low is not None and self.high is not None:
            return int(rng.integers(self.low, self.high + 1))
        return self.choices[int(rng.integers(len(self.choices)))]


def _dimension(name: str, value: Any) -> _Dimension:
    """Parse one entry of a space specification."""
    if isinstance(value, range):
        value = list(value)
    if isinstance(value, tuple) and len(value) == 2 and all(map(_is_int, value)):
        low, high = int(value[0]), int(value[1])
        if low > high:
            raise ValueError(f"Space entry {name!r}: the range {value!r} has low > high.")
        return _Dimension(name, low=low, high=high)
    if isinstance(value, list | tuple | np.ndarray):
        choices = tuple(value)
        if not choices:
            raise ValueError(f"Space entry {name!r} has no values.")
        return _Dimension(name, choices=choices)
    return _Dimension(name, choices=(value,))


class ParameterSpace:
    """Searchable settings and their admissible values.

    Parameters
    ----------
    space : mapping
        Setting name to values: a list of choices, an inclusive integer range
        ``(low, high)`` or a fixed value (see the module notes).

    Raises
    ------
    ValueError
        If the space is empty, a name is not a non-empty string, a list is empty or a
        range has ``low > high``.

    Examples
    --------
    >>> space = ParameterSpace({"n_factors": [1, 2], "n_series": (3, 5), "factor_lags": 1})
    >>> space.size
    6
    >>> space.grid()[:2]
    [{'n_factors': 1, 'n_series': 3, 'factor_lags': 1}, {'n_factors': 1, 'n_series': 4, 'factor_lags': 1}]
    >>> draws = space.draw(3, random_state=0)
    >>> draws == space.draw(3, random_state=0)
    True
    >>> space.draw(2, random_state=0) == draws[:2]  # draw i does not depend on n
    True
    """

    def __init__(self, space: Mapping[str, Any]) -> None:
        if not isinstance(space, Mapping) or not space:
            raise ValueError("space must be a non-empty mapping {setting: values}.")
        dims = []
        for name, value in space.items():
            if not isinstance(name, str) or not name:
                raise ValueError(f"Space keys must be non-empty strings, got {name!r}.")
            dims.append(_dimension(name, value))
        self._dims: tuple[_Dimension, ...] = tuple(dims)

    @property
    def names(self) -> list[str]:
        """Names of the settings, in order.

        Examples
        --------
        >>> ParameterSpace({"a": [1], "b": (1, 2)}).names
        ['a', 'b']
        """
        return [d.name for d in self._dims]

    @property
    def size(self) -> int:
        """Number of distinct specifications (size of the exhaustive grid).

        Examples
        --------
        >>> ParameterSpace({"a": [1, 2, 3], "b": (1, 4)}).size
        12
        """
        return math.prod(d.size for d in self._dims)

    def values(self, name: str) -> tuple[Any, ...]:
        """Every admissible value of one setting (ranges expanded).

        Parameters
        ----------
        name : str
            Setting name.

        Returns
        -------
        tuple
            Values.

        Raises
        ------
        KeyError
            If the setting is not in the space.

        Examples
        --------
        >>> ParameterSpace({"n_series": (2, 4)}).values("n_series")
        (2, 3, 4)
        """
        return self._dimension(name).values()

    def endpoints(self, name: str) -> tuple[Any, ...]:
        """Values to validate: the bounds of a range or every choice.

        Parameters
        ----------
        name : str
            Setting name.

        Returns
        -------
        tuple
            Values.

        Raises
        ------
        KeyError
            If the setting is not in the space.

        Examples
        --------
        >>> ParameterSpace({"n_series": (2, 40)}).endpoints("n_series")
        (2, 40)
        """
        return self._dimension(name).endpoints()

    def _dimension(self, name: str) -> _Dimension:
        for dim in self._dims:
            if dim.name == name:
                return dim
        raise KeyError(f"{name!r} is not in the space; settings: {self.names}.")

    def __contains__(self, name: object) -> bool:
        return any(d.name == name for d in self._dims)

    def __iter__(self) -> Iterator[str]:
        return iter(self.names)

    def grid(self) -> list[dict[str, Any]]:
        """Every specification of the space (exhaustive grid, in lexicographic order).

        Returns
        -------
        list of dict
            Specifications.

        Examples
        --------
        >>> ParameterSpace({"a": [1, 2], "b": ["x", "y"]}).grid()[-1]
        {'a': 2, 'b': 'y'}
        """
        names = self.names
        products = itertools.product(*(d.values() for d in self._dims))
        return [dict(zip(names, combo, strict=True)) for combo in products]

    def draw(self, n: int, random_state: RandomState = None) -> list[dict[str, Any]]:
        """Random specifications, each setting drawn uniformly over its values.

        Parameters
        ----------
        n : int
            Number of draws (duplicates are possible).
        random_state : int, numpy.random.SeedSequence or None
            Seed; draw ``i`` uses the ``i``-th child of ``SeedSequence(random_state)``
            (a given ``SeedSequence`` is copied, never spawned from, so repeated calls
            give the same draws).

        Returns
        -------
        list of dict
            Specifications.

        Raises
        ------
        ValueError
            If ``n`` is not a positive integer.

        Examples
        --------
        >>> len(ParameterSpace({"a": (1, 9)}).draw(4, random_state=1))
        4
        """
        if not _is_int(n) or n < 1:
            raise ValueError(f"n must be a positive integer, got {n!r}.")
        out = []
        for child in _seed_sequence(random_state).spawn(int(n)):
            rng = np.random.default_rng(child)
            out.append({d.name: d.draw(rng) for d in self._dims})
        return out

    def __repr__(self) -> str:
        parts = []
        for d in self._dims:
            text = f"({d.low}, {d.high})" if d.is_range else repr(list(d.choices))
            parts.append(f"{d.name}={text}")
        return f"ParameterSpace({', '.join(parts)})"
