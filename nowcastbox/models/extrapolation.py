r"""Pluggable extrapolation of indicators to the end of the target period.

Bridge equations need every indicator up to the last (high-frequency) period of the
target periods being nowcast, but at the ragged edge the most recent months are not
released yet. An *extrapolator* completes them. It receives the whole panel at once, so
that multivariate methods (a large BVAR, Phase 3 of the ECB-parity plan) can be plugged
in next to the univariate AR completion (Baffigi, Golinelli & Parigi, 2004; Diron,
2008; Bańbura, Belousova, Bodnár & Tóth, 2023).

An extrapolator is any callable

.. code-block:: python

    def extrapolate(
        data: MixedFrequencyData, columns: Sequence[str], end: pd.Period
    ) -> dict[str, pd.Series]: ...

returning, for each name in ``columns``, the series on its contiguous **native** grid
from its first period up to the last native period contained in ``end``, observed
values unchanged and missing values after the last observation filled. Named
extrapolators are created by factories kept in a registry:

* :func:`register_extrapolator` - add a factory (``"bvar"`` is registered by
  :mod:`nowcastbox.models.bvar_extrapolation`);
* :func:`make_extrapolator` - build one from a name (with options) or pass a callable
  through;
* :func:`available_extrapolators` - registered names.

``"ar"`` (:class:`ARExtrapolator`) is the iterated AR(:math:`p`) completion of
:func:`nowcastbox.models.bridge.ar_extend`, series by series.

Examples
--------
>>> import numpy as np, pandas as pd
>>> from nowcastbox.core.data import MixedFrequencyData
>>> from nowcastbox.models.extrapolation import make_extrapolator
>>> idx = pd.period_range("2020-01", periods=5, freq="M")
>>> mfd = MixedFrequencyData(pd.DataFrame({"x": [1.0, 2, 3, 4, np.nan]}, index=idx), "M")
>>> ext = make_extrapolator("ar", ar_lags=1)
>>> ext(mfd, ["x"], pd.Period("2020Q2", "Q"))["x"].tolist()
[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from nowcastbox._logging import get_logger
from nowcastbox.core.data import MixedFrequencyData
from nowcastbox.core.exceptions import NowcastDataError
from nowcastbox.models.bridge import ar_extend

__all__ = [
    "ARExtrapolator",
    "Extrapolator",
    "available_extrapolators",
    "make_extrapolator",
    "native_until",
    "register_extrapolator",
]

logger = get_logger(__name__)


@runtime_checkable
class Extrapolator(Protocol):
    """Protocol of an indicator extrapolator (see the module docstring).

    Examples
    --------
    >>> from nowcastbox.models.extrapolation import ARExtrapolator, Extrapolator
    >>> isinstance(ARExtrapolator(), Extrapolator)
    True
    """

    def __call__(
        self, data: MixedFrequencyData, columns: Sequence[str], end: pd.Period
    ) -> dict[str, pd.Series]:
        """Complete ``columns`` of ``data`` up to the native periods contained in ``end``."""
        ...  # pragma: no cover


ExtrapolatorFactory = Callable[..., Extrapolator]

_REGISTRY: dict[str, ExtrapolatorFactory] = {}


def native_until(data: MixedFrequencyData, column: str, end: pd.Period) -> pd.Series:
    """Series ``column`` on its contiguous native grid, up to the native period of ``end``.

    Parameters
    ----------
    data : MixedFrequencyData
        Panel.
    column : str
        Series name.
    end : pandas.Period
        Last period to cover (normally a target period); the grid stops at the native
        period that contains the last day of ``end``.

    Returns
    -------
    pandas.Series
        Values (NaN = not observed) indexed by a contiguous :class:`pandas.PeriodIndex`
        of the series' native frequency, starting at the first period of the panel.

    Examples
    --------
    >>> import pandas as pd
    >>> from nowcastbox.core.data import MixedFrequencyData
    >>> idx = pd.period_range("2020-01", periods=3, freq="M")
    >>> mfd = MixedFrequencyData(pd.DataFrame({"x": [1.0, 2.0, 3.0]}, index=idx), "M")
    >>> native_until(mfd, "x", pd.Period("2020Q2", "Q")).index.astype(str).tolist()
    ['2020-01', '2020-02', '2020-03', '2020-04', '2020-05', '2020-06']
    """
    freq = data.metadata[column].frequency.pandas_freq
    native = data.to_native(column)
    last = end.asfreq(freq, how="E")
    grid = pd.period_range(native.index[0], last, freq=freq)
    return native.reindex(grid).astype(float)


@dataclass(frozen=True)
class ARExtrapolator:
    r"""Iterated AR(:math:`p`) completion of each indicator (series by series).

    Each series :math:`x_t = c + \sum_{j=1}^p \rho_j x_{t-j} + u_t` is estimated by OLS
    on its own observed history and extended with iterated forecasts after its last
    observation (:func:`nowcastbox.models.bridge.ar_extend`); this is the ragged-edge
    treatment of :class:`~nowcastbox.models.BridgeEquation`.

    Parameters
    ----------
    ar_lags : int, default 1
        AR order :math:`p` (0 = complete with the sample mean).

    Raises
    ------
    ValueError
        If ``ar_lags`` is not a non-negative integer.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from nowcastbox.core.data import MixedFrequencyData
    >>> idx = pd.period_range("2020-01", periods=4, freq="M")
    >>> mfd = MixedFrequencyData(pd.DataFrame({"x": [1.0, 3.0, np.nan, np.nan]}, index=idx), "M")
    >>> ARExtrapolator(ar_lags=0)(mfd, ["x"], pd.Period("2020-04", "M"))["x"].tolist()
    [1.0, 3.0, 2.0, 2.0]
    """

    ar_lags: int = 1

    def __post_init__(self) -> None:
        value = self.ar_lags
        if isinstance(value, bool) or not isinstance(value, int | np.integer) or value < 0:
            raise ValueError(f"ar_lags must be a non-negative integer, got {value!r}.")

    def __call__(
        self, data: MixedFrequencyData, columns: Sequence[str], end: pd.Period
    ) -> dict[str, pd.Series]:
        """Complete every series in ``columns`` up to ``end``.

        Parameters
        ----------
        data : MixedFrequencyData
            Panel (the vintage's information set).
        columns : sequence of str
            Series to complete.
        end : pandas.Period
            Last period to cover.

        Returns
        -------
        dict of str to pandas.Series
            Completed series on their native grids.

        Raises
        ------
        NowcastDataError
            If a series has no observation or too few for the AR model.

        Examples
        --------
        >>> import pandas as pd
        >>> from nowcastbox.core.data import MixedFrequencyData
        >>> idx = pd.period_range("2020-01", periods=4, freq="M")
        >>> mfd = MixedFrequencyData(pd.DataFrame({"x": [1.0, 2.0, 4.0, 3.0]}, index=idx), "M")
        >>> len(ARExtrapolator()(mfd, ["x"], pd.Period("2020-06", "M"))["x"])
        6
        """
        out: dict[str, pd.Series] = {}
        for col in columns:
            native = native_until(data, col, end)
            values = native.to_numpy(dtype=float)
            observed = np.flatnonzero(np.isfinite(values))
            if observed.size == 0:
                raise NowcastDataError(f"Predictor {col!r} has no observations.")
            n_ahead = len(values) - int(observed[-1]) - 1
            filled = ar_extend(values, n_ahead, int(self.ar_lags))
            out[col] = pd.Series(filled, index=native.index, name=col)
        logger.debug("AR(%d) extrapolation of %d series", self.ar_lags, len(columns))
        return out


def register_extrapolator(
    name: str, factory: ExtrapolatorFactory, *, overwrite: bool = False
) -> None:
    """Register a named extrapolator factory.

    Parameters
    ----------
    name : str
        Name used in ``extrapolation=`` options (e.g. ``"bvar"``).
    factory : callable
        Called with the keyword options of :func:`make_extrapolator`; returns an
        :class:`Extrapolator`.
    overwrite : bool, default False
        Replace an existing registration.

    Raises
    ------
    ValueError
        If ``name`` is empty or already registered (and ``overwrite`` is False).
    TypeError
        If ``factory`` is not callable.

    Examples
    --------
    >>> from nowcastbox.models.extrapolation import (
    ...     ARExtrapolator,
    ...     available_extrapolators,
    ...     register_extrapolator,
    ... )
    >>> register_extrapolator("ar2", lambda: ARExtrapolator(ar_lags=2), overwrite=True)
    >>> "ar2" in available_extrapolators()
    True
    """
    if not isinstance(name, str) or not name:
        raise ValueError("The extrapolator name must be a non-empty string.")
    if not callable(factory):
        raise TypeError("factory must be callable.")
    if name in _REGISTRY and not overwrite:
        raise ValueError(f"Extrapolator {name!r} is already registered.")
    _REGISTRY[name] = factory


def available_extrapolators() -> list[str]:
    """Names of the registered extrapolators.

    Returns
    -------
    list of str
        Sorted names.

    Examples
    --------
    >>> from nowcastbox.models.extrapolation import available_extrapolators
    >>> "ar" in available_extrapolators()
    True
    """
    return sorted(_REGISTRY)


def make_extrapolator(spec: str | Extrapolator, **options: Any) -> Extrapolator:
    """Build an extrapolator from a registered name or pass a callable through.

    Parameters
    ----------
    spec : str or callable
        Registered name (``"ar"``, ``"bvar"``) or an extrapolator.
    **options
        Keyword options of the factory (e.g. ``ar_lags=2`` for ``"ar"``); not allowed
        with a callable ``spec``.

    Returns
    -------
    Extrapolator
        The extrapolator.

    Raises
    ------
    ValueError
        If the name is unknown or options are given with a callable.
    TypeError
        If ``spec`` is neither a string nor callable, or the factory rejects an option.

    Examples
    --------
    >>> from nowcastbox.models.extrapolation import make_extrapolator
    >>> make_extrapolator("ar", ar_lags=3)
    ARExtrapolator(ar_lags=3)
    """
    if isinstance(spec, str):
        if spec not in _REGISTRY:
            raise ValueError(
                f"Unknown extrapolation {spec!r}; available: {available_extrapolators()}."
            )
        return _REGISTRY[spec](**options)
    if not callable(spec):
        raise TypeError(f"extrapolation must be a name or a callable, got {spec!r}.")
    if options:
        raise ValueError("Extrapolation options are only used with a registered name.")
    return spec


register_extrapolator("ar", ARExtrapolator)
