r"""Comparison of two data vintages: new releases, revisions and removed values.

Between an old information set :math:`\Omega_v` and a new one :math:`\Omega_{v+1}` a
cell of the panel can be

* **released** - missing in the old vintage, observed in the new one (source of
  *news*);
* **revised** - observed in both with different values (data revision);
* **removed** - observed in the old vintage only (rare: discontinued series,
  re-basing).

The news decomposition (Bańbura & Modugno, 2014) first moves from :math:`\Omega_v` to
the old pattern with revised values, then drops removed cells, and finally adds the new
releases, so that :math:`\Omega_v \to \Omega^{\ast} \subseteq \Omega_{v+1}` is nested.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from nowcastbox.news._model import to_frame

__all__ = ["VintageDiff", "compare_vintages", "data_revisions"]

BoolArray = NDArray[np.bool_]


@dataclass(frozen=True, eq=False)
class VintageDiff:
    """Cell masks separating two vintages on a common grid.

    Attributes
    ----------
    released : numpy.ndarray of bool
        Missing in the old vintage, observed in the new one.
    revised : numpy.ndarray of bool
        Observed in both, different values.
    removed : numpy.ndarray of bool
        Observed in the old vintage only.

    Examples
    --------
    >>> import numpy as np
    >>> d = compare_vintages(np.array([[1.0, np.nan]]), np.array([[2.0, 3.0]]))
    >>> d.released.tolist(), d.revised.tolist(), d.removed.tolist()
    ([[False, True]], [[True, False]], [[False, False]])
    """

    released: BoolArray
    revised: BoolArray
    removed: BoolArray

    @property
    def n_released(self) -> int:
        """Number of new releases."""
        return int(self.released.sum())

    @property
    def n_revised(self) -> int:
        """Number of revised values."""
        return int(self.revised.sum())

    @property
    def n_removed(self) -> int:
        """Number of removed values."""
        return int(self.removed.sum())


def compare_vintages(old: NDArray[np.float64], new: NDArray[np.float64]) -> VintageDiff:
    """Masks of released, revised and removed cells between two aligned arrays.

    Parameters
    ----------
    old, new : numpy.ndarray
        Values of the two vintages on the same grid (``NaN`` = missing).

    Returns
    -------
    VintageDiff
        Cell masks.

    Raises
    ------
    ValueError
        If the shapes differ.

    Examples
    --------
    >>> import numpy as np
    >>> compare_vintages(np.array([1.0]), np.array([np.nan])).n_removed
    1
    """
    old_a = np.asarray(old, dtype=np.float64)
    new_a = np.asarray(new, dtype=np.float64)
    if old_a.shape != new_a.shape:
        raise ValueError(f"Vintages have different shapes {old_a.shape} and {new_a.shape}.")
    obs_old = ~np.isnan(old_a)
    obs_new = ~np.isnan(new_a)
    both = obs_old & obs_new
    revised = np.zeros_like(both)
    revised[both] = old_a[both] != new_a[both]
    return VintageDiff(
        released=obs_new & ~obs_old,
        revised=revised,
        removed=obs_old & ~obs_new,
    )


def data_revisions(old_data: object, new_data: object) -> pd.DataFrame:
    """Long table of the values that changed between two vintages.

    Parameters
    ----------
    old_data, new_data : MixedFrequencyData or pandas.DataFrame
        Panels on the same base frequency (PeriodIndex). Only common columns are
        compared; the grids are aligned on their union.

    Returns
    -------
    pandas.DataFrame
        Columns ``series``, ``period`` (base-grid period), ``kind``
        (``"released"``, ``"revised"`` or ``"removed"``), ``old``, ``new`` and
        ``revision`` (``new - old``, NaN unless revised); ordered by series and period.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> idx = pd.period_range("2020-01", periods=2, freq="M")
    >>> old = pd.DataFrame({"a": [1.0, np.nan]}, index=idx)
    >>> new = pd.DataFrame({"a": [1.5, 2.0]}, index=idx)
    >>> data_revisions(old, new)[["kind", "revision"]].values.tolist()
    [['revised', 0.5], ['released', nan]]
    """
    old_f = to_frame(old_data)
    new_f = to_frame(new_data)
    columns = [c for c in new_f.columns if c in old_f.columns]
    index = old_f.index.union(new_f.index)
    a = old_f.reindex(index=index, columns=columns).to_numpy(dtype=np.float64)
    b = new_f.reindex(index=index, columns=columns).to_numpy(dtype=np.float64)
    diff = compare_vintages(a, b)
    rows: list[dict[str, object]] = []
    for j, name in enumerate(columns):
        for kind, mask in (
            ("released", diff.released),
            ("revised", diff.revised),
            ("removed", diff.removed),
        ):
            for t in np.flatnonzero(mask[:, j]):
                rows.append(
                    {
                        "series": name,
                        "period": index[t],
                        "kind": kind,
                        "old": a[t, j],
                        "new": b[t, j],
                        "revision": b[t, j] - a[t, j] if kind == "revised" else np.nan,
                    }
                )
    frame = pd.DataFrame(rows, columns=["series", "period", "kind", "old", "new", "revision"])
    if frame.empty:
        return frame
    order = {name: i for i, name in enumerate(columns)}
    frame["_o"] = frame["series"].map(order)
    frame = frame.sort_values(["_o", "period"], kind="stable").drop(columns="_o")
    return frame.reset_index(drop=True)
