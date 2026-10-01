r"""Convergence diagnostics of the EM algorithm.

The EM algorithm of Bańbura & Modugno (2014) increases the log-likelihood
:math:`\ell_k` at every iteration (Dempster, Laird & Rubin, 1977); convergence is
declared when the relative change

.. math:: c_k = \frac{\ell_k - \ell_{k-1}}{(|\ell_k| + |\ell_{k-1}| + \epsilon)/2}

falls below ``tol`` (Doz, Giannone & Reichlin, 2012). Non-monotone steps
(:math:`c_k < 0` beyond numerical noise) signal numerical problems, approximations in
the M-step (e.g. the initial-state terms ignored by Bańbura-Modugno) or identification
issues; they are reported and trigger a
:class:`~nowcastbox.core.exceptions.ConvergenceWarning`.

References
----------
Bańbura, M., & Modugno, M. (2014). Maximum likelihood estimation of factor models on
datasets with arbitrary pattern of missing data. *Journal of Applied Econometrics*,
29(1), 133-160.

Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). Maximum likelihood from
incomplete data via the EM algorithm. *Journal of the Royal Statistical Society B*,
39(1), 1-38.

Doz, C., Giannone, D., & Reichlin, L. (2012). A quasi-maximum likelihood approach for
large, approximate dynamic factor models. *Review of Economics and Statistics*, 94(4),
1014-1024.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from nowcastbox.core.exceptions import ConvergenceWarning, NowcastDataError

__all__ = ["ConvergenceDiagnostics", "em_convergence", "loglikelihood_path"]

_EPS = float(np.finfo(np.float64).eps)


@dataclass(frozen=True, eq=False)
class ConvergenceDiagnostics:
    r"""Diagnostics of the log-likelihood path of an iterative estimator.

    Parameters
    ----------
    path : pandas.DataFrame
        Indexed by ``iteration`` (0 = initial parameters): ``loglikelihood``,
        ``change`` (:math:`\ell_k - \ell_{k-1}`), ``relative_change`` (:math:`c_k`)
        and ``decrease`` (non-monotone step).
    converged : bool
        Convergence flag (from the results, or :math:`|c_K| <` ``tol``).
    n_iter : int
        Number of iterations (``len(path) - 1``).
    tol : float or None
        Tolerance on the relative change.
    n_decreases : int
        Number of non-monotone steps.
    largest_decrease : float
        Largest relative decrease (0 when monotone).
    last_relative_change : float
        :math:`c_K` (``NaN`` without iterations).

    Examples
    --------
    >>> from nowcastbox.diagnostics import em_convergence
    >>> diag = em_convergence([-120.0, -101.0, -100.5, -100.49], tol=1e-3)
    >>> diag.converged, diag.n_iter, diag.monotone
    (True, 3, True)
    """

    path: pd.DataFrame
    converged: bool
    n_iter: int
    tol: float | None
    n_decreases: int
    largest_decrease: float
    last_relative_change: float

    @property
    def monotone(self) -> bool:
        """Whether the log-likelihood never decreased."""
        return self.n_decreases == 0

    def to_frame(self) -> pd.DataFrame:
        """Copy of :attr:`path`.

        Returns
        -------
        pandas.DataFrame
            Log-likelihood path.

        Examples
        --------
        >>> from nowcastbox.diagnostics import em_convergence
        >>> em_convergence([-3.0, -2.0]).to_frame()["change"].tolist()
        [nan, 1.0]
        """
        return self.path.copy()

    def summary_frame(self) -> pd.DataFrame:
        """One-row overview (tidy).

        Returns
        -------
        pandas.DataFrame
            ``converged``, ``n_iter``, ``tol``, ``monotone``, ``n_decreases``,
            ``largest_decrease``, ``last_relative_change``, ``initial_loglikelihood``,
            ``final_loglikelihood``.

        Examples
        --------
        >>> from nowcastbox.diagnostics import em_convergence
        >>> int(em_convergence([-3.0, -2.0]).summary_frame().loc[0, "n_iter"])
        1
        """
        ll = self.path["loglikelihood"]
        return pd.DataFrame(
            [
                {
                    "converged": self.converged,
                    "n_iter": self.n_iter,
                    "tol": self.tol,
                    "monotone": self.monotone,
                    "n_decreases": self.n_decreases,
                    "largest_decrease": self.largest_decrease,
                    "last_relative_change": self.last_relative_change,
                    "initial_loglikelihood": float(ll.iloc[0]),
                    "final_loglikelihood": float(ll.iloc[-1]),
                }
            ]
        )

    def summary(self) -> str:
        """Text summary.

        Returns
        -------
        str
            Convergence overview.

        Examples
        --------
        >>> from nowcastbox.diagnostics import em_convergence
        >>> print(em_convergence([-3.0, -2.0], tol=1.0).summary())  # doctest: +ELLIPSIS
        EM convergence
          Converged ...
        """
        row = self.summary_frame().iloc[0]
        tol = "-" if self.tol is None else f"{self.tol:.3g}"
        lines = [
            "EM convergence",
            f"  {'Converged':<26}{bool(row['converged'])} (tol={tol})",
            f"  {'Iterations':<26}{self.n_iter}",
            f"  {'Log-likelihood':<26}{row['initial_loglikelihood']:.4f} -> "
            f"{row['final_loglikelihood']:.4f}",
            f"  {'Last relative change':<26}{self.last_relative_change:.3g}",
            f"  {'Non-monotone steps':<26}{self.n_decreases}"
            + (f" (largest {self.largest_decrease:.3g})" if self.n_decreases else ""),
        ]
        return "\n".join(lines)


def loglikelihood_path(source: Any) -> np.ndarray | None:
    """Extract the log-likelihood path from results (or return ``None``).

    Parameters
    ----------
    source : results object or array_like
        :class:`~nowcastbox.models.MixedFreqDFMResults` (``loglikelihood_path``),
        any results with ``info["loglikelihood_path"]``, or the path itself.

    Returns
    -------
    numpy.ndarray or None
        Path as a float array, ``None`` when unavailable.

    Examples
    --------
    >>> from nowcastbox.diagnostics import loglikelihood_path
    >>> loglikelihood_path([1, 2]).tolist()
    [1.0, 2.0]
    """
    path = getattr(source, "loglikelihood_path", None)
    if path is None:
        info = getattr(source, "info", None)
        if isinstance(info, dict):
            path = info.get("loglikelihood_path")
        elif isinstance(source, list | tuple | np.ndarray | pd.Series):
            path = source
    if path is None:
        return None
    return np.asarray(path, dtype=np.float64).ravel()


def em_convergence(
    source: Any,
    *,
    tol: float | None = None,
    decrease_tol: float = 1e-9,
    warn: bool = True,
) -> ConvergenceDiagnostics:
    r"""Diagnose the convergence of the EM algorithm from its log-likelihood path.

    Parameters
    ----------
    source : results object or array_like
        :class:`~nowcastbox.models.MixedFreqDFMResults`, results storing
        ``info["loglikelihood_path"]``, or a path :math:`(\ell_0, \dots, \ell_K)`.
    tol : float, optional
        Tolerance on the relative change; defaults to ``results.model_params["tol"]``.
    decrease_tol : float, default 1e-9
        Relative changes below ``-decrease_tol`` count as non-monotone steps.
    warn : bool, default True
        Issue a :class:`~nowcastbox.core.exceptions.ConvergenceWarning` for
        non-monotone steps or lack of convergence.

    Returns
    -------
    ConvergenceDiagnostics
        Path and summary.

    Raises
    ------
    NowcastDataError
        If no (finite, non-empty) log-likelihood path is available.

    Warns
    -----
    ConvergenceWarning
        Non-monotone log-likelihood or no convergence (``warn=True``).

    Examples
    --------
    >>> import warnings
    >>> from nowcastbox.diagnostics import em_convergence
    >>> with warnings.catch_warnings(record=True) as w:
    ...     warnings.simplefilter("always")
    ...     diag = em_convergence([-10.0, -9.0, -9.5, -8.99], tol=1e-6)
    >>> diag.n_decreases, len(w) > 0
    (1, True)
    """
    path = loglikelihood_path(source)
    if path is None or path.size == 0 or not np.isfinite(path).all():
        raise NowcastDataError("No finite log-likelihood path is available for the EM diagnostics.")
    if tol is None:
        params = getattr(source, "model_params", None)
        tol = params.get("tol") if isinstance(params, dict) else None
    change = np.concatenate([[np.nan], np.diff(path)])
    scale = (np.abs(path[1:]) + np.abs(path[:-1]) + _EPS) / 2.0
    rel = np.concatenate([[np.nan], np.diff(path) / scale])
    decrease = np.concatenate([[False], rel[1:] < -decrease_tol])
    frame = pd.DataFrame(
        {"loglikelihood": path, "change": change, "relative_change": rel, "decrease": decrease},
        index=pd.RangeIndex(path.size, name="iteration"),
    )
    last = float(rel[-1])
    flag = getattr(source, "converged", None)
    if isinstance(flag, bool | np.bool_):
        converged = bool(flag)
    else:
        converged = bool(tol is not None and path.size > 1 and abs(last) < float(tol))
    n_dec = int(decrease.sum())
    diag = ConvergenceDiagnostics(
        path=frame,
        converged=converged,
        n_iter=path.size - 1,
        tol=None if tol is None else float(tol),
        n_decreases=n_dec,
        largest_decrease=float(-rel[1:][decrease[1:]].min()) if n_dec else 0.0,
        last_relative_change=last,
    )
    if warn:
        _warn(diag)
    return diag


def _warn(diag: ConvergenceDiagnostics) -> None:
    if diag.n_decreases:
        steps = diag.path.index[diag.path["decrease"].to_numpy(bool)].tolist()
        warnings.warn(
            f"The EM log-likelihood decreased in {diag.n_decreases} iteration(s) "
            f"{steps[:10]} (largest relative decrease {diag.largest_decrease:.3g}).",
            ConvergenceWarning,
            stacklevel=3,
        )
    if not diag.converged and diag.n_iter > 0:
        warnings.warn(
            f"The EM algorithm did not converge after {diag.n_iter} iterations "
            f"(last relative change {diag.last_relative_change:.3g}).",
            ConvergenceWarning,
            stacklevel=3,
        )
