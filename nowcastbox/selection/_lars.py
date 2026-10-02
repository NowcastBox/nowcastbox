r"""Least angle regression (Efron, Hastie, Johnstone & Tibshirani, 2004).

Implementation of the LARS algorithm of Section 2 of the paper, with the optional lasso
modification of Section 3.1. Starting from :math:`\hat\mu_0 = 0`, at every step

1. the current correlations :math:`\hat c = X'(y - \hat\mu)` are computed and the
   active set :math:`\mathcal A` holds the predictors with the largest absolute
   correlation :math:`\hat C = \max_j |\hat c_j|`, with signs :math:`s_j =
   \operatorname{sign}(\hat c_j)`;
2. the equiangular direction is (eqs. 2.4-2.6)

   .. math::

       X_{\mathcal A} = (s_j x_j)_{j\in\mathcal A},\quad
       \mathcal G_{\mathcal A} = X_{\mathcal A}'X_{\mathcal A},\quad
       A_{\mathcal A} = (1'\mathcal G_{\mathcal A}^{-1}1)^{-1/2},\quad
       w_{\mathcal A} = A_{\mathcal A}\mathcal G_{\mathcal A}^{-1}1,\quad
       u_{\mathcal A} = X_{\mathcal A} w_{\mathcal A};

3. with :math:`a = X'u_{\mathcal A}`, the step length is (eq. 2.13)

   .. math::

       \hat\gamma = {\min_{j\in\mathcal A^c}}^{+}
       \Big\{\frac{\hat C - \hat c_j}{A_{\mathcal A} - a_j},
             \frac{\hat C + \hat c_j}{A_{\mathcal A} + a_j}\Big\},

   and the minimising predictor joins the active set. When no inactive predictor is
   left the last step goes to the least-squares fit, :math:`\hat\gamma = \hat C /
   A_{\mathcal A}`.

**Lasso modification** (eqs. 3.4-3.6). With :math:`d_j = s_j w_{\mathcal A j}`, the
coefficient of an active predictor changes sign at :math:`\gamma_j = -\hat\beta_j/d_j`;
if :math:`\tilde\gamma = \min^{+}\gamma_j < \hat\gamma` the step stops at
:math:`\tilde\gamma` and that predictor leaves the active set.

The penalties reported along the path follow the usual convention
:math:`\alpha = \hat C / n`.

References
----------
Efron, B., Hastie, T., Johnstone, I., & Tibshirani, R. (2004). Least angle regression.
*The Annals of Statistics*, 32(2), 407-499.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from nowcastbox._logging import get_logger
from nowcastbox.core.exceptions import ConvergenceWarning, NowcastDataError

__all__ = ["LarsPath", "lars_path"]

logger = get_logger(__name__)

LarsMethod = Literal["lar", "lasso"]


@dataclass(frozen=True)
class LarsPath:
    r"""Least angle regression path (output of :func:`lars_path`).

    Attributes
    ----------
    alphas : numpy.ndarray
        ``(K + 1,)`` penalties :math:`\hat C / n` at the start of the path and after
        each of the ``K`` steps (decreasing).
    coefs : numpy.ndarray
        ``(p, K + 1)`` coefficients at each knot (the first column is zero).
    active : list[int]
        Active set at the end of the path, in order of entry.
    entry_order : list[int]
        Every predictor that entered the path, in order of **first** entry (with the
        lasso modification a dropped predictor keeps its first position).
    drops : list[int]
        Predictors dropped by the lasso modification, in order.
    method : str
        ``"lar"`` or ``"lasso"``.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    >>> path = lars_path(X, np.array([2.0, 1.0, 0.0]))
    >>> path.entry_order, path.n_steps
    ([0, 1], 2)
    """

    alphas: np.ndarray
    coefs: np.ndarray
    active: list[int]
    entry_order: list[int]
    drops: list[int] = field(default_factory=list)
    method: str = "lar"

    @property
    def n_steps(self) -> int:
        """Number of LARS steps (knots after the origin)."""
        return int(self.alphas.shape[0]) - 1

    @property
    def entry_step(self) -> np.ndarray:
        """Step (1-based) at which each predictor first entered; NaN if it never did.

        Returns
        -------
        numpy.ndarray
            ``(p,)`` float array.

        Examples
        --------
        >>> import numpy as np
        >>> X = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0, 0, 0]])
        >>> lars_path(X, np.array([0.0, 2.0, 1.0, 0.0]), max_steps=1).entry_step.tolist()
        [nan, 1.0, 2.0]
        """
        steps = np.full(self.coefs.shape[0], np.nan)
        for pos, j in enumerate(self.entry_order):
            steps[j] = pos + 1
        return steps


@dataclass
class _State:
    """Mutable state of the LARS iterations."""

    beta: np.ndarray
    active: list[int]
    entry_order: list[int]
    drops: list[int] = field(default_factory=list)
    alphas: list[float] = field(default_factory=list)
    coefs: list[np.ndarray] = field(default_factory=list)


def _check_args(method: str, max_steps: int | None, tol: float) -> None:
    if method not in ("lar", "lasso"):
        raise ValueError(f"method must be 'lar' or 'lasso'; got {method!r}.")
    if max_steps is not None and (
        isinstance(max_steps, bool) or not isinstance(max_steps, int | np.integer) or max_steps < 1
    ):
        raise ValueError(f"max_steps must be a positive integer; got {max_steps!r}.")
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError(f"tol must be positive; got {tol!r}.")


def _validate(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if X.ndim != 2 or X.shape[0] != y.shape[0] or X.shape[1] == 0:
        raise NowcastDataError(
            f"X must be 2-D with {y.shape[0]} rows and at least one column; got {X.shape}."
        )
    if not (np.isfinite(X).all() and np.isfinite(y).all()):
        raise NowcastDataError("X and y must be finite (remove missing values first).")
    return X, y


def _direction(X: np.ndarray, signs: np.ndarray, active: list[int]) -> tuple[float, np.ndarray]:
    """Return ``A_A`` and the weights ``w_A`` of the equiangular direction (eq. 2.5)."""
    xa = X[:, active] * signs
    gram = xa.T @ xa
    ones = np.ones(len(active))
    try:
        chol = np.linalg.cholesky(gram)
    except np.linalg.LinAlgError as exc:
        raise np.linalg.LinAlgError("degenerate active set") from exc
    ginv1 = np.linalg.solve(chol.T, np.linalg.solve(chol, ones))
    denom = float(ones @ ginv1)
    a_a = 1.0 / np.sqrt(denom)
    return a_a, a_a * ginv1


def _entry_step(
    c: np.ndarray, a: np.ndarray, big_c: float, a_a: float, inactive: np.ndarray
) -> tuple[float, int | None]:
    """Smallest positive step at which an inactive predictor joins (eq. 2.13)."""
    full = big_c / a_a
    if not inactive.any():
        return full, None
    with np.errstate(divide="ignore", invalid="ignore"):
        g1 = (big_c - c) / (a_a - a)
        g2 = (big_c + c) / (a_a + a)
    # Filter the two crossings separately: a predictor just dropped by the lasso
    # modification sits on one of the boundaries (a crossing at ~0) and may still
    # re-enter later through the other one.
    eps = 1e-12 * full
    cand = np.minimum(
        np.where(np.isfinite(g1) & (g1 > eps), g1, np.inf),
        np.where(np.isfinite(g2) & (g2 > eps), g2, np.inf),
    )
    cand[~inactive] = np.inf
    j = int(np.argmin(cand))
    if cand[j] >= full * (1.0 - 1e-9):  # joins only at the least-squares fit
        return full, None
    return float(cand[j]), j


def _drop_step(beta: np.ndarray, d: np.ndarray, active: list[int]) -> tuple[float, int | None]:
    """Smallest positive step at which an active coefficient crosses zero (eq. 3.4)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        gam = -beta[active] / d
    gam = np.where(np.isfinite(gam) & (gam > 1e-14), gam, np.inf)
    k = int(np.argmin(gam))
    if not np.isfinite(gam[k]):
        return np.inf, None
    return float(gam[k]), active[k]


def _one_step(X: np.ndarray, y: np.ndarray, state: _State, method: str, cap: int) -> bool:
    """Perform one LARS step; return False when the path is complete."""
    n, p = X.shape
    c = X.T @ (y - X @ state.beta)
    big_c = float(np.max(np.abs(c[state.active])))
    signs = np.sign(c[state.active])
    try:
        a_a, w = _direction(X, signs, state.active)
    except np.linalg.LinAlgError:
        warnings.warn(
            f"LARS stopped after {len(state.alphas) - 1} steps: the active set is collinear.",
            ConvergenceWarning,
            stacklevel=4,
        )
        return False
    a = X.T @ (X[:, state.active] @ (signs * w))
    inactive = np.ones(p, dtype=bool)
    inactive[state.active] = False
    if len(state.active) >= cap:
        inactive[:] = False
    gamma, joining = _entry_step(c, a, big_c, a_a, inactive)
    d = signs * w
    leaving = None
    if method == "lasso":
        gam_drop, leaving = _drop_step(state.beta, d, state.active)
        if leaving is not None and gam_drop < gamma:
            gamma, joining = gam_drop, None
        else:
            leaving = None
    state.beta[state.active] += gamma * d
    if leaving is not None:
        state.beta[leaving] = 0.0
        state.active.remove(leaving)
        state.drops.append(leaving)
    state.alphas.append(max(big_c - gamma * a_a, 0.0) / n)
    state.coefs.append(state.beta.copy())
    if joining is not None:
        state.active.append(joining)
        if joining not in state.entry_order:
            state.entry_order.append(joining)
    return joining is not None or leaving is not None


def lars_path(
    X: np.ndarray,
    y: np.ndarray,
    *,
    method: LarsMethod = "lar",
    max_steps: int | None = None,
    tol: float = 1e-12,
) -> LarsPath:
    """Compute the least angle regression path (Efron et al., 2004).

    The data are used as given (no centring or scaling); centre ``y`` and the columns
    of ``X`` beforehand to fit an intercept, and standardise ``X`` so that the entry
    order does not depend on the units of the predictors.

    Parameters
    ----------
    X : numpy.ndarray
        ``(n, p)`` design matrix (finite).
    y : numpy.ndarray
        ``(n,)`` response.
    method : {"lar", "lasso"}, default "lar"
        Plain LARS (Section 2) or the lasso modification (Section 3.1), which drops a
        predictor whose coefficient crosses zero.
    max_steps : int, optional
        Maximum number of steps (default: no limit; at most ``min(n, p)`` predictors
        are ever active simultaneously).
    tol : float, default 1e-12
        The path stops when the largest absolute correlation falls below ``tol``
        times its initial value.

    Returns
    -------
    LarsPath
        Penalties, coefficients and entry order.

    Raises
    ------
    ValueError
        Invalid ``method``, ``max_steps`` or ``tol``.
    NowcastDataError
        Non-finite or misaligned data.

    Warns
    -----
    ConvergenceWarning
        If the path stops early because the active predictors are collinear.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(50, 4))
    >>> y = 3 * X[:, 2] - X[:, 0] + 0.1 * rng.normal(size=50)
    >>> path = lars_path(X, y)
    >>> path.entry_order[:2]
    [2, 0]
    >>> (np.round(path.coefs[:, -1], 1) + 0.0).tolist()
    [-1.0, 0.0, 3.0, 0.0]
    """
    _check_args(method, max_steps, tol)
    X, y = _validate(X, y)
    n, p = X.shape
    c0 = X.T @ y
    c_max = float(np.max(np.abs(c0)))
    state = _State(beta=np.zeros(p), active=[], entry_order=[])
    state.alphas.append(c_max / n)
    state.coefs.append(state.beta.copy())
    if c_max > 0:
        first = int(np.argmax(np.abs(c0)))
        state.active.append(first)
        state.entry_order.append(first)
    limit = max_steps if max_steps is not None else 8 * max(n, p)
    while state.active and len(state.alphas) - 1 < limit and state.alphas[-1] * n > tol * c_max:
        if not _one_step(X, y, state, method, min(n, p)):
            break
    logger.debug("LARS (%s): %d steps, %d active", method, len(state.alphas) - 1, len(state.active))
    return LarsPath(
        alphas=np.asarray(state.alphas),
        coefs=np.column_stack(state.coefs),
        active=list(state.active),
        entry_order=list(state.entry_order),
        drops=list(state.drops),
        method=method,
    )
