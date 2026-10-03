r"""Normal-Inverse-Wishart prior machinery of the large Bayesian VAR.

Building blocks of :mod:`nowcastbox.models.bvar` (Phase 3 of the ECB-parity plan): the
conjugate prior of a VAR(:math:`p`), its closed-form posterior and marginal likelihood,
posterior simulation and the hierarchical choice of the prior tightness.

Model
-----
The VAR is written in stacked form

.. math::

    Y = X B + E, \qquad \operatorname{vec}(E) \sim N(0, \Sigma \otimes I_T),

where row :math:`t` of :math:`Y` (:math:`T \times n`) is :math:`y_t'` and row :math:`t`
of :math:`X` (:math:`T \times k`, :math:`k = 1 + np`) is
:math:`x_t' = (1, y_{t-1}', \dots, y_{t-p}')` (the constant is optional). The
Normal-Inverse-Wishart (NIW) prior is

.. math::

    \Sigma \sim IW(\Psi, d), \qquad
    \operatorname{vec}(B) \mid \Sigma \sim N(\operatorname{vec}(b), \Sigma \otimes \Omega).

**Minnesota prior** (Litterman, 1986; Doan, Litterman & Sims, 1984) in the
parameterisation of Giannone, Lenza & Primiceri (2015, GLP): :math:`\Psi =
\operatorname{diag}(\psi)`, :math:`d = n + 2`, :math:`\Omega` diagonal with
:math:`\lambda^2 (d - n - 1) / (l^{\kappa} \psi_j)` for lag :math:`l` of variable
:math:`j` (:math:`\kappa = 2`) and a diffuse variance for the constant; :math:`b` has
:math:`\delta_i` (1 = random walk, 0 = white noise) on the own first lag of variable
:math:`i` and zeros elsewhere. Hence
:math:`\operatorname{Var}(B_{l,ji} \mid \Sigma) = \lambda^2 \Sigma_{ii} / (l^\kappa
\operatorname{E}[\Sigma_{jj}])`.

**Sum-of-coefficients** (Doan, Litterman & Sims, 1984) and **dummy-initial-observation**
(Sims, 1993; Sims & Zha, 1998) priors are dummy observations :math:`(Y^+, X^+)`
prepended to the data, with :math:`\bar y_0` the mean of the first :math:`p`
observations: :math:`n` rows :math:`Y^+ = \operatorname{diag}(\bar y_0)/\mu`,
:math:`X^+ = (0, Y^+, \dots, Y^+)`, and one row :math:`y^{++} = \bar y_0'/\delta`,
:math:`x^{++} = (1/\delta, y^{++}, \dots, y^{++})` (Bańbura, Giannone & Reichlin, 2010;
GLP 2015, §3). Small :math:`\mu` pushes :math:`\sum_l B_l` to the identity (unit roots,
no cointegration); small :math:`\delta` pushes the VAR towards a common stochastic trend
started at :math:`\bar y_0`.

**Blocked random walk.** :class:`PriorSettings` also takes a full prior mean
:math:`\operatorname{E}[A_1]` (not only own-lag means) and *unit-root groups*: variables
of one group share a single unit root in the two dummy priors (one sum-of-coefficients
row per group, common starting level). In the blocked VAR of
:mod:`nowcastbox.models.bvar` the three monthly blocks :math:`x^{(1)}, x^{(2)}, x^{(3)}`
of a monthly random walk satisfy :math:`x^{(m)}_t = x^{(3)}_{t-1} + \text{noise}`
(each month is centred on the last month of the previous quarter), i.e.
:math:`\operatorname{E}[A_1]` has ones in the column of :math:`x^{(3)}_{t-1}` and
:math:`\sum_l A_l \iota_g = \iota_g` for the group :math:`g` of the three blocks. The
Minnesota variances keep the quarterly lag decay (the conjugate prior needs
:math:`\Omega` common to all equations).

Posterior and marginal likelihood
---------------------------------
With :math:`\bar\Omega = (X'X + \Omega^{-1})^{-1}` and
:math:`\bar B = \bar\Omega (X'Y + \Omega^{-1} b)`, the posterior is
:math:`\Sigma \mid Y \sim IW(\bar\Psi, d + T)`, :math:`\operatorname{vec}(B) \mid
\Sigma, Y \sim N(\operatorname{vec}(\bar B), \Sigma \otimes \bar\Omega)` with
:math:`\bar\Psi = \Psi + \hat E'\hat E + (\bar B - b)'\Omega^{-1}(\bar B - b)`,
:math:`\hat E = Y - X\bar B`, and the marginal likelihood is (GLP 2015, appendix)

.. math::

    p(Y) = \pi^{-nT/2} \frac{\Gamma_n(\frac{d+T}{2})}{\Gamma_n(\frac d2)}
    |\Omega|^{-n/2} |\Psi|^{d/2} |X'X + \Omega^{-1}|^{-n/2} |\bar\Psi|^{-(d+T)/2}.

It is computed with :math:`|\Omega|\,|X'X + \Omega^{-1}| = |I_k + \Omega^{1/2} X'X
\Omega^{1/2}| = |R|^2`, where :math:`R` is the triangular factor of the thin QR
decomposition of the stacked matrix :math:`(\Omega^{1/2}X', I_k)'`; :math:`\bar B` is the
least-squares solution of that stacked system. The cross-product :math:`X'X` is never
formed: it squares the condition number, which loses all accuracy when the dummy rows are
large (data in levels with a tight sum-of-coefficients prior). With dummy
observations the likelihood of the actual data is the ratio
:math:`p(Y \mid Y^+) = p(Y, Y^+) / p(Y^+)`.

Hyperparameters
---------------
:func:`select_hyperparameters` maximises :math:`\log p(Y \mid \lambda, \mu, \delta,
\psi) + \log p(\lambda, \mu, \delta, \psi)` over the logarithms of the
hyperparameters (L-BFGS-B), with GLP's hyperpriors: Gamma with mode 0.2 and standard
deviation 0.4 for :math:`\lambda`, mode 1 and standard deviation 1 for :math:`\mu` and
:math:`\delta`, inverse-Gamma with shape and scale :math:`0.02^2` for each
:math:`\psi_j`. It also returns a numerical inverse Hessian of the objective on the log
scale (proposal covariance for a later Metropolis step).

Examples
--------
>>> import numpy as np
>>> from nowcastbox.models._bvar_prior import (
...     BVARHyperparameters,
...     VARSystem,
...     log_marginal_likelihood,
...     posterior,
... )
>>> rng = np.random.default_rng(0)
>>> y = np.cumsum(rng.standard_normal((60, 2)), axis=0)
>>> system = VARSystem.from_array(y, lags=2)
>>> hyper = BVARHyperparameters(lambda_=0.2, psi=system.ar_residual_variances(), mu=1.0, delta=1.0)
>>> post = posterior(system, hyper)
>>> post.mean.shape, post.dof
((5, 2), 65.0)
>>> bool(np.isfinite(log_marginal_likelihood(system, hyper)))
True
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass, field, replace

import numpy as np
import scipy.linalg
import scipy.optimize
from scipy.special import gammaln

from nowcastbox._logging import get_logger
from nowcastbox.core.exceptions import ConvergenceWarning, NowcastDataError

__all__ = [
    "GLP_HYPERPRIORS",
    "BVARHyperparameters",
    "GLPHyperpriors",
    "GammaHyperprior",
    "HyperGradient",
    "HyperparameterSelection",
    "InverseGammaHyperprior",
    "NIWPosterior",
    "NIWPrior",
    "PriorSettings",
    "VARSystem",
    "dummy_initial_observation_dummies",
    "implied_prior_from_dummies",
    "log_hyperprior",
    "log_hyperprior_gradient",
    "log_marginal_likelihood",
    "log_marginal_likelihood_gradient",
    "minnesota_dummies",
    "minnesota_prior",
    "niw_posterior",
    "numerical_hessian",
    "posterior",
    "prior_dummies",
    "select_hyperparameters",
    "sum_of_coefficients_dummies",
    "var_design",
]

logger = get_logger(__name__)

_PENALTY = 1e10
_ESTIMABLE = ("lambda", "mu", "delta", "psi")


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def var_design(y: np.ndarray, lags: int, *, constant: bool = True) -> tuple[np.ndarray, np.ndarray]:
    r"""Stacked regression form :math:`Y = XB + E` of a VAR(:math:`p`).

    Parameters
    ----------
    y : numpy.ndarray
        Data, shape ``(T + p, n)`` (rows = periods, oldest first), without missing values.
    lags : int
        VAR order :math:`p \ge 1`.
    constant : bool, default True
        Include a constant as the first column of :math:`X`.

    Returns
    -------
    Y : numpy.ndarray
        Shape ``(T, n)``: observations ``p, ..., T + p - 1``.
    X : numpy.ndarray
        Shape ``(T, k)``, ``k = constant + n p``; columns ``(1, y_{t-1}', ..., y_{t-p}')``.

    Raises
    ------
    NowcastDataError
        If ``y`` is not 2-D, has missing/infinite values or not more than ``lags`` rows.
    ValueError
        If ``lags < 1``.

    Examples
    --------
    >>> import numpy as np
    >>> Y, X = var_design(np.arange(8.0).reshape(4, 2), lags=1)
    >>> Y.tolist(), X.tolist()[0]
    ([[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]], [1.0, 0.0, 1.0])
    """
    if lags < 1:
        raise ValueError(f"lags must be >= 1, got {lags}")
    arr = np.asarray(y, dtype=float)
    if arr.ndim != 2:
        raise NowcastDataError(f"VAR data must be 2-D (periods x variables), got {arr.ndim}-D")
    if not np.all(np.isfinite(arr)):
        raise NowcastDataError("VAR data contain missing or infinite values")
    if arr.shape[0] <= lags:
        raise NowcastDataError(f"need more than {lags} observations, got {arr.shape[0]}")
    T = arr.shape[0] - lags
    blocks = [arr[lags - lag : lags - lag + T] for lag in range(1, lags + 1)]
    if constant:
        blocks.insert(0, np.ones((T, 1)))
    return arr[lags:].copy(), np.hstack(blocks)


@dataclass(frozen=True)
class VARSystem:
    """Data of a VAR(:math:`p`) in stacked regression form.

    Parameters
    ----------
    Y : numpy.ndarray
        Left-hand side, shape ``(T, n)``.
    X : numpy.ndarray
        Regressors, shape ``(T, k)`` (see :func:`var_design`).
    y0_mean : numpy.ndarray
        Mean of the first ``lags`` (pre-sample) observations, shape ``(n,)``; used by the
        sum-of-coefficients and dummy-initial-observation priors.
    lags : int
        VAR order :math:`p`.
    constant : bool, default True
        Whether the first column of ``X`` is a constant.

    Examples
    --------
    >>> import numpy as np
    >>> s = VARSystem.from_array(np.arange(10.0).reshape(5, 2), lags=2)
    >>> s.n, s.n_obs, s.k, s.y0_mean.tolist()
    (2, 3, 5, [1.0, 2.0])
    """

    Y: np.ndarray
    X: np.ndarray
    y0_mean: np.ndarray
    lags: int
    constant: bool = True

    @classmethod
    def from_array(cls, y: np.ndarray, lags: int, *, constant: bool = True) -> VARSystem:
        """Build the system from a data matrix.

        Parameters
        ----------
        y : numpy.ndarray
            Data, shape ``(T + p, n)``, oldest first, no missing values.
        lags : int
            VAR order :math:`p`.
        constant : bool, default True
            Include a constant.

        Returns
        -------
        VARSystem
            The stacked system.

        Raises
        ------
        NowcastDataError
            See :func:`var_design`.
        ValueError
            If ``lags < 1``.

        Examples
        --------
        >>> import numpy as np
        >>> VARSystem.from_array(np.ones((4, 3)), lags=1, constant=False).k
        3
        """
        Y, X = var_design(y, lags, constant=constant)
        y0 = np.asarray(y, dtype=float)[:lags].mean(axis=0)
        return cls(Y=Y, X=X, y0_mean=y0, lags=lags, constant=constant)

    @property
    def n(self) -> int:
        """Number of variables.

        Returns
        -------
        int
            :math:`n`.

        Examples
        --------
        >>> import numpy as np
        >>> VARSystem.from_array(np.ones((4, 3)), lags=1).n
        3
        """
        return int(self.Y.shape[1])

    @property
    def n_obs(self) -> int:
        """Number of observations in the likelihood (after the pre-sample).

        Returns
        -------
        int
            :math:`T`.

        Examples
        --------
        >>> import numpy as np
        >>> VARSystem.from_array(np.ones((6, 3)), lags=2).n_obs
        4
        """
        return int(self.Y.shape[0])

    @property
    def k(self) -> int:
        r"""Number of coefficients per equation.

        Returns
        -------
        int
            :math:`k = \text{constant} + np`.

        Examples
        --------
        >>> import numpy as np
        >>> VARSystem.from_array(np.ones((6, 3)), lags=2).k
        7
        """
        return int(self.X.shape[1])

    def lag_column(self, lag: int, variable: int) -> int:
        r"""Column of :math:`X` (row of :math:`B`) holding ``variable`` at ``lag``.

        Parameters
        ----------
        lag : int
            Lag :math:`l \in \{1, \dots, p\}`.
        variable : int
            Variable index :math:`j \in \{0, \dots, n-1\}`.

        Returns
        -------
        int
            Column index.

        Examples
        --------
        >>> import numpy as np
        >>> VARSystem.from_array(np.ones((6, 3)), lags=2).lag_column(2, 1)
        5
        """
        return int(self.constant) + (lag - 1) * self.n + variable

    def ar_residual_variances(self, lags: int | None = None) -> np.ndarray:
        r"""Residual variances of univariate AR regressions (default scale :math:`\psi`).

        Each variable is regressed by OLS on the constant (if any) and its own first
        ``lags`` lags on the sample of the system — the usual choice of the Minnesota
        scales (Litterman, 1986) and GLP's starting value.

        Parameters
        ----------
        lags : int, optional
            AR order (default: the VAR order; at most the VAR order).

        Returns
        -------
        numpy.ndarray
            Shape ``(n,)``, positive (floored at ``1e-12`` times the sample variance).

        Raises
        ------
        ValueError
            If ``lags`` is not in ``1..p``.

        Examples
        --------
        >>> import numpy as np
        >>> rng = np.random.default_rng(1)
        >>> s = VARSystem.from_array(2.0 * rng.standard_normal((400, 2)), lags=1)
        >>> bool(np.all(np.abs(s.ar_residual_variances() - 4.0) < 1.0))
        True
        """
        q = self.lags if lags is None else lags
        if not 1 <= q <= self.lags:
            raise ValueError(f"lags must be in 1..{self.lags}, got {q}")
        out = np.empty(self.n)
        for j in range(self.n):
            cols = [self.lag_column(lag, j) for lag in range(1, q + 1)]
            if self.constant:
                cols.insert(0, 0)
            Xj = self.X[:, cols]
            coef = np.linalg.lstsq(Xj, self.Y[:, j], rcond=None)[0]
            resid = self.Y[:, j] - Xj @ coef
            df = max(self.n_obs - len(cols), 1)
            floor = 1e-12 * max(float(np.var(self.Y[:, j])), 1.0)
            out[j] = max(float(resid @ resid) / df, floor)
        return out


# ---------------------------------------------------------------------------
# prior
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PriorSettings:
    r"""Fixed (not estimated) settings of the Minnesota prior.

    Parameters
    ----------
    prior_mean : {"random_walk", "white_noise"}, float, sequence of float or matrix
        Prior mean of the first-lag coefficient matrix :math:`A_1` (all other lags have
        mean zero). A name, a number or one number per variable set the own first-lag
        means :math:`\delta_i`: 1 (``"random_walk"``, for levels), 0 (``"white_noise"``,
        for growth rates), a common value or one value per variable. An ``(n, n)``
        matrix gives the full :math:`\operatorname{E}[A_1]`: entry ``(i, j)`` is the
        prior mean of the coefficient of :math:`y_{j,t-1}` in equation :math:`i` (used
        by the blocked random walk of :class:`~nowcastbox.models.LargeBVAR`, where every
        month of a monthly series is centred on its last month of the previous quarter).
    lag_decay : float, default 2.0
        Exponent :math:`\kappa` of the lag decay :math:`1/l^\kappa` of the prior
        variances (GLP use 2).
    intercept_variance : float, default 1e6
        Prior variance factor :math:`\Omega_{cc}` of the constant (diffuse).
    dof : float, optional
        Prior degrees of freedom :math:`d` of :math:`\Sigma`; default :math:`n + 2`
        (the smallest integer with a finite prior mean of :math:`\Sigma`).
    unit_root_groups : sequence of hashable, optional
        One label per variable; variables with the same label share **one** unit root
        in the sum-of-coefficients and dummy-initial-observation priors (the three
        monthly blocks of a series in the blocked VAR). ``None``: every variable is its
        own group (the usual priors). See :func:`sum_of_coefficients_dummies`.

    Raises
    ------
    ValueError
        If ``prior_mean`` is an unknown string or a non-square/non-finite matrix, or
        ``lag_decay``/``intercept_variance`` are not positive.

    Examples
    --------
    >>> PriorSettings().own_lag_mean(2).tolist()
    [1.0, 1.0]
    >>> PriorSettings(prior_mean=[1, 0]).own_lag_mean(2).tolist()
    [1.0, 0.0]
    >>> PriorSettings().degrees_of_freedom(3)
    5.0
    >>> blocked = PriorSettings(prior_mean=[[0, 1], [0, 1]], unit_root_groups=["x", "x"])
    >>> blocked.first_lag_mean(2).tolist(), blocked.group_index(2).tolist()
    ([[0.0, 1.0], [0.0, 1.0]], [0, 0])
    """

    prior_mean: str | float | Sequence[float] | Sequence[Sequence[float]] | np.ndarray = (
        "random_walk"
    )
    lag_decay: float = 2.0
    intercept_variance: float = 1e6
    dof: float | None = None
    unit_root_groups: Sequence[object] | None = None

    def __post_init__(self) -> None:
        """Validate the settings (matrices and groups are stored as tuples)."""
        if isinstance(self.prior_mean, str):
            if self.prior_mean not in ("random_walk", "white_noise"):
                raise ValueError(
                    f"prior_mean must be 'random_walk', 'white_noise' or numeric, "
                    f"got {self.prior_mean!r}"
                )
        elif np.ndim(self.prior_mean) == 2:
            mat = np.asarray(self.prior_mean, dtype=float)
            if mat.shape[0] != mat.shape[1] or not np.all(np.isfinite(mat)):
                raise ValueError(f"a prior_mean matrix must be square and finite, got {mat.shape}")
            object.__setattr__(self, "prior_mean", tuple(map(tuple, mat.tolist())))
        if self.unit_root_groups is not None:
            object.__setattr__(self, "unit_root_groups", tuple(self.unit_root_groups))
        if self.lag_decay <= 0 or self.intercept_variance <= 0:
            raise ValueError("lag_decay and intercept_variance must be positive")

    def own_lag_mean(self, n: int) -> np.ndarray:
        """Prior means of the own first-lag coefficients (diagonal of E[A_1]).

        Parameters
        ----------
        n : int
            Number of variables.

        Returns
        -------
        numpy.ndarray
            Shape ``(n,)``.

        Raises
        ------
        ValueError
            If a sequence or matrix of the wrong size was given.

        Examples
        --------
        >>> PriorSettings("white_noise").own_lag_mean(3).tolist()
        [0.0, 0.0, 0.0]
        """
        if isinstance(self.prior_mean, str):
            value = 1.0 if self.prior_mean == "random_walk" else 0.0
            return np.full(n, value)
        arr = np.asarray(self.prior_mean, dtype=float)
        if arr.ndim == 0:
            return np.full(n, float(arr))
        if arr.ndim == 2:
            return np.diag(self.first_lag_mean(n)).copy()
        if arr.shape != (n,):
            raise ValueError(f"prior_mean has {arr.size} values for {n} variables")
        return arr.copy()

    def first_lag_mean(self, n: int) -> np.ndarray:
        r"""Prior mean :math:`\operatorname{E}[A_1]` of the first-lag coefficients.

        Parameters
        ----------
        n : int
            Number of variables.

        Returns
        -------
        numpy.ndarray
            Shape ``(n, n)``; entry ``(i, j)`` is the mean of the coefficient of
            :math:`y_{j,t-1}` in equation :math:`i` (diagonal unless a matrix was given).

        Raises
        ------
        ValueError
            If ``prior_mean`` does not match ``n`` variables.

        Examples
        --------
        >>> PriorSettings(prior_mean=[1, 0]).first_lag_mean(2).tolist()
        [[1.0, 0.0], [0.0, 0.0]]
        """
        if isinstance(self.prior_mean, str) or np.ndim(self.prior_mean) < 2:
            return np.diag(self.own_lag_mean(n))
        mat = np.asarray(self.prior_mean, dtype=float)
        if mat.shape != (n, n):
            raise ValueError(f"prior_mean matrix has shape {mat.shape} for {n} variables")
        return mat

    def group_index(self, n: int) -> np.ndarray:
        """Unit-root group (``0..G-1``, in order of appearance) of every variable.

        Parameters
        ----------
        n : int
            Number of variables.

        Returns
        -------
        numpy.ndarray
            Integer array of shape ``(n,)``; ``arange(n)`` without ``unit_root_groups``.

        Raises
        ------
        ValueError
            If ``unit_root_groups`` does not have ``n`` labels.

        Examples
        --------
        >>> PriorSettings(unit_root_groups=["b", "b", "a"]).group_index(3).tolist()
        [0, 0, 1]
        >>> PriorSettings().group_index(2).tolist()
        [0, 1]
        """
        labels = self.unit_root_groups
        if labels is None:
            return np.arange(n)
        if len(labels) != n:
            raise ValueError(f"unit_root_groups has {len(labels)} labels for {n} variables")
        order: dict[object, int] = {}
        return np.array([order.setdefault(label, len(order)) for label in labels], dtype=int)

    def degrees_of_freedom(self, n: int) -> float:
        r"""Prior degrees of freedom :math:`d`.

        Parameters
        ----------
        n : int
            Number of variables.

        Returns
        -------
        float
            ``dof`` or :math:`n + 2`.

        Raises
        ------
        ValueError
            If ``dof <= n + 1`` (the Minnesota scaling needs a finite
            :math:`\operatorname{E}[\Sigma]`).

        Examples
        --------
        >>> PriorSettings(dof=10).degrees_of_freedom(3)
        10.0
        """
        d = float(n + 2) if self.dof is None else float(self.dof)
        if d <= n + 1:
            raise ValueError(f"dof must exceed n + 1 = {n + 1}, got {d}")
        return d


@dataclass(frozen=True)
class BVARHyperparameters:
    r"""Hyperparameters of the GLP prior.

    Parameters
    ----------
    lambda_ : float
        Overall Minnesota tightness :math:`\lambda > 0`.
    psi : array-like
        Diagonal :math:`\psi` of the prior scale of :math:`\Sigma` (shape ``(n,)``,
        positive).
    mu : float, optional
        Sum-of-coefficients tightness :math:`\mu > 0`; ``None`` = prior not used.
    delta : float, optional
        Dummy-initial-observation tightness :math:`\delta > 0`; ``None`` = not used.

    Raises
    ------
    ValueError
        If a value is not positive and finite.

    Examples
    --------
    >>> h = BVARHyperparameters(0.2, [1.0, 2.0], mu=1.0)
    >>> h.psi.tolist(), h.delta is None
    ([1.0, 2.0], True)
    """

    lambda_: float
    psi: np.ndarray
    mu: float | None = None
    delta: float | None = None

    def __post_init__(self) -> None:
        """Validate and coerce ``psi`` to a float array."""
        psi = np.atleast_1d(np.asarray(self.psi, dtype=float)).copy()
        object.__setattr__(self, "psi", psi)
        values = [self.lambda_, *psi.tolist()]
        values += [v for v in (self.mu, self.delta) if v is not None]
        if not all(np.isfinite(v) and v > 0 for v in values):
            raise ValueError("BVAR hyperparameters must be positive and finite")


@dataclass(frozen=True)
class NIWPrior:
    r"""Normal-Inverse-Wishart prior with diagonal :math:`\Omega`.

    Parameters
    ----------
    mean : numpy.ndarray
        Prior mean :math:`b` of :math:`B`, shape ``(k, n)``.
    omega : numpy.ndarray
        Diagonal of :math:`\Omega`, shape ``(k,)``, positive.
    scale : numpy.ndarray
        Scale :math:`\Psi` of the inverse-Wishart, shape ``(n, n)``, positive definite.
    dof : float
        Degrees of freedom :math:`d > n - 1`.

    Examples
    --------
    >>> import numpy as np
    >>> p = NIWPrior(np.zeros((2, 1)), np.ones(2), np.eye(1) * 2.0, 4.0)
    >>> p.sigma_mean.tolist()
    [[1.0]]
    """

    mean: np.ndarray
    omega: np.ndarray
    scale: np.ndarray
    dof: float

    @property
    def sigma_mean(self) -> np.ndarray:
        r"""Prior mean :math:`\Psi / (d - n - 1)` of :math:`\Sigma`.

        Returns
        -------
        numpy.ndarray
            Shape ``(n, n)`` (``inf`` when :math:`d \le n + 1`).

        Examples
        --------
        >>> import numpy as np
        >>> NIWPrior(np.zeros((1, 2)), np.ones(1), np.eye(2), 6.0).sigma_mean[0, 0]
        np.float64(0.3333333333333333)
        """
        n = self.scale.shape[0]
        denom = self.dof - n - 1
        return self.scale / denom if denom > 0 else np.full_like(self.scale, np.inf)


def minnesota_prior(
    system: VARSystem, hyper: BVARHyperparameters, settings: PriorSettings | None = None
) -> NIWPrior:
    r"""Minnesota prior in NIW form (GLP 2015 parameterisation).

    Parameters
    ----------
    system : VARSystem
        VAR data (dimensions and lag layout).
    hyper : BVARHyperparameters
        Uses ``lambda_`` and ``psi``.
    settings : PriorSettings, optional
        Prior means, lag decay, constant variance and degrees of freedom.

    Returns
    -------
    NIWPrior
        :math:`b` (first lag = :math:`\operatorname{E}[A_1]'`, by default
        :math:`\delta_i` on the own lag), diagonal :math:`\Omega`
        (:math:`\lambda^2 (d-n-1)/(l^\kappa \psi_j)`; constant:
        ``intercept_variance``), :math:`\Psi = \operatorname{diag}(\psi)`, :math:`d`.

    Raises
    ------
    ValueError
        If ``psi`` does not have ``n`` entries.

    Examples
    --------
    >>> import numpy as np
    >>> s = VARSystem.from_array(np.ones((5, 2)), lags=2)
    >>> p = minnesota_prior(s, BVARHyperparameters(0.5, [1.0, 4.0]))
    >>> p.omega.tolist()
    [1000000.0, 0.25, 0.0625, 0.0625, 0.015625]
    >>> p.mean[:, 0].tolist()
    [0.0, 1.0, 0.0, 0.0, 0.0]
    """
    settings = settings or PriorSettings()
    n = system.n
    if hyper.psi.shape != (n,):
        raise ValueError(f"psi has {hyper.psi.size} entries for {n} variables")
    d = settings.degrees_of_freedom(n)
    lag = np.repeat(np.arange(1, system.lags + 1, dtype=float), n)
    psi_rep = np.tile(hyper.psi, system.lags)
    omega_lags = hyper.lambda_**2 * (d - n - 1) / (lag**settings.lag_decay * psi_rep)
    omega = np.concatenate([[settings.intercept_variance], omega_lags])
    omega = omega if system.constant else omega[1:]
    mean = np.zeros((system.k, n))
    first = [system.lag_column(1, j) for j in range(n)]
    mean[first, :] = settings.first_lag_mean(n).T  # row j of B = regressor y_{j,t-1}
    return NIWPrior(mean=mean, omega=omega, scale=np.diag(hyper.psi), dof=d)


def minnesota_dummies(
    system: VARSystem, hyper: BVARHyperparameters, settings: PriorSettings | None = None
) -> tuple[np.ndarray, np.ndarray]:
    r"""Minnesota prior as dummy observations (Bańbura, Giannone & Reichlin, 2010).

    With :math:`\sigma_j = \sqrt{\psi_j}` and :math:`\tilde\lambda = \lambda
    \sqrt{d - n - 1}`: for each lag :math:`l` and variable :math:`j` one row with
    :math:`X = l^{\kappa/2}\sigma_j/\tilde\lambda` in the column of :math:`y_{j,t-l}`
    and :math:`Y = \delta_j \sigma_j/\tilde\lambda` in column :math:`j` when
    :math:`l = 1`; :math:`n` rows :math:`Y = \operatorname{diag}(\sigma)`,
    :math:`X = 0` for :math:`\Sigma`; one row for the constant with
    :math:`X = \Omega_{cc}^{-1/2}`. Combined with a flat prior, the implied prior
    (:func:`implied_prior_from_dummies`) has the mean, :math:`\Omega` and
    :math:`\Psi` of :func:`minnesota_prior`; the degrees of freedom are not encoded.

    Parameters
    ----------
    system : VARSystem
        VAR data.
    hyper : BVARHyperparameters
        Uses ``lambda_`` and ``psi``.
    settings : PriorSettings, optional
        Prior settings.

    Returns
    -------
    Yd : numpy.ndarray
        Shape ``(n p + n + constant, n)``.
    Xd : numpy.ndarray
        Shape ``(n p + n + constant, k)``.

    Examples
    --------
    >>> import numpy as np
    >>> s = VARSystem.from_array(np.ones((5, 2)), lags=1)
    >>> Yd, Xd = minnesota_dummies(s, BVARHyperparameters(0.5, [1.0, 4.0]))
    >>> Yd.shape, Xd.shape
    ((5, 2), (5, 3))
    """
    prior = minnesota_prior(system, hyper, settings)
    n, k = system.n, system.k
    sd_coef = 1.0 / np.sqrt(prior.omega)  # 1/sqrt(Omega_jj) on the diagonal
    Xd = np.vstack([np.diag(sd_coef), np.zeros((n, k))])
    Yd = np.vstack([prior.mean * sd_coef[:, None], np.diag(np.sqrt(hyper.psi))])
    return Yd, Xd


def _group_levels(
    system: VARSystem, groups: Sequence[int] | np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    r"""Group indicator (``(G, n)``) and per-variable common level :math:`\bar y_{0,g(i)}`."""
    n = system.n
    index = np.arange(n) if groups is None else np.asarray(groups, dtype=int)
    if index.shape != (n,):
        raise ValueError(f"groups has {index.size} labels for {n} variables")
    _, index = np.unique(index, return_inverse=True)
    indicator = (index[None, :] == np.arange(int(index.max()) + 1)[:, None]).astype(float)
    level = (indicator @ system.y0_mean) / indicator.sum(axis=1)
    return indicator, level[index]


def sum_of_coefficients_dummies(
    system: VARSystem, mu: float, groups: Sequence[int] | np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    r"""Sum-of-coefficients dummy observations (Doan, Litterman & Sims, 1984).

    One row per unit-root group :math:`g` (by default per variable): with
    :math:`\bar y_{0,g}` the mean of :math:`\bar y_0` over the members of :math:`g`,
    :math:`Y^+_{g,i} = \bar y_{0,g}/\mu` for :math:`i \in g` (zero otherwise) and
    :math:`X^+_g = (0, Y^+_g, \dots, Y^+_g)`. In equation :math:`i` the row says
    :math:`\bar y_{0,g}\,1\{i \in g\} = \bar y_{0,g} \sum_l \sum_{j \in g}
    (A_l)_{ij} + \text{noise}`: if every member of the group sits at a common level in
    all lags, the members stay there and the other variables do not react. With
    singleton groups this is the usual prior :math:`\sum_l A_l = I`; with the three
    monthly blocks of a series as one group it is the blocked unit root
    :math:`\sum_l A_l \iota_g = \iota_g` (one unit root per series), which the blocked
    random walk satisfies exactly.

    Parameters
    ----------
    system : VARSystem
        VAR data (uses ``y0_mean``).
    mu : float
        Tightness :math:`\mu > 0` (smaller = tighter).
    groups : array-like of int, optional
        Unit-root group of every variable (see :meth:`PriorSettings.group_index`);
        default: one group per variable.

    Returns
    -------
    Yd : numpy.ndarray
        Shape ``(G, n)`` (:math:`\operatorname{diag}(\bar y_0)/\mu` by default).
    Xd : numpy.ndarray
        :math:`(0, Y_d, \dots, Y_d)`, shape ``(G, k)``.

    Raises
    ------
    ValueError
        If ``groups`` does not have one label per variable.

    Examples
    --------
    >>> import numpy as np
    >>> s = VARSystem.from_array(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), lags=1)
    >>> Yd, Xd = sum_of_coefficients_dummies(s, 0.5)
    >>> Yd.tolist(), Xd.tolist()
    ([[2.0, 0.0], [0.0, 4.0]], [[0.0, 2.0, 0.0], [0.0, 0.0, 4.0]])
    >>> sum_of_coefficients_dummies(s, 0.5, groups=[0, 0])[0].tolist()
    [[3.0, 3.0]]
    """
    indicator, level = _group_levels(system, groups)
    Yd = indicator * level[None, :] / mu
    blocks = [Yd] * system.lags
    if system.constant:
        blocks.insert(0, np.zeros((Yd.shape[0], 1)))
    return Yd, np.hstack(blocks)


def dummy_initial_observation_dummies(
    system: VARSystem, delta: float, groups: Sequence[int] | np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    r"""Dummy-initial-observation row (Sims, 1993; Sims & Zha, 1998).

    The row places every variable at its starting level :math:`\bar y_0` (with
    ``groups``: the common level of its group, so that a group sharing one unit root
    starts from one level and the row is satisfied exactly by the blocked random
    walk).

    Parameters
    ----------
    system : VARSystem
        VAR data (uses ``y0_mean``).
    delta : float
        Tightness :math:`\delta > 0` (smaller = tighter).
    groups : array-like of int, optional
        Unit-root group of every variable; default: one group per variable.

    Returns
    -------
    Yd : numpy.ndarray
        :math:`\bar y_0'/\delta`, shape ``(1, n)``.
    Xd : numpy.ndarray
        :math:`(1/\delta, Y_d, \dots, Y_d)`, shape ``(1, k)``.

    Raises
    ------
    ValueError
        If ``groups`` does not have one label per variable.

    Examples
    --------
    >>> import numpy as np
    >>> s = VARSystem.from_array(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), lags=1)
    >>> Yd, Xd = dummy_initial_observation_dummies(s, 0.5)
    >>> Yd.tolist(), Xd.tolist()
    ([[2.0, 4.0]], [[2.0, 2.0, 4.0]])
    >>> dummy_initial_observation_dummies(s, 0.5, groups=[1, 1])[0].tolist()
    [[3.0, 3.0]]
    """
    _, level = _group_levels(system, groups)
    Yd = level[None, :] / delta
    blocks = [Yd] * system.lags
    if system.constant:
        blocks.insert(0, np.full((1, 1), 1.0 / delta))
    return Yd, np.hstack(blocks)


def prior_dummies(
    system: VARSystem, hyper: BVARHyperparameters, settings: PriorSettings | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Stacked sum-of-coefficients and dummy-initial-observation rows in use.

    Parameters
    ----------
    system : VARSystem
        VAR data.
    hyper : BVARHyperparameters
        ``mu``/``delta`` (``None`` = that prior is off).
    settings : PriorSettings, optional
        Its ``unit_root_groups`` set the groups of the dummies (default: one per
        variable).

    Returns
    -------
    Yd : numpy.ndarray
        Shape ``(m, n)`` with ``m`` in ``{0, 1, G, G + 1}`` (``G`` groups).
    Xd : numpy.ndarray
        Shape ``(m, k)``.

    Examples
    --------
    >>> import numpy as np
    >>> s = VARSystem.from_array(np.ones((4, 3)), lags=1)
    >>> prior_dummies(s, BVARHyperparameters(0.2, np.ones(3), mu=1.0, delta=1.0))[0].shape
    (4, 3)
    >>> prior_dummies(s, BVARHyperparameters(0.2, np.ones(3)))[1].shape
    (0, 4)
    >>> grouped = PriorSettings(unit_root_groups=["a", "a", "b"])
    >>> prior_dummies(s, BVARHyperparameters(0.2, np.ones(3), mu=1.0), grouped)[0].shape
    (2, 3)
    """
    groups = None if settings is None else settings.group_index(system.n)
    ys, xs = [np.empty((0, system.n))], [np.empty((0, system.k))]
    if hyper.mu is not None:
        Yd, Xd = sum_of_coefficients_dummies(system, hyper.mu, groups)
        ys.append(Yd)
        xs.append(Xd)
    if hyper.delta is not None:
        Yd, Xd = dummy_initial_observation_dummies(system, hyper.delta, groups)
        ys.append(Yd)
        xs.append(Xd)
    return np.vstack(ys), np.vstack(xs)


def implied_prior_from_dummies(
    Yd: np.ndarray, Xd: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Prior moments encoded by dummy observations (combined with a flat prior).

    Parameters
    ----------
    Yd : numpy.ndarray
        Dummy left-hand side, shape ``(m, n)``.
    Xd : numpy.ndarray
        Dummy regressors, shape ``(m, k)``; :math:`X_d'X_d` must be non-singular.

    Returns
    -------
    mean : numpy.ndarray
        :math:`b = (X_d'X_d)^{-1} X_d'Y_d`, shape ``(k, n)``.
    omega : numpy.ndarray
        :math:`\Omega = (X_d'X_d)^{-1}`, shape ``(k, k)``.
    scale : numpy.ndarray
        :math:`\Psi = (Y_d - X_d b)'(Y_d - X_d b)`, shape ``(n, n)``.

    Raises
    ------
    numpy.linalg.LinAlgError
        If :math:`X_d'X_d` is singular.

    Examples
    --------
    >>> import numpy as np
    >>> s = VARSystem.from_array(np.ones((5, 2)), lags=1)
    >>> b, om, ps = implied_prior_from_dummies(
    ...     *minnesota_dummies(s, BVARHyperparameters(0.5, [1.0, 4.0]))
    ... )
    >>> np.round(np.diag(om), 6).tolist(), np.diag(ps).tolist()
    ([1000000.0, 0.25, 0.0625], [1.0, 4.0])
    """
    G = Xd.T @ Xd
    chol = scipy.linalg.cho_factor(G, lower=True)
    mean = scipy.linalg.cho_solve(chol, Xd.T @ Yd)
    omega = scipy.linalg.cho_solve(chol, np.eye(G.shape[0]))
    resid = Yd - Xd @ mean
    return mean, omega, resid.T @ resid


# ---------------------------------------------------------------------------
# posterior and marginal likelihood
# ---------------------------------------------------------------------------
def _multigammaln(a: float, n: int) -> float:
    r"""Log of the multivariate Gamma function :math:`\Gamma_n(a)`."""
    j = np.arange(n)
    return float(n * (n - 1) / 4 * np.log(np.pi) + np.sum(gammaln(a - j / 2)))


def _chol_logdet(A: np.ndarray) -> float:
    """Log-determinant of a symmetric positive definite matrix via Cholesky."""
    try:
        L = np.linalg.cholesky(A)
    except np.linalg.LinAlgError as exc:
        raise NowcastDataError("matrix is not positive definite") from exc
    return float(2.0 * np.sum(np.log(np.diag(L))))


def _precision_root(xs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r"""Thin QR factorisation of :math:`A = (X_s', I_k)'` (shape ``(T + k, k)``).

    :math:`A'A = I_k + X_s'X_s = R'R`, so :math:`R` is a square root of the posterior
    precision factor without forming :math:`X_s'X_s`. Forming the cross-product squares
    the condition number, which destroys the identity part when the dummy rows are large
    (data in levels with a tight sum-of-coefficients prior); the QR factor keeps full
    accuracy. The top block of :math:`Q` holds the leverages :math:`x_t'(I + X_s'X_s)^{-1}x_t`
    as squared row norms.
    """
    k = xs.shape[1]
    Q, R = np.linalg.qr(np.vstack([xs, np.eye(k)]))
    return Q, R


@dataclass(frozen=True)
class NIWPosterior:
    r"""Normal-Inverse-Wishart posterior of a VAR.

    Parameters
    ----------
    mean : numpy.ndarray
        Posterior mean :math:`\bar B`, shape ``(k, n)``.
    scale : numpy.ndarray
        :math:`\bar\Psi`, shape ``(n, n)``.
    dof : float
        :math:`\bar d = d + T` (dummy rows count as observations).
    log_ml : float
        Log marginal likelihood of the data (conditional on the dummy observations, if
        any; see :func:`posterior`).
    omega_sqrt : numpy.ndarray
        :math:`\Omega^{1/2}` (prior, diagonal), shape ``(k,)``.
    root : numpy.ndarray
        Upper-triangular :math:`R` with :math:`R'R = I_k + X_s'X_s`, :math:`X_s =
        X\Omega^{1/2}` (all rows used, dummies included), shape ``(k, k)``;
        :math:`\bar\Omega = \Omega^{1/2}R^{-1}R^{-\top}\Omega^{1/2}`.
    leverage : numpy.ndarray
        :math:`x_{s,t}'(I_k + X_s'X_s)^{-1}x_{s,t}` for every row used, shape ``(T,)``.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> s = VARSystem.from_array(rng.standard_normal((50, 2)), lags=1)
    >>> post = posterior(s, BVARHyperparameters(0.2, [1.0, 1.0]))
    >>> B, S = post.draw(3, rng=1)
    >>> B.shape, S.shape
    ((3, 3, 2), (3, 2, 2))
    """

    mean: np.ndarray
    scale: np.ndarray
    dof: float
    log_ml: float
    omega_sqrt: np.ndarray = field(repr=False)
    root: np.ndarray = field(repr=False)
    leverage: np.ndarray = field(repr=False)

    @property
    def n(self) -> int:
        """Number of variables.

        Returns
        -------
        int
            :math:`n`.

        Examples
        --------
        >>> import numpy as np
        >>> s = VARSystem.from_array(np.random.default_rng(0).standard_normal((9, 2)), 1)
        >>> posterior(s, BVARHyperparameters(0.2, [1.0, 1.0])).n
        2
        """
        return int(self.mean.shape[1])

    def _root_inverse(self) -> np.ndarray:
        r""":math:`R^{-1}` (upper triangular); :math:`(I + X_s'X_s)^{-1} = R^{-1}R^{-\top}`."""
        k = self.root.shape[0]
        return scipy.linalg.solve_triangular(self.root, np.eye(k), lower=False)

    @property
    def omega(self) -> np.ndarray:
        r"""Posterior :math:`\bar\Omega = (X'X + \Omega^{-1})^{-1}`.

        Returns
        -------
        numpy.ndarray
            Shape ``(k, k)``.

        Examples
        --------
        >>> import numpy as np
        >>> s = VARSystem.from_array(np.random.default_rng(0).standard_normal((9, 1)), 1)
        >>> posterior(s, BVARHyperparameters(0.2, [1.0])).omega.shape
        (2, 2)
        """
        W = self.omega_sqrt[:, None] * self._root_inverse()
        return W @ W.T

    @property
    def sigma_mean(self) -> np.ndarray:
        r"""Posterior mean :math:`\bar\Psi/(\bar d - n - 1)` of :math:`\Sigma`.

        Returns
        -------
        numpy.ndarray
            Shape ``(n, n)``.

        Examples
        --------
        >>> import numpy as np
        >>> s = VARSystem.from_array(np.random.default_rng(0).standard_normal((99, 1)), 1)
        >>> float(posterior(s, BVARHyperparameters(0.2, [1.0])).sigma_mean[0, 0]) > 0
        True
        """
        return self.scale / (self.dof - self.n - 1)

    def draw(
        self, n_draws: int, rng: np.random.Generator | int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""Draw :math:`(B, \Sigma)` from the posterior.

        :math:`\Sigma = G G'` with :math:`G = C A^{-\top}`, :math:`\bar\Psi = CC'` and
        :math:`A` the Bartlett factor of a Wishart(:math:`I`, :math:`\bar d`) draw, so
        :math:`\Sigma^{-1} \sim W(\bar\Psi^{-1}, \bar d)`; then
        :math:`B = \bar B + \Omega^{1/2} R^{-1} Z G'` with :math:`R'R = I + X_s'X_s`
        and :math:`Z` standard normal, so that :math:`\operatorname{vec}(B) \mid \Sigma
        \sim N(\operatorname{vec}(\bar B), \Sigma \otimes \bar\Omega)`.

        Parameters
        ----------
        n_draws : int
            Number of draws.
        rng : numpy.random.Generator or int, optional
            Random generator or seed.

        Returns
        -------
        B : numpy.ndarray
            Shape ``(n_draws, k, n)``.
        Sigma : numpy.ndarray
            Shape ``(n_draws, n, n)``.

        Raises
        ------
        ValueError
            If ``n_draws < 1``.

        Examples
        --------
        >>> import numpy as np
        >>> s = VARSystem.from_array(np.random.default_rng(0).standard_normal((30, 2)), 1)
        >>> B, S = posterior(s, BVARHyperparameters(0.2, [1.0, 1.0])).draw(2, rng=0)
        >>> bool(np.all(np.linalg.eigvalsh(S) > 0))
        True
        """
        if n_draws < 1:
            raise ValueError(f"n_draws must be >= 1, got {n_draws}")
        gen = np.random.default_rng(rng)
        G = self._sigma_roots(n_draws, gen)
        k, n = self.mean.shape
        Z = gen.standard_normal((k, n_draws * n))
        W = scipy.linalg.solve_triangular(self.root, Z, lower=False)
        W = self.omega_sqrt[:, None] * W
        W = W.reshape(k, n_draws, n).transpose(1, 0, 2)
        B = self.mean[None] + W @ G.transpose(0, 2, 1)
        return B, G @ G.transpose(0, 2, 1)

    def _sigma_roots(self, n_draws: int, gen: np.random.Generator) -> np.ndarray:
        """Square roots :math:`G` (``(n_draws, n, n)``) of inverse-Wishart draws."""
        n = self.n
        C = np.linalg.cholesky(self.scale)
        A = np.zeros((n_draws, n, n))
        rows, cols = np.tril_indices(n, -1)
        A[:, rows, cols] = gen.standard_normal((n_draws, rows.size))
        diag = np.sqrt(gen.chisquare(self.dof - np.arange(n), size=(n_draws, n)))
        A[:, np.arange(n), np.arange(n)] = diag
        # G' = A^{-1} C'  =>  G G' = C A^{-T} A^{-1} C' = (C^{-T} A A' C^{-1})^{-1}
        Ct = np.ascontiguousarray(C.T)
        G = np.empty_like(A)
        for i in range(n_draws):  # triangular solves: cheaper than a batched dense solve
            G[i] = scipy.linalg.solve_triangular(A[i], Ct, lower=True, check_finite=False).T
        return G


def niw_posterior(Y: np.ndarray, X: np.ndarray, prior: NIWPrior) -> NIWPosterior:
    r"""Conjugate NIW posterior and log marginal likelihood of :math:`Y = XB + E`.

    Parameters
    ----------
    Y : numpy.ndarray
        Shape ``(T, n)``.
    X : numpy.ndarray
        Shape ``(T, k)``.
    prior : NIWPrior
        Prior with diagonal :math:`\Omega`.

    Returns
    -------
    NIWPosterior
        :math:`\bar B`, :math:`\bar\Psi`, :math:`d + T` and :math:`\log p(Y)` (GLP
        2015, appendix A).

    Raises
    ------
    NowcastDataError
        If the prior or posterior scale is not positive definite.

    Examples
    --------
    >>> import numpy as np
    >>> prior = NIWPrior(np.zeros((1, 1)), np.ones(1), np.eye(1), 3.0)
    >>> post = niw_posterior(np.array([[1.0], [1.0]]), np.ones((2, 1)), prior)
    >>> round(float(post.mean[0, 0]), 4), round(float(post.scale[0, 0]), 4)
    (0.6667, 1.6667)
    """
    T, n = Y.shape
    d_sqrt = np.sqrt(prior.omega)
    xs = X * d_sqrt
    dev0 = Y - X @ prior.mean
    Q, R = _precision_root(xs)
    # z = (B_bar - b) / sqrt(Omega): least-squares solution of [X_s; I] z ~ [Y - Xb; 0]
    z = scipy.linalg.solve_triangular(R, Q[:T].T @ dev0, lower=False)
    mean = prior.mean + d_sqrt[:, None] * z
    resid = dev0 - xs @ z
    scale = prior.scale + resid.T @ resid + z.T @ z
    scale = (scale + scale.T) / 2
    logdet_m = 2.0 * float(np.sum(np.log(np.abs(np.diag(R)))))
    d0 = float(prior.dof)
    log_ml = (
        -0.5 * n * T * np.log(np.pi)
        + _multigammaln((d0 + T) / 2, n)
        - _multigammaln(d0 / 2, n)
        + 0.5 * d0 * _chol_logdet(prior.scale)
        - 0.5 * n * logdet_m
        - 0.5 * (d0 + T) * _chol_logdet(scale)
    )
    return NIWPosterior(
        mean=mean,
        scale=scale,
        dof=d0 + T,
        log_ml=float(log_ml),
        omega_sqrt=d_sqrt,
        root=R,
        leverage=np.sum(Q[:T] ** 2, axis=1),
    )


def posterior(
    system: VARSystem, hyper: BVARHyperparameters, settings: PriorSettings | None = None
) -> NIWPosterior:
    r"""Posterior of the VAR under the GLP prior (Minnesota + optional dummies).

    Parameters
    ----------
    system : VARSystem
        VAR data.
    hyper : BVARHyperparameters
        Hyperparameters (``mu``/``delta`` ``None`` = no dummy rows).
    settings : PriorSettings, optional
        Prior settings.

    Returns
    -------
    NIWPosterior
        Posterior given the data and the dummy rows; ``log_ml`` is
        :math:`\log p(Y \mid Y^+) = \log p(Y, Y^+) - \log p(Y^+)`.

    Raises
    ------
    NowcastDataError
        If a scale matrix is not positive definite.

    Examples
    --------
    >>> import numpy as np
    >>> s = VARSystem.from_array(np.random.default_rng(0).standard_normal((40, 2)), 1)
    >>> posterior(s, BVARHyperparameters(0.2, [1.0, 1.0], mu=1.0)).dof
    45.0
    """
    prior = minnesota_prior(system, hyper, settings)
    Yd, Xd = prior_dummies(system, hyper, settings)
    if Yd.shape[0] == 0:
        return niw_posterior(system.Y, system.X, prior)
    full = niw_posterior(np.vstack([Yd, system.Y]), np.vstack([Xd, system.X]), prior)
    dummies_only = niw_posterior(Yd, Xd, prior)
    return replace(full, log_ml=full.log_ml - dummies_only.log_ml)


def log_marginal_likelihood(
    system: VARSystem, hyper: BVARHyperparameters, settings: PriorSettings | None = None
) -> float:
    r"""Log marginal likelihood :math:`\log p(Y \mid \lambda, \mu, \delta, \psi)`.

    Parameters
    ----------
    system : VARSystem
        VAR data.
    hyper : BVARHyperparameters
        Hyperparameters.
    settings : PriorSettings, optional
        Prior settings.

    Returns
    -------
    float
        Log density of :math:`Y` (conditional on the pre-sample and the dummy rows).

    Raises
    ------
    NowcastDataError
        If a scale matrix is not positive definite.

    Examples
    --------
    >>> import numpy as np
    >>> s = VARSystem.from_array(np.random.default_rng(0).standard_normal((40, 2)), 1)
    >>> tight = log_marginal_likelihood(
    ...     s, BVARHyperparameters(0.01, [1.0, 1.0]), PriorSettings("white_noise")
    ... )
    >>> loose = log_marginal_likelihood(
    ...     s, BVARHyperparameters(5.0, [1.0, 1.0]), PriorSettings("white_noise")
    ... )
    >>> bool(tight > loose)  # white-noise data favour a tight white-noise prior
    True
    """
    return posterior(system, hyper, settings).log_ml


@dataclass(frozen=True)
class HyperGradient:
    r"""Derivatives with respect to the **logarithms** of the hyperparameters.

    Parameters
    ----------
    lambda_ : float
        :math:`\partial/\partial\log\lambda`.
    psi : numpy.ndarray
        :math:`\partial/\partial\log\psi_j`, shape ``(n,)``.
    mu : float, optional
        :math:`\partial/\partial\log\mu` (``None`` when the prior is off).
    delta : float, optional
        :math:`\partial/\partial\log\delta` (``None`` when the prior is off).

    Examples
    --------
    >>> import numpy as np
    >>> g = HyperGradient(1.0, np.array([0.5]), mu=2.0)
    >>> (g + g).lambda_, (g + g).mu, (g + g).delta
    (2.0, 4.0, None)
    """

    lambda_: float
    psi: np.ndarray
    mu: float | None = None
    delta: float | None = None

    def __add__(self, other: HyperGradient) -> HyperGradient:
        """Elementwise sum (``None`` + ``None`` = ``None``; ``None`` counts as 0)."""

        def plus(a: float | None, b: float | None) -> float | None:
            if a is None and b is None:
                return None
            return (a or 0.0) + (b or 0.0)

        return HyperGradient(
            lambda_=self.lambda_ + other.lambda_,
            psi=self.psi + other.psi,
            mu=plus(self.mu, other.mu),
            delta=plus(self.delta, other.delta),
        )


def _gradient_parts(
    Y: np.ndarray, X: np.ndarray, prior: NIWPrior, post: NIWPosterior, n_rows: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Derivatives of one NIW log marginal likelihood (envelope theorem).

    Returns :math:`\partial\ell/\partial\log\Omega_{ii}` (``(k,)``),
    :math:`\partial\ell/\partial\log\psi_j` through :math:`\Psi` only (``(n,)``) and
    :math:`\partial\ell/\partial\log s_r` for scaling each of the first ``n_rows`` rows
    of :math:`(Y, X)` by :math:`s_r` (``(n_rows,)``). With :math:`\bar\Psi = \Psi +
    \min_B S(B)` the derivative of :math:`\bar\Psi` is that of :math:`S` at fixed
    :math:`\bar B`.
    """
    n = Y.shape[1]
    # diag of (I + X_s'X_s)^{-1}
    diag_minv = np.sum(post._root_inverse() ** 2, axis=1)  # pyright: ignore[reportPrivateUsage]
    lever = post.leverage[:n_rows]
    psi_inv = scipy.linalg.cho_solve(scipy.linalg.cho_factor(post.scale, lower=True), np.eye(n))
    dev = (post.mean - prior.mean) / post.omega_sqrt[:, None]
    quad_dev = np.einsum("ij,jk,ik->i", dev, psi_inv, dev)
    g_omega = -0.5 * n * (1.0 - diag_minv) + 0.5 * post.dof * quad_dev
    g_psi = 0.5 * prior.dof - 0.5 * post.dof * np.diag(psi_inv) * np.diag(prior.scale)
    resid = Y[:n_rows] - X[:n_rows] @ post.mean
    g_rows = -n * lever - post.dof * np.einsum("ij,jk,ik->i", resid, psi_inv, resid)
    return g_omega, g_psi, g_rows


def log_marginal_likelihood_gradient(
    system: VARSystem, hyper: BVARHyperparameters, settings: PriorSettings | None = None
) -> tuple[float, HyperGradient]:
    r"""Log marginal likelihood and its gradient in the log hyperparameters.

    Analytic derivatives (envelope theorem on :math:`\bar\Psi`): for one NIW block with
    :math:`\bar d = d + T`,

    .. math::

        \frac{\partial \ell}{\partial \log\Omega_{ii}} =
        -\frac n2\Big(1 - \frac{\bar\Omega_{ii}}{\Omega_{ii}}\Big)
        + \frac{\bar d}{2}\,\frac{(\bar B - b)_{i\cdot}\bar\Psi^{-1}(\bar B - b)_{i\cdot}'}
        {\Omega_{ii}},
        \qquad
        \frac{\partial \ell}{\partial \log\psi_j}\Big|_{\Psi} =
        \frac d2 - \frac{\bar d}{2}\psi_j(\bar\Psi^{-1})_{jj},

    and scaling dummy row :math:`r` by :math:`s` gives
    :math:`\partial\ell/\partial\log s = -n\, x_r'\bar\Omega x_r - \bar d\, \hat e_r'
    \bar\Psi^{-1}\hat e_r`; then the chain rule through :math:`\Omega_{ii} \propto
    \lambda^2/\psi_j` and the dummy rows :math:`\propto 1/\mu, 1/\delta`, for both
    :math:`p(Y, Y^+)` and :math:`p(Y^+)`.

    Parameters
    ----------
    system : VARSystem
        VAR data.
    hyper : BVARHyperparameters
        Point.
    settings : PriorSettings, optional
        Prior settings.

    Returns
    -------
    value : float
        :func:`log_marginal_likelihood`.
    gradient : HyperGradient
        Derivatives with respect to :math:`\log\lambda, \log\psi, \log\mu, \log\delta`.

    Raises
    ------
    NowcastDataError
        If a scale matrix is not positive definite.

    Examples
    --------
    >>> import numpy as np
    >>> s = VARSystem.from_array(np.random.default_rng(0).standard_normal((40, 2)), 1)
    >>> h = BVARHyperparameters(0.2, [1.0, 1.0], mu=1.0, delta=1.0)
    >>> value, grad = log_marginal_likelihood_gradient(s, h)
    >>> bool(np.isclose(value, log_marginal_likelihood(s, h))), grad.psi.shape
    (True, (2,))
    """
    prior = minnesota_prior(system, hyper, settings)
    Yd, Xd = prior_dummies(system, hyper, settings)
    m = Yd.shape[0]
    Y_all, X_all = np.vstack([Yd, system.Y]), np.vstack([Xd, system.X])
    full = niw_posterior(Y_all, X_all, prior)
    g_omega, g_psi, g_rows = _gradient_parts(Y_all, X_all, prior, full, m)
    value = full.log_ml
    if m:
        dummy = niw_posterior(Yd, Xd, prior)
        d_omega, d_psi, d_rows = _gradient_parts(Yd, Xd, prior, dummy, m)
        value -= dummy.log_ml
        g_omega, g_psi, g_rows = g_omega - d_omega, g_psi - d_psi, g_rows - d_rows
    n_soc = m - int(hyper.delta is not None)  # sum-of-coefficients rows come first
    return value, _chain_rule(system, hyper, g_omega, g_psi, g_rows, n_soc)


def _chain_rule(
    system: VARSystem,
    hyper: BVARHyperparameters,
    g_omega: np.ndarray,
    g_psi: np.ndarray,
    g_rows: np.ndarray,
    n_soc: int,
) -> HyperGradient:
    """Map block derivatives to the log hyperparameters."""
    g_lags = g_omega[int(system.constant) :]
    return HyperGradient(
        lambda_=float(2.0 * np.sum(g_lags)),
        psi=g_psi - g_lags.reshape(system.lags, system.n).sum(axis=0),
        mu=None if hyper.mu is None else -float(np.sum(g_rows[:n_soc])),
        delta=None if hyper.delta is None else -float(g_rows[n_soc]),
    )


# ---------------------------------------------------------------------------
# hyperpriors
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GammaHyperprior:
    r"""Gamma hyperprior parameterised by its mode and standard deviation (GLP 2015).

    Parameters
    ----------
    mode : float
        Mode :math:`m \ge 0`.
    sd : float
        Standard deviation :math:`s > 0`.

    Examples
    --------
    >>> g = GammaHyperprior(0.2, 0.4)
    >>> round(g.shape, 4), round(g.scale, 4)
    (1.6404, 0.3123)
    """

    mode: float
    sd: float

    @property
    def scale(self) -> float:
        r"""Scale :math:`\theta = (\sqrt{m^2 + 4s^2} - m)/2`.

        Returns
        -------
        float
            :math:`\theta`, from mode :math:`(a-1)\theta = m` and variance
            :math:`a\theta^2 = s^2`.

        Examples
        --------
        >>> GammaHyperprior(1.0, 1.0).scale > 0.0
        True
        """
        return float((np.sqrt(self.mode**2 + 4 * self.sd**2) - self.mode) / 2)

    @property
    def shape(self) -> float:
        r"""Shape :math:`a = 1 + m/\theta`.

        Returns
        -------
        float
            :math:`a`.

        Examples
        --------
        >>> round(GammaHyperprior(1.0, 1.0).shape, 4)
        2.618
        """
        return 1.0 + self.mode / self.scale

    def logpdf(self, x: float) -> float:
        r"""Log density at ``x``.

        Parameters
        ----------
        x : float
            Point (positive).

        Returns
        -------
        float
            :math:`\log p(x)`.

        Examples
        --------
        >>> import scipy.stats
        >>> g = GammaHyperprior(0.2, 0.4)
        >>> bool(abs(g.logpdf(0.3) - scipy.stats.gamma.logpdf(0.3, g.shape, scale=g.scale)) < 1e-12)
        True
        """
        a, th = self.shape, self.scale
        return float((a - 1) * np.log(x) - x / th - gammaln(a) - a * np.log(th))

    def dlogpdf_dlog(self, x: float) -> float:
        r"""Derivative of the log density with respect to :math:`\log x`.

        Parameters
        ----------
        x : float
            Point (positive).

        Returns
        -------
        float
            :math:`(a - 1) - x/\theta`.

        Examples
        --------
        >>> abs(GammaHyperprior(0.2, 0.4).dlogpdf_dlog(0.2)) < 1e-12  # zero at the mode
        True
        """
        return float(self.shape - 1.0 - x / self.scale)


@dataclass(frozen=True)
class InverseGammaHyperprior:
    r"""Inverse-Gamma hyperprior :math:`p(x) \propto x^{-a-1} e^{-\beta/x}`.

    Parameters
    ----------
    shape : float
        Shape :math:`a > 0`.
    scale : float
        Scale :math:`\beta > 0`.

    Examples
    --------
    >>> import scipy.stats
    >>> ig = InverseGammaHyperprior(2.0, 3.0)
    >>> bool(abs(ig.logpdf(1.5) - scipy.stats.invgamma.logpdf(1.5, 2.0, scale=3.0)) < 1e-12)
    True
    """

    shape: float
    scale: float

    def logpdf(self, x: float) -> float:
        r"""Log density at ``x``.

        Parameters
        ----------
        x : float
            Point (positive).

        Returns
        -------
        float
            :math:`\log p(x)`.

        Examples
        --------
        >>> round(InverseGammaHyperprior(1.0, 1.0).logpdf(1.0), 6)
        -1.0
        """
        a, b = self.shape, self.scale
        return float(a * np.log(b) - gammaln(a) - (a + 1) * np.log(x) - b / x)

    def dlogpdf_dlog(self, x: float) -> float:
        r"""Derivative of the log density with respect to :math:`\log x`.

        Parameters
        ----------
        x : float
            Point (positive).

        Returns
        -------
        float
            :math:`-(a + 1) + \beta/x`.

        Examples
        --------
        >>> InverseGammaHyperprior(1.0, 2.0).dlogpdf_dlog(1.0)
        0.0
        """
        return float(-(self.shape + 1.0) + self.scale / x)


@dataclass(frozen=True)
class GLPHyperpriors:
    r"""Hyperpriors of :math:`(\lambda, \mu, \delta, \psi)`; ``None`` = flat.

    Parameters
    ----------
    lambda_ : GammaHyperprior, optional
        Default: mode 0.2, sd 0.4 (GLP 2015).
    mu : GammaHyperprior, optional
        Default: mode 1, sd 1.
    delta : GammaHyperprior, optional
        Default: mode 1, sd 1.
    psi : InverseGammaHyperprior, optional
        Default: shape = scale = :math:`0.02^2` (each :math:`\psi_j`).

    Examples
    --------
    >>> GLPHyperpriors().lambda_.mode
    0.2
    """

    lambda_: GammaHyperprior | None = field(default_factory=lambda: GammaHyperprior(0.2, 0.4))
    mu: GammaHyperprior | None = field(default_factory=lambda: GammaHyperprior(1.0, 1.0))
    delta: GammaHyperprior | None = field(default_factory=lambda: GammaHyperprior(1.0, 1.0))
    psi: InverseGammaHyperprior | None = field(
        default_factory=lambda: InverseGammaHyperprior(0.02**2, 0.02**2)
    )


GLP_HYPERPRIORS = GLPHyperpriors()


def log_hyperprior(hyper: BVARHyperparameters, hyperpriors: GLPHyperpriors | None = None) -> float:
    r"""Log hyperprior density :math:`\log p(\lambda, \mu, \delta, \psi)`.

    Parameters
    ----------
    hyper : BVARHyperparameters
        Point; ``mu``/``delta`` equal to ``None`` contribute nothing.
    hyperpriors : GLPHyperpriors, optional
        Hyperpriors (default GLP's); ``None`` fields are flat (contribute 0).

    Returns
    -------
    float
        Sum of the independent log densities.

    Examples
    --------
    >>> h = BVARHyperparameters(0.2, [1.0])
    >>> flat = GLPHyperpriors(lambda_=None, mu=None, delta=None, psi=None)
    >>> log_hyperprior(h, flat)
    0.0
    """
    hp = hyperpriors or GLP_HYPERPRIORS
    total = 0.0
    pairs = ((hp.lambda_, hyper.lambda_), (hp.mu, hyper.mu), (hp.delta, hyper.delta))
    for dist, value in pairs:
        if dist is not None and value is not None:
            total += dist.logpdf(value)
    if hp.psi is not None:
        total += sum(hp.psi.logpdf(float(v)) for v in hyper.psi)
    return float(total)


def log_hyperprior_gradient(
    hyper: BVARHyperparameters, hyperpriors: GLPHyperpriors | None = None
) -> HyperGradient:
    r"""Gradient of :func:`log_hyperprior` in the log hyperparameters.

    Parameters
    ----------
    hyper : BVARHyperparameters
        Point.
    hyperpriors : GLPHyperpriors, optional
        Hyperpriors (default GLP's); ``None`` fields contribute 0.

    Returns
    -------
    HyperGradient
        Derivatives with respect to :math:`\log\lambda, \log\psi, \log\mu,
        \log\delta` (``mu``/``delta`` ``None`` when the prior is off).

    Examples
    --------
    >>> g = log_hyperprior_gradient(BVARHyperparameters(0.2, [1.0], mu=1.0))
    >>> round(g.lambda_, 4), g.delta is None
    (0.0, True)
    """
    hp = hyperpriors or GLP_HYPERPRIORS

    def term(dist: GammaHyperprior | None, value: float | None) -> float | None:
        if value is None:
            return None
        return 0.0 if dist is None else dist.dlogpdf_dlog(value)

    psi_prior = hp.psi
    psi = (
        np.zeros_like(hyper.psi)
        if psi_prior is None
        else np.array([psi_prior.dlogpdf_dlog(float(v)) for v in hyper.psi])
    )
    return HyperGradient(
        lambda_=float(term(hp.lambda_, hyper.lambda_) or 0.0),
        psi=psi,
        mu=term(hp.mu, hyper.mu),
        delta=term(hp.delta, hyper.delta),
    )


# ---------------------------------------------------------------------------
# hyperparameter selection
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class HyperparameterSelection:
    """Result of :func:`select_hyperparameters`.

    Parameters
    ----------
    hyperparameters : BVARHyperparameters
        Posterior mode of the hyperparameters.
    log_ml : float
        Log marginal likelihood at the mode.
    log_posterior : float
        Log ML + log hyperprior at the mode (the maximised objective).
    names : tuple of str
        Labels of the estimated (log-transformed) parameters, e.g.
        ``("lambda", "mu", "delta", "psi[0]", ...)``.
    x : numpy.ndarray
        Mode on the log scale.
    inverse_hessian : numpy.ndarray or None
        Inverse of the Hessian of the negative objective on the log scale (proposal
        covariance for a random-walk Metropolis step); ``None`` if not requested.
    success : bool
        Optimiser convergence flag.
    n_evaluations : int
        Objective evaluations (including the Hessian).
    message : str
        Optimiser message.

    Examples
    --------
    >>> import numpy as np
    >>> s = VARSystem.from_array(np.random.default_rng(0).standard_normal((60, 2)), 1)
    >>> sel = select_hyperparameters(
    ...     s, PriorSettings("white_noise"), estimate=("lambda",), hessian=False
    ... )
    >>> sel.names, sel.inverse_hessian is None
    (('lambda',), True)
    """

    hyperparameters: BVARHyperparameters
    log_ml: float
    log_posterior: float
    names: tuple[str, ...]
    x: np.ndarray
    inverse_hessian: np.ndarray | None
    success: bool
    n_evaluations: int
    message: str

    @property
    def standard_errors(self) -> np.ndarray | None:
        """Approximate posterior standard deviations of the log hyperparameters.

        Returns
        -------
        numpy.ndarray or None
            Square roots of the diagonal of ``inverse_hessian``.

        Examples
        --------
        >>> import numpy as np
        >>> s = VARSystem.from_array(np.random.default_rng(0).standard_normal((60, 2)), 1)
        >>> sel = select_hyperparameters(s, PriorSettings("white_noise"), estimate=("lambda",))
        >>> sel.standard_errors.shape
        (1,)
        """
        if self.inverse_hessian is None:
            return None
        return np.sqrt(np.clip(np.diag(self.inverse_hessian), 0.0, None))


@dataclass(frozen=True)
class _Packer:
    """Maps between hyperparameters and the log-scale optimisation vector."""

    start: BVARHyperparameters
    estimate: tuple[str, ...]

    @property
    def names(self) -> tuple[str, ...]:
        """Labels of the vector entries."""
        out: list[str] = []
        for name in self.estimate:
            if name == "psi":
                out += [f"psi[{j}]" for j in range(self.start.psi.size)]
            else:
                out.append(name)
        return tuple(out)

    def pack(self, hyper: BVARHyperparameters) -> np.ndarray:
        """Log-scale vector of the estimated entries."""
        parts: list[np.ndarray] = []
        for name in self.estimate:
            value = hyper.psi if name == "psi" else getattr(hyper, _attr(name))
            parts.append(np.log(np.atleast_1d(value)))
        return np.concatenate(parts)

    def flatten(self, grad: HyperGradient) -> np.ndarray:
        """Gradient entries in the order of :meth:`pack`."""
        parts = [np.atleast_1d(np.asarray(getattr(grad, _attr(n)), float)) for n in self.estimate]
        return np.concatenate(parts)

    def unpack(self, z: np.ndarray) -> BVARHyperparameters:
        """Hyperparameters with the estimated entries replaced by ``exp(z)``."""
        values = np.exp(z)
        changes: dict[str, object] = {}
        pos = 0
        for name in self.estimate:
            if name == "psi":
                size = self.start.psi.size
                changes["psi"] = values[pos : pos + size]
                pos += size
            else:
                changes[_attr(name)] = float(values[pos])
                pos += 1
        return replace(self.start, **changes)  # type: ignore[arg-type]

    def bounds(self) -> list[tuple[float, float]]:
        """Log-scale box constraints of the estimated entries."""
        out: list[tuple[float, float]] = []
        for name in self.estimate:
            if name == "psi":
                logs = np.log(self.start.psi)
                out += [(float(v - np.log(1e3)), float(v + np.log(1e3))) for v in logs]
            else:
                out.append(_LOG_BOUNDS[name])
        return out


_LOG_BOUNDS = {
    "lambda": (float(np.log(1e-4)), float(np.log(10.0))),
    "mu": (float(np.log(1e-4)), float(np.log(100.0))),
    "delta": (float(np.log(1e-4)), float(np.log(100.0))),
}


def _attr(name: str) -> str:
    """Attribute of :class:`BVARHyperparameters` for an estimable name."""
    return "lambda_" if name == "lambda" else name


def _default_start(
    system: VARSystem, sum_of_coefficients: bool, initial_observation: bool
) -> BVARHyperparameters:
    """GLP starting point: lambda 0.2, mu = delta = 1, psi from AR(1) regressions."""
    return BVARHyperparameters(
        lambda_=0.2,
        psi=system.ar_residual_variances(1),
        mu=1.0 if sum_of_coefficients else None,
        delta=1.0 if initial_observation else None,
    )


def _check_estimate(estimate: Collection[str], start: BVARHyperparameters) -> tuple[str, ...]:
    """Validate the names to estimate (canonical order)."""
    estimate = (estimate,) if isinstance(estimate, str) else estimate
    unknown = set(estimate) - set(_ESTIMABLE)
    if unknown:
        raise ValueError(f"unknown hyperparameters {sorted(unknown)}; choose from {_ESTIMABLE}")
    names = tuple(name for name in _ESTIMABLE if name in estimate)
    if not names:
        raise ValueError("estimate must name at least one hyperparameter")
    off = [name for name in ("mu", "delta") if name in names and getattr(start, name) is None]
    if off:
        raise ValueError(f"cannot estimate {off}: the corresponding prior is switched off")
    return names


def _objective(
    system: VARSystem,
    packer: _Packer,
    settings: PriorSettings,
    hyperpriors: GLPHyperpriors,
    counter: list[int],
) -> Callable[[np.ndarray], tuple[float, np.ndarray]]:
    """Negative log posterior of the hyperparameters and its gradient (log scale)."""

    def fun(z: np.ndarray) -> tuple[float, np.ndarray]:
        counter[0] += 1
        z = np.asarray(z, dtype=float)
        hyper = packer.unpack(z)
        try:
            value, grad = log_marginal_likelihood_gradient(system, hyper, settings)
        except (NowcastDataError, np.linalg.LinAlgError):
            return _PENALTY, np.zeros_like(z)
        value += log_hyperprior(hyper, hyperpriors)
        grad = grad + log_hyperprior_gradient(hyper, hyperpriors)
        if not np.isfinite(value):
            return _PENALTY, np.zeros_like(z)
        return -value, -packer.flatten(grad)

    return fun


def numerical_hessian(
    gradient: Callable[[np.ndarray], np.ndarray], x: np.ndarray, step: float = 1e-4
) -> np.ndarray:
    r"""Hessian by central differences of an (analytic) gradient.

    Parameters
    ----------
    gradient : callable
        Gradient of the scalar function, mapping a 1-D array to a 1-D array.
    x : numpy.ndarray
        Point, shape ``(m,)``.
    step : float, default 1e-4
        Step size (absolute; the BVAR uses log-scale parameters).

    Returns
    -------
    numpy.ndarray
        Symmetrised ``(m, m)`` Hessian (:math:`2m` gradient evaluations).

    Examples
    --------
    >>> import numpy as np
    >>> grad = lambda v: np.array([2 * v[0] + 3 * v[1], 3 * v[0]])
    >>> np.round(numerical_hessian(grad, np.zeros(2)), 6).tolist()
    [[2.0, 3.0], [3.0, 0.0]]
    """
    x = np.asarray(x, dtype=float)
    eye = np.eye(x.size) * step
    cols = [(gradient(x + e) - gradient(x - e)) / (2 * step) for e in eye]
    H = np.column_stack(cols)
    return (H + H.T) / 2


def _safe_inverse(H: np.ndarray) -> np.ndarray:
    """Inverse of a Hessian; eigenvalues floored when not positive definite."""
    try:
        chol = scipy.linalg.cho_factor(H, lower=True)
        return scipy.linalg.cho_solve(chol, np.eye(H.shape[0]))
    except np.linalg.LinAlgError:
        vals, vecs = np.linalg.eigh((H + H.T) / 2)
        floor = max(1e-8, 1e-8 * float(np.max(np.abs(vals))))
        logger.warning("Hessian at the BVAR hyperparameter mode is not positive definite")
        return (vecs / np.maximum(vals, floor)) @ vecs.T


def select_hyperparameters(
    system: VARSystem,
    settings: PriorSettings | None = None,
    *,
    hyperpriors: GLPHyperpriors | None = None,
    estimate: Collection[str] | None = None,
    start: BVARHyperparameters | None = None,
    sum_of_coefficients: bool = True,
    initial_observation: bool = True,
    hessian: bool = True,
    max_iter: int = 500,
) -> HyperparameterSelection:
    r"""Hierarchical choice of the prior hyperparameters (Giannone, Lenza & Primiceri, 2015).

    Maximises :math:`\log p(Y \mid \theta) + \log p(\theta)` over
    :math:`\log\theta` with L-BFGS-B and the analytic gradient of
    :func:`log_marginal_likelihood_gradient` (box constraints: :math:`\lambda \in
    [10^{-4}, 10]`, :math:`\mu, \delta \in [10^{-4}, 100]`, :math:`\psi_j` within a
    factor :math:`10^3` of the start); non-estimated entries stay at ``start``.

    Parameters
    ----------
    system : VARSystem
        VAR data.
    settings : PriorSettings, optional
        Fixed prior settings.
    hyperpriors : GLPHyperpriors, optional
        Hyperpriors (default GLP's; ``GLPHyperpriors(None, None, None, None)`` gives the
        maximum marginal likelihood, i.e. empirical Bayes).
    estimate : collection of {"lambda", "mu", "delta", "psi"}, optional
        Hyperparameters to optimise; default: all of them that are in use (``mu`` and
        ``delta`` only when their prior is switched on).
    start : BVARHyperparameters, optional
        Starting point and fixed values; default :math:`\lambda = 0.2`,
        :math:`\mu = \delta = 1` (when switched on) and :math:`\psi` from AR(1)
        regressions.
    sum_of_coefficients, initial_observation : bool, default True
        Use the sum-of-coefficients / dummy-initial-observation priors (ignored when
        ``start`` is given: its ``mu``/``delta`` decide).
    hessian : bool, default True
        Compute the inverse Hessian at the mode (central differences of the analytic
        gradient: :math:`2m` evaluations for :math:`m` estimated entries).
    max_iter : int, default 500
        Maximum optimiser iterations.

    Returns
    -------
    HyperparameterSelection
        Mode, objective values and inverse Hessian.

    Raises
    ------
    ValueError
        If ``estimate`` is empty, has unknown names or names a switched-off prior.

    Warns
    -----
    ConvergenceWarning
        If the optimiser does not report convergence.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> y = np.cumsum(rng.standard_normal((80, 3)), axis=0)
    >>> s = VARSystem.from_array(y, lags=2)
    >>> sel = select_hyperparameters(s, hessian=False)
    >>> sel.names[:3], bool(0 < sel.hyperparameters.lambda_ < 10)
    (('lambda', 'mu', 'delta'), True)
    """
    settings = settings or PriorSettings()
    hp = hyperpriors or GLP_HYPERPRIORS
    start = start or _default_start(system, sum_of_coefficients, initial_observation)
    if estimate is None:
        estimate = [name for name in _ESTIMABLE if getattr(start, _attr(name)) is not None]
    packer = _Packer(start, _check_estimate(estimate, start))
    counter = [0]
    fun = _objective(system, packer, settings, hp, counter)
    res = scipy.optimize.minimize(
        fun,
        packer.pack(start),
        jac=True,
        method="L-BFGS-B",
        bounds=packer.bounds(),
        options={"maxiter": max_iter},
    )
    if not res.success:
        warnings.warn(
            f"BVAR hyperparameter optimisation did not converge: {res.message}",
            ConvergenceWarning,
            stacklevel=2,
        )
    inv_h = _safe_inverse(numerical_hessian(lambda z: fun(z)[1], res.x)) if hessian else None
    best = packer.unpack(res.x)
    log_ml = log_marginal_likelihood(system, best, settings)
    logger.debug("BVAR hyperparameters: %s (log ML %.3f)", best, log_ml)
    return HyperparameterSelection(
        hyperparameters=best,
        log_ml=log_ml,
        log_posterior=-float(res.fun),
        names=packer.names,
        x=np.asarray(res.x, dtype=float),
        inverse_hessian=inv_h,
        success=bool(res.success),
        n_evaluations=counter[0],
        message=str(res.message),
    )
