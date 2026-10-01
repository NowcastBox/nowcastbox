r"""Principal components, factor VAR and shock loadings for two-step estimation.

Building blocks of the first step of the two-step estimator of Giannone, Reichlin &
Small (2008) and Doz, Giannone & Reichlin (2011), written from the papers:

1. :func:`principal_components` - eigen-decomposition of the sample covariance of a
   balanced (standardised) panel :math:`X` (:math:`T \times N`),
   :math:`S = X'X/T = V D V'`. The loadings are the first :math:`r` eigenvectors
   :math:`\hat\Lambda = V_r` (so :math:`\hat\Lambda'\hat\Lambda = I_r`) and the factors
   :math:`\hat F = X V_r`, with :math:`\hat F'\hat F/T = D_r`.
2. :func:`fit_factor_var` - VAR(:math:`p`) without constant on the factors by OLS,
   :math:`f_t = \sum_{i=1}^p A_i f_{t-i} + v_t`.
3. :func:`shock_loadings` - rank-:math:`q` decomposition of the VAR residual
   covariance, :math:`\Sigma_v \approx B B'` with :math:`B = P_q M_q^{1/2}`
   (:math:`P_q`, :math:`M_q` the leading eigenvectors / eigenvalues).

The functions are pure NumPy and work on arrays without missing values.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nowcastbox.core.exceptions import NowcastDataError

__all__ = [
    "FactorVAR",
    "PrincipalComponents",
    "fit_factor_var",
    "principal_components",
    "shock_loadings",
]


@dataclass(frozen=True)
class PrincipalComponents:
    r"""Principal-component factors of a balanced panel.

    Parameters
    ----------
    factors : numpy.ndarray, shape (n_periods, n_factors)
        :math:`\hat F = X V_r`.
    loadings : numpy.ndarray, shape (n_series, n_factors)
        :math:`\hat\Lambda = V_r` (orthonormal columns).
    eigenvalues : numpy.ndarray, shape (n_series,)
        All eigenvalues of :math:`X'X/T`, in decreasing order.
    residuals : numpy.ndarray, shape (n_periods, n_series)
        Idiosyncratic residuals :math:`X - \hat F \hat\Lambda'`.

    Examples
    --------
    >>> import numpy as np
    >>> from nowcastbox.models._pca import principal_components
    >>> x = np.random.default_rng(0).standard_normal((50, 4))
    >>> pc = principal_components(x, 2)
    >>> pc.factors.shape, pc.loadings.shape
    ((50, 2), (4, 2))
    """

    factors: np.ndarray
    loadings: np.ndarray
    eigenvalues: np.ndarray
    residuals: np.ndarray

    @property
    def n_factors(self) -> int:
        """Number of factors :math:`r`."""
        return int(self.loadings.shape[1])

    @property
    def explained_variance_ratio(self) -> np.ndarray:
        """Share of the total variance explained by each principal component."""
        total = float(self.eigenvalues.sum())
        return self.eigenvalues / total if total > 0 else np.zeros_like(self.eigenvalues)

    @property
    def idiosyncratic_variance(self) -> np.ndarray:
        r"""Mean squared residual of each series, :math:`\mathrm{diag}(E'E/T)`."""
        return np.mean(self.residuals**2, axis=0)


def principal_components(x: np.ndarray, n_factors: int) -> PrincipalComponents:
    r"""Principal-component estimates of factors and loadings.

    Parameters
    ----------
    x : numpy.ndarray, shape (n_periods, n_series)
        Balanced panel without missing values (usually standardised).
    n_factors : int
        Number of factors :math:`r`, ``1 <= r <= min(n_periods, n_series)``.

    Returns
    -------
    PrincipalComponents
        Factors :math:`XV_r`, loadings :math:`V_r`, all eigenvalues of
        :math:`X'X/T` and residuals.

    Raises
    ------
    NowcastDataError
        If ``x`` is not a finite 2-D array.
    ValueError
        If ``n_factors`` is out of range.

    Notes
    -----
    The sign of each eigenvector is fixed so that its loadings sum to a non-negative
    number (the sign of a principal component is not identified).

    Examples
    --------
    >>> import numpy as np
    >>> from nowcastbox.models._pca import principal_components
    >>> rng = np.random.default_rng(1)
    >>> f = rng.standard_normal((200, 1))
    >>> x = f @ np.ones((1, 5)) + 0.1 * rng.standard_normal((200, 5))
    >>> pc = principal_components(x, 1)
    >>> bool(abs(np.corrcoef(pc.factors[:, 0], f[:, 0])[0, 1]) > 0.99)
    True
    """
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        raise NowcastDataError(f"x must be a non-empty 2-D array, got shape {arr.shape}.")
    if not bool(np.isfinite(arr).all()):
        raise NowcastDataError("x must not contain missing or infinite values.")
    n_periods, n_series = arr.shape
    if isinstance(n_factors, bool) or not isinstance(n_factors, int | np.integer):
        raise ValueError(f"n_factors must be an integer, got {n_factors!r}.")
    if not 1 <= n_factors <= min(n_periods, n_series):
        raise ValueError(
            f"n_factors must be between 1 and min(n_periods, n_series) = "
            f"{min(n_periods, n_series)}, got {n_factors}."
        )
    cov = arr.T @ arr / n_periods
    eigval, eigvec = np.linalg.eigh(0.5 * (cov + cov.T))
    order = np.argsort(eigval)[::-1]
    eigval = np.clip(eigval[order], 0.0, None)
    eigvec = eigvec[:, order]
    loadings = eigvec[:, :n_factors].copy()
    signs = np.where(loadings.sum(axis=0) < 0, -1.0, 1.0)
    loadings *= signs
    factors = arr @ loadings
    residuals = arr - factors @ loadings.T
    return PrincipalComponents(
        factors=factors, loadings=loadings, eigenvalues=eigval, residuals=residuals
    )


@dataclass(frozen=True)
class FactorVAR:
    r"""OLS estimate of a VAR(p) without intercept.

    Parameters
    ----------
    coefficients : numpy.ndarray, shape (r, r * p)
        Stacked :math:`[A_1, \dots, A_p]`.
    residuals : numpy.ndarray, shape (n_used, r)
        OLS residuals :math:`\hat v_t` of the usable periods.
    residual_cov : numpy.ndarray, shape (r, r)
        :math:`\hat\Sigma_v = \hat V'\hat V/n_{used}` (maximum-likelihood divisor).
    factor_lags : int
        VAR order :math:`p`.

    Examples
    --------
    >>> import numpy as np
    >>> from nowcastbox.models._pca import fit_factor_var
    >>> f = np.random.default_rng(0).standard_normal((100, 2))
    >>> fit_factor_var(f, 2).coefficients.shape
    (2, 4)
    """

    coefficients: np.ndarray
    residuals: np.ndarray
    residual_cov: np.ndarray
    factor_lags: int

    def matrices(self) -> list[np.ndarray]:
        r"""Return :math:`[A_1, \dots, A_p]` as a list of ``(r, r)`` arrays.

        Returns
        -------
        list of numpy.ndarray
            VAR coefficient matrices.

        Examples
        --------
        >>> import numpy as np
        >>> f = np.random.default_rng(0).standard_normal((60, 2))
        >>> len(fit_factor_var(f, 3).matrices())
        3
        """
        r = self.coefficients.shape[0]
        return [self.coefficients[:, i * r : (i + 1) * r].copy() for i in range(self.factor_lags)]


def fit_factor_var(factors: np.ndarray, factor_lags: int) -> FactorVAR:
    r"""Estimate :math:`f_t = \sum_{i=1}^p A_i f_{t-i} + v_t` by least squares.

    Parameters
    ----------
    factors : numpy.ndarray, shape (n_periods, r)
        Factor estimates on a contiguous time grid. Periods with missing values
        (NaN rows) are allowed: a regression row is used only when the period and
        all its ``p`` lags are observed.
    factor_lags : int
        VAR order :math:`p \ge 1`.

    Returns
    -------
    FactorVAR
        Coefficients, residuals and residual covariance.

    Raises
    ------
    NowcastDataError
        If there are fewer than ``r * p + 1`` usable regression rows or the factors
        contain infinite values.
    ValueError
        If ``factor_lags`` is not a positive integer.

    Examples
    --------
    >>> import numpy as np
    >>> from nowcastbox.models._pca import fit_factor_var
    >>> rng = np.random.default_rng(3)
    >>> f = np.zeros((2000, 1))
    >>> for t in range(1, 2000):
    ...     f[t] = 0.7 * f[t - 1] + rng.standard_normal()
    >>> round(float(fit_factor_var(f, 1).coefficients[0, 0]), 1)
    0.7
    """
    f = np.asarray(factors, dtype=float)
    if f.ndim == 1:
        f = f[:, None]
    if f.ndim != 2 or bool(np.isinf(f).any()):
        raise NowcastDataError("factors must be a 2-D array without infinite values.")
    if isinstance(factor_lags, bool) or not isinstance(factor_lags, int | np.integer):
        raise ValueError(f"factor_lags must be an integer, got {factor_lags!r}.")
    p = int(factor_lags)
    if p < 1:
        raise ValueError(f"factor_lags must be >= 1, got {p}.")
    n, r = f.shape
    y = f[p:]
    lagged = (
        np.hstack([f[p - i : n - i] for i in range(1, p + 1)]) if n > p else np.empty((0, r * p))
    )
    usable = np.isfinite(y).all(axis=1) & np.isfinite(lagged).all(axis=1)
    y, lagged = y[usable], lagged[usable]
    n_eff = y.shape[0]
    if n_eff < r * p + 1:
        raise NowcastDataError(
            f"Only {n_eff} usable periods to estimate a VAR({p}) with {r} factors; "
            f"need at least {r * p + 1}."
        )
    coef, *_ = np.linalg.lstsq(lagged, y, rcond=None)
    resid = y - lagged @ coef
    cov = resid.T @ resid / n_eff
    return FactorVAR(
        coefficients=coef.T.copy(),
        residuals=resid,
        residual_cov=0.5 * (cov + cov.T),
        factor_lags=p,
    )


def shock_loadings(residual_cov: np.ndarray, n_shocks: int) -> tuple[np.ndarray, np.ndarray]:
    r"""Rank-:math:`q` factorisation :math:`\Sigma_v \approx B B'`.

    Parameters
    ----------
    residual_cov : numpy.ndarray, shape (r, r)
        Covariance of the VAR residuals.
    n_shocks : int
        Number of dynamic shocks :math:`q`, ``1 <= q <= r``.

    Returns
    -------
    shock_loadings : numpy.ndarray, shape (r, q)
        :math:`B = P_q M_q^{1/2}`, eigenvectors scaled by the square roots of the
        :math:`q` largest eigenvalues.
    eigenvalues : numpy.ndarray, shape (r,)
        All eigenvalues of :math:`\Sigma_v` in decreasing order.

    Raises
    ------
    ValueError
        If ``n_shocks`` is out of range or the matrix is not square.

    Examples
    --------
    >>> import numpy as np
    >>> from nowcastbox.models._pca import shock_loadings
    >>> b, ev = shock_loadings(np.array([[2.0, 0.0], [0.0, 1.0]]), 2)
    >>> np.allclose(b @ b.T, [[2.0, 0.0], [0.0, 1.0]])
    True
    """
    cov = np.asarray(residual_cov, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError(f"residual_cov must be square, got shape {cov.shape}.")
    r = cov.shape[0]
    if isinstance(n_shocks, bool) or not isinstance(n_shocks, int | np.integer):
        raise ValueError(f"n_shocks must be an integer, got {n_shocks!r}.")
    if not 1 <= n_shocks <= r:
        raise ValueError(f"n_shocks must be between 1 and {r}, got {n_shocks}.")
    eigval, eigvec = np.linalg.eigh(0.5 * (cov + cov.T))
    order = np.argsort(eigval)[::-1]
    eigval = np.clip(eigval[order], 0.0, None)
    eigvec = eigvec[:, order]
    vec = eigvec[:, :n_shocks]
    vec = vec * np.where(vec.sum(axis=0) < 0, -1.0, 1.0)
    return vec * np.sqrt(eigval[:n_shocks]), eigval
