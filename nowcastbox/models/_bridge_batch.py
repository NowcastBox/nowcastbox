r"""Batched OLS for many small bridge equations sharing one feature matrix.

Every bridge equation of :class:`~nowcastbox.models.BridgeCombination` is a regression
of the target on a few columns of one target-frequency feature matrix :math:`F`
(``(T, n_features)``: constant, aggregated indicators and their lags, target lags).
Equations with the same number of coefficients :math:`k` are stacked into a tensor
:math:`X` of shape ``(E, T, k)`` and estimated at once. Each equation keeps only the
periods where the target and all its regressors are observed: excluded rows are set to
zero in both :math:`X_e` and :math:`y`, which leaves the OLS solution on the remaining
rows unchanged. The stacked problems are solved by a batched QR decomposition
(:math:`X_e = Q_e R_e`, :math:`\hat\beta_e = R_e^{-1} Q_e' y`); an equation is flagged
invalid when :math:`R_e` is numerically singular (collinear regressors) or when it has
fewer than :math:`k + 1` usable periods.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "BatchFit",
    "batched_ols",
    "design_tensor",
    "one_step_errors",
    "recursive_predict",
    "training_mask",
]


@dataclass(frozen=True)
class BatchFit:
    """OLS estimates of a batch of equations.

    Parameters
    ----------
    beta : numpy.ndarray
        Coefficients ``(E, k)`` (NaN for invalid equations).
    ssr : numpy.ndarray
        Residual sums of squares ``(E,)``.
    tss : numpy.ndarray
        Centred total sums of squares of the target on each sample ``(E,)``.
    n_obs : numpy.ndarray
        Number of periods used ``(E,)``.
    valid : numpy.ndarray
        Whether the equation could be estimated ``(E,)``.
    residuals : numpy.ndarray
        Residuals ``(E, T)`` (NaN outside each sample).

    Examples
    --------
    >>> import numpy as np
    >>> X = np.column_stack([np.ones(5), np.arange(5.0)])[None]
    >>> fit = batched_ols(X, 1.0 + 2.0 * np.arange(5.0), np.ones((1, 5), bool))
    >>> np.round(fit.beta, 8).tolist(), fit.valid.tolist()
    ([[1.0, 2.0]], [True])
    """

    beta: np.ndarray
    ssr: np.ndarray
    tss: np.ndarray
    n_obs: np.ndarray
    valid: np.ndarray
    residuals: np.ndarray


def design_tensor(features: np.ndarray, columns: np.ndarray) -> np.ndarray:
    """Stack the regressor matrices of a batch of equations.

    Parameters
    ----------
    features : numpy.ndarray
        Feature matrix ``(T, n_features)``.
    columns : numpy.ndarray
        Integer feature indices ``(E, k)`` of each equation.

    Returns
    -------
    numpy.ndarray
        Tensor ``(E, T, k)``.

    Examples
    --------
    >>> import numpy as np
    >>> design_tensor(np.arange(6.0).reshape(3, 2), np.array([[1, 0]])).shape
    (1, 3, 2)
    """
    return np.ascontiguousarray(np.transpose(features[:, columns], (1, 0, 2)))


def training_mask(X: np.ndarray, y: np.ndarray, end: int | None = None) -> np.ndarray:
    """Periods usable by each equation: target and all regressors observed.

    Parameters
    ----------
    X : numpy.ndarray
        Regressors ``(E, T, k)``.
    y : numpy.ndarray
        Target ``(T,)`` (NaN = not observed).
    end : int, optional
        Only periods before position ``end`` are used (pseudo out-of-sample training).

    Returns
    -------
    numpy.ndarray
        Boolean mask ``(E, T)``.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.ones((1, 3, 1))
    >>> training_mask(X, np.array([1.0, np.nan, 2.0])).tolist()
    [[True, False, True]]
    """
    mask = np.isfinite(X).all(axis=2) & np.isfinite(y)[None, :]
    if end is not None:
        mask[:, end:] = False
    return mask


def batched_ols(X: np.ndarray, y: np.ndarray, mask: np.ndarray) -> BatchFit:
    """Estimate a batch of OLS regressions on their own samples.

    Parameters
    ----------
    X : numpy.ndarray
        Regressors ``(E, T, k)`` (may contain NaN outside ``mask``).
    y : numpy.ndarray
        Target ``(T,)``.
    mask : numpy.ndarray
        Periods used by each equation ``(E, T)``.

    Returns
    -------
    BatchFit
        Estimates; invalid equations (fewer than ``k + 1`` periods or a singular
        design) have NaN coefficients.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.stack([np.column_stack([np.ones(4), np.ones(4)])])  # collinear
    >>> batched_ols(X, np.arange(4.0), np.ones((1, 4), bool)).valid.tolist()
    [False]
    """
    n_eq, n_per, k = X.shape
    Xm = np.where(mask[:, :, None], X, 0.0)
    ym = np.where(mask, np.nan_to_num(y)[None, :], 0.0)
    n_obs = mask.sum(axis=1)
    if n_per < k:
        valid = np.zeros(n_eq, dtype=bool)
        beta = np.full((n_eq, k), np.nan)
    else:
        q, r = np.linalg.qr(Xm)
        diag = np.abs(np.diagonal(r, axis1=1, axis2=2))
        scale = np.maximum(diag.max(axis=1, initial=0.0), np.finfo(float).tiny)
        tol = max(n_per, k) * np.finfo(float).eps * 1e3
        valid = (diag > tol * scale[:, None]).all(axis=1) & (n_obs >= k + 1)
        r_safe = np.where(valid[:, None, None], r, np.eye(k)[None])
        qty = np.einsum("etk,et->ek", q, ym)
        beta = np.linalg.solve(r_safe, qty[:, :, None])[:, :, 0]
        beta[~valid] = np.nan
    resid = np.where(mask, ym - np.einsum("etk,ek->et", Xm, np.nan_to_num(beta)), np.nan)
    resid[~valid] = np.nan
    ssr = np.where(valid, np.nansum(resid**2, axis=1), np.nan)
    count = np.maximum(n_obs, 1)
    ybar = ym.sum(axis=1) / count
    tss = np.where(mask, (ym - ybar[:, None]) ** 2, 0.0).sum(axis=1)
    return BatchFit(beta=beta, ssr=ssr, tss=tss, n_obs=n_obs, valid=valid, residuals=resid)


def recursive_predict(
    X: np.ndarray, beta: np.ndarray, observed: np.ndarray, n_target_lags: int
) -> np.ndarray:
    """Predictions of every equation for every period, iterating unobserved target lags.

    The last ``n_target_lags`` regressors of each equation are the target lags
    ``1..n_target_lags``; where the target is not observed they are replaced by the
    equation's own earlier predictions (iterated multi-step forecasts).

    Parameters
    ----------
    X : numpy.ndarray
        Regressors ``(E, T, k)`` (the values of the target-lag columns are not used:
        lags are taken from ``observed`` or from earlier predictions).
    beta : numpy.ndarray
        Coefficients ``(E, k)``.
    observed : numpy.ndarray
        Target ``(T,)`` (NaN = not observed).
    n_target_lags : int
        Number of target lags.

    Returns
    -------
    numpy.ndarray
        Predictions ``(E, T)`` (NaN where a regressor is missing).

    Examples
    --------
    >>> import numpy as np
    >>> X = np.ones((1, 3, 2))  # constant + one target lag
    >>> recursive_predict(X, np.array([[1.0, 0.5]]), np.array([2.0, np.nan, np.nan]), 1).tolist()
    [[nan, 2.0, 2.0]]
    """
    n_static = X.shape[2] - n_target_lags
    static = np.einsum("etk,ek->et", X[:, :, :n_static], beta[:, :n_static])
    if n_target_lags == 0:
        return static
    phi = beta[:, n_static:]
    n_per = X.shape[1]
    history = np.tile(np.asarray(observed, dtype=float), (X.shape[0], 1))
    preds = np.full_like(static, np.nan)
    for t in range(n_per):
        lags = np.full((X.shape[0], n_target_lags), np.nan)
        for j in range(1, min(t, n_target_lags) + 1):
            lags[:, j - 1] = history[:, t - j]
        preds[:, t] = static[:, t] + np.einsum("ej,ej->e", lags, phi)
        if not np.isfinite(observed[t]):
            history[:, t] = preds[:, t]
    return preds


def one_step_errors(
    X: np.ndarray, y: np.ndarray, positions: np.ndarray, min_train: int
) -> np.ndarray:
    """Pseudo out-of-sample one-step errors of a batch of equations.

    For each evaluation position :math:`s` every equation is re-estimated on the
    periods before :math:`s` and predicts :math:`y_s` from its regressors at
    :math:`s` (target lags at their observed values).

    Parameters
    ----------
    X : numpy.ndarray
        Regressors ``(E, T, k)``.
    y : numpy.ndarray
        Target ``(T,)``.
    positions : numpy.ndarray
        Evaluation positions (periods with an observed target).
    min_train : int
        Minimum number of training periods; fewer gives a NaN error.

    Returns
    -------
    numpy.ndarray
        Errors ``(E, len(positions))``.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.column_stack([np.ones(6), np.arange(6.0)])[None]
    >>> errors = one_step_errors(X, 2.0 * np.arange(6.0), np.array([4, 5]), 3)
    >>> errors.shape, bool(np.abs(errors).max() < 1e-8)
    ((1, 2), True)
    """
    base = training_mask(X, y)
    errors = np.full((X.shape[0], len(positions)), np.nan)
    for i, s in enumerate(np.asarray(positions, dtype=int)):
        mask = base.copy()
        mask[:, s:] = False
        fit = batched_ols(X, y, mask)
        pred = np.einsum("ek,ek->e", X[:, s, :], fit.beta)
        ok = fit.valid & (fit.n_obs >= min_train)
        errors[:, i] = np.where(ok, y[s] - pred, np.nan)
    return errors
