"""Kalman filter with arbitrary missing-data patterns.

Two algorithms share the same output (:class:`FilterResult`):

* ``"univariate"`` - the univariate treatment of multivariate observations
  (Koopman & Durbin, 2000; Durbin & Koopman, 2012, sec. 6.4): with diagonal ``H`` the
  observed series of each period are processed one at a time, so no ``n x n`` matrix
  is ever formed or inverted and missing values are simply skipped. This is the default
  and the fast path for large panels (innovation I2).
* ``"multivariate"`` - the textbook filter on the observed rows ``W_t y_t`` of each
  period (Durbin & Koopman, 2012, sec. 4.10), used for non-diagonal ``H``.

Optionally the observation vector is first collapsed to the rank of ``Z``
(:func:`~nowcastbox.statespace.collapse_observations`; Jungbacker & Koopman, 2015),
which makes the cost per period independent of the number of series.

Time-varying observation equations (``StateSpace(..., obs_index=...)``) are handled by
passing the observation store and the per-period index to the kernels. For dynamic
factor models with idiosyncratic states, :func:`~nowcastbox.statespace.structured_smoother`
is usually much faster than the dense filter/smoother pair.

The kernels are compiled with Numba; the first call of a session pays the compilation
(cached on disk afterwards).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from nowcastbox._logging import get_logger
from nowcastbox.core.exceptions import NowcastDataError
from nowcastbox.statespace import _kernels
from nowcastbox.statespace.representation import StateSpace

__all__ = [
    "FilterMethod",
    "FilterResult",
    "kalman_filter",
    "loglikelihood",
    "prepare_observations",
    "transition_csr",
]

logger = get_logger(__name__)

FloatArray = NDArray[np.float64]
FilterMethod = Literal["auto", "univariate", "multivariate"]
_METHODS = ("auto", "univariate", "multivariate")


def prepare_observations(model: StateSpace, observations: ArrayLike) -> FloatArray:
    """Validate observations against ``model`` and return a float64 ``(n_periods, n_obs)`` copy.

    Parameters
    ----------
    model : StateSpace
        Model whose ``n_obs`` the observations must match.
    observations : array_like or pandas.DataFrame
        Observations, ``NaN`` = missing. A 1-D array is accepted when ``n_obs == 1``.

    Returns
    -------
    numpy.ndarray, shape (n_periods, n_obs)
        C-contiguous float64 copy.

    Raises
    ------
    nowcastbox.core.exceptions.NowcastDataError
        On non-numeric data, wrong shape, no periods or infinite values.

    Examples
    --------
    >>> from nowcastbox.statespace import StateSpace, prepare_observations
    >>> ssm = StateSpace([[0.5]], [[1.0]], [[1.0]], [1.0])
    >>> prepare_observations(ssm, [1.0, 2.0]).shape
    (2, 1)
    """
    if isinstance(observations, (pd.DataFrame, pd.Series)):
        observations = observations.to_numpy(dtype=np.float64, na_value=np.nan)
    try:
        y = np.array(observations, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as err:
        msg = "observations must be numeric"
        raise NowcastDataError(msg) from err
    if y.ndim == 1 and model.n_obs == 1:
        y = y.reshape(-1, 1)
    if y.ndim != 2 or y.shape[1] != model.n_obs:
        msg = f"observations must have shape (n_periods, {model.n_obs}), got {y.shape}"
        raise NowcastDataError(msg)
    if y.shape[0] == 0:
        msg = "observations must contain at least one period"
        raise NowcastDataError(msg)
    if model.n_periods is not None and y.shape[0] != model.n_periods:
        msg = (
            f"the model has a time-varying observation equation for {model.n_periods} "
            f"periods, got {y.shape[0]} periods of observations"
        )
        raise NowcastDataError(msg)
    if np.any(np.isinf(y)):
        msg = "observations contain infinite values"
        raise NowcastDataError(msg)
    return np.ascontiguousarray(y)


@dataclass(frozen=True)
class FilterResult:
    """Output of :func:`kalman_filter`.

    Periods are 0-based: row ``t`` refers to the ``(t+1)``-th observation period.

    Attributes
    ----------
    model : StateSpace
        Model that was filtered.
    observations : numpy.ndarray, shape (n_periods, n_obs)
        Observations (``NaN`` = missing).
    predicted_state : numpy.ndarray, shape (n_periods + 1, n_states)
        ``a_t = E[alpha_t | y_1..y_{t-1}]``; the last row is the one-step-ahead
        prediction beyond the sample.
    predicted_state_cov : numpy.ndarray, shape (n_periods + 1, n_states, n_states)
        ``P_t = Var[alpha_t | y_1..y_{t-1}]``.
    filtered_state : numpy.ndarray, shape (n_periods, n_states)
        ``a_{t|t} = E[alpha_t | y_1..y_t]``.
    filtered_state_cov : numpy.ndarray, shape (n_periods, n_states, n_states)
        ``P_{t|t} = Var[alpha_t | y_1..y_t]``.
    loglikelihood_obs : numpy.ndarray, shape (n_periods,)
        Gaussian log-likelihood contribution of each period (0 when nothing observed).
    n_observed : numpy.ndarray of int, shape (n_periods,)
        Number of non-missing series per period.
    method : str
        ``"univariate"`` or ``"multivariate"``.
    collapsed : bool
        Whether the observation vector was collapsed (Jungbacker & Koopman, 2015).
    smoother_score : numpy.ndarray or None, shape (n_periods, n_states)
        ``Z_t' F_t^{-1} v_t`` per period (input of the smoother).
    smoother_information : numpy.ndarray or None, shape (n_periods, n_states, n_states)
        ``Z_t' F_t^{-1} Z_t`` per period (input of the smoother).

    Examples
    --------
    >>> import numpy as np
    >>> from nowcastbox.statespace import StateSpace, kalman_filter
    >>> ssm = StateSpace([[0.5]], [[1.0]], [[1.0]], [1.0])
    >>> res = kalman_filter(ssm, np.array([[1.0], [np.nan]]))
    >>> res.n_periods, res.predicted_state.shape
    (2, (3, 1))
    """

    model: StateSpace
    observations: FloatArray = field(repr=False)
    predicted_state: FloatArray = field(repr=False)
    predicted_state_cov: FloatArray = field(repr=False)
    filtered_state: FloatArray = field(repr=False)
    filtered_state_cov: FloatArray = field(repr=False)
    loglikelihood_obs: FloatArray = field(repr=False)
    n_observed: NDArray[np.int64] = field(repr=False)
    method: str
    collapsed: bool = False
    smoother_score: FloatArray | None = field(default=None, repr=False)
    smoother_information: FloatArray | None = field(default=None, repr=False)

    @property
    def loglikelihood(self) -> float:
        """Total Gaussian log-likelihood ``sum_t log p(y_t | y_1..y_{t-1})``."""
        return float(np.sum(self.loglikelihood_obs))

    @property
    def n_periods(self) -> int:
        """Number of periods ``n``."""
        return int(self.filtered_state.shape[0])

    @property
    def n_states(self) -> int:
        """Number of states ``m``."""
        return int(self.filtered_state.shape[1])

    @property
    def has_smoother_quantities(self) -> bool:
        """Whether the quantities needed by the smoother were stored."""
        return self.smoother_score is not None and self.smoother_information is not None

    def forecasts(self) -> FloatArray:
        """One-step-ahead forecasts ``E[y_t | y_1..y_{t-1}] = Z a_t + d``.

        Returns
        -------
        numpy.ndarray, shape (n_periods, n_obs)
            Forecasts of every series (also for missing entries).

        Examples
        --------
        >>> import numpy as np
        >>> from nowcastbox.statespace import StateSpace, kalman_filter
        >>> ssm = StateSpace([[0.5]], [[2.0]], [[1.0]], [1.0], initial_state=[1.0])
        >>> kalman_filter(ssm, np.array([[np.nan]])).forecasts()
        array([[2.]])
        """
        n = self.n_periods
        index = self.model.period_index(n)
        z = self.model.designs(n)
        return (
            np.einsum("tij,tj->ti", z, self.predicted_state[:-1])
            + (self.model.obs_intercept_store[index])
        )

    def forecast_errors(self) -> FloatArray:
        """One-step-ahead forecast errors ``v_t = y_t - Z a_t - d`` (``NaN`` where missing).

        Returns
        -------
        numpy.ndarray, shape (n_periods, n_obs)
            Prediction errors.

        Examples
        --------
        >>> import numpy as np
        >>> from nowcastbox.statespace import StateSpace, kalman_filter
        >>> ssm = StateSpace([[0.5]], [[1.0]], [[1.0]], [1.0])
        >>> kalman_filter(ssm, np.array([[3.0]])).forecast_errors()
        array([[3.]])
        """
        return self.observations - self.forecasts()

    def forecast_cov(self, period: int) -> FloatArray:
        """Covariance ``F_t = Z P_t Z' + H`` of the one-step-ahead forecast of all series.

        Parameters
        ----------
        period : int
            0-based period ``t`` (``-1`` .. ``-n_periods`` count from the end).

        Returns
        -------
        numpy.ndarray, shape (n_obs, n_obs)
            Forecast covariance.

        Raises
        ------
        IndexError
            If ``period`` is out of range.

        Examples
        --------
        >>> import numpy as np
        >>> from nowcastbox.statespace import StateSpace, kalman_filter
        >>> ssm = StateSpace([[0.0]], [[1.0]], [[1.0]], [1.0])
        >>> kalman_filter(ssm, np.array([[0.0]])).forecast_cov(0)
        array([[2.]])
        """
        if not -self.n_periods <= period < self.n_periods:
            msg = f"period {period} out of range for {self.n_periods} periods"
            raise IndexError(msg)
        period = period % self.n_periods
        z_mat = self.model.design_at(period)
        return z_mat @ self.predicted_state_cov[period] @ z_mat.T + self.model.obs_cov_at(period)


def _resolve_method(model: StateSpace, method: str, collapse: bool) -> str:
    if method not in _METHODS:
        msg = f"method must be one of {_METHODS}, got {method!r}"
        raise ValueError(msg)
    if collapse:
        if method == "multivariate":
            msg = "collapse=True uses the univariate filter; method='multivariate' is not allowed"
            raise ValueError(msg)
        return "univariate"
    if method == "auto":
        return "univariate" if model.obs_cov_is_diagonal else "multivariate"
    if method == "univariate" and not model.obs_cov_is_diagonal:
        msg = (
            "the univariate filter requires a diagonal obs_cov; use method='multivariate' "
            "or collapse=True for a full H"
        )
        raise ValueError(msg)
    return method


def transition_csr(
    transition: FloatArray,
) -> tuple[NDArray[np.int64], NDArray[np.int64], FloatArray, bool]:
    """CSR form of ``T`` and whether the kernels should use it (sparse or small ``T``).

    Parameters
    ----------
    transition : numpy.ndarray, shape (m, m)
        Transition matrix.

    Returns
    -------
    indptr, indices : numpy.ndarray of int
        CSR row pointers and column indices.
    values : numpy.ndarray
        Non-zero values.
    sparse : bool
        True when at most 30 % of the entries are non-zero or ``m <= 16`` (explicit
        loops beat BLAS calls); otherwise the kernels use dense BLAS products.

    Examples
    --------
    >>> import numpy as np
    >>> from nowcastbox.statespace.kalman import transition_csr
    >>> ptr, idx, val, sparse = transition_csr(np.array([[0.5, 0.0], [1.0, 0.0]]))
    >>> ptr.tolist(), idx.tolist(), val.tolist(), sparse
    ([0, 1, 2], [0, 0], [0.5, 1.0], True)
    """
    t_mat = np.ascontiguousarray(transition, dtype=np.float64)
    m = t_mat.shape[0]
    rows, cols = np.nonzero(t_mat)
    indptr = np.zeros(m + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(np.bincount(rows, minlength=m))
    sparse = bool(rows.size <= 0.3 * m * m or m <= 16)
    return indptr, cols.astype(np.int64), np.ascontiguousarray(t_mat[rows, cols]), sparse


def kalman_filter(
    model: StateSpace,
    observations: ArrayLike,
    *,
    method: FilterMethod = "auto",
    collapse: bool = False,
    store_smoother: bool = True,
) -> FilterResult:
    """Kalman filter with missing observations.

    Parameters
    ----------
    model : StateSpace
        State-space model (time-invariant state equation; the observation equation may
        be time varying).
    observations : array_like, shape (n_periods, n_obs)
        Observations; ``NaN`` marks missing values (any pattern, including periods with
        no observation at all).
    method : {"auto", "univariate", "multivariate"}, default "auto"
        ``"auto"`` uses the univariate treatment (Koopman & Durbin, 2000) when ``H`` is
        diagonal and the multivariate filter otherwise.
    collapse : bool, default False
        Collapse the observation vector to the rank of ``Z`` before filtering
        (Jungbacker & Koopman, 2015). Recommended when ``n_obs`` is much larger than
        the number of factors; results are identical up to rounding.
    store_smoother : bool, default True
        Store the per-period quantities needed by
        :func:`~nowcastbox.statespace.smooth` (``O(n_periods m^2)`` memory).

    Returns
    -------
    FilterResult
        Predicted and filtered states/covariances and log-likelihood.

    Raises
    ------
    ValueError
        On an invalid ``method`` or incompatible ``method``/``H`` combination.
    numpy.linalg.LinAlgError
        If a forecast covariance is not positive definite (multivariate filter).
    nowcastbox.core.exceptions.NowcastDataError
        On malformed observations.

    Examples
    --------
    >>> import numpy as np
    >>> from nowcastbox.statespace import StateSpace, kalman_filter
    >>> ssm = StateSpace([[0.8]], [[1.0], [0.5]], [[1.0]], [0.5, 0.2])
    >>> y = np.array([[0.3, 0.1], [np.nan, 0.4], [np.nan, np.nan]])
    >>> res = kalman_filter(ssm, y)
    >>> res.method, res.filtered_state.shape, bool(np.isfinite(res.loglikelihood))
    ('univariate', (3, 1), True)
    """
    resolved = _resolve_method(model, method, collapse)
    y = prepare_observations(model, observations)
    n_observed = np.sum(~np.isnan(y), axis=1).astype(np.int64)
    index = model.period_index(y.shape[0])
    common = (
        np.ascontiguousarray(model.T),
        np.ascontiguousarray(model.c),
        np.ascontiguousarray(model.state_disturbance_cov),
        np.ascontiguousarray(model.a0),
        np.ascontiguousarray(model.P0),
        bool(store_smoother),
        *transition_csr(model.T),
    )
    adjustment: FloatArray | None = None
    if collapse:
        from nowcastbox.statespace.collapse import collapse_observations

        col = collapse_observations(model, y)
        k_max = col.max_dimension
        n_pat = col.n_patterns
        out = _kernels.univariate_filter(
            np.ascontiguousarray(col.observations),
            np.ascontiguousarray(col.design),
            np.zeros((n_pat, k_max)),
            np.ones((n_pat, k_max)),
            col.pattern_index,
            *common,
        )
        adjustment = col.loglikelihood_adjustment
        logger.debug(
            "collapsed %d series to at most %d per period (%d patterns)",
            model.n_obs,
            k_max,
            n_pat,
        )
    elif resolved == "univariate":
        out = _kernels.univariate_filter(
            y,
            np.ascontiguousarray(model.design_store),
            np.ascontiguousarray(model.obs_intercept_store),
            np.ascontiguousarray(model.obs_cov_diagonal_store),
            index,
            *common,
        )
    else:
        try:
            out = _kernels.multivariate_filter(
                y,
                np.ascontiguousarray(model.design_store),
                np.ascontiguousarray(model.obs_intercept_store),
                np.ascontiguousarray(model.obs_cov_matrix_store),
                index,
                *common,
            )
        except np.linalg.LinAlgError as err:
            msg = (
                "forecast error covariance is not positive definite in the multivariate "
                "filter (check obs_cov and initial_state_cov)"
            )
            raise np.linalg.LinAlgError(msg) from err
    a_pred, p_pred, a_filt, p_filt, ll_obs, _n_used, score, info = out
    if adjustment is not None:
        ll_obs = ll_obs + adjustment
    return FilterResult(
        model=model,
        observations=y,
        predicted_state=a_pred,
        predicted_state_cov=p_pred,
        filtered_state=a_filt,
        filtered_state_cov=p_filt,
        loglikelihood_obs=ll_obs,
        n_observed=n_observed,
        method=resolved,
        collapsed=bool(collapse),
        smoother_score=score if store_smoother else None,
        smoother_information=info if store_smoother else None,
    )


def loglikelihood(
    model: StateSpace,
    observations: ArrayLike,
    *,
    method: FilterMethod = "auto",
    collapse: bool = False,
) -> float:
    """Gaussian log-likelihood of ``observations`` under ``model`` (prediction-error form).

    Parameters
    ----------
    model : StateSpace
        State-space model.
    observations : array_like, shape (n_periods, n_obs)
        Observations with ``NaN`` for missing values.
    method : {"auto", "univariate", "multivariate"}, default "auto"
        Filter algorithm (see :func:`kalman_filter`).
    collapse : bool, default False
        Collapse the observation vector first (Jungbacker & Koopman, 2015).

    Returns
    -------
    float
        ``sum_t log p(y_t | y_1..y_{t-1})``.

    Examples
    --------
    >>> import numpy as np
    >>> from nowcastbox.statespace import StateSpace, loglikelihood
    >>> ssm = StateSpace([[0.0]], [[1.0]], [[1.0]], [1.0])
    >>> round(loglikelihood(ssm, np.array([[0.0]])), 6)  # N(0, 2) density at 0
    -1.265512
    """
    return kalman_filter(
        model, observations, method=method, collapse=collapse, store_smoother=False
    ).loglikelihood
