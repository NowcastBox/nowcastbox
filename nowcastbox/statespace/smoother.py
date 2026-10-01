"""Fixed-interval state smoother with lag-one covariances.

Computes ``E[alpha_t | Y_n]``, ``Var[alpha_t | Y_n]`` and the lag-one covariances
``Cov(alpha_{t+1}, alpha_t | Y_n)`` required by the E-step of the EM algorithm
(Shumway & Stoffer, 1982; Bańbura & Modugno, 2014).

The results coincide with the Rauch-Tung-Striebel (1965) smoother
``a_{t|n} = a_{t|t} + J_t (a_{t+1|n} - a_{t+1})``, ``J_t = P_{t|t} T' P_{t+1}^{-1}``, but
are computed with the inversion-free backward recursions of de Jong (1989) and Durbin &
Koopman (2012, sec. 4.4)::

    r_{t-1} = Z_t' F_t^{-1} v_t + L_t' r_t,      a_{t|n} = a_t + P_t r_{t-1}
    N_{t-1} = Z_t' F_t^{-1} Z_t + L_t' N_t L_t,  V_t    = P_t - P_t N_{t-1} P_t

with ``L_t = T (I - P_t Z_t' F_t^{-1} Z_t)``, so singular predicted covariances (e.g.
states that are lags of other states) are handled exactly. The lag-one covariance is
``Cov(alpha_{t+1}, alpha_t | Y_n) = (I - P_{t+1} N_t) T P_{t|t}`` (Durbin & Koopman,
2012, eq. 4.69). The per-period quantities ``Z_t' F_t^{-1} v_t`` and
``Z_t' F_t^{-1} Z_t`` are accumulated by the filter, also in its univariate form, so the
smoother is shared by all filter variants.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from nowcastbox.statespace import _kernels
from nowcastbox.statespace.kalman import (
    FilterMethod,
    FilterResult,
    kalman_filter,
    transition_csr,
)
from nowcastbox.statespace.representation import StateSpace

__all__ = ["SmootherResult", "kalman_smoother", "smooth"]

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SmootherResult:
    """Output of :func:`smooth` / :func:`kalman_smoother`.

    Attributes
    ----------
    filter_result : FilterResult
        The filter output the smoother was run on.
    smoothed_state : numpy.ndarray, shape (n_periods, n_states)
        ``E[alpha_t | Y_n]``.
    smoothed_state_cov : numpy.ndarray, shape (n_periods, n_states, n_states)
        ``Var[alpha_t | Y_n]``.
    smoothed_state_autocov : numpy.ndarray, shape (n_periods, n_states, n_states)
        Row ``t`` holds ``Cov(alpha_{t+1}, alpha_t | Y_n)`` (0-based ``t``); the last row
        involves the out-of-sample state ``alpha_{n+1}``. The EM E-step uses rows
        ``0 .. n-2`` for the pairs ``(alpha_{t}, alpha_{t-1})``, ``t = 2..n``.

    Examples
    --------
    >>> import numpy as np
    >>> from nowcastbox.statespace import StateSpace, kalman_smoother
    >>> ssm = StateSpace([[0.5]], [[1.0]], [[1.0]], [1.0])
    >>> res = kalman_smoother(ssm, np.array([[1.0], [np.nan], [0.5]]))
    >>> res.smoothed_state_cov.shape
    (3, 1, 1)
    """

    filter_result: FilterResult = field(repr=False)
    smoothed_state: FloatArray = field(repr=False)
    smoothed_state_cov: FloatArray = field(repr=False)
    smoothed_state_autocov: FloatArray = field(repr=False)

    @property
    def model(self) -> StateSpace:
        """The state-space model."""
        return self.filter_result.model

    @property
    def loglikelihood(self) -> float:
        """Log-likelihood from the filter pass."""
        return self.filter_result.loglikelihood

    @property
    def n_periods(self) -> int:
        """Number of periods."""
        return int(self.smoothed_state.shape[0])

    def smoothed_signal(self) -> FloatArray:
        """Smoothed observations ``E[Z alpha_t + d | Y_n]`` for every series.

        Returns
        -------
        numpy.ndarray, shape (n_periods, n_obs)
            Smoothed signal (fills missing values with their conditional mean minus noise).

        Examples
        --------
        >>> import numpy as np
        >>> from nowcastbox.statespace import StateSpace, kalman_smoother
        >>> ssm = StateSpace([[0.5]], [[2.0]], [[1.0]], [1.0])
        >>> kalman_smoother(ssm, np.array([[np.nan]])).smoothed_signal()
        array([[0.]])
        """
        n = self.n_periods
        index = self.model.period_index(n)
        z = self.model.designs(n)
        return (
            np.einsum("tij,tj->ti", z, self.smoothed_state)
            + (self.model.obs_intercept_store[index])
        )

    def smoothed_signal_variance(self) -> FloatArray:
        """Variance of the smoothed signal ``Var(Z_t alpha_t | Y_n)`` of every series.

        Same interface as
        :meth:`~nowcastbox.statespace.StructuredSmootherResult.smoothed_signal_variance`.

        Returns
        -------
        numpy.ndarray, shape (n_periods, n_obs)
            Diagonal of ``Z_t V_t Z_t'``.

        Examples
        --------
        >>> import numpy as np
        >>> from nowcastbox.statespace import StateSpace, kalman_smoother
        >>> ssm = StateSpace([[0.0]], [[2.0]], [[1.0]], [1.0])
        >>> kalman_smoother(ssm, np.array([[np.nan]])).smoothed_signal_variance()
        array([[4.]])
        """
        z = self.model.designs(self.n_periods)
        return np.einsum("tij,tjk,tik->ti", z, self.smoothed_state_cov, z)

    def smoothed_signal_cov(self, period: int) -> FloatArray:
        """Covariance ``Z V_t Z'`` of the smoothed signal in one period.

        Parameters
        ----------
        period : int
            0-based period (negative values count from the end).

        Returns
        -------
        numpy.ndarray, shape (n_obs, n_obs)
            Covariance of ``Z alpha_t`` given all observations.

        Raises
        ------
        IndexError
            If ``period`` is out of range.

        Examples
        --------
        >>> import numpy as np
        >>> from nowcastbox.statespace import StateSpace, kalman_smoother
        >>> ssm = StateSpace([[0.0]], [[1.0]], [[1.0]], [1.0])
        >>> kalman_smoother(ssm, np.array([[np.nan]])).smoothed_signal_cov(0)
        array([[1.]])
        """
        if not -self.n_periods <= period < self.n_periods:
            msg = f"period {period} out of range for {self.n_periods} periods"
            raise IndexError(msg)
        z_mat = self.model.design_at(period % self.n_periods)
        return z_mat @ self.smoothed_state_cov[period] @ z_mat.T


def smooth(filter_result: FilterResult) -> SmootherResult:
    """Run the fixed-interval smoother on an existing filter output.

    Parameters
    ----------
    filter_result : FilterResult
        Output of :func:`~nowcastbox.statespace.kalman_filter` computed with
        ``store_smoother=True``.

    Returns
    -------
    SmootherResult
        Smoothed states, covariances and lag-one covariances.

    Raises
    ------
    ValueError
        If the filter was run with ``store_smoother=False``.

    Examples
    --------
    >>> import numpy as np
    >>> from nowcastbox.statespace import StateSpace, kalman_filter, smooth
    >>> ssm = StateSpace([[0.5]], [[1.0]], [[1.0]], [1.0])
    >>> fres = kalman_filter(ssm, np.array([[1.0], [2.0]]))
    >>> sres = smooth(fres)
    >>> bool(np.allclose(sres.smoothed_state[-1], fres.filtered_state[-1]))
    True
    """
    if filter_result.smoother_score is None or filter_result.smoother_information is None:
        msg = "the filter was run with store_smoother=False; rerun it with store_smoother=True"
        raise ValueError(msg)
    a_s, v_s, autocov = _kernels.fixed_interval_smoother(
        filter_result.predicted_state,
        filter_result.predicted_state_cov,
        filter_result.filtered_state_cov,
        filter_result.smoother_score,
        filter_result.smoother_information,
        np.ascontiguousarray(filter_result.model.T),
        *transition_csr(filter_result.model.T),
    )
    return SmootherResult(
        filter_result=filter_result,
        smoothed_state=a_s,
        smoothed_state_cov=v_s,
        smoothed_state_autocov=autocov,
    )


def kalman_smoother(
    model: StateSpace,
    observations: ArrayLike,
    *,
    method: FilterMethod = "auto",
    collapse: bool = False,
) -> SmootherResult:
    """Kalman filter followed by the fixed-interval smoother.

    Parameters
    ----------
    model : StateSpace
        State-space model.
    observations : array_like, shape (n_periods, n_obs)
        Observations with ``NaN`` for missing values.
    method : {"auto", "univariate", "multivariate"}, default "auto"
        Filter algorithm (see :func:`~nowcastbox.statespace.kalman_filter`).
    collapse : bool, default False
        Collapse the observation vector first (Jungbacker & Koopman, 2015).

    Returns
    -------
    SmootherResult
        Smoothed states, covariances, lag-one covariances and the filter output.

    Examples
    --------
    >>> import numpy as np
    >>> from nowcastbox.statespace import StateSpace, kalman_smoother
    >>> ssm = StateSpace([[0.9]], [[1.0], [1.0]], [[1.0]], [1.0, 1.0])
    >>> y = np.array([[1.0, 1.2], [np.nan, 0.8], [np.nan, np.nan]])
    >>> res = kalman_smoother(ssm, y)
    >>> res.smoothed_state.shape, res.smoothed_state_autocov.shape
    ((3, 1), (3, 1, 1))
    """
    return smooth(kalman_filter(model, observations, method=method, collapse=collapse))
