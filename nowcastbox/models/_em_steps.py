r"""Building blocks of the EM algorithm of :class:`~nowcastbox.models.MixedFreqDFM`.

The model (Bańbura & Modugno, 2014; Bańbura, Giannone & Reichlin, 2011) is written on
the base (highest) frequency grid. For a series :math:`i` with aggregation weights
:math:`w_i = (w_{i,0}, \dots, w_{i,L_i-1})` (``w = (1,)`` for series observed at the
base frequency, the Mariano-Murasawa triangle ``(1, 2, 3, 2, 1)`` for quarterly growth
rates on a monthly grid, and any other vector produced by
:func:`nowcastbox.preprocessing.aggregation.aggregation_weights`, innovation I1):

.. math::

    x_{i,t} = \sum_{l=0}^{L_i-1} \lambda_{i,l}' f^{(i)}_{t-l}
              + \sum_{l=0}^{L_i-1} w_{i,l}\, e_{i,t-l} + \varepsilon_{i,t},
    \qquad R_i (\lambda_{i,0}', \dots, \lambda_{i,L_i-1}')' = 0,

where :math:`R_i` encodes :math:`\lambda_{i,l} = (w_{i,l}/w_{i,0})\lambda_{i,0}`
(:func:`~nowcastbox.preprocessing.aggregation.loading_constraints`),
:math:`f^{(i)}_t` stacks the factors of the blocks series :math:`i` loads on, each block
follows an independent VAR(p) :math:`f^b_t = \sum_k A^b_k f^b_{t-k} + u^b_t`, and the
idiosyncratic component is either an AR(1) :math:`e_{i,t} = \rho_i e_{i,t-1} + v_{i,t}`
kept in the state (``"ar1"``, with a small fixed :math:`\varepsilon` variance) or white
noise absorbed in :math:`\varepsilon` (``"iid"``). Optionally (innovation I4, Antolin-Diaz,
Drechsel & Petrella, 2017) a random-walk long-run mean :math:`\mu_t = \mu_{t-1} + \zeta_t`
enters the target (loading fixed at one) and selected series (estimated loadings) as one
extra state appended at the end of the state vector.

State vector (Durbin & Koopman notation, :class:`~nowcastbox.statespace.StateSpace`)::

    alpha_t = (f^1_t, ..., f^1_{t-s_1+1}, ..., f^B_t, ..., f^B_{t-s_B+1},
               e_{1,t}, ..., e_{1,t-L_1+1}, ..., e_{N,t}, ..., e_{N,t-L_N+1})

with :math:`s_b = \max(p, \max_{i \in b} L_i)`. The E-step is the Kalman smoother of
:mod:`nowcastbox.statespace` (univariate treatment, Koopman & Durbin, 2000); the M-step
is in closed form (Shumway & Stoffer, 1982; Watson & Engle, 1983) with restricted least
squares for the loadings of lower-frequency series (Bańbura & Modugno, 2014, eq. 9).

**Calendar-aware aggregation** (innovation I1). When the number of base periods per
native period varies (monthly or quarterly series on a weekly grid, daily data...), the
weights depend on the period, :math:`w_{i,t}`
(:class:`~nowcastbox.preprocessing.aggregation.CalendarAggregation`), and the series
loads on :math:`\sum_l w_{i,t,l}\,\lambda_i' f^{(i)}_{t-l} + \sum_l w_{i,t,l}\,e_{i,t-l}`:
the design matrix is time varying (``StateSpace(obs_index=...)``, one design per
distinct calendar pattern) and only :math:`\lambda_i`, the loading of the latent
base-frequency series, is estimated (the aggregation restrictions are built into the
regressors :math:`\sum_l w_{i,t,l} f_{t-l}`; Bańbura, Giannone & Reichlin, 2011).

**Initial-state term of the M-step** (monotone EM). The first state is drawn from the
stationary distribution implied by the parameters, :math:`\alpha_1 \sim N(0, P_0(\theta))`,
so the expected complete-data log-likelihood contains
:math:`-\tfrac12\{\log|P_0(\theta)| + \operatorname{tr}(P_0(\theta)^{-1}E[\alpha_1\alpha_1'])\}`,
which has no closed-form maximiser. :func:`m_step` computes the classical closed-form
update of each transition block (factor VAR, idiosyncratic AR(1)) and then checks it
against the *exact* expected complete-data log-likelihood of that block, including the
initial-state term; when the closed-form update does not increase it, the step is
shortened by halving towards the previous value (at worst the previous value is kept).
Every M-step therefore increases the exact expected log-likelihood, a generalized EM
(Dempster, Laird & Rubin, 1977; Wu, 1983) whose log-likelihood is monotone.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from functools import partial

import numpy as np
import pandas as pd
import scipy.linalg
from numpy.typing import NDArray

from nowcastbox.core.frequency import Frequency
from nowcastbox.models.long_run import (
    LONG_RUN_INITIAL_VARIANCE,
    long_run_restrictions,
    random_walk_variance,
)
from nowcastbox.preprocessing.aggregation import CalendarAggregation, loading_constraints
from nowcastbox.statespace import (
    SmootherResult,
    StateSpace,
    StructuredSmootherResult,
    kalman_smoother,
    smoothed_moments,
    stationary_initial_cov,
)

__all__ = [
    "EMParameters",
    "StateLayout",
    "SufficientStatistics",
    "build_state_space",
    "e_step",
    "em_converged",
    "m_step",
    "restricted_least_squares",
    "signal_mean",
    "signal_variance",
    "smoother_with_obs_var",
    "transition_matrix",
    "trim_weights",
]

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

VARIANCE_FLOOR = 1e-6
"""Lower bound of every estimated variance (keeps the model well defined)."""

_STRUCTURED_METHODS = ("auto", "structured")
"""E-step methods routed through :func:`~nowcastbox.statespace.smoothed_moments`."""

MAX_AR_COEFFICIENT = 0.995
"""Idiosyncratic AR(1) coefficients are clipped to ``[-0.995, 0.995]``."""

_NONSTATIONARY_INITIAL_VARIANCE = 10.0
_IDIOSYNCRATIC_KINDS = ("ar1", "iid")
_MAX_STEP_HALVINGS = 30


def trim_weights(weights: Sequence[float] | FloatArray) -> FloatArray:
    """Drop trailing zero aggregation weights (keeps at least the first weight).

    Parameters
    ----------
    weights : sequence of float
        Aggregation weights, most recent period first (non-zero first element).

    Returns
    -------
    numpy.ndarray
        Weights without trailing zeros.

    Raises
    ------
    ValueError
        If the weights are empty, not finite or start with zero.

    Examples
    --------
    >>> from nowcastbox.models._em_steps import trim_weights
    >>> trim_weights([1.0, 0.0, 0.0]).tolist()
    [1.0]
    >>> trim_weights([1, 2, 3, 2, 1]).tolist()
    [1.0, 2.0, 3.0, 2.0, 1.0]
    """
    w = np.asarray(weights, dtype=np.float64).ravel()
    if w.size == 0 or not bool(np.all(np.isfinite(w))) or w[0] == 0.0:
        msg = f"aggregation weights must be finite with a non-zero first element, got {w}"
        raise ValueError(msg)
    last = int(np.flatnonzero(w)[-1])
    return np.ascontiguousarray(w[: last + 1])


class StateLayout:
    """Position of every factor and idiosyncratic component in the state vector.

    Parameters
    ----------
    series : sequence of str
        Series names (rows of the design matrix), length ``N``.
    block_names : sequence of str
        Factor block names, length ``B``.
    n_factors : sequence of int
        Number of factors of each block.
    factor_lags : int
        VAR order ``p`` of the factors.
    membership : array_like of bool, shape (N, B)
        ``membership[i, b]`` is True when series ``i`` loads on block ``b``. Every series
        must load on at least one block and every block must have a series.
    weights : sequence of array_like
        Aggregation weights of each series (``[1.0]`` for base-frequency series).
    idiosyncratic : {"ar1", "iid"}
        Idiosyncratic specification.
    block_prefix : bool, default True
        Name factors ``"<block>_f<k>"`` (``"f<k>"`` when False).
    long_run : sequence of int, default ()
        Series loading on a random-walk long-run mean state (innovation I4). The first
        one (the target) has its loading fixed at one (normalisation); the others have
        estimated loadings. Empty: no long-run state.
    long_run_variance : float, optional
        Fixed variance of the long-run mean innovations (standardised units); ``None``
        estimates it.
    calendar : sequence of CalendarAggregation or None, optional
        Per series, a calendar-aware aggregation (innovation I1) for pairs of
        frequencies whose ratio varies (``None`` for the others). The ``weights`` of
        such a series are replaced by its
        :attr:`~nowcastbox.preprocessing.aggregation.CalendarAggregation.reference_weights`
        (the longest weight vector, which sets the number of lags in the state) and the
        actual weights of every period come from :meth:`weight_path`.
    start : pandas.Period, optional
        First period of the base grid (required with calendar aggregations).

    Attributes
    ----------
    block_lags : tuple of int
        Number of lags ``s_b`` of each block kept in the state.
    n_factor_states : int
        Size of the factor part of the state.
    n_states : int
        Total state dimension.
    n_shocks : int
        Number of state disturbances (factor shocks plus idiosyncratic shocks plus the
        long-run mean shock).
    trend_index : int
        State index of the long-run mean (``-1`` without one).
    calendar : tuple of CalendarAggregation or None
        Calendar-aware aggregation of each series (``None``: fixed weights).

    Raises
    ------
    ValueError
        On inconsistent dimensions or memberships.

    Examples
    --------
    >>> import numpy as np
    >>> from nowcastbox.models._em_steps import StateLayout
    >>> lay = StateLayout(
    ...     ["ip", "gdp"],
    ...     ["global"],
    ...     [1],
    ...     1,
    ...     np.ones((2, 1), bool),
    ...     [[1.0], [1, 2, 3, 2, 1]],
    ...     "ar1",
    ... )
    >>> lay.block_lags, lay.n_factor_states, lay.n_states
    ((5,), 5, 11)
    >>> lay.loading_index(1).tolist()
    [0, 1, 2, 3, 4]
    """

    def __init__(
        self,
        series: Sequence[str],
        block_names: Sequence[str],
        n_factors: Sequence[int],
        factor_lags: int,
        membership: NDArray[np.bool_] | Sequence[Sequence[bool]],
        weights: Sequence[Sequence[float] | FloatArray],
        idiosyncratic: str,
        *,
        block_prefix: bool = True,
        long_run: Sequence[int] = (),
        long_run_variance: float | None = None,
        calendar: Sequence[CalendarAggregation | None] | None = None,
        start: pd.Period | None = None,
    ) -> None:
        self.series = tuple(str(s) for s in series)
        self.block_names = tuple(str(b) for b in block_names)
        self.n_factors = tuple(int(r) for r in n_factors)
        self.factor_lags = int(factor_lags)
        self.membership = np.array(membership, dtype=bool, copy=True).reshape(
            len(self.series), len(self.block_names)
        )
        specs = [None] * len(self.series) if calendar is None else list(calendar)
        if len(specs) != len(self.series):
            raise ValueError("calendar must give one entry per series")
        self.calendar: tuple[CalendarAggregation | None, ...] = tuple(
            None if c is None or c.is_fixed else c for c in specs
        )
        self.start = None if start is None else pd.Period(start)
        if len(weights) != len(self.series):
            raise ValueError("weights must give one vector per series")
        self.weights = tuple(
            trim_weights(w if c is None else c.reference_weights)
            for w, c in zip(weights, self.calendar, strict=True)
        )
        self._paths: dict[tuple[int, int], FloatArray] = {}
        self.idiosyncratic = str(idiosyncratic)
        self.block_prefix = bool(block_prefix)
        self.long_run = tuple(int(i) for i in long_run)
        self.long_run_variance = None if long_run_variance is None else float(long_run_variance)
        self._validate()
        n_series = len(self.series)
        lengths = np.array([w.size for w in self.weights])
        self.block_lags = tuple(
            max(self.factor_lags, int(lengths[self.membership[:, b]].max()))
            for b in range(self.n_blocks)
        )
        sizes = [r * s for r, s in zip(self.n_factors, self.block_lags, strict=True)]
        self.block_offsets = tuple(int(o) for o in np.concatenate([[0], np.cumsum(sizes)[:-1]]))
        self.n_factor_states = int(sum(sizes))
        if self.idiosyncratic == "ar1":
            offsets = self.n_factor_states + np.concatenate([[0], np.cumsum(lengths)[:-1]])
            self.idio_offsets: IntArray = offsets.astype(np.int64)
            self.n_states = self.n_factor_states + int(lengths.sum())
            self.n_shocks = self.total_factors + n_series
        else:
            self.idio_offsets = np.full(n_series, -1, dtype=np.int64)
            self.n_states = self.n_factor_states
            self.n_shocks = self.total_factors
        self.trend_index = -1
        if self.long_run:
            self.trend_index = self.n_states
            self.n_states += 1
            self.n_shocks += 1

    def _validate(self) -> None:
        if len(set(self.series)) != len(self.series) or not self.series:
            raise ValueError("series names must be unique and non-empty")
        if len(self.n_factors) != len(self.block_names) or not self.block_names:
            raise ValueError("n_factors must give one positive integer per block")
        if min(self.n_factors) < 1 or self.factor_lags < 1:
            raise ValueError("n_factors and factor_lags must be positive integers")
        if self.idiosyncratic not in _IDIOSYNCRATIC_KINDS:
            raise ValueError(f"idiosyncratic must be one of {_IDIOSYNCRATIC_KINDS}")
        if not self.membership.any(axis=1).all():
            raise ValueError("every series must load on at least one block")
        if not self.membership.any(axis=0).all():
            raise ValueError("every block must contain at least one series")
        valid = all(0 <= i < len(self.series) for i in self.long_run)
        if not valid or len(set(self.long_run)) != len(self.long_run):
            raise ValueError("long_run must hold distinct series positions")
        if self.long_run_variance is not None and not self.long_run_variance > 0.0:
            raise ValueError("long_run_variance must be positive")
        self._validate_calendar()

    def _validate_calendar(self) -> None:
        highs = {c.high for c in self.calendar if c is not None}
        if highs and (len(highs) > 1 or self.start is None):
            raise ValueError("calendar aggregations need a common base frequency and a start")
        start = self.start
        if highs and start is not None and Frequency.from_value(start.freqstr) not in highs:
            raise ValueError("start must be a period of the base frequency of the calendar")

    # ------------------------------------------------------------------ sizes
    @property
    def n_series(self) -> int:
        """Number of observed series ``N``."""
        return len(self.series)

    @property
    def n_blocks(self) -> int:
        """Number of factor blocks ``B``."""
        return len(self.block_names)

    @property
    def total_factors(self) -> int:
        """Total number of factors ``r = sum_b r_b``."""
        return int(sum(self.n_factors))

    @property
    def has_long_run(self) -> bool:
        """Whether the state contains a long-run mean (innovation I4)."""
        return self.trend_index >= 0

    @property
    def is_time_varying(self) -> bool:
        """Whether some series has calendar-dependent weights (time-varying design)."""
        return any(c is not None for c in self.calendar)

    def weight_path(self, series: int, n_periods: int) -> FloatArray:
        """Aggregation weights of a series in every period of the base grid.

        Parameters
        ----------
        series : int
            Series position.
        n_periods : int
            Number of periods from :attr:`start`.

        Returns
        -------
        numpy.ndarray, shape (n_periods, L_i)
            Row ``t``: weights on ``(x_t, x_{t-1}, ...)`` (constant rows for series with
            fixed weights; calendar weights, truncated at ``t`` outside the storage
            slots, for calendar aggregations).
        """
        spec = self.calendar[series]
        width = self.weights[series].size
        if spec is None:
            return np.broadcast_to(self.weights[series], (int(n_periods), width))
        key = (series, int(n_periods))
        path = self._paths.get(key)
        if path is None:
            assert self.start is not None  # noqa: S101 - checked in _validate
            index = pd.period_range(self.start, periods=int(n_periods), freq=self.start.freq)
            path = np.ascontiguousarray(spec.weight_matrix(index)[:, :width])
            path.setflags(write=False)
            self._paths[key] = path
        return path

    @property
    def factor_names(self) -> list[str]:
        """Factor labels (``"<block>_f<k>"`` or ``"f<k>"``), in state order."""
        names: list[str] = []
        for block, r in zip(self.block_names, self.n_factors, strict=True):
            prefix = f"{block}_" if self.block_prefix else ""
            names += [f"{prefix}f{k + 1}" for k in range(r)]
        return names

    # ------------------------------------------------------------------ indices
    def factor_index(self, block: int, lag: int = 0) -> IntArray:
        """State indices of ``f^b_{t-lag}``.

        Parameters
        ----------
        block : int
            Block position.
        lag : int, default 0
            Lag (``0 <= lag < block_lags[block]``).

        Returns
        -------
        numpy.ndarray of int
            ``n_factors[block]`` indices.
        """
        r = self.n_factors[block]
        start = self.block_offsets[block] + lag * r
        return np.arange(start, start + r, dtype=np.int64)

    def block_index(self, block: int) -> IntArray:
        """All state indices of block ``block`` (every lag)."""
        start = self.block_offsets[block]
        return np.arange(start, start + self.n_factors[block] * self.block_lags[block])

    @property
    def current_factor_index(self) -> IntArray:
        """State indices of the current factors ``f_t`` of every block."""
        return np.concatenate([self.factor_index(b, 0) for b in range(self.n_blocks)])

    def series_blocks(self, series: int) -> list[int]:
        """Positions of the blocks series ``series`` loads on."""
        return [int(b) for b in np.flatnonzero(self.membership[series])]

    def loading_index(self, series: int) -> IntArray:
        """State indices of the loadings of a series, lag-major across its blocks.

        The order ``(f^{(i)}_t, f^{(i)}_{t-1}, ...)`` matches the layout expected by
        :func:`~nowcastbox.preprocessing.aggregation.loading_constraints`.

        Parameters
        ----------
        series : int
            Series position.

        Returns
        -------
        numpy.ndarray of int
            ``L_i * r_i`` indices.
        """
        blocks = self.series_blocks(series)
        parts = [
            self.factor_index(b, lag) for lag in range(self.weights[series].size) for b in blocks
        ]
        return np.concatenate(parts)

    def idio_index(self, series: int) -> IntArray:
        """State indices of the idiosyncratic segment of a series (empty for ``"iid"``)."""
        start = int(self.idio_offsets[series])
        if start < 0:
            return np.zeros(0, dtype=np.int64)
        return np.arange(start, start + self.weights[series].size, dtype=np.int64)

    def regressor_index(self, series: int) -> IntArray:
        """State indices regressed on in the M-step of a series.

        :meth:`loading_index` followed by the long-run mean state when the series loads
        on it.
        """
        idx = self.loading_index(series)
        if series in self.long_run:
            idx = np.append(idx, self.trend_index)
        return idx

    def n_series_factors(self, series: int) -> int:
        """Number of factors ``r_i`` a series loads on (sum over its blocks)."""
        return int(sum(self.n_factors[b] for b in self.series_blocks(series)))

    def factor_constraints(self, series: int) -> tuple[FloatArray, FloatArray]:
        """Aggregation restrictions ``R lambda = q`` on the factor loadings of a series.

        For a calendar aggregation the coefficients are the ``r_i`` loadings of the
        latent base-frequency series (the weights enter the regressors): no restriction.
        """
        r_i = self.n_series_factors(series)
        if self.calendar[series] is not None:
            return np.zeros((0, r_i)), np.zeros(0)
        return loading_constraints(self.weights[series], r_i)

    def constraints(self, series: int) -> tuple[FloatArray, FloatArray]:
        """Restrictions ``R beta = q`` on the coefficients of :meth:`regressor_index`.

        Aggregation restrictions on the factor loadings (Bańbura & Modugno, 2014) and,
        for the first :attr:`long_run` series, the unit loading on the long-run mean.
        """
        R, q = self.factor_constraints(series)
        if series not in self.long_run:
            return R, q
        return long_run_restrictions(R, q, fixed=series == self.long_run[0])

    def is_compatible(self, other: StateLayout) -> bool:
        """Whether ``other`` describes the same model structure."""
        return (
            self.series == other.series
            and self.block_names == other.block_names
            and self.n_factors == other.n_factors
            and self.factor_lags == other.factor_lags
            and self.idiosyncratic == other.idiosyncratic
            and self.long_run == other.long_run
            and bool(np.array_equal(self.membership, other.membership))
            and all(np.array_equal(a, b) for a, b in zip(self.weights, other.weights, strict=True))
            and self.calendar == other.calendar
            and (not self.is_time_varying or self.start == other.start)
        )

    def __repr__(self) -> str:
        return (
            f"StateLayout(n_series={self.n_series}, blocks={self.block_names}, "
            f"n_factors={self.n_factors}, factor_lags={self.factor_lags}, "
            f"idiosyncratic={self.idiosyncratic!r}, n_states={self.n_states})"
        )


@dataclass(frozen=True, eq=False)
class EMParameters:
    r"""Parameters of the mixed-frequency dynamic factor model.

    Parameters
    ----------
    transition : tuple of numpy.ndarray
        Per block, the stacked VAR matrices ``[A_1, ..., A_p]`` of shape ``(r_b, r_b p)``.
    factor_cov : tuple of numpy.ndarray
        Per block, the factor innovation covariance ``(r_b, r_b)``.
    loadings : numpy.ndarray, shape (N, n_factor_states)
        Factor part of the design matrix (loadings on every factor lag in the state).
        For a series with a calendar aggregation (time-varying design) only the
        loadings on the current factors are stored: the loadings :math:`\lambda_i` of
        the latent base-frequency series (the design row of period ``t`` is
        :math:`w_{i,t} \otimes \lambda_i`).
    idio_ar : numpy.ndarray, shape (N,)
        Idiosyncratic AR(1) coefficients (zeros for ``"iid"``).
    idio_var : numpy.ndarray, shape (N,)
        Idiosyncratic innovation variances (AR(1) innovations or white-noise variances).
    obs_var : numpy.ndarray, shape (N,)
        Diagonal of the measurement-noise covariance ``H``.
    trend_loadings : numpy.ndarray, shape (N,), optional
        Loadings on the long-run mean state (layouts with ``long_run`` only).
    trend_var : float, default 0.0
        Variance of the long-run mean innovations.

    Examples
    --------
    >>> import numpy as np
    >>> from nowcastbox.models._em_steps import EMParameters
    >>> p = EMParameters(
    ...     (np.array([[0.5]]),), (np.eye(1),), np.ones((2, 1)), np.zeros(2), np.ones(2), np.ones(2)
    ... )
    >>> p.loadings.shape
    (2, 1)
    """

    transition: tuple[FloatArray, ...]
    factor_cov: tuple[FloatArray, ...]
    loadings: FloatArray
    idio_ar: FloatArray
    idio_var: FloatArray
    obs_var: FloatArray
    trend_loadings: FloatArray | None = None
    trend_var: float = 0.0

    def copy(self) -> EMParameters:
        """Deep copy of the parameters."""
        return EMParameters(
            tuple(a.copy() for a in self.transition),
            tuple(q.copy() for q in self.factor_cov),
            self.loadings.copy(),
            self.idio_ar.copy(),
            self.idio_var.copy(),
            self.obs_var.copy(),
            None if self.trend_loadings is None else self.trend_loadings.copy(),
            float(self.trend_var),
        )

    def max_abs_difference(self, other: EMParameters) -> float:
        """Largest absolute difference between two parameter sets."""
        pairs = [
            *zip(self.transition, other.transition, strict=True),
            *zip(self.factor_cov, other.factor_cov, strict=True),
            (self.loadings, other.loadings),
            (self.idio_ar, other.idio_ar),
            (self.idio_var, other.idio_var),
            (self.obs_var, other.obs_var),
            (np.atleast_1d(self.trend_var), np.atleast_1d(other.trend_var)),
        ]
        if self.trend_loadings is not None and other.trend_loadings is not None:
            pairs.append((self.trend_loadings, other.trend_loadings))
        return float(max(np.max(np.abs(a - b)) for a, b in pairs))


# ---------------------------------------------------------------------- state space
def _block_companion(transition: FloatArray, n_lags: int) -> FloatArray:
    r, rp = transition.shape
    comp = np.zeros((r * n_lags, r * n_lags))
    comp[:r, :rp] = transition
    if n_lags > 1:
        comp[r:, : r * (n_lags - 1)] = np.eye(r * (n_lags - 1))
    return comp


def _block_initial_cov(companion: FloatArray, shock_cov: FloatArray) -> FloatArray:
    r = shock_cov.shape[0]
    full = np.zeros_like(companion)
    full[:r, :r] = shock_cov
    try:
        return stationary_initial_cov(companion, full)
    except ValueError:
        # non-stationary VAR: proper but uninformative prior (diffuse is not supported)
        return _NONSTATIONARY_INITIAL_VARIANCE * np.eye(companion.shape[0])


def _ar1_segment_cov(rho: float, variance: float, length: int) -> FloatArray:
    lags = np.abs(np.subtract.outer(np.arange(length), np.arange(length)))
    return variance / (1.0 - rho**2) * rho**lags


@dataclass(frozen=True, eq=False)
class _StateEquation:
    """Time-invariant part of the model: ``T``, ``R``, ``Q`` and ``P0``."""

    T: FloatArray
    R: FloatArray
    Q: FloatArray
    P0: FloatArray


def _state_equation(params: EMParameters, layout: StateLayout) -> _StateEquation:
    m, g = layout.n_states, layout.n_shocks
    T = np.zeros((m, m))
    R = np.zeros((m, g))
    Q = np.zeros((g, g))
    P0 = np.zeros((m, m))
    shock = 0
    for b in range(layout.n_blocks):
        idx = layout.block_index(b)
        r = layout.n_factors[b]
        comp = _block_companion(params.transition[b], layout.block_lags[b])
        T[np.ix_(idx, idx)] = comp
        R[layout.factor_index(b, 0), shock + np.arange(r)] = 1.0
        Q[shock : shock + r, shock : shock + r] = params.factor_cov[b]
        P0[np.ix_(idx, idx)] = _block_initial_cov(comp, params.factor_cov[b])
        shock += r
    for i in range(layout.n_series):
        idx = layout.idio_index(i)
        if idx.size == 0:
            continue
        rho, var = float(params.idio_ar[i]), float(params.idio_var[i])
        T[idx[0], idx[0]] = rho
        T[idx[1:], idx[:-1]] = 1.0
        R[idx[0], shock] = 1.0
        Q[shock, shock] = var
        P0[np.ix_(idx, idx)] = _ar1_segment_cov(rho, var, idx.size)
        shock += 1
    if layout.has_long_run:
        k = layout.trend_index
        T[k, k] = 1.0
        R[k, shock] = 1.0
        Q[shock, shock] = params.trend_var
        P0[k, k] = LONG_RUN_INITIAL_VARIANCE
    return _StateEquation(T, R, Q, P0)


def transition_matrix(params: EMParameters, layout: StateLayout) -> FloatArray:
    """Transition matrix ``T`` of the model (no observation equation needed).

    Parameters
    ----------
    params : EMParameters
        Model parameters.
    layout : StateLayout
        State layout.

    Returns
    -------
    numpy.ndarray, shape (n_states, n_states)
        Block-diagonal companion / AR(1) / random-walk transition.

    Examples
    --------
    >>> import numpy as np
    >>> from nowcastbox.models._em_steps import EMParameters, StateLayout, transition_matrix
    >>> lay = StateLayout(["a", "b"], ["g"], [1], 1, np.ones((2, 1), bool), [[1.0], [1.0]], "iid")
    >>> p = EMParameters(
    ...     (np.array([[0.5]]),), (np.eye(1),), np.ones((2, 1)), np.zeros(2), np.ones(2), np.ones(2)
    ... )
    >>> transition_matrix(p, lay).tolist()
    [[0.5]]
    """
    return _state_equation(params, layout).T


def _base_design(params: EMParameters, layout: StateLayout) -> FloatArray:
    """Design matrix of the series with fixed weights (calendar rows left empty)."""
    Z = np.zeros((layout.n_series, layout.n_states))
    Z[:, : layout.n_factor_states] = params.loadings
    for i in range(layout.n_series):
        if layout.calendar[i] is not None:
            Z[i, : layout.n_factor_states] = 0.0
            continue
        idx = layout.idio_index(i)
        if idx.size:
            Z[i, idx] = layout.weights[i]
    if layout.has_long_run:
        if params.trend_loadings is None:
            raise ValueError("the layout has a long-run mean but trend_loadings is None")
        Z[:, layout.trend_index] = params.trend_loadings
    return Z


def _calendar_designs(
    params: EMParameters, layout: StateLayout, base: FloatArray, n_periods: int
) -> tuple[FloatArray, NDArray[np.int64]]:
    """Store of distinct design matrices and the per-period index (calendar layouts)."""
    series = [i for i in range(layout.n_series) if layout.calendar[i] is not None]
    codes = np.zeros((n_periods, len(series)), dtype=np.int64)
    patterns: list[FloatArray] = []
    for j, i in enumerate(series):
        unique, inverse = np.unique(layout.weight_path(i, n_periods), axis=0, return_inverse=True)
        codes[:, j] = inverse.ravel()
        patterns.append(unique)
    combos, obs_index = np.unique(codes, axis=0, return_inverse=True)
    store = np.repeat(base[None], combos.shape[0], axis=0)
    for j, i in enumerate(series):
        load = layout.loading_index(i)
        lam = params.loadings[i, load[: layout.n_series_factors(i)]]
        idio = layout.idio_index(i)
        for k, code in enumerate(combos[:, j]):
            w = patterns[j][code]
            store[k, i, load] = np.kron(w, lam)
            if idio.size:
                store[k, i, idio] = w
    return store, np.ascontiguousarray(obs_index.ravel(), dtype=np.int64)


def build_state_space(
    params: EMParameters, layout: StateLayout, n_periods: int | None = None
) -> StateSpace:
    """Assemble the :class:`~nowcastbox.statespace.StateSpace` of the model.

    The first state ``alpha_1`` is initialised at its stationary distribution under
    ``params`` (zero mean; a ``10 I`` covariance for a non-stationary factor block and
    for the random-walk long-run mean).

    Parameters
    ----------
    params : EMParameters
        Model parameters.
    layout : StateLayout
        State layout.
    n_periods : int, optional
        Number of periods of the base grid (from ``layout.start``). Required for
        layouts with calendar aggregations, whose observation equation is time varying
        (``obs_index``: one design per distinct calendar pattern); ignored otherwise.

    Returns
    -------
    StateSpace
        Model with ``N`` observations, ``layout.n_states`` states and
        ``layout.n_shocks`` disturbances.

    Raises
    ------
    ValueError
        If the layout has a long-run mean but ``params.trend_loadings`` is missing, or
        a calendar layout is built without ``n_periods``.

    Examples
    --------
    >>> import numpy as np
    >>> from nowcastbox.models._em_steps import EMParameters, StateLayout, build_state_space
    >>> lay = StateLayout(["a", "b"], ["g"], [1], 1, np.ones((2, 1), bool), [[1.0], [1.0]], "iid")
    >>> p = EMParameters(
    ...     (np.array([[0.5]]),), (np.eye(1),), np.ones((2, 1)), np.zeros(2), np.ones(2), np.ones(2)
    ... )
    >>> build_state_space(p, lay).n_states
    1
    """
    state = _state_equation(params, layout)
    Z = _base_design(params, layout)
    common = {
        "selection": state.R,
        "initial_state": np.zeros(layout.n_states),
        "initial_state_cov": state.P0,
    }
    if not layout.is_time_varying:
        return StateSpace(state.T, Z, state.Q, params.obs_var, **common)
    if n_periods is None or int(n_periods) < 1:
        raise ValueError("layouts with calendar aggregations need n_periods (time-varying Z)")
    store, obs_index = _calendar_designs(params, layout, Z, int(n_periods))
    obs_cov = np.repeat(params.obs_var[None], store.shape[0], axis=0)
    return StateSpace(state.T, store, state.Q, obs_cov, obs_index=obs_index, **common)


def signal_mean(model: StateSpace, states: FloatArray) -> FloatArray:
    """``Z_t a_t`` for every period (time-invariant or time-varying design).

    Parameters
    ----------
    model : StateSpace
        Model (with ``obs_index`` its number of periods must match ``states``).
    states : numpy.ndarray, shape (n_periods, n_states)
        State vectors.

    Returns
    -------
    numpy.ndarray, shape (n_periods, n_obs)
        Signals (without the intercept ``d``).

    Examples
    --------
    >>> import numpy as np
    >>> from nowcastbox.statespace import StateSpace
    >>> from nowcastbox.models._em_steps import signal_mean
    >>> tv = StateSpace([[0.5]], [[[1.0]], [[2.0]]], [[1.0]], [[1.0], [1.0]], obs_index=[0, 1])
    >>> signal_mean(tv, np.ones((2, 1))).ravel().tolist()
    [1.0, 2.0]
    """
    if not model.is_time_varying:
        return states @ model.Z.T
    return np.einsum("tij,tj->ti", model.designs(states.shape[0]), states)


def signal_variance(model: StateSpace, covariances: FloatArray) -> FloatArray:
    """``diag(Z_t P_t Z_t')`` for every period (time-invariant or time-varying design).

    Parameters
    ----------
    model : StateSpace
        Model (with ``obs_index`` its number of periods must match ``covariances``).
    covariances : numpy.ndarray, shape (n_periods, n_states, n_states)
        State covariance matrices.

    Returns
    -------
    numpy.ndarray, shape (n_periods, n_obs)
        Signal variances.

    Examples
    --------
    >>> import numpy as np
    >>> from nowcastbox.statespace import StateSpace
    >>> from nowcastbox.models._em_steps import signal_variance
    >>> ssm = StateSpace([[0.5]], [[2.0]], [[1.0]], [1.0])
    >>> signal_variance(ssm, np.ones((3, 1, 1))).ravel().tolist()
    [4.0, 4.0, 4.0]
    """
    if not model.is_time_varying:
        Z = model.Z
        return np.einsum("ij,tjk,ik->ti", Z, covariances, Z, optimize=True)
    Zs = model.designs(covariances.shape[0])
    return np.einsum("tij,tjk,tik->ti", Zs, covariances, Zs, optimize=True)


# ---------------------------------------------------------------------- E-step
@dataclass(frozen=True, eq=False)
class SufficientStatistics:
    """Smoothed moments used by the M-step.

    Attributes
    ----------
    smoother : SmootherResult or StructuredSmootherResult
        Smoother output (``E[alpha_t | Y]``, ``Var[alpha_t | Y]``, lag-one covariances).
        The exact structured smoother (:func:`~nowcastbox.statespace.structured_smoother`)
        does not compute covariances between two different private (idiosyncratic)
        groups; the M-step never uses them.
    s11 : numpy.ndarray, shape (m, m)
        ``sum_{t=2}^n E[alpha_t alpha_t' | Y]``.
    s00 : numpy.ndarray, shape (m, m)
        ``sum_{t=2}^n E[alpha_{t-1} alpha_{t-1}' | Y]``.
    s10 : numpy.ndarray, shape (m, m)
        ``sum_{t=2}^n E[alpha_t alpha_{t-1}' | Y]``.
    n_pairs : int
        Number of transitions ``n - 1``.
    """

    smoother: SmootherResult | StructuredSmootherResult
    s11: FloatArray = field(repr=False)
    s00: FloatArray = field(repr=False)
    s10: FloatArray = field(repr=False)
    n_pairs: int

    @property
    def loglikelihood(self) -> float:
        """Log-likelihood of the data under the parameters of the E-step."""
        return self.smoother.loglikelihood


def smoother_with_obs_var(
    model: StateSpace, observations: FloatArray, obs_var: FloatArray
) -> SmootherResult:
    r"""Kalman smoother with a time-varying diagonal measurement variance.

    Uses the public time-varying observation equation of
    :class:`~nowcastbox.statespace.StateSpace` (one observation system per period,
    ``obs_index``) and the univariate treatment (Koopman & Durbin, 2000). Needed by the
    Student-t E-step, where the variance of :math:`\varepsilon_{i,t}` is
    :math:`\psi_i / E[\lambda_{i,t}]` (scale mixture of normals). ``model.H`` is
    ignored (its diagonal is replaced by ``obs_var``); ``model`` may itself have a
    time-varying design (calendar aggregations) for ``n_periods`` periods.

    Parameters
    ----------
    model : StateSpace
        Model (its ``Z_t``, ``d_t``, ``T``, ``R Q R'`` and initial conditions are used).
    observations : numpy.ndarray, shape (n_periods, N)
        Observations (``NaN`` = missing).
    obs_var : numpy.ndarray, shape (n_periods, N)
        Measurement variance of every observation (non-negative).

    Returns
    -------
    SmootherResult
        Smoother output; its ``filter_result.model`` is ``model`` (so
        ``smoothed_signal`` and the measurement scale refer to the base model).

    Raises
    ------
    ValueError
        If ``obs_var`` has the wrong shape or negative / non-finite entries.

    Examples
    --------
    >>> import numpy as np
    >>> from nowcastbox.statespace import StateSpace, kalman_smoother
    >>> from nowcastbox.models._em_steps import smoother_with_obs_var
    >>> ssm = StateSpace([[0.5]], [[1.0]], [[1.0]], [2.0])
    >>> y = np.array([[1.0], [0.5], [-0.3]])
    >>> a = smoother_with_obs_var(ssm, y, np.full((3, 1), 2.0)).smoothed_state
    >>> bool(np.allclose(a, kalman_smoother(ssm, y).smoothed_state))
    True
    """
    y = np.asarray(observations, dtype=np.float64)
    n, n_obs = y.shape
    h = np.ascontiguousarray(obs_var, dtype=np.float64)
    if h.shape != (n, n_obs) or not bool(np.all(np.isfinite(h))) or bool(np.any(h < 0)):
        raise ValueError(f"obs_var must be a finite non-negative array of shape {(n, n_obs)}")
    index = model.period_index(n)
    varying = StateSpace(
        model.T,
        np.ascontiguousarray(model.designs(n)),
        model.Q,
        h,
        selection=model.R,
        state_intercept=model.c,
        obs_intercept=model.obs_intercept_store[index],
        initial_state=model.a0,
        initial_state_cov=model.P0,
        obs_index=np.arange(n),
    )
    sm = kalman_smoother(varying, y, method="univariate")
    # downstream code reads Z, d and the scale H from the (time-invariant) base model
    return replace(sm, filter_result=replace(sm.filter_result, model=model))


def _profile_free_transitions(
    a: FloatArray,
    s11: FloatArray,
    s00: FloatArray,
    s10: FloatArray,
    free_periods: NDArray[np.bool_],
    free_states: IntArray,
) -> None:
    r"""Drop the mean part of the moments of transitions with free intercepts (in place).

    With an unrestricted intercept :math:`D_t` in the transition into period ``t``, its
    ML value removes the mean of the residual, so only the conditional covariances of
    :math:`(\alpha_t, \alpha_{t-1})` contribute to the M-step of ``T`` and ``Q``.
    """
    periods = np.flatnonzero(free_periods[1:]) + 1
    if periods.size == 0:
        return
    ix = np.ix_(free_states, free_states)
    cur, prev = a[np.ix_(periods, free_states)], a[np.ix_(periods - 1, free_states)]
    s11[ix] -= cur.T @ cur
    s00[ix] -= prev.T @ prev
    s10[ix] -= cur.T @ prev


def e_step(
    model: StateSpace,
    observations: FloatArray,
    *,
    method: str = "univariate",
    obs_var: FloatArray | None = None,
    state_offset: FloatArray | None = None,
    free_periods: NDArray[np.bool_] | None = None,
    free_states: IntArray | None = None,
) -> SufficientStatistics:
    r"""Run the Kalman smoother and accumulate the smoothed second moments.

    Parameters
    ----------
    model : StateSpace
        Model at the current parameters.
    observations : numpy.ndarray, shape (n_periods, N)
        Standardised data (``NaN`` = missing).
    method : {"univariate", "multivariate", "auto", "structured"}, default "univariate"
        ``"univariate"``/``"multivariate"``: dense Kalman smoother with that filter
        (:func:`~nowcastbox.statespace.kalman_filter`). ``"auto"``: the exact structured
        smoother (:func:`~nowcastbox.statespace.smoothed_moments`, innovation I2) when the
        model has the structure and it is estimated to be cheaper, the dense smoother
        otherwise. ``"structured"`` forces the structured smoother. The structured path
        is only used without ``obs_var``, ``state_offset`` and ``free_periods`` (the
        robust E-steps always use the dense univariate treatment).
    obs_var : numpy.ndarray, shape (n_periods, N), optional
        Time-varying measurement variances (Student-t E-step, see
        :func:`smoother_with_obs_var`).
    state_offset : numpy.ndarray, shape (n_periods, m), optional
        Deterministic part :math:`g_t` of the state (intervention dummies):
        :math:`\alpha_t = \tilde\alpha_t + g_t`. The smoother runs on
        :math:`y_t - Z g_t` and ``g`` is added back to the smoothed states.
    free_periods : numpy.ndarray of bool, shape (n_periods,), optional
        Periods whose transition from the previous period has an unrestricted intercept
        on ``free_states``; the mean part of their moments is profiled out.
    free_states : numpy.ndarray of int, optional
        States with free intercepts (required with ``free_periods``).

    Returns
    -------
    SufficientStatistics
        Smoother output (smoothed states include ``state_offset``) and moment sums.

    Raises
    ------
    ValueError
        If there are fewer than two periods.

    Examples
    --------
    >>> import numpy as np
    >>> from nowcastbox.statespace import StateSpace
    >>> from nowcastbox.models._em_steps import e_step
    >>> ssm = StateSpace([[0.5]], [[1.0]], [[1.0]], [1.0])
    >>> e_step(ssm, np.array([[1.0], [np.nan], [0.2]])).n_pairs
    2
    """
    n = observations.shape[0]
    if n < 2:
        raise ValueError("the EM algorithm needs at least two periods")
    plain = obs_var is None and state_offset is None and free_periods is None
    if method in _STRUCTURED_METHODS:
        if plain:
            mom = smoothed_moments(model, observations, method=method)  # type: ignore[arg-type]
            return SufficientStatistics(mom.smoother, mom.s11, mom.s00, mom.s10, mom.n_pairs)
        method = "univariate"
    y = observations if state_offset is None else observations - signal_mean(model, state_offset)
    if obs_var is None:
        sm = kalman_smoother(model, y, method=method)  # type: ignore[arg-type]
    else:
        sm = smoother_with_obs_var(model, y, obs_var)
    if state_offset is not None:
        sm = replace(sm, smoothed_state=sm.smoothed_state + state_offset)
    a, P, C = sm.smoothed_state, sm.smoothed_state_cov, sm.smoothed_state_autocov
    s11 = a[1:].T @ a[1:] + P[1:].sum(axis=0)
    s00 = a[:-1].T @ a[:-1] + P[:-1].sum(axis=0)
    s10 = a[1:].T @ a[:-1] + C[:-1].sum(axis=0)
    if free_periods is not None and free_states is not None:
        _profile_free_transitions(a, s11, s00, s10, free_periods, free_states)
    return SufficientStatistics(sm, s11, s00, s10, n - 1)


# ---------------------------------------------------------------------- M-step
def _solve_psd(matrix: FloatArray, rhs: FloatArray) -> FloatArray:
    """Solve ``matrix @ x = rhs`` for a symmetric PSD matrix (pseudo-inverse fallback)."""
    try:
        return scipy.linalg.solve(matrix, rhs, assume_a="pos")
    except (np.linalg.LinAlgError, scipy.linalg.LinAlgError):
        return np.linalg.pinv(matrix, hermitian=True) @ rhs


def restricted_least_squares(
    cross_moment: FloatArray,
    rhs: FloatArray,
    restriction: FloatArray | None = None,
    restriction_rhs: FloatArray | None = None,
) -> FloatArray:
    r"""Minimise :math:`\beta' S \beta - 2 \beta' s` subject to :math:`R\beta = q`.

    The solution is :math:`\beta = \beta_u - S^{-1}R'(R S^{-1} R')^{-1}(R\beta_u - q)`
    with :math:`\beta_u = S^{-1}s` (restricted least squares; Bańbura & Modugno, 2014,
    eq. 9 for the loadings of quarterly series).

    Parameters
    ----------
    cross_moment : numpy.ndarray, shape (k, k)
        Symmetric positive (semi-)definite matrix :math:`S`.
    rhs : numpy.ndarray, shape (k,)
        Vector :math:`s`.
    restriction : numpy.ndarray, shape (c, k), optional
        Restriction matrix :math:`R` (no restriction when omitted or empty).
    restriction_rhs : numpy.ndarray, shape (c,), optional
        Right-hand side :math:`q` (zeros by default).

    Returns
    -------
    numpy.ndarray, shape (k,)
        Restricted solution.

    Examples
    --------
    >>> import numpy as np
    >>> from nowcastbox.models._em_steps import restricted_least_squares
    >>> beta = restricted_least_squares(np.eye(2), np.array([1.0, 3.0]), np.array([[1.0, -1.0]]))
    >>> beta.tolist()
    [2.0, 2.0]
    """
    beta = _solve_psd(cross_moment, rhs)
    if restriction is None or restriction.shape[0] == 0:
        return beta
    q = np.zeros(restriction.shape[0]) if restriction_rhs is None else restriction_rhs
    s_inv_rt = _solve_psd(cross_moment, restriction.T)
    middle = restriction @ s_inv_rt
    correction = np.linalg.lstsq(middle, restriction @ beta - q, rcond=None)[0]
    return beta - s_inv_rt @ correction


def _m_step_factors(
    stats: SufficientStatistics, layout: StateLayout
) -> tuple[tuple[FloatArray, ...], tuple[FloatArray, ...]]:
    p = layout.factor_lags
    transitions: list[FloatArray] = []
    covs: list[FloatArray] = []
    for b in range(layout.n_blocks):
        cur = layout.factor_index(b, 0)
        lagged = layout.block_index(b)[: layout.n_factors[b] * p]
        s10 = stats.s10[np.ix_(cur, lagged)]
        s00 = stats.s00[np.ix_(lagged, lagged)]
        A = _solve_psd(s00, s10.T).T
        Q = (stats.s11[np.ix_(cur, cur)] - A @ s10.T) / stats.n_pairs
        Q = 0.5 * (Q + Q.T)
        eigval, eigvec = np.linalg.eigh(Q)
        Q = (eigvec * np.maximum(eigval, VARIANCE_FLOOR)) @ eigvec.T
        transitions.append(A)
        covs.append(0.5 * (Q + Q.T))
    return tuple(transitions), tuple(covs)


def _m_step_idiosyncratic(
    stats: SufficientStatistics, layout: StateLayout, previous: EMParameters
) -> tuple[FloatArray, FloatArray]:
    rho = previous.idio_ar.copy()
    var = previous.idio_var.copy()
    for i in range(layout.n_series):
        k = int(layout.idio_offsets[i])  # "ar1" layouts: every series has a segment
        s00, s10, s11 = stats.s00[k, k], stats.s10[k, k], stats.s11[k, k]
        if s00 > 0.0:
            rho_i = float(np.clip(s10 / s00, -MAX_AR_COEFFICIENT, MAX_AR_COEFFICIENT))
        else:  # degenerate (identically zero) component: keep the previous coefficient
            rho_i = float(previous.idio_ar[i])
        var_i = (s11 - 2.0 * rho_i * s10 + rho_i**2 * s00) / stats.n_pairs
        rho[i] = rho_i
        var[i] = max(float(var_i), VARIANCE_FLOOR)
    return rho, var


def _calendar_reduction(
    layout: StateLayout, series: int, path: FloatArray
) -> tuple[FloatArray, IntArray]:
    r"""Per-period maps from the regressor states to the reduced regressors.

    For a calendar aggregation the regressors of series ``i`` are
    :math:`G_t = (w_{i,t} \otimes I_{r_i}) F_t` (plus the long-run mean), with :math:`F_t`
    the stacked factor lags of :meth:`StateLayout.loading_index`.

    Returns
    -------
    D : numpy.ndarray, shape (n, q, K)
        Reduction matrices (``q = r_i`` (+1), ``K`` = size of the regressor index).
    index : numpy.ndarray of int
        State indices of the reduced coefficients (current factors, + long-run mean).
    """
    r = layout.n_series_factors(series)
    n, L = path.shape
    full = layout.regressor_index(series)
    trend = full.size > L * r
    D = np.zeros((n, r + int(trend), full.size))
    D[:, :r, : L * r] = (path[:, None, :, None] * np.eye(r)[None, :, None, :]).reshape(n, r, L * r)
    index = layout.loading_index(series)[:r]
    if trend:
        D[:, r, -1] = 1.0
        index = np.append(index, layout.trend_index)
    return D, index


def _calendar_moments(
    stats: SufficientStatistics,
    layout: StateLayout,
    observations: FloatArray,
    series: int,
    weights: FloatArray | None = None,
) -> tuple[FloatArray, FloatArray, float, int, IntArray]:
    """:func:`_series_moments` for a series with time-varying (calendar) weights."""
    observed = np.flatnonzero(~np.isnan(observations[:, series]))
    full = layout.regressor_index(series)
    a = stats.smoother.smoothed_state
    P = stats.smoother.smoothed_state_cov
    path = layout.weight_path(series, observations.shape[0])[observed]
    D, index = _calendar_reduction(layout, series, path)
    x = observations[observed, series]
    omega = np.ones(observed.size) if weights is None else weights[observed, series]
    a_f = a[np.ix_(observed, full)]
    second = P[np.ix_(observed, full, full)] + a_f[:, :, None] * a_f[:, None, :]
    S = np.einsum("t,tqk,tkl,tpl->qp", omega, D, second, D, optimize=True)
    g = np.einsum("tqk,tk->tq", D, a_f)
    s = g.T @ (omega * x)
    e_idx = layout.idio_index(series)
    if e_idx.size:  # "ar1": never combined with Student-t weights
        e_agg = np.einsum("tl,tl->t", a[np.ix_(observed, e_idx)], path)
        cross = a_f * e_agg[:, None] + np.einsum(
            "tkl,tl->tk", P[np.ix_(observed, full, e_idx)], path
        )
        s = s - np.einsum("tqk,tk->q", D, cross)
    return S, s, float(omega @ (x * x)), int(observed.size), index


def _series_moments(
    stats: SufficientStatistics,
    layout: StateLayout,
    observations: FloatArray,
    series: int,
    weights: FloatArray | None = None,
) -> tuple[FloatArray, FloatArray, float, int, IntArray]:
    r"""``S = sum w E[G G']``, ``s = sum w (x E[G] - E[G e_agg])``, ``sum w x^2``, count, index.

    ``weights`` (``(n_periods, N)``, optional) are the Student-t precision weights
    :math:`E[\lambda_{i,t}]` of the scale-mixture representation. ``index`` holds the
    state indices of the estimated coefficients.
    """
    if layout.calendar[series] is not None:
        return _calendar_moments(stats, layout, observations, series, weights)
    observed = np.flatnonzero(~np.isnan(observations[:, series]))
    idx = layout.regressor_index(series)
    a = stats.smoother.smoothed_state
    P = stats.smoother.smoothed_state_cov
    x = observations[observed, series]
    omega = np.ones(observed.size) if weights is None else weights[observed, series]
    a_g = a[np.ix_(observed, idx)]
    S = a_g.T @ (omega[:, None] * a_g) + np.einsum(
        "t,tij->ij", omega, P[np.ix_(observed, idx, idx)]
    )
    s = a_g.T @ (omega * x)
    e_idx = layout.idio_index(series)
    if e_idx.size:  # "ar1": never combined with Student-t weights
        w = layout.weights[series]
        e_agg = a[np.ix_(observed, e_idx)] @ w
        s = s - a_g.T @ e_agg - P[np.ix_(observed, idx, e_idx)].sum(axis=0) @ w
    return S, s, float(omega @ (x * x)), int(observed.size), idx


def _m_step_loadings(
    stats: SufficientStatistics,
    layout: StateLayout,
    observations: FloatArray,
    previous: EMParameters,
    weights: FloatArray | None = None,
) -> tuple[FloatArray, FloatArray, FloatArray | None]:
    loadings = np.zeros_like(previous.loadings)
    idio_var = previous.idio_var.copy()
    trend = None if previous.trend_loadings is None else previous.trend_loadings.copy()
    for i in range(layout.n_series):
        S, s, xx, n_obs, idx = _series_moments(stats, layout, observations, i, weights)
        if n_obs == 0:
            factor_idx = layout.loading_index(i)
            loadings[i, factor_idx] = previous.loadings[i, factor_idx]
            continue
        R, q = layout.constraints(i)
        lam = restricted_least_squares(S, s, R, q)
        if trend is not None and i in layout.long_run:
            trend[i] = lam[-1]
            loadings[i, idx[:-1]] = lam[:-1]
        else:
            loadings[i, idx] = lam
        if layout.idiosyncratic == "iid":
            psi = (xx - 2.0 * lam @ s + lam @ S @ lam) / n_obs
            idio_var[i] = max(float(psi), VARIANCE_FLOOR)
    return loadings, idio_var, trend


def _first_period_cov(stats: SufficientStatistics) -> FloatArray:
    """``Var(alpha_1 | Y)`` (``NaN`` between private groups of the structured smoother)."""
    sm = stats.smoother
    if not isinstance(sm, StructuredSmootherResult):
        return np.asarray(sm.smoothed_state_cov[0], dtype=np.float64)
    m = sm.smoothed_state.shape[1]
    out = np.full((m, m), np.nan)
    common = sm.structure.groups[0].states  # may be empty (assignment is then a no-op)
    out[np.ix_(common, common)] = sm.common_cov[0]
    for g in range(1, len(sm.structure.groups)):
        states, cov, _ = sm.group_moments(g)
        out[np.ix_(states, states)] = cov[0]
    return out


def _initial_second_moment(
    stats: SufficientStatistics, index: IntArray, first_cov: FloatArray | None = None
) -> FloatArray:
    """``E[alpha_1 alpha_1' | Y]`` restricted to ``index``."""
    a0 = stats.smoother.smoothed_state[0, index]
    cov = _first_period_cov(stats) if first_cov is None else first_cov
    return np.outer(a0, a0) + cov[np.ix_(index, index)]


def _gaussian_term(cov: FloatArray, second_moment: FloatArray, weight: float = 1.0) -> float:
    r"""Gaussian term :math:`-\tfrac12\{w\log|\Sigma| + \operatorname{tr}(\Sigma^{-1}M)\}`.

    ``-inf`` when :math:`\Sigma` is not positive definite.
    """
    try:
        chol = scipy.linalg.cholesky(cov, lower=True)
    except (np.linalg.LinAlgError, scipy.linalg.LinAlgError):
        return -np.inf
    logdet = 2.0 * float(np.sum(np.log(np.diag(chol))))
    trace = float(np.trace(scipy.linalg.cho_solve((chol, True), second_moment)))
    return -0.5 * (weight * logdet + trace)


def _factor_objective(
    A: FloatArray,
    Q: FloatArray,
    block: int,
    stats: SufficientStatistics,
    layout: StateLayout,
    initial: FloatArray,
) -> float:
    """Exact expected complete-data log-likelihood of one factor block (up to constants)."""
    cur = layout.factor_index(block, 0)
    lagged = layout.block_index(block)[: layout.n_factors[block] * layout.factor_lags]
    s10 = stats.s10[np.ix_(cur, lagged)]
    resid = stats.s11[np.ix_(cur, cur)] - A @ s10.T - s10 @ A.T
    resid = resid + A @ stats.s00[np.ix_(lagged, lagged)] @ A.T
    transitions = _gaussian_term(Q, resid, float(stats.n_pairs))
    comp = _block_companion(A, layout.block_lags[block])
    return transitions + _gaussian_term(_block_initial_cov(comp, Q), initial)


def _factor_path(
    step: float,
    *,
    block: int,
    start: tuple[FloatArray, FloatArray],
    delta: tuple[FloatArray, FloatArray],
    stats: SufficientStatistics,
    layout: StateLayout,
    initial: FloatArray,
) -> float:
    """:func:`_factor_objective` at ``start + step * delta``."""
    A = start[0] + step * delta[0]
    Q = start[1] + step * delta[1]
    return _factor_objective(A, Q, block, stats, layout, initial)


def _safeguarded_step(objective: Callable[[float], float], tolerance: float = 0.0) -> float:
    """Largest step ``2^-h`` (``h = 0, 1, ...``) not decreasing ``objective`` (0 if none)."""
    target = objective(0.0) - tolerance
    step = 1.0
    for _ in range(_MAX_STEP_HALVINGS):
        if objective(step) >= target:
            return step
        step *= 0.5
    return 0.0


def _safeguard_factors(
    stats: SufficientStatistics,
    layout: StateLayout,
    previous: EMParameters,
    transition: tuple[FloatArray, ...],
    factor_cov: tuple[FloatArray, ...],
) -> tuple[tuple[FloatArray, ...], tuple[FloatArray, ...]]:
    """Shorten the closed-form factor updates that lower the exact objective."""
    out_a, out_q = list(transition), list(factor_cov)
    first_cov = _first_period_cov(stats)
    for b in range(layout.n_blocks):
        initial = _initial_second_moment(stats, layout.block_index(b), first_cov)
        A0, Q0 = previous.transition[b], previous.factor_cov[b]
        dA, dQ = transition[b] - A0, factor_cov[b] - Q0

        objective = partial(
            _factor_path,
            block=b,
            start=(A0, Q0),
            delta=(dA, dQ),
            stats=stats,
            layout=layout,
            initial=initial,
        )
        step = _safeguarded_step(objective)
        if step < 1.0:
            out_a[b], out_q[b] = A0 + step * dA, Q0 + step * dQ
    return tuple(out_a), tuple(out_q)


def _ar1_initial_moments(
    stats: SufficientStatistics, layout: StateLayout
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    """Per series: ``E[e_1^2]`` and sums over the segment of the lag products at ``t = 1``.

    For the segment ``(e_t, e_{t-1}, ..., e_{t-L+1})`` of the first state: ``m00`` =
    ``E[x_0^2]``, ``cur`` = ``sum_j E[x_j^2]``, ``cross`` = ``sum_j E[x_j x_{j-1}]``,
    ``prev`` = ``sum_j E[x_{j-1}^2]`` (``j = 1..L-1``) and the lengths ``L``.
    """
    first_cov = _first_period_cov(stats)
    a1 = stats.smoother.smoothed_state[0]
    n = layout.n_series
    m00, cur, cross, prev, length = (np.zeros(n) for _ in range(5))
    for i in range(n):
        idx = layout.idio_index(i)
        M = np.outer(a1[idx], a1[idx]) + first_cov[np.ix_(idx, idx)]
        diag = np.diag(M)
        m00[i], length[i] = diag[0], idx.size
        cur[i], prev[i] = diag[1:].sum(), diag[:-1].sum()
        cross[i] = np.trace(M, offset=-1)
    return m00, cur, cross, prev, length


def _ar1_objectives(
    rho: FloatArray,
    var: FloatArray,
    transitions: tuple[FloatArray, FloatArray, FloatArray],
    initial: tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray],
    n_pairs: int,
) -> FloatArray:
    r"""Exact expected complete-data log-likelihood of every idiosyncratic AR(1) chain.

    The first state of a chain is the stationary segment
    :math:`(e_1, e_0, \dots, e_{2-L})` whose density, by time reversibility, is
    :math:`N(0, \sigma^2/(1-\rho^2))` for :math:`e_1` times :math:`L-1` conditional
    :math:`N(\rho x_{j-1}, \sigma^2)` terms (same as ``_ar1_segment_cov``).
    """
    s11, s10, s00 = transitions
    m00, cur, cross, prev, length = initial
    ssr = s11 - 2.0 * rho * s10 + rho**2 * s00
    first = (1.0 - rho**2) * m00 + cur - 2.0 * rho * cross + rho**2 * prev
    with np.errstate(divide="ignore", invalid="ignore"):
        value = (n_pairs + length) * np.log(var) - np.log1p(-(rho**2)) + (ssr + first) / var
    return np.where(np.isfinite(value), -0.5 * value, -np.inf)


def _safeguard_idiosyncratic(
    stats: SufficientStatistics,
    layout: StateLayout,
    previous: EMParameters,
    rho: FloatArray,
    var: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Shorten the closed-form AR(1) updates that lower the exact objective."""
    k = layout.idio_offsets
    transitions = (stats.s11[k, k], stats.s10[k, k], stats.s00[k, k])
    initial = _ar1_initial_moments(stats, layout)
    r0, v0 = previous.idio_ar, previous.idio_var
    steps = np.append(0.5 ** np.arange(_MAX_STEP_HALVINGS), 0.0)
    cand_r = r0[:, None] + steps[None, :] * (rho - r0)[:, None]
    cand_v = v0[:, None] + steps[None, :] * (var - v0)[:, None]
    values = _ar1_objectives(
        cand_r,
        cand_v,
        tuple(x[:, None] for x in transitions),  # type: ignore[arg-type]
        tuple(x[:, None] for x in initial),  # type: ignore[arg-type]
        stats.n_pairs,
    )
    accepted = values >= values[:, -1:]
    choice = np.argmax(accepted, axis=1)  # first (largest) accepted step; step 0 always is
    rows = np.arange(layout.n_series)
    return cand_r[rows, choice], cand_v[rows, choice]


def m_step(
    stats: SufficientStatistics,
    layout: StateLayout,
    observations: FloatArray,
    previous: EMParameters,
    *,
    weights: FloatArray | None = None,
    initial_term: bool = True,
) -> EMParameters:
    r"""Closed-form M-step (Shumway & Stoffer, 1982; Bańbura & Modugno, 2014).

    With ``weights`` (Student-t precision weights :math:`E[\lambda_{i,t}]`), loadings
    and idiosyncratic variances solve the weighted least-squares problems of the
    ECM algorithm for the scale mixture of normals (Lange, Little & Taylor, 1989;
    Liu & Rubin, 1995). The long-run mean variance (I4) is
    :math:`\sum_t E[(\mu_t - \mu_{t-1})^2] / (n - 1)` unless fixed by the layout.

    The transition parameters (factor VARs and idiosyncratic AR(1)s) also enter the
    stationary distribution of the first state. With ``initial_term=True`` (default)
    each closed-form update is accepted only if it does not lower the exact expected
    complete-data log-likelihood of its block, initial-state term included; otherwise
    the step towards it is halved until it does (generalized EM: the log-likelihood is
    monotone, see the module notes).

    Parameters
    ----------
    stats : SufficientStatistics
        Output of :func:`e_step` at ``previous``.
    layout : StateLayout
        State layout.
    observations : numpy.ndarray, shape (n_periods, N)
        Standardised data (``NaN`` = missing).
    previous : EMParameters
        Parameters used in the E-step (the measurement-noise variance of the ``"ar1"``
        specification is kept fixed).
    weights : numpy.ndarray, shape (n_periods, N), optional
        Precision weights of the observations (``"iid"`` layouts only).
    initial_term : bool, default True
        Safeguard the transition updates with the initial-state term (``False``: the
        classical closed-form update of Bańbura & Modugno, 2014, which ignores it).

    Returns
    -------
    EMParameters
        Updated parameters.

    Examples
    --------
    >>> import numpy as np
    >>> from nowcastbox.models._em_steps import (
    ...     EMParameters,
    ...     StateLayout,
    ...     build_state_space,
    ...     e_step,
    ...     m_step,
    ... )
    >>> lay = StateLayout(["a", "b"], ["g"], [1], 1, np.ones((2, 1), bool), [[1.0], [1.0]], "iid")
    >>> p = EMParameters(
    ...     (np.array([[0.5]]),), (np.eye(1),), np.ones((2, 1)), np.zeros(2), np.ones(2), np.ones(2)
    ... )
    >>> y = np.random.default_rng(0).standard_normal((20, 2))
    >>> new = m_step(e_step(build_state_space(p, lay), y), lay, y, p)
    >>> new.loadings.shape
    (2, 1)
    """
    transition, factor_cov = _m_step_factors(stats, layout)
    if initial_term:
        transition, factor_cov = _safeguard_factors(stats, layout, previous, transition, factor_cov)
    loadings, idio_var, trend = _m_step_loadings(stats, layout, observations, previous, weights)
    if layout.idiosyncratic == "ar1":
        idio_ar, idio_var = _m_step_idiosyncratic(stats, layout, previous)
        if initial_term:
            idio_ar, idio_var = _safeguard_idiosyncratic(stats, layout, previous, idio_ar, idio_var)
        obs_var = previous.obs_var.copy()
    else:
        idio_ar = np.zeros(layout.n_series)
        obs_var = idio_var.copy()
    trend_var = float(previous.trend_var)
    if layout.has_long_run and layout.long_run_variance is None:
        k = layout.trend_index
        trend_var = random_walk_variance(
            stats.s11[k, k], stats.s10[k, k], stats.s00[k, k], stats.n_pairs, VARIANCE_FLOOR
        )
    return EMParameters(
        transition, factor_cov, loadings, idio_ar, idio_var, obs_var, trend, trend_var
    )


def em_converged(loglikelihood: float, previous: float, tol: float) -> tuple[bool, float]:
    r"""Relative change criterion of the EM algorithm.

    :math:`c = (\ell_k - \ell_{k-1}) / \{(|\ell_k| + |\ell_{k-1}| + \epsilon)/2\}`;
    convergence when :math:`|c| < \text{tol}` (Doz, Giannone & Reichlin, 2012;
    Bańbura & Modugno, 2014).

    Parameters
    ----------
    loglikelihood : float
        Current log-likelihood.
    previous : float
        Log-likelihood of the previous iteration.
    tol : float
        Tolerance.

    Returns
    -------
    converged : bool
        Whether ``|c| < tol``.
    change : float
        The signed relative change ``c``.

    Examples
    --------
    >>> from nowcastbox.models._em_steps import em_converged
    >>> converged, change = em_converged(-100.0, -100.001, 1e-4)
    >>> converged, round(change, 8)
    (True, 1e-05)
    """
    average = (abs(loglikelihood) + abs(previous) + np.finfo(float).eps) / 2.0
    change = (loglikelihood - previous) / average
    return bool(abs(change) < tol), float(change)
