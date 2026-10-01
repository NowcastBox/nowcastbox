r"""Robust EM for :class:`~nowcastbox.models.MixedFreqDFM` (innovation I3).

Three complementary devices protect the factor model against outliers and against the
2020-2021 pandemic observations without discarding the sample (Antolin-Diaz, Drechsel
& Petrella, 2017, 2024):

**Student-t idiosyncratic errors** (``idiosyncratic="student_t"``). The idiosyncratic
component is white noise with a scaled Student-t distribution, written as a scale
mixture of normals (Andrews & Mallows, 1974; Lange, Little & Taylor, 1989):

.. math::

    \varepsilon_{i,t} \mid \lambda_{i,t} \sim N(0, \psi_i / \lambda_{i,t}), \qquad
    \lambda_{i,t} \sim \text{Gamma}(\nu/2, \nu/2).

The EM algorithm treats :math:`(\alpha_t, \lambda_{i,t})` as missing data. Because the
joint posterior of states and mixing weights is not available in closed form, the
E-step uses the factorisation :math:`q(\alpha)\,q(\lambda)` (variational EM; Beal &
Ghahramani, 2003): :math:`q(\alpha)` is the Kalman smoother of the Gaussian model with
measurement variances :math:`\psi_i / w_{i,t}`, :math:`w_{i,t} = E_q[\lambda_{i,t}]`, and
:math:`q(\lambda_{i,t}) = \text{Gamma}((\nu+1)/2, (\nu + \delta_{i,t})/2)` with
:math:`\delta_{i,t} = E_q[(x_{i,t} - Z_i\alpha_t)^2]/\psi_i`, hence

.. math::

    w_{i,t} = \frac{\nu + 1}{\nu + \delta_{i,t}}

(large residuals get small weights). The M-step is the ECM of Liu & Rubin (1995):
weighted least squares for loadings and :math:`\psi_i`, then (``df=None``) a
one-dimensional maximisation over :math:`\nu` of
:math:`\sum_{i,t} [\tfrac{\nu}{2}\log\tfrac{\nu}{2} - \log\Gamma(\tfrac{\nu}{2})
+ (\tfrac{\nu}{2} - 1) E\log\lambda_{i,t} - \tfrac{\nu}{2} E\lambda_{i,t}]`.
Every step increases the evidence lower bound

.. math::

    \mathcal F = \ell_G(\theta; w) + \sum_{i,t}\Big[\tfrac12\big(E\log\lambda_{i,t}
    - \log w_{i,t}\big) - \text{KL}\big(q(\lambda_{i,t}) \,\|\, p(\lambda_{i,t}\mid\nu)\big)\Big]

(:math:`\ell_G`: Gaussian log-likelihood with variances :math:`\psi_i/w_{i,t}`), a lower
bound of the Student-t log-likelihood; this is the objective reported (and monitored
for convergence) by the robust EM.

**Automatic outlier handling** (``outliers="auto"``). After the EM converges, the
standardised one-step-ahead prediction errors
:math:`v_{i,t}/\sqrt{F_{ii,t}}` (Durbin & Koopman, 2012, sec. 2.12 and 7.5) of every
observation are computed; those above ``outlier_threshold`` in absolute value are
treated as missing and the EM is re-run, until the set of flagged observations is
stable (at most ``max_outlier_passes`` passes). Unlike pre-cleaning (Stock & Watson,
2002) the outliers are judged against the model's own forecasts, so common movements
(recessions) are not flagged.

**Pandemic / excluded periods.** ``exclude_periods`` and ``covid="mask"`` treat every
observation of the given periods as missing in the E-step (the states are still
smoothed through them; Schorfheide & Song, 2021, drop the 2020 observations in the same
spirit). ``covid="dummy"`` instead adds an unrestricted intercept (impulse dummy;
Lütkepohl, 2005, ch. 10) to the factor VAR in every period of the window,
:math:`f_t = \sum_k A_k f_{t-k} + D_t + u_t`: the data of the window still identify the
factors (the pandemic swing is visible in the factors and the nowcasts), but the
swing is absorbed by :math:`D_t` instead of distorting :math:`A` and :math:`Q`. In the
EM the dummies enter as a deterministic state offset :math:`g_t = T g_{t-1} + S D_t`,
their ML value given the other parameters is :math:`D_t = E[f_t - \sum_k A_k
f_{t-k} \mid Y]`, and profiling them out leaves only the conditional covariances of
those transitions in the M-step of :math:`A` and :math:`Q` (exact ECM).

References
----------
Andrews, D. F., & Mallows, C. L. (1974). Scale mixtures of normal distributions.
*JRSS-B*, 36(1), 99-102.

Antolin-Diaz, J., Drechsel, T., & Petrella, I. (2017). Tracking the slowdown in long-run
GDP growth. *Review of Economics and Statistics*, 99(2), 343-356.

Antolin-Diaz, J., Drechsel, T., & Petrella, I. (2024). Advances in nowcasting economic
activity: the role of heterogeneous dynamics and fat tails. *Journal of Econometrics*,
238(2), 105634.

Beal, M. J., & Ghahramani, Z. (2003). The variational Bayesian EM algorithm for
incomplete data. *Bayesian Statistics 7*, 453-464.

Lange, K. L., Little, R. J. A., & Taylor, J. M. G. (1989). Robust statistical modeling
using the t distribution. *JASA*, 84(408), 881-896.

Liu, C., & Rubin, D. B. (1995). ML estimation of the t distribution using EM and its
extensions, ECM and ECME. *Statistica Sinica*, 5(1), 19-39.

Schorfheide, F., & Song, D. (2021). Real-time forecasting with a (standard) mixed-
frequency VAR during a pandemic. NBER Working Paper 29535.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd
import scipy.optimize
import scipy.special
from numpy.typing import NDArray

from nowcastbox._logging import get_logger
from nowcastbox.core.exceptions import NowcastDataError
from nowcastbox.core.frequency import Frequency
from nowcastbox.models._em_steps import (
    EMParameters,
    StateLayout,
    SufficientStatistics,
    build_state_space,
    e_step,
    em_converged,
    m_step,
    signal_mean,
    signal_variance,
    transition_matrix,
)
from nowcastbox.statespace import SmootherResult

__all__ = [
    "COVID_MODES",
    "COVID_WINDOW",
    "DF_BOUNDS",
    "OUTLIER_MODES",
    "RobustEMResult",
    "RobustSpec",
    "RobustState",
    "StudentTWeights",
    "expected_squared_residuals",
    "flag_outliers",
    "intervention_offset",
    "period_mask",
    "robust_smoother",
    "run_robust_em",
    "standardized_innovations",
    "student_t_bound",
    "update_df",
    "update_interventions",
]

logger = get_logger(__name__)

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

COVID_MODES = ("none", "mask", "dummy")
"""Accepted values of ``MixedFreqDFM(covid=...)``."""

COVID_WINDOW = ("2020-03", "2021-12")
"""Default pandemic window (first and last base period), valid for Brazil and the US."""

OUTLIER_MODES = ("none", "auto")
"""Accepted values of ``MixedFreqDFM(outliers=...)``."""

DF_BOUNDS = (2.0, 200.0)
"""Search interval of the estimated degrees of freedom (finite variance at the bound)."""

DF_START = 10.0
"""Starting value of the degrees of freedom when they are estimated."""

_DECREASE_TOL = 1e-9


# ====================================================================== periods
def _period_bounds(value: Any) -> tuple[pd.Period, pd.Period]:
    if isinstance(value, tuple | list) and len(value) == 2:
        return _period_bounds(value[0])[0], _period_bounds(value[1])[1]
    try:
        period = value if isinstance(value, pd.Period) else pd.Period(str(value))
    except (TypeError, ValueError) as err:
        raise ValueError(f"Cannot interpret {value!r} as a period.") from err
    return period, period


def _on_or_after(index: pd.PeriodIndex, period: pd.Period, after: bool) -> BoolArray:
    """Base periods on/after (``after``) or on/before ``period`` (calendar aware)."""
    base = Frequency.from_index(index)
    if Frequency.from_value(period.freqstr).is_lower_than(base):
        # a base period belongs to the native period containing its last day
        native = index.asfreq(period.freqstr)
        return np.asarray(native >= period if after else native <= period, dtype=bool)
    own = period.asfreq(index.freqstr, how="start" if after else "end")
    return np.asarray(index >= own if after else index <= own, dtype=bool)


def period_mask(index: pd.PeriodIndex, periods: Any) -> BoolArray:
    """Boolean mask of the base periods covered by a period specification.

    Parameters
    ----------
    index : pandas.PeriodIndex
        Base grid.
    periods : None, period-like, (start, end) tuple or sequence of those
        Periods to select. A single period (``"2020-04"``, ``"2020Q2"``, ``"2020"``,
        :class:`pandas.Period`) covers all its base periods; a 2-tuple ``(start, end)``
        covers the closed range. A list combines several specifications.

    Returns
    -------
    numpy.ndarray of bool, shape (len(index),)
        Mask of the selected periods. On weekly or daily grids a base period is covered
        when the period containing its last day is (a week belongs to the month in
        which it ends).

    Raises
    ------
    ValueError
        If an entry cannot be interpreted as a period or a range is reversed.

    Examples
    --------
    >>> import pandas as pd
    >>> from nowcastbox.models.robust import period_mask
    >>> idx = pd.period_range("2020-01", periods=8, freq="M")
    >>> period_mask(idx, ["2020Q1", ("2020-06", "2020-07")]).astype(int).tolist()
    [1, 1, 1, 0, 0, 1, 1, 0]
    """
    mask = np.zeros(len(index), dtype=bool)
    if periods is None:
        return mask
    if isinstance(periods, str | pd.Period) or (
        isinstance(periods, tuple) and len(periods) == 2 and not isinstance(periods[0], tuple)
    ):
        specs: Sequence[Any] = [periods]
    else:
        specs = list(periods)
    freq = index.freqstr
    for spec in specs:
        start, end = _period_bounds(spec)
        if end.asfreq(freq, how="end") < start.asfreq(freq, how="start"):
            raise ValueError(f"Period range {spec!r} ends before it starts.")
        mask |= _on_or_after(index, start, after=True) & _on_or_after(index, end, after=False)
    return mask


# ====================================================================== specification
@dataclass(frozen=True, eq=False)
class RobustSpec:
    """Robustness options of the EM algorithm.

    Parameters
    ----------
    student_t : bool, default False
        Student-t idiosyncratic errors (``"iid"`` layouts).
    df : float, optional
        Fixed degrees of freedom; ``None`` estimates them.
    outliers : bool, default False
        Automatic outlier handling.
    outlier_threshold : float, default 4.0
        Threshold on the absolute standardised prediction error.
    max_outlier_passes : int, default 3
        Maximum number of EM runs of the outlier procedure.
    excluded : numpy.ndarray of bool, shape (n_periods, N), optional
        Observations treated as missing (``exclude_periods``, ``covid="mask"``).
    intervention : numpy.ndarray of bool, shape (n_periods,), optional
        Periods with impulse dummies in the factor VAR (``covid="dummy"``).

    Examples
    --------
    >>> from nowcastbox.models.robust import RobustSpec
    >>> RobustSpec(student_t=True).is_active
    True
    >>> RobustSpec().is_active
    False
    """

    student_t: bool = False
    df: float | None = None
    outliers: bool = False
    outlier_threshold: float = 4.0
    max_outlier_passes: int = 3
    excluded: BoolArray | None = field(default=None, repr=False)
    intervention: BoolArray | None = field(default=None, repr=False)

    @property
    def is_active(self) -> bool:
        """Whether any robustness device is requested."""
        return bool(
            self.student_t
            or self.outliers
            or (self.excluded is not None and self.excluded.any())
            or (self.intervention is not None and self.intervention[1:].any())
        )

    @property
    def has_intervention(self) -> bool:
        """Whether impulse dummies are active (some period after the first)."""
        return self.intervention is not None and bool(self.intervention[1:].any())


# ====================================================================== Student-t
@dataclass(frozen=True, eq=False)
class StudentTWeights:
    r"""Variational distribution :math:`q(\lambda_{i,t}) = \text{Gamma}(a, b_{i,t})`.

    Parameters
    ----------
    shape : float
        Common shape :math:`a`.
    rate : numpy.ndarray, shape (n_periods, N)
        Rates :math:`b_{i,t}` (positive).

    Examples
    --------
    >>> import numpy as np
    >>> from nowcastbox.models.robust import StudentTWeights
    >>> q = StudentTWeights.prior((2, 1), 4.0)
    >>> q.weights.ravel().tolist()
    [1.0, 1.0]
    """

    shape: float
    rate: FloatArray

    @classmethod
    def prior(cls, size: tuple[int, int], df: float) -> StudentTWeights:
        r"""Prior :math:`\text{Gamma}(\nu/2, \nu/2)` (unit weights)."""
        return cls(df / 2.0, np.full(size, df / 2.0))

    @classmethod
    def posterior(cls, delta: FloatArray, df: float) -> StudentTWeights:
        r"""Update given the scaled expected squared residuals :math:`\delta_{i,t}`.

        Missing entries (``NaN``) keep unit weights.
        """
        shape = (df + 1.0) / 2.0
        rate = np.where(np.isnan(delta), shape, (df + np.nan_to_num(delta)) / 2.0)
        return cls(shape, rate)

    @property
    def weights(self) -> FloatArray:
        r""":math:`E[\lambda_{i,t}] = a / b_{i,t}`."""
        return self.shape / self.rate

    @property
    def log_weights(self) -> FloatArray:
        r""":math:`E[\log\lambda_{i,t}] = \psi(a) - \log b_{i,t}`."""
        return float(scipy.special.digamma(self.shape)) - np.log(self.rate)


def expected_squared_residuals(smoother: SmootherResult, observations: FloatArray) -> FloatArray:
    r""":math:`E[(x_{i,t} - Z_i\alpha_t)^2 \mid Y] = (x_{i,t} - Z_i a_{t|n})^2 + Z_i V_t Z_i'`.

    Parameters
    ----------
    smoother : SmootherResult
        Smoother output (smoothed states including any deterministic offset).
    observations : numpy.ndarray, shape (n_periods, N)
        Data (``NaN`` = missing).

    Returns
    -------
    numpy.ndarray, shape (n_periods, N)
        Expected squared residuals (``NaN`` where data are missing).

    Examples
    --------
    >>> import numpy as np
    >>> from nowcastbox.statespace import StateSpace, kalman_smoother
    >>> from nowcastbox.models.robust import expected_squared_residuals
    >>> ssm = StateSpace([[0.0]], [[1.0]], [[1.0]], [1.0])
    >>> y = np.array([[2.0], [np.nan]])
    >>> expected_squared_residuals(kalman_smoother(ssm, y), y).ravel().tolist()
    [1.5, nan]
    """
    model = smoother.model
    resid = observations - signal_mean(model, smoother.smoothed_state)
    return resid**2 + signal_variance(model, smoother.smoothed_state_cov)


def _df_objective(df: float, log_weights: FloatArray, weights: FloatArray) -> float:
    half = df / 2.0
    count = weights.size
    value = count * (half * np.log(half) - scipy.special.gammaln(half))
    value += (half - 1.0) * float(log_weights.sum()) - half * float(weights.sum())
    return float(value)


def update_df(
    q: StudentTWeights, observed: BoolArray, bounds: tuple[float, float] = DF_BOUNDS
) -> float:
    r"""ECM update of the degrees of freedom :math:`\nu` (Liu & Rubin, 1995).

    Maximises :math:`\sum_{(i,t)\,\text{observed}} E_q[\log p(\lambda_{i,t} \mid \nu)]`
    over ``bounds`` (one-dimensional bounded search on :math:`\log\nu`).

    Parameters
    ----------
    q : StudentTWeights
        Current distribution of the mixing weights.
    observed : numpy.ndarray of bool, shape (n_periods, N)
        Observations entering the likelihood.
    bounds : tuple of float, default DF_BOUNDS
        Search interval.

    Returns
    -------
    float
        Updated degrees of freedom.

    Examples
    --------
    >>> import numpy as np
    >>> from nowcastbox.models.robust import StudentTWeights, update_df
    >>> rng = np.random.default_rng(0)
    >>> lam = rng.gamma(2.5, 1 / 2.5, size=(4000, 1))  # nu = 5
    >>> q = StudentTWeights(1e6, 1e6 / lam)  # (almost) known weights
    >>> round(update_df(q, np.ones_like(lam, bool)))
    5
    """
    if not observed.any():
        return float(np.sqrt(bounds[0] * bounds[1]))
    elog = q.log_weights[observed]
    w = q.weights[observed]
    result = scipy.optimize.minimize_scalar(
        lambda log_df: -_df_objective(float(np.exp(log_df)), elog, w),
        bounds=(np.log(bounds[0]), np.log(bounds[1])),
        method="bounded",
        options={"xatol": 1e-6},
    )
    return float(np.exp(result.x))


def student_t_bound(q: StudentTWeights, df: float, observed: BoolArray) -> float:
    r"""Correction turning the Gaussian log-likelihood into the evidence lower bound.

    :math:`\sum_{(i,t)} [\tfrac12(E\log\lambda_{i,t} - \log w_{i,t})
    - \text{KL}(\text{Gamma}(a, b_{i,t}) \,\|\, \text{Gamma}(\nu/2, \nu/2))]` over the
    observed entries.

    Parameters
    ----------
    q : StudentTWeights
        Distribution of the mixing weights.
    df : float
        Degrees of freedom of the prior.
    observed : numpy.ndarray of bool
        Observations entering the likelihood.

    Returns
    -------
    float
        Correction (non-positive).

    Examples
    --------
    >>> import numpy as np
    >>> from nowcastbox.models.robust import StudentTWeights, student_t_bound
    >>> q = StudentTWeights.prior((3, 1), 1e8)  # nu -> inf: lambda = 1
    >>> abs(student_t_bound(q, 1e8, np.ones((3, 1), bool))) < 1e-6
    True
    """
    a = q.shape
    b = q.rate[observed]
    elog = float(scipy.special.digamma(a)) - np.log(b)
    half = df / 2.0
    kl = (
        (a - half) * scipy.special.digamma(a)
        - scipy.special.gammaln(a)
        + scipy.special.gammaln(half)
        + half * (np.log(b) - np.log(half))
        + a * (half - b) / b
    )
    return float(np.sum(0.5 * (elog - np.log(a / b)) - kl))


# ====================================================================== interventions
def _shock_rows(layout: StateLayout) -> NDArray[np.int64]:
    return layout.current_factor_index


def intervention_offset(
    dummies: FloatArray, params: EMParameters, layout: StateLayout
) -> FloatArray:
    r"""Deterministic state path :math:`g_t = T g_{t-1} + S D_t` of the impulse dummies.

    Parameters
    ----------
    dummies : numpy.ndarray, shape (n_periods, r)
        Dummies :math:`D_t` on the current factors (zero outside the window).
    params : EMParameters
        Parameters (define ``T``).
    layout : StateLayout
        State layout.

    Returns
    -------
    numpy.ndarray, shape (n_periods, n_states)
        Offset :math:`g_t` (:math:`g_0 = S D_0`).

    Examples
    --------
    >>> import numpy as np
    >>> from nowcastbox.models._em_steps import EMParameters, StateLayout
    >>> from nowcastbox.models.robust import intervention_offset
    >>> lay = StateLayout(["a", "b"], ["g"], [1], 1, np.ones((2, 1), bool), [[1.0]] * 2, "iid")
    >>> p = EMParameters(
    ...     (np.array([[0.5]]),), (np.eye(1),), np.ones((2, 1)), np.zeros(2), np.ones(2), np.ones(2)
    ... )
    >>> intervention_offset(np.array([[0.0], [2.0], [0.0]]), p, lay).ravel().tolist()
    [0.0, 2.0, 1.0]
    """
    T = transition_matrix(params, layout)
    rows = _shock_rows(layout)
    g = np.zeros((dummies.shape[0], layout.n_states))
    g[0, rows] = dummies[0]
    for t in range(1, dummies.shape[0]):
        g[t] = T @ g[t - 1]
        g[t, rows] += dummies[t]
    return g


def update_interventions(
    state: FloatArray, params: EMParameters, layout: StateLayout, periods: BoolArray
) -> FloatArray:
    r"""ML impulse dummies :math:`D_t = S'(a_{t|n} - T a_{t-1|n})` in the window.

    Parameters
    ----------
    state : numpy.ndarray, shape (n_periods, n_states)
        Smoothed states (including the current offset).
    params : EMParameters
        Updated parameters (define ``T``).
    layout : StateLayout
        State layout.
    periods : numpy.ndarray of bool, shape (n_periods,)
        Window (the first period never gets a dummy).

    Returns
    -------
    numpy.ndarray, shape (n_periods, r)
        Dummies (zero outside the window).

    Examples
    --------
    >>> import numpy as np
    >>> from nowcastbox.models._em_steps import EMParameters, StateLayout
    >>> from nowcastbox.models.robust import update_interventions
    >>> lay = StateLayout(["a", "b"], ["g"], [1], 1, np.ones((2, 1), bool), [[1.0]] * 2, "iid")
    >>> p = EMParameters(
    ...     (np.array([[0.5]]),), (np.eye(1),), np.ones((2, 1)), np.zeros(2), np.ones(2), np.ones(2)
    ... )
    >>> a = np.array([[1.0], [3.0], [1.0]])
    >>> update_interventions(a, p, lay, np.array([True, True, False])).ravel().tolist()
    [0.0, 2.5, 0.0]
    """
    T = transition_matrix(params, layout)
    rows = _shock_rows(layout)
    dummies = np.zeros((state.shape[0], rows.size))
    window = np.flatnonzero(periods[1:]) + 1
    predicted = state[window - 1] @ T.T
    dummies[window] = state[np.ix_(window, rows)] - predicted[:, rows]
    return dummies


# ====================================================================== outliers
def standardized_innovations(
    smoother: SmootherResult, observations: FloatArray, offset: FloatArray | None = None
) -> FloatArray:
    r"""Standardised one-step-ahead prediction errors of every observation.

    :math:`(x_{i,t} - Z_i a_t) / \sqrt{Z_i P_t Z_i' + H_{ii}}` with the predicted state
    :math:`a_t` and covariance :math:`P_t` of the filter run of ``smoother`` (plus the
    deterministic ``offset``) and :math:`H` of its model. Observations that were missing
    in the filter run are standardised too (useful to re-assess flagged outliers).

    Parameters
    ----------
    smoother : SmootherResult
        Smoother output.
    observations : numpy.ndarray, shape (n_periods, N)
        Data to standardise (``NaN`` = missing).
    offset : numpy.ndarray, shape (n_periods, n_states), optional
        Deterministic state offset of the run.

    Returns
    -------
    numpy.ndarray, shape (n_periods, N)
        Standardised errors (``NaN`` where data are missing).

    Examples
    --------
    >>> import numpy as np
    >>> from nowcastbox.statespace import StateSpace, kalman_smoother
    >>> from nowcastbox.models.robust import standardized_innovations
    >>> ssm = StateSpace([[0.0]], [[1.0]], [[1.0]], [3.0])
    >>> y = np.array([[4.0]])
    >>> standardized_innovations(kalman_smoother(ssm, y), y).item()
    2.0
    """
    fres = smoother.filter_result
    model = fres.model
    a = fres.predicted_state[:-1]
    if offset is not None:
        a = a + offset
    var = signal_variance(model, fres.predicted_state_cov[:-1])
    var = var + model.obs_cov_diagonal_store[model.period_index(a.shape[0])]
    return (observations - signal_mean(model, a)) / np.sqrt(np.maximum(var, 1e-12))


def flag_outliers(innovations: FloatArray, threshold: float) -> BoolArray:
    """Observations whose absolute standardised prediction error exceeds ``threshold``.

    Parameters
    ----------
    innovations : numpy.ndarray
        Standardised prediction errors (``NaN`` = missing, never flagged).
    threshold : float
        Positive threshold.

    Returns
    -------
    numpy.ndarray of bool
        Flags.

    Examples
    --------
    >>> import numpy as np
    >>> from nowcastbox.models.robust import flag_outliers
    >>> flag_outliers(np.array([0.5, -5.0, np.nan]), 4.0).tolist()
    [False, True, False]
    """
    return np.abs(np.nan_to_num(innovations, nan=0.0)) > threshold


# ====================================================================== EM
@dataclass(frozen=True, eq=False)
class RobustState:
    """Latent quantities of the robust EM besides the model parameters.

    Parameters
    ----------
    q : StudentTWeights, optional
        Distribution of the Student-t mixing weights.
    df : float, optional
        Degrees of freedom.
    dummies : numpy.ndarray, shape (n_periods, r), optional
        Impulse dummies of the factor VAR.
    flags : numpy.ndarray of bool, shape (n_periods, N)
        Observations flagged as outliers.

    Examples
    --------
    >>> import numpy as np
    >>> from nowcastbox.models.robust import RobustState
    >>> RobustState(flags=np.zeros((3, 2), bool)).padded(5).flags.shape
    (5, 2)
    """

    flags: BoolArray
    q: StudentTWeights | None = None
    df: float | None = None
    dummies: FloatArray | None = None

    def padded(self, n_periods: int) -> RobustState:
        """Extend the per-period arrays with neutral values to ``n_periods`` rows."""
        extra = n_periods - self.flags.shape[0]
        flags = np.vstack([self.flags, np.zeros((extra, self.flags.shape[1]), bool)])
        q = self.q
        if q is not None:
            pad = np.full((extra, q.rate.shape[1]), q.shape)
            q = StudentTWeights(q.shape, np.vstack([q.rate, pad]))
        dummies = self.dummies
        if dummies is not None:
            dummies = np.vstack([dummies, np.zeros((extra, dummies.shape[1]))])
        return RobustState(flags, q, self.df, dummies)


@dataclass(frozen=True, eq=False)
class RobustEMResult:
    """Output of :func:`run_robust_em`.

    Attributes
    ----------
    params : EMParameters
        Final parameters.
    loglikelihood_path : numpy.ndarray
        Objective at every E-step of the last pass (Gaussian log-likelihood, or the
        evidence lower bound with Student-t errors).
    converged : bool
        Whether the last pass converged.
    n_iter : int
        M-steps of the last pass.
    n_decreases : int
        Decreases of the objective in the last pass.
    last_change : float
        Last relative change.
    state : RobustState
        Weights, degrees of freedom, dummies and outlier flags.
    n_passes : int
        Number of EM runs (outlier procedure).
    total_iter : int
        M-steps over all passes.
    """

    params: EMParameters
    loglikelihood_path: FloatArray
    converged: bool
    n_iter: int
    n_decreases: int
    last_change: float
    state: RobustState
    n_passes: int = 1
    total_iter: int = 0


def _masked(observations: FloatArray, spec: RobustSpec, flags: BoolArray) -> FloatArray:
    excluded = flags if spec.excluded is None else (flags | spec.excluded)
    return np.where(excluded, np.nan, observations)


def _dense(stats: SufficientStatistics) -> SmootherResult:
    """Dense smoother output of a robust E-step (never the structured smoother)."""
    if not isinstance(stats.smoother, SmootherResult):  # pragma: no cover - invariant
        raise TypeError("the robust EM needs the dense Kalman smoother output")
    return stats.smoother


def robust_smoother(
    params: EMParameters,
    layout: StateLayout,
    observations: FloatArray,
    spec: RobustSpec,
    state: RobustState,
    method: str = "univariate",
) -> SufficientStatistics:
    """E-step of the robust EM (also used for the final smoothing).

    Parameters
    ----------
    params : EMParameters
        Parameters.
    layout : StateLayout
        State layout.
    observations : numpy.ndarray, shape (n_periods, N)
        Data already masked (excluded observations and outliers set to ``NaN``).
    spec : RobustSpec
        Options (intervention window, padded to ``n_periods`` if needed).
    state : RobustState
        Weights and dummies (``n_periods`` rows).
    method : str, default "univariate"
        Filter variant (Gaussian errors). The robust E-step always uses the dense Kalman
        smoother: ``"auto"`` and ``"structured"`` mean ``"univariate"``.

    Returns
    -------
    SufficientStatistics
        Smoother output (states include the dummy offset) and moments (profiled over
        the dummies).

    Examples
    --------
    >>> import numpy as np
    >>> from nowcastbox.models._em_steps import EMParameters, StateLayout
    >>> from nowcastbox.models.robust import RobustSpec, RobustState, robust_smoother
    >>> lay = StateLayout(["a", "b"], ["g"], [1], 1, np.ones((2, 1), bool), [[1.0]] * 2, "iid")
    >>> p = EMParameters(
    ...     (np.array([[0.5]]),), (np.eye(1),), np.ones((2, 1)), np.zeros(2), np.ones(2), np.ones(2)
    ... )
    >>> y = np.random.default_rng(0).standard_normal((10, 2))
    >>> st = RobustState(np.zeros((10, 2), bool))
    >>> robust_smoother(p, lay, y, RobustSpec(), st).n_pairs
    9
    """
    model = build_state_space(params, layout, observations.shape[0])
    obs_var = None
    if state.q is not None:
        obs_var = params.obs_var[None, :] / state.q.weights
    offset = None
    free_states = None
    free_periods = None
    if spec.has_intervention and state.dummies is not None:
        offset = intervention_offset(state.dummies, params, layout)
        free_periods = spec.intervention
        free_states = np.arange(layout.n_factor_states)
    return e_step(
        model,
        observations,
        # the outlier pass needs the filter's one-step predictions: dense smoother only
        method="multivariate" if method == "multivariate" else "univariate",
        obs_var=obs_var,
        state_offset=offset,
        free_periods=free_periods,
        free_states=free_states,
    )


def _objective(stats: SufficientStatistics, state: RobustState, observed: BoolArray) -> float:
    value = stats.loglikelihood
    if state.q is not None and state.df is not None:
        value += student_t_bound(state.q, state.df, observed)
    return value


def _robust_m_step(
    stats: SufficientStatistics,
    params: EMParameters,
    layout: StateLayout,
    observations: FloatArray,
    spec: RobustSpec,
    state: RobustState,
) -> tuple[EMParameters, RobustState]:
    q, df, dummies = state.q, state.df, state.dummies
    weights = None
    if q is not None and df is not None:
        delta = expected_squared_residuals(_dense(stats), observations) / params.obs_var
        q = StudentTWeights.posterior(delta, df)
        weights = q.weights
    new = m_step(stats, layout, observations, params, weights=weights)
    if q is not None and spec.df is None:
        df = update_df(q, ~np.isnan(observations))
    if spec.has_intervention:
        assert spec.intervention is not None  # noqa: S101
        dummies = update_interventions(
            stats.smoother.smoothed_state, new, layout, spec.intervention
        )
    return new, RobustState(state.flags, q, df, dummies)


def _em_pass(
    params: EMParameters,
    layout: StateLayout,
    observations: FloatArray,
    spec: RobustSpec,
    state: RobustState,
    *,
    max_iter: int,
    tol: float,
    method: str,
) -> tuple[RobustEMResult, SufficientStatistics]:
    observed = ~np.isnan(observations)
    path: list[float] = []
    converged, n_decreases, change, n_iter = False, 0, float("nan"), 0
    while True:
        stats = robust_smoother(params, layout, observations, spec, state, method)
        value = _objective(stats, state, observed)
        if not np.isfinite(value):
            raise NowcastDataError(f"Non-finite objective at robust EM iteration {n_iter}.")
        path.append(value)
        if n_iter > 0:
            converged, change = em_converged(value, path[-2], tol)
            n_decreases += int(change < -_DECREASE_TOL)
        logger.debug("robust EM iteration %d: objective %.6f", n_iter, value)
        if converged or n_iter == max_iter:
            break
        params, state = _robust_m_step(stats, params, layout, observations, spec, state)
        n_iter += 1
    result = RobustEMResult(params, np.asarray(path), converged, n_iter, n_decreases, change, state)
    return result, stats


def _initial_state(shape: tuple[int, int], layout: StateLayout, spec: RobustSpec) -> RobustState:
    q, df, dummies = None, None, None
    if spec.student_t:
        df = DF_START if spec.df is None else float(spec.df)
        q = StudentTWeights.prior(shape, df)
    if spec.has_intervention:
        dummies = np.zeros((shape[0], layout.total_factors))
    return RobustState(np.zeros(shape, dtype=bool), q, df, dummies)


def run_robust_em(
    params: EMParameters,
    layout: StateLayout,
    observations: FloatArray,
    spec: RobustSpec,
    *,
    max_iter: int = 500,
    tol: float = 1e-4,
    method: str = "univariate",
) -> RobustEMResult:
    """Robust EM: Student-t errors, outlier passes, excluded periods and dummies.

    Parameters
    ----------
    params : EMParameters
        Starting values.
    layout : StateLayout
        State layout (``"iid"`` when ``spec.student_t``).
    observations : numpy.ndarray, shape (n_periods, N)
        Standardised data (``NaN`` = missing).
    spec : RobustSpec
        Robustness options.
    max_iter : int, default 500
        Maximum number of M-steps per pass.
    tol : float, default 1e-4
        Relative-change tolerance of the objective.
    method : str, default "univariate"
        Kalman filter variant (Gaussian errors).

    Returns
    -------
    RobustEMResult
        Parameters, objective path of the last pass and latent quantities.

    Raises
    ------
    ValueError
        If Student-t errors are requested with an ``"ar1"`` layout.
    NowcastDataError
        If the objective becomes non-finite.

    Examples
    --------
    >>> import numpy as np
    >>> from nowcastbox.models._em_steps import EMParameters, StateLayout
    >>> from nowcastbox.models.robust import RobustSpec, run_robust_em
    >>> lay = StateLayout(["a", "b"], ["g"], [1], 1, np.ones((2, 1), bool), [[1.0]] * 2, "iid")
    >>> p = EMParameters(
    ...     (np.array([[0.5]]),), (np.eye(1),), np.ones((2, 1)), np.zeros(2), np.ones(2), np.ones(2)
    ... )
    >>> y = np.random.default_rng(0).standard_t(3, size=(60, 2))
    >>> out = run_robust_em(p, lay, y, RobustSpec(student_t=True), max_iter=20)
    >>> bool(np.all(np.diff(out.loglikelihood_path) > -1e-6)), out.state.df is not None
    (True, True)
    """
    if spec.student_t and layout.idiosyncratic != "iid":
        raise ValueError("Student-t errors require an 'iid' idiosyncratic layout.")
    state = _initial_state(observations.shape, layout, spec)
    total_iter, n_passes = 0, 0
    while True:
        n_passes += 1
        masked = _masked(observations, spec, state.flags)
        result, stats = _em_pass(
            params, layout, masked, spec, state, max_iter=max_iter, tol=tol, method=method
        )
        params, state = result.params, result.state
        total_iter += result.n_iter
        if not spec.outliers or n_passes >= spec.max_outlier_passes:
            break
        offset = None
        if spec.has_intervention and state.dummies is not None:
            offset = intervention_offset(state.dummies, params, layout)
        unflagged = _masked(observations, spec, np.zeros_like(state.flags))
        z = standardized_innovations(_dense(stats), unflagged, offset)
        flags = flag_outliers(z, spec.outlier_threshold)
        logger.debug("outlier pass %d: %d observations flagged", n_passes, int(flags.sum()))
        if np.array_equal(flags, state.flags):
            break
        state = replace(state, flags=flags)
    return replace(result, state=state, n_passes=n_passes, total_iter=total_iter)
