r"""Blocking of mixed-frequency panels and conditional forecasts of a VAR.

Building blocks of :mod:`nowcastbox.models.bvar` (Phase 3 of the ECB-parity plan).

Blocking (stacking)
-------------------
Following Cimadomo, Giannone, Lenza, Monti & Sokol (2022, section 2.2), every monthly
series :math:`x_{t_m}` is re-organised as three quarterly series, one per month of the
quarter,

.. math::

    x^q_{t_q} = (x_{3t_q - 2},\; x_{3t_q - 1},\; x_{3t_q}),

and quarterly series enter as they are. The blocked vector
:math:`Y_{t_q} = (y_{t_q}', x^{q\,\prime}_{t_q})'` of size :math:`n = Q + 3M` follows a
quarterly VAR(:math:`p`)

.. math::

    Y_{t} = c + A_1 Y_{t-1} + \dots + A_p Y_{t-p} + \varepsilon_t, \qquad
    \varepsilon_t \sim N(0, \Sigma).

Because each month is its own variable, no aggregation rule (flow, stock, average) is
imposed: the VAR learns the relation between monthly and quarterly variables.

Conditional forecasts
---------------------
At the ragged edge some entries of the last blocked vectors are released and others
are not. With the parameters fixed, the nowcast is the expectation of the missing
entries conditional on the released ones (Waggoner & Zha, 1999; Bańbura, Giannone &
Lenza, 2015). It is computed with the Kalman smoother of :mod:`nowcastbox.statespace`
on the companion form (Durbin & Koopman notation)

.. math::

    \alpha_{t+1} = T\alpha_t + c^\ast + R\eta_t,\quad
    \alpha_t = (Y_t', \dots, Y_{t-p+1}')',\quad
    Y_t = Z\alpha_t \;(H = 0),

with :math:`T` the companion matrix, :math:`R = (I_n, 0)'`, :math:`Q = \Sigma`, and
``NaN`` for the unreleased entries. The state of the first period of the edge is
initialised from the :math:`p` pre-sample rows (:func:`initial_state`): when they are
complete the initial state is exact, so only the data from the last :math:`p` complete
quarters onwards matter (Markov property of the VAR).

:func:`conditional_moments` computes the same moments in closed form from the moving
average representation :math:`Y_{T_0+i} = \mu_i + \sum_{s \le i} \Phi_{i-s}
\varepsilon_{T_0+s}` (a joint Gaussian vector conditioned on its observed entries); it
is used for the posterior draws, where it is much cheaper than a smoother pass.

References
----------
Bańbura, M., Giannone, D., & Lenza, M. (2015). Conditional forecasts and scenario
analysis with vector autoregressions for large cross-sections. *International Journal
of Forecasting*, 31(3), 739-756.

Cimadomo, J., Giannone, D., Lenza, M., Monti, F., & Sokol, A. (2022). Nowcasting with
large Bayesian vector autoregressions. *Journal of Econometrics*, 231(2), 500-519.

Durbin, J., & Koopman, S. J. (2012). *Time Series Analysis by State Space Methods*
(2nd ed.). Oxford University Press.

Waggoner, D. F., & Zha, T. (1999). Conditional forecasts in dynamic multivariate
models. *Review of Economics and Statistics*, 81(4), 639-651.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.linalg

from nowcastbox.core.data import MixedFrequencyData
from nowcastbox.core.exceptions import NowcastDataError
from nowcastbox.core.frequency import Frequency
from nowcastbox.statespace import SmootherResult, StateSpace, companion_matrix, kalman_smoother

__all__ = [
    "MONTHS_PER_QUARTER",
    "BlockedLayout",
    "ConditionalForecast",
    "VARParameters",
    "balanced_run",
    "block_values",
    "blocked_layout",
    "conditional_forecast",
    "conditional_moments",
    "initial_state",
    "native_periods",
    "unblock_values",
    "var_state_space",
    "window_end",
]

MONTHS_PER_QUARTER = 3
_DIFFUSE_VARIANCE = 1e4
_SUPPORTED = (Frequency.MONTHLY, Frequency.QUARTERLY)


# ====================================================================== blocking
@dataclass(frozen=True)
class BlockedLayout:
    """Correspondence between the series of a panel and the blocked quarterly variables.

    Parameters
    ----------
    series : tuple of str
        Original series, in panel order.
    frequencies : tuple of Frequency
        Native frequency of each series (monthly or quarterly).
    columns : tuple of str
        Names of the blocked variables: ``"<series>[m1]"``, ``"<series>[m2]"``,
        ``"<series>[m3]"`` for a monthly series, ``"<series>"`` for a quarterly one.
    source : tuple of int
        Index (in ``series``) of the series of each blocked variable.
    month : tuple of int
        Month of the quarter (1, 2, 3) of each blocked variable; 3 (the storage slot)
        for quarterly series.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from nowcastbox.core.data import MixedFrequencyData
    >>> idx = pd.period_range("2020-01", periods=6, freq="M")
    >>> df = pd.DataFrame({"ip": np.arange(6.0), "gdp": [np.nan, np.nan, 1.0] * 2}, index=idx)
    >>> layout = blocked_layout(MixedFrequencyData(df, {"ip": "M", "gdp": "Q"}))
    >>> layout.columns
    ('ip[m1]', 'ip[m2]', 'ip[m3]', 'gdp')
    >>> layout.columns_of("ip")
    [0, 1, 2]
    """

    series: tuple[str, ...]
    frequencies: tuple[Frequency, ...]
    columns: tuple[str, ...]
    source: tuple[int, ...]
    month: tuple[int, ...]

    @property
    def n(self) -> int:
        """Number of blocked variables.

        Returns
        -------
        int
            :math:`n = Q + 3M`.

        Examples
        --------
        >>> BlockedLayout(
        ...     ("a",), (Frequency.MONTHLY,), ("a[m1]", "a[m2]", "a[m3]"), (0, 0, 0), (1, 2, 3)
        ... ).n
        3
        """
        return len(self.columns)

    def columns_of(self, name: str) -> list[int]:
        """Blocked variables of one series.

        Parameters
        ----------
        name : str
            Series name.

        Returns
        -------
        list of int
            Positions in :attr:`columns` (three for a monthly series, one otherwise).

        Raises
        ------
        KeyError
            If ``name`` is not a series of the layout.

        Examples
        --------
        >>> BlockedLayout(("q",), (Frequency.QUARTERLY,), ("q",), (0,), (3,)).columns_of("q")
        [0]
        """
        if name not in self.series:
            raise KeyError(f"Unknown series {name!r}.")
        i = self.series.index(name)
        return [c for c, s in enumerate(self.source) if s == i]

    def source_name(self, column: int) -> str:
        """Original series of a blocked variable.

        Parameters
        ----------
        column : int
            Position in :attr:`columns`.

        Returns
        -------
        str
            Series name.

        Examples
        --------
        >>> BlockedLayout(("q",), (Frequency.QUARTERLY,), ("q",), (0,), (3,)).source_name(0)
        'q'
        """
        return self.series[self.source[column]]


def blocked_layout(data: MixedFrequencyData) -> BlockedLayout:
    """Blocked layout of a monthly panel with monthly and quarterly series.

    Parameters
    ----------
    data : MixedFrequencyData
        Panel on a monthly base grid.

    Returns
    -------
    BlockedLayout
        One blocked variable per month for monthly series, one for quarterly series.

    Raises
    ------
    NowcastDataError
        If the base grid is not monthly or a series is neither monthly nor quarterly.

    Examples
    --------
    >>> import pandas as pd
    >>> from nowcastbox.core.data import MixedFrequencyData
    >>> idx = pd.period_range("2020-01", periods=3, freq="M")
    >>> blocked_layout(MixedFrequencyData(pd.DataFrame({"x": [1.0, 2, 3]}, idx), "M")).month
    (1, 2, 3)
    """
    if data.base_frequency is not Frequency.MONTHLY:
        raise NowcastDataError(
            f"The blocked BVAR needs a monthly base grid, got {data.base_frequency.label}."
        )
    series, freqs, columns, source, month = [], [], [], [], []
    for i, name in enumerate(data.columns):
        freq = data.metadata[name].frequency
        if freq not in _SUPPORTED:
            raise NowcastDataError(
                f"Series {name!r} is {freq.label}; the blocked BVAR supports monthly and "
                "quarterly series only."
            )
        series.append(name)
        freqs.append(freq)
        months = (1, 2, 3) if freq is Frequency.MONTHLY else (MONTHS_PER_QUARTER,)
        for m in months:
            columns.append(f"{name}[m{m}]" if freq is Frequency.MONTHLY else name)
            source.append(i)
            month.append(m)
    return BlockedLayout(tuple(series), tuple(freqs), tuple(columns), tuple(source), tuple(month))


def _quarter_grid(index: pd.PeriodIndex) -> tuple[pd.PeriodIndex, pd.PeriodIndex]:
    """Quarters covering a monthly index and the months of those quarters."""
    first = index[0].asfreq("Q")
    last = index[-1].asfreq("Q")
    quarters = pd.period_range(first, last, freq="Q")
    months = pd.period_range(first.asfreq("M", how="S"), last.asfreq("M", how="E"), freq="M")
    return quarters, months


def block_values(frame: pd.DataFrame, layout: BlockedLayout) -> tuple[pd.PeriodIndex, np.ndarray]:
    """Blocked quarterly matrix of a monthly panel.

    Parameters
    ----------
    frame : pandas.DataFrame
        Panel on a monthly :class:`pandas.PeriodIndex` containing every series of the
        layout (extra columns are ignored). Months outside the panel are missing.
    layout : BlockedLayout
        Layout.

    Returns
    -------
    quarters : pandas.PeriodIndex
        Quarters covering the panel.
    values : numpy.ndarray
        Shape ``(len(quarters), layout.n)``; ``NaN`` = missing.

    Raises
    ------
    NowcastDataError
        If the frame is not monthly or lacks series of the layout.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> lay = BlockedLayout(
    ...     ("x",), (Frequency.MONTHLY,), ("x[m1]", "x[m2]", "x[m3]"), (0, 0, 0), (1, 2, 3)
    ... )
    >>> idx = pd.period_range("2020-02", periods=5, freq="M")
    >>> q, v = block_values(pd.DataFrame({"x": np.arange(5.0)}, index=idx), lay)
    >>> [str(p) for p in q], v.tolist()
    (['2020Q1', '2020Q2'], [[nan, 0.0, 1.0], [2.0, 3.0, 4.0]])
    """
    index = frame.index
    if not isinstance(index, pd.PeriodIndex) or index.freqstr != "M" or len(index) == 0:
        raise NowcastDataError("The panel must be indexed by a non-empty monthly PeriodIndex.")
    missing = [s for s in layout.series if s not in frame.columns]
    if missing:
        raise NowcastDataError(f"The data do not contain the model series {missing}.")
    quarters, months = _quarter_grid(index)
    raw = frame.loc[:, list(layout.series)].reindex(months).to_numpy(dtype=np.float64)
    cube = raw.reshape(len(quarters), MONTHS_PER_QUARTER, len(layout.series))
    month = np.asarray(layout.month) - 1
    return quarters, cube[:, month, np.asarray(layout.source)]


def unblock_values(
    values: np.ndarray, quarters: pd.PeriodIndex, layout: BlockedLayout
) -> pd.DataFrame:
    """Monthly panel from a blocked quarterly matrix (inverse of :func:`block_values`).

    Parameters
    ----------
    values : numpy.ndarray
        Shape ``(len(quarters), layout.n)``.
    quarters : pandas.PeriodIndex
        Quarterly grid.
    layout : BlockedLayout
        Layout.

    Returns
    -------
    pandas.DataFrame
        Monthly grid of the quarters; quarterly series in their storage slot (third
        month), ``NaN`` elsewhere.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> lay = BlockedLayout(("q",), (Frequency.QUARTERLY,), ("q",), (0,), (3,))
    >>> q = pd.period_range("2020Q1", periods=1, freq="Q")
    >>> unblock_values(np.array([[2.0]]), q, lay)["q"].tolist()
    [nan, nan, 2.0]
    """
    n_q = len(quarters)
    cube = np.full((n_q, MONTHS_PER_QUARTER, len(layout.series)), np.nan)
    cube[:, np.asarray(layout.month) - 1, np.asarray(layout.source)] = values
    months = pd.period_range(
        quarters[0].asfreq("M", how="S"), quarters[-1].asfreq("M", how="E"), freq="M"
    )
    flat = cube.reshape(n_q * MONTHS_PER_QUARTER, len(layout.series))
    return pd.DataFrame(flat, index=months, columns=list(layout.series))


def native_periods(quarters: pd.PeriodIndex, layout: BlockedLayout, column: int) -> pd.PeriodIndex:
    """Native periods of the observations of one blocked variable.

    Parameters
    ----------
    quarters : pandas.PeriodIndex
        Quarterly grid.
    layout : BlockedLayout
        Layout.
    column : int
        Blocked variable.

    Returns
    -------
    pandas.PeriodIndex
        The quarters (quarterly series) or the given month of each quarter.

    Examples
    --------
    >>> import pandas as pd
    >>> lay = BlockedLayout(
    ...     ("x",), (Frequency.MONTHLY,), ("x[m1]", "x[m2]", "x[m3]"), (0, 0, 0), (1, 2, 3)
    ... )
    >>> q = pd.period_range("2020Q1", periods=2, freq="Q")
    >>> [str(p) for p in native_periods(q, lay, 1)]
    ['2020-02', '2020-05']
    """
    if layout.frequencies[layout.source[column]] is Frequency.QUARTERLY:
        return quarters
    return quarters.asfreq("M", how="S").shift(layout.month[column] - 1)


def balanced_run(values: np.ndarray) -> tuple[int, int]:
    """Longest run of complete rows ending at the last complete row.

    Parameters
    ----------
    values : numpy.ndarray
        Blocked matrix ``(T, n)`` with ``NaN`` for missing entries.

    Returns
    -------
    tuple of int
        ``(start, end)`` rows (inclusive) of the balanced estimation sample.

    Raises
    ------
    NowcastDataError
        If no row is complete.

    Examples
    --------
    >>> import numpy as np
    >>> v = np.array([[1.0], [np.nan], [1.0], [2.0], [np.nan]])
    >>> balanced_run(v)
    (2, 3)
    """
    complete = np.asarray(np.isfinite(values).all(axis=1), dtype=bool)
    rows = np.flatnonzero(complete)
    if rows.size == 0:
        raise NowcastDataError("No quarter has every blocked variable observed.")
    end = int(rows[-1])
    start = end
    while start > 0 and bool(complete[start - 1]):
        start -= 1
    return start, end


def window_end(values: np.ndarray, lags: int) -> int:
    """Last row ``r`` such that rows ``r - lags + 1 .. r`` are complete.

    Parameters
    ----------
    values : numpy.ndarray
        Blocked matrix ``(T, n)``.
    lags : int
        VAR order :math:`p`.

    Returns
    -------
    int
        Row of the end of the last complete window of ``lags`` quarters.

    Raises
    ------
    NowcastDataError
        If no window of ``lags`` complete quarters exists.

    Examples
    --------
    >>> import numpy as np
    >>> v = np.array([[1.0], [2.0], [np.nan], [3.0]])
    >>> window_end(v, 2), window_end(v, 1)
    (1, 3)
    """
    complete = np.isfinite(values).all(axis=1).astype(int)
    if complete.size >= lags:
        runs = np.convolve(complete, np.ones(lags, dtype=int), mode="valid")
        ends = np.flatnonzero(np.equal(runs, lags))
        if ends.size:
            return int(ends[-1]) + lags - 1
    raise NowcastDataError(f"The data have no {lags} consecutive complete quarters.")


# ====================================================================== VAR parameters
@dataclass(frozen=True)
class VARParameters:
    r"""Parameters of :math:`Y_t = c + \sum_{l=1}^p A_l Y_{t-l} + \varepsilon_t`.

    Parameters
    ----------
    intercept : numpy.ndarray
        :math:`c`, shape ``(n,)``.
    coefficients : numpy.ndarray
        :math:`(A_1, \dots, A_p)`, shape ``(p, n, n)``.
    sigma : numpy.ndarray
        :math:`\Sigma`, shape ``(n, n)``.

    Examples
    --------
    >>> import numpy as np
    >>> B = np.array([[0.1], [0.5]])  # constant, lag 1 (stacked form, k x n)
    >>> par = VARParameters.from_stacked(B, np.eye(1), lags=1)
    >>> par.coefficients.tolist(), par.intercept.tolist()
    ([[[0.5]]], [0.1])
    """

    intercept: np.ndarray
    coefficients: np.ndarray
    sigma: np.ndarray

    @classmethod
    def from_stacked(
        cls, B: np.ndarray, sigma: np.ndarray, lags: int, *, constant: bool = True
    ) -> VARParameters:
        """Parameters from the stacked form :math:`Y = XB + E`.

        Parameters
        ----------
        B : numpy.ndarray
            Shape ``(k, n)``: the constant row (if any) then lag 1 to lag ``p`` blocks
            of ``n`` rows (:class:`~nowcastbox.models._bvar_prior.VARSystem` order).
        sigma : numpy.ndarray
            Shape ``(n, n)``.
        lags : int
            :math:`p`.
        constant : bool, default True
            Whether the first row of ``B`` is the constant.

        Returns
        -------
        VARParameters
            Parameters with :math:`A_l = B_l'`.

        Examples
        --------
        >>> import numpy as np
        >>> VARParameters.from_stacked(np.eye(2), np.eye(2), 1, constant=False).lags
        1
        """
        B = np.asarray(B, dtype=np.float64)
        n = B.shape[1]
        start = int(constant)
        intercept = B[0].copy() if constant else np.zeros(n)
        blocks = B[start : start + lags * n].reshape(lags, n, n)
        return cls(intercept, np.ascontiguousarray(blocks.transpose(0, 2, 1)), np.asarray(sigma))

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
        >>> VARParameters(np.zeros(2), np.zeros((1, 2, 2)), np.eye(2)).n
        2
        """
        return int(self.intercept.shape[0])

    @property
    def lags(self) -> int:
        """VAR order.

        Returns
        -------
        int
            :math:`p`.

        Examples
        --------
        >>> import numpy as np
        >>> VARParameters(np.zeros(1), np.zeros((3, 1, 1)), np.eye(1)).lags
        3
        """
        return int(self.coefficients.shape[0])

    def companion(self) -> np.ndarray:
        """Companion matrix :math:`T` of the VAR.

        Returns
        -------
        numpy.ndarray
            Shape ``(n p, n p)``.

        Examples
        --------
        >>> import numpy as np
        >>> VARParameters(np.zeros(1), np.array([[[0.5]], [[0.2]]]), np.eye(1)).companion()
        array([[0.5, 0.2],
               [1. , 0. ]])
        """
        return companion_matrix(list(self.coefficients))

    def one_step(self, history: np.ndarray) -> np.ndarray:
        r"""One-step prediction :math:`c + \sum_l A_l y_{t-l}`.

        Parameters
        ----------
        history : numpy.ndarray
            The last ``p`` observations, shape ``(p, n)``, oldest first.

        Returns
        -------
        numpy.ndarray
            Shape ``(n,)``.

        Examples
        --------
        >>> import numpy as np
        >>> par = VARParameters(np.ones(1), np.array([[[0.5]]]), np.eye(1))
        >>> par.one_step(np.array([[2.0]])).tolist()
        [2.0]
        """
        out = self.intercept.copy()
        for lag in range(1, self.lags + 1):
            out += self.coefficients[lag - 1] @ history[-lag]
        return out

    def fitted(self, values: np.ndarray) -> np.ndarray:
        """One-step predictions for every row with ``p`` previous rows.

        Parameters
        ----------
        values : numpy.ndarray
            Shape ``(T, n)``.

        Returns
        -------
        numpy.ndarray
            Shape ``(T, n)``: row ``t`` is the prediction from rows ``t-p .. t-1``
            (``NaN`` for the first ``p`` rows).

        Examples
        --------
        >>> import numpy as np
        >>> par = VARParameters(np.zeros(1), np.array([[[0.5]]]), np.eye(1))
        >>> par.fitted(np.array([[2.0], [4.0]])).ravel().tolist()
        [nan, 1.0]
        """
        out = np.full_like(values, np.nan, dtype=np.float64)
        out[self.lags :] = self.intercept
        for lag in range(1, self.lags + 1):
            out[self.lags :] += (
                values[self.lags - lag : len(values) - lag] @ self.coefficients[lag - 1].T
            )
        return out

    def impulse_responses(self, horizon: int) -> np.ndarray:
        r"""Moving-average coefficients :math:`\Phi_0 = I, \Phi_j = \sum_l A_l \Phi_{j-l}`.

        Parameters
        ----------
        horizon : int
            Number of coefficients :math:`h` (:math:`\Phi_0 .. \Phi_{h-1}`).

        Returns
        -------
        numpy.ndarray
            Shape ``(h, n, n)``.

        Examples
        --------
        >>> import numpy as np
        >>> par = VARParameters(np.zeros(1), np.array([[[0.5]]]), np.eye(1))
        >>> par.impulse_responses(3).ravel().tolist()
        [1.0, 0.5, 0.25]
        """
        n = self.n
        phi = np.zeros((horizon, n, n))
        if horizon:
            phi[0] = np.eye(n)
        for j in range(1, horizon):
            for lag in range(1, min(j, self.lags) + 1):
                phi[j] += self.coefficients[lag - 1] @ phi[j - lag]
        return phi


# ====================================================================== state space
def initial_state(params: VARParameters, presample: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r"""Mean and covariance of :math:`\alpha_s = (Y_s', \dots, Y_{s-p+1}')'` given the pre-sample.

    The ``p`` pre-sample rows :math:`Y_{s-p}, \dots, Y_{s-1}` are taken as known;
    missing entries get mean 0 and a large variance (approximate diffuse, standardised
    units). Then :math:`Y_s = c + \sum_l A_l Y_{s-l} + \varepsilon_s`.

    Parameters
    ----------
    params : VARParameters
        VAR parameters.
    presample : numpy.ndarray
        Shape ``(p, n)``, oldest first, ``NaN`` = missing.

    Returns
    -------
    a0 : numpy.ndarray
        Shape ``(n p,)``.
    P0 : numpy.ndarray
        Shape ``(n p, n p)``; exact (lag blocks zero, first block :math:`\Sigma`) when
        the pre-sample is complete.

    Raises
    ------
    ValueError
        If ``presample`` does not have ``p`` rows of ``n`` values.

    Examples
    --------
    >>> import numpy as np
    >>> par = VARParameters(np.zeros(1), np.array([[[0.5]]]), np.eye(1))
    >>> a0, P0 = initial_state(par, np.array([[2.0]]))
    >>> a0.tolist(), P0.tolist()
    ([1.0], [[1.0]])
    """
    n, p = params.n, params.lags
    if presample.shape != (p, n):
        raise ValueError(f"presample must have shape {(p, n)}, got {presample.shape}.")
    missing = ~np.isfinite(presample)
    known = np.where(missing, 0.0, presample)
    var = np.where(missing, _DIFFUSE_VARIANCE, 0.0)  # (p, n), row -l = lag l
    a0 = np.zeros(n * p)
    P0 = np.zeros((n * p, n * p))
    a0[:n] = params.one_step(known)
    P0[:n, :n] = params.sigma
    for lag in range(1, p + 1):
        A = params.coefficients[lag - 1]
        v = var[-lag]
        P0[:n, :n] += (A * v) @ A.T
        if lag < p:
            block = slice(lag * n, (lag + 1) * n)
            a0[block] = known[-lag]
            P0[block, block] = np.diag(v)
            P0[:n, block] = A * v
            P0[block, :n] = (A * v).T
    return a0, 0.5 * (P0 + P0.T)


def var_state_space(params: VARParameters, presample: np.ndarray) -> StateSpace:
    """Companion-form state-space model of the VAR, initialised from the pre-sample.

    Parameters
    ----------
    params : VARParameters
        VAR parameters.
    presample : numpy.ndarray
        Shape ``(p, n)``: the ``p`` rows before the first period of the model.

    Returns
    -------
    StateSpace
        ``alpha_{t+1} = T alpha_t + c* + R eta_t``, ``Y_t = Z alpha_t`` with ``H = 0``.

    Examples
    --------
    >>> import numpy as np
    >>> par = VARParameters(np.zeros(2), np.zeros((2, 2, 2)), np.eye(2))
    >>> var_state_space(par, np.zeros((2, 2))).n_states
    4
    """
    n, p = params.n, params.lags
    m = n * p
    Z = np.zeros((n, m))
    Z[:, :n] = np.eye(n)
    R = np.zeros((m, n))
    R[:n] = np.eye(n)
    c = np.zeros(m)
    c[:n] = params.intercept
    a0, P0 = initial_state(params, presample)
    return StateSpace(
        params.companion(),
        Z,
        params.sigma,
        np.zeros(n),
        selection=R,
        state_intercept=c,
        initial_state=a0,
        initial_state_cov=P0,
    )


@dataclass(frozen=True)
class ConditionalForecast:
    r"""Moments of the edge of the blocked panel given the released data.

    Parameters
    ----------
    mean : numpy.ndarray
        :math:`E[Y_t \mid \text{data}]`, shape ``(h, n)`` (released entries unchanged).
    variance : numpy.ndarray
        Conditional variances, shape ``(h, n)`` (zero for released entries).
    smoother : SmootherResult
        Output of the Kalman smoother.

    Examples
    --------
    >>> import numpy as np
    >>> par = VARParameters(np.zeros(1), np.array([[[0.5]]]), np.eye(1))
    >>> fc = conditional_forecast(par, np.array([[2.0]]), np.array([[np.nan]]))
    >>> fc.mean.tolist(), fc.variance.tolist()
    ([[1.0]], [[1.0]])
    """

    mean: np.ndarray
    variance: np.ndarray
    smoother: SmootherResult


def conditional_forecast(
    params: VARParameters, presample: np.ndarray, future: np.ndarray
) -> ConditionalForecast:
    """Conditional forecast of the edge with the Kalman smoother.

    Parameters
    ----------
    params : VARParameters
        VAR parameters.
    presample : numpy.ndarray
        Shape ``(p, n)``: rows before the edge (normally complete).
    future : numpy.ndarray
        Shape ``(h, n)``: the edge, ``NaN`` for unreleased entries.

    Returns
    -------
    ConditionalForecast
        Smoothed means and variances of :math:`Y_t` on the edge.

    Examples
    --------
    >>> import numpy as np
    >>> par = VARParameters(np.zeros(2), np.array([[[0.5, 0.0], [0.0, 0.5]]]), np.eye(2))
    >>> fc = conditional_forecast(par, np.zeros((1, 2)), np.array([[1.0, np.nan]]))
    >>> fc.mean.tolist()
    [[1.0, 0.0]]
    """
    model = var_state_space(params, presample)
    sm = kalman_smoother(model, future, method="univariate")
    n = params.n
    mean = sm.smoothed_state[:, :n].copy()
    var = np.einsum("tii->ti", sm.smoothed_state_cov[:, :n, :n]).copy()
    observed = np.isfinite(future)
    mean[observed] = future[observed]
    var[observed] = 0.0
    return ConditionalForecast(mean, np.maximum(var, 0.0), sm)


def _forecast_path(params: VARParameters, presample: np.ndarray, horizon: int) -> np.ndarray:
    """Unconditional forecasts of ``horizon`` rows after a complete pre-sample."""
    history = list(presample)
    out = np.empty((horizon, params.n))
    for i in range(horizon):
        out[i] = params.one_step(np.asarray(history[-params.lags :]))
        history.append(out[i])
    return out


def _ma_matrix(params: VARParameters, horizon: int) -> np.ndarray:
    r"""Block lower-triangular :math:`M` with :math:`\operatorname{vec}(Y') = \mu + M u`, :math:`u \sim N(0, I)`."""
    n = params.n
    G = np.linalg.cholesky(params.sigma)
    phi_g = params.impulse_responses(horizon) @ G
    M = np.zeros((horizon * n, horizon * n))
    for i in range(horizon):
        for s in range(i + 1):
            M[i * n : (i + 1) * n, s * n : (s + 1) * n] = phi_g[i - s]
    return M


def conditional_moments(
    params: VARParameters, presample: np.ndarray, future: np.ndarray, cells: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    r"""Closed-form conditional mean and variance of some edge entries.

    With a complete pre-sample the edge is the Gaussian vector
    :math:`\operatorname{vec}(Y') = \mu + M u` (:math:`M` from the moving-average
    coefficients and :math:`\Sigma^{1/2}`); conditioning on the observed entries
    :math:`o` gives :math:`\mu_c + M_c M_o'(M_o M_o')^{-1}(y_o - \mu_o)` and
    :math:`M_c M_c' - M_c M_o'(M_o M_o')^{-1} M_o M_c'` (diagonal only).

    Parameters
    ----------
    params : VARParameters
        VAR parameters.
    presample : numpy.ndarray
        Shape ``(p, n)``, complete.
    future : numpy.ndarray
        Shape ``(h, n)``, ``NaN`` for unreleased entries.
    cells : numpy.ndarray
        Shape ``(c, 2)``: ``(row, column)`` pairs of the wanted entries.

    Returns
    -------
    mean, variance : numpy.ndarray
        Shape ``(c,)`` each.

    Raises
    ------
    ValueError
        If the pre-sample has missing values.

    Examples
    --------
    >>> import numpy as np
    >>> par = VARParameters(np.zeros(1), np.array([[[0.5]]]), np.eye(1))
    >>> m, v = conditional_moments(
    ...     par, np.array([[2.0]]), np.full((2, 1), np.nan), np.array([[1, 0]])
    ... )
    >>> m.tolist(), v.tolist()
    ([0.5], [1.25])
    """
    if not np.isfinite(presample).all():
        raise ValueError("conditional_moments needs a complete pre-sample.")
    h, n = future.shape
    mu = _forecast_path(params, presample, h).ravel()
    M = _ma_matrix(params, h)
    flat = future.ravel()
    obs = np.flatnonzero(np.isfinite(flat))
    idx = np.asarray(cells, dtype=np.intp).reshape(-1, 2)
    tgt = idx[:, 0] * n + idx[:, 1]
    Mt = M[tgt]
    mean = mu[tgt].copy()
    var = np.einsum("ij,ij->i", Mt, Mt)
    if obs.size:
        Mo = M[obs]
        chol = scipy.linalg.cholesky(Mo @ Mo.T, lower=True)
        W = scipy.linalg.solve_triangular(chol, Mo @ Mt.T, lower=True)
        z = scipy.linalg.solve_triangular(chol, flat[obs] - mu[obs], lower=True)
        mean += W.T @ z
        var -= np.einsum("ij,ij->j", W, W)
    return mean, np.maximum(var, 0.0)
