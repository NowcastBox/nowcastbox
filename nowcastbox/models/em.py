r"""Mixed-frequency dynamic factor model estimated by EM (Bańbura & Modugno, 2014).

:class:`MixedFreqDFM` estimates by maximum likelihood, with the EM algorithm, a dynamic
factor model for panels with an arbitrary pattern of missing data (ragged edges,
series starting at different dates, mixed frequencies):

.. math::

    x_{i,t} = \sum_{l} \lambda_{i,l}' f^{(i)}_{t-l} + \sum_l w_{i,l}\, e_{i,t-l}
              + \varepsilon_{i,t}, \qquad
    f^b_t = \sum_{k=1}^{p} A^b_k f^b_{t-k} + u^b_t, \qquad
    e_{i,t} = \rho_i e_{i,t-1} + v_{i,t},

on the base grid (monthly, weekly or daily). Series observed at a lower frequency (e.g.
quarterly GDP growth, stored in the last base period of the quarter) are linked to the
latent base-frequency series by aggregation weights :math:`w_i` (Mariano & Murasawa,
2003: ``(1, 2, 3, 2, 1)`` by default; flow, average and stock aggregation through the
series metadata). For fixed-ratio pairs (M -> Q, M -> A, Q -> A, D -> W) the weights
come from :func:`nowcastbox.preprocessing.aggregation.aggregation_weights` and impose the
linear restrictions :math:`\lambda_{i,l} = (w_{i,l}/w_{i,0})\lambda_{i,0}`, enforced by a
restricted M-step. For pairs whose ratio varies over the calendar (weekly or daily base
grid: W -> M, W -> Q, D -> M, D -> Q, W -> A...) the weights depend on the period
(:class:`~nowcastbox.preprocessing.aggregation.CalendarAggregation`, exact sums, means,
end-of-period values or growth rates of the low-frequency average) and the observation
equation is time varying (innovation I1): e.g. weekly financial series, monthly
activity indicators and quarterly GDP in one model on a weekly grid. Factors can be
organised in blocks (global, real, nominal, soft...), each block following an
independent VAR(p). The idiosyncratic components are AR(1)
processes kept in the state vector (``idiosyncratic="ar1"``) or white noise
(``"iid"``). The E-step uses the Kalman smoother of :mod:`nowcastbox.statespace` with
the univariate treatment of the observations (Koopman & Durbin, 2000; innovation I2).

Robustness (innovation I3, :mod:`nowcastbox.models.robust`): Student-t idiosyncratic
errors (``idiosyncratic="student_t"``, scale mixture of normals inside the EM),
automatic outlier handling (``outliers="auto"``) and explicit treatments of the
2020-2021 pandemic (``covid="mask"|"dummy"``, ``exclude_periods``). Time-varying
long-run mean of the target (innovation I4, :mod:`nowcastbox.models.long_run`):
``long_run_mean="time_varying"`` adds a random-walk mean (Antolin-Diaz, Drechsel &
Petrella, 2017).

References
----------
Antolin-Diaz, J., Drechsel, T., & Petrella, I. (2017). Tracking the slowdown in long-run
GDP growth. *Review of Economics and Statistics*, 99(2), 343-356.

Bańbura, M., & Modugno, M. (2014). Maximum likelihood estimation of factor models on
datasets with arbitrary pattern of missing data. *Journal of Applied Econometrics*,
29(1), 133-160.

Bańbura, M., Giannone, D., & Reichlin, L. (2011). Nowcasting. In *Oxford Handbook of
Economic Forecasting*, 193-224.

Doz, C., Giannone, D., & Reichlin, L. (2012). A quasi-maximum likelihood approach for
large, approximate dynamic factor models. *Review of Economics and Statistics*, 94(4).

Mariano, R. S., & Murasawa, Y. (2003). A new coincident index of business cycles based
on monthly and quarterly series. *Journal of Applied Econometrics*, 18(4), 427-443.

Shumway, R. H., & Stoffer, D. S. (1982). An approach to time series smoothing and
forecasting using the EM algorithm. *Journal of Time Series Analysis*, 3(4), 253-264.
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd
import scipy.linalg
import scipy.stats
from numpy.typing import NDArray

from nowcastbox._logging import get_logger
from nowcastbox.core.base import BaseNowcaster
from nowcastbox.core.data import MixedFrequencyData, as_mixed_frequency_data
from nowcastbox.core.exceptions import ConvergenceWarning, DataQualityWarning, NowcastDataError
from nowcastbox.core.frequency import (
    AggregationType,
    Frequency,
    base_to_native,
    is_fixed_ratio,
    is_period_end,
    native_period_bounds,
)
from nowcastbox.core.results import FactorResults, build_nowcast_frame
from nowcastbox.models._em_steps import (
    EMParameters,
    StateLayout,
    build_state_space,
    e_step,
    em_converged,
    m_step,
    signal_mean,
)
from nowcastbox.models._init_conditions import BLOCK_ORDERS, pca_initial_parameters
from nowcastbox.models.long_run import LONG_RUN_MODES, long_run_frame, resolve_long_run_series
from nowcastbox.models.robust import (
    COVID_MODES,
    COVID_WINDOW,
    OUTLIER_MODES,
    RobustSpec,
    RobustState,
    intervention_offset,
    period_mask,
    robust_smoother,
    run_robust_em,
)
from nowcastbox.preprocessing.aggregation import (
    CalendarAggregation,
    aggregation_weights,
    calendar_aggregation,
)
from nowcastbox.statespace import (
    SmootherResult,
    StateSpace,
    StructuredSmootherResult,
    kalman_smoother,
    loglikelihood,
    smoothed_moments,
)

__all__ = [
    "EMParameters",
    "MixedFreqDFM",
    "MixedFreqDFMResults",
    "StateLayout",
    "series_calendar",
]

logger = get_logger(__name__)

FloatArray = NDArray[np.float64]

_FILTER_METHODS = ("auto", "structured", "univariate", "multivariate")
_INIT_OPTIONS = ("pca", *(f"pca_{order}" for order in BLOCK_ORDERS))
_IDIOSYNCRATIC = ("ar1", "iid", "student_t")
_DEFAULT_BLOCK = "global"
_DECREASE_TOL = 1e-9
_Z68 = float(scipy.stats.norm.ppf(0.84))
_Z90 = float(scipy.stats.norm.ppf(0.95))


# ====================================================================== results
@dataclass(frozen=True, kw_only=True, eq=False, repr=False)
class MixedFreqDFMResults(FactorResults):
    r"""Results of :class:`MixedFreqDFM`.

    Besides the :class:`~nowcastbox.core.results.FactorResults` fields (``factors``:
    smoothed current factors ``<block>_f<k>`` on the base grid, including the forecast
    periods; ``loadings``: loadings on the current factors, i.e. the :math:`C` matrix
    restricted to ``f_t``; ``transition``: block-diagonal :math:`[A_1, \dots, A_p]`;
    ``shock_loadings``: :math:`Q^{1/2}`; ``idiosyncratic_variance``: idiosyncratic
    innovation variances), it stores everything needed to re-run the Kalman smoother
    on other vintages (news decomposition): the final state-space model, the state
    layout, the parameters and the standardisation statistics.

    The ``nowcast`` frame holds, at the target's native frequency, ``observed``,
    ``in_sample`` (common component of the target where it is observed),
    ``out_of_sample`` (``E[y_t | data]`` where it is not observed: backcasts,
    nowcasts, forecasts), ``common`` (common component for every period), ``std``
    (standard deviation of ``y_t`` given the data, out-of-sample periods only) and
    the Gaussian bounds ``lower_68``/``upper_68``/``lower_90``/``upper_90``.

    Parameters
    ----------
    em_parameters : EMParameters, optional
        Final parameters (standardised units).
    state_layout : StateLayout, optional
        Position of factors and idiosyncratic components in the state vector.
    state_space : StateSpace, optional
        Final state-space representation.
    loglikelihood_path : numpy.ndarray, optional
        Log-likelihood at every EM iteration (first entry: initial parameters).
    idiosyncratic_ar : pandas.Series, optional
        Idiosyncratic AR(1) coefficients (zeros for ``idiosyncratic="iid"``).
    smoothed_data : pandas.DataFrame, optional
        ``E[x_t | data]`` for every series on the (extended) base grid, original units.
    common_component : pandas.DataFrame, optional
        Common component of every series on the (extended) base grid, original units.
    smoothed_state : numpy.ndarray, optional
        Smoothed state vector on :attr:`grid`.
    grid : pandas.PeriodIndex, optional
        Base grid of the estimation sample extended to the forecast horizon.
    filter_method : str, default "auto"
        Smoother used in the E-step (see :class:`MixedFreqDFM`).
    observation_weights : pandas.DataFrame, optional
        Student-t precision weights :math:`E[\lambda_{i,t} \mid Y]` of every observation
        used in the estimation (``NaN`` elsewhere); ``idiosyncratic="student_t"`` only.
        Weights well below one mark observations treated as outliers.
    student_t_df : float, optional
        (Estimated or fixed) degrees of freedom of the Student-t errors.
    outlier_flags : pandas.DataFrame, optional
        Observations flagged as outliers and treated as missing (``outliers="auto"``).
    excluded_observations : pandas.DataFrame, optional
        Every observation treated as missing by the robust options (excluded periods,
        ``covid="mask"`` and flagged outliers); applied also by :meth:`observations`,
        :meth:`smooth` and :meth:`predict`.
    interventions : pandas.DataFrame, optional
        Impulse dummies :math:`D_t` of the factor VAR in the pandemic window
        (``covid="dummy"``; standardised units, one column per factor).
    state_offset : numpy.ndarray, optional
        Deterministic state path :math:`g_t` generated by the dummies on :attr:`grid`
        (propagated with ``T`` beyond it).
    long_run_mean : pandas.DataFrame, optional
        ``long_run_mean="time_varying"``: smoothed long-run mean of the target and of
        the ``long_run_series`` on :attr:`grid` (original units; columns ``<series>``
        and ``<series>_std``).

    Notes
    -----
    Every field of :class:`~nowcastbox.core.results.FactorResults` is also accepted (keyword-only).

    Examples
    --------
    >>> res = MixedFreqDFM().fit(data, target="gdp")  # doctest: +SKIP
    >>> res.get_nowcast()  # doctest: +SKIP
    """

    em_parameters: EMParameters | None = None
    state_layout: StateLayout | None = None
    state_space: StateSpace | None = None
    loglikelihood_path: FloatArray | None = field(default=None)
    idiosyncratic_ar: pd.Series | None = None
    smoothed_data: pd.DataFrame | None = None
    common_component: pd.DataFrame | None = None
    smoothed_state: FloatArray | None = field(default=None)
    grid: pd.PeriodIndex | None = None
    filter_method: str = "auto"
    observation_weights: pd.DataFrame | None = None
    student_t_df: float | None = None
    outlier_flags: pd.DataFrame | None = None
    excluded_observations: pd.DataFrame | None = None
    interventions: pd.DataFrame | None = None
    state_offset: FloatArray | None = field(default=None)
    long_run_mean: pd.DataFrame | None = None

    # ------------------------------------------------------------------ helpers
    def state_space_model(
        self, n_periods: int | None = None
    ) -> tuple[StateSpace, StateLayout, pd.PeriodIndex]:
        """Estimated state-space model, its state layout and the base grid.

        Public access for the analysis modules (news decomposition, density,
        diagnostics) that re-run the Kalman smoother with the estimated parameters.

        Parameters
        ----------
        n_periods : int, optional
            Number of periods (from the start of :attr:`grid`) the model must cover.
            Only matters for models with calendar aggregations (weekly or daily base
            grids), whose observation equation is time varying
            (``StateSpace(obs_index=...)``) and therefore tied to a number of periods;
            the stored model covers :attr:`grid`.

        Returns
        -------
        state_space : StateSpace
            Final state-space representation (standardised units); time varying for
            calendar aggregations (use ``designs(n)`` rather than ``Z``).
        state_layout : StateLayout
            Position of factors and idiosyncratic components in the state vector.
        grid : pandas.PeriodIndex
            Estimation grid extended to the forecast horizon.

        Raises
        ------
        ValueError
            If the results do not contain the model or the standardisation statistics.

        Examples
        --------
        >>> model, layout, grid = res.state_space_model()  # doctest: +SKIP
        """
        if self.state_space is None or self.state_layout is None or self.grid is None:
            raise ValueError("These results do not contain a state-space model.")
        if self.standardization is None:
            raise ValueError("These results do not contain standardisation statistics.")
        model = self.state_space
        layout = self.state_layout
        if (
            n_periods is not None
            and layout.is_time_varying
            and model.n_periods != int(n_periods)
            and self.em_parameters is not None
        ):
            model = build_state_space(self.em_parameters, layout, int(n_periods))
        return model, layout, self.grid

    def block_factors(self, block: str) -> pd.DataFrame:
        """Smoothed factors of one block.

        Parameters
        ----------
        block : str
            Block name.

        Returns
        -------
        pandas.DataFrame
            Columns of :attr:`factors` belonging to ``block``.

        Raises
        ------
        KeyError
            If ``block`` is unknown.

        Examples
        --------
        >>> res.block_factors("global")  # doctest: +SKIP
        """
        _, layout, _ = self.state_space_model()
        if block not in layout.block_names or self.factors is None:
            raise KeyError(f"Unknown block {block!r}; blocks: {list(layout.block_names)}.")
        b = layout.block_names.index(block)
        start = sum(layout.n_factors[:b])
        return self.factors.iloc[:, start : start + layout.n_factors[b]].copy()

    def observations(
        self, data: MixedFrequencyData | pd.DataFrame | None = None
    ) -> tuple[pd.PeriodIndex, FloatArray]:
        """Standardised observation matrix of a (new) vintage on the model grid.

        Parameters
        ----------
        data : MixedFrequencyData or pandas.DataFrame, optional
            Panel containing every model series (extra columns are ignored), e.g. a
            later vintage. Defaults to the estimation data. The grid starts at the
            first period of the estimation grid and ends at the later of the model grid
            end and the end of ``data``.

        Returns
        -------
        grid : pandas.PeriodIndex
            Base grid.
        values : numpy.ndarray, shape (n_periods, n_series)
            Data standardised with the estimation statistics (``NaN`` = missing). The
            observations in :attr:`excluded_observations` are set to ``NaN``.

        Raises
        ------
        NowcastDataError
            If series of the model are missing from ``data``.

        Examples
        --------
        >>> grid, y = res.observations(new_vintage)  # doctest: +SKIP
        """
        _, layout, grid = self.state_space_model()
        assert self.standardization is not None  # noqa: S101
        panel = self.data if data is None else data
        if panel is None:
            raise ValueError("No data given and none stored in the results.")
        if isinstance(panel, pd.DataFrame):
            frame = panel
        else:
            frame = as_mixed_frequency_data(panel).data
        missing = [s for s in layout.series if s not in frame.columns]
        if missing:
            raise NowcastDataError(f"The data do not contain the model series {missing}.")
        if not isinstance(frame.index, pd.PeriodIndex):
            raise NowcastDataError("The data must be indexed by a pandas PeriodIndex.")
        end = max(grid[-1], frame.index.asfreq(grid.freqstr)[-1])
        full = pd.period_range(grid[0], end, freq=grid.freqstr)
        aligned = frame.loc[:, list(layout.series)].reindex(full)
        values = np.array(self.standardization.transform(aligned), dtype=np.float64)
        if self.excluded_observations is not None:
            mask = self.excluded_observations.reindex(full, fill_value=False)
            values[mask.to_numpy(dtype=bool)] = np.nan
        return full, values

    def _offset(self, n_periods: int) -> FloatArray | None:
        """Dummy offset ``g_t`` on ``n_periods`` periods from the start of :attr:`grid`."""
        if self.state_offset is None or self.state_space is None:
            return None
        g = self.state_offset[:n_periods]
        if n_periods > g.shape[0]:
            extra = np.zeros((n_periods - g.shape[0], g.shape[1]))
            last = g[-1]
            for t in range(extra.shape[0]):
                last = self.state_space.T @ last
                extra[t] = last
            g = np.vstack([g, extra])
        return g

    def smooth(self, data: MixedFrequencyData | pd.DataFrame | None = None) -> SmootherResult:
        r"""Run the Kalman smoother with the estimated parameters on a vintage.

        Parameters
        ----------
        data : MixedFrequencyData or pandas.DataFrame, optional
            Panel (see :meth:`observations`); defaults to the estimation data.

        Returns
        -------
        SmootherResult
            Smoothed states (standardised units) on the grid of :meth:`observations`,
            including the deterministic offset of the pandemic dummies. Always the dense
            Kalman smoother (full covariances, filter output for the news
            decomposition).

        Notes
        -----
        With Student-t errors the smoother uses the Gaussian model with the scale
        :math:`\psi_i` as measurement variance (unit weights): new observations have no
        fitted weights.

        Examples
        --------
        >>> res.smooth(new_vintage).smoothed_state.shape  # doctest: +SKIP
        """
        _, values = self.observations(data)
        model, _, _ = self.state_space_model(values.shape[0])
        offset = self._offset(values.shape[0])
        method = _dense_method(self.filter_method)
        if offset is None:
            return kalman_smoother(model, values, method=method)  # type: ignore[arg-type]
        y = values - signal_mean(model, offset)
        sm = kalman_smoother(model, y, method=method)  # type: ignore[arg-type]
        return replace(sm, smoothed_state=sm.smoothed_state + offset)

    def predict(self, data: MixedFrequencyData | pd.DataFrame | None = None) -> pd.DataFrame:
        """``E[x_t | data]`` for every series, in original units.

        Parameters
        ----------
        data : MixedFrequencyData or pandas.DataFrame, optional
            Panel (see :meth:`observations`); defaults to the estimation data.

        Returns
        -------
        pandas.DataFrame
            Smoothed and forecast values on the base grid (lower-frequency series are
            meaningful in their storage slots).

        Examples
        --------
        >>> res.predict(new_vintage).tail()  # doctest: +SKIP
        """
        _, layout, _ = self.state_space_model()
        assert self.standardization is not None  # noqa: S101
        grid, _ = self.observations(data)
        sm = self.smooth(data)
        frame = pd.DataFrame(sm.smoothed_signal(), index=grid, columns=list(layout.series))
        return self.standardization.inverse_transform(frame)

    def _summary_sections(self) -> list[tuple[str, list[str]]]:
        sections = super()._summary_sections()
        layout = self.state_layout
        if layout is None:
            return sections
        lines = [
            f"  {'Blocks':<22}"
            + ", ".join(
                f"{b} ({r})" for b, r in zip(layout.block_names, layout.n_factors, strict=True)
            ),
            f"  {'Idiosyncratic':<22}{self.model_params.get('idiosyncratic', layout.idiosyncratic)}",
            f"  {'State dimension':<22}{layout.n_states}",
            f"  {'E-step filter':<22}{self.filter_method}",
        ]
        lines += self._robust_summary_lines()
        if self.loglikelihood_path is not None and self.loglikelihood_path.size > 1:
            first = float(self.loglikelihood_path[0])
            lines.append(f"  {'Initial log-lik.':<22}{first:.4f}")
        sections.append(("EM estimation", lines))
        return sections

    def _robust_summary_lines(self) -> list[str]:
        lines: list[str] = []
        if self.student_t_df is not None:
            lines.append(f"  {'Student-t df':<22}{self.student_t_df:.3f}")
        if self.outlier_flags is not None:
            lines.append(f"  {'Outliers flagged':<22}{int(self.outlier_flags.to_numpy().sum())}")
        if self.excluded_observations is not None:
            n_excluded = int(self.excluded_observations.to_numpy().sum())
            lines.append(f"  {'Excluded obs.':<22}{n_excluded}")
        if self.interventions is not None:
            lines.append(f"  {'COVID dummies':<22}{len(self.interventions)} periods")
        if self.long_run_mean is not None:
            lines.append(
                f"  {'Long-run mean':<22}time-varying ({self.long_run_mean.shape[1] // 2})"
            )
        return lines


# ====================================================================== estimator
class MixedFreqDFM(BaseNowcaster):
    r"""Mixed-frequency dynamic factor model estimated by EM (Bańbura & Modugno, 2014).

    Parameters
    ----------
    n_factors : int or mapping of str to int, default 1
        Number of factors of every block, or one number per block name.
    factor_lags : int, default 1
        Order ``p`` of the factor VAR of each block.
    blocks : None, "data", pandas.DataFrame, mapping or array_like, default None
        Block structure. ``None``: a single ``"global"`` block with every series.
        ``"data"``: the block metadata of the panel (``MixedFrequencyData.blocks``).
        A DataFrame (series x blocks, 0/1 or bool, rows indexed by series name), a
        mapping ``{series: block or [blocks]}``, or an array ``(n_series, n_blocks)``
        aligned with the panel columns (blocks named ``block1, block2, ...``). Every
        series must load on at least one block; blocks without series are dropped with
        a :class:`~nowcastbox.core.exceptions.DataQualityWarning`.
    idiosyncratic : {"ar1", "iid", "student_t"}, default "ar1"
        ``"ar1"``: AR(1) idiosyncratic components in the state (lower-frequency series
        aggregate a latent base-frequency AR(1) with their weights) and a small fixed
        measurement noise ``obs_noise_var``. ``"iid"``: white-noise idiosyncratic
        components with estimated variances. ``"student_t"``: white-noise idiosyncratic
        components with a scaled Student-t distribution (scale mixture of normals,
        estimated by ECM; innovation I3, :mod:`nowcastbox.models.robust`); fitted
        weights in ``results.observation_weights``.
    max_iter : int, default 500
        Maximum number of EM iterations (``0`` only evaluates the initial parameters).
    tol : float, default 1e-4
        Convergence tolerance on the relative change of the log-likelihood
        ``(l_k - l_{k-1}) / ((|l_k| + |l_{k-1}|) / 2)``.
    init : str, MixedFreqDFMResults or EMParameters, default "pca"
        Starting values: principal components of the spline-filled panel (Bańbura &
        Modugno, 2014), or a warm start from previous results / parameters of the
        same model structure. With several blocks, ``"pca"`` computes the block
        components in each order of
        :data:`~nowcastbox.models._init_conditions.BLOCK_ORDERS` and starts from the
        one with the highest log-likelihood (``info["initialization"]``);
        ``"pca_given"``, ``"pca_reversed"``, ``"pca_specific_first"`` and
        ``"pca_independent"`` force one order (``"pca_given"`` is the sequential scheme
        of Bańbura & Modugno).
    obs_noise_var : float, default 1e-4
        Fixed measurement-noise variance (standardised units) of the ``"ar1"``
        specification.
    filter_method : {"auto", "structured", "univariate", "multivariate"}, default "auto"
        Smoother of the E-step and of the final smoothing pass. ``"auto"``: the exact
        structured smoother of :func:`~nowcastbox.statespace.smoothed_moments`
        (innovation I2: sparse precision of the state path, banded Cholesky of the
        idiosyncratic chains, ``O(N n^2)``) whenever the model has the required
        structure and it is estimated to be cheaper, the dense Kalman smoother
        otherwise. ``"structured"`` forces it (``ValueError`` if not applicable, e.g.
        ``obs_noise_var=0``). ``"univariate"``/``"multivariate"`` force the dense Kalman
        smoother with that filter (the univariate treatment of Koopman & Durbin, 2000,
        avoids inverting ``N x N`` matrices). Both paths give the same estimates up to
        rounding. The robust E-steps (Student-t, outliers, pandemic options) and
        :meth:`MixedFreqDFMResults.smooth` always use the dense smoother.
    df : float, optional
        Degrees of freedom of the Student-t errors (``> 0``); ``None`` estimates them in
        ``[2, 200]`` by one-dimensional likelihood maximisation. Only with
        ``idiosyncratic="student_t"``.
    outliers : {"none", "auto"}, default "none"
        ``"auto"``: after the EM converges, observations whose standardised one-step-ahead
        prediction error exceeds ``outlier_threshold`` are treated as missing and the EM
        is re-run (at most 3 passes, until the flags are stable); flags in
        ``results.outlier_flags``.
    outlier_threshold : float, default 4.0
        Threshold (in standard deviations) of the outlier procedure.
    exclude_periods : period-like, (start, end) tuple or list of those, optional
        Periods whose observations (every series) are treated as missing in the
        estimation (standardisation, starting values and E-step), e.g.
        ``[("2020-03", "2020-06")]`` or ``"2020Q2"``. The states are still smoothed
        through them, and ``results.nowcast["observed"]`` keeps the data.
    covid : {"none", "mask", "dummy"}, default "none"
        Treatment of the pandemic window ``covid_window``. ``"mask"``: its
        observations are treated as missing. ``"dummy"``: impulse dummies in the
        factor VAR for every period of the window (the factors follow the data, the
        swing does not distort the VAR and its covariance); dummies in
        ``results.interventions``.
    covid_window : tuple of two period-likes, default ("2020-03", "2021-12")
        First and last period of the pandemic window (Brazil and US).
    long_run_mean : {"constant", "time_varying"}, default "constant"
        ``"constant"``: the mean of every series is constant (removed by the
        standardisation). ``"time_varying"``: the target has a random-walk long-run
        mean (local level, one extra state; innovation I4, Antolin-Diaz, Drechsel &
        Petrella, 2017); smoothed path in ``results.long_run_mean`` and in the
        ``long_run_mean`` column of ``results.nowcast``.
    long_run_series : str or list of str, optional
        Further series loading on the long-run mean (estimated loadings).
    long_run_variance : float, optional
        Fix the variance of the long-run mean innovations (standardised units, i.e. a
        signal-to-noise ratio relative to the target variance), avoiding the pile-up
        of its ML estimate at zero; ``None`` estimates it.

    Notes
    -----
    Data are standardised internally (``MixedFrequencyData.standardize``); results are
    reported in the original units. The default aggregation of lower-frequency series
    is Mariano-Murasawa (``AggregationType.GROWTH_RATE``); set the ``aggregation``
    metadata of a series (``"flow"``, ``"average"``, ``"stock"``) to change it.

    Weekly or daily data (innovation I1): put the panel on a weekly (``"W"``) or daily
    base grid; monthly and quarterly series are stored in the last week (day) that ends
    in their period. Their aggregation weights then vary with the calendar (4 or 5 weeks
    per month, 12 to 14 per quarter), the design matrix is time varying and only the
    loading of the latent base-frequency series is estimated (see
    :func:`series_calendar`). Monthly + quarterly panels on a monthly grid keep the
    fixed-weight model. The factor VAR then runs at the base frequency and the state
    holds as many factor lags as the longest weight vector (27 weeks for quarterly
    growth rates on a weekly grid).

    The transition parameters are updated by a generalized EM step that accounts for
    the stationary initial-state distribution (see :func:`~nowcastbox.models._em_steps.m_step`),
    so the log-likelihood never decreases; ``results.info["n_loglikelihood_decreases"]``
    counts violations beyond rounding (expected to be zero).

    ``fit`` accepts ``horizon`` (int, default 0): number of target periods to forecast
    beyond the period containing the end of the sample (or beyond the first period
    after the last target observation when the target is up to date).

    With Student-t errors the reported ``loglikelihood`` (and ``loglikelihood_path``) is
    the evidence lower bound maximised by the variational ECM (a lower bound of the
    Student-t log-likelihood, monotone over iterations). With ``outliers="auto"`` the
    path is the one of the last EM pass and ``n_iter`` counts every pass.

    References
    ----------
    Bańbura, M., & Modugno, M. (2014). *Journal of Applied Econometrics*, 29(1), 133-160.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from nowcastbox.models import MixedFreqDFM
    >>> rng = np.random.default_rng(0)
    >>> n = 120
    >>> f = np.zeros(n)
    >>> for t in range(1, n):
    ...     f[t] = 0.7 * f[t - 1] + rng.standard_normal()
    >>> x = np.outer(f, [1.0, 0.8, 0.6, 0.9]) + 0.5 * rng.standard_normal((n, 4))
    >>> gdp = np.convolve(f, [1, 2, 3, 2, 1])[:n] / 3 + 0.3 * rng.standard_normal(n)
    >>> gdp[np.arange(n) % 3 != 2] = np.nan
    >>> idx = pd.period_range("2010-01", periods=n, freq="M")
    >>> df = pd.DataFrame(x, index=idx, columns=["a", "b", "c", "d"]).assign(gdp=gdp)
    >>> df.iloc[-2:, :] = np.nan
    >>> model = MixedFreqDFM(n_factors=1, max_iter=50)
    >>> res = model.fit(df, target="gdp", frequency={"gdp": "Q"})
    >>> res.converged, res.factors.shape[1]
    (True, 1)
    >>> bool(np.isfinite(res.get_nowcast()))
    True
    """

    def __init__(
        self,
        n_factors: int | Mapping[str, int] = 1,
        factor_lags: int = 1,
        blocks: Any = None,
        idiosyncratic: str = "ar1",
        max_iter: int = 500,
        tol: float = 1e-4,
        init: Any = "pca",
        obs_noise_var: float = 1e-4,
        filter_method: str = "auto",
        df: float | None = None,
        outliers: str = "none",
        outlier_threshold: float = 4.0,
        exclude_periods: Any = None,
        covid: str = "none",
        covid_window: tuple[Any, Any] = COVID_WINDOW,
        long_run_mean: str = "constant",
        long_run_series: str | list[str] | None = None,
        long_run_variance: float | None = None,
    ) -> None:
        self.n_factors = n_factors
        self.factor_lags = factor_lags
        self.blocks = blocks
        self.idiosyncratic = idiosyncratic
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.obs_noise_var = obs_noise_var
        self.filter_method = filter_method
        self.df = df
        self.outliers = outliers
        self.outlier_threshold = outlier_threshold
        self.exclude_periods = exclude_periods
        self.covid = covid
        self.covid_window = covid_window
        self.long_run_mean = long_run_mean
        self.long_run_series = long_run_series
        self.long_run_variance = long_run_variance

    # ------------------------------------------------------------------ validation
    def _validate_params(self) -> None:
        """Check the hyper-parameters (raises ``ValueError``)."""
        _check_n_factors(self.n_factors)
        _check_int(self.factor_lags, "factor_lags", 1)
        _check_int(self.max_iter, "max_iter", 0)
        _check_nonnegative(self.tol, "tol")
        _check_nonnegative(self.obs_noise_var, "obs_noise_var")
        if self.idiosyncratic not in _IDIOSYNCRATIC:
            raise ValueError(
                f"idiosyncratic must be one of {_IDIOSYNCRATIC}, got {self.idiosyncratic!r}."
            )
        if self.filter_method not in _FILTER_METHODS:
            raise ValueError(
                f"filter_method must be one of {_FILTER_METHODS}, got {self.filter_method!r}."
            )
        valid_init = isinstance(self.init, MixedFreqDFMResults | EMParameters) or (
            isinstance(self.init, str) and self.init in _INIT_OPTIONS
        )
        if not valid_init:
            raise ValueError(
                f"init must be one of {_INIT_OPTIONS}, a MixedFreqDFMResults or "
                f"EMParameters object, got {self.init!r}."
            )
        self._validate_robust_params()
        self._validate_long_run_params()

    def _validate_robust_params(self) -> None:
        """Check the I3 options (raises ``ValueError``)."""
        if self.df is not None:
            if self.idiosyncratic != "student_t":
                raise ValueError("df is only used with idiosyncratic='student_t'.")
            _check_positive(self.df, "df")
        if self.outliers not in OUTLIER_MODES:
            raise ValueError(f"outliers must be one of {OUTLIER_MODES}, got {self.outliers!r}.")
        _check_positive(self.outlier_threshold, "outlier_threshold")
        if self.covid not in COVID_MODES:
            raise ValueError(f"covid must be one of {COVID_MODES}, got {self.covid!r}.")
        window = self.covid_window
        if not isinstance(window, tuple | list) or len(window) != 2:
            raise ValueError(f"covid_window must be a (start, end) pair, got {window!r}.")
        probe = pd.period_range("2000-01", periods=1, freq="M")
        period_mask(probe, [tuple(window)])
        period_mask(probe, self.exclude_periods)

    def _validate_long_run_params(self) -> None:
        """Check the I4 options (raises ``ValueError``)."""
        if self.long_run_mean not in LONG_RUN_MODES:
            raise ValueError(
                f"long_run_mean must be one of {LONG_RUN_MODES}, got {self.long_run_mean!r}."
            )
        if self.long_run_mean == "constant" and (
            self.long_run_series is not None or self.long_run_variance is not None
        ):
            raise ValueError(
                "long_run_series and long_run_variance require long_run_mean='time_varying'."
            )
        if self.long_run_variance is not None:
            _check_positive(self.long_run_variance, "long_run_variance")

    # ------------------------------------------------------------------ fit
    def _fit(self, data: MixedFrequencyData, target: str, **fit_kwargs: Any) -> MixedFreqDFMResults:
        """Estimate the model by EM and compute the nowcasts of ``target``.

        Parameters
        ----------
        data : MixedFrequencyData
            Panel with the target and its predictors.
        target : str
            Target series.
        **fit_kwargs
            Only ``horizon`` (int, default 0): additional target periods to forecast.

        Returns
        -------
        MixedFreqDFMResults
            Estimation results.

        Raises
        ------
        TypeError
            On unknown fit options.
        """
        horizon = fit_kwargs.pop("horizon", 0)
        if fit_kwargs:
            raise TypeError(f"Unknown fit options: {sorted(fit_kwargs)}.")
        _check_int(horizon, "horizon", 0)
        started = time.perf_counter()
        spec = self._robust_spec(data)
        estimation_data = data
        if spec.excluded is not None and spec.excluded.any():
            # excluded observations influence neither standardisation nor start values
            frame = data.data.mask(pd.DataFrame(spec.excluded, data.index, data.columns))
            estimation_data = data.with_data(frame)
        standardized, stats = estimation_data.standardize()
        layout = build_layout(
            standardized,
            self.n_factors,
            self.factor_lags,
            self.blocks,
            self.idiosyncratic,
            long_run=resolve_long_run_series(
                self.long_run_mean, self.long_run_series, list(data.columns), target
            ),
            long_run_variance=self.long_run_variance,
        )
        params, init_info = self._initial_parameters(standardized, layout)
        if spec.is_active:
            outcome = _run_robust(self, params, layout, standardized.values, spec)
        else:
            outcome = run_em(
                params,
                layout,
                standardized.values,
                max_iter=int(self.max_iter),
                tol=float(self.tol),
                method=self.filter_method,
            )
        if not outcome.converged and self.max_iter > 0:
            warnings.warn(
                f"EM did not converge in {self.max_iter} iterations (last relative "
                f"change {outcome.last_change:.3g} > tol={self.tol}).",
                ConvergenceWarning,
                stacklevel=3,
            )
        full = standardized if estimation_data is data else data.standardize(stats)[0]
        extended = full.extend(_extension_periods(data, target, int(horizon)))
        info = {
            "elapsed_seconds": time.perf_counter() - started,
            "n_loglikelihood_decreases": outcome.n_decreases,
            "last_relative_change": outcome.last_change,
            "horizon": int(horizon),
            "n_states": layout.n_states,
            "initialization": init_info,
        }
        if outcome.robust is not None:
            info.update(outcome.info)
        return _assemble_results(self, data, target, stats, layout, extended, outcome, info)

    def _robust_spec(self, panel: MixedFrequencyData) -> RobustSpec:
        """Robustness options on the estimation grid."""
        index = panel.index
        excluded = period_mask(index, self.exclude_periods)
        window = period_mask(index, [tuple(self.covid_window)])
        if self.covid != "none" and not window.any():
            warnings.warn(
                f"covid={self.covid!r} but the window {tuple(self.covid_window)} does not "
                "overlap the sample; it has no effect.",
                DataQualityWarning,
                stacklevel=4,
            )
        if self.covid == "mask":
            excluded = excluded | window
        intervention = window if self.covid == "dummy" else None
        return RobustSpec(
            student_t=self.idiosyncratic == "student_t",
            df=None if self.df is None else float(self.df),
            outliers=self.outliers == "auto",
            outlier_threshold=float(self.outlier_threshold),
            excluded=np.repeat(excluded[:, None], panel.n_series, axis=1),
            intervention=intervention,
        )

    def _initial_parameters(
        self, standardized: MixedFrequencyData, layout: StateLayout
    ) -> tuple[EMParameters, dict[str, Any]]:
        if isinstance(self.init, MixedFreqDFMResults):
            if self.init.state_layout is None or self.init.em_parameters is None:
                raise ValueError("init results do not contain EM parameters.")
            if not layout.is_compatible(self.init.state_layout):
                raise ValueError("init results were estimated with a different model structure.")
            return self.init.em_parameters.copy(), {"method": "results"}
        if isinstance(self.init, EMParameters):
            _check_parameter_shapes(self.init, layout)
            return self.init.copy(), {"method": "parameters"}
        noise = float(self.obs_noise_var)
        assert isinstance(self.init, str)  # noqa: S101 - validated
        if self.init != "pca" or layout.n_blocks == 1:
            order = "given" if self.init == "pca" else self.init.removeprefix("pca_")
            params = pca_initial_parameters(
                standardized, layout, obs_noise_var=noise, block_order=order
            )
            return params, {"method": "pca", "block_order": order}
        return _best_pca_start(standardized, layout, noise)


def _best_pca_start(
    standardized: MixedFrequencyData, layout: StateLayout, obs_noise_var: float
) -> tuple[EMParameters, dict[str, Any]]:
    """Block-PCA starting values of every block order; keep the most likely one.

    With nested blocks (a global block plus thematic blocks) the sequential scheme of
    Bańbura & Modugno depends on the order of the blocks and can start the EM in the
    basin of a poor local maximum. Each order of ``BLOCK_ORDERS`` costs one PCA and one
    Kalman filter pass; failures (e.g. a degenerate block) are skipped.
    """
    values = standardized.values
    candidates: dict[str, EMParameters] = {}
    scores: dict[str, float] = {}
    errors: list[Exception] = []
    for order in BLOCK_ORDERS:
        try:
            params = pca_initial_parameters(
                standardized, layout, obs_noise_var=obs_noise_var, block_order=order
            )
            model = build_state_space(params, layout, standardized.n_periods)
            score = float(loglikelihood(model, values))
        except (ValueError, np.linalg.LinAlgError) as err:  # NowcastDataError is a ValueError
            errors.append(err)
            continue
        if np.isfinite(score):
            candidates[order], scores[order] = params, score
    if not candidates:
        raise errors[0] if errors else NowcastDataError("No finite starting log-likelihood.")
    best = max(scores, key=lambda k: (scores[k], -BLOCK_ORDERS.index(k)))
    logger.debug("EM start: block order %s (log-likelihoods %s)", best, scores)
    return candidates[best], {"method": "pca", "block_order": best, "loglikelihood": scores}


# ====================================================================== EM loop
@dataclass(frozen=True, eq=False)
class EMOutcome:
    """Output of :func:`run_em`.

    Attributes
    ----------
    params : EMParameters
        Parameters at the last E-step (the log-likelihood of ``loglikelihood_path[-1]``).
    loglikelihood_path : numpy.ndarray
        Log-likelihood at each E-step.
    converged : bool
        Whether the relative change fell below ``tol``.
    n_iter : int
        Number of M-steps performed.
    n_decreases : int
        Number of iterations in which the log-likelihood decreased (beyond rounding).
    last_change : float
        Last relative change of the log-likelihood (NaN with ``max_iter=0``).
    robust : RobustState, optional
        Latent quantities of the robust EM (weights, df, dummies, outlier flags).
    spec : RobustSpec, optional
        Robustness options used.
    info : dict
        Extra diagnostics of the robust EM.
    """

    params: EMParameters
    loglikelihood_path: FloatArray
    converged: bool
    n_iter: int
    n_decreases: int
    last_change: float
    robust: RobustState | None = None
    spec: RobustSpec | None = None
    info: dict[str, Any] = field(default_factory=dict)


def run_em(
    params: EMParameters,
    layout: StateLayout,
    observations: FloatArray,
    *,
    max_iter: int = 500,
    tol: float = 1e-4,
    method: str = "univariate",
) -> EMOutcome:
    """Iterate E- and M-steps until the relative log-likelihood change is below ``tol``.

    Parameters
    ----------
    params : EMParameters
        Starting values.
    layout : StateLayout
        State layout.
    observations : numpy.ndarray, shape (n_periods, N)
        Standardised data (``NaN`` = missing).
    max_iter : int, default 500
        Maximum number of M-steps.
    tol : float, default 1e-4
        Relative-change tolerance (see :func:`~nowcastbox.models._em_steps.em_converged`).
    method : str, default "univariate"
        Kalman filter variant.

    Returns
    -------
    EMOutcome
        Final parameters and diagnostics.

    Examples
    --------
    >>> import numpy as np
    >>> from nowcastbox.models._em_steps import EMParameters, StateLayout
    >>> from nowcastbox.models.em import run_em
    >>> lay = StateLayout(["a", "b"], ["g"], [1], 1, np.ones((2, 1), bool), [[1.0], [1.0]], "iid")
    >>> p = EMParameters(
    ...     (np.array([[0.5]]),), (np.eye(1),), np.ones((2, 1)), np.zeros(2), np.ones(2), np.ones(2)
    ... )
    >>> y = np.random.default_rng(0).standard_normal((30, 2))
    >>> out = run_em(p, lay, y, max_iter=5, tol=0.0)
    >>> out.n_iter, out.loglikelihood_path.size
    (5, 6)
    """
    path: list[float] = []
    converged, n_decreases, change, n_iter = False, 0, float("nan"), 0
    iteration = 0
    while True:
        model = build_state_space(params, layout, observations.shape[0])
        stats = e_step(model, observations, method=method)
        loglik = stats.loglikelihood
        if not np.isfinite(loglik):
            raise NowcastDataError(f"Non-finite log-likelihood at EM iteration {iteration}.")
        path.append(loglik)
        if iteration > 0:
            converged, change = em_converged(loglik, path[-2], tol)
            if change < -_DECREASE_TOL:
                n_decreases += 1
                logger.debug("log-likelihood decreased at iteration %d (%.3g)", iteration, change)
        logger.debug("EM iteration %d: loglik %.6f", iteration, loglik)
        if converged or iteration == max_iter:
            break
        params = m_step(stats, layout, observations, params)
        n_iter += 1
        iteration += 1
    return EMOutcome(params, np.asarray(path), converged, n_iter, n_decreases, change)


def _run_robust(
    estimator: MixedFreqDFM,
    params: EMParameters,
    layout: StateLayout,
    observations: FloatArray,
    spec: RobustSpec,
) -> EMOutcome:
    """Run :func:`~nowcastbox.models.robust.run_robust_em` and wrap its output."""
    out = run_robust_em(
        params,
        layout,
        observations,
        spec,
        max_iter=int(estimator.max_iter),
        tol=float(estimator.tol),
        method=estimator.filter_method,
    )
    info = {
        "objective": "evidence_lower_bound" if spec.student_t else "loglikelihood",
        "n_outlier_passes": out.n_passes,
        "n_outliers": int(out.state.flags.sum()),
        "n_iter_last_pass": out.n_iter,
    }
    return EMOutcome(
        out.params,
        out.loglikelihood_path,
        out.converged,
        out.total_iter,
        out.n_decreases,
        out.last_change,
        robust=out.state,
        spec=spec,
        info=info,
    )


# ====================================================================== layout
def _check_int(value: object, name: str, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int | np.integer) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}.")


def _check_nonnegative(value: object, name: str) -> None:
    ok = isinstance(value, int | float | np.floating | np.integer) and not isinstance(value, bool)
    if not ok or not np.isfinite(float(value)) or float(value) < 0:  # type: ignore[arg-type]
        raise ValueError(f"{name} must be a finite non-negative number, got {value!r}.")


def _check_positive(value: object, name: str) -> None:
    try:
        _check_nonnegative(value, name)
    except ValueError:
        raise ValueError(f"{name} must be a finite positive number, got {value!r}.") from None
    if float(value) <= 0:  # type: ignore[arg-type]
        raise ValueError(f"{name} must be a finite positive number, got {value!r}.")


def _check_n_factors(n_factors: object) -> None:
    if isinstance(n_factors, Mapping):
        if not n_factors:
            raise ValueError("n_factors mapping must not be empty.")
        for block, value in n_factors.items():
            _check_int(value, f"n_factors[{block!r}]", 1)
        return
    _check_int(n_factors, "n_factors", 1)


def _check_parameter_shapes(params: EMParameters, layout: StateLayout) -> None:
    expected = (layout.n_series, layout.n_factor_states)
    ok = (
        len(params.transition) == layout.n_blocks
        and params.loadings.shape == expected
        and all(
            a.shape == (r, r * layout.factor_lags)
            for a, r in zip(params.transition, layout.n_factors, strict=True)
        )
        and params.idio_var.shape == (layout.n_series,)
        and (
            not layout.has_long_run
            or (
                params.trend_loadings is not None
                and params.trend_loadings.shape == (layout.n_series,)
            )
        )
    )
    if not ok:
        raise ValueError("init parameters do not match the model structure.")


def _membership_from_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    missing = [c for c in columns if c not in frame.index]
    if missing:
        raise ValueError(f"blocks has no row for the series {missing}.")
    values = frame.loc[columns]
    try:
        numeric = values.astype(float)
    except (TypeError, ValueError) as err:
        raise ValueError("blocks must contain 0/1 or boolean values.") from err
    if not bool(numeric.isin([0.0, 1.0]).all().all()):
        raise ValueError("blocks must contain 0/1 or boolean values.")
    return numeric.astype(bool).set_axis([str(c) for c in values.columns], axis=1)


def _membership_from_mapping(spec: Mapping[str, Any], columns: list[str]) -> pd.DataFrame:
    names: list[str] = []
    rows: dict[str, list[str]] = {}
    for column in columns:
        if column not in spec:
            raise ValueError(f"blocks has no entry for the series {column!r}.")
        value = spec[column]
        entries = [value] if isinstance(value, str) else list(value)
        rows[column] = [str(e) for e in entries]
        names += [e for e in rows[column] if e not in names]
    frame = pd.DataFrame(False, index=columns, columns=names)
    for column, entries in rows.items():
        frame.loc[column, entries] = True
    return frame


def _membership_from_array(spec: Any, columns: list[str]) -> pd.DataFrame:
    array = np.asarray(spec)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or array.shape[0] != len(columns):
        raise ValueError(
            f"blocks array must have shape (n_series={len(columns)}, n_blocks), got {array.shape}."
        )
    names = [f"block{k + 1}" for k in range(array.shape[1])]
    return _membership_from_frame(pd.DataFrame(array, index=columns, columns=names), columns)


def resolve_blocks(blocks: Any, panel: MixedFrequencyData) -> tuple[pd.DataFrame, bool]:
    """Block-membership matrix (series x blocks) from a ``blocks`` specification.

    Parameters
    ----------
    blocks : None, "data", pandas.DataFrame, mapping or array_like
        See :class:`MixedFreqDFM`.
    panel : MixedFrequencyData
        Panel (its columns define the series order).

    Returns
    -------
    membership : pandas.DataFrame
        Boolean matrix indexed by series, one column per (non-empty) block.
    block_prefix : bool
        Whether factor names carry the block name (False for the default single block).

    Raises
    ------
    ValueError
        On malformed specifications or series without block.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from nowcastbox.core.data import MixedFrequencyData
    >>> from nowcastbox.models.em import resolve_blocks
    >>> idx = pd.period_range("2020-01", periods=3, freq="M")
    >>> mfd = MixedFrequencyData(pd.DataFrame({"a": [1.0, 2, 3], "b": [3.0, 1, 2]}, idx), "M")
    >>> m, prefix = resolve_blocks({"a": ["g", "r"], "b": "g"}, mfd)
    >>> m.astype(int).values.tolist(), prefix
    ([[1, 1], [1, 0]], True)
    """
    columns = list(panel.columns)
    if blocks is None:
        return pd.DataFrame(True, index=columns, columns=[_DEFAULT_BLOCK]), False
    if isinstance(blocks, str):
        if blocks != "data":
            raise ValueError(
                f"blocks must be None, 'data' or a block specification, got {blocks!r}."
            )
        membership = panel.blocks
        if membership.shape[1] == 0:
            raise ValueError("blocks='data' but the panel has no block metadata.")
    elif isinstance(blocks, pd.DataFrame):
        membership = _membership_from_frame(blocks, columns)
    elif isinstance(blocks, Mapping):
        membership = _membership_from_mapping(blocks, columns)
    else:
        membership = _membership_from_array(blocks, columns)
    return _clean_membership(membership.loc[columns]), True


def _clean_membership(membership: pd.DataFrame) -> pd.DataFrame:
    orphans = [str(s) for s in membership.index[~membership.any(axis=1)]]
    if orphans:
        raise ValueError(f"Series {orphans} do not load on any block.")
    empty = [str(b) for b in membership.columns[~membership.any(axis=0)]]
    if empty:
        warnings.warn(
            f"Blocks {empty} contain no series and were dropped.", DataQualityWarning, stacklevel=4
        )
        membership = membership.drop(columns=empty)
    return membership.astype(bool)


def _resolve_n_factors(n_factors: int | Mapping[str, int], names: list[str]) -> list[int]:
    if not isinstance(n_factors, Mapping):
        return [int(n_factors)] * len(names)
    lookup = {str(k): int(v) for k, v in n_factors.items()}
    missing = [b for b in names if b not in lookup]
    if missing:
        raise ValueError(f"n_factors has no entry for the blocks {missing}.")
    return [lookup[b] for b in names]


def series_weights(panel: MixedFrequencyData) -> list[FloatArray]:
    """Aggregation weights linking each series to the base-frequency latent series.

    Parameters
    ----------
    panel : MixedFrequencyData
        Panel; ``SeriesMetadata.aggregation`` selects the aggregation of each
        lower-frequency series (default Mariano-Murasawa).

    Returns
    -------
    list of numpy.ndarray
        Weights (most recent period first); ``[1.0]`` for base-frequency series. For
        calendar aggregations (see :func:`series_calendar`) the longest weight vector
        (the weights actually vary with the period).

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from nowcastbox.core.data import MixedFrequencyData
    >>> from nowcastbox.models.em import series_weights
    >>> idx = pd.period_range("2020-01", periods=3, freq="M")
    >>> df = pd.DataFrame({"a": [1.0, 2, 3], "q": [np.nan, np.nan, 1.0]}, index=idx)
    >>> [w.tolist() for w in series_weights(MixedFrequencyData(df, {"a": "M", "q": "Q"}))]
    [[1.0], [1.0, 2.0, 3.0, 2.0, 1.0]]
    """
    base = panel.base_frequency
    out: list[FloatArray] = []
    for column, spec in zip(panel.columns, series_calendar(panel), strict=True):
        meta = panel.metadata[column]
        if meta.frequency == base:
            out.append(np.ones(1))
        elif spec is not None:
            out.append(np.asarray(spec.reference_weights, float))
        else:
            aggregation = meta.aggregation or AggregationType.GROWTH_RATE
            out.append(np.asarray(aggregation_weights(base, meta.frequency, aggregation), float))
    return out


def series_calendar(panel: MixedFrequencyData) -> list[CalendarAggregation | None]:
    """Calendar-aware aggregation of every series whose ratio to the base grid varies.

    Parameters
    ----------
    panel : MixedFrequencyData
        Panel; ``SeriesMetadata.aggregation`` selects the aggregation (default
        Mariano-Murasawa growth rate).

    Returns
    -------
    list of CalendarAggregation or None
        ``None`` for base-frequency series and fixed-ratio pairs (e.g. quarterly series
        on a monthly grid), a
        :class:`~nowcastbox.preprocessing.aggregation.CalendarAggregation` otherwise
        (e.g. monthly or quarterly series on a weekly grid).

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from nowcastbox.core.data import MixedFrequencyData
    >>> from nowcastbox.models.em import series_calendar
    >>> idx = pd.period_range("2020-01-05", periods=9, freq="W")
    >>> df = pd.DataFrame({"w": np.arange(9.0), "m": np.nan}, index=idx)
    >>> df.loc[idx[3], "m"] = 1.0  # last week ending in January
    >>> [
    ...     None if c is None else c.low.value
    ...     for c in series_calendar(MixedFrequencyData(df, {"w": "W", "m": "M"}))
    ... ]
    [None, 'M']
    """
    base = panel.base_frequency
    out: list[CalendarAggregation | None] = []
    for column in panel.columns:
        meta = panel.metadata[column]
        if is_fixed_ratio(base, meta.frequency):
            out.append(None)
            continue
        aggregation = meta.aggregation or AggregationType.GROWTH_RATE
        out.append(calendar_aggregation(base, meta.frequency, aggregation))
    return out


def build_layout(
    panel: MixedFrequencyData,
    n_factors: int | Mapping[str, int],
    factor_lags: int,
    blocks: Any,
    idiosyncratic: str,
    *,
    long_run: tuple[int, ...] = (),
    long_run_variance: float | None = None,
) -> StateLayout:
    """State layout of a :class:`MixedFreqDFM` for a panel.

    Parameters
    ----------
    panel : MixedFrequencyData
        Estimation panel.
    n_factors : int or mapping
        Factors per block.
    factor_lags : int
        VAR order.
    blocks : see :class:`MixedFreqDFM`
        Block specification.
    idiosyncratic : {"ar1", "iid", "student_t"}
        Idiosyncratic specification (``"student_t"`` has the ``"iid"`` state layout).
    long_run : tuple of int, default ()
        Series loading on the long-run mean state (target first).
    long_run_variance : float, optional
        Fixed long-run mean innovation variance.

    Returns
    -------
    StateLayout
        Layout of the state vector.

    Raises
    ------
    ValueError
        On inconsistent blocks/factors (e.g. more factors than series in a block).

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from nowcastbox.core.data import MixedFrequencyData
    >>> from nowcastbox.models.em import build_layout
    >>> idx = pd.period_range("2020-01", periods=3, freq="M")
    >>> df = pd.DataFrame({"a": [1.0, 2, 3], "q": [np.nan, np.nan, 1.0]}, index=idx)
    >>> build_layout(MixedFrequencyData(df, {"a": "M", "q": "Q"}), 1, 2, None, "ar1").n_states
    11
    """
    membership, prefix = resolve_blocks(blocks, panel)
    names = [str(b) for b in membership.columns]
    factors = _resolve_n_factors(n_factors, names)
    sizes = membership.sum(axis=0).to_numpy()
    for name, r, size in zip(names, factors, sizes, strict=True):
        if r > size:
            raise ValueError(f"Block {name!r} has {int(size)} series but {r} factors.")
    return StateLayout(
        panel.columns,
        names,
        factors,
        factor_lags,
        membership.to_numpy(),
        series_weights(panel),
        "iid" if idiosyncratic == "student_t" else idiosyncratic,
        block_prefix=prefix,
        long_run=long_run,
        long_run_variance=long_run_variance,
        calendar=series_calendar(panel),
        start=panel.index[0],
    )


def _extension_periods(panel: MixedFrequencyData, target: str, horizon: int) -> int:
    """Base periods to append so the current target period (and ``horizon`` more) fit.

    Calendar aware: the slot of a native period is its last base period (a week belongs
    to the period containing its last day).
    """
    freq = panel.metadata[target].frequency
    end = panel.index[-1]
    current = base_to_native(panel.index[-1:], freq)[0]
    observed = panel.to_native(target, dropna=True)
    last = current + horizon
    if len(observed) and observed.index[-1] >= current:
        last = last + 1
    slot = native_period_bounds(pd.PeriodIndex([last]), panel.base_frequency)[1][0]
    return max(int((slot - end).n), 0)


# ====================================================================== results
def _block_diagonal_transition(params: EMParameters, layout: StateLayout) -> FloatArray:
    p = layout.factor_lags
    mats = []
    for lag in range(p):
        blocks = [
            a[:, lag * r : (lag + 1) * r]
            for a, r in zip(params.transition, layout.n_factors, strict=True)
        ]
        mats.append(scipy.linalg.block_diag(*blocks))
    return np.hstack(mats)


def _matrix_sqrt(cov: FloatArray) -> FloatArray:
    eigval, eigvec = np.linalg.eigh(0.5 * (cov + cov.T))
    order = np.argsort(eigval)[::-1]
    return eigvec[:, order] * np.sqrt(np.maximum(eigval[order], 0.0))


def _state_labels(layout: StateLayout) -> list[str]:
    labels: list[str] = []
    for b, block in enumerate(layout.block_names):
        prefix = f"{block}_" if layout.block_prefix else ""
        for lag in range(layout.block_lags[b]):
            labels += [f"{prefix}f{k + 1}_L{lag}" for k in range(layout.n_factors[b])]
    return labels


def _common_component(model: StateSpace, layout: StateLayout, state: FloatArray) -> FloatArray:
    """Factor (and long-run mean) part of the signal of every series."""
    common_states = np.zeros_like(state)
    common_states[:, : layout.n_factor_states] = state[:, : layout.n_factor_states]
    if layout.has_long_run:
        common_states[:, layout.trend_index] = state[:, layout.trend_index]
    return signal_mean(model, common_states)


def _calendar_weights_dict(layout: StateLayout, grid: pd.PeriodIndex) -> dict[str, Any]:
    """Representative weights and weight paths of the calendar-aggregated series."""
    weights: dict[str, FloatArray] = {}
    paths: dict[str, pd.DataFrame] = {}
    for i, name in enumerate(layout.series):
        spec = layout.calendar[i]
        if spec is None:
            continue
        path = layout.weight_path(i, len(grid))
        slots = np.flatnonzero(is_period_end(grid, spec.low))
        weights[name] = path[slots[-1]].copy() if slots.size else layout.weights[i].copy()
        paths[name] = pd.DataFrame(
            path, index=grid, columns=[f"lag{lag}" for lag in range(path.shape[1])]
        )
    return {"weights": weights, "paths": paths}


def _params_dict(
    params: EMParameters, layout: StateLayout, model: StateSpace, grid: pd.PeriodIndex
) -> dict[str, Any]:
    series = list(layout.series)
    calendar = _calendar_weights_dict(layout, grid)
    weights = {s: w.copy() for s, w in zip(series, layout.weights, strict=True)}
    weights.update(calendar["weights"])
    out: dict[str, Any] = {
        "transition_blocks": dict(zip(layout.block_names, params.transition, strict=True)),
        "factor_cov_blocks": dict(zip(layout.block_names, params.factor_cov, strict=True)),
        "factor_cov": np.asarray(scipy.linalg.block_diag(*params.factor_cov), float),
        "loadings_all_lags": pd.DataFrame(
            params.loadings, index=series, columns=_state_labels(layout)
        ),
        "design": model.designs(len(grid)).copy() if model.is_time_varying else model.Z.copy(),
        "obs_cov": pd.Series(params.obs_var, index=series, name="obs_cov"),
        "idiosyncratic_ar": pd.Series(params.idio_ar, index=series, name="idiosyncratic_ar"),
        "idiosyncratic_var": pd.Series(params.idio_var, index=series, name="idiosyncratic_var"),
        "aggregation_weights": weights,
    }
    if calendar["paths"]:
        out["aggregation_weight_paths"] = calendar["paths"]
    if layout.has_long_run and params.trend_loadings is not None:
        out["long_run_loadings"] = pd.Series(
            params.trend_loadings, index=series, name="long_run_loadings"
        )
        out["long_run_variance"] = float(params.trend_var)
    return out


def _target_frame(
    data: MixedFrequencyData,
    target: str,
    stats: Any,
    grid: pd.PeriodIndex,
    smoother: SmootherResult | StructuredSmootherResult,
    common: FloatArray,
    layout: StateLayout,
    noise_var: float,
    long_run: pd.DataFrame | None = None,
) -> pd.DataFrame:
    j = layout.series.index(target)
    freq: Frequency = data.metadata[target].frequency
    slots = is_period_end(grid, freq)
    native = base_to_native(grid[slots], freq)
    variance = smoother.smoothed_signal_variance()[slots, j] + noise_var
    signal = smoother.smoothed_signal()[slots, j]
    estimate = pd.Series(stats.inverse_series(signal, target), index=native)
    common_s = pd.Series(stats.inverse_series(common[slots, j], target), index=native)
    std = pd.Series(
        stats.inverse_series(np.sqrt(np.maximum(variance, 0.0)), target, scale_only=True),
        index=native,
    )
    observed = data.to_native(target).reindex(native)
    has_obs = observed.notna()
    std = std.where(~has_obs)
    extra = {"common": common_s, "std": std}
    for label, zq in (("68", _Z68), ("90", _Z90)):
        extra[f"lower_{label}"] = estimate - zq * std
        extra[f"upper_{label}"] = estimate + zq * std
    if long_run is not None:
        extra["long_run_mean"] = pd.Series(long_run[target].to_numpy()[slots], index=native)
    # Periods before the first observation of the target are not backcasts of the
    # nowcasting exercise: leave them out of ``out_of_sample`` (``common`` keeps the
    # model's estimate of them).
    observed_flags = has_obs.to_numpy(dtype=bool)
    presample = ~np.logical_or.accumulate(observed_flags)
    for key in [k for k in extra if k != "common"]:
        extra[key] = extra[key].where(~presample)
    estimate = estimate.where(~presample)
    return build_nowcast_frame(observed, estimate.where(~has_obs, common_s), extra=extra)


@dataclass(frozen=True, eq=False)
class _Smoothed:
    """Final smoothing on the extended grid and the robust by-products."""

    smoother: SmootherResult | StructuredSmootherResult
    noise_var: FloatArray
    fields: dict[str, Any]


def _dense_method(method: str) -> str:
    """Dense Kalman filter variant for ``filter_method`` (``"structured"`` -> univariate)."""
    return method if method in ("univariate", "multivariate", "auto") else "univariate"


def _smooth_final(
    model: StateSpace, values: FloatArray, method: str
) -> SmootherResult | StructuredSmootherResult:
    """Final smoothing pass: structured smoother when chosen by ``method`` (I2)."""
    if method in ("auto", "structured"):
        return smoothed_moments(model, values, method=method).smoother  # type: ignore[arg-type]
    return kalman_smoother(model, values, method=method)  # type: ignore[arg-type]


def _final_smoothing(
    estimator: MixedFreqDFM,
    layout: StateLayout,
    extended: MixedFrequencyData,
    outcome: EMOutcome,
) -> _Smoothed:
    params = outcome.params
    state, spec = outcome.robust, outcome.spec
    if state is None or spec is None:
        model = build_state_space(params, layout, extended.n_periods)
        sm = _smooth_final(model, extended.values, estimator.filter_method)
        return _Smoothed(sm, params.obs_var, {})
    n_est, n_ext = state.flags.shape[0], extended.n_periods
    pad = np.zeros(n_ext - n_est, dtype=bool)
    excluded = np.vstack([spec.excluded, np.zeros((pad.size, layout.n_series), bool)])  # type: ignore[list-item]
    window = None if spec.intervention is None else np.concatenate([spec.intervention, pad])
    spec_ext = replace(spec, excluded=excluded, intervention=window)
    state_ext = state.padded(n_ext)
    values = np.where(excluded | state_ext.flags, np.nan, extended.values)
    sm = robust_smoother(
        params, layout, values, spec_ext, state_ext, estimator.filter_method
    ).smoother
    noise_var = params.obs_var
    if state.df is not None and state.df > 2.0:
        noise_var = params.obs_var * state.df / (state.df - 2.0)
    missing = np.isnan(extended.values[:n_est])
    fields = _robust_fields(layout, extended.index[:n_est], spec, state, missing)
    if spec.has_intervention and state_ext.dummies is not None:
        fields["state_offset"] = intervention_offset(state_ext.dummies, params, layout)
    return _Smoothed(sm, noise_var, fields)


def _robust_fields(
    layout: StateLayout,
    index: pd.PeriodIndex,
    spec: RobustSpec,
    state: RobustState,
    missing: NDArray[np.bool_],
) -> dict[str, Any]:
    series = list(layout.series)
    excluded = state.flags if spec.excluded is None else (spec.excluded | state.flags)
    fields: dict[str, Any] = {
        "excluded_observations": pd.DataFrame(excluded, index=index, columns=series),
    }
    if spec.outliers:
        fields["outlier_flags"] = pd.DataFrame(state.flags, index=index, columns=series)
    if state.q is not None:
        fields["observation_weights"] = pd.DataFrame(
            state.q.weights, index=index, columns=series
        ).where(~(excluded | missing))
        fields["student_t_df"] = state.df
    if spec.has_intervention and spec.intervention is not None and state.dummies is not None:
        window = np.flatnonzero(spec.intervention[1:]) + 1
        fields["interventions"] = pd.DataFrame(
            state.dummies[window], index=index[window], columns=layout.factor_names
        )
    return fields


def _long_run_results(
    layout: StateLayout,
    params: EMParameters,
    stats: Any,
    sm: SmootherResult | StructuredSmootherResult,
    grid: pd.PeriodIndex,
) -> pd.DataFrame | None:
    if not layout.has_long_run or params.trend_loadings is None:
        return None
    k = layout.trend_index
    names = [layout.series[i] for i in layout.long_run]
    return long_run_frame(
        sm.smoothed_state[:, k],
        sm.smoothed_state_cov[:, k, k],
        params.trend_loadings[list(layout.long_run)],
        stats.mean[names].to_numpy(dtype=np.float64),
        stats.std[names].to_numpy(dtype=np.float64),
        grid,
        names,
    )


def _assemble_results(
    estimator: MixedFreqDFM,
    data: MixedFrequencyData,
    target: str,
    stats: Any,
    layout: StateLayout,
    extended: MixedFrequencyData,
    outcome: EMOutcome,
    info: dict[str, Any],
) -> MixedFreqDFMResults:
    params = outcome.params
    grid = extended.index
    model = build_state_space(params, layout, len(grid))
    smoothed = _final_smoothing(estimator, layout, extended, outcome)
    sm = smoothed.smoother
    series = list(layout.series)
    state = sm.smoothed_state
    common = _common_component(model, layout, state)
    long_run = _long_run_results(layout, params, stats, sm, grid)
    j = layout.series.index(target)
    current = layout.current_factor_index
    names = layout.factor_names
    return MixedFreqDFMResults(
        target=target,
        nowcast=_target_frame(
            data, target, stats, grid, sm, common, layout, float(smoothed.noise_var[j]), long_run
        ),
        model_name="MixedFreqDFM",
        model_params=estimator.get_params(),
        factors=pd.DataFrame(state[:, current], index=grid, columns=names),
        loadings=pd.DataFrame(params.loadings[:, current], index=series, columns=names),
        params=_params_dict(params, layout, model, grid),
        loglikelihood=float(outcome.loglikelihood_path[-1]),
        n_iter=outcome.n_iter,
        converged=outcome.converged,
        data=data,
        standardization=stats,
        info=info,
        transition=_block_diagonal_transition(params, layout),
        shock_loadings=_matrix_sqrt(np.asarray(scipy.linalg.block_diag(*params.factor_cov), float)),
        factor_lags=layout.factor_lags,
        n_shocks=layout.total_factors,
        idiosyncratic_variance=pd.Series(params.idio_var, index=series, name="idiosyncratic_var"),
        em_parameters=params,
        state_layout=layout,
        state_space=model,
        loglikelihood_path=outcome.loglikelihood_path,
        idiosyncratic_ar=pd.Series(params.idio_ar, index=series, name="idiosyncratic_ar"),
        smoothed_data=stats.inverse_transform(
            pd.DataFrame(sm.smoothed_signal(), index=grid, columns=series)
        ),
        common_component=stats.inverse_transform(pd.DataFrame(common, index=grid, columns=series)),
        smoothed_state=state,
        grid=grid,
        filter_method=estimator.filter_method,
        long_run_mean=long_run,
        **smoothed.fields,
    )
