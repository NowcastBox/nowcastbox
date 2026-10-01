r"""Two-step dynamic factor model for nowcasting.

Implements the estimator of Giannone, Reichlin & Small (2008) with the asymptotic
justification of Doz, Giannone & Reichlin (2011) (plan sections 3.1 and 4.2):

.. math::

    x_t = \Lambda f_t + \varepsilon_t, \quad \varepsilon_t \sim N(0, \Psi),
    \qquad
    f_t = \sum_{i=1}^{p} A_i f_{t-i} + B u_t, \quad u_t \sim N(0, I_q).

**Step 1 (principal components).** On the balanced part of the standardised panel
of base-frequency predictors (the periods where every one of them is observed), the
first :math:`r` eigenvectors :math:`V_r` of the sample covariance give
:math:`\hat\Lambda = V_r` and :math:`\hat f_t = \hat\Lambda' x_t`. A VAR(:math:`p`)
fitted by OLS to :math:`\hat f_t` gives :math:`\hat A_i`; the :math:`q` leading
eigenvectors/eigenvalues of its residual covariance give
:math:`\hat B = P_q M_q^{1/2}`; :math:`\hat\Psi` is the diagonal of the covariance of
the idiosyncratic residuals :math:`x_t - \hat\Lambda\hat f_t`.

**Step 2 (Kalman smoother).** With these parameters the model is cast in state-space
form and the Kalman filter and smoother (:mod:`nowcastbox.statespace`) are run on the
full, unbalanced panel - including the ragged edge and a forecast horizon - giving
:math:`\hat f_{t|T}` for every period.

**Bridge equation.** A low-frequency target is linked to the factors by OLS
(:mod:`nowcastbox.models.bridge`),

.. math:: y_t^Q = \alpha + \beta' \bar f_t^Q + e_t,

with two variants (plan section 3.1):

* ``aggregate="factors"`` (``"2s_agg"``): :math:`\bar f^Q_t = \sum_j w_j f_{t-j}` with
  Mariano-Murasawa weights :math:`\tfrac19(1,2,3,2,1)` for months/quarters (or the
  chosen aggregation), evaluated at the last month of each quarter;
* ``aggregate="variables"`` (``"2s"``): the monthly predictors are first filtered with
  the aggregation weights so that, observed in the last month of a quarter, they are
  quarterly quantities (Giannone, Reichlin & Small, 2008, section 3); the factors of
  that panel are used directly at the quarter-end months.

Lower-frequency predictors (e.g. quarterly series other than the target) do not
enter the principal components; their loadings are estimated by OLS on the
(aggregated) principal-component factors and they enter the Kalman smoother with the
aggregation structure of Mariano & Murasawa (2003) on the lagged factors, so the
design works for any pair of frequencies with a fixed ratio (plan innovation I1).
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, overload

import numpy as np
import pandas as pd

from nowcastbox._logging import get_logger
from nowcastbox.core.base import BaseNowcaster
from nowcastbox.core.data import (
    FrequencySpec,
    MixedFrequencyData,
    StandardizationStats,
    as_mixed_frequency_data,
)
from nowcastbox.core.exceptions import DataQualityWarning, NowcastDataError
from nowcastbox.core.frequency import AggregationType, Frequency, is_period_end
from nowcastbox.core.results import FactorResults, build_nowcast_frame
from nowcastbox.models._pca import (
    FactorVAR,
    PrincipalComponents,
    fit_factor_var,
    principal_components,
    shock_loadings,
)
from nowcastbox.models.bridge import (
    BridgeRegression,
    fit_bridge_regression,
    resolve_aggregation_weights,
)
from nowcastbox.preprocessing.aggregation import rolling_aggregate
from nowcastbox.statespace import FilterMethod, SmootherResult, StateSpace, kalman_smoother

__all__ = ["AGGREGATE_OPTIONS", "TwoStepDFM", "TwoStepResults"]

logger = get_logger(__name__)

AGGREGATE_OPTIONS: dict[str, str] = {
    "factors": "factors",
    "2s_agg": "factors",
    "variables": "variables",
    "2s": "variables",
}
"""Accepted values of ``TwoStepDFM(aggregate=...)`` and their canonical name."""

AggregationSpec = str | AggregationType | Sequence[float] | np.ndarray | None


# ---------------------------------------------------------------------- results
@dataclass(frozen=True, kw_only=True, eq=False, repr=False)
class TwoStepResults(FactorResults):
    r"""Results of :class:`TwoStepDFM`.

    Besides the :class:`~nowcastbox.core.results.FactorResults` fields
    (``nowcast``, ``factors``, ``loadings``, ``transition``, ``shock_loadings``,
    ``idiosyncratic_variance`` ...) it holds:

    Parameters
    ----------
    eigenvalues : pandas.Series, optional
        Eigenvalues of the covariance matrix of the balanced standardised panel used
        for the principal components (index ``1..N``, for scree plots).
    bridge : BridgeRegression, optional
        Bridge regression of the target on the aggregated factors.
    aggregated_factors : pandas.DataFrame, optional
        Regressors of the bridge equation: factors aggregated to the target frequency
        (index = target periods).
    x_forecast : pandas.DataFrame, optional
        Smoothed common component plus mean of every predictor in the state-space
        model, **in original units** (destandardised), on the extended base grid
        (for ``aggregate="variables"`` the units are those of the filtered series).
    x_filled : pandas.DataFrame, optional
        Predictors with observed values kept and every storage slot without a value
        (gaps, ragged edge, forecast horizon) filled with ``x_forecast``.
    aggregate : str, optional
        ``"factors"`` or ``"variables"``.
    aggregation_weights : numpy.ndarray, optional
        Weights used to aggregate the factors to the target frequency (most recent
        base period first); ``[1.0]`` for ``aggregate="variables"``.
    state_space : StateSpace, optional
        State-space model used in the Kalman smoother (standardised data).
    model_data : MixedFrequencyData, optional
        The panel the model actually observes, on the grid of ``data`` and in original
        units: the target and every predictor, with the base-frequency predictors
        **filtered** by their aggregation weights for ``aggregate="variables"`` (equal
        to the selected estimation panel for ``aggregate="factors"``). It is what the
        state-space model's observation equation refers to, so linear analyses that
        need the observations of the model (news decompositions, parametric bootstrap,
        idiosyncratic residuals) should use it, or :meth:`filter_panel` for another
        vintage. ``TwoStepDFM().fit(model_data, target, prefiltered=True)``
        reproduces the estimates.
    variable_weights : dict of str to numpy.ndarray, optional
        Filter of each base-frequency predictor (most recent period first) for
        ``aggregate="variables"``; empty for ``aggregate="factors"`` or a target at the
        base frequency.

    Notes
    -----
    Every field of :class:`~nowcastbox.core.results.FactorResults` is also accepted (keyword-only).

    Examples
    --------
    >>> from nowcastbox.models import TwoStepDFM
    >>> from nowcastbox.models.two_step import simulate_two_step_example
    >>> data = simulate_two_step_example(random_state=0)
    >>> res = TwoStepDFM(n_factors=1).fit(data, "gdp")
    >>> res.A.shape, res.B.shape, res.Psi.shape
    ((1, 1), (1, 1), (10,))
    """

    eigenvalues: pd.Series | None = None
    bridge: BridgeRegression | None = None
    aggregated_factors: pd.DataFrame | None = None
    x_forecast: pd.DataFrame | None = None
    x_filled: pd.DataFrame | None = None
    aggregate: str | None = None
    aggregation_weights: np.ndarray | None = None
    state_space: StateSpace | None = None
    model_data: MixedFrequencyData | None = None
    variable_weights: dict[str, np.ndarray] | None = None

    @property
    def is_filtered(self) -> bool:
        """Whether the state-space model observes filtered predictors (``"variables"``)."""
        return bool(self.variable_weights)

    def filter_panel(
        self,
        data: MixedFrequencyData | pd.DataFrame,
        *,
        frequency: FrequencySpec | None = None,
    ) -> MixedFrequencyData:
        r"""Apply the predictor filters of the fit to another panel (e.g. a vintage).

        For ``aggregate="variables"`` every base-frequency predictor :math:`x_{i,t}`
        is replaced by :math:`\tilde x_{i,t} = \sum_j w_{i,j} x_{i,t-j}` (NaN when a
        value with a non-zero weight is missing), exactly as in :meth:`TwoStepDFM.fit`;
        the target and lower-frequency predictors are unchanged. For
        ``aggregate="factors"`` the panel is returned unchanged (restricted to the
        model's series). The result is linear in the raw values, so a raw release
        maps to the filtered observations that it completes (and a revision to every
        filtered value whose window contains it).

        Parameters
        ----------
        data : MixedFrequencyData or pandas.DataFrame
            Panel with the target and every predictor of the fit (raw, unfiltered).
        frequency : frequency specification, optional
            Per-series frequencies for DataFrame input.

        Returns
        -------
        MixedFrequencyData
            Panel observed by the state-space model, on the grid of ``data``.

        Raises
        ------
        ValueError
            If the results do not store the model panel (built by hand).
        NowcastDataError
            If ``data`` lacks a series of the model.

        Examples
        --------
        >>> from nowcastbox.models import TwoStepDFM
        >>> from nowcastbox.models.two_step import simulate_two_step_example
        >>> data = simulate_two_step_example(random_state=0)
        >>> res = TwoStepDFM(n_factors=1, aggregate="variables").fit(data, "gdp")
        >>> filtered = res.filter_panel(data)
        >>> bool(filtered.equals(res.model_data))
        True
        >>> int(filtered.n_observations()["x1"]) == int(data.n_observations()["x1"]) - 4
        True
        """
        if self.model_data is None or self.variable_weights is None:
            raise ValueError("These TwoStepDFM results do not store the model panel.")
        panel = as_mixed_frequency_data(data, frequency)
        needed = list(self.model_data.columns)
        missing = [c for c in needed if c not in panel.columns]
        if missing:
            raise NowcastDataError(f"The panel lacks the series {missing} of the model.")
        return _apply_filters(panel.select(needed), self.variable_weights)

    @property
    def A(self) -> np.ndarray:  # noqa: N802
        r"""Stacked VAR matrices :math:`[A_1, \dots, A_p]` (``transition``)."""
        if self.transition is None:
            raise ValueError("No transition matrix stored.")
        return self.transition.copy()

    @property
    def B(self) -> np.ndarray:  # noqa: N802
        """Shock loadings :math:`B` (``shock_loadings``), shape ``(r, q)``."""
        if self.shock_loadings is None:
            raise ValueError("No shock loadings stored.")
        return self.shock_loadings.copy()

    @property
    def BB(self) -> np.ndarray:  # noqa: N802
        """Covariance :math:`BB'` of the factor innovations (rank ``q``)."""
        b = self.B
        return b @ b.T

    @property
    def Psi(self) -> pd.Series:  # noqa: N802
        r"""Diagonal of the idiosyncratic covariance :math:`\Psi` (standardised units)."""
        if self.idiosyncratic_variance is None:
            raise ValueError("No idiosyncratic variances stored.")
        return self.idiosyncratic_variance.copy()

    @property
    def explained_variance_ratio(self) -> pd.Series:
        """Share of the variance of the balanced panel explained by each component."""
        if self.eigenvalues is None:
            raise ValueError("No eigenvalues stored.")
        ev = self.eigenvalues
        return (ev / float(ev.sum())).rename("explained_variance_ratio")

    def _summary_sections(self) -> list[tuple[str, list[str]]]:
        sections = super()._summary_sections()
        lines = [f"  {'Aggregation':<22}{self.aggregate}"]
        if self.eigenvalues is not None and self.n_factors is not None:
            share = float(self.explained_variance_ratio.iloc[: self.n_factors].sum())
            lines.append(f"  {'Variance explained':<22}{share:.4f}")
        sections.append(("Principal components", lines))
        if self.bridge is not None:
            blines = [
                f"  {'Observations':<22}{self.bridge.n_obs}",
                f"  {'R-squared':<22}{self.bridge.rsquared:.4f}",
                f"  {'Residual std':<22}{self.bridge.sigma:.4f}",
            ]
            blines += [f"  {k:<22}{v:.4f}" for k, v in self.bridge.params.items()]
            sections.append(("Bridge equation", blines))
        return sections


# ---------------------------------------------------------------------- spec
@dataclass(frozen=True)
class _FittedSpec:
    """Everything needed to re-run step 2 and the bridge on a new panel."""

    target: str
    predictors: list[str]
    state_columns: list[str]
    stats: StandardizationStats
    state_space: StateSpace
    n_factors: int
    factor_weights: np.ndarray
    variable_weights: dict[str, np.ndarray]
    bridge: BridgeRegression


@dataclass(frozen=True)
class _FirstStep:
    """Output of the principal-component step."""

    pca: PrincipalComponents
    var: FactorVAR
    shock_loadings: np.ndarray
    shock_eigenvalues: np.ndarray
    factors_on_grid: np.ndarray
    n_balanced: int
    balanced_start: pd.Period
    balanced_end: pd.Period


_FILTER_METHODS = ("auto", "univariate", "multivariate")


# ---------------------------------------------------------------------- estimator
class TwoStepDFM(BaseNowcaster):
    r"""Two-step dynamic factor model (Giannone, Reichlin & Small, 2008).

    Parameters
    ----------
    n_factors : int, default 2
        Number of static factors :math:`r`.
    factor_lags : int, default 1
        Order :math:`p` of the factor VAR.
    n_shocks : int, optional
        Number of dynamic shocks :math:`q \le r` (default :math:`q = r`).
    aggregate : {"factors", "variables", "2s_agg", "2s"}, default "factors"
        How the target frequency is bridged: aggregate the monthly factors
        (``"factors"``/``"2s_agg"``) or estimate the factors on predictors already
        filtered to target-frequency quantities (``"variables"``/``"2s"``).
    aggregation : str, AggregationType or sequence of float, optional
        Temporal aggregation linking base-frequency quantities to lower-frequency
        series: ``"mariano_murasawa"``, ``"average"``, ``"flow"``, ``"stock"`` or
        explicit weights (most recent period first). Default: each lower-frequency
        series' ``aggregation`` metadata, else Mariano-Murasawa. For the factors the
        named weights are normalised to sum one (:math:`\tfrac19(1,2,3,2,1)`), for
        filtered variables by the frequency ratio (:math:`\tfrac13(1,2,3,2,1)`).
    horizon : int, default 1
        Number of target periods forecast after the target period that contains the
        last base period of the panel.
    filter_method : {"auto", "univariate", "multivariate"}, default "auto"
        Kalman filter variant (:func:`nowcastbox.statespace.kalman_filter`).
    collapse : bool, default False
        Collapse the observation vector (Jungbacker & Koopman, 2015) - faster for
        large panels, identical results.
    idio_variance_floor : float, default 1e-4
        Floor on :math:`\Psi_{ii}` (standardised units); flooring emits a
        :class:`~nowcastbox.core.exceptions.DataQualityWarning`.
    ddof : int, default 1
        Delta degrees of freedom of the standardisation.
    bridge_cov_type : str, default "nonrobust"
        Covariance estimator of the bridge coefficients (statsmodels ``cov_type``).

    Attributes
    ----------
    results_ : TwoStepResults
        Results of the last fit.

    Notes
    -----
    The ``nowcast`` frame has an extra column ``std``: the standard deviation of the
    bridge prediction, :math:`\sqrt{\beta' V_t \beta + \sigma_e^2}`, with
    :math:`V_t` the smoothed covariance of the aggregated factors (filtering
    uncertainty) and :math:`\sigma_e` the bridge residual standard error; parameter
    uncertainty is ignored.

    References
    ----------
    Giannone, D., Reichlin, L. & Small, D. (2008). Nowcasting: The real-time
    informational content of macroeconomic data. *Journal of Monetary Economics*,
    55(4), 665-676.

    Doz, C., Giannone, D. & Reichlin, L. (2011). A two-step estimator for large
    approximate dynamic factor models based on Kalman filtering. *Journal of
    Econometrics*, 164(1), 188-205.

    Mariano, R. S. & Murasawa, Y. (2003). A new coincident index of business cycles
    based on monthly and quarterly series. *Journal of Applied Econometrics*, 18(4),
    427-443.

    Examples
    --------
    >>> from nowcastbox.models import TwoStepDFM
    >>> from nowcastbox.models.two_step import simulate_two_step_example
    >>> data = simulate_two_step_example(random_state=0)
    >>> model = TwoStepDFM(n_factors=1, factor_lags=1, aggregate="factors")
    >>> res = model.fit(data, target="gdp")
    >>> res.factors.shape[1]
    1
    >>> list(res.nowcast.columns)
    ['observed', 'in_sample', 'out_of_sample', 'std']
    >>> bool(res.bridge.rsquared > 0.5)
    True
    """

    def __init__(
        self,
        n_factors: int = 2,
        factor_lags: int = 1,
        n_shocks: int | None = None,
        aggregate: Literal["factors", "variables", "2s_agg", "2s"] = "factors",
        aggregation: AggregationSpec = None,
        horizon: int = 1,
        filter_method: Literal["auto", "univariate", "multivariate"] = "auto",
        collapse: bool = False,
        idio_variance_floor: float = 1e-4,
        ddof: int = 1,
        bridge_cov_type: str = "nonrobust",
    ) -> None:
        self.n_factors = n_factors
        self.factor_lags = factor_lags
        self.n_shocks = n_shocks
        self.aggregate = aggregate
        self.aggregation = aggregation
        self.horizon = horizon
        self.filter_method = filter_method
        self.collapse = collapse
        self.idio_variance_floor = idio_variance_floor
        self.ddof = ddof
        self.bridge_cov_type = bridge_cov_type

    _spec: _FittedSpec | None = None

    # ------------------------------------------------------------------ validation
    def _validate_params(self) -> None:
        r = _check_int("n_factors", self.n_factors, 1)
        _check_int("factor_lags", self.factor_lags, 1)
        _check_int("horizon", self.horizon, 0)
        _check_int("ddof", self.ddof, 0)
        if self.n_shocks is not None:
            q = _check_int("n_shocks", self.n_shocks, 1)
            if q > r:
                raise ValueError(f"n_shocks ({q}) cannot exceed n_factors ({r}).")
        if self.aggregate not in AGGREGATE_OPTIONS:
            raise ValueError(
                f"aggregate must be one of {sorted(AGGREGATE_OPTIONS)}, got {self.aggregate!r}."
            )
        if self.filter_method not in _FILTER_METHODS:
            raise ValueError(
                f"filter_method must be one of {_FILTER_METHODS}, got {self.filter_method!r}."
            )
        if not isinstance(self.collapse, bool):
            raise ValueError(f"collapse must be a bool, got {self.collapse!r}.")
        floor = self.idio_variance_floor
        if isinstance(floor, bool) or not isinstance(floor, int | float) or not floor > 0:
            raise ValueError(f"idio_variance_floor must be > 0, got {floor!r}.")
        _check_aggregation(self.aggregation)

    def _reset(self) -> None:
        super()._reset()
        self._spec = None

    @property
    def _mode(self) -> str:
        return AGGREGATE_OPTIONS[str(self.aggregate)]

    @property
    def _method(self) -> FilterMethod:
        method = str(self.filter_method)
        if method == "univariate":
            return "univariate"
        if method == "multivariate":
            return "multivariate"
        return "auto"

    # ------------------------------------------------------------------ helpers
    def _weights_for(
        self, data: MixedFrequencyData, column: str, low: Frequency, normalize: str
    ) -> np.ndarray:
        """Aggregation weights from the base frequency to ``low`` for ``column``."""
        spec: AggregationSpec = self.aggregation
        if spec is None:
            spec = data.metadata[column].aggregation
        return resolve_aggregation_weights(spec, data.base_frequency, low, normalize=normalize)

    def _model_panel(
        self,
        data: MixedFrequencyData,
        target: str,
        predictors: list[str],
        variable_weights: dict[str, np.ndarray],
        prefiltered: bool = False,
    ) -> tuple[MixedFrequencyData, MixedFrequencyData, pd.PeriodIndex]:
        """Model panel (target + predictors, filtered for ``"variables"``) and extended grid.

        Returns the filtered panel on the grid of ``data`` (stored as
        ``results.model_data``), the predictor panel extended to the forecast horizon
        and the target periods.
        """
        target_freq = data.metadata[target].frequency
        pf = target_freq.pandas_freq
        periods = pd.period_range(
            data.start.asfreq(pf), data.end.asfreq(pf) + int(self.horizon), freq=pf
        )
        grid_end = periods[-1].asfreq(data.base_frequency.pandas_freq, how="E")
        model_data = data.select([target, *predictors])
        if not prefiltered:
            model_data = _apply_filters(model_data, variable_weights)
        panel = model_data.select(predictors).extend(int((grid_end - data.end).n))
        return model_data, panel, pd.PeriodIndex(periods)

    def _aggregation_setup(
        self, data: MixedFrequencyData, target: str, base_cols: list[str]
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Factor aggregation weights and per-variable filters for the chosen variant."""
        target_freq = data.metadata[target].frequency
        if target_freq == data.base_frequency:
            return np.ones(1), {}
        if self._mode == "factors":
            return self._weights_for(data, target, target_freq, "sum"), {}
        return np.ones(1), {c: self._weights_for(data, c, target_freq, "ratio") for c in base_cols}

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        data: MixedFrequencyData | pd.DataFrame,
        target: str,
        *,
        frequency: FrequencySpec | None = None,
        **fit_kwargs: Any,
    ) -> TwoStepResults:
        """Estimate the two-step model and nowcast ``target``.

        Parameters
        ----------
        data : MixedFrequencyData or pandas.DataFrame
            Panel on the base (monthly) grid; quarterly values in the third month.
        target : str
            Target name or formula (``"gdp ~ ."``, ``"gdp ~ ip + pmi"``).
        frequency : frequency specification, optional
            Per-series frequencies for DataFrame input.
        **fit_kwargs
            ``prefiltered=True`` declares that the base-frequency predictors of
            ``data`` are already filtered to target-frequency quantities (e.g.
            ``results.model_data`` or a panel simulated from the fitted state-space
            model, as in a parametric bootstrap), so the
            ``aggregate="variables"`` filter is not applied again; it has no effect
            for ``aggregate="factors"``. Any other option raises ``TypeError``.

        Returns
        -------
        TwoStepResults
            Estimation results (also stored in :attr:`results_`).

        Raises
        ------
        NowcastDataError
            If the data cannot support the model (too few predictors, too short
            balanced panel, target without observations ...).
        ValueError
            If a hyper-parameter is invalid.

        Examples
        --------
        >>> from nowcastbox.models import TwoStepDFM
        >>> from nowcastbox.models.two_step import simulate_two_step_example
        >>> res = TwoStepDFM(n_factors=1).fit(simulate_two_step_example(random_state=0), "gdp")
        >>> str(res.nowcast.index[-1])
        '2015Q1'
        """
        results = super().fit(data, target, frequency=frequency, **fit_kwargs)
        assert isinstance(results, TwoStepResults)  # noqa: S101
        return results

    def _fit(self, data: MixedFrequencyData, target: str, **fit_kwargs: Any) -> TwoStepResults:
        prefiltered = _check_prefiltered(fit_kwargs.pop("prefiltered", False))
        if fit_kwargs:
            raise TypeError(f"Unexpected fit options {sorted(fit_kwargs)}.")
        started = time.perf_counter()
        r = int(self.n_factors)
        base = data.base_frequency
        predictors = [c for c in data.columns if c != target]
        base_cols = [c for c in predictors if data.metadata[c].frequency == base]
        low_cols = [c for c in predictors if data.metadata[c].frequency != base]
        if len(base_cols) < r:
            raise NowcastDataError(
                f"TwoStepDFM needs at least n_factors={r} {base.label} predictors for the "
                f"principal components; got {len(base_cols)}."
            )
        if int(data.n_observations()[target]) == 0:
            raise NowcastDataError(f"The target {target!r} has no observations.")
        factor_weights, variable_weights = self._aggregation_setup(data, target, base_cols)
        model_data, x_panel, periods = self._model_panel(
            data, target, predictors, variable_weights, prefiltered
        )
        z_panel, stats = x_panel.standardize(ddof=int(self.ddof))
        z = z_panel.to_frame()

        first = self._first_step(z, base_cols, base.label)
        low = self._low_frequency_loadings(data, z, low_cols, first.factors_on_grid)
        state_cols = base_cols + [c for c, *_ in low]
        lam_all = np.vstack([first.pca.loadings, *(lam for _, lam, _, _ in low)])
        psi_all = self._floor_psi(
            state_cols,
            np.concatenate([first.pca.idiosyncratic_variance, [psi for *_, psi, _ in low]]),
        )
        weights = [np.ones(1)] * len(base_cols) + [w for *_, w in low]
        ssm = _build_state_space(
            first.var.coefficients,
            first.shock_loadings,
            lam_all,
            psi_all,
            weights,
            max(int(self.factor_lags), factor_weights.size, *(w.size for w in weights)),
            first.pca.factors,
        )
        smoothed, agg_factors, agg_cov = self._second_step(
            ssm, z[state_cols], factor_weights, periods
        )
        y_obs = data.to_native(target).reindex(periods).rename(target)
        regression = fit_bridge_regression(
            y_obs, agg_factors, add_constant=True, cov_type=self.bridge_cov_type
        )
        self._spec = _FittedSpec(
            target=target,
            predictors=predictors,
            state_columns=state_cols,
            stats=stats,
            state_space=ssm,
            n_factors=r,
            factor_weights=factor_weights,
            variable_weights=variable_weights,
            bridge=regression,
        )
        info: dict[str, Any] = {
            "n_balanced_periods": first.n_balanced,
            "balanced_start": first.balanced_start,
            "balanced_end": first.balanced_end,
            "shock_eigenvalues": first.shock_eigenvalues,
            "excluded_predictors": sorted(set(low_cols) - set(state_cols)),
            "prefiltered": prefiltered,
        }
        eigenvalues = pd.Series(
            first.pca.eigenvalues,
            index=pd.RangeIndex(1, len(first.pca.eigenvalues) + 1, name="component"),
            name="eigenvalue",
        )
        results = self._assemble(
            data=data,
            smoothed=smoothed,
            agg=(agg_factors, agg_cov),
            y_obs=y_obs,
            spec=self._spec,
            x_panel=x_panel,
            model_data=model_data,
            loadings=pd.DataFrame(lam_all, index=state_cols, columns=_factor_names(r)),
            eigenvalues=eigenvalues,
            residual_cov=first.var.residual_cov,
            info=info,
        )
        results.info["fit_time"] = time.perf_counter() - started
        logger.debug("TwoStepDFM fitted in %.3fs", results.info["fit_time"])
        return results

    def _first_step(self, z: pd.DataFrame, base_cols: list[str], label: str) -> _FirstStep:
        """Principal components, factor VAR and shock loadings on the balanced panel."""
        r = int(self.n_factors)
        p = int(self.factor_lags)
        q = r if self.n_shocks is None else int(self.n_shocks)
        zb = z[base_cols].to_numpy(dtype=float)
        complete = np.isfinite(zb).all(axis=1)
        n_bal = int(complete.sum())
        if n_bal < max(2 * r, p + r * p + 1):
            raise NowcastDataError(
                f"The balanced part of the panel has only {n_bal} periods; increase the "
                "sample, drop short series or reduce n_factors/factor_lags."
            )
        idx = np.flatnonzero(complete)
        interior = int((~complete[idx[0] : idx[-1] + 1]).sum())
        if interior:
            warnings.warn(
                f"{interior} periods inside the sample have missing values in some "
                f"{label} predictor and are excluded from the principal components.",
                DataQualityWarning,
                stacklevel=5,
            )
        pca = principal_components(zb[complete], r)
        f_full = np.full((len(z), r), np.nan)
        f_full[complete] = pca.factors
        var = fit_factor_var(f_full, p)
        b_mat, shock_eig = shock_loadings(var.residual_cov, q)
        return _FirstStep(
            pca=pca,
            var=var,
            shock_loadings=b_mat,
            shock_eigenvalues=shock_eig,
            factors_on_grid=f_full,
            n_balanced=n_bal,
            balanced_start=z.index[idx[0]],
            balanced_end=z.index[idx[-1]],
        )

    def _low_frequency_loadings(
        self,
        data: MixedFrequencyData,
        z: pd.DataFrame,
        low_cols: list[str],
        factors: np.ndarray,
    ) -> list[tuple[str, np.ndarray, float, np.ndarray]]:
        """OLS loadings of lower-frequency predictors on aggregated PC factors."""
        out: list[tuple[str, np.ndarray, float, np.ndarray]] = []
        r = factors.shape[1]
        for c in low_cols:
            freq_c = data.metadata[c].frequency
            w_c = (
                self._weights_for(data, c, freq_c, "ratio")
                if self._mode == "factors"
                else np.ones(1)
            )
            g = np.column_stack([rolling_aggregate(factors[:, k], w_c) for k in range(r)])
            yc = z[c].to_numpy(dtype=float)
            ok = np.isfinite(yc) & np.isfinite(g).all(axis=1)
            if int(ok.sum()) < r + 2:
                warnings.warn(
                    f"Predictor {c!r} has only {int(ok.sum())} observations overlapping the "
                    "balanced panel; it is excluded from the model.",
                    DataQualityWarning,
                    stacklevel=5,
                )
                continue
            lam, *_ = np.linalg.lstsq(g[ok], yc[ok], rcond=None)
            resid = yc[ok] - g[ok] @ lam
            out.append((c, lam, float(np.mean(resid**2)), w_c))
        return out

    def _floor_psi(self, columns: list[str], psi: np.ndarray) -> np.ndarray:
        """Apply :attr:`idio_variance_floor`, warning about the affected series."""
        floor = float(self.idio_variance_floor)
        floored = [c for c, v in zip(columns, psi, strict=True) if v < floor]
        if floored:
            warnings.warn(
                f"Idiosyncratic variance of {floored} below {floor}; set to the floor "
                "(idio_variance_floor).",
                DataQualityWarning,
                stacklevel=5,
            )
        return np.maximum(psi, floor)

    def _second_step(
        self,
        ssm: StateSpace,
        z: pd.DataFrame,
        factor_weights: np.ndarray,
        periods: pd.PeriodIndex,
    ) -> tuple[SmootherResult, pd.DataFrame, np.ndarray]:
        """Kalman smoother on the unbalanced panel and aggregation to target periods."""
        smoothed = kalman_smoother(
            ssm, z.to_numpy(dtype=float), method=self._method, collapse=bool(self.collapse)
        )
        index = z.index
        assert isinstance(index, pd.PeriodIndex)  # noqa: S101
        agg_factors, agg_cov = _aggregate_state(
            smoothed, int(self.n_factors), factor_weights, index, periods
        )
        return smoothed, agg_factors, agg_cov

    def _assemble(
        self,
        *,
        data: MixedFrequencyData,
        smoothed: SmootherResult,
        agg: tuple[pd.DataFrame, np.ndarray],
        y_obs: pd.Series,
        spec: _FittedSpec,
        x_panel: MixedFrequencyData,
        model_data: MixedFrequencyData,
        loadings: pd.DataFrame,
        eigenvalues: pd.Series | None,
        residual_cov: np.ndarray | None,
        info: dict[str, Any],
    ) -> TwoStepResults:
        r = spec.n_factors
        ssm = spec.state_space
        agg_factors, agg_cov = agg
        grid = x_panel.index
        factors = pd.DataFrame(smoothed.smoothed_state[:, :r], index=grid, columns=_factor_names(r))
        factors.index.name = "period"
        regression = spec.bridge
        estimate = regression.predict(agg_factors)
        first = y_obs.first_valid_index()
        if first is not None:  # periods before the first target observation: no backcast
            estimate = estimate.where(estimate.index >= first)
        beta = regression.params.to_numpy(dtype=float)[1:]
        factor_var = np.einsum("i,tij,j->t", beta, agg_cov, beta)
        std = pd.Series(np.sqrt(factor_var + regression.sigma**2), index=agg_factors.index)
        frame = build_nowcast_frame(y_obs, estimate, extra={"std": std.where(estimate.notna())})

        signal = pd.DataFrame(smoothed.smoothed_signal(), index=grid, columns=spec.state_columns)
        x_forecast = spec.stats.inverse_transform(signal)
        observed_x = x_panel.to_frame()[spec.state_columns]
        slots = _slot_mask(x_panel, spec.state_columns)
        x_filled = observed_x.where(observed_x.notna(), x_forecast.where(slots))

        psi = pd.Series(ssm.obs_cov_diagonal, index=spec.state_columns, name="Psi")
        transition = ssm.T[:r, : r * int(self.factor_lags)].copy()
        b_mat = ssm.R[:r].copy()
        params: dict[str, Any] = {
            "A": transition,
            "B": b_mat,
            "Psi": psi.copy(),
            "loadings": loadings.copy(),
            "bridge_coefficients": regression.params.copy(),
            "bridge_sigma": regression.sigma,
            "factor_aggregation_weights": spec.factor_weights.copy(),
        }
        params["residual_cov"] = None if residual_cov is None else residual_cov.copy()
        params["eigenvalues"] = None if eigenvalues is None else eigenvalues.copy()
        return TwoStepResults(
            target=spec.target,
            nowcast=frame,
            model_name="TwoStepDFM",
            model_params=self.get_params(deep=False),
            factors=factors,
            loadings=loadings,
            params=params,
            loglikelihood=smoothed.loglikelihood,
            data=data,
            standardization=spec.stats,
            info=info,
            transition=transition,
            shock_loadings=b_mat,
            factor_lags=int(self.factor_lags),
            n_shocks=int(b_mat.shape[1]),
            idiosyncratic_variance=psi,
            eigenvalues=eigenvalues,
            bridge=regression,
            aggregated_factors=agg_factors,
            x_forecast=x_forecast,
            x_filled=x_filled,
            aggregate=self._mode,
            aggregation_weights=spec.factor_weights.copy(),
            state_space=ssm,
            model_data=model_data,
            variable_weights={k: v.copy() for k, v in spec.variable_weights.items()},
        )

    # ------------------------------------------------------------------ update
    def update(
        self,
        data: MixedFrequencyData | pd.DataFrame,
        *,
        frequency: FrequencySpec | None = None,
        prefiltered: bool = False,
    ) -> TwoStepResults:
        r"""Re-run the Kalman smoother and the bridge on new data with fixed parameters.

        The parameters estimated by :meth:`fit` (standardisation, loadings, VAR,
        :math:`\Psi`, bridge coefficients) are kept; only the smoothed factors and the
        nowcasts are recomputed. This is the information update used in real-time
        exercises and news decompositions (new releases of the same panel).

        Parameters
        ----------
        data : MixedFrequencyData or pandas.DataFrame
            New panel (e.g. a later vintage) containing the target and every
            predictor used in the fit, with the same frequencies.
        frequency : frequency specification, optional
            Per-series frequencies for DataFrame input.
        prefiltered : bool, default False
            The base-frequency predictors are already filtered (see :meth:`fit`).

        Returns
        -------
        TwoStepResults
            Results on the new information set (the fitted results in
            :attr:`results_` are not changed).

        Raises
        ------
        ModelNotFittedError
            If the model has not been fitted.
        NowcastDataError
            If series are missing from ``data`` or their frequency changed.

        Examples
        --------
        >>> from nowcastbox.models import TwoStepDFM
        >>> from nowcastbox.models.two_step import simulate_two_step_example
        >>> data = simulate_two_step_example(random_state=0)
        >>> model = TwoStepDFM(n_factors=1)
        >>> res = model.fit(data, "gdp")
        >>> again = model.update(data)
        >>> bool(abs(again.get_nowcast() - res.get_nowcast()) < 1e-10)
        True
        """
        self._check_fitted()
        spec = self._spec
        fitted = self.results_
        assert spec is not None  # noqa: S101
        assert isinstance(fitted, TwoStepResults)  # noqa: S101
        assert fitted.data is not None  # noqa: S101
        panel = as_mixed_frequency_data(data, frequency)
        needed = [spec.target, *spec.predictors]
        missing = [c for c in needed if c not in panel.columns]
        if missing:
            raise NowcastDataError(f"update() data lack the series {missing}.")
        panel = panel.select(needed)
        if panel.base_frequency != fitted.data.base_frequency:
            raise NowcastDataError("The base frequency differs from the fitted data.")
        changed = [
            c for c in needed if panel.metadata[c].frequency != fitted.data.metadata[c].frequency
        ]
        if changed:
            raise NowcastDataError(f"The frequency of {changed} differs from the fitted data.")
        model_data, x_panel, periods = self._model_panel(
            panel,
            spec.target,
            spec.predictors,
            spec.variable_weights,
            _check_prefiltered(prefiltered),
        )
        z_panel, _ = x_panel.standardize(spec.stats)
        z = z_panel.to_frame()[spec.state_columns]
        smoothed, agg_factors, agg_cov = self._second_step(
            spec.state_space, z, spec.factor_weights, periods
        )
        y_obs = panel.to_native(spec.target).reindex(periods).rename(spec.target)
        info = dict(fitted.info)
        info["updated"] = True
        assert fitted.loadings is not None  # noqa: S101
        return self._assemble(
            data=panel,
            smoothed=smoothed,
            agg=(agg_factors, agg_cov),
            y_obs=y_obs,
            spec=spec,
            x_panel=x_panel,
            model_data=model_data,
            loadings=fitted.loadings,
            eigenvalues=fitted.eigenvalues,
            residual_cov=fitted.params.get("residual_cov"),
            info=info,
        )


def _factor_names(n_factors: int) -> list[str]:
    """Column names ``f1..fr`` of the factors (core contract)."""
    return [f"f{k + 1}" for k in range(n_factors)]


def _check_int(name: str, value: object, minimum: int) -> int:
    """Validate an integer hyper-parameter (``ValueError`` otherwise)."""
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        raise ValueError(f"{name} must be an integer, got {value!r}.")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}.")
    return int(value)


def _slot_mask(panel: MixedFrequencyData, columns: list[str]) -> pd.DataFrame:
    """Storage-slot mask of ``columns`` (one :func:`is_period_end` call per frequency)."""
    by_freq: dict[Frequency, np.ndarray] = {}
    out: dict[str, np.ndarray] = {}
    for col in columns:
        freq = panel.metadata[col].frequency
        if freq not in by_freq:
            by_freq[freq] = (
                np.ones(panel.n_periods, dtype=bool)
                if freq == panel.base_frequency
                else is_period_end(panel.index, freq)
            )
        out[col] = by_freq[freq]
    return pd.DataFrame(out, index=panel.index, columns=columns)


def _check_prefiltered(value: object) -> bool:
    """Validate the ``prefiltered`` option (``ValueError`` otherwise)."""
    if not isinstance(value, bool | np.bool_):
        raise ValueError(f"prefiltered must be a bool, got {value!r}.")
    return bool(value)


def _apply_filters(panel: MixedFrequencyData, weights: dict[str, np.ndarray]) -> MixedFrequencyData:
    """Filter the columns of ``weights`` with :func:`rolling_aggregate` (others unchanged)."""
    if not weights:
        return panel
    frame = panel.to_frame()
    for col, w in weights.items():
        frame[col] = rolling_aggregate(frame[col].to_numpy(dtype=float), w)
    return panel.with_data(frame)


def _check_aggregation(aggregation: object) -> None:
    """Validate the ``aggregation`` hyper-parameter (``ValueError`` otherwise)."""
    if aggregation is None or isinstance(aggregation, AggregationType):
        return
    if isinstance(aggregation, str):
        AggregationType.from_value(aggregation)
        return
    try:
        w = np.asarray(aggregation, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid aggregation {aggregation!r}.") from exc
    if w.ndim != 1 or w.size == 0 or not bool(np.isfinite(w).all()) or not w.any():
        raise ValueError("Explicit aggregation weights must be a non-zero, finite 1-D array.")


def _build_state_space(
    coefficients: np.ndarray,
    b_mat: np.ndarray,
    lam: np.ndarray,
    psi: np.ndarray,
    weights: list[np.ndarray],
    n_lags: int,
    pc_factors: np.ndarray,
) -> StateSpace:
    r"""State-space form with state :math:`\alpha_t = (f_t', \dots, f_{t-L+1}')'`.

    Transition: companion form of the VAR padded with zero blocks up to ``n_lags``;
    selection :math:`(B', 0)'` with :math:`Q = I_q`; series ``i`` loads on
    :math:`(w_{i,0}\lambda_i', w_{i,1}\lambda_i', \dots)` (aggregation structure);
    :math:`H = \mathrm{diag}(\Psi)`. The initial state is the stationary distribution,
    or - for a non-stationary VAR, with a warning - zero mean and the sample
    covariance of the principal-component factors.
    """
    r = coefficients.shape[0]
    p = coefficients.shape[1] // r
    m = r * n_lags
    transition = np.zeros((m, m))
    transition[:r, : r * p] = coefficients
    if n_lags > 1:
        transition[r:, : m - r] = np.eye(m - r)
    selection = np.zeros((m, b_mat.shape[1]))
    selection[:r] = b_mat
    design = np.zeros((lam.shape[0], m))
    for i, w in enumerate(weights):
        for j, wj in enumerate(w):
            design[i, j * r : (j + 1) * r] = wj * lam[i]
    initial_cov = None
    radius = float(np.abs(np.linalg.eigvals(transition)).max())
    if radius >= 1.0 - 1e-8:
        warnings.warn(
            f"The estimated factor VAR is not stationary (max |eigenvalue| = {radius:.4f}); "
            "the Kalman smoother starts from the sample covariance of the "
            "principal-component factors.",
            DataQualityWarning,
            stacklevel=4,
        )
        sample_cov = np.atleast_2d(np.cov(pc_factors, rowvar=False))
        initial_cov = np.kron(np.eye(n_lags), sample_cov)
    return StateSpace(
        transition,
        design,
        np.eye(b_mat.shape[1]),
        psi,
        selection=selection,
        initial_state_cov=initial_cov,
    )


def _aggregate_state(
    smoothed: SmootherResult,
    n_factors: int,
    weights: np.ndarray,
    base_index: pd.PeriodIndex,
    periods: pd.PeriodIndex,
) -> tuple[pd.DataFrame, np.ndarray]:
    r"""Aggregated factors :math:`\sum_j w_j f_{t-j|T}` and their covariances at target periods."""
    r = n_factors
    m = smoothed.smoothed_state.shape[1]
    sel = np.zeros((m, r))
    for j, wj in enumerate(weights):
        sel[j * r : (j + 1) * r] = wj * np.eye(r)
    ends = periods.asfreq(base_index.freqstr, how="E")
    pos = base_index.get_indexer(ends)
    values = np.full((len(periods), r), np.nan)
    covs = np.full((len(periods), r, r), np.nan)
    ok = pos >= weights.size - 1
    values[ok] = smoothed.smoothed_state[pos[ok]] @ sel
    covs[ok] = np.einsum("ki,tkl,lj->tij", sel, smoothed.smoothed_state_cov[pos[ok]], sel)
    frame = pd.DataFrame(values, index=periods, columns=[f"f{k + 1}" for k in range(r)])
    frame.index.name = "period"
    return frame, covs


@overload
def simulate_two_step_example(
    n_periods: int = ...,
    n_series: int = ...,
    *,
    n_factors: int = ...,
    ragged_edge: Sequence[int] | None = ...,
    return_factors: Literal[False] = ...,
    random_state: int | np.random.Generator | None = ...,
) -> MixedFrequencyData: ...


@overload
def simulate_two_step_example(
    n_periods: int = ...,
    n_series: int = ...,
    *,
    n_factors: int = ...,
    ragged_edge: Sequence[int] | None = ...,
    return_factors: Literal[True],
    random_state: int | np.random.Generator | None = ...,
) -> tuple[MixedFrequencyData, pd.DataFrame]: ...


def simulate_two_step_example(
    n_periods: int = 180,
    n_series: int = 10,
    *,
    n_factors: int = 1,
    ragged_edge: Sequence[int] | None = None,
    return_factors: bool = False,
    random_state: int | np.random.Generator | None = None,
) -> MixedFrequencyData | tuple[MixedFrequencyData, pd.DataFrame]:
    r"""Simulate a small monthly panel with a quarterly target driven by common factors.

    Monthly predictors follow :math:`x_t = \Lambda f_t + e_t` with a VAR(1) factor;
    the quarterly target (stored in the third month) is
    :math:`y_t^Q = 0.5 + \tfrac19\sum_j w_j\,\iota' f_{t-j} + u_t` with
    Mariano-Murasawa weights. The last quarter of the target is missing and the
    predictors have a ragged edge. Used in examples and tests.

    Parameters
    ----------
    n_periods : int, default 180
        Number of months (multiple of 3 recommended).
    n_series : int, default 10
        Number of monthly predictors ``x1..xN``.
    n_factors : int, default 1
        Number of true factors.
    ragged_edge : sequence of int, optional
        Number of missing final months of each predictor (cycled over series).
        Default ``(0, 1, 2)``.
    return_factors : bool, default False
        Also return the true factors.
    random_state : int or numpy.random.Generator, optional
        Seed.

    Returns
    -------
    data : MixedFrequencyData
        Panel with columns ``x1..xN`` (monthly) and ``gdp`` (quarterly).
    factors : pandas.DataFrame
        True factors (only when ``return_factors=True``).

    Examples
    --------
    >>> from nowcastbox.models.two_step import simulate_two_step_example
    >>> data = simulate_two_step_example(random_state=1)
    >>> data.n_series, data.frequencies["gdp"]
    (11, <Frequency.QUARTERLY: 'Q'>)
    """
    rng = np.random.default_rng(random_state)
    r = int(n_factors)
    burn = 50
    total = n_periods + burn
    a_mat = 0.7 * np.eye(r)
    f = np.zeros((total, r))
    for t in range(1, total):
        f[t] = a_mat @ f[t - 1] + rng.standard_normal(r)
    f = f[burn:]
    lam = rng.uniform(0.5, 1.5, size=(n_series, r)) * rng.choice([-1.0, 1.0], size=(n_series, r))
    x = f @ lam.T + 0.7 * rng.standard_normal((n_periods, n_series))
    edges = list(ragged_edge) if ragged_edge is not None else [0, 1, 2]
    for i in range(n_series):
        k = edges[i % len(edges)]
        if k > 0:
            x[-k:, i] = np.nan
    w = AggregationType.GROWTH_RATE.weights(3) / 9.0
    agg = np.column_stack([rolling_aggregate(f[:, k], w) for k in range(r)]).sum(axis=1)
    y = 0.5 + agg + 0.2 * rng.standard_normal(n_periods)
    index = pd.period_range("2000-01", periods=n_periods, freq="M")
    y_series = pd.Series(y, index=index).where(index.month % 3 == 0)
    last_q_end = np.flatnonzero(index.month % 3 == 0)[-1]
    y_series.iloc[last_q_end] = np.nan
    frame = pd.DataFrame(x, index=index, columns=[f"x{i + 1}" for i in range(n_series)])
    frame["gdp"] = y_series
    freqs = dict.fromkeys(frame.columns[:-1], "M") | {"gdp": "Q"}
    data = MixedFrequencyData(frame, freqs)
    if return_factors:
        true = pd.DataFrame(f, index=index, columns=[f"f{k + 1}" for k in range(r)])
        return data, true
    return data
