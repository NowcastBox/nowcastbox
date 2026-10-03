r"""Large mixed-frequency Bayesian VAR for nowcasting (Cimadomo et al., 2022).

:class:`LargeBVAR` is the "blocking" (B-BVAR) model of Cimadomo, Giannone, Lenza, Monti
& Sokol (2022): every monthly series is split into three quarterly variables, one per
month of the quarter, quarterly series enter as they are, and the blocked vector
:math:`Y_t` (:math:`n = Q + 3M` variables) follows a quarterly VAR(:math:`p`)

.. math::

    Y_t = c + A_1 Y_{t-1} + \dots + A_p Y_{t-p} + \varepsilon_t,\qquad
    \varepsilon_t \sim N(0, \Sigma),

estimated with the Normal-Inverse-Wishart Minnesota prior of Giannone, Lenza &
Primiceri (2015) - optionally with the sum-of-coefficients and dummy-initial-observation
priors - and a hierarchical choice of the tightness (posterior mode of :math:`\lambda`,
and of :math:`\mu`, :math:`\delta`, :math:`\psi` when requested;
:mod:`nowcastbox.models._bvar_prior`).

* **Estimation sample.** The posterior is computed on the longest run of complete
  quarters ending at the last complete one (the balanced part of the blocked panel).
* **Ragged edge.** Backcasts, nowcasts and forecasts are conditional forecasts given
  every entry released at the vintage (Waggoner & Zha, 1999; Bańbura, Giannone & Lenza,
  2015), computed with the Kalman smoother of :mod:`nowcastbox.statespace` on the
  companion form of the VAR at the posterior mean of :math:`(B, \Sigma)`
  (:mod:`nowcastbox.models._bvar_blocking`). The released months of the current quarter
  are partially observed entries of :math:`Y_t`.
* **Densities.** With ``n_draws > 0`` (or ``results.distribution(n_draws=...)``) the
  predictive distribution integrates the parameter uncertainty: for every posterior draw
  :math:`(B^{(i)}, \Sigma^{(i)})` the target is Gaussian given the data (closed-form
  conditional moments), so the distribution is the mixture of these Gaussians
  (Rao-Blackwellised conditional simulation), a
  :class:`~nowcastbox.density.NowcastDistribution` usable for CRPS, PIT and fan charts.
* **News.** With the parameters fixed the model is linear and Gaussian, so
  ``results.news(old, new)`` and ``results.level_contributions()`` reuse
  :mod:`nowcastbox.news` on the blocked quarterly representation and report releases by
  original series and month.

Data are standardised like in the factor models (``standardize=True``); results are in
the original units of the target.

References
----------
Bańbura, M., Giannone, D., & Lenza, M. (2015). Conditional forecasts and scenario
analysis with vector autoregressions for large cross-sections. *International Journal
of Forecasting*, 31(3), 739-756.

Bańbura, M., Giannone, D., & Reichlin, L. (2010). Large Bayesian vector auto
regressions. *Journal of Applied Econometrics*, 25(1), 71-92.

Cimadomo, J., Giannone, D., Lenza, M., Monti, F., & Sokol, A. (2022). Nowcasting with
large Bayesian vector autoregressions. *Journal of Econometrics*, 231(2), 500-519.

Giannone, D., Lenza, M., & Primiceri, G. E. (2015). Prior selection for vector
autoregressions. *Review of Economics and Statistics*, 97(2), 436-451.

Waggoner, D. F., & Zha, T. (1999). Conditional forecasts in dynamic multivariate
models. *Review of Economics and Statistics*, 81(4), 639-651.

Examples
--------
>>> import nowcastbox as nb
>>> from nowcastbox.models import LargeBVAR
>>> sim = nb.simulate.dfm(n_series=6, n_factors=1, n_periods=120, random_state=0)
>>> res = LargeBVAR(lags=1).fit(sim.data, "gdp")
>>> res.model_name, res.n_variables
('LargeBVAR', 19)
>>> bool(res.out_of_sample.notna().any())
True
"""

from __future__ import annotations

import dataclasses
import time
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import scipy.stats

from nowcastbox._logging import get_logger
from nowcastbox.core.base import BaseNowcaster
from nowcastbox.core.data import (
    MixedFrequencyData,
    StandardizationStats,
    as_mixed_frequency_data,
)
from nowcastbox.core.exceptions import DataQualityWarning, NowcastDataError
from nowcastbox.core.frequency import Frequency
from nowcastbox.core.results import NowcastResults, build_nowcast_frame
from nowcastbox.models._bvar_blocking import (
    BlockedLayout,
    VARParameters,
    balanced_run,
    block_values,
    blocked_layout,
    conditional_forecast,
    conditional_moments,
    native_periods,
    unblock_values,
    var_state_space,
    window_end,
)
from nowcastbox.models._bvar_prior import (
    BVARHyperparameters,
    HyperparameterSelection,
    NIWPosterior,
    PriorSettings,
    VARSystem,
    posterior,
    select_hyperparameters,
)

if TYPE_CHECKING:
    from nowcastbox.density import EmpiricalQuantileDistribution, NowcastDistribution
    from nowcastbox.news import LevelContributions, NewsResults, NowcastTracker
    from nowcastbox.news._model import LinearNowcastModel

__all__ = [
    "BlockedBVAR",
    "LargeBVAR",
    "LargeBVARResults",
    "fit_blocked_bvar",
]

logger = get_logger(__name__)

_BLOCKED_RW = "blocked_random_walk"
_PRIOR_MEANS = ("white_noise", "random_walk", _BLOCKED_RW)
_PRIOR_KEYS = frozenset({"lambda", "mu", "delta", "psi"})
_MIN_OBSERVATIONS = 3
_DRAW_CHUNK = 50
_DEFAULT_DRAWS = 500
_SCALE_FLOOR = 1e-12
_DENSITY_METHODS = ("auto", "posterior", "analytic")


# ====================================================================== helpers
def _panel_frame(data: object) -> pd.DataFrame:
    """Wide monthly frame of a panel (MixedFrequencyData or DataFrame)."""
    if isinstance(data, MixedFrequencyData):
        return data.data
    if isinstance(data, pd.DataFrame) and isinstance(data.index, pd.PeriodIndex):
        return data.astype(float)
    return as_mixed_frequency_data(data).data  # type: ignore[arg-type]


def _identity_stats(columns: Sequence[str]) -> StandardizationStats:
    """Statistics that leave the data unchanged (``standardize=False``)."""
    index = pd.Index(list(columns))
    return StandardizationStats(pd.Series(0.0, index=index), pd.Series(1.0, index=index))


def _mean_value(value: object) -> float:
    """Own-lag prior mean of one entry (``"random_walk"`` = 1, ``"white_noise"`` = 0).

    ``"blocked_random_walk"`` also counts as 1 (its own-lag mean for a quarterly series;
    :func:`_prior_settings` moves it to the last month for monthly series).
    """
    if isinstance(value, str):
        if value not in _PRIOR_MEANS:
            raise ValueError(f"prior_mean must be one of {_PRIOR_MEANS} or numeric, got {value!r}.")
        return 0.0 if value == "white_noise" else 1.0
    if isinstance(value, bool) or not isinstance(value, int | float | np.integer | np.floating):
        raise ValueError(f"prior_mean values must be numeric or a name, got {value!r}.")
    if not np.isfinite(float(value)):
        raise ValueError("prior_mean values must be finite.")
    return float(value)


def _is_blocked(value: object) -> bool:
    return isinstance(value, str) and value == _BLOCKED_RW


def _prior_settings(
    names: Sequence[str],
    prior_mean: object,
    lag_decay: float,
    months: Sequence[int] | None = None,
) -> PriorSettings:
    """Minnesota settings of a VAR on ``names`` (mapping prior means expanded per variable).

    ``names`` holds the source series of every VAR variable (repeated for the three
    monthly blocks of a blocked series) and ``months`` the month of the quarter of each
    (default: 3, i.e. every variable is its own last month). ``"blocked_random_walk"``
    (globally or for some series of a mapping) gives the full first-lag mean matrix of
    the blocked random walk and one unit-root group per series.
    """
    if isinstance(prior_mean, Mapping):
        raw = [prior_mean.get(name, 0.0) for name in names]
    else:
        raw = [prior_mean] * len(names)
    if any(_is_blocked(v) for v in raw):
        return _blocked_settings(names, raw, months, lag_decay)
    if isinstance(prior_mean, Mapping):
        means = [_mean_value(v) for v in raw]
        return PriorSettings(prior_mean=means, lag_decay=float(lag_decay))
    if isinstance(prior_mean, str):
        return PriorSettings(prior_mean=prior_mean, lag_decay=float(lag_decay))
    return PriorSettings(prior_mean=_mean_value(prior_mean), lag_decay=float(lag_decay))


def _blocked_settings(
    names: Sequence[str], raw: Sequence[object], months: Sequence[int] | None, lag_decay: float
) -> PriorSettings:
    r"""Blocked random-walk prior: :math:`\operatorname{E}[x^{(m)}_t] = x^{(3)}_{t-1}`.

    Every blocked variable with value ``"blocked_random_walk"`` gets mean 1 on the
    first lag of the last month of its series (itself for quarterly series), the
    others the usual own-lag mean; the blocks of one series form one unit-root group.
    """
    n = len(names)
    month = [3] * n if months is None else [int(m) for m in months]
    last: dict[str, int] = {}
    for col, name in enumerate(names):
        if name not in last or month[col] > month[last[name]]:
            last[name] = col
    A1 = np.zeros((n, n))
    for i, value in enumerate(raw):
        if _is_blocked(value):
            A1[i, last[names[i]]] = 1.0
        else:
            A1[i, i] = _mean_value(value)
    return PriorSettings(prior_mean=A1, lag_decay=float(lag_decay), unit_root_groups=list(names))


def _fixed_hyperparameters(
    prior: Mapping[str, Any], start: BVARHyperparameters
) -> BVARHyperparameters:
    """Hyperparameters from a mapping (missing ``psi``: AR residual variances)."""
    psi = np.broadcast_to(
        np.asarray(prior.get("psi", start.psi), dtype=float), start.psi.shape
    ).copy()
    return BVARHyperparameters(
        lambda_=float(prior["lambda"]),
        psi=psi,
        mu=None if start.mu is None else float(prior.get("mu", start.mu)),
        delta=None if start.delta is None else float(prior.get("delta", start.delta)),
    )


def _resolve_prior(
    system: VARSystem,
    settings: PriorSettings,
    options: _EstimationOptions,
) -> tuple[BVARHyperparameters, HyperparameterSelection | None]:
    """Hyperparameters: hierarchical GLP mode or the fixed values given."""
    prior = options.prior
    if isinstance(prior, BVARHyperparameters):
        if prior.psi.shape != (system.n,):
            raise ValueError(f"prior.psi must have {system.n} values (one per VAR variable).")
        return prior, None
    start = BVARHyperparameters(
        lambda_=0.2,
        psi=system.ar_residual_variances(1),
        mu=1.0 if options.sum_of_coefficients else None,
        delta=1.0 if options.initial_observation else None,
    )
    if isinstance(prior, Mapping):
        return _fixed_hyperparameters(prior, start), None
    estimate = ["lambda"]
    estimate += ["mu"] if options.sum_of_coefficients else []
    estimate += ["delta"] if options.initial_observation else []
    estimate += ["psi"] if options.estimate_psi else []
    sel = select_hyperparameters(
        system,
        settings,
        start=start,
        estimate=estimate,
        hessian=False,
        max_iter=options.max_iter,
    )
    return sel.hyperparameters, sel


@dataclass(frozen=True)
class _EstimationOptions:
    """Hyper-parameters of the estimation (shared by the model and the extrapolator)."""

    lags: int
    prior: Any
    prior_mean: Any
    sum_of_coefficients: bool
    initial_observation: bool
    estimate_psi: bool
    lag_decay: float
    standardize: bool
    max_iter: int


# ====================================================================== fitted engine
@dataclass(frozen=True, eq=False)
class BlockedBVAR:
    r"""Fitted blocked BVAR (target free): data map, posterior and conditional forecasts.

    Built by :func:`fit_blocked_bvar`; used by :class:`LargeBVAR` and by the ``"bvar"``
    indicator extrapolator.

    Parameters
    ----------
    layout : BlockedLayout
        Blocked variables of the panel.
    standardization : StandardizationStats
        Statistics of the original series (identity when not standardised).
    parameters : VARParameters
        Posterior mean of :math:`(c, A_1..A_p)` and of :math:`\Sigma` (standardised
        units).
    posterior : NIWPosterior
        Normal-Inverse-Wishart posterior.
    hyperparameters : BVARHyperparameters
        Hyperparameters used.
    selection : HyperparameterSelection or None
        Hierarchical selection (``prior="glp"``) or None (fixed hyperparameters).
    settings : PriorSettings
        Fixed prior settings (prior means, lag decay).
    quarters : pandas.PeriodIndex
        Quarterly grid of the estimation panel.
    values : numpy.ndarray
        Standardised blocked data on :attr:`quarters`, shape ``(len(quarters), n)``.
    sample : tuple of int
        First and last rows (inclusive) of the balanced estimation sample.

    Examples
    --------
    >>> import nowcastbox as nb
    >>> from nowcastbox.models.bvar import fit_blocked_bvar
    >>> sim = nb.simulate.dfm(n_series=4, n_factors=1, n_periods=96, random_state=0)
    >>> eng = fit_blocked_bvar(sim.data, lags=1)
    >>> eng.layout.n, eng.parameters.lags
    (13, 1)
    """

    layout: BlockedLayout
    standardization: StandardizationStats
    parameters: VARParameters
    posterior: NIWPosterior = field(repr=False)
    hyperparameters: BVARHyperparameters
    selection: HyperparameterSelection | None = field(repr=False)
    settings: PriorSettings
    quarters: pd.PeriodIndex = field(repr=False)
    values: np.ndarray = field(repr=False)
    sample: tuple[int, int]

    @property
    def lags(self) -> int:
        """Quarterly VAR order :math:`p`.

        Returns
        -------
        int
            Number of lags.

        Examples
        --------
        >>> import nowcastbox as nb
        >>> from nowcastbox.models.bvar import fit_blocked_bvar
        >>> sim = nb.simulate.dfm(n_series=3, n_factors=1, n_periods=96, random_state=0)
        >>> eng = fit_blocked_bvar(sim.data, prior={"lambda": 0.2})
        >>> eng.lags
        1
        """
        return self.parameters.lags

    def standardized_blocks(self, data: object) -> tuple[pd.PeriodIndex, np.ndarray]:
        """Standardised blocked matrix of a panel, at least as long as the estimation grid.

        Parameters
        ----------
        data : MixedFrequencyData or pandas.DataFrame
            Monthly panel containing every series of the model.

        Returns
        -------
        quarters : pandas.PeriodIndex
            Quarterly grid (extended with empty quarters up to the end of
            :attr:`quarters` when shorter).
        values : numpy.ndarray
            Blocked standardised values.

        Raises
        ------
        NowcastDataError
            If series are missing or the panel is not monthly.

        Examples
        --------
        >>> import nowcastbox as nb
        >>> from nowcastbox.models.bvar import fit_blocked_bvar
        >>> sim = nb.simulate.dfm(n_series=3, n_factors=1, n_periods=96, random_state=0)
        >>> eng = fit_blocked_bvar(sim.data, prior={"lambda": 0.2})
        >>> q, v = eng.standardized_blocks(sim.data)
        >>> v.shape
        (32, 10)
        """
        frame = _panel_frame(data)
        missing = [s for s in self.layout.series if s not in frame.columns]
        if missing:
            raise NowcastDataError(f"The data do not contain the model series {missing}.")
        std = self.standardization.transform(frame.loc[:, list(self.layout.series)])
        quarters, values = block_values(std, self.layout)
        end = self.quarters[-1]
        if quarters[-1] < end:
            extra = int((end - quarters[-1]).n)
            values = np.vstack([values, np.full((extra, self.layout.n), np.nan)])
            quarters = pd.period_range(quarters[0], end, freq="Q")
        return quarters, values

    def edge(self, values: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
        """Conditional forecast of everything after the last complete window.

        Parameters
        ----------
        values : numpy.ndarray
            Standardised blocked matrix.

        Returns
        -------
        end : int
            Last row of the last window of :attr:`lags` complete quarters.
        mean : numpy.ndarray
            ``values`` with the rows after ``end`` replaced by their conditional
            expectation.
        variance : numpy.ndarray
            Conditional variances (zero up to ``end`` and for released entries).

        Raises
        ------
        NowcastDataError
            If the data have no window of :attr:`lags` complete quarters.

        Examples
        --------
        >>> import nowcastbox as nb
        >>> from nowcastbox.models.bvar import fit_blocked_bvar
        >>> sim = nb.simulate.dfm(n_series=3, n_factors=1, n_periods=96, random_state=0)
        >>> eng = fit_blocked_bvar(sim.data, prior={"lambda": 0.2})
        >>> end, mean, var = eng.edge(eng.values)
        >>> bool(np.isfinite(mean[end + 1 :]).all()), bool((var[: end + 1] == 0).all())
        (True, True)
        """
        end = window_end(values, self.lags)
        mean = values.copy()
        var = np.zeros_like(values)
        if end + 1 < len(values):
            presample = values[end - self.lags + 1 : end + 1]
            fc = conditional_forecast(self.parameters, presample, values[end + 1 :])
            mean[end + 1 :] = fc.mean
            var[end + 1 :] = fc.variance
        return end, mean, var

    def to_original(self, values: np.ndarray, quarters: pd.PeriodIndex) -> pd.DataFrame:
        """Monthly frame in original units from standardised blocked values.

        Parameters
        ----------
        values : numpy.ndarray
            Standardised blocked values.
        quarters : pandas.PeriodIndex
            Their quarters.

        Returns
        -------
        pandas.DataFrame
            Monthly grid; quarterly series in their storage slot.

        Examples
        --------
        >>> import nowcastbox as nb
        >>> from nowcastbox.models.bvar import fit_blocked_bvar
        >>> sim = nb.simulate.dfm(n_series=3, n_factors=1, n_periods=96, random_state=0)
        >>> eng = fit_blocked_bvar(sim.data, prior={"lambda": 0.2})
        >>> eng.to_original(eng.values, eng.quarters).shape
        (96, 4)
        """
        frame = unblock_values(values, quarters, self.layout)
        return self.standardization.inverse_transform(frame)

    def predict(self, data: object) -> pd.DataFrame:
        """``E[x_t | data]`` for every series, in original units (posterior-mean parameters).

        Parameters
        ----------
        data : MixedFrequencyData or pandas.DataFrame
            Vintage containing every model series.

        Returns
        -------
        pandas.DataFrame
            Monthly grid covering the data and the estimation grid: released values
            unchanged, entries after the last window of :attr:`lags` complete quarters
            replaced by their conditional expectation (earlier gaps stay ``NaN``).

        Examples
        --------
        >>> import nowcastbox as nb
        >>> from nowcastbox.models.bvar import fit_blocked_bvar
        >>> sim = nb.simulate.dfm(n_series=3, n_factors=1, n_periods=96, random_state=0)
        >>> eng = fit_blocked_bvar(sim.data, prior={"lambda": 0.2})
        >>> bool(eng.predict(sim.data).iloc[-1].notna().all())
        True
        """
        quarters, values = self.standardized_blocks(data)
        _, mean, _ = self.edge(values)
        return self.to_original(mean, quarters)

    def mixture(
        self,
        values: np.ndarray,
        cells: np.ndarray,
        n_draws: int,
        random_state: int | np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""Conditional means and standard deviations of edge entries under posterior draws.

        Parameters
        ----------
        values : numpy.ndarray
            Standardised blocked matrix.
        cells : numpy.ndarray
            ``(row, column)`` pairs (rows of ``values``) after the last complete window.
        n_draws : int
            Number of draws of :math:`(B, \Sigma)`.
        random_state : int, numpy.random.Generator or None
            Seed.

        Returns
        -------
        locs, scales : numpy.ndarray
            Shape ``(len(cells), n_draws)`` each (standardised units).

        Raises
        ------
        ValueError
            If ``n_draws < 1`` or a cell lies before the edge.

        Examples
        --------
        >>> import nowcastbox as nb
        >>> from nowcastbox.models.bvar import fit_blocked_bvar
        >>> sim = nb.simulate.dfm(n_series=3, n_factors=1, n_periods=96, random_state=0)
        >>> eng = fit_blocked_bvar(sim.data, prior={"lambda": 0.2})
        >>> end = window_end(eng.values, eng.lags)
        >>> locs, scales = eng.mixture(eng.values, [(end + 1, 9)], 20, random_state=0)
        >>> locs.shape, bool((scales > 0).all())
        ((1, 20), True)
        """
        if n_draws < 1:
            raise ValueError(f"n_draws must be >= 1, got {n_draws}.")
        end = window_end(values, self.lags)
        cells = np.asarray(cells, dtype=np.intp).reshape(-1, 2)
        if cells.size and int(cells[:, 0].min()) <= end:
            raise ValueError("Every cell must lie after the last complete window.")
        presample = values[end - self.lags + 1 : end + 1]
        future = values[end + 1 :]
        edge_cells = cells - np.array([end + 1, 0])
        gen = np.random.default_rng(random_state)
        locs = np.empty((len(cells), n_draws))
        scales = np.empty((len(cells), n_draws))
        done = 0
        while done < n_draws:
            size = min(_DRAW_CHUNK, n_draws - done)
            B, S = self.posterior.draw(size, gen)
            for i in range(size):
                par = VARParameters.from_stacked(B[i], S[i], self.lags)
                mean, var = conditional_moments(par, presample, future, edge_cells)
                locs[:, done + i] = mean
                scales[:, done + i] = np.sqrt(np.maximum(var, _SCALE_FLOOR))
            done += size
        return locs, scales


def _validate_options(options: _EstimationOptions) -> None:
    """Raise ``ValueError`` on invalid estimation options."""
    _check_int(options.lags, "lags", 1)
    _check_int(options.max_iter, "max_iter", 1)
    _check_prior(options.prior)
    _check_prior_mean(options.prior_mean)
    for name in ("sum_of_coefficients", "initial_observation", "estimate_psi", "standardize"):
        if not isinstance(getattr(options, name), bool):
            raise ValueError(f"{name} must be a bool.")
    decay = options.lag_decay
    if isinstance(decay, bool) or not isinstance(decay, int | float) or not decay > 0:
        raise ValueError(f"lag_decay must be a positive number, got {decay!r}.")


def _check_int(value: object, name: str, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int | np.integer) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}.")


def _check_prior(prior: object) -> None:
    if isinstance(prior, BVARHyperparameters) or prior == "glp":
        return
    if not isinstance(prior, Mapping):
        raise ValueError(
            "prior must be 'glp', a mapping of fixed hyperparameters or BVARHyperparameters, "
            f"got {prior!r}."
        )
    unknown = set(prior) - _PRIOR_KEYS
    if unknown or "lambda" not in prior:
        raise ValueError(
            f"A fixed prior needs 'lambda' and may give {sorted(_PRIOR_KEYS)}; got {sorted(prior)}."
        )


def _check_prior_mean(prior_mean: object) -> None:
    if isinstance(prior_mean, Mapping):
        for value in prior_mean.values():
            _mean_value(value)
    else:
        _mean_value(prior_mean)


def fit_blocked_bvar(
    data: MixedFrequencyData | pd.DataFrame,
    *,
    lags: int = 1,
    prior: Any = "glp",
    prior_mean: Any = "white_noise",
    sum_of_coefficients: bool = False,
    initial_observation: bool = False,
    estimate_psi: bool = False,
    lag_decay: float = 2.0,
    standardize: bool = True,
    max_iter: int = 500,
) -> BlockedBVAR:
    """Estimate the blocked BVAR of a monthly/quarterly panel (no target needed).

    Parameters
    ----------
    data : MixedFrequencyData or pandas.DataFrame
        Panel on a monthly base grid with monthly and quarterly series.
    lags, prior, prior_mean, sum_of_coefficients, initial_observation, estimate_psi, \
lag_decay, standardize, max_iter
        See :class:`LargeBVAR`.

    Returns
    -------
    BlockedBVAR
        Fitted engine.

    Raises
    ------
    ValueError
        On invalid options.
    NowcastDataError
        If the panel is not monthly/quarterly or has fewer than ``lags + 3`` complete
        consecutive quarters.

    Examples
    --------
    >>> import nowcastbox as nb
    >>> sim = nb.simulate.dfm(n_series=3, n_factors=1, n_periods=96, random_state=0)
    >>> eng = fit_blocked_bvar(sim.data, lags=1, prior={"lambda": 0.2})
    >>> eng.hyperparameters.lambda_
    0.2
    """
    options = _EstimationOptions(
        lags=lags,
        prior=prior,
        prior_mean=prior_mean,
        sum_of_coefficients=sum_of_coefficients,
        initial_observation=initial_observation,
        estimate_psi=estimate_psi,
        lag_decay=lag_decay,
        standardize=standardize,
        max_iter=max_iter,
    )
    _validate_options(options)
    panel = as_mixed_frequency_data(data)
    return _fit_engine(panel, options)


def _warn_discarded(
    values: np.ndarray, periods: pd.PeriodIndex, names: Sequence[str], start: int
) -> None:
    """Warn when periods with data before the balanced sample are left out of it."""
    discarded = int(np.isfinite(values[:start]).any(axis=1).sum())
    if discarded <= 1:  # at most a partial first period
        return
    gaps = sorted({names[int(c)] for c in np.flatnonzero(np.isnan(values[start - 1]))})
    shown = ", ".join(gaps[:5]) + (f" and {len(gaps) - 5} more" if len(gaps) > 5 else "")
    unit = "quarters" if periods.freqstr.startswith("Q") else "periods"
    warnings.warn(
        f"The BVAR is estimated on {periods[start]} onwards: {discarded} earlier {unit} "
        f"with data are left out because of missing values in {periods[start - 1]} "
        f"({shown}).",
        DataQualityWarning,
        stacklevel=5,
    )


@dataclass(frozen=True)
class _Estimate:
    """Posterior of a VAR estimated on the balanced part of a standardised matrix."""

    parameters: VARParameters
    posterior: NIWPosterior
    hyperparameters: BVARHyperparameters
    selection: HyperparameterSelection | None
    settings: PriorSettings
    sample: tuple[int, int]


def _estimate_var(
    values: np.ndarray,
    periods: pd.PeriodIndex,
    names: Sequence[str],
    options: _EstimationOptions,
    months: Sequence[int] | None = None,
) -> _Estimate:
    """BVAR posterior on the longest balanced run of ``values`` (rows = ``periods``).

    Shared by the blocked (quarterly) and the monthly engines; ``names`` gives the source
    series of every column (prior means by series, warnings) and ``months`` the month of
    the quarter of every blocked column (blocked random-walk prior; ``None`` for the
    monthly engine, where every column is its own last month).
    """
    start, end = balanced_run(values)
    if end - start + 1 < options.lags + _MIN_OBSERVATIONS:
        raise NowcastDataError(
            f"The BVAR needs at least {options.lags + _MIN_OBSERVATIONS} consecutive "
            f"complete periods; the balanced sample has {end - start + 1} "
            f"({periods[start]}..{periods[end]})."
        )
    _warn_discarded(values, periods, names, start)
    system = VARSystem.from_array(values[start : end + 1], options.lags)
    settings = _prior_settings(names, options.prior_mean, options.lag_decay, months)
    try:
        hyper, selection = _resolve_prior(system, settings, options)
        post = posterior(system, hyper, settings)
    except np.linalg.LinAlgError as err:
        raise NowcastDataError(f"The BVAR posterior could not be computed: {err}") from err
    params = VARParameters.from_stacked(post.mean, post.sigma_mean, options.lags)
    logger.debug(
        "BVAR: n=%d, p=%d, sample %s..%s, lambda=%.4g",
        len(names),
        options.lags,
        periods[start],
        periods[end],
        hyper.lambda_,
    )
    return _Estimate(params, post, hyper, selection, settings, (start, end))


def _fit_engine(panel: MixedFrequencyData, options: _EstimationOptions) -> BlockedBVAR:
    layout = blocked_layout(panel)
    if options.standardize:
        stats = panel.standardization_stats()
    else:
        stats = _identity_stats(layout.series)
    std = stats.transform(panel.data.loc[:, list(layout.series)])
    quarters, values = block_values(std, layout)
    names = [layout.source_name(c) for c in range(layout.n)]
    est = _estimate_var(values, quarters, names, options, layout.month)
    return BlockedBVAR(
        layout=layout,
        standardization=stats,
        parameters=est.parameters,
        posterior=est.posterior,
        hyperparameters=est.hyperparameters,
        selection=est.selection,
        settings=est.settings,
        quarters=quarters,
        values=values,
        sample=est.sample,
    )


# ====================================================================== results
@dataclass(frozen=True, kw_only=True, eq=False, repr=False)
class LargeBVARResults(NowcastResults):
    r"""Results of :class:`LargeBVAR`.

    The ``nowcast`` frame holds, on the target's native periods, ``observed``,
    ``in_sample`` (one-quarter-ahead VAR prediction where the target is observed),
    ``out_of_sample`` (conditional expectation given the released data: backcasts,
    nowcast, forecasts), ``std`` (predictive standard deviation: Kalman variance at the
    posterior mean or, with posterior draws, the standard deviation of the mixture) and
    the bounds ``lower_68``/``upper_68``/``lower_90``/``upper_90``.

    Parameters
    ----------
    bvar : BlockedBVAR, optional
        Fitted engine (layout, posterior, standardised blocked data).
    smoothed_data : pandas.DataFrame, optional
        ``E[x_t | data]`` for every series on the monthly grid (original units).
    window_end : pandas.Period, optional
        Last quarter of the last window of ``lags`` complete quarters: later quarters
        are conditional forecasts.
    draw_locs, draw_scales : pandas.DataFrame, optional
        Posterior mixture of the out-of-sample target periods (rows: periods, columns:
        draws; original units), when ``n_draws > 0``.

    Notes
    -----
    Every field of :class:`~nowcastbox.core.results.NowcastResults` is also accepted
    (keyword-only).

    Examples
    --------
    >>> import nowcastbox as nb
    >>> sim = nb.simulate.dfm(n_series=3, n_factors=1, n_periods=96, random_state=0)
    >>> res = LargeBVAR().fit(sim.data, "gdp")
    >>> res.coefficients().shape
    (11, 10)
    """

    bvar: BlockedBVAR | None = None
    smoothed_data: pd.DataFrame | None = None
    window_end: pd.Period | None = None
    draw_locs: pd.DataFrame | None = None
    draw_scales: pd.DataFrame | None = None

    # ------------------------------------------------------------------ access
    def _engine(self) -> BlockedBVAR:
        if self.bvar is None:
            raise ValueError("These results do not contain the fitted BVAR.")
        return self.bvar

    @property
    def n_variables(self) -> int:
        """Number of blocked variables :math:`n = Q + 3M`.

        Returns
        -------
        int
            Size of :math:`Y_t`.

        Examples
        --------
        >>> import nowcastbox as nb
        >>> from nowcastbox.models import LargeBVAR
        >>> sim = nb.simulate.dfm(n_series=3, n_factors=1, n_periods=96, random_state=0)
        >>> res = LargeBVAR().fit(sim.data, "gdp")
        >>> res.n_variables
        10
        """
        return self._engine().layout.n

    def coefficients(self) -> pd.DataFrame:
        r"""Posterior mean of the stacked coefficients :math:`\bar B` (standardised units).

        Returns
        -------
        pandas.DataFrame
            Rows ``"const"`` and ``"<variable> (lag l)"``; columns = blocked variables
            (equations).

        Examples
        --------
        >>> import nowcastbox as nb
        >>> from nowcastbox.models import LargeBVAR
        >>> sim = nb.simulate.dfm(n_series=3, n_factors=1, n_periods=96, random_state=0)
        >>> res = LargeBVAR().fit(sim.data, "gdp")
        >>> res.coefficients().loc["const"].shape
        (10,)
        """
        eng = self._engine()
        cols = list(eng.layout.columns)
        rows = ["const"] + [f"{c} (lag {lag})" for lag in range(1, eng.lags + 1) for c in cols]
        return pd.DataFrame(eng.posterior.mean, index=rows, columns=cols)

    def blocked_data(self, data: object = None) -> pd.DataFrame:
        """Blocked quarterly panel in original units.

        Parameters
        ----------
        data : MixedFrequencyData or pandas.DataFrame, optional
            Monthly panel (default: the estimation data).

        Returns
        -------
        pandas.DataFrame
            Quarterly :class:`pandas.PeriodIndex`, one column per blocked variable
            (``"ip[m1]"``, ``"ip[m2]"``, ``"ip[m3]"``, ``"gdp"``).

        Raises
        ------
        ValueError
            If no data are given and none are stored.

        Examples
        --------
        >>> import nowcastbox as nb
        >>> from nowcastbox.models import LargeBVAR
        >>> sim = nb.simulate.dfm(n_series=3, n_factors=1, n_periods=96, random_state=0)
        >>> res = LargeBVAR().fit(sim.data, "gdp")
        >>> res.blocked_data().columns[:3].tolist()
        ['x01[m1]', 'x01[m2]', 'x01[m3]']
        """
        eng = self._engine()
        panel = self.data if data is None else data
        if panel is None:
            raise ValueError("No data given and none stored in the results.")
        frame = _panel_frame(panel)
        quarters, values = block_values(frame, eng.layout)
        out = pd.DataFrame(values, index=quarters, columns=list(eng.layout.columns))
        out.index.name = "period"
        return out

    def predict(self, data: object = None) -> pd.DataFrame:
        """``E[x_t | data]`` for every series (posterior-mean parameters), original units.

        Parameters
        ----------
        data : MixedFrequencyData or pandas.DataFrame, optional
            Vintage containing every model series (default: the estimation data).

        Returns
        -------
        pandas.DataFrame
            Monthly grid (quarterly series meaningful in their storage slot); see
            :meth:`BlockedBVAR.predict`.

        Raises
        ------
        ValueError
            If no data are given and none are stored.

        Examples
        --------
        >>> import nowcastbox as nb
        >>> from nowcastbox.models import LargeBVAR
        >>> sim = nb.simulate.dfm(n_series=3, n_factors=1, n_periods=96, random_state=0)
        >>> res = LargeBVAR().fit(sim.data, "gdp")
        >>> old = sim.data.truncate(end="2007-10")
        >>> bool(abs(res.predict(old)["gdp"].iloc[-1]) < 10)
        True
        """
        panel = self.data if data is None else data
        if panel is None:
            raise ValueError("No data given and none stored in the results.")
        return self._engine().predict(panel)

    # ------------------------------------------------------------------ densities
    def _target_cells(self) -> tuple[pd.PeriodIndex, np.ndarray]:
        """Out-of-sample target periods and their ``(row, column)`` cells."""
        eng = self._engine()
        end = window_end(eng.values, eng.lags)
        periods, cells = [], []
        for c in eng.layout.columns_of(self.target):
            native = native_periods(eng.quarters, eng.layout, c)
            for r in range(end + 1, len(eng.quarters)):
                if not np.isfinite(eng.values[r, c]):
                    periods.append(native[r])
                    cells.append((r, c))
        order = np.argsort([p.ordinal for p in periods], kind="stable")
        index = pd.PeriodIndex([periods[i] for i in order], freq=self.target_frequency.pandas_freq)
        return index, np.asarray([cells[i] for i in order], dtype=np.intp).reshape(-1, 2)

    def posterior_mixture(
        self, n_draws: int = _DEFAULT_DRAWS, random_state: int | np.random.Generator | None = None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        r"""Posterior mixture components of the out-of-sample target periods.

        For each draw :math:`(B^{(i)}, \Sigma^{(i)})` of the Normal-Inverse-Wishart
        posterior, the conditional mean and standard deviation of the target given the
        released data (closed form; equal to the Kalman smoother).

        Parameters
        ----------
        n_draws : int, default 500
            Number of posterior draws.
        random_state : int, numpy.random.Generator or None
            Seed.

        Returns
        -------
        locs, scales : pandas.DataFrame
            Rows: out-of-sample target periods; columns: draws (original units).

        Raises
        ------
        ValueError
            If ``n_draws < 1``.
        NowcastDataError
            If no target period is out of sample.

        Examples
        --------
        >>> import nowcastbox as nb
        >>> from nowcastbox.models import LargeBVAR
        >>> sim = nb.simulate.dfm(n_series=3, n_factors=1, n_periods=96, random_state=0)
        >>> res = LargeBVAR().fit(sim.data, "gdp")
        >>> locs, scales = res.posterior_mixture(20, random_state=0)
        >>> locs.shape
        (1, 20)
        """
        _check_int(n_draws, "n_draws", 1)
        eng = self._engine()
        index, cells = self._target_cells()
        if len(index) == 0:
            raise NowcastDataError("No out-of-sample target period.")
        locs, scales = eng.mixture(eng.values, cells, int(n_draws), random_state)
        stats = eng.standardization
        locs_o = stats.inverse_series(locs, self.target)
        scales_o = stats.inverse_series(scales, self.target, scale_only=True)
        return pd.DataFrame(locs_o, index=index), pd.DataFrame(scales_o, index=index)

    def distribution(self, **kwargs: Any) -> NowcastDistribution | EmpiricalQuantileDistribution:
        """Predictive distribution of the nowcast.

        Parameters
        ----------
        **kwargs
            ``method``: ``"auto"`` (default: the posterior mixture when draws are
            stored or ``n_draws`` is given, else the Gaussian of the ``std`` column),
            ``"posterior"`` (posterior mixture, computed with ``n_draws`` draws -
            default 500 - when none are stored), ``"analytic"`` (Gaussian of the
            ``std`` column) or ``"empirical"`` (empirical error bands, see
            :meth:`NowcastResults.distribution`); ``n_draws``, ``random_state`` and
            ``periods``.

        Returns
        -------
        NowcastDistribution or EmpiricalQuantileDistribution
            Mixture with one Gaussian component per posterior draw (CRPS, PIT, fan
            charts), or the other distributions.

        Raises
        ------
        ValueError
            On an unknown method.
        TypeError
            On unknown options.

        Examples
        --------
        >>> import nowcastbox as nb
        >>> from nowcastbox.models import LargeBVAR
        >>> sim = nb.simulate.dfm(n_series=3, n_factors=1, n_periods=96, random_state=0)
        >>> res = LargeBVAR().fit(sim.data, "gdp")
        >>> res.distribution(n_draws=20, random_state=0).n_components
        20
        """
        method = kwargs.get("method", "auto")
        if method == "empirical" or kwargs.get("n_boot"):
            return super().distribution(**kwargs)
        options = dict(kwargs)
        options.pop("method", None)
        n_draws = options.pop("n_draws", None)
        random_state = options.pop("random_state", None)
        periods = options.pop("periods", None)
        if options:
            raise TypeError(f"Unexpected options {sorted(options)} for distribution().")
        if method not in _DENSITY_METHODS:
            raise ValueError(f"method must be one of {(*_DENSITY_METHODS, 'empirical')}.")
        use_draws = method == "posterior" or (
            method == "auto" and (n_draws is not None or self.draw_locs is not None)
        )
        if not use_draws:
            from nowcastbox.density import analytic_distribution

            return analytic_distribution(self, periods)
        return self._mixture_distribution(n_draws, random_state, periods)

    def _mixture_distribution(
        self, n_draws: int | None, random_state: Any, periods: Any
    ) -> NowcastDistribution:
        from nowcastbox.density import NowcastDistribution

        if n_draws is None and self.draw_locs is not None and self.draw_scales is not None:
            locs, scales = self.draw_locs, self.draw_scales
        else:
            n = _DEFAULT_DRAWS if n_draws is None else n_draws
            locs, scales = self.posterior_mixture(n, random_state)
        if periods is not None:
            index = _period_index(periods, self.target_frequency)
            bad = [str(p) for p in index if p not in locs.index]
            if bad:
                raise NowcastDataError(f"No out-of-sample estimate for the periods {bad}.")
            locs, scales = locs.loc[index], scales.loc[index]
        point = self.estimate.reindex(locs.index).to_numpy(dtype=float)
        return NowcastDistribution(
            pd.PeriodIndex(locs.index),
            locs.to_numpy(),
            scales.to_numpy(),
            target=self.target,
            point=point,
            info={"source": "posterior", "model": self.model_name, "n_draws": locs.shape[1]},
        )

    # ------------------------------------------------------------------ news
    def linear_nowcast_model(self) -> LinearNowcastModel:
        """Linear-Gaussian representation at the posterior mean, on the blocked quarterly grid.

        Used by :mod:`nowcastbox.news` (through :meth:`news` and
        :meth:`level_contributions`, which pass blocked quarterly vintages). The grid
        starts ``lags`` quarters after the start of the balanced estimation sample; the
        initial state is set from the estimation data before it.

        Returns
        -------
        LinearNowcastModel
            Model with the blocked variables as series.

        Raises
        ------
        NotImplementedError
            If the target is not quarterly.

        Examples
        --------
        >>> import nowcastbox as nb
        >>> from nowcastbox.models import LargeBVAR
        >>> sim = nb.simulate.dfm(n_series=3, n_factors=1, n_periods=96, random_state=0)
        >>> res = LargeBVAR().fit(sim.data, "gdp")
        >>> res.linear_nowcast_model().series[:2]
        ('x01[m1]', 'x01[m2]')
        """
        from nowcastbox.news._model import LinearNowcastModel

        self._require_quarterly_target()
        eng = self._engine()
        layout, stats, p = eng.layout, eng.standardization, eng.lags
        start = eng.sample[0]
        model = var_state_space(eng.parameters, eng.values[start : start + p])
        names = [layout.source_name(c) for c in range(layout.n)]
        mean = stats.mean.reindex(names).to_numpy(dtype=np.float64)
        std = stats.std.reindex(names).to_numpy(dtype=np.float64)
        j = layout.columns.index(self.target)
        gain = np.zeros(model.n_states)
        gain[j] = 1.0
        categories, blocks = self._blocked_labels()
        return LinearNowcastModel(
            model_name=self.model_name,
            target=self.target,
            target_frequency=Frequency.QUARTERLY,
            base_frequency=Frequency.QUARTERLY,
            series=layout.columns,
            mean=mean,
            std=std,
            state_space=model,
            grid_start=eng.quarters[start + p],
            min_grid_end=eng.quarters[-1],
            gain=gain,
            intercept=0.0,
            offset=float(mean[j]),
            scale=float(std[j]),
            min_lag=0,
            method="univariate",
            categories=categories,
            blocks=blocks,
            frequencies=dict.fromkeys(layout.columns, Frequency.QUARTERLY),
        )

    def _require_quarterly_target(self) -> None:
        if self.target_frequency is not Frequency.QUARTERLY:
            raise NotImplementedError(
                "News and level contributions of LargeBVAR need a quarterly target."
            )

    def _blocked_labels(self) -> tuple[dict[str, str], dict[str, str]]:
        """Category and block label of every blocked variable (from its series)."""
        from nowcastbox.news._model import UNCATEGORIZED

        layout = self._engine().layout
        categories: dict[str, str] = {}
        blocks: dict[str, str] = {}
        meta = {} if self.data is None else self.data.metadata
        for c, name in enumerate(layout.columns):
            info = meta.get(layout.source_name(c))
            category = None if info is None else info.category
            categories[name] = (
                UNCATEGORIZED if category is None else str(getattr(category, "value", category))
            )
            blocks[name] = "+".join(info.blocks) if info is not None and info.blocks else "all"
        return categories, blocks

    def _blocked_categories(self, categories: Mapping[str, object] | None) -> dict[str, object]:
        """User category overrides by series, expanded to the blocked variables."""
        if not categories:
            return {}
        layout = self._engine().layout
        return {
            layout.columns[c]: categories[layout.source_name(c)]
            for c in range(layout.n)
            if layout.source_name(c) in categories
        }

    def news(self, old: Any, new: Any, target_period: Any = None, **kwargs: Any) -> NewsResults:
        """News decomposition of the nowcast revision between two vintages.

        Delegates to :func:`nowcastbox.news.news_decomposition` on the blocked quarterly
        representation (parameters fixed at the posterior mean) and maps the releases
        back to the original series and months.

        Parameters
        ----------
        old, new : MixedFrequencyData or pandas.DataFrame
            Old and new monthly vintages.
        target_period : period-like, optional
            Target quarter (default: the first quarter after the last target
            observation in ``old``).
        **kwargs
            ``new_results`` (re-estimated :class:`LargeBVARResults`) and ``categories``
            (``{series: category}``).

        Returns
        -------
        NewsResults
            Releases by series and month (``slot`` = month of release on the monthly
            grid); ``info["blocked_series"]`` lists the blocked variables.

        Raises
        ------
        NotImplementedError
            If the target is not quarterly.
        TypeError
            On unknown options.

        Examples
        --------
        >>> import nowcastbox as nb
        >>> from nowcastbox.models import LargeBVAR
        >>> sim = nb.simulate.dfm(n_series=3, n_factors=1, n_periods=96, random_state=0)
        >>> res = LargeBVAR().fit(sim.data, "gdp")
        >>> old = sim.data.truncate(end="2007-11")
        >>> res.news(old, sim.data).check_identity()
        True
        """
        from nowcastbox.news import news_decomposition

        self._require_quarterly_target()
        new_results = kwargs.pop("new_results", None)
        categories = kwargs.pop("categories", None)
        if kwargs:
            raise TypeError(f"Unexpected options {sorted(kwargs)} for news().")
        raw = news_decomposition(
            self,
            self.blocked_data(old),
            self.blocked_data(new),
            target_period,
            new_results=new_results,
            categories=self._blocked_categories(categories),
        )
        return _unblock_news(raw, self._engine().layout)

    def level_contributions(
        self, data: Any = None, target_period: Any = None, **kwargs: Any
    ) -> LevelContributions:
        """Contribution of every series to the level of the nowcast.

        Delegates to :func:`nowcastbox.news.level_contributions` on the blocked
        representation and sums the contributions of the months of each series. The
        baseline includes the effect of the estimation data before the grid of
        :meth:`linear_nowcast_model` (initial state).

        Parameters
        ----------
        data : MixedFrequencyData or pandas.DataFrame, optional
            Monthly vintage (default: the estimation data).
        target_period : period-like, optional
            Target quarter.
        **kwargs
            ``categories`` override (``{series: category}``).

        Returns
        -------
        LevelContributions
            Baseline plus one contribution per original series.

        Raises
        ------
        NotImplementedError
            If the target is not quarterly.

        Examples
        --------
        >>> import nowcastbox as nb
        >>> from nowcastbox.models import LargeBVAR
        >>> sim = nb.simulate.dfm(n_series=3, n_factors=1, n_periods=96, random_state=0)
        >>> res = LargeBVAR().fit(sim.data, "gdp")
        >>> res.level_contributions().check_identity()
        True
        """
        from nowcastbox.news import level_contributions

        self._require_quarterly_target()
        categories = self._blocked_categories(kwargs.pop("categories", None))
        lc = level_contributions(
            self, self.blocked_data(data), target_period, categories=categories, **kwargs
        )
        layout = self._engine().layout
        frame = lc.contributions.copy()
        frame["source"] = [layout.source_name(layout.columns.index(s)) for s in frame.index]
        grouped = frame.groupby("source", sort=False).agg(
            contribution=("contribution", "sum"),
            n_obs=("n_obs", "sum"),
            category=("category", "first"),
            block=("block", "first"),
        )
        grouped.index.name = "series"
        return dataclasses.replace(lc, contributions=grouped)

    def nowcast_tracker(
        self,
        data: Any,
        calendar: Any = None,
        target_period: Any = None,
        start: Any = None,
        end: Any = None,
        **kwargs: Any,
    ) -> NowcastTracker:
        """Not available for :class:`LargeBVAR` (use :meth:`news` between vintages).

        Parameters
        ----------
        data, calendar, target_period, start, end, **kwargs
            Ignored.

        Returns
        -------
        NowcastTracker
            Never returns.

        Raises
        ------
        NotImplementedError
            Always.

        Examples
        --------
        >>> import nowcastbox as nb
        >>> from nowcastbox.models import LargeBVAR
        >>> sim = nb.simulate.dfm(n_series=3, n_factors=1, n_periods=96, random_state=0)
        >>> res = LargeBVAR().fit(sim.data, "gdp")
        >>> res.nowcast_tracker(sim.data)
        Traceback (most recent call last):
        NotImplementedError: ...
        """
        raise NotImplementedError(
            "nowcast_tracker is not available for LargeBVAR; compute the news between "
            "consecutive vintages with results.news(old, new)."
        )

    # ------------------------------------------------------------------ summary
    def _summary_sections(self) -> list[tuple[str, list[str]]]:
        sections = super()._summary_sections()
        if self.bvar is None:
            return sections
        eng = self.bvar
        hyper = eng.hyperparameters
        start, end = eng.sample
        lines = [
            f"  {'Blocked variables':<22}{eng.layout.n}",
            f"  {'Quarterly lags':<22}{eng.lags}",
            f"  {'Estimation sample':<22}{eng.quarters[start]}..{eng.quarters[end]}",
            f"  {'Prior':<22}{'GLP hierarchical' if eng.selection else 'fixed'}",
            f"  {'lambda':<22}{hyper.lambda_:.4g}",
        ]
        if hyper.mu is not None:
            lines.append(f"  {'mu (SoC)':<22}{hyper.mu:.4g}")
        if hyper.delta is not None:
            lines.append(f"  {'delta (DIO)':<22}{hyper.delta:.4g}")
        if self.window_end is not None:
            lines.append(f"  {'Conditioning from':<22}{self.window_end + 1}")
        sections.append(("Bayesian VAR", lines))
        return sections


def _period_index(periods: Any, freq: Frequency) -> pd.PeriodIndex:
    items = [periods] if isinstance(periods, str | pd.Period) else list(periods)
    return pd.PeriodIndex([pd.Period(p, freq=freq.pandas_freq) for p in items])


def _blocked_reference(layout: BlockedLayout, column: str, quarter: pd.Period) -> pd.Period:
    """Native period of the observation of a blocked variable in a quarter."""
    c = layout.columns.index(column)
    q = pd.PeriodIndex([quarter.asfreq("Q")], freq="Q")
    return native_periods(q, layout, c)[0]


def _unblock_news(raw: NewsResults, layout: BlockedLayout) -> NewsResults:
    """Map a news decomposition on blocked variables back to series and months."""
    releases = raw.releases.copy()
    if len(releases):
        refs = [
            _blocked_reference(layout, s, q)
            for s, q in zip(releases["series"], releases["slot"], strict=True)
        ]
        releases["reference_period"] = refs
        releases["slot"] = [p.asfreq("M", how="E") for p in refs]
        releases["series"] = [
            layout.source_name(layout.columns.index(s)) for s in releases["series"]
        ]
    revisions = raw.revisions.copy()
    if len(revisions):
        revisions["source"] = [layout.source_name(layout.columns.index(s)) for s in revisions.index]
        revisions = revisions.groupby("source", sort=False).agg(
            n_revised=("n_revised", "sum"),
            impact=("impact", "sum"),
            category=("category", "first"),
            block=("block", "first"),
        )
        revisions.index.name = "series"
    info = dict(raw.info)
    info["blocked_series"] = list(info.get("series", layout.columns))
    info["series"] = list(layout.series)
    labels = dict(zip(info["blocked_series"], info.get("block_labels", []), strict=False))
    info["block_labels"] = [
        labels.get(layout.columns[layout.columns_of(s)[0]], "all") for s in layout.series
    ]
    return dataclasses.replace(raw, releases=releases, revisions=revisions, info=info)


# ====================================================================== estimator
class LargeBVAR(BaseNowcaster):
    r"""Large mixed-frequency Bayesian VAR with blocking (Cimadomo et al., 2022).

    Parameters
    ----------
    lags : int, default 1
        Order :math:`p` of the quarterly VAR of the blocked vector (Cimadomo et al. use
        5 with data in log-levels; 1-2 suit growth rates).
    prior : "glp", mapping or BVARHyperparameters, default "glp"
        ``"glp"``: posterior mode of the hyperparameters under the hyperpriors of
        Giannone, Lenza & Primiceri (2015) (:math:`\lambda`; :math:`\mu`/:math:`\delta`
        when the corresponding prior is on; :math:`\psi` with ``estimate_psi``).
        A mapping fixes them: ``{"lambda": 0.2, "mu": 1.0, "delta": 1.0, "psi": ...}``
        (``lambda`` required; ``psi`` defaults to AR(1) residual variances; ``mu`` and
        ``delta`` are used only when their prior is switched on, default 1). A
        :class:`~nowcastbox.models._bvar_prior.BVARHyperparameters` is used as given
        (its ``mu``/``delta`` decide which dummy priors are on).
    prior_mean : {"white_noise", "random_walk", "blocked_random_walk"}, float or mapping, \
default "white_noise"
        Prior mean of the first-lag coefficients: ``"white_noise"`` (0) for stationary
        data (growth rates, the usual nowcastbox panels); ``"random_walk"`` (1 on the own
        first lag of every blocked variable, as in Cimadomo et al., 2022, eq. 2: each
        month is centred on the *same month* of the previous quarter);
        ``"blocked_random_walk"`` for (log-)levels: every month of a monthly series is
        centred on the *last month* of the previous quarter,
        :math:`\operatorname{E}[x^{(m)}_t] = x^{(3)}_{t-1}` - the blocked form of a
        monthly random walk - with one shared unit root per series in the
        sum-of-coefficients and dummy-initial-observation priors (quarterly series:
        same as ``"random_walk"``); a number; or a mapping ``{series: value}`` (missing
        series: 0; values may be any of the names, e.g. ``"white_noise"`` for a
        stationary survey in a log-level panel).
    sum_of_coefficients : bool, default False
        Add the sum-of-coefficients prior (Doan, Litterman & Sims, 1984) - for data in
        levels.
    initial_observation : bool, default False
        Add the dummy-initial-observation prior (Sims, 1993) - for data in levels.
    estimate_psi : bool, default False
        Also estimate the prior scales :math:`\psi` (``prior="glp"``); by default they
        are the AR(1) residual variances, as in Cimadomo et al. (2022).
    lag_decay : float, default 2.0
        Exponent of the lag decay of the Minnesota prior variances.
    standardize : bool, default True
        Standardise the series before blocking (statistics of the estimation panel).
    n_draws : int, default 0
        Posterior draws stored for the density nowcast (mixture over draws). ``0``:
        the ``std`` column is the Kalman standard deviation at the posterior mean;
        draws can still be computed later with ``results.distribution(n_draws=...)``.
    random_state : int, numpy.random.Generator or None, default None
        Seed of the posterior draws.
    max_iter : int, default 500
        Maximum iterations of the hyperparameter optimisation.

    Notes
    -----
    ``fit`` accepts ``horizon`` (int, default 0): additional target periods to
    forecast. The panel must be on a monthly grid with monthly and quarterly series.
    The posterior is computed on the balanced part of the blocked panel (the longest
    run of complete quarters ending at the last complete one); the ragged edge after
    the last window of ``lags`` complete quarters is handled by the conditional
    forecast. Series starting late (or interior gaps) therefore shorten the estimation
    sample; a :class:`~nowcastbox.core.exceptions.DataQualityWarning` names the series
    responsible when quarters with data are left out.

    Examples
    --------
    >>> import nowcastbox as nb
    >>> sim = nb.simulate.dfm(n_series=5, n_factors=1, n_periods=120, random_state=2)
    >>> res = LargeBVAR(lags=1, n_draws=50, random_state=0).fit(sim.data, "gdp")
    >>> dist = res.distribution()
    >>> dist.n_components
    50
    >>> bool((res.nowcast["std"].dropna() > 0).all())
    True
    """

    def __init__(
        self,
        lags: int = 1,
        prior: Any = "glp",
        prior_mean: Any = "white_noise",
        sum_of_coefficients: bool = False,
        initial_observation: bool = False,
        estimate_psi: bool = False,
        lag_decay: float = 2.0,
        standardize: bool = True,
        n_draws: int = 0,
        random_state: int | np.random.Generator | None = None,
        max_iter: int = 500,
    ) -> None:
        self.lags = lags
        self.prior = prior
        self.prior_mean = prior_mean
        self.sum_of_coefficients = sum_of_coefficients
        self.initial_observation = initial_observation
        self.estimate_psi = estimate_psi
        self.lag_decay = lag_decay
        self.standardize = standardize
        self.n_draws = n_draws
        self.random_state = random_state
        self.max_iter = max_iter

    def fit(
        self,
        data: MixedFrequencyData | pd.DataFrame,
        target: str,
        *,
        frequency: Any = None,
        **fit_kwargs: Any,
    ) -> LargeBVARResults:
        """Estimate the blocked BVAR and nowcast ``target``.

        Parameters
        ----------
        data : MixedFrequencyData or pandas.DataFrame
            Monthly panel with monthly and quarterly series.
        target : str
            Target series (quarterly or monthly) or a formula (``"gdp ~ ip + pmi"``).
        frequency : frequency specification, optional
            Per-series frequencies when ``data`` is a DataFrame.
        **fit_kwargs
            ``horizon`` (int, default 0): additional target periods to forecast.

        Returns
        -------
        LargeBVARResults
            Results (also in :attr:`results_`).

        Raises
        ------
        NowcastDataError
            If the panel is not monthly/quarterly or has too few complete quarters.
        ValueError
            On invalid hyper-parameters.
        TypeError
            On unknown fit options.

        Examples
        --------
        >>> import nowcastbox as nb
        >>> sim = nb.simulate.dfm(n_series=3, n_factors=1, n_periods=96, random_state=0)
        >>> res = LargeBVAR(prior={"lambda": 0.2}).fit(sim.data, "gdp ~ x01 + x02", horizon=1)
        >>> len(res.out_of_sample.dropna())
        2
        """
        results = super().fit(data, target, frequency=frequency, **fit_kwargs)
        assert isinstance(results, LargeBVARResults)  # noqa: S101
        return results

    def _options(self) -> _EstimationOptions:
        return _EstimationOptions(
            lags=self.lags,
            prior=self.prior,
            prior_mean=self.prior_mean,
            sum_of_coefficients=self.sum_of_coefficients,
            initial_observation=self.initial_observation,
            estimate_psi=self.estimate_psi,
            lag_decay=self.lag_decay,
            standardize=self.standardize,
            max_iter=self.max_iter,
        )

    def _validate_params(self) -> None:
        _validate_options(self._options())
        _check_int(self.n_draws, "n_draws", 0)

    def _fit(self, data: MixedFrequencyData, target: str, **fit_kwargs: Any) -> LargeBVARResults:
        """Estimate the blocked BVAR and nowcast ``target``.

        Parameters
        ----------
        data : MixedFrequencyData
            Panel with the target and its predictors.
        target : str
            Target series (quarterly or monthly).
        **fit_kwargs
            Only ``horizon`` (int, default 0).

        Returns
        -------
        LargeBVARResults
            Results.

        Raises
        ------
        TypeError
            On unknown fit options.
        """
        started = time.perf_counter()
        horizon = fit_kwargs.pop("horizon", 0)
        if fit_kwargs:
            raise TypeError(f"Unexpected fit options {sorted(fit_kwargs)} for LargeBVAR.")
        _check_int(horizon, "horizon", 0)
        panel = data.extend(_extension_periods(data, target, int(horizon)))
        engine = _fit_engine(panel, self._options())
        end, mean, var = engine.edge(engine.values)
        frame_parts = _TargetParts.build(engine, target, mean, var, end)
        draws = None
        if self.n_draws:
            draws = _stored_draws(engine, target, frame_parts, self.n_draws, self.random_state)
        nowcast = _nowcast_frame(panel, target, engine, frame_parts, draws)
        elapsed = time.perf_counter() - started
        return _assemble(self, panel, target, engine, nowcast, mean, end, draws, elapsed)


@dataclass(frozen=True)
class _TargetParts:
    """Standardised estimates of the target on its native periods."""

    index: pd.PeriodIndex
    rows: np.ndarray
    cols: np.ndarray
    estimate: np.ndarray
    variance: np.ndarray
    out_of_sample: np.ndarray

    @classmethod
    def build(
        cls, engine: BlockedBVAR, target: str, mean: np.ndarray, var: np.ndarray, end: int
    ) -> _TargetParts:
        layout, values = engine.layout, engine.values
        fitted = engine.parameters.fitted(mean)
        periods: list[pd.Period] = []
        rows: list[int] = []
        cols: list[int] = []
        for c in layout.columns_of(target):
            periods += list(native_periods(engine.quarters, layout, c))
            rows += list(range(len(engine.quarters)))
            cols += [c] * len(engine.quarters)
        index = pd.PeriodIndex(periods)
        order = np.argsort([p.ordinal for p in periods], kind="stable")
        r, c = np.asarray(rows)[order], np.asarray(cols)[order]
        observed = np.isfinite(values[r, c])
        oos = ~observed & (r > end)
        estimate = np.where(observed, fitted[r, c], np.where(oos, mean[r, c], np.nan))
        variance = np.where(oos, var[r, c], np.nan)
        return cls(index[order], r, c, estimate, variance, oos)


def _stored_draws(
    engine: BlockedBVAR, target: str, parts: _TargetParts, n_draws: int, random_state: Any
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Posterior mixture of the out-of-sample target periods (original units)."""
    if not parts.out_of_sample.any():
        return None
    cells = np.column_stack([parts.rows, parts.cols])[parts.out_of_sample]
    locs, scales = engine.mixture(engine.values, cells, n_draws, random_state)
    stats = engine.standardization
    index = parts.index[parts.out_of_sample]
    return (
        pd.DataFrame(stats.inverse_series(locs, target), index=index),
        pd.DataFrame(stats.inverse_series(scales, target, scale_only=True), index=index),
    )


def _nowcast_frame(
    panel: MixedFrequencyData,
    target: str,
    engine: BlockedBVAR,
    parts: _TargetParts,
    draws: tuple[pd.DataFrame, pd.DataFrame] | None,
) -> pd.DataFrame:
    stats = engine.standardization
    estimate = pd.Series(stats.inverse_series(parts.estimate, target), index=parts.index)
    std_values = stats.inverse_series(np.sqrt(parts.variance), target, scale_only=True)
    std = pd.Series(std_values, index=parts.index)
    extra: dict[str, pd.Series] = {"std": std}
    for label, level in (("68", 0.68), ("90", 0.9)):
        z = float(scipy.stats.norm.ppf(0.5 + level / 2))
        extra[f"lower_{label}"] = estimate - z * std
        extra[f"upper_{label}"] = estimate + z * std
    if draws is not None:
        _mixture_columns(extra, estimate, draws, target)
    observed = panel.to_native(target).reindex(parts.index)
    return build_nowcast_frame(observed, estimate, extra=extra)


def _mixture_columns(
    extra: dict[str, pd.Series],
    estimate: pd.Series,
    draws: tuple[pd.DataFrame, pd.DataFrame],
    target: str,
) -> None:
    """Replace ``std`` and the bounds by those of the posterior mixture."""
    from nowcastbox.density import NowcastDistribution

    locs, scales = draws
    point = estimate.reindex(locs.index).to_numpy(dtype=float)
    dist = NowcastDistribution(
        pd.PeriodIndex(locs.index), locs.to_numpy(), scales.to_numpy(), target=target, point=point
    )
    extra["std"] = dist.std.combine_first(extra["std"])
    for label, level in (("68", 0.68), ("90", 0.9)):
        bounds = dist.interval(level)
        extra[f"lower_{label}"] = bounds["lower"].combine_first(extra[f"lower_{label}"])
        extra[f"upper_{label}"] = bounds["upper"].combine_first(extra[f"upper_{label}"])


def _extension_periods(panel: MixedFrequencyData, target: str, horizon: int) -> int:
    """Months to append so the current target period (+ ``horizon``) and its quarter fit."""
    freq = panel.metadata[target].frequency.pandas_freq
    end = panel.index[-1]
    current = end.asfreq(freq)
    observed = panel.to_native(target, dropna=True)
    last = current + horizon
    if len(observed) and observed.index[-1] >= current:
        last = last + 1
    quarter_end = last.asfreq("M", how="E").asfreq("Q").asfreq("M", how="E")
    return max(int((quarter_end - end).n), 0)


def _assemble(
    model: LargeBVAR,
    panel: MixedFrequencyData,
    target: str,
    engine: BlockedBVAR,
    nowcast: pd.DataFrame,
    mean: np.ndarray,
    end: int,
    draws: tuple[pd.DataFrame, pd.DataFrame] | None,
    elapsed: float,
) -> LargeBVARResults:
    hyper = engine.hyperparameters
    start, stop = engine.sample
    params = {
        "intercept": engine.parameters.intercept,
        "coefficients": engine.parameters.coefficients,
        "sigma": engine.parameters.sigma,
        "lambda": hyper.lambda_,
        "mu": hyper.mu,
        "delta": hyper.delta,
        "psi": hyper.psi,
    }
    sel = engine.selection
    info = {
        "time": elapsed,
        "n_variables": engine.layout.n,
        "estimation_sample": (engine.quarters[start], engine.quarters[stop]),
        "n_draws": 0 if draws is None else int(draws[0].shape[1]),
        "hyperparameter_search": None if sel is None else sel.message,
    }
    return LargeBVARResults(
        target=target,
        nowcast=nowcast,
        model_name="LargeBVAR",
        model_params=model.get_params(),
        params=params,
        loglikelihood=float(engine.posterior.log_ml),
        converged=None if sel is None else sel.success,
        n_iter=None if sel is None else sel.n_evaluations,
        data=panel,
        standardization=engine.standardization,
        info=info,
        bvar=engine,
        smoothed_data=engine.to_original(mean, engine.quarters),
        window_end=engine.quarters[end],
        draw_locs=None if draws is None else draws[0],
        draw_scales=None if draws is None else draws[1],
    )
