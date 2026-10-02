r"""BVAR extrapolation of indicators (``extrapolation="bvar"``).

Bridge equations need every indicator up to the end of the target period. The ``"bvar"``
extrapolator of the registry (:mod:`nowcastbox.models.extrapolation`) completes all the
requested indicators **jointly** with the conditional forecast of one Bayesian VAR,
estimated with the Normal-Inverse-Wishart Minnesota prior and the hierarchical choice of
the tightness of Giannone, Lenza & Primiceri (2015), at the posterior mean of
:math:`(B, \Sigma)`. The unreleased entries are the conditional expectation given every
released value (Waggoner & Zha, 1999; Bańbura, Giannone & Lenza, 2015), computed with the
Kalman smoother of :mod:`nowcastbox.statespace` on the companion form of the VAR.

Two VARs are available:

* **Monthly BVAR** (:class:`MonthlyBVAR`, :func:`fit_monthly_bvar`) - a VAR(:math:`p`)
  on the indicators at the base frequency,

  .. math::

      x_t = c + A_1 x_{t-1} + \dots + A_p x_{t-p} + \varepsilon_t,\qquad
      \varepsilon_t \sim N(0, \Sigma),

  the large monthly BVAR of Bańbura, Giannone & Reichlin (2010). Used by default when
  every requested series is at the base (monthly) frequency, which is the usual case for
  bridge-equation indicators.
* **Blocked quarterly BVAR** (:class:`~nowcastbox.models.BlockedBVAR`) - the "blocking"
  model of Cimadomo, Giannone, Lenza, Monti & Sokol (2022) used by
  :class:`~nowcastbox.models.LargeBVAR`: each monthly series becomes three quarterly
  variables. Used by default when quarterly indicators are requested too, because it is
  the joint model of monthly and quarterly series.

Why the monthly VAR for monthly indicators. Extrapolating monthly indicators is a
monthly forecasting problem. The monthly VAR has :math:`N` variables and three times as
many observations as the blocked VAR, which has :math:`3N` variables (150 for 50
indicators) on quarterly data. It also uses every complete month of the ragged edge
directly. Blocking is needed only to put quarterly variables in the same VAR.

The VAR is estimated once per data set and cached in the extrapolator instance (keyed by
a hash of the selected panel), so calls for other ends (longer horizons) with the same
columns on the same vintage reuse it; another vintage or another set of columns is a new
estimation.

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
>>> import pandas as pd
>>> from nowcastbox.models.extrapolation import make_extrapolator
>>> sim = nb.simulate.dfm(n_series=3, n_factors=1, n_periods=96, random_state=0)
>>> ext = make_extrapolator("bvar")
>>> out = ext(sim.data, ["x01", "x02"], pd.Period("2008Q1", "Q"))
>>> str(out["x01"].index[-1]), bool(out["x01"].notna().all())
('2008-03', True)
"""

from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from nowcastbox._logging import get_logger
from nowcastbox.core.data import MixedFrequencyData, StandardizationStats, as_mixed_frequency_data
from nowcastbox.core.exceptions import NowcastDataError
from nowcastbox.models._bvar_blocking import VARParameters, conditional_forecast, window_end
from nowcastbox.models._bvar_prior import (
    BVARHyperparameters,
    HyperparameterSelection,
    NIWPosterior,
    PriorSettings,
)
from nowcastbox.models.bvar import (
    BlockedBVAR,
    _estimate_var,  # pyright: ignore[reportPrivateUsage]
    _EstimationOptions,  # pyright: ignore[reportPrivateUsage]
    _fit_engine,  # pyright: ignore[reportPrivateUsage]
    _identity_stats,  # pyright: ignore[reportPrivateUsage]
    _panel_frame,  # pyright: ignore[reportPrivateUsage]
    _validate_options,  # pyright: ignore[reportPrivateUsage]
)
from nowcastbox.models.extrapolation import native_until, register_extrapolator

__all__ = ["BVARExtrapolator", "MonthlyBVAR", "fit_monthly_bvar"]

logger = get_logger(__name__)

_MONTHLY_LAGS = 3
_BLOCKED_LAGS = 1
_CACHE_SIZE = 4


# ====================================================================== monthly engine
@dataclass(frozen=True, eq=False)
class MonthlyBVAR:
    r"""Fitted BVAR on series of the base frequency (monthly): conditional forecasts.

    Built by :func:`fit_monthly_bvar`; used by the ``"bvar"`` extrapolator.

    Parameters
    ----------
    columns : tuple of str
        Series of the VAR, in order.
    standardization : StandardizationStats
        Statistics of the series (identity when not standardised).
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
    index : pandas.PeriodIndex
        Base grid of the estimation panel.
    values : numpy.ndarray
        Standardised data on :attr:`index`, shape ``(len(index), n)``.
    sample : tuple of int
        First and last rows (inclusive) of the balanced estimation sample.

    Examples
    --------
    >>> import nowcastbox as nb
    >>> sim = nb.simulate.dfm(n_series=3, n_factors=1, n_periods=96, random_state=0)
    >>> eng = fit_monthly_bvar(sim.data.select(["x01", "x02"]), prior={"lambda": 0.2})
    >>> eng.columns, eng.lags
    (('x01', 'x02'), 3)
    """

    columns: tuple[str, ...]
    standardization: StandardizationStats
    parameters: VARParameters
    posterior: NIWPosterior = field(repr=False)
    hyperparameters: BVARHyperparameters
    selection: HyperparameterSelection | None = field(repr=False)
    settings: PriorSettings
    index: pd.PeriodIndex = field(repr=False)
    values: np.ndarray = field(repr=False)
    sample: tuple[int, int]

    @property
    def lags(self) -> int:
        """VAR order :math:`p` (base periods).

        Returns
        -------
        int
            Number of lags.

        Examples
        --------
        >>> import nowcastbox as nb
        >>> sim = nb.simulate.dfm(n_series=2, n_factors=1, n_periods=96, random_state=0)
        >>> fit_monthly_bvar(sim.data.select(["x01"]), lags=2, prior={"lambda": 0.2}).lags
        2
        """
        return self.parameters.lags

    def standardized(self, data: object) -> tuple[pd.PeriodIndex, np.ndarray]:
        """Standardised matrix of a panel, at least as long as the estimation grid.

        Parameters
        ----------
        data : MixedFrequencyData or pandas.DataFrame
            Panel on the base grid containing every model series.

        Returns
        -------
        index : pandas.PeriodIndex
            Base grid (extended with empty periods up to the end of :attr:`index` when
            shorter).
        values : numpy.ndarray
            Standardised values, one column per series of :attr:`columns`.

        Raises
        ------
        NowcastDataError
            If series are missing.

        Examples
        --------
        >>> import nowcastbox as nb
        >>> sim = nb.simulate.dfm(n_series=2, n_factors=1, n_periods=96, random_state=0)
        >>> eng = fit_monthly_bvar(sim.data.select(["x01"]), prior={"lambda": 0.2})
        >>> eng.standardized(sim.data.truncate(end="2003-12"))[1].shape
        (96, 1)
        """
        frame = _panel_frame(data)
        missing = [s for s in self.columns if s not in frame.columns]
        if missing:
            raise NowcastDataError(f"The data do not contain the model series {missing}.")
        std = self.standardization.transform(frame.loc[:, list(self.columns)])
        index = pd.PeriodIndex(std.index)
        values = std.to_numpy(dtype=np.float64)
        end = self.index[-1]
        if index[-1] < end:
            extra = int((end - index[-1]).n)
            values = np.vstack([values, np.full((extra, len(self.columns)), np.nan)])
            index = pd.period_range(index[0], end, freq=index.freq)
        return index, values

    def edge(self, values: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
        """Conditional forecast of everything after the last complete window.

        Parameters
        ----------
        values : numpy.ndarray
            Standardised matrix (rows = base periods).

        Returns
        -------
        end : int
            Last row of the last window of :attr:`lags` complete periods.
        mean : numpy.ndarray
            ``values`` with the rows after ``end`` replaced by their conditional
            expectation (released entries unchanged).
        variance : numpy.ndarray
            Conditional variances (zero up to ``end`` and for released entries).

        Raises
        ------
        NowcastDataError
            If the data have no window of :attr:`lags` complete periods.

        Examples
        --------
        >>> import nowcastbox as nb
        >>> sim = nb.simulate.dfm(n_series=2, n_factors=1, n_periods=96, random_state=0)
        >>> eng = fit_monthly_bvar(sim.data.select(["x01", "x02"]), prior={"lambda": 0.2})
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

    def predict(self, data: object) -> pd.DataFrame:
        """``E[x_t | data]`` for every series, in original units (posterior-mean parameters).

        Parameters
        ----------
        data : MixedFrequencyData or pandas.DataFrame
            Vintage containing every model series.

        Returns
        -------
        pandas.DataFrame
            Base grid covering the data and the estimation grid: released values
            unchanged, entries after the last window of :attr:`lags` complete periods
            replaced by their conditional expectation (earlier gaps stay ``NaN``).

        Examples
        --------
        >>> import nowcastbox as nb
        >>> sim = nb.simulate.dfm(n_series=2, n_factors=1, n_periods=96, random_state=0)
        >>> panel = sim.data.select(["x01", "x02"])
        >>> eng = fit_monthly_bvar(panel, prior={"lambda": 0.2})
        >>> bool(eng.predict(panel.extend(3)).iloc[-1].notna().all())
        True
        """
        index, values = self.standardized(data)
        _, mean, _ = self.edge(values)
        frame = pd.DataFrame(mean, index=index, columns=list(self.columns))
        return self.standardization.inverse_transform(frame)


def _options(
    lags: int,
    prior: Any,
    prior_mean: Any,
    sum_of_coefficients: bool,
    initial_observation: bool,
    estimate_psi: bool,
    lag_decay: float,
    standardize: bool,
    max_iter: int,
) -> _EstimationOptions:
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
    return options


def fit_monthly_bvar(
    data: MixedFrequencyData | pd.DataFrame,
    *,
    lags: int = _MONTHLY_LAGS,
    prior: Any = "glp",
    prior_mean: Any = "white_noise",
    sum_of_coefficients: bool = False,
    initial_observation: bool = False,
    estimate_psi: bool = False,
    lag_decay: float = 2.0,
    standardize: bool = True,
    max_iter: int = 500,
) -> MonthlyBVAR:
    """Estimate a BVAR on every series of a panel at the base frequency.

    Parameters
    ----------
    data : MixedFrequencyData or pandas.DataFrame
        Panel whose series all have the base frequency (normally monthly).
    lags : int, default 3
        VAR order in base periods.
    prior, prior_mean, sum_of_coefficients, initial_observation, estimate_psi, \
lag_decay, standardize, max_iter
        See :class:`~nowcastbox.models.LargeBVAR`.

    Returns
    -------
    MonthlyBVAR
        Fitted engine (posterior on the longest run of complete periods ending at the
        last complete one).

    Raises
    ------
    ValueError
        On invalid options.
    NowcastDataError
        If a series is not at the base frequency or has no observation, or the
        balanced sample has fewer than ``lags + 3`` periods.

    Examples
    --------
    >>> import nowcastbox as nb
    >>> sim = nb.simulate.dfm(n_series=3, n_factors=1, n_periods=96, random_state=0)
    >>> eng = fit_monthly_bvar(sim.data.select(["x01", "x02", "x03"]), lags=2)
    >>> eng.parameters.coefficients.shape
    (2, 3, 3)
    """
    options = _options(
        lags,
        prior,
        prior_mean,
        sum_of_coefficients,
        initial_observation,
        estimate_psi,
        lag_decay,
        standardize,
        max_iter,
    )
    return _fit_monthly(as_mixed_frequency_data(data), options)


def _check_base_frequency(panel: MixedFrequencyData) -> None:
    other = [c for c in panel.columns if panel.metadata[c].frequency != panel.base_frequency]
    if other:
        raise NowcastDataError(
            f"The monthly BVAR needs every series at the base frequency "
            f"({panel.base_frequency.value}); {other} are not (use the blocked BVAR)."
        )
    empty = [c for c in panel.columns if not panel.data[c].notna().any()]
    if empty:
        raise NowcastDataError(f"Series {empty} have no observations.")


def _fit_monthly(panel: MixedFrequencyData, options: _EstimationOptions) -> MonthlyBVAR:
    _check_base_frequency(panel)
    columns = tuple(panel.columns)
    stats = panel.standardization_stats() if options.standardize else _identity_stats(columns)
    std = stats.transform(panel.data.loc[:, list(columns)])
    index = pd.PeriodIndex(std.index)
    values = std.to_numpy(dtype=np.float64)
    est = _estimate_var(values, index, list(columns), options)
    return MonthlyBVAR(
        columns=columns,
        standardization=stats,
        parameters=est.parameters,
        posterior=est.posterior,
        hyperparameters=est.hyperparameters,
        selection=est.selection,
        settings=est.settings,
        index=index,
        values=values,
        sample=est.sample,
    )


# ====================================================================== extrapolator
def _fingerprint(panel: MixedFrequencyData) -> str:
    """Hash of the values, grid, names and frequencies of a panel."""
    digest = hashlib.blake2b(digest_size=16)
    digest.update(repr(tuple(panel.columns)).encode())
    digest.update(repr([panel.metadata[c].frequency.value for c in panel.columns]).encode())
    digest.update(str(panel.index[0]).encode() + panel.index.freqstr.encode())
    digest.update(np.ascontiguousarray(panel.data.to_numpy(dtype=np.float64)).tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class BVARExtrapolator:
    """Indicator extrapolation with a Bayesian VAR (``extrapolation="bvar"``).

    All the requested indicators are completed jointly by the conditional forecast of
    one BVAR estimated on them (posterior mean of the parameters), for
    :class:`~nowcastbox.models.BridgeCombination` and the other users of the extrapolator
    registry. By default a monthly VAR is used when every requested series is at the
    base frequency, and the blocked quarterly VAR of
    :class:`~nowcastbox.models.LargeBVAR` otherwise (see the module docstring for the
    reasons). Fitted VARs are cached in the instance, keyed by a hash of the data.

    Parameters
    ----------
    lags : int, optional
        VAR order in periods of the VAR: default 3 (months) for the monthly VAR and 1
        (quarter) for the blocked VAR, the same one-quarter memory.
    blocking : bool, optional
        ``False``: monthly VAR (every series must be at the base frequency); ``True``:
        blocked quarterly VAR; ``None`` (default): monthly when possible, blocked
        otherwise.
    prior, prior_mean, sum_of_coefficients, initial_observation, estimate_psi, \
lag_decay, standardize, max_iter
        See :class:`~nowcastbox.models.LargeBVAR`.

    Raises
    ------
    ValueError
        On invalid options.

    Examples
    --------
    >>> import nowcastbox as nb
    >>> import pandas as pd
    >>> sim = nb.simulate.dfm(n_series=3, n_factors=1, n_periods=96, random_state=0)
    >>> ext = BVARExtrapolator(prior={"lambda": 0.2})
    >>> out = ext(sim.data, ["x01", "gdp"], pd.Period("2008Q2", "Q"))  # blocked VAR
    >>> str(out["gdp"].index[-1]), bool(out["gdp"].iloc[-3:].notna().all())
    ('2008Q2', True)
    >>> ext.mode(sim.data, ["x01", "x02"]), ext.mode(sim.data, ["x01", "gdp"])
    ('monthly', 'blocked')
    """

    lags: int | None = None
    blocking: bool | None = None
    prior: Any = "glp"
    prior_mean: Any = "white_noise"
    sum_of_coefficients: bool = False
    initial_observation: bool = False
    estimate_psi: bool = False
    lag_decay: float = 2.0
    standardize: bool = True
    max_iter: int = 500
    _engines: dict[str, MonthlyBVAR | BlockedBVAR] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self.blocking is not None and not isinstance(self.blocking, bool):
            raise ValueError(f"blocking must be a bool or None, got {self.blocking!r}.")
        self._estimation_options(blocked=bool(self.blocking))

    def _estimation_options(self, *, blocked: bool) -> _EstimationOptions:
        values = {
            f.name: getattr(self, f.name)
            for f in dataclasses.fields(self)
            if f.init and f.name != "blocking"
        }
        if values["lags"] is None:
            values["lags"] = _BLOCKED_LAGS if blocked else _MONTHLY_LAGS
        return _options(**values)

    def mode(self, data: MixedFrequencyData, columns: Sequence[str]) -> str:
        """VAR used for ``columns``: ``"monthly"`` or ``"blocked"``.

        Parameters
        ----------
        data : MixedFrequencyData
            Panel.
        columns : sequence of str
            Series to complete.

        Returns
        -------
        str
            ``"blocked"`` when ``blocking=True`` or (``blocking=None``) some series is
            not at the base frequency, ``"monthly"`` otherwise.

        Examples
        --------
        >>> import nowcastbox as nb
        >>> sim = nb.simulate.dfm(n_series=2, n_factors=1, n_periods=96, random_state=0)
        >>> BVARExtrapolator(blocking=True).mode(sim.data, ["x01"])
        'blocked'
        """
        if self.blocking is not None:
            return "blocked" if self.blocking else "monthly"
        base = data.base_frequency
        same = all(data.metadata[c].frequency == base for c in columns)
        return "monthly" if same else "blocked"

    def engine(self, data: MixedFrequencyData, columns: Sequence[str]) -> MonthlyBVAR | BlockedBVAR:
        """Fitted VAR on ``columns`` of ``data`` (estimated once, then cached).

        Parameters
        ----------
        data : MixedFrequencyData
            Panel (the vintage's information set).
        columns : sequence of str
            Series of the VAR.

        Returns
        -------
        MonthlyBVAR or BlockedBVAR
            Fitted engine.

        Raises
        ------
        NowcastDataError
            If the VAR cannot be estimated.

        Examples
        --------
        >>> import nowcastbox as nb
        >>> sim = nb.simulate.dfm(n_series=2, n_factors=1, n_periods=96, random_state=0)
        >>> ext = BVARExtrapolator(prior={"lambda": 0.2})
        >>> ext.engine(sim.data, ["x01"]) is ext.engine(sim.data, ["x01"])
        True
        """
        mode = self.mode(data, columns)
        panel = data.select(list(columns))
        key = f"{mode}:{_fingerprint(panel)}"
        cached = self._engines.get(key)
        if cached is not None:
            return cached
        options = self._estimation_options(blocked=mode == "blocked")
        engine: MonthlyBVAR | BlockedBVAR
        if mode == "blocked":
            engine = _fit_engine(panel, options)
        else:
            engine = _fit_monthly(panel, options)
        if len(self._engines) >= _CACHE_SIZE:
            self._engines.pop(next(iter(self._engines)))
        self._engines[key] = engine
        return engine

    def __call__(
        self, data: MixedFrequencyData, columns: Sequence[str], end: pd.Period
    ) -> dict[str, pd.Series]:
        """Complete ``columns`` up to the native periods contained in ``end``.

        Parameters
        ----------
        data : MixedFrequencyData
            Panel (the vintage's information set).
        columns : sequence of str
            Series to complete.
        end : pandas.Period
            Last period to cover.

        Returns
        -------
        dict of str to pandas.Series
            Series on their native grids; observed values unchanged, values after the
            last observation filled by the conditional forecast.

        Raises
        ------
        NowcastDataError
            If the BVAR cannot be estimated (too few complete periods, unsupported
            frequencies) or a series has no observation.

        Examples
        --------
        >>> import nowcastbox as nb
        >>> sim = nb.simulate.dfm(n_series=3, n_factors=1, n_periods=96, random_state=0)
        >>> out = BVARExtrapolator()(sim.data, ["x01", "x02"], pd.Period("2008Q2", "Q"))
        >>> str(out["x02"].index[-1]), bool(out["x02"].iloc[-6:].notna().all())
        ('2008-06', True)
        """
        columns = list(columns)
        empty = [c for c in columns if not data.data[c].notna().any()]
        if empty:
            raise NowcastDataError(f"Predictors {empty} have no observations.")
        engine = self.engine(data, columns)
        panel = data.select(columns)
        last = end.asfreq(data.base_frequency.pandas_freq, how="E")
        if isinstance(engine, BlockedBVAR):  # whole quarters on the blocked grid
            last = last.asfreq("Q").asfreq(data.base_frequency.pandas_freq, how="E")
        if last > panel.end:
            panel = panel.extend(int((last - panel.end).n))
        predicted = engine.predict(panel)
        out = {col: _fill(data, col, end, predicted[col]) for col in columns}
        logger.debug(
            "BVAR (%s) extrapolation of %d series up to %s",
            self.mode(data, columns),
            len(columns),
            end,
        )
        return out


def _fill(data: MixedFrequencyData, col: str, end: pd.Period, predicted: pd.Series) -> pd.Series:
    """Native series of ``col`` up to ``end`` with the values after its last observation filled."""
    native = native_until(data, col, end)
    values = native.to_numpy(dtype=float).copy()
    observed = np.flatnonzero(np.isfinite(values))  # non-empty: checked before the fit
    source = predicted.dropna()
    freq = data.metadata[col].frequency.pandas_freq
    source.index = pd.PeriodIndex(source.index).asfreq(freq)
    fill = source.reindex(native.index).to_numpy(dtype=float)
    after = np.arange(len(values)) > observed[-1]
    values[after] = fill[after]
    return pd.Series(values, index=native.index, name=col)


register_extrapolator("bvar", BVARExtrapolator, overwrite=True)
