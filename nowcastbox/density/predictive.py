r"""Predictive distribution of the nowcast of a fitted model (innovation I5).

Two sources of uncertainty are combined:

**(a) Filtering/smoothing uncertainty** (analytic, Gaussian). Given the estimated
parameters the target is Gaussian conditional on the data, with variance

.. math::

    \operatorname{Var}(y_t \mid \Omega) = Z_y P_{t|T} Z_y' + H_{yy}

for :class:`~nowcastbox.models.MixedFreqDFM` (the row :math:`Z_y` of the target
already contains the aggregation weights of the low-frequency target, so this is the
variance of the aggregated latent series plus its measurement noise), and
:math:`\beta' V_t \beta + \sigma_e^2` for :class:`~nowcastbox.models.TwoStepDFM`
(:math:`V_t` = smoothed covariance of the aggregated factors, :math:`\sigma_e` = bridge
residual standard error). Both models report it in the ``std`` column of
``results.nowcast``; :func:`analytic_distribution` turns it into a
:class:`~nowcastbox.density.NowcastDistribution` with one Gaussian component.

**(b) Parameter uncertainty** (bootstrap, :mod:`nowcastbox.density.bootstrap`). With
re-estimated parameters :math:`\hat\theta^*_b`, :math:`b = 1..B`, the predictive
distribution is the mixture (Hamilton, 1986)

.. math::

    p(y_t \mid \Omega) \approx \frac{1}{B}\sum_{b=1}^{B}
    N\!\left(\hat y_t + \bigl(\hat y_t(\hat\theta^*_b) - \bar y^*_t\bigr),\;
             s_t^2(\hat\theta^*_b)\right),

centred at the point nowcast :math:`\hat y_t` (``center=True``), so that

.. math::

    \operatorname{Var}(y_t \mid \Omega) =
    \underbrace{\tfrac1B\textstyle\sum_b s_t^2(\hat\theta^*_b)}_{\text{filtering}} +
    \underbrace{\tfrac1B\textstyle\sum_b(\hat y_t(\hat\theta^*_b) - \bar y^*_t)^2}
    _{\text{parameter}},

the bootstrap decomposition of the prediction MSE of Pfeffermann & Tiller (2005) and
Rodríguez & Ruiz (2012).

References
----------
Hamilton, J. D. (1986). A standard error for the estimated state vector of a
state-space model. *Journal of Econometrics*, 33(3), 387-397.

Pfeffermann, D. & Tiller, R. (2005). *Journal of Time Series Analysis*, 26(6), 893-916.

Rodríguez, A. & Ruiz, E. (2012). *Computational Statistics & Data Analysis*, 56(1),
62-74.

Durbin, J. & Koopman, S. J. (2012). *Time Series Analysis by State Space Methods*,
2nd ed., Oxford University Press, section 4.4.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from nowcastbox.core.exceptions import NowcastDataError
from nowcastbox.core.results import NowcastResults
from nowcastbox.density.bootstrap import (
    BootstrapMethod,
    BootstrapNowcasts,
    RandomState,
    bootstrap_nowcasts,
    default_periods,
)
from nowcastbox.density.distribution import NowcastDistribution

__all__ = ["analytic_distribution", "combine_bootstrap", "nowcast_distribution"]


def _as_period_index(
    periods: Iterable[pd.Period | str] | pd.Period | str, freq: str
) -> pd.PeriodIndex:
    """Coerce a period specification to a non-empty PeriodIndex of frequency ``freq``."""
    items = [periods] if isinstance(periods, str | pd.Period) else list(periods)
    if not items:
        raise NowcastDataError("periods must not be empty.")
    try:
        return pd.PeriodIndex([pd.Period(p, freq=freq) for p in items])
    except (TypeError, ValueError) as exc:
        raise NowcastDataError(f"Invalid target periods {items!r}.") from exc


def _resolve_periods(
    results: NowcastResults, periods: Iterable[pd.Period | str] | pd.Period | str | None
) -> pd.PeriodIndex:
    """Requested target periods, checked against the ``nowcast`` frame."""
    if periods is None:
        return default_periods(results)
    if "std" not in results.nowcast.columns:
        default_periods(results)  # raises the explicit error
    index = _as_period_index(periods, results.target_frequency.pandas_freq)
    estimate = results.estimate.reindex(index)
    std = results.nowcast["std"].reindex(index)
    bad = [str(p) for p, e, s in zip(index, estimate, std, strict=True) if not (e == e and s > 0)]
    if bad:
        raise NowcastDataError(f"No estimate with a standard deviation for the periods {bad}.")
    return index


def analytic_distribution(
    results: NowcastResults,
    periods: Iterable[pd.Period | str] | pd.Period | str | None = None,
) -> NowcastDistribution:
    """Gaussian predictive distribution from the Kalman smoother (parameters fixed).

    Parameters
    ----------
    results : NowcastResults
        Fitted results whose ``nowcast`` frame has a ``std`` column
        (``TwoStepDFM``, ``MixedFreqDFM``).
    periods : period, str or iterable, optional
        Target periods (default: every out-of-sample period with a standard
        deviation - backcasts, nowcast and forecasts).

    Returns
    -------
    NowcastDistribution
        One Gaussian component per period, mean = model estimate, standard deviation
        = ``nowcast["std"]``.

    Raises
    ------
    NowcastDataError
        If there is no ``std`` column or a requested period has no estimate/std.

    Examples
    --------
    >>> from nowcastbox.models import TwoStepDFM
    >>> from nowcastbox.models.two_step import simulate_two_step_example
    >>> from nowcastbox.density import analytic_distribution
    >>> res = TwoStepDFM(n_factors=1).fit(simulate_two_step_example(random_state=0), "gdp")
    >>> dist = analytic_distribution(res)
    >>> dist.is_gaussian, [str(p) for p in dist.index]
    (True, ['2014Q4', '2015Q1'])
    """
    index = _resolve_periods(results, periods)
    mean = results.estimate.reindex(index).to_numpy(dtype=np.float64)
    std = results.nowcast["std"].reindex(index).to_numpy(dtype=np.float64)
    decomposition = pd.DataFrame(
        {"filtering": std**2, "parameter": np.zeros(len(index)), "total": std**2}, index=index
    )
    decomposition.index.name = "period"
    return NowcastDistribution(
        index,
        mean,
        std,
        target=results.target,
        point=mean,
        variance_decomposition=decomposition,
        info={"source": "analytic", "model": results.model_name},
    )


def combine_bootstrap(
    analytic: NowcastDistribution, bootstrap: BootstrapNowcasts, *, center: bool = True
) -> NowcastDistribution:
    """Combine the analytic distribution with bootstrapped nowcasts (Gaussian mixture).

    Parameters
    ----------
    analytic : NowcastDistribution
        Gaussian distribution of the point estimates (:func:`analytic_distribution`).
    bootstrap : BootstrapNowcasts
        Bootstrapped nowcasts for (at least) the same periods.
    center : bool, default True
        Shift the bootstrap nowcasts so that their average equals the point nowcast
        (the mixture mean is then the model estimate).

    Returns
    -------
    NowcastDistribution
        Mixture with one component per successful replication.

    Raises
    ------
    NowcastDataError
        If a period of ``analytic`` is missing from ``bootstrap``.

    Examples
    --------
    >>> from nowcastbox.models import TwoStepDFM
    >>> from nowcastbox.models.two_step import simulate_two_step_example
    >>> from nowcastbox.density import analytic_distribution, bootstrap_nowcasts
    >>> from nowcastbox.density import combine_bootstrap
    >>> res = TwoStepDFM(n_factors=1).fit(simulate_two_step_example(random_state=0), "gdp")
    >>> mix = combine_bootstrap(analytic_distribution(res), bootstrap_nowcasts(res, 5))
    >>> mix.n_components
    5
    """
    index = analytic.index
    covered = set(bootstrap.means.columns)
    missing = [str(p) for p in index if p not in covered]
    if missing:
        raise NowcastDataError(f"The bootstrap does not cover the periods {missing}.")
    means = bootstrap.means.reindex(columns=index).to_numpy(dtype=np.float64)
    stds = bootstrap.stds.reindex(columns=index).to_numpy(dtype=np.float64)
    point = analytic.point.to_numpy()
    locs = (point[None, :] + means - means.mean(axis=0)) if center else means
    filtering = (stds**2).mean(axis=0)
    parameter = means.var(axis=0)
    decomposition = pd.DataFrame(
        {"filtering": filtering, "parameter": parameter, "total": filtering + parameter},
        index=index,
    )
    info: dict[str, Any] = analytic.info | {
        "source": "bootstrap",
        "method": bootstrap.method,
        "n_boot": bootstrap.n_boot,
        "n_failed": bootstrap.n_failed,
        "n_convergence_warnings": bootstrap.n_convergence_warnings,
        "analytic_std": analytic.std,
        "center": center,
    }
    return NowcastDistribution(
        index,
        locs.T,
        stds.T,
        target=analytic.target,
        point=point,
        variance_decomposition=decomposition,
        info=info,
    )


def nowcast_distribution(
    results: NowcastResults,
    *,
    n_boot: int = 0,
    method: BootstrapMethod = "auto",
    periods: Iterable[pd.Period | str] | pd.Period | str | None = None,
    random_state: RandomState = None,
    n_jobs: int | None = None,
    refit_params: Mapping[str, Any] | None = None,
    warm_start: bool = True,
    block_length: int | None = None,
    center: bool = True,
) -> NowcastDistribution:
    """Predictive distribution of the nowcast, optionally with parameter uncertainty.

    Parameters
    ----------
    results : NowcastResults
        Fitted ``TwoStepDFM`` or ``MixedFreqDFM`` results (any results with a ``std``
        column when ``n_boot=0``).
    n_boot : int, default 0
        Bootstrap replications; ``0`` returns the analytic Gaussian distribution
        (filtering uncertainty only).
    method : {"auto", "parametric", "block"}, default "auto"
        Bootstrap scheme (see :func:`~nowcastbox.density.bootstrap_nowcasts`).
    periods : period, str or iterable, optional
        Target periods (default: out-of-sample periods with a standard deviation).
    random_state : int, numpy.random.Generator, SeedSequence or None
        Seed of the bootstrap.
    n_jobs : int, optional
        joblib workers for the bootstrap.
    refit_params : mapping, optional
        Estimator hyper-parameters overriding the original ones in the refits.
    warm_start : bool, default True
        Start EM refits from the original estimates.
    block_length : int, optional
        Block length (target periods) of the block bootstrap.
    center : bool, default True
        Centre the bootstrap mixture at the point nowcast.

    Returns
    -------
    NowcastDistribution
        Gaussian (``n_boot=0``) or Gaussian mixture with ``variance_decomposition``
        (filtering vs. parameter uncertainty).

    Raises
    ------
    NowcastDataError
        If the results provide no standard deviation for the requested periods.
    ValueError
        On invalid ``n_boot``/``method``.

    Examples
    --------
    >>> from nowcastbox.models import TwoStepDFM
    >>> from nowcastbox.models.two_step import simulate_two_step_example
    >>> from nowcastbox.density import nowcast_distribution
    >>> res = TwoStepDFM(n_factors=1).fit(simulate_two_step_example(random_state=0), "gdp")
    >>> dist = nowcast_distribution(res, n_boot=20, random_state=0)
    >>> dist.n_components
    20
    >>> bool((dist.std >= dist.variance_decomposition["filtering"] ** 0.5).all())
    True
    >>> list(dist.to_frame(levels=[0.68, 0.9]).columns)
    ['point', 'mean', 'std', 'median', 'lower_68', 'upper_68', 'lower_90', 'upper_90']
    """
    if isinstance(n_boot, bool) or not isinstance(n_boot, int | np.integer) or n_boot < 0:
        raise ValueError(f"n_boot must be a non-negative integer, got {n_boot!r}.")
    analytic = analytic_distribution(results, periods)
    if n_boot == 0:
        return analytic
    boot = bootstrap_nowcasts(
        results,
        int(n_boot),
        method=method,
        periods=analytic.index,
        random_state=random_state,
        n_jobs=n_jobs,
        refit_params=refit_params,
        warm_start=warm_start,
        block_length=block_length,
    )
    return combine_bootstrap(analytic, boot, center=center)
