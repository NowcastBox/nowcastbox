r"""Deterministic simulation of a mixed-frequency dynamic factor model.

The data-generating process is the textbook DFM of Giannone, Reichlin & Small (2008)
and Bańbura & Modugno (2014) with a quarterly variable linked to its latent monthly
counterpart by the Mariano & Murasawa (2003) restriction:

.. math::

    f_t &= A f_{t-1} + u_t, & u_t &\sim N(0, Q), \quad A = \operatorname{diag}(\phi),
    \; Q = I - A A^\top, \\
    x_{it} &= \lambda_i^\top f_t + e_{it}, & e_{it} &\sim N(0, \psi_i), \\
    y^*_t &= \beta^\top f_t + \varepsilon_t, & \varepsilon_t &\sim N(0, \sigma^2), \\
    y^Q_t &= \tfrac13 (y^*_t + 2 y^*_{t-1} + 3 y^*_{t-2} + 2 y^*_{t-3} + y^*_{t-4}),

where :math:`y^Q_t` is observed only in the third month of each quarter. The factors
have unit stationary variance, so the share of common variance of series :math:`i` is
:math:`R^2_i = \lambda_i^\top\lambda_i / (\lambda_i^\top\lambda_i + \psi_i)`; the
:math:`R^2_i` are drawn uniformly on ``r2_range``. A ragged edge is imposed by giving
every monthly series a publication lag of 0, 1 or 2 months and leaving the last quarter
of the quarterly target unobserved.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from nowcastbox.core.frequency import AggregationType

__all__ = ["simulate_mixed_frequency_dfm"]

_LAG_DELAY_DAYS = {0: 5, 1: 35, 2: 65}
_TARGET_DELAY_DAYS = 45


def _check_int(value: object, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}.")
    return int(value)


def simulate_mixed_frequency_dfm(
    n_monthly: int = 20,
    n_factors: int = 2,
    n_periods: int = 240,
    *,
    start: str = "2000-01",
    r2_range: tuple[float, float] = (0.3, 0.8),
    target_r2: float = 0.7,
    ragged_edge: bool = True,
    random_state: int | np.random.Generator | None = 0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Simulate a monthly panel plus a quarterly target from a known DFM.

    Parameters
    ----------
    n_monthly : int, default 20
        Number of monthly indicators.
    n_factors : int, default 2
        Number of common factors.
    n_periods : int, default 240
        Number of months (at least 24).
    start : str, default "2000-01"
        First month.
    r2_range : (float, float), default (0.3, 0.8)
        Range of the common-variance shares of the monthly series, within (0, 1).
    target_r2 : float, default 0.7
        Common-variance share of the latent monthly target, within (0, 1).
    ragged_edge : bool, default True
        Impose publication lags (0-2 months) and an unobserved last quarter.
    random_state : int, numpy.random.Generator or None, default 0
        Seed (the default makes the output deterministic).

    Returns
    -------
    data : pandas.DataFrame
        Monthly PeriodIndex; columns ``x01``.. (monthly) and ``gdp`` (quarterly, third
        month of each quarter).
    truth : dict
        True parameters and latent quantities: ``factors`` (DataFrame), ``loadings``
        (DataFrame, monthly series and ``gdp``), ``transition`` (A), ``factor_cov``
        (Q), ``idiosyncratic_var`` (Series), ``gdp_monthly`` (latent monthly target),
        ``aggregation_weights`` and ``publication_lags`` (months, Series).

    Raises
    ------
    ValueError
        On invalid sizes or shares.

    Examples
    --------
    >>> from nowcastbox.datasets._simulated import simulate_mixed_frequency_dfm
    >>> data, truth = simulate_mixed_frequency_dfm(n_monthly=5, n_factors=1, n_periods=36)
    >>> data.shape, truth["loadings"].shape
    ((36, 6), (6, 1))
    """
    n = _check_int(n_monthly, "n_monthly", 1)
    r = _check_int(n_factors, "n_factors", 1)
    t_len = _check_int(n_periods, "n_periods", 24)
    lo, hi = (float(v) for v in r2_range)
    if not 0.0 < lo <= hi < 1.0:
        raise ValueError(f"r2_range must satisfy 0 < low <= high < 1, got {r2_range!r}.")
    if not 0.0 < float(target_r2) < 1.0:
        raise ValueError(f"target_r2 must be in (0, 1), got {target_r2!r}.")
    rng = np.random.default_rng(random_state)
    index = pd.period_range(start, periods=t_len, freq="M")

    phi = np.linspace(0.8, 0.4, r) if r > 1 else np.array([0.8])
    transition = np.diag(phi)
    factor_cov = np.eye(r) - transition @ transition.T
    factors = np.zeros((t_len, r))
    factors[0] = rng.standard_normal(r)  # stationary start: unit variance
    shocks = rng.standard_normal((t_len, r)) * np.sqrt(np.diag(factor_cov))
    for t in range(1, t_len):
        factors[t] = transition @ factors[t - 1] + shocks[t]

    loadings = rng.standard_normal((n + 1, r))
    r2 = np.append(rng.uniform(lo, hi, n), float(target_r2))
    common_var = np.sum(loadings**2, axis=1)
    psi = common_var * (1.0 - r2) / r2
    common = factors @ loadings.T
    noise = rng.standard_normal((t_len, n + 1)) * np.sqrt(psi)
    observed = common + noise

    weights = AggregationType.GROWTH_RATE.weights(3, normalize=True)
    latent = observed[:, n]
    gdp = np.full(t_len, np.nan)
    months = index.month.to_numpy()
    for t in range(len(weights) - 1, t_len):
        if months[t] % 3 == 0:
            gdp[t] = float(np.dot(weights, latent[t - np.arange(len(weights))]))

    names = [f"x{i + 1:02d}" for i in range(n)]
    values = observed[:, :n].copy()
    lags = pd.Series(0, index=[*names, "gdp"], dtype=int)
    if ragged_edge:
        drawn = rng.integers(0, 3, size=n)
        lags[names] = drawn
        for i, lag in enumerate(drawn):
            if lag > 0:
                values[t_len - int(lag) :, i] = np.nan
        slots = np.flatnonzero(~np.isnan(gdp))
        gdp[slots[-1]] = np.nan
    data = pd.DataFrame(values, index=index, columns=names)
    data["gdp"] = gdp

    delays = lags.map(_LAG_DELAY_DAYS).astype(int)
    delays["gdp"] = _TARGET_DELAY_DAYS
    truth: dict[str, Any] = {
        "factors": pd.DataFrame(factors, index=index, columns=[f"f{j + 1}" for j in range(r)]),
        "loadings": pd.DataFrame(
            loadings, index=[*names, "gdp"], columns=[f"f{j + 1}" for j in range(r)]
        ),
        "transition": transition,
        "factor_cov": factor_cov,
        "idiosyncratic_var": pd.Series(psi, index=[*names, "gdp"]),
        "gdp_monthly": pd.Series(latent, index=index, name="gdp_monthly"),
        "aggregation_weights": weights.tolist(),
        "publication_lags": lags,
        "delay_days": delays,
    }
    return data, truth
