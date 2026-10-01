r"""Simulation of mixed-frequency dynamic factor models with known parameters.

Two data-generating processes, both written from the textbook DFM (Giannone, Reichlin &
Small, 2008; Bańbura & Modugno, 2014) with the Mariano & Murasawa (2003) link between a
quarterly growth rate and its latent higher-frequency counterpart:

* :func:`dfm` - monthly indicators plus a quarterly target on a monthly grid (the
  generator of :func:`~nowcastbox.datasets.load_simulated_dfm`):

  .. math::

      f_t = A f_{t-1} + u_t,\quad x_{it} = \lambda_i' f_t + e_{it},\quad
      y^Q_t = \tfrac13 (y^*_t + 2 y^*_{t-1} + 3 y^*_{t-2} + 2 y^*_{t-3} + y^*_{t-4});

* :func:`weekly_dfm` - one weekly AR(1) factor with weekly series, monthly calendar
  averages (4 or 5 weeks) and quarterly GDP growth whose Mariano-Murasawa weights follow
  the calendar (12 to 14 weeks per quarter), on a weekly grid (innovation I1).

Both return a :class:`SimulatedDFM`: the panel as a
:class:`~nowcastbox.core.data.MixedFrequencyData` (frequencies, release delays and
aggregation metadata set) and the true parameters/latent quantities, for tests,
tutorials and Monte Carlo studies.

References
----------
Bańbura, M., & Modugno, M. (2014). Maximum likelihood estimation of factor models on
datasets with arbitrary pattern of missing data. *Journal of Applied Econometrics*,
29(1), 133-160.

Mariano, R. S., & Murasawa, Y. (2003). A new coincident index of business cycles based
on monthly and quarterly series. *Journal of Applied Econometrics*, 18(4), 427-443.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from nowcastbox.core.data import MixedFrequencyData
from nowcastbox.core.frequency import is_period_end
from nowcastbox.datasets._simulated import simulate_mixed_frequency_dfm
from nowcastbox.preprocessing.aggregation import calendar_aggregation

__all__ = ["SimulatedDFM", "dfm", "weekly_dfm"]

RandomState = int | np.random.Generator | None

_WEEKLY_DELAYS = {"W": 3, "M": 20, "Q": 45}


@dataclass(frozen=True)
class SimulatedDFM:
    """A simulated panel and the parameters that generated it.

    Attributes
    ----------
    data : MixedFrequencyData
        Simulated panel (target ``"gdp"``), with frequencies and release delays.
    truth : dict
        True parameters and latent quantities (see :func:`dfm` / :func:`weekly_dfm`).

    Examples
    --------
    >>> from nowcastbox.simulate import dfm
    >>> sim = dfm(n_series=5, n_factors=1, n_periods=48)
    >>> data, truth = sim
    >>> data.shape, sorted(truth)[:2]
    ((48, 6), ['aggregation_weights', 'delay_days'])
    """

    data: MixedFrequencyData
    truth: dict[str, Any] = field(default_factory=dict)

    def __iter__(self) -> Any:
        """Unpack as ``data, truth``."""
        return iter((self.data, self.truth))


def dfm(
    n_series: int = 20,
    n_factors: int = 2,
    n_periods: int = 240,
    *,
    start: str = "2000-01",
    r2_range: tuple[float, float] = (0.3, 0.8),
    target_r2: float = 0.7,
    ragged_edge: bool = True,
    random_state: RandomState = 0,
) -> SimulatedDFM:
    """Simulate monthly indicators and a quarterly target from a known DFM.

    The factors follow a diagonal VAR(1) with unit stationary variance; the
    common-variance shares of the monthly series are drawn on ``r2_range``; the
    quarterly target aggregates its latent monthly counterpart with the
    Mariano-Murasawa weights. With ``ragged_edge`` every monthly series gets a
    publication lag of 0, 1 or 2 months (release delays of 5, 35 or 65 days) and the
    last quarter of the target is not observed (delay 45 days).

    Parameters
    ----------
    n_series : int, default 20
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
        Impose publication lags and an unobserved last quarter.
    random_state : int, numpy.random.Generator or None, default 0
        Seed (the default makes the output deterministic).

    Returns
    -------
    SimulatedDFM
        ``data`` (columns ``x01``... monthly and ``gdp`` quarterly) and ``truth`` with
        ``factors``, ``loadings``, ``transition``, ``factor_cov``,
        ``idiosyncratic_var``, ``gdp_monthly``, ``aggregation_weights``,
        ``publication_lags`` and ``delay_days``.

    Raises
    ------
    ValueError
        On invalid sizes or shares.

    Examples
    --------
    >>> import nowcastbox as nb
    >>> sim = nb.simulate.dfm(n_series=8, n_factors=1, n_periods=120, random_state=1)
    >>> res = nb.MixedFreqDFM(n_factors=1, max_iter=30).fit(sim.data, "gdp")
    >>> bool(abs(res.factors["f1"].corr(sim.truth["factors"]["f1"])) > 0.9)
    True
    """
    frame, truth = simulate_mixed_frequency_dfm(
        n_monthly=n_series,
        n_factors=n_factors,
        n_periods=n_periods,
        start=start,
        r2_range=r2_range,
        target_r2=target_r2,
        ragged_edge=ragged_edge,
        random_state=random_state,
    )
    frequencies = {name: ("Q" if name == "gdp" else "M") for name in frame.columns}
    data = MixedFrequencyData(
        frame,
        frequencies,
        release_delays={k: int(v) for k, v in truth["delay_days"].items()},
        aggregations={"gdp": "mariano_murasawa"},
    )
    return SimulatedDFM(data, truth)


def _check_count(value: object, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int | np.integer) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}.")
    return int(value)


def weekly_dfm(
    n_weeks: int = 520,
    n_weekly: int = 4,
    n_monthly: int = 5,
    *,
    start: str = "2010-01-03",
    phi: float = 0.9,
    noise: float = 0.6,
    ragged_edge: bool = True,
    random_state: RandomState = 0,
) -> SimulatedDFM:
    r"""Simulate weekly, monthly and quarterly series from one weekly factor (I1).

    The factor is a weekly AR(1) with unit variance. Weekly series load on it directly,
    monthly series are calendar **averages** of a latent weekly series (4 or 5 weeks)
    and the quarterly target ``gdp`` is the growth rate of the quarterly average of a
    latent weekly log level (calendar Mariano-Murasawa weights over 12 to 14 plus 12 to
    14 weeks, :func:`~nowcastbox.preprocessing.aggregation.calendar_aggregation`). White
    measurement noise is added at the observation dates; monthly and quarterly values
    are stored in the last week ending in their period.

    Parameters
    ----------
    n_weeks : int, default 520
        Number of weeks (at least 60).
    n_weekly, n_monthly : int, default 4 and 5
        Number of weekly and monthly series.
    start : str, default "2010-01-03"
        A day of the first week.
    phi : float, default 0.9
        AR(1) coefficient of the weekly factor, within (-1, 1).
    noise : float, default 0.6
        Standard deviation of the measurement noise of the weekly series (0.3x for the
        monthly series, 0.2 for the target).
    ragged_edge : bool, default True
        Leave the last quarter of the target and the last two weeks of the monthly
        series unobserved (release delays: 3 days weekly, 20 monthly, 45 quarterly).
    random_state : int, numpy.random.Generator or None, default 0
        Seed.

    Returns
    -------
    SimulatedDFM
        ``data`` on a weekly grid (columns ``w0``..., ``m0``..., ``gdp``) and ``truth``
        with ``factor`` (Series), ``common`` (DataFrame of common components at the
        observation dates), ``loadings`` (Series) and ``phi``.

    Raises
    ------
    ValueError
        On invalid sizes, ``phi`` or ``noise``.

    Examples
    --------
    >>> from nowcastbox.simulate import weekly_dfm
    >>> sim = weekly_dfm(n_weeks=104, random_state=1)
    >>> sim.data.base_frequency.value, sim.data.frequencies["gdp"].value
    ('W', 'Q')
    """
    n = _check_count(n_weeks, "n_weeks", 60)
    k_w = _check_count(n_weekly, "n_weekly", 1)
    k_m = _check_count(n_monthly, "n_monthly", 0)
    if not -1.0 < float(phi) < 1.0:
        raise ValueError(f"phi must be in (-1, 1), got {phi!r}.")
    if not float(noise) >= 0.0:
        raise ValueError(f"noise must be non-negative, got {noise!r}.")
    rng = np.random.default_rng(random_state)
    idx = pd.period_range(start, periods=n, freq="W")
    f = np.zeros(n)
    f[0] = rng.standard_normal()
    for t in range(1, n):
        f[t] = phi * f[t - 1] + np.sqrt(1.0 - phi**2) * rng.standard_normal()
    factor = pd.Series(f, index=idx, name="factor")
    data: dict[str, np.ndarray] = {}
    common: dict[str, np.ndarray] = {}
    loadings: dict[str, float] = {}
    for j in range(k_w):
        lam = float(rng.uniform(0.6, 1.2))
        loadings[f"w{j}"] = lam
        common[f"w{j}"] = lam * f
        data[f"w{j}"] = lam * f + noise * rng.standard_normal(n)
    month_end = is_period_end(idx, "M")
    monthly = calendar_aggregation("W", "M", "average").apply(factor).to_numpy()
    for j in range(k_m):
        lam = float(rng.uniform(0.6, 1.2))
        loadings[f"m{j}"] = lam
        c = np.where(month_end, lam * monthly, np.nan)
        common[f"m{j}"] = c
        data[f"m{j}"] = c + 0.3 * noise * rng.standard_normal(n)
    quarter_end = is_period_end(idx, "Q")
    quarterly = calendar_aggregation("W", "Q").apply(factor).to_numpy()
    loadings["gdp"] = 0.8
    common["gdp"] = np.where(quarter_end, 0.8 * quarterly, np.nan)
    data["gdp"] = common["gdp"] + 0.2 * rng.standard_normal(n)
    frame = pd.DataFrame(data, index=idx)
    if ragged_edge:
        last = np.flatnonzero(quarter_end & np.isfinite(frame["gdp"].to_numpy()))[-1]
        frame.loc[frame.index[last], "gdp"] = np.nan
        monthly_cols = [c for c in frame.columns if c.startswith("m")]
        frame.loc[frame.index[-2:], monthly_cols] = np.nan
    frequencies = {c: ("W" if c.startswith("w") else "M") for c in frame.columns}
    frequencies["gdp"] = "Q"
    panel = MixedFrequencyData(
        frame,
        frequencies,
        aggregations={**{f"m{j}": "average" for j in range(k_m)}, "gdp": "mariano_murasawa"},
        release_delays={c: _WEEKLY_DELAYS[frequencies[c]] for c in frame.columns},
    )
    truth: dict[str, Any] = {
        "factor": factor,
        "common": pd.DataFrame(common, index=idx),
        "loadings": pd.Series(loadings, name="loading"),
        "phi": float(phi),
    }
    return SimulatedDFM(panel, truth)
