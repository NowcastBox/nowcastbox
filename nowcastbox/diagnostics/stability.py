r"""Tests for structural breaks in factor loadings (Breitung & Eickmeier, 2011).

For each series :math:`i` with (standardised) observations :math:`x_{it}` and
estimated factors :math:`\hat F_t` (aggregated with the series' weights when the
series is observed at a lower frequency), Breitung & Eickmeier (2011) compare the
restricted regression

.. math:: x_{it} = c_i + \lambda_i' \hat F_t + \varepsilon_{it}

with the unrestricted regression that lets the loadings change from the period
:math:`T^*` on,

.. math:: x_{it} = c_i + \lambda_i' \hat F_t + \delta_i' \hat F_t\, d_t(T^*)
          + \varepsilon_{it}, \qquad d_t(T^*) = 1(t \ge T^*),

and test :math:`H_0: \delta_i = 0` (no break in the loadings of series :math:`i`).
With :math:`S_r`, :math:`S_u` the restricted and unrestricted residual sums of
squares and :math:`n` the number of observations,

.. math::

    LM = n\,\frac{S_r - S_u}{S_r}, \qquad
    W = n\,\frac{S_r - S_u}{S_u}, \qquad
    LR = n \log\frac{S_r}{S_u},

so :math:`W \ge LR \ge LM`. The LM statistic equals :math:`n R^2` of the auxiliary
regression of the restricted residuals on :math:`(1, \hat F_t, \hat F_t d_t)` (the form
used by Breitung & Eickmeier). For a known break date the three statistics are
asymptotically :math:`\chi^2(r)`. Autocorrelated idiosyncratic components are
accommodated by adding ``ar_lags`` lags of the restricted residuals to both
regressions (Breitung & Eickmeier, 2011, section 3).

When the break date is unknown, the statistics are computed for every candidate
:math:`T^* \in [\pi_0 n, (1 - \pi_0) n]` and the supremum is taken (Andrews, 1993).
Its asymptotic null distribution,
:math:`\sup_{\pi \in [\pi_0, 1 - \pi_0]} \|W(\pi) - \pi W(1)\|^2 / (\pi(1 - \pi))`
with :math:`W` an :math:`r`-dimensional Brownian motion, is approximated by Monte Carlo
simulation of discretised Brownian motions (cached per :math:`(r, \pi_0)`), which
reproduces the critical values tabulated by Andrews (1993, table 1) and gives p-values
directly (instead of the response surfaces of Hansen, 1997).

Each series is tested separately; the multiple-testing summary reports the number
of rejections with and without a Holm / Bonferroni / Benjamini-Hochberg correction
and a binomial test of the number of rejections against its expectation
:math:`\alpha N` (exact under independence across series).

References
----------
Andrews, D. W. K. (1993). Tests for parameter instability and structural change with
unknown change point. *Econometrica*, 61(4), 821-856.

Breitung, J., & Eickmeier, S. (2011). Testing for structural breaks in dynamic factor
models. *Journal of Econometrics*, 163(1), 71-84.

Hansen, B. E. (1997). Approximate asymptotic p values for structural-change tests.
*Journal of Business & Economic Statistics*, 15(1), 60-67.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
import scipy.stats
from numpy.typing import NDArray

from nowcastbox._logging import get_logger
from nowcastbox.core.data import FrequencySpec, MixedFrequencyData
from nowcastbox.core.exceptions import NowcastDataError
from nowcastbox.diagnostics._common import (
    FloatArray,
    adjust_pvalues,
    align_factors,
    as_panel,
    check_alpha,
    check_correction,
    check_int,
    pca_factors,
    series_design,
    series_weights,
)

__all__ = [
    "LoadingStabilityResult",
    "andrews_critical_values",
    "loading_stability_test",
    "sup_break_pvalue",
]

logger = get_logger(__name__)

StabilityStatistic = Literal["lm", "wald", "lr"]
_STATISTICS: tuple[str, ...] = ("lm", "wald", "lr")
_N_SIM = 20_000
_N_GRID = 1_000
_SEED = 20_110_163
_CHUNK_ELEMENTS = 4_000_000


# ====================================================================== null distribution
def _check_trim(trim: float) -> float:
    if isinstance(trim, bool) or not isinstance(trim, int | float | np.floating):
        raise ValueError(f"trim must be a number in (0, 0.5), got {trim!r}.")
    value = float(trim)
    if not 0.0 < value < 0.5:
        raise ValueError(f"trim must be in (0, 0.5), got {trim!r}.")
    return value


@functools.lru_cache(maxsize=64)
def _sup_null_draws(k: int, trim: float, n_sim: int, n_grid: int, seed: int) -> FloatArray:
    """Sorted Monte Carlo draws of the sup-Wald limiting distribution."""
    rng = np.random.default_rng(seed)
    grid = np.arange(1, n_grid + 1) / n_grid
    lo = max(math.ceil(trim * n_grid), 1)
    hi = min(math.floor((1.0 - trim) * n_grid), n_grid - 1)
    pi = grid[lo - 1 : hi]
    scale = pi * (1.0 - pi)
    chunk = max(1, _CHUNK_ELEMENTS // (n_grid * k))
    draws: list[FloatArray] = []
    done = 0
    while done < n_sim:
        m = min(chunk, n_sim - done)
        w = np.cumsum(rng.standard_normal((m, n_grid, k)), axis=1) / math.sqrt(n_grid)
        bridge = w[:, lo - 1 : hi, :] - pi[None, :, None] * w[:, -1:, :]
        draws.append(((bridge**2).sum(axis=2) / scale).max(axis=1))
        done += m
    return np.sort(np.concatenate(draws))


def sup_break_pvalue(
    statistic: Any,
    n_breaking: int,
    trim: float = 0.15,
    *,
    n_sim: int = _N_SIM,
    n_grid: int = _N_GRID,
    random_state: int = _SEED,
) -> FloatArray:
    r"""Asymptotic p-value of a sup-LM/Wald/LR break statistic (Andrews, 1993).

    Parameters
    ----------
    statistic : float or array_like
        Observed sup statistic(s); ``NaN`` gives ``NaN``.
    n_breaking : int
        Number of parameters allowed to break (the number of factors :math:`r`).
    trim : float, default 0.15
        Trimming :math:`\pi_0` of the candidate break fractions.
    n_sim : int, default 20000
        Monte Carlo replications of the limiting distribution.
    n_grid : int, default 1000
        Grid points of each simulated Brownian motion.
    random_state : int, default 20110163
        Seed of the (cached) simulation.

    Returns
    -------
    numpy.ndarray
        p-values :math:`(1 + \#\{draws \ge s\}) / (1 + n_{sim})` (shape of ``statistic``).

    Raises
    ------
    ValueError
        On invalid arguments.

    Examples
    --------
    >>> from nowcastbox.diagnostics import sup_break_pvalue
    >>> float(sup_break_pvalue(8.68, 1, 0.15).round(2))
    0.05
    """
    k = check_int(n_breaking, "n_breaking", 1)
    pi0 = _check_trim(trim)
    draws = _sup_null_draws(
        k, pi0, check_int(n_sim, "n_sim", 100), check_int(n_grid, "n_grid", 10), random_state
    )
    s = np.asarray(statistic, dtype=np.float64)
    exceed = draws.size - np.searchsorted(draws, np.nan_to_num(s, nan=0.0), side="left")
    p = (1.0 + exceed) / (1.0 + draws.size)
    return np.where(np.isnan(s), np.nan, p)


def andrews_critical_values(
    n_breaking: int,
    trim: float = 0.15,
    levels: Sequence[float] = (0.10, 0.05, 0.01),
    **kwargs: Any,
) -> pd.Series:
    r"""Asymptotic critical values of the sup-Wald/LM/LR statistics (Andrews, 1993).

    Parameters
    ----------
    n_breaking : int
        Number of parameters allowed to break.
    trim : float, default 0.15
        Trimming :math:`\pi_0`.
    levels : sequence of float, default (0.10, 0.05, 0.01)
        Significance levels.
    **kwargs
        ``n_sim``, ``n_grid``, ``random_state`` of :func:`sup_break_pvalue`.

    Returns
    -------
    pandas.Series
        Critical values indexed by level.

    Notes
    -----
    The supremum is taken over a grid of ``n_grid`` points, so the simulated values
    are slightly below the continuous-time values of Andrews (1993, table 1) (about
    0.1-0.2 at the 5 % level for one or two breaking parameters with 15 % trimming),
    a discreteness that matches the finite set of candidate dates of the tests.

    Raises
    ------
    ValueError
        On invalid arguments.

    Examples
    --------
    >>> from nowcastbox.diagnostics import andrews_critical_values
    >>> andrews_critical_values(1).round(1).tolist()
    [7.0, 8.5, 12.2]
    """
    k = check_int(n_breaking, "n_breaking", 1)
    pi0 = _check_trim(trim)
    n_sim = check_int(kwargs.pop("n_sim", _N_SIM), "n_sim", 100)
    n_grid = check_int(kwargs.pop("n_grid", _N_GRID), "n_grid", 10)
    seed = kwargs.pop("random_state", _SEED)
    if kwargs:
        raise TypeError(f"Unknown arguments {sorted(kwargs)}.")
    draws = _sup_null_draws(k, pi0, n_sim, n_grid, seed)
    lv = [check_alpha(a) for a in levels]
    values = [float(np.quantile(draws, 1.0 - a)) for a in lv]
    return pd.Series(values, index=pd.Index(lv, name="level"), name="critical_value")


# ====================================================================== regressions
def _reverse_cumsum(a: FloatArray) -> FloatArray:
    return np.cumsum(a[::-1], axis=0)[::-1]


def _ssr(y: FloatArray, X: FloatArray) -> tuple[float, FloatArray]:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return float(resid @ resid), resid


def break_ssr(
    y: FloatArray, W: FloatArray, Fb: FloatArray, starts: Any
) -> tuple[float, FloatArray]:
    r"""Restricted and unrestricted residual sums of squares for candidate breaks.

    Parameters
    ----------
    y : numpy.ndarray, shape (n,)
        Dependent variable.
    W : numpy.ndarray, shape (n, m)
        Regressors of the restricted model (constant, factors, lags).
    Fb : numpy.ndarray, shape (n, r)
        Regressors whose coefficients may break.
    starts : array_like of int
        Candidate first rows of the post-break regime (``0 < k < n``).

    Returns
    -------
    ssr_restricted : float
        :math:`S_r`.
    ssr_unrestricted : numpy.ndarray
        :math:`S_u(k)` for every ``k`` in ``starts``.

    Notes
    -----
    The unrestricted regression adds :math:`F_t 1(t \ge k)` to ``W``. Its residual sum
    of squares is computed for all ``k`` at once from reverse cumulative cross-products
    of the restricted residuals :math:`e` (orthogonal to ``W``):
    :math:`S_u(k) = S_r - b_k' X_k' e` with :math:`X_k' e = (0, \sum_{t \ge k} F_t e_t)`.

    Examples
    --------
    >>> import numpy as np
    >>> from nowcastbox.diagnostics.stability import break_ssr
    >>> f = np.arange(1.0, 9.0)[:, None]
    >>> y = np.where(np.arange(8) < 4, 1.0, 3.0) * f[:, 0]
    >>> W = np.column_stack([np.ones(8), f])
    >>> sr, su = break_ssr(y, W, f, [4])
    >>> sr > 0, bool(abs(su[0]) < 1e-9)
    (True, True)
    """
    ks = np.asarray(starts, dtype=np.int64)
    ssr_r, e = _ssr(y, W)
    m = W.shape[1]
    r = Fb.shape[1]
    WtW = W.T @ W
    G = _reverse_cumsum(Fb[:, :, None] * Fb[:, None, :])[ks]
    H = _reverse_cumsum(W[:, :, None] * Fb[:, None, :])[ks]
    g = _reverse_cumsum(Fb * e[:, None])[ks]
    xtx = np.empty((ks.size, m + r, m + r))
    xtx[:, :m, :m] = WtW
    xtx[:, :m, m:] = H
    xtx[:, m:, :m] = np.transpose(H, (0, 2, 1))
    xtx[:, m:, m:] = G
    xte = np.concatenate([np.zeros((ks.size, m)), g], axis=1)
    b = np.einsum("kij,kj->ki", np.linalg.pinv(xtx, hermitian=True), xte)
    explained = np.einsum("ki,ki->k", b, xte)
    ssr_u = np.clip(ssr_r - explained, ssr_r * 1e-12, ssr_r)
    return ssr_r, ssr_u


def _statistics(n: int, ssr_r: float, ssr_u: FloatArray) -> dict[str, FloatArray]:
    return {
        "lm": n * (ssr_r - ssr_u) / ssr_r,
        "wald": n * (ssr_r - ssr_u) / ssr_u,
        "lr": n * np.log(ssr_r / ssr_u),
    }


def _lagged_design(
    y: FloatArray, F: FloatArray, ar_lags: int
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Restricted regressors ``[1, F, e_{t-1..p}]``, dropping the first ``p`` rows."""
    n = y.size
    base = np.column_stack([np.ones(n), F])
    if ar_lags == 0:
        return y, base, F
    _, e = _ssr(y, base)
    lags = np.column_stack([e[ar_lags - j - 1 : n - j - 1] for j in range(ar_lags)])
    return y[ar_lags:], np.column_stack([base[ar_lags:], lags]), F[ar_lags:]


@dataclass(frozen=True)
class _SeriesTest:
    n_obs: int
    break_row: int | None
    stats: dict[str, float] = field(default_factory=dict)
    note: str = ""


def _candidate_rows(n: int, r: int, trim: float) -> NDArray[np.int64]:
    lo = max(math.ceil(trim * n), r + 1)
    hi = min(math.floor((1.0 - trim) * n), n - r - 1)
    return np.arange(lo, hi + 1)


def _test_one_series(
    index: pd.PeriodIndex,
    y: FloatArray,
    F: FloatArray,
    *,
    break_period: pd.Period | None,
    trim: float,
    ar_lags: int,
) -> tuple[_SeriesTest, pd.PeriodIndex]:
    r = F.shape[1]
    idx = index[ar_lags:]
    if y.size < 2 * r + ar_lags + 3:
        return _SeriesTest(max(y.size - ar_lags, 0), None, note="too few observations"), idx
    y2, W, Fb = _lagged_design(y, F, ar_lags)
    n = y2.size
    if break_period is not None:
        k = int(idx.searchsorted(break_period))
        if min(k, n - k) <= r:
            return _SeriesTest(n, None, note="break date too close to sample ends"), idx
        rows = np.array([k])
    else:
        rows = _candidate_rows(n, r, trim)
        if rows.size == 0:
            return _SeriesTest(n, None, note="too few observations for trimming"), idx
    ssr_r, ssr_u = break_ssr(y2, W, Fb, rows)
    if ssr_r <= 1e-12 * float(y2 @ y2):
        return _SeriesTest(n, None, note="perfect fit"), idx
    stats = _statistics(n, ssr_r, ssr_u)
    best = int(np.argmax(stats["wald"]))
    return _SeriesTest(n, int(rows[best]), {k: float(v[best]) for k, v in stats.items()}), idx


# ====================================================================== result
@dataclass(frozen=True, eq=False)
class LoadingStabilityResult:
    r"""Breitung-Eickmeier loading-stability tests, one row per series.

    Parameters
    ----------
    table : pandas.DataFrame
        Indexed by series: ``frequency``, ``n_obs``, ``break_period`` (first base
        period of the post-break regime; the estimated date for sup tests),
        ``lm_stat``/``lm_pvalue``, ``wald_stat``/``wald_pvalue``,
        ``lr_stat``/``lr_pvalue``, ``df`` (number of factors), ``pvalue_adj``
        (adjusted p-value of ``statistic``), ``reject`` and ``note``.
    test : {"known", "sup"}
        Known break date or sup statistic over a trimmed range.
    statistic : {"lm", "wald", "lr"}
        Statistic used for ``pvalue_adj`` and ``reject``.
    alpha : float
        Significance level.
    correction : str
        Multiple-testing correction.
    trim : float
        Trimming of the sup tests.
    ar_lags : int
        Lags of the residuals in the test regressions.
    break_date : pandas.Period or None
        Known break date (``None`` for sup tests).

    Examples
    --------
    >>> from nowcastbox.models.two_step import simulate_two_step_example
    >>> from nowcastbox.diagnostics import loading_stability_test
    >>> data = simulate_two_step_example(random_state=0).drop(["gdp"])
    >>> res = loading_stability_test(data, n_factors=1, break_date="2007-01")
    >>> res.table.shape[0], res.test
    (10, 'known')
    """

    table: pd.DataFrame
    test: str
    statistic: str
    alpha: float
    correction: str
    trim: float
    ar_lags: int
    break_date: pd.Period | None = None

    @property
    def rejected(self) -> list[str]:
        """Series whose loadings are unstable after the multiple-testing correction."""
        return [str(s) for s in self.table.index[self.table["reject"].to_numpy(bool)]]

    def to_frame(self) -> pd.DataFrame:
        """Copy of :attr:`table`.

        Returns
        -------
        pandas.DataFrame
            One row per series.

        Examples
        --------
        >>> res.to_frame().columns[:3].tolist()  # doctest: +SKIP
        ['frequency', 'n_obs', 'break_period']
        """
        return self.table.copy()

    def multiple_testing(self) -> pd.DataFrame:
        r"""Multiple-testing summary for each statistic.

        Returns
        -------
        pandas.DataFrame
            Indexed by statistic: ``n_tests``, ``n_reject`` (raw p-values below
            ``alpha``), ``share_reject``, ``expected_reject`` (:math:`\alpha N`),
            ``binomial_pvalue`` (:math:`P(X \ge n_{reject})`, :math:`X \sim
            Bin(N, \alpha)`) and ``n_reject_adjusted`` (after :attr:`correction`).

        Examples
        --------
        >>> res.multiple_testing().loc["lm", "n_tests"]  # doctest: +SKIP
        10
        """
        rows = {}
        for stat in _STATISTICS:
            p = self.table[f"{stat}_pvalue"].to_numpy(dtype=np.float64)
            ok = np.isfinite(p)
            n = int(ok.sum())
            k = int((p[ok] < self.alpha).sum())
            adj = adjust_pvalues(p, self.correction)
            rows[stat] = {
                "n_tests": n,
                "n_reject": k,
                "share_reject": k / n if n else np.nan,
                "expected_reject": self.alpha * n,
                "binomial_pvalue": float(scipy.stats.binom.sf(k - 1, n, self.alpha))
                if n
                else np.nan,
                "n_reject_adjusted": int((adj[ok] < self.alpha).sum()),
            }
        out = pd.DataFrame.from_dict(rows, orient="index")
        out.index.name = "statistic"
        return out

    def summary(self) -> str:
        """Text summary.

        Returns
        -------
        str
            Test description, multiple-testing table and rejected series.

        Examples
        --------
        >>> print(res.summary())  # doctest: +SKIP
        """
        kind = (
            f"known break date {self.break_date}"
            if self.test == "known"
            else f"unknown break date (sup over [{self.trim:.2f}, {1 - self.trim:.2f}])"
        )
        lines = [
            "Loading stability (Breitung & Eickmeier, 2011)",
            f"  Test: {kind}; statistic for decisions: {self.statistic.upper()}; "
            f"alpha={self.alpha}; correction={self.correction}",
            self.multiple_testing().to_string(float_format=lambda v: f"{v:.4g}"),
        ]
        rejected = self.rejected
        shown = ", ".join(rejected[:10]) + (" ..." if len(rejected) > 10 else "")
        lines.append(f"  Unstable loadings ({len(rejected)}): {shown or '-'}")
        return "\n".join(lines)


# ====================================================================== entry point
def _to_base_period(value: Any, index: pd.PeriodIndex) -> pd.Period:
    if isinstance(value, pd.Timestamp):
        period = pd.Period(value, freq=index.freqstr)
    else:
        period = value if isinstance(value, pd.Period) else pd.Period(str(value))
        period = period.asfreq(index.freqstr, how="start")
    if not index[0] < period <= index[-1]:
        raise ValueError(
            f"break_date {value!r} must lie inside the sample ({index[0]} - {index[-1]})."
        )
    return period


def _pvalues(stat: FloatArray, r: int, test: str, trim: float) -> FloatArray:
    if test == "known":
        return np.asarray(scipy.stats.chi2.sf(stat, r), dtype=np.float64)
    return sup_break_pvalue(stat, r, trim)


def _resolve_factors(
    panel: MixedFrequencyData, factors: pd.DataFrame | None, n_factors: int | None
) -> pd.DataFrame:
    if factors is None:
        if n_factors is None:
            raise ValueError("Pass either factors or n_factors (principal components).")
        return pca_factors(panel, n_factors)
    if n_factors is not None:
        raise ValueError("Pass factors or n_factors, not both.")
    return align_factors(factors, panel.index)


def _check_options(statistic: str, ar_lags: int, trim: float) -> tuple[int, float]:
    if statistic not in _STATISTICS:
        raise ValueError(f"statistic must be one of {_STATISTICS}, got {statistic!r}.")
    return check_int(ar_lags, "ar_lags", 0), _check_trim(trim)


def loading_stability_test(
    data: MixedFrequencyData | pd.DataFrame,
    factors: pd.DataFrame | None = None,
    *,
    n_factors: int | None = None,
    break_date: Any = None,
    trim: float = 0.15,
    ar_lags: int = 0,
    statistic: StabilityStatistic = "lm",
    alpha: float = 0.05,
    correction: str = "holm",
    series: Sequence[str] | None = None,
    weights: Mapping[str, Any] | None = None,
    frequency: FrequencySpec | None = None,
) -> LoadingStabilityResult:
    r"""Breitung & Eickmeier (2011) tests for breaks in the factor loadings.

    Parameters
    ----------
    data : MixedFrequencyData or pandas.DataFrame
        Panel on a base-frequency PeriodIndex (each series is standardised).
    factors : pandas.DataFrame, optional
        Estimated factors on the base grid (e.g. ``results.factors``). When omitted,
        ``n_factors`` principal components of the balanced base-frequency panel are
        used, as in Breitung & Eickmeier (2011).
    n_factors : int, optional
        Number of principal components (only without ``factors``).
    break_date : period-like, optional
        First period of the post-break regime (``"2008-09"``, ``"2008Q3"``,
        :class:`pandas.Period`). ``None`` computes sup statistics over the trimmed
        range of candidate dates (unknown break date).
    trim : float, default 0.15
        Trimming :math:`\pi_0` of the sup tests.
    ar_lags : int, default 0
        Lags of the restricted residuals added to the test regressions (serially
        correlated idiosyncratic components).
    statistic : {"lm", "wald", "lr"}, default "lm"
        Statistic used for the multiple-testing decision ``reject``.
    alpha : float, default 0.05
        Significance level.
    correction : {"holm", "bonferroni", "fdr_bh", "none"}, default "holm"
        Multiple-testing correction across series.
    series : sequence of str, optional
        Series to test (default: every series of ``data``).
    weights : mapping, optional
        Aggregation weights per series (most recent first) linking lower-frequency
        series to the factors; defaults follow the series' ``aggregation`` metadata
        (Mariano-Murasawa for growth rates).
    frequency : frequency specification, optional
        Frequencies of a DataFrame ``data``.

    Returns
    -------
    LoadingStabilityResult
        Per-series statistics and p-values.

    Raises
    ------
    ValueError
        On invalid options or a break date outside the sample.
    NowcastDataError
        On unusable data or factors.

    Examples
    --------
    >>> from nowcastbox.models.two_step import simulate_two_step_example
    >>> from nowcastbox.diagnostics import loading_stability_test
    >>> data = simulate_two_step_example(random_state=1)
    >>> res = loading_stability_test(data, n_factors=1)
    >>> res.test, len(res.table)
    ('sup', 11)
    """
    ar, pi0 = _check_options(statistic, ar_lags, trim)
    check_correction(correction)
    alpha = check_alpha(alpha)
    panel = as_panel(data, frequency)
    names = list(panel.columns if series is None else series)
    unknown = [s for s in names if s not in panel.columns]
    if unknown:
        raise NowcastDataError(f"Unknown series {unknown}.")
    F = _resolve_factors(panel, factors, n_factors)
    r = F.shape[1]
    period = None if break_date is None else _to_base_period(break_date, panel.index)
    test = "sup" if period is None else "known"
    w = series_weights(panel, names, weights)
    rows: list[dict[str, Any]] = []
    for name in names:
        idx, y, X = series_design(panel, F, name, w[name])
        res, used = _test_one_series(idx, y, X, break_period=period, trim=pi0, ar_lags=ar)
        rows.append(
            {
                "series": name,
                "frequency": panel.metadata[name].frequency.value,
                "n_obs": res.n_obs,
                "break_period": str(used[res.break_row]) if res.break_row is not None else None,
                **{f"{s}_stat": res.stats.get(s, np.nan) for s in _STATISTICS},
                "note": res.note,
            }
        )
    table = pd.DataFrame(rows).set_index("series")
    for s in _STATISTICS:
        table[f"{s}_pvalue"] = _pvalues(table[f"{s}_stat"].to_numpy(float), r, test, pi0)
    table["df"] = r
    table["pvalue_adj"] = adjust_pvalues(table[f"{statistic}_pvalue"].to_numpy(), correction)
    table["reject"] = table["pvalue_adj"].to_numpy() < alpha
    order = ["frequency", "n_obs", "break_period"]
    order += [f"{s}_{k}" for s in _STATISTICS for k in ("stat", "pvalue")]
    table = table[[*order, "df", "pvalue_adj", "reject", "note"]]
    logger.debug("Loading stability: %d series, %d rejections", len(table), table["reject"].sum())
    return LoadingStabilityResult(
        table=table,
        test=test,
        statistic=statistic,
        alpha=alpha,
        correction=correction,
        trim=pi0,
        ar_lags=ar,
        break_date=period,
    )
