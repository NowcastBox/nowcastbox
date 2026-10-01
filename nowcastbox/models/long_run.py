r"""Time-varying long-run mean of the target (innovation I4).

Antolin-Diaz, Drechsel & Petrella (2017) let the long-run growth rate of GDP drift as a
random walk, so that a decline in trend growth (e.g. the post-2000 slowdown in the US or
in Brazil) does not bias the nowcast towards the full-sample mean. In
:class:`~nowcastbox.models.MixedFreqDFM` (``long_run_mean="time_varying"``) the target
and, optionally, selected series get an extra additive component

.. math::

    x_{i,t} = \dots + \beta_i \mu_t, \qquad \mu_t = \mu_{t-1} + \zeta_t, \qquad
    \zeta_t \sim N(0, \sigma^2_\zeta),

with :math:`\beta_{\text{target}} = 1` (normalisation: :math:`\mu_t` is the long-run mean
of the target in standardised units) and :math:`\beta_i` estimated for the other
series. :math:`\mu_t` is one extra state of the EM model; :math:`\sigma^2_\zeta` is
estimated in the M-step (:func:`random_walk_variance`) or fixed by the user
(``long_run_variance``). Fixing it is the frequentist counterpart of the tight prior of
Antolin-Diaz et al. (2017) and avoids the *pile-up* of the maximum-likelihood estimate
at zero when the trend variance is small relative to the noise (Stock & Watson, 1998).

This module holds the helpers that do not depend on the EM internals.

References
----------
Antolin-Diaz, J., Drechsel, T., & Petrella, I. (2017). Tracking the slowdown in
long-run GDP growth. *Review of Economics and Statistics*, 99(2), 343-356.

Stock, J. H., & Watson, M. W. (1998). Median unbiased estimation of coefficient variance
in a time-varying parameter model. *Journal of the American Statistical Association*,
93(441), 349-358.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

__all__ = [
    "LONG_RUN_INITIAL_VARIANCE",
    "LONG_RUN_MODES",
    "LONG_RUN_START_VARIANCE",
    "long_run_frame",
    "long_run_restrictions",
    "random_walk_variance",
    "resolve_long_run_series",
]

FloatArray = NDArray[np.float64]

LONG_RUN_MODES = ("constant", "time_varying")
"""Accepted values of ``MixedFreqDFM(long_run_mean=...)``."""

LONG_RUN_INITIAL_VARIANCE = 10.0
"""Variance of the (proper, nearly uninformative) prior of the first long-run mean state."""

LONG_RUN_START_VARIANCE = 1e-3
"""Starting value of the long-run mean innovation variance (standardised units)."""


def random_walk_variance(
    s11: float, s10: float, s00: float, n_pairs: int, floor: float = 1e-6
) -> float:
    r"""M-step of the innovation variance of a random walk.

    :math:`\hat\sigma^2 = \sum_t E[(\mu_t - \mu_{t-1})^2 \mid Y] / (n - 1)
    = (S_{11} - 2 S_{10} + S_{00}) / (n - 1)` (Shumway & Stoffer, 1982, with the
    transition coefficient fixed at one).

    Parameters
    ----------
    s11, s10, s00 : float
        Smoothed moment sums :math:`\sum E[\mu_t^2]`, :math:`\sum E[\mu_t\mu_{t-1}]`,
        :math:`\sum E[\mu_{t-1}^2]` over ``t = 2..n``.
    n_pairs : int
        Number of transitions ``n - 1`` (positive).
    floor : float, default 1e-6
        Lower bound of the estimate.

    Returns
    -------
    float
        Estimated variance (at least ``floor``).

    Raises
    ------
    ValueError
        If ``n_pairs`` is not positive.

    Examples
    --------
    >>> from nowcastbox.models.long_run import random_walk_variance
    >>> random_walk_variance(5.0, 4.5, 4.5, 2)
    0.25
    """
    if n_pairs <= 0:
        raise ValueError(f"n_pairs must be positive, got {n_pairs}")
    value = (float(s11) - 2.0 * float(s10) + float(s00)) / n_pairs
    return max(value, float(floor))


def long_run_restrictions(
    restriction: FloatArray, rhs: FloatArray, *, fixed: bool
) -> tuple[FloatArray, FloatArray]:
    """Append the long-run loading to linear loading restrictions ``R beta = q``.

    The long-run loading is the last coefficient. It is free (a zero column is added)
    or, for the normalising series (``fixed=True``), restricted to one.

    Parameters
    ----------
    restriction : numpy.ndarray, shape (c, k)
        Restrictions on the factor loadings.
    rhs : numpy.ndarray, shape (c,)
        Right-hand side.
    fixed : bool
        Whether the long-run loading is fixed at one.

    Returns
    -------
    restriction : numpy.ndarray, shape (c or c + 1, k + 1)
        Extended restriction matrix.
    rhs : numpy.ndarray
        Extended right-hand side.

    Examples
    --------
    >>> import numpy as np
    >>> from nowcastbox.models.long_run import long_run_restrictions
    >>> R, q = long_run_restrictions(np.zeros((0, 2)), np.zeros(0), fixed=True)
    >>> R.tolist(), q.tolist()
    ([[0.0, 0.0, 1.0]], [1.0])
    """
    R = np.hstack([restriction, np.zeros((restriction.shape[0], 1))])
    q = np.asarray(rhs, dtype=np.float64)
    if not fixed:
        return R, q
    unit = np.zeros((1, R.shape[1]))
    unit[0, -1] = 1.0
    return np.vstack([R, unit]), np.append(q, 1.0)


def resolve_long_run_series(
    mode: str, series: Sequence[str] | str | None, columns: Sequence[str], target: str
) -> tuple[int, ...]:
    """Positions of the series loading on the long-run mean (target first).

    Parameters
    ----------
    mode : {"constant", "time_varying"}
        Long-run mean specification.
    series : str, sequence of str or None
        Additional series with a long-run mean component (besides the target).
    columns : sequence of str
        Panel columns (series order of the model).
    target : str
        Target series.

    Returns
    -------
    tuple of int
        Empty for ``"constant"``; otherwise the target position followed by the
        positions of ``series``.

    Raises
    ------
    ValueError
        On an unknown mode, unknown / duplicated series, or ``series`` given with
        ``mode="constant"``.

    Examples
    --------
    >>> from nowcastbox.models.long_run import resolve_long_run_series
    >>> resolve_long_run_series("time_varying", ["c"], ["a", "gdp", "c"], "gdp")
    (1, 2)
    >>> resolve_long_run_series("constant", None, ["a", "gdp"], "gdp")
    ()
    """
    if mode not in LONG_RUN_MODES:
        raise ValueError(f"long_run_mean must be one of {LONG_RUN_MODES}, got {mode!r}.")
    extra = [series] if isinstance(series, str) else list(series or [])
    if mode == "constant":
        if extra:
            raise ValueError("long_run_series requires long_run_mean='time_varying'.")
        return ()
    names = list(columns)
    unknown = [s for s in extra if s not in names]
    if unknown:
        raise ValueError(f"long_run_series contains unknown series {unknown}.")
    ordered = [target, *[s for s in extra if s != target]]
    if len(set(ordered)) != len(ordered):
        raise ValueError("long_run_series contains duplicated series.")
    return tuple(names.index(s) for s in ordered)


def long_run_frame(
    path: FloatArray,
    variance: FloatArray,
    loadings: FloatArray,
    means: FloatArray,
    scales: FloatArray,
    index: pd.PeriodIndex,
    names: Sequence[str],
) -> pd.DataFrame:
    r"""Long-run means of the selected series in original units.

    Column ``name`` holds :math:`m_i + s_i \beta_i \hat\mu_{t|n}` and column
    ``name + "_std"`` its smoothed standard deviation
    :math:`s_i |\beta_i| \sqrt{V_t}` (``m_i``, ``s_i``: standardisation mean and
    standard deviation).

    Parameters
    ----------
    path : numpy.ndarray, shape (n_periods,)
        Smoothed long-run mean state (standardised units).
    variance : numpy.ndarray, shape (n_periods,)
        Its smoothed variance.
    loadings : numpy.ndarray, shape (k,)
        Loadings :math:`\beta_i` of the ``k`` series.
    means, scales : numpy.ndarray, shape (k,)
        Standardisation statistics of the series.
    index : pandas.PeriodIndex
        Base grid.
    names : sequence of str
        Series names.

    Returns
    -------
    pandas.DataFrame
        ``2k`` columns indexed by ``index``.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from nowcastbox.models.long_run import long_run_frame
    >>> idx = pd.period_range("2020-01", periods=2, freq="M")
    >>> f = long_run_frame(
    ...     np.array([0.0, 1.0]),
    ...     np.array([0.25, 0.25]),
    ...     np.array([1.0]),
    ...     np.array([2.0]),
    ...     np.array([3.0]),
    ...     idx,
    ...     ["gdp"],
    ... )
    >>> f["gdp"].tolist(), f["gdp_std"].tolist()
    ([2.0, 5.0], [1.5, 1.5])
    """
    columns: dict[str, FloatArray] = {}
    sd = np.sqrt(np.maximum(variance, 0.0))
    for k, name in enumerate(names):
        columns[name] = means[k] + scales[k] * loadings[k] * path
        columns[f"{name}_std"] = scales[k] * abs(loadings[k]) * sd
    return pd.DataFrame(columns, index=index)
