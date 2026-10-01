r"""Starting values of the EM algorithm of :class:`~nowcastbox.models.MixedFreqDFM`.

Following Bańbura & Modugno (2014, sec. 3) and Bańbura, Giannone & Reichlin (2011):

1. every series of the (standardised) panel is filled in on its native grid by a
   not-a-knot cubic spline, with the edges filled by local medians
   (:func:`nowcastbox.preprocessing.missing.fill_missing`); lower-frequency series are
   then linearly interpolated over the base grid;
2. blocks are processed in order: the factors of block :math:`b` are the first
   :math:`r_b` principal components of the filled base-frequency members of the block
   (lower-frequency members are used only when the block has too few base-frequency
   series), and the projection of the panel on them is removed before the next block.
   The order matters with nested blocks (a global block containing every series):
   :data:`BLOCK_ORDERS` lists the variants (``"given"`` - the model's block order, as
   in Bańbura & Modugno; ``"reversed"``; ``"specific_first"`` - smaller blocks first,
   the global block last; ``"independent"`` - each block on the whole panel, no
   projection removed).
   :class:`~nowcastbox.models.MixedFreqDFM` with ``init="pca"`` evaluates the
   log-likelihood of every variant and starts from the best (several starting points
   are the standard remedy for the local maxima of EM, McLachlan & Krishnan, 2008,
   sec. 2.7);
3. a VAR(p) estimated by OLS on the principal components gives :math:`A^b` and
   :math:`Q^b`;
4. loadings are (restricted) least-squares regressions of the **observed** values on the
   factors (and their lags, aggregated with the series weights); the residuals give the
   idiosyncratic AR(1) coefficient and innovation variance (for lower-frequency series
   the AR coefficient starts at zero and the variance is scaled by
   :math:`1/\sum_l w_l^2`). For calendar aggregations (weekly or daily base grids) the
   regressors are the calendar-weighted factor sums :math:`\sum_l w_{t,l} f_{t-l}` and
   the coefficients are the loadings of the latent base-frequency series;
5. with a long-run mean state (innovation I4), a centred moving average of the filled
   target is a preliminary long-run mean; it enters the regressions of step 4 of the
   selected series (loading fixed at one for the target), and the innovation variance
   starts at
   :data:`~nowcastbox.models.long_run.LONG_RUN_START_VARIANCE` (or the fixed value).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from nowcastbox.core.data import MixedFrequencyData
from nowcastbox.core.exceptions import NowcastDataError
from nowcastbox.core.frequency import is_fixed_ratio, native_to_base
from nowcastbox.models._em_steps import (
    VARIANCE_FLOOR,
    EMParameters,
    StateLayout,
    restricted_least_squares,
)
from nowcastbox.models.long_run import LONG_RUN_START_VARIANCE
from nowcastbox.preprocessing.missing import fill_missing

__all__ = [
    "BLOCK_ORDERS",
    "fill_for_initialization",
    "initial_long_run",
    "pca_initial_parameters",
    "preliminary_trend",
    "principal_components",
]

FloatArray = NDArray[np.float64]

_MIN_AR_PAIRS = 3

BLOCK_ORDERS: tuple[str, ...] = ("given", "reversed", "specific_first", "independent")
"""Orders of the block principal components (see the module notes)."""


def fill_for_initialization(panel: MixedFrequencyData) -> FloatArray:
    """Balanced panel used to compute the initial principal components.

    Parameters
    ----------
    panel : MixedFrequencyData
        Standardised panel.

    Returns
    -------
    numpy.ndarray, shape (n_periods, n_series)
        Values without missing entries: interior gaps filled by spline, edges by
        local medians; lower-frequency series interpolated linearly over the base grid
        (constant beyond the first and last slots).

    Raises
    ------
    NowcastDataError
        If a series has no observation.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from nowcastbox.core.data import MixedFrequencyData
    >>> from nowcastbox.models._init_conditions import fill_for_initialization
    >>> idx = pd.period_range("2020-01", periods=6, freq="M")
    >>> df = pd.DataFrame({"a": [1.0, np.nan, 3, 4, 5, np.nan]}, index=idx)
    >>> fill_for_initialization(MixedFrequencyData(df, "M"))[:, 0].round(2).tolist()
    [1.0, 2.0, 3.0, 4.0, 5.0, 4.0]
    """
    empty = [c for c, n in panel.n_observations().items() if n == 0]
    if empty:
        raise NowcastDataError(f"Series without observations: {empty}.")
    base = panel.base_frequency
    if all(is_fixed_ratio(base, m.frequency) for m in panel.metadata.values()):
        frame = fill_missing(panel, "spline", fill_ragged_edge=True, fill_leading=True).data
    else:
        frame = _fill_native(panel)
    for column in panel.columns:
        if panel.metadata[column].frequency != base:
            frame[column] = frame[column].interpolate(method="linear", limit_direction="both")
    values = frame.to_numpy(dtype=np.float64)
    # series whose single observation defeats interpolation fall back to zero (the mean)
    return np.where(np.isnan(values), 0.0, values)


def _fill_native(panel: MixedFrequencyData) -> pd.DataFrame:
    """Spline-fill every series on its native grid and write it back on its slots.

    Calendar-aware variant of the panel path of
    :func:`~nowcastbox.preprocessing.missing.fill_missing` (weekly or daily base grids).
    """
    frame = panel.data
    for column in panel.columns:
        native = panel.to_native(column)
        filled = fill_missing(
            native.to_frame(column), "spline", fill_ragged_edge=True, fill_leading=True
        )[column]
        slots = native_to_base(filled.index, panel.base_frequency)  # type: ignore[arg-type]
        values = pd.Series(filled.to_numpy(), index=slots).reindex(panel.index)
        frame[column] = values.to_numpy()
    return frame


def principal_components(values: FloatArray, n_components: int) -> FloatArray:
    """First principal components ``X V_r`` of a balanced panel.

    Parameters
    ----------
    values : numpy.ndarray, shape (n_periods, n_series)
        Balanced panel (columns roughly standardised).
    n_components : int
        Number of components ``r``.

    Returns
    -------
    numpy.ndarray, shape (n_periods, r)
        Principal components, ordered by decreasing variance; the sign is fixed so that
        the loadings sum to a non-negative number.

    Raises
    ------
    ValueError
        If ``n_components`` exceeds the number of series.

    Examples
    --------
    >>> import numpy as np
    >>> from nowcastbox.models._init_conditions import principal_components
    >>> x = np.outer(np.arange(5.0) - 2, [1.0, 2.0])
    >>> principal_components(x, 1).ravel().round(3).tolist()
    [-4.472, -2.236, 0.0, 2.236, 4.472]
    """
    n_series = values.shape[1]
    if n_components > n_series:
        raise ValueError(f"cannot extract {n_components} components from {n_series} series")
    cov = np.cov(values, rowvar=False).reshape(n_series, n_series)
    eigval, eigvec = np.linalg.eigh(cov)
    vectors = eigvec[:, np.argsort(eigval)[::-1][:n_components]]
    signs = np.where(vectors.sum(axis=0) < 0, -1.0, 1.0)
    return values @ (vectors * signs)


def _block_order(layout: StateLayout, order: str) -> list[int]:
    """Processing order of the blocks for ``order`` in :data:`BLOCK_ORDERS`."""
    blocks = list(range(layout.n_blocks))
    if order == "reversed":
        return blocks[::-1]
    if order == "specific_first":
        sizes = layout.membership.sum(axis=0)
        return sorted(blocks, key=lambda b: (int(sizes[b]), b))
    return blocks


def _block_factors(
    filled: FloatArray,
    layout: StateLayout,
    base_mask: NDArray[np.bool_],
    order: str = "given",
) -> list[FloatArray]:
    """Block principal components (projection removed after each block, see notes)."""
    if order not in BLOCK_ORDERS:
        raise ValueError(f"block_order must be one of {BLOCK_ORDERS}, got {order!r}.")
    centred = filled - filled.mean(axis=0)
    residual = centred
    factors: list[FloatArray] = [np.empty((0, 0))] * layout.n_blocks
    for b in _block_order(layout, order):
        members = layout.membership[:, b]
        use = members & base_mask
        if use.sum() < layout.n_factors[b]:
            use = members
        if use.sum() < layout.n_factors[b]:
            msg = (
                f"block {layout.block_names[b]!r} has {int(use.sum())} series but "
                f"{layout.n_factors[b]} factors were requested"
            )
            raise ValueError(msg)
        f = principal_components(residual[:, use], layout.n_factors[b])
        factors[b] = f
        if order != "independent":
            coef = np.linalg.lstsq(f, residual, rcond=None)[0]
            residual = residual - f @ coef
    return factors


def _lag_matrix(values: FloatArray, n_lags: int) -> FloatArray:
    """``[x_t, x_{t-1}, ..., x_{t-n_lags+1}]`` with NaN for pre-sample lags."""
    n, k = values.shape
    out = np.full((n, k * n_lags), np.nan)
    for lag in range(n_lags):
        out[lag:, lag * k : (lag + 1) * k] = values[: n - lag]
    return out


def _initial_var(factors: FloatArray, p: int) -> tuple[FloatArray, FloatArray]:
    r = factors.shape[1]
    lagged = _lag_matrix(factors, p + 1)[p:]
    y, x = lagged[:, :r], lagged[:, r:]
    if y.shape[0] <= x.shape[1]:
        raise NowcastDataError(
            f"Too few periods ({factors.shape[0]}) to initialise a VAR({p}) of {r} factors."
        )
    A = np.linalg.lstsq(x, y, rcond=None)[0].T
    resid = y - x @ A.T
    Q = resid.T @ resid / resid.shape[0]
    eigval, eigvec = np.linalg.eigh(0.5 * (Q + Q.T))
    Q = (eigvec * np.maximum(eigval, VARIANCE_FLOOR)) @ eigvec.T
    return A, 0.5 * (Q + Q.T)


def _ar1_from_residuals(resid: FloatArray, observed: NDArray[np.intp]) -> tuple[float, float]:
    """AR(1) coefficient and innovation variance from residuals on observed periods."""
    variance = max(float(np.mean(resid**2)), VARIANCE_FLOOR)
    consecutive = np.flatnonzero(np.diff(observed) == 1)
    if consecutive.size < _MIN_AR_PAIRS:
        return 0.0, variance
    e_lag, e_cur = resid[consecutive], resid[consecutive + 1]
    denom = float(e_lag @ e_lag)
    rho = float(np.clip(e_lag @ e_cur / denom, -0.9, 0.9)) if denom > 0 else 0.0
    return rho, max(variance * (1.0 - rho**2), VARIANCE_FLOOR)


def preliminary_trend(values: FloatArray, window: int = 61) -> FloatArray:
    """Centred moving average used as a starting long-run mean (innovation I4).

    Parameters
    ----------
    values : numpy.ndarray, shape (n_periods,)
        Filled (balanced) target series on the base grid.
    window : int, default 61
        Maximum window length (base periods); capped at a third of the sample.

    Returns
    -------
    numpy.ndarray, shape (n_periods,)
        Moving average (shorter windows at the edges), demeaned.

    Examples
    --------
    >>> import numpy as np
    >>> from nowcastbox.models._init_conditions import preliminary_trend
    >>> preliminary_trend(np.arange(9.0), window=3).tolist()
    [-3.5, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 3.5]
    """
    n = values.size
    width = max(1, min(int(window), n // 3))
    trend = pd.Series(values).rolling(width, center=True, min_periods=1).mean().to_numpy()
    return np.asarray(trend - trend.mean(), dtype=np.float64)


def _regress_series(
    x: FloatArray, design: FloatArray, R: FloatArray, q: FloatArray
) -> tuple[FloatArray, FloatArray]:
    k = design.shape[1]
    lam = restricted_least_squares(design.T @ design + 1e-8 * np.eye(k), design.T @ x, R, q)
    return lam, x - design @ lam


def _series_design(
    layout: StateLayout, lagged: FloatArray, series: int
) -> tuple[FloatArray, NDArray[np.int64], FloatArray]:
    r"""Regressors of a series, the state indices of their coefficients and the weights.

    Fixed weights: the factor lags of :meth:`StateLayout.loading_index` (restricted
    later). Calendar aggregations: :math:`G_t = \sum_l w_{t,l} f_{t-l}` (``NaN`` when a
    lag with a non-zero weight is missing), whose coefficients are the loadings on the
    current factors.
    """
    idx = layout.loading_index(series)
    n = lagged.shape[0]
    path = np.asarray(layout.weight_path(series, n), dtype=np.float64)
    if layout.calendar[series] is None:
        return lagged[:, idx], idx, path
    r = layout.n_series_factors(series)
    lags = lagged[:, idx].reshape(n, path.shape[1], r)
    used = path != 0.0
    incomplete = (used[:, :, None] & np.isnan(lags)).any(axis=(1, 2))
    design = np.einsum("tl,tlr->tr", path, np.where(used[:, :, None], lags, 0.0))
    design[incomplete] = np.nan
    return design, idx[:r], path


def _initial_loadings(
    observations: FloatArray,
    layout: StateLayout,
    lagged: FloatArray,
    trend: FloatArray | None = None,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray | None]:
    """Loadings (factor part of ``Z``), idiosyncratic AR(1), variances, long-run loadings."""
    n_series = layout.n_series
    loadings = np.zeros((n_series, layout.n_factor_states))
    rho, var = np.zeros(n_series), np.ones(n_series)
    trend_loadings = None if trend is None else np.zeros(n_series)
    for i in range(n_series):
        design, idx, path = _series_design(layout, lagged, i)
        with_trend = trend is not None and i in layout.long_run
        if trend is not None and with_trend:
            design = np.column_stack([design, trend])
        ok = ~np.isnan(observations[:, i]) & ~np.isnan(design).any(axis=1)
        observed = np.flatnonzero(ok)
        if observed.size == 0:
            raise NowcastDataError(
                f"Series {layout.series[i]!r} has no observation with complete factor lags."
            )
        R, q = layout.constraints(i) if with_trend else layout.factor_constraints(i)
        lam, resid = _regress_series(observations[observed, i], design[observed], R, q)
        loadings[i, idx] = lam[: idx.size]
        if with_trend and trend_loadings is not None:
            trend_loadings[i] = lam[-1]
        if path.shape[1] == 1 and layout.idiosyncratic == "ar1":
            rho[i], var[i] = _ar1_from_residuals(resid, observed)
        else:
            scale = 1.0
            if layout.idiosyncratic == "ar1":
                scale = float(np.mean(np.einsum("tl,tl->t", path[observed], path[observed])))
            var[i] = max(float(np.mean(resid**2)) / scale, VARIANCE_FLOOR)
    return loadings, rho, var, trend_loadings


def pca_initial_parameters(
    panel: MixedFrequencyData,
    layout: StateLayout,
    *,
    obs_noise_var: float = 1e-4,
    block_order: str = "given",
) -> EMParameters:
    """Starting values from principal components of the spline-filled panel.

    Parameters
    ----------
    panel : MixedFrequencyData
        Standardised panel whose columns are ``layout.series``.
    layout : StateLayout
        State layout of the model.
    obs_noise_var : float, default 1e-4
        Fixed measurement-noise variance of the ``"ar1"`` specification.
    block_order : {"given", "reversed", "specific_first", "independent"}, default "given"
        Order of the block principal components (:data:`BLOCK_ORDERS`; irrelevant
        with a single block).

    Returns
    -------
    EMParameters
        Initial parameters.

    Raises
    ------
    NowcastDataError
        If the panel is too short or a series cannot be related to the factors.
    ValueError
        If a block has fewer series than factors or ``block_order`` is unknown.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from nowcastbox.core.data import MixedFrequencyData
    >>> from nowcastbox.models._em_steps import StateLayout
    >>> from nowcastbox.models._init_conditions import pca_initial_parameters
    >>> rng = np.random.default_rng(0)
    >>> f = np.cumsum(rng.standard_normal(60)) * 0.1
    >>> x = np.outer(f, [1.0, 0.8, 0.6]) + 0.3 * rng.standard_normal((60, 3))
    >>> idx = pd.period_range("2015-01", periods=60, freq="M")
    >>> z, _ = MixedFrequencyData(pd.DataFrame(x, idx, ["a", "b", "c"]), "M").standardize()
    >>> lay = StateLayout(list("abc"), ["g"], [1], 1, np.ones((3, 1), bool), [[1.0]] * 3, "ar1")
    >>> pca_initial_parameters(z, lay).loadings.shape
    (3, 1)
    """
    if list(panel.columns) != list(layout.series):
        raise ValueError("panel columns and layout series differ")
    filled = fill_for_initialization(panel)
    base = panel.base_frequency
    base_mask = np.array([panel.metadata[c].frequency == base for c in panel.columns])
    factors = _block_factors(filled, layout, base_mask, block_order)
    transitions, covs = zip(*(_initial_var(f, layout.factor_lags) for f in factors), strict=True)
    lagged = np.concatenate(
        [_lag_matrix(f, s) for f, s in zip(factors, layout.block_lags, strict=True)], axis=1
    )
    observations = panel.values
    trend = None
    if layout.has_long_run:
        trend = preliminary_trend(filled[:, layout.long_run[0]])
    loadings, rho, var, trend_loadings = _initial_loadings(observations, layout, lagged, trend)
    if layout.idiosyncratic == "ar1":
        obs_var = np.full(layout.n_series, float(obs_noise_var))
    else:
        obs_var = var.copy()
    _, trend_var = initial_long_run(layout)
    return EMParameters(
        tuple(transitions), tuple(covs), loadings, rho, var, obs_var, trend_loadings, trend_var
    )


def initial_long_run(layout: StateLayout) -> tuple[FloatArray | None, float]:
    """Default long-run mean loadings and innovation variance (innovation I4).

    :func:`pca_initial_parameters` replaces the loadings by regressions on a
    :func:`preliminary_trend`; the variance is used as is.

    Parameters
    ----------
    layout : StateLayout
        State layout.

    Returns
    -------
    trend_loadings : numpy.ndarray or None
        One for the first ``layout.long_run`` series (the target), zero otherwise;
        ``None`` without a long-run state.
    trend_var : float
        Fixed ``layout.long_run_variance`` or
        :data:`~nowcastbox.models.long_run.LONG_RUN_START_VARIANCE` (``0.0`` without
        a long-run state).

    Examples
    --------
    >>> import numpy as np
    >>> from nowcastbox.models._em_steps import StateLayout
    >>> from nowcastbox.models._init_conditions import initial_long_run
    >>> lay = StateLayout(
    ...     ["a", "b"], ["g"], [1], 1, np.ones((2, 1), bool), [[1.0]] * 2, "iid", long_run=[1]
    ... )
    >>> loads, var = initial_long_run(lay)
    >>> loads.tolist(), var
    ([0.0, 1.0], 0.001)
    """
    if not layout.has_long_run:
        return None, 0.0
    loads = np.zeros(layout.n_series)
    loads[layout.long_run[0]] = 1.0
    if layout.long_run_variance is not None:
        return loads, layout.long_run_variance
    return loads, LONG_RUN_START_VARIANCE
