r"""Number of primitive (dynamic) shocks: Bai & Ng (2007).

In the factor model ``x_t = \Lambda F_t + e_t`` with VAR dynamics
``F_t = A_1 F_{t-1} + \dots + A_p F_{t-p} + u_t`` and ``u_t = R \varepsilon_t``, where
``R`` is ``r x q`` with rank ``q <= r``, the covariance of ``u_t`` has rank ``q``. Bai &
Ng (2007, section 4, Proposition 2) estimate ``q`` as follows:

1. estimate ``F_t`` by principal components (``r`` factors) on the standardised panel
   under the normalisation ``\Lambda'\Lambda / N = I_r`` (so that the variance of each
   factor is proportional to its eigenvalue, Lemma 2);
2. fit a VAR(``p``) to ``\hat F_t`` by OLS and compute the residual covariance
   ``\hat\Sigma_u = T^{-1}\sum_t \hat u_t \hat u_t'`` (or the residual correlation
   matrix ``\hat S_u``, which is scale invariant);
3. with its eigenvalues ``\hat c_1 \ge \dots \ge \hat c_r``, compute (section 2)

   .. math::

       \hat D_{1,k} = \left(\frac{\hat c_{k+1}^2}{\sum_{j=1}^r \hat c_j^2}\right)^{1/2},
       \qquad
       \hat D_{2,k} = \left(\frac{\sum_{j=k+1}^r \hat c_j^2}
                                 {\sum_{j=1}^r \hat c_j^2}\right)^{1/2};

4. set ``\hat q_3 = \min\{k : \hat D_{1,k} < M_{NT}\}`` and
   ``\hat q_4 = \min\{k : \hat D_{2,k} < M_{NT}\}`` (eqs. 14-15) with the bound
   ``M_{NT} = m / \min(N^{1/2-\delta}, T^{1/2-\delta})``, ``0 < \delta < 1/2``, ``m > 0``.

Since ``\hat D_{\cdot, r} = 0``, ``1 \le \hat q \le r`` always.

**Tuning constants (Bai & Ng, 2007, section 5).** The simulations use
``\delta = 0.1`` (so ``M_{NT} = m / \min(N^{2/5}, T^{2/5})``) and a VAR(2) in the
factors ("selecting too few lags will be problematic"). With the covariance matrix,
``m = 1`` works for both statistics; with the correlation matrix the authors recommend
``m = 1.25`` for ``\hat D_1`` and ``m = 2.25`` for ``\hat D_2``. These are the defaults
here (``m=None`` picks them; see :data:`DEFAULT_M`). ``\hat D_1`` "tends to have better
properties when N or T is small".

References
----------
Bai, J., & Ng, S. (2007). Determining the number of primitive shocks in factor models.
*Journal of Business & Economic Statistics*, 25(1), 52-60.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from nowcastbox._logging import get_logger
from nowcastbox.core.exceptions import NowcastDataError
from nowcastbox.selection._panel import (
    MissingPolicy,
    PanelInput,
    fit_var_ols,
    prepare_panel,
    principal_components,
)
from nowcastbox.selection._plot import Backend, plot_lines
from nowcastbox.selection.bai_ng_factors import select_factors

__all__ = [
    "DEFAULT_M",
    "ShockSelectionResult",
    "ShockStatistic",
    "select_shocks",
    "shock_bound",
    "shock_statistics",
]

logger = get_logger(__name__)

ShockStatistic = Literal["D1", "D2"]

DEFAULT_M: dict[tuple[str, str], float] = {
    ("covariance", "D1"): 1.0,
    ("covariance", "D2"): 1.0,
    ("correlation", "D1"): 1.25,
    ("correlation", "D2"): 2.25,
}
"""Scale ``m`` of the bound recommended by Bai & Ng (2007, section 5), by
``(matrix, statistic)``."""


def shock_bound(n_series: int, n_periods: int, delta: float = 0.1, m: float = 1.0) -> float:
    """Threshold ``M_NT = m / min(N^(1/2 - delta), T^(1/2 - delta))``.

    Parameters
    ----------
    n_series, n_periods : int
        Panel dimensions ``N`` and ``T``.
    delta : float, default 0.1
        Rate parameter, ``0 < delta < 1/2``.
    m : float, default 1.0
        Scale parameter, ``m > 0``.

    Returns
    -------
    float
        The bound.

    Raises
    ------
    ValueError
        Parameters out of range.

    Examples
    --------
    >>> round(shock_bound(100, 100, delta=0.1, m=1.0), 6)
    0.158489
    """
    _check_delta_m(delta, m)
    if n_series < 1 or n_periods < 1:
        raise ValueError("n_series and n_periods must be positive.")
    expo = 0.5 - delta
    return float(m / min(n_series**expo, n_periods**expo))


def _check_delta_m(delta: float, m: float) -> None:
    if not np.isfinite(delta) or not 0.0 < delta < 0.5:
        raise ValueError(f"delta must lie in (0, 1/2); got {delta!r}.")
    if not np.isfinite(m) or m <= 0:
        raise ValueError(f"m must be positive; got {m!r}.")


def shock_statistics(eigenvalues: np.ndarray) -> pd.DataFrame:
    """Bai-Ng (2007) statistics ``D1_k`` and ``D2_k`` for ``k = 1, ..., r``.

    Parameters
    ----------
    eigenvalues : numpy.ndarray
        Eigenvalues of the residual covariance (any order; sorted internally).

    Returns
    -------
    pandas.DataFrame
        Index ``k`` (``n_shocks``), columns ``D1`` and ``D2``.

    Raises
    ------
    NowcastDataError
        If all eigenvalues are zero.

    Examples
    --------
    >>> shock_statistics(np.array([3.0, 4.0, 0.0]))["D2"].round(2).tolist()
    [0.6, 0.0, 0.0]
    """
    c = np.sort(np.clip(np.asarray(eigenvalues, dtype=float), 0.0, None))[::-1]
    sq = c**2
    total = sq.sum()
    if total <= 0:
        raise NowcastDataError("Residual covariance is zero; cannot compute D statistics.")
    r = len(c)
    tail = np.concatenate((np.cumsum(sq[::-1])[::-1], [0.0]))  # tail[k] = sum_{j>=k} (0-based)
    next_sq = np.concatenate((sq, [0.0]))
    k = np.arange(1, r + 1)
    d1 = np.sqrt(next_sq[k] / total)
    d2 = np.sqrt(tail[k] / total)
    return pd.DataFrame({"D1": d1, "D2": d2}, index=pd.Index(k, name="n_shocks"))


@dataclass(frozen=True)
class ShockSelectionResult:
    """Outcome of :func:`select_shocks`.

    Attributes
    ----------
    q_star : int
        Selected number of primitive shocks under ``statistic``.
    statistic : str
        ``"D1"`` or ``"D2"``.
    q_by_statistic : dict[str, int]
        Selection under each statistic.
    statistics : pandas.DataFrame
        ``D1_k`` and ``D2_k`` for ``k = 1..r``.
    bound : float
        Threshold ``M_NT`` of ``statistic``.
    eigenvalues : pandas.Series
        Eigenvalues ``c_1 >= ... >= c_r`` of the residual covariance/correlation.
    n_factors, factor_lags : int
        ``r`` and the VAR order ``p``.
    delta, m : float
        Tuning parameters of the bound (``m`` of ``statistic``).
    matrix : str
        ``"covariance"`` or ``"correlation"``.
    n_series, n_periods : int
        Dimensions of the balanced panel.
    residual_covariance : numpy.ndarray
        ``r x r`` matrix whose eigenvalues were used.
    columns : list[str]
        Series used.
    m_by_statistic : dict[str, float]
        Scale ``m`` used for each statistic.
    bound_by_statistic : dict[str, float]
        Bound ``M_NT`` of each statistic.
    """

    q_star: int
    statistic: str
    q_by_statistic: dict[str, int]
    statistics: pd.DataFrame
    bound: float
    eigenvalues: pd.Series
    n_factors: int
    factor_lags: int
    delta: float
    m: float
    matrix: str
    n_series: int
    n_periods: int
    residual_covariance: np.ndarray
    columns: list[str] = field(default_factory=list)
    m_by_statistic: dict[str, float] = field(default_factory=dict)
    bound_by_statistic: dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        """Plain-text summary.

        Returns
        -------
        str
            Settings, bound, selections and the statistics table.

        Examples
        --------
        >>> import numpy as np
        >>> x = np.random.default_rng(0).normal(size=(60, 20))
        >>> "Bai & Ng (2007)" in select_shocks(x, n_factors=2).summary()
        True
        """
        lines = [
            "Number of primitive shocks - Bai & Ng (2007)",
            "=" * 46,
            f"N (series): {self.n_series}   T (periods): {self.n_periods}",
            f"r (factors): {self.n_factors}   p (VAR lags): {self.factor_lags}   "
            f"matrix: {self.matrix}",
            f"delta: {self.delta}   "
            + "   ".join(
                f"{k}: m={self.m_by_statistic.get(k, self.m)}, "
                f"M_NT={self.bound_by_statistic.get(k, self.bound):.5f}"
                for k in ("D1", "D2")
            ),
            f"Selected ({self.statistic}): q* = {self.q_star}",
            "By statistic: " + ", ".join(f"{k}={v}" for k, v in self.q_by_statistic.items()),
            "",
            pd.concat([self.eigenvalues.rename("eigenvalue"), self.statistics], axis=1)
            .rename_axis("k")
            .to_string(float_format=lambda v: f"{v: .5f}"),
        ]
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()

    def plot(self, *, backend: Backend = "matplotlib", ax: Any = None) -> Any:
        """Plot ``D1_k`` and ``D2_k`` against ``k`` with the bound ``M_NT``.

        Parameters
        ----------
        backend : {"matplotlib", "plotly"}, default "matplotlib"
            Plotting library.
        ax : matplotlib.axes.Axes, optional
            Existing axes (Matplotlib only).

        Returns
        -------
        matplotlib.axes.Axes or plotly.graph_objects.Figure
            The plot.

        Examples
        --------
        >>> import matplotlib
        >>> matplotlib.use("Agg")
        >>> import numpy as np
        >>> res = select_shocks(np.random.default_rng(0).normal(size=(60, 20)), n_factors=2)
        >>> ax = res.plot()
        """
        return plot_lines(
            self.statistics,
            selected=dict(self.q_by_statistic),
            title="Bai & Ng (2007) primitive shocks",
            xlabel="number of shocks k",
            ylabel="D statistic",
            hline=self.bound,
            hline_label="bound M_NT",
            backend=backend,
            ax=ax,
        )


def _check_positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer; got {value!r}.")
    return int(value)


def select_shocks(
    x: PanelInput,
    n_factors: int | None = None,
    factor_lags: int = 2,
    delta: float = 0.1,
    m: float | None = None,
    *,
    statistic: ShockStatistic = "D1",
    matrix: Literal["covariance", "correlation"] = "covariance",
    standardize: bool = True,
    missing: MissingPolicy = "drop_rows",
) -> ShockSelectionResult:
    """Select the number of primitive dynamic shocks ``q`` (Bai & Ng, 2007).

    Parameters
    ----------
    x : DataFrame, ndarray or MixedFrequencyData
        ``T x N`` panel (rows = periods). For :class:`MixedFrequencyData` only series at
        the base frequency are used.
    n_factors : int, optional
        Number of static factors ``r``. When omitted it is chosen with
        :func:`~nowcastbox.selection.select_factors` (``IC2``, ``rmax=min(10, min(N,T)-1)``).
    factor_lags : int, default 2
        Order ``p`` of the VAR fitted to the factors (Bai & Ng, 2007, report VAR(2);
        too few lags leave serial correlation in ``u_t``).
    delta : float, default 0.1
        Rate parameter of the bound, ``0 < delta < 1/2`` (Bai & Ng, 2007, section 5).
    m : float, optional
        Scale parameter of the bound, ``m > 0``, used for both statistics. Default: the
        values recommended by Bai & Ng (2007, section 5) - ``1`` with the covariance
        matrix, ``1.25`` (``D1``) and ``2.25`` (``D2``) with the correlation matrix
        (:data:`DEFAULT_M`).
    statistic : {"D1", "D2"}, default "D1"
        Statistic defining ``q_star`` (both are computed).
    matrix : {"covariance", "correlation"}, default "covariance"
        Use the eigenvalues of the residual covariance or correlation matrix.
    standardize : bool, default True
        Standardise the series before principal components.
    missing : {"drop_rows", "drop_columns", "raise"}, default "drop_rows"
        Handling of missing values (see :func:`select_factors`).

    Returns
    -------
    ShockSelectionResult
        ``q_star``, ``D1``/``D2`` statistics, bound and residual eigenvalues.

    Raises
    ------
    ValueError
        Invalid tuning parameters.
    NowcastDataError
        Invalid data or too few periods for the VAR.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> t, n = 300, 100
    >>> s = np.zeros(t + 1)
    >>> for i in range(1, t + 1):
    ...     s[i] = 0.6 * s[i - 1] + rng.normal()
    >>> f = np.column_stack([s[1:], s[:-1]])  # r = 2 static factors, q = 1 shock
    >>> x = f @ rng.normal(size=(n, 2)).T + rng.normal(size=(t, n))
    >>> select_shocks(x, n_factors=2, factor_lags=1).q_star
    1
    """
    if statistic not in ("D1", "D2"):
        raise ValueError(f"statistic must be 'D1' or 'D2'; got {statistic!r}.")
    if matrix not in ("covariance", "correlation"):
        raise ValueError(f"matrix must be 'covariance' or 'correlation'; got {matrix!r}.")
    m_by = {k: DEFAULT_M[(matrix, k)] if m is None else m for k in ("D1", "D2")}
    for value in m_by.values():
        _check_delta_m(delta, value)
    factor_lags = _check_positive_int("factor_lags", factor_lags)
    panel = prepare_panel(x, standardize=standardize, missing=missing)
    n_periods, n_series = panel.n_periods, panel.n_series
    if n_factors is None:
        rmax = min(10, min(n_series, n_periods) - 1)
        n_factors = select_factors(
            panel.values, rmax=rmax, criterion="IC2", standardize=False
        ).r_star
        if n_factors == 0:
            raise NowcastDataError(
                "select_factors (IC2) found no factor; pass n_factors explicitly."
            )
        logger.info("select_shocks: n_factors chosen by IC2: %d", n_factors)
    n_factors = _check_positive_int("n_factors", n_factors)
    if n_factors > min(n_series, n_periods):
        raise ValueError(f"n_factors={n_factors} exceeds min(N, T) = {min(n_series, n_periods)}.")
    factors, _, singular = principal_components(panel.values, n_factors)
    # Bai-Ng normalisation Lambda'Lambda/N = I: F = X V / sqrt(N) = U S / sqrt(N),
    # i.e. the unit-variance factors rescaled by s_k / sqrt(N T).
    factors = factors * (singular[:n_factors] / np.sqrt(n_series * n_periods))
    _, residuals = fit_var_ols(factors, factor_lags, trend=True)
    sigma = residuals.T @ residuals / residuals.shape[0]
    if matrix == "correlation":
        sd = np.sqrt(np.diag(sigma))
        if np.any(sd <= 1e-10):
            raise NowcastDataError("A VAR residual has zero variance; use matrix='covariance'.")
        sigma = sigma / np.outer(sd, sd)
    eig = np.sort(np.linalg.eigvalsh((sigma + sigma.T) / 2))[::-1]
    stats = shock_statistics(eig)
    bounds = {k: shock_bound(n_series, n_periods, delta=delta, m=v) for k, v in m_by.items()}
    q_by = {}
    for name in ("D1", "D2"):
        below = np.flatnonzero(stats[name].to_numpy() < bounds[name])
        q_by[name] = int(stats.index[below[0]]) if below.size else n_factors
    logger.debug("Bai-Ng shocks: bounds=%s -> %s", bounds, q_by)
    return ShockSelectionResult(
        q_star=q_by[statistic],
        statistic=statistic,
        q_by_statistic=q_by,
        statistics=stats,
        bound=bounds[statistic],
        eigenvalues=pd.Series(
            np.clip(eig, 0.0, None),
            index=pd.RangeIndex(1, n_factors + 1, name="k"),
            name="eigenvalue",
        ),
        n_factors=n_factors,
        factor_lags=factor_lags,
        delta=float(delta),
        m=float(m_by[statistic]),
        matrix=matrix,
        n_series=n_series,
        n_periods=n_periods,
        residual_covariance=sigma,
        columns=[str(c) for c in panel.columns],
        m_by_statistic={k: float(v) for k, v in m_by.items()},
        bound_by_statistic=bounds,
    )
