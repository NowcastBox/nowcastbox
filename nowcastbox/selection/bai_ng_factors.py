r"""Number of static factors: Bai & Ng (2002) information criteria.

For a balanced, standardised ``T x N`` panel ``X`` estimated by principal components
with ``r`` factors, let

.. math::

    V(r) = \frac{1}{NT} \sum_{i=1}^{N} \sum_{t=1}^{T} (x_{it} - \hat\lambda_i'\hat F_t)^2
         = \frac{1}{NT} \sum_{j > r} s_j^2,

where ``s_j`` are the singular values of ``X``. With ``C_{NT}^2 = \min(N, T)`` the
penalties of Bai & Ng (2002, eqs. 9 and 10) are

.. math::

    g_1 = \frac{N+T}{NT}\ln\frac{NT}{N+T},\quad
    g_2 = \frac{N+T}{NT}\ln C_{NT}^2,\quad
    g_3 = \frac{\ln C_{NT}^2}{C_{NT}^2},

and the criteria are

.. math::

    PC_{p,k}(r) = V(r) + r\,\hat\sigma^2 g_k, \qquad
    IC_{p,k}(r) = \ln V(r) + r\, g_k, \qquad k = 1, 2, 3,

with ``\hat\sigma^2 = V(r_{max})``. The estimate is ``\hat r = \arg\min_{0 \le r \le
r_{max}}`` of the chosen criterion.

**Small samples.** The criteria are asymptotic in both ``N`` and ``T``. In the Monte
Carlo of Bai & Ng (2002, section 5, Tables I-VIII; ``r_{max} = 8``) the estimates are
precise when ``min(N, T) >= 40``; with ``min(N, T) = 10`` every criterion selects
``r_{max}``, and with ``min(N, T) = 20`` the ``PC_p`` criteria overestimate ``r``
(``IC_p`` are still accurate for ``r = 1`` but underestimate larger ``r``). ``PC_p``
also depends on ``r_{max}`` through ``\hat\sigma^2``. :func:`select_factors` therefore
emits a :class:`~nowcastbox.core.exceptions.DataQualityWarning` when
``min(N, T) <`` :data:`MIN_RELIABLE_DIM` (20).

References
----------
Bai, J., & Ng, S. (2002). Determining the number of factors in approximate factor
models. *Econometrica*, 70(1), 191-221.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from nowcastbox._logging import get_logger
from nowcastbox.core.exceptions import DataQualityWarning
from nowcastbox.selection._panel import MissingPolicy, PanelInput, prepare_panel
from nowcastbox.selection._plot import Backend, plot_lines

__all__ = [
    "CRITERIA",
    "MIN_RELIABLE_DIM",
    "FactorCriterion",
    "FactorSelectionResult",
    "bai_ng_penalty",
    "factor_criteria",
    "select_factors",
]

logger = get_logger(__name__)

FactorCriterion = Literal["IC1", "IC2", "IC3", "PC1", "PC2", "PC3"]
CRITERIA: tuple[str, ...] = ("IC1", "IC2", "IC3", "PC1", "PC2", "PC3")

MIN_RELIABLE_DIM = 20
"""Below this ``min(N, T)`` the Bai-Ng (2002) criteria are unreliable (they tend to
select ``rmax``); :func:`select_factors` warns."""


def bai_ng_penalty(n_series: int, n_periods: int, k: int) -> float:
    """Penalty ``g_k(N, T)`` of Bai & Ng (2002).

    Parameters
    ----------
    n_series : int
        Cross-section dimension ``N``.
    n_periods : int
        Time dimension ``T``.
    k : {1, 2, 3}
        Penalty number.

    Returns
    -------
    float
        Penalty per factor.

    Raises
    ------
    ValueError
        If ``k`` is not 1, 2 or 3 or the dimensions are not positive.

    Examples
    --------
    >>> round(bai_ng_penalty(100, 100, 3), 6)
    0.046052
    """
    if n_series < 1 or n_periods < 1:
        raise ValueError("n_series and n_periods must be positive.")
    n, t = float(n_series), float(n_periods)
    c2 = min(n, t)
    if k == 1:
        return (n + t) / (n * t) * np.log(n * t / (n + t))
    if k == 2:
        return (n + t) / (n * t) * np.log(c2)
    if k == 3:
        return float(np.log(c2) / c2)
    raise ValueError(f"Penalty k must be 1, 2 or 3; got {k!r}.")


def factor_criteria(
    singular_values: np.ndarray, n_series: int, n_periods: int, rmax: int
) -> tuple[pd.Series, pd.DataFrame]:
    """Compute ``V(r)`` and the six Bai-Ng criteria for ``r = 0, ..., rmax``.

    Parameters
    ----------
    singular_values : numpy.ndarray
        Singular values of the balanced ``T x N`` panel, in decreasing order.
    n_series, n_periods : int
        Panel dimensions ``N`` and ``T``.
    rmax : int
        Largest number of factors considered.

    Returns
    -------
    ssr : pandas.Series
        ``V(r)`` indexed by ``r``.
    criteria : pandas.DataFrame
        Columns ``IC1, IC2, IC3, PC1, PC2, PC3`` indexed by ``r``.

    Examples
    --------
    >>> import numpy as np
    >>> s = np.linalg.svd(np.random.default_rng(0).normal(size=(30, 10)), compute_uv=False)
    >>> v, c = factor_criteria(s, 10, 30, 3)
    >>> list(c.columns)
    ['IC1', 'IC2', 'IC3', 'PC1', 'PC2', 'PC3']
    """
    sq = np.asarray(singular_values, dtype=float) ** 2
    total = sq.sum()
    explained = np.concatenate(([0.0], np.cumsum(sq[:rmax])))
    v = np.maximum(total - explained, 0.0) / (n_series * n_periods)
    r = np.arange(rmax + 1, dtype=float)
    sigma2 = v[rmax]
    data: dict[str, np.ndarray] = {}
    with np.errstate(divide="ignore"):
        log_v = np.log(v)
    for k in (1, 2, 3):
        data[f"IC{k}"] = log_v + r * bai_ng_penalty(n_series, n_periods, k)
    for k in (1, 2, 3):
        data[f"PC{k}"] = v + r * sigma2 * bai_ng_penalty(n_series, n_periods, k)
    index = pd.RangeIndex(rmax + 1, name="n_factors")
    return pd.Series(v, index=index, name="V"), pd.DataFrame(data, index=index)


@dataclass(frozen=True)
class FactorSelectionResult:
    """Outcome of :func:`select_factors`.

    Attributes
    ----------
    r_star : int
        Selected number of factors under ``criterion``.
    criterion : str
        Criterion used for ``r_star``.
    criteria : pandas.DataFrame
        Values of every criterion (columns) for ``r = 0, ..., rmax`` (index).
    r_star_by_criterion : dict[str, int]
        Minimiser of each criterion.
    ssr : pandas.Series
        ``V(r)``, the mean squared idiosyncratic residual.
    eigenvalues : pandas.Series
        Eigenvalues of ``X'X / T`` (indexed from 1), i.e. of the sample correlation
        matrix up to the ``(T-1)/T`` factor when the panel is standardised.
    explained_variance_ratio : pandas.Series
        Share of total variance explained by each principal component.
    rmax : int
        Largest number of factors considered.
    n_series, n_periods : int
        Dimensions ``N`` and ``T`` of the balanced panel.
    columns : list[str]
        Series used.
    dropped_rows : pandas.Index
        Periods dropped because of missing values.
    dropped_columns : list[str]
        Series dropped (missing values / lower frequency).
    """

    r_star: int
    criterion: str
    criteria: pd.DataFrame
    r_star_by_criterion: dict[str, int]
    ssr: pd.Series
    eigenvalues: pd.Series
    explained_variance_ratio: pd.Series
    rmax: int
    n_series: int
    n_periods: int
    columns: list[str] = field(default_factory=list)
    dropped_rows: pd.Index = field(default_factory=lambda: pd.Index([]))
    dropped_columns: list[str] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        """Criteria table with ``V(r)`` as first column.

        Returns
        -------
        pandas.DataFrame
            Index ``r``; columns ``V, IC1, ..., PC3``.

        Examples
        --------
        >>> import numpy as np
        >>> x = np.random.default_rng(0).normal(size=(40, 20))
        >>> select_factors(x, rmax=3).to_frame().shape
        (4, 7)
        """
        return pd.concat([self.ssr, self.criteria], axis=1)

    def summary(self) -> str:
        """Plain-text summary.

        Returns
        -------
        str
            Dimensions, selections per criterion and the criteria table.

        Examples
        --------
        >>> import numpy as np
        >>> x = np.random.default_rng(0).normal(size=(40, 20))
        >>> "Bai & Ng (2002)" in select_factors(x, rmax=3).summary()
        True
        """
        lines = [
            "Number of factors - Bai & Ng (2002)",
            "=" * 40,
            f"N (series): {self.n_series}   T (periods): {self.n_periods}   rmax: {self.rmax}",
            f"Selected ({self.criterion}): r* = {self.r_star}",
            "By criterion: " + ", ".join(f"{k}={v}" for k, v in self.r_star_by_criterion.items()),
        ]
        if len(self.dropped_rows):
            lines.append(f"Dropped periods (missing values): {len(self.dropped_rows)}")
        if self.dropped_columns:
            lines.append(f"Dropped series: {len(self.dropped_columns)}")
        lines += ["", self.to_frame().to_string(float_format=lambda v: f"{v: .5f}")]
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()

    def plot(
        self,
        kind: Literal["criteria", "eigenvalues"] = "criteria",
        *,
        criteria: list[str] | None = None,
        backend: Backend = "matplotlib",
        ax: Any = None,
    ) -> Any:
        """Plot the criteria against ``r`` or the scree plot of eigenvalues.

        Parameters
        ----------
        kind : {"criteria", "eigenvalues"}, default "criteria"
            What to draw.
        criteria : list[str], optional
            Criteria to draw (default: the selected one; ``kind="criteria"`` only).
        backend : {"matplotlib", "plotly"}, default "matplotlib"
            Plotting library.
        ax : matplotlib.axes.Axes, optional
            Existing axes (Matplotlib only).

        Returns
        -------
        matplotlib.axes.Axes or plotly.graph_objects.Figure
            The plot.

        Raises
        ------
        ValueError
            Unknown ``kind`` or criterion.

        Examples
        --------
        >>> import matplotlib
        >>> matplotlib.use("Agg")
        >>> import numpy as np
        >>> res = select_factors(np.random.default_rng(0).normal(size=(40, 20)), rmax=3)
        >>> ax = res.plot()
        """
        if kind == "criteria":
            cols = [self.criterion] if criteria is None else list(criteria)
            unknown = [c for c in cols if c not in self.criteria.columns]
            if unknown:
                raise ValueError(f"Unknown criteria {unknown}; choose from {CRITERIA}.")
            return plot_lines(
                self.criteria[cols].replace(-np.inf, np.nan),
                selected={c: self.r_star_by_criterion[c] for c in cols},
                title="Bai & Ng (2002) information criteria",
                xlabel="number of factors r",
                ylabel="criterion",
                backend=backend,
                ax=ax,
            )
        if kind == "eigenvalues":
            return plot_lines(
                self.eigenvalues.to_frame("eigenvalue"),
                selected={"eigenvalue": self.r_star},
                title="Scree plot",
                xlabel="component",
                ylabel="eigenvalue of X'X/T",
                backend=backend,
                ax=ax,
            )
        raise ValueError(f"kind must be 'criteria' or 'eigenvalues'; got {kind!r}.")


def _normalize_criterion(criterion: str) -> str:
    if not isinstance(criterion, str):
        raise ValueError(f"criterion must be a string; got {criterion!r}.")
    # Accept "IC2", "ic2", "ICp2", "IC_p2", "PCp1", ...
    key = criterion.upper().replace("_", "")
    if len(key) == 4 and key[2] == "P":
        key = key[:2] + key[3]
    if key not in CRITERIA:
        raise ValueError(f"criterion must be one of {CRITERIA}; got {criterion!r}.")
    return key


def select_factors(
    x: PanelInput,
    rmax: int = 10,
    criterion: FactorCriterion | str = "IC2",
    *,
    standardize: bool = True,
    missing: MissingPolicy = "drop_rows",
    warn_small_sample: bool = True,
) -> FactorSelectionResult:
    """Select the number of static factors with the Bai & Ng (2002) criteria.

    Parameters
    ----------
    x : DataFrame, ndarray or MixedFrequencyData
        ``T x N`` panel (rows = periods). For :class:`MixedFrequencyData` only series at
        the base frequency are used.
    rmax : int, default 10
        Largest number of factors considered; must satisfy ``1 <= rmax < min(N, T)``.
    criterion : {"IC1", "IC2", "IC3", "PC1", "PC2", "PC3"}, default "IC2"
        Criterion defining ``r_star`` (all six are computed anyway). Variants such as
        ``"ICp2"`` / ``"IC_p2"`` are accepted.
    standardize : bool, default True
        Standardise each series before principal components (recommended; the
        criteria are not scale invariant).
    missing : {"drop_rows", "drop_columns", "raise"}, default "drop_rows"
        How missing values are removed to obtain a balanced panel (a
        :class:`~nowcastbox.core.exceptions.DataQualityWarning` reports any drop).
    warn_small_sample : bool, default True
        Emit a :class:`~nowcastbox.core.exceptions.DataQualityWarning` when
        ``min(N, T) <`` :data:`MIN_RELIABLE_DIM` (see the module notes).

    Returns
    -------
    FactorSelectionResult
        ``r_star``, criteria per ``r`` (``r = 0..rmax``), eigenvalues and a plot hook.

    Raises
    ------
    ValueError
        Invalid ``rmax`` or ``criterion``.
    NowcastDataError
        Invalid data (non-finite, constant series, too small after dropping NaN).

    Warns
    -----
    DataQualityWarning
        If ``min(N, T) < 20`` (small-sample unreliability of the criteria).

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> f = rng.normal(size=(200, 2))
    >>> lam = rng.normal(size=(50, 2))
    >>> x = f @ lam.T + rng.normal(size=(200, 50))
    >>> select_factors(x, rmax=8, criterion="IC2").r_star
    2
    """
    crit = _normalize_criterion(criterion)
    if isinstance(rmax, bool) or not isinstance(rmax, (int, np.integer)) or rmax < 1:
        raise ValueError(f"rmax must be a positive integer; got {rmax!r}.")
    rmax = int(rmax)
    panel = prepare_panel(x, standardize=standardize, missing=missing)
    n_periods, n_series = panel.n_periods, panel.n_series
    if rmax >= min(n_series, n_periods):
        raise ValueError(
            f"rmax={rmax} must be smaller than min(N, T) = {min(n_series, n_periods)}."
        )
    if warn_small_sample and min(n_series, n_periods) < MIN_RELIABLE_DIM:
        warnings.warn(
            f"Bai-Ng criteria are unreliable when min(N, T) < {MIN_RELIABLE_DIM} "
            f"(N={n_series}, T={n_periods}); they tend to select rmax. Consider fixing "
            "the number of factors.",
            DataQualityWarning,
            stacklevel=2,
        )
    s = np.linalg.svd(panel.values, compute_uv=False)
    ssr, criteria = factor_criteria(s, n_series, n_periods, rmax)
    by_crit = {c: int(criteria[c].to_numpy().argmin()) for c in CRITERIA}
    eig = s**2 / n_periods
    comp = pd.RangeIndex(1, len(s) + 1, name="component")
    logger.debug("Bai-Ng factor selection: N=%d T=%d -> %s", n_series, n_periods, by_crit)
    return FactorSelectionResult(
        r_star=by_crit[crit],
        criterion=crit,
        criteria=criteria,
        r_star_by_criterion=by_crit,
        ssr=ssr,
        eigenvalues=pd.Series(eig, index=comp, name="eigenvalue"),
        explained_variance_ratio=pd.Series(
            s**2 / np.sum(s**2), index=comp, name="explained_variance_ratio"
        ),
        rmax=rmax,
        n_series=n_series,
        n_periods=n_periods,
        columns=[str(c) for c in panel.columns],
        dropped_rows=panel.dropped_rows,
        dropped_columns=[str(c) for c in panel.dropped_columns],
    )
