r"""Contribution of each factor to each series (variance shares and commonality).

Each standardised series :math:`x_i` is projected on the estimated factors
(aggregated with the series' weights for lower-frequency series, as in the
measurement equation of mixed-frequency factor models):

.. math:: x_{it} = c_i + \sum_k \beta_{ik} \hat F_{kt} + e_{it}.

The *commonality* of the series is the :math:`R^2` of this regression, the share of
its variance explained by the common factors (Stock & Watson, 2002; Bai & Ng, 2006,
use the same measure to relate observed series to estimated factors). With
:math:`c_{ikt} = \beta_{ik}(\hat F_{kt} - \bar F_k)` and :math:`c_{it} = \sum_k
c_{ikt}`, the variance share of factor :math:`k` is

.. math:: s_{ik} = \frac{\operatorname{Cov}(c_{ik}, c_i)}{\operatorname{Var}(x_i)},
          \qquad \sum_k s_{ik} = R_i^2,

which equals :math:`\operatorname{Var}(c_{ik})/\operatorname{Var}(x_i)` for orthogonal
factors (principal components) and splits the covariance terms evenly otherwise (a
share can then be negative). Block commonalities are the :math:`R^2` of the
projection on the factors of one block (Bańbura, Giannone & Reichlin, 2011, block
structure).

References
----------
Bai, J., & Ng, S. (2006). Evaluating latent and observed factors in macroeconomics and
finance. *Journal of Econometrics*, 131(1-2), 507-537.

Bańbura, M., Giannone, D., & Reichlin, L. (2011). Nowcasting. In *Oxford Handbook of
Economic Forecasting*, 193-224.

Stock, J. H., & Watson, M. W. (2002). Macroeconomic forecasting using diffusion
indexes. *Journal of Business & Economic Statistics*, 20(2), 147-162.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from nowcastbox.core.data import FrequencySpec, MixedFrequencyData
from nowcastbox.core.exceptions import NowcastDataError
from nowcastbox.diagnostics._common import (
    FloatArray,
    align_factors,
    as_panel,
    factor_block_map,
    series_design,
    series_weights,
)

__all__ = ["FactorContribution", "factor_contributions", "projection_residuals"]


@dataclass(frozen=True)
class _Projection:
    index: pd.PeriodIndex
    y: FloatArray
    beta: FloatArray
    shares: FloatArray
    r2: float
    adj_r2: float
    residuals: FloatArray


def _project(index: pd.PeriodIndex, y: FloatArray, X: FloatArray) -> _Projection:
    n, r = X.shape
    if n < r + 2:
        nan = np.full(r, np.nan)
        return _Projection(index, y, nan, nan, np.nan, np.nan, np.full(n, np.nan))
    Xc = X - X.mean(axis=0)
    yc = y - y.mean()
    beta, *_ = np.linalg.lstsq(Xc, yc, rcond=None)
    contrib = Xc * beta
    common = contrib.sum(axis=1)
    sst = float(yc @ yc)
    resid = yc - common
    if sst <= 0:
        nan = np.full(r, np.nan)
        return _Projection(index, y, beta, nan, np.nan, np.nan, resid)
    shares = contrib.T @ common / sst
    r2 = 1.0 - float(resid @ resid) / sst
    adj = 1.0 - (1.0 - r2) * (n - 1) / (n - r - 1) if n > r + 1 else np.nan
    return _Projection(index, y, beta, shares, r2, adj, resid)


@dataclass(frozen=True, eq=False)
class FactorContribution:
    """Variance shares of the factors and commonality of each series.

    Parameters
    ----------
    shares : pandas.DataFrame
        Long table: ``series``, ``factor``, ``block``, ``loading`` (projection
        coefficient, standardised units) and ``share`` (:math:`s_{ik}`).
    r2 : pandas.DataFrame
        Indexed by series: ``frequency``, ``n_obs``, ``r2`` (commonality), ``adj_r2``.
    block_r2 : pandas.DataFrame
        Long table: ``series``, ``block``, ``r2`` (projection on the block factors) and
        ``member`` (whether the series belongs to the block in the panel metadata;
        always True without block metadata).
    factor_blocks : dict of str to list of str
        Factors of each block.

    Examples
    --------
    >>> from nowcastbox.models.two_step import simulate_two_step_example
    >>> from nowcastbox.diagnostics import factor_contributions
    >>> from nowcastbox.diagnostics._common import pca_factors
    >>> data = simulate_two_step_example(random_state=0)
    >>> fc = factor_contributions(data, pca_factors(data, 1))
    >>> bool((fc.r2["r2"] > 0.3).all())
    True
    """

    shares: pd.DataFrame
    r2: pd.DataFrame
    block_r2: pd.DataFrame
    factor_blocks: dict[str, list[str]]

    def share_matrix(self) -> pd.DataFrame:
        """Variance shares as a series x factor matrix.

        Returns
        -------
        pandas.DataFrame
            Rows: series, columns: factors.

        Examples
        --------
        >>> fc.share_matrix().shape  # doctest: +SKIP
        (11, 1)
        """
        wide = self.shares.pivot_table(index="series", columns="factor", values="share")
        return wide.reindex(
            index=self.r2.index, columns=[f for fs in self.factor_blocks.values() for f in fs]
        )

    def by_factor(self) -> pd.DataFrame:
        """Average variance share of each factor across series.

        Returns
        -------
        pandas.DataFrame
            Indexed by factor: ``block``, ``mean_share``, ``median_share``,
            ``max_share``.

        Examples
        --------
        >>> fc.by_factor().columns.tolist()  # doctest: +SKIP
        ['block', 'mean_share', 'median_share', 'max_share']
        """
        g = self.shares.groupby("factor", sort=False)
        out = pd.DataFrame(
            {
                "block": g["block"].first(),
                "mean_share": g["share"].mean(),
                "median_share": g["share"].median(),
                "max_share": g["share"].max(),
            }
        )
        out.index.name = "factor"
        return out

    def by_block(self) -> pd.DataFrame:
        """Commonality of each block over its member series.

        Returns
        -------
        pandas.DataFrame
            Indexed by block: ``n_series`` (members), ``mean_r2`` (projection on the
            block factors, members only) and ``mean_total_r2`` (all factors, members).

        Examples
        --------
        >>> fc.by_block().columns.tolist()  # doctest: +SKIP
        ['n_series', 'mean_r2', 'mean_total_r2']
        """
        rows = {}
        for block, frame in self.block_r2.groupby("block", sort=False):
            members = frame.loc[frame["member"].to_numpy(bool)]
            total = self.r2["r2"].reindex(members["series"].to_numpy())
            rows[block] = {
                "n_series": len(members),
                "mean_r2": float(members["r2"].mean()) if len(members) else np.nan,
                "mean_total_r2": float(total.mean()) if len(members) else np.nan,
            }
        out = pd.DataFrame.from_dict(rows, orient="index")
        out.index.name = "block"
        return out

    def summary(self, n_series: int = 3) -> str:
        """Text summary.

        Parameters
        ----------
        n_series : int, default 3
            Number of series with the highest / lowest commonality to list.

        Returns
        -------
        str
            Overview.

        Examples
        --------
        >>> print(fc.summary())  # doctest: +SKIP
        """
        r2 = self.r2["r2"].dropna().sort_values()
        fmt = ", ".join
        lines = [
            "Factor contribution (commonality)",
            f"  {'Mean R2':<22}{r2.mean():.3f} (median {r2.median():.3f})",
            f"  {'Highest R2':<22}"
            + fmt(f"{s} ({v:.2f})" for s, v in r2.iloc[::-1][:n_series].items()),
            f"  {'Lowest R2':<22}" + fmt(f"{s} ({v:.2f})" for s, v in r2[:n_series].items()),
        ]
        for factor, row in self.by_factor().iterrows():
            lines.append(f"  {'Share ' + str(factor):<22}mean {row['mean_share']:.3f}")
        if len(self.factor_blocks) > 1:
            for block, row in self.by_block().iterrows():
                lines.append(
                    f"  {'Block ' + str(block):<22}mean R2 {row['mean_r2']:.3f} "
                    f"({int(row['n_series'])} series)"
                )
        return "\n".join(lines)


def _blocks_of(panel: MixedFrequencyData, factor_blocks: Mapping[str, list[str]]) -> pd.DataFrame:
    """Series x block membership (all True when the metadata has no such block)."""
    meta = panel.blocks
    out = {}
    for block in factor_blocks:
        if block in meta.columns:
            out[block] = meta[block].reindex(panel.columns).fillna(False).astype(bool)
        else:
            out[block] = pd.Series(True, index=panel.columns)
    return pd.DataFrame(out, index=panel.columns)


def _check_factor_blocks(
    factor_blocks: Mapping[str, Sequence[str]] | None,
    F: pd.DataFrame,
    panel: MixedFrequencyData,
) -> dict[str, list[str]]:
    if factor_blocks is None:
        return factor_block_map([str(c) for c in F.columns], panel.block_names)
    out = {str(b): [str(f) for f in fs] for b, fs in factor_blocks.items()}
    unknown = sorted({f for fs in out.values() for f in fs} - {str(c) for c in F.columns})
    if unknown or not out or any(not fs for fs in out.values()):
        raise ValueError(
            f"factor_blocks must map blocks to non-empty lists of factor columns; "
            f"unknown factors {unknown}."
        )
    return out


def _projections(
    data: MixedFrequencyData | pd.DataFrame,
    factors: pd.DataFrame,
    series: Sequence[str] | None,
    weights: Mapping[str, Any] | None,
    frequency: FrequencySpec | None,
) -> tuple[MixedFrequencyData, pd.DataFrame, dict[str, tuple[_Projection, FloatArray]]]:
    panel = as_panel(data, frequency)
    names = list(panel.columns if series is None else series)
    unknown = [s for s in names if s not in panel.columns]
    if unknown:
        raise NowcastDataError(f"Unknown series {unknown}.")
    F = align_factors(factors, panel.index)
    w = series_weights(panel, names, weights)
    out = {}
    for name in names:
        idx, y, X = series_design(panel, F, name, w[name])
        out[name] = (_project(idx, y, X), X)
    return panel, F, out


def factor_contributions(
    data: MixedFrequencyData | pd.DataFrame,
    factors: pd.DataFrame,
    *,
    factor_blocks: Mapping[str, Sequence[str]] | None = None,
    series: Sequence[str] | None = None,
    weights: Mapping[str, Any] | None = None,
    frequency: FrequencySpec | None = None,
) -> FactorContribution:
    """Variance share of each factor in each series and commonality per series/block.

    Parameters
    ----------
    data : MixedFrequencyData or pandas.DataFrame
        Panel on a base-frequency PeriodIndex.
    factors : pandas.DataFrame
        Estimated factors on the base grid (e.g. ``results.factors``).
    factor_blocks : mapping of str to list of str, optional
        Factor columns of each block. Default: from ``"<block>_f<k>"`` column names
        matching the panel blocks, otherwise a single block ``"all"``.
    series : sequence of str, optional
        Series to analyse (default: all).
    weights : mapping, optional
        Aggregation weights per series (see
        :func:`~nowcastbox.diagnostics.loading_stability_test`).
    frequency : frequency specification, optional
        Frequencies of a DataFrame ``data``.

    Returns
    -------
    FactorContribution
        Tidy tables of shares and commonalities.

    Raises
    ------
    ValueError
        On an invalid ``factor_blocks``.
    NowcastDataError
        On unusable data or factors.

    Examples
    --------
    >>> from nowcastbox.models import TwoStepDFM
    >>> from nowcastbox.models.two_step import simulate_two_step_example
    >>> from nowcastbox.diagnostics import factor_contributions
    >>> data = simulate_two_step_example(random_state=0)
    >>> res = TwoStepDFM(n_factors=1).fit(data, "gdp")
    >>> fc = factor_contributions(data, res.factors)
    >>> float(fc.r2["r2"].mean()) > 0.5
    True
    """
    panel, F, proj = _projections(data, factors, series, weights, frequency)
    blocks = _check_factor_blocks(factor_blocks, F, panel)
    block_of = {f: b for b, fs in blocks.items() for f in fs}
    membership = _blocks_of(panel, blocks)
    cols = [str(c) for c in F.columns]
    share_rows, r2_rows, block_rows = [], [], []
    for name, (p, X) in proj.items():
        r2_rows.append(
            {
                "series": name,
                "frequency": panel.metadata[name].frequency.value,
                "n_obs": int(p.y.size),
                "r2": p.r2,
                "adj_r2": p.adj_r2,
            }
        )
        for k, factor in enumerate(cols):
            share_rows.append(
                {
                    "series": name,
                    "factor": factor,
                    "block": block_of.get(factor, "all"),
                    "loading": float(p.beta[k]),
                    "share": float(p.shares[k]),
                }
            )
        for block, fs in blocks.items():
            pos = [cols.index(f) for f in fs]
            r2_b = p.r2 if len(blocks) == 1 else _project(p.index, p.y, X[:, pos]).r2
            block_rows.append(
                {
                    "series": name,
                    "block": block,
                    "r2": r2_b,
                    "member": bool(membership.loc[name, block]),
                }
            )
    return FactorContribution(
        shares=pd.DataFrame(share_rows),
        r2=pd.DataFrame(r2_rows).set_index("series"),
        block_r2=pd.DataFrame(block_rows),
        factor_blocks=blocks,
    )


def projection_residuals(
    data: MixedFrequencyData | pd.DataFrame,
    factors: pd.DataFrame,
    *,
    series: Sequence[str] | None = None,
    weights: Mapping[str, Any] | None = None,
    frequency: FrequencySpec | None = None,
) -> dict[str, pd.Series]:
    """Residuals of the projection of each standardised series on the factors.

    Parameters
    ----------
    data : MixedFrequencyData or pandas.DataFrame
        Panel.
    factors : pandas.DataFrame
        Factors on the base grid.
    series : sequence of str, optional
        Series (default: all).
    weights : mapping, optional
        Aggregation weights per series.
    frequency : frequency specification, optional
        Frequencies of a DataFrame ``data``.

    Returns
    -------
    dict of str to pandas.Series
        Residuals on each series' native PeriodIndex (observed periods only).

    Examples
    --------
    >>> from nowcastbox.models.two_step import simulate_two_step_example
    >>> from nowcastbox.diagnostics import projection_residuals
    >>> from nowcastbox.diagnostics._common import pca_factors
    >>> data = simulate_two_step_example(random_state=0)
    >>> res = projection_residuals(data, pca_factors(data, 1))
    >>> str(res["gdp"].index.freqstr)
    'Q-DEC'
    """
    panel, _, proj = _projections(data, factors, series, weights, frequency)
    out = {}
    for name, (p, _) in proj.items():
        freq = panel.metadata[name].frequency
        idx = p.index.asfreq(freq.pandas_freq)
        out[name] = pd.Series(p.residuals, index=idx, name=name)
    return out
