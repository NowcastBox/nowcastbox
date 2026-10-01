"""Combined diagnostics of a fitted dynamic factor model (innovation I9).

:func:`run_diagnostics` gathers, for fitted results (:class:`~nowcastbox.models.TwoStepResults`,
:class:`~nowcastbox.models.MixedFreqDFMResults` or any
:class:`~nowcastbox.core.results.NowcastResults` with factors):

1. loading stability tests (Breitung & Eickmeier, 2011),
2. EM convergence diagnostics (log-likelihood path),
3. factor contributions and commonalities,
4. a data-quality report,
5. residual diagnostics (Ljung-Box, Jarque-Bera) of the idiosyncratic and bridge
   residuals,

returned as tidy DataFrames in a :class:`DiagnosticsReport` with a text
:meth:`~DiagnosticsReport.summary`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import pandas as pd

from nowcastbox._logging import get_logger
from nowcastbox.core.data import MixedFrequencyData
from nowcastbox.core.exceptions import NowcastDataError
from nowcastbox.core.results import NowcastResults
from nowcastbox.diagnostics._common import align_factors, as_panel, factor_block_map, to_native
from nowcastbox.diagnostics.contribution import (
    FactorContribution,
    factor_contributions,
    projection_residuals,
)
from nowcastbox.diagnostics.convergence import (
    ConvergenceDiagnostics,
    em_convergence,
    loglikelihood_path,
)
from nowcastbox.diagnostics.data_quality import DataQualityReport, data_quality_report
from nowcastbox.diagnostics.residuals import ResidualDiagnostics, residual_diagnostics
from nowcastbox.diagnostics.stability import (
    LoadingStabilityResult,
    StabilityStatistic,
    loading_stability_test,
)

__all__ = [
    "COMPONENTS",
    "DiagnosticsReport",
    "effective_loadings",
    "idiosyncratic_residuals",
    "run_diagnostics",
]

logger = get_logger(__name__)

COMPONENTS: tuple[str, ...] = (
    "data_quality",
    "convergence",
    "stability",
    "contribution",
    "residuals",
)
"""Diagnostics computed by :func:`run_diagnostics`."""


@dataclass(frozen=True, eq=False)
class DiagnosticsReport:
    """Diagnostics of a fitted nowcasting model.

    Parameters
    ----------
    model_name : str
        Estimator name.
    target : str
        Target series.
    data_quality : DataQualityReport or None
        Missing data, ragged edge, outliers, weak loadings, publication delays.
    convergence : ConvergenceDiagnostics or None
        EM log-likelihood path (``None`` for non-iterative estimators).
    stability : LoadingStabilityResult or None
        Breitung-Eickmeier loading-stability tests.
    contribution : FactorContribution or None
        Variance shares and commonalities.
    residuals : ResidualDiagnostics or None
        Ljung-Box / Jarque-Bera tests of the residuals.
    notes : list of str
        Components skipped and caveats.

    Examples
    --------
    >>> from nowcastbox.models import TwoStepDFM
    >>> from nowcastbox.models.two_step import simulate_two_step_example
    >>> from nowcastbox.diagnostics import run_diagnostics
    >>> data = simulate_two_step_example(random_state=0)
    >>> res = TwoStepDFM(n_factors=1).fit(data, "gdp")
    >>> rep = run_diagnostics(res)
    >>> sorted(rep.to_frames())[:2]
    ['bridge_residuals', 'contribution_block_r2']
    """

    model_name: str
    target: str
    data_quality: DataQualityReport | None = None
    convergence: ConvergenceDiagnostics | None = None
    stability: LoadingStabilityResult | None = None
    contribution: FactorContribution | None = None
    residuals: ResidualDiagnostics | None = None
    notes: list[str] = field(default_factory=list[str])

    def series_overview(self) -> pd.DataFrame:
        """One row per series combining the main per-series diagnostics.

        Returns
        -------
        pandas.DataFrame
            Columns (when available): ``na_share``, ``ragged_edge``, ``n_outliers``,
            ``near_zero_loading``, ``r2``, ``break_pvalue_adj``, ``unstable_loadings``,
            ``lb_pvalue``, ``jb_pvalue``, ``flags``.

        Examples
        --------
        >>> rep.series_overview().columns[:3].tolist()  # doctest: +SKIP
        ['na_share', 'ragged_edge', 'n_outliers']
        """
        parts: list[pd.DataFrame] = []
        if self.data_quality is not None:
            dq = self.data_quality.table
            parts.append(dq[["na_share", "ragged_edge", "n_outliers", "near_zero_loading"]])
        if self.contribution is not None:
            parts.append(self.contribution.r2[["r2"]])
        if self.stability is not None:
            st = self.stability.table
            parts.append(
                pd.DataFrame(
                    {"break_pvalue_adj": st["pvalue_adj"], "unstable_loadings": st["reject"]}
                )
            )
        if self.residuals is not None:
            rt = self.residuals.table
            parts.append(rt.loc[rt["kind"] != "bridge", ["lb_pvalue", "jb_pvalue"]])
        if not parts:
            return pd.DataFrame()
        out = pd.concat(parts, axis=1)
        out["flags"] = [self._flags(out.loc[s]) for s in out.index]
        out.index.name = "series"
        return out

    def _flags(self, row: pd.Series) -> str:
        flags = []
        if self.data_quality is not None:
            dq_flags = self.data_quality.table["flags"]
            text = dq_flags.get(row.name, "")
            if isinstance(text, str) and text:
                flags.append(text)
        unstable = row.get("unstable_loadings", np.nan)
        if pd.notna(unstable) and bool(unstable):
            flags.append("unstable_loadings")
        alpha = self.residuals.alpha if self.residuals is not None else 0.05
        lb = row.get("lb_pvalue", np.nan)
        if pd.notna(lb) and float(lb) < alpha:
            flags.append("autocorrelated_residuals")
        return ";".join(flags)

    @property
    def flagged_series(self) -> list[str]:
        """Series with at least one diagnostic flag."""
        ov = self.series_overview()
        if ov.empty:
            return []
        return [str(s) for s in ov.index[ov["flags"] != ""]]

    def to_frames(self) -> dict[str, pd.DataFrame]:
        """All diagnostics as tidy DataFrames.

        Returns
        -------
        dict of str to pandas.DataFrame
            Keys among ``data_quality``, ``outliers``, ``publication_delays``,
            ``convergence_path``, ``convergence``, ``stability``,
            ``stability_multiple_testing``, ``contribution_shares``,
            ``contribution_r2``, ``contribution_block_r2``, ``contribution_by_factor``,
            ``contribution_by_block``, ``residuals``, ``bridge_residuals``,
            ``series_overview``.

        Examples
        --------
        >>> "series_overview" in rep.to_frames()  # doctest: +SKIP
        True
        """
        out: dict[str, pd.DataFrame] = {}
        if self.data_quality is not None:
            out["data_quality"] = self.data_quality.table.copy()
            out["outliers"] = self.data_quality.outliers.copy()
            out["publication_delays"] = self.data_quality.publication.copy()
        if self.convergence is not None:
            out["convergence_path"] = self.convergence.to_frame()
            out["convergence"] = self.convergence.summary_frame()
        if self.stability is not None:
            out["stability"] = self.stability.to_frame()
            out["stability_multiple_testing"] = self.stability.multiple_testing()
        if self.contribution is not None:
            c = self.contribution
            out["contribution_shares"] = c.shares.copy()
            out["contribution_r2"] = c.r2.copy()
            out["contribution_block_r2"] = c.block_r2.copy()
            out["contribution_by_factor"] = c.by_factor()
            out["contribution_by_block"] = c.by_block()
        if self.residuals is not None:
            t = self.residuals.table
            out["residuals"] = t.loc[t["kind"] != "bridge"].copy()
            out["bridge_residuals"] = t.loc[t["kind"] == "bridge"].copy()
        out["series_overview"] = self.series_overview()
        return out

    def summary(self) -> str:
        """Text report of every computed diagnostic.

        Returns
        -------
        str
            Multi-section summary.

        Examples
        --------
        >>> print(rep.summary())  # doctest: +SKIP
        """
        bar = "=" * 78
        header = f"Diagnostics: {self.model_name or 'model'} (target: {self.target})"
        sections = [bar, header, bar]
        for part in (
            self.data_quality,
            self.convergence,
            self.stability,
            self.contribution,
            self.residuals,
        ):
            if part is not None:
                sections += [part.summary(), "-" * 78]
        flagged = self.flagged_series
        sections.append(
            f"Flagged series ({len(flagged)}): "
            + (", ".join(flagged[:15]) + (" ..." if len(flagged) > 15 else "") or "-")
        )
        if self.notes:
            sections.append("Notes:")
            sections += [f"  - {n}" for n in self.notes]
        sections.append(bar)
        return "\n".join(sections)

    def __str__(self) -> str:
        return self.summary()


# ====================================================================== residuals
def _model_common(results: Any, panel: MixedFrequencyData) -> pd.DataFrame | None:
    """Common component (+ mean) of the model series in original units, if stored."""
    common = getattr(results, "common_component", None)
    if isinstance(common, pd.DataFrame):
        return common
    x_fc = getattr(results, "x_forecast", None)
    if isinstance(x_fc, pd.DataFrame) and getattr(results, "aggregate", None) == "factors":
        base = panel.base_frequency
        cols = [c for c in x_fc.columns if c in panel.columns]
        return x_fc[[c for c in cols if panel.metadata[c].frequency == base]]
    return None


def _filtered_residuals(results: Any, panel: MixedFrequencyData) -> dict[str, pd.Series]:
    r"""Residuals of ``TwoStepDFM(aggregate="variables")`` on the filtered predictors.

    The state-space model observes :math:`\tilde x_{i,t} = \sum_j w_{i,j} x_{i,t-j}`, so
    the idiosyncratic residuals are the filtered observations minus the fitted signal
    (``x_forecast``) on the base-frequency predictors.
    """
    x_fc = getattr(results, "x_forecast", None)
    filter_panel = getattr(results, "filter_panel", None)
    if not (
        isinstance(x_fc, pd.DataFrame)
        and getattr(results, "aggregate", None) == "variables"
        and getattr(results, "model_data", None) is not None
        and callable(filter_panel)
    ):
        return {}
    try:
        filtered_panel = filter_panel(panel)
    except (NowcastDataError, ValueError):
        return {}
    filtered = cast("MixedFrequencyData", filtered_panel).to_frame()
    base = panel.base_frequency
    names = [
        c
        for c in x_fc.columns
        if c in filtered.columns and c in panel.columns and panel.metadata[c].frequency == base
    ]
    fitted = x_fc.reindex(filtered.index)
    return {c: (filtered[c] - fitted[c]).dropna().rename(c) for c in names}


def idiosyncratic_residuals(
    results: NowcastResults, data: MixedFrequencyData | pd.DataFrame | None = None
) -> tuple[dict[str, pd.Series], dict[str, str]]:
    """Idiosyncratic residuals of each series on its native grid.

    Uses ``x - common component`` when the results store the common component
    (:class:`~nowcastbox.models.MixedFreqDFMResults`; ``TwoStepResults`` with
    ``aggregate="factors"`` for base-frequency series; with ``aggregate="variables"``
    the filtered predictors minus the fitted signal), and the residuals of the
    projection of each standardised series on the factors otherwise.

    Parameters
    ----------
    results : NowcastResults
        Fitted results with ``factors``.
    data : MixedFrequencyData or pandas.DataFrame, optional
        Panel (defaults to ``results.data``).

    Returns
    -------
    residuals : dict of str to pandas.Series
        Residuals (observed periods, native index).
    source : dict of str to str
        ``"model"`` or ``"projection"`` for each series.

    Raises
    ------
    NowcastDataError
        If no panel or factors are available.

    Examples
    --------
    >>> from nowcastbox.models import MixedFreqDFM
    >>> from nowcastbox.models.two_step import simulate_two_step_example
    >>> from nowcastbox.diagnostics import idiosyncratic_residuals
    >>> data = simulate_two_step_example(random_state=0)
    >>> res = MixedFreqDFM(max_iter=20).fit(data, "gdp")
    >>> resid, source = idiosyncratic_residuals(res)
    >>> source["gdp"], str(resid["gdp"].index.freqstr)
    ('model', 'Q-DEC')
    """
    panel = _resolve_data(results, data)
    common = _model_common(results, panel)
    out: dict[str, pd.Series] = _filtered_residuals(results, panel)
    source: dict[str, str] = dict.fromkeys(out, "model")
    if common is not None:
        frame = panel.to_frame()
        aligned = common.reindex(panel.index)
        for name in [c for c in common.columns if c in panel.columns]:
            resid = to_native((frame[name] - aligned[name]).rename(name), panel)
            out[name] = resid.dropna()
            source[name] = "model"
    rest = [c for c in panel.columns if c not in out]
    if rest:
        if results.factors is None:
            raise NowcastDataError("The results contain no factors.")
        proj = projection_residuals(panel, results.factors, series=rest)
        for name in rest:
            out[name] = proj[name]
            source[name] = "projection"
    return {c: out[c] for c in panel.columns}, {c: source[c] for c in panel.columns}


# ====================================================================== entry point
def _resolve_data(
    results: NowcastResults, data: MixedFrequencyData | pd.DataFrame | None
) -> MixedFrequencyData:
    if data is not None:
        return as_panel(data)
    if results.data is None:
        raise NowcastDataError("Pass data: the results do not store the estimation panel.")
    return results.data


def _check_components(components: Iterable[str] | None) -> tuple[str, ...]:
    if components is None:
        return COMPONENTS
    chosen = tuple(components)
    unknown = sorted(set(chosen) - set(COMPONENTS))
    if unknown:
        raise ValueError(f"Unknown components {unknown}; choose from {COMPONENTS}.")
    return chosen


def _model_weights(results: Any) -> dict[str, Any] | None:
    params = getattr(results, "params", None)
    weights = params.get("aggregation_weights") if isinstance(params, dict) else None
    return dict(weights) if isinstance(weights, dict) else None


def effective_loadings(results: NowcastResults) -> pd.DataFrame | None:
    r"""Loadings on the aggregated factors (response to a persistent factor shift).

    For a series observed at a lower frequency with aggregation weights :math:`w` and
    restricted loadings :math:`\lambda_l = (w_l / w_0)\lambda_0` on the lagged factors
    (``MixedFreqDFM``), the total loading is :math:`\lambda_0 \sum_l w_l / w_0`; the
    stored ``loadings`` only hold :math:`\lambda_0`.

    Parameters
    ----------
    results : NowcastResults
        Fitted results.

    Returns
    -------
    pandas.DataFrame or None
        Rescaled copy of ``results.loadings`` (unchanged without aggregation weights).

    Examples
    --------
    >>> from nowcastbox.models import MixedFreqDFM
    >>> from nowcastbox.models.two_step import simulate_two_step_example
    >>> from nowcastbox.diagnostics.report import effective_loadings
    >>> res = MixedFreqDFM(max_iter=10).fit(simulate_two_step_example(random_state=0), "gdp")
    >>> ratio = effective_loadings(res).loc["gdp"] / res.loadings.loc["gdp"]
    >>> float(ratio.iloc[0])
    9.0
    """
    if results.loadings is None:
        return None
    out = results.loadings.copy()
    for name, w in (_model_weights(results) or {}).items():
        w = np.asarray(w, dtype=np.float64)
        if name in out.index and w.size and w[0] != 0:
            out.loc[name] = out.loc[name] * float(w.sum() / w[0])
    return out


def _model_factor_blocks(results: Any, factors: pd.DataFrame) -> dict[str, list[str]]:
    layout = getattr(results, "state_layout", None)
    names = [str(c) for c in factors.columns]
    if layout is not None and sum(layout.n_factors) == len(names):
        out, start = {}, 0
        for block, r in zip(layout.block_names, layout.n_factors, strict=True):
            out[str(block)] = names[start : start + r]
            start += r
        return out
    return factor_block_map(names, [])


@dataclass
class _Options:
    break_date: Any
    trim: float
    ar_lags: int
    statistic: StabilityStatistic
    alpha: float
    correction: str
    lags: int | None
    outlier_threshold: float
    loading_tol: float
    max_na_share: float
    warn: bool


def _run_convergence(
    results: Any, opts: _Options, notes: list[str]
) -> ConvergenceDiagnostics | None:
    if loglikelihood_path(results) is None:
        notes.append("EM convergence: no log-likelihood path (non-iterative estimator).")
        return None
    return em_convergence(results, warn=opts.warn)


def _run_factor_parts(
    results: NowcastResults,
    panel: MixedFrequencyData,
    opts: _Options,
    chosen: tuple[str, ...],
) -> tuple[LoadingStabilityResult | None, FactorContribution | None]:
    assert results.factors is not None  # noqa: S101
    factors = align_factors(results.factors, panel.index)
    weights = _model_weights(results)
    paths = getattr(results, "params", {}).get("aggregation_weight_paths")
    if weights is not None and isinstance(paths, dict):
        weights.update(paths)  # calendar aggregation: per-period weights
    stability = None
    contribution = None
    if "stability" in chosen:
        stability = loading_stability_test(
            panel,
            factors,
            break_date=opts.break_date,
            trim=opts.trim,
            ar_lags=opts.ar_lags,
            statistic=opts.statistic,
            alpha=opts.alpha,
            correction=opts.correction,
            weights=weights,
        )
    if "contribution" in chosen:
        contribution = factor_contributions(
            panel,
            factors,
            factor_blocks=_model_factor_blocks(results, factors),
            weights=weights,
        )
    return stability, contribution


def _run_residuals(
    results: NowcastResults, panel: MixedFrequencyData, opts: _Options
) -> ResidualDiagnostics | None:
    resid: dict[str, pd.Series] = {}
    kinds: dict[str, str] = {}
    if results.factors is not None:
        resid, source = idiosyncratic_residuals(results, panel)
        kinds = {n: "idiosyncratic" if s == "model" else "projection" for n, s in source.items()}
    bridge = getattr(results, "bridge", None)
    if bridge is not None:
        name = f"{results.target} (bridge)"
        resid[name] = bridge.residuals.rename(name)
        kinds[name] = "bridge"
    if not resid:
        return None
    return residual_diagnostics(resid, lags=opts.lags, alpha=opts.alpha, kind=kinds)


def run_diagnostics(
    results: NowcastResults,
    data: MixedFrequencyData | pd.DataFrame | None = None,
    *,
    break_date: Any = None,
    trim: float = 0.15,
    ar_lags: int = 0,
    statistic: StabilityStatistic = "lm",
    alpha: float = 0.05,
    correction: str = "holm",
    lags: int | None = None,
    outlier_threshold: float = 4.0,
    loading_tol: float = 0.1,
    max_na_share: float = 0.5,
    components: Iterable[str] | None = None,
    warn: bool = True,
) -> DiagnosticsReport:
    """Run the dynamic-factor-model diagnostics (innovation I9) on fitted results.

    Parameters
    ----------
    results : NowcastResults
        Fitted results (``TwoStepResults``, ``MixedFreqDFMResults``, ...).
    data : MixedFrequencyData or pandas.DataFrame, optional
        Panel to diagnose (defaults to ``results.data``, the estimation panel).
    break_date : period-like, optional
        Known break date of the loading-stability tests (first period of the new
        regime); ``None`` runs sup tests over the trimmed sample.
    trim : float, default 0.15
        Trimming of the sup tests.
    ar_lags : int, default 0
        Residual lags in the stability regressions.
    statistic : {"lm", "wald", "lr"}, default "lm"
        Stability statistic used for decisions.
    alpha : float, default 0.05
        Significance level of every test.
    correction : {"holm", "bonferroni", "fdr_bh", "none"}, default "holm"
        Multiple-testing correction of the stability tests.
    lags : int, optional
        Ljung-Box lags (default ``min(10, n // 5)``).
    outlier_threshold : float, default 4.0
        IQR multiple of the outlier rule.
    loading_tol : float, default 0.1
        Near-zero loading threshold (standardised units).
    max_na_share : float, default 0.5
        Missing-share threshold of the data-quality flags.
    components : iterable of str, optional
        Subset of :data:`COMPONENTS` (default: all).
    warn : bool, default True
        Emit :class:`~nowcastbox.core.exceptions.ConvergenceWarning` /
        :class:`~nowcastbox.core.exceptions.DataQualityWarning` for problems found.

    Returns
    -------
    DiagnosticsReport
        Tidy diagnostics and a text summary.

    Raises
    ------
    TypeError
        If ``results`` is not a :class:`~nowcastbox.core.results.NowcastResults`.
    ValueError
        On invalid options.
    NowcastDataError
        If no panel is available.

    Examples
    --------
    >>> from nowcastbox.models import MixedFreqDFM
    >>> from nowcastbox.models.two_step import simulate_two_step_example
    >>> from nowcastbox.diagnostics import run_diagnostics
    >>> data = simulate_two_step_example(random_state=0)
    >>> res = MixedFreqDFM(n_factors=1, max_iter=50).fit(data, "gdp")
    >>> rep = run_diagnostics(res)
    >>> rep.convergence.converged, rep.stability.test
    (True, 'sup')
    >>> "Loading stability" in rep.summary()
    True
    """
    if not isinstance(results, NowcastResults):
        raise TypeError(f"results must be a NowcastResults, got {type(results).__name__}.")
    chosen = _check_components(components)
    panel = _resolve_data(results, data)
    opts = _Options(
        break_date,
        trim,
        ar_lags,
        statistic,
        alpha,
        correction,
        lags,
        outlier_threshold,
        loading_tol,
        max_na_share,
        warn,
    )
    notes: list[str] = []
    parts: dict[str, Any] = {}
    if "data_quality" in chosen:
        parts["data_quality"] = data_quality_report(
            panel,
            loadings=effective_loadings(results),
            outlier_threshold=outlier_threshold,
            loading_tol=loading_tol,
            max_na_share=max_na_share,
            warn=warn,
        )
    if "convergence" in chosen:
        parts["convergence"] = _run_convergence(results, opts, notes)
    factor_parts = {"stability", "contribution", "residuals"} & set(chosen)
    if factor_parts and results.factors is None:
        bridge_only = "residuals" in chosen and getattr(results, "bridge", None) is not None
        notes.append(
            f"{', '.join(sorted(factor_parts))}: the results contain no factors"
            + (" (only the bridge residuals are tested)." if bridge_only else ".")
        )
    elif factor_parts - {"residuals"}:
        parts["stability"], parts["contribution"] = _run_factor_parts(results, panel, opts, chosen)
    if "residuals" in chosen:
        parts["residuals"] = _run_residuals(results, panel, opts)
    if getattr(results, "aggregate", None) == "variables":
        notes.append(
            "TwoStepDFM(aggregate='variables'): the factors summarise the filtered "
            "predictors; stability and contribution use the unfiltered series, the "
            "idiosyncratic residuals the filtered ones."
        )
    logger.debug("Diagnostics computed: %s", sorted(k for k, v in parts.items() if v is not None))
    return DiagnosticsReport(
        model_name=results.model_name,
        target=results.target,
        notes=notes,
        **parts,
    )
