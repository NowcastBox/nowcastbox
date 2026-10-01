"""Diagnostics of dynamic factor models (innovation I9).

- :func:`loading_stability_test` - Breitung & Eickmeier (2011) LM/Wald/LR tests for
  breaks in the factor loadings, at a known date or sup tests over a trimmed range
  (Andrews, 1993), with a multiple-testing summary;
- :func:`em_convergence` - log-likelihood path, relative changes, non-monotone steps;
- :func:`factor_contributions` - variance share of each factor in each series and
  commonality (R2) per series and block;
- :func:`data_quality_report` - missing data, ragged edge, outliers, near-zero
  loadings, publication delays;
- :func:`residual_diagnostics` - Ljung-Box and Jarque-Bera tests of idiosyncratic and
  bridge residuals;
- :func:`run_diagnostics` - everything above for fitted results, in a
  :class:`DiagnosticsReport` with tidy DataFrames and ``summary()``.

Examples
--------
>>> from nowcastbox.models import TwoStepDFM
>>> from nowcastbox.models.two_step import simulate_two_step_example
>>> from nowcastbox.diagnostics import run_diagnostics
>>> data = simulate_two_step_example(random_state=0)
>>> report = run_diagnostics(TwoStepDFM(n_factors=1).fit(data, "gdp"))
>>> report.stability.table.shape[0]
11
"""

from nowcastbox.diagnostics._common import adjust_pvalues
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
from nowcastbox.diagnostics.report import (
    COMPONENTS,
    DiagnosticsReport,
    effective_loadings,
    idiosyncratic_residuals,
    run_diagnostics,
)
from nowcastbox.diagnostics.residuals import (
    ResidualDiagnostics,
    jarque_bera,
    ljung_box,
    residual_diagnostics,
)
from nowcastbox.diagnostics.stability import (
    LoadingStabilityResult,
    andrews_critical_values,
    loading_stability_test,
    sup_break_pvalue,
)

__all__ = [
    "COMPONENTS",
    "ConvergenceDiagnostics",
    "DataQualityReport",
    "DiagnosticsReport",
    "FactorContribution",
    "LoadingStabilityResult",
    "ResidualDiagnostics",
    "adjust_pvalues",
    "andrews_critical_values",
    "data_quality_report",
    "effective_loadings",
    "em_convergence",
    "factor_contributions",
    "idiosyncratic_residuals",
    "jarque_bera",
    "ljung_box",
    "loading_stability_test",
    "loglikelihood_path",
    "projection_residuals",
    "residual_diagnostics",
    "run_diagnostics",
    "sup_break_pvalue",
]
