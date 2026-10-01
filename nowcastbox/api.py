r"""High-level one-call nowcasting API (plan §6.2).

:func:`nowcast` chains the main building blocks of the library: optional preprocessing
(:func:`~nowcastbox.preprocessing.prepare_panel`), selection of the number of factors
with the Bai & Ng (2002) information criteria
(:func:`~nowcastbox.selection.select_factors`) and, optionally, of the number of
dynamic shocks (Bai & Ng, 2007; :func:`~nowcastbox.selection.select_shocks`),
estimation of either the two-step DFM (Giannone, Reichlin & Small, 2008) or the
mixed-frequency DFM estimated by EM (Bańbura & Modugno, 2014; with the robust and
time-varying-mean options of innovations I3/I4), and optionally a density nowcast
(innovation I5, :func:`~nowcastbox.density.nowcast_distribution`).

References
----------
Bai, J. & Ng, S. (2002). Determining the number of factors in approximate factor
models. *Econometrica*, 70(1), 191-221.

Bai, J. & Ng, S. (2007). Determining the number of primitive shocks in factor models.
*Journal of Business & Economic Statistics*, 25(1), 52-60.

Giannone, D., Reichlin, L. & Small, D. (2008). Nowcasting: The real-time informational
content of macroeconomic data. *Journal of Monetary Economics*, 55(4), 665-676.

Bańbura, M. & Modugno, M. (2014). Maximum likelihood estimation of factor models on
datasets with arbitrary pattern of missing data. *Journal of Applied Econometrics*,
29(1), 133-160.
"""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
import pandas as pd

from nowcastbox._logging import get_logger
from nowcastbox.core.data import FrequencySpec, MixedFrequencyData, as_mixed_frequency_data
from nowcastbox.core.exceptions import DataQualityWarning, NowcastDataError
from nowcastbox.core.formula import resolve_target
from nowcastbox.core.frequency import is_fixed_ratio
from nowcastbox.core.results import NowcastResults
from nowcastbox.density import nowcast_distribution
from nowcastbox.models import MixedFreqDFM, TwoStepDFM
from nowcastbox.preprocessing import apply_transforms, prepare_panel
from nowcastbox.selection import (
    MIN_RELIABLE_DIM,
    FactorSelectionResult,
    ShockSelectionResult,
    select_factors,
    select_shocks,
)

__all__ = [
    "DENSITY_COLUMNS",
    "DENSITY_LEVELS",
    "METHODS",
    "add_density",
    "nowcast",
    "select_n_factors",
]

logger = get_logger(__name__)

_MIN_RELIABLE_DIM = MIN_RELIABLE_DIM
"""Below this ``min(N, T)`` the Bai-Ng criteria are known to over-select (Bai & Ng, 2002)."""

METHODS: tuple[str, ...] = ("em", "two_step")
"""Estimation methods accepted by :func:`nowcast`."""

DENSITY_LEVELS: tuple[float, ...] = (0.68, 0.9)
"""Interval levels of the density columns added by ``nowcast(..., density=True)``."""

DENSITY_COLUMNS: tuple[str, ...] = (
    "std",
    "median",
    "lower_68",
    "upper_68",
    "lower_90",
    "upper_90",
)
"""Columns of ``results.nowcast`` (re)written by ``nowcast(..., density=True)``."""

_EM_OPTIONS = ("idiosyncratic", "long_run_mean", "outliers")
"""Estimator options of :func:`nowcast` that only apply to ``method="em"``."""

_METHOD_ALIASES: dict[str, str] = {
    "em": "em",
    "mixedfreqdfm": "em",
    "mixed_freq_dfm": "em",
    "bm": "em",
    "two_step": "two_step",
    "twostep": "two_step",
    "twostepdfm": "two_step",
    "2s": "two_step",
}


def _normalize_method(method: str) -> str:
    """Canonical method name (raises ``ValueError``)."""
    key = str(method).strip().lower().replace("-", "_")
    if key not in _METHOD_ALIASES:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}.")
    return _METHOD_ALIASES[key]


def select_n_factors(
    panel: MixedFrequencyData,
    predictors: list[str],
    *,
    rmax: int = 8,
    criterion: str = "IC2",
) -> tuple[int, FactorSelectionResult]:
    """Number of factors for :func:`nowcast` from the Bai & Ng (2002) criteria.

    The criteria are computed on the base-frequency ``predictors`` (rows with missing
    values, e.g. the ragged edge, are dropped silently). ``rmax`` is capped at ``min(N, T) - 1`` and the selected
    number is floored at one, since a nowcasting model needs at least one factor; a
    :class:`~nowcastbox.core.exceptions.DataQualityWarning` reports the floor. The
    criteria tend to select ``rmax`` when ``min(N, T) < 20`` (Bai & Ng, 2002, Monte
    Carlo with ``N = 10``); a warning is emitted in that case.

    Parameters
    ----------
    panel : MixedFrequencyData
        Panel holding the predictors.
    predictors : list of str
        Series used for the selection (only those at the base frequency are used).
    rmax : int, default 8
        Largest number of factors considered.
    criterion : str, default "IC2"
        Bai-Ng criterion (``"IC1"``-``"IC3"``, ``"PC1"``-``"PC3"``).

    Returns
    -------
    n_factors : int
        Selected number of factors (at least one).
    selection : FactorSelectionResult
        Full output of :func:`~nowcastbox.selection.select_factors`.

    Raises
    ------
    NowcastDataError
        If fewer than two base-frequency predictors are available.

    Examples
    --------
    >>> from nowcastbox.models.two_step import simulate_two_step_example
    >>> data = simulate_two_step_example(random_state=0)
    >>> preds = [c for c in data.columns if c != "gdp"]
    >>> import warnings
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")  # N = 10 < 20: small-N warning
    ...     r, sel = select_n_factors(data, preds, rmax=4)
    >>> r >= 1 and sel.rmax <= 4
    True
    """
    base = [c for c in predictors if panel.metadata[c].frequency == panel.base_frequency]
    if len(base) < 2:
        raise NowcastDataError(
            "Selecting the number of factors needs at least two base-frequency predictors, "
            f"got {base}. Pass n_factors explicitly."
        )
    complete = _complete_rows(panel, base)
    n_rows = int(complete.shape[0])
    cap = min(len(base), n_rows) - 1
    if cap < 1:
        raise NowcastDataError(
            "Too few complete rows of the predictors to select the number of factors; "
            "pass n_factors explicitly."
        )
    if min(len(base), n_rows) < _MIN_RELIABLE_DIM:
        warnings.warn(
            f"Bai-Ng criteria are unreliable when min(N, T) < {_MIN_RELIABLE_DIM} "
            f"(N={len(base)}, T={n_rows}); they tend to select rmax. Consider passing "
            "n_factors explicitly.",
            DataQualityWarning,
            stacklevel=3,
        )
    selection = select_factors(
        complete, rmax=min(int(rmax), cap), criterion=criterion, warn_small_sample=False
    )
    n_factors = int(selection.r_star)
    if n_factors < 1:
        warnings.warn(
            f"Bai-Ng criterion {selection.criterion} selected 0 factors; using 1.",
            DataQualityWarning,
            stacklevel=3,
        )
        n_factors = 1
    return n_factors, selection


def _prepare(
    panel: MixedFrequencyData,
    target: str,
    transform: Any,
    preprocess: bool | Mapping[str, Any],
) -> MixedFrequencyData:
    """Apply the requested preprocessing (target always kept)."""
    if preprocess is False:
        if transform is None:
            return panel
        out = apply_transforms(panel, transform)
        assert isinstance(out, MixedFrequencyData)  # noqa: S101 - panel in, panel out
        return out
    options: dict[str, Any] = {} if preprocess is True else dict(preprocess)
    keep = options.pop("keep", None)
    clean_target = bool(options.pop("clean_target", False))
    keep_list = [keep] if isinstance(keep, str) else list(keep or [])
    out = prepare_panel(panel, transform, keep=[*keep_list, target], **options)
    assert isinstance(out, MixedFrequencyData)  # noqa: S101
    if clean_target:
        return out
    return _restore_target(panel, out, target, transform, options)


def _restore_target(
    panel: MixedFrequencyData,
    cleaned: MixedFrequencyData,
    target: str,
    transform: Any,
    options: Mapping[str, Any],
) -> MixedFrequencyData:
    """Put the transformed but *uncleaned* target back into a prepared panel.

    Outlier replacement and gap filling are meant for the predictors; applied to the
    target they would also change the values the nowcast is evaluated against (e.g. the
    2020 GDP contraction would be winsorised and reported as ``observed``).
    """
    raw = prepare_panel(
        panel,
        transform,
        keep=[target],
        replace_outliers=False,
        replace_na=False,
        max_na_prop=None,
        aggregate=False,
        drop_leading_empty=bool(options.get("drop_leading_empty", True)),
    )
    assert isinstance(raw, MixedFrequencyData)  # noqa: S101
    frame = cleaned.data
    frame[target] = raw.data[target].reindex(frame.index)
    return cleaned.with_data(frame)


def _build_model(
    method: str,
    n_factors: Any,
    factor_lags: int,
    n_shocks: int | None,
    blocks: Any,
    horizon: int,
    model_kwargs: dict[str, Any],
) -> MixedFreqDFM | TwoStepDFM:
    """Instantiate the estimator for ``method``."""
    if method == "em":
        if n_shocks is not None:
            raise ValueError("n_shocks is only used by method='two_step'.")
        return MixedFreqDFM(
            n_factors=n_factors, factor_lags=factor_lags, blocks=blocks, **model_kwargs
        )
    if blocks is not None:
        raise ValueError("blocks are only supported by method='em'.")
    return TwoStepDFM(
        n_factors=n_factors,
        factor_lags=factor_lags,
        n_shocks=n_shocks,
        horizon=horizon,
        **model_kwargs,
    )


def nowcast(
    data: MixedFrequencyData | pd.DataFrame,
    target: str,
    method: Literal["em", "two_step"] | str = "em",
    *,
    n_factors: int | Mapping[str, int] | None = None,
    factor_lags: int = 1,
    n_shocks: int | Literal["auto"] | None = None,
    blocks: Any = None,
    horizon: int = 1,
    frequency: FrequencySpec | None = None,
    transform: Any = None,
    preprocess: bool | Mapping[str, Any] = False,
    rmax: int = 8,
    criterion: str = "IC2",
    idiosyncratic: str | None = None,
    long_run_mean: str | None = None,
    outliers: str | None = None,
    density: bool = False,
    n_boot: int = 0,
    random_state: int | np.random.Generator | None = None,
    **model_kwargs: Any,
) -> NowcastResults:
    """Nowcast ``target`` in one call: (preprocess) -> select r -> estimate.

    Parameters
    ----------
    data : MixedFrequencyData or pandas.DataFrame
        Panel on its base grid: monthly (quarterly values in the third month) or, for
        ``method="em"``, weekly or daily with calendar-aware aggregation of the monthly
        and quarterly series (innovation I1).
    target : str
        Target series, or a formula selecting the predictors (``"gdp ~ ."``,
        ``"gdp ~ ip + pmi"``).
    method : {"em", "two_step"}, default "em"
        ``"em"``: :class:`~nowcastbox.models.MixedFreqDFM` (Bańbura & Modugno, 2014);
        ``"two_step"``: :class:`~nowcastbox.models.TwoStepDFM` (Giannone, Reichlin &
        Small, 2008).
    n_factors : int or mapping, optional
        Number of factors (per block for ``method="em"`` with a mapping). ``None``:
        Bai & Ng (2002) criterion ``criterion`` on the base-frequency predictors when
        there are no blocks; one factor per block when ``blocks`` are given.
    factor_lags : int, default 1
        Order of the factor VAR.
    n_shocks : int or "auto", optional
        Number of dynamic shocks of the two-step model (default: ``n_factors``);
        ``"auto"`` uses the Bai & Ng (2007) statistic.
    blocks : optional
        Block structure for ``method="em"`` (see :class:`~nowcastbox.models.MixedFreqDFM`;
        ``"data"`` uses the ``blocks`` metadata of the panel; ``"auto"`` uses it when
        the panel has block metadata and a single global block otherwise).
    horizon : int, default 1
        Additional target periods forecast after the current one.
    frequency : frequency specification, optional
        Per-series native frequencies for DataFrame input (inferred when omitted).
    transform : optional
        Transformations applied before estimation (see
        :func:`~nowcastbox.preprocessing.apply_transforms`). ``None`` keeps the data
        as given (metadata transformations are applied only through ``preprocess``).
    preprocess : bool or mapping, default False
        Run :func:`~nowcastbox.preprocessing.prepare_panel` (outliers, interior gaps,
        sparse series); a mapping passes options to it. The target is always kept and,
        unless the mapping sets ``clean_target=True``, only transformed: its outliers
        and gaps are left as observed, so that e.g. the 2020 GDP contraction is not
        winsorised.
    rmax : int, default 8
        Largest number of factors considered by the Bai-Ng selection.
    criterion : str, default "IC2"
        Bai-Ng criterion used when ``n_factors`` is ``None``.
    idiosyncratic : {"ar1", "iid", "student_t"}, optional
        Idiosyncratic specification of ``method="em"`` (``"student_t"``: robust
        Student-t errors, innovation I3). ``None`` keeps the estimator default.
    long_run_mean : {"constant", "time_varying"}, optional
        ``"time_varying"``: random-walk long-run mean of the target (innovation I4,
        ``method="em"`` only).
    outliers : {"none", "auto"}, optional
        ``"auto"``: automatic outlier detection inside the EM (I3, ``method="em"``).
    density : bool, default False
        Add a density nowcast (innovation I5): the predictive distribution of
        :func:`~nowcastbox.density.nowcast_distribution` (filtering uncertainty plus,
        with ``n_boot > 0``, parameter uncertainty from a bootstrap with
        re-estimation) overwrites the columns :data:`DENSITY_COLUMNS` (``std``,
        ``median`` and the 68 %/90 % bounds) of ``results.nowcast`` for the
        out-of-sample periods; the distribution is stored in
        ``info["distribution"]``.
    n_boot : int, default 0
        Bootstrap replications of the density nowcast (``0``: Gaussian, filtering
        uncertainty only).
    random_state : int or numpy.random.Generator, optional
        Seed of the bootstrap.
    **model_kwargs
        Further estimator arguments (e.g. ``max_iter``, ``tol``, ``covid``, ``df`` for
        EM; ``aggregate`` for the two-step model).

    Returns
    -------
    NowcastResults
        :class:`~nowcastbox.models.MixedFreqDFMResults` or
        :class:`~nowcastbox.models.TwoStepResults`. ``info["selection"]`` holds the
        :class:`~nowcastbox.selection.FactorSelectionResult` (and
        ``info["shock_selection"]`` the shock selection) when they were run;
        ``info["method"]`` the estimation method; ``info["distribution"]`` the
        :class:`~nowcastbox.density.NowcastDistribution` with ``density=True``.

    Raises
    ------
    ValueError
        On an unknown ``method`` or options that do not apply to it.
    NowcastDataError
        If the data cannot support the model (or, with ``density=True``, the
        results have no standard deviation for any out-of-sample period).

    Examples
    --------
    >>> import nowcastbox as nb
    >>> from nowcastbox.models.two_step import simulate_two_step_example
    >>> data = simulate_two_step_example(random_state=0)
    >>> res = nb.nowcast(data, target="gdp", method="two_step", n_factors=1)
    >>> res.info["method"], res.n_factors
    ('two_step', 1)
    >>> dens = nb.nowcast(data, target="gdp", method="two_step", n_factors=1, density=True)
    >>> {"median", "lower_90", "upper_90"} <= set(dens.nowcast.columns)
    True
    """
    method_name = _normalize_method(method)
    em_options = _em_options(method_name, idiosyncratic, long_run_mean, outliers)
    _check_options(n_shocks, density, n_boot)
    panel = as_mixed_frequency_data(data, frequency)
    _check_calendar_support(method_name, panel)
    target_name, predictors = resolve_target(target, panel.columns)
    panel = _prepare(panel, target_name, transform, preprocess)
    predictors = tuple(p for p in predictors if p in panel.columns)
    if not predictors:
        raise NowcastDataError(f"No predictors of {target_name!r} left after preprocessing.")

    if isinstance(blocks, str) and blocks == "auto":
        blocks = "data" if len(panel.block_names) > 0 else None

    extra_info: dict[str, Any] = {"method": method_name}
    if n_factors is None:
        if blocks is not None:
            n_factors = 1
        else:
            n_factors, selection = select_n_factors(
                panel, list(predictors), rmax=rmax, criterion=criterion
            )
            extra_info["selection"] = selection
    if n_shocks == "auto":
        if method_name != "two_step" or not isinstance(n_factors, int):
            raise ValueError("n_shocks='auto' requires method='two_step' and an integer r.")
        shock_sel = _select_shocks(panel, list(predictors), n_factors, factor_lags)
        n_shocks = max(1, int(shock_sel.q_star)) if shock_sel is not None else None
        extra_info["shock_selection"] = shock_sel

    model = _build_model(
        method_name,
        n_factors,
        factor_lags,
        n_shocks,
        blocks,
        horizon,
        {**model_kwargs, **em_options},
    )
    formula = f"{target_name} ~ " + " + ".join(predictors)
    logger.info("nowcast: method=%s n_factors=%s target=%s", method_name, n_factors, target_name)
    if method_name == "em":
        results = model.fit(panel, target=formula, horizon=horizon)
    else:
        results = model.fit(panel, target=formula)
    results = dataclasses.replace(results, info={**results.info, **extra_info})
    if density:
        results = add_density(results, n_boot=n_boot, random_state=random_state)
    return results


def _check_calendar_support(method: str, panel: MixedFrequencyData) -> None:
    """Reject weekly/daily (calendar) panels for the two-step model (explicit error)."""
    if method != "two_step":
        return
    base = panel.base_frequency
    calendar = sorted(
        {
            m.frequency.label
            for m in panel.metadata.values()
            if not is_fixed_ratio(base, m.frequency)
        }
    )
    if calendar:
        raise ValueError(
            f"method='two_step' needs a fixed number of {base.label} periods per period of "
            f"every series; {calendar} series on a {base.label} grid need calendar-aware "
            "aggregation: use method='em' (innovation I1)."
        )


def _check_options(n_shocks: object, density: bool, n_boot: int) -> None:
    """Validate the options that do not depend on the data (raises ``ValueError``)."""
    if isinstance(n_shocks, str) and n_shocks != "auto":
        raise ValueError(f"n_shocks must be an integer, None or 'auto', got {n_shocks!r}.")
    if n_boot and not density:
        raise ValueError("n_boot is only used with density=True.")


def _em_options(
    method: str, idiosyncratic: str | None, long_run_mean: str | None, outliers: str | None
) -> dict[str, str]:
    """EM-only estimator options given explicitly (``ValueError`` for other methods)."""
    values = (idiosyncratic, long_run_mean, outliers)
    options = {k: v for k, v in zip(_EM_OPTIONS, values, strict=True) if v is not None}
    if options and method != "em":
        raise ValueError(f"{sorted(options)} only apply to method='em'.")
    return options


def add_density(
    results: NowcastResults,
    *,
    n_boot: int = 0,
    random_state: int | np.random.Generator | None = None,
    **kwargs: Any,
) -> NowcastResults:
    """Overwrite the uncertainty columns of ``results.nowcast`` with a density nowcast.

    Parameters
    ----------
    results : NowcastResults
        Fitted results with a ``std`` column.
    n_boot : int, default 0
        Bootstrap replications (``0``: Gaussian filtering uncertainty only).
    random_state : int or numpy.random.Generator, optional
        Seed of the bootstrap.
    **kwargs
        Further options of :func:`~nowcastbox.density.nowcast_distribution`.

    Returns
    -------
    NowcastResults
        Copy whose ``nowcast`` has the columns :data:`DENSITY_COLUMNS` taken from the
        predictive distribution on its periods (``NaN`` elsewhere for ``median``), and
        ``info["distribution"]`` the :class:`~nowcastbox.density.NowcastDistribution`.

    Raises
    ------
    NowcastDataError
        If no out-of-sample period has a standard deviation.

    Examples
    --------
    >>> from nowcastbox.models import TwoStepDFM
    >>> from nowcastbox.models.two_step import simulate_two_step_example
    >>> res = TwoStepDFM(n_factors=1).fit(simulate_two_step_example(random_state=0), "gdp")
    >>> out = add_density(res)
    >>> float(out.nowcast["median"].dropna().iloc[-1]) == float(
    ...     res.nowcast["out_of_sample"].iloc[-1]
    ... )
    True
    """
    dist = nowcast_distribution(results, n_boot=n_boot, random_state=random_state, **kwargs)
    bands = dist.to_frame(levels=DENSITY_LEVELS)
    frame = results.nowcast.copy()
    for column in DENSITY_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
        frame.loc[bands.index, column] = bands[column].to_numpy(dtype=float)
    info = {**results.info, "distribution": dist, "density_n_boot": int(n_boot)}
    return dataclasses.replace(results, nowcast=frame, info=info)


def _select_shocks(
    panel: MixedFrequencyData, predictors: list[str], n_factors: int, factor_lags: int
) -> ShockSelectionResult | None:
    """Bai-Ng (2007) shock selection on the base-frequency predictors (``None`` if r=1)."""
    if n_factors < 2:
        return None
    base = [c for c in predictors if panel.metadata[c].frequency == panel.base_frequency]
    complete = _complete_rows(panel, base)
    return select_shocks(complete, n_factors=n_factors, factor_lags=factor_lags)


def _complete_rows(panel: MixedFrequencyData, columns: list[str]) -> pd.DataFrame:
    """Rows of ``columns`` without missing values (the ragged edge is expected)."""
    return panel.select(columns).to_frame().dropna()
