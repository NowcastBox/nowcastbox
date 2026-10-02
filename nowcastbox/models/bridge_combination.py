r"""Combination of many small bridge equations (thick modelling).

Following Bańbura, Belousova, Bodnár & Tóth (2023, ECB Working Paper 2815), the target
is nowcast by **every** bridge equation (Baffigi, Golinelli & Parigi, 2004; Diron,
2008) with :math:`1, \dots, m` higher-frequency (e.g. monthly) indicators and
:math:`0, \dots, q` indicators of the target frequency (e.g. quarterly),

.. math::

    y_t = \alpha_e + \sum_{k \in S_e} \sum_{l=0}^{L} \beta_{e,k,l}\, \bar x^{(k)}_{t-l}
          + \sum_{j=1}^{J} \phi_{e,j}\, y_{t-j} + u_{e,t},
    \qquad e = 1, \dots, E,

and the :math:`E` predictions are pooled. The indicators are completed up to the end of
the forecast horizon **once per fit** by a pluggable extrapolator
(:mod:`nowcastbox.models.extrapolation`; iterated AR forecasts by default), aggregated to
the target frequency and shared by all equations, which are then estimated by batched
OLS (:mod:`nowcastbox.models._bridge_batch`). Combinations:

* ``"mean"`` - equal weights (the default of Bańbura et al., 2023);
* ``"median"`` - cross-equation median;
* ``"inverse_mse"`` - weights :math:`w_e \propto 1/\mathrm{MSE}_e` (Stock & Watson,
  2004; Timmermann, 2006), the MSE being the (optionally discounted) mean squared
  in-sample residual or one-step pseudo out-of-sample error.

``trim`` discards the worst equations by MSE before combining (Timmermann, 2006).

The number of equations is
:math:`\sum_{a=m_0}^{m}\binom{N_m}{a}\cdot\sum_{b=0}^{q}\binom{N_q}{b}`
(:func:`bridge_equation_count`).

References
----------
Bańbura, M., Belousova, I., Bodnár, K. & Tóth, M. B. (2023). Nowcasting employment in
the euro area. ECB Working Paper No. 2815.

Stock, J. H. & Watson, M. W. (2004). Combination forecasts of output growth in a
seven-country data set. *Journal of Forecasting*, 23(6), 405-430.

Timmermann, A. (2006). Forecast combinations. In *Handbook of Economic Forecasting*,
vol. 1, 135-196. Elsevier.
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from math import comb
from typing import Any

import numpy as np
import pandas as pd

from nowcastbox._logging import get_logger
from nowcastbox.core.base import BaseNowcaster
from nowcastbox.core.data import FrequencySpec, MixedFrequencyData
from nowcastbox.core.exceptions import DataQualityWarning, NowcastDataError
from nowcastbox.core.frequency import AggregationType, Frequency
from nowcastbox.core.results import NowcastResults, build_nowcast_frame
from nowcastbox.models._bridge_batch import (
    batched_ols,
    design_tensor,
    one_step_errors,
    recursive_predict,
    training_mask,
)
from nowcastbox.models.bridge import aggregate_to_target, resolve_aggregation_weights
from nowcastbox.models.extrapolation import Extrapolator, make_extrapolator, native_until

__all__ = [
    "BridgeCombination",
    "BridgeCombinationResults",
    "bridge_equation_count",
]

logger = get_logger(__name__)

_COMBINE = ("mean", "median", "inverse_mse")
_MSE = ("in_sample", "out_of_sample")
_BY = ("equation", "indicator")
_CHUNK = 2048


# ---------------------------------------------------------------------- helpers
def bridge_equation_count(
    n_monthly: int,
    n_quarterly: int = 0,
    max_monthly: int = 2,
    max_quarterly: int = 1,
    min_monthly: int = 1,
    *,
    include_empty: bool = False,
) -> int:
    r"""Number of bridge equations of a :class:`BridgeCombination`.

    Parameters
    ----------
    n_monthly, n_quarterly : int
        Numbers of higher-frequency and target-frequency indicators.
    max_monthly, max_quarterly, min_monthly : int
        Size limits of the indicator subsets (see :class:`BridgeCombination`).
    include_empty : bool, default False
        Count the equation without indicators (only possible with target lags).

    Returns
    -------
    int
        :math:`\sum_{a=m_0}^{m}\binom{N_m}{a}\sum_{b=0}^{q}\binom{N_q}{b}`
        (minus the empty equation unless ``include_empty``).

    Examples
    --------
    >>> from nowcastbox.models import bridge_equation_count
    >>> bridge_equation_count(50)
    1275
    >>> bridge_equation_count(10, 3)
    220
    """
    high = sum(comb(n_monthly, a) for a in range(min_monthly, max_monthly + 1))
    low = sum(comb(n_quarterly, b) for b in range(0, max_quarterly + 1))
    empty = int(min_monthly == 0 and not include_empty)
    return high * low - empty


def _check_int(name: str, value: object, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int | np.integer) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}.")
    return int(value)


def _target_periods(data: MixedFrequencyData, freq: Frequency, horizon: int) -> pd.PeriodIndex:
    """Target periods spanned by the panel plus ``horizon`` extra periods."""
    pf = freq.pandas_freq
    return pd.period_range(data.start.asfreq(pf), data.end.asfreq(pf) + horizon, freq=pf)


def _split_predictors(data: MixedFrequencyData, target: str) -> tuple[list[str], list[str]]:
    """Higher-frequency and same-frequency predictors (validated).

    Predictors without any observation in the panel (e.g. series that start after an
    early pseudo real-time vintage) are dropped with a :class:`DataQualityWarning`, so
    that one late series does not stop the whole combination.
    """
    target_freq = data.metadata[target].frequency
    high: list[str] = []
    low: list[str] = []
    empty: list[str] = []
    for col in data.columns:
        if col == target:
            continue
        freq = data.metadata[col].frequency
        if freq.is_lower_than(target_freq):
            raise NowcastDataError(
                f"Predictor {col!r} ({freq.label}) has a lower frequency than the target "
                f"({target_freq.label})."
            )
        if not data[col].notna().any():
            empty.append(col)
            continue
        (high if freq.is_higher_than(target_freq) else low).append(col)
    if empty:
        _warn_dropped(empty, "they have no observations")
    return high, low


def _warn_dropped(names: Sequence[str], reason: str) -> None:
    warnings.warn(
        f"Predictors {list(names)} are left out of the bridge combination: {reason}.",
        DataQualityWarning,
        stacklevel=2,
    )


def _complete_each(
    extrapolator: Callable[..., dict[str, pd.Series]],
    data: MixedFrequencyData,
    columns: Sequence[str],
    end: pd.Period,
    error: NowcastDataError,
) -> dict[str, pd.Series]:
    """Extrapolate series by series, leaving out those the extrapolator rejects."""
    completed: dict[str, pd.Series] = {}
    failed: list[str] = []
    for col in columns:
        try:
            completed.update(extrapolator(data, [col], end))
        except NowcastDataError:
            failed.append(col)
    if not completed:
        raise error
    if failed:
        _warn_dropped(failed, "they cannot be extrapolated (too few observations)")
    return completed


def _enumerate(
    high: Sequence[str], low: Sequence[str], sizes: tuple[int, int, int], allow_empty: bool
) -> list[tuple[tuple[str, ...], tuple[str, ...]]]:
    """All (higher-frequency subset, same-frequency subset) pairs, grouped by size."""
    min_high, max_high, max_low = sizes
    out: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    for n_high in range(min_high, min(max_high, len(high)) + 1):
        for n_low in range(0, min(max_low, len(low)) + 1):
            if n_high + n_low == 0 and not allow_empty:
                continue
            for h in combinations(high, n_high):
                out.extend((h, lo) for lo in combinations(low, n_low))
    return out


def _chunks(columns: list[list[int]]) -> Iterator[tuple[int, int]]:
    """Consecutive runs of equations with the same number of coefficients."""
    start = 0
    while start < len(columns):
        k = len(columns[start])
        stop = start + 1
        while stop < len(columns) and stop - start < _CHUNK and len(columns[stop]) == k:
            stop += 1
        yield start, stop
        start = stop


def _discounted_mse(errors: np.ndarray, positions: np.ndarray, discount: float) -> np.ndarray:
    """Discounted mean squared error of each row (NaN errors ignored)."""
    if positions.size == 0:
        return np.full(errors.shape[0], np.nan)
    w = float(discount) ** (positions.max() - positions).astype(float)
    finite = np.isfinite(errors)
    den = (finite * w[None, :]).sum(axis=1)
    num = (np.where(finite, errors, 0.0) ** 2 * w[None, :]).sum(axis=1)
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)


def _combine(estimates: np.ndarray, weights: np.ndarray, method: str) -> np.ndarray:
    """Combine equation estimates ``(E, T)`` period by period (missing ones skipped)."""
    finite = np.isfinite(estimates)
    out = np.full(estimates.shape[1], np.nan)
    cols = finite.any(axis=0)
    if method == "median":
        out[cols] = np.nanmedian(estimates[:, cols], axis=0)
        return out
    w = weights[:, None] * finite
    den = w.sum(axis=0)
    num = (w * np.where(finite, estimates, 0.0)).sum(axis=0)
    ok = den > 0
    out[ok] = num[ok] / den[ok]
    return out


def _dispersion(estimates: np.ndarray, index: pd.PeriodIndex) -> dict[str, pd.Series]:
    """Cross-equation standard deviation, range and count per period."""
    finite = np.isfinite(estimates)
    cols = finite.any(axis=0)
    stats = {name: np.full(estimates.shape[1], np.nan) for name in ("std", "min", "max")}
    sub = estimates[:, cols]
    stats["std"][cols] = np.nanstd(sub, axis=0)
    stats["min"][cols] = np.nanmin(sub, axis=0)
    stats["max"][cols] = np.nanmax(sub, axis=0)
    out = {f"equation_{k}": pd.Series(v, index=index) for k, v in stats.items()}
    out["n_equations"] = pd.Series(finite.sum(axis=0).astype(float), index=index)
    return out


# ---------------------------------------------------------------------- features
@dataclass(frozen=True)
class _Features:
    """Target-frequency feature matrix shared by all equations."""

    values: np.ndarray
    names: list[str]
    observed: np.ndarray
    positions: dict[str, list[int]]
    fixed: list[int]
    target_lags: list[int]

    def columns(self, regressors: Sequence[str]) -> list[int]:
        """Feature indices of one equation: constant, indicators (with lags), target lags."""
        middle = [p for name in regressors for p in self.positions[name]]
        return [*self.fixed, *middle, *self.target_lags]


def _build_features(
    indicators: pd.DataFrame,
    observed: pd.Series,
    regressor_lags: int,
    target_lags: int,
    add_constant: bool,
) -> _Features:
    cols: list[np.ndarray] = []
    names: list[str] = []
    fixed: list[int] = []
    if add_constant:
        fixed.append(0)
        cols.append(np.ones(len(observed)))
        names.append("const")
    positions: dict[str, list[int]] = {}
    for name in indicators.columns:
        positions[str(name)] = []
        for lag in range(regressor_lags + 1):
            positions[str(name)].append(len(cols))
            cols.append(indicators[name].shift(lag).to_numpy(dtype=float))
            names.append(str(name) if lag == 0 else f"{name}_lag{lag}")
    lag_pos: list[int] = []
    for j in range(1, target_lags + 1):
        lag_pos.append(len(cols))
        cols.append(observed.shift(j).to_numpy(dtype=float))
        names.append(f"{observed.name}_lag{j}")
    return _Features(
        values=np.column_stack(cols),
        names=names,
        observed=observed.to_numpy(dtype=float),
        positions=positions,
        fixed=fixed,
        target_lags=lag_pos,
    )


@dataclass
class _Batch:
    """Estimates of all equations (concatenated over the chunks)."""

    coefficients: list[np.ndarray]
    estimates: np.ndarray
    errors: np.ndarray
    ssr: np.ndarray
    tss: np.ndarray
    n_obs: np.ndarray
    valid: np.ndarray


@dataclass(frozen=True)
class _Fitted:
    """Everything the results need from the estimation step."""

    equations: list[tuple[tuple[str, ...], tuple[str, ...]]]
    columns: list[list[int]]
    names: list[str]
    batch: _Batch
    mse: np.ndarray
    included: np.ndarray
    weights: np.ndarray
    eval_positions: np.ndarray
    started: float


def _fit_all(
    features: _Features,
    columns: list[list[int]],
    eval_positions: np.ndarray,
    oos: bool,
    min_train: int,
) -> _Batch:
    n_eq, n_per = len(columns), len(features.observed)
    batch = _Batch(
        coefficients=[],
        estimates=np.full((n_eq, n_per), np.nan),
        errors=np.full((n_eq, eval_positions.size), np.nan),
        ssr=np.full(n_eq, np.nan),
        tss=np.full(n_eq, np.nan),
        n_obs=np.zeros(n_eq, dtype=int),
        valid=np.zeros(n_eq, dtype=bool),
    )
    y = features.observed
    n_lags = len(features.target_lags)
    for start, stop in _chunks(columns):
        cols = np.asarray(columns[start:stop], dtype=int)
        X = design_tensor(features.values, cols)
        fit = batched_ols(X, y, training_mask(X, y))
        batch.coefficients.extend(fit.beta)
        batch.estimates[start:stop] = recursive_predict(X, fit.beta, y, n_lags)
        if oos:
            minimum = max(min_train, cols.shape[1] + 1)
            batch.errors[start:stop] = one_step_errors(X, y, eval_positions, minimum)
        else:
            batch.errors[start:stop] = fit.residuals[:, eval_positions]
        batch.ssr[start:stop], batch.tss[start:stop] = fit.ssr, fit.tss
        batch.n_obs[start:stop], batch.valid[start:stop] = fit.n_obs, fit.valid
    return batch


# ---------------------------------------------------------------------- results
@dataclass(frozen=True, kw_only=True, eq=False, repr=False)
class BridgeCombinationResults(NowcastResults):
    """Results of :class:`BridgeCombination`.

    The ``nowcast`` frame holds the combined estimate (``observed``, ``in_sample``,
    ``out_of_sample``) plus the cross-equation ``equation_std``, ``equation_min``,
    ``equation_max`` and ``n_equations`` (equations entering each period's combination).

    Parameters
    ----------
    specs : pandas.DataFrame, optional
        One row per equation (index ``eq0001``...): ``spec`` (formula), ``monthly`` and
        ``quarterly`` (indicator tuples), ``n_coefficients``, ``n_obs``, ``rsquared``,
        ``sigma``, ``mse``, ``valid``, ``included`` and ``weight`` (NaN for the median).
    estimates : pandas.DataFrame, optional
        Estimates of every equation (rows: target periods, columns: equation ids).
    indicators : pandas.DataFrame, optional
        Extrapolated indicators aggregated to the target frequency (no lags).
    extrapolated : dict of str to pandas.Series
        Indicators completed by the extrapolator, on their native grids.

    Notes
    -----
    Every field of :class:`~nowcastbox.core.results.NowcastResults` is also accepted
    (keyword-only).

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from nowcastbox.models import BridgeCombination
    >>> rng = np.random.default_rng(0)
    >>> idx = pd.period_range("2000-01", periods=90, freq="M")
    >>> x = rng.standard_normal((90, 3))
    >>> y = pd.Series(x[:, 0], index=idx).rolling(3).mean().where(idx.month % 3 == 0)
    >>> df = pd.DataFrame({"a": x[:, 0], "b": x[:, 1], "c": x[:, 2], "y": y}, index=idx)
    >>> res = BridgeCombination().fit(df, "y", frequency={"a": "M", "b": "M", "c": "M", "y": "Q"})
    >>> res.equations().shape[0]
    6
    """

    specs: pd.DataFrame | None = None
    estimates: pd.DataFrame | None = None
    indicators: pd.DataFrame | None = None
    extrapolated: dict[str, pd.Series] = field(default_factory=dict[str, pd.Series])

    def _tables(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        if self.specs is None or self.estimates is None:
            raise ValueError("These results hold no equation table.")
        return self.specs, self.estimates

    def _period(self, period: pd.Period | str | None) -> pd.Period:
        _, estimates = self._tables()
        if period is not None:
            return pd.Period(period, freq=self.target_frequency.pandas_freq)
        last = self.nowcast["observed"].last_valid_index()
        index = estimates.index
        later = index if last is None else index[index > last]
        if len(later) == 0:
            raise KeyError("No period after the last observation of the target.")
        return later[0]

    @property
    def weights(self) -> pd.Series:
        """Combination weights by equation id (0 for excluded equations).

        Raises
        ------
        ValueError
            If the results hold no equation table.
        """
        specs, _ = self._tables()
        return specs["weight"].copy()

    @property
    def equation_estimates(self) -> pd.DataFrame:
        """Estimates of every equation (periods x equation ids).

        Raises
        ------
        ValueError
            If the results hold no equation table.
        """
        _, estimates = self._tables()
        return estimates.copy()

    def equations(
        self, period: pd.Period | str | None = None, *, included_only: bool = False
    ) -> pd.DataFrame:
        """Specification, fit statistics, weight and estimate of every equation.

        Parameters
        ----------
        period : pandas.Period or str, optional
            Target period of the ``nowcast`` column. Default: the first period after the
            last observation of the target (as :meth:`get_nowcast`).
        included_only : bool, default False
            Keep only the equations entering the combination.

        Returns
        -------
        pandas.DataFrame
            :attr:`specs` plus a ``nowcast`` column, indexed by equation id.

        Raises
        ------
        KeyError
            If ``period`` is not in the estimates.
        ValueError
            If the results hold no equation table.

        Examples
        --------
        >>> import numpy as np, pandas as pd
        >>> from nowcastbox.models import BridgeCombination
        >>> idx = pd.period_range("2000-01", periods=60, freq="M")
        >>> x = np.sin(np.arange(60.0))
        >>> y = pd.Series(x, index=idx).rolling(3).mean().where(idx.month % 3 == 0)
        >>> df = pd.DataFrame({"x": x, "y": y}, index=idx)
        >>> res = BridgeCombination().fit(df, "y", frequency={"x": "M", "y": "Q"})
        >>> res.equations()[["spec", "weight"]].to_dict("records")
        [{'spec': 'y ~ x', 'weight': 1.0}]
        """
        specs, _ = self._tables()
        out = specs.copy()
        out["nowcast"] = self._row(self._period(period))
        return out[out["included"]] if included_only else out

    def _row(self, period: pd.Period) -> np.ndarray:
        """Estimates of every equation for one period."""
        _, estimates = self._tables()
        matches = np.flatnonzero(estimates.index == period)
        if matches.size == 0:
            raise KeyError(f"Period {period} is not in the estimates.")
        return estimates.iloc[int(matches[0])].to_numpy(dtype=float)

    def _effective(self, period: pd.Period) -> pd.DataFrame:
        """Estimates and period-specific (renormalised) weights, indexed by spec."""
        specs, _ = self._tables()
        values = self._row(period)
        weights = specs["weight"].to_numpy(dtype=float) * np.isfinite(values)
        total = np.nansum(weights)
        if total > 0:
            weights = weights / total
        return pd.DataFrame({"estimate": values, "weight": weights}, index=specs["spec"].to_numpy())

    def nowcast_change(
        self,
        previous: BridgeCombinationResults,
        period: pd.Period | str | None = None,
        *,
        by: str = "equation",
    ) -> pd.DataFrame:
        r"""Approximate *news*: revision of the combined nowcast between two vintages.

        Bridge combinations have no state-space form, so the exact news decomposition of
        :meth:`news` is not available. The revision
        :math:`\sum_e w^{new}_e \hat y^{new}_e - \sum_e w^{old}_e \hat y^{old}_e` is split
        into equation contributions :math:`w^{new}_e \hat y^{new}_e - w^{old}_e
        \hat y^{old}_e` (weights renormalised over the equations with an estimate);
        with ``by="indicator"`` each equation's contribution is shared equally by its
        indicators. Contributions mix new data, data revisions and re-estimation and are
        NaN for the median combination.

        Parameters
        ----------
        previous : BridgeCombinationResults
            Results of the earlier vintage.
        period : pandas.Period or str, optional
            Target period (default: this result's current nowcast period).
        by : {"equation", "indicator"}, default "equation"
            Level of the decomposition.

        Returns
        -------
        pandas.DataFrame
            ``by="equation"``: columns ``previous``, ``current``, ``previous_weight``,
            ``current_weight`` and ``contribution`` indexed by equation formula;
            ``by="indicator"``: column ``contribution`` indexed by indicator. The
            contributions sum to ``self.get_nowcast(period) -
            previous.get_nowcast(period)`` (mean and inverse-MSE combinations).

        Raises
        ------
        ValueError
            If ``by`` is invalid or a result holds no equation table.
        KeyError
            If ``period`` is missing from one of the results.

        Examples
        --------
        >>> import numpy as np, pandas as pd
        >>> from nowcastbox.models import BridgeCombination
        >>> rng = np.random.default_rng(3)
        >>> idx = pd.period_range("2000-01", periods=63, freq="M")
        >>> x = rng.standard_normal((63, 2))
        >>> y = pd.Series(x[:, 0], index=idx).rolling(3).mean().where(idx.month % 3 == 0)
        >>> y.iloc[-1] = np.nan
        >>> df = pd.DataFrame({"a": x[:, 0], "b": x[:, 1], "y": y}, index=idx)
        >>> freq = {"a": "M", "b": "M", "y": "Q"}
        >>> old = BridgeCombination().fit(df.iloc[:-1], "y", frequency=freq)
        >>> new = BridgeCombination().fit(df, "y", frequency=freq)
        >>> change = new.nowcast_change(old, by="indicator")
        >>> bool(np.isclose(change["contribution"].sum(), new.get_nowcast() - old.get_nowcast()))
        True
        """
        if by not in _BY:
            raise ValueError(f"by must be one of {_BY}, got {by!r}.")
        key = self._period(period)
        current = self._effective(key)
        before = previous._effective(key)
        frame = pd.DataFrame(
            {
                "previous": before["estimate"],
                "current": current["estimate"],
                "previous_weight": before["weight"],
                "current_weight": current["weight"],
            }
        )
        frame.index.name = "equation"
        new_term = frame["current_weight"].fillna(0.0) * frame["current"].fillna(0.0)
        old_term = frame["previous_weight"].fillna(0.0) * frame["previous"].fillna(0.0)
        frame["contribution"] = new_term - old_term
        if self.model_params.get("combine") == "median":
            frame["contribution"] = np.nan
        if by == "equation":
            return frame
        return self._by_indicator(frame["contribution"], previous)

    def _by_indicator(
        self, contribution: pd.Series, previous: BridgeCombinationResults | None = None
    ) -> pd.DataFrame:
        specs, _ = self._tables()
        members: dict[str, list[str]] = {}
        # equations of the previous vintage only (e.g. a dropped indicator) keep their own
        # indicators; the current table takes precedence for common equations
        for table in (previous.specs if previous is not None else None, specs):
            if table is None:
                continue
            for spec, m, q in zip(table["spec"], table["monthly"], table["quarterly"], strict=True):
                members[str(spec)] = list(m) + list(q) or ["(target lags)"]
        rows: dict[str, float] = {}
        for spec, value in contribution.items():
            names = members.get(str(spec), ["(other)"])
            for name in names:
                rows[name] = rows.get(name, 0.0) + float(value) / len(names)
        out = pd.DataFrame({"contribution": pd.Series(rows, dtype=float)})
        out.index.name = "indicator"
        return out

    def _summary_sections(self) -> list[tuple[str, list[str]]]:
        sections = super()._summary_sections()
        if self.specs is not None:
            specs = self.specs
            lines = [
                f"  {'Equations':<22}{len(specs)}",
                f"  {'Estimable':<22}{int(specs['valid'].sum())}",
                f"  {'Combined':<22}{int(specs['included'].sum())}",
                f"  {'Combination':<22}{self.model_params.get('combine', '-')}",
            ]
            top = specs[specs["included"]].sort_values("mse").head(5)
            lines.append("  Lowest MSE:")
            lines += [
                f"    {s:<38}mse={v:.4f}" for s, v in zip(top["spec"], top["mse"], strict=True)
            ]
            sections.append(("Bridge combination", lines))
        return sections


# ---------------------------------------------------------------------- estimator
class BridgeCombination(BaseNowcaster):
    r"""Nowcast by combining all small bridge equations (Bańbura et al., 2023).

    Every subset of ``min_monthly..max_monthly`` higher-frequency indicators and
    ``0..max_quarterly`` target-frequency indicators defines a bridge equation (as in
    :class:`~nowcastbox.models.BridgeEquation`, with the same aggregation, lags and
    iterated target lags); all of them are estimated by batched OLS and their
    predictions combined. Indicators are extrapolated to the end of the horizon once per
    fit, so the cost is dominated by the (vectorised) regressions: 50 monthly
    indicators give 1,275 equations.

    Parameters
    ----------
    max_monthly : int, default 2
        Largest number of higher-frequency (e.g. monthly) indicators per equation.
    max_quarterly : int, default 1
        Largest number of target-frequency (e.g. quarterly) indicators per equation.
    min_monthly : int, default 1
        Smallest number of higher-frequency indicators per equation (0 allows
        equations with target-frequency indicators or target lags only).
    target_lags : int, default 0
        Autoregressive lags of the target in every equation (iterated when unobserved).
    regressor_lags : int, default 0
        Lags (target periods) of each aggregated indicator.
    aggregation : str, AggregationType or sequence of float, default "average"
        Aggregation of higher-frequency indicators (see
        :class:`~nowcastbox.models.BridgeEquation`).
    add_constant : bool, default True
        Include an intercept in every equation.
    combine : {"mean", "median", "inverse_mse"}, default "mean"
        Combination of the equation predictions.
    mse : {"in_sample", "out_of_sample"}, default "in_sample"
        Accuracy measure behind ``inverse_mse`` and ``trim``: mean squared in-sample
        residual, or one-step pseudo out-of-sample error (each equation re-estimated
        recursively on the periods before each evaluation period of the vintage).
    mse_window : int, optional
        Use only the last ``mse_window`` periods with an observed target (default:
        all).
    discount : float, default 1.0
        Discount factor :math:`\delta \in (0, 1]` of past squared errors (Stock &
        Watson, 2004): the error :math:`h` periods before the last one has weight
        :math:`\delta^h`.
    min_train : int, default 8
        Minimum estimation sample of the pseudo out-of-sample errors.
    trim : float, default 0.0
        Share of the estimable equations with the largest MSE discarded before
        combining, in :math:`[0, 1)`.
    extrapolation : str, callable or None, default "ar"
        Completion of the indicators up to the end of the horizon: a registered name
        (:func:`~nowcastbox.models.extrapolation.available_extrapolators`), a callable
        following :class:`~nowcastbox.models.extrapolation.Extrapolator`, or None (no
        completion: periods with incomplete indicators get no estimate). Indicators the
        extrapolator rejects (too few observations) and indicators without any
        observation are left out with a
        :class:`~nowcastbox.core.exceptions.DataQualityWarning`.
    ar_lags : int, default 1
        AR order of the ``"ar"`` extrapolation.
    extrapolation_options : mapping, optional
        Extra options of a named extrapolator.
    horizon : int, default 1
        Target periods forecast after the one containing the last base period.

    See Also
    --------
    nowcastbox.benchmarks.BridgeCombinationBenchmark : The same model as a benchmark.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from nowcastbox.models import BridgeCombination
    >>> rng = np.random.default_rng(1)
    >>> idx = pd.period_range("2000-01", periods=120, freq="M")
    >>> x = rng.standard_normal((120, 4))
    >>> y = 0.5 + pd.Series(x[:, :2].sum(axis=1), index=idx).rolling(3).mean()
    >>> y = y.where(idx.month % 3 == 0)
    >>> y.iloc[-1] = np.nan
    >>> df = pd.DataFrame(x, index=idx, columns=["a", "b", "c", "d"]).assign(y=y)
    >>> freq = {"a": "M", "b": "M", "c": "M", "d": "M", "y": "Q"}
    >>> res = BridgeCombination(combine="inverse_mse").fit(df, "y", frequency=freq)
    >>> res.info["n_equations"]
    10
    >>> res.equations().sort_values("weight").index[-1]
    'eq0005'
    >>> round(float(res.weights.sum()), 10)
    1.0
    """

    def __init__(
        self,
        max_monthly: int = 2,
        max_quarterly: int = 1,
        min_monthly: int = 1,
        target_lags: int = 0,
        regressor_lags: int = 0,
        aggregation: str | AggregationType | Sequence[float] = "average",
        add_constant: bool = True,
        combine: str = "mean",
        mse: str = "in_sample",
        mse_window: int | None = None,
        discount: float = 1.0,
        min_train: int = 8,
        trim: float = 0.0,
        extrapolation: str | Extrapolator | None = "ar",
        ar_lags: int = 1,
        extrapolation_options: Mapping[str, Any] | None = None,
        horizon: int = 1,
    ) -> None:
        self.max_monthly = max_monthly
        self.max_quarterly = max_quarterly
        self.min_monthly = min_monthly
        self.target_lags = target_lags
        self.regressor_lags = regressor_lags
        self.aggregation = aggregation
        self.add_constant = add_constant
        self.combine = combine
        self.mse = mse
        self.mse_window = mse_window
        self.discount = discount
        self.min_train = min_train
        self.trim = trim
        self.extrapolation = extrapolation
        self.ar_lags = ar_lags
        self.extrapolation_options = extrapolation_options
        self.horizon = horizon

    # ------------------------------------------------------------------ validation
    def _validate_params(self) -> None:
        for name in ("max_quarterly", "min_monthly", "target_lags", "regressor_lags"):
            _check_int(name, getattr(self, name), 0)
        for name in ("ar_lags", "horizon"):
            _check_int(name, getattr(self, name), 0)
        _check_int("max_monthly", self.max_monthly, max(int(self.min_monthly), 1))
        _check_int("min_train", self.min_train, 1)
        if self.mse_window is not None:
            _check_int("mse_window", self.mse_window, 1)
        self._validate_choices()

    def _validate_choices(self) -> None:
        if self.combine not in _COMBINE:
            raise ValueError(f"combine must be one of {_COMBINE}, got {self.combine!r}.")
        if self.mse not in _MSE:
            raise ValueError(f"mse must be one of {_MSE}, got {self.mse!r}.")
        if not 0.0 < float(self.discount) <= 1.0:
            raise ValueError(f"discount must be in (0, 1], got {self.discount!r}.")
        if not 0.0 <= float(self.trim) < 1.0:
            raise ValueError(f"trim must be in [0, 1), got {self.trim!r}.")
        if not isinstance(self.add_constant, bool):
            raise ValueError(f"add_constant must be a bool, got {self.add_constant!r}.")
        if self.extrapolation_options is not None and not isinstance(
            self.extrapolation_options, Mapping
        ):
            raise ValueError("extrapolation_options must be a mapping or None.")

    # ------------------------------------------------------------------ pieces
    def _extrapolator(self) -> Callable[..., dict[str, pd.Series]] | None:
        spec = self.extrapolation
        if spec is None:
            return None
        options = dict(self.extrapolation_options or {})
        if spec == "ar":
            options.setdefault("ar_lags", self.ar_lags)
        return make_extrapolator(spec, **options)

    def _extrapolate(
        self, data: MixedFrequencyData, columns: list[str], end: pd.Period
    ) -> dict[str, pd.Series]:
        extrapolator = self._extrapolator()
        if extrapolator is None:
            return {col: native_until(data, col, end) for col in columns}
        try:
            completed = extrapolator(data, columns, end)
        except NowcastDataError as err:
            completed = _complete_each(extrapolator, data, columns, end, err)
            columns = [c for c in columns if c in completed]
        missing = [c for c in columns if c not in completed]
        if missing:
            raise NowcastDataError(f"The extrapolator returned no values for {missing}.")
        return {col: completed[col] for col in columns}

    def _indicators(
        self,
        data: MixedFrequencyData,
        completed: dict[str, pd.Series],
        target_freq: Frequency,
        periods: pd.PeriodIndex,
    ) -> pd.DataFrame:
        columns: dict[str, pd.Series] = {}
        for col, series in completed.items():
            freq = data.metadata[col].frequency
            weights = resolve_aggregation_weights(
                self.aggregation, freq, target_freq, normalize="ratio"
            )
            columns[col] = aggregate_to_target(series, weights, periods)
        return pd.DataFrame(columns, index=periods)

    def _eval_positions(self, observed: np.ndarray) -> np.ndarray:
        positions = np.flatnonzero(np.isfinite(observed))
        if self.mse_window is not None:
            positions = positions[-int(self.mse_window) :]
        return positions

    def _select(self, mse: np.ndarray, valid: np.ndarray) -> np.ndarray:
        """Equations entering the combination (estimable, MSE known if needed, trimmed)."""
        needs_mse = self.combine == "inverse_mse" or float(self.trim) > 0
        included = valid & np.isfinite(mse) if needs_mse else valid.copy()
        if not included.any():
            raise NowcastDataError(
                "No bridge equation could be estimated (or none has an MSE); check the "
                "sample length, min_train and mse_window."
            )
        n_drop = int(np.floor(float(self.trim) * included.sum()))
        n_drop = min(n_drop, int(included.sum()) - 1)
        if n_drop > 0:
            candidates = np.flatnonzero(included)
            worst = candidates[np.argsort(mse[candidates], kind="stable")[::-1][:n_drop]]
            included[worst] = False
        return included

    def _weights(self, mse: np.ndarray, included: np.ndarray) -> np.ndarray:
        weights = np.zeros(len(mse))
        if self.combine == "inverse_mse":
            sub = mse[included]
            floor = max(float(np.max(sub)) * np.finfo(float).eps, np.finfo(float).tiny)
            inv = 1.0 / np.maximum(sub, floor)
            weights[included] = inv / inv.sum()
        else:
            weights[included] = 1.0 / included.sum()
        return weights

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        data: MixedFrequencyData | pd.DataFrame,
        target: str,
        *,
        frequency: FrequencySpec | None = None,
        **fit_kwargs: Any,
    ) -> BridgeCombinationResults:
        """Estimate every bridge equation and combine their nowcasts of ``target``.

        Parameters
        ----------
        data : MixedFrequencyData or pandas.DataFrame
            Panel on the base grid.
        target : str
            Target name or formula (``"gdp ~ ."``, ``"gdp ~ ip + pmi + sent"``)
            selecting the candidate indicators.
        frequency : frequency specification, optional
            Per-series frequencies for DataFrame input.
        **fit_kwargs
            ``as_of`` (date-like): fit on the pseudo real-time vintage
            ``data.as_of(as_of)`` (needs release delays), so that the estimates,
            MSEs, weights and trimming use only the information released by then.

        Returns
        -------
        BridgeCombinationResults
            Combined nowcasts and the equation table (also in :attr:`results_`).

        Raises
        ------
        NowcastDataError
            If no equation can be estimated or a predictor is invalid.
        ValueError
            If a hyper-parameter is invalid.
        TypeError
            If an unknown fit option is given.

        Examples
        --------
        >>> import numpy as np, pandas as pd
        >>> idx = pd.period_range("2000-01", periods=48, freq="M")
        >>> x = np.cos(np.arange(48.0))
        >>> y = pd.Series(x, index=idx).rolling(3).mean().where(idx.month % 3 == 0)
        >>> df = pd.DataFrame({"x": x, "y": y}, index=idx)
        >>> res = BridgeCombination().fit(df, "y", frequency={"x": "M", "y": "Q"})
        >>> res.model_name
        'BridgeCombination'
        """
        results = super().fit(data, target, frequency=frequency, **fit_kwargs)
        assert isinstance(results, BridgeCombinationResults)  # noqa: S101
        return results

    def _prepare(
        self, data: MixedFrequencyData, target: str
    ) -> tuple[pd.PeriodIndex, pd.Series, dict[str, pd.Series], pd.DataFrame, list[str], list[str]]:
        target_freq = data.metadata[target].frequency
        high, low = _split_predictors(data, target)
        periods = _target_periods(data, target_freq, int(self.horizon))
        observed = data.to_native(target).reindex(periods).rename(target)
        if int(observed.notna().sum()) == 0:
            raise NowcastDataError(f"The target {target!r} has no observations.")
        completed = self._extrapolate(data, [*high, *low], periods[-1])
        high = [c for c in high if c in completed]
        low = [c for c in low if c in completed]
        indicators = self._indicators(data, completed, target_freq, periods)
        return periods, observed, completed, indicators, high, low

    def _fit(
        self, data: MixedFrequencyData, target: str, **fit_kwargs: Any
    ) -> BridgeCombinationResults:
        as_of = fit_kwargs.pop("as_of", None)
        if fit_kwargs:
            raise TypeError(f"Unexpected fit options {sorted(fit_kwargs)}.")
        if as_of is not None:
            data = data.as_of(as_of)
        started = time.perf_counter()
        periods, observed, completed, indicators, high, low = self._prepare(data, target)
        features = _build_features(
            indicators,
            observed,
            int(self.regressor_lags),
            int(self.target_lags),
            self.add_constant,
        )
        sizes = (int(self.min_monthly), int(self.max_monthly), int(self.max_quarterly))
        allow_empty = int(self.target_lags) > 0
        equations = _enumerate(high, low, sizes, allow_empty)
        if not equations:
            raise NowcastDataError(
                "No bridge equation can be formed: the panel needs at least "
                f"{self.min_monthly} higher-frequency indicator(s)."
            )
        columns = [features.columns([*h, *lo]) for h, lo in equations]
        eval_positions = self._eval_positions(features.observed)
        batch = _fit_all(
            features, columns, eval_positions, self.mse == "out_of_sample", int(self.min_train)
        )
        mse = _discounted_mse(batch.errors, eval_positions, float(self.discount))
        included = self._select(mse, batch.valid)
        weights = self._weights(mse, included)
        n_invalid = int((~batch.valid).sum())
        if n_invalid:
            _warn_invalid(n_invalid, len(equations))
        fitted = _Fitted(
            equations=equations,
            columns=columns,
            names=features.names,
            batch=batch,
            mse=mse,
            included=included,
            weights=weights,
            eval_positions=eval_positions,
            started=started,
        )
        return self._build_results(data, target, (periods, observed, completed, indicators), fitted)

    def _build_results(
        self,
        data: MixedFrequencyData,
        target: str,
        prepared: tuple[pd.PeriodIndex, pd.Series, dict[str, pd.Series], pd.DataFrame],
        fitted: _Fitted,
    ) -> BridgeCombinationResults:
        periods, observed, completed, indicators = prepared
        ids = [f"eq{i + 1:04d}" for i in range(len(fitted.equations))]
        specs = _spec_frame(ids, target, fitted)
        if self.combine == "median":
            specs["weight"] = np.where(fitted.included, np.nan, 0.0)
        estimates = pd.DataFrame(fitted.batch.estimates.T, index=periods, columns=ids)
        used = fitted.batch.estimates[fitted.included]
        combined = _combine(used, fitted.weights[fitted.included], str(self.combine))
        frame = build_nowcast_frame(
            observed, pd.Series(combined, index=periods), extra=_dispersion(used, periods)
        )
        coefficients = _coefficient_series(ids, fitted)
        logger.debug(
            "BridgeCombination: %d equations (%d combined)", len(ids), int(fitted.included.sum())
        )
        return BridgeCombinationResults(
            target=target,
            nowcast=frame,
            model_name="BridgeCombination",
            model_params=self.get_params(deep=False),
            params={"coefficients": coefficients, "weights": specs["weight"].copy()},
            data=data,
            info={
                "fit_time": time.perf_counter() - fitted.started,
                "n_equations": len(ids),
                "n_valid": int(fitted.batch.valid.sum()),
                "n_included": int(fitted.included.sum()),
                "mse_periods": periods[fitted.eval_positions],
            },
            specs=specs,
            estimates=estimates,
            indicators=indicators,
            extrapolated=completed,
        )


def _warn_invalid(n_invalid: int, n_total: int) -> None:
    warnings.warn(
        f"{n_invalid} of {n_total} bridge equations could not be estimated (too few "
        "observations or collinear regressors) and are excluded.",
        DataQualityWarning,
        stacklevel=4,
    )


def _spec_frame(ids: list[str], target: str, fitted: _Fitted) -> pd.DataFrame:
    batch, equations = fitted.batch, fitted.equations
    n_coef = np.array([len(c) for c in fitted.columns])
    dof = np.maximum(batch.n_obs - n_coef, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rsquared = np.where(batch.tss > 0, 1.0 - batch.ssr / batch.tss, np.nan)
    return pd.DataFrame(
        {
            "spec": [f"{target} ~ " + (" + ".join([*h, *lo]) or "1") for h, lo in equations],
            "monthly": [h for h, _ in equations],
            "quarterly": [lo for _, lo in equations],
            "n_coefficients": n_coef,
            "n_obs": batch.n_obs,
            "rsquared": rsquared,
            "sigma": np.sqrt(batch.ssr / dof),
            "mse": fitted.mse,
            "valid": batch.valid,
            "included": fitted.included,
            "weight": fitted.weights,
        },
        index=pd.Index(ids, name="equation"),
    )


def _coefficient_series(ids: list[str], fitted: _Fitted) -> pd.Series:
    names = fitted.names
    keys = [(eq, names[c]) for eq, cols in zip(ids, fitted.columns, strict=True) for c in cols]
    values = np.concatenate([np.asarray(b, dtype=float) for b in fitted.batch.coefficients])
    index = pd.MultiIndex.from_tuples(keys, names=["equation", "term"])
    return pd.Series(values, index=index, name="coefficient")
