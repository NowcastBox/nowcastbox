r"""Data-quality report of a nowcasting panel.

Per series: share of missing storage slots, missing values before the first
observation, inside the sample and at the ragged edge (Bańbura, Giannone & Reichlin,
2011: the jagged end of the panel created by publication lags), outliers flagged by
the IQR rule of :func:`nowcastbox.preprocessing.outliers.detect_outliers` (Stock &
Watson, 2002), and series whose estimated loadings are all close to zero (series that
contribute little to the common factors and mostly add noise; Boivin & Ng, 2006).
A publication-delay summary groups the series by frequency and release delay.

References
----------
Bańbura, M., Giannone, D., & Reichlin, L. (2011). Nowcasting. In *Oxford Handbook of
Economic Forecasting*, 193-224.

Boivin, J., & Ng, S. (2006). Are more data always better for factor analysis?
*Journal of Econometrics*, 132(1), 169-194.

Stock, J. H., & Watson, M. W. (2002). Macroeconomic forecasting using diffusion
indexes. *Journal of Business & Economic Statistics*, 20(2), 147-162.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from nowcastbox.core.data import FrequencySpec, MixedFrequencyData
from nowcastbox.core.exceptions import DataQualityWarning
from nowcastbox.core.frequency import base_to_native
from nowcastbox.diagnostics._common import as_panel
from nowcastbox.preprocessing.outliers import detect_outliers

__all__ = ["DataQualityReport", "data_quality_report"]


@dataclass(frozen=True, eq=False)
class DataQualityReport:
    """Data-quality diagnostics of a panel.

    Parameters
    ----------
    table : pandas.DataFrame
        Indexed by series: ``frequency``, ``category``, ``release_delay`` (days),
        ``n_slots``, ``n_obs``, ``na_share``, ``leading_missing``,
        ``interior_missing``, ``ragged_edge`` (native periods after the last
        observation), ``first_observed``, ``last_observed``, ``n_outliers``,
        ``max_abs_loading``, ``near_zero_loading`` and ``flags``.
    outliers : pandas.DataFrame
        Long table of flagged observations: ``series``, ``period`` (native), ``value``.
    publication : pandas.DataFrame
        Publication-delay summary by ``frequency`` and ``release_delay``:
        ``n_series``, ``mean_ragged_edge``, ``max_ragged_edge``, ``series``.
    end : pandas.Period
        Last period of the panel.
    thresholds : dict
        ``outlier_threshold``, ``loading_tol``, ``max_na_share``.

    Examples
    --------
    >>> from nowcastbox.models.two_step import simulate_two_step_example
    >>> from nowcastbox.diagnostics import data_quality_report
    >>> rep = data_quality_report(simulate_two_step_example(random_state=0))
    >>> int(rep.table.loc["x3", "ragged_edge"])
    2
    """

    table: pd.DataFrame
    outliers: pd.DataFrame
    publication: pd.DataFrame
    end: pd.Period
    thresholds: dict[str, float]

    @property
    def flagged(self) -> list[str]:
        """Series with at least one flag."""
        return [str(s) for s in self.table.index[self.table["flags"] != ""]]

    def to_frame(self) -> pd.DataFrame:
        """Copy of :attr:`table`.

        Returns
        -------
        pandas.DataFrame
            One row per series.

        Examples
        --------
        >>> rep.to_frame().shape[0]  # doctest: +SKIP
        11
        """
        return self.table.copy()

    def summary(self) -> str:
        """Text summary.

        Returns
        -------
        str
            Overview of missing data, ragged edge, outliers and flags.

        Examples
        --------
        >>> print(rep.summary())  # doctest: +SKIP
        """
        t = self.table
        total_slots = int(t["n_slots"].sum())
        na = 1.0 - int(t["n_obs"].sum()) / total_slots if total_slots else np.nan
        lines = [
            "Data quality",
            f"  {'Series':<26}{len(t)} (panel ends {self.end})",
            f"  {'Missing share (slots)':<26}{na:.3f}",
            f"  {'Ragged edge (periods)':<26}max {int(t['ragged_edge'].max())}, "
            f"mean {t['ragged_edge'].mean():.2f}",
            f"  {'Outliers':<26}{int(t['n_outliers'].sum())} in "
            f"{int((t['n_outliers'] > 0).sum())} series",
            f"  {'Near-zero loadings':<26}"
            + (", ".join(t.index[t["near_zero_loading"].to_numpy(bool)].astype(str)) or "-"),
        ]
        for name in self.flagged:
            lines.append(f"  {'! ' + name:<26}{t.loc[name, 'flags']}")
        return "\n".join(lines)


def _missing_parts(panel: MixedFrequencyData) -> pd.DataFrame:
    slots = panel.slot_mask().to_numpy()
    obs = panel.observation_mask().to_numpy()
    ragged = panel.ragged_edge_mask().to_numpy()
    before_first = np.cumsum(obs, axis=0) == 0
    leading = slots & before_first & ~ragged
    interior = slots & ~obs & ~leading & ~ragged
    return pd.DataFrame(
        {
            "n_slots": slots.sum(axis=0),
            "n_obs": (obs & slots).sum(axis=0),
            "leading_missing": leading.sum(axis=0),
            "interior_missing": interior.sum(axis=0),
            "ragged_edge": ragged.sum(axis=0),
        },
        index=panel.columns,
    )


def _native_str(period: Any, panel: MixedFrequencyData, name: str) -> str | None:
    if period is None or pd.isna(period):
        return None
    freq = panel.metadata[name].frequency
    native = base_to_native(pd.PeriodIndex([period]), freq)
    return str(native[0])


def _category(value: Any) -> str | None:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    return str(getattr(value, "value", value))


def _outlier_table(panel: MixedFrequencyData, threshold: float) -> pd.DataFrame:
    mask = detect_outliers(panel, threshold)
    assert isinstance(mask, pd.DataFrame)  # noqa: S101
    frame = panel.to_frame()
    rows = []
    for name in panel.columns:
        flags = mask[name].to_numpy(bool)
        values = frame[name].to_numpy(dtype=np.float64)[flags]
        for period, value in zip(mask.index[flags], values, strict=True):
            rows.append(
                {
                    "series": name,
                    "period": _native_str(period, panel, name),
                    "value": float(value),
                }
            )
    return pd.DataFrame(rows, columns=["series", "period", "value"])


def _loading_info(loadings: pd.DataFrame | None, names: list[str], tol: float) -> pd.DataFrame:
    if loadings is None:
        nan = pd.Series(np.nan, index=names)
        return pd.DataFrame({"max_abs_loading": nan, "near_zero_loading": False})
    mx = loadings.abs().max(axis=1).reindex(names).astype(float)
    return pd.DataFrame({"max_abs_loading": mx, "near_zero_loading": (mx < tol).to_numpy()})


def _flags(row: pd.Series, max_na_share: float) -> str:
    out = []
    if row["n_obs"] == 0:
        out.append("no_observations")
    elif row["na_share"] > max_na_share:
        out.append(f"na_share>{max_na_share:g}")
    if row["n_outliers"] > 0:
        out.append(f"outliers({int(row['n_outliers'])})")
    if bool(row["near_zero_loading"]):
        out.append("near_zero_loading")
    return ";".join(out)


def _publication(table: pd.DataFrame) -> pd.DataFrame:
    keyed = table.assign(release_delay=table["release_delay"].astype("Float64").fillna(-1))
    rows = []
    for (freq, delay), g in keyed.groupby(["frequency", "release_delay"], sort=True):
        rows.append(
            {
                "frequency": freq,
                "release_delay": None if delay == -1 else int(delay),
                "n_series": len(g),
                "mean_ragged_edge": float(g["ragged_edge"].mean()),
                "max_ragged_edge": int(g["ragged_edge"].max()),
                "series": ", ".join(map(str, g.index)),
            }
        )
    return pd.DataFrame(rows)


def _check_positive(value: float, name: str, upper: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float | np.floating):
        raise ValueError(f"{name} must be a positive number, got {value!r}.")
    v = float(value)
    if not np.isfinite(v) or v <= 0 or (upper is not None and v > upper):
        bound = f" and <= {upper}" if upper is not None else ""
        raise ValueError(f"{name} must be positive{bound}, got {value!r}.")
    return v


def data_quality_report(
    data: MixedFrequencyData | pd.DataFrame,
    *,
    loadings: pd.DataFrame | None = None,
    outlier_threshold: float = 4.0,
    loading_tol: float = 0.1,
    max_na_share: float = 0.5,
    warn: bool = False,
    frequency: FrequencySpec | None = None,
) -> DataQualityReport:
    """Missing data, ragged edge, outliers, weak loadings and publication delays.

    Parameters
    ----------
    data : MixedFrequencyData or pandas.DataFrame
        Panel on a base-frequency PeriodIndex.
    loadings : pandas.DataFrame, optional
        Estimated loadings (series x factors, standardised units, e.g.
        ``results.loadings``); series whose largest absolute loading is below
        ``loading_tol`` are flagged.
    outlier_threshold : float, default 4.0
        IQR multiple of the outlier rule.
    loading_tol : float, default 0.1
        Threshold of the near-zero-loading flag.
    max_na_share : float, default 0.5
        Flag series whose share of missing slots exceeds this value.
    warn : bool, default False
        Issue one :class:`~nowcastbox.core.exceptions.DataQualityWarning` listing the
        flagged series.
    frequency : frequency specification, optional
        Frequencies of a DataFrame ``data``.

    Returns
    -------
    DataQualityReport
        Tidy tables and summary.

    Raises
    ------
    ValueError
        On invalid thresholds.

    Warns
    -----
    DataQualityWarning
        When ``warn=True`` and some series are flagged.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from nowcastbox.diagnostics import data_quality_report
    >>> idx = pd.period_range("2020-01", periods=12, freq="M")
    >>> x = np.r_[np.nan, np.arange(9.0), 50.0, np.nan]
    >>> rep = data_quality_report(pd.DataFrame({"x": x}, index=idx), frequency="M")
    >>> rep.table.loc["x", ["leading_missing", "ragged_edge", "n_outliers"]].tolist()
    [1, 1, 1]
    """
    thr = _check_positive(outlier_threshold, "outlier_threshold")
    tol = _check_positive(loading_tol, "loading_tol")
    max_na = _check_positive(max_na_share, "max_na_share", 1.0)
    panel = as_panel(data, frequency)
    names = list(panel.columns)
    table = _missing_parts(panel)
    table["na_share"] = 1.0 - table["n_obs"] / table["n_slots"].clip(lower=1)
    first = {n: panel.to_frame()[n].first_valid_index() for n in names}
    last = panel.last_observed()
    table["first_observed"] = [_native_str(first[n], panel, n) for n in names]
    table["last_observed"] = [_native_str(last[n], panel, n) for n in names]
    outliers = _outlier_table(panel, thr)
    table["n_outliers"] = outliers.groupby("series").size().reindex(names).fillna(0).astype(int)
    table = table.join(_loading_info(loadings, names, tol))
    table.insert(0, "frequency", [panel.metadata[n].frequency.value for n in names])
    cats = panel.categories
    table.insert(1, "category", [_category(c) for c in cats.reindex(names)])
    table.insert(2, "release_delay", panel.release_delays.reindex(names))
    table["flags"] = [_flags(table.loc[n], max_na) for n in names]
    table.index.name = "series"
    report = DataQualityReport(
        table=table,
        outliers=outliers,
        publication=_publication(table),
        end=panel.end,
        thresholds={
            "outlier_threshold": thr,
            "loading_tol": tol,
            "max_na_share": max_na,
        },
    )
    if warn and report.flagged:
        warnings.warn(
            f"Data-quality flags in {len(report.flagged)} series: "
            + "; ".join(f"{n} [{table.loc[n, 'flags']}]" for n in report.flagged[:10]),
            DataQualityWarning,
            stacklevel=2,
        )
    return report
