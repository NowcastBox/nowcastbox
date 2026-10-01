"""Panel preparation pipeline and standardisation helpers.

:func:`prepare_panel` chains the usual pre-processing steps of the dynamic-factor
nowcasting literature (Giannone, Reichlin & Small, 2008; Bańbura & Modugno, 2014):

1. stationarity transformations, series by series on their native grids
   (:mod:`~nowcastbox.preprocessing.transforms`);
2. removal of the leading periods without any observation (lost to lags);
3. removal of series with too many missing values (``max_na_prop``, default 1/3);
4. outlier correction - IQR rule and centred moving median
   (:mod:`~nowcastbox.preprocessing.outliers`);
5. filling of interior gaps, never the ragged edge unless requested
   (:mod:`~nowcastbox.preprocessing.missing`);
6. optionally, Mariano-Murasawa filtering of the high-frequency series
   (:mod:`~nowcastbox.preprocessing.aggregation`).

Standardisation is left to the models (contract: results are reported in the units
of the prepared panel); :func:`standardize` / :func:`destandardize` are provided for
convenience and work on DataFrames as well as on panels.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from nowcastbox._logging import get_logger
from nowcastbox.core.data import FrequencySpec, MixedFrequencyData, StandardizationStats
from nowcastbox.core.exceptions import DataQualityWarning, NowcastDataError
from nowcastbox.core.frequency import AggregationTypeLike
from nowcastbox.preprocessing._utils import PanelLike, as_panel, return_like
from nowcastbox.preprocessing.aggregation import aggregate_panel
from nowcastbox.preprocessing.missing import EdgeMethod, FillMethod, fill_missing
from nowcastbox.preprocessing.outliers import OutlierReplacement
from nowcastbox.preprocessing.outliers import replace_outliers as _replace_outliers
from nowcastbox.preprocessing.transforms import (
    apply_transforms,
    get_transform,
    resolve_transforms,
)

__all__ = ["PanelReport", "destandardize", "prepare_panel", "standardize"]

logger = get_logger(__name__)


@dataclass(frozen=True)
class PanelReport:
    """What :func:`prepare_panel` did to each series.

    Parameters
    ----------
    transforms : dict of str to str
        Canonical specification of the transformation applied to each input series.
    dropped : list of str
        Series removed for having too many missing values.
    missing_proportion : pandas.Series
        Proportion of missing storage slots of each transformed series (before
        dropping and filling).
    n_outliers : pandas.Series
        Number of outliers replaced in each kept series.
    n_filled : pandas.Series
        Number of missing values filled in each kept series.
    aggregated : bool
        Whether the high-frequency series were aggregated.
    start : pandas.Period or None
        First period of the prepared panel.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from nowcastbox.preprocessing.panel import prepare_panel
    >>> idx = pd.period_range("2020-01", periods=12, freq="M")
    >>> df = pd.DataFrame({"a": np.arange(1.0, 13.0) ** 2}, index=idx)
    >>> _, report = prepare_panel(df, "diff", frequency="M", return_report=True)
    >>> report.transforms, report.dropped
    ({'a': 'diff(1)'}, [])
    """

    transforms: dict[str, str]
    dropped: list[str]
    missing_proportion: pd.Series
    n_outliers: pd.Series
    n_filled: pd.Series
    aggregated: bool = False
    start: pd.Period | None = None
    info: dict[str, Any] = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        """One row per input series with the transformation and the counts.

        Returns
        -------
        pandas.DataFrame
            Columns ``transform``, ``missing_proportion``, ``dropped``,
            ``n_outliers``, ``n_filled``.

        Examples
        --------
        >>> import pandas as pd
        >>> from nowcastbox.preprocessing.panel import prepare_panel
        >>> idx = pd.period_range("2020-01", periods=6, freq="M")
        >>> df = pd.DataFrame({"a": [1.0, 4, 2, 8, 5, 7]}, index=idx)
        >>> bool(
        ...     prepare_panel(df, frequency="M", return_report=True)[1]
        ...     .to_frame()
        ...     .loc["a", "dropped"]
        ... )
        False
        """
        names = list(self.transforms)
        frame = pd.DataFrame(
            {
                "transform": pd.Series(self.transforms),
                "missing_proportion": self.missing_proportion.reindex(names),
                "dropped": pd.Series({n: n in self.dropped for n in names}),
                "n_outliers": self.n_outliers.reindex(names).fillna(0).astype(int),
                "n_filled": self.n_filled.reindex(names).fillna(0).astype(int),
            },
            index=names,
        )
        frame.index.name = "series"
        return frame


def _check_prepare_args(max_na_prop: float | None) -> None:
    if max_na_prop is None:
        return
    if isinstance(max_na_prop, bool) or not isinstance(max_na_prop, int | float | np.floating):
        raise ValueError(f"max_na_prop must be a number in [0, 1], got {max_na_prop!r}.")
    if not 0.0 <= float(max_na_prop) <= 1.0:
        raise ValueError(f"max_na_prop must be in [0, 1], got {max_na_prop!r}.")


def _drop_leading_empty(panel: MixedFrequencyData) -> MixedFrequencyData:
    has_obs = panel.observation_mask().any(axis=1).to_numpy()
    if not has_obs.any():
        raise NowcastDataError("The transformed panel has no observation.")
    first = int(np.argmax(has_obs))
    if first == 0:
        return panel
    return panel.truncate(start=panel.index[first])


def _drop_sparse(
    panel: MixedFrequencyData, max_na_prop: float | None, keep: list[str]
) -> tuple[MixedFrequencyData, pd.Series, list[str]]:
    slots = panel.slot_mask().sum()
    prop = (panel.missing_mask().sum() / slots.clip(lower=1)).rename("missing_proportion")
    if max_na_prop is None:
        return panel, prop, []
    dropped = [c for c in panel.columns if prop[c] > max_na_prop and c not in keep]
    if len(dropped) == panel.n_series:
        raise NowcastDataError(
            f"Every series has more than {max_na_prop:.0%} missing values; nothing is left."
        )
    if dropped:
        warnings.warn(
            f"Dropped {len(dropped)} series with more than {max_na_prop:.1%} missing "
            f"values: {dropped}.",
            DataQualityWarning,
            stacklevel=3,
        )
        panel = panel.drop(dropped)
    return panel, prop, dropped


def prepare_panel(
    data: PanelLike,
    transform: Any = None,
    *,
    frequency: FrequencySpec | None = None,
    replace_outliers: bool = True,
    outlier_threshold: float = 4.0,
    outlier_window: int = 3,
    outlier_replacement: OutlierReplacement = "moving_median",
    replace_na: bool = True,
    na_method: FillMethod = "spline",
    na_window: int = 3,
    fill_ragged_edge: bool = False,
    edge_method: EdgeMethod = "median",
    max_na_prop: float | None = 1 / 3,
    keep: list[str] | str | None = None,
    aggregate: bool = False,
    aggregation: AggregationTypeLike | None = None,
    drop_leading_empty: bool = True,
    return_report: bool = False,
) -> Any:
    """Transform and clean a mixed-frequency panel before estimation.

    Parameters
    ----------
    data : pandas.DataFrame or MixedFrequencyData
        Panel in levels on a base grid (quarterly values in the third month).
    transform : transform-like, mapping, Series or sequence, optional
        Transformations (see :func:`~nowcastbox.preprocessing.transforms.apply_transforms`):
        one for all series, per series (mapping/Series by name, or a sequence aligned
        with the columns, e.g. codes 0-7 of a legend). Defaults to the ``transform``
        metadata of a :class:`MixedFrequencyData`, else levels.
    frequency : frequency specification, optional
        Per-series native frequencies for DataFrame input (inferred when omitted).
    replace_outliers : bool, default True
        Replace outliers (``|x - median| > outlier_threshold * IQR``).
    outlier_threshold : float, default 4.0
        IQR multiple of the outlier rule.
    outlier_window : int, default 3
        Window (native periods) of the centred moving median used as replacement.
    outlier_replacement : {"moving_median", "median", "nan"}, default "moving_median"
        Replacement of outliers.
    replace_na : bool, default True
        Fill interior gaps (between the first and last observation of each series).
    na_method : {"spline", "linear", "moving_median"}, default "spline"
        Interior filling method.
    na_window : int, default 3
        Window of the ``"moving_median"`` filling and of the edge median.
    fill_ragged_edge : bool, default False
        Also fill the ragged edge (only with ``replace_na=True``). Off by default:
        nowcasting models exploit it.
    edge_method : {"median", "last", "mean"}, default "median"
        Filling of the ragged edge when ``fill_ragged_edge=True``.
    max_na_prop : float or None, default 1/3
        Drop series whose proportion of missing storage slots (after transformation)
        exceeds this value; ``None`` keeps every series.
    keep : str or list of str, optional
        Series never dropped (e.g. the nowcast target).
    aggregate : bool, default False
        Filter the base-frequency series with the aggregation weights of the lowest
        frequency in the panel (Mariano-Murasawa by default, normalised), as in the
        "aggregate the variables" two-step variant.
    aggregation : AggregationType or str, optional
        Aggregation used when ``aggregate=True`` (default: series metadata, else
        Mariano-Murasawa).
    drop_leading_empty : bool, default True
        Remove the leading periods in which no series is observed (typically the
        periods lost to the lags of the transformations).
    return_report : bool, default False
        Also return a :class:`PanelReport`.

    Returns
    -------
    panel : pandas.DataFrame or MixedFrequencyData
        Prepared panel, of the input type. Series keep their native frequencies and
        the storage convention.
    report : PanelReport
        Only when ``return_report=True``.

    Raises
    ------
    NowcastDataError
        On invalid data, when every series would be dropped, or when the transformed
        panel is empty.
    ValueError
        On invalid options.

    Warns
    -----
    DataQualityWarning
        When series are dropped.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from nowcastbox.preprocessing.panel import prepare_panel
    >>> idx = pd.period_range("2019-01", periods=24, freq="M")
    >>> rng = np.random.default_rng(0)
    >>> ip = 100 * np.exp(np.cumsum(rng.normal(0.002, 0.01, 24)))
    >>> gdp = np.where(np.arange(24) % 3 == 2, np.linspace(100, 108, 24), np.nan)
    >>> df = pd.DataFrame({"ip": ip, "gdp": gdp}, index=idx)
    >>> out = prepare_panel(
    ...     df, {"ip": "dlog", "gdp": "pct_change"}, frequency={"ip": "M", "gdp": "Q"}
    ... )
    >>> str(out.index[0]), int(out["gdp"].notna().sum())
    ('2019-02', 7)
    """
    _check_prepare_args(max_na_prop)
    panel, was_frame = as_panel(data, frequency)
    specs = resolve_transforms(panel, transform)
    applied = {c: get_transform(s).to_spec() for c, s in specs.items()}
    keep_list = [keep] if isinstance(keep, str) else list(keep or [])
    unknown = sorted(set(keep_list) - set(panel.columns))
    if unknown:
        raise NowcastDataError(f"keep refers to unknown series {unknown}.")

    out = apply_transforms(panel, specs)
    assert isinstance(out, MixedFrequencyData)  # noqa: S101 - panel in, panel out
    if drop_leading_empty:
        out = _drop_leading_empty(out)
    out, prop, dropped = _drop_sparse(out, max_na_prop, keep_list)

    n_outliers = pd.Series(0, index=out.columns, dtype=int)
    if replace_outliers:
        out, mask = _replace_outliers(
            out,
            outlier_threshold,
            window=outlier_window,
            replacement=outlier_replacement,
            return_mask=True,
        )
        n_outliers = mask.sum().astype(int)

    observed_before = out.observation_mask().sum()
    if replace_na:
        out = fill_missing(
            out,
            na_method,
            window=na_window,
            fill_ragged_edge=fill_ragged_edge,
            edge_method=edge_method,
        )
    n_filled = (out.observation_mask().sum() - observed_before).astype(int)

    if aggregate:
        out = aggregate_panel(out, aggregation)
    assert isinstance(out, MixedFrequencyData)  # noqa: S101

    report = PanelReport(
        transforms=applied,
        dropped=dropped,
        missing_proportion=prop,
        n_outliers=n_outliers.rename("n_outliers"),
        n_filled=n_filled.rename("n_filled"),
        aggregated=aggregate,
        start=out.start,
    )
    logger.info(
        "prepare_panel: %d series kept, %d dropped, %d outliers, %d values filled",
        out.n_series,
        len(dropped),
        int(n_outliers.sum()),
        int(n_filled.sum()),
    )
    result = return_like(out, was_frame)
    return (result, report) if return_report else result


# ---------------------------------------------------------------------- standardisation
def standardize(
    data: PanelLike, stats: StandardizationStats | None = None, *, ddof: int = 1
) -> tuple[PanelLike, StandardizationStats]:
    """Standardise every series: ``(x - mean) / std`` over the observed values.

    Parameters
    ----------
    data : pandas.DataFrame or MixedFrequencyData
        Panel.
    stats : StandardizationStats, optional
        Statistics to reuse (e.g. from a training sample); computed when omitted.
    ddof : int, default 1
        Delta degrees of freedom of the standard deviation.

    Returns
    -------
    standardized : pandas.DataFrame or MixedFrequencyData
        Standardised panel of the input type.
    stats : StandardizationStats
        Statistics for :func:`destandardize`.

    Raises
    ------
    NowcastDataError
        If a series is constant or has too few observations, or ``stats`` does not
        cover every series.

    Examples
    --------
    >>> import pandas as pd
    >>> from nowcastbox.preprocessing.panel import destandardize, standardize
    >>> idx = pd.period_range("2020-01", periods=3, freq="M")
    >>> z, stats = standardize(pd.DataFrame({"a": [1.0, 2.0, 3.0]}, index=idx))
    >>> z["a"].tolist(), destandardize(z, stats)["a"].tolist()
    ([-1.0, 0.0, 1.0], [1.0, 2.0, 3.0])
    """
    if isinstance(data, MixedFrequencyData):
        return data.standardize(stats, ddof=ddof)
    if not isinstance(data, pd.DataFrame):
        raise NowcastDataError(
            f"data must be a DataFrame or MixedFrequencyData, got {type(data).__name__}."
        )
    frame = data.astype(float)
    if stats is None:
        mean = frame.mean(skipna=True)
        std = frame.std(skipna=True, ddof=ddof)
        bad = std.index[~np.isfinite(std.to_numpy()) | (std.to_numpy() <= 0)].tolist()
        if bad:
            raise NowcastDataError(
                f"Cannot standardise series {bad}: too few observations or zero variance."
            )
        stats = StandardizationStats(mean=mean, std=std, ddof=ddof)
    return stats.transform(frame), stats


def destandardize(data: PanelLike, stats: StandardizationStats) -> PanelLike:
    """Map standardised data back to original units: ``z * std + mean``.

    Parameters
    ----------
    data : pandas.DataFrame or MixedFrequencyData
        Standardised panel.
    stats : StandardizationStats
        Statistics returned by :func:`standardize`.

    Returns
    -------
    pandas.DataFrame or MixedFrequencyData
        Panel in original units.

    Raises
    ------
    NowcastDataError
        If ``stats`` does not cover every series.

    Examples
    --------
    >>> import pandas as pd
    >>> from nowcastbox.core.data import StandardizationStats
    >>> from nowcastbox.preprocessing.panel import destandardize
    >>> stats = StandardizationStats(pd.Series({"a": 10.0}), pd.Series({"a": 2.0}))
    >>> idx = pd.period_range("2020-01", periods=2, freq="M")
    >>> destandardize(pd.DataFrame({"a": [0.0, 1.0]}, index=idx), stats)["a"].tolist()
    [10.0, 12.0]
    """
    if isinstance(data, MixedFrequencyData):
        return data.destandardize(stats)
    if not isinstance(data, pd.DataFrame):
        raise NowcastDataError(
            f"data must be a DataFrame or MixedFrequencyData, got {type(data).__name__}."
        )
    return stats.inverse_transform(data.astype(float))
