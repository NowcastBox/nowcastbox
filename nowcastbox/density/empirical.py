r"""Empirical error bands from past nowcast errors (Reifschneider-Tulip / ECB style).

Instead of a model-based predictive distribution, the uncertainty around a point
nowcast :math:`\hat y_{\tau \mid v}` made at vintage :math:`v` is measured by the
**past errors of the same model at the same horizon** (months to the end of the target
period, :math:`h`), as in the projection ranges of Reifschneider & Tulip (2019) and of
the Eurosystem/ECB staff projections (ECB, 2009), used by the ECB nowcasting toolbox
(Linzenich & Meunier, 2024, section 2.3).

Information set
---------------
Only errors that were **known at the vintage** enter the bands: an error
:math:`e_{s,\tau'} = y_{\tau'} - \hat y_{\tau' \mid s}` is used for the nowcast made at
:math:`v` when

* its forecast was made strictly before the vintage (:math:`s < v`) and inside the
  rolling window (:math:`s \ge v - W`, default :math:`W` = 10 years);
* its outcome had been released by the vintage (``availability="release"``, the
  default): :math:`s + \text{days\_to\_release} \le v`. When the release date is
  unknown the target period must have ended before :math:`v`. With
  ``availability="vintage"`` only :math:`s < v` is required.

Scale of the bands
------------------
With :math:`n_h` available errors :math:`e_1..e_{n_h}` at horizon :math:`h`:

``"mae"``
    :math:`\mathrm{MAE}_h = n_h^{-1}\sum_i |e_i|`. The ECB band is
    :math:`\hat y \pm \mathrm{MAE}_h`, which covers about 57.5 % of the outcomes under
    normality, since :math:`\operatorname{E}|e| = \sigma\sqrt{2/\pi}` for
    :math:`e \sim N(0, \sigma^2)`. Bands of any level :math:`\ell` follow from the
    Gaussian distribution with :math:`\hat\sigma_h = \mathrm{MAE}_h\sqrt{\pi/2}`.
``"rmse"``
    :math:`\hat\sigma_h = (n_h^{-1}\sum_i e_i^2)^{1/2}`.
``"quantile"``
    Non-parametric: the predictive distribution is the empirical distribution of
    :math:`\hat y + e_i` (:class:`EmpiricalQuantileDistribution`).

The bands are centred at the point nowcast (past errors are not de-meaned, as in the
ECB procedure). Optionally the past errors are adjusted for outliers before computing
the scale: errors outside ``median ± threshold × 1.4826 × MAD`` are clipped
(``outliers="winsorize"``) or dropped (``outliers="exclude"``).

References
----------
Reifschneider, D. & Tulip, P. (2019). Gauging the uncertainty of the economic outlook
using historical forecasting errors: the Federal Reserve's approach. *International
Journal of Forecasting*, 35(4), 1564-1582.

European Central Bank (2009). *New procedure for constructing Eurosystem and ECB staff
projection ranges*. ECB, Frankfurt am Main.

Linzenich, J. & Meunier, B. (2024). Nowcasting made easier: a toolbox for economists.
ECB Working Paper No. 3004.

Hyndman, R. J. & Fan, Y. (1996). Sample quantiles in statistical packages. *The
American Statistician*, 50(4), 361-365.

Examples
--------
>>> import numpy as np, pandas as pd
>>> from nowcastbox.density import empirical_error_scales
>>> past = pd.DataFrame(
...     {
...         "vintage": pd.to_datetime(["2019-01-15", "2019-04-15", "2019-07-15"]),
...         "target_period": pd.PeriodIndex(["2019Q1", "2019Q2", "2019Q3"], freq="Q"),
...         "months_to_end": 2,
...         "days_to_release": 100.0,
...         "error": [0.5, -1.0, 0.3],
...     }
... )
>>> empirical_error_scales(past, "2020-01-15", min_errors=2)[["n", "mae"]].round(3)
                n  mae
months_to_end
2               3  0.6
"""

from __future__ import annotations

import math
import re
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, Protocol

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from nowcastbox._logging import get_logger
from nowcastbox.core.exceptions import DataQualityWarning, NowcastDataError
from nowcastbox.density.distribution import NowcastDistribution

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from nowcastbox.core.results import NowcastResults

__all__ = [
    "EMPIRICAL_LEVELS",
    "EMPIRICAL_METHODS",
    "MAE_TO_SIGMA",
    "EmpiricalGaussianDistribution",
    "EmpiricalQuantileDistribution",
    "available_errors",
    "backtest_empirical_bands",
    "empirical_bands",
    "empirical_error_scales",
]

logger = get_logger(__name__)

FloatArray = NDArray[np.float64]
EmpiricalMethod = Literal["mae", "rmse", "quantile"]
OutlierRule = Literal["winsorize", "exclude"]
Availability = Literal["release", "vintage"]
WindowLike = str | pd.DateOffset | pd.Timedelta | None

EMPIRICAL_METHODS: tuple[str, ...] = ("mae", "rmse", "quantile")
"""Scale estimators of :func:`empirical_bands`."""

EMPIRICAL_LEVELS: tuple[float, ...] = (0.575, 0.68, 0.9)
"""Default band levels: the ECB ±1 MAE band (57.5 %), 68 % and 90 %."""

MAE_TO_SIGMA: float = math.sqrt(math.pi / 2.0)
r"""Factor :math:`\sqrt{\pi/2}` turning a Gaussian MAE into a standard deviation."""

_REQUIRED = ("vintage", "target_period", "months_to_end", "error")
_HORIZON = "months_to_end"
_KNOWN = "_known_from"
_MAD_TO_SIGMA = 1.4826
_WINDOW_RE = re.compile(r"^\s*(\d+)\s*([YMWD])\s*$", re.IGNORECASE)
_WINDOW_UNITS: dict[str, Callable[[int], pd.DateOffset]] = {
    "Y": lambda n: pd.DateOffset(years=n),
    "M": lambda n: pd.DateOffset(months=n),
    "W": lambda n: pd.DateOffset(weeks=n),
    "D": lambda n: pd.DateOffset(days=n),
}


class _HasForecastTable(Protocol):
    """Anything with a long forecast table (e.g. :class:`BacktestResults`)."""

    def to_frame(self) -> pd.DataFrame: ...  # pragma: no cover


BacktestLike = _HasForecastTable | pd.DataFrame


# ====================================================================== validation
def _level_label(level: float) -> str:
    return f"{100.0 * level:.10g}"


def _check_levels(levels: float | Iterable[float]) -> list[float]:
    values = [float(levels)] if np.isscalar(levels) else [float(v) for v in levels]  # type: ignore[arg-type]
    if not values:
        raise ValueError("At least one band level is required.")
    for v in values:
        if not 0.0 < v < 1.0:
            raise ValueError(f"Band levels must be in (0, 1), got {v}.")
    return values


def _check_probabilities(q: float | Iterable[float]) -> FloatArray:
    arr = np.atleast_1d(np.asarray(q, dtype=np.float64))
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("Probabilities must be a scalar or a non-empty 1-D sequence.")
    if not bool(np.all((arr > 0.0) & (arr < 1.0))):
        raise ValueError(f"Probabilities must lie in (0, 1), got {arr.tolist()}.")
    return arr


def _check_choice(value: Any, name: str, choices: Sequence[Any]) -> None:
    if value not in choices:
        raise ValueError(f"{name} must be one of {list(choices)}, got {value!r}.")


def _window_offset(window: WindowLike) -> pd.DateOffset | pd.Timedelta | None:
    """Parse the rolling window (``"10Y"``, ``"36M"``, DateOffset, Timedelta, None)."""
    if window is None or (isinstance(window, str) and window.lower() == "expanding"):
        return None
    if isinstance(window, pd.DateOffset | pd.Timedelta):
        return window
    match = _WINDOW_RE.match(str(window))
    if match is None or int(match.group(1)) < 1:
        raise ValueError(
            f"window must look like '10Y', '36M', '520W' or '3650D' (or be None), got {window!r}."
        )
    return _WINDOW_UNITS[match.group(2).upper()](int(match.group(1)))


def _forecast_table(backtest: BacktestLike, model: str | None) -> tuple[pd.DataFrame, str | None]:
    """Long forecast table of one model (columns of ``BacktestResults.forecasts``)."""
    frame = backtest.copy() if isinstance(backtest, pd.DataFrame) else backtest.to_frame()
    missing = [c for c in _REQUIRED if c not in frame.columns]
    if missing:
        raise ValueError(f"The backtest table lacks the columns {missing}.")
    if "model" in frame.columns:
        names = list(getattr(backtest, "models", None) or pd.unique(frame["model"]))
        chosen = names[0] if model is None else model
        if chosen not in set(frame["model"]):
            raise ValueError(f"Model {chosen!r} is not in the backtest; available: {names}.")
        frame = frame[frame["model"] == chosen]
    else:
        chosen = model
    frame = frame.assign(vintage=pd.to_datetime(frame["vintage"]).dt.normalize())
    frame = frame.reset_index(drop=True)
    frame[_KNOWN] = _known_dates(frame)
    return frame, chosen


def _known_dates(frame: pd.DataFrame) -> pd.Series:
    """Date from which the outcome of each forecast is known.

    ``vintage + days_to_release`` when the release date is known, otherwise the day
    after the end of the target period.
    """
    periods = pd.PeriodIndex([pd.Period(p) for p in frame["target_period"]])
    fallback = pd.Series(periods.end_time.normalize() + pd.Timedelta(days=1), index=frame.index)
    if "days_to_release" not in frame.columns:
        return fallback
    days = pd.to_numeric(frame["days_to_release"], errors="coerce")
    release = frame["vintage"] + pd.to_timedelta(days.fillna(0.0), unit="D")
    return release.where(days.notna(), fallback)


def _available(
    frame: pd.DataFrame,
    date: pd.Timestamp,
    offset: pd.DateOffset | pd.Timedelta | None,
    availability: Availability,
) -> pd.DataFrame:
    """Rows of ``frame`` whose error was known at ``date``."""
    keep = (frame["vintage"] < date).to_numpy() & np.isfinite(
        pd.to_numeric(frame["error"], errors="coerce").to_numpy(dtype=float)
    )
    if offset is not None:
        keep &= (frame["vintage"] >= date - offset).to_numpy()
    if availability == "release":
        keep &= (frame[_KNOWN] <= date).to_numpy()
    return frame[keep]


# ====================================================================== error statistics
def _robust_bounds(errors: FloatArray, threshold: float) -> tuple[float, float] | None:
    """``median ± threshold * 1.4826 * MAD`` (``None`` when the MAD is zero)."""
    med = float(np.median(errors))
    mad = _MAD_TO_SIGMA * float(np.median(np.abs(errors - med)))
    if mad <= 0.0:
        return None
    return med - threshold * mad, med + threshold * mad


def _adjust_outliers(
    errors: FloatArray, outliers: OutlierRule | None, threshold: float
) -> FloatArray:
    """Winsorize or drop outlying past errors."""
    if outliers is None or errors.size < 3:
        return errors
    bounds = _robust_bounds(errors, threshold)
    if bounds is None:
        return errors
    lo, hi = bounds
    if outliers == "winsorize":
        return np.clip(errors, lo, hi)
    return errors[(errors >= lo) & (errors <= hi)]


def _horizon_errors(
    sub: pd.DataFrame, horizon: int, outliers: OutlierRule | None, threshold: float
) -> FloatArray:
    values = sub.loc[sub[_HORIZON] == horizon, "error"].to_numpy(dtype=np.float64)
    return _adjust_outliers(values, outliers, threshold)


def _scale(errors: FloatArray, method: str) -> float:
    """Standard deviation implied by the errors (MAE·√(π/2) or RMSE)."""
    if method == "mae":
        return float(np.mean(np.abs(errors))) * MAE_TO_SIGMA
    return float(np.sqrt(np.mean(errors**2)))


def _stats_row(errors: FloatArray, min_errors: int) -> dict[str, float]:
    n = int(errors.size)
    if n < min_errors or n == 0:
        nan = float("nan")
        return {"n": n, "bias": nan, "mae": nan, "rmse": nan, "sigma_mae": nan}
    mae = float(np.mean(np.abs(errors)))
    return {
        "n": n,
        "bias": float(np.mean(errors)),
        "mae": mae,
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "sigma_mae": mae * MAE_TO_SIGMA,
    }


def _check_common(
    method: str, outliers: Any, availability: Any, min_errors: int, threshold: float
) -> None:
    _check_choice(method, "method", EMPIRICAL_METHODS)
    _check_choice(outliers, "outliers", (None, "winsorize", "exclude"))
    _check_choice(availability, "availability", ("release", "vintage"))
    if isinstance(min_errors, bool) or not isinstance(min_errors, int | np.integer):
        raise ValueError(f"min_errors must be a positive integer, got {min_errors!r}.")
    if min_errors < 1:
        raise ValueError(f"min_errors must be a positive integer, got {min_errors!r}.")
    if not threshold > 0.0:
        raise ValueError(f"outlier_threshold must be positive, got {threshold!r}.")


# ====================================================================== public: errors
def available_errors(
    backtest: BacktestLike,
    vintage: pd.Timestamp | str,
    *,
    model: str | None = None,
    window: WindowLike = "10Y",
    availability: Availability = "release",
) -> pd.DataFrame:
    """Past forecast errors that were known at a vintage date.

    Parameters
    ----------
    backtest : BacktestResults or pandas.DataFrame
        Backtest results (or their long :meth:`~BacktestResults.to_frame` table) with
        at least the columns ``vintage``, ``target_period``, ``months_to_end`` and
        ``error``; ``days_to_release`` and ``model`` are used when present.
    vintage : Timestamp or str
        Date of the information set.
    model : str, optional
        Model whose errors are used (default: the main model of the backtest).
    window : str, DateOffset, Timedelta or None, default "10Y"
        Rolling window before the vintage (``"10Y"``, ``"36M"``...); ``None`` or
        ``"expanding"`` uses every earlier error.
    availability : {"release", "vintage"}, default "release"
        ``"release"`` also requires the outcome to have been released by the vintage
        (no look-ahead); ``"vintage"`` only requires the forecast vintage to be
        strictly earlier.

    Returns
    -------
    pandas.DataFrame
        The selected rows of the forecast table.

    Raises
    ------
    ValueError
        On a malformed table, an unknown model or invalid options.

    Examples
    --------
    >>> import pandas as pd
    >>> from nowcastbox.density import available_errors
    >>> past = pd.DataFrame(
    ...     {
    ...         "vintage": pd.to_datetime(["2019-01-15", "2019-12-15"]),
    ...         "target_period": pd.PeriodIndex(["2019Q1", "2019Q4"], freq="Q"),
    ...         "months_to_end": [2, 0],
    ...         "days_to_release": [100.0, 100.0],
    ...         "error": [0.5, -1.0],
    ...     }
    ... )
    >>> available_errors(past, "2020-01-15")["error"].tolist()  # Q4 not yet released
    [0.5]
    """
    _check_choice(availability, "availability", ("release", "vintage"))
    frame, _ = _forecast_table(backtest, model)
    date = pd.Timestamp(vintage).normalize()
    sub = _available(frame, date, _window_offset(window), availability)
    return sub.drop(columns=_KNOWN).reset_index(drop=True)


def empirical_error_scales(
    backtest: BacktestLike,
    vintage: pd.Timestamp | str,
    *,
    model: str | None = None,
    window: WindowLike = "10Y",
    outliers: OutlierRule | None = None,
    outlier_threshold: float = 3.0,
    availability: Availability = "release",
    min_errors: int = 8,
) -> pd.DataFrame:
    r"""Error statistics by horizon from the errors known at a vintage.

    Parameters
    ----------
    backtest : BacktestResults or pandas.DataFrame
        Backtest results or their long forecast table (see :func:`available_errors`).
    vintage : Timestamp or str
        Date of the information set.
    model : str, optional
        Model whose errors are used (default: the main model).
    window : str, DateOffset, Timedelta or None, default "10Y"
        Rolling window (see :func:`available_errors`).
    outliers : {"winsorize", "exclude"} or None, default None
        Outlier adjustment of the past errors (robust ``median ± k·MAD`` rule).
    outlier_threshold : float, default 3.0
        :math:`k` of the outlier rule (in robust standard deviations).
    availability : {"release", "vintage"}, default "release"
        Information-set rule (see :func:`available_errors`).
    min_errors : int, default 8
        Minimum number of errors per horizon; below it the statistics are NaN.

    Returns
    -------
    pandas.DataFrame
        Index ``months_to_end`` (every horizon of the backtest), columns ``n``,
        ``bias``, ``mae``, ``rmse`` and ``sigma_mae`` (:math:`\mathrm{MAE}\sqrt{\pi/2}`).

    Raises
    ------
    ValueError
        On a malformed table or invalid options.

    Examples
    --------
    >>> import pandas as pd
    >>> from nowcastbox.density import empirical_error_scales
    >>> past = pd.DataFrame(
    ...     {
    ...         "vintage": pd.to_datetime(["2018-01-15", "2018-04-15"]),
    ...         "target_period": pd.PeriodIndex(["2018Q1", "2018Q2"], freq="Q"),
    ...         "months_to_end": 2,
    ...         "error": [1.0, -1.0],
    ...     }
    ... )
    >>> empirical_error_scales(past, "2020-01-01", min_errors=2)["rmse"].tolist()
    [1.0]
    """
    _check_common("mae", outliers, availability, min_errors, outlier_threshold)
    frame, _ = _forecast_table(backtest, model)
    date = pd.Timestamp(vintage).normalize()
    sub = _available(frame, date, _window_offset(window), availability)
    horizons = sorted(int(h) for h in pd.unique(frame[_HORIZON]))
    rows = [
        _stats_row(_horizon_errors(sub, h, outliers, outlier_threshold), min_errors)
        for h in horizons
    ]
    out = pd.DataFrame(rows, index=pd.Index(horizons, name=_HORIZON))
    return out.astype({"n": int})


# ====================================================================== quantile distribution
class EmpiricalQuantileDistribution:
    r"""Non-parametric predictive distribution: point nowcast plus past errors.

    For each target period :math:`t` the distribution is the empirical distribution of
    :math:`\hat y_t + e_{t,i}`, :math:`i = 1..n_t`, where :math:`e_{t,i}` are the past
    errors at the horizon of :math:`t`. Quantiles use the linear interpolation of
    Hyndman & Fan (1996, definition 7). The API mirrors
    :class:`~nowcastbox.density.NowcastDistribution` (``interval``, ``quantiles``,
    ``to_frame``, ``fan_chart_frame``, ``plot``, ``sample``, ``cdf``) and adds an exact
    ensemble :meth:`crps`.

    Parameters
    ----------
    index : pandas.PeriodIndex or sequence of Period/str
        Target periods.
    point : array_like, shape (n_periods,)
        Point nowcasts.
    errors : sequence of array_like
        Past errors of each period (at least one finite value each).
    target : str, default ""
        Name of the target series.
    levels : float or iterable of float, default (0.575, 0.68, 0.9)
        Default band levels of :meth:`to_frame`, :meth:`fan_chart_frame` and
        :meth:`plot`.
    info : mapping, optional
        Extra information.

    Raises
    ------
    ValueError
        On inconsistent lengths, non-finite values, empty error sets or invalid levels.

    Examples
    --------
    >>> from nowcastbox.density import EmpiricalQuantileDistribution
    >>> d = EmpiricalQuantileDistribution(["2020Q1"], [1.0], [[-1.0, 0.0, 1.0]], target="gdp")
    >>> d.interval(0.5).to_numpy().tolist()
    [[0.5, 1.5]]
    >>> d
    EmpiricalQuantileDistribution('gdp', 2020Q1, 3 errors)
    """

    def __init__(
        self,
        index: pd.PeriodIndex | Sequence[pd.Period | str],
        point: ArrayLike,
        errors: Sequence[ArrayLike],
        *,
        target: str = "",
        levels: float | Iterable[float] = EMPIRICAL_LEVELS,
        info: Mapping[str, Any] | None = None,
    ) -> None:
        idx = self._coerce_index(index)
        pt = np.asarray(point, dtype=np.float64).reshape(-1)
        if pt.shape != (len(idx),) or not bool(np.isfinite(pt).all()):
            raise ValueError(f"point must be a finite array of length {len(idx)}.")
        errs = [np.asarray(e, dtype=np.float64).reshape(-1) for e in errors]
        if len(errs) != len(idx):
            raise ValueError(f"errors must hold one array per period ({len(idx)}).")
        if any(e.size == 0 or not bool(np.isfinite(e).all()) for e in errs):
            raise ValueError("Every period needs at least one error and all must be finite.")
        self._index = idx
        self._point = pt
        self._errors = [np.sort(e) for e in errs]
        self._target = str(target)
        self._levels = tuple(_check_levels(levels))
        self._info: dict[str, Any] = dict(info or {})

    @staticmethod
    def _coerce_index(index: pd.PeriodIndex | Sequence[pd.Period | str]) -> pd.PeriodIndex:
        items = list(index)
        if not items:
            raise ValueError("index must contain at least one period.")
        out = (
            index.copy()
            if isinstance(index, pd.PeriodIndex)
            else pd.PeriodIndex([pd.Period(p) for p in items])
        )
        if out.has_duplicates:
            raise ValueError("index must not contain duplicated periods.")
        out.name = "period"
        return out

    # ------------------------------------------------------------------ attributes
    @property
    def index(self) -> pd.PeriodIndex:
        """Target periods."""
        return self._index.copy()

    @property
    def target(self) -> str:
        """Name of the target series."""
        return self._target

    @property
    def levels(self) -> tuple[float, ...]:
        """Default band levels."""
        return self._levels

    @property
    def n_periods(self) -> int:
        """Number of target periods."""
        return len(self._index)

    @property
    def is_gaussian(self) -> bool:
        """Always ``False`` (non-parametric distribution)."""
        return False

    @property
    def info(self) -> dict[str, Any]:
        """Extra information (copy)."""
        return dict(self._info)

    @property
    def errors(self) -> list[FloatArray]:
        """Sorted past errors of each period (copies)."""
        return [e.copy() for e in self._errors]

    @property
    def n_errors(self) -> pd.Series:
        """Number of past errors per period."""
        return pd.Series([e.size for e in self._errors], index=self.index, name="n_errors")

    def __len__(self) -> int:
        return self.n_periods

    def __repr__(self) -> str:
        span = f"{self._index[0]}" if len(self) == 1 else f"{self._index[0]}..{self._index[-1]}"
        name = f"{self._target!r}, " if self._target else ""
        n = sum(e.size for e in self._errors)
        return f"EmpiricalQuantileDistribution({name}{span}, {n} errors)"

    # ------------------------------------------------------------------ selection
    def _position(self, period: pd.Period | str | int) -> int:
        if isinstance(period, int | np.integer) and not isinstance(period, bool):
            pos = int(period)
            if not -self.n_periods <= pos < self.n_periods:
                raise KeyError(f"Position {pos} out of range for {self.n_periods} periods.")
            return pos % self.n_periods
        freq = self._index.freqstr
        key = period.asfreq(freq) if isinstance(period, pd.Period) else pd.Period(str(period), freq)
        matches = np.flatnonzero(self._index == key)
        if matches.size == 0:
            raise KeyError(f"Period {key} is not in the distribution index.")
        return int(matches[0])

    def __getitem__(self, period: pd.Period | str | int) -> EmpiricalQuantileDistribution:
        """Distribution of a single period (label or integer position)."""
        return self.select([period])

    def select(self, periods: Iterable[pd.Period | str | int]) -> EmpiricalQuantileDistribution:
        """Distribution restricted to some periods.

        Parameters
        ----------
        periods : iterable of Period, str or int
            Period labels or integer positions.

        Returns
        -------
        EmpiricalQuantileDistribution
            New object with the selected periods.

        Raises
        ------
        KeyError
            If a period is not in :attr:`index`.

        Examples
        --------
        >>> from nowcastbox.density import EmpiricalQuantileDistribution
        >>> d = EmpiricalQuantileDistribution(["2020Q1", "2020Q2"], [0.0, 1.0], [[1.0], [2.0]])
        >>> d.select(["2020Q2"]).point.tolist()
        [1.0]
        """
        pos = [self._position(p) for p in periods]
        return EmpiricalQuantileDistribution(
            self._index[pos],
            self._point[pos],
            [self._errors[i] for i in pos],
            target=self._target,
            levels=self._levels,
            info=self._info,
        )

    # ------------------------------------------------------------------ moments
    @property
    def point(self) -> pd.Series:
        """Point nowcasts (centre of the bands)."""
        return pd.Series(self._point, index=self.index, name="point")

    @property
    def mean(self) -> pd.Series:
        """Mean of the empirical distribution (point plus mean past error)."""
        values = [p + float(e.mean()) for p, e in zip(self._point, self._errors, strict=True)]
        return pd.Series(values, index=self.index, name="mean")

    @property
    def std(self) -> pd.Series:
        """Standard deviation of the empirical distribution (``ddof=0``)."""
        return pd.Series([float(e.std()) for e in self._errors], index=self.index, name="std")

    @property
    def median(self) -> pd.Series:
        """Median of the empirical distribution."""
        return self.quantiles(0.5).iloc[:, 0].rename("median")

    # ------------------------------------------------------------------ quantiles / cdf
    def _ppf(self, probs: FloatArray) -> FloatArray:
        rows = [p + np.quantile(e, probs) for p, e in zip(self._point, self._errors, strict=True)]
        return np.asarray(rows, dtype=np.float64).reshape(self.n_periods, probs.size)

    def ppf(self, q: float | Iterable[float]) -> FloatArray:
        """Quantile function (linear interpolation of the empirical quantiles).

        Parameters
        ----------
        q : float or sequence of float
            Probabilities in ``(0, 1)``.

        Returns
        -------
        numpy.ndarray, shape (n_periods, len(q))
            Quantiles.

        Raises
        ------
        ValueError
            If a probability is outside ``(0, 1)``.

        Examples
        --------
        >>> from nowcastbox.density import EmpiricalQuantileDistribution
        >>> d = EmpiricalQuantileDistribution(["2020Q1"], [0.0], [[-1.0, 1.0]])
        >>> d.ppf([0.5]).tolist()
        [[0.0]]
        """
        return self._ppf(_check_probabilities(q))

    def quantiles(self, q: float | Iterable[float]) -> pd.DataFrame:
        """Quantiles of every period.

        Parameters
        ----------
        q : float or sequence of float
            Probabilities in ``(0, 1)``.

        Returns
        -------
        pandas.DataFrame
            Index = periods, one column per probability.

        Raises
        ------
        ValueError
            If a probability is outside ``(0, 1)``.

        Examples
        --------
        >>> from nowcastbox.density import EmpiricalQuantileDistribution
        >>> d = EmpiricalQuantileDistribution(["2020Q1"], [0.0], [[-1.0, 0.0, 1.0]])
        >>> d.quantiles([0.25, 0.75]).to_numpy().tolist()
        [[-0.5, 0.5]]
        """
        probs = _check_probabilities(q)
        return pd.DataFrame(self._ppf(probs), index=self.index, columns=pd.Index(probs, name="q"))

    def interval(self, level: float = 0.9) -> pd.DataFrame:
        """Central (equal-tailed) band.

        Parameters
        ----------
        level : float, default 0.9
            Nominal coverage in ``(0, 1)``.

        Returns
        -------
        pandas.DataFrame
            Columns ``lower`` and ``upper``.

        Raises
        ------
        ValueError
            If ``level`` is outside ``(0, 1)``.

        Examples
        --------
        >>> from nowcastbox.density import EmpiricalQuantileDistribution
        >>> d = EmpiricalQuantileDistribution(["2020Q1"], [0.0], [[-1.0, 0.0, 1.0]])
        >>> d.interval(0.5).to_numpy().tolist()
        [[-0.5, 0.5]]
        """
        (lvl,) = _check_levels(level)
        q = self._ppf(np.array([0.5 * (1.0 - lvl), 0.5 * (1.0 + lvl)]))
        return pd.DataFrame(q, index=self.index, columns=["lower", "upper"])

    def cdf(self, x: ArrayLike) -> FloatArray:
        r"""Empirical distribution function (step function).

        Parameters
        ----------
        x : array_like
            One point per period (broadcast to ``(n_periods,)``).

        Returns
        -------
        numpy.ndarray, shape (n_periods,)
            Share of :math:`\hat y_t + e_{t,i}` that are ``<= x_t``.

        Examples
        --------
        >>> from nowcastbox.density import EmpiricalQuantileDistribution
        >>> d = EmpiricalQuantileDistribution(["2020Q1"], [0.0], [[-1.0, 0.0, 1.0, 2.0]])
        >>> d.cdf(0.0).tolist()
        [0.5]
        """
        xb = np.broadcast_to(np.asarray(x, dtype=np.float64), (self.n_periods,))
        values = [
            np.searchsorted(e, v - p, side="right") / e.size
            for v, p, e in zip(xb, self._point, self._errors, strict=True)
        ]
        return np.asarray(values, dtype=np.float64)

    def crps(self, observed: ArrayLike) -> FloatArray:
        r"""Exact CRPS of the empirical distribution (ensemble form).

        :math:`\mathrm{CRPS} = \operatorname{E}|X - y| - \tfrac12 \operatorname{E}|X - X'|`
        with :math:`X, X'` independent draws from the empirical distribution
        (Gneiting & Raftery, 2007).

        Parameters
        ----------
        observed : array_like
            One outcome per period.

        Returns
        -------
        numpy.ndarray, shape (n_periods,)
            CRPS per period.

        Examples
        --------
        >>> from nowcastbox.density import EmpiricalQuantileDistribution
        >>> d = EmpiricalQuantileDistribution(["2020Q1"], [0.0], [[-1.0, 1.0]])
        >>> d.crps(0.0).tolist()
        [0.5]
        """
        obs = np.broadcast_to(np.asarray(observed, dtype=np.float64), (self.n_periods,))
        out = []
        for y, p, e in zip(obs, self._point, self._errors, strict=True):
            x = p + e
            spread = float(np.abs(x[:, None] - x[None, :]).mean())
            out.append(float(np.abs(x - y).mean()) - 0.5 * spread)
        return np.asarray(out, dtype=np.float64)

    def sample(
        self, n: int, *, random_state: int | np.random.Generator | None = None
    ) -> pd.DataFrame:
        """Resample the empirical distribution (independently across periods).

        Parameters
        ----------
        n : int
            Number of draws per period (``>= 1``).
        random_state : int, numpy.random.Generator or None
            Seed or generator.

        Returns
        -------
        pandas.DataFrame
            Shape ``(n, n_periods)``; columns = periods.

        Raises
        ------
        ValueError
            If ``n`` is not a positive integer.

        Examples
        --------
        >>> from nowcastbox.density import EmpiricalQuantileDistribution
        >>> d = EmpiricalQuantileDistribution(["2020Q1"], [0.0], [[-1.0, 1.0]])
        >>> d.sample(10, random_state=0).shape
        (10, 1)
        """
        if isinstance(n, bool) or not isinstance(n, int | np.integer) or n < 1:
            raise ValueError(f"n must be a positive integer, got {n!r}.")
        rng = np.random.default_rng(random_state)
        draws = np.column_stack(
            [p + rng.choice(e, size=int(n)) for p, e in zip(self._point, self._errors, strict=True)]
        )
        return pd.DataFrame(draws, columns=self.index)

    # ------------------------------------------------------------------ export
    def to_frame(self, levels: float | Iterable[float] | None = None) -> pd.DataFrame:
        """Summary table: point, mean, std, median, bands and number of errors.

        Parameters
        ----------
        levels : float or iterable of float, optional
            Band levels (default: :attr:`levels`); columns ``lower_<100*level>`` and
            ``upper_<100*level>``.

        Returns
        -------
        pandas.DataFrame
            One row per period.

        Raises
        ------
        ValueError
            If a level is outside ``(0, 1)``.

        Examples
        --------
        >>> from nowcastbox.density import EmpiricalQuantileDistribution
        >>> d = EmpiricalQuantileDistribution(["2020Q1"], [0.0], [[-1.0, 1.0]])
        >>> list(d.to_frame(0.9).columns)
        ['point', 'mean', 'std', 'median', 'lower_90', 'upper_90', 'n_errors']
        """
        lv = _check_levels(self._levels if levels is None else levels)
        frame = pd.DataFrame(
            {
                "point": self._point,
                "mean": self.mean.to_numpy(),
                "std": self.std.to_numpy(),
                "median": self.median.to_numpy(),
            },
            index=self.index,
        )
        for v in lv:
            band = self.interval(v)
            frame[f"lower_{_level_label(v)}"] = band["lower"].to_numpy()
            frame[f"upper_{_level_label(v)}"] = band["upper"].to_numpy()
        frame["n_errors"] = self.n_errors.to_numpy()
        return frame

    def fan_chart_frame(self, levels: float | Iterable[float] | None = None) -> pd.DataFrame:
        """Long-format bands for fan charts (widest band first).

        Parameters
        ----------
        levels : float or iterable of float, optional
            Band levels (default: :attr:`levels`).

        Returns
        -------
        pandas.DataFrame
            Columns ``period``, ``level``, ``lower``, ``upper``, ``median``.

        Raises
        ------
        ValueError
            If a level is outside ``(0, 1)``.

        Examples
        --------
        >>> from nowcastbox.density import EmpiricalQuantileDistribution
        >>> d = EmpiricalQuantileDistribution(["2020Q1"], [0.0], [[-1.0, 1.0]])
        >>> d.fan_chart_frame()["level"].tolist()
        [0.9, 0.68, 0.575]
        """
        lv = sorted(_check_levels(self._levels if levels is None else levels), reverse=True)
        median = self.median.to_numpy()
        parts = []
        for v in lv:
            band = self.interval(v)
            parts.append(
                pd.DataFrame(
                    {
                        "period": self._index,
                        "level": v,
                        "lower": band["lower"].to_numpy(),
                        "upper": band["upper"].to_numpy(),
                        "median": median,
                    }
                )
            )
        return pd.concat(parts, ignore_index=True)

    def plot(
        self,
        *,
        levels: float | Iterable[float] | None = None,
        history: pd.Series | None = None,
        ax: Axes | None = None,
        color: str = "C0",
        title: str | None = None,
    ) -> Axes:
        """Fan chart of the empirical bands (matplotlib).

        Parameters
        ----------
        levels : float or iterable of float, optional
            Band levels (default: :attr:`levels`), drawn as nested bands.
        history : pandas.Series, optional
            Observed target values (PeriodIndex of the same frequency).
        ax : matplotlib.axes.Axes, optional
            Axes to draw on (default: a new figure).
        color : str, default "C0"
            Colour of the bands and of the point line.
        title : str, optional
            Axes title (default: ``"Empirical error bands: <target>"``).

        Returns
        -------
        matplotlib.axes.Axes
            The axes.

        Examples
        --------
        >>> import matplotlib
        >>> matplotlib.use("Agg")
        >>> from nowcastbox.density import EmpiricalQuantileDistribution
        >>> d = EmpiricalQuantileDistribution(["2020Q1", "2020Q2"], [0.0, 1.0], [[-1, 1], [-2, 2]])
        >>> len(d.plot().collections)
        3
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 4))
        bands = self.fan_chart_frame(levels)
        x = self._index.to_timestamp(how="end").normalize()
        lv = sorted(set(bands["level"]), reverse=True)
        for i, v in enumerate(lv):
            band = bands[bands["level"] == v]
            ax.fill_between(
                x,
                band["lower"].to_numpy(),
                band["upper"].to_numpy(),
                color=color,
                alpha=0.15 + 0.5 * (i + 1) / (len(lv) + 1),
                linewidth=0,
                label=f"{_level_label(v)}%",
            )
        ax.plot(x, self._point, color=color, marker="o", label="nowcast")
        if history is not None and isinstance(history.index, pd.PeriodIndex):
            hist = history.dropna()
            hist_index = pd.PeriodIndex(hist.index)
            ax.plot(
                hist_index.to_timestamp(how="end").normalize(),
                hist.to_numpy(dtype=float),
                color="black",
                label="observed",
            )
        default = f"Empirical error bands: {self._target}".strip()
        ax.set_title(title if title is not None else default)
        ax.legend(loc="best")
        return ax


# ====================================================================== gaussian bands
class EmpiricalGaussianDistribution(NowcastDistribution):
    r"""Gaussian empirical bands (methods ``"mae"`` and ``"rmse"``).

    A :class:`~nowcastbox.density.NowcastDistribution` centred at the point nowcasts
    with the scales implied by the past errors, whose :meth:`to_frame`,
    :meth:`fan_chart_frame` and :meth:`plot` default to the band levels of the
    empirical bands (57.5 % -- the ECB :math:`\pm 1` MAE band --, 68 % and 90 %) instead
    of the generic :data:`~nowcastbox.density.DEFAULT_LEVELS`. Everything else
    (quantiles, cdf/pdf, sampling, :func:`~nowcastbox.evaluation.scoring.crps`, PIT...)
    is inherited.

    Parameters
    ----------
    index : pandas.PeriodIndex or sequence of Period/str
        Target periods.
    point : array_like, shape (n_periods,)
        Point nowcasts (centres of the bands).
    scales : array_like, shape (n_periods,)
        Standard deviations implied by the past errors.
    target : str, default ""
        Name of the target series.
    levels : float or iterable of float, default (0.575, 0.68, 0.9)
        Default band levels (also stored in ``info["levels"]``).
    info : mapping, optional
        Extra information.

    Raises
    ------
    ValueError
        On inconsistent shapes, non-positive scales or invalid levels.

    Examples
    --------
    >>> from nowcastbox.density import EmpiricalGaussianDistribution
    >>> d = EmpiricalGaussianDistribution(["2020Q1"], [1.0], [0.5], target="gdp")
    >>> list(d.to_frame().columns)  # doctest: +NORMALIZE_WHITESPACE
    ['point', 'mean', 'std', 'median', 'lower_57.5', 'upper_57.5', 'lower_68', 'upper_68',
     'lower_90', 'upper_90']
    >>> d
    EmpiricalGaussianDistribution('gdp', 2020Q1, Gaussian)
    """

    def __init__(
        self,
        index: pd.PeriodIndex | Sequence[pd.Period | str],
        point: ArrayLike,
        scales: ArrayLike,
        *,
        target: str = "",
        levels: float | Iterable[float] = EMPIRICAL_LEVELS,
        info: Mapping[str, Any] | None = None,
    ) -> None:
        lv = tuple(_check_levels(levels))
        super().__init__(
            index,
            point,
            scales,
            target=target,
            point=point,
            info={**dict(info or {}), "levels": lv},
        )
        self._levels = lv

    @property
    def levels(self) -> tuple[float, ...]:
        """Default band levels."""
        return self._levels

    def __repr__(self) -> str:
        return super().__repr__().replace("NowcastDistribution", type(self).__name__, 1)

    def __getitem__(self, period: pd.Period | str | int) -> EmpiricalGaussianDistribution:
        """Distribution of a single period (label or integer position)."""
        return self.select([period])

    def select(self, periods: Iterable[pd.Period | str | int]) -> EmpiricalGaussianDistribution:
        """Bands restricted to some periods (keeping the default levels).

        Parameters
        ----------
        periods : iterable of Period, str or int
            Period labels or integer positions.

        Returns
        -------
        EmpiricalGaussianDistribution
            New object with the selected periods.

        Raises
        ------
        KeyError
            If a period is not in :attr:`index`.

        Examples
        --------
        >>> from nowcastbox.density import EmpiricalGaussianDistribution
        >>> d = EmpiricalGaussianDistribution(["2020Q1", "2020Q2"], [0.0, 1.0], [1.0, 2.0])
        >>> d.select(["2020Q2"]).std.tolist()
        [2.0]
        """
        base = super().select(periods)
        return EmpiricalGaussianDistribution(
            base.index,
            base.point.to_numpy(),
            base.scales[:, 0],
            target=self.target,
            levels=self._levels,
            info=self.info,
        )

    def to_frame(self, levels: float | Iterable[float] | None = None) -> pd.DataFrame:
        """Summary table: point, mean, std, median and band bounds.

        Parameters
        ----------
        levels : float or iterable of float, optional
            Band levels (default: :attr:`levels`).

        Returns
        -------
        pandas.DataFrame
            One row per period.

        Raises
        ------
        ValueError
            If a level is outside ``(0, 1)``.

        Examples
        --------
        >>> from nowcastbox.density import EmpiricalGaussianDistribution
        >>> d = EmpiricalGaussianDistribution(["2020Q1"], [0.0], [1.0], levels=[0.9])
        >>> list(d.to_frame().columns)
        ['point', 'mean', 'std', 'median', 'lower_90', 'upper_90']
        """
        return super().to_frame(self._levels if levels is None else levels)

    def fan_chart_frame(self, levels: float | Iterable[float] | None = None) -> pd.DataFrame:
        """Long-format bands for fan charts (widest band first).

        Parameters
        ----------
        levels : float or iterable of float, optional
            Band levels (default: :attr:`levels`).

        Returns
        -------
        pandas.DataFrame
            Columns ``period``, ``level``, ``lower``, ``upper``, ``median``.

        Raises
        ------
        ValueError
            If a level is outside ``(0, 1)``.

        Examples
        --------
        >>> from nowcastbox.density import EmpiricalGaussianDistribution
        >>> d = EmpiricalGaussianDistribution(["2020Q1"], [0.0], [1.0])
        >>> d.fan_chart_frame()["level"].tolist()
        [0.9, 0.68, 0.575]
        """
        return super().fan_chart_frame(self._levels if levels is None else levels)

    def plot(
        self,
        *,
        levels: float | Iterable[float] | None = None,
        history: pd.Series | None = None,
        ax: Axes | None = None,
        color: str = "C0",
        title: str | None = None,
    ) -> Axes:
        """Fan chart of the empirical bands (matplotlib).

        Parameters
        ----------
        levels : float or iterable of float, optional
            Band levels (default: :attr:`levels`), drawn as nested bands.
        history : pandas.Series, optional
            Observed target values (PeriodIndex of the same frequency).
        ax : matplotlib.axes.Axes, optional
            Axes to draw on (default: a new figure).
        color : str, default "C0"
            Colour of the bands and of the median line.
        title : str, optional
            Axes title (default: ``"Empirical error bands: <target>"``).

        Returns
        -------
        matplotlib.axes.Axes
            The axes.

        Examples
        --------
        >>> import matplotlib
        >>> matplotlib.use("Agg")
        >>> from nowcastbox.density import EmpiricalGaussianDistribution
        >>> d = EmpiricalGaussianDistribution(["2020Q1", "2020Q2"], [0.0, 1.0], [1.0, 2.0])
        >>> d.plot().get_title()
        'Empirical error bands:'
        """
        default = f"Empirical error bands: {self.target}".strip()
        return super().plot(
            levels=self._levels if levels is None else levels,
            history=history,
            ax=ax,
            color=color,
            title=default if title is None else title,
        )


# ====================================================================== bands for results
def _resolve_vintage(results: NowcastResults, vintage: pd.Timestamp | str | None) -> pd.Timestamp:
    """Vintage date of a nowcast: explicit, ``results.info["vintage"]`` or the data edge."""
    if vintage is not None:
        return pd.Timestamp(vintage).normalize()
    if results.info.get("vintage") is not None:
        return pd.Timestamp(results.info["vintage"]).normalize()
    if results.data is None:
        raise ValueError(
            "Cannot infer the vintage date: pass vintage=... (the results carry no data)."
        )
    observed = results.data.data.dropna(how="all")
    if observed.empty:
        raise ValueError("Cannot infer the vintage date: the estimation data are empty.")
    last = observed.index[-1]
    date = (last + 1).start_time.normalize()
    logger.info("Vintage date inferred from the data edge (%s): %s.", last, date.date())
    return date


def _point_nowcasts(
    results: NowcastResults, periods: Iterable[pd.Period | str] | pd.Period | str | None
) -> pd.Series:
    """Point nowcasts to wrap in bands (default: every out-of-sample period)."""
    if periods is None:
        point = results.out_of_sample.dropna()
        if point.empty:
            raise NowcastDataError("The results have no out-of-sample estimate.")
        return point
    items = [periods] if isinstance(periods, str | pd.Period) else list(periods)
    estimate = results.estimate
    freq = estimate.index.freqstr  # type: ignore[attr-defined]
    keys = [p.asfreq(freq) if isinstance(p, pd.Period) else pd.Period(str(p), freq) for p in items]
    selected = estimate.reindex(pd.PeriodIndex(keys))
    missing = [str(k) for k, v in zip(keys, selected.to_numpy(), strict=True) if pd.isna(v)]
    if missing:
        raise NowcastDataError(f"No estimate for the periods {missing}.")
    return selected


def _months_to_end(period: pd.Period, date: pd.Timestamp) -> int:
    return int(period.asfreq("M", how="E").ordinal - pd.Period(date, freq="M").ordinal)


def _warn_insufficient(skipped: dict[str, int], min_errors: int) -> None:
    if skipped:
        warnings.warn(
            f"Too few past errors (< {min_errors}) or a zero error scale for the periods "
            f"{skipped} (horizon in months to the end of the period); they get no bands.",
            DataQualityWarning,
            stacklevel=3,
        )


def empirical_bands(
    results: NowcastResults,
    backtest: BacktestLike,
    *,
    vintage: pd.Timestamp | str | None = None,
    model: str | None = None,
    window: WindowLike = "10Y",
    method: EmpiricalMethod = "mae",
    levels: float | Iterable[float] = EMPIRICAL_LEVELS,
    outliers: OutlierRule | None = None,
    outlier_threshold: float = 3.0,
    availability: Availability = "release",
    min_errors: int = 8,
    periods: Iterable[pd.Period | str] | pd.Period | str | None = None,
) -> NowcastDistribution | EmpiricalQuantileDistribution:
    r"""Empirical error bands around a nowcast from past errors at the same horizon.

    Parameters
    ----------
    results : NowcastResults
        Results holding the point nowcasts.
    backtest : BacktestResults or pandas.DataFrame
        Backtest of the same model (or its long forecast table) supplying the past
        errors; see :func:`available_errors`.
    vintage : Timestamp or str, optional
        Date of the nowcast's information set. Default: ``results.info["vintage"]``
        when present, otherwise the first day after the last base period with data.
    model : str, optional
        Backtest model whose errors are used (default: the main model).
    window : str, DateOffset, Timedelta or None, default "10Y"
        Rolling window of past vintages (``None`` = expanding).
    method : {"mae", "rmse", "quantile"}, default "mae"
        ``"mae"``: Gaussian with :math:`\sigma = \mathrm{MAE}\sqrt{\pi/2}` (ECB);
        ``"rmse"``: Gaussian with :math:`\sigma = \mathrm{RMSE}`; ``"quantile"``:
        empirical quantiles of the past errors.
    levels : float or iterable of float, default (0.575, 0.68, 0.9)
        Band levels (stored in ``info["levels"]``; default levels of the quantile
        object).
    outliers : {"winsorize", "exclude"} or None, default None
        Outlier adjustment of the past errors.
    outlier_threshold : float, default 3.0
        Threshold of the robust outlier rule (robust standard deviations).
    availability : {"release", "vintage"}, default "release"
        Information-set rule for the past errors (see :func:`available_errors`).
    min_errors : int, default 8
        Minimum number of past errors at a horizon; periods below it get no band
        (with a :class:`~nowcastbox.core.exceptions.DataQualityWarning`).
    periods : period, str or iterable, optional
        Target periods (default: every out-of-sample period).

    Returns
    -------
    NowcastDistribution or EmpiricalQuantileDistribution
        Gaussian :class:`EmpiricalGaussianDistribution` (a
        :class:`~nowcastbox.density.NowcastDistribution` whose tables and plots default
        to ``levels``) centred at the point nowcast (``"mae"``/``"rmse"``) or the
        empirical distribution (``"quantile"``). ``info`` holds the method,
        vintage, window, model, levels, the horizon and number of errors of each
        period and the periods left without bands.

    Raises
    ------
    NowcastDataError
        If no period has enough past errors, or a requested period has no estimate.
    ValueError
        On invalid options, a malformed backtest table or an unknown vintage.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from nowcastbox.core.results import NowcastResults, build_nowcast_frame
    >>> from nowcastbox.density import empirical_bands
    >>> vint = pd.date_range("2000-01-15", periods=80, freq="3MS") + pd.Timedelta(days=14)
    >>> past = pd.DataFrame(
    ...     {
    ...         "vintage": vint,
    ...         "target_period": pd.PeriodIndex(vint, freq="Q"),
    ...         "months_to_end": 2,
    ...         "days_to_release": 120.0,
    ...         "error": np.random.default_rng(0).normal(size=80),
    ...     }
    ... )
    >>> idx = pd.PeriodIndex(["2020Q1"], freq="Q")
    >>> res = NowcastResults(
    ...     target="gdp",
    ...     nowcast=build_nowcast_frame(
    ...         pd.Series([np.nan], index=idx), pd.Series([0.4], index=idx)
    ...     ),
    ... )
    >>> bands = empirical_bands(res, past, vintage="2020-01-20")
    >>> bands.info["horizons"], bands.info["n_errors"]
    ({'2020Q1': 2}, {'2020Q1': 39})
    >>> bool(bands.interval(0.575)["upper"].iloc[0] > 0.4)
    True
    """
    _check_common(method, outliers, availability, min_errors, outlier_threshold)
    lv = _check_levels(levels)
    date = _resolve_vintage(results, vintage)
    point = _point_nowcasts(results, periods)
    frame, chosen = _forecast_table(backtest, model)
    sub = _available(frame, date, _window_offset(window), availability)
    kept: dict[pd.Period, FloatArray] = {}
    horizons: dict[str, int] = {}
    skipped: dict[str, int] = {}
    for period in point.index:
        h = _months_to_end(period, date)
        errs = _horizon_errors(sub, h, outliers, outlier_threshold)
        if errs.size < min_errors or (method != "quantile" and _scale(errs, method) <= 0.0):
            skipped[str(period)] = h
            continue
        kept[period] = errs
        horizons[str(period)] = h
    _warn_insufficient(skipped, min_errors)
    if not kept:
        raise NowcastDataError(
            f"No target period has at least {min_errors} past errors at its horizon "
            f"before {date.date()}."
        )
    info = {
        "method": method,
        "vintage": date,
        "window": window,
        "model": chosen,
        "levels": tuple(lv),
        "outliers": outliers,
        "availability": availability,
        "horizons": horizons,
        "n_errors": {k: int(v.size) for k, v in zip(horizons, kept.values(), strict=True)},
        "skipped": skipped,
    }
    index = pd.PeriodIndex(list(kept))
    centre = point.loc[index].to_numpy(dtype=float)
    if method == "quantile":
        return EmpiricalQuantileDistribution(
            index, centre, list(kept.values()), target=results.target, levels=lv, info=info
        )
    scales = np.array([_scale(e, method) for e in kept.values()])
    return EmpiricalGaussianDistribution(
        index, centre, scales, target=results.target, levels=lv, info=info
    )


# ====================================================================== bands along a backtest
def _band_columns(
    errs: FloatArray, forecast: float, method: str, levels: list[float]
) -> dict[str, float]:
    """Sigma and band bounds of one forecast from its past errors."""
    out: dict[str, float] = {}
    if method == "quantile":
        out["sigma"] = float("nan")
        for v in levels:
            lo, hi = np.quantile(errs, [0.5 * (1 - v), 0.5 * (1 + v)])
            out[f"lower_{_level_label(v)}"] = forecast + float(lo)
            out[f"upper_{_level_label(v)}"] = forecast + float(hi)
        return out
    from scipy.stats import norm

    sigma = _scale(errs, method)
    out["sigma"] = sigma
    for v in levels:
        z = float(norm.ppf(0.5 * (1 + v)))
        out[f"lower_{_level_label(v)}"] = forecast - z * sigma
        out[f"upper_{_level_label(v)}"] = forecast + z * sigma
    return out


def _empty_band(levels: list[float]) -> dict[str, float]:
    nan = float("nan")
    out = {"sigma": nan}
    for v in levels:
        out[f"lower_{_level_label(v)}"] = nan
        out[f"upper_{_level_label(v)}"] = nan
    return out


def backtest_empirical_bands(
    backtest: BacktestLike,
    *,
    model: str | None = None,
    window: WindowLike = "10Y",
    method: EmpiricalMethod = "mae",
    levels: float | Iterable[float] = EMPIRICAL_LEVELS,
    outliers: OutlierRule | None = None,
    outlier_threshold: float = 3.0,
    availability: Availability = "release",
    min_errors: int = 8,
) -> pd.DataFrame:
    """Real-time empirical bands for every forecast of a backtest.

    Each forecast made at vintage :math:`v` gets the band computed only from the
    errors known at :math:`v` (same rules as :func:`empirical_bands`), so the
    coverage of the bands can be evaluated out of sample, e.g. with
    :func:`nowcastbox.evaluation.scoring.interval_coverage`.

    Parameters
    ----------
    backtest : BacktestResults or pandas.DataFrame
        Backtest results or their long forecast table (needs ``forecast`` and
        ``actual`` besides the columns of :func:`available_errors`).
    model : str, optional
        Model (default: the main model).
    window : str, DateOffset, Timedelta or None, default "10Y"
        Rolling window.
    method : {"mae", "rmse", "quantile"}, default "mae"
        Band method (see :func:`empirical_bands`).
    levels : float or iterable of float, default (0.575, 0.68, 0.9)
        Band levels.
    outliers : {"winsorize", "exclude"} or None, default None
        Outlier adjustment of the past errors.
    outlier_threshold : float, default 3.0
        Threshold of the robust outlier rule.
    availability : {"release", "vintage"}, default "release"
        Information-set rule.
    min_errors : int, default 8
        Minimum number of past errors; below it the band is NaN.

    Returns
    -------
    pandas.DataFrame
        Columns ``vintage``, ``target_period``, ``months_to_end``, ``forecast``,
        ``actual``, ``n_errors``, ``sigma`` (NaN for ``"quantile"``) and
        ``lower_<level>``/``upper_<level>``, one row per forecast.

    Raises
    ------
    ValueError
        On invalid options or a malformed table.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from nowcastbox.density import backtest_empirical_bands
    >>> vint = pd.date_range("2000-01-15", periods=60, freq="3MS") + pd.Timedelta(days=14)
    >>> err = np.random.default_rng(1).normal(size=60)
    >>> table = pd.DataFrame(
    ...     {
    ...         "vintage": vint,
    ...         "target_period": pd.PeriodIndex(vint, freq="Q"),
    ...         "months_to_end": 2,
    ...         "days_to_release": 120.0,
    ...         "forecast": 0.0,
    ...         "actual": err,
    ...         "error": err,
    ...     }
    ... )
    >>> out = backtest_empirical_bands(table, levels=[0.9], min_errors=8)
    >>> int(out["sigma"].notna().sum())
    51
    """
    _check_common(method, outliers, availability, min_errors, outlier_threshold)
    lv = _check_levels(levels)
    frame, _ = _forecast_table(backtest, model)
    offset = _window_offset(window)
    rows = []
    for date, group in frame.groupby("vintage", sort=True):
        sub = _available(frame, pd.Timestamp(date), offset, availability)  # type: ignore[arg-type]
        for _, row in group.iterrows():
            errs = _horizon_errors(sub, int(row[_HORIZON]), outliers, outlier_threshold)
            enough = errs.size >= min_errors
            forecast = float(row.get("forecast", np.nan))
            band = _band_columns(errs, forecast, method, lv) if enough else _empty_band(lv)
            rows.append(
                {
                    "vintage": date,
                    "target_period": row["target_period"],
                    _HORIZON: int(row[_HORIZON]),
                    "forecast": forecast,
                    "actual": float(row.get("actual", np.nan)),
                    "n_errors": int(errs.size),
                    **band,
                }
            )
    return pd.DataFrame(rows)
