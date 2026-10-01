r"""Targeted predictors (Bai & Ng, 2008): pre-selection of the series used by a factor model.

Two rules are implemented (innovation I7 of the development plan):

**Hard thresholding.** For each candidate predictor ``x_i`` the regression

.. math::

    y_{t+h} = \alpha' W_t + \beta_i x_{it} + \varepsilon_{t+h}

is estimated by OLS, where ``W_t`` holds a constant and ``y_lags`` recent values of the
target. Predictor ``i`` is kept when ``|t_i| > `` ``threshold`` (Bai & Ng use the
critical values 1.28, 1.65 and 2.58). The ``t`` statistic uses either the classical or
the Newey & West (1987) HAC standard error (Bartlett kernel), which is appropriate for
the serially correlated errors that overlapping ``h``-step forecasts generate.

**Soft thresholding.** ``y_{t+h}`` is regressed on all standardised predictors with the
elastic net (Zou & Hastie, 2005)

.. math::

    \min_\beta \frac{1}{2T}\lVert y - X\beta \rVert_2^2
    + \alpha\Big(\rho \lVert\beta\rVert_1 + \frac{1-\rho}{2}\lVert\beta\rVert_2^2\Big),

solved by cyclic coordinate descent (Friedman, Hastie & Tibshirani, 2010) along a
decreasing grid of penalties with warm starts. As in Bai & Ng (2008), who rank series
by the order in which LARS-EN activates them, predictors are ranked by the point of the
path at which they first enter the active set and the first ``n_predictors`` are kept
(or, with a fixed ``alpha``, all predictors with non-zero coefficient).

Timing convention
-----------------
Rows where the target is missing are removed **before** shifting, so ``horizon`` and
``y_lags`` count *observations of the target*. A quarterly target stored in the third
month of each quarter on a monthly grid is therefore paired with the predictors of that
month, and ``horizon=1`` means one quarter ahead. With ``horizon=0`` the controls start
at ``y_{t-1}`` (``y_t`` itself is the left-hand side).

References
----------
Bai, J., & Ng, S. (2008). Forecasting economic time series using targeted predictors.
*Journal of Econometrics*, 146(2), 304-317.

Zou, H., & Hastie, T. (2005). Regularization and variable selection via the elastic
net. *Journal of the Royal Statistical Society B*, 67(2), 301-320.

Friedman, J., Hastie, T., & Tibshirani, R. (2010). Regularization paths for
generalized linear models via coordinate descent. *Journal of Statistical Software*,
33(1), 1-22.

Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity
and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703-708.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from nowcastbox._logging import get_logger
from nowcastbox.core.data import MixedFrequencyData
from nowcastbox.core.exceptions import ConvergenceWarning, DataQualityWarning, NowcastDataError

try:  # pragma: no cover - exercised implicitly
    from numba import njit
except ImportError:  # pragma: no cover
    njit = None

__all__ = [
    "TargetedPredictorsResult",
    "elastic_net",
    "elastic_net_path",
    "hard_threshold",
    "newey_west_lags",
    "select_targeted_predictors",
    "soft_threshold",
]

logger = get_logger(__name__)

TargetLike = pd.Series | np.ndarray | str
PredictorsLike = pd.DataFrame | np.ndarray | MixedFrequencyData


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TargetedPredictorsResult:
    """Outcome of :func:`hard_threshold` / :func:`soft_threshold`.

    Attributes
    ----------
    selected : list[str]
        Selected predictors, ordered by importance (``|t|`` for the hard rule, entry
        order along the elastic-net path for the soft rule).
    method : str
        ``"hard"`` or ``"soft"``.
    scores : pandas.Series
        Score of every candidate: ``|t|`` statistic (hard) or the elastic-net
        coefficient on standardised predictors at the selected penalty (soft).
    ranking : pandas.Series
        Rank of every candidate (1 = most important; soft rule: entry step on the path,
        ``NaN`` if never active).
    horizon : int
        Forecast horizon ``h`` (in observations of the target).
    n_obs : int
        Number of observations used (soft rule) or the maximum over predictors (hard).
    params : dict
        Settings: threshold / covariance type (hard), ``alpha`` / ``l1_ratio`` (soft).
    """

    selected: list[str]
    method: str
    scores: pd.Series
    ranking: pd.Series
    horizon: int
    n_obs: int
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def n_selected(self) -> int:
        """Number of selected predictors."""
        return len(self.selected)

    def transform(self, x: pd.DataFrame | MixedFrequencyData) -> Any:
        """Keep only the selected predictors (plus nothing else).

        Parameters
        ----------
        x : DataFrame or MixedFrequencyData
            Panel containing the selected columns.

        Returns
        -------
        DataFrame or MixedFrequencyData
            Same type as ``x`` restricted to :attr:`selected`.

        Raises
        ------
        NowcastDataError
            If a selected column is missing from ``x``.

        Examples
        --------
        >>> import numpy as np, pandas as pd
        >>> rng = np.random.default_rng(0)
        >>> x = pd.DataFrame(rng.normal(size=(100, 3)), columns=["a", "b", "c"])
        >>> y = 2 * x["a"] + rng.normal(scale=0.1, size=100)
        >>> list(hard_threshold(x, y).transform(x).columns)
        ['a']
        """
        cols = list(x.columns)
        missing = [c for c in self.selected if c not in cols]
        if missing:
            raise NowcastDataError(f"Selected predictors not found: {missing}.")
        if isinstance(x, MixedFrequencyData):
            return x.select(self.selected)
        return x.loc[:, self.selected]

    def summary(self) -> str:
        """Plain-text summary.

        Returns
        -------
        str
            Method, settings and the ranked selection.

        Examples
        --------
        >>> import numpy as np, pandas as pd
        >>> rng = np.random.default_rng(0)
        >>> x = pd.DataFrame(rng.normal(size=(100, 3)), columns=["a", "b", "c"])
        >>> y = 2 * x["a"] + rng.normal(scale=0.1, size=100)
        >>> "Targeted predictors" in hard_threshold(x, y).summary()
        True
        """
        settings = ", ".join(f"{k}={v}" for k, v in self.params.items())
        lines = [
            f"Targeted predictors - Bai & Ng (2008), {self.method} thresholding",
            "=" * 60,
            f"horizon: {self.horizon}   observations: {self.n_obs}   {settings}",
            f"Selected {self.n_selected} of {len(self.scores)} predictors:",
        ]
        lines += [
            f"  {i + 1:>3}. {name}  ({self.scores[name]: .4f})"
            for i, name in enumerate(self.selected)
        ]
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


# ---------------------------------------------------------------------------
# Data alignment
# ---------------------------------------------------------------------------
def _predictor_frame(x: PredictorsLike) -> pd.DataFrame:
    if isinstance(x, MixedFrequencyData):
        return x.to_frame()
    if isinstance(x, pd.DataFrame):
        return x.copy()
    if isinstance(x, np.ndarray) and x.ndim == 2:
        return pd.DataFrame(x, columns=[f"x{i + 1}" for i in range(x.shape[1])])
    raise TypeError("x must be a DataFrame, a 2-D ndarray or a MixedFrequencyData.")


def _target_series(y: pd.Series | np.ndarray, frame: pd.DataFrame, positional: bool) -> pd.Series:
    if isinstance(y, pd.Series):
        if y.index.equals(frame.index):
            return y.copy()
        if positional and len(y) == len(frame):
            return pd.Series(y.to_numpy(), index=frame.index)
        target = y.reindex(frame.index)
        if target.notna().sum() == 0:
            raise NowcastDataError("y and x share no index labels.")
        return target
    arr = np.asarray(y, dtype=float).ravel()
    if len(arr) != len(frame):
        raise NowcastDataError(f"y has {len(arr)} values but x has {len(frame)} rows.")
    return pd.Series(arr, index=frame.index)


def _split_inputs(x: PredictorsLike, y: TargetLike) -> tuple[pd.DataFrame, pd.Series]:
    frame = _predictor_frame(x)
    if isinstance(y, str):
        if y not in frame.columns:
            raise NowcastDataError(f"Target {y!r} is not a column of x.")
        target = frame.pop(y)
    else:
        target = _target_series(y, frame, positional=isinstance(x, np.ndarray))
    try:
        frame = frame.astype(float)
        target = target.astype(float)
    except (TypeError, ValueError) as exc:
        raise NowcastDataError(f"Non-numeric data: {exc}") from exc
    if np.isinf(frame.to_numpy()).any() or np.isinf(target.to_numpy()).any():
        raise NowcastDataError("Data contain infinite values.")
    if frame.shape[1] == 0:
        raise NowcastDataError("No candidate predictors.")
    return frame, target


def _check_nonneg_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer; got {value!r}.")
    return int(value)


def _align(
    x: PredictorsLike, y: TargetLike, horizon: int, y_lags: int
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Return predictors at ``t``, target at ``t+h`` and the controls ``W_t``."""
    frame, target = _split_inputs(x, y)
    observed = target.notna().to_numpy()
    frame = frame.loc[observed]
    yv = target.to_numpy()[observed]
    first = 0 if horizon > 0 else 1  # most recent usable lag of y
    start = max(0, first + y_lags - 1) if y_lags else 0
    n = len(yv)
    stop = n - horizon
    if stop - start < 3:
        raise NowcastDataError(
            f"Too few target observations ({n}) for horizon={horizon} and y_lags={y_lags}."
        )
    rows = np.arange(start, stop)
    lhs = yv[rows + horizon]
    controls = [np.ones(len(rows))] + [yv[rows - first - j] for j in range(y_lags)]
    return frame.iloc[rows], lhs, np.column_stack(controls)


# ---------------------------------------------------------------------------
# Hard thresholding
# ---------------------------------------------------------------------------
def newey_west_lags(n_obs: int) -> int:
    """Default Bartlett bandwidth ``floor(4 (T/100)^(2/9))`` (Newey & West, 1994).

    Parameters
    ----------
    n_obs : int
        Sample size.

    Returns
    -------
    int
        Number of lags.

    Examples
    --------
    >>> newey_west_lags(100)
    4
    """
    return int(np.floor(4.0 * (n_obs / 100.0) ** (2.0 / 9.0)))


def _ols_tstat(design: np.ndarray, lhs: np.ndarray, cov_type: str, hac_lags: int) -> float:
    """Return the t statistic of the last regressor."""
    n_obs, k = design.shape
    xtx_inv = np.linalg.pinv(design.T @ design)
    beta = xtx_inv @ design.T @ lhs
    resid = lhs - design @ beta
    if cov_type == "nonrobust":
        var = resid @ resid / (n_obs - k) * xtx_inv[-1, -1]
    else:
        scores = design * resid[:, None]
        meat = scores.T @ scores
        for lag in range(1, min(hac_lags, n_obs - 1) + 1):
            weight = 1.0 - lag / (hac_lags + 1.0)
            gamma = scores[lag:].T @ scores[:-lag]
            meat += weight * (gamma + gamma.T)
        var = (xtx_inv @ meat @ xtx_inv)[-1, -1]
    if not var > 0:
        return 0.0
    return float(beta[-1] / np.sqrt(var))


def hard_threshold(
    x: PredictorsLike,
    y: TargetLike,
    *,
    horizon: int = 0,
    y_lags: int = 0,
    threshold: float = 1.65,
    cov_type: Literal["hac", "nonrobust"] = "hac",
    hac_lags: int | None = None,
    max_predictors: int | None = None,
    min_obs: int = 10,
) -> TargetedPredictorsResult:
    """Targeted predictors by hard thresholding (Bai & Ng, 2008, §3.1).

    Parameters
    ----------
    x : DataFrame, ndarray or MixedFrequencyData
        Candidate predictors (rows = periods).
    y : Series, ndarray or str
        Target aligned with ``x`` (a column name of ``x`` is accepted and removed from
        the candidates).
    horizon : int, default 0
        Forecast horizon ``h`` in observations of the target.
    y_lags : int, default 0
        Number of recent target values in the controls ``W_t`` (a constant is always
        included).
    threshold : float, default 1.65
        Critical value for ``|t|`` (Bai & Ng use 1.28, 1.65 and 2.58).
    cov_type : {"hac", "nonrobust"}, default "hac"
        Newey-West (Bartlett) or classical OLS standard errors.
    hac_lags : int, optional
        HAC bandwidth; default :func:`newey_west_lags` of the sample size.
    max_predictors : int, optional
        Keep at most this many predictors (the largest ``|t|``).
    min_obs : int, default 10
        Minimum number of complete observations per regression; predictors with fewer
        are skipped (``|t|`` = NaN) with a :class:`DataQualityWarning`.

    Returns
    -------
    TargetedPredictorsResult
        Selected predictors ordered by ``|t|``; ``scores`` holds ``|t|`` for all.

    Raises
    ------
    ValueError
        Invalid tuning parameters.
    NowcastDataError
        Invalid or insufficient data.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> x = pd.DataFrame(rng.normal(size=(200, 4)), columns=list("abcd"))
    >>> y = x["a"] - x["c"] + rng.normal(size=200)
    >>> hard_threshold(x, y, threshold=2.58).selected
    ['c', 'a']
    """
    horizon = _check_nonneg_int("horizon", horizon)
    y_lags = _check_nonneg_int("y_lags", y_lags)
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError(f"threshold must be a non-negative number; got {threshold!r}.")
    if cov_type not in ("hac", "nonrobust"):
        raise ValueError(f"cov_type must be 'hac' or 'nonrobust'; got {cov_type!r}.")
    if hac_lags is not None:
        hac_lags = _check_nonneg_int("hac_lags", hac_lags)
    if max_predictors is not None and _check_nonneg_int("max_predictors", max_predictors) == 0:
        raise ValueError("max_predictors must be positive.")
    preds, lhs, controls = _align(x, y, horizon, y_lags)
    tstats, n_used = _hard_tstats(preds, lhs, controls, cov_type, hac_lags, min_obs)
    abs_t = pd.Series(np.abs(tstats), index=[str(c) for c in preds.columns], name="abs_t")
    order = abs_t.sort_values(ascending=False, na_position="last", kind="mergesort")
    selected = [c for c in order.index if order[c] > threshold]
    if max_predictors is not None:
        selected = selected[:max_predictors]
    if not selected:
        warnings.warn(
            f"No predictor has |t| > {threshold}; the selection is empty.",
            DataQualityWarning,
            stacklevel=2,
        )
    ranking = abs_t.rank(ascending=False, method="first").rename("rank")
    return TargetedPredictorsResult(
        selected=selected,
        method="hard",
        scores=abs_t,
        ranking=ranking,
        horizon=horizon,
        n_obs=n_used,
        params={"threshold": threshold, "cov_type": cov_type, "y_lags": y_lags},
    )


def _hard_tstats(
    preds: pd.DataFrame,
    lhs: np.ndarray,
    controls: np.ndarray,
    cov_type: str,
    hac_lags: int | None,
    min_obs: int,
) -> tuple[np.ndarray, int]:
    values = preds.to_numpy()
    tstats = np.full(values.shape[1], np.nan)
    skipped: list[str] = []
    n_used = 0
    for i in range(values.shape[1]):
        ok = ~np.isnan(values[:, i])
        n_ok = int(ok.sum())
        if n_ok < max(min_obs, controls.shape[1] + 2):
            skipped.append(str(preds.columns[i]))
            continue
        col = values[ok, i]
        if np.ptp(col) == 0:
            skipped.append(str(preds.columns[i]))
            continue
        design = np.column_stack([controls[ok], col])
        lags = newey_west_lags(n_ok) if hac_lags is None else hac_lags
        tstats[i] = _ols_tstat(design, lhs[ok], cov_type, lags)
        n_used = max(n_used, n_ok)
    if skipped:
        warnings.warn(
            f"{len(skipped)} predictors skipped (constant or too few observations): {skipped}.",
            DataQualityWarning,
            stacklevel=3,
        )
    if n_used == 0:
        raise NowcastDataError("No predictor has enough observations for the regression.")
    return tstats, n_used


# ---------------------------------------------------------------------------
# Elastic net (coordinate descent)
# ---------------------------------------------------------------------------
def _cd_loop(
    gram: np.ndarray,
    xty: np.ndarray,
    beta: np.ndarray,
    l1: float,
    l2: float,
    max_iter: int,
    tol: float,
) -> int:
    """Cyclic coordinate descent on the covariance form; updates ``beta`` in place.

    Minimises ``0.5 b'Gb - c'b + l1 |b|_1 + 0.5 l2 |b|^2``. Returns the number of sweeps
    (``max_iter + 1`` when not converged).
    """
    p = beta.shape[0]
    g_beta = gram @ beta
    for sweep in range(max_iter):
        max_change = 0.0
        max_coef = 0.0
        for j in range(p):
            gjj = gram[j, j]
            if gjj <= 0.0:
                beta[j] = 0.0
                continue
            old = beta[j]
            rho = xty[j] - g_beta[j] + gjj * old
            new = np.sign(rho) * max(abs(rho) - l1, 0.0) / (gjj + l2)
            delta = new - old
            if delta != 0.0:
                beta[j] = new
                for i in range(p):
                    g_beta[i] += gram[i, j] * delta
                if abs(delta) > max_change:
                    max_change = abs(delta)
            if abs(new) > max_coef:
                max_coef = abs(new)
        if max_change <= tol * max(max_coef, 1.0):
            return sweep + 1
    return max_iter + 1


_cd_loop_fast = njit(cache=True)(_cd_loop) if njit is not None else _cd_loop


def _check_en_params(alpha: float, l1_ratio: float, max_iter: int, tol: float) -> None:
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError(f"alpha must be a non-negative number; got {alpha!r}.")
    if not np.isfinite(l1_ratio) or not 0.0 < l1_ratio <= 1.0:
        raise ValueError(f"l1_ratio must lie in (0, 1]; got {l1_ratio!r}.")
    if isinstance(max_iter, bool) or not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError(f"max_iter must be a positive integer; got {max_iter!r}.")
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError(f"tol must be positive; got {tol!r}.")


def _prepare_xy(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if X.ndim != 2 or X.shape[0] != y.shape[0]:
        raise NowcastDataError(f"X must be 2-D with {y.shape[0]} rows; got shape {X.shape}.")
    if not (np.isfinite(X).all() and np.isfinite(y).all()):
        raise NowcastDataError("X and y must be finite (remove missing values first).")
    return X, y


def elastic_net(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    l1_ratio: float = 0.5,
    *,
    fit_intercept: bool = True,
    max_iter: int = 10_000,
    tol: float = 1e-8,
    coef_init: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    r"""Elastic-net regression by cyclic coordinate descent.

    Minimises ``(1/(2n)) ||y - c - X b||^2 + alpha (l1_ratio ||b||_1 +
    (1 - l1_ratio)/2 ||b||_2^2)`` (Zou & Hastie, 2005; Friedman et al., 2010). The
    intercept ``c`` is not penalised.

    Parameters
    ----------
    X : numpy.ndarray
        ``(n, p)`` design matrix (finite).
    y : numpy.ndarray
        ``(n,)`` response.
    alpha : float
        Overall penalty (``>= 0``; 0 gives least squares when well posed).
    l1_ratio : float, default 0.5
        Mixing parameter ``rho`` in ``(0, 1]`` (1 = lasso).
    fit_intercept : bool, default True
        Center ``X`` and ``y`` and return the intercept.
    max_iter : int, default 10000
        Maximum number of coordinate sweeps.
    tol : float, default 1e-8
        Convergence tolerance on the largest coefficient change (relative).
    coef_init : numpy.ndarray, optional
        Warm start.

    Returns
    -------
    coef : numpy.ndarray
        ``(p,)`` coefficients.
    intercept : float
        Intercept (0 when ``fit_intercept=False``).

    Raises
    ------
    ValueError
        Invalid tuning parameters.
    NowcastDataError
        Non-finite or misaligned data.

    Warns
    -----
    ConvergenceWarning
        If ``max_iter`` sweeps are reached.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.eye(4) * 2.0
    >>> coef, _ = elastic_net(X, np.array([4.0, 0.2, 0.0, 0.0]), 0.1, 1.0, fit_intercept=False)
    >>> coef.round(3).tolist()
    [1.9, 0.0, 0.0, 0.0]
    """
    _check_en_params(alpha, l1_ratio, max_iter, tol)
    X, y = _prepare_xy(X, y)
    n = X.shape[0]
    if fit_intercept:
        x_mean, y_mean = X.mean(axis=0), y.mean()
        Xc, yc = X - x_mean, y - y_mean
    else:
        x_mean, y_mean = np.zeros(X.shape[1]), 0.0
        Xc, yc = X, y
    gram = Xc.T @ Xc / n
    xty = Xc.T @ yc / n
    beta = np.zeros(X.shape[1]) if coef_init is None else np.array(coef_init, dtype=float)
    if beta.shape != (X.shape[1],):
        raise ValueError(f"coef_init must have shape ({X.shape[1]},).")
    sweeps = _cd_loop_fast(
        gram, xty, beta, alpha * l1_ratio, alpha * (1.0 - l1_ratio), int(max_iter), tol
    )
    if sweeps > max_iter:
        warnings.warn(
            f"Coordinate descent did not converge in {max_iter} sweeps (alpha={alpha:.4g}).",
            ConvergenceWarning,
            stacklevel=2,
        )
    return beta, float(y_mean - x_mean @ beta)


def elastic_net_path(
    X: np.ndarray,
    y: np.ndarray,
    l1_ratio: float = 0.5,
    *,
    n_alphas: int = 100,
    eps: float = 1e-3,
    alphas: np.ndarray | None = None,
    max_iter: int = 10_000,
    tol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Elastic-net coefficients along a decreasing penalty grid (warm starts).

    The default grid is log-spaced from ``alpha_max = max_j |x_j'(y - ybar)| / (n
    l1_ratio)`` (smallest penalty with all coefficients zero) down to ``eps *
    alpha_max``. An intercept is always fitted (data are centred).

    Parameters
    ----------
    X : numpy.ndarray
        ``(n, p)`` design.
    y : numpy.ndarray
        ``(n,)`` response.
    l1_ratio : float, default 0.5
        Mixing parameter in ``(0, 1]``.
    n_alphas : int, default 100
        Grid size (ignored if ``alphas`` is given).
    eps : float, default 1e-3
        Ratio ``alpha_min / alpha_max``.
    alphas : numpy.ndarray, optional
        Explicit grid (sorted decreasingly internally).
    max_iter, tol
        Passed to the coordinate-descent solver.

    Returns
    -------
    alphas : numpy.ndarray
        ``(n_alphas,)`` decreasing penalties.
    coefs : numpy.ndarray
        ``(n_alphas, p)`` coefficients.

    Raises
    ------
    ValueError
        Invalid grid parameters.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(50, 5))
    >>> y = X[:, 0] + rng.normal(size=50)
    >>> a, b = elastic_net_path(X, y, 1.0, n_alphas=10)
    >>> bool(np.all(b[0] == 0)), bool(b[-1, 0] != 0)
    (True, True)
    """
    X, y = _prepare_xy(X, y)
    if alphas is None:
        if isinstance(n_alphas, bool) or not isinstance(n_alphas, (int, np.integer)):
            raise ValueError(f"n_alphas must be an integer; got {n_alphas!r}.")
        if n_alphas < 2 or not 0 < eps < 1:
            raise ValueError("n_alphas must be >= 2 and eps in (0, 1).")
        _check_en_params(0.0, l1_ratio, max_iter, tol)
        xty = (X - X.mean(axis=0)).T @ (y - y.mean()) / X.shape[0]
        alpha_max = float(np.max(np.abs(xty))) / l1_ratio
        if alpha_max <= 0:
            alpha_max = 1.0
        grid = alpha_max * np.logspace(0.0, np.log10(eps), int(n_alphas))
    else:
        grid = np.sort(np.asarray(alphas, dtype=float).ravel())[::-1]
        if grid.size == 0:
            raise ValueError("alphas must not be empty.")
    coefs = np.zeros((grid.size, X.shape[1]))
    beta = np.zeros(X.shape[1])
    for i, alpha in enumerate(grid):
        beta, _ = elastic_net(X, y, alpha, l1_ratio, max_iter=max_iter, tol=tol, coef_init=beta)
        coefs[i] = beta
    return grid, coefs


# ---------------------------------------------------------------------------
# Soft thresholding
# ---------------------------------------------------------------------------
def _complete_rows(preds: pd.DataFrame, lhs: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    ok = ~preds.isna().any(axis=1).to_numpy()
    if not ok.all():
        warnings.warn(
            f"Dropped {int((~ok).sum())} of {len(ok)} observations with missing predictors.",
            DataQualityWarning,
            stacklevel=3,
        )
    preds, lhs = preds.loc[ok], lhs[ok]
    if len(lhs) < 3:
        raise NowcastDataError("Too few complete observations for the elastic net.")
    return preds, lhs


def _entry_steps(coefs: np.ndarray) -> np.ndarray:
    active = coefs != 0
    ever = active.any(axis=0)
    steps = active.argmax(axis=0).astype(float)
    steps[~ever] = np.nan
    return steps


def soft_threshold(
    x: PredictorsLike,
    y: TargetLike,
    *,
    horizon: int = 0,
    n_predictors: int | None = None,
    alpha: float | None = None,
    l1_ratio: float = 0.5,
    n_alphas: int = 100,
    eps: float = 1e-3,
    max_iter: int = 10_000,
    tol: float = 1e-8,
) -> TargetedPredictorsResult:
    """Targeted predictors by soft thresholding with the elastic net (Bai & Ng, 2008, §3.2).

    Predictors are standardised (``ddof=0``) on the complete observations. With a fixed
    ``alpha`` every predictor with a non-zero coefficient is kept, ordered by ``|coef|``.
    Otherwise the elastic-net path is computed and the first ``n_predictors`` series to
    enter the active set are kept (default ``min(30, N)``, the size used by Bai & Ng).

    Parameters
    ----------
    x : DataFrame, ndarray or MixedFrequencyData
        Candidate predictors (rows = periods).
    y : Series, ndarray or str
        Target aligned with ``x`` (or a column name of ``x``).
    horizon : int, default 0
        Forecast horizon ``h`` in observations of the target.
    n_predictors : int, optional
        Number of predictors to keep (path mode).
    alpha : float, optional
        Fixed penalty; mutually exclusive with ``n_predictors``.
    l1_ratio : float, default 0.5
        Elastic-net mixing parameter in ``(0, 1]``.
    n_alphas, eps : int, float
        Path grid (see :func:`elastic_net_path`).
    max_iter, tol
        Coordinate-descent settings.

    Returns
    -------
    TargetedPredictorsResult
        Selected predictors; ``scores`` are the coefficients at the selected penalty
        (stored in ``params["alpha"]``) and ``ranking`` the entry step on the path.

    Raises
    ------
    ValueError
        Invalid or conflicting tuning parameters.
    NowcastDataError
        Invalid or insufficient data.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> x = pd.DataFrame(rng.normal(size=(200, 6)), columns=list("abcdef"))
    >>> y = 2 * x["b"] - x["e"] + rng.normal(scale=0.5, size=200)
    >>> soft_threshold(x, y, n_predictors=2).selected
    ['b', 'e']
    """
    horizon = _check_nonneg_int("horizon", horizon)
    if alpha is not None and n_predictors is not None:
        raise ValueError("Pass either alpha or n_predictors, not both.")
    preds, lhs, _ = _align(x, y, horizon, 0)
    preds, lhs = _complete_rows(preds, lhs)
    names = [str(c) for c in preds.columns]
    values = preds.to_numpy()
    std = values.std(axis=0)
    if np.any(std == 0):
        raise NowcastDataError(
            f"Constant predictors: {[n for n, s in zip(names, std, strict=True) if s == 0]}."
        )
    z = (values - values.mean(axis=0)) / std
    if alpha is not None:
        coef, _ = elastic_net(z, lhs, alpha, l1_ratio, max_iter=max_iter, tol=tol)
        scores = pd.Series(coef, index=names, name="coef")
        nonzero = scores[scores != 0]
        selected = list(nonzero.abs().sort_values(ascending=False, kind="mergesort").index)
        ranking = scores.abs().where(scores != 0).rank(ascending=False, method="first")
        chosen_alpha = float(alpha)
    else:
        selected, scores, ranking, chosen_alpha = _soft_path_selection(
            z, lhs, names, n_predictors, l1_ratio, n_alphas, eps, max_iter, tol
        )
    if not selected:
        warnings.warn("The elastic net selected no predictor.", DataQualityWarning, stacklevel=2)
    return TargetedPredictorsResult(
        selected=selected,
        method="soft",
        scores=scores,
        ranking=ranking.rename("rank"),
        horizon=horizon,
        n_obs=len(lhs),
        params={"alpha": chosen_alpha, "l1_ratio": l1_ratio},
    )


def _soft_path_selection(
    z: np.ndarray,
    lhs: np.ndarray,
    names: list[str],
    n_predictors: int | None,
    l1_ratio: float,
    n_alphas: int,
    eps: float,
    max_iter: int,
    tol: float,
) -> tuple[list[str], pd.Series, pd.Series, float]:
    p = z.shape[1]
    k = min(30, p) if n_predictors is None else _check_nonneg_int("n_predictors", n_predictors)
    if k < 1 or k > p:
        raise ValueError(f"n_predictors must lie in [1, {p}]; got {n_predictors!r}.")
    grid, coefs = elastic_net_path(
        z, lhs, l1_ratio, n_alphas=n_alphas, eps=eps, max_iter=max_iter, tol=tol
    )
    n_active = (coefs != 0).sum(axis=1)
    reached = np.flatnonzero(n_active >= k)
    step = int(reached[0]) if reached.size else len(grid) - 1
    if not reached.size:
        warnings.warn(
            f"Only {int(n_active[-1])} predictors became active along the path "
            f"(requested {k}); decrease eps or increase n_alphas.",
            DataQualityWarning,
            stacklevel=3,
        )
    entry = _entry_steps(coefs)
    # Order by entry step, ties broken by |coef| at the selected penalty
    order = sorted(
        (j for j in range(p) if not np.isnan(entry[j]) and entry[j] <= step),
        key=lambda j: (entry[j], -abs(coefs[step, j])),
    )
    selected = [names[j] for j in order[:k]]
    scores = pd.Series(coefs[step], index=names, name="coef")
    ranking = pd.Series(np.nan, index=names)
    for pos, j in enumerate(order):
        ranking.iloc[j] = pos + 1
    return selected, scores, ranking, float(grid[step])


def select_targeted_predictors(
    x: PredictorsLike,
    y: TargetLike,
    method: Literal["hard", "soft"] = "hard",
    **kwargs: Any,
) -> TargetedPredictorsResult:
    """Dispatch to :func:`hard_threshold` or :func:`soft_threshold`.

    Parameters
    ----------
    x : DataFrame, ndarray or MixedFrequencyData
        Candidate predictors.
    y : Series, ndarray or str
        Target (or a column name of ``x``).
    method : {"hard", "soft"}, default "hard"
        Thresholding rule.
    **kwargs
        Keyword arguments of the chosen rule.

    Returns
    -------
    TargetedPredictorsResult
        The selection.

    Raises
    ------
    ValueError
        Unknown method.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(1)
    >>> df = pd.DataFrame(rng.normal(size=(150, 3)), columns=["a", "b", "c"])
    >>> df["y"] = 3 * df["b"] + rng.normal(size=150)
    >>> select_targeted_predictors(df, "y", method="soft", n_predictors=1).selected
    ['b']
    """
    if method == "hard":
        return hard_threshold(x, y, **kwargs)
    if method == "soft":
        return soft_threshold(x, y, **kwargs)
    raise ValueError(f"method must be 'hard' or 'soft'; got {method!r}.")
