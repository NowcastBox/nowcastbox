r"""Tests of equal predictive accuracy and the Model Confidence Set.

* :func:`diebold_mariano` - Diebold & Mariano (1995) test of
  :math:`H_0: E[d_t] = 0` for the loss differential
  :math:`d_t = L(e_{1t}) - L(e_{2t})`, with the long-run variance
  :math:`\hat\gamma_0 + 2\sum_{k=1}^{h-1}\hat\gamma_k` (rectangular truncation at
  :math:`h-1` for :math:`h`-step forecasts) and the small-sample correction of
  Harvey, Leybourne & Newbold (1997):
  :math:`DM^* = DM\,\sqrt{(n + 1 - 2h + h(h-1)/n)/n}` compared with Student's
  :math:`t_{n-1}`.
* :func:`clark_west` - Clark & West (2007) test of equal MSPE for *nested* models:
  the larger model's squared error is adjusted for the noise of estimating
  parameters that are zero under :math:`H_0`,
  :math:`d_t = e_{1t}^2 - (e_{1t} - e_{2t})^2 - e_{2t}^2` (model 1 nests model 2),
  and :math:`\sqrt{n}\,\bar d / \hat\sigma` is compared with the standard normal.
* :func:`giacomini_white` - Giacomini & White (2006) conditional test of
  :math:`H_0: E[d_t \mid \mathcal F_{t-h}] = 0` through the moment conditions
  :math:`E[h_{t-h} d_t] = 0`: :math:`GW = n\,\bar Z' \hat\Omega^{-1} \bar Z
  \sim \chi^2_q` with :math:`Z_t = h_{t-h} d_t` (default instruments
  :math:`h_{t-h} = (1, d_{t-h})'`) and :math:`\hat\Omega` the sample second-moment
  matrix of :math:`Z_t` (Newey-West with :math:`h-1` lags for :math:`h > 1`).
* :func:`pesaran_timmermann` - Pesaran & Timmermann (1992) test of directional
  predictability: with :math:`\hat P` the share of correctly predicted signs of the
  change, :math:`P_y`, :math:`P_x` the shares of positive actual and predicted
  changes and :math:`P_* = P_y P_x + (1-P_y)(1-P_x)` the hit rate expected under
  independence, :math:`S_n = (\hat P - \hat P_*) / \{\hat V(\hat P) -
  \hat V(\hat P_*)\}^{1/2}` is asymptotically standard normal under :math:`H_0`
  (no directional predictability).
* :func:`model_confidence_set` - Model Confidence Set of Hansen, Lunde & Nason (2011)
  with the :math:`T_{\max,\mathcal M}` or :math:`T_{R,\mathcal M}` statistic, the
  corresponding elimination rules and a (circular) moving-block bootstrap
  (Künsch, 1989) with a fixed seed by default.

References
----------
Diebold, F. X. & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of
Business & Economic Statistics*, 13(3), 253-263.

Harvey, D., Leybourne, S. & Newbold, P. (1997). Testing the equality of prediction mean
squared errors. *International Journal of Forecasting*, 13(2), 281-291.

Clark, T. E. & West, K. D. (2007). Approximately normal tests for equal predictive
accuracy in nested models. *Journal of Econometrics*, 138(1), 291-311.

Giacomini, R. & White, H. (2006). Tests of conditional predictive ability.
*Econometrica*, 74(6), 1545-1578.

Hansen, P. R., Lunde, A. & Nason, J. M. (2011). The model confidence set.
*Econometrica*, 79(2), 453-497.

Pesaran, M. H. & Timmermann, A. (1992). A simple nonparametric test of predictive
performance. *Journal of Business & Economic Statistics*, 10(4), 461-465.

Blaskowitz, O. & Herwartz, H. (2011). On economic evaluation of directional forecasts.
*International Journal of Forecasting*, 27(4), 1058-1065.

Künsch, H. R. (1989). The jackknife and the bootstrap for general stationary
observations. *Annals of Statistics*, 17(3), 1217-1241.

Newey, W. K. & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity
and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703-708.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from nowcastbox.core.exceptions import DataQualityWarning, NowcastDataError
from nowcastbox.evaluation.metrics import (
    ArrayLike,
    LossLike,
    directional_changes,
    loss_values,
)

__all__ = [
    "ClarkWestResult",
    "DieboldMarianoResult",
    "GiacominiWhiteResult",
    "ModelConfidenceSetResult",
    "PesaranTimmermannResult",
    "clark_west",
    "clark_west_differential",
    "clark_west_from_differential",
    "diebold_mariano",
    "giacomini_white",
    "model_confidence_set",
    "pesaran_timmermann",
]

_ALTERNATIVES = ("two-sided", "less", "greater")
_MCS_STATISTICS = ("max", "range")


# ====================================================================== helpers
def _check_horizon(h: object) -> int:
    if isinstance(h, bool) or not isinstance(h, int | np.integer) or h < 1:
        raise ValueError(f"h must be a positive integer, got {h!r}.")
    return int(h)


def _loss_differential(errors1: ArrayLike, errors2: ArrayLike, loss: LossLike) -> np.ndarray:
    l1 = loss_values(errors1, loss)
    l2 = loss_values(errors2, loss)
    if l1.shape != l2.shape:
        raise ValueError("The two error series have different lengths.")
    return l1 - l2


def _autocovariance(x: np.ndarray, lag: int) -> float:
    """Sample autocovariance (divisor ``n``) of a centred series."""
    n = x.size
    return float(x[lag:] @ x[: n - lag]) / n


def _p_value(statistic: float, alternative: str, dist: Any) -> float:
    if alternative == "two-sided":
        return float(2.0 * dist.sf(abs(statistic)))
    if alternative == "less":
        return float(dist.cdf(statistic))
    return float(dist.sf(statistic))


# ====================================================================== Diebold-Mariano
@dataclass(frozen=True)
class DieboldMarianoResult:
    r"""Result of :func:`diebold_mariano`.

    Attributes
    ----------
    statistic : float
        Test statistic (HLN-corrected when ``hln=True``).
    pvalue : float
        p-value under the chosen alternative.
    mean_loss_differential : float
        :math:`\bar d` (negative: the first forecast has the lower loss).
    n_obs : int
        Number of loss differentials used.
    h : int
        Forecast horizon used in the long-run variance.
    alternative : str
        ``"two-sided"``, ``"less"`` (first forecast more accurate) or ``"greater"``.
    hln : bool
        Whether the Harvey-Leybourne-Newbold correction (and :math:`t_{n-1}`) was used.

    Examples
    --------
    >>> res = DieboldMarianoResult(-2.1, 0.04, -0.3, 40, 1, "two-sided", True)
    >>> res.reject(0.05)
    True
    """

    statistic: float
    pvalue: float
    mean_loss_differential: float
    n_obs: int
    h: int
    alternative: str
    hln: bool

    def reject(self, alpha: float = 0.05) -> bool:
        """Whether :math:`H_0` (equal accuracy) is rejected at level ``alpha``.

        Parameters
        ----------
        alpha : float, default 0.05
            Significance level.

        Returns
        -------
        bool
            ``pvalue < alpha`` (False when the p-value is NaN).

        Examples
        --------
        >>> DieboldMarianoResult(0.1, 0.9, 0.01, 30, 1, "two-sided", True).reject()
        False
        """
        return bool(self.pvalue < alpha)


def diebold_mariano(
    errors1: ArrayLike,
    errors2: ArrayLike,
    *,
    h: int = 1,
    loss: LossLike = "squared",
    alternative: str = "two-sided",
    hln: bool = True,
) -> DieboldMarianoResult:
    r"""Diebold-Mariano test of equal predictive accuracy (with HLN correction).

    Parameters
    ----------
    errors1, errors2 : array-like
        Aligned forecast errors of the two competing forecasts (pairs with a NaN are
        dropped).
    h : int, default 1
        Forecast horizon: autocovariances up to lag :math:`h - 1` enter the long-run
        variance of :math:`\bar d`.
    loss : {"squared", "absolute"} or callable, default "squared"
        Loss function.
    alternative : {"two-sided", "less", "greater"}, default "two-sided"
        ``"less"``: the first forecast is more accurate (:math:`E[d_t] < 0`);
        ``"greater"``: the second one is.
    hln : bool, default True
        Apply the Harvey, Leybourne & Newbold (1997) small-sample correction and use
        :math:`t_{n-1}` critical values (otherwise the standard normal).

    Returns
    -------
    DieboldMarianoResult
        Statistic and p-value.

    Raises
    ------
    ValueError
        If ``h`` or ``alternative`` is invalid or the inputs differ in length.
    NowcastDataError
        If fewer than 3 complete pairs are available.

    Warns
    -----
    DataQualityWarning
        If the truncated long-run variance is not positive (the Bartlett-weighted
        estimator is used instead), or the loss differential is constant (NaN
        statistic).

    Examples
    --------
    >>> import numpy as np
    >>> from nowcastbox.evaluation import diebold_mariano
    >>> rng = np.random.default_rng(0)
    >>> e_good, e_bad = rng.normal(0, 1, 200), rng.normal(0, 2, 200)
    >>> res = diebold_mariano(e_good, e_bad, alternative="less")
    >>> res.statistic < 0, res.pvalue < 0.01
    (True, True)
    """
    h = _check_horizon(h)
    if alternative not in _ALTERNATIVES:
        raise ValueError(f"alternative must be one of {_ALTERNATIVES}, got {alternative!r}.")
    d = _loss_differential(errors1, errors2, loss)
    d = d[np.isfinite(d)]
    n = d.size
    if n < 3:
        raise NowcastDataError(f"The Diebold-Mariano test needs at least 3 pairs, got {n}.")
    d_bar = float(d.mean())
    variance = _long_run_variance(d - d_bar, h)
    if variance <= 0:
        warnings.warn(
            "Constant loss differential: the Diebold-Mariano statistic is undefined.",
            DataQualityWarning,
            stacklevel=2,
        )
        return DieboldMarianoResult(np.nan, np.nan, d_bar, n, h, alternative, hln)
    statistic = d_bar / math.sqrt(variance / n)
    if hln:
        statistic *= math.sqrt(max(n + 1 - 2 * h + h * (h - 1) / n, 0.0) / n)
        dist = stats.t(df=n - 1)
    else:
        dist = stats.norm()
    pvalue = _p_value(statistic, alternative, dist)
    return DieboldMarianoResult(statistic, pvalue, d_bar, n, h, alternative, hln)


def _long_run_variance(centred: np.ndarray, h: int) -> float:
    """Truncated (rectangular) long-run variance; Bartlett fallback if not positive."""
    gammas = [_autocovariance(centred, k) for k in range(min(h, centred.size))]
    variance = gammas[0] + 2.0 * sum(gammas[1:])
    if variance <= 0 < gammas[0]:
        warnings.warn(
            "Non-positive truncated long-run variance; using Bartlett (Newey-West) weights.",
            DataQualityWarning,
            stacklevel=3,
        )
        variance = gammas[0] + 2.0 * sum(
            (1.0 - k / h) * g for k, g in enumerate(gammas[1:], start=1)
        )
    return float(variance)


# ====================================================================== Clark-West
@dataclass(frozen=True)
class ClarkWestResult:
    r"""Result of :func:`clark_west`.

    Attributes
    ----------
    statistic : float
        :math:`\sqrt{n}\,\bar d / \hat\sigma` (negative: the larger model is more
        accurate after the adjustment).
    pvalue : float
        p-value under the chosen alternative (standard normal).
    mean_loss_differential : float
        Mean adjusted loss differential :math:`\bar d`.
    n_obs : int
        Number of differentials used.
    h : int
        Forecast horizon used in the long-run variance.
    alternative : str
        ``"less"`` (default: the larger model is more accurate), ``"greater"`` or
        ``"two-sided"``.

    Examples
    --------
    >>> ClarkWestResult(-2.0, 0.02, -0.1, 40, 1, "less").reject(0.05)
    True
    """

    statistic: float
    pvalue: float
    mean_loss_differential: float
    n_obs: int
    h: int
    alternative: str

    def reject(self, alpha: float = 0.05) -> bool:
        """Whether :math:`H_0` (equal MSPE of the nested models) is rejected at ``alpha``.

        Parameters
        ----------
        alpha : float, default 0.05
            Significance level.

        Returns
        -------
        bool
            ``pvalue < alpha`` (False when the p-value is NaN).

        Examples
        --------
        >>> ClarkWestResult(0.3, 0.6, 0.01, 40, 1, "less").reject()
        False
        """
        return bool(self.pvalue < alpha)


def clark_west_differential(errors_large: ArrayLike, errors_small: ArrayLike) -> np.ndarray:
    r"""Clark-West adjusted squared-loss differential of two nested forecasts.

    With forecast errors :math:`e = y - \hat y`, the difference of the two forecasts
    is :math:`\hat y_2 - \hat y_1 = e_1 - e_2`, so the adjusted differential is
    :math:`d_t = e_{1t}^2 - (e_{1t} - e_{2t})^2 - e_{2t}^2`.

    Parameters
    ----------
    errors_large : array-like
        Errors of the larger model (the one that nests the other).
    errors_small : array-like
        Errors of the nested (parsimonious) benchmark.

    Returns
    -------
    numpy.ndarray
        Adjusted differentials (NaN where an error is missing).

    Raises
    ------
    ValueError
        If the inputs differ in length.

    Examples
    --------
    >>> from nowcastbox.evaluation import clark_west_differential
    >>> clark_west_differential([1.0], [2.0]).tolist()
    [-4.0]
    """
    e1 = np.asarray(errors_large, dtype=float).ravel()
    e2 = np.asarray(errors_small, dtype=float).ravel()
    if e1.shape != e2.shape:
        raise ValueError("The two error series have different lengths.")
    return e1**2 - (e1 - e2) ** 2 - e2**2


def clark_west(
    errors_large: ArrayLike,
    errors_small: ArrayLike,
    *,
    h: int = 1,
    alternative: str = "less",
) -> ClarkWestResult:
    r"""Clark-West test of equal predictive accuracy for nested models.

    Under :math:`H_0` the extra parameters of the larger model are zero, and its
    sample MSPE is biased upwards by the noise of estimating them; the adjusted
    differential of :func:`clark_west_differential` removes that bias. The statistic
    uses the truncated long-run variance of :func:`diebold_mariano` and standard
    normal critical values (Clark & West, 2007). The test is usually one-sided.

    Parameters
    ----------
    errors_large : array-like
        Errors of the larger model (pairs with a NaN are dropped).
    errors_small : array-like
        Errors of the nested benchmark.
    h : int, default 1
        Forecast horizon: autocovariances up to lag :math:`h - 1` enter the
        long-run variance.
    alternative : {"less", "greater", "two-sided"}, default "less"
        ``"less"``: the larger model is more accurate (:math:`E[d_t] < 0`).

    Returns
    -------
    ClarkWestResult
        Statistic and p-value.

    Raises
    ------
    ValueError
        If ``h`` or ``alternative`` is invalid or the inputs differ in length.
    NowcastDataError
        If fewer than 3 complete pairs are available.

    Warns
    -----
    DataQualityWarning
        If the differential is constant (NaN statistic).

    Examples
    --------
    >>> import numpy as np
    >>> from nowcastbox.evaluation import clark_west
    >>> rng = np.random.default_rng(0)
    >>> y = rng.normal(size=200)
    >>> signal = 0.8 * y + rng.normal(0, 0.3, 200)
    >>> res = clark_west(y - signal, y)
    >>> res.statistic < 0, res.pvalue < 0.01
    (True, True)
    """
    return clark_west_from_differential(
        clark_west_differential(errors_large, errors_small), h=h, alternative=alternative
    )


def clark_west_from_differential(
    differential: ArrayLike, *, h: int = 1, alternative: str = "less"
) -> ClarkWestResult:
    """Clark-West test on precomputed adjusted differentials.

    Useful when the differentials are transformed before testing, e.g. averaged
    within each target period (see :meth:`BacktestResults.clark_west`).

    Parameters
    ----------
    differential : array-like
        Adjusted differentials from :func:`clark_west_differential` (NaNs dropped).
    h : int, default 1
        Forecast horizon of the long-run variance.
    alternative : {"less", "greater", "two-sided"}, default "less"
        ``"less"``: the larger model is more accurate.

    Returns
    -------
    ClarkWestResult
        Statistic and p-value.

    Raises
    ------
    ValueError
        If ``h`` or ``alternative`` is invalid.
    NowcastDataError
        If fewer than 3 differentials are available.

    Examples
    --------
    >>> from nowcastbox.evaluation import clark_west_from_differential
    >>> res = clark_west_from_differential([-1.0, -0.5, -2.0, 0.1, -0.7])
    >>> res.statistic < 0
    True
    """
    h = _check_horizon(h)
    if alternative not in _ALTERNATIVES:
        raise ValueError(f"alternative must be one of {_ALTERNATIVES}, got {alternative!r}.")
    d = np.asarray(differential, dtype=float).ravel()
    d = d[np.isfinite(d)]
    n = d.size
    if n < 3:
        raise NowcastDataError(f"The Clark-West test needs at least 3 pairs, got {n}.")
    d_bar = float(d.mean())
    variance = _long_run_variance(d - d_bar, h)
    if variance <= 0:
        warnings.warn(
            "Constant adjusted differential: the Clark-West statistic is undefined.",
            DataQualityWarning,
            stacklevel=3,
        )
        return ClarkWestResult(np.nan, np.nan, d_bar, n, h, alternative)
    statistic = d_bar / math.sqrt(variance / n)
    pvalue = _p_value(statistic, alternative, stats.norm())
    return ClarkWestResult(statistic, pvalue, d_bar, n, h, alternative)


# ====================================================================== Giacomini-White
@dataclass(frozen=True)
class GiacominiWhiteResult:
    r"""Result of :func:`giacomini_white`.

    Attributes
    ----------
    statistic : float
        Wald statistic :math:`n \bar Z' \hat\Omega^{-1} \bar Z`.
    pvalue : float
        p-value from :math:`\chi^2_q`.
    df : int
        Number of instruments :math:`q`.
    mean_loss_differential : float
        Mean loss differential on the test sample (negative favours the first
        forecast).
    n_obs : int
        Number of observations used.
    h : int
        Forecast horizon.

    Examples
    --------
    >>> GiacominiWhiteResult(7.0, 0.03, 2, -0.2, 50, 1).reject(0.05)
    True
    """

    statistic: float
    pvalue: float
    df: int
    mean_loss_differential: float
    n_obs: int
    h: int

    def reject(self, alpha: float = 0.05) -> bool:
        """Whether equal conditional predictive ability is rejected at ``alpha``.

        Parameters
        ----------
        alpha : float, default 0.05
            Significance level.

        Returns
        -------
        bool
            ``pvalue < alpha``.

        Examples
        --------
        >>> GiacominiWhiteResult(1.0, 0.6, 2, 0.0, 50, 1).reject()
        False
        """
        return bool(self.pvalue < alpha)


def _default_instruments(d: np.ndarray, h: int) -> np.ndarray:
    lagged = np.full(d.size, np.nan)
    lagged[h:] = d[:-h]
    return np.column_stack([np.ones(d.size), lagged])


def _hac_second_moment(Z: np.ndarray, h: int) -> np.ndarray:
    """Uncentred second-moment matrix of ``Z`` with Newey-West weights up to ``h - 1``."""
    n = Z.shape[0]
    omega = Z.T @ Z / n
    for k in range(1, h):
        gamma = Z[k:].T @ Z[:-k] / n
        omega += (1.0 - k / h) * (gamma + gamma.T)
    return omega


def giacomini_white(
    errors1: ArrayLike,
    errors2: ArrayLike,
    *,
    h: int = 1,
    loss: LossLike = "squared",
    instruments: np.ndarray | pd.DataFrame | None = None,
) -> GiacominiWhiteResult:
    r"""Giacomini-White test of equal conditional predictive ability.

    Parameters
    ----------
    errors1, errors2 : array-like
        Aligned forecast errors, ordered in time.
    h : int, default 1
        Forecast horizon (instruments are lagged by ``h``; Newey-West with ``h - 1``
        lags for :math:`h > 1`).
    loss : {"squared", "absolute"} or callable, default "squared"
        Loss function.
    instruments : array-like of shape (n, q), optional
        Test functions :math:`h_{t-h}`, row :math:`t` aligned with the
        :math:`t`-th loss differential (already lagged, i.e. known when the
        forecasts of row :math:`t` were made). Default: a constant and
        :math:`d_{t-h}`. A single constant column gives the unconditional test.

    Returns
    -------
    GiacominiWhiteResult
        Statistic, p-value and degrees of freedom.

    Raises
    ------
    ValueError
        If the inputs are inconsistent.
    NowcastDataError
        If fewer complete observations than instruments + 2 are available or the
        moment matrix is singular.

    Examples
    --------
    >>> import numpy as np
    >>> from nowcastbox.evaluation import giacomini_white
    >>> rng = np.random.default_rng(1)
    >>> res = giacomini_white(rng.normal(0, 1, 300), rng.normal(0, 1.5, 300))
    >>> res.df, res.reject(0.05)
    (2, True)
    """
    h = _check_horizon(h)
    d = _loss_differential(errors1, errors2, loss)
    if instruments is None:
        H = _default_instruments(d, h)
    else:
        H = np.asarray(instruments, dtype=float)
        H = H.reshape(-1, 1) if H.ndim == 1 else H
        if H.shape[0] != d.size:
            raise ValueError("instruments must have one row per loss differential.")
    Z = H * d[:, None]
    Z = Z[np.isfinite(Z).all(axis=1)]
    n, q = Z.shape
    if n < q + 2:
        raise NowcastDataError(f"Too few complete observations ({n}) for {q} instruments.")
    z_bar = Z.mean(axis=0)
    omega = _hac_second_moment(Z, h)
    try:
        solved = np.linalg.solve(omega, z_bar)
    except np.linalg.LinAlgError as err:
        raise NowcastDataError("Singular moment matrix in the Giacomini-White test.") from err
    statistic = float(n * z_bar @ solved)
    pvalue = float(stats.chi2.sf(statistic, df=q))
    d_ok = d[np.isfinite(d)]
    return GiacominiWhiteResult(statistic, pvalue, q, float(d_ok.mean()), n, h)


# ====================================================================== MCS
@dataclass(frozen=True)
class ModelConfidenceSetResult:
    r"""Result of :func:`model_confidence_set`.

    Attributes
    ----------
    included : list of str
        Models in the :math:`(1-\alpha)` Model Confidence Set.
    eliminated : list of str
        Models in elimination order (worst first; excludes the last survivor).
    pvalues : pandas.Series
        MCS p-values (monotone along the elimination sequence; the last survivor has
        p-value 1), indexed by model in elimination order.
    mean_loss : pandas.Series
        Average loss of each model on the common sample.
    alpha : float
        Level of the set.
    statistic : str
        ``"max"`` (:math:`T_{\max}`) or ``"range"`` (:math:`T_R`).
    n_bootstrap : int
        Bootstrap replications.
    block_length : int
        Block length of the moving-block bootstrap.
    n_obs : int
        Number of time periods used.

    Examples
    --------
    >>> import pandas as pd
    >>> res = ModelConfidenceSetResult(
    ...     ["A"],
    ...     ["B"],
    ...     pd.Series({"B": 0.01, "A": 1.0}),
    ...     pd.Series({"A": 1.0, "B": 2.0}),
    ...     0.1,
    ...     "max",
    ...     100,
    ...     2,
    ...     50,
    ... )
    >>> res.to_frame()["included"].to_dict()
    {'A': True, 'B': False}
    """

    included: list[str]
    eliminated: list[str]
    pvalues: pd.Series
    mean_loss: pd.Series
    alpha: float
    statistic: str
    n_bootstrap: int
    block_length: int
    n_obs: int = field(default=0)

    def to_frame(self) -> pd.DataFrame:
        """Table with the mean loss, MCS p-value and membership of every model.

        Returns
        -------
        pandas.DataFrame
            Indexed by model, sorted by MCS p-value (best first); columns
            ``mean_loss``, ``pvalue``, ``included``.

        Examples
        --------
        >>> import pandas as pd
        >>> res = ModelConfidenceSetResult(
        ...     ["A"],
        ...     ["B"],
        ...     pd.Series({"B": 0.0, "A": 1.0}),
        ...     pd.Series({"A": 1.0, "B": 3.0}),
        ...     0.1,
        ...     "max",
        ...     10,
        ...     1,
        ...     20,
        ... )
        >>> res.to_frame().index.tolist()
        ['A', 'B']
        """
        frame = pd.DataFrame(
            {
                "mean_loss": self.mean_loss.reindex(self.pvalues.index),
                "pvalue": self.pvalues,
                "included": [m in self.included for m in self.pvalues.index],
            }
        )
        return frame.iloc[::-1]

    def summary(self) -> str:
        """Plain-text summary of the Model Confidence Set.

        Returns
        -------
        str
            Report.

        Examples
        --------
        >>> import pandas as pd
        >>> res = ModelConfidenceSetResult(
        ...     ["A"], [], pd.Series({"A": 1.0}), pd.Series({"A": 1.0}), 0.1, "max", 10, 1, 20
        ... )
        >>> "Model Confidence Set" in res.summary()
        True
        """
        lines = [
            f"Model Confidence Set (Hansen, Lunde & Nason, 2011) - {1 - self.alpha:.0%} level",
            f"statistic: T_{self.statistic}; bootstrap: {self.n_bootstrap} replications, "
            f"block length {self.block_length}; T = {self.n_obs}",
            "",
            self.to_frame().to_string(float_format=lambda v: f"{v:.4f}"),
        ]
        return "\n".join(lines)


def _block_indices(
    n: int, block_length: int, n_bootstrap: int, rng: np.random.Generator
) -> np.ndarray:
    """Circular moving-block bootstrap time indices, shape ``(n_bootstrap, n)``."""
    n_blocks = -(-n // block_length)
    starts = rng.integers(0, n, size=(n_bootstrap, n_blocks))
    offsets = np.arange(block_length)
    idx = (starts[:, :, None] + offsets[None, None, :]) % n
    return idx.reshape(n_bootstrap, -1)[:, :n]


def _safe_ratio(num: np.ndarray, var: np.ndarray) -> np.ndarray:
    scale = np.sqrt(np.where(var > 0, var, np.inf))
    return num / scale


def _step_max(mean: np.ndarray, zeta: np.ndarray) -> tuple[float, int]:
    r"""p-value of :math:`T_{\max}` and index of the model to eliminate."""
    d_dot = mean - mean.mean()
    z_dot = zeta - zeta.mean(axis=1, keepdims=True)
    var = np.mean(z_dot**2, axis=0)
    t = _safe_ratio(d_dot, var)
    t_boot = _safe_ratio(z_dot, var).max(axis=1)
    return float(np.mean(t_boot >= t.max())), int(np.argmax(t))


def _step_range(mean: np.ndarray, zeta: np.ndarray) -> tuple[float, int]:
    """p-value of :math:`T_R` and index of the model to eliminate."""
    d = mean[:, None] - mean[None, :]
    z = zeta[:, :, None] - zeta[:, None, :]
    var = np.mean(z**2, axis=0)
    t = _safe_ratio(d, var)
    t_boot = np.abs(_safe_ratio(z, var)).reshape(z.shape[0], -1).max(axis=1)
    return float(np.mean(t_boot >= np.abs(t).max())), int(np.argmax(t.max(axis=1)))


def _as_loss_frame(losses: pd.DataFrame | np.ndarray, names: Sequence[str] | None) -> pd.DataFrame:
    if isinstance(losses, pd.DataFrame):
        frame = losses.astype(float)
        if names is not None:
            frame.columns = list(names)
    else:
        values = np.asarray(losses, dtype=float)
        if values.ndim != 2:
            raise ValueError("losses must be a 2-D array (time x models).")
        cols = (
            list(names) if names is not None else [f"model{i + 1}" for i in range(values.shape[1])]
        )
        frame = pd.DataFrame(values, columns=cols)
    frame.columns = [str(c) for c in frame.columns]
    if frame.shape[1] < 2 or len(set(frame.columns)) != frame.shape[1]:
        raise ValueError("The MCS needs at least two models with distinct names.")
    return frame.dropna()


def model_confidence_set(
    losses: pd.DataFrame | np.ndarray,
    *,
    alpha: float = 0.1,
    statistic: str = "max",
    n_bootstrap: int = 1000,
    block_length: int | None = None,
    random_state: int | np.random.Generator | None = 0,
    names: Sequence[str] | None = None,
) -> ModelConfidenceSetResult:
    r"""Model Confidence Set of Hansen, Lunde & Nason (2011).

    Starting from all models, the equivalence hypothesis
    :math:`H_{0,\mathcal M}: E[d_{ij,t}] = 0\ \forall i, j \in \mathcal M` is tested
    with a bootstrap of :math:`T_{\max,\mathcal M} = \max_i \bar d_{i\cdot} /
    \widehat{\mathrm{sd}}(\bar d_{i\cdot})` (``statistic="max"``) or
    :math:`T_{R,\mathcal M} = \max_{i,j} |\bar d_{ij}| / \widehat{\mathrm{sd}}(\bar d_{ij})`
    (``statistic="range"``); while it is rejected the worst model (largest
    standardised relative loss) is eliminated. MCS p-values are the running maximum of
    the test p-values along the elimination sequence, and the set at level
    :math:`1 - \alpha` contains the models with p-value :math:`\ge \alpha`.

    Parameters
    ----------
    losses : pandas.DataFrame or numpy.ndarray of shape (T, M)
        Loss of each model (columns) in each period (rows); rows with a NaN are dropped
        (common sample).
    alpha : float, default 0.1
        Level: the set contains the best models with probability :math:`\ge 1-\alpha`
        asymptotically.
    statistic : {"max", "range"}, default "max"
        Test statistic / elimination rule.
    n_bootstrap : int, default 1000
        Bootstrap replications.
    block_length : int, optional
        Block length of the circular moving-block bootstrap (default
        :math:`\lceil T^{1/3} \rceil`).
    random_state : int, numpy.random.Generator or None, default 0
        Seed (fixed by default for reproducibility).
    names : sequence of str, optional
        Model names (columns of an array; default ``model1, model2, ...``).

    Returns
    -------
    ModelConfidenceSetResult
        Set, elimination order and MCS p-values.

    Raises
    ------
    ValueError
        If a parameter is invalid or fewer than two models are given.
    NowcastDataError
        If fewer than 3 complete periods are available.

    Examples
    --------
    >>> import numpy as np
    >>> from nowcastbox.evaluation import model_confidence_set
    >>> rng = np.random.default_rng(0)
    >>> e = rng.standard_normal((200, 3)) * np.array([1.0, 1.02, 2.0])
    >>> res = model_confidence_set(e**2, names=["A", "B", "C"], n_bootstrap=500)
    >>> sorted(res.included), res.eliminated[0]
    (['A', 'B'], 'C')
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}.")
    if statistic not in _MCS_STATISTICS:
        raise ValueError(f"statistic must be one of {_MCS_STATISTICS}, got {statistic!r}.")
    _check_horizon(n_bootstrap)
    frame = _as_loss_frame(losses, names)
    n = frame.shape[0]
    if n < 3:
        raise NowcastDataError(f"The MCS needs at least 3 complete periods, got {n}.")
    length = math.ceil(n ** (1 / 3)) if block_length is None else _check_horizon(block_length)
    rng = np.random.default_rng(random_state)
    values = frame.to_numpy()
    mean = values.mean(axis=0)
    boot = values[_block_indices(n, length, int(n_bootstrap), rng)].mean(axis=1)
    zeta = boot - mean
    step = _step_max if statistic == "max" else _step_range
    alive = list(range(values.shape[1]))
    order: list[int] = []
    pvals: list[float] = []
    running = 0.0
    while len(alive) > 1:
        p, worst = step(mean[alive], zeta[:, alive])
        running = max(running, p)
        order.append(alive.pop(worst))
        pvals.append(running)
    order.append(alive[0])
    pvals.append(1.0)
    models = list(frame.columns)
    pvalues = pd.Series(pvals, index=[models[i] for i in order], name="pvalue")
    return ModelConfidenceSetResult(
        included=[m for m in pvalues.index if pvalues[m] >= alpha],
        eliminated=list(pvalues.index[:-1]),
        pvalues=pvalues,
        mean_loss=pd.Series(mean, index=models, name="mean_loss"),
        alpha=float(alpha),
        statistic=statistic,
        n_bootstrap=int(n_bootstrap),
        block_length=int(length),
        n_obs=int(n),
    )


# ====================================================================== Pesaran-Timmermann
@dataclass(frozen=True)
class PesaranTimmermannResult:
    r"""Result of :func:`pesaran_timmermann`.

    Attributes
    ----------
    statistic : float
        :math:`S_n` (positive: more correct directions than expected by chance).
    pvalue : float
        p-value under the chosen alternative (standard normal).
    hit_rate : float
        :math:`\hat P`, share of correctly predicted directions.
    expected_hit_rate : float
        :math:`\hat P_*`, hit rate expected if forecasts and outcomes were independent.
    n_obs : int
        Number of forecasts used.
    alternative : str
        ``"greater"`` (default: directional predictability), ``"two-sided"`` or
        ``"less"``.

    Examples
    --------
    >>> PesaranTimmermannResult(2.5, 0.006, 0.8, 0.5, 40, "greater").reject(0.05)
    True
    """

    statistic: float
    pvalue: float
    hit_rate: float
    expected_hit_rate: float
    n_obs: int
    alternative: str

    def reject(self, alpha: float = 0.05) -> bool:
        """Whether :math:`H_0` (no directional predictability) is rejected at ``alpha``.

        Parameters
        ----------
        alpha : float, default 0.05
            Significance level.

        Returns
        -------
        bool
            ``pvalue < alpha`` (False when the p-value is NaN).

        Examples
        --------
        >>> PesaranTimmermannResult(0.2, 0.42, 0.55, 0.52, 40, "greater").reject()
        False
        """
        return bool(self.pvalue < alpha)


def _pt_statistic(up_actual: np.ndarray, up_forecast: np.ndarray) -> tuple[float, float, float]:
    """Hit rate, expected hit rate and variance of their difference (PT 1992, eq. 3-6)."""
    n = up_actual.size
    hit = float(np.mean(up_actual == up_forecast))
    p_y = float(np.mean(up_actual))
    p_x = float(np.mean(up_forecast))
    p_star = p_y * p_x + (1.0 - p_y) * (1.0 - p_x)
    var_p = p_star * (1.0 - p_star) / n
    var_star = (
        (2.0 * p_y - 1.0) ** 2 * p_x * (1.0 - p_x) / n
        + (2.0 * p_x - 1.0) ** 2 * p_y * (1.0 - p_y) / n
        + 4.0 * p_y * p_x * (1.0 - p_y) * (1.0 - p_x) / n**2
    )
    return hit, p_star, var_p - var_star


def pesaran_timmermann(
    actual: ArrayLike,
    forecast: ArrayLike,
    previous: ArrayLike,
    *,
    alternative: str = "greater",
) -> PesaranTimmermannResult:
    r"""Pesaran-Timmermann (1992) test of directional forecast accuracy.

    The direction of the actual change :math:`y_t - y^{p}_t` is compared with the
    direction of the predicted change :math:`\hat y_t - y^{p}_t` (a change is "up"
    when it is positive). Under :math:`H_0` the two directions are independent and
    the expected hit rate is :math:`P_* = P_y P_x + (1-P_y)(1-P_x)`; the statistic

    .. math::

        S_n = \frac{\hat P - \hat P_*}{\sqrt{\hat V(\hat P) - \hat V(\hat P_*)}},
        \quad \hat V(\hat P) = \frac{\hat P_*(1-\hat P_*)}{n},

        \hat V(\hat P_*) = \frac{(2\hat P_y-1)^2 \hat P_x(1-\hat P_x)}{n}
        + \frac{(2\hat P_x-1)^2 \hat P_y(1-\hat P_y)}{n}
        + \frac{4 \hat P_y \hat P_x (1-\hat P_y)(1-\hat P_x)}{n^2},

    is asymptotically standard normal. With continuous data the hit rate equals the
    forecast directional accuracy :func:`~nowcastbox.evaluation.directional_accuracy`.

    Parameters
    ----------
    actual, forecast, previous : array-like
        Realisations, forecasts and previous values (triples with a NaN are dropped).
    alternative : {"greater", "two-sided", "less"}, default "greater"
        ``"greater"``: the forecasts predict the direction better than chance.

    Returns
    -------
    PesaranTimmermannResult
        Statistic, p-value, hit rate and expected hit rate.

    Raises
    ------
    ValueError
        If ``alternative`` is invalid or the inputs differ in length.
    NowcastDataError
        If fewer than 3 complete triples are available.

    Warns
    -----
    DataQualityWarning
        If the variance of the statistic is not positive (e.g. every forecast or
        every outcome has the same direction); the statistic is then NaN.

    Examples
    --------
    >>> import numpy as np
    >>> from nowcastbox.evaluation import pesaran_timmermann
    >>> rng = np.random.default_rng(0)
    >>> change = rng.standard_normal(200)
    >>> forecast = change + 0.5 * rng.standard_normal(200)
    >>> res = pesaran_timmermann(change, forecast, np.zeros(200))
    >>> res.hit_rate > 0.75, res.pvalue < 0.01
    (True, True)
    """
    if alternative not in _ALTERNATIVES:
        raise ValueError(f"alternative must be one of {_ALTERNATIVES}, got {alternative!r}.")
    dy, dyhat = directional_changes(actual, forecast, previous)
    n = dy.size
    if n < 3:
        raise NowcastDataError(f"The Pesaran-Timmermann test needs at least 3 forecasts, got {n}.")
    hit, p_star, variance = _pt_statistic(dy > 0, dyhat > 0)
    if variance <= 1e-15:
        warnings.warn(
            "Degenerate directions (no variation in the actual or predicted signs): the "
            "Pesaran-Timmermann statistic is undefined.",
            DataQualityWarning,
            stacklevel=2,
        )
        return PesaranTimmermannResult(np.nan, np.nan, hit, p_star, n, alternative)
    statistic = (hit - p_star) / math.sqrt(variance)
    pvalue = _p_value(statistic, alternative, stats.norm())
    return PesaranTimmermannResult(statistic, pvalue, hit, p_star, n, alternative)
