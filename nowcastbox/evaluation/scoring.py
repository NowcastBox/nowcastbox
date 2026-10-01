r"""Scoring rules and calibration tests for density nowcasts (innovation I5).

Proper scoring rules (Gneiting & Raftery, 2007), all **negatively oriented** (smaller is
better) except the log score:

* :func:`crps` - continuous ranked probability score. Closed forms for the Gaussian
  (Gneiting, Raftery, Westveld & Goldman, 2005) and Gaussian mixtures (Grimit,
  Gneiting, Berrocal & Johnson, 2006); for samples the energy form
  :math:`\mathrm{CRPS}(F, y) = E|X - y| - \tfrac12 E|X - X'|` (Gneiting & Raftery,
  2007, eq. 21), optionally with the unbiased "fair" estimator (Ferro, 2014).
* :func:`log_score` - log predictive density :math:`\log p(y)` (**positively**
  oriented, larger is better); kernel density estimate for samples.
* :func:`quantile_score`, :func:`weighted_quantile_score` - pinball loss and the
  quantile-weighted CRPS of Gneiting & Ranjan (2011); :func:`interval_score`
  (Gneiting & Raftery, 2007, sec. 6.2).

Calibration:

* :func:`pit` - probability integral transform :math:`u_t = F_t(y_t)` (Dawid, 1984;
  Diebold, Gunther & Tay, 1998), uniform and (for one-step-ahead forecasts) i.i.d.
  under correct calibration;
* :func:`berkowitz_test` - likelihood-ratio test of Berkowitz (2001) on
  :math:`z_t = \Phi^{-1}(u_t)` against a Gaussian AR(1);
* :func:`ks_uniformity_test` - Kolmogorov-Smirnov test of uniformity;
* :func:`interval_coverage` / :func:`christoffersen_test` - empirical coverage and the
  unconditional, independence and conditional coverage LR tests of Christoffersen
  (1998).

References
----------
Berkowitz, J. (2001). Testing density forecasts, with applications to risk management.
*Journal of Business & Economic Statistics*, 19(4), 465-474.

Christoffersen, P. F. (1998). Evaluating interval forecasts. *International Economic
Review*, 39(4), 841-862.

Dawid, A. P. (1984). Statistical theory: the prequential approach. *JRSS-A*, 147(2),
278-292.

Diebold, F. X., Gunther, T. A. & Tay, A. S. (1998). Evaluating density forecasts with
applications to financial risk management. *International Economic Review*, 39(4),
863-883.

Ferro, C. A. T. (2014). Fair scores for ensemble forecasts. *QJRMS*, 140(683),
1917-1923.

Gneiting, T., Raftery, A. E., Westveld, A. H. & Goldman, T. (2005). Calibrated
probabilistic forecasting using ensemble model output statistics and minimum CRPS
estimation. *Monthly Weather Review*, 133(5), 1098-1118.

Gneiting, T. & Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and
estimation. *JASA*, 102(477), 359-378.

Gneiting, T. & Ranjan, R. (2011). Comparing density forecasts using threshold- and
quantile-weighted scoring rules. *JBES*, 29(3), 411-422.

Grimit, E. P., Gneiting, T., Berrocal, V. J. & Johnson, N. A. (2006). The continuous
ranked probability score for circular variables and its application to mesoscale
forecast ensemble verification. *QJRMS*, 132(621C), 2925-2942.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import scipy.optimize
import scipy.special
import scipy.stats
from numpy.typing import ArrayLike, NDArray

from nowcastbox.density.distribution import NowcastDistribution

__all__ = [
    "BerkowitzTestResult",
    "CoverageTestResult",
    "UniformityTestResult",
    "berkowitz_test",
    "christoffersen_test",
    "crps",
    "crps_gaussian",
    "crps_mixture",
    "crps_sample",
    "interval_coverage",
    "interval_hits",
    "interval_score",
    "ks_uniformity_test",
    "log_score",
    "log_score_gaussian",
    "log_score_sample",
    "pit",
    "quantile_score",
    "weighted_quantile_score",
]

FloatArray = NDArray[np.float64]
QuantileWeight = Literal["uniform", "center", "left", "right"]

_INV_SQRT_PI = 1.0 / np.sqrt(np.pi)
_LOG_SQRT_2PI = 0.5 * float(np.log(2.0 * np.pi))


# ====================================================================== helpers
def _finite(values: ArrayLike, name: str) -> FloatArray:
    arr = np.asarray(values, dtype=np.float64)
    if not bool(np.isfinite(arr).all()):
        raise ValueError(f"{name} must be finite.")
    return arr


def _positive(values: ArrayLike, name: str) -> FloatArray:
    arr = _finite(values, name)
    if not bool(np.all(arr > 0.0)):
        raise ValueError(f"{name} must be strictly positive.")
    return arr


def _level(level: float) -> float:
    lvl = float(level)
    if not 0.0 < lvl < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level!r}.")
    return lvl


def _samples(samples: ArrayLike, observed: FloatArray) -> FloatArray:
    """Sample array with draws on the last axis, broadcast against ``observed``."""
    arr = _finite(samples, "samples")
    if arr.ndim == 0 or arr.shape[-1] == 0:
        raise ValueError("samples must have at least one draw on the last axis.")
    np.broadcast_shapes(arr.shape[:-1], observed.shape)
    return arr


def _a_function(mu: FloatArray, var: FloatArray) -> FloatArray:
    r""":math:`A(\mu, \sigma^2) = E|X|`, :math:`X \sim N(\mu, \sigma^2)` (Grimit et al., 2006)."""
    sd = np.sqrt(var)
    z = mu / sd
    return 2.0 * sd * np.exp(-0.5 * z**2 - _LOG_SQRT_2PI) + mu * (2.0 * scipy.special.ndtr(z) - 1.0)


# ====================================================================== CRPS
def crps_gaussian(observed: ArrayLike, mean: ArrayLike, std: ArrayLike) -> FloatArray:
    r"""CRPS of a Gaussian forecast (closed form).

    .. math::

        \mathrm{CRPS}(N(\mu, \sigma^2), y) =
        \sigma\left[z\,(2\Phi(z) - 1) + 2\varphi(z) - \tfrac{1}{\sqrt\pi}\right],
        \qquad z = \frac{y - \mu}{\sigma}

    (Gneiting et al., 2005, eq. 5).

    Parameters
    ----------
    observed : array_like
        Realisations :math:`y`.
    mean, std : array_like
        Forecast means and standard deviations (``std > 0``); broadcast together.

    Returns
    -------
    numpy.ndarray
        CRPS (same units as ``observed``; smaller is better).

    Raises
    ------
    ValueError
        On non-finite input or non-positive ``std``.

    Examples
    --------
    >>> from nowcastbox.evaluation.scoring import crps_gaussian
    >>> round(float(crps_gaussian(0.0, 0.0, 1.0)), 6)
    0.233695
    """
    y = _finite(observed, "observed")
    mu = _finite(mean, "mean")
    sd = _positive(std, "std")
    z = (y - mu) / sd
    pdf = np.exp(-0.5 * z**2 - _LOG_SQRT_2PI)
    return sd * (z * (2.0 * scipy.special.ndtr(z) - 1.0) + 2.0 * pdf - _INV_SQRT_PI)


def crps_mixture(
    observed: ArrayLike,
    locs: ArrayLike,
    scales: ArrayLike,
    weights: ArrayLike | None = None,
) -> FloatArray:
    r"""CRPS of a Gaussian mixture (closed form of Grimit et al., 2006).

    .. math::

        \mathrm{CRPS} = \sum_i w_i A(y - \mu_i, \sigma_i^2)
        - \tfrac12 \sum_{i,j} w_i w_j A(\mu_i - \mu_j, \sigma_i^2 + \sigma_j^2),

    with :math:`A(\mu, \sigma^2) = 2\sigma\varphi(\mu/\sigma) + \mu(2\Phi(\mu/\sigma) - 1)`.

    Parameters
    ----------
    observed : array_like, shape (...)
        Realisations.
    locs, scales : array_like, shape (..., K)
        Component means and standard deviations (components on the last axis).
    weights : array_like, shape (K,), optional
        Component weights (normalised; default equal).

    Returns
    -------
    numpy.ndarray
        CRPS with the broadcast leading shape.

    Raises
    ------
    ValueError
        On inconsistent shapes or invalid values.

    Examples
    --------
    >>> import numpy as np
    >>> from nowcastbox.evaluation.scoring import crps_gaussian, crps_mixture
    >>> bool(np.isclose(crps_mixture(0.3, [0.0], [1.0]), crps_gaussian(0.3, 0.0, 1.0)))
    True
    """
    y = _finite(observed, "observed")
    mu = np.atleast_1d(_finite(locs, "locs"))
    sd = np.atleast_1d(_positive(scales, "scales"))
    if mu.shape != sd.shape:
        raise ValueError("locs and scales must have the same shape.")
    k = mu.shape[-1]
    if weights is None:
        w = np.full(k, 1.0 / k)
    else:
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        if w.shape != (k,) or bool((w < 0).any()) or not float(w.sum()) > 0:
            raise ValueError(f"weights must be {k} non-negative numbers with a positive sum.")
        w = w / w.sum()
    var = sd**2
    first = _a_function(y[..., None] - mu, var) @ w
    diff = mu[..., :, None] - mu[..., None, :]
    pair = _a_function(diff, var[..., :, None] + var[..., None, :])
    second = np.einsum("...ij,i,j->...", pair, w, w)
    return np.asarray(first - 0.5 * second, dtype=np.float64)


def crps_sample(observed: ArrayLike, samples: ArrayLike, *, fair: bool = False) -> FloatArray:
    r"""Sample (ensemble) CRPS in the energy form.

    .. math::

        \widehat{\mathrm{CRPS}} = \frac1m\sum_i |x_i - y|
        - \frac{1}{2 m^2}\sum_{i,j}|x_i - x_j|,

    computed in :math:`O(m \log m)` from the order statistics. With ``fair=True`` the
    second term uses :math:`m(m-1)` (Ferro, 2014), an unbiased estimator of the CRPS
    of the distribution the draws come from.

    Parameters
    ----------
    observed : array_like, shape (...)
        Realisations.
    samples : array_like, shape (..., m)
        Draws from the forecast distribution (last axis).
    fair : bool, default False
        Use the fair (unbiased) estimator (requires ``m >= 2``).

    Returns
    -------
    numpy.ndarray
        CRPS with the broadcast leading shape.

    Raises
    ------
    ValueError
        On empty or non-finite samples, or ``fair=True`` with one draw.

    Examples
    --------
    >>> from nowcastbox.evaluation.scoring import crps_sample
    >>> float(crps_sample(0.0, [-1.0, 1.0]))
    0.5
    >>> float(crps_sample(0.0, [-1.0, 1.0], fair=True))
    0.0
    """
    y = _finite(observed, "observed")
    x = np.sort(_samples(samples, y), axis=-1)
    m = x.shape[-1]
    if fair and m < 2:
        raise ValueError("fair=True needs at least two draws.")
    first = np.abs(x - y[..., None]).mean(axis=-1)
    coef = 2.0 * np.arange(1, m + 1) - m - 1.0
    pair_sum = 2.0 * (x @ coef)  # sum_{i,j} |x_i - x_j|
    denom = m * (m - 1) if fair else m * m
    return np.asarray(first - 0.5 * pair_sum / denom, dtype=np.float64)


def crps(forecast: NowcastDistribution | ArrayLike, observed: ArrayLike) -> FloatArray:
    """CRPS of a density nowcast.

    Parameters
    ----------
    forecast : NowcastDistribution or array_like
        A :class:`~nowcastbox.density.NowcastDistribution` (exact Gaussian/mixture
        formula; ``observed`` is broadcast against its periods) or draws with the
        sample axis last (energy form, :func:`crps_sample`).
    observed : array_like
        Realisations.

    Returns
    -------
    numpy.ndarray
        CRPS (smaller is better).

    Examples
    --------
    >>> from nowcastbox.density import NowcastDistribution
    >>> from nowcastbox.evaluation.scoring import crps
    >>> d = NowcastDistribution(["2020Q1", "2020Q2"], [0.0, 1.0], [1.0, 1.0])
    >>> crps(d, [0.0, 1.0]).round(6).tolist()
    [0.233695, 0.233695]
    """
    if isinstance(forecast, NowcastDistribution):
        y = np.broadcast_to(_finite(observed, "observed"), (forecast.n_periods,))
        if forecast.is_gaussian:
            return crps_gaussian(y, forecast.locs[:, 0], forecast.scales[:, 0])
        return crps_mixture(y, forecast.locs, forecast.scales, forecast.weights)
    return crps_sample(observed, forecast)


# ====================================================================== log score
def log_score_gaussian(observed: ArrayLike, mean: ArrayLike, std: ArrayLike) -> FloatArray:
    r"""Log predictive density :math:`\log \varphi((y-\mu)/\sigma) - \log\sigma`.

    Parameters
    ----------
    observed : array_like
        Realisations.
    mean, std : array_like
        Forecast means and standard deviations (``std > 0``).

    Returns
    -------
    numpy.ndarray
        Log score (larger is better).

    Raises
    ------
    ValueError
        On non-finite input or non-positive ``std``.

    Examples
    --------
    >>> from nowcastbox.evaluation.scoring import log_score_gaussian
    >>> round(float(log_score_gaussian(0.0, 0.0, 1.0)), 4)
    -0.9189
    """
    y = _finite(observed, "observed")
    mu = _finite(mean, "mean")
    sd = _positive(std, "std")
    z = (y - mu) / sd
    return -0.5 * z**2 - np.log(sd) - _LOG_SQRT_2PI


def log_score_sample(
    observed: ArrayLike, samples: ArrayLike, *, bandwidth: str | float | None = None
) -> FloatArray:
    """Log score of a sample forecast via a Gaussian kernel density estimate.

    Parameters
    ----------
    observed : array_like, shape (...)
        Realisations.
    samples : array_like, shape (..., m)
        Draws (last axis); at least two distinct values per forecast.
    bandwidth : str, float or None
        ``bw_method`` of :class:`scipy.stats.gaussian_kde` (default Scott's rule).

    Returns
    -------
    numpy.ndarray
        Estimated log predictive density.

    Raises
    ------
    ValueError
        On invalid or degenerate samples.

    Examples
    --------
    >>> import numpy as np
    >>> from nowcastbox.evaluation.scoring import log_score_sample
    >>> draws = np.random.default_rng(0).standard_normal(5000)
    >>> bool(abs(float(log_score_sample(0.0, draws)) + 0.9189) < 0.1)
    True
    """
    y = _finite(observed, "observed")
    x = _samples(samples, y)
    shape = np.broadcast_shapes(x.shape[:-1], y.shape)
    xb = np.broadcast_to(x, (*shape, x.shape[-1])).reshape(-1, x.shape[-1])
    yb = np.broadcast_to(y, shape).reshape(-1)
    out = np.empty(yb.size)
    for i in range(yb.size):
        if np.ptp(xb[i]) == 0.0:
            raise ValueError("Each sample needs at least two distinct values for a KDE.")
        kde = scipy.stats.gaussian_kde(xb[i], bw_method=bandwidth)
        out[i] = float(kde.logpdf(yb[i])[0])
    return out.reshape(shape)


def log_score(forecast: NowcastDistribution | ArrayLike, observed: ArrayLike) -> FloatArray:
    r"""Log score (log predictive density) of a density nowcast.

    Parameters
    ----------
    forecast : NowcastDistribution or array_like
        Distribution (exact mixture density) or draws with the sample axis last
        (kernel density estimate, :func:`log_score_sample`).
    observed : array_like
        Realisations.

    Returns
    -------
    numpy.ndarray
        :math:`\log p(y)` (positively oriented: larger is better).

    Examples
    --------
    >>> from nowcastbox.density import NowcastDistribution
    >>> from nowcastbox.evaluation.scoring import log_score
    >>> d = NowcastDistribution(["2020Q1"], [0.0], [1.0])
    >>> log_score(d, 0.0).round(4).tolist()
    [-0.9189]
    """
    if isinstance(forecast, NowcastDistribution):
        return forecast.logpdf(_finite(observed, "observed"))
    return log_score_sample(observed, forecast)


# ====================================================================== PIT
def pit(
    forecast: NowcastDistribution | ArrayLike,
    observed: ArrayLike,
    *,
    randomize: bool = False,
    random_state: int | np.random.Generator | None = None,
) -> FloatArray:
    r"""Probability integral transform :math:`u = F(y)`.

    Parameters
    ----------
    forecast : NowcastDistribution or array_like
        Distribution (exact cdf) or draws with the sample axis last (empirical cdf).
    observed : array_like
        Realisations.
    randomize : bool, default False
        For samples: randomised PIT :math:`(\#\{x < y\} + V\,(\#\{x = y\} + 1))/(m + 1)`,
        :math:`V \sim U(0, 1)`, exactly uniform for exchangeable draws (no ties
        between the draws and the outcome needed). Otherwise the empirical cdf
        :math:`\#\{x \le y\}/m`.
    random_state : int, numpy.random.Generator or None
        Seed of the randomisation.

    Returns
    -------
    numpy.ndarray
        PIT values in ``[0, 1]``.

    Examples
    --------
    >>> from nowcastbox.density import NowcastDistribution
    >>> from nowcastbox.evaluation.scoring import pit
    >>> d = NowcastDistribution(["2020Q1"], [0.0], [1.0])
    >>> pit(d, 0.0).tolist()
    [0.5]
    >>> pit([1.0, 2.0, 3.0, 4.0], 2.5).tolist()
    0.5
    """
    y = _finite(observed, "observed")
    if isinstance(forecast, NowcastDistribution):
        return forecast.cdf(y)
    x = _samples(forecast, y)
    below = (x < y[..., None]).sum(axis=-1)
    if not randomize:
        return np.asarray((x <= y[..., None]).mean(axis=-1), dtype=np.float64)
    ties = (x == y[..., None]).sum(axis=-1)
    v = np.random.default_rng(random_state).uniform(size=below.shape)
    return np.asarray((below + v * (ties + 1)) / (x.shape[-1] + 1), dtype=np.float64)


def _pit_values(values: ArrayLike, minimum: int) -> FloatArray:
    u = np.asarray(values, dtype=np.float64).reshape(-1)
    if u.size < minimum:
        raise ValueError(f"At least {minimum} PIT values are required, got {u.size}.")
    if not bool(np.isfinite(u).all()) or bool(((u < 0) | (u > 1)).any()):
        raise ValueError("PIT values must be finite and lie in [0, 1].")
    return u


@dataclass(frozen=True)
class UniformityTestResult:
    """Result of a uniformity test of PIT values.

    Parameters
    ----------
    statistic : float
        Test statistic.
    pvalue : float
        p-value.
    test : str
        Test name.
    nobs : int
        Number of PIT values.

    Examples
    --------
    >>> from nowcastbox.evaluation.scoring import ks_uniformity_test
    >>> ks_uniformity_test([0.1, 0.4, 0.6, 0.9]).test
    'kolmogorov-smirnov'
    """

    statistic: float
    pvalue: float
    test: str
    nobs: int

    def reject(self, alpha: float = 0.05) -> bool:
        """Whether uniformity is rejected at level ``alpha``."""
        return self.pvalue < alpha


def ks_uniformity_test(pit_values: ArrayLike) -> UniformityTestResult:
    """Kolmogorov-Smirnov test that PIT values are :math:`U(0, 1)`.

    The asymptotic distribution assumes independent PIT values (one-step-ahead
    nowcasts); with serially correlated PITs the test is oversized.

    Parameters
    ----------
    pit_values : array_like
        PIT values in ``[0, 1]`` (at least 2).

    Returns
    -------
    UniformityTestResult
        KS statistic and p-value.

    Raises
    ------
    ValueError
        On invalid PIT values.

    Examples
    --------
    >>> import numpy as np
    >>> from nowcastbox.evaluation.scoring import ks_uniformity_test
    >>> u = np.random.default_rng(0).uniform(size=500)
    >>> ks_uniformity_test(u).reject(0.01)
    False
    """
    u = _pit_values(pit_values, 2)
    res = scipy.stats.kstest(u, "uniform")
    return UniformityTestResult(
        statistic=float(res.statistic),  # type: ignore[attr-defined]
        pvalue=float(res.pvalue),  # type: ignore[attr-defined]
        test="kolmogorov-smirnov",
        nobs=int(u.size),
    )


@dataclass(frozen=True)
class BerkowitzTestResult:
    r"""Result of the Berkowitz (2001) likelihood-ratio test.

    Parameters
    ----------
    statistic : float
        :math:`LR_3 = -2(L(0, 0, 1) - L(\hat\mu, \hat\rho, \hat\sigma^2))`,
        :math:`\chi^2(3)` under :math:`H_0`.
    pvalue : float
        p-value of ``statistic``.
    independence_statistic : float
        :math:`LR_{ind} = -2(L(\hat\mu_0, 0, \hat\sigma^2_0) - L(\hat\mu, \hat\rho,
        \hat\sigma^2))`, :math:`\chi^2(1)`.
    independence_pvalue : float
        p-value of ``independence_statistic``.
    mean, rho, variance : float
        Maximum-likelihood estimates of the AR(1) model for
        :math:`z_t = \Phi^{-1}(u_t)`.
    nobs : int
        Number of PIT values.

    Examples
    --------
    >>> import numpy as np
    >>> from nowcastbox.evaluation.scoring import berkowitz_test
    >>> res = berkowitz_test(np.random.default_rng(1).uniform(size=300))
    >>> res.reject(0.01)
    False
    """

    statistic: float
    pvalue: float
    independence_statistic: float
    independence_pvalue: float
    mean: float
    rho: float
    variance: float
    nobs: int

    def reject(self, alpha: float = 0.05) -> bool:
        """Whether the joint null (mean 0, variance 1, no autocorrelation) is rejected."""
        return self.pvalue < alpha


def _ar1_loglik(z: FloatArray, mu: float, rho: float, var: float) -> float:
    """Exact Gaussian AR(1) log-likelihood (stationary first observation)."""
    v1 = var / (1.0 - rho**2)
    e = z[1:] - mu - rho * (z[:-1] - mu)
    n = z.size
    return float(
        -0.5 * n * np.log(2.0 * np.pi)
        - 0.5 * np.log(v1)
        - 0.5 * (z[0] - mu) ** 2 / v1
        - 0.5 * (n - 1) * np.log(var)
        - 0.5 * float(e @ e) / var
    )


def _fit_ar1(z: FloatArray) -> tuple[float, float, float]:
    """Exact ML of a stationary Gaussian AR(1) (starting from conditional OLS)."""
    x, y = z[:-1], z[1:]
    xc = x - x.mean()
    denom = float(xc @ xc)
    rho0 = float(np.clip((xc @ (y - y.mean())) / denom if denom > 0 else 0.0, -0.95, 0.95))
    var0 = max(float(np.var(z)) * (1.0 - rho0**2), 1e-8)

    def negloglik(theta: FloatArray) -> float:
        return -_ar1_loglik(z, float(theta[0]), float(np.tanh(theta[1])), float(np.exp(theta[2])))

    start = np.array([float(z.mean()), float(np.arctanh(rho0)), float(np.log(var0))])
    options = {"xatol": 1e-8, "fatol": 1e-10, "maxiter": 4000}
    opt = scipy.optimize.minimize(negloglik, start, method="Nelder-Mead", options=options)
    best = opt.x if opt.fun <= negloglik(start) else start
    return float(best[0]), float(np.tanh(best[1])), float(np.exp(best[2]))


def berkowitz_test(pit_values: ArrayLike, *, clip: float = 1e-10) -> BerkowitzTestResult:
    r"""Berkowitz (2001) likelihood-ratio test of density-forecast calibration.

    The PIT values are mapped to :math:`z_t = \Phi^{-1}(u_t)`, which are i.i.d.
    :math:`N(0, 1)` under correct calibration. A Gaussian AR(1)
    :math:`z_t - \mu = \rho(z_{t-1} - \mu) + \varepsilon_t` is fitted by exact
    maximum likelihood and :math:`H_0: \mu = 0, \rho = 0, \sigma^2 = 1` is tested with
    :math:`LR_3 \sim \chi^2(3)`; :math:`LR_{ind}` tests :math:`\rho = 0` alone.

    Parameters
    ----------
    pit_values : array_like
        PIT values in time order (at least 3).
    clip : float, default 1e-10
        PIT values are clipped to ``[clip, 1 - clip]`` before the inverse normal.

    Returns
    -------
    BerkowitzTestResult
        Statistics, p-values and AR(1) estimates.

    Raises
    ------
    ValueError
        On invalid or constant PIT values, or fewer than 3 observations.

    Examples
    --------
    >>> import numpy as np
    >>> from nowcastbox.evaluation.scoring import berkowitz_test
    >>> u = np.random.default_rng(0).uniform(size=400) ** 2  # miscalibrated
    >>> berkowitz_test(u).reject(0.01)
    True
    """
    u = _pit_values(pit_values, 3)
    if not 0.0 < clip < 0.5:
        raise ValueError(f"clip must be in (0, 0.5), got {clip}.")
    z = scipy.special.ndtri(np.clip(u, clip, 1.0 - clip))
    if float(np.ptp(z)) == 0.0:
        raise ValueError("PIT values are constant; the Berkowitz test is undefined.")
    mu, rho, var = _fit_ar1(z)
    l_full = _ar1_loglik(z, mu, rho, var)
    l_null = float(scipy.stats.norm.logpdf(z).sum())
    mu0, var0 = float(z.mean()), max(float(np.var(z)), 1e-12)
    l_iid = float(scipy.stats.norm.logpdf(z, mu0, np.sqrt(var0)).sum())
    lr = _nonnegative(-2.0 * (l_null - l_full))
    lr_ind = _nonnegative(-2.0 * (l_iid - l_full))
    return BerkowitzTestResult(
        statistic=lr,
        pvalue=float(scipy.stats.chi2.sf(lr, 3)),
        independence_statistic=lr_ind,
        independence_pvalue=float(scipy.stats.chi2.sf(lr_ind, 1)),
        mean=mu,
        rho=rho,
        variance=var,
        nobs=int(u.size),
    )


# ====================================================================== coverage
def interval_hits(observed: ArrayLike, lower: ArrayLike, upper: ArrayLike) -> NDArray[np.bool_]:
    """Indicator that each realisation falls inside its interval (bounds inclusive).

    Parameters
    ----------
    observed, lower, upper : array_like
        Realisations and interval bounds (broadcast together, ``lower <= upper``).

    Returns
    -------
    numpy.ndarray of bool
        Hit sequence.

    Raises
    ------
    ValueError
        On non-finite input or ``lower > upper``.

    Examples
    --------
    >>> from nowcastbox.evaluation.scoring import interval_hits
    >>> interval_hits([0.0, 2.0], -1.0, 1.0).tolist()
    [True, False]
    """
    y = _finite(observed, "observed")
    lo = _finite(lower, "lower")
    hi = _finite(upper, "upper")
    if bool(np.any(lo > hi)):
        raise ValueError("lower must not exceed upper.")
    return (y >= lo) & (y <= hi)


def interval_coverage(observed: ArrayLike, lower: ArrayLike, upper: ArrayLike) -> float:
    """Empirical coverage rate of interval forecasts.

    Parameters
    ----------
    observed, lower, upper : array_like
        Realisations and interval bounds.

    Returns
    -------
    float
        Share of realisations inside their interval.

    Raises
    ------
    ValueError
        On invalid input or no observations.

    Examples
    --------
    >>> from nowcastbox.evaluation.scoring import interval_coverage
    >>> interval_coverage([0.0, 2.0, 0.5, -3.0], -1.0, 1.0)
    0.5
    """
    hits = interval_hits(observed, lower, upper)
    if hits.size == 0:
        raise ValueError("No observations.")
    return float(hits.mean())


def interval_score(
    observed: ArrayLike, lower: ArrayLike, upper: ArrayLike, level: float
) -> FloatArray:
    r"""Interval score of a central :math:`(1-\alpha)` interval (Gneiting & Raftery, 2007).

    .. math::

        S_\alpha(l, u; y) = (u - l) + \tfrac{2}{\alpha}(l - y)\mathbb 1\{y < l\}
        + \tfrac{2}{\alpha}(y - u)\mathbb 1\{y > u\}.

    Parameters
    ----------
    observed, lower, upper : array_like
        Realisations and bounds.
    level : float
        Nominal coverage :math:`1 - \alpha` in ``(0, 1)``.

    Returns
    -------
    numpy.ndarray
        Interval scores (smaller is better).

    Raises
    ------
    ValueError
        On invalid input.

    Examples
    --------
    >>> from nowcastbox.evaluation.scoring import interval_score
    >>> interval_score([0.0, 2.0], -1.0, 1.0, 0.9).round(6).tolist()
    [2.0, 22.0]
    """
    alpha = 1.0 - _level(level)
    interval_hits(observed, lower, upper)  # validation
    y = np.asarray(observed, dtype=np.float64)
    lo = np.asarray(lower, dtype=np.float64)
    hi = np.asarray(upper, dtype=np.float64)
    below = np.maximum(lo - y, 0.0)
    above = np.maximum(y - hi, 0.0)
    return (hi - lo) + (2.0 / alpha) * (below + above)


@dataclass(frozen=True)
class CoverageTestResult:
    r"""Christoffersen (1998) interval-forecast evaluation.

    Parameters
    ----------
    coverage : float
        Nominal coverage :math:`p`.
    hit_rate : float
        Empirical coverage :math:`\hat\pi`.
    nobs : int
        Number of forecasts.
    lr_uc, pvalue_uc : float
        Unconditional coverage LR statistic (:math:`\chi^2(1)`) and p-value.
    lr_ind, pvalue_ind : float
        Independence LR statistic against a first-order Markov chain
        (:math:`\chi^2(1)`) and p-value.
    lr_cc, pvalue_cc : float
        Conditional coverage :math:`LR_{cc} = LR_{uc} + LR_{ind}` (:math:`\chi^2(2)`).
    transitions : pandas.DataFrame
        Counts :math:`n_{ij}` of transitions from state ``i`` to ``j`` (1 = hit).

    Examples
    --------
    >>> from nowcastbox.evaluation.scoring import christoffersen_test
    >>> res = christoffersen_test([1, 1, 0, 1, 1, 1, 1, 0, 1, 1], 0.8)
    >>> res.hit_rate, res.nobs
    (0.8, 10)
    """

    coverage: float
    hit_rate: float
    nobs: int
    lr_uc: float
    pvalue_uc: float
    lr_ind: float
    pvalue_ind: float
    lr_cc: float
    pvalue_cc: float
    transitions: pd.DataFrame

    def reject(self, alpha: float = 0.05, *, test: str = "cc") -> bool:
        """Whether the chosen null (``"uc"``, ``"ind"`` or ``"cc"``) is rejected."""
        pvalues = {"uc": self.pvalue_uc, "ind": self.pvalue_ind, "cc": self.pvalue_cc}
        if test not in pvalues:
            raise ValueError(f"test must be one of {sorted(pvalues)}, got {test!r}.")
        return pvalues[test] < alpha


def _nonnegative(value: float) -> float:
    """Clip a likelihood-ratio statistic at zero (removes rounding and ``-0.0``)."""
    return float(value) if value > 0.0 else 0.0


def _xlogy(count: float, prob: float) -> float:
    """``count * log(prob)`` with the convention ``0 * log(0) = 0``."""
    return float(scipy.special.xlogy(count, prob))


def _bernoulli_loglik(n1: float, n0: float, p: float) -> float:
    return _xlogy(n1, p) + _xlogy(n0, 1.0 - p)


def christoffersen_test(hits: ArrayLike, coverage: float) -> CoverageTestResult:
    r"""Christoffersen (1998) tests of unconditional and conditional coverage.

    Parameters
    ----------
    hits : array_like of bool or {0, 1}
        Hit sequence in time order (1 = realisation inside the interval), e.g. from
        :func:`interval_hits`. At least 2 values.
    coverage : float
        Nominal coverage :math:`p` in ``(0, 1)``.

    Returns
    -------
    CoverageTestResult
        :math:`LR_{uc}` (:math:`H_0: \pi = p`), :math:`LR_{ind}` (hits independent
        vs. first-order Markov) and :math:`LR_{cc} = LR_{uc} + LR_{ind}`.

    Raises
    ------
    ValueError
        On invalid hits or coverage.

    Notes
    -----
    Uses :math:`0\log 0 = 0`; when a transition row is empty its probability is
    unidentified and the corresponding term vanishes.

    Examples
    --------
    >>> from nowcastbox.evaluation.scoring import christoffersen_test
    >>> res = christoffersen_test([1] * 45 + [0] * 5, 0.9)
    >>> round(res.lr_uc, 6), res.pvalue_uc == 1.0
    (0.0, True)
    """
    p = _level(coverage)
    h = np.asarray(hits)
    if h.ndim != 1 or h.size < 2:
        raise ValueError("hits must be a 1-D sequence with at least 2 values.")
    if not bool(np.isin(h, [0, 1]).all()):
        raise ValueError("hits must contain only 0/1 or booleans.")
    h = h.astype(int)
    n = h.size
    n1 = float(h.sum())
    pi = n1 / n
    lr_uc = _nonnegative(
        -2.0 * (_bernoulli_loglik(n1, n - n1, p) - _bernoulli_loglik(n1, n - n1, pi))
    )
    prev, curr = h[:-1], h[1:]
    counts = np.array([[np.sum((prev == i) & (curr == j)) for j in (0, 1)] for i in (0, 1)], float)
    row = counts.sum(axis=1)
    pi01 = counts[0, 1] / row[0] if row[0] > 0 else 0.0
    pi11 = counts[1, 1] / row[1] if row[1] > 0 else 0.0
    pi_t = (counts[0, 1] + counts[1, 1]) / (n - 1)
    l_markov = _bernoulli_loglik(counts[0, 1], counts[0, 0], pi01) + _bernoulli_loglik(
        counts[1, 1], counts[1, 0], pi11
    )
    l_indep = _bernoulli_loglik(counts[0, 1] + counts[1, 1], counts[0, 0] + counts[1, 0], pi_t)
    lr_ind = _nonnegative(-2.0 * (l_indep - l_markov))
    lr_cc = lr_uc + lr_ind
    transitions = pd.DataFrame(
        counts.astype(int), index=pd.Index([0, 1], name="from"), columns=pd.Index([0, 1], name="to")
    )
    return CoverageTestResult(
        coverage=p,
        hit_rate=pi,
        nobs=n,
        lr_uc=lr_uc,
        pvalue_uc=float(scipy.stats.chi2.sf(lr_uc, 1)),
        lr_ind=lr_ind,
        pvalue_ind=float(scipy.stats.chi2.sf(lr_ind, 1)),
        lr_cc=lr_cc,
        pvalue_cc=float(scipy.stats.chi2.sf(lr_cc, 2)),
        transitions=transitions,
    )


# ====================================================================== quantile scores
def quantile_score(observed: ArrayLike, quantile: ArrayLike, level: ArrayLike) -> FloatArray:
    r"""Quantile (pinball) score :math:`QS_\tau(q, y) = (\mathbb 1\{y < q\} - \tau)(q - y)`.

    Parameters
    ----------
    observed : array_like
        Realisations.
    quantile : array_like
        Forecast quantiles :math:`q`.
    level : array_like
        Quantile levels :math:`\tau \in (0, 1)`; all three arguments broadcast.

    Returns
    -------
    numpy.ndarray
        Non-negative scores (smaller is better).

    Raises
    ------
    ValueError
        On invalid input.

    Examples
    --------
    >>> from nowcastbox.evaluation.scoring import quantile_score
    >>> quantile_score([0.0, 2.0], 1.0, 0.9).round(2).tolist()
    [0.1, 0.9]
    """
    y = _finite(observed, "observed")
    q = _finite(quantile, "quantile")
    tau = _finite(level, "level")
    if not bool(np.all((tau > 0) & (tau < 1))):
        raise ValueError("Quantile levels must lie in (0, 1).")
    return ((y < q).astype(float) - tau) * (q - y)


_QUANTILE_WEIGHTS: dict[str, Callable[[FloatArray], FloatArray]] = {
    "uniform": lambda t: np.ones_like(t),
    "center": lambda t: t * (1.0 - t),
    "left": lambda t: (1.0 - t) ** 2,
    "right": lambda t: t**2,
}


def _quantile_weights(weights: QuantileWeight | ArrayLike, tau: FloatArray) -> FloatArray:
    """Weights of :func:`weighted_quantile_score` at the levels ``tau``."""
    if isinstance(weights, str):
        if weights not in _QUANTILE_WEIGHTS:
            raise ValueError(
                f"weights must be one of {sorted(_QUANTILE_WEIGHTS)}, got {weights!r}."
            )
        return _QUANTILE_WEIGHTS[weights](tau)
    w = np.atleast_1d(_finite(weights, "weights"))
    if w.shape != tau.shape or bool((w < 0).any()):
        raise ValueError("weights must be non-negative, one per level.")
    return w


def weighted_quantile_score(
    observed: ArrayLike,
    quantiles: ArrayLike | pd.DataFrame,
    levels: Sequence[float] | ArrayLike | None = None,
    *,
    weights: QuantileWeight | ArrayLike = "uniform",
) -> FloatArray:
    r"""Quantile-weighted score of Gneiting & Ranjan (2011).

    .. math::

        \mathrm{wQS}(y) = \frac{2}{J}\sum_{j=1}^{J} w(\tau_j)\,QS_{\tau_j}(q_{\tau_j}, y).

    With ``weights="uniform"`` and an equally spaced grid of levels it approximates
    the CRPS; ``"center"`` (:math:`\tau(1-\tau)`), ``"left"`` (:math:`(1-\tau)^2`) and
    ``"right"`` (:math:`\tau^2`) emphasise the centre or one tail.

    Parameters
    ----------
    observed : array_like, shape (n,)
        Realisations.
    quantiles : array_like or pandas.DataFrame, shape (n, J)
        Forecast quantiles; a DataFrame whose columns are the levels (e.g. from
        :meth:`NowcastDistribution.quantiles`) needs no ``levels``.
    levels : sequence of float, optional
        Quantile levels (required unless ``quantiles`` is a DataFrame).
    weights : {"uniform", "center", "left", "right"} or array_like, default "uniform"
        Weight function or explicit weights (length ``J``).

    Returns
    -------
    numpy.ndarray, shape (n,)
        Scores (smaller is better).

    Raises
    ------
    ValueError
        On inconsistent shapes, invalid levels or unknown weight names.

    Examples
    --------
    >>> import numpy as np
    >>> from nowcastbox.density import NowcastDistribution
    >>> from nowcastbox.evaluation.scoring import weighted_quantile_score, crps
    >>> d = NowcastDistribution(["2020Q1"], [0.0], [1.0])
    >>> taus = (np.arange(1, 200) - 0.5) / 199
    >>> wqs = weighted_quantile_score([0.3], d.quantiles(taus))
    >>> bool(abs(wqs[0] - crps(d, 0.3)[0]) < 5e-3)
    True
    """
    if isinstance(quantiles, pd.DataFrame) and levels is None:
        levels = np.asarray(quantiles.columns, dtype=np.float64)
    if levels is None:
        raise ValueError("levels are required when quantiles is not a DataFrame.")
    tau = np.atleast_1d(_finite(levels, "levels"))
    q = np.atleast_2d(_finite(np.asarray(quantiles, dtype=np.float64), "quantiles"))
    y = np.atleast_1d(_finite(observed, "observed"))
    if tau.ndim != 1 or q.shape[-1] != tau.size:
        raise ValueError("quantiles must have one column per level.")
    if q.shape[0] != y.size:
        raise ValueError("quantiles must have one row per observation.")
    w = _quantile_weights(weights, tau)
    qs = quantile_score(y[:, None], q, tau[None, :])
    return np.asarray(2.0 * (qs @ w) / tau.size, dtype=np.float64)
