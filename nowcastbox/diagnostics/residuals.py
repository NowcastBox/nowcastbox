r"""Residual diagnostics: serial correlation (Ljung-Box) and normality (Jarque-Bera).

For a residual series :math:`e_1, \dots, e_n` with sample autocorrelations
:math:`\hat\rho_k`, the Ljung & Box (1978) statistic

.. math:: Q(h) = n(n + 2) \sum_{k=1}^{h} \frac{\hat\rho_k^2}{n - k}
          \sim \chi^2(h - m)

tests the absence of autocorrelation up to lag :math:`h` (:math:`m` = parameters of a
fitted ARMA filter, 0 for raw residuals). Missing values inside the sample are
handled by computing the autocovariances over the available pairs (the series is
kept on its native grid, so gaps do not shift the lags). The Jarque & Bera (1980)
statistic

.. math:: JB = \frac{n}{6}\left(S^2 + \frac{(K - 3)^2}{4}\right) \sim \chi^2(2)

uses the sample skewness :math:`S` and kurtosis :math:`K`.

References
----------
Jarque, C. M., & Bera, A. K. (1980). Efficient tests for normality,
homoscedasticity and serial independence of regression residuals. *Economics
Letters*, 6(3), 255-259.

Ljung, G. M., & Box, G. E. P. (1978). On a measure of lack of fit in time series
models. *Biometrika*, 65(2), 297-303.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import scipy.stats

from nowcastbox.diagnostics._common import FloatArray, check_alpha, check_int

__all__ = ["ResidualDiagnostics", "jarque_bera", "ljung_box", "residual_diagnostics"]


def _as_vector(x: Any) -> FloatArray:
    arr = np.asarray(x, dtype=np.float64).ravel()
    if np.isinf(arr).any():
        raise ValueError("Residuals must not contain infinite values.")
    return arr


def autocorrelation(x: Any, nlags: int) -> FloatArray:
    r"""Sample autocorrelations at lags ``1..nlags`` over the available pairs.

    Parameters
    ----------
    x : array_like
        Series on a regular grid (``NaN`` = missing).
    nlags : int
        Largest lag.

    Returns
    -------
    numpy.ndarray
        :math:`\hat\rho_1, \dots, \hat\rho_h` (denominator ``n`` as in Box-Jenkins).

    Examples
    --------
    >>> from nowcastbox.diagnostics.residuals import autocorrelation
    >>> autocorrelation([1.0, -1.0, 1.0, -1.0], 1).round(2).tolist()
    [-0.75]
    """
    v = _as_vector(x)
    ok = np.isfinite(v)
    d = np.where(ok, v - v[ok].mean(), 0.0)
    g0 = float(d @ d)
    out = np.empty(nlags)
    for k in range(1, nlags + 1):
        out[k - 1] = float(d[k:] @ d[:-k]) / g0 if g0 > 0 else np.nan
    return out


def ljung_box(x: Any, lags: int | None = None, *, model_df: int = 0) -> tuple[float, float, int]:
    r"""Ljung-Box test of no autocorrelation up to ``lags``.

    Parameters
    ----------
    x : array_like
        Residuals on a regular grid (``NaN`` = missing, ignored pairwise).
    lags : int, optional
        Number of lags :math:`h`; default ``min(10, n // 5)`` (at least 1).
    model_df : int, default 0
        Degrees of freedom absorbed by a fitted model (subtracted from ``lags``).

    Returns
    -------
    statistic : float
        :math:`Q(h)`.
    pvalue : float
        :math:`P(\chi^2_{h - m} > Q)`.
    lags : int
        Lags used.

    Raises
    ------
    ValueError
        If there are too few observations or ``lags <= model_df``.

    Examples
    --------
    >>> import numpy as np
    >>> from nowcastbox.diagnostics import ljung_box
    >>> e = np.random.default_rng(0).standard_normal(200)
    >>> stat, p, h = ljung_box(e, 10)
    >>> h, p > 0.05
    (10, True)
    """
    v = _as_vector(x)
    n = int(np.isfinite(v).sum())
    h = max(1, min(10, n // 5)) if lags is None else check_int(lags, "lags", 1)
    m = check_int(model_df, "model_df", 0)
    if h <= m:
        raise ValueError(f"lags ({h}) must exceed model_df ({m}).")
    if n < h + 2:
        raise ValueError(f"Ljung-Box needs more than {h + 1} observations, got {n}.")
    rho = autocorrelation(v, h)
    stat = float(n * (n + 2) * np.sum(rho**2 / (n - np.arange(1, h + 1))))
    return stat, float(scipy.stats.chi2.sf(stat, h - m)), h


def jarque_bera(x: Any) -> tuple[float, float, float, float]:
    r"""Jarque-Bera normality test.

    Parameters
    ----------
    x : array_like
        Residuals (``NaN`` ignored).

    Returns
    -------
    statistic : float
        :math:`JB`.
    pvalue : float
        :math:`P(\chi^2_2 > JB)`.
    skewness : float
        Sample skewness.
    kurtosis : float
        Sample (non-excess) kurtosis.

    Raises
    ------
    ValueError
        With fewer than 3 observations or a constant series.

    Examples
    --------
    >>> import numpy as np
    >>> from nowcastbox.diagnostics import jarque_bera
    >>> e = np.random.default_rng(1).standard_t(3, 2000)
    >>> jarque_bera(e)[1] < 0.01
    True
    """
    v = _as_vector(x)
    v = v[np.isfinite(v)]
    n = v.size
    if n < 3:
        raise ValueError(f"Jarque-Bera needs at least 3 observations, got {n}.")
    d = v - v.mean()
    m2 = float(np.mean(d**2))
    if m2 <= 0:
        raise ValueError("Jarque-Bera is undefined for a constant series.")
    skew = float(np.mean(d**3)) / m2**1.5
    kurt = float(np.mean(d**4)) / m2**2
    stat = n / 6.0 * (skew**2 + (kurt - 3.0) ** 2 / 4.0)
    return stat, float(scipy.stats.chi2.sf(stat, 2)), skew, kurt


@dataclass(frozen=True, eq=False)
class ResidualDiagnostics:
    """Ljung-Box and Jarque-Bera tests for several residual series.

    Parameters
    ----------
    table : pandas.DataFrame
        Indexed by series: ``kind``, ``n_obs``, ``mean``, ``std``, ``lb_lags``,
        ``lb_stat``, ``lb_pvalue``, ``lb_reject``, ``skewness``, ``kurtosis``,
        ``jb_stat``, ``jb_pvalue``, ``jb_reject``, ``note``.
    alpha : float
        Significance level of the ``*_reject`` columns.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from nowcastbox.diagnostics import residual_diagnostics
    >>> rng = np.random.default_rng(0)
    >>> e = pd.DataFrame({"a": rng.standard_normal(120), "b": rng.standard_normal(120)})
    >>> rd = residual_diagnostics(e)
    >>> rd.table["lb_lags"].tolist()
    [10, 10]
    """

    table: pd.DataFrame
    alpha: float

    def to_frame(self) -> pd.DataFrame:
        """Copy of :attr:`table`.

        Returns
        -------
        pandas.DataFrame
            One row per residual series.

        Examples
        --------
        >>> rd.to_frame().shape[0]  # doctest: +SKIP
        2
        """
        return self.table.copy()

    def summary(self) -> str:
        """Text summary.

        Returns
        -------
        str
            Rejection counts by residual kind.

        Examples
        --------
        >>> print(rd.summary())  # doctest: +SKIP
        """
        lines = [f"Residual diagnostics (alpha={self.alpha})"]
        for kind, g in self.table.groupby("kind", sort=False):
            n = int(g["lb_pvalue"].notna().sum())
            lines.append(
                f"  {kind!s:<16}{len(g)} series; Ljung-Box rejects {int(g['lb_reject'].sum())}"
                f"/{n}, Jarque-Bera rejects {int(g['jb_reject'].sum())}"
                f"/{int(g['jb_pvalue'].notna().sum())}"
            )
            bad = g.index[g["lb_reject"].to_numpy(bool)].astype(str).tolist()
            if bad:
                lines.append(
                    f"  {'':<16}autocorrelated: {', '.join(bad[:10])}"
                    + (" ..." if len(bad) > 10 else "")
                )
        return "\n".join(lines)


def _row(values: FloatArray, lags: int | None, model_df: int) -> dict[str, Any]:
    ok = values[np.isfinite(values)]
    row: dict[str, Any] = {
        "n_obs": int(ok.size),
        "mean": float(ok.mean()) if ok.size else np.nan,
        "std": float(ok.std(ddof=1)) if ok.size > 1 else np.nan,
    }
    notes = []
    try:
        row["lb_stat"], row["lb_pvalue"], row["lb_lags"] = ljung_box(
            values, lags, model_df=model_df
        )
    except ValueError as exc:
        row.update(lb_stat=np.nan, lb_pvalue=np.nan, lb_lags=lags)
        notes.append(f"Ljung-Box: {exc}")
    try:
        row["jb_stat"], row["jb_pvalue"], row["skewness"], row["kurtosis"] = jarque_bera(values)
    except ValueError as exc:
        row.update(jb_stat=np.nan, jb_pvalue=np.nan, skewness=np.nan, kurtosis=np.nan)
        notes.append(f"Jarque-Bera: {exc}")
    row["note"] = "; ".join(notes)
    return row


def _as_mapping(residuals: Any) -> dict[str, FloatArray]:
    if isinstance(residuals, pd.Series):
        return {str(residuals.name or "residuals"): residuals.to_numpy(dtype=np.float64)}
    if isinstance(residuals, pd.DataFrame):
        return {str(c): residuals[c].to_numpy(dtype=np.float64) for c in residuals.columns}
    if isinstance(residuals, Mapping):
        return {str(k): _as_vector(v) for k, v in residuals.items()}
    raise TypeError(
        "residuals must be a pandas Series, DataFrame or a mapping of name to series, "
        f"got {type(residuals).__name__}."
    )


def residual_diagnostics(
    residuals: pd.DataFrame | pd.Series | Mapping[str, Any],
    *,
    lags: int | None = None,
    model_df: int = 0,
    alpha: float = 0.05,
    kind: str | Mapping[str, str] = "residual",
) -> ResidualDiagnostics:
    """Ljung-Box and Jarque-Bera tests for each residual series.

    Parameters
    ----------
    residuals : pandas.DataFrame, pandas.Series or mapping
        Residual series, each on its own regular (native) grid; ``NaN`` = missing.
    lags : int, optional
        Ljung-Box lags (default ``min(10, n // 5)`` per series).
    model_df : int, default 0
        Degrees of freedom subtracted in the Ljung-Box test.
    alpha : float, default 0.05
        Significance level of the reject flags.
    kind : str or mapping, default "residual"
        Label of every series (or per-series labels, e.g. ``"idiosyncratic"``,
        ``"bridge"``).

    Returns
    -------
    ResidualDiagnostics
        Tidy table of tests.

    Raises
    ------
    TypeError
        If ``residuals`` has an unsupported type.
    ValueError
        On invalid options or infinite residuals.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from nowcastbox.diagnostics import residual_diagnostics
    >>> rng = np.random.default_rng(2)
    >>> e = np.zeros(300)
    >>> for t in range(1, 300):
    ...     e[t] = 0.8 * e[t - 1] + rng.standard_normal()
    >>> bool(residual_diagnostics(pd.Series(e, name="ar1")).table.loc["ar1", "lb_reject"])
    True
    """
    a = check_alpha(alpha)
    m = check_int(model_df, "model_df", 0)
    if lags is not None:
        check_int(lags, "lags", 1)
    series = _as_mapping(residuals)
    if not series:
        raise ValueError("No residual series given.")
    rows = []
    for name, values in series.items():
        label = kind.get(name, "residual") if isinstance(kind, Mapping) else kind
        rows.append({"series": name, "kind": label, **_row(values, lags, m)})
    cols = [
        "kind",
        "n_obs",
        "mean",
        "std",
        "lb_lags",
        "lb_stat",
        "lb_pvalue",
        "lb_reject",
        "skewness",
        "kurtosis",
        "jb_stat",
        "jb_pvalue",
        "jb_reject",
        "note",
    ]
    table = pd.DataFrame(rows).set_index("series")
    table["lb_reject"] = table["lb_pvalue"].to_numpy(float) < a
    table["jb_reject"] = table["jb_pvalue"].to_numpy(float) < a
    return ResidualDiagnostics(table=table[cols], alpha=a)
