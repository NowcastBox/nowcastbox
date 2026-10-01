"""``NowcastExperiment``: fit several model specifications on the same data and compare.

Follows the ``panelbox.experiment.PanelExperiment`` pattern: an experiment holds one
dataset and target, models are added by name, fitted, and compared in a table
(current nowcast, log-likelihood, in-sample fit, timings). An optional backtest hook
runs any pseudo-real-time evaluation per model and collects RMSFE by horizon.
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from nowcastbox._logging import get_logger
from nowcastbox.core.exceptions import ModelNotFittedError, NowcastBoxWarning
from nowcastbox.core.results import NowcastResults
from nowcastbox.visualization import plot_rmsfe_by_horizon, rmsfe_frame
from nowcastbox.visualization._common import (
    as_float_array,
    finish_mpl,
    finish_plotly,
    mpl_context,
    new_axes,
    new_plotly_figure,
    resolve,
    to_plot_index,
)
from nowcastbox.visualization.themes import Theme

__all__ = ["BacktestHook", "NowcastExperiment", "in_sample_metrics", "pseudo_real_time_hook"]

logger = get_logger(__name__)

BacktestHook = Callable[..., Any]
"""``hook(estimator, data, target, **kwargs) -> result``; see :meth:`NowcastExperiment.run_backtest`."""


def pseudo_real_time_hook(estimator: Any, data: Any, target: str, **kwargs: Any) -> Any:
    """Default backtest hook of :meth:`NowcastExperiment.run_backtest`.

    Runs a :class:`~nowcastbox.evaluation.PseudoRealTimeBacktest` of ``estimator``
    alone (no benchmarks, so that :meth:`NowcastExperiment.rmsfe_table` gets one
    column per model).

    Parameters
    ----------
    estimator : BaseNowcaster
        Unfitted model.
    data : MixedFrequencyData or pandas.DataFrame
        Final data (vintages are built with ``delay``/``calendar``) or a
        :class:`~nowcastbox.vintages.VintageStore`.
    target : str
        Target series.
    **kwargs
        Options of :class:`~nowcastbox.evaluation.PseudoRealTimeBacktest` (``delay`` or
        ``calendar``, ``start``, ``end``, ``step``...) plus ``n_jobs`` for ``run``.

    Returns
    -------
    BacktestResults
        Forecasts of the model at every vintage.

    Examples
    --------
    >>> exp.run_backtest(delay=30, start="2014-01-01", end="2014-06-01")  # doctest: +SKIP
    """
    from nowcastbox.evaluation import PseudoRealTimeBacktest

    n_jobs = kwargs.pop("n_jobs", None)
    backtest = PseudoRealTimeBacktest(model=estimator, data=data, target=target, **kwargs)
    return backtest.run(n_jobs=n_jobs)


def in_sample_metrics(results: NowcastResults) -> dict[str, float]:
    """In-sample fit of the target: RMSE, MAE and R-squared over observed periods.

    Parameters
    ----------
    results : NowcastResults
        Fitted results.

    Returns
    -------
    dict
        ``n_in_sample``, ``rmse_in_sample``, ``mae_in_sample``, ``r2_in_sample``
        (NaN when fewer than two periods have both observation and fit).

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from nowcastbox.core.results import build_nowcast_frame
    >>> idx = pd.period_range("2020Q1", periods=3, freq="Q")
    >>> f = build_nowcast_frame(
    ...     pd.Series([1.0, 2.0, 3.0], index=idx), pd.Series([1.0, 2.0, 4.0], index=idx)
    ... )
    >>> m = in_sample_metrics(NowcastResults(target="y", nowcast=f))
    >>> round(m["mae_in_sample"], 4), round(m["r2_in_sample"], 4)
    (0.3333, 0.5)
    """
    frame = results.nowcast[["observed", "in_sample"]].dropna()
    n = len(frame)
    out = {
        "n_in_sample": float(n),
        "rmse_in_sample": np.nan,
        "mae_in_sample": np.nan,
        "r2_in_sample": np.nan,
    }
    if n < 2:
        return out
    y = frame["observed"].to_numpy(dtype=float)
    err = y - frame["in_sample"].to_numpy(dtype=float)
    out["rmse_in_sample"] = float(np.sqrt(np.mean(err**2)))
    out["mae_in_sample"] = float(np.mean(np.abs(err)))
    tss = float(np.sum((y - y.mean()) ** 2))
    out["r2_in_sample"] = float(1.0 - np.sum(err**2) / tss) if tss > 0 else np.nan
    return out


class NowcastExperiment:
    """Fit and compare several nowcasting models on the same data.

    Parameters
    ----------
    data : MixedFrequencyData or pandas.DataFrame
        Panel passed unchanged to every estimator's ``fit``.
    target : str
        Target name or formula (``"gdp"``, ``"gdp ~ ."``).
    frequency : optional
        Frequency specification forwarded to ``fit`` (for DataFrames).
    models : mapping of str to estimator, optional
        Initial models (see :meth:`add_model`).

    Attributes
    ----------
    failures : dict
        ``{name: exception}`` of fits that failed with ``errors="warn"``.

    Raises
    ------
    ValueError
        Empty target.

    Examples
    --------
    >>> from nowcastbox.models import TwoStepDFM
    >>> from nowcastbox.models.two_step import simulate_two_step_example
    >>> data = simulate_two_step_example(random_state=0)
    >>> exp = NowcastExperiment(data, "gdp")
    >>> exp.add_model("dfm1", TwoStepDFM(n_factors=1)).add_model("dfm2", TwoStepDFM(n_factors=2))
    NowcastExperiment(target='gdp', models=['dfm1', 'dfm2'], fitted=[])
    >>> _ = exp.fit_all()
    >>> exp.compare().index.tolist()
    ['dfm1', 'dfm2']
    """

    def __init__(
        self,
        data: Any,
        target: str,
        *,
        frequency: Any = None,
        models: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(target, str) or not target.strip():
            raise ValueError("target must be a non-empty string.")
        self.data = data
        self.target = target
        self.frequency = frequency
        self._models: dict[str, Any] = {}
        self._results: dict[str, NowcastResults] = {}
        self._fit_seconds: dict[str, float] = {}
        self._backtests: dict[str, Any] = {}
        self.failures: dict[str, BaseException] = {}
        for name, estimator in (models or {}).items():
            self.add_model(name, estimator)

    # ------------------------------------------------------------------ models
    def add_model(self, name: str, estimator: Any, *, overwrite: bool = False) -> NowcastExperiment:
        """Add a model specification (not fitted yet).

        Parameters
        ----------
        name : str
            Unique model name.
        estimator : estimator
            Object with ``fit(data, target, frequency=...)`` returning
            :class:`NowcastResults` (any :class:`~nowcastbox.core.base.BaseNowcaster`).
        overwrite : bool, default False
            Replace an existing model (its results are discarded).

        Returns
        -------
        NowcastExperiment
            ``self`` (chainable).

        Raises
        ------
        ValueError
            Empty or duplicated name.
        TypeError
            Estimator without a ``fit`` method.

        Examples
        --------
        >>> from nowcastbox.models import TwoStepDFM
        >>> NowcastExperiment(None, "gdp").add_model("dfm", TwoStepDFM()).list_models()
        ['dfm']
        """
        if not isinstance(name, str) or not name:
            raise ValueError("Model name must be a non-empty string.")
        if name in self._models and not overwrite:
            raise ValueError(f"Model {name!r} already exists; pass overwrite=True.")
        if not callable(getattr(estimator, "fit", None)):
            raise TypeError(f"Estimator for {name!r} has no fit() method.")
        self._models[name] = estimator
        for store in (self._results, self._fit_seconds, self._backtests, self.failures):
            store.pop(name, None)
        return self

    def list_models(self) -> list[str]:
        """Names of the models, in insertion order.

        Returns
        -------
        list of str
            Model names.

        Examples
        --------
        >>> NowcastExperiment(None, "gdp").list_models()
        []
        """
        return list(self._models)

    def get_model(self, name: str) -> Any:
        """Return the estimator registered under ``name``.

        Parameters
        ----------
        name : str
            Model name.

        Returns
        -------
        estimator
            The estimator.

        Raises
        ------
        KeyError
            Unknown model.

        Examples
        --------
        >>> from nowcastbox.models import TwoStepDFM
        >>> exp = NowcastExperiment(None, "gdp", models={"dfm": TwoStepDFM()})
        >>> type(exp.get_model("dfm")).__name__
        'TwoStepDFM'
        """
        if name not in self._models:
            raise KeyError(f"Unknown model {name!r}; available: {self.list_models()}.")
        return self._models[name]

    # ------------------------------------------------------------------ fitting
    def fit_model(self, name: str, estimator: Any = None, **fit_kwargs: Any) -> NowcastResults:
        """Fit one model (adding it first when ``estimator`` is given).

        Parameters
        ----------
        name : str
            Model name.
        estimator : estimator, optional
            New estimator to add (replaces an existing one with the same name).
        **fit_kwargs
            Passed to ``estimator.fit``.

        Returns
        -------
        NowcastResults
            The results (also stored in the experiment).

        Raises
        ------
        KeyError
            Unknown model and no estimator.
        TypeError
            ``fit`` did not return :class:`NowcastResults`.

        Examples
        --------
        >>> from nowcastbox.models import TwoStepDFM
        >>> from nowcastbox.models.two_step import simulate_two_step_example
        >>> exp = NowcastExperiment(simulate_two_step_example(random_state=0), "gdp")
        >>> exp.fit_model("dfm", TwoStepDFM(n_factors=1)).model_name
        'TwoStepDFM'
        """
        if estimator is not None:
            self.add_model(name, estimator, overwrite=True)
        model = self.get_model(name)
        if self.frequency is not None:
            fit_kwargs.setdefault("frequency", self.frequency)
        start = time.perf_counter()
        results = model.fit(self.data, self.target, **fit_kwargs)
        elapsed = time.perf_counter() - start
        if not isinstance(results, NowcastResults):
            raise TypeError(
                f"{name!r}: fit() returned {type(results).__name__}, expected NowcastResults."
            )
        self._results[name] = results
        self._fit_seconds[name] = elapsed
        self.failures.pop(name, None)
        logger.info("Fitted %s in %.3f s", name, elapsed)
        return results

    def fit_all(
        self, *, errors: Literal["raise", "warn"] = "raise", refit: bool = False
    ) -> dict[str, NowcastResults]:
        """Fit every model.

        Parameters
        ----------
        errors : {"raise", "warn"}, default "raise"
            On a failing fit, re-raise, or issue a
            :class:`~nowcastbox.core.exceptions.NowcastBoxWarning` and record the
            exception in :attr:`failures`.
        refit : bool, default False
            Refit models that are already fitted.

        Returns
        -------
        dict
            ``{name: results}`` of the fitted models.

        Raises
        ------
        ValueError
            Invalid ``errors`` or no models.

        Examples
        --------
        >>> NowcastExperiment(None, "gdp").fit_all()
        Traceback (most recent call last):
        ...
        ValueError: The experiment has no models; use add_model().
        """
        if errors not in ("raise", "warn"):
            raise ValueError(f"errors must be 'raise' or 'warn'; got {errors!r}.")
        if not self._models:
            raise ValueError("The experiment has no models; use add_model().")
        for name in self._models:
            if name in self._results and not refit:
                continue
            try:
                self.fit_model(name)
            except Exception as exc:
                if errors == "raise":
                    raise
                self.failures[name] = exc
                warnings.warn(
                    f"Model {name!r} failed to fit: {exc}", NowcastBoxWarning, stacklevel=2
                )
        return self.results

    @property
    def results(self) -> dict[str, NowcastResults]:
        """``{name: results}`` of the fitted models (copy of the mapping)."""
        return dict(self._results)

    def get_results(self, name: str) -> NowcastResults:
        """Results of a fitted model.

        Parameters
        ----------
        name : str
            Model name.

        Returns
        -------
        NowcastResults
            The results.

        Raises
        ------
        KeyError
            Unknown model.
        ModelNotFittedError
            Model not fitted yet.

        Examples
        --------
        >>> exp = NowcastExperiment(None, "gdp")
        >>> exp.get_results("dfm")
        Traceback (most recent call last):
        ...
        KeyError: "Unknown model 'dfm'; available: []."
        """
        self.get_model(name)
        if name not in self._results:
            raise ModelNotFittedError(f"Model {name!r} is not fitted; call fit_model/fit_all.")
        return self._results[name]

    def _fitted(self) -> dict[str, NowcastResults]:
        if not self._results:
            raise ModelNotFittedError("No fitted models; call fit_all() first.")
        return {n: self._results[n] for n in self._models if n in self._results}

    # ------------------------------------------------------------------ comparison
    def compare(self, period: pd.Period | str | None = None) -> pd.DataFrame:
        """Comparison table of the fitted models.

        Parameters
        ----------
        period : pandas.Period or str, optional
            Target period of the reported nowcast (default: each model's current
            nowcast, the first period after the last observation).

        Returns
        -------
        pandas.DataFrame
            One row per model with ``model_name``, ``nowcast_period``, ``nowcast``,
            ``std``, ``lower_90``, ``upper_90`` (NaN when not available),
            ``loglikelihood``, ``n_factors``, ``n_iter``, ``converged``, the
            :func:`in_sample_metrics` and ``fit_seconds``.

        Raises
        ------
        ModelNotFittedError
            No fitted model.

        Examples
        --------
        >>> exp.compare()[["nowcast", "rmse_in_sample"]]  # doctest: +SKIP
        """
        rows = {}
        for name, res in self._fitted().items():
            target_period = _resolve_period(res, period)
            row: dict[str, Any] = {
                "model_name": res.model_name,
                "nowcast_period": None if target_period is None else str(target_period),
            }
            row.update(_period_values(res, target_period))
            row.update(
                {
                    "loglikelihood": res.loglikelihood,
                    "n_factors": res.n_factors,
                    "n_iter": res.n_iter,
                    "converged": res.converged,
                }
            )
            row.update(in_sample_metrics(res))
            row["fit_seconds"] = self._fit_seconds.get(name, np.nan)
            rows[name] = row
        table = pd.DataFrame.from_dict(rows, orient="index")
        table.index.name = "model"
        return table

    def nowcast_table(
        self, *, kind: Literal["estimate", "out_of_sample"] = "estimate"
    ) -> pd.DataFrame:
        """Target estimates of every fitted model side by side.

        Parameters
        ----------
        kind : {"estimate", "out_of_sample"}, default "estimate"
            In-sample fit combined with out-of-sample estimates, or out-of-sample only.

        Returns
        -------
        pandas.DataFrame
            Index = target periods (union), first column ``observed``, then one column
            per model.

        Raises
        ------
        ValueError
            Invalid ``kind``.
        ModelNotFittedError
            No fitted model.

        Examples
        --------
        >>> exp.nowcast_table().columns.tolist()  # doctest: +SKIP
        ['observed', 'dfm1', 'dfm2']
        """
        if kind not in ("estimate", "out_of_sample"):
            raise ValueError(f"kind must be 'estimate' or 'out_of_sample'; got {kind!r}.")
        fitted = self._fitted()
        columns = {
            name: (res.estimate if kind == "estimate" else res.out_of_sample)
            for name, res in fitted.items()
        }
        observed = [res.observed for res in fitted.values()]
        obs = pd.concat(observed, axis=1).bfill(axis=1).iloc[:, 0].rename("observed")
        return pd.concat([obs, pd.DataFrame(columns)], axis=1).sort_index()

    # ------------------------------------------------------------------ backtest
    def run_backtest(
        self,
        hook: BacktestHook | None = None,
        *,
        models: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run a backtest for each model (pseudo real-time by default).

        ``hook`` is called as ``hook(estimator, data, target, **kwargs)`` for each model.
        Without a hook, :func:`pseudo_real_time_hook` runs a
        :class:`~nowcastbox.evaluation.PseudoRealTimeBacktest` of the model (pass its
        options, e.g. ``delay``/``calendar``, ``start``, ``end``, as ``**kwargs``). The
        return value is stored; when it is a Series of RMSFE by horizon, a DataFrame, or
        an object with ``rmsfe_by_horizon()`` (e.g.
        :class:`~nowcastbox.evaluation.BacktestResults`), :meth:`rmsfe_table` can
        combine the models.

        Parameters
        ----------
        hook : callable, optional
            ``hook(estimator, data, target, **kwargs) -> result``; default
            :func:`pseudo_real_time_hook`.
        models : list of str, optional
            Models to evaluate (default: all).
        **kwargs
            Passed to ``hook``.

        Returns
        -------
        dict
            ``{name: result}``.

        Raises
        ------
        TypeError
            ``hook`` not callable.
        KeyError
            Unknown model.

        Examples
        --------
        >>> import pandas as pd
        >>> from nowcastbox.models import TwoStepDFM
        >>> exp = NowcastExperiment(None, "gdp", models={"dfm": TwoStepDFM()})
        >>> out = exp.run_backtest(lambda est, data, target: pd.Series([0.5, 0.7], name="rmsfe"))
        >>> exp.rmsfe_table()["dfm"].tolist()
        [0.5, 0.7]
        """
        if hook is None:
            hook = pseudo_real_time_hook
        if not callable(hook):
            raise TypeError("hook must be callable.")
        names = self.list_models() if models is None else models
        out: dict[str, Any] = {}
        for name in names:
            out[name] = hook(self.get_model(name), self.data, self.target, **kwargs)
        self._backtests.update(out)
        return out

    @property
    def backtests(self) -> dict[str, Any]:
        """Results returned by the backtest hook, by model."""
        return dict(self._backtests)

    def rmsfe_table(self) -> pd.DataFrame:
        """Combine the backtest results into a table of RMSFE by horizon.

        Returns
        -------
        pandas.DataFrame
            Rows = horizons, columns = models.

        Raises
        ------
        ValueError
            No backtest results, or a result that cannot be read as RMSFE by horizon
            (a DataFrame must have exactly one column).

        Examples
        --------
        >>> NowcastExperiment(None, "gdp").rmsfe_table()
        Traceback (most recent call last):
        ...
        ValueError: No backtest results; call run_backtest() first.
        """
        if not self._backtests:
            raise ValueError("No backtest results; call run_backtest() first.")
        columns: dict[str, pd.Series] = {}
        for name, result in self._backtests.items():
            try:
                table = rmsfe_frame(result)
            except TypeError as exc:
                raise ValueError(f"Backtest result of {name!r} is not RMSFE by horizon.") from exc
            if table.shape[1] != 1:
                raise ValueError(
                    f"Backtest result of {name!r} has {table.shape[1]} columns; expected one."
                )
            columns[name] = table.iloc[:, 0]
        return pd.DataFrame(columns)

    # ------------------------------------------------------------------ output
    def plot(
        self,
        kind: Literal["nowcasts", "rmsfe"] = "nowcasts",
        *,
        backend: str = "plotly",
        theme: Theme | str | None = None,
        n_periods: int | None = 20,
        relative_to: str | None = None,
        title: str | None = None,
        ax: Any = None,
    ) -> Any:
        """Plot the models' estimates against the target, or the backtest RMSFE.

        Parameters
        ----------
        kind : {"nowcasts", "rmsfe"}, default "nowcasts"
            ``"nowcasts"``: observed target and each model's estimate (last
            ``n_periods``); ``"rmsfe"``: :meth:`rmsfe_table` by horizon.
        backend : {"plotly", "matplotlib"}, default "plotly"
            Plotting library.
        theme : Theme or str, optional
            Visual theme.
        n_periods : int or None, default 20
            Number of most recent periods (``"nowcasts"`` only).
        relative_to : str, optional
            Reference model for relative RMSFE (``"rmsfe"`` only).
        title : str, optional
            Title.
        ax : matplotlib.axes.Axes, optional
            Axes to draw on (Matplotlib only).

        Returns
        -------
        plotly.graph_objects.Figure or matplotlib.figure.Figure
            The figure.

        Raises
        ------
        ValueError
            Unknown ``kind`` or backend.
        ModelNotFittedError
            No fitted model (``"nowcasts"``).

        Examples
        --------
        >>> exp.plot("nowcasts", backend="matplotlib")  # doctest: +SKIP
        """
        if kind == "rmsfe":
            return plot_rmsfe_by_horizon(
                self.rmsfe_table(),
                relative_to=relative_to,
                backend=backend,
                theme=theme,
                title=title,
                ax=ax,
            )
        if kind != "nowcasts":
            raise ValueError(f"kind must be 'nowcasts' or 'rmsfe'; got {kind!r}.")
        th, be = resolve(theme, backend, ax)
        table = self.nowcast_table()
        if n_periods is not None:
            table = table.iloc[-n_periods:]
        title = f"Model comparison: {self.target}" if title is None else title
        x = to_plot_index(table.index)
        models = [c for c in table.columns if c != "observed"]
        if be == "plotly":
            import plotly.graph_objects as go

            fig = new_plotly_figure()
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=table["observed"],
                    name="Observed",
                    mode="lines+markers",
                    line={"color": th.observed_color, "width": th.line_width},
                )
            )
            for i, name in enumerate(models):
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=table[name],
                        name=name,
                        mode="lines+markers",
                        line={"color": th.color(i), "width": th.line_width},
                    )
                )
            return finish_plotly(fig, th, title=title)
        with mpl_context(th):
            fig, axes = new_axes(ax, th, None)
            axes.plot(
                x,
                as_float_array(table["observed"]),
                color=th.observed_color,
                marker="o",
                markersize=th.marker_size / 2,
                label="Observed",
            )
            for i, name in enumerate(models):
                axes.plot(
                    x,
                    as_float_array(table[name]),
                    color=th.color(i),
                    marker="o",
                    markersize=th.marker_size / 2.5,
                    label=name,
                )
            finish_mpl(axes, th, title=title, dates=True)
        return fig

    def summary(self) -> str:
        """Text summary: comparison table of the fitted models and failures.

        Returns
        -------
        str
            Formatted summary.

        Examples
        --------
        >>> print(NowcastExperiment(None, "gdp").summary())
        NowcastExperiment: target gdp, 0 model(s), 0 fitted
        """
        lines = [
            f"NowcastExperiment: target {self.target}, {len(self._models)} model(s), "
            f"{len(self._results)} fitted"
        ]
        if self._results:
            cols = [
                "nowcast_period",
                "nowcast",
                "loglikelihood",
                "rmse_in_sample",
                "r2_in_sample",
                "fit_seconds",
            ]
            with pd.option_context("display.width", 120, "display.max_columns", 20):
                lines.append(self.compare()[cols].to_string(float_format=lambda v: f"{v:.4f}"))
        for name, exc in self.failures.items():
            lines.append(f"FAILED {name}: {exc}")
        return "\n".join(lines)

    def report(self, name: str, path: str | Path | None = None, **kwargs: Any) -> str:
        """HTML report (:class:`~nowcastbox.reports.NowcastReport`) of one model.

        Parameters
        ----------
        name : str
            Fitted model.
        path : str or pathlib.Path, optional
            Output file.
        **kwargs
            Passed to :class:`~nowcastbox.reports.NowcastReport` (``news``,
            ``tracker``, ...). The model's backtest result is used as ``backtest``
            unless given.

        Returns
        -------
        str
            The HTML document.

        Examples
        --------
        >>> exp.report("dfm1", "dfm1.html")  # doctest: +SKIP
        """
        from nowcastbox.reports import NowcastReport

        results = self.get_results(name)
        if "backtest" not in kwargs and self._backtests:
            try:
                kwargs["backtest"] = self.rmsfe_table()
            except ValueError:
                logger.warning("Backtest results are not RMSFE tables; not added to the report.")
        kwargs.setdefault("title", f"Nowcast report: {self.target} ({name})")
        return NowcastReport(results, **kwargs).to_html(path)

    def __repr__(self) -> str:
        return (
            f"NowcastExperiment(target={self.target!r}, models={self.list_models()}, "
            f"fitted={[n for n in self._models if n in self._results]})"
        )


def _resolve_period(results: NowcastResults, period: pd.Period | str | None) -> pd.Period | None:
    if period is not None:
        return pd.Period(period, freq=results.target_frequency.pandas_freq)
    last = results.nowcast["observed"].last_valid_index()
    index = results.nowcast.index
    later = index if last is None else index[index > last]
    return None if len(later) == 0 else later[0]


def _period_values(results: NowcastResults, period: pd.Period | None) -> dict[str, float]:
    """Estimate and interval columns of ``results`` at ``period`` (NaN if absent)."""
    names = ("std", "lower_90", "upper_90")
    out = {"nowcast": np.nan, **dict.fromkeys(names, np.nan)}
    if period is None:
        return out
    frame = results.nowcast
    positions = np.flatnonzero(frame.index == period)
    if positions.size == 0:
        return out
    pos = int(positions[0])
    out["nowcast"] = float(results.estimate.to_numpy(dtype=float)[pos])
    for col in names:
        if col in frame.columns:
            out[col] = float(frame[col].to_numpy(dtype=float)[pos])
    return out
