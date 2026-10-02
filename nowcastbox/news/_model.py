r"""Linear view of a fitted factor model used by the news decomposition.

Every supported model produces its nowcast of a target period as an affine function of
the smoothed state at the target's storage slot,

.. math::

    \hat y_{\tau}(\Omega) = o + s\,\bigl(g'\,\mathbb{E}[\alpha_{t_\tau} \mid \Omega] + h\bigr),

where :math:`\Omega` is the information set (pattern and values of the standardised
observations), :math:`g` a fixed vector, :math:`h` an intercept and :math:`(o, s)` map
standardised units to the target's original units:

* :class:`~nowcastbox.models.MixedFreqDFMResults`: :math:`g = Z_{y}` (row of the design
  matrix for the target), :math:`h = d_y`, :math:`o, s` = mean and standard deviation of
  the target (:math:`\hat y` = ``E[y_t | data]``, the ``out_of_sample`` column);
* :class:`~nowcastbox.models.two_step.TwoStepResults` with ``aggregate="factors"``:
  :math:`g = S\beta` with :math:`S` the factor-aggregation selector, :math:`o` the bridge
  intercept and :math:`s = 1`;
* :class:`~nowcastbox.models.two_step.TwoStepResults` with ``aggregate="variables"``: the
  state-space model observes the *filtered* predictors
  :math:`\tilde x_{i,t} = \sum_j w_{i,j} x_{i,t-j}`, so every vintage is first passed
  through :meth:`~nowcastbox.models.two_step.TwoStepResults.filter_panel`
  (:attr:`LinearNowcastModel.prefilter`); :math:`g = \beta` (the bridge uses the factors
  at the target slot). A raw release completes one filtered observation and is
  attributed to it; a revision of :math:`x_{i,t-j}` changes the filtered values
  :math:`t, \dots, t+L-1` and is counted with the data revisions.

For calendar aggregations (weekly or daily base grids, innovation I1) the observation
equation is time varying: the model is rebuilt for the length of each grid
(:meth:`~nowcastbox.models.MixedFreqDFMResults.state_space_model`) and :math:`g` is the
target row of :math:`Z_{t_\tau}`.

Given a fixed missing-data pattern the Kalman smoother is linear in the observed
values, which is what the decomposition exploits (Bańbura & Modugno, 2014).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from nowcastbox.core.data import MixedFrequencyData, SeriesCategory, as_mixed_frequency_data
from nowcastbox.core.exceptions import NowcastDataError
from nowcastbox.core.frequency import Frequency, period_to_base
from nowcastbox.core.results import NowcastResults
from nowcastbox.news._linear import batched_functional
from nowcastbox.statespace import SmootherResult, StateSpace, kalman_smoother

__all__ = ["LinearNowcastModel", "linear_model", "position", "to_frame", "to_period"]

FloatArray = NDArray[np.float64]

UNCATEGORIZED = "uncategorized"


def to_period(value: object, freq: Frequency, name: str = "target_period") -> pd.Period:
    """Coerce ``value`` to a :class:`pandas.Period` of frequency ``freq``.

    Parameters
    ----------
    value : period-like
        ``pandas.Period``, string (``"2020Q2"``) or timestamp.
    freq : Frequency
        Target frequency.
    name : str, default "target_period"
        Argument name used in error messages.

    Returns
    -------
    pandas.Period
        Period at frequency ``freq``.

    Raises
    ------
    ValueError
        If ``value`` cannot be parsed.

    Examples
    --------
    >>> from nowcastbox.core.frequency import Frequency, period_to_base
    >>> to_period("2020Q2", Frequency.QUARTERLY)
    Period('2020Q2', 'Q-DEC')
    """
    try:
        if isinstance(value, pd.Period):
            return value.asfreq(freq.pandas_freq, how="E")
        return pd.Period(value, freq=freq.pandas_freq)  # type: ignore[arg-type]
    except (TypeError, ValueError) as err:
        raise ValueError(f"Invalid {name} {value!r}.") from err


def position(grid: pd.PeriodIndex, period: pd.Period) -> int:
    """Integer position of ``period`` on ``grid``.

    Parameters
    ----------
    grid : pandas.PeriodIndex
        Grid.
    period : pandas.Period
        Period on the grid.

    Returns
    -------
    int
        Position (``-1`` when absent).

    Examples
    --------
    >>> import pandas as pd
    >>> grid = pd.period_range("2020-01", periods=3, freq="M")
    >>> position(grid, pd.Period("2020-02", freq="M"))
    1
    """
    return int(grid.get_indexer(pd.PeriodIndex([period], freq=grid.freqstr))[0])


def to_frame(data: object) -> pd.DataFrame:
    """Wide DataFrame on a :class:`pandas.PeriodIndex` from a panel.

    Parameters
    ----------
    data : MixedFrequencyData or pandas.DataFrame
        Panel (a DataFrame without a PeriodIndex is coerced with
        :func:`~nowcastbox.core.data.as_mixed_frequency_data`).

    Returns
    -------
    pandas.DataFrame
        Copy of the values (``NaN`` = missing).

    Raises
    ------
    TypeError
        If ``data`` is not a panel.

    Examples
    --------
    >>> import pandas as pd
    >>> idx = pd.period_range("2020-01", periods=2, freq="M")
    >>> to_frame(pd.DataFrame({"a": [1.0, 2.0]}, index=idx)).shape
    (2, 1)
    """
    if isinstance(data, MixedFrequencyData):
        return data.data
    if isinstance(data, pd.DataFrame):
        if isinstance(data.index, pd.PeriodIndex):
            return data.astype(float)
        return as_mixed_frequency_data(data).data
    raise TypeError(
        f"Expected a MixedFrequencyData or pandas.DataFrame panel, got {type(data).__name__}."
    )


@dataclass(frozen=True, eq=False)
class LinearNowcastModel:
    """Fixed-parameter linear representation of a fitted nowcasting model.

    Parameters
    ----------
    model_name : str
        Name of the model the results come from.
    target : str
        Target series.
    target_frequency : Frequency
        Native frequency of the target.
    base_frequency : Frequency
        Frequency of the base grid.
    series : tuple of str
        Series entering the state-space model (rows of ``Z``), in order.
    mean, std : numpy.ndarray
        Standardisation statistics of ``series``.
    state_space : StateSpace
        Model on standardised data.
    grid_start : pandas.Period
        First period of the estimation grid (the smoother starts there).
    min_grid_end : pandas.Period
        Last period of the estimation grid.
    gain : numpy.ndarray
        Vector ``g`` (length ``n_states``).
    intercept : float
        ``h``.
    offset, scale : float
        Map to original units: ``nowcast = offset + scale * (g' a + h)``.
    min_lag : int
        Number of base periods before the slot the state must cover.
    method : str
        Kalman filter variant.
    categories : dict of str to str
        Category label of each series.
    blocks : dict of str to str
        Block label of each series (``"+"``-joined when several).
    frequencies : dict of str to Frequency
        Native frequency of each series.
    model_builder : callable, optional
        ``n_periods -> StateSpace``: the model covering a grid of ``n_periods`` periods,
        for time-varying observation equations (calendar aggregation). ``None``: the
        time-invariant :attr:`state_space` is used for every grid.
    target_row : int, optional
        Row of the target in ``Z_t``: when given and the model is time varying, the
        gain and intercept are read from the observation equation at the target slot
        (``g = Z_t[row]``, ``h = d_t[row]``) instead of :attr:`gain`/:attr:`intercept`.
    prefilter : callable, optional
        ``DataFrame -> DataFrame`` applied to every panel before standardisation (the
        predictor filters of ``TwoStepDFM(aggregate="variables")``).

    Examples
    --------
    >>> lin = linear_model(res)  # doctest: +SKIP
    >>> lin.series  # doctest: +SKIP
    """

    model_name: str
    target: str
    target_frequency: Frequency
    base_frequency: Frequency
    series: tuple[str, ...]
    mean: FloatArray
    std: FloatArray
    state_space: StateSpace
    grid_start: pd.Period
    min_grid_end: pd.Period
    gain: FloatArray
    intercept: float
    offset: float
    scale: float
    min_lag: int
    method: str
    categories: dict[str, str]
    blocks: dict[str, str]
    frequencies: dict[str, Frequency]
    model_builder: Callable[[int], StateSpace] | None = None
    target_row: int | None = None
    prefilter: Callable[[pd.DataFrame], pd.DataFrame] | None = None
    _models: dict[int, StateSpace] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    # ------------------------------------------------------------------ model
    def model_for(self, n_periods: int) -> StateSpace:
        """State-space model covering a grid of ``n_periods`` periods.

        Parameters
        ----------
        n_periods : int
            Length of the grid (from :attr:`grid_start`).

        Returns
        -------
        StateSpace
            :attr:`state_space`, or the model rebuilt by :attr:`model_builder` for
            time-varying observation equations (cached per length).

        Examples
        --------
        >>> lin.model_for(120).n_states  # doctest: +SKIP
        """
        if self.model_builder is None:
            return self.state_space
        n = int(n_periods)
        if n not in self._models:
            self._models[n] = self.model_builder(n)
        return self._models[n]

    def gain_at(self, model: StateSpace, position: int) -> tuple[FloatArray, float]:
        """Gain ``g`` and intercept ``h`` of the target functional at a grid position.

        Parameters
        ----------
        model : StateSpace
            Model of the grid (see :meth:`model_for`).
        position : int
            Position of the target slot.

        Returns
        -------
        tuple of (numpy.ndarray, float)
            ``(g, h)``.

        Examples
        --------
        >>> g, h = lin.gain_at(lin.model_for(n), pos)  # doctest: +SKIP
        """
        if self.target_row is not None and model.is_time_varying:
            row = int(self.target_row)
            gain = np.asarray(model.design_at(position)[row], dtype=np.float64)
            return gain, float(model.obs_intercept_at(position)[row])
        return self.gain, self.intercept

    # ------------------------------------------------------------------ grid
    def target_slot(self, period: object) -> pd.Period:
        """Base-grid storage slot of a target period.

        Parameters
        ----------
        period : period-like
            Target period at the target's native frequency.

        Returns
        -------
        pandas.Period
            Last base period of ``period``.

        Raises
        ------
        ValueError
            If the slot lies before the start of the grid (plus the aggregation lags).

        Examples
        --------
        >>> lin.target_slot("2020Q2")  # doctest: +SKIP
        Period('2020-06', 'M')
        """
        native = to_period(period, self.target_frequency)
        slot = period_to_base(native, self.base_frequency)
        if (slot - self.grid_start).n < self.min_lag:
            raise ValueError(
                f"target_period {native} is too early for the model grid starting at "
                f"{self.grid_start}."
            )
        return slot

    def grid(self, slot: pd.Period, *frames: pd.DataFrame) -> pd.PeriodIndex:
        """Common base grid covering the estimation grid, ``slot`` and every frame.

        Parameters
        ----------
        slot : pandas.Period
            Target slot.
        *frames : pandas.DataFrame
            Panels on base-frequency PeriodIndexes.

        Returns
        -------
        pandas.PeriodIndex
            Contiguous grid starting at :attr:`grid_start`.

        Raises
        ------
        NowcastDataError
            If a frame is not on a base-frequency PeriodIndex.

        Examples
        --------
        >>> lin.grid(slot, old_frame, new_frame)  # doctest: +SKIP
        """
        freq = self.grid_start.freqstr
        for frame in frames:
            index = frame.index
            if not isinstance(index, pd.PeriodIndex) or index.freqstr != freq:
                raise NowcastDataError(
                    f"The data must be indexed by a {self.base_frequency.label} PeriodIndex "
                    f"({freq})."
                )
        end = max([self.min_grid_end, slot, *(f.index[-1] for f in frames if len(f))])
        return pd.period_range(self.grid_start, end, freq=freq)

    def standardize(self, frame: pd.DataFrame, grid: pd.PeriodIndex) -> FloatArray:
        """Standardised observation matrix of a panel on ``grid``.

        Parameters
        ----------
        frame : pandas.DataFrame
            Panel containing every model series (extra columns are ignored); passed
            through :attr:`prefilter` first when set.
        grid : pandas.PeriodIndex
            Base grid.

        Returns
        -------
        numpy.ndarray, shape (len(grid), n_series)
            ``(x - mean) / std`` with ``NaN`` for missing values.

        Raises
        ------
        NowcastDataError
            If series are missing.

        Examples
        --------
        >>> lin.standardize(frame, grid).shape  # doctest: +SKIP
        """
        missing = [s for s in self.series if s not in frame.columns]
        if missing:
            raise NowcastDataError(f"The data do not contain the model series {missing}.")
        if self.prefilter is not None:
            frame = self.prefilter(frame)
        values = frame.loc[:, list(self.series)].reindex(grid).to_numpy(dtype=np.float64)
        return (values - self.mean) / self.std

    # ------------------------------------------------------------------ evaluation
    def smooth(self, values: FloatArray) -> SmootherResult:
        """Kalman smoother on standardised observations.

        Parameters
        ----------
        values : numpy.ndarray, shape (n_periods, n_series)
            Standardised data (``NaN`` = missing).

        Returns
        -------
        SmootherResult
            Smoother output.

        Examples
        --------
        >>> lin.smooth(values).smoothed_state.shape  # doctest: +SKIP
        """
        model = self.model_for(values.shape[0])
        return kalman_smoother(model, values, method=self.method)  # type: ignore[arg-type]

    def functional(self, smoothed: SmootherResult, position: int) -> float:
        """``g' E[alpha | data] + h`` at a grid position (standardised units).

        Parameters
        ----------
        smoothed : SmootherResult
            Smoother output.
        position : int
            Position of the target slot on the grid.

        Returns
        -------
        float
            Standardised nowcast.

        Examples
        --------
        >>> lin.functional(lin.smooth(values), pos)  # doctest: +SKIP
        """
        model = self.model_for(smoothed.smoothed_state.shape[0])
        gain, intercept = self.gain_at(model, position)
        return float(gain @ smoothed.smoothed_state[position]) + intercept

    def evaluate(self, values: FloatArray, position: int) -> float:
        """Standardised nowcast ``g' E[alpha | data] + h`` for a data matrix.

        Parameters
        ----------
        values : numpy.ndarray
            Standardised data on the grid.
        position : int
            Position of the target slot.

        Returns
        -------
        float
            Standardised nowcast.

        Examples
        --------
        >>> lin.evaluate(values, pos)  # doctest: +SKIP
        """
        return self.functional(self.smooth(values), position)

    def evaluate_many(
        self, pattern: NDArray[np.bool_], data: FloatArray, position: int
    ) -> FloatArray:
        """Standardised nowcasts for many data sets sharing one observation pattern.

        Parameters
        ----------
        pattern : numpy.ndarray of bool, shape (n_periods, n_series)
            Observed cells.
        data : numpy.ndarray, shape (n_periods, n_series, K)
            Data sets (only cells in ``pattern`` are read).
        position : int
            Position of the target slot.

        Returns
        -------
        numpy.ndarray, shape (K,)
            ``g' E[alpha | data_k] + h`` (see
            :func:`~nowcastbox.news._linear.batched_functional`).

        Examples
        --------
        >>> lin.evaluate_many(pattern, data, pos)  # doctest: +SKIP
        """
        model = self.model_for(pattern.shape[0])
        gain, intercept = self.gain_at(model, position)
        return batched_functional(model, gain, intercept, pattern, data, position)

    def to_original(self, value: float) -> float:
        """Map a standardised nowcast to the target's original units.

        Parameters
        ----------
        value : float
            Standardised value.

        Returns
        -------
        float
            ``offset + scale * value``.

        Examples
        --------
        >>> lin.to_original(0.0)  # doctest: +SKIP
        """
        return self.offset + self.scale * value

    def nowcast(self, frame: pd.DataFrame, period: object) -> float:
        """Nowcast of a target period given a panel, in original units.

        Parameters
        ----------
        frame : pandas.DataFrame
            Panel.
        period : period-like
            Target period.

        Returns
        -------
        float
            Model estimate of the target.

        Examples
        --------
        >>> lin.nowcast(frame, "2020Q2")  # doctest: +SKIP
        """
        slot = self.target_slot(period)
        grid = self.grid(slot, frame)
        values = self.standardize(frame, grid)
        return self.to_original(self.evaluate(values, position(grid, slot)))

    def same_parameters(self, other: LinearNowcastModel) -> bool:
        """Whether two linear models are numerically identical.

        Parameters
        ----------
        other : LinearNowcastModel
            Other model.

        Returns
        -------
        bool
            True when series, statistics, system matrices and target map coincide.

        Examples
        --------
        >>> lin.same_parameters(lin)  # doctest: +SKIP
        True
        """
        if self.series != other.series or self.grid_start != other.grid_start:
            return False
        mine, theirs = self.state_space.to_dict(), other.state_space.to_dict()
        arrays = [(self.mean, other.mean), (self.std, other.std), (self.gain, other.gain)]
        arrays += [(mine[k], theirs[k]) for k in mine]
        same_shape = all(a.shape == b.shape for a, b in arrays)
        scalars = (self.offset, self.scale, self.intercept) == (
            other.offset,
            other.scale,
            other.intercept,
        )
        return same_shape and scalars and all(np.array_equal(a, b) for a, b in arrays)


# ====================================================================== builders
def _category_label(value: object) -> str:
    if value is None:
        return UNCATEGORIZED
    if isinstance(value, SeriesCategory):
        return value.value
    return str(value)


def _metadata_labels(
    results: NowcastResults, series: tuple[str, ...], base: Frequency
) -> tuple[dict[str, str], dict[str, str], dict[str, Frequency]]:
    """Category, (metadata) block and frequency of ``series`` from the stored panel."""
    categories = dict.fromkeys(series, UNCATEGORIZED)
    blocks = dict.fromkeys(series, "all")
    frequencies = dict.fromkeys(series, base)
    data = results.data
    if data is None:
        return categories, blocks, frequencies
    for name in series:
        meta = data.metadata.get(name)
        if meta is None:
            continue
        categories[name] = _category_label(meta.category)
        frequencies[name] = meta.frequency
        if meta.blocks:
            blocks[name] = "+".join(meta.blocks)
    return categories, blocks, frequencies


def _from_em(results: Any) -> LinearNowcastModel:
    model, layout, grid = results.state_space_model()
    stats = results.standardization
    series = tuple(layout.series)
    target = results.target
    j = series.index(target)
    base = Frequency.from_index(grid)
    categories, _, frequencies = _metadata_labels(results, series, base)
    blocks = {
        name: "+".join(
            b for b, member in zip(layout.block_names, layout.membership[i], strict=True) if member
        )
        for i, name in enumerate(series)
    }
    mean = stats.mean.reindex(list(series)).to_numpy(dtype=np.float64)
    std = stats.std.reindex(list(series)).to_numpy(dtype=np.float64)
    builder: Callable[[int], StateSpace] | None = None
    if model.is_time_varying:  # calendar aggregation: Z_t depends on the period

        def _build(n_periods: int) -> StateSpace:
            return results.state_space_model(n_periods)[0]

        builder = _build

        gain = np.asarray(model.design_at(len(grid) - 1)[j], dtype=np.float64).copy()
        intercept = float(model.obs_intercept_at(len(grid) - 1)[j])
    else:
        gain = np.asarray(model.Z[j], dtype=np.float64).copy()
        intercept = float(model.d[j])
    return LinearNowcastModel(
        model_name=results.model_name,
        target=target,
        target_frequency=results.target_frequency,
        base_frequency=base,
        series=series,
        mean=mean,
        std=std,
        state_space=model,
        grid_start=grid[0],
        min_grid_end=grid[-1],
        gain=gain,
        intercept=intercept,
        offset=float(mean[j]),
        scale=float(std[j]),
        min_lag=0,
        method=str(results.filter_method),
        categories=categories,
        blocks=blocks,
        frequencies=frequencies,
        model_builder=builder,
        target_row=j,
    )


def _two_step_prefilter(results: Any) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Predictor filters of ``TwoStepDFM(aggregate="variables")`` as a frame map."""
    model_data = results.model_data
    frequencies = {name: model_data.metadata[name].frequency for name in model_data.columns}

    def prefilter(frame: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in model_data.columns if c not in frame.columns]
        if missing:
            raise NowcastDataError(f"The data do not contain the model series {missing}.")
        sub = frame.loc[:, list(model_data.columns)]
        return results.filter_panel(sub, frequency=frequencies).data

    return prefilter


def _from_two_step(results: Any) -> LinearNowcastModel:
    prefilter = None
    if results.aggregate != "factors":
        if not results.is_filtered or results.model_data is None:
            raise NotImplementedError(
                "News for TwoStepDFM(aggregate='variables') need results that store the "
                "filtered model panel (model_data); refit with the current version."
            )
        prefilter = _two_step_prefilter(results)
    model = results.state_space
    loadings = results.loadings
    factors = results.factors
    stats = results.standardization
    bridge = results.bridge
    weights = results.aggregation_weights
    if any(x is None for x in (model, loadings, factors, stats, bridge, weights)):
        raise ValueError("These TwoStepDFM results do not contain the state-space model.")
    series = tuple(str(s) for s in loadings.index)
    r = int(loadings.shape[1])
    beta = bridge.params.to_numpy(dtype=np.float64)
    const = float(beta[0]) if bridge.add_constant else 0.0
    slopes = beta[1:] if bridge.add_constant else beta
    gain = np.zeros(model.n_states)
    for lag, w in enumerate(np.asarray(weights, dtype=np.float64)):
        gain[lag * r : (lag + 1) * r] += w * slopes
    grid = factors.index
    base = Frequency.from_index(grid)
    categories, blocks, frequencies = _metadata_labels(results, series, base)
    params = results.model_params
    return LinearNowcastModel(
        model_name=results.model_name,
        target=results.target,
        target_frequency=results.target_frequency,
        base_frequency=base,
        series=series,
        mean=stats.mean.reindex(list(series)).to_numpy(dtype=np.float64),
        std=stats.std.reindex(list(series)).to_numpy(dtype=np.float64),
        state_space=model,
        grid_start=grid[0],
        min_grid_end=grid[-1],
        gain=gain,
        intercept=0.0,
        offset=const,
        scale=1.0,
        min_lag=int(np.asarray(weights).size) - 1,
        method=str(params.get("filter_method", "auto")),
        categories=categories,
        blocks=blocks,
        frequencies=frequencies,
        prefilter=prefilter,
    )


def linear_model(
    results: NowcastResults, categories: Mapping[str, object] | None = None
) -> LinearNowcastModel:
    """Linear (fixed-parameter) view of fitted results.

    Parameters
    ----------
    results : NowcastResults
        Results of :class:`~nowcastbox.models.MixedFreqDFM` (any base grid, including
        weekly/daily calendar aggregation) or of :class:`~nowcastbox.models.TwoStepDFM`
        (``aggregate="factors"`` or ``"variables"``), or any results with a
        ``linear_nowcast_model()`` method returning a :class:`LinearNowcastModel`
        (:class:`~nowcastbox.models.LargeBVARResults`, whose grid is the blocked
        quarterly one).
    categories : mapping of str to category, optional
        Override of the category label of some series (default: the
        ``SeriesMetadata.category`` of the estimation panel).

    Returns
    -------
    LinearNowcastModel
        Linear representation.

    Raises
    ------
    TypeError
        If the results come from an unsupported model.
    NotImplementedError
        For ``TwoStepDFM(aggregate="variables")`` results without the filtered panel.

    Examples
    --------
    >>> lin = linear_model(res)  # doctest: +SKIP
    """
    from nowcastbox.models.em import MixedFreqDFMResults
    from nowcastbox.models.two_step import TwoStepResults

    hook = getattr(results, "linear_nowcast_model", None)
    if isinstance(results, MixedFreqDFMResults):
        lin = _from_em(results)
    elif isinstance(results, TwoStepResults):
        lin = _from_two_step(results)
    elif callable(hook):  # other linear-Gaussian models (e.g. LargeBVARResults)
        lin = hook()
        if not isinstance(lin, LinearNowcastModel):
            raise TypeError("linear_nowcast_model() must return a LinearNowcastModel.")
    else:
        raise TypeError(
            "News decompositions need MixedFreqDFMResults or TwoStepResults (state-space "
            f"models); got {type(results).__name__}."
        )
    if categories:
        merged = dict(lin.categories)
        merged.update({str(k): _category_label(v) for k, v in categories.items()})
        object.__setattr__(lin, "categories", merged)
    return lin
