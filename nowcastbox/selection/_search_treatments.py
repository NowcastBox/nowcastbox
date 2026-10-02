"""Treatments of the Covid-19 observations compared by the robustness step of the search.

The ECB toolbox (Linzenich & Meunier, 2024, §2.3) re-evaluates the best
specifications with several treatments of the pandemic observations and keeps the most
accurate one after the pandemic. Each treatment here maps to an option that already
exists in the library:

=============  ==========================================================================
``"none"``     no treatment (``covid="none"`` and ``outliers="none"`` when the model has
               those options)
``"dummy"``    impulse dummies in the factor VAR for every period of the pandemic
               window (``covid="dummy"`` of :class:`~nowcastbox.models.MixedFreqDFM`)
``"mask"``     pandemic observations treated as missing (``covid="mask"``)
``"outliers"`` automatic outlier handling of the model (``outliers="auto"``) when it has
               that option; otherwise the IQR correction of
               :func:`~nowcastbox.preprocessing.replace_outliers` applied to the predictors
               of every vintage (never to the target), so there is no look-ahead
=============  ==========================================================================

A treatment is a function ``(model, target) -> TreatmentPlan``; a plan that is not
``supported`` makes the robustness step skip that (specification, treatment) pair with
the plan's note. Custom treatments can be passed to
:meth:`~nowcastbox.selection.SearchResults.covid_robustness` as a mapping.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from nowcastbox.core.data import MixedFrequencyData
from nowcastbox.preprocessing.outliers import replace_outliers

__all__ = [
    "COVID_TREATMENTS",
    "OutlierCorrection",
    "Treatment",
    "TreatmentPlan",
    "resolve_treatments",
]

_COVID_KEYS = ("covid", "outliers")


@dataclass(frozen=True)
class TreatmentPlan:
    """How a treatment changes a model: parameters, data preprocessing and a note.

    Parameters
    ----------
    params : mapping, optional
        Model parameters to set (``set_params``).
    preprocess : callable, optional
        Function applied to every vintage's information set before estimation.
    note : str, default ""
        Description shown in the robustness table (or why the treatment is skipped).
    supported : bool, default True
        False when the model cannot apply the treatment.

    Examples
    --------
    >>> TreatmentPlan({"covid": "dummy"}, note="dummies").supported
    True
    >>> TreatmentPlan.unsupported("no covid option").supported
    False
    """

    params: Mapping[str, Any] = field(default_factory=dict)
    preprocess: Callable[[MixedFrequencyData], MixedFrequencyData] | None = None
    note: str = ""
    supported: bool = True

    @classmethod
    def unsupported(cls, note: str) -> TreatmentPlan:
        """A plan that marks the treatment as not applicable to the model.

        Parameters
        ----------
        note : str
            Reason.

        Returns
        -------
        TreatmentPlan
            Plan with ``supported=False``.

        Examples
        --------
        >>> TreatmentPlan.unsupported("why").note
        'why'
        """
        return cls(note=note, supported=False)


Treatment = Callable[[Any, str], TreatmentPlan]
"""A treatment: ``(model, target) -> TreatmentPlan``."""


class OutlierCorrection:
    """Per-vintage IQR outlier correction of the predictors (picklable callable).

    Parameters
    ----------
    exclude : sequence of str, default ()
        Series left untouched (the target).
    threshold : float, default 4.0
        Multiple of the inter-quartile range defining an outlier.
    window : int, default 3
        Centred moving-median window of the replacement (native periods).

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from nowcastbox.core.data import MixedFrequencyData
    >>> idx = pd.period_range("2020-01", periods=8, freq="M")
    >>> x = [0.0, 1.0, -1.0, 0.5, 30.0, -0.5, 0.2, 0.1]
    >>> panel = MixedFrequencyData(pd.DataFrame({"x": x, "y": x}, index=idx))
    >>> out = OutlierCorrection(exclude=("y",))(panel)
    >>> out["x"].tolist()[4], out["y"].tolist()[4]
    (0.0, 30.0)
    """

    def __init__(
        self, exclude: Sequence[str] = (), threshold: float = 4.0, window: int = 3
    ) -> None:
        self.exclude = tuple(exclude)
        self.threshold = float(threshold)
        self.window = int(window)

    def __call__(self, panel: MixedFrequencyData) -> MixedFrequencyData:
        columns = [c for c in panel.columns if c not in self.exclude]
        if not columns:
            return panel
        return replace_outliers(panel, self.threshold, window=self.window, columns=columns)

    def __repr__(self) -> str:
        return (
            f"OutlierCorrection(exclude={self.exclude!r}, threshold={self.threshold!r}, "
            f"window={self.window!r})"
        )


def _model_params(model: Any) -> dict[str, Any]:
    get_params = getattr(model, "get_params", None)
    if not callable(get_params):
        return {}
    params = get_params(deep=False)
    return dict(params) if isinstance(params, Mapping) else {}


def _neutral(params: Mapping[str, Any]) -> dict[str, Any]:
    """Switch off the Covid/outlier options the model has."""
    return {key: "none" for key in _COVID_KEYS if key in params}


def treat_none(model: Any, target: str) -> TreatmentPlan:
    """No treatment of the pandemic observations.

    Parameters
    ----------
    model : estimator
        The configured model.
    target : str
        Target series.

    Returns
    -------
    TreatmentPlan
        ``covid="none"``/``outliers="none"`` when the model has these options.

    Examples
    --------
    >>> from nowcastbox.models import MixedFreqDFM
    >>> treat_none(MixedFreqDFM(covid="dummy"), "gdp").params
    {'covid': 'none', 'outliers': 'none'}
    """
    return TreatmentPlan(_neutral(_model_params(model)), note="no treatment")


def _covid_option(model: Any, mode: str, note: str) -> TreatmentPlan:
    params = _model_params(model)
    if "covid" not in params:
        return TreatmentPlan.unsupported(
            f"{type(model).__name__} has no 'covid' option (covid={mode!r} needs e.g. "
            "MixedFreqDFM); skipped"
        )
    return TreatmentPlan({**_neutral(params), "covid": mode}, note=note)


def treat_dummy(model: Any, target: str) -> TreatmentPlan:
    """Impulse dummies for the pandemic window in the factor VAR (``covid="dummy"``).

    Parameters
    ----------
    model : estimator
        The configured model.
    target : str
        Target series.

    Returns
    -------
    TreatmentPlan
        Supported only when the model has a ``covid`` option.

    Examples
    --------
    >>> from nowcastbox.models import MixedFreqDFM, TwoStepDFM
    >>> treat_dummy(MixedFreqDFM(), "gdp").params["covid"]
    'dummy'
    >>> treat_dummy(TwoStepDFM(), "gdp").supported
    False
    """
    return _covid_option(model, "dummy", "pandemic impulse dummies (covid='dummy')")


def treat_mask(model: Any, target: str) -> TreatmentPlan:
    """Pandemic observations treated as missing (``covid="mask"``).

    Parameters
    ----------
    model : estimator
        The configured model.
    target : str
        Target series.

    Returns
    -------
    TreatmentPlan
        Supported only when the model has a ``covid`` option.

    Examples
    --------
    >>> from nowcastbox.models import MixedFreqDFM
    >>> treat_mask(MixedFreqDFM(), "gdp").params["covid"]
    'mask'
    """
    return _covid_option(model, "mask", "pandemic observations excluded (covid='mask')")


def treat_outliers(model: Any, target: str) -> TreatmentPlan:
    """Outlier correction: the model's ``outliers="auto"`` or a per-vintage IQR rule.

    Parameters
    ----------
    model : estimator
        The configured model.
    target : str
        Target series (never corrected by the IQR rule).

    Returns
    -------
    TreatmentPlan
        ``outliers="auto"`` when the model has that option, otherwise a plan whose
        ``preprocess`` is :class:`OutlierCorrection` of the predictors.

    Examples
    --------
    >>> from nowcastbox.models import MixedFreqDFM, TwoStepDFM
    >>> treat_outliers(MixedFreqDFM(), "gdp").params["outliers"]
    'auto'
    >>> treat_outliers(TwoStepDFM(), "gdp").preprocess
    OutlierCorrection(exclude=('gdp',), threshold=4.0, window=3)
    """
    params = _model_params(model)
    if "outliers" in params:
        return TreatmentPlan(
            {**_neutral(params), "outliers": "auto"},
            note="automatic outlier handling of the model (outliers='auto')",
        )
    return TreatmentPlan(
        _neutral(params),
        preprocess=OutlierCorrection(exclude=(target,)),
        note="IQR outlier correction of the predictors at each vintage",
    )


COVID_TREATMENTS: dict[str, Treatment] = {
    "none": treat_none,
    "dummy": treat_dummy,
    "mask": treat_mask,
    "outliers": treat_outliers,
}
"""Built-in treatments of the robustness step."""


def resolve_treatments(
    treatments: Sequence[str] | Mapping[str, Treatment | str],
) -> dict[str, Treatment]:
    """Treatments by name (built-in names or custom functions).

    Parameters
    ----------
    treatments : sequence of str or mapping
        Names of :data:`COVID_TREATMENTS`, or a mapping ``{label: name or function}``.

    Returns
    -------
    dict
        Label to treatment function, in the given order.

    Raises
    ------
    ValueError
        If no treatment is given or a name is unknown.
    TypeError
        If a mapping value is neither a name nor a callable.

    Examples
    --------
    >>> list(resolve_treatments(("none", "dummy")))
    ['none', 'dummy']
    """
    if isinstance(treatments, str):
        treatments = (treatments,)
    items = treatments.items() if isinstance(treatments, Mapping) else [(t, t) for t in treatments]
    out: dict[str, Treatment] = {}
    for label, value in items:
        if isinstance(value, str):
            if value not in COVID_TREATMENTS:
                raise ValueError(
                    f"Unknown treatment {value!r}; built-in: {sorted(COVID_TREATMENTS)}."
                )
            out[str(label)] = COVID_TREATMENTS[value]
        elif callable(value):
            out[str(label)] = value
        else:
            raise TypeError(f"Treatment {label!r} must be a name or a callable.")
    if not out:
        raise ValueError("Give at least one treatment.")
    return out
