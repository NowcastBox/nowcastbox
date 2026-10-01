"""Exceptions and warnings raised by nowcastbox.

Every error raised deliberately by the library derives from :class:`NowcastBoxError`
and every warning from :class:`NowcastBoxWarning`, so users can catch or filter all
library-specific conditions at once. Errors about invalid *data* also derive from
:class:`ValueError` and the not-fitted error from :class:`AttributeError` (as in
scikit-learn), so generic handlers keep working.

Hierarchy
---------
::

    NowcastBoxError (Exception)
    ├── NowcastDataError (NowcastBoxError, ValueError)
    ├── ModelNotFittedError (NowcastBoxError, AttributeError)
    └── FormulaError (NowcastBoxError, ValueError)

    NowcastBoxWarning (UserWarning)
    ├── ConvergenceWarning
    └── DataQualityWarning
"""

from __future__ import annotations

__all__ = [
    "ConvergenceWarning",
    "DataQualityWarning",
    "FormulaError",
    "ModelNotFittedError",
    "NowcastBoxError",
    "NowcastBoxWarning",
    "NowcastDataError",
]


class NowcastBoxError(Exception):
    """Base class for all errors raised by nowcastbox.

    Examples
    --------
    >>> from nowcastbox.core.exceptions import NowcastBoxError, NowcastDataError
    >>> issubclass(NowcastDataError, NowcastBoxError)
    True
    """


class NowcastDataError(NowcastBoxError, ValueError):
    """Invalid, inconsistent or insufficient input data.

    Raised e.g. when a quarterly series has values outside the third month of the
    quarter, when the index is not a valid period grid, or when metadata refer to
    unknown series.

    Examples
    --------
    >>> from nowcastbox.core.exceptions import NowcastDataError
    >>> try:
    ...     raise NowcastDataError("duplicated periods in index")
    ... except ValueError as err:
    ...     print(err)
    duplicated periods in index
    """


class ModelNotFittedError(NowcastBoxError, AttributeError):
    """A method requiring a fitted estimator was called before ``fit``.

    Examples
    --------
    >>> from nowcastbox.core.exceptions import ModelNotFittedError
    >>> issubclass(ModelNotFittedError, AttributeError)
    True
    """


class FormulaError(NowcastBoxError, ValueError):
    """Malformed model formula such as ``"y ~ x1 +"``.

    Examples
    --------
    >>> from nowcastbox.core.exceptions import FormulaError
    >>> issubclass(FormulaError, ValueError)
    True
    """


class NowcastBoxWarning(UserWarning):
    """Base class for all warnings issued by nowcastbox.

    Examples
    --------
    >>> import warnings
    >>> from nowcastbox.core.exceptions import NowcastBoxWarning
    >>> warnings.simplefilter("ignore", NowcastBoxWarning)
    """


class ConvergenceWarning(NowcastBoxWarning):
    """An iterative algorithm (EM, optimiser) stopped before converging.

    Examples
    --------
    >>> import warnings
    >>> from nowcastbox.core.exceptions import ConvergenceWarning
    >>> with warnings.catch_warnings(record=True) as w:
    ...     warnings.simplefilter("always")
    ...     warnings.warn("EM did not converge in 500 iterations", ConvergenceWarning)
    >>> w[0].category.__name__
    'ConvergenceWarning'
    """


class DataQualityWarning(NowcastBoxWarning):
    """Suspicious but usable data (all-NaN series, very short samples, outliers...).

    Examples
    --------
    >>> from nowcastbox.core.exceptions import DataQualityWarning, NowcastBoxWarning
    >>> issubclass(DataQualityWarning, NowcastBoxWarning)
    True
    """
