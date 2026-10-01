from __future__ import annotations

import warnings

import pytest

from nowcastbox.core import exceptions as ex


@pytest.mark.parametrize(
    ("cls", "bases"),
    [
        (ex.NowcastDataError, (ex.NowcastBoxError, ValueError)),
        (ex.ModelNotFittedError, (ex.NowcastBoxError, AttributeError)),
        (ex.FormulaError, (ex.NowcastBoxError, ValueError)),
        (ex.ConvergenceWarning, (ex.NowcastBoxWarning, UserWarning)),
        (ex.DataQualityWarning, (ex.NowcastBoxWarning, UserWarning)),
    ],
)
def test_hierarchy(cls, bases):
    for base in bases:
        assert issubclass(cls, base)


def test_errors_carry_message():
    with pytest.raises(ValueError, match="bad data"):
        raise ex.NowcastDataError("bad data")


def test_warnings_can_be_filtered_together():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("ignore", ex.NowcastBoxWarning)
        warnings.warn("x", ex.ConvergenceWarning, stacklevel=1)
        warnings.warn("y", ex.DataQualityWarning, stacklevel=1)
    assert caught == []
