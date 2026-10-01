"""Convenience utilities (plan §2.1: ``nb.utils.month_to_quarter``).

Thin re-exports of frequently used helpers that live in more specific subpackages.

Examples
--------
>>> import pandas as pd
>>> import nowcastbox as nb
>>> q = pd.Series([1.0, 2.0], index=pd.period_range("2020Q1", periods=2, freq="Q"))
>>> nb.utils.quarter_to_month(q, "end").dropna().tolist()
[1.0, 2.0]
"""

from __future__ import annotations

from nowcastbox.preprocessing.aggregation import month_to_quarter, quarter_to_month

__all__ = ["month_to_quarter", "quarter_to_month"]
