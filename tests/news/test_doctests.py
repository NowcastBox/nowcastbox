"""Run the doctests of the news modules."""

from __future__ import annotations

import doctest

import matplotlib
import matplotlib.pyplot as plt
import pytest

import nowcastbox.news as news_pkg
import nowcastbox.news._model as model_mod
import nowcastbox.news.contributions as contrib_mod
import nowcastbox.news.decomposition as decomp_mod
import nowcastbox.news.plotting as plot_mod
import nowcastbox.news.revisions as rev_mod
import nowcastbox.news.tracker as tracker_mod

matplotlib.use("Agg")


@pytest.mark.parametrize(
    "module",
    [news_pkg, model_mod, contrib_mod, decomp_mod, plot_mod, rev_mod, tracker_mod],
    ids=lambda m: m.__name__,
)
def test_doctests(module):
    result = doctest.testmod(module, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)
    plt.close("all")
    assert result.failed == 0
