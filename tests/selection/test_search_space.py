"""Tests of the parameter space of the specification search."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nowcastbox.selection import ParameterSpace
from nowcastbox.selection._search_space import canonical, spec_key


class TestParsing:
    def test_list_range_and_fixed(self):
        space = ParameterSpace({"a": [1, 2, 3], "b": (2, 5), "c": "x", "d": {"k": 1}})
        assert space.names == ["a", "b", "c", "d"]
        assert space.values("a") == (1, 2, 3)
        assert space.values("b") == (2, 3, 4, 5)
        assert space.values("c") == ("x",)
        assert space.values("d") == ({"k": 1},)
        assert space.size == 12

    def test_python_range_is_a_list_of_choices(self):
        assert ParameterSpace({"a": range(1, 4)}).values("a") == (1, 2, 3)

    def test_tuple_that_is_not_an_int_pair_is_a_list_of_choices(self):
        space = ParameterSpace({"start": ("2005-01", "2010-01"), "t": (1, 2, 5)})
        assert space.values("start") == ("2005-01", "2010-01")
        assert space.values("t") == (1, 2, 5)
        assert space.endpoints("t") == (1, 2, 5)

    def test_bool_pair_is_not_a_range(self):
        assert ParameterSpace({"flag": (False, True)}).values("flag") == (False, True)

    def test_numpy_int_range(self):
        space = ParameterSpace({"n": (np.int64(1), np.int64(3))})
        assert space.values("n") == (1, 2, 3)
        assert space.endpoints("n") == (1, 3)

    def test_array_choices(self):
        assert ParameterSpace({"a": np.array([0.5, 1.0])}).size == 2

    def test_contains_and_iter(self):
        space = ParameterSpace({"a": [1], "b": [2]})
        assert "a" in space
        assert "z" not in space
        assert list(space) == ["a", "b"]

    def test_repr(self):
        text = repr(ParameterSpace({"a": [1, 2], "b": (1, 3)}))
        assert text == "ParameterSpace(a=[1, 2], b=(1, 3))"

    @pytest.mark.parametrize(
        ("space", "message"),
        [
            ({}, "non-empty mapping"),
            ([("a", 1)], "non-empty mapping"),
            ({"": [1]}, "non-empty strings"),
            ({1: [1]}, "non-empty strings"),
            ({"a": []}, "has no values"),
            ({"a": (5, 2)}, "low > high"),
        ],
    )
    def test_invalid(self, space, message):
        with pytest.raises(ValueError, match=message):
            ParameterSpace(space)

    def test_unknown_setting(self):
        with pytest.raises(KeyError, match="not in the space"):
            ParameterSpace({"a": [1]}).values("b")


class TestGridAndDraws:
    def test_grid_is_exhaustive_and_ordered(self):
        space = ParameterSpace({"a": [1, 2], "b": (1, 3)})
        grid = space.grid()
        assert len(grid) == space.size == 6
        assert grid[0] == {"a": 1, "b": 1}
        assert grid[-1] == {"a": 2, "b": 3}
        assert len({spec_key(s) for s in grid}) == 6

    def test_draws_are_reproducible(self):
        space = ParameterSpace({"a": [1, 2, 3], "b": (10, 99)})
        assert space.draw(20, random_state=7) == space.draw(20, random_state=7)
        assert space.draw(20, random_state=7) != space.draw(20, random_state=8)

    def test_draw_i_does_not_depend_on_n(self):
        space = ParameterSpace({"a": (1, 1000), "b": ["x", "y"]})
        assert space.draw(10, random_state=3)[:4] == space.draw(4, random_state=3)

    def test_seed_sequence_accepted(self):
        space = ParameterSpace({"a": (1, 1000)})
        seed = np.random.SeedSequence(5)
        assert space.draw(3, seed) == space.draw(3, np.random.SeedSequence(5))

    def test_seed_sequence_is_not_mutated(self):
        space = ParameterSpace({"a": (1, 1000)})
        seed = np.random.SeedSequence(5)
        first = space.draw(3, seed)
        assert space.draw(3, seed) == first
        assert seed.n_children_spawned == 0

    def test_spawned_seed_sequence_keeps_its_spawn_key(self):
        space = ParameterSpace({"a": (1, 1000)})
        child_a, child_b = np.random.SeedSequence(5).spawn(2)
        assert space.draw(5, child_a) != space.draw(5, child_b)

    def test_draws_stay_in_the_space(self):
        space = ParameterSpace({"a": [1, 2], "b": (3, 6), "c": "fixed"})
        for spec in space.draw(200, random_state=0):
            assert spec["a"] in (1, 2)
            assert 3 <= spec["b"] <= 6
            assert isinstance(spec["b"], int)
            assert spec["c"] == "fixed"

    def test_uniform_coverage(self):
        draws = ParameterSpace({"b": (1, 4)}).draw(2000, random_state=1)
        counts = pd.Series([d["b"] for d in draws]).value_counts()
        assert set(counts.index) == {1, 2, 3, 4}
        assert counts.min() > 400

    @pytest.mark.parametrize("n", [0, -1, 1.5, True, "3"])
    def test_invalid_number_of_draws(self, n):
        with pytest.raises(ValueError, match="positive integer"):
            ParameterSpace({"a": [1]}).draw(n)


class TestKeys:
    def test_canonical(self):
        assert canonical(np.float64(1.5)) == 1.5
        assert canonical(pd.Timestamp("2020-01-01")) == "2020-01-01 00:00:00"
        assert canonical(np.array([1, 2])) == (1, 2)
        assert canonical({"b": 1, "a": [np.int64(2)]}) == (("a", (2,)), ("b", 1))
        assert canonical("x") == "x"

    def test_spec_key_identifies_equal_specs(self):
        assert spec_key({"a": np.int64(2), "b": [1]}) == spec_key({"a": 2, "b": (1,)})
        assert spec_key({"a": 2}) != spec_key({"a": 3})
