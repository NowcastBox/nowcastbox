from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from nowcastbox.core.exceptions import FormulaError
from nowcastbox.core.formula import ParsedFormula, is_formula, parse_formula, resolve_target

COLUMNS = ["ip", "gdp", "pmi", "retail sales"]


class TestParse:
    def test_dot(self):
        p = parse_formula("gdp ~ .")
        assert p == ParsedFormula(target="gdp", regressors=None, excluded=(), uses_dot=True)
        assert p.resolve(COLUMNS) == ("ip", "pmi", "retail sales")

    def test_explicit(self):
        p = parse_formula("gdp~ip+pmi")
        assert p.regressors == ("ip", "pmi")
        assert not p.uses_dot

    def test_backticks(self):
        p = parse_formula("gdp ~ `retail sales` + ip", columns=COLUMNS)
        assert p.regressors == ("retail sales", "ip")

    def test_dot_minus(self):
        assert parse_formula("gdp ~ . - pmi", COLUMNS).regressors == ("ip", "retail sales")
        assert parse_formula("gdp ~ . - pmi - ip", COLUMNS).regressors == ("retail sales",)

    def test_leading_minus_with_dot(self):
        p = parse_formula("gdp ~ -pmi + .", COLUMNS)
        assert p.regressors == ("ip", "retail sales")
        assert p.excluded == ("pmi",)

    def test_dot_with_explicit_terms(self):
        assert parse_formula("gdp ~ ip + .", COLUMNS).regressors == ("ip", "pmi", "retail sales")

    def test_explicit_with_exclusion_only(self):
        p = parse_formula("gdp ~ ip - pmi", COLUMNS)
        assert p.regressors == ("ip",)

    def test_names_with_dots_and_digits(self):
        p = parse_formula("y.q ~ x.1 + x_2")
        assert p.target == "y.q"
        assert p.regressors == ("x.1", "x_2")

    @pytest.mark.parametrize(
        ("formula", "match"),
        [
            ("gdp", "exactly one"),
            ("gdp ~ ip ~ pmi", "exactly one"),
            ("~ ip", "single series"),
            ("gdp pmi ~ ip", "single series"),
            ("gdp ~", "Empty right-hand"),
            ("gdp ~ ip +", "ends with an operator"),
            ("gdp ~ ip pmi", "Missing"),
            ("gdp ~ + ip", "Unexpected"),
            ("gdp ~ ip + + pmi", "Unexpected"),
            ("gdp ~ - .", "not allowed"),
            ("gdp ~ gdp", "cannot appear"),
            ("gdp ~ ip + ip", "Repeated"),
            ("gdp ~ `unterminated", "Cannot parse"),
        ],
    )
    def test_malformed(self, formula, match):
        with pytest.raises(FormulaError, match=match):
            parse_formula(formula)

    def test_not_a_string(self):
        with pytest.raises(FormulaError, match="string"):
            parse_formula(3)  # type: ignore[arg-type]

    def test_unknown_columns(self):
        with pytest.raises(FormulaError, match="Unknown series"):
            parse_formula("gdp ~ zzz", COLUMNS)
        with pytest.raises(FormulaError, match="Unknown series"):
            parse_formula("zzz ~ .", COLUMNS)

    def test_no_predictor_left(self):
        with pytest.raises(FormulaError, match="no predictor"):
            parse_formula("gdp ~ .", ["gdp"])
        with pytest.raises(FormulaError, match="no predictor"):
            parse_formula("gdp ~ -ip", COLUMNS)


def test_is_formula():
    assert is_formula("a ~ b")
    assert not is_formula("a")


class TestResolveTarget:
    def test_name(self):
        assert resolve_target("gdp", COLUMNS) == ("gdp", ("ip", "pmi", "retail sales"))

    def test_formula(self):
        assert resolve_target("gdp ~ pmi", COLUMNS) == ("gdp", ("pmi",))

    def test_unknown_target(self):
        with pytest.raises(FormulaError, match="not a column"):
            resolve_target("zzz", COLUMNS)

    def test_univariate(self):
        assert resolve_target("gdp", ["gdp"]) == ("gdp", ())


_names = st.from_regex(r"[a-z][a-z0-9_]{0,6}", fullmatch=True)


@given(st.lists(_names, min_size=2, max_size=8, unique=True))
def test_roundtrip_property(names):
    target, *regs = names
    formula = f"{target} ~ " + " + ".join(regs)
    parsed = parse_formula(formula, names)
    assert parsed.target == target
    assert parsed.regressors == tuple(regs)
    dot = parse_formula(f"{target} ~ .", list(reversed(names)))
    assert set(dot.regressors or ()) == set(regs)
