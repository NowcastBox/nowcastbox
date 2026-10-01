"""Minimal model formulas: ``"y ~ ."``, ``"y ~ x1 + x2"``, ``"y ~ . - x3"``.

Formulas only *select variables*: the left-hand side names the target and the
right-hand side the series used as predictors. There are no interactions,
transformations or intercept terms (transformations belong to
``nowcastbox.preprocessing``). Names that are not plain identifiers can be quoted with
backticks: ``"y ~ `ipca 12m` + x2"``.

Grammar
-------
::

    formula := name "~" rhs
    rhs     := ["-"] term (("+" | "-") term)*
    term    := "." | name
    name    := identifier-like token | `backtick quoted`

``.`` means "every column except the target"; ``- name`` removes a series.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from nowcastbox.core.exceptions import FormulaError

__all__ = ["ParsedFormula", "is_formula", "parse_formula", "resolve_target"]

_TOKEN = re.compile(r"\s*(?:(`[^`]+`)|([+\-~])|(\.)(?![\w.])|([^\s+\-~`]+))")


@dataclass(frozen=True)
class ParsedFormula:
    """Result of :func:`parse_formula`.

    Parameters
    ----------
    target : str
        Target series (left-hand side).
    regressors : tuple of str or None
        Explicit predictors, in formula order. ``None`` means "all columns except the
        target and ``excluded``" and is only returned when no columns were given to
        :func:`parse_formula`.
    excluded : tuple of str, default ()
        Series removed with ``- name``.
    uses_dot : bool, default False
        Whether the formula contains ``.``.

    Examples
    --------
    >>> from nowcastbox.core.formula import parse_formula
    >>> parse_formula("y ~ x1 + x2")
    ParsedFormula(target='y', regressors=('x1', 'x2'), excluded=(), uses_dot=False)
    """

    target: str
    regressors: tuple[str, ...] | None
    excluded: tuple[str, ...] = ()
    uses_dot: bool = False

    def resolve(self, columns: Sequence[str]) -> tuple[str, ...]:
        """Return the explicit list of predictors given the available columns.

        Parameters
        ----------
        columns : sequence of str
            Available series.

        Returns
        -------
        tuple of str
            Predictors, in column order for ``.`` and formula order otherwise.

        Raises
        ------
        FormulaError
            If a referenced series is not in ``columns`` or no predictor remains.

        Examples
        --------
        >>> parse_formula("y ~ . - b").resolve(["y", "a", "b", "c"])
        ('a', 'c')
        """
        cols = list(columns)
        referenced = [self.target, *self.excluded, *(self.regressors or ())]
        unknown = sorted(set(referenced) - set(cols))
        if unknown:
            raise FormulaError(f"Unknown series in formula: {unknown}.")
        dropped = {self.target, *self.excluded}
        out = tuple(c for c in cols if c not in dropped) if self.uses_dot else self.regressors or ()
        if not out:
            raise FormulaError("The formula leaves no predictor.")
        return out


def is_formula(text: str) -> bool:
    """Return True if ``text`` looks like a formula (contains ``~``).

    Parameters
    ----------
    text : str
        Candidate target name or formula.

    Returns
    -------
    bool
        Whether ``text`` is a string containing a tilde (``False`` for non-strings).

    Examples
    --------
    >>> is_formula("gdp ~ ."), is_formula("gdp")
    (True, False)
    """
    return isinstance(text, str) and "~" in text


def _tokenize(formula: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    pos = 0
    text = formula.rstrip()
    while pos < len(text):
        match = _TOKEN.match(text, pos)
        if match is None or match.end() == pos:
            raise FormulaError(f"Cannot parse formula {formula!r} at position {pos}.")
        quoted, op, dot, name = match.groups()
        if quoted is not None:
            tokens.append(("name", quoted[1:-1].strip()))
        elif op is not None:
            tokens.append(("op", op))
        elif dot is not None:
            tokens.append(("dot", "."))
        else:
            tokens.append(("name", name))
        pos = match.end()
    return tokens


def _rhs_terms(tokens: list[tuple[str, str]], formula: str) -> list[tuple[str, str, str]]:
    """Group right-hand-side tokens into ``(sign, kind, value)`` terms."""
    if not tokens:
        raise FormulaError(f"Empty right-hand side in formula {formula!r}.")
    if tokens[0] == ("op", "-"):
        signs, rest = ["-"], tokens[1:]
    else:
        signs, rest = ["+"], tokens
    terms: list[tuple[str, str, str]] = []
    for position, (kind, value) in enumerate(rest):
        expect_term = position % 2 == 0
        if expect_term and kind == "op":
            raise FormulaError(f"Unexpected {value!r} in formula {formula!r}.")
        if expect_term:
            terms.append((signs[-1], kind, value))
        elif kind != "op" or value == "~":
            raise FormulaError(f"Missing '+' or '-' before {value!r} in {formula!r}.")
        else:
            signs.append(value)
    if len(signs) > len(terms):
        raise FormulaError(f"Formula {formula!r} ends with an operator.")
    return terms


def _parse_rhs(tokens: list[tuple[str, str]], formula: str) -> tuple[list[str], list[str], bool]:
    include: list[str] = []
    exclude: list[str] = []
    uses_dot = False
    for sign, kind, value in _rhs_terms(tokens, formula):
        if kind == "dot" and sign == "-":
            raise FormulaError("'- .' is not allowed.")
        if kind == "dot":
            uses_dot = True
        else:
            (include if sign == "+" else exclude).append(value)
    return include, exclude, uses_dot


def _split_sides(tokens: list[tuple[str, str]], formula: str) -> tuple[str, list[tuple[str, str]]]:
    tildes = [i for i, token in enumerate(tokens) if token == ("op", "~")]
    if len(tildes) != 1:
        raise FormulaError(f"A formula needs exactly one '~', got {formula!r}.")
    lhs, rhs = tokens[: tildes[0]], tokens[tildes[0] + 1 :]
    if len(lhs) != 1 or lhs[0][0] != "name":
        raise FormulaError(f"The left-hand side must be a single series name in {formula!r}.")
    return lhs[0][1], rhs


def parse_formula(formula: str, columns: Sequence[str] | None = None) -> ParsedFormula:
    """Parse a variable-selection formula.

    Parameters
    ----------
    formula : str
        Formula such as ``"y ~ ."``, ``"y ~ x1 + x2"`` or ``"y ~ . - x3"``.
    columns : sequence of str, optional
        Available series. When given, names are validated and ``.`` is expanded, so
        ``regressors`` is always a tuple.

    Returns
    -------
    ParsedFormula
        Target and predictors.

    Raises
    ------
    FormulaError
        If the formula is malformed, the target appears on the right-hand side, a
        name is repeated, or (with ``columns``) a series is unknown.

    Examples
    --------
    >>> from nowcastbox.core.formula import parse_formula
    >>> parse_formula("gdp ~ .", columns=["ip", "gdp", "pmi"]).regressors
    ('ip', 'pmi')
    >>> parse_formula("gdp ~ `retail sales` + ip").regressors
    ('retail sales', 'ip')
    """
    if not isinstance(formula, str):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise FormulaError(f"A formula must be a string, got {type(formula).__name__}.")
    target, rhs = _split_sides(_tokenize(formula), formula)
    include, exclude, uses_dot = _parse_rhs(rhs, formula)
    names = include + exclude
    if target in names:
        raise FormulaError(f"The target {target!r} cannot appear on the right-hand side.")
    if len(set(names)) != len(names):
        raise FormulaError(f"Repeated series in formula {formula!r}.")
    parsed = ParsedFormula(
        target=target,
        regressors=tuple(include) if (include or not uses_dot) else None,
        excluded=tuple(exclude),
        uses_dot=uses_dot,
    )
    if columns is None:
        return parsed
    return ParsedFormula(
        target=target,
        regressors=parsed.resolve(columns),
        excluded=tuple(exclude),
        uses_dot=uses_dot,
    )


def resolve_target(target: str, columns: Sequence[str]) -> tuple[str, tuple[str, ...]]:
    """Resolve a target name *or* formula against the available columns.

    Parameters
    ----------
    target : str
        Either a series name (all other columns are predictors) or a formula.
    columns : sequence of str
        Available series.

    Returns
    -------
    target : str
        Target series name.
    regressors : tuple of str
        Predictor series.

    Raises
    ------
    FormulaError
        If ``target`` is not a string, the formula is invalid or the target is not in
        ``columns``.

    Examples
    --------
    >>> from nowcastbox.core.formula import resolve_target
    >>> resolve_target("gdp", ["ip", "gdp"])
    ('gdp', ('ip',))
    >>> resolve_target("gdp ~ ip", ["ip", "pmi", "gdp"])
    ('gdp', ('ip',))
    """
    if not isinstance(target, str):
        raise FormulaError(
            f"target must be a series name or a formula string, got {type(target).__name__}."
        )
    if is_formula(target):
        parsed = parse_formula(target, columns)
        assert parsed.regressors is not None  # noqa: S101 - resolved with columns
        return parsed.target, parsed.regressors
    if target not in columns:
        raise FormulaError(f"Target {target!r} is not a column of the data.")
    return target, tuple(c for c in columns if c != target)
