"""Tests for the hand-rolled PEG parser core.

This module pins down the minimal combinator set we need before wiring the
turn grammar:

  lit / seq / alt / opt

All state is immutable (frozen dataclasses) per the project coding-style
rule — parsers never mutate ParseState, they return a new one. The
plan-of-record is ~/.claude/plans/jazzy-wishing-starfish.md.
"""

from __future__ import annotations

from agent_grammar.parser import (
    ParseFailure,
    ParseState,
    ParseSuccess,
    alt,
    lit,
    opt,
    seq,
)


def _run(parser, text: str):
    return parser(ParseState(text=text, pos=0))


def test_parse_state_is_immutable() -> None:
    state = ParseState(text="abc", pos=0)
    advanced = state.advance(2)
    assert state.pos == 0
    assert advanced.pos == 2
    assert advanced.text == "abc"


def test_lit_matches_prefix() -> None:
    result = _run(lit("hello"), "hello world")
    assert isinstance(result, ParseSuccess)
    assert result.value == "hello"
    assert result.state.pos == 5


def test_lit_fails_when_no_match() -> None:
    result = _run(lit("hi"), "hello")
    assert isinstance(result, ParseFailure)
    assert result.pos == 0
    assert "hi" in result.message


def test_lit_matches_at_nonzero_position() -> None:
    state = ParseState(text="hi there", pos=3)
    result = lit("there")(state)
    assert isinstance(result, ParseSuccess)
    assert result.value == "there"
    assert result.state.pos == 8


def test_seq_concatenates_successes() -> None:
    parser = seq(lit("foo"), lit("bar"))
    result = _run(parser, "foobar")
    assert isinstance(result, ParseSuccess)
    assert result.value == ("foo", "bar")
    assert result.state.pos == 6


def test_seq_fails_if_any_member_fails() -> None:
    parser = seq(lit("foo"), lit("bar"))
    result = _run(parser, "fooBAZ")
    assert isinstance(result, ParseFailure)
    assert result.pos == 3


def test_seq_does_not_consume_input_on_partial_failure() -> None:
    """PEG: failed seq must report failure but not advance the caller's state."""
    parser = seq(lit("foo"), lit("bar"))
    state = ParseState(text="fooBAZ", pos=0)
    result = parser(state)
    assert isinstance(result, ParseFailure)
    # The original state is untouched (immutability) — the caller still
    # holds pos=0 and can retry an alternative.
    assert state.pos == 0


def test_alt_picks_first_success() -> None:
    parser = alt(lit("foo"), lit("bar"))
    result = _run(parser, "bar")
    assert isinstance(result, ParseSuccess)
    assert result.value == "bar"


def test_alt_picks_first_when_multiple_match() -> None:
    """PEG semantics: ordered choice, not longest match."""
    parser = alt(lit("foo"), lit("foobar"))
    result = _run(parser, "foobar")
    assert isinstance(result, ParseSuccess)
    assert result.value == "foo"
    assert result.state.pos == 3


def test_alt_fails_when_all_alternatives_fail() -> None:
    parser = alt(lit("foo"), lit("bar"))
    result = _run(parser, "baz")
    assert isinstance(result, ParseFailure)


def test_opt_returns_none_on_failure() -> None:
    parser = opt(lit("x"))
    result = _run(parser, "abc")
    assert isinstance(result, ParseSuccess)
    assert result.value is None
    assert result.state.pos == 0


def test_opt_returns_value_on_success() -> None:
    parser = opt(lit("a"))
    result = _run(parser, "abc")
    assert isinstance(result, ParseSuccess)
    assert result.value == "a"
    assert result.state.pos == 1


def test_combinators_compose() -> None:
    """opt + seq + alt should compose into a useful grammar fragment."""
    # Match "hi" or "hello" optionally followed by " world"
    greeting = alt(lit("hello"), lit("hi"))
    parser = seq(greeting, opt(lit(" world")))

    result = _run(parser, "hello world")
    assert isinstance(result, ParseSuccess)
    assert result.value == ("hello", " world")
    assert result.state.pos == 11

    result2 = _run(parser, "hi")
    assert isinstance(result2, ParseSuccess)
    assert result2.value == ("hi", None)
    assert result2.state.pos == 2
