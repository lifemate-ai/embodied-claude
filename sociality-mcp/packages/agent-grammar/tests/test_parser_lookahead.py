"""Tests for lookahead combinators: not_p (!) and and_p (&).

Neither consumes input. Both are zero-width assertions in classical PEG.

  and_p(p): succeeds iff p succeeds at the current position. Does not advance.
  not_p(p): succeeds iff p fails at the current position. Does not advance.
"""

from __future__ import annotations

from agent_grammar.parser import (
    ParseFailure,
    ParseState,
    ParseSuccess,
    and_p,
    lit,
    not_p,
    seq,
)


def _run(parser, text: str):
    return parser(ParseState(text=text, pos=0))


class TestNotP:
    def test_not_p_succeeds_when_inner_fails(self) -> None:
        result = _run(not_p(lit("foo")), "bar")
        assert isinstance(result, ParseSuccess)
        assert result.value is None
        assert result.state.pos == 0

    def test_not_p_fails_when_inner_succeeds(self) -> None:
        result = _run(not_p(lit("foo")), "foobar")
        assert isinstance(result, ParseFailure)
        assert result.pos == 0

    def test_not_p_does_not_consume_input_on_success(self) -> None:
        parser = seq(not_p(lit("foo")), lit("bar"))
        result = _run(parser, "bar")
        assert isinstance(result, ParseSuccess)
        assert result.value == (None, "bar")
        assert result.state.pos == 3

    def test_not_p_useful_for_negative_terminator(self) -> None:
        # eat any 'a' that is not followed by 'b'
        parser = seq(not_p(seq(lit("a"), lit("b"))), lit("a"))
        # "ac" should match (a not followed by b)
        result = _run(parser, "ac")
        assert isinstance(result, ParseSuccess)
        # "ab" should not (a is followed by b)
        result2 = _run(parser, "ab")
        assert isinstance(result2, ParseFailure)


class TestAndP:
    def test_and_p_succeeds_when_inner_succeeds(self) -> None:
        result = _run(and_p(lit("foo")), "foobar")
        assert isinstance(result, ParseSuccess)
        assert result.value is None
        assert result.state.pos == 0

    def test_and_p_fails_when_inner_fails(self) -> None:
        result = _run(and_p(lit("foo")), "bar")
        assert isinstance(result, ParseFailure)
        assert result.pos == 0

    def test_and_p_does_not_consume_input(self) -> None:
        parser = seq(and_p(lit("foo")), lit("foo"))
        result = _run(parser, "foobar")
        assert isinstance(result, ParseSuccess)
        assert result.value == (None, "foo")
        assert result.state.pos == 3
