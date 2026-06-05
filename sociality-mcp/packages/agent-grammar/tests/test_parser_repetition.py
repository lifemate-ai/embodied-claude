"""Tests for repetition combinators: star (*) and plus (+).

  star: 0 or more occurrences; never fails, returns list
  plus: 1 or more occurrences; fails if no match

Both are greedy (PEG semantics — no backtracking from inside the loop).
"""

from __future__ import annotations

from agent_grammar.parser import (
    ParseFailure,
    ParseState,
    ParseSuccess,
    lit,
    plus,
    seq,
    star,
)


def _run(parser, text: str):
    return parser(ParseState(text=text, pos=0))


class TestStar:
    def test_star_zero_matches_succeeds_with_empty_list(self) -> None:
        result = _run(star(lit("a")), "xyz")
        assert isinstance(result, ParseSuccess)
        assert result.value == []
        assert result.state.pos == 0

    def test_star_consumes_all_matches_greedily(self) -> None:
        result = _run(star(lit("a")), "aaab")
        assert isinstance(result, ParseSuccess)
        assert result.value == ["a", "a", "a"]
        assert result.state.pos == 3

    def test_star_with_seq_pattern(self) -> None:
        # match any number of "ab" pairs
        parser = star(seq(lit("a"), lit("b")))
        result = _run(parser, "ababab")
        assert isinstance(result, ParseSuccess)
        assert len(result.value) == 3
        assert result.state.pos == 6


class TestPlus:
    def test_plus_zero_matches_fails(self) -> None:
        result = _run(plus(lit("a")), "xyz")
        assert isinstance(result, ParseFailure)
        assert result.pos == 0

    def test_plus_single_match_succeeds(self) -> None:
        result = _run(plus(lit("a")), "ab")
        assert isinstance(result, ParseSuccess)
        assert result.value == ["a"]
        assert result.state.pos == 1

    def test_plus_multiple_matches(self) -> None:
        result = _run(plus(lit("a")), "aaab")
        assert isinstance(result, ParseSuccess)
        assert result.value == ["a", "a", "a"]
        assert result.state.pos == 3

    def test_plus_in_seq(self) -> None:
        parser = seq(plus(lit("a")), lit("b"))
        result = _run(parser, "aaab")
        assert isinstance(result, ParseSuccess)
        assert result.value == (["a", "a", "a"], "b")
