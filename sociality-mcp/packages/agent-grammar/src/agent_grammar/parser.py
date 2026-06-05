"""Hand-rolled PEG parser core.

Implements the minimal combinator set used to build the agent turn grammar:

  lit  : match a literal string
  seq  : ordered concatenation; all sub-parsers must succeed
  alt  : PEG ordered choice; first success wins (NOT longest-match)
  opt  : 0 or 1 occurrence; never fails, returns None on miss

State is immutable: parsers take a ParseState and return a fresh
ParseSuccess (with an advanced state) or a ParseFailure. The caller's
state is never mutated, so an outer alt can retry alternatives without
needing to "rewind" anything.

Future extensions tracked in the plan-of-record
(~/.claude/plans/jazzy-wishing-starfish.md):

  - plus / star (repetition)
  - and_p / not_p (lookahead)
  - cut (PEG-with-cut, à la Onion) for BoundaryCheck commits
  - packrat memoization
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ParseState:
    """Immutable parser position into the input text."""

    text: str
    pos: int = 0

    def advance(self, n: int) -> ParseState:
        return ParseState(text=self.text, pos=self.pos + n)


@dataclass(frozen=True)
class ParseSuccess:
    value: Any
    state: ParseState
    committed: bool = False


@dataclass(frozen=True)
class ParseFailure:
    pos: int
    message: str
    committed: bool = False


ParseResult = ParseSuccess | ParseFailure
Parser = Callable[[ParseState], ParseResult]


def lit(s: str) -> Parser:
    """Match the exact literal string s at the current position."""

    def parse(state: ParseState) -> ParseResult:
        if state.text.startswith(s, state.pos):
            return ParseSuccess(value=s, state=state.advance(len(s)))
        return ParseFailure(pos=state.pos, message=f"expected literal {s!r}")

    return parse


def seq(*parsers: Parser) -> Parser:
    """Run sub-parsers in order; fail fast on any failure.

    Once any sub-parser succeeds with committed=True (i.e. a cut() was
    crossed), subsequent failures inside this seq propagate as committed
    failures so that an enclosing alt() will not try further alternatives.
    """

    def parse(state: ParseState) -> ParseResult:
        values: list[Any] = []
        current = state
        committed = False
        for sub in parsers:
            result = sub(current)
            if isinstance(result, ParseFailure):
                if committed and not result.committed:
                    return ParseFailure(
                        pos=result.pos,
                        message=result.message,
                        committed=True,
                    )
                return result
            if result.committed:
                committed = True
            values.append(result.value)
            current = result.state
        return ParseSuccess(value=tuple(values), state=current, committed=committed)

    return parse


def alt(*parsers: Parser) -> Parser:
    """PEG ordered choice: return the first success; fail only if all fail.

    A committed failure (one produced after a cut() inside the failing
    branch) short-circuits the choice: alt returns it immediately rather
    than trying further alternatives.
    """

    def parse(state: ParseState) -> ParseResult:
        last_failure: ParseFailure | None = None
        for sub in parsers:
            result = sub(state)
            if isinstance(result, ParseSuccess):
                return result
            if result.committed:
                return result
            last_failure = result
        if last_failure is not None:
            return last_failure
        return ParseFailure(pos=state.pos, message="alt() called with no parsers")

    return parse


def opt(parser: Parser) -> Parser:
    """0 or 1 occurrence. Always succeeds; value is None on miss."""

    def parse(state: ParseState) -> ParseResult:
        result = parser(state)
        if isinstance(result, ParseSuccess):
            return result
        return ParseSuccess(value=None, state=state)

    return parse


def star(parser: Parser) -> Parser:
    """0 or more occurrences (greedy). Never fails; returns list of values."""

    def parse(state: ParseState) -> ParseResult:
        values: list[Any] = []
        current = state
        while True:
            result = parser(current)
            if isinstance(result, ParseFailure):
                return ParseSuccess(value=values, state=current)
            values.append(result.value)
            current = result.state

    return parse


def plus(parser: Parser) -> Parser:
    """1 or more occurrences (greedy). Fails if zero matches."""

    def parse(state: ParseState) -> ParseResult:
        first = parser(state)
        if isinstance(first, ParseFailure):
            return first
        values: list[Any] = [first.value]
        current = first.state
        while True:
            result = parser(current)
            if isinstance(result, ParseFailure):
                return ParseSuccess(value=values, state=current)
            values.append(result.value)
            current = result.state

    return parse


def and_p(parser: Parser) -> Parser:
    """Positive lookahead. Succeeds iff parser succeeds; never consumes input."""

    def parse(state: ParseState) -> ParseResult:
        result = parser(state)
        if isinstance(result, ParseSuccess):
            return ParseSuccess(value=None, state=state)
        return result

    return parse


def not_p(parser: Parser) -> Parser:
    """Negative lookahead. Succeeds iff parser fails; never consumes input."""

    def parse(state: ParseState) -> ParseResult:
        result = parser(state)
        if isinstance(result, ParseSuccess):
            return ParseFailure(
                pos=state.pos,
                message="negative lookahead matched (expected miss)",
            )
        return ParseSuccess(value=None, state=state)

    return parse


def cut() -> Parser:
    """Commit point for PEG-with-cut semantics.

    Following Mizukami & Mizushima (2014) and Onion: once cut() is crossed
    inside a seq, subsequent failures in that seq propagate as committed
    failures, and an enclosing alt() will not try further alternatives.

    Use this when a partial match is enough to confirm we are in the right
    branch (e.g. after a BoundaryCheck rule has matched a policy concern),
    so that other response paths are not silently considered.
    """

    def parse(state: ParseState) -> ParseResult:
        return ParseSuccess(value=None, state=state, committed=True)

    return parse
