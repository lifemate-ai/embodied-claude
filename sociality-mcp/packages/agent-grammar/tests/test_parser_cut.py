"""Tests for the PEG-with-cut commitment operator.

Cut (à la Mizukami & Mizushima 2014, and Onion) marks a commitment point:
once seq(..., cut(), ...) has passed cut(), subsequent failures inside the
same seq are propagated as committed failures, and an enclosing alt() will
NOT try further alternatives — it will fail immediately.

Use case in agent-grammar: BoundaryCheck. Once a policy violation is
confirmed, we don't want the orchestrator to silently try another response
move that ignores the boundary. Cut makes that propagation explicit.
"""

from __future__ import annotations

from agent_grammar.parser import (
    ParseFailure,
    ParseState,
    ParseSuccess,
    alt,
    cut,
    lit,
    seq,
)


def _run(parser, text: str):
    return parser(ParseState(text=text, pos=0))


class TestCutItself:
    def test_cut_alone_succeeds_with_committed_flag(self) -> None:
        result = _run(cut(), "anything")
        assert isinstance(result, ParseSuccess)
        assert result.value is None
        assert result.committed is True
        # cut never consumes input
        assert result.state.pos == 0


class TestCutInSeq:
    def test_seq_without_cut_propagates_uncommitted_failure(self) -> None:
        parser = seq(lit("a"), lit("b"))
        result = _run(parser, "ax")
        assert isinstance(result, ParseFailure)
        assert result.committed is False

    def test_seq_after_cut_propagates_committed_failure(self) -> None:
        parser = seq(lit("a"), cut(), lit("b"))
        result = _run(parser, "ax")
        assert isinstance(result, ParseFailure)
        assert result.committed is True

    def test_seq_success_after_cut_still_succeeds(self) -> None:
        parser = seq(lit("a"), cut(), lit("b"))
        result = _run(parser, "ab")
        assert isinstance(result, ParseSuccess)
        # the seq's tuple skips the cut() value (None) — implementation
        # choice: we include it as None for transparency.
        assert "a" in result.value
        assert "b" in result.value


class TestCutInAlt:
    def test_alt_without_cut_tries_next_alternative_on_failure(self) -> None:
        # Without cut: if "ab" branch fails after seeing "a", alt tries "ax"
        parser = alt(seq(lit("a"), lit("b")), seq(lit("a"), lit("x")))
        result = _run(parser, "ax")
        assert isinstance(result, ParseSuccess)
        assert result.value == ("a", "x")

    def test_alt_with_cut_does_not_try_next_alternative(self) -> None:
        # With cut after "a": once "a" matches and cut commits, failure on
        # "b" is committed, so the second alternative is NOT tried even
        # though it would have matched.
        parser = alt(
            seq(lit("a"), cut(), lit("b")),
            seq(lit("a"), lit("x")),
        )
        result = _run(parser, "ax")
        assert isinstance(result, ParseFailure)
        assert result.committed is True

    def test_alt_with_cut_still_succeeds_when_committed_branch_completes(self) -> None:
        parser = alt(
            seq(lit("a"), cut(), lit("b")),
            seq(lit("a"), lit("x")),
        )
        result = _run(parser, "ab")
        assert isinstance(result, ParseSuccess)
        assert "a" in result.value
        assert "b" in result.value

    def test_alt_without_cut_picks_correct_alternative_when_first_fails_early(
        self,
    ) -> None:
        # Pure PEG: failure before any cut is uncommitted, alt backtracks.
        parser = alt(lit("foo"), lit("bar"))
        result = _run(parser, "bar")
        assert isinstance(result, ParseSuccess)
        assert result.value == "bar"
