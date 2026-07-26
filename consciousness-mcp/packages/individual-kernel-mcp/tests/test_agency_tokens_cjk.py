"""Mismatch tokenization handles non-ASCII summaries via character bigrams."""

from __future__ import annotations

from individual_kernel_mcp.agency import _tokens


class TestTokenizerCjk:
    def test_ascii_behavior_unchanged(self) -> None:
        assert _tokens("Camera moved left as predicted") == {
            "camera",
            "moved",
            "left",
            "predicted",
        }

    def test_japanese_identical_summaries_fully_overlap(self) -> None:
        expected = _tokens("カメラが左に動いた")
        actual = _tokens("カメラが左に動いた")
        assert expected
        assert len(expected & actual) / len(expected) == 1.0

    def test_japanese_related_summaries_share_tokens(self) -> None:
        expected = _tokens("カメラが左に動いた")
        actual = _tokens(
            "カメラが左に動いて視界が変わった"
        )
        overlap = len(expected & actual) / len(expected)
        assert overlap > 0.5

    def test_mixed_text_keeps_both_kinds_of_tokens(self) -> None:
        tokens = _tokens("cameraを左へ")
        assert "camera" in tokens
        assert "を左" in tokens
