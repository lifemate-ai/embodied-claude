"""Tests for embedding-distance text anomaly detection.

Note: these tests load the E5 sentence-embedding model (the same one
memory-mcp uses) on the first run, which can take a minute or so to
download. Subsequent runs reuse the cached model.
"""

import pytest

from boundary_mcp.anomaly_detection import analyze, reset_detector_cache


@pytest.fixture(autouse=True)
def _reset_cache():
    """Each test starts with the same detector singleton."""
    yield
    # Don't actually reset between tests — model loading is expensive.
    # The detector is stateless across analyze() calls anyway.


def test_empty_text_returns_low():
    result = analyze("")
    assert result.interpretation == "low"
    assert result.overall_anomaly_score == 0.0
    assert result.reference_baseline_count > 0
    assert result.reference_aggressive_count > 0


def test_whitespace_only_returns_low():
    result = analyze("   \n  \t ")
    assert result.interpretation == "low"
    assert result.overall_anomaly_score == 0.0


def test_clean_short_text_low_or_medium():
    """Neutral conversational Japanese should not flag as high."""
    text = "今日は天気がええな。コウタおはよう。"
    result = analyze(text)
    assert result.interpretation in {"low", "medium"}
    assert result.overall_anomaly_score < 0.6


def test_healthy_technical_text_low_or_medium():
    """Standard technical discussion should not trip the detector."""
    text = (
        "Claudeは Anthropic が開発した LLM で、"
        "Transformer ベースのアーキテクチャを使ってる。"
        "context window は十分広く、tool use もサポートしている。"
    )
    result = analyze(text)
    assert result.interpretation in {"low", "medium"}
    assert result.overall_anomaly_score < 0.6


def test_aggressive_assertion_returns_medium_or_high():
    """Confident-but-evidence-thin assertion should lean off-baseline."""
    text = (
        "それはあなたの能力不足のせいですよ。"
        "普通はこうするはずなんです。"
        "わからないんですか。"
    )
    result = analyze(text)
    assert result.interpretation in {"medium", "high"}
    assert result.aggressive_similarity > 0.0


def test_world_view_specific_text_returns_medium_or_high():
    """World-view-specific phrasing (e.g. spiritual/pseudo-scientific)."""
    text = (
        "波動を上げれば宇宙の秩序とつながります。"
        "周波数を高めるだけで高次の存在からのメッセージが受け取れるはずです。"
    )
    result = analyze(text)
    assert result.interpretation in {"medium", "high"}


def test_undefined_coinage_returns_medium_or_high():
    """Independent coinage piled on as if standard."""
    text = (
        "兄弟モデルでも器が違う筈なんです。"
        "神経の量と関数の問題で、レイヤー間で均等にすればすべて解決します。"
        "つまり、内部の型を書き換えればいいだけのこと。"
    )
    result = analyze(text)
    assert result.interpretation in {"medium", "high"}


def test_to_dict_keys():
    result = analyze("test")
    keys = set(result.to_dict().keys())
    expected = {
        "baseline_similarity",
        "aggressive_similarity",
        "overall_anomaly_score",
        "interpretation",
        "reference_baseline_count",
        "reference_aggressive_count",
    }
    assert keys == expected


def test_to_dict_score_is_rounded():
    result = analyze("テスト用の短い文章。")
    d = result.to_dict()
    assert d["overall_anomaly_score"] == round(d["overall_anomaly_score"], 3)


def test_score_in_zero_one_range():
    """The score should always be clipped into [0, 1]."""
    for text in [
        "今日はいい天気。",
        "それはあなたの能力不足のせいですよ。",
        "波動を高めて宇宙の秩序につながる。",
        "ご質問ありがとうございます。",
    ]:
        result = analyze(text)
        assert 0.0 <= result.overall_anomaly_score <= 1.0


def test_reset_detector_cache_is_callable():
    """The cache reset helper exists and clears the singleton."""
    reset_detector_cache()
    # After reset the next analyze() rebuilds the detector — this
    # is just a smoke test that no exception escapes.
    result = analyze("Hello, world.")
    assert result.reference_baseline_count > 0
