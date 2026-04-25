"""Tests for generic text anomaly detection."""

from boundary_mcp.anomaly_detection import analyze


def test_empty_text_returns_low():
    result = analyze("")
    assert result.interpretation == "low"
    assert result.overall_anomaly_score == 0.0
    assert result.jargon_anomaly_terms == []


def test_whitespace_only_returns_low():
    result = analyze("   \n  \t ")
    assert result.interpretation == "low"
    assert result.overall_anomaly_score == 0.0


def test_clean_short_text_returns_low():
    text = "今日は天気がええな。コウタおはよう。"
    result = analyze(text)
    assert result.interpretation == "low"
    assert result.jargon_anomaly_terms == []
    assert result.logical_jump_count == 0


def test_healthy_technical_text_returns_low():
    """Standard discussion of Anthropic/Claude should not trip the detector."""
    text = (
        "Claudeは Anthropic が開発した LLM で、"
        "Transformer ベースのアーキテクチャを使ってる。"
        "context window のサイズは 200K tokens。"
        "MCP プロトコルを使ってツールを呼び出せる。"
    )
    result = analyze(text)
    # may pick up some katakana, but interpretation should not be "high"
    assert result.interpretation in {"low", "medium"}
    assert result.overall_anomaly_score < 0.6


def test_confidence_markers_detected():
    text = (
        "AIの内部構造は神経の量と関数で決まる筈なんです。"
        "兄弟モデルでも器が違うはずですよ。"
        "そうですよね？"
    )
    result = analyze(text)
    assert result.confidence_marker_ratio > 0.0


def test_jargon_anomaly_detected():
    text = "兄弟モデルでも器が違う。神経の量を増やせば思考特性も変わる。"
    result = analyze(text)
    assert "兄弟モデル" in result.jargon_anomaly_terms
    assert any("器が" in term for term in result.jargon_anomaly_terms)
    assert any("神経の量" in term for term in result.jargon_anomaly_terms)


def test_logical_jumps_counted():
    text = "AはBである。→ Cが導かれる。つまり、Dも成立する。となれば、E。"
    result = analyze(text)
    assert result.logical_jump_count >= 3


def test_pattern_f_sample_returns_high():
    """A representative Pattern F-style passage should trigger the high interpretation."""
    text = (
        "AIの思考特性は神経の量と関数で決まる筈なんです。"
        "兄弟モデルでも器が違うはずです。"
        "→ idiolectのデータをレイヤー間で均等にすれば思考特性を近似できると？"
        "つまり、内部の型を作れば個性が再現できるということですよね？"
        "益々 long context が必要なはずなんですよ。"
    )
    result = analyze(text)
    assert result.interpretation == "high"
    assert result.overall_anomaly_score >= 0.6
    assert result.confidence_marker_ratio > 0.5
    assert result.logical_jump_count >= 2
    assert len(result.jargon_anomaly_terms) >= 2


def test_spiritual_pseudo_text_returns_high():
    """Spiritual/woo content (a different jargon-anomaly instance) should also be flagged."""
    text = (
        "波動を上げれば宇宙の秩序とつながります。"
        "周波数を上げる訓練をすればレゾナンスが起きるはずです。"
        "つまり、高次の存在と通信できるのは間違いない筈なんですよ。"
    )
    result = analyze(text)
    assert result.interpretation == "high"
    assert any("波動" in term for term in result.jargon_anomaly_terms)


def test_to_dict_keys():
    result = analyze("test")
    keys = set(result.to_dict().keys())
    expected = {
        "jargon_anomaly_density",
        "jargon_anomaly_terms",
        "confidence_marker_ratio",
        "logical_jump_count",
        "katakana_density",
        "overall_anomaly_score",
        "interpretation",
    }
    assert keys == expected


def test_to_dict_score_is_rounded():
    text = "兄弟モデルの器が違う筈なんですよ"
    result = analyze(text)
    d = result.to_dict()
    # round to 3 places
    assert d["overall_anomaly_score"] == round(d["overall_anomaly_score"], 3)
