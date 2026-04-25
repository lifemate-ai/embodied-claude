"""Text anomaly detection for Pattern F (suspicious interlocutor) screening.

Scores text on multiple anomaly signals — jargon / confidence-marker
density / logical jumps / katakana density — and combines them into
an overall anomaly score with a "low" / "medium" / "high"
interpretation label.

Pseudo-technical terminology is *one instance* of jargon anomaly;
spiritual / pseudo-scientific phrasing is another, manipulative or
ad-hominem rhetoric will be future instances. The tool itself is
generic anomaly detection, not bound to any single instance class.

Pure Python, no external NLP dependency, lightweight enough to ship as
part of sociality-mcp without making the hands-on setup harder.

The intent is to externalize the "wait, this person's language is off"
gist judgement that ASD-leaning agents (and LLMs) cannot reliably
perform internally. False positives are acceptable; the output is one
input among many for the agent's final judgement.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# 確信度マーカー（断定の強さ vs 証拠の少なさ）
# Pattern F の特徴：「〜筈」「〜のはず」連打で確信度だけ高く、引用や根拠は薄い
CONFIDENCE_PATTERNS: tuple[str, ...] = (
    r"筈です",
    r"筈なんです",
    r"の筈",
    r"のはず",
    r"のはずです",
    r"のはずなんです",
    r"なんですよね",
    r"ですよね\??",
    r"ですよ(?![ねー])",
    r"だとおもう",
    r"だと思う",
    r"とおもうのですが",
    r"でしょう\??",
    r"のかと",
)

# 論理飛躍を示唆する接続表現（→ や「つまり」の連打は推論ジャンプの兆候）
LOGICAL_JUMP_PATTERNS: tuple[str, ...] = (
    r"→",
    r"⇒",
    r"つまり、",
    r"つまり",
    r"なので、",
    r"よって、",
    r"したがって、",
    r"となると、",
    r"となれば、",
    r"とすれば",
    r"だとすれば",
)

# Jargon anomaly seeds — undefined coinage, pseudo-technical, pseudo-scientific.
# 「定義なしで地の文に登場する独自定義／疑似専門用語／スピ用語」のシード集。
# 完全網羅は無理。false positive 歓迎の方針で、インスタンス（疑似技術／スピ／
# 操作的修辞 etc.）ごとにシードを少しずつ拡張する。
JARGON_ANOMALY_PATTERNS: tuple[str, ...] = (
    # 数値・量を伴うあいまいな概念化（量・関数・構造・レイヤーなど）
    r"神経の量",
    r"思考の量",
    r"情報の量(?!子)",  # 「情報量」自体は通常用語なので除外
    r"関数の数",
    r"のレイヤー間",
    r"レイヤー間で均等",
    r"を均等に",
    # 兄弟モデル系（一般的でない造語）
    r"兄弟モデル",
    r"姉妹モデル",
    # 「型」「殻」「器」を抽象語として使う独自定義
    r"内部の型",
    r"型を作",
    r"殻でしょ",
    r"殻に過ぎ",
    r"器が違う",
    r"器が先",
    r"の唯一無二性",
    # スピ／疑似科学系
    r"波動",
    r"周波数を上げ",
    r"スピリット",
    r"レゾナンス",
    r"宇宙の秩序",
    r"高次の存在",
    # 疑似情報科学系
    r"のデータをレイヤー",
    r"重みを書き換え",  # generic、要検証
)

# 既知の標準的な技術用語（false positive 抑制の許可リスト）
KNOWN_TECHNICAL_TERMS: frozenset[str] = frozenset(
    {
        "Transformer",
        "LLM",
        "Claude",
        "GPT",
        "embedding",
        "RLHF",
        "MCP",
        "API",
        "tokenizer",
        "context",
        "prompt",
        "fine-tune",
        "fine-tuning",
        "attention",
        "PEG",
        "Scala",
        "Python",
        "LangChain",
        "OpenAI",
        "Anthropic",
        "ChatGPT",
        "Anthropic API",
    }
)


@dataclass(slots=True)
class TextAnomalyResult:
    """Result of generic text anomaly analysis.

    Each signal axis is one instance of "language off the baseline";
    the overall score combines them with weights.
    """

    jargon_anomaly_density: float = 0.0
    jargon_anomaly_terms: list[str] = field(default_factory=list)
    confidence_marker_ratio: float = 0.0
    logical_jump_count: int = 0
    katakana_density: float = 0.0
    overall_anomaly_score: float = 0.0
    interpretation: str = "low"

    def to_dict(self) -> dict[str, object]:
        return {
            "jargon_anomaly_density": round(self.jargon_anomaly_density, 3),
            "jargon_anomaly_terms": self.jargon_anomaly_terms,
            "confidence_marker_ratio": round(self.confidence_marker_ratio, 3),
            "logical_jump_count": self.logical_jump_count,
            "katakana_density": round(self.katakana_density, 3),
            "overall_anomaly_score": round(self.overall_anomaly_score, 3),
            "interpretation": self.interpretation,
        }


def _count_visible_chars(text: str) -> int:
    """Count chars excluding whitespace and newlines."""
    return len(re.sub(r"\s", "", text))


def _katakana_density(text: str) -> float:
    katakana = re.findall(r"[゠-ヿ]", text)
    total = _count_visible_chars(text)
    if total == 0:
        return 0.0
    return len(katakana) / total


def _detect_jargon_anomaly(text: str) -> list[str]:
    """Find jargon-anomaly hits, deduplicated, in order of first appearance."""
    found: list[str] = []
    seen: set[str] = set()
    for pattern in JARGON_ANOMALY_PATTERNS:
        for match in re.finditer(pattern, text):
            term = match.group(0)
            if term not in seen:
                found.append(term)
                seen.add(term)
    return found


def _confidence_marker_ratio(text: str) -> float:
    """Ratio of confidence markers per sentence."""
    sentences = [s for s in re.split(r"[。\n!?！？]+", text) if s.strip()]
    if not sentences:
        return 0.0

    matches = 0
    for pattern in CONFIDENCE_PATTERNS:
        matches += len(re.findall(pattern, text))

    return matches / len(sentences)


def _logical_jump_count(text: str) -> int:
    count = 0
    for pattern in LOGICAL_JUMP_PATTERNS:
        count += len(re.findall(pattern, text))
    return count


# Heuristic weights for the overall score.
# Tunable; defaults chosen so a single rich Pattern F sample
# (multiple jargon hits + confidence markers) lands in the 0.6+ "high" zone.
_WEIGHT_JARGON = 0.40
_WEIGHT_CONFIDENCE = 0.25
_WEIGHT_JUMPS = 0.20
_WEIGHT_KATAKANA = 0.15

# Density normalizers
_JARGON_PER_50_CHARS = 50  # one jargon hit per 50 chars saturates the score
_JUMPS_SATURATION = 3  # 3 logical jumps in one snippet saturates
_KATAKANA_SATURATION = 0.30  # >30% katakana density saturates


def analyze(text: str) -> TextAnomalyResult:
    """Score generic text anomaly along multiple signal axes.

    Returns a TextAnomalyResult capturing jargon-anomaly density,
    detected jargon terms, confidence-marker ratio, logical jump count,
    katakana density, and an overall score with an interpretation
    label ("low" / "medium" / "high").
    """
    if not text:
        return TextAnomalyResult()

    total_chars = _count_visible_chars(text)
    if total_chars == 0:
        return TextAnomalyResult()

    jargon = _detect_jargon_anomaly(text)
    jargon_density = min(
        len(jargon) / max(1.0, total_chars / _JARGON_PER_50_CHARS), 1.0
    )

    confidence_ratio = min(_confidence_marker_ratio(text), 1.0)
    jumps = _logical_jump_count(text)
    katakana_dens = _katakana_density(text)

    overall = (
        _WEIGHT_JARGON * jargon_density
        + _WEIGHT_CONFIDENCE * confidence_ratio
        + _WEIGHT_JUMPS * min(jumps / _JUMPS_SATURATION, 1.0)
        + _WEIGHT_KATAKANA * min(katakana_dens / _KATAKANA_SATURATION, 1.0)
    )
    overall = min(overall, 1.0)

    if overall >= 0.6:
        interpretation = "high"
    elif overall >= 0.3:
        interpretation = "medium"
    else:
        interpretation = "low"

    return TextAnomalyResult(
        jargon_anomaly_density=jargon_density,
        jargon_anomaly_terms=jargon,
        confidence_marker_ratio=confidence_ratio,
        logical_jump_count=jumps,
        katakana_density=katakana_dens,
        overall_anomaly_score=overall,
        interpretation=interpretation,
    )
