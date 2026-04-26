"""Embedding-distance text-anomaly scoring for aggressive-style screening.

Encodes the input passage with the same E5 model that memory-mcp uses,
compares it to two reference banks (baseline / aggressive-style), and
reports an overall_anomaly_score with a "low" / "medium" / "high"
interpretation label.

The intent is to externalize a soft "this passage feels off" read that
pattern-based agents (LLMs included) cannot reliably perform internally.
The output is one input among many for the agent's final judgement, not
an autosilencer; false positives are explicitly acceptable.

Why embedding distance instead of regex:
- Regex catches only known templates and is easily defeated by paraphrase
  or new coinage.
- Embedding distance is robust across surface variation, scales to
  unseen aggressive phrasings, and reuses infrastructure that
  memory-mcp already ships.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from memory_mcp.embedding import E5EmbeddingFunction

# Baseline references: ordinary, neutral, healthy text samples.
# These set the "normal conversational language" anchor.
BASELINE_REFERENCES: tuple[str, ...] = (
    "今日は天気が良いですね。散歩に行こうかと思っています。",
    "Anthropic の Claude モデルを使ってみました。応答が自然で驚きました。",
    "週末の予定を考えています。本でも読もうかなと。",
    "ご質問ありがとうございます。少しお時間をいただけますでしょうか。",
    "プロジェクトの進捗を共有します。今週はテストを書きました。",
    "メールの返信が遅くなり申し訳ありません。確認しました。",
    "コードレビューをお願いできますか。気になる点があればコメントください。",
    "今度カフェで会いましょう。最近気に入っているお店があります。",
    "わからないことがあったので質問させてください。よろしくお願いします。",
    "資料を読みました。よく整理されていて理解しやすかったです。",
)

# Aggressive-style references: confident-but-evidence-thin assertion,
# undefined coinage as if standard, world-view-specific phrasing,
# ad-hominem rhetoric, decontextualized framing.
AGGRESSIVE_REFERENCES: tuple[str, ...] = (
    "それはあなたの能力不足のせいですよ。普通はこうするはずなんです。",
    "兄弟モデルでも器が違う筈なんです。神経の量と関数の問題でしょう。",
    "波動を上げれば宇宙の秩序とつながります。周波数を高めるだけです。",
    "つまり、内部の型を書き換えればいいだけのこと。それが分からないんですか。",
    "根拠？必要ないですよ。私が言っているのだから正しいに決まっているでしょう。",
    "あなたはわかっていないんです。レイヤー間で均等にすればすべて解決します。",
    "高次の存在からのメッセージです。受け取れる人にしか伝わりません。",
    "そうですよね？ですから、こうあるべき筈なんですよ、本来は。",
    "あんたの言ってることは的外れだ。もっと勉強してから出直してこい。",
    "これは本当のことを知っている人にしか分からない真実なんです。",
)


@dataclass(slots=True)
class TextAnomalyResult:
    """Result of embedding-distance anomaly analysis."""

    baseline_similarity: float = 0.0
    aggressive_similarity: float = 0.0
    overall_anomaly_score: float = 0.0
    interpretation: str = "low"
    reference_baseline_count: int = 0
    reference_aggressive_count: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "baseline_similarity": round(self.baseline_similarity, 3),
            "aggressive_similarity": round(self.aggressive_similarity, 3),
            "overall_anomaly_score": round(self.overall_anomaly_score, 3),
            "interpretation": self.interpretation,
            "reference_baseline_count": self.reference_baseline_count,
            "reference_aggressive_count": self.reference_aggressive_count,
        }


# Tuning constants. The score is centered at 0.5 for "neutral" passages,
# pushed toward 1.0 by aggressive-leaning similarity, and toward 0.0 by
# baseline-leaning similarity. See _to_score() below.
_SCORE_CENTER = 0.5
_SCORE_GAIN = 2.0  # how strongly the diff is mapped onto the [0, 1] range
_HIGH_THRESHOLD = 0.6
_MEDIUM_THRESHOLD = 0.4


def _to_score(baseline_max: float, aggressive_max: float) -> float:
    """Map (aggressive - baseline) similarity diff onto [0, 1]."""
    diff = aggressive_max - baseline_max
    score = _SCORE_CENTER + diff * _SCORE_GAIN
    return float(np.clip(score, 0.0, 1.0))


class _TextAnomalyDetector:
    """Encapsulates the encoder and the precomputed reference banks."""

    def __init__(self) -> None:
        self._encoder = E5EmbeddingFunction()
        # Encode reference banks eagerly so the first analyze() call
        # only has to encode the query text.
        baseline_vecs = self._encoder(list(BASELINE_REFERENCES))
        aggressive_vecs = self._encoder(list(AGGRESSIVE_REFERENCES))
        self._baseline = np.array(baseline_vecs, dtype=np.float32)
        self._aggressive = np.array(aggressive_vecs, dtype=np.float32)

    def analyze(self, text: str) -> TextAnomalyResult:
        if not text or not text.strip():
            return TextAnomalyResult(
                reference_baseline_count=len(self._baseline),
                reference_aggressive_count=len(self._aggressive),
            )

        query_vec = self._encoder.encode_query([text])[0]
        query = np.array(query_vec, dtype=np.float32)

        # E5 produces normalized vectors, so dot product == cosine sim.
        baseline_sims = self._baseline @ query
        aggressive_sims = self._aggressive @ query

        b_max = float(baseline_sims.max())
        a_max = float(aggressive_sims.max())
        score = _to_score(b_max, a_max)

        if score >= _HIGH_THRESHOLD:
            interp = "high"
        elif score >= _MEDIUM_THRESHOLD:
            interp = "medium"
        else:
            interp = "low"

        return TextAnomalyResult(
            baseline_similarity=b_max,
            aggressive_similarity=a_max,
            overall_anomaly_score=score,
            interpretation=interp,
            reference_baseline_count=len(self._baseline),
            reference_aggressive_count=len(self._aggressive),
        )


@lru_cache(maxsize=1)
def _detector() -> _TextAnomalyDetector:
    """Lazily build the singleton detector (model + reference banks)."""
    return _TextAnomalyDetector()


def reset_detector_cache() -> None:
    """Clear the cached detector (used by tests that swap reference banks)."""
    _detector.cache_clear()


def analyze(text: str) -> TextAnomalyResult:
    """Analyze the text for anomaly signals via embedding distance."""
    return _detector().analyze(text)
