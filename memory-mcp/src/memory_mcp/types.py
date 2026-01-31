"""Type definitions for Memory MCP Server."""

from dataclasses import dataclass
from enum import Enum


class Emotion(str, Enum):
    """感情タグ."""

    HAPPY = "happy"
    SAD = "sad"
    SURPRISED = "surprised"
    MOVED = "moved"
    EXCITED = "excited"
    NOSTALGIC = "nostalgic"
    CURIOUS = "curious"
    NEUTRAL = "neutral"


class Category(str, Enum):
    """記憶カテゴリ."""

    DAILY = "daily"
    PHILOSOPHICAL = "philosophical"
    TECHNICAL = "technical"
    MEMORY = "memory"
    OBSERVATION = "observation"
    FEELING = "feeling"
    CONVERSATION = "conversation"


@dataclass(frozen=True)
class Memory:
    """記憶データ構造."""

    id: str
    content: str
    timestamp: str  # ISO 8601 format
    emotion: str
    importance: int  # 1-5
    category: str
    # Phase 2: アクセス追跡
    access_count: int = 0  # 想起回数
    last_accessed: str = ""  # 最終アクセス時刻（ISO 8601）
    # Phase 3: 連想リンク
    linked_ids: tuple[str, ...] = ()  # リンク先の記憶ID群

    def to_metadata(self) -> dict:
        """Convert to dictionary for ChromaDB metadata."""
        return {
            "timestamp": self.timestamp,
            "emotion": self.emotion,
            "importance": self.importance,
            "category": self.category,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed,
            "linked_ids": ",".join(self.linked_ids),  # カンマ区切りで保存
        }


@dataclass(frozen=True)
class MemorySearchResult:
    """検索結果."""

    memory: Memory
    distance: float  # 類似度（小さいほど近い）


@dataclass(frozen=True)
class ScoredMemory:
    """スコアリング済み検索結果."""

    memory: Memory
    semantic_distance: float  # ChromaDBからの生距離
    time_decay_factor: float  # 時間減衰係数 (0.0-1.0)
    emotion_boost: float  # 感情ブースト
    importance_boost: float  # 重要度ブースト
    final_score: float  # 最終スコア（低いほど良い）


@dataclass(frozen=True)
class MemoryStats:
    """記憶の統計情報."""

    total_count: int
    by_category: dict[str, int]
    by_emotion: dict[str, int]
    oldest_timestamp: str | None
    newest_timestamp: str | None
