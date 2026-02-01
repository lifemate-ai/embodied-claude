"""Working memory buffer for fast access to recent memories."""

import asyncio
from collections import deque
from datetime import datetime, timedelta, timezone
from itertools import islice
from typing import TYPE_CHECKING

from .types import Memory

if TYPE_CHECKING:
    from .memory import MemoryStore


class WorkingMemoryBuffer:
    """作業記憶（短期記憶）バッファ - インメモリのみ.

    人間の短期記憶と同様に、最近の記憶を高速にアクセスできるバッファ。
    セッション終了で自然に忘れる（永続化しない）。
    """

    def __init__(self, capacity: int = 20):
        """Initialize working memory buffer.

        Args:
            capacity: バッファの最大容量（デフォルト20）
        """
        self._buffer: deque[Memory] = deque(maxlen=capacity)
        self._lock = asyncio.Lock()

    async def add(self, memory: Memory) -> None:
        """記憶を追加（古いものは自動削除）.

        Args:
            memory: 追加する記憶
        """
        async with self._lock:
            self._buffer.append(memory)

    async def get_recent(self, n: int = 10) -> list[Memory]:
        """最近のn件を取得.

        Args:
            n: 取得する記憶の数

        Returns:
            最新のn件の記憶（新しい順）
        """
        async with self._lock:
            return list(islice(reversed(self._buffer), n))

    async def get_all(self) -> list[Memory]:
        """バッファ内の全記憶を取得.

        Returns:
            全記憶（新しい順）
        """
        async with self._lock:
            return list(reversed(self._buffer))

    async def clear(self) -> None:
        """バッファをクリア."""
        async with self._lock:
            self._buffer.clear()

    async def refresh_important(
        self,
        memory_store: "MemoryStore",
    ) -> None:
        """重要な記憶を長期記憶から再ロード.

        以下の条件を満たす記憶を再ロード：
        - importance >= 4
        - access_count >= 5
        - last_accessed が直近1週間以内

        Args:
            memory_store: 長期記憶ストア
        """
        # 直近1週間の閾値
        one_week_ago = (
            datetime.now(timezone.utc) - timedelta(days=7)
        ).isoformat()

        # 重要度の高い記憶を検索
        # （memory_storeのメソッドを使う - 実装はmemory.pyで）
        important_memories = await memory_store.search_important_memories(
            min_importance=4,
            min_access_count=5,
            since=one_week_ago,
            n_results=10,
        )

        # バッファに追加（重複排除）
        async with self._lock:
            existing_ids = {m.id for m in self._buffer}
            for memory in important_memories:
                if memory.id not in existing_ids:
                    self._buffer.append(memory)

    def size(self) -> int:
        """現在のバッファサイズを取得.

        Returns:
            バッファ内の記憶数
        """
        return len(self._buffer)
