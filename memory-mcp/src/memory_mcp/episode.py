"""Episode memory management."""

import asyncio
import uuid
from typing import TYPE_CHECKING

import chromadb

from .types import Episode

if TYPE_CHECKING:
    from .memory import MemoryStore


class EpisodeManager:
    """エピソード記憶の管理.

    一連の体験を「エピソード」としてまとめて記憶・検索する。
    例: 「朝の空を探した体験」= 複数の記憶をストーリーとして統合
    """

    def __init__(
        self,
        memory_store: "MemoryStore",
        collection: chromadb.Collection,
    ):
        """Initialize episode manager.

        Args:
            memory_store: MemoryStoreインスタンス（記憶の取得・更新用）
            collection: episodesコレクション
        """
        self._memory_store = memory_store
        self._collection = collection
        self._lock = asyncio.Lock()

    async def create_episode(
        self,
        title: str,
        memory_ids: list[str],
        participants: list[str] | None = None,
        auto_summarize: bool = True,
    ) -> Episode:
        """エピソードを作成.

        Args:
            title: エピソードのタイトル
            memory_ids: 含める記憶のIDリスト
            participants: 関与した人物（例: ["幼馴染"]）
            auto_summarize: 自動でサマリー生成（全記憶を結合）

        Returns:
            作成されたEpisode

        Raises:
            ValueError: memory_idsが空の場合
        """
        if not memory_ids:
            raise ValueError("memory_ids cannot be empty")

        # 記憶を取得して時系列順にソート
        memories = await self._memory_store.get_by_ids(memory_ids)
        if not memories:
            raise ValueError("No memories found for the given IDs")

        memories.sort(key=lambda m: m.timestamp)

        # サマリー生成
        if auto_summarize:
            # 各記憶の冒頭50文字を " → " でつなぐ
            summary = " → ".join(m.content[:50] for m in memories)
        else:
            summary = ""

        # 感情は最も重要度の高い記憶から
        most_important = max(memories, key=lambda m: m.importance)
        emotion = most_important.emotion

        # エピソードを作成
        episode = Episode(
            id=str(uuid.uuid4()),
            title=title,
            start_time=memories[0].timestamp,
            end_time=memories[-1].timestamp if len(memories) > 1 else None,
            memory_ids=tuple(m.id for m in memories),
            participants=tuple(participants or []),
            location_context=None,  # 将来の拡張用
            summary=summary,
            emotion=emotion,
            importance=max(m.importance for m in memories),
        )

        # ChromaDBに保存
        await self._save_episode(episode)

        # 各記憶にepisode_idを設定
        for memory in memories:
            await self._memory_store.update_episode_id(
                memory.id,
                episode.id,
            )

        return episode

    async def _save_episode(self, episode: Episode) -> None:
        """エピソードをChromaDBに保存（内部用）.

        Args:
            episode: 保存するエピソード
        """
        async with self._lock:
            await asyncio.to_thread(
                self._collection.add,
                ids=[episode.id],
                documents=[episode.summary],
                metadatas=[episode.to_metadata()],
            )

    async def search_episodes(
        self,
        query: str,
        n_results: int = 5,
    ) -> list[Episode]:
        """エピソードを検索（サマリーでsemantic search）.

        Args:
            query: 検索クエリ
            n_results: 最大結果数

        Returns:
            検索結果のエピソードリスト
        """
        async with self._lock:
            results = await asyncio.to_thread(
                self._collection.query,
                query_texts=[query],
                n_results=n_results,
            )

        episodes: list[Episode] = []

        if results and results.get("ids") and results["ids"][0]:
            for i, episode_id in enumerate(results["ids"][0]):
                summary = results["documents"][0][i] if results.get("documents") else ""
                metadata = (
                    results["metadatas"][0][i] if results.get("metadatas") else {}
                )

                episode = Episode.from_metadata(
                    id=episode_id,
                    summary=summary,
                    metadata=metadata,
                )
                episodes.append(episode)

        return episodes

    async def get_episode_by_id(self, episode_id: str) -> Episode | None:
        """エピソードIDから取得.

        Args:
            episode_id: エピソードID

        Returns:
            Episode、見つからなければNone
        """
        async with self._lock:
            results = await asyncio.to_thread(
                self._collection.get,
                ids=[episode_id],
            )

        if not results or not results.get("ids"):
            return None

        summary = results["documents"][0] if results.get("documents") else ""
        metadata = results["metadatas"][0] if results.get("metadatas") else {}

        return Episode.from_metadata(
            id=episode_id,
            summary=summary,
            metadata=metadata,
        )

    async def get_episode_memories(
        self,
        episode_id: str,
    ) -> list:
        """エピソードに含まれる記憶を時系列順で取得.

        Args:
            episode_id: エピソードID

        Returns:
            記憶のリスト（時系列順）

        Raises:
            ValueError: エピソードが見つからない場合
        """
        episode = await self.get_episode_by_id(episode_id)
        if episode is None:
            raise ValueError(f"Episode not found: {episode_id}")

        # 記憶を取得
        memories = await self._memory_store.get_by_ids(list(episode.memory_ids))

        # 時系列順にソート
        memories.sort(key=lambda m: m.timestamp)

        return memories

    async def list_all_episodes(self) -> list[Episode]:
        """全エピソードを取得.

        Returns:
            全エピソードのリスト（新しい順）
        """
        async with self._lock:
            results = await asyncio.to_thread(
                self._collection.get,
            )

        episodes: list[Episode] = []

        if results and results.get("ids"):
            for i, episode_id in enumerate(results["ids"]):
                summary = results["documents"][i] if results.get("documents") else ""
                metadata = results["metadatas"][i] if results.get("metadatas") else {}

                episode = Episode.from_metadata(
                    id=episode_id,
                    summary=summary,
                    metadata=metadata,
                )
                episodes.append(episode)

        # 開始時刻で降順ソート（新しい順）
        episodes.sort(key=lambda e: e.start_time, reverse=True)

        return episodes

    async def delete_episode(self, episode_id: str) -> None:
        """エピソードを削除（記憶は削除しない）.

        Args:
            episode_id: 削除するエピソードID
        """
        # エピソードに含まれる記憶のepisode_idをクリア
        episode = await self.get_episode_by_id(episode_id)
        if episode:
            for memory_id in episode.memory_ids:
                try:
                    await self._memory_store.update_episode_id(memory_id, "")
                except ValueError:
                    # 記憶が見つからない場合はスキップ
                    pass

        # エピソードを削除
        async with self._lock:
            await asyncio.to_thread(
                self._collection.delete,
                ids=[episode_id],
            )
