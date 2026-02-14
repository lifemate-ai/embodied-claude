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
        # Phase 7: ジョブ分離
        memory_type: str = "global",
        job_id: str | None = None,
        shared_group_ids: tuple[str, ...] = (),
    ) -> Episode:
        """エピソードを作成.

        Args:
            title: エピソードのタイトル
            memory_ids: 含める記憶のIDリスト
            participants: 関与した人物（例: ["幼馴染"]）
            auto_summarize: 自動でサマリー生成（全記憶を結合）
            memory_type: メモリタイプ ("global" | "job" | "shared")
            job_id: ジョブ固有エピソードの場合のジョブID
            shared_group_ids: 共有エピソードの場合の共有グループID

        Returns:
            作成されたEpisode

        Raises:
            ValueError: memory_idsが空の場合
        """
        if not memory_ids:
            raise ValueError("memory_ids cannot be empty")

        # 記憶を取得して時系列順にソート（ジョブ分離適用）
        memories = await self._memory_store.get_by_ids(
            memory_ids,
            job_id=job_id,
            include_global=True,
            include_shared=True,
        )
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
            # Phase 7: ジョブ分離
            memory_type=memory_type,
            job_id=job_id,
            shared_group_ids=shared_group_ids,
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
        # Phase 7: ジョブ分離
        job_id: str | None = None,
        include_global: bool = True,
        include_shared: bool = True,
    ) -> list[Episode]:
        """エピソードを検索（サマリーでsemantic search）.

        Args:
            query: 検索クエリ
            n_results: 最大結果数
            job_id: ジョブ固有エピソードを検索する場合のジョブID
            include_global: グローバルエピソードを含めるか
            include_shared: 共有エピソードを含めるか

        Returns:
            検索結果のエピソードリスト
        """
        # Phase 7: ジョブ分離 - 検索対象のエピソードタイプを制御
        job_conditions: list[dict[str, any]] = []

        if include_global:
            job_conditions.append({"memory_type": {"$eq": "global"}})

        if job_id:
            job_conditions.append(
                {
                    "$and": [
                        {"memory_type": {"$eq": "job"}},
                        {"job_id": {"$eq": job_id}},
                    ]
                }
            )

            if include_shared:
                job_conditions.append({"memory_type": {"$eq": "shared"}})

        where: dict[str, any] | None = None
        if job_conditions:
            if len(job_conditions) == 1:
                where = job_conditions[0]
            else:
                where = {"$or": job_conditions}

        async with self._lock:
            results = await asyncio.to_thread(
                self._collection.query,
                query_texts=[query],
                n_results=n_results,
                where=where,
            )

        episodes: list[Episode] = []

        # ジョブが参照する共有グループIDを取得
        allowed_shared_groups: set[str] = set()
        if job_id and include_shared:
            job_config = self._memory_store._job_configs.get(job_id)
            if job_config and job_config.shared_group_ids:
                allowed_shared_groups = set(job_config.shared_group_ids)

        if results and results.get("ids") and results["ids"][0]:
            for i, episode_id in enumerate(results["ids"][0]):
                summary = results["documents"][0][i] if results.get("documents") else ""
                metadata = results["metadatas"][0][i] if results.get("metadatas") else {}

                episode = Episode.from_metadata(
                    id=episode_id,
                    summary=summary,
                    metadata=metadata,
                )

                # 共有エピソードの場合、shared_group_idsの所有権を検証
                if episode.memory_type == "shared" and include_shared:
                    episode_group_ids = set(episode.shared_group_ids)
                    if not episode_group_ids.intersection(allowed_shared_groups):
                        continue  # このジョブが参照していない共有グループのエピソードはスキップ

                episodes.append(episode)

        return episodes

    async def get_episode_by_id(
        self,
        episode_id: str,
        # Phase 7: ジョブ分離
        job_id: str | None = None,
        include_global: bool = True,
        include_shared: bool = True,
    ) -> Episode | None:
        """エピソードIDから取得.

        Args:
            episode_id: エピソードID
            job_id: ジョブ固有エピソードを検索する場合のジョブID（所有権検証用）
            include_global: グローバルエピソードを含めるか
            include_shared: 共有エピソードを含めるか

        Returns:
            Episode、見つからなければNone、所有権がない場合もNone
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

        episode = Episode.from_metadata(
            id=episode_id,
            summary=summary,
            metadata=metadata,
        )

        # Phase 7: ジョブ分離 - 所有権検証
        if job_id:
            if episode.memory_type == "global":
                if not include_global:
                    return None
            elif episode.memory_type == "job":
                if episode.job_id != job_id:
                    return None
            elif episode.memory_type == "shared":
                if not include_shared:
                    return None
                job_config = self._memory_store._job_configs.get(job_id)
                if job_config and job_config.shared_group_ids:
                    episode_group_ids = set(episode.shared_group_ids)
                    job_group_ids = set(job_config.shared_group_ids)
                    if not episode_group_ids.intersection(job_group_ids):
                        return None
                else:
                    return None

        return episode

    async def get_episode_memories(
        self,
        episode_id: str,
        # Phase 7: ジョブ分離
        job_id: str | None = None,
        include_global: bool = True,
        include_shared: bool = True,
    ) -> list:
        """エピソードに含まれる記憶を時系列順で取得.

        Args:
            episode_id: エピソードID
            job_id: ジョブ固有エピソードを検索する場合のジョブID（所有権検証用）
            include_global: グローバルエピソードを含めるか
            include_shared: 共有エピソードを含めるか

        Returns:
            記憶のリスト（時系列順）

        Raises:
            ValueError: エピソードが見つからない場合、または所有権がない場合
        """
        episode = await self.get_episode_by_id(
            episode_id,
            job_id=job_id,
            include_global=include_global,
            include_shared=include_shared,
        )
        if episode is None:
            raise ValueError(f"Episode not found: {episode_id}")

        # 記憶を取得（ジョブ分離適用）
        memories = await self._memory_store.get_by_ids(
            list(episode.memory_ids),
            job_id=job_id,
            include_global=include_global,
            include_shared=include_shared,
        )

        # 時系列順にソート
        memories.sort(key=lambda m: m.timestamp)

        return memories

    async def list_all_episodes(
        self,
        # Phase 7: ジョブ分離
        job_id: str | None = None,
        include_global: bool = True,
        include_shared: bool = True,
    ) -> list[Episode]:
        """全エピソードを取得.

        Args:
            job_id: ジョブ固有エピソードを検索する場合のジョブID
            include_global: グローバルエピソードを含めるか
            include_shared: 共有エピソードを含めるか

        Returns:
            全エピソードのリスト（新しい順）
        """
        # Phase 7: ジョブ分離 - 検索対象のエピソードタイプを制御
        job_conditions: list[dict[str, any]] = []

        if include_global:
            job_conditions.append({"memory_type": {"$eq": "global"}})

        if job_id:
            job_conditions.append(
                {
                    "$and": [
                        {"memory_type": {"$eq": "job"}},
                        {"job_id": {"$eq": job_id}},
                    ]
                }
            )

            if include_shared:
                job_conditions.append({"memory_type": {"$eq": "shared"}})

        where: dict[str, any] | None = None
        if job_conditions:
            if len(job_conditions) == 1:
                where = job_conditions[0]
            else:
                where = {"$or": job_conditions}

        async with self._lock:
            results = await asyncio.to_thread(
                self._collection.get,
                where=where,
            )

        episodes: list[Episode] = []

        # ジョブが参照する共有グループIDを取得
        allowed_shared_groups: set[str] = set()
        if job_id and include_shared:
            job_config = self._memory_store._job_configs.get(job_id)
            if job_config and job_config.shared_group_ids:
                allowed_shared_groups = set(job_config.shared_group_ids)

        if results and results.get("ids"):
            for i, episode_id in enumerate(results["ids"]):
                summary = results["documents"][i] if results.get("documents") else ""
                metadata = results["metadatas"][i] if results.get("metadatas") else {}

                episode = Episode.from_metadata(
                    id=episode_id,
                    summary=summary,
                    metadata=metadata,
                )

                # 共有エピソードの場合、shared_group_idsの所有権を検証
                if episode.memory_type == "shared" and include_shared:
                    episode_group_ids = set(episode.shared_group_ids)
                    if not episode_group_ids.intersection(allowed_shared_groups):
                        continue  # このジョブが参照していない共有グループのエピソードはスキップ

                episodes.append(episode)

        # 開始時刻で降順ソート（新しい順）
        episodes.sort(key=lambda e: e.start_time, reverse=True)

        return episodes

    async def delete_episode(
        self,
        episode_id: str,
        # Phase 7: ジョブ分離
        job_id: str | None = None,
        include_global: bool = True,
        include_shared: bool = True,
    ) -> bool:
        """エピソードを削除（記憶は削除しない）.

        Args:
            episode_id: 削除するエピソードID
            job_id: ジョブ固有エピソードを削除する場合のジョブID（所有権検証用）
            include_global: グローバルエピソードを含めるか
            include_shared: 共有エピソードを含めるか

        Returns:
            削除成功時はTrue、所有権がないまたは見つからない場合はFalse
        """
        # エピソードを取得（所有権検証）
        episode = await self.get_episode_by_id(
            episode_id,
            job_id=job_id,
            include_global=include_global,
            include_shared=include_shared,
        )
        if episode is None:
            return False

        # エピソードに含まれる記憶のepisode_idをクリア
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

        return True
