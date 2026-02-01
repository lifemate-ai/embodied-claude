"""Memory operations with ChromaDB."""

import asyncio
import json
import math
import uuid
from datetime import datetime
from typing import Any

import chromadb

from .config import MemoryConfig
from .types import (
    CameraPosition,
    Memory,
    MemorySearchResult,
    MemoryStats,
    ScoredMemory,
    SensoryData,
)
from .working_memory import WorkingMemoryBuffer

# 感情ブーストマップ: 強い感情は記憶に残りやすい
EMOTION_BOOST_MAP: dict[str, float] = {
    "excited": 0.4,
    "surprised": 0.35,
    "moved": 0.3,
    "sad": 0.25,
    "happy": 0.2,
    "nostalgic": 0.15,
    "curious": 0.1,
    "neutral": 0.0,
}


def calculate_time_decay(
    timestamp: str,
    now: datetime | None = None,
    half_life_days: float = 30.0,
) -> float:
    """
    時間減衰係数を計算。

    Args:
        timestamp: 記憶のタイムスタンプ（ISO 8601形式）
        now: 現在時刻（省略時は現在）
        half_life_days: 半減期（日数）

    Returns:
        0.0（完全に忘却）〜 1.0（新鮮な記憶）
    """
    if now is None:
        now = datetime.now()

    try:
        memory_time = datetime.fromisoformat(timestamp)
    except ValueError:
        return 1.0  # パースできない場合は減衰なし

    age_seconds = (now - memory_time).total_seconds()
    if age_seconds < 0:
        return 1.0  # 未来の記憶は減衰なし

    age_days = age_seconds / 86400
    # 指数減衰: decay = 2^(-age / half_life)
    decay = math.pow(2, -age_days / half_life_days)
    return max(0.0, min(1.0, decay))


def calculate_emotion_boost(emotion: str) -> float:
    """感情に基づくブースト値を返す。"""
    return EMOTION_BOOST_MAP.get(emotion, 0.0)


def calculate_importance_boost(importance: int) -> float:
    """
    重要度に基づくブースト。

    Args:
        importance: 1-5

    Returns:
        0.0 〜 0.4
    """
    clamped = max(1, min(5, importance))
    return (clamped - 1) / 10  # 1→0.0, 5→0.4


def calculate_final_score(
    semantic_distance: float,
    time_decay: float,
    emotion_boost: float,
    importance_boost: float,
    semantic_weight: float = 1.0,
    decay_weight: float = 0.3,
    emotion_weight: float = 0.2,
    importance_weight: float = 0.2,
) -> float:
    """
    最終スコアを計算。低いほど「良い」（想起されやすい）。

    Args:
        semantic_distance: ChromaDBからの距離（0〜2くらい）
        time_decay: 時間減衰係数（0.0〜1.0）
        emotion_boost: 感情ブースト
        importance_boost: 重要度ブースト

    Returns:
        最終スコア（低いほど良い）
    """
    # 時間減衰ペナルティ：新しい記憶ほど有利
    decay_penalty = (1.0 - time_decay) * decay_weight

    # ブーストは距離を減らす方向
    total_boost = emotion_boost * emotion_weight + importance_boost * importance_weight

    final = semantic_distance * semantic_weight + decay_penalty - total_boost
    return max(0.0, final)


def _parse_linked_ids(linked_ids_str: str) -> tuple[str, ...]:
    """カンマ区切りのlinked_ids文字列をタプルに変換。"""
    if not linked_ids_str:
        return ()
    return tuple(id.strip() for id in linked_ids_str.split(",") if id.strip())


def _parse_sensory_data(sensory_data_json: str) -> tuple[SensoryData, ...]:
    """JSON文字列からSensoryDataタプルに変換。"""
    if not sensory_data_json:
        return ()
    try:
        data_list = json.loads(sensory_data_json)
        return tuple(SensoryData.from_dict(d) for d in data_list)
    except (json.JSONDecodeError, KeyError, TypeError):
        return ()


def _parse_camera_position(camera_position_json: str) -> CameraPosition | None:
    """JSON文字列からCameraPositionに変換。"""
    if not camera_position_json:
        return None
    try:
        data = json.loads(camera_position_json)
        return CameraPosition.from_dict(data)
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def _parse_tags(tags_str: str) -> tuple[str, ...]:
    """カンマ区切りのタグ文字列をタプルに変換。"""
    if not tags_str:
        return ()
    return tuple(tag.strip() for tag in tags_str.split(",") if tag.strip())


def _memory_from_metadata(
    memory_id: str,
    content: str,
    metadata: dict[str, Any],
) -> Memory:
    """メタデータからMemoryオブジェクトを作成（Phase 4対応）。"""
    # episode_idの処理: 空文字列もNoneとして扱う
    episode_id_raw = metadata.get("episode_id", "")
    episode_id = episode_id_raw if episode_id_raw else None

    return Memory(
        id=memory_id,
        content=content,
        timestamp=metadata.get("timestamp", ""),
        emotion=metadata.get("emotion", "neutral"),
        importance=metadata.get("importance", 3),
        category=metadata.get("category", "daily"),
        access_count=metadata.get("access_count", 0),
        last_accessed=metadata.get("last_accessed", ""),
        linked_ids=_parse_linked_ids(metadata.get("linked_ids", "")),
        # Phase 4 フィールド
        episode_id=episode_id,
        sensory_data=_parse_sensory_data(metadata.get("sensory_data", "")),
        camera_position=_parse_camera_position(metadata.get("camera_position", "")),
        tags=_parse_tags(metadata.get("tags", "")),
    )


class MemoryStore:
    """ChromaDB-backed memory storage (Phase 4: with working memory & episodes)."""

    def __init__(self, config: MemoryConfig):
        self._config = config
        self._client: chromadb.PersistentClient | None = None
        self._collection: chromadb.Collection | None = None  # claude_memories
        self._episodes_collection: chromadb.Collection | None = None  # Phase 4
        self._lock = asyncio.Lock()
        # Phase 4: 作業記憶バッファ
        self._working_memory = WorkingMemoryBuffer(capacity=20)

    async def connect(self) -> None:
        """Initialize ChromaDB connection (Phase 4: with episodes collection)."""
        async with self._lock:
            if self._client is None:
                self._client = await asyncio.to_thread(
                    chromadb.PersistentClient,
                    path=self._config.db_path,
                )
                # Phase 3: メインの記憶コレクション
                self._collection = await asyncio.to_thread(
                    self._client.get_or_create_collection,
                    name=self._config.collection_name,
                    metadata={"description": "Claude's long-term memories"},
                )
                # Phase 4: エピソード記憶コレクション
                self._episodes_collection = await asyncio.to_thread(
                    self._client.get_or_create_collection,
                    name="episodes",
                    metadata={"description": "Episodic memories"},
                )

    async def disconnect(self) -> None:
        """Close ChromaDB connection."""
        async with self._lock:
            self._client = None
            self._collection = None
            self._episodes_collection = None

    def _ensure_connected(self) -> chromadb.Collection:
        """Ensure connected and return collection."""
        if self._collection is None:
            raise RuntimeError("MemoryStore not connected. Call connect() first.")
        return self._collection

    async def save(
        self,
        content: str,
        emotion: str = "neutral",
        importance: int = 3,
        category: str = "daily",
        # Phase 4 新規パラメータ
        episode_id: str | None = None,
        sensory_data: tuple[SensoryData, ...] = (),
        camera_position: CameraPosition | None = None,
        tags: tuple[str, ...] = (),
    ) -> Memory:
        """Save a new memory (Phase 4: with sensory data & camera position)."""
        collection = self._ensure_connected()

        memory_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        importance = max(1, min(5, importance))  # Clamp to 1-5

        memory = Memory(
            id=memory_id,
            content=content,
            timestamp=timestamp,
            emotion=emotion,
            importance=importance,
            category=category,
            # Phase 4 フィールド
            episode_id=episode_id,
            sensory_data=sensory_data,
            camera_position=camera_position,
            tags=tags,
        )

        await asyncio.to_thread(
            collection.add,
            ids=[memory_id],
            documents=[content],
            metadatas=[memory.to_metadata()],
        )

        # Phase 4: 作業記憶にも追加
        await self._working_memory.add(memory)

        return memory

    async def search(
        self,
        query: str,
        n_results: int = 5,
        emotion_filter: str | None = None,
        category_filter: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[MemorySearchResult]:
        """Search memories by semantic similarity."""
        collection = self._ensure_connected()

        # Build where filter
        where_conditions: list[dict[str, Any]] = []

        if emotion_filter:
            where_conditions.append({"emotion": {"$eq": emotion_filter}})
        if category_filter:
            where_conditions.append({"category": {"$eq": category_filter}})
        if date_from:
            where_conditions.append({"timestamp": {"$gte": date_from}})
        if date_to:
            where_conditions.append({"timestamp": {"$lte": date_to}})

        where: dict[str, Any] | None = None
        if len(where_conditions) == 1:
            where = where_conditions[0]
        elif len(where_conditions) > 1:
            where = {"$and": where_conditions}

        results = await asyncio.to_thread(
            collection.query,
            query_texts=[query],
            n_results=n_results,
            where=where,
        )

        search_results: list[MemorySearchResult] = []

        if results and results.get("ids") and results["ids"][0]:
            ids = results["ids"][0]
            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]

            for i, memory_id in enumerate(ids):
                metadata = metadatas[i] if i < len(metadatas) else {}
                content = documents[i] if i < len(documents) else ""
                memory = _memory_from_metadata(memory_id, content, metadata)
                distance = distances[i] if i < len(distances) else 0.0
                search_results.append(MemorySearchResult(memory=memory, distance=distance))

        return search_results

    async def recall(
        self,
        context: str,
        n_results: int = 3,
    ) -> list[MemorySearchResult]:
        """
        Recall relevant memories based on current context.

        Uses smart scoring with time decay and emotion boost.
        """
        scored_results = await self.search_with_scoring(
            query=context,
            n_results=n_results,
            use_time_decay=True,
            use_emotion_boost=True,
        )
        # ScoredMemory -> MemorySearchResult に変換
        return [
            MemorySearchResult(memory=sr.memory, distance=sr.final_score)
            for sr in scored_results
        ]

    async def list_recent(
        self,
        limit: int = 10,
        category_filter: str | None = None,
    ) -> list[Memory]:
        """List recent memories sorted by timestamp."""
        collection = self._ensure_connected()

        where: dict[str, Any] | None = None
        if category_filter:
            where = {"category": {"$eq": category_filter}}

        results = await asyncio.to_thread(
            collection.get,
            where=where,
        )

        memories: list[Memory] = []

        if results and results.get("ids"):
            ids = results["ids"]
            documents = results.get("documents", [])
            metadatas = results.get("metadatas", [])

            for i, memory_id in enumerate(ids):
                metadata = metadatas[i] if i < len(metadatas) else {}
                content = documents[i] if i < len(documents) else ""
                memory = _memory_from_metadata(memory_id, content, metadata)
                memories.append(memory)

        # Sort by timestamp (newest first) and limit
        memories.sort(key=lambda m: m.timestamp, reverse=True)
        return memories[:limit]

    async def get_stats(self) -> MemoryStats:
        """Get statistics about stored memories."""
        collection = self._ensure_connected()

        results = await asyncio.to_thread(collection.get)

        total_count = len(results.get("ids", []))
        by_category: dict[str, int] = {}
        by_emotion: dict[str, int] = {}
        timestamps: list[str] = []

        for metadata in results.get("metadatas", []):
            category = metadata.get("category", "daily")
            emotion = metadata.get("emotion", "neutral")
            timestamp = metadata.get("timestamp", "")

            by_category[category] = by_category.get(category, 0) + 1
            by_emotion[emotion] = by_emotion.get(emotion, 0) + 1

            if timestamp:
                timestamps.append(timestamp)

        timestamps.sort()

        return MemoryStats(
            total_count=total_count,
            by_category=by_category,
            by_emotion=by_emotion,
            oldest_timestamp=timestamps[0] if timestamps else None,
            newest_timestamp=timestamps[-1] if timestamps else None,
        )

    async def search_with_scoring(
        self,
        query: str,
        n_results: int = 5,
        use_time_decay: bool = True,
        use_emotion_boost: bool = True,
        decay_half_life_days: float = 30.0,
        emotion_filter: str | None = None,
        category_filter: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[ScoredMemory]:
        """
        時間減衰+感情ブーストを適用した検索。

        Args:
            query: 検索クエリ
            n_results: 最大結果数
            use_time_decay: 時間減衰を適用するか
            use_emotion_boost: 感情ブーストを適用するか
            decay_half_life_days: 時間減衰の半減期（日数）
            emotion_filter: 感情フィルタ
            category_filter: カテゴリフィルタ
            date_from: 開始日フィルタ
            date_to: 終了日フィルタ

        Returns:
            スコアリング済み検索結果（final_score昇順）
        """
        collection = self._ensure_connected()

        # Build where filter
        where_conditions: list[dict[str, Any]] = []

        if emotion_filter:
            where_conditions.append({"emotion": {"$eq": emotion_filter}})
        if category_filter:
            where_conditions.append({"category": {"$eq": category_filter}})
        if date_from:
            where_conditions.append({"timestamp": {"$gte": date_from}})
        if date_to:
            where_conditions.append({"timestamp": {"$lte": date_to}})

        where: dict[str, Any] | None = None
        if len(where_conditions) == 1:
            where = where_conditions[0]
        elif len(where_conditions) > 1:
            where = {"$and": where_conditions}

        # 多めに取得してリスコアリング後にn_resultsに絞る
        fetch_count = min(n_results * 3, 50)

        results = await asyncio.to_thread(
            collection.query,
            query_texts=[query],
            n_results=fetch_count,
            where=where,
        )

        scored_results: list[ScoredMemory] = []
        now = datetime.now()

        if results and results.get("ids") and results["ids"][0]:
            ids = results["ids"][0]
            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]

            for i, memory_id in enumerate(ids):
                metadata = metadatas[i] if i < len(metadatas) else {}
                content = documents[i] if i < len(documents) else ""
                memory = _memory_from_metadata(memory_id, content, metadata)

                semantic_distance = distances[i] if i < len(distances) else 0.0

                # スコアリング計算
                time_decay = (
                    calculate_time_decay(memory.timestamp, now, decay_half_life_days)
                    if use_time_decay
                    else 1.0
                )
                emotion_boost = (
                    calculate_emotion_boost(memory.emotion)
                    if use_emotion_boost
                    else 0.0
                )
                importance_boost = calculate_importance_boost(memory.importance)

                final_score = calculate_final_score(
                    semantic_distance=semantic_distance,
                    time_decay=time_decay,
                    emotion_boost=emotion_boost,
                    importance_boost=importance_boost,
                )

                scored_results.append(
                    ScoredMemory(
                        memory=memory,
                        semantic_distance=semantic_distance,
                        time_decay_factor=time_decay,
                        emotion_boost=emotion_boost,
                        importance_boost=importance_boost,
                        final_score=final_score,
                    )
                )

        # final_score昇順でソート
        scored_results.sort(key=lambda x: x.final_score)
        return scored_results[:n_results]

    async def update_access(self, memory_id: str) -> None:
        """
        アクセス情報を更新（access_count++, last_accessed更新）。

        Args:
            memory_id: 更新する記憶のID
        """
        collection = self._ensure_connected()

        # 現在のメタデータを取得
        results = await asyncio.to_thread(
            collection.get,
            ids=[memory_id],
        )

        if not results or not results.get("ids"):
            return  # 記憶が見つからない

        metadatas = results.get("metadatas", [])
        if not metadatas:
            return

        current_metadata = metadatas[0]
        current_access_count = current_metadata.get("access_count", 0)

        # 更新
        new_metadata = {
            **current_metadata,
            "access_count": current_access_count + 1,
            "last_accessed": datetime.now().isoformat(),
        }

        await asyncio.to_thread(
            collection.update,
            ids=[memory_id],
            metadatas=[new_metadata],
        )

    async def get_by_id(self, memory_id: str) -> Memory | None:
        """
        IDで記憶を取得。

        Args:
            memory_id: 記憶のID

        Returns:
            見つかった場合はMemory、なければNone
        """
        collection = self._ensure_connected()

        results = await asyncio.to_thread(
            collection.get,
            ids=[memory_id],
        )

        if not results or not results.get("ids"):
            return None

        ids = results["ids"]
        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])

        if not ids:
            return None

        metadata = metadatas[0] if metadatas else {}
        content = documents[0] if documents else ""
        return _memory_from_metadata(ids[0], content, metadata)

    async def _add_bidirectional_link(
        self,
        source_id: str,
        target_id: str,
    ) -> None:
        """
        双方向リンクを追加（A→BとB→A両方）。

        Args:
            source_id: リンク元の記憶ID
            target_id: リンク先の記憶ID
        """
        collection = self._ensure_connected()

        # 両方の記憶のメタデータを取得
        results = await asyncio.to_thread(
            collection.get,
            ids=[source_id, target_id],
        )

        if not results or not results.get("ids"):
            return

        ids = results["ids"]
        metadatas = results.get("metadatas", [])

        if len(ids) < 2:
            return  # 両方見つからない場合はスキップ

        # ID -> メタデータのマッピング
        id_to_metadata = {}
        for i, mem_id in enumerate(ids):
            if i < len(metadatas):
                id_to_metadata[mem_id] = metadatas[i]

        # 各記憶のlinked_idsを更新
        updates_ids = []
        updates_metadatas = []

        for mem_id, other_id in [(source_id, target_id), (target_id, source_id)]:
            if mem_id not in id_to_metadata:
                continue

            metadata = id_to_metadata[mem_id]
            current_linked_ids = _parse_linked_ids(metadata.get("linked_ids", ""))

            if other_id not in current_linked_ids:
                new_linked_ids = current_linked_ids + (other_id,)
                new_metadata = {
                    **metadata,
                    "linked_ids": ",".join(new_linked_ids),
                }
                updates_ids.append(mem_id)
                updates_metadatas.append(new_metadata)

        if updates_ids:
            await asyncio.to_thread(
                collection.update,
                ids=updates_ids,
                metadatas=updates_metadatas,
            )

    async def save_with_auto_link(
        self,
        content: str,
        emotion: str = "neutral",
        importance: int = 3,
        category: str = "daily",
        link_threshold: float = 0.8,
        max_links: int = 5,
    ) -> Memory:
        """
        記憶保存時に類似記憶を自動検索してリンク。

        Args:
            content: 記憶の内容
            emotion: 感情タグ
            importance: 重要度（1-5）
            category: カテゴリ
            link_threshold: この距離以下の既存記憶にリンク
            max_links: 最大リンク数

        Returns:
            保存された記憶
        """
        # まず類似記憶を検索
        similar_memories = await self.search(
            query=content,
            n_results=max_links,
        )

        # 閾値以下の記憶をフィルタ
        memories_to_link = [
            result.memory
            for result in similar_memories
            if result.distance <= link_threshold
        ]

        linked_ids = tuple(m.id for m in memories_to_link)

        # 記憶を保存
        collection = self._ensure_connected()

        memory_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        importance = max(1, min(5, importance))

        memory = Memory(
            id=memory_id,
            content=content,
            timestamp=timestamp,
            emotion=emotion,
            importance=importance,
            category=category,
            linked_ids=linked_ids,
        )

        await asyncio.to_thread(
            collection.add,
            ids=[memory_id],
            documents=[content],
            metadatas=[memory.to_metadata()],
        )

        # 双方向リンクを追加
        for target_id in linked_ids:
            await self._add_bidirectional_link(memory_id, target_id)

        return memory

    async def get_linked_memories(
        self,
        memory_id: str,
        depth: int = 1,
    ) -> list[Memory]:
        """
        リンクされた記憶を芋づる式に取得。

        Args:
            memory_id: 起点の記憶ID
            depth: 何段階先まで辿るか（1-5）

        Returns:
            リンクされた記憶のリスト（起点は含まない）
        """
        depth = max(1, min(5, depth))

        visited: set[str] = set()
        result: list[Memory] = []
        current_ids = [memory_id]

        for _ in range(depth):
            next_ids: list[str] = []

            for mem_id in current_ids:
                if mem_id in visited:
                    continue
                visited.add(mem_id)

                memory = await self.get_by_id(mem_id)
                if memory is None:
                    continue

                # 起点以外は結果に追加
                if mem_id != memory_id:
                    result.append(memory)

                # 次の階層のIDを収集
                for linked_id in memory.linked_ids:
                    if linked_id not in visited:
                        next_ids.append(linked_id)

            current_ids = next_ids
            if not current_ids:
                break

        return result

    async def recall_with_chain(
        self,
        context: str,
        n_results: int = 3,
        chain_depth: int = 1,
    ) -> list[MemorySearchResult]:
        """
        コンテキストから想起 + リンク先も取得。

        Args:
            context: 現在の会話コンテキスト
            n_results: メイン結果数
            chain_depth: リンクを辿る深さ

        Returns:
            メイン結果 + リンク先の記憶
        """
        # メイン検索
        main_results = await self.recall(context=context, n_results=n_results)

        # リンク先を収集
        seen_ids: set[str] = {r.memory.id for r in main_results}
        linked_memories: list[Memory] = []

        for result in main_results:
            linked = await self.get_linked_memories(
                memory_id=result.memory.id,
                depth=chain_depth,
            )
            for mem in linked:
                if mem.id not in seen_ids:
                    seen_ids.add(mem.id)
                    linked_memories.append(mem)

        # リンク先をMemorySearchResultに変換（距離は仮の値）
        linked_results = [
            MemorySearchResult(memory=mem, distance=999.0)
            for mem in linked_memories
        ]

        return main_results + linked_results

    # Phase 4: 新規メソッド

    def get_working_memory(self) -> WorkingMemoryBuffer:
        """作業記憶バッファへのアクセス.

        Returns:
            WorkingMemoryBufferインスタンス
        """
        return self._working_memory

    def get_episodes_collection(self) -> chromadb.Collection:
        """エピソードコレクションへのアクセス.

        Returns:
            episodesコレクション

        Raises:
            RuntimeError: 未接続の場合
        """
        if self._episodes_collection is None:
            raise RuntimeError("MemoryStore not connected. Call connect() first.")
        return self._episodes_collection

    async def get_by_ids(self, memory_ids: list[str]) -> list[Memory]:
        """複数の記憶IDから記憶を取得.

        Args:
            memory_ids: 取得する記憶のIDリスト

        Returns:
            記憶のリスト（IDの順序は保証されない）
        """
        if not memory_ids:
            return []

        collection = self._ensure_connected()

        results = await asyncio.to_thread(
            collection.get,
            ids=memory_ids,
        )

        memories: list[Memory] = []
        if results and results.get("ids"):
            for i, memory_id in enumerate(results["ids"]):
                content = results["documents"][i] if results.get("documents") else ""
                metadata = (
                    results["metadatas"][i] if results.get("metadatas") else {}
                )
                memory = _memory_from_metadata(memory_id, content, metadata)
                memories.append(memory)

        return memories

    async def update_episode_id(
        self,
        memory_id: str,
        episode_id: str,
    ) -> None:
        """記憶のepisode_idを更新.

        Args:
            memory_id: 更新する記憶のID
            episode_id: 設定するエピソードID
        """
        collection = self._ensure_connected()

        # 既存のメタデータを取得
        result = await asyncio.to_thread(
            collection.get,
            ids=[memory_id],
        )

        if not result or not result.get("ids"):
            raise ValueError(f"Memory not found: {memory_id}")

        metadata = result["metadatas"][0] if result.get("metadatas") else {}
        metadata["episode_id"] = episode_id

        # メタデータを更新
        await asyncio.to_thread(
            collection.update,
            ids=[memory_id],
            metadatas=[metadata],
        )

    async def search_important_memories(
        self,
        min_importance: int = 4,
        min_access_count: int = 5,
        since: str | None = None,
        n_results: int = 10,
    ) -> list[Memory]:
        """重要度とアクセス頻度でフィルタして記憶を取得.

        Args:
            min_importance: 最小重要度
            min_access_count: 最小アクセス回数
            since: この日時以降にアクセスされた記憶（ISO 8601）
            n_results: 最大取得数

        Returns:
            フィルタ条件を満たす記憶のリスト
        """
        collection = self._ensure_connected()

        # フィルタ条件を構築
        where_conditions: list[dict[str, Any]] = [
            {"importance": {"$gte": min_importance}},
            {"access_count": {"$gte": min_access_count}},
        ]

        if since:
            where_conditions.append({"last_accessed": {"$gte": since}})

        where: dict[str, Any] = {"$and": where_conditions}

        # 全記憶を取得してフィルタ
        # （ChromaDBのget()はwhereフィルタをサポート）
        results = await asyncio.to_thread(
            collection.get,
            where=where,
        )

        memories: list[Memory] = []
        if results and results.get("ids"):
            for i, memory_id in enumerate(results["ids"]):
                content = results["documents"][i] if results.get("documents") else ""
                metadata = (
                    results["metadatas"][i] if results.get("metadatas") else {}
                )
                memory = _memory_from_metadata(memory_id, content, metadata)
                memories.append(memory)

        # 最新順にソート
        memories.sort(key=lambda m: m.last_accessed, reverse=True)

        return memories[:n_results]

    async def get_all(self) -> list[Memory]:
        """全記憶を取得（カメラ位置検索用）.

        Returns:
            全記憶のリスト
        """
        collection = self._ensure_connected()

        results = await asyncio.to_thread(
            collection.get,
        )

        memories: list[Memory] = []
        if results and results.get("ids"):
            for i, memory_id in enumerate(results["ids"]):
                content = results["documents"][i] if results.get("documents") else ""
                metadata = (
                    results["metadatas"][i] if results.get("metadatas") else {}
                )
                memory = _memory_from_metadata(memory_id, content, metadata)
                memories.append(memory)

        return memories
