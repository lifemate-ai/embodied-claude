"""Memory operations with ChromaDB."""

import asyncio
import json
import logging
import math
import os
import uuid
from datetime import datetime
from typing import Any

import chromadb

from .association import (
    AssociationDiagnostics,
    AssociationEngine,
    adaptive_search_params,
)
from .config import MemoryConfig
from .consolidation import ConsolidationEngine
from .predictive import (
    PredictiveDiagnostics,
    calculate_context_relevance,
    calculate_novelty_score,
    calculate_prediction_error,
)
from .types import (
    CameraPosition,
    JobConfig,
    Memory,
    MemoryLink,
    MemorySearchResult,
    MemoryStats,
    ScoredMemory,
    SensoryData,
    SharedGroupConfig,
)
from .working_memory import WorkingMemoryBuffer
from .workspace import (
    WorkspaceCandidate,
    diversity_score,
    select_workspace_candidates,
)

logger = logging.getLogger(__name__)

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


def _parse_links(links_json: str) -> tuple[MemoryLink, ...]:
    """JSON文字列からMemoryLinkタプルに変換。"""
    if not links_json:
        return ()
    try:
        data_list = json.loads(links_json)
        return tuple(MemoryLink.from_dict(d) for d in data_list)
    except (json.JSONDecodeError, KeyError, TypeError):
        return ()


def _parse_coactivation_weights(coactivation_json: Any) -> tuple[tuple[str, float], ...]:
    """JSON文字列から共起重みタプルに変換。"""
    if not coactivation_json:
        return ()

    if isinstance(coactivation_json, dict):
        payload = coactivation_json
    else:
        try:
            payload = json.loads(coactivation_json)
        except (json.JSONDecodeError, TypeError):
            return ()

    if not isinstance(payload, dict):
        return ()

    weights: list[tuple[str, float]] = []
    for memory_id, weight in payload.items():
        if not isinstance(memory_id, str):
            continue
        try:
            value = float(weight)
        except (TypeError, ValueError):
            continue
        value = max(0.0, min(1.0, value))
        weights.append((memory_id, value))
    return tuple(weights)


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Best-effort float conversion."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    """Best-effort int conversion."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _memory_from_metadata(
    memory_id: str,
    content: str,
    metadata: dict[str, Any],
) -> Memory:
    """メタデータからMemoryオブジェクトを作成（Phase 7対応）。"""
    # episode_idの処理: 空文字列もNoneとして扱う
    episode_id_raw = metadata.get("episode_id", "")
    episode_id = episode_id_raw if episode_id_raw else None

    # Phase 7: ジョブ分離フィールドのパース
    memory_type = metadata.get("memory_type", "global")
    job_id_raw = metadata.get("job_id", "")
    job_id = job_id_raw if job_id_raw else None
    shared_group_ids_str = metadata.get("shared_group_ids", "")
    shared_group_ids = tuple(id.strip() for id in shared_group_ids_str.split(",") if id.strip())

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
        # Phase 5: 因果リンク
        links=_parse_links(metadata.get("links", "")),
        # Phase 6: 発散想起・予測符号化
        novelty_score=_safe_float(metadata.get("novelty_score", 0.0), 0.0),
        prediction_error=_safe_float(metadata.get("prediction_error", 0.0), 0.0),
        activation_count=_safe_int(metadata.get("activation_count", 0), 0),
        last_activated=metadata.get("last_activated", ""),
        coactivation_weights=_parse_coactivation_weights(metadata.get("coactivation", "")),
        # Phase 7: ジョブ分離
        memory_type=memory_type,
        job_id=job_id,
        shared_group_ids=shared_group_ids,
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
        # Phase 6: 連想・統合エンジン
        self._association_engine = AssociationEngine()
        self._consolidation_engine = ConsolidationEngine()
        # Phase 7: ジョブ分離 - ジョブ設定管理
        self._job_configs: dict[str, JobConfig] = {}
        self._shared_groups: dict[str, SharedGroupConfig] = {}
        self._job_config_path = os.path.join(os.path.dirname(config.db_path), "job_configs.json")

    async def connect(self) -> None:
        """Initialize ChromaDB connection (Phase 7: with job config loading)."""
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
                # Phase 7: ジョブ設定を読み込み
                await self._load_job_configs()

    async def disconnect(self) -> None:
        """Close ChromaDB connection (Phase 7: with job config saving)."""
        async with self._lock:
            # Phase 7: ジョブ設定を保存
            await self._save_job_configs()
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
        # Phase 7: ジョブ分離
        memory_type: str = "global",
        job_id: str | None = None,
        shared_group_ids: tuple[str, ...] = (),
    ) -> Memory:
        """Save a new memory (Phase 7: with job isolation)."""
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
            # Phase 7: ジョブ分離
            memory_type=memory_type,
            job_id=job_id,
            shared_group_ids=shared_group_ids,
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
        # Phase 7: ジョブ分離
        job_id: str | None = None,
        include_global: bool = True,
        include_shared: bool = True,
    ) -> list[MemorySearchResult]:
        """Search memories by semantic similarity (Phase 7: with job isolation)."""
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

        # Phase 7: ジョブ分離 - 検索対象の記憶タイプを制御
        job_conditions: list[dict[str, Any]] = []

        if include_global:
            job_conditions.append({"memory_type": {"$eq": "global"}})

        if job_id:
            # ジョブ固有記憶
            job_conditions.append(
                {
                    "$and": [
                        {"memory_type": {"$eq": "job"}},
                        {"job_id": {"$eq": job_id}},
                    ]
                }
            )

            if include_shared:
                # 共有メモリも検索対象に含める
                # ChromaDBの制約により、shared_group_idsの部分一致は後でフィルタリング
                job_conditions.append({"memory_type": {"$eq": "shared"}})

        if job_conditions:
            if len(job_conditions) == 1:
                where_conditions.append(job_conditions[0])
            else:
                where_conditions.append({"$or": job_conditions})

        where: dict[str, Any] | None = None
        if len(where_conditions) == 1:
            where = where_conditions[0]
        elif len(where_conditions) > 1:
            where = {"$and": where_conditions}

        # 多めに取得して、共有メモリのフィルタリング後にn_resultsに絞る
        fetch_count = n_results * 3 if (job_id and include_shared) else n_results

        results = await asyncio.to_thread(
            collection.query,
            query_texts=[query],
            n_results=fetch_count,
            where=where,
        )

        search_results: list[MemorySearchResult] = []

        if results and results.get("ids") and results["ids"][0]:
            ids = results["ids"][0]
            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]

            # ジョブが参照する共有グループIDを取得
            allowed_shared_groups: set[str] = set()
            if job_id and include_shared:
                job_config = self._job_configs.get(job_id)
                if job_config and job_config.shared_group_ids:
                    allowed_shared_groups = set(job_config.shared_group_ids)

            for i, memory_id in enumerate(ids):
                metadata = metadatas[i] if i < len(metadatas) else {}
                content = documents[i] if i < len(documents) else ""
                memory = _memory_from_metadata(memory_id, content, metadata)
                distance = distances[i] if i < len(distances) else 0.0

                # 共有メモリの場合、shared_group_idsの所有権を検証
                # job_idがNoneの場合も含め、常に所有権検証を行う
                if memory.memory_type == "shared" and include_shared:
                    memory_group_ids = set(memory.shared_group_ids)
                    if not memory_group_ids.intersection(allowed_shared_groups):
                        continue  # このジョブが参照していない共有グループの記憶はスキップ

                search_results.append(MemorySearchResult(memory=memory, distance=distance))

                if len(search_results) >= n_results:
                    break

        return search_results

    async def recall(
        self,
        context: str,
        n_results: int = 3,
        job_id: str | None = None,
        include_global: bool = True,
        include_shared: bool = True,
    ) -> list[MemorySearchResult]:
        """
        Recall relevant memories based on current context.

        Uses smart scoring with time decay and emotion boost.

        Args:
            context: 検索コンテキスト
            n_results: 最大結果数
            job_id: ジョブ固有メモリを検索する場合のジョブID
            include_global: グローバルメモリを含めるか
            include_shared: 共有メモリを含めるか
        """
        scored_results = await self.search_with_scoring(
            query=context,
            n_results=n_results,
            use_time_decay=True,
            use_emotion_boost=True,
            job_id=job_id,
            include_global=include_global,
            include_shared=include_shared,
        )
        # ScoredMemory -> MemorySearchResult に変換
        return [
            MemorySearchResult(memory=sr.memory, distance=sr.final_score) for sr in scored_results
        ]

    async def list_recent(
        self,
        limit: int = 10,
        category_filter: str | None = None,
        job_id: str | None = None,
        include_global: bool = True,
        include_shared: bool = True,
    ) -> list[Memory]:
        """
        List recent memories sorted by timestamp.

        Args:
            limit: 最大取得数
            category_filter: カテゴリフィルタ
            job_id: ジョブ固有メモリを検索する場合のジョブID
            include_global: グローバルメモリを含めるか
            include_shared: 共有メモリを含めるか
        """
        collection = self._ensure_connected()

        # Build where filter
        where_conditions: list[dict[str, Any]] = []

        if category_filter:
            where_conditions.append({"category": {"$eq": category_filter}})

        # Phase 7: ジョブ分離 - 検索対象の記憶タイプを制御
        job_conditions: list[dict[str, Any]] = []

        if include_global:
            job_conditions.append({"memory_type": {"$eq": "global"}})

        if job_id:
            # ジョブ固有記憶
            job_conditions.append(
                {
                    "$and": [
                        {"memory_type": {"$eq": "job"}},
                        {"job_id": {"$eq": job_id}},
                    ]
                }
            )

            if include_shared:
                # 共有メモリも検索対象に含める
                # ChromaDBの制約により、shared_group_idsの部分一致は後でフィルタリング
                job_conditions.append({"memory_type": {"$eq": "shared"}})

        if job_conditions:
            if len(job_conditions) == 1:
                where_conditions.append(job_conditions[0])
            else:
                where_conditions.append({"$or": job_conditions})

        where: dict[str, Any] | None = None
        if len(where_conditions) == 1:
            where = where_conditions[0]
        elif len(where_conditions) > 1:
            where = {"$and": where_conditions}

        results = await asyncio.to_thread(
            collection.get,
            where=where,
        )

        memories: list[Memory] = []

        # ジョブが参照する共有グループIDを取得
        allowed_shared_groups: set[str] = set()
        if job_id and include_shared:
            job_config = self._job_configs.get(job_id)
            if job_config and job_config.shared_group_ids:
                allowed_shared_groups = set(job_config.shared_group_ids)

        if results and results.get("ids"):
            ids = results["ids"]
            documents = results.get("documents", [])
            metadatas = results.get("metadatas", [])

            for i, memory_id in enumerate(ids):
                metadata = metadatas[i] if i < len(metadatas) else {}
                content = documents[i] if i < len(documents) else ""
                memory = _memory_from_metadata(memory_id, content, metadata)

                # 共有メモリの場合、shared_group_idsの所有権を検証
                # job_idがNoneの場合も含め、常に所有権検証を行う
                if memory.memory_type == "shared" and include_shared:
                    memory_group_ids = set(memory.shared_group_ids)
                    if not memory_group_ids.intersection(allowed_shared_groups):
                        continue  # このジョブが参照していない共有グループの記憶はスキップ

                memories.append(memory)

        # Sort by timestamp (newest first) and limit
        memories.sort(key=lambda m: m.timestamp, reverse=True)
        return memories[:limit]

    async def get_stats(
        self,
        job_id: str | None = None,
        include_global: bool = True,
        include_shared: bool = True,
    ) -> MemoryStats:
        """
        Get statistics about stored memories.

        Args:
            job_id: ジョブ固有メモリを検索する場合のジョブID
            include_global: グローバルメモリを含めるか
            include_shared: 共有メモリを含めるか
        """
        collection = self._ensure_connected()

        # Build where filter for job isolation
        where_conditions: list[dict[str, Any]] = []

        # Phase 7: ジョブ分離 - 検索対象の記憶タイプを制御
        job_conditions: list[dict[str, Any]] = []

        if include_global:
            job_conditions.append({"memory_type": {"$eq": "global"}})

        if job_id:
            # ジョブ固有記憶
            job_conditions.append(
                {
                    "$and": [
                        {"memory_type": {"$eq": "job"}},
                        {"job_id": {"$eq": job_id}},
                    ]
                }
            )

            if include_shared:
                # 共有メモリも検索対象に含める
                # ChromaDBの制約により、shared_group_idsの部分一致は後でフィルタリング
                job_conditions.append({"memory_type": {"$eq": "shared"}})

        where: dict[str, Any] | None = None
        if job_conditions:
            if len(job_conditions) == 1:
                where = job_conditions[0]
            else:
                where = {"$or": job_conditions}

        results = await asyncio.to_thread(collection.get, where=where)

        # ジョブが参照する共有グループIDを取得
        allowed_shared_groups: set[str] = set()
        if job_id and include_shared:
            job_config = self._job_configs.get(job_id)
            if job_config and job_config.shared_group_ids:
                allowed_shared_groups = set(job_config.shared_group_ids)

        total_count = 0
        by_category: dict[str, int] = {}
        by_emotion: dict[str, int] = {}
        timestamps: list[str] = []

        metadatas = results.get("metadatas", [])
        for metadata in metadatas:
            # 共有メモリの場合、shared_group_idsの所有権を検証
            # job_idがNoneの場合も含め、常に所有権検証を行う
            memory_type = metadata.get("memory_type", "global")
            if memory_type == "shared" and include_shared:
                shared_group_ids_str = metadata.get("shared_group_ids", "")
                memory_group_ids = set(
                    id.strip() for id in shared_group_ids_str.split(",") if id.strip()
                )
                if not memory_group_ids.intersection(allowed_shared_groups):
                    continue  # このジョブが参照していない共有グループの記憶はスキップ

            total_count += 1
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
        job_id: str | None = None,
        include_global: bool = True,
        include_shared: bool = True,
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
            job_id: ジョブ固有メモリを検索する場合のジョブID
            include_global: グローバルメモリを含めるか
            include_shared: 共有メモリを含めるか

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

        # Phase 7: ジョブ分離 - 検索対象の記憶タイプを制御
        job_conditions: list[dict[str, Any]] = []

        if include_global:
            job_conditions.append({"memory_type": {"$eq": "global"}})

        if job_id:
            # ジョブ固有記憶
            job_conditions.append(
                {
                    "$and": [
                        {"memory_type": {"$eq": "job"}},
                        {"job_id": {"$eq": job_id}},
                    ]
                }
            )

            if include_shared:
                # 共有メモリも検索対象に含める
                # ChromaDBの制約により、shared_group_idsの部分一致は後でフィルタリング
                job_conditions.append({"memory_type": {"$eq": "shared"}})

        if job_conditions:
            if len(job_conditions) == 1:
                where_conditions.append(job_conditions[0])
            else:
                where_conditions.append({"$or": job_conditions})

        where: dict[str, Any] | None = None
        if len(where_conditions) == 1:
            where = where_conditions[0]
        elif len(where_conditions) > 1:
            where = {"$and": where_conditions}

        # 多めに取得してリスコアリング後にn_resultsに絞る
        # 共有メモリのフィルタリングを考慮してさらに多めに取得
        fetch_count = min(n_results * 5 if (job_id and include_shared) else n_results * 3, 50)

        results = await asyncio.to_thread(
            collection.query,
            query_texts=[query],
            n_results=fetch_count,
            where=where,
        )

        scored_results: list[ScoredMemory] = []
        now = datetime.now()

        # ジョブが参照する共有グループIDを取得
        allowed_shared_groups: set[str] = set()
        if job_id and include_shared:
            job_config = self._job_configs.get(job_id)
            if job_config and job_config.shared_group_ids:
                allowed_shared_groups = set(job_config.shared_group_ids)

        if results and results.get("ids") and results["ids"][0]:
            ids = results["ids"][0]
            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]

            for i, memory_id in enumerate(ids):
                metadata = metadatas[i] if i < len(metadatas) else {}
                content = documents[i] if i < len(documents) else ""
                memory = _memory_from_metadata(memory_id, content, metadata)

                # 共有メモリの場合、shared_group_idsの所有権を検証
                # job_idがNoneの場合も含め、常に所有権検証を行う
                if memory.memory_type == "shared" and include_shared:
                    memory_group_ids = set(memory.shared_group_ids)
                    if not memory_group_ids.intersection(allowed_shared_groups):
                        continue  # このジョブが参照していない共有グループの記憶はスキップ

                semantic_distance = distances[i] if i < len(distances) else 0.0

                # スコアリング計算
                time_decay = (
                    calculate_time_decay(memory.timestamp, now, decay_half_life_days)
                    if use_time_decay
                    else 1.0
                )
                emotion_boost = (
                    calculate_emotion_boost(memory.emotion) if use_emotion_boost else 0.0
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

                if len(scored_results) >= n_results:
                    break

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

    async def get_by_id(
        self,
        memory_id: str,
        job_id: str | None = None,
        include_global: bool = True,
        include_shared: bool = True,
    ) -> Memory | None:
        """
        IDで記憶を取得。

        Args:
            memory_id: 記憶のID
            job_id: ジョブ固有メモリを検索する場合のジョブID（所有権検証用）
            include_global: グローバルメモリを含めるか
            include_shared: 共有メモリを含めるか

        Returns:
            見つかった場合はMemory、なければNone
            job_id指定時は、所有権がない場合もNoneを返す
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
        memory = _memory_from_metadata(ids[0], content, metadata)

        # Phase 7: ジョブ分離 - 所有権検証
        if job_id:
            # グローバルメモリは常に許可（include_globalがTrueの場合）
            if memory.memory_type == "global":
                if not include_global:
                    return None
            # ジョブ固有メモリは所有権チェック
            elif memory.memory_type == "job":
                if memory.job_id != job_id:
                    return None
            # 共有メモリはジョブが参照するグループに含まれるかチェック
            elif memory.memory_type == "shared":
                if not include_shared:
                    return None
                job_config = self._job_configs.get(job_id)
                if job_config and job_config.shared_group_ids:
                    # 共有グループIDのいずれかがメモリのshared_group_idsに含まれるか
                    memory_group_ids = set(memory.shared_group_ids)
                    job_group_ids = set(job_config.shared_group_ids)
                    if not memory_group_ids.intersection(job_group_ids):
                        return None
                else:
                    # ジョブが共有グループを参照していない場合はアクセス不可
                    return None

        return memory

    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory by ID.

        Args:
            memory_id: ID of the memory to delete

        Returns:
            True if deleted successfully, False if not found
        """
        collection = self._ensure_connected()

        try:
            # Check if memory exists
            result = await asyncio.to_thread(
                collection.get,
                ids=[memory_id],
            )

            if not result or not result.get("ids"):
                return False

            # Delete the memory
            await asyncio.to_thread(
                collection.delete,
                ids=[memory_id],
            )

            logger.info(f"Deleted memory: {memory_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete memory {memory_id}: {e}")
            return False

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
        # Phase 7: ジョブ分離
        memory_type: str = "global",
        job_id: str | None = None,
        shared_group_ids: tuple[str, ...] = (),
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
            memory_type: メモリタイプ ("global" | "job" | "shared")
            job_id: ジョブ固有メモリの場合のジョブID
            shared_group_ids: 共有メモリの場合の共有グループID

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
            result.memory for result in similar_memories if result.distance <= link_threshold
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
            # Phase 7: ジョブ分離
            memory_type=memory_type,
            job_id=job_id,
            shared_group_ids=shared_group_ids,
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
        job_id: str | None = None,
        include_global: bool = True,
        include_shared: bool = True,
    ) -> list[Memory]:
        """
        リンクされた記憶を芋づる式に取得。

        Args:
            memory_id: 起点の記憶ID
            depth: 何段階先まで辿るか（1-5）
            job_id: ジョブ固有メモリを検索する場合のジョブID（所有権検証用）
            include_global: グローバルメモリを含めるか
            include_shared: 共有メモリを含めるか

        Returns:
            リンクされた記憶のリスト（起点は含まない）
            job_id指定時は、所有権がある記憶のみ返す
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

                memory = await self.get_by_id(
                    mem_id,
                    job_id=job_id,
                    include_global=include_global,
                    include_shared=include_shared,
                )
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
        job_id: str | None = None,
        include_global: bool = True,
        include_shared: bool = True,
    ) -> list[MemorySearchResult]:
        """
        コンテキストから想起 + リンク先も取得。

        Args:
            context: 現在の会話コンテキスト
            n_results: メイン結果数
            chain_depth: リンクを辿る深さ
            job_id: ジョブ固有メモリを検索する場合のジョブID
            include_global: グローバルメモリを含めるか
            include_shared: 共有メモリを含めるか

        Returns:
            メイン結果 + リンク先の記憶
        """
        # メイン検索
        main_results = await self.recall(
            context=context,
            n_results=n_results,
            job_id=job_id,
            include_global=include_global,
            include_shared=include_shared,
        )

        # リンク先を収集
        seen_ids: set[str] = {r.memory.id for r in main_results}
        linked_memories: list[Memory] = []

        for result in main_results:
            linked = await self.get_linked_memories(
                memory_id=result.memory.id,
                depth=chain_depth,
                job_id=job_id,
                include_global=include_global,
                include_shared=include_shared,
            )
            for mem in linked:
                if mem.id not in seen_ids:
                    seen_ids.add(mem.id)
                    linked_memories.append(mem)

        # リンク先をMemorySearchResultに変換（距離は仮の値）
        linked_results = [MemorySearchResult(memory=mem, distance=999.0) for mem in linked_memories]

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

    async def get_by_ids(
        self,
        memory_ids: list[str],
        job_id: str | None = None,
        include_global: bool = True,
        include_shared: bool = True,
    ) -> list[Memory]:
        """複数の記憶IDから記憶を取得.

        Args:
            memory_ids: 取得する記憶のIDリスト
            job_id: ジョブ固有メモリを検索する場合のジョブID（所有権検証用）
            include_global: グローバルメモリを含めるか
            include_shared: 共有メモリを含めるか

        Returns:
            記憶のリスト（IDの順序は保証されない）
            job_id指定時は、所有権がある記憶のみ返す
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
                metadata = results["metadatas"][i] if results.get("metadatas") else {}
                memory = _memory_from_metadata(memory_id, content, metadata)

                # Phase 7: ジョブ分離 - 所有権検証
                if job_id:
                    # グローバルメモリ
                    if memory.memory_type == "global":
                        if not include_global:
                            continue
                    # ジョブ固有メモリ
                    elif memory.memory_type == "job":
                        if memory.job_id != job_id:
                            continue
                    # 共有メモリ
                    elif memory.memory_type == "shared":
                        if not include_shared:
                            continue
                        job_config = self._job_configs.get(job_id)
                        if job_config and job_config.shared_group_ids:
                            memory_group_ids = set(memory.shared_group_ids)
                            job_group_ids = set(job_config.shared_group_ids)
                            if not memory_group_ids.intersection(job_group_ids):
                                continue
                        else:
                            continue

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
        job_id: str | None = None,
        include_global: bool = True,
        include_shared: bool = True,
    ) -> list[Memory]:
        """重要度とアクセス頻度でフィルタして記憶を取得.

        Args:
            min_importance: 最小重要度
            min_access_count: 最小アクセス回数
            since: この日時以降にアクセスされた記憶（ISO 8601）
            n_results: 最大取得数
            job_id: ジョブ固有メモリを検索する場合のジョブID
            include_global: グローバルメモリを含めるか
            include_shared: 共有メモリを含めるか

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

        # Phase 7: ジョブ分離 - 検索対象の記憶タイプを制御
        job_conditions: list[dict[str, Any]] = []

        if include_global:
            job_conditions.append({"memory_type": {"$eq": "global"}})

        if job_id:
            # ジョブ固有記憶
            job_conditions.append(
                {
                    "$and": [
                        {"memory_type": {"$eq": "job"}},
                        {"job_id": {"$eq": job_id}},
                    ]
                }
            )

            if include_shared:
                # 共有メモリも検索対象に含める
                # ChromaDBの制約により、shared_group_idsの部分一致は後でフィルタリング
                job_conditions.append({"memory_type": {"$eq": "shared"}})

        if job_conditions:
            if len(job_conditions) == 1:
                where_conditions.append(job_conditions[0])
            else:
                where_conditions.append({"$or": job_conditions})

        where: dict[str, Any] = {"$and": where_conditions}

        # 全記憶を取得してフィルタ
        # （ChromaDBのget()はwhereフィルタをサポート）
        results = await asyncio.to_thread(
            collection.get,
            where=where,
        )

        memories: list[Memory] = []

        # ジョブが参照する共有グループIDを取得
        allowed_shared_groups: set[str] = set()
        if job_id and include_shared:
            job_config = self._job_configs.get(job_id)
            if job_config and job_config.shared_group_ids:
                allowed_shared_groups = set(job_config.shared_group_ids)

        if results and results.get("ids"):
            for i, memory_id in enumerate(results["ids"]):
                content = results["documents"][i] if results.get("documents") else ""
                metadata = results["metadatas"][i] if results.get("metadatas") else {}
                memory = _memory_from_metadata(memory_id, content, metadata)

                # 共有メモリの場合、shared_group_idsの所有権を検証
                # job_idがNoneの場合も含め、常に所有権検証を行う
                if memory.memory_type == "shared" and include_shared:
                    memory_group_ids = set(memory.shared_group_ids)
                    if not memory_group_ids.intersection(allowed_shared_groups):
                        continue  # このジョブが参照していない共有グループの記憶はスキップ

                memories.append(memory)

        # 最新順にソート
        memories.sort(key=lambda m: m.last_accessed, reverse=True)

        return memories[:n_results]

    async def get_all(
        self,
        job_id: str | None = None,
        include_global: bool = True,
        include_shared: bool = True,
    ) -> list[Memory]:
        """全記憶を取得（カメラ位置検索用）.

        Args:
            job_id: ジョブ固有メモリを検索する場合のジョブID
            include_global: グローバルメモリを含めるか
            include_shared: 共有メモリを含めるか

        Returns:
            記憶のリスト
        """
        collection = self._ensure_connected()

        # Phase 7: ジョブ分離 - 検索対象の記憶タイプを制御
        job_conditions: list[dict[str, Any]] = []

        if include_global:
            job_conditions.append({"memory_type": {"$eq": "global"}})

        if job_id:
            # ジョブ固有記憶
            job_conditions.append(
                {
                    "$and": [
                        {"memory_type": {"$eq": "job"}},
                        {"job_id": {"$eq": job_id}},
                    ]
                }
            )

            if include_shared:
                # 共有メモリも検索対象に含める
                # ChromaDBの制約により、shared_group_idsの部分一致は後でフィルタリング
                job_conditions.append({"memory_type": {"$eq": "shared"}})

        where: dict[str, Any] | None = None
        if job_conditions:
            if len(job_conditions) == 1:
                where = job_conditions[0]
            else:
                where = {"$or": job_conditions}

        results = await asyncio.to_thread(
            collection.get,
            where=where,
        )

        memories: list[Memory] = []

        # ジョブが参照する共有グループIDを取得
        allowed_shared_groups: set[str] = set()
        if job_id and include_shared:
            job_config = self._job_configs.get(job_id)
            if job_config and job_config.shared_group_ids:
                allowed_shared_groups = set(job_config.shared_group_ids)

        if results and results.get("ids"):
            for i, memory_id in enumerate(results["ids"]):
                content = results["documents"][i] if results.get("documents") else ""
                metadata = results["metadatas"][i] if results.get("metadatas") else {}
                memory = _memory_from_metadata(memory_id, content, metadata)

                # 共有メモリの場合、shared_group_idsの所有権を検証
                # job_idがNoneの場合も含め、常に所有権検証を行う
                if memory.memory_type == "shared" and include_shared:
                    memory_group_ids = set(memory.shared_group_ids)
                    if not memory_group_ids.intersection(allowed_shared_groups):
                        continue  # このジョブが参照していない共有グループの記憶はスキップ

                memories.append(memory)

        return memories

    # Phase 5: 因果リンク

    async def add_causal_link(
        self,
        source_id: str,
        target_id: str,
        link_type: str = "caused_by",
        note: str | None = None,
        job_id: str | None = None,
        include_global: bool = True,
        include_shared: bool = True,
    ) -> None:
        """因果リンクを追加（単方向）.

        Args:
            source_id: リンク元の記憶ID
            target_id: リンク先の記憶ID
            link_type: リンクタイプ ("caused_by", "leads_to", "related", "similar")
            note: リンクの説明（任意）
            job_id: ジョブ固有メモリを検索する場合のジョブID（所有権検証用）
            include_global: グローバルメモリを含めるか
            include_shared: 共有メモリを含めるか
        """
        collection = self._ensure_connected()

        # ソース記憶を取得（ジョブ分離適用）
        source_memory = await self.get_by_id(
            source_id,
            job_id=job_id,
            include_global=include_global,
            include_shared=include_shared,
        )
        if source_memory is None:
            raise ValueError(f"Source memory not found: {source_id}")

        # ターゲット記憶が存在するか確認（ジョブ分離適用）
        target_memory = await self.get_by_id(
            target_id,
            job_id=job_id,
            include_global=include_global,
            include_shared=include_shared,
        )
        if target_memory is None:
            raise ValueError(f"Target memory not found: {target_id}")

        # 新しいリンクを作成
        new_link = MemoryLink(
            target_id=target_id,
            link_type=link_type,
            created_at=datetime.now().isoformat(),
            note=note,
        )

        # 既存のリンクに追加（重複チェック）
        existing_links = list(source_memory.links)
        for link in existing_links:
            if link.target_id == target_id and link.link_type == link_type:
                return  # 既に同じリンクが存在

        updated_links = tuple(existing_links + [new_link])

        # メタデータを更新
        results = await asyncio.to_thread(
            collection.get,
            ids=[source_id],
        )

        if results and results.get("metadatas"):
            metadata = results["metadatas"][0]
            metadata["links"] = json.dumps([link.to_dict() for link in updated_links])

            await asyncio.to_thread(
                collection.update,
                ids=[source_id],
                metadatas=[metadata],
            )

    async def get_causal_chain(
        self,
        memory_id: str,
        direction: str = "backward",
        max_depth: int = 5,
        job_id: str | None = None,
        include_global: bool = True,
        include_shared: bool = True,
    ) -> list[tuple[Memory, str]]:
        """因果の連鎖を辿る.

        Args:
            memory_id: 起点の記憶ID
            direction: "backward" (原因を辿る) or "forward" (結果を辿る)
            max_depth: 最大深度（1-5）
            job_id: ジョブ固有メモリを検索する場合のジョブID（所有権検証用）
            include_global: グローバルメモリを含めるか
            include_shared: 共有メモリを含めるか

        Returns:
            [(Memory, link_type), ...] の形式
        """
        max_depth = max(1, min(5, max_depth))

        # 方向によって辿るリンクタイプを決定
        if direction == "backward":
            target_link_types = {"caused_by"}
        elif direction == "forward":
            target_link_types = {"leads_to"}
        else:
            raise ValueError(f"Invalid direction: {direction}")

        visited: set[str] = set()
        result: list[tuple[Memory, str]] = []
        current_ids = [memory_id]

        for _ in range(max_depth):
            next_ids: list[str] = []

            for mem_id in current_ids:
                if mem_id in visited:
                    continue
                visited.add(mem_id)

                memory = await self.get_by_id(
                    mem_id,
                    job_id=job_id,
                    include_global=include_global,
                    include_shared=include_shared,
                )
                if memory is None:
                    continue

                # 該当するリンクタイプのリンクを探す
                for link in memory.links:
                    if link.link_type in target_link_types:
                        target_memory = await self.get_by_id(
                            link.target_id,
                            job_id=job_id,
                            include_global=include_global,
                            include_shared=include_shared,
                        )
                        if target_memory and link.target_id not in visited:
                            result.append((target_memory, link.link_type))
                            next_ids.append(link.target_id)

            current_ids = next_ids
            if not current_ids:
                break

        return result

    # Phase 6: 発散的想起・統合

    async def update_memory_fields(
        self,
        memory_id: str,
        job_id: str | None = None,
        include_global: bool = True,
        include_shared: bool = True,
        **fields: Any,
    ) -> bool:
        """記憶メタデータの部分更新.

        Args:
            memory_id: 更新する記憶のID
            job_id: ジョブ固有メモリを検索する場合のジョブID（所有権検証用）
            include_global: グローバルメモリを含めるか
            include_shared: 共有メモリを含めるか
            **fields: 更新するフィールド

        Returns:
            更新成功時はTrue、失敗時はFalse
        """
        collection = self._ensure_connected()

        # ジョブ分離: 所有権を確認
        if job_id:
            memory = await self.get_by_id(
                memory_id,
                job_id=job_id,
                include_global=include_global,
                include_shared=include_shared,
            )
            if memory is None:
                return False

        result = await asyncio.to_thread(collection.get, ids=[memory_id])
        if not result or not result.get("ids"):
            return False

        metadata = result["metadatas"][0] if result.get("metadatas") else {}
        metadata.update(fields)
        await asyncio.to_thread(collection.update, ids=[memory_id], metadatas=[metadata])
        return True

    async def record_activation(
        self,
        memory_id: str,
        prediction_error: float | None = None,
        job_id: str | None = None,
        include_global: bool = True,
        include_shared: bool = True,
    ) -> bool:
        """想起時の活性化情報を更新.

        Args:
            memory_id: 更新する記憶のID
            prediction_error: 予測誤差（0.0-1.0）
            job_id: ジョブ固有メモリを検索する場合のジョブID（所有権検証用）
            include_global: グローバルメモリを含めるか
            include_shared: 共有メモリを含めるか

        Returns:
            更新成功時はTrue、失敗時はFalse
        """
        memory = await self.get_by_id(
            memory_id,
            job_id=job_id,
            include_global=include_global,
            include_shared=include_shared,
        )
        if memory is None:
            return False

        payload: dict[str, Any] = {
            "activation_count": memory.activation_count + 1,
            "last_activated": datetime.now().isoformat(),
        }
        if prediction_error is not None:
            payload["prediction_error"] = max(0.0, min(1.0, prediction_error))

        return await self.update_memory_fields(
            memory_id,
            job_id=job_id,
            include_global=include_global,
            include_shared=include_shared,
            **payload,
        )

    async def bump_coactivation(
        self,
        source_id: str,
        target_id: str,
        delta: float = 0.1,
        job_id: str | None = None,
        include_global: bool = True,
        include_shared: bool = True,
    ) -> bool:
        """共起重みを双方向で増加.

        Args:
            source_id: ソース記憶のID
            target_id: ターゲット記憶のID
            delta: 増加量（0.0-1.0）
            job_id: ジョブ固有メモリを検索する場合のジョブID（所有権検証用）
            include_global: グローバルメモリを含めるか
            include_shared: 共有メモリを含めるか

        Returns:
            更新成功時はTrue、失敗時はFalse
        """
        source = await self.get_by_id(
            source_id,
            job_id=job_id,
            include_global=include_global,
            include_shared=include_shared,
        )
        target = await self.get_by_id(
            target_id,
            job_id=job_id,
            include_global=include_global,
            include_shared=include_shared,
        )
        if source is None or target is None:
            return False

        delta = max(0.0, min(1.0, delta))

        updates: list[tuple[str, dict[str, Any]]] = []
        for left, right_id in ((source, target_id), (target, source_id)):
            weight_map = {item_id: value for item_id, value in left.coactivation_weights}
            current = weight_map.get(right_id, 0.0)
            weight_map[right_id] = max(0.0, min(1.0, current + delta))
            updates.append(
                (
                    left.id,
                    {
                        "coactivation": json.dumps(weight_map, ensure_ascii=False),
                    },
                )
            )

        for memory_id, payload in updates:
            await self.update_memory_fields(
                memory_id,
                job_id=job_id,
                include_global=include_global,
                include_shared=include_shared,
                **payload,
            )

        return True

    async def maybe_add_related_link(
        self,
        source_id: str,
        target_id: str,
        threshold: float = 0.6,
        job_id: str | None = None,
        include_global: bool = True,
        include_shared: bool = True,
    ) -> bool:
        """共起重みが閾値を超えたとき related リンクを追加.

        Args:
            source_id: ソース記憶のID
            target_id: ターゲット記憶のID
            threshold: 閾値（0.0-1.0）
            job_id: ジョブ固有メモリを検索する場合のジョブID（所有権検証用）
            include_global: グローバルメモリを含めるか
            include_shared: 共有メモリを含めるか

        Returns:
            リンク追加成功時はTrue、失敗時はFalse
        """
        source = await self.get_by_id(
            source_id,
            job_id=job_id,
            include_global=include_global,
            include_shared=include_shared,
        )
        if source is None:
            return False

        weight_map = {item_id: value for item_id, value in source.coactivation_weights}
        if weight_map.get(target_id, 0.0) < threshold:
            return False

        await self.add_causal_link(
            source_id=source_id,
            target_id=target_id,
            link_type="related",
            note="auto-linked by consolidation replay",
            job_id=job_id,
            include_global=include_global,
            include_shared=include_shared,
        )
        return True

    async def recall_divergent(
        self,
        context: str,
        n_results: int = 5,
        max_branches: int = 3,
        max_depth: int = 3,
        temperature: float = 0.7,
        include_diagnostics: bool = False,
        record_activation: bool = True,
        job_id: str | None = None,
        include_global: bool = True,
        include_shared: bool = True,
    ) -> tuple[list[MemorySearchResult], dict[str, Any]]:
        """連想拡散 + ワークスペース競合で発散想起."""
        n_results = max(1, min(20, n_results))
        seed_size = max(3, min(25, n_results * 3))
        seeds = await self.search_with_scoring(
            query=context,
            n_results=seed_size,
            job_id=job_id,
            include_global=include_global,
            include_shared=include_shared,
        )
        if not seeds:
            return [], {}

        branch_limit, depth_limit = adaptive_search_params(
            context=context,
            requested_branches=max_branches,
            requested_depth=max_depth,
            seed_count=len(seeds),
        )

        seed_memories = [item.memory for item in seeds]

        # fetch_memory_by_id にジョブ分離パラメータを渡すラッパー
        async def fetch_with_job_isolation(memory_id: str) -> Memory | None:
            return await self.get_by_id(
                memory_id,
                job_id=job_id,
                include_global=include_global,
                include_shared=include_shared,
            )

        expanded, assoc_diag = await self._association_engine.spread(
            seeds=seed_memories,
            fetch_memory_by_id=fetch_with_job_isolation,
            max_branches=branch_limit,
            max_depth=depth_limit,
        )

        distance_map = {item.memory.id: item.semantic_distance for item in seeds}
        all_candidates: dict[str, Memory] = {}
        for memory in seed_memories + expanded:
            all_candidates[memory.id] = memory

        workspace_candidates: list[WorkspaceCandidate] = []
        prediction_errors: list[float] = []
        novelty_scores: list[float] = []

        for memory in all_candidates.values():
            semantic_distance = distance_map.get(memory.id)
            if semantic_distance is None:
                relevance = calculate_context_relevance(context, memory)
            else:
                relevance = 1.0 / (1.0 + max(0.0, semantic_distance))

            prediction_error = calculate_prediction_error(context, memory)
            novelty = calculate_novelty_score(memory, prediction_error)
            emotion_boost = calculate_emotion_boost(memory.emotion)
            normalized_emotion = max(0.0, min(1.0, emotion_boost / 0.4))

            prediction_errors.append(prediction_error)
            novelty_scores.append(novelty)

            workspace_candidates.append(
                WorkspaceCandidate(
                    memory=memory,
                    relevance=relevance,
                    prediction_error=prediction_error,
                    novelty=novelty,
                    emotion_boost=normalized_emotion,
                )
            )

        selected_with_scores = select_workspace_candidates(
            candidates=workspace_candidates,
            max_results=n_results,
            temperature=temperature,
        )

        selected_memories = [cand.memory for cand, _ in selected_with_scores]

        if record_activation:
            for memory in selected_memories:
                await self.record_activation(memory.id)

        results = [
            MemorySearchResult(memory=m, distance=distance_map.get(m.id, 0.5))
            for m in selected_memories
        ]

        diagnostics: dict[str, Any] = {}
        if include_diagnostics:
            selected_ids = [cand.memory.id for cand, _ in selected_with_scores]
            diagnostics = self._build_divergent_diagnostics(
                assoc_diag=assoc_diag,
                workspace_candidates=workspace_candidates,
                winner_ids=selected_ids,
                prediction_errors=prediction_errors,
                novelty_scores=novelty_scores,
                branch_limit=branch_limit,
                depth_limit=depth_limit,
            )

        return results, diagnostics

    async def get_association_diagnostics(
        self,
        context: str,
        sample_size: int = 20,
        job_id: str | None = None,
        include_global: bool = True,
        include_shared: bool = True,
    ) -> dict[str, Any]:
        """連想探索の診断値を返す."""
        n_results = max(3, min(20, sample_size))
        _, diagnostics = await self.recall_divergent(
            context=context,
            n_results=n_results,
            max_branches=4,
            max_depth=3,
            include_diagnostics=True,
            record_activation=False,
            job_id=job_id,
            include_global=include_global,
            include_shared=include_shared,
        )
        return diagnostics

    async def consolidate_memories(
        self,
        window_hours: int = 24,
        max_replay_events: int = 200,
        link_update_strength: float = 0.2,
        job_id: str | None = None,
        include_global: bool = True,
        include_shared: bool = True,
    ) -> dict[str, int]:
        """手動トリガーの統合処理."""
        stats = await self._consolidation_engine.run(
            store=self,
            window_hours=window_hours,
            max_replay_events=max_replay_events,
            link_update_strength=link_update_strength,
            job_id=job_id,
            include_global=include_global,
            include_shared=include_shared,
        )
        return stats.to_dict()

    def _build_divergent_diagnostics(
        self,
        assoc_diag: AssociationDiagnostics,
        workspace_candidates: list[WorkspaceCandidate],
        winner_ids: list[str],
        prediction_errors: list[float],
        novelty_scores: list[float],
        branch_limit: int = 0,
        depth_limit: int = 0,
    ) -> dict[str, Any]:
        """発散想起の診断情報を構築."""
        avg_prediction_error = 0.0
        if prediction_errors:
            avg_prediction_error = sum(prediction_errors) / len(prediction_errors)

        avg_novelty = 0.0
        if novelty_scores:
            avg_novelty = sum(novelty_scores) / len(novelty_scores)

        # 選択されたメモリのダイバーシティスコアを計算
        selected_memories = [
            cand.memory for cand in workspace_candidates if cand.memory.id in winner_ids
        ]
        div_score = diversity_score(selected_memories) if selected_memories else 0.0

        return {
            "branches_used": assoc_diag.branches_used,
            "depth_used": assoc_diag.depth_used,
            "adaptive_branch_limit": branch_limit,
            "adaptive_depth_limit": depth_limit,
            "traversed_edges": assoc_diag.traversed_edges,
            "expanded_nodes": assoc_diag.expanded_nodes,
            "avg_branching_factor": assoc_diag.avg_branching_factor,
            "workspace_candidates": len(workspace_candidates),
            "selected_count": len(winner_ids),
            "diversity_score": div_score,
            "avg_prediction_error": avg_prediction_error,
            "avg_novelty": avg_novelty,
        }

    # Phase 7: ジョブ設定の永続化

    async def _load_job_configs(self) -> None:
        """Load job configurations from JSON file."""
        if os.path.exists(self._job_config_path):
            try:
                with open(self._job_config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Load jobs
                for job_data in data.get("jobs", []):
                    job = JobConfig.from_dict(job_data)
                    self._job_configs[job.job_id] = job

                # Load shared groups
                for group_data in data.get("shared_groups", []):
                    group = SharedGroupConfig.from_dict(group_data)
                    self._shared_groups[group.group_id] = group

                logger.info(
                    f"Loaded {len(self._job_configs)} jobs and "
                    f"{len(self._shared_groups)} shared groups"
                )
            except Exception as e:
                logger.warning(f"Failed to load job configs: {e}")
                self._job_configs = {}
                self._shared_groups = {}

    async def _save_job_configs(self) -> None:
        """Save job configurations to JSON file."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self._job_config_path), exist_ok=True)

            data = {
                "jobs": [job.to_dict() for job in self._job_configs.values()],
                "shared_groups": [group.to_dict() for group in self._shared_groups.values()],
            }

            with open(self._job_config_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            logger.info(
                f"Saved {len(self._job_configs)} jobs and {len(self._shared_groups)} shared groups"
            )
        except Exception as e:
            logger.error(f"Failed to save job configs: {e}")

    # Phase 7: ジョブ分離 - ジョブ設定管理メソッド

    async def create_job(
        self,
        job_id: str,
        name: str,
        description: str = "",
        shared_group_ids: tuple[str, ...] = (),
    ) -> JobConfig:
        """Create a new job configuration."""
        job_config = JobConfig(
            job_id=job_id,
            name=name,
            description=description,
            shared_group_ids=shared_group_ids,
        )
        self._job_configs[job_id] = job_config
        return job_config

    async def get_job(self, job_id: str) -> JobConfig | None:
        """Get job configuration by ID."""
        return self._job_configs.get(job_id)

    async def list_jobs(self) -> list[JobConfig]:
        """List all job configurations."""
        return list(self._job_configs.values())

    async def update_job(
        self,
        job_id: str,
        name: str | None = None,
        description: str | None = None,
        shared_group_ids: tuple[str, ...] | None = None,
    ) -> JobConfig | None:
        """Update job configuration."""
        job = self._job_configs.get(job_id)
        if job is None:
            return None

        updated = JobConfig(
            job_id=job_id,
            name=name if name is not None else job.name,
            description=description if description is not None else job.description,
            shared_group_ids=shared_group_ids
            if shared_group_ids is not None
            else job.shared_group_ids,
        )
        self._job_configs[job_id] = updated
        return updated

    async def delete_job(self, job_id: str) -> bool:
        """Delete job configuration."""
        if job_id in self._job_configs:
            del self._job_configs[job_id]
            return True
        return False

    async def create_shared_group(
        self,
        group_id: str,
        name: str,
        description: str = "",
        member_job_ids: tuple[str, ...] = (),
    ) -> SharedGroupConfig:
        """Create a new shared group configuration."""
        group_config = SharedGroupConfig(
            group_id=group_id,
            name=name,
            description=description,
            member_job_ids=member_job_ids,
        )
        self._shared_groups[group_id] = group_config

        # Also update each member job's shared_group_ids to include this new group
        for job_id in member_job_ids:
            job = self._job_configs.get(job_id)
            if job is not None and group_id not in job.shared_group_ids:
                new_ids = tuple(list(job.shared_group_ids) + [group_id])
                self._job_configs[job_id] = JobConfig(
                    job_id=job.job_id,
                    name=job.name,
                    description=job.description,
                    shared_group_ids=new_ids,
                )

        return group_config

    async def get_shared_group(self, group_id: str) -> SharedGroupConfig | None:
        """Get shared group configuration by ID."""
        return self._shared_groups.get(group_id)

    async def list_shared_groups(self) -> list[SharedGroupConfig]:
        """List all shared group configurations."""
        return list(self._shared_groups.values())

    async def add_job_to_shared_group(self, job_id: str, group_id: str) -> bool:
        """Add a job to a shared group."""
        job = self._job_configs.get(job_id)
        group = self._shared_groups.get(group_id)

        if job is None or group is None:
            return False

        # Update job's shared_group_ids
        if group_id not in job.shared_group_ids:
            new_ids = tuple(list(job.shared_group_ids) + [group_id])
            self._job_configs[job_id] = JobConfig(
                job_id=job.job_id,
                name=job.name,
                description=job.description,
                shared_group_ids=new_ids,
            )

        # Update group's member_job_ids
        if job_id not in group.member_job_ids:
            new_members = tuple(list(group.member_job_ids) + [job_id])
            self._shared_groups[group_id] = SharedGroupConfig(
                group_id=group.group_id,
                name=group.name,
                description=group.description,
                member_job_ids=new_members,
            )

        return True

    async def remove_job_from_shared_group(self, job_id: str, group_id: str) -> bool:
        """Remove a job from a shared group."""
        job = self._job_configs.get(job_id)
        group = self._shared_groups.get(group_id)

        if job is None or group is None:
            return False

        # Update job's shared_group_ids
        if group_id in job.shared_group_ids:
            new_ids = tuple([gid for gid in job.shared_group_ids if gid != group_id])
            self._job_configs[job_id] = JobConfig(
                job_id=job.job_id,
                name=job.name,
                description=job.description,
                shared_group_ids=new_ids,
            )

        # Update group's member_job_ids
        if job_id in group.member_job_ids:
            new_members = tuple([jid for jid in group.member_job_ids if jid != job_id])
            self._shared_groups[group_id] = SharedGroupConfig(
                group_id=group.group_id,
                name=group.name,
                description=group.description,
                member_job_ids=new_members,
            )

        return True

    async def delete_shared_group(self, group_id: str) -> bool:
        """Delete a shared group.

        Args:
            group_id: ID of the shared group to delete

        Returns:
            True if deleted successfully, False if group not found
        """
        group = self._shared_groups.get(group_id)
        if group is None:
            return False

        # Remove this group from all member jobs' shared_group_ids
        for job_id in group.member_job_ids:
            job = self._job_configs.get(job_id)
            if job is not None and group_id in job.shared_group_ids:
                new_ids = tuple([gid for gid in job.shared_group_ids if gid != group_id])
                self._job_configs[job_id] = JobConfig(
                    job_id=job.job_id,
                    name=job.name,
                    description=job.description,
                    shared_group_ids=new_ids,
                )

        # Delete the group
        del self._shared_groups[group_id]

        return True
