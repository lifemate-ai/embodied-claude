"""Sensory data integration with memories."""

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from .types import CameraPosition, Memory, SensoryData

if TYPE_CHECKING:
    from .memory import MemoryStore


class SensoryIntegration:
    """感覚データの記憶統合.

    視覚・聴覚データをカメラ位置と共に記憶に紐付ける。
    例: 「朝の空を見つけた」記憶 + 画像パス + カメラ位置(pan=60, tilt=-30)
    """

    def __init__(self, memory_store: "MemoryStore"):
        """Initialize sensory integration.

        Args:
            memory_store: MemoryStoreインスタンス
        """
        self._memory_store = memory_store

    async def save_visual_memory(
        self,
        content: str,
        image_path: str,
        camera_position: CameraPosition,
        emotion: str = "neutral",
        importance: int = 3,
        category: str = "observation",
        auto_describe: bool = False,
    ) -> Memory:
        """視覚記憶を保存（画像パス + カメラ位置）.

        Args:
            content: 記憶の内容（例: "朝の空を見つけた"）
            image_path: 画像ファイルパス
            camera_position: カメラの向き
            emotion: 感情
            importance: 重要度（1-5）
            category: カテゴリ
            auto_describe: 画像説明を自動生成（Phase 4.3では未実装）

        Returns:
            保存された記憶
        """
        # 感覚データを作成
        sensory_data = SensoryData(
            sensory_type="visual",
            file_path=image_path,
            metadata={
                "camera_position": camera_position.to_dict(),
            },
            description=None,  # Phase 4.3では説明生成なし
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        # 記憶を保存（感覚データとカメラ位置を含む）
        return await self._memory_store.save(
            content=content,
            emotion=emotion,
            importance=importance,
            category=category,
            sensory_data=(sensory_data,),
            camera_position=camera_position,
        )

    async def save_audio_memory(
        self,
        content: str,
        audio_path: str,
        transcript: str,
        emotion: str = "neutral",
        importance: int = 3,
        category: str = "observation",
    ) -> Memory:
        """聴覚記憶を保存（音声パス + 文字起こし）.

        Args:
            content: 記憶の内容（例: "幼馴染の声を聞いた"）
            audio_path: 音声ファイルパス
            transcript: Whisperなどでの文字起こし
            emotion: 感情
            importance: 重要度（1-5）
            category: カテゴリ

        Returns:
            保存された記憶
        """
        # 感覚データを作成
        sensory_data = SensoryData(
            sensory_type="audio",
            file_path=audio_path,
            metadata={"transcript": transcript},
            description=transcript,  # 文字起こしを説明として使用
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        # 記憶を保存
        return await self._memory_store.save(
            content=content,
            emotion=emotion,
            importance=importance,
            category=category,
            sensory_data=(sensory_data,),
        )

    async def recall_by_camera_position(
        self,
        pan_angle: int,
        tilt_angle: int,
        tolerance: int = 15,
    ) -> list[Memory]:
        """カメラ位置から記憶を想起.

        「この方向を見た時に何を見たっけ?」という問いに答える。

        Args:
            pan_angle: パン角度（-90 to +90）
            tilt_angle: チルト角度（-90 to +90）
            tolerance: 角度の許容範囲（デフォルト±15度）

        Returns:
            条件を満たす記憶のリスト（新しい順）
        """
        # 全記憶を取得
        all_memories = await self._memory_store.get_all()

        # カメラ位置でフィルタ
        results = []
        for memory in all_memories:
            if memory.camera_position is None:
                continue

            pan_diff = abs(memory.camera_position.pan_angle - pan_angle)
            tilt_diff = abs(memory.camera_position.tilt_angle - tilt_angle)

            if pan_diff <= tolerance and tilt_diff <= tolerance:
                results.append(memory)

        # 時系列逆順（新しい順）
        results.sort(key=lambda m: m.timestamp, reverse=True)

        return results

    async def get_memories_with_sensory_data(
        self,
        sensory_type: str | None = None,
    ) -> list[Memory]:
        """感覚データを持つ記憶を取得.

        Args:
            sensory_type: フィルタする感覚タイプ（"visual", "audio"など）
                         Noneの場合は全ての感覚データ付き記憶を返す

        Returns:
            感覚データを持つ記憶のリスト（新しい順）
        """
        all_memories = await self._memory_store.get_all()

        # 感覚データを持つ記憶をフィルタ
        results = []
        for memory in all_memories:
            if not memory.sensory_data:
                continue

            # タイプでフィルタ
            if sensory_type:
                if any(sd.sensory_type == sensory_type for sd in memory.sensory_data):
                    results.append(memory)
            else:
                results.append(memory)

        # 時系列逆順
        results.sort(key=lambda m: m.timestamp, reverse=True)

        return results
