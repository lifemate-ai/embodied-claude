"""Image utilities for visual memory storage."""

import base64
import logging
from io import BytesIO

from PIL import Image
from PIL.Image import Image as PilImage

logger = logging.getLogger(__name__)

RESOLUTION_PRESETS: dict[str, tuple[int, int]] = {
    "low": (160, 120),
    "medium": (320, 240),
    "high": (640, 480),
    "full_hd": (1920, 1080),
}


def encode_image_for_memory(
    image_path: str,
    max_width: int = 320,
    max_height: int = 240,
    quality: int = 60,
) -> str | None:
    """画像を読み込み、リサイズし、JPEG base64文字列を返す.

    人間の記憶もぼんやりしているように、解像度を落として保存する。

    Args:
        image_path: 画像ファイルパス
        max_width: 最大幅（デフォルト320）
        max_height: 最大高さ（デフォルト240）
        quality: JPEG品質（デフォルト60）

    Returns:
        base64エンコードされたJPEG文字列、失敗時はNone
    """
    try:
        with Image.open(image_path) as img_file:
            img: PilImage = img_file
            # RGBA等をRGBに変換（JPEG保存のため）
            if img.mode in ("RGBA", "LA", "P"):
                img = img.convert("RGB")

            # アスペクト比を維持してリサイズ
            img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)

            # JPEGとしてバッファに書き出し
            buffer = BytesIO()
            img.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)

            return base64.b64encode(buffer.read()).decode("ascii")
    except Exception:
        logger.exception("Failed to encode image: %s", image_path)
        return None


QUALITY_PRESETS: dict[str, int] = {
    "low": 40,
    "medium": 60,
    "high": 75,
    "full_hd": 85,
}


def resolve_resolution(resolution: str | None) -> tuple[int, int]:
    """解像度プリセット名をサイズに変換する.

    Args:
        resolution: "low", "medium", "high", "full_hd" または None

    Returns:
        (max_width, max_height) タプル
    """
    if resolution is None:
        return RESOLUTION_PRESETS["medium"]
    return RESOLUTION_PRESETS.get(resolution, RESOLUTION_PRESETS["medium"])


def resolve_quality(resolution: str | None) -> int:
    """解像度プリセットに応じたJPEG品質を返す."""
    if resolution is None:
        return QUALITY_PRESETS["medium"]
    return QUALITY_PRESETS.get(resolution, QUALITY_PRESETS["medium"])
