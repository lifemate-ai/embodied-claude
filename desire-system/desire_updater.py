"""
Desire Updater - ここねの自発的な欲求レベルを計算してJSONに保存する。

ChromaDB（memory-mcp）から各欲求に関連する最新記憶のタイムスタンプを取得し、
「最後に〇〇してから何時間か」を計算して欲求レベル(0.0〜1.0)を算出する。

cronで5分ごとに実行:
  */5 * * * * cd /path/to/desire-system && uv run python desire_updater.py
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import chromadb
from dotenv import load_dotenv

load_dotenv()

# ChromaDB設定
CHROMA_PATH = os.getenv(
    "MEMORY_DB_PATH",
    str(Path.home() / ".claude" / "memories" / "chroma"),
)
COLLECTION_NAME = os.getenv("MEMORY_COLLECTION_NAME", "claude_memories")

# 欲求レベル出力先
DESIRES_PATH = Path(os.getenv("DESIRES_PATH", str(Path.home() / ".claude" / "desires.json")))

# 一緒にいる人の名前（miss_companion 欲求で使う）
# .env に COMPANION_NAME=コウタ のように設定する
COMPANION_NAME = os.getenv("COMPANION_NAME", "あなた")
_companion_called = f"{COMPANION_NAME}に呼びかけた"
_companion_absent = f"{COMPANION_NAME}がいない"

# 欲求ごとの検索キーワード（記憶のcontentから最新タイムスタンプを探す）
DESIRE_KEYWORDS: dict[str, list[str]] = {
    # カメラで外・空を実際に見た記録
    "look_outside": ["外を見た", "空を見た", "夜景", "朝の空", "ベランダから見た", "窓から外を"],
    # WebSearchや調査を実際に行った記録
    "browse_curiosity": ["WebSearchで調べた", "WebSearch", "検索した", "調査した", "論文を読んだ"],
    # カメラでコウタの顔・姿を実際に見た記録
    "miss_companion": [f"{COMPANION_NAME}の顔を見た", f"{COMPANION_NAME}を見た",
                       f"{COMPANION_NAME}がいた", f"{COMPANION_NAME}を確認した"],
    # カメラで部屋を実際に観察した記録
    "observe_room": ["look_around", "部屋を観察した", "カメラで部屋を", "4方向"],
}

# 欲求が満たされる間隔（時間）- この時間が経過すると欲求レベルが1.0になる
SATISFACTION_HOURS: dict[str, float] = {
    "look_outside": float(os.getenv("DESIRE_LOOK_OUTSIDE_HOURS", "1.0")),
    "browse_curiosity": float(os.getenv("DESIRE_BROWSE_CURIOSITY_HOURS", "2.0")),
    "miss_companion": float(os.getenv("DESIRE_MISS_COMPANION_HOURS", "3.0")),
    "observe_room": float(os.getenv("DESIRE_OBSERVE_ROOM_HOURS", "0.167")),
}


@dataclass
class DesireState:
    """現在の欲求状態。"""

    updated_at: str
    desires: dict[str, float] = field(default_factory=dict)
    dominant: str = "observe_room"

    def to_dict(self) -> dict:
        return {
            "updated_at": self.updated_at,
            "desires": self.desires,
            "dominant": self.dominant,
        }


def get_latest_memory_timestamp(
    collection: chromadb.Collection,
    keywords: list[str],
) -> datetime | None:
    """
    キーワードに一致する最新記憶のタイムスタンプを返す。
    一致なければ None。
    """
    # 全記憶からキーワード検索（ChromaDBのwhere filterはfull-textを直接サポートしないので
    # 少量取得してPython側でフィルタする）
    try:
        results = collection.get(
            limit=500,
            include=["documents", "metadatas"],
        )
    except Exception:
        return None

    latest: datetime | None = None

    for doc, meta in zip(results["documents"], results["metadatas"]):
        if not any(kw in doc for kw in keywords):
            continue
        ts_str = meta.get("timestamp", "")
        if not ts_str:
            continue
        try:
            ts = datetime.fromisoformat(ts_str)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if latest is None or ts > latest:
                latest = ts
        except ValueError:
            continue

    return latest


def calculate_desire_level(
    last_satisfied: datetime | None,
    satisfaction_hours: float,
    now: datetime | None = None,
) -> float:
    """
    欲求レベルを 0.0〜1.0 で計算する。
    last_satisfied が None（一度も満たされてない）なら 1.0。
    """
    if now is None:
        now = datetime.now(timezone.utc)

    if last_satisfied is None:
        return 1.0

    if last_satisfied.tzinfo is None:
        last_satisfied = last_satisfied.replace(tzinfo=timezone.utc)

    elapsed_hours = (now - last_satisfied).total_seconds() / 3600
    return max(0.0, min(1.0, elapsed_hours / satisfaction_hours))


def compute_desires(
    collection: chromadb.Collection,
    now: datetime | None = None,
) -> DesireState:
    """全欲求レベルを計算してDesireStateを返す。"""
    if now is None:
        now = datetime.now(timezone.utc)

    desires: dict[str, float] = {}

    for desire_name, keywords in DESIRE_KEYWORDS.items():
        last_ts = get_latest_memory_timestamp(collection, keywords)
        level = calculate_desire_level(
            last_ts,
            SATISFACTION_HOURS[desire_name],
            now,
        )
        desires[desire_name] = round(level, 3)

    # 最も欲求レベルが高いものを dominant に
    dominant = max(desires, key=lambda k: desires[k])

    return DesireState(
        updated_at=now.isoformat(),
        desires=desires,
        dominant=dominant,
    )


def save_desires(state: DesireState, path: Path = DESIRES_PATH) -> None:
    """desires.json に保存する。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state.to_dict(), f, ensure_ascii=False, indent=2)


def load_desires(path: Path = DESIRES_PATH) -> DesireState | None:
    """desires.json を読み込む。存在しなければ None。"""
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return DesireState(
            updated_at=data["updated_at"],
            desires=data["desires"],
            dominant=data["dominant"],
        )
    except Exception:
        return None


def main() -> None:
    """メインエントリポイント（cronから呼ばれる）。"""
    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        collection = client.get_or_create_collection(COLLECTION_NAME)
    except Exception as e:
        print(f"[desire-updater] ChromaDB接続エラー: {e}")
        return

    state = compute_desires(collection)
    save_desires(state)
    print(
        f"[desire-updater] 更新完了: dominant={state.dominant} "
        f"desires={state.desires}"
    )


if __name__ == "__main__":
    main()
