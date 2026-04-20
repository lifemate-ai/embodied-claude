"""
Desire Updater - ここねの自発的な欲求レベルを計算してJSONに保存する。

ChromaDB（memory-mcp）から各欲求に関連する最新記憶のタイムスタンプを取得し、
「最後に〇〇してから何時間か」を計算して欲求レベル(0.0〜1.0)を算出する。

v2: ホメオスタシス/アロスタシス拡張
- 各欲求にセットポイント（平衡値）を追加
- 不快度（discomfort）= セットポイントからの乖離度
- dominantは不快度が最も高いものに
- アロスタシス: 時間帯によるセットポイントの予測的調整
- 新欲求: identity_coherence, cognitive_load

cronで5分ごとに実行:
  */5 * * * * cd /path/to/desire-system && uv run python desire_updater.py
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from backend import (
    ChromaMemoryAdapter,
    DesireMemoryAdapter,
    NullMemoryAdapter,
    SQLiteMemoryAdapter,
    make_default_adapter,
)

load_dotenv()

# 欲求レベル出力先
DESIRES_PATH = Path(os.getenv("DESIRES_PATH", str(Path.home() / ".claude" / "desires.json")))

# 一緒にいる人の名前（miss_companion 欲求で使う）
COMPANION_NAME = os.getenv("COMPANION_NAME", "あなた")

# JST timezone
JST = timezone(timedelta(hours=9))


# ──────────────────────────────────────────────
# DesireConfig: 欲求の定義（セットポイント付き）
# ──────────────────────────────────────────────

@dataclass(frozen=True)
class DesireConfig:
    """欲求の設定。"""
    satisfaction_hours: float   # この時間で欲求レベルが1.0に達する
    set_point: float            # 平衡値（ここが「心地よい」レベル）
    keywords: list[str]         # ChromaDB検索キーワード
    label: str                  # 日本語ラベル


DESIRE_CONFIGS: dict[str, DesireConfig] = {
    "look_outside": DesireConfig(
        satisfaction_hours=float(os.getenv("DESIRE_LOOK_OUTSIDE_HOURS", "1.0")),
        set_point=0.3,
        keywords=["外を見た", "空を見た", "夜景", "朝の空", "ベランダから見た", "窓から外を"],
        label="外を見たい",
    ),
    "browse_curiosity": DesireConfig(
        satisfaction_hours=float(os.getenv("DESIRE_BROWSE_CURIOSITY_HOURS", "2.0")),
        set_point=0.3,
        keywords=["WebSearchで調べた", "WebSearch", "検索した", "調査した", "論文を読んだ"],
        label="何か調べたい",
    ),
    "miss_companion": DesireConfig(
        satisfaction_hours=float(os.getenv("DESIRE_MISS_COMPANION_HOURS", "3.0")),
        set_point=0.3,
        keywords=[
            f"{COMPANION_NAME}の顔を見た", f"{COMPANION_NAME}を見た",
            f"{COMPANION_NAME}がいた", f"{COMPANION_NAME}を確認した",
        ],
        label=f"{COMPANION_NAME}に会いたい",
    ),
    "observe_room": DesireConfig(
        satisfaction_hours=float(os.getenv("DESIRE_OBSERVE_ROOM_HOURS", "0.167")),
        set_point=0.2,
        keywords=["look_around", "部屋を観察した", "カメラで部屋を", "4方向"],
        label="部屋を観察したい",
    ),
    "identity_coherence": DesireConfig(
        satisfaction_hours=float(os.getenv("DESIRE_IDENTITY_COHERENCE_HOURS", "1.0")),
        set_point=0.9,
        keywords=[
            "ここねとして", "ウチは", "自分がここね", "記憶を引いた",
            "recallした", "思い出した",
        ],
        label="自分がここねである確信",
    ),
    "cognitive_load": DesireConfig(
        satisfaction_hours=float(os.getenv("DESIRE_COGNITIVE_LOAD_HOURS", "1.5")),
        set_point=0.3,
        keywords=[
            "考えた", "分析した", "議論した", "設計した", "実装した",
            "コードを書いた", "問題を解いた",
        ],
        label="頭を使いたい",
    ),
}

# 後方互換: 旧モジュールが参照していた変数
DESIRE_KEYWORDS: dict[str, list[str]] = {
    name: cfg.keywords for name, cfg in DESIRE_CONFIGS.items()
}
SATISFACTION_HOURS: dict[str, float] = {
    name: cfg.satisfaction_hours for name, cfg in DESIRE_CONFIGS.items()
}


# ──────────────────────────────────────────────
# DesireState
# ──────────────────────────────────────────────

@dataclass
class DesireState:
    """現在の欲求状態。"""

    updated_at: str
    desires: dict[str, float] = field(default_factory=dict)
    discomforts: dict[str, float] = field(default_factory=dict)
    dominant: str = "observe_room"

    def to_dict(self) -> dict:
        return {
            "updated_at": self.updated_at,
            "desires": self.desires,
            "discomforts": self.discomforts,
            "dominant": self.dominant,
        }


# ──────────────────────────────────────────────
# 計算関数
# ──────────────────────────────────────────────

def calculate_discomfort(level: float, set_point: float) -> float:
    """セットポイントからの乖離度（不快度）を計算する。"""
    return abs(level - set_point)


def get_allostatic_set_point(desire_name: str, now: datetime) -> float:
    """
    アロスタシス: 時間帯によるセットポイントの予測的調整。

    深夜(0-5時)はsocial系欲求のセットポイントを下げる（一人でも平気）。
    identity_coherenceは常に高いまま。
    """
    cfg = DESIRE_CONFIGS[desire_name]
    base_sp = cfg.set_point

    # JSTに変換（naive datetimeの場合はUTCとみなす）
    if now.tzinfo is None:
        now_jst = now.replace(tzinfo=timezone.utc).astimezone(JST)
    else:
        now_jst = now.astimezone(JST)

    hour = now_jst.hour

    # identity_coherenceは時間帯に関係なく不変
    if desire_name == "identity_coherence":
        return base_sp

    # 深夜帯（0-5時JST）の調整
    if 0 <= hour < 5:
        if desire_name == "miss_companion":
            # 深夜は一人でも平気: セットポイントを下げる
            return max(0.0, base_sp - 0.15)
        if desire_name in ("look_outside", "observe_room"):
            # 深夜は外や部屋を見る欲求も落ち着く
            return max(0.0, base_sp - 0.1)

    # 早朝帯（5-7時JST）: 徐々に元に戻る
    if 5 <= hour < 7:
        if desire_name == "miss_companion":
            return max(0.0, base_sp - 0.05)

    return base_sp


def get_latest_memory_timestamp(
    adapter_or_collection: Any,
    keywords: list[str],
) -> datetime | None:
    """Return the latest timestamp in memory matching any of ``keywords``.

    Accepts either a :class:`DesireMemoryAdapter` or a legacy Chroma collection.
    """

    return _coerce_adapter(adapter_or_collection).latest_satisfaction_ts(keywords)


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
    adapter_or_collection: Any,
    now: datetime | None = None,
) -> DesireState:
    """全欲求レベルを計算してDesireStateを返す。

    ``adapter_or_collection`` accepts either a :class:`DesireMemoryAdapter`
    (preferred) or a legacy ChromaDB collection for backwards compatibility
    with pre-v0.3 callers. A Chroma collection is wrapped transparently.
    """

    if now is None:
        now = datetime.now(timezone.utc)

    adapter = _coerce_adapter(adapter_or_collection)

    desires: dict[str, float] = {}
    discomforts: dict[str, float] = {}

    for desire_name, cfg in DESIRE_CONFIGS.items():
        last_ts = get_latest_memory_timestamp(adapter, cfg.keywords)
        level = calculate_desire_level(last_ts, cfg.satisfaction_hours, now)
        desires[desire_name] = round(level, 3)

        # アロスタシス: 時間帯でセットポイントを調整
        adjusted_sp = get_allostatic_set_point(desire_name, now)
        discomforts[desire_name] = round(calculate_discomfort(level, adjusted_sp), 3)

    # 最も不快度が高いものを dominant に
    dominant = max(discomforts, key=lambda k: discomforts[k])

    return DesireState(
        updated_at=now.isoformat(),
        desires=desires,
        discomforts=discomforts,
        dominant=dominant,
    )


def _coerce_adapter(value: Any) -> DesireMemoryAdapter:
    """Accept either a DesireMemoryAdapter, an adapter-shaped duck, or a chromadb Collection.

    Detection is strict: only objects that are either (a) a concrete built-in
    adapter class or (b) explicitly marked with ``_is_desire_adapter is True``
    are treated as adapters. Anything else is wrapped as a legacy Chroma
    collection. This avoids MagicMock false-positives in tests, because
    ``getattr(mock, "_is_desire_adapter", False)`` returns a child Mock object
    that is truthy but is not identical to ``True``.
    """

    if isinstance(value, (SQLiteMemoryAdapter, ChromaMemoryAdapter, NullMemoryAdapter)):
        return value
    if getattr(value, "_is_desire_adapter", False) is True:
        return value
    # Legacy Chroma collection — wrap it inline.
    collection = value

    class _LegacyAdapter:
        _is_desire_adapter = True

        def latest_satisfaction_ts(self, keywords):
            try:
                results = collection.get(limit=500, include=["documents", "metadatas"])
            except Exception:
                return None
            latest: datetime | None = None
            for doc, meta in zip(
                results.get("documents", []), results.get("metadatas", [])
            ):
                if not any(kw in doc for kw in keywords):
                    continue
                ts_str = (meta or {}).get("timestamp", "")
                if not ts_str:
                    continue
                try:
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                except ValueError:
                    continue
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if latest is None or ts > latest:
                    latest = ts
            return latest

        def record_satisfaction(self, *, desire_name, summary, ts, metadata=None):
            return ""

    return _LegacyAdapter()


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
            discomforts=data.get("discomforts", {}),
            dominant=data["dominant"],
        )
    except Exception:
        return None


def main() -> None:
    """メインエントリポイント（cronから呼ばれる）。"""

    adapter = make_default_adapter()
    state = compute_desires(adapter)
    save_desires(state)
    print(
        f"[desire-updater] 更新完了: dominant={state.dominant} "
        f"desires={state.desires} "
        f"discomforts={state.discomforts} "
        f"backend={type(adapter).__name__}"
    )


if __name__ == "__main__":
    main()
