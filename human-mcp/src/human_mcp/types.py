"""Type definitions for Human as MCP."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class TaskType(str, Enum):
    """What kind of task requires a human."""

    OBSERVATION = "observation"      # 現実世界の観測（写真、状況確認）
    JUDGMENT = "judgment"            # 価値判断（複数の妥当な選択肢から選ぶ）
    APPROVAL = "approval"           # 承認（法的・金銭的・倫理的な決定）
    PHYSICAL_ACTION = "physical_action"  # 物理操作（モノを動かす、ボタンを押す）


class ResponseFormat(str, Enum):
    """Expected format of the human's response."""

    CHOICE = "choice"          # 選択肢から1つ選ぶ
    YES_NO = "yes_no"          # yes/no
    SHORT_TEXT = "short_text"  # 50文字以内の短文
    NUMBER = "number"          # 数値
    PHOTO = "photo"            # 写真パス


class Urgency(str, Enum):
    """How urgently the human's response is needed."""

    LOW = "low"        # 急ぎじゃない（1時間以内）
    NORMAL = "normal"  # 普通（5分以内）
    HIGH = "high"      # 急ぎ（1分以内）


@dataclass
class HumanRequest:
    """A structured request to a human."""

    task_type: TaskType
    question: str
    reason: str
    expected_format: ResponseFormat = ResponseFormat.SHORT_TEXT
    options: list[str] = field(default_factory=list)
    urgency: Urgency = Urgency.NORMAL
    batch_id: str | None = None
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class HumanResponse:
    """A structured response from a human."""

    choice: str | None = None
    yes_no: bool | None = None
    short_text: str | None = None
    number: float | None = None
    photo_path: str | None = None
    responded_at: datetime = field(default_factory=datetime.now)
    response_time_seconds: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dict, omitting None values."""
        result: dict = {}
        if self.choice is not None:
            result["choice"] = self.choice
        if self.yes_no is not None:
            result["yes_no"] = self.yes_no
        if self.short_text is not None:
            result["short_text"] = self.short_text
        if self.number is not None:
            result["number"] = self.number
        if self.photo_path is not None:
            result["photo_path"] = self.photo_path
        result["responded_at"] = self.responded_at.isoformat()
        result["response_time_seconds"] = self.response_time_seconds
        return result
