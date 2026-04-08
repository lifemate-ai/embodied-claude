"""Metacognition module — hypothesis tracking and self-monitoring.

Enables the AI to:
1. Register hypotheses before acting ("I think X is the cause")
2. Verify hypotheses after observing results
3. Detect when it's stuck in a failing approach

Designed to integrate with the existing memory system.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class HypothesisStatus(str, Enum):
    ACTIVE = "active"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"


@dataclass
class Hypothesis:
    id: str
    hypothesis: str  # "tiltが逆なのはコードのバグ"
    context: str  # "カメラが下を向かない問題の調査中"
    approach: str  # "camera.pyのtilt方向を反転させる"
    status: HypothesisStatus = HypothesisStatus.ACTIVE
    created_at: str = ""
    resolved_at: str = ""
    outcome: str = ""  # "コード修正したが改善せず"
    rejection_count: int = 0  # この文脈で棄却された仮説の数

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "hypothesis": self.hypothesis,
            "context": self.context,
            "approach": self.approach,
            "status": self.status.value,
            "created_at": self.created_at,
            "resolved_at": self.resolved_at,
            "outcome": self.outcome,
            "rejection_count": self.rejection_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Hypothesis":
        return cls(
            id=data["id"],
            hypothesis=data["hypothesis"],
            context=data["context"],
            approach=data["approach"],
            status=HypothesisStatus(data["status"]),
            created_at=data.get("created_at", ""),
            resolved_at=data.get("resolved_at", ""),
            outcome=data.get("outcome", ""),
            rejection_count=data.get("rejection_count", 0),
        )


class MetacognitionTracker:
    """Tracks hypotheses and detects stuck patterns."""

    APPROACH_CHANGE_THRESHOLD = 2  # 同じ文脈で2回棄却→アプローチ変更を推奨
    MAX_ACTIVE_HYPOTHESES = 10

    def __init__(self, state_path: Path):
        self._state_path = state_path
        self._hypotheses: list[Hypothesis] = []
        self._counter = 0
        self._load()

    def _load(self) -> None:
        if self._state_path.exists():
            try:
                data = json.loads(self._state_path.read_text())
                self._hypotheses = [Hypothesis.from_dict(h) for h in data.get("hypotheses", [])]
                self._counter = data.get("counter", 0)
            except (json.JSONDecodeError, KeyError):
                logger.warning("Failed to load metacognition state, starting fresh")
                self._hypotheses = []
                self._counter = 0

    def _save(self) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "hypotheses": [h.to_dict() for h in self._hypotheses],
            "counter": self._counter,
        }
        self._state_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))

    def hypothesize(self, hypothesis: str, context: str, approach: str) -> Hypothesis:
        """Register a new hypothesis before taking action."""
        self._counter += 1
        now = datetime.now(timezone.utc).isoformat()

        # Count rejections in this context
        context_rejections = sum(
            1
            for h in self._hypotheses
            if h.context == context and h.status == HypothesisStatus.REJECTED
        )

        h = Hypothesis(
            id=f"hyp_{self._counter}",
            hypothesis=hypothesis,
            context=context,
            approach=approach,
            created_at=now,
            rejection_count=context_rejections,
        )

        # Supersede previous active hypotheses in same context
        for existing in self._hypotheses:
            if existing.context == context and existing.status == HypothesisStatus.ACTIVE:
                existing.status = HypothesisStatus.SUPERSEDED
                existing.resolved_at = now

        self._hypotheses.append(h)

        # Trim old resolved hypotheses
        if len(self._hypotheses) > 50:
            resolved = [h for h in self._hypotheses if h.status != HypothesisStatus.ACTIVE]
            active = [h for h in self._hypotheses if h.status == HypothesisStatus.ACTIVE]
            self._hypotheses = resolved[-30:] + active

        self._save()
        return h

    def verify(self, hypothesis_id: str, outcome: str, succeeded: bool) -> Hypothesis | None:
        """Record the result of testing a hypothesis."""
        now = datetime.now(timezone.utc).isoformat()

        for h in self._hypotheses:
            if h.id == hypothesis_id:
                h.status = HypothesisStatus.CONFIRMED if succeeded else HypothesisStatus.REJECTED
                h.resolved_at = now
                h.outcome = outcome

                if not succeeded:
                    # Update rejection count for the context
                    for other in self._hypotheses:
                        if other.context == h.context and other.status == HypothesisStatus.ACTIVE:
                            other.rejection_count = sum(
                                1
                                for x in self._hypotheses
                                if x.context == h.context
                                and x.status == HypothesisStatus.REJECTED
                            )

                self._save()
                return h

        return None

    def get_status(self) -> dict:
        """Get metacognition status — active hypotheses, warnings, patterns."""
        active = [h for h in self._hypotheses if h.status == HypothesisStatus.ACTIVE]
        recent_rejected = [
            h
            for h in self._hypotheses
            if h.status == HypothesisStatus.REJECTED
        ][-5:]

        # Detect stuck patterns per context
        warnings = []
        contexts_seen: dict[str, int] = {}
        for h in self._hypotheses:
            if h.status == HypothesisStatus.REJECTED:
                contexts_seen[h.context] = contexts_seen.get(h.context, 0) + 1

        for context, count in contexts_seen.items():
            if count >= self.APPROACH_CHANGE_THRESHOLD:
                # Check if there's still an active hypothesis in this context
                has_active = any(
                    h.context == context and h.status == HypothesisStatus.ACTIVE
                    for h in self._hypotheses
                )
                if has_active:
                    warnings.append(
                        f"'{context}' で仮説が{count}回棄却されています。"
                        f"アプローチ自体を変えるか、人間に相談してください。"
                    )

        # Summary stats
        total = len(self._hypotheses)
        confirmed = sum(1 for h in self._hypotheses if h.status == HypothesisStatus.CONFIRMED)
        rejected = sum(1 for h in self._hypotheses if h.status == HypothesisStatus.REJECTED)

        return {
            "active_hypotheses": [h.to_dict() for h in active],
            "recent_rejections": [h.to_dict() for h in recent_rejected],
            "warnings": warnings,
            "stats": {
                "total": total,
                "confirmed": confirmed,
                "rejected": rejected,
                "confirmation_rate": f"{confirmed / max(confirmed + rejected, 1):.0%}",
            },
        }

    def get_context_history(self, context: str) -> list[Hypothesis]:
        """Get all hypotheses for a specific context."""
        return [h for h in self._hypotheses if h.context == context]
