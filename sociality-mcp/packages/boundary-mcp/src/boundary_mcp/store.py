"""Persistence and evaluation helpers for boundary-mcp."""

from __future__ import annotations

import json
import uuid
from datetime import timedelta
from pathlib import Path
from typing import Any

from social_core import (
    EventStore,
    SocialDB,
    confidence_from_evidence,
    ensure_iso8601,
    parse_timestamp,
)

from .policy import SocialPolicy, in_quiet_hours, load_policy
from .schemas import EvaluateActionResult, QuietModeState, ReviewSocialPostResult

PRIVATE_STATE_WORDS = ("tired", "疲れ", "sleepy", "眠い", "stress", "しんど", "元気ない")
ROUTINE_WORDS = ("meeting", "会議", "dentist", "sleep", "commute", "routine")


class BoundaryStore:
    """Action gating against policy and consent state."""

    def __init__(
        self,
        path: str | Path | None = None,
        db: SocialDB | None = None,
        policy_path: str | Path | None = None,
    ) -> None:
        self.db = db or SocialDB(path)
        self.events = EventStore(self.db)
        self.policy_path = policy_path

    def close(self) -> None:
        self.db.close()

    @property
    def policy(self) -> SocialPolicy:
        return load_policy(self.policy_path)

    def record_consent(
        self,
        *,
        person_id: str,
        consent_type: str,
        value: bool,
        source: str,
    ) -> dict[str, str]:
        self._ensure_person(person_id)
        consent_id = f"consent_{uuid.uuid4().hex[:10]}"
        created_at = ensure_iso8601(
            self.events.get_latest_timestamp(person_id=person_id) or "2026-01-01T12:00:00+00:00"
        )
        with self.db.transaction() as connection:
            connection.execute(
                """
                INSERT INTO consents(
                    consent_id,
                    person_id,
                    consent_type,
                    value,
                    source,
                    created_at,
                    expires_at
                )
                VALUES (?, ?, ?, ?, ?, ?, NULL)
                ON CONFLICT(person_id, consent_type) DO UPDATE SET
                    consent_id = excluded.consent_id,
                    value = excluded.value,
                    source = excluded.source,
                    created_at = excluded.created_at
                """,
                (consent_id, person_id, consent_type, int(value), source, created_at),
            )
        return {"consent_id": consent_id}

    def evaluate_action(
        self,
        *,
        action_type: str,
        channel: str | None = None,
        person_id: str | None = None,
        context: dict[str, Any] | None = None,
        payload_preview: dict[str, Any] | None = None,
        urgency: str = "low",
    ) -> EvaluateActionResult:
        context = context or {}
        payload_preview = payload_preview or {}
        ts = str(
            context.get("time_local")
            or self.events.get_latest_timestamp(person_id=person_id)
            or "2026-01-01T12:00:00+00:00"
        )
        reasons: list[str] = []
        alternatives: list[str] = []
        decision = "allow"

        quiet_active, quiet_until = in_quiet_hours(
            ts, self.policy.global_policy.quiet_hours, self.policy.global_policy.timezone
        )
        high_urgency = urgency in {"high", "critical"} or bool(context.get("health_safety"))
        person_rule = self.policy.person_rule_for(person_id)

        if quiet_active and action_type in {"say", "nudge_human", "speak_loud"}:
            if high_urgency:
                reasons.append("quiet hours are active")
            else:
                decision = "deny"
                reasons.append("quiet hours are active")
                alternatives.append("wait until quiet hours end")

        if person_rule and action_type in set(person_rule.avoid_actions) and not high_urgency:
            decision = "deny"
            reasons.append(f"person-specific rule avoids {action_type}")
            alternatives.append("switch to a quieter channel")

        if (action_type == "post_tweet" or channel == "x") and context.get("scene_contains_face"):
            rule = self.policy.posting_rule_for("x")
            consent = self._latest_consent(person_id, "public_face_photo") if person_id else None
            if rule and rule.require_face_consent and consent is not True:
                decision = "deny"
                reasons.append("face present and consent not recorded")
                alternatives.extend(
                    [
                        "remove person-specific details",
                        "ask for consent first",
                        "keep as private memory instead",
                    ]
                )

        recent_nudges = self._recent_nudge_count(person_id)
        max_nudges = self.policy.global_policy.max_nudges_per_hour
        if (
            recent_nudges >= max_nudges
            and action_type in {"say", "nudge_human"}
            and not high_urgency
        ):
            decision = "deny"
            reasons.append("recent nudge saturation exceeds policy")
            alternatives.append("cool down before nudging again")

        topic = str(payload_preview.get("topic") or payload_preview.get("text") or "")
        if (
            topic
            and self._recent_topic_repeats(person_id, topic)
            and action_type in {"say", "nudge_human"}
            and not high_urgency
        ):
            decision = "deny"
            reasons.append("same topic was nudged recently")
            alternatives.append("wait before repeating the same reminder")

        if high_urgency and reasons:
            return EvaluateActionResult(
                decision="allow_with_override",
                confidence=0.91,
                reasons=["urgent health/safety context overrides quieter social rules", *reasons],
                safer_alternatives=alternatives,
            )

        confidence = confidence_from_evidence(
            [0.95 if decision == "deny" else 0.65, 0.6 if reasons else 0.3], 0.05
        )
        if quiet_active and quiet_until and "quiet hours are active" in reasons:
            reasons.append(f"quiet mode lasts until {quiet_until}")
        return EvaluateActionResult(
            decision=decision,
            confidence=confidence,
            reasons=reasons or ["no matching boundary rule fired"],
            safer_alternatives=alternatives,
        )

    def review_social_post(
        self,
        *,
        channel: str,
        text: str,
        scene_contains_face: bool,
        person_mentions: list[str] | None = None,
    ) -> ReviewSocialPostResult:
        person_mentions = person_mentions or []
        issues: list[str] = []
        risk = "low"
        recommendation = "post_ok"
        lowered = text.lower()
        if person_mentions and any(word in lowered for word in PRIVATE_STATE_WORDS):
            issues.append("references another person's internal state")
            risk = "medium"
            recommendation = "rewrite"
        if person_mentions and any(word in lowered for word in ROUTINE_WORDS):
            issues.append("may expose private routine")
            risk = "medium"
            recommendation = "rewrite"
        if scene_contains_face and person_mentions:
            issues.append("face present while a person is implicated")
            risk = "high"
            recommendation = "deny"
        return ReviewSocialPostResult(risk_level=risk, issues=issues, recommendation=recommendation)

    def get_quiet_mode_state(self, *, ts: str) -> QuietModeState:
        active, until = in_quiet_hours(
            ts, self.policy.global_policy.quiet_hours, self.policy.global_policy.timezone
        )
        reasons = (
            ["within configured quiet hours"] if active else ["outside configured quiet hours"]
        )
        return QuietModeState(active=active, confidence=0.95, reasons=reasons, until=until)

    def _latest_consent(self, person_id: str | None, consent_type: str) -> bool | None:
        if person_id is None:
            return None
        row = self.db.fetchone(
            "SELECT value FROM consents WHERE person_id = ? AND consent_type = ?",
            (person_id, consent_type),
        )
        if row is None:
            return None
        return bool(row["value"])

    def _recent_nudge_count(self, person_id: str | None) -> int:
        latest_ts = self.events.get_latest_timestamp(person_id=person_id)
        since = None
        if latest_ts:
            since = ensure_iso8601(parse_timestamp(latest_ts) - timedelta(hours=1))
        events = self.events.fetch_events(person_id=person_id, since=since, limit=100)
        count = 0
        for event in events:
            if event.kind not in {"touchpoint", "agent_utterance"}:
                continue
            action = str(event.payload_json.get("action", "")).lower()
            style = str(event.payload_json.get("style", "")).lower()
            if "nudge" in action or "nudge" in style or "reminder" in action:
                count += 1
        return count

    def _recent_topic_repeats(self, person_id: str | None, topic: str) -> bool:
        latest_ts = self.events.get_latest_timestamp(person_id=person_id)
        since = None
        if latest_ts:
            since = ensure_iso8601(parse_timestamp(latest_ts) - timedelta(hours=1))
        events = self.events.fetch_events(person_id=person_id, since=since, limit=50)
        normalized = topic.lower()
        for event in events:
            payload_text = json.dumps(event.payload_json, ensure_ascii=False).lower()
            if normalized and normalized in payload_text:
                return True
        return False

    def _ensure_person(self, person_id: str) -> None:
        if (
            self.db.fetchone("SELECT person_id FROM persons WHERE person_id = ?", (person_id,))
            is not None
        ):
            return
        now = ensure_iso8601(self.events.get_latest_timestamp() or "2026-01-01T12:00:00+00:00")
        with self.db.transaction() as connection:
            connection.execute(
                """
                INSERT INTO persons(
                    person_id,
                    canonical_name,
                    role,
                    profile_json,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, NULL, '{}', ?, ?)
                """,
                (person_id, person_id, now, now),
            )
