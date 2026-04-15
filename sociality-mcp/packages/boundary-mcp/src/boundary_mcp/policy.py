"""TOML-backed policy loading for boundary-mcp."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path


@dataclass(slots=True)
class GlobalPolicy:
    quiet_hours: list[str] = field(default_factory=list)
    max_nudges_per_hour: int = 2


@dataclass(slots=True)
class PrivacyZone:
    name: str
    camera_presets: list[str] = field(default_factory=list)
    deny_actions: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PostingRule:
    channel: str
    require_face_consent: bool = False
    require_review_if_person_present: bool = False


@dataclass(slots=True)
class PersonRule:
    person_id: str
    avoid_actions: list[str] = field(default_factory=list)
    preferred_nudge_style: str | None = None


@dataclass(slots=True)
class SocialPolicy:
    global_policy: GlobalPolicy = field(default_factory=GlobalPolicy)
    privacy_zones: list[PrivacyZone] = field(default_factory=list)
    posting_rules: list[PostingRule] = field(default_factory=list)
    person_rules: list[PersonRule] = field(default_factory=list)

    def posting_rule_for(self, channel: str) -> PostingRule | None:
        return next((rule for rule in self.posting_rules if rule.channel == channel), None)

    def person_rule_for(self, person_id: str | None) -> PersonRule | None:
        if person_id is None:
            return None
        return next((rule for rule in self.person_rules if rule.person_id == person_id), None)


def get_policy_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path).expanduser()
    env = os.environ.get("SOCIAL_POLICY_PATH")
    if env:
        return Path(env).expanduser()
    return Path.cwd() / "socialPolicy.toml"


def load_policy(path: str | Path | None = None) -> SocialPolicy:
    policy_path = get_policy_path(path)
    if not policy_path.exists():
        return SocialPolicy(
            global_policy=GlobalPolicy(quiet_hours=["00:00-07:00"], max_nudges_per_hour=2)
        )
    data = tomllib.loads(policy_path.read_text(encoding="utf-8"))
    global_block = data.get("global", {})
    return SocialPolicy(
        global_policy=GlobalPolicy(
            quiet_hours=list(global_block.get("quiet_hours", ["00:00-07:00"])),
            max_nudges_per_hour=int(global_block.get("max_nudges_per_hour", 2)),
        ),
        privacy_zones=[
            PrivacyZone(
                name=item["name"],
                camera_presets=list(item.get("camera_presets", [])),
                deny_actions=list(item.get("deny_actions", [])),
            )
            for item in data.get("privacy_zones", [])
        ],
        posting_rules=[
            PostingRule(
                channel=item["channel"],
                require_face_consent=bool(item.get("require_face_consent", False)),
                require_review_if_person_present=bool(
                    item.get("require_review_if_person_present", False)
                ),
            )
            for item in data.get("posting_rules", [])
        ],
        person_rules=[
            PersonRule(
                person_id=item["person_id"],
                avoid_actions=list(item.get("avoid_actions", [])),
                preferred_nudge_style=item.get("preferred_nudge_style"),
            )
            for item in data.get("person_rules", [])
        ],
    )


def in_quiet_hours(ts: str, windows: list[str]) -> tuple[bool, str | None]:
    reference = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    current_minutes = reference.hour * 60 + reference.minute
    for window in windows:
        start_text, end_text = window.split("-", 1)
        start = _to_minutes(start_text)
        end = _to_minutes(end_text)
        if start <= end and start <= current_minutes < end:
            return True, _window_end(reference, end).isoformat(timespec="seconds")
        if start > end and (current_minutes >= start or current_minutes < end):
            return True, _window_end(reference, end).isoformat(timespec="seconds")
    return False, None


def _to_minutes(value: str) -> int:
    hours, minutes = value.split(":", 1)
    return int(hours) * 60 + int(minutes)


def _window_end(reference: datetime, end_minutes: int) -> datetime:
    day_offset = 1 if end_minutes <= (reference.hour * 60 + reference.minute) else 0
    midnight = reference.replace(hour=0, minute=0, second=0, microsecond=0)
    return midnight + timedelta(days=day_offset, minutes=end_minutes)
