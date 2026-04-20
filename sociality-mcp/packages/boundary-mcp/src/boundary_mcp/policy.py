"""TOML-backed policy loading for boundary-mcp."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from social_core import DEFAULT_POLICY_TIMEZONE
from social_core import in_quiet_hours as _in_quiet_hours_core


@dataclass(slots=True)
class GlobalPolicy:
    quiet_hours: list[str] = field(default_factory=list)
    max_nudges_per_hour: int = 2
    timezone: str = DEFAULT_POLICY_TIMEZONE


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
    """Resolve the socialPolicy.toml location.

    Search order:
    1. Explicit ``path`` argument (for tests / programmatic callers).
    2. ``SOCIAL_POLICY_PATH`` environment variable.
    3. Walk upward from ``cwd`` (bounded to 6 parents) looking for
       ``socialPolicy.toml`` — lets the MCP find the file whether it is
       launched from the repo root or from a sub-package's directory.
    4. Fall back to ``cwd / socialPolicy.toml`` even if missing, so the
       caller can surface the intended-but-absent path in errors.
    """

    if path is not None:
        return Path(path).expanduser()
    env = os.environ.get("SOCIAL_POLICY_PATH")
    if env:
        return Path(env).expanduser()

    cwd = Path.cwd()
    for candidate in (cwd, *cwd.parents):
        policy = candidate / "socialPolicy.toml"
        if policy.exists():
            return policy
        # Stop at $HOME or once we have walked 6 levels up, whichever first.
        try:
            if candidate == Path.home():
                break
        except RuntimeError:  # HOME not set
            pass
        if len(candidate.parts) <= max(1, len(cwd.parts) - 6):
            break
    return cwd / "socialPolicy.toml"


def load_policy(path: str | Path | None = None) -> SocialPolicy:
    policy_path = get_policy_path(path)
    if not policy_path.exists():
        return SocialPolicy(
            global_policy=GlobalPolicy(
                quiet_hours=["00:00-07:00"],
                max_nudges_per_hour=2,
                timezone=DEFAULT_POLICY_TIMEZONE,
            )
        )
    data = tomllib.loads(policy_path.read_text(encoding="utf-8"))
    global_block = data.get("global", {})
    return SocialPolicy(
        global_policy=GlobalPolicy(
            quiet_hours=list(global_block.get("quiet_hours", ["00:00-07:00"])),
            max_nudges_per_hour=int(global_block.get("max_nudges_per_hour", 2)),
            timezone=str(global_block.get("timezone", DEFAULT_POLICY_TIMEZONE)),
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


def in_quiet_hours(
    ts: str,
    windows: list[str],
    tz_name: str = DEFAULT_POLICY_TIMEZONE,
) -> tuple[bool, str | None]:
    """Delegate to the shared timezone-aware helper in social_core."""

    return _in_quiet_hours_core(ts, windows, tz_name)
