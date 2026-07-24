from __future__ import annotations

import json
import os
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

import pytest

from scripts.onboarding import (
    BASE_EMBEDDING_MODEL,
    SMALL_EMBEDDING_MODEL,
    FeatureSelection,
    build_mcp_config,
    configs_equivalent,
    missing_environment,
    redact_config,
)
from scripts.setup_io import (
    ConfigAction,
    ConfigConflictError,
    apply_config_plan,
    copy_policy_if_missing,
    plan_config_write,
)

CORE_SERVERS = {
    "memory",
    "desire-system",
    "sociality",
    "individual-kernel",
}


def test_core_profile_has_exactly_the_hardware_free_servers() -> None:
    config = build_mcp_config(FeatureSelection(), {})

    assert set(config["mcpServers"]) == CORE_SERVERS
    assert config["mcpServers"]["memory"]["env"]["MEMORY_EMBEDDING_MODEL"] == (
        SMALL_EMBEDDING_MODEL
    )
    assert config["mcpServers"]["individual-kernel"]["env"] == {
        "SOCIAL_DB_PATH": "~/.claude/sociality/social.db",
        "MEMORY_HTTP_PORT": "18900",
    }


def test_generated_servers_use_root_workspace_entrypoints() -> None:
    config = build_mcp_config(
        FeatureSelection(
            camera="usb",
            voice="voicevox",
            x_enabled=True,
            system_temperature=True,
        ),
        {
            "XAI_API_KEY": "xai-secret",
            "X_CONSUMER_KEY": "consumer-key",
            "X_CONSUMER_SECRET": "consumer-secret",
            "X_ACCESS_TOKEN": "access-token",
            "X_ACCESS_TOKEN_SECRET": "access-secret",
        },
    )

    expected = {
        "memory": ("memory-mcp", "memory-mcp"),
        "desire-system": ("desire-system", "desire-system"),
        "sociality": ("sociality-mcp", "sociality-mcp"),
        "individual-kernel": ("individual-kernel-mcp", "individual-kernel-mcp"),
        "usb-webcam": ("usb-webcam-mcp", "usb-webcam-mcp"),
        "tts": ("tts-mcp", "tts-mcp"),
        "x-mcp": ("x-mcp", "x-mcp"),
        "system-temperature": ("system-temperature-mcp", "system-temperature-mcp"),
    }
    for name, (package, entrypoint) in expected.items():
        server = config["mcpServers"][name]
        assert server["command"] == "uv"
        assert server["args"] == [
            "run",
            "--package",
            package,
            entrypoint,
        ]


def test_base_embedding_model_is_explicitly_selectable() -> None:
    config = build_mcp_config(FeatureSelection(embedding_model="base"), {})

    assert config["mcpServers"]["memory"]["env"]["MEMORY_EMBEDDING_MODEL"] == (
        BASE_EMBEDDING_MODEL
    )


@pytest.mark.parametrize(
    ("selection", "environment", "added_server"),
    [
        (FeatureSelection(camera="usb"), {}, "usb-webcam"),
        (
            FeatureSelection(camera="tapo"),
            {
                "TAPO_CAMERA_HOST": "192.0.2.10",
                "TAPO_USERNAME": "camera-user",
                "TAPO_PASSWORD": "camera-password",
            },
            "wifi-cam",
        ),
        (FeatureSelection(voice="voicevox"), {}, "tts"),
        (
            FeatureSelection(voice="elevenlabs"),
            {"ELEVENLABS_API_KEY": "eleven-secret"},
            "tts",
        ),
        (
            FeatureSelection(x_enabled=True),
            {
                "XAI_API_KEY": "xai-secret",
                "X_CONSUMER_KEY": "consumer-key",
                "X_CONSUMER_SECRET": "consumer-secret",
                "X_ACCESS_TOKEN": "access-token",
                "X_ACCESS_TOKEN_SECRET": "access-secret",
            },
            "x-mcp",
        ),
        (
            FeatureSelection(system_temperature=True),
            {},
            "system-temperature",
        ),
    ],
)
def test_optional_capabilities_add_only_the_selected_server(
    selection: FeatureSelection,
    environment: dict[str, str],
    added_server: str,
) -> None:
    config = build_mcp_config(selection, environment)

    assert set(config["mcpServers"]) == CORE_SERVERS | {added_server}


def test_voicevox_uses_local_default_url() -> None:
    config = build_mcp_config(FeatureSelection(voice="voicevox"), {})

    assert config["mcpServers"]["tts"]["env"] == {
        "VOICEVOX_URL": "http://localhost:50021",
        "TTS_DEFAULT_ENGINE": "voicevox",
    }


def test_elevenlabs_voice_id_is_optional_but_preserved_when_present() -> None:
    without_voice = build_mcp_config(
        FeatureSelection(voice="elevenlabs"),
        {"ELEVENLABS_API_KEY": "eleven-secret"},
    )
    with_voice = build_mcp_config(
        FeatureSelection(voice="elevenlabs"),
        {
            "ELEVENLABS_API_KEY": "eleven-secret",
            "ELEVENLABS_VOICE_ID": "voice-id",
        },
    )

    assert without_voice["mcpServers"]["tts"]["env"] == {
        "ELEVENLABS_API_KEY": "eleven-secret",
        "TTS_DEFAULT_ENGINE": "elevenlabs",
    }
    assert with_voice["mcpServers"]["tts"]["env"]["ELEVENLABS_VOICE_ID"] == "voice-id"


def test_missing_environment_lists_every_selected_requirement() -> None:
    selection = FeatureSelection(camera="tapo", x_enabled=True)

    assert missing_environment(selection, {"TAPO_CAMERA_HOST": "192.0.2.10"}) == (
        "TAPO_USERNAME",
        "TAPO_PASSWORD",
        "XAI_API_KEY",
        "X_CONSUMER_KEY",
        "X_CONSUMER_SECRET",
        "X_ACCESS_TOKEN",
        "X_ACCESS_TOKEN_SECRET",
    )


def test_build_rejects_missing_or_placeholder_credentials() -> None:
    with pytest.raises(ValueError, match="TAPO_PASSWORD"):
        build_mcp_config(
            FeatureSelection(camera="tapo"),
            {
                "TAPO_CAMERA_HOST": "192.168.1.xxx",
                "TAPO_USERNAME": "your-username",
                "TAPO_PASSWORD": "your-password",
            },
        )


def test_redaction_is_recursive_and_does_not_mutate_input() -> None:
    config = build_mcp_config(
        FeatureSelection(camera="tapo", voice="elevenlabs"),
        {
            "TAPO_CAMERA_HOST": "192.0.2.10",
            "TAPO_USERNAME": "camera-user",
            "TAPO_PASSWORD": "camera-password",
            "ELEVENLABS_API_KEY": "eleven-secret",
            "ELEVENLABS_VOICE_ID": "voice-id",
        },
    )
    original = deepcopy(config)

    redacted = redact_config(config)

    assert redacted["mcpServers"]["wifi-cam"]["env"] == {
        "TAPO_CAMERA_HOST": "192.0.2.10",
        "TAPO_USERNAME": "<redacted>",
        "TAPO_PASSWORD": "<redacted>",
    }
    assert redacted["mcpServers"]["tts"]["env"]["ELEVENLABS_API_KEY"] == "<redacted>"
    assert redacted["mcpServers"]["tts"]["env"]["ELEVENLABS_VOICE_ID"] == "<redacted>"
    assert config == original


def test_config_equivalence_ignores_mapping_order() -> None:
    left = {"mcpServers": {"memory": {"command": "uv", "args": ["run"]}}}
    right = {"mcpServers": {"memory": {"args": ["run"], "command": "uv"}}}

    assert configs_equivalent(left, right)


def test_new_config_is_created_atomically_with_private_mode(tmp_path: Path) -> None:
    destination = tmp_path / ".mcp.json"
    config = build_mcp_config(FeatureSelection(), {})
    plan = plan_config_write(destination, config)

    assert plan.action is ConfigAction.CREATE
    assert plan.backup is None

    apply_config_plan(plan, config)

    assert json.loads(destination.read_text()) == config
    if os.name == "posix":
        assert destination.stat().st_mode & 0o777 == 0o600
    assert list(tmp_path.glob(".mcp.json.*.tmp")) == []


def test_equivalent_existing_config_is_kept_byte_for_byte(tmp_path: Path) -> None:
    destination = tmp_path / ".mcp.json"
    config = build_mcp_config(FeatureSelection(), {})
    original = json.dumps(config, separators=(",", ":"))
    destination.write_text(original)

    plan = plan_config_write(destination, config)
    apply_config_plan(plan, config)

    assert plan.action is ConfigAction.KEEP
    assert destination.read_text() == original


def test_different_existing_config_is_refused_without_force(tmp_path: Path) -> None:
    destination = tmp_path / ".mcp.json"
    destination.write_text('{"mcpServers":{"custom":{"command":"custom"}}}\n')

    with pytest.raises(ConfigConflictError, match="--force"):
        plan_config_write(
            destination,
            build_mcp_config(FeatureSelection(), {}),
        )


def test_invalid_existing_config_is_refused_without_force(tmp_path: Path) -> None:
    destination = tmp_path / ".mcp.json"
    destination.write_text("{invalid json")

    with pytest.raises(ConfigConflictError, match="valid JSON"):
        plan_config_write(
            destination,
            build_mcp_config(FeatureSelection(), {}),
        )


def test_force_backs_up_before_replacing_config(tmp_path: Path) -> None:
    destination = tmp_path / ".mcp.json"
    old_content = '{"mcpServers":{"custom":{"command":"custom"}}}\n'
    destination.write_text(old_content)
    config = build_mcp_config(FeatureSelection(), {})
    now = datetime(2026, 7, 24, 12, 34, 56, tzinfo=UTC)

    plan = plan_config_write(destination, config, force=True, now=now)
    apply_config_plan(plan, config)

    assert plan.action is ConfigAction.REPLACE
    assert plan.backup == tmp_path / ".mcp.json.backup-20260724-123456"
    assert plan.backup.read_text() == old_content
    assert json.loads(destination.read_text()) == config


def test_force_chooses_a_unique_backup_name(tmp_path: Path) -> None:
    destination = tmp_path / ".mcp.json"
    destination.write_text('{"old":true}\n')
    occupied = tmp_path / ".mcp.json.backup-20260724-123456"
    occupied.write_text("existing backup")
    now = datetime(2026, 7, 24, 12, 34, 56, tzinfo=UTC)

    plan = plan_config_write(
        destination,
        build_mcp_config(FeatureSelection(), {}),
        force=True,
        now=now,
    )

    assert plan.backup == tmp_path / ".mcp.json.backup-20260724-123456-1"


def test_social_policy_is_copied_only_when_missing(tmp_path: Path) -> None:
    source = tmp_path / "socialPolicy.example.toml"
    destination = tmp_path / "socialPolicy.toml"
    source.write_text('version = 1\nname = "example"\n')

    assert copy_policy_if_missing(source, destination)
    assert destination.read_text() == source.read_text()

    destination.write_text('version = 1\nname = "custom"\n')
    assert not copy_policy_if_missing(source, destination)
    assert 'name = "custom"' in destination.read_text()
