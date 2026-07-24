"""Pure configuration and validation helpers for repository onboarding."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

SMALL_EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
BASE_EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
DEFAULT_VOICEVOX_URL = "http://localhost:50021"

CORE_SERVER_NAMES = (
    "memory",
    "desire-system",
    "sociality",
    "individual-kernel",
)


@dataclass(frozen=True)
class FeatureSelection:
    """Capabilities selected for a generated MCP configuration."""

    profile: str = "core"
    camera: str | None = None
    voice: str | None = None
    x_enabled: bool = False
    system_temperature: bool = False
    embedding_model: str = "small"

    def __post_init__(self) -> None:
        if self.profile != "core":
            raise ValueError(f"Unsupported profile: {self.profile}")
        if self.camera not in {None, "usb", "tapo"}:
            raise ValueError(f"Unsupported camera: {self.camera}")
        if self.voice not in {None, "voicevox", "elevenlabs"}:
            raise ValueError(f"Unsupported voice: {self.voice}")
        if self.embedding_model not in {"small", "base"}:
            raise ValueError(f"Unsupported embedding model: {self.embedding_model}")


@dataclass(frozen=True)
class ServerSpec:
    """Canonical workspace command for one setup-managed MCP server."""

    name: str
    package: str
    entrypoint: str

    def command(self, environment: Mapping[str, str] | None = None) -> dict[str, Any]:
        result: dict[str, Any] = {
            "command": "uv",
            "args": ["run", "--package", self.package, self.entrypoint],
        }
        if environment:
            result["env"] = dict(environment)
        return result


SERVER_SPECS = {
    "memory": ServerSpec("memory", "memory-mcp", "memory-mcp"),
    "desire-system": ServerSpec("desire-system", "desire-system", "desire-system"),
    "sociality": ServerSpec("sociality", "sociality-mcp", "sociality-mcp"),
    "individual-kernel": ServerSpec(
        "individual-kernel",
        "individual-kernel-mcp",
        "individual-kernel-mcp",
    ),
    "usb-webcam": ServerSpec("usb-webcam", "usb-webcam-mcp", "usb-webcam-mcp"),
    "wifi-cam": ServerSpec("wifi-cam", "wifi-cam-mcp", "wifi-cam-mcp"),
    "tts": ServerSpec("tts", "tts-mcp", "tts-mcp"),
    "x-mcp": ServerSpec("x-mcp", "x-mcp", "x-mcp"),
    "system-temperature": ServerSpec(
        "system-temperature",
        "system-temperature-mcp",
        "system-temperature-mcp",
    ),
}

TAPO_REQUIRED_ENVIRONMENT = (
    "TAPO_CAMERA_HOST",
    "TAPO_USERNAME",
    "TAPO_PASSWORD",
)
ELEVENLABS_REQUIRED_ENVIRONMENT = ("ELEVENLABS_API_KEY",)
ELEVENLABS_OPTIONAL_ENVIRONMENT = ("ELEVENLABS_VOICE_ID",)
X_REQUIRED_ENVIRONMENT = (
    "XAI_API_KEY",
    "X_CONSUMER_KEY",
    "X_CONSUMER_SECRET",
    "X_ACCESS_TOKEN",
    "X_ACCESS_TOKEN_SECRET",
)

_PRIVATE_KEYS = {
    "TAPO_USERNAME",
    "ELEVENLABS_VOICE_ID",
}
_SECRET_KEY_PARTS = ("KEY", "TOKEN", "SECRET", "PASSWORD")
_PLACEHOLDER_MARKERS = (
    "<redacted>",
    "changeme",
    "placeholder",
    "your-",
    "your_",
    ".xxx",
    "example.com",
)


def required_environment(selection: FeatureSelection) -> tuple[str, ...]:
    """Return required environment keys in stable prompt/report order."""

    required: list[str] = []
    if selection.camera == "tapo":
        required.extend(TAPO_REQUIRED_ENVIRONMENT)
    if selection.voice == "elevenlabs":
        required.extend(ELEVENLABS_REQUIRED_ENVIRONMENT)
    if selection.x_enabled:
        required.extend(X_REQUIRED_ENVIRONMENT)
    return tuple(required)


def missing_environment(
    selection: FeatureSelection,
    environment: Mapping[str, str],
) -> tuple[str, ...]:
    """Return selected required keys whose values are empty."""

    return tuple(
        key for key in required_environment(selection) if not environment.get(key, "").strip()
    )


def is_placeholder_value(value: str) -> bool:
    """Return whether a value looks like an example rather than real config."""

    normalized = value.strip().lower()
    return not normalized or any(marker in normalized for marker in _PLACEHOLDER_MARKERS)


def placeholder_environment(
    selection: FeatureSelection,
    environment: Mapping[str, str],
) -> tuple[str, ...]:
    """Return required keys that contain common example/placeholder values."""

    return tuple(
        key
        for key in required_environment(selection)
        if environment.get(key) and is_placeholder_value(environment[key])
    )


def _selected_environment(
    environment: Mapping[str, str],
    required: tuple[str, ...],
    optional: tuple[str, ...] = (),
) -> dict[str, str]:
    result = {key: environment[key] for key in required}
    result.update(
        (key, environment[key])
        for key in optional
        if environment.get(key, "").strip()
    )
    return result


def build_mcp_config(
    selection: FeatureSelection,
    environment: Mapping[str, str],
) -> dict[str, Any]:
    """Build a deterministic MCP configuration for the selected capabilities."""

    invalid = missing_environment(selection, environment) + placeholder_environment(
        selection,
        environment,
    )
    if invalid:
        joined = ", ".join(dict.fromkeys(invalid))
        raise ValueError(f"Missing or placeholder environment values: {joined}")

    embedding_model = (
        SMALL_EMBEDDING_MODEL
        if selection.embedding_model == "small"
        else BASE_EMBEDDING_MODEL
    )
    servers: dict[str, dict[str, Any]] = {
        "memory": SERVER_SPECS["memory"].command(
            {"MEMORY_EMBEDDING_MODEL": embedding_model}
        ),
        "desire-system": SERVER_SPECS["desire-system"].command(),
        "sociality": SERVER_SPECS["sociality"].command(),
        "individual-kernel": SERVER_SPECS["individual-kernel"].command(
            {
                "SOCIAL_DB_PATH": "~/.claude/sociality/social.db",
                "MEMORY_HTTP_PORT": "18900",
            }
        ),
    }

    if selection.camera == "usb":
        servers["usb-webcam"] = SERVER_SPECS["usb-webcam"].command()
    elif selection.camera == "tapo":
        servers["wifi-cam"] = SERVER_SPECS["wifi-cam"].command(
            _selected_environment(environment, TAPO_REQUIRED_ENVIRONMENT)
        )

    if selection.voice == "voicevox":
        servers["tts"] = SERVER_SPECS["tts"].command(
            {
                "VOICEVOX_URL": environment.get("VOICEVOX_URL", "").strip()
                or DEFAULT_VOICEVOX_URL,
                "TTS_DEFAULT_ENGINE": "voicevox",
            }
        )
    elif selection.voice == "elevenlabs":
        voice_environment = _selected_environment(
            environment,
            ELEVENLABS_REQUIRED_ENVIRONMENT,
            ELEVENLABS_OPTIONAL_ENVIRONMENT,
        )
        voice_environment["TTS_DEFAULT_ENGINE"] = "elevenlabs"
        servers["tts"] = SERVER_SPECS["tts"].command(voice_environment)

    if selection.x_enabled:
        servers["x-mcp"] = SERVER_SPECS["x-mcp"].command(
            _selected_environment(environment, X_REQUIRED_ENVIRONMENT)
        )

    if selection.system_temperature:
        servers["system-temperature"] = SERVER_SPECS["system-temperature"].command()

    return {"mcpServers": servers}


def _is_private_key(key: str) -> bool:
    upper = key.upper()
    return key in _PRIVATE_KEYS or any(part in upper for part in _SECRET_KEY_PARTS)


def redact_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return a deep redacted copy suitable for terminal output."""

    def redact(value: Any, key: str | None = None) -> Any:
        if key is not None and _is_private_key(key):
            return "<redacted>"
        if isinstance(value, Mapping):
            return {item_key: redact(item, item_key) for item_key, item in value.items()}
        if isinstance(value, list):
            return [redact(item) for item in value]
        return deepcopy(value)

    return redact(config)


def configs_equivalent(left: object, right: object) -> bool:
    """Compare parsed JSON values without depending on mapping order."""

    return left == right
