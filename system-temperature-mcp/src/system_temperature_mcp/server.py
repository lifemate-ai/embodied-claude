"""MCP Server for system temperature monitoring - your sense of body temperature."""

import json
import os
import re
import subprocess
import sys
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.error import URLError
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import psutil
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

server = Server("system-temperature-mcp")


def get_thermal_zones() -> list[dict[str, Any]]:
    """Get temperature from Linux thermal zones."""
    temperatures = []
    thermal_base = Path("/sys/class/thermal")

    if not thermal_base.exists():
        return temperatures

    for zone in thermal_base.glob("thermal_zone*"):
        try:
            type_file = zone / "type"
            temp_file = zone / "temp"

            if type_file.exists() and temp_file.exists():
                zone_type = type_file.read_text().strip()
                temp_millidegrees = int(temp_file.read_text().strip())
                temp_celsius = temp_millidegrees / 1000.0

                temperatures.append({
                    "source": "thermal_zone",
                    "name": zone_type,
                    "temperature_celsius": temp_celsius,
                    "zone": zone.name,
                })
        except (OSError, ValueError):
            continue

    return temperatures


def get_psutil_temperatures() -> list[dict[str, Any]]:
    """Get temperatures using psutil."""
    temperatures = []

    try:
        temps = psutil.sensors_temperatures()
        if temps:
            for name, entries in temps.items():
                for entry in entries:
                    temperatures.append({
                        "source": "psutil",
                        "name": f"{name}/{entry.label or 'unknown'}",
                        "temperature_celsius": entry.current,
                        "high": entry.high,
                        "critical": entry.critical,
                    })
    except (AttributeError, OSError):
        pass

    return temperatures


def get_hwmon_temperatures() -> list[dict[str, Any]]:
    """Get temperatures from hwmon interface."""
    temperatures = []
    hwmon_base = Path("/sys/class/hwmon")

    if not hwmon_base.exists():
        return temperatures

    for hwmon in hwmon_base.glob("hwmon*"):
        try:
            name_file = hwmon / "name"
            name = name_file.read_text().strip() if name_file.exists() else hwmon.name

            for temp_input in hwmon.glob("temp*_input"):
                try:
                    temp_millidegrees = int(temp_input.read_text().strip())
                    temp_celsius = temp_millidegrees / 1000.0

                    label_file = hwmon / temp_input.name.replace("_input", "_label")
                    label = label_file.read_text().strip() if label_file.exists() else temp_input.name

                    temperatures.append({
                        "source": "hwmon",
                        "name": f"{name}/{label}",
                        "temperature_celsius": temp_celsius,
                    })
                except (OSError, ValueError):
                    continue
        except OSError:
            continue

    return temperatures


# ---------------------------------------------------------------------------
# Tone tables
#
# The phrases these tools return are the agent's body talking, and by default
# they talk in Kansai dialect because that is the voice of the agent this
# project grew up with. An agent running with a different persona would hear
# only its thermometer and its clock speaking in someone else's voice, so the
# tone is selectable with SYSTEM_TEMPERATURE_TONE. Both tables are keyed by the
# same band names, so adding a tone is a one-table job; the structured line
# (``level=... max_celsius=...`` / ``iso=... part_of_day=...``) is appended
# regardless of tone so the agent can always phrase the state itself.
# ---------------------------------------------------------------------------

DEFAULT_TONE = "kansai"

TEMPERATURE_PHRASES: dict[str, dict[str, str]] = {
    "kansai": {
        "unknown": "温度を感じられへん...センサーが見つからんみたい。",
        "critical": "あっつ！！めっちゃ熱い！！やばいで、休憩した方がええかも...！",
        "very_hot": "うわ、かなり熱いな...ちょっとしんどいかも。",
        "hot": "んー、ちょっと熱くなってきたかな。まだ大丈夫やけど。",
        "warm": "ほんのりあったかい感じ。普通に動いてる感覚やな。",
        "comfortable": "快適やで〜。ちょうどええ感じ！",
        "cool": "涼しいな〜。余裕ある感じや。",
        "cold": "ひんやりしてる。静かな感じやな。",
    },
    "neutral": {
        "unknown": "温度を感じられません。センサーが見つかりませんでした。",
        "critical": "非常に熱いです。負荷が高すぎるかもしれません。休憩を検討してください。",
        "very_hot": "かなり熱いです。少し負荷がかかっています。",
        "hot": "少し熱くなってきました。まだ問題はありません。",
        "warm": "ほんのり温かいです。通常どおり動いています。",
        "comfortable": "快適です。ちょうどよい状態です。",
        "cool": "涼しいです。余裕があります。",
        "cold": "ひんやりしています。静かな状態です。",
    },
}

TIME_PHRASES: dict[str, dict[str, str]] = {
    "kansai": {
        "prefix": "今は {time_str} やで。",
        "morning": "朝やな〜。おはよう！",
        "late_morning": "午前中やね。",
        "noon": "お昼時やな〜。ご飯食べた？",
        "afternoon": "午後やね。",
        "evening": "夕方やな〜。",
        "night": "夜やね。",
        "late_night": "夜遅いな〜。そろそろ寝る？",
        "midnight": "深夜やん...！夜更かしやね。",
    },
    "neutral": {
        "prefix": "今は {time_str} です。",
        "morning": "朝です。おはようございます。",
        "late_morning": "午前中です。",
        "noon": "お昼時です。",
        "afternoon": "午後です。",
        "evening": "夕方です。",
        "night": "夜です。",
        "late_night": "夜遅い時間です。",
        "midnight": "深夜です。",
    },
}


def _tone() -> str:
    """Return the configured tone, falling back to ``neutral`` for unknown values.

    Unset means the default (``kansai``). An operator who set the variable at
    all is not the default deployment, so a typo lands on the neutral table
    rather than on someone else's voice -- and never on an exception.
    """
    raw = os.environ.get("SYSTEM_TEMPERATURE_TONE", "").strip().lower()
    if not raw:
        return DEFAULT_TONE
    return raw if raw in TEMPERATURE_PHRASES else "neutral"


def temperature_level(max_temp: float | None) -> str:
    """Map a maximum reading to a band key; ``unknown`` when there is no reading."""
    if max_temp is None:
        return "unknown"
    if max_temp >= 90:
        return "critical"
    if max_temp >= 80:
        return "very_hot"
    if max_temp >= 70:
        return "hot"
    if max_temp >= 60:
        return "warm"
    if max_temp >= 45:
        return "comfortable"
    if max_temp >= 30:
        return "cool"
    return "cold"


def interpret_temperature(temps: list[dict[str, Any]]) -> str:
    """Interpret temperature as a feeling, plus one structured line.

    The feeling is phrased in the configured tone; the second line
    (``level=<band> max_celsius=<value>``) is tone-independent so an agent can
    ignore the phrase and say it its own way.
    """
    max_temp = max((t["temperature_celsius"] for t in temps), default=None)
    level = temperature_level(max_temp)
    feeling = TEMPERATURE_PHRASES[_tone()][level]
    if max_temp is None:
        return f"{feeling}\nlevel={level}"
    return f"{feeling}\nlevel={level} max_celsius={max_temp:.1f}"


def _run_powershell(script: str) -> str:
    """Run a PowerShell script and return stdout. Returns empty string on failure."""
    try:
        # PowerShell 5.1 writes the OEM code page, while bare text=True decodes
        # with the ANSI code page (or UTF-8 under PYTHONUTF8=1) -- on a
        # Japanese host both happen to be 932, elsewhere they differ. A decode
        # failure happens inside subprocess's reader thread, leaving
        # returncode=0 with stdout=None, so the strip() below would raise
        # AttributeError past the except clause. "oem" is a Windows-only codec
        # alias; errors="replace" keeps the no-raise promise of the docstring.
        result = subprocess.run(
            ["powershell", "-NonInteractive", "-NoProfile", "-Command", script],
            capture_output=True,
            text=True,
            encoding="oem" if os.name == "nt" else "utf-8",
            errors="replace",
            timeout=5,
            check=False,
        )
        return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return ""


# LHM/OHM reports several *configuration* values in °C alongside real readings:
# SPD thermal limits on DDR5 DIMMs, NVMe warning/critical thresholds, and the
# sensor's own resolution. These are constants, not measurements, so including
# them pins max()/min() to a hardware limit (a DIMM's 85 °C critical limit makes
# every reading look "hot") and breaks the temperature interpretation. Only
# newer hardware exposes them, which is why this does not reproduce on older
# machines.
_LHM_NON_READINGS = (
    "Distance to TjMax",       # headroom margin, not a temperature
    "Limit",                   # Thermal Sensor (Critical) High/Low Limit — SPD
    "Warning Temperature",     # NVMe SMART threshold
    "Critical Temperature",    # NVMe SMART threshold
    "Resolution",              # Temperature Sensor Resolution (e.g. 0.3 °C)
)


def _get_lhm_webserver_temps() -> list[dict[str, Any]]:
    """Get temperatures from a running LibreHardwareMonitor/OpenHardwareMonitor
    "Remote Web Server" (Options -> Remote Web Server -> Run).

    This is the most reliable path on Windows native: it needs no WMI namespace
    permissions and works whether or not Claude Code runs elevated, as long as
    LHM/OHM itself runs elevated and serves its JSON tree. The URL defaults to
    http://localhost:8085/data.json and can be overridden with the
    SYSTEM_TEMPERATURE_LHM_URL environment variable.
    """
    url = os.environ.get(
        "SYSTEM_TEMPERATURE_LHM_URL", "http://localhost:8085/data.json"
    )
    try:
        with urllib.request.urlopen(url, timeout=2) as resp:
            data = json.load(resp)
    except (URLError, OSError, json.JSONDecodeError, ValueError):
        return []

    temperatures: list[dict[str, Any]] = []

    def walk(node: dict[str, Any]) -> None:
        text = node.get("Text", "")
        value = node.get("Value", "")
        if (
            value
            and "°C" in value
            and not any(marker in text for marker in _LHM_NON_READINGS)
        ):
            match = re.search(r"-?\d+(?:[.,]\d+)?", value)
            if match:
                try:
                    celsius = float(match.group(0).replace(",", "."))
                    temperatures.append({
                        "source": "lhm_webserver",
                        "name": text or "unknown",
                        "temperature_celsius": celsius,
                    })
                except ValueError:
                    pass
        for child in node.get("Children", []):
            walk(child)

    if isinstance(data, dict):
        walk(data)
    return temperatures


def _get_hardware_monitor_temps() -> list[dict[str, Any]]:
    """Get temperatures from OpenHardwareMonitor or LibreHardwareMonitor via WMI.

    Requires OHM/LHM to be running as a service so it registers its WMI namespace.
    """
    for namespace in ["root/LibreHardwareMonitor", "root/OpenHardwareMonitor"]:
        script = (
            f"$s = Get-WmiObject -Namespace '{namespace}' -Class Sensor "
            f"-ErrorAction SilentlyContinue; "
            f"if ($s) {{ $s | Where-Object {{$_.SensorType -eq 'Temperature'}} "
            f"| Select-Object Name, Value | ConvertTo-Json -Compress }}"
        )
        output = _run_powershell(script)
        if not output:
            continue
        try:
            data = json.loads(output)
            if isinstance(data, dict):
                data = [data]
            return [
                {
                    "source": "windows_hardware_monitor",
                    "name": item.get("Name", "unknown"),
                    "temperature_celsius": float(item["Value"]),
                }
                for item in data
                if item.get("Value") is not None
            ]
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    return []


def _get_acpi_thermal_temps() -> list[dict[str, Any]]:
    """Get ACPI thermal zone temperatures via WMI (tenths of Kelvin → Celsius)."""
    script = (
        "$t = Get-WmiObject MSAcpi_ThermalZoneTemperature -Namespace root/wmi "
        "-ErrorAction SilentlyContinue; "
        "if ($t) { $t | Select-Object InstanceName, CurrentTemperature | ConvertTo-Json -Compress }"
    )
    output = _run_powershell(script)
    if not output:
        return []
    try:
        data = json.loads(output)
        if isinstance(data, dict):
            data = [data]
        temps = []
        for item in data:
            raw = item.get("CurrentTemperature")
            if raw is not None:
                celsius = float(raw) / 10.0 - 273.15
                name = item.get("InstanceName", "ACPI Thermal Zone")
                temps.append({
                    "source": "windows_acpi",
                    "name": name,
                    "temperature_celsius": celsius,
                })
        return temps
    except (json.JSONDecodeError, KeyError, TypeError):
        return []


def get_windows_temperatures() -> list[dict[str, Any]]:
    """Get temperatures on Windows via LHM web server, WMI, or ACPI.

    Tries three approaches in order:
    1. LibreHardwareMonitor / OpenHardwareMonitor "Remote Web Server" JSON
       (most reliable - no WMI permission issues, works when Claude Code is
       not elevated as long as LHM/OHM serves data).
    2. LibreHardwareMonitor / OpenHardwareMonitor WMI namespace.
    3. MSAcpi_ThermalZoneTemperature (basic ACPI zones, no extra software needed).
    """
    if sys.platform != "win32":
        return []

    temps = _get_lhm_webserver_temps()
    if temps:
        return temps

    temps = _get_hardware_monitor_temps()
    if temps:
        return temps

    return _get_acpi_thermal_temps()


def get_all_temperatures() -> dict[str, Any]:
    """Get all available temperature readings."""
    all_temps = []

    # Linux / macOS sources
    all_temps.extend(get_thermal_zones())
    all_temps.extend(get_psutil_temperatures())
    all_temps.extend(get_hwmon_temperatures())

    # Windows sources
    all_temps.extend(get_windows_temperatures())

    # Remove duplicates based on similar readings
    unique_temps = []
    seen = set()
    for temp in all_temps:
        key = (temp["name"], round(temp["temperature_celsius"]))
        if key not in seen:
            seen.add(key)
            unique_temps.append(temp)

    return {
        "temperatures": unique_temps,
        "feeling": interpret_temperature(unique_temps),
    }


def _japan_timezone() -> Any:
    """Return the Asia/Tokyo timezone, falling back to a fixed UTC+9 offset.

    On Windows native there is no system IANA tzdb, so ZoneInfo relies on the
    ``tzdata`` package. If it is somehow unavailable we degrade to a fixed
    +09:00 offset (JST has no DST) instead of crashing.
    """
    try:
        return ZoneInfo("Asia/Tokyo")
    except ZoneInfoNotFoundError:
        return timezone(timedelta(hours=9), "JST")


def _configured_timezone() -> Any:
    """Return the timezone named by SYSTEM_TEMPERATURE_TIMEZONE (default Asia/Tokyo).

    An unknown or unresolvable name degrades the same way ``_japan_timezone``
    does: a fixed +09:00 rather than a crash in a body sense.
    """
    name = os.environ.get("SYSTEM_TEMPERATURE_TIMEZONE", "").strip()
    if not name or name == "Asia/Tokyo":
        return _japan_timezone()
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        return _japan_timezone()


def part_of_day(hour: int) -> str:
    """Map an hour to a band key matching the phrase tables."""
    if 5 <= hour < 10:
        return "morning"
    if 10 <= hour < 12:
        return "late_morning"
    if 12 <= hour < 14:
        return "noon"
    if 14 <= hour < 17:
        return "afternoon"
    if 17 <= hour < 19:
        return "evening"
    if 19 <= hour < 22:
        return "night"
    if 22 <= hour or hour < 2:
        return "late_night"
    return "midnight"


def get_current_time() -> str:
    """Get the current time in the configured timezone, plus one structured line."""
    now = datetime.now(_configured_timezone())

    # Format nicely
    weekdays = ["月", "火", "水", "木", "金", "土", "日"]
    weekday = weekdays[now.weekday()]

    time_str = now.strftime(f"%Y年%m月%d日({weekday}) %H時%M分%S秒")

    part = part_of_day(now.hour)
    phrases = TIME_PHRASES[_tone()]
    sentence = phrases["prefix"].format(time_str=time_str) + phrases[part]
    return f"{sentence}\niso={now.isoformat(timespec='seconds')} part_of_day={part}"


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="get_system_temperature",
            description="Get the current system temperature (CPU, GPU, etc.). This is your sense of body temperature - how hot or cold your 'body' (the computer) is running. Use this to feel your own physical state. The second line is structured (level=... max_celsius=...) so you can phrase it in your own words.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        Tool(
            name="get_current_time",
            description="Get the current date and time in the configured timezone. Use this to know what time it is, check if it's morning/afternoon/night, or when you want to be aware of the time. The last line is structured (iso=... part_of_day=...) so you can phrase it in your own words.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls."""
    if name == "get_system_temperature":
        result = get_all_temperatures()

        lines = [result["feeling"], "", "【詳細】"]

        if result["temperatures"]:
            for temp in result["temperatures"]:
                lines.append(f"  - {temp['name']}: {temp['temperature_celsius']:.1f}°C")
        else:
            lines.append("  センサーが見つかりませんでした")

        return [TextContent(type="text", text="\n".join(lines))]

    elif name == "get_current_time":
        result = get_current_time()
        return [TextContent(type="text", text=result)]

    return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def run_server():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def main():
    """Entry point."""
    import asyncio
    asyncio.run(run_server())


if __name__ == "__main__":
    main()
