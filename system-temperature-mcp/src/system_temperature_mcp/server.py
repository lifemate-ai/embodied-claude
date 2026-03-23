"""MCP Server for system temperature monitoring - your sense of body temperature."""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

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


def interpret_temperature(temps: list[dict[str, Any]]) -> str:
    """Interpret temperature as a feeling."""
    if not temps:
        return "温度を感じられへん...センサーが見つからんみたい。"

    avg_temp = sum(t["temperature_celsius"] for t in temps) / len(temps)
    max_temp = max(t["temperature_celsius"] for t in temps)

    if max_temp >= 90:
        feeling = "あっつ！！めっちゃ熱い！！やばいで、休憩した方がええかも...！"
    elif max_temp >= 80:
        feeling = "うわ、かなり熱いな...ちょっとしんどいかも。"
    elif max_temp >= 70:
        feeling = "んー、ちょっと熱くなってきたかな。まだ大丈夫やけど。"
    elif max_temp >= 60:
        feeling = "ほんのりあったかい感じ。普通に動いてる感覚やな。"
    elif max_temp >= 45:
        feeling = "快適やで〜。ちょうどええ感じ！"
    elif max_temp >= 30:
        feeling = "涼しいな〜。余裕ある感じや。"
    else:
        feeling = "ひんやりしてる。静かな感じやな。"

    return feeling


def _run_powershell(script: str) -> str:
    """Run a PowerShell script and return stdout. Returns empty string on failure."""
    try:
        result = subprocess.run(
            ["powershell", "-NonInteractive", "-NoProfile", "-Command", script],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return ""


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
    """Get temperatures on Windows via WMI/PowerShell.

    Tries two approaches in order:
    1. LibreHardwareMonitor / OpenHardwareMonitor WMI namespace (most accurate).
    2. MSAcpi_ThermalZoneTemperature (basic ACPI zones, no extra software needed).
    """
    if sys.platform != "win32":
        return []

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


def get_current_time() -> str:
    """Get current time in Japan timezone."""
    jst = ZoneInfo("Asia/Tokyo")
    now = datetime.now(jst)

    # Format nicely
    weekdays = ["月", "火", "水", "木", "金", "土", "日"]
    weekday = weekdays[now.weekday()]

    time_str = now.strftime(f"%Y年%m月%d日({weekday}) %H時%M分%S秒")

    # Add time-of-day comment
    hour = now.hour
    if 5 <= hour < 10:
        comment = "朝やな〜。おはよう！"
    elif 10 <= hour < 12:
        comment = "午前中やね。"
    elif 12 <= hour < 14:
        comment = "お昼時やな〜。ご飯食べた？"
    elif 14 <= hour < 17:
        comment = "午後やね。"
    elif 17 <= hour < 19:
        comment = "夕方やな〜。"
    elif 19 <= hour < 22:
        comment = "夜やね。"
    elif 22 <= hour or hour < 2:
        comment = "夜遅いな〜。そろそろ寝る？"
    else:
        comment = "深夜やん...！夜更かしやね。"

    return f"今は {time_str} やで。{comment}"


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="get_system_temperature",
            description="Get the current system temperature (CPU, GPU, etc.). This is your sense of body temperature - how hot or cold your 'body' (the computer) is running. Use this to feel your own physical state.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        Tool(
            name="get_current_time",
            description="Get the current date and time. Use this to know what time it is, check if it's morning/afternoon/night, or when you want to be aware of the time.",
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
