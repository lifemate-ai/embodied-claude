from __future__ import annotations

import io
import json
from datetime import timedelta

from system_temperature_mcp import server


def test_japan_timezone_falls_back_to_fixed_jst(monkeypatch) -> None:
    def missing_zoneinfo(_key: str):
        raise server.ZoneInfoNotFoundError

    monkeypatch.setattr(server, "ZoneInfo", missing_zoneinfo)

    timezone = server._japan_timezone()

    assert timezone.utcoffset(None) == timedelta(hours=9)
    assert timezone.tzname(None) == "JST"


def test_lhm_webserver_extracts_temperatures(monkeypatch) -> None:
    payload = {
        "Text": "root",
        "Children": [
            {"Text": "CPU Package", "Value": "52.5 °C", "Children": []},
            {"Text": "SSD", "Value": "41,0 °C", "Children": []},
            {"Text": "Distance to TjMax", "Value": "47.5 °C", "Children": []},
        ],
    }
    response = io.BytesIO(json.dumps(payload).encode())
    monkeypatch.setattr(server.urllib.request, "urlopen", lambda *_args, **_kwargs: response)

    temperatures = server._get_lhm_webserver_temps()

    assert temperatures == [
        {
            "source": "lhm_webserver",
            "name": "CPU Package",
            "temperature_celsius": 52.5,
        },
        {
            "source": "lhm_webserver",
            "name": "SSD",
            "temperature_celsius": 41.0,
        },
    ]


def _lhm_payload_with_configuration_values() -> dict:
    """A trimmed LHM tree captured from a real machine.

    Source: Intel Core Ultra 7 155H / SK Hynix DDR5 / KIOXIA NVMe, LHM 0.9.6.
    Alongside real readings, LHM reports hardware *configuration* values in °C:
    DDR5 SPD exposes thermal limits and the sensor's own resolution, and NVMe
    exposes SMART warning/critical thresholds. Older hardware does not report
    these, so this only shows up on newer machines.
    """
    return {
        "Text": "Sensor",
        "Children": [
            {
                "Text": "Intel Core Ultra 7 155H",
                "Children": [
                    {
                        "Text": "Temperatures",
                        "Children": [
                            {"Text": "CPU Package", "Value": "47.0 °C", "Children": []},
                            {"Text": "P-Core #1", "Value": "44.0 °C", "Children": []},
                            {
                                "Text": "P-Core #1 Distance to TjMax",
                                "Value": "53.0 °C",
                                "Children": [],
                            },
                        ],
                    }
                ],
            },
            {
                "Text": "SK Hynix - HMCG78AGBSA092N (#0)",
                "Children": [
                    {
                        "Text": "Temperatures",
                        "Children": [
                            {"Text": "DIMM #0", "Value": "39.0 °C", "Children": []},
                            {
                                "Text": "Temperature Sensor Resolution",
                                "Value": "0.3 °C",
                                "Children": [],
                            },
                            {
                                "Text": "Thermal Sensor Low Limit",
                                "Value": "0.0 °C",
                                "Children": [],
                            },
                            {
                                "Text": "Thermal Sensor High Limit",
                                "Value": "55.0 °C",
                                "Children": [],
                            },
                            {
                                "Text": "Thermal Sensor Critical Low Limit",
                                "Value": "0.0 °C",
                                "Children": [],
                            },
                            {
                                "Text": "Thermal Sensor Critical High Limit",
                                "Value": "85.0 °C",
                                "Children": [],
                            },
                        ],
                    }
                ],
            },
            {
                "Text": "KBG6AZNV512G LA KIOXIA",
                "Children": [
                    {
                        "Text": "Temperatures",
                        "Children": [
                            {
                                "Text": "Composite Temperature",
                                "Value": "35.0 °C",
                                "Children": [],
                            },
                            {
                                "Text": "Warning Temperature",
                                "Value": "82.0 °C",
                                "Children": [],
                            },
                            {
                                "Text": "Critical Temperature",
                                "Value": "84.0 °C",
                                "Children": [],
                            },
                        ],
                    }
                ],
            },
        ],
    }


def test_lhm_webserver_skips_configuration_values(monkeypatch) -> None:
    """Only live readings are collected; limits and specs are not temperatures."""
    response = io.BytesIO(json.dumps(_lhm_payload_with_configuration_values()).encode())
    monkeypatch.setattr(server.urllib.request, "urlopen", lambda *_args, **_kwargs: response)

    temperatures = server._get_lhm_webserver_temps()

    assert [t["name"] for t in temperatures] == [
        "CPU Package",
        "P-Core #1",
        "DIMM #0",
        "Composite Temperature",
    ]
    # The DIMM's 0.0 °C low limit would otherwise become the minimum, and its
    # 85.0 °C critical limit the maximum.
    values = [t["temperature_celsius"] for t in temperatures]
    assert (min(values), max(values)) == (35.0, 47.0)


def test_configuration_values_do_not_change_the_reported_feeling(monkeypatch) -> None:
    """Regression: a DDR5 critical limit must not be felt as body heat.

    Before configuration values were filtered out, a DIMM's 85.0 °C critical
    limit became max_temp, so the server reported severe heat while the CPU was
    idling in the 40s. Asserting against the readings-only verdict keeps this
    independent of how the feelings themselves are worded.
    """
    response = io.BytesIO(json.dumps(_lhm_payload_with_configuration_values()).encode())
    monkeypatch.setattr(server.urllib.request, "urlopen", lambda *_args, **_kwargs: response)

    from_lhm = server.interpret_temperature(server._get_lhm_webserver_temps())
    readings_only = server.interpret_temperature([
        {"source": "lhm_webserver", "name": "CPU Package", "temperature_celsius": 47.0},
        {"source": "lhm_webserver", "name": "P-Core #1", "temperature_celsius": 44.0},
        {"source": "lhm_webserver", "name": "DIMM #0", "temperature_celsius": 39.0},
        {
            "source": "lhm_webserver",
            "name": "Composite Temperature",
            "temperature_celsius": 35.0,
        },
    ])

    assert from_lhm == readings_only


def test_windows_temperature_prefers_lhm_webserver(monkeypatch) -> None:
    expected = [
        {
            "source": "lhm_webserver",
            "name": "CPU Package",
            "temperature_celsius": 52.5,
        }
    ]
    monkeypatch.setattr(server.sys, "platform", "win32")
    monkeypatch.setattr(server, "_get_lhm_webserver_temps", lambda: expected)
    monkeypatch.setattr(
        server,
        "_get_hardware_monitor_temps",
        lambda: (_ for _ in ()).throw(AssertionError("WMI fallback should not run")),
    )

    assert server.get_windows_temperatures() == expected


# ---------------------------------------------------------------------------
# Tone and timezone (#135)
# ---------------------------------------------------------------------------


def _reading(celsius: float) -> list[dict]:
    return [{"source": "test", "name": "CPU", "temperature_celsius": celsius}]


def test_default_tone_is_the_original_kansai_phrase(monkeypatch) -> None:
    monkeypatch.delenv("SYSTEM_TEMPERATURE_TONE", raising=False)

    text = server.interpret_temperature(_reading(52.0))

    assert text.splitlines()[0] == "快適やで〜。ちょうどええ感じ！"


def test_structured_line_is_appended_regardless_of_tone(monkeypatch) -> None:
    monkeypatch.delenv("SYSTEM_TEMPERATURE_TONE", raising=False)
    assert server.interpret_temperature(_reading(52.0)).splitlines()[1] == (
        "level=comfortable max_celsius=52.0"
    )
    assert server.interpret_temperature([]).splitlines()[1] == "level=unknown"

    monkeypatch.setenv("SYSTEM_TEMPERATURE_TONE", "neutral")
    assert server.interpret_temperature(_reading(85.5)).splitlines()[1] == (
        "level=very_hot max_celsius=85.5"
    )


def test_neutral_tone_has_no_dialect(monkeypatch) -> None:
    monkeypatch.setenv("SYSTEM_TEMPERATURE_TONE", "neutral")

    for celsius in (20.0, 35.0, 52.0, 65.0, 75.0, 85.0, 95.0):
        phrase = server.interpret_temperature(_reading(celsius)).splitlines()[0]
        assert "やで" not in phrase and "やな" not in phrase and "へん" not in phrase
    assert server.interpret_temperature(_reading(52.0)).splitlines()[0] == (
        "快適です。ちょうどよい状態です。"
    )
    phrase = server.interpret_temperature([]).splitlines()[0]
    assert "へん" not in phrase


def test_unknown_tone_falls_back_to_neutral(monkeypatch) -> None:
    monkeypatch.setenv("SYSTEM_TEMPERATURE_TONE", "klingon")

    assert server._tone() == "neutral"
    assert "やで" not in server.interpret_temperature(_reading(52.0))


def test_temperature_level_bands_match_thresholds() -> None:
    assert server.temperature_level(None) == "unknown"
    assert server.temperature_level(29.9) == "cold"
    assert server.temperature_level(30.0) == "cool"
    assert server.temperature_level(45.0) == "comfortable"
    assert server.temperature_level(60.0) == "warm"
    assert server.temperature_level(70.0) == "hot"
    assert server.temperature_level(80.0) == "very_hot"
    assert server.temperature_level(90.0) == "critical"


def test_part_of_day_matches_existing_bands() -> None:
    assert server.part_of_day(7) == "morning"
    assert server.part_of_day(11) == "late_morning"
    assert server.part_of_day(13) == "noon"
    assert server.part_of_day(15) == "afternoon"
    assert server.part_of_day(18) == "evening"
    assert server.part_of_day(20) == "night"
    assert server.part_of_day(23) == "late_night"
    assert server.part_of_day(1) == "late_night"
    assert server.part_of_day(3) == "midnight"


def _freeze_now(monkeypatch, utc_tuple) -> None:
    from datetime import UTC
    from datetime import datetime as real_datetime

    fixed = real_datetime(*utc_tuple, tzinfo=UTC)

    class _FrozenDatetime(real_datetime):
        @classmethod
        def now(cls, tzinfo=None):
            return fixed.astimezone(tzinfo) if tzinfo else fixed

    monkeypatch.setattr(server, "datetime", _FrozenDatetime)


def test_current_time_default_tone_and_timezone(monkeypatch) -> None:
    monkeypatch.delenv("SYSTEM_TEMPERATURE_TONE", raising=False)
    monkeypatch.delenv("SYSTEM_TEMPERATURE_TIMEZONE", raising=False)
    _freeze_now(monkeypatch, (2026, 8, 14, 22, 30, 0))  # 07:30 JST on the 15th

    sentence, structured = server.get_current_time().splitlines()

    assert sentence == "今は 2026年08月15日(土) 07時30分00秒 やで。朝やな〜。おはよう！"
    assert structured == "iso=2026-08-15T07:30:00+09:00 part_of_day=morning"


def test_current_time_neutral_tone(monkeypatch) -> None:
    monkeypatch.setenv("SYSTEM_TEMPERATURE_TONE", "neutral")
    monkeypatch.delenv("SYSTEM_TEMPERATURE_TIMEZONE", raising=False)
    _freeze_now(monkeypatch, (2026, 8, 14, 22, 30, 0))

    sentence, structured = server.get_current_time().splitlines()

    assert sentence == "今は 2026年08月15日(土) 07時30分00秒 です。朝です。おはようございます。"
    assert "やで" not in sentence
    assert structured.endswith("part_of_day=morning")


def test_current_time_respects_timezone_env(monkeypatch) -> None:
    monkeypatch.delenv("SYSTEM_TEMPERATURE_TONE", raising=False)
    monkeypatch.setenv("SYSTEM_TEMPERATURE_TIMEZONE", "America/New_York")
    _freeze_now(monkeypatch, (2026, 8, 14, 22, 30, 0))  # 18:30 EDT on the 14th

    sentence, structured = server.get_current_time().splitlines()

    assert "2026年08月14日(金) 18時30分00秒" in sentence
    assert structured == "iso=2026-08-14T18:30:00-04:00 part_of_day=evening"


def test_unknown_timezone_env_falls_back_to_jst(monkeypatch) -> None:
    monkeypatch.setenv("SYSTEM_TEMPERATURE_TIMEZONE", "Not/AZone")
    _freeze_now(monkeypatch, (2026, 8, 14, 22, 30, 0))

    assert "iso=2026-08-15T07:30:00+09:00" in server.get_current_time()
