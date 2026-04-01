"""Tests for types module."""

from datetime import datetime

from human_mcp.types import (
    HumanRequest,
    HumanResponse,
    ResponseFormat,
    TaskType,
    Urgency,
)


def test_task_type_values():
    assert TaskType.OBSERVATION.value == "observation"
    assert TaskType.JUDGMENT.value == "judgment"
    assert TaskType.APPROVAL.value == "approval"
    assert TaskType.PHYSICAL_ACTION.value == "physical_action"


def test_response_format_values():
    assert ResponseFormat.CHOICE.value == "choice"
    assert ResponseFormat.YES_NO.value == "yes_no"
    assert ResponseFormat.SHORT_TEXT.value == "short_text"
    assert ResponseFormat.NUMBER.value == "number"
    assert ResponseFormat.PHOTO.value == "photo"


def test_urgency_values():
    assert Urgency.LOW.value == "low"
    assert Urgency.NORMAL.value == "normal"
    assert Urgency.HIGH.value == "high"


def test_human_request_defaults():
    req = HumanRequest(
        task_type=TaskType.OBSERVATION,
        question="Is the door locked?",
        reason="Cannot verify remotely",
    )
    assert req.expected_format == ResponseFormat.SHORT_TEXT
    assert req.options == []
    assert req.urgency == Urgency.NORMAL
    assert req.batch_id is None
    assert isinstance(req.created_at, datetime)


def test_human_request_with_options():
    req = HumanRequest(
        task_type=TaskType.JUDGMENT,
        question="Which color?",
        reason="Subjective preference",
        expected_format=ResponseFormat.CHOICE,
        options=["Red", "Blue", "Green"],
        urgency=Urgency.HIGH,
    )
    assert req.options == ["Red", "Blue", "Green"]
    assert req.urgency == Urgency.HIGH


def test_human_response_to_dict_choice():
    resp = HumanResponse(choice="Red", response_time_seconds=2.5)
    d = resp.to_dict()
    assert d["choice"] == "Red"
    assert d["response_time_seconds"] == 2.5
    assert "yes_no" not in d
    assert "short_text" not in d
    assert "number" not in d
    assert "photo_path" not in d


def test_human_response_to_dict_yes_no():
    resp = HumanResponse(yes_no=True, response_time_seconds=1.0)
    d = resp.to_dict()
    assert d["yes_no"] is True
    assert "choice" not in d


def test_human_response_to_dict_short_text():
    resp = HumanResponse(short_text="Looks fine", response_time_seconds=3.2)
    d = resp.to_dict()
    assert d["short_text"] == "Looks fine"


def test_human_response_to_dict_number():
    resp = HumanResponse(number=42.0, response_time_seconds=1.5)
    d = resp.to_dict()
    assert d["number"] == 42.0


def test_human_response_to_dict_photo():
    resp = HumanResponse(photo_path="/tmp/photo.jpg", response_time_seconds=10.0)
    d = resp.to_dict()
    assert d["photo_path"] == "/tmp/photo.jpg"


def test_human_response_to_dict_has_responded_at():
    resp = HumanResponse(short_text="test")
    d = resp.to_dict()
    assert "responded_at" in d
