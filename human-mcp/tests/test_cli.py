"""Tests for CLI prompt module."""

from unittest.mock import patch

from human_mcp.cli import prompt_human
from human_mcp.types import HumanRequest, ResponseFormat, TaskType, Urgency


def _make_request(**kwargs):
    defaults = {
        "task_type": TaskType.OBSERVATION,
        "question": "Test question?",
        "reason": "Testing",
        "urgency": Urgency.NORMAL,
    }
    defaults.update(kwargs)
    return HumanRequest(**defaults)


@patch("builtins.input", return_value="hello world")
def test_prompt_short_text(mock_input):
    req = _make_request(expected_format=ResponseFormat.SHORT_TEXT)
    resp = prompt_human(req)
    assert resp.short_text == "hello world"
    assert resp.response_time_seconds >= 0


@patch("builtins.input", return_value="y")
def test_prompt_yes_no_yes(mock_input):
    req = _make_request(expected_format=ResponseFormat.YES_NO)
    resp = prompt_human(req)
    assert resp.yes_no is True


@patch("builtins.input", return_value="n")
def test_prompt_yes_no_no(mock_input):
    req = _make_request(expected_format=ResponseFormat.YES_NO)
    resp = prompt_human(req)
    assert resp.yes_no is False


@patch("builtins.input", return_value="はい")
def test_prompt_yes_no_japanese_yes(mock_input):
    req = _make_request(expected_format=ResponseFormat.YES_NO)
    resp = prompt_human(req)
    assert resp.yes_no is True


@patch("builtins.input", return_value="いいえ")
def test_prompt_yes_no_japanese_no(mock_input):
    req = _make_request(expected_format=ResponseFormat.YES_NO)
    resp = prompt_human(req)
    assert resp.yes_no is False


@patch("builtins.input", return_value="2")
def test_prompt_choice(mock_input):
    req = _make_request(
        expected_format=ResponseFormat.CHOICE,
        options=["Apple", "Banana", "Cherry"],
    )
    resp = prompt_human(req)
    assert resp.choice == "Banana"


@patch("builtins.input", return_value="42.5")
def test_prompt_number(mock_input):
    req = _make_request(expected_format=ResponseFormat.NUMBER)
    resp = prompt_human(req)
    assert resp.number == 42.5


@patch("builtins.input", return_value="/tmp/photo.jpg")
def test_prompt_photo(mock_input):
    req = _make_request(expected_format=ResponseFormat.PHOTO)
    resp = prompt_human(req)
    assert resp.photo_path == "/tmp/photo.jpg"


@patch("builtins.input", return_value="1")
def test_prompt_choice_first_option(mock_input):
    req = _make_request(
        expected_format=ResponseFormat.CHOICE,
        options=["Yes", "No"],
    )
    resp = prompt_human(req)
    assert resp.choice == "Yes"


@patch("builtins.input", return_value="3")
def test_prompt_choice_last_option(mock_input):
    req = _make_request(
        expected_format=ResponseFormat.CHOICE,
        options=["A", "B", "C"],
    )
    resp = prompt_human(req)
    assert resp.choice == "C"
