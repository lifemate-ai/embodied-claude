"""post_tweet must return a URL that resolves for whichever account posted."""

from __future__ import annotations

from types import SimpleNamespace

from x_mcp import server


def test_posted_url_is_account_independent(monkeypatch) -> None:
    captured: dict = {}

    class _Client:
        def create_tweet(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(data={"id": "1234567890"})

    monkeypatch.setattr(server, "_tweepy_client", lambda: _Client())

    result = server.post_tweet("hello")

    assert captured == {"text": "hello"}
    assert result == "Posted! https://x.com/i/status/1234567890"
    assert "xai_kokone" not in result


def test_tool_description_names_no_account() -> None:
    assert "xai_kokone" not in (server.post_tweet.__doc__ or "")
