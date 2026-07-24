from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from memory_mcp.config import MemoryConfig
from memory_mcp.embedding import E5EmbeddingFunction
from memory_mcp.server import MemoryMCPServer
from memory_mcp.store import MemoryStore


def test_embedding_warmup_loads_model_once(monkeypatch: pytest.MonkeyPatch) -> None:
    constructor_calls: list[str] = []

    class FakeSentenceTransformer:
        def __init__(self, model_name: str) -> None:
            constructor_calls.append(model_name)

    fake_module = ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = FakeSentenceTransformer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    embedding = E5EmbeddingFunction("fixture/model")

    embedding.warmup()
    embedding.warmup()

    assert constructor_calls == ["fixture/model"]


def test_store_warmup_delegates_synchronously(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = MemoryStore(
        MemoryConfig(db_path=":memory:", collection_name="warmup-test")
    )
    calls: list[str] = []
    monkeypatch.setattr(
        store._embedding_fn,
        "warmup",
        lambda: calls.append("warmup"),
        raising=False,
    )

    result = store.warmup()

    assert result is None
    assert calls == ["warmup"]


@pytest.mark.asyncio
async def test_server_warms_on_main_path_before_async_connect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class FakeStore:
        def __init__(self, _config: object) -> None:
            events.append("construct")

        def warmup(self) -> None:
            events.append("warmup")

        async def connect(self) -> None:
            events.append("connect")

    monkeypatch.setattr(
        "memory_mcp.server.MemoryConfig.from_env",
        lambda: SimpleNamespace(db_path=":memory:"),
    )
    monkeypatch.setattr("memory_mcp.server.MemoryStore", FakeStore)
    monkeypatch.setattr(
        "memory_mcp.server.EpisodeManager",
        lambda _store: events.append("episode"),
    )
    monkeypatch.setattr(
        "memory_mcp.server.SensoryIntegration",
        lambda _store: events.append("sensory"),
    )
    monkeypatch.setattr(
        "memory_mcp.server.MetacognitionTracker",
        lambda _path: events.append("metacognition"),
    )
    server = MemoryMCPServer()

    await server.connect_memory()

    assert events[:3] == ["construct", "warmup", "connect"]
