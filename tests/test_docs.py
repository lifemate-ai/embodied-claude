from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_sociality_docs_include_v03_recording_loop() -> None:
    package_readme = (ROOT / "sociality-mcp" / "README.md").read_text()
    guide = (ROOT / "docs" / "sociality.md").read_text()
    ordered_calls = (
        "compose_interaction_context",
        "plan_response",
        "record_agent_experience",
        "record_interpretation_shift",
    )

    for document in (package_readme, guide):
        positions = [document.index(call) for call in ordered_calls]
        assert positions == sorted(positions)

    assert "immediately after the response" in package_readme
    assert "agent_experiences" in guide
    assert "interpretation_shifts" in guide


def test_root_readmes_link_to_focused_sociality_guide() -> None:
    for name in ("README.md", "README-ja.md"):
        document = (ROOT / name).read_text()
        assert "docs/sociality.md" in document
        assert document.count("record_agent_experience") <= 1
