"""Graph-specific behavior tests."""

from __future__ import annotations

from pathlib import Path

from src.agent.graph import (
    build_agent_graph,
    create_memory_checkpointer,
    create_sqlite_checkpointer,
    invoke_new_run,
    resume_run,
)
from src.agent.state import create_initial_state
from src.observability.trace_manager import TraceManager
from src.tools.registry import load_tool_registry


def test_sqlite_checkpointer_creates_missing_parent_and_database(
    tmp_path: Path,
) -> None:
    """The runtime checkpoint path is created on first initialization."""

    db_path = tmp_path / "missing" / "checkpoints.sqlite"

    _, context = create_sqlite_checkpointer(db_path)
    try:
        assert db_path.is_file()
    finally:
        context.__exit__(None, None, None)


def test_failed_review_after_two_revision_rounds(tmp_path: Path, monkeypatch) -> None:
    """The graph fails review and does not generate cover letters after max revisions."""

    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "outputs"))
    memory_file = tmp_path / "memory.json"
    memory_file.write_text("[]", encoding="utf-8")
    app = build_agent_graph(
        tools=load_tool_registry(),
        checkpointer=create_memory_checkpointer(),
        tracer=TraceManager(enabled=False),
    )
    state = create_initial_state(
        thread_id="thread-fail-review",
        run_id="run-fail-review",
        memory_file=str(memory_file),
    )
    result = invoke_new_run(app, state)

    for _ in range(3):
        payload = result["__interrupt__"][0].value
        feedback = {
            job_id: {"decision": "reject", "comment": "I have used GraphQL."}
            for job_id in payload["resumes"]
        }
        result = resume_run(app, "thread-fail-review", feedback)
        if result.get("status") == "FAILED_REVIEW":
            break

    assert result["status"] == "FAILED_REVIEW"
    assert result["cover_letter_results"] == {}
