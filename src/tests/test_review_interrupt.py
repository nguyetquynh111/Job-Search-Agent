"""LangGraph human interrupt tests."""

from __future__ import annotations

from pathlib import Path

from src.agent.graph import build_agent_graph, create_memory_checkpointer, invoke_new_run
from src.agent.state import create_initial_state
from src.observability.trace_manager import TraceManager
from src.tools.registry import load_tool_registry


def test_one_interrupt_payload_contains_all_three_resumes(tmp_path: Path, monkeypatch) -> None:
    """The review interrupt contains one payload with all selected resumes."""

    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "outputs"))
    memory_file = tmp_path / "memory.json"
    memory_file.write_text("[]", encoding="utf-8")
    app = build_agent_graph(
        tools=load_tool_registry(),
        checkpointer=create_memory_checkpointer(),
        tracer=TraceManager(enabled=False),
    )
    state = create_initial_state(
        thread_id="thread-review",
        run_id="run-review",
        memory_file=str(memory_file),
    )
    result = invoke_new_run(app, state)

    interrupts = result.get("__interrupt__", [])
    assert len(interrupts) == 1
    payload = interrupts[0].value
    assert payload["review_round"] == 1
    assert payload["max_revision_rounds"] == 2
    assert len(payload["resumes"]) == 3
    assert set(payload["resumes"]) == set(result["top_3_job_ids"])
    assert result["status"] == "WAITING_FOR_REVIEW"
    assert all(job_id in result["tailoring_results"] for job_id in result["top_3_job_ids"])
