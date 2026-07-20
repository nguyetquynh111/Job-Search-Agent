"""Full workflow tests."""

from __future__ import annotations

from pathlib import Path

from src.agent.graph import build_agent_graph, create_memory_checkpointer, invoke_new_run, resume_run
from src.agent.state import create_initial_state
from src.observability.trace_manager import TraceManager
from src.tools.registry import load_tool_registry


def test_full_workflow_completes_end_to_end(tmp_path: Path, monkeypatch) -> None:
    """Tool implementations support the full run, including rejection, memory, revision, and cover letters."""

    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "outputs"))
    memory_file = tmp_path / "memory.json"
    memory_file.write_text("[]", encoding="utf-8")
    tracer = TraceManager(enabled=False)
    app = build_agent_graph(
        tools=load_tool_registry(),
        checkpointer=create_memory_checkpointer(),
        tracer=tracer,
    )
    state = create_initial_state(
        thread_id="thread-e2e",
        run_id="run-e2e",
        memory_file=str(memory_file),
    )

    first = invoke_new_run(app, state)
    first_payload = first["__interrupt__"][0].value
    rejected_job_id = list(first_payload["resumes"])[1]
    first_feedback = {
        job_id: {"decision": "approve", "comment": ""}
        for job_id in first_payload["resumes"]
    }
    first_feedback[rejected_job_id] = {
        "decision": "reject",
        "comment": "Add GraphQL. I have used it in previous projects.",
    }

    second = resume_run(app, "thread-e2e", first_feedback)
    assert second["status"] == "WAITING_FOR_REVIEW"
    assert second["revision_round"] == 1
    assert any(fact["canonical_value"] == "GraphQL" for fact in second["memory_facts"])
    revision_decisions = [
        item
        for item in second["agent_decisions"]
        if item["selected_tool"] == "tailor_resume"
        and item["arguments"]["job"]["job_id"] == rejected_job_id
        and item["arguments"].get("revision_feedback")
    ]
    assert revision_decisions
    assert any(
        evidence["evidence_id"].startswith("mem-")
        for evidence in revision_decisions[-1]["arguments"]["candidate_evidence"]
    )

    second_payload = second["__interrupt__"][0].value
    second_feedback = {
        job_id: {"decision": "approve", "comment": ""}
        for job_id in second_payload["resumes"]
    }
    final = resume_run(app, "thread-e2e", second_feedback)

    assert final["status"] == "COMPLETED"
    assert final["phase"] == "COMPLETE"
    assert len(final["cover_letter_results"]) == 3
    assert not final["errors"]
    assert final["trace_id"] == "trace-run-e2e"
    assert [event.name for event in tracer.events].count("job_search_agent_run") == 1
