"""Workflow integration test using real LaTeX and PDF extraction."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from pypdf import PdfReader

from src.agent.graph import (
    build_agent_graph,
    create_sqlite_checkpointer,
    invoke_new_run,
    resume_run,
)
from src.agent.state import create_initial_state
from src.observability.trace_manager import TraceManager
from src.tools.registry import load_tool_registry


@pytest.mark.skipif(
    shutil.which("pdflatex") is None,
    reason="pdflatex is not installed; real PDF generation cannot be exercised",
)
def test_real_pdflatex_complete_workflow_with_review_memory_and_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run all real business tools; only the controller uses the explicit test policy."""

    output_dir = tmp_path / "outputs"
    memory_file = output_dir / "memory.json"
    monkeypatch.setenv("OUTPUT_DIR", str(output_dir))
    tracer = TraceManager(enabled=False)
    checkpointer, context = create_sqlite_checkpointer(output_dir / "checkpoints.sqlite")
    try:
        app = build_agent_graph(
            tools=load_tool_registry(),
            checkpointer=checkpointer,
            tracer=tracer,
        )
        state = create_initial_state(
            thread_id="thread-real-pdflatex",
            run_id="run-real-pdflatex",
            memory_file=str(memory_file),
        )

        waiting = invoke_new_run(app, state)
        payload = waiting["__interrupt__"][0].value
        feedback = {
            job_id: {"decision": "approve", "comment": ""}
            for job_id in payload["resumes"]
        }
        feedback["J028"] = {
            "decision": "reject",
            "comment": "Add LangGraph. I have used it in previous projects.",
        }
        final = resume_run(app, state["thread_id"], feedback)
    finally:
        context.__exit__(None, None, None)

    assert final["status"] == "COMPLETED"
    assert final["revision_round"] == 1
    assert len(final["review_history"]) == 1
    assert [event.name for event in tracer.events].count("human_review_pause") == 1
    assert any(
        fact["canonical_value"] == "LangGraph" for fact in final["memory_facts"]
    )
    for job_id in final["top_3_job_ids"]:
        resume = final["tailoring_results"][job_id]
        letter = final["cover_letter_results"][job_id]
        assert len(PdfReader(resume["output_pdf_path"]).pages) == 1
        assert len(PdfReader(letter["output_pdf_path"]).pages) == 1
        assert Path(resume["output_tex_path"]).is_file()
        assert Path(letter["output_tex_path"]).is_file()
