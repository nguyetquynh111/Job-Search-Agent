"""Graph-specific behavior tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from pypdf import PdfWriter

from src.agent.graph import (
    ToolExecutionError,
    _apply_tool_output,
    _validate_artifact_output,
    build_agent_graph,
    create_memory_checkpointer,
    create_sqlite_checkpointer,
    invoke_new_run,
    resume_run,
)
from src.agent.state import create_initial_state
from src.observability.trace_manager import TraceManager
from src.tools.registry import load_tool_registry
from src.tools import generate_cover_letter as cover_module
from src.tools import tailor_resume as tailoring_module
from src.schemas.tailoring import TailorResumeOutput


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


def test_graph_rejects_placeholder_artifact_paths() -> None:
    output = TailorResumeOutput(
        job_id="J001",
        status="OK",
        output_tex_path="outputs/J001/resume.tex",
        output_pdf_path="outputs/J001/resume.pdf",
        page_count=1,
        errors=[],
    )

    with pytest.raises(ToolExecutionError, match="does not exist"):
        _validate_artifact_output("tailor_resume", output)


def test_graph_rejects_a_real_two_page_pdf(tmp_path: Path) -> None:
    tex_path = tmp_path / "resume.tex"
    pdf_path = tmp_path / "resume.pdf"
    tex_path.write_text("\\documentclass{article}")
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.add_blank_page(width=612, height=792)
    with pdf_path.open("wb") as handle:
        writer.write(handle)
    output = TailorResumeOutput(
        job_id="J001",
        status="OK",
        output_tex_path=str(tex_path),
        output_pdf_path=str(pdf_path),
        page_count=1,
    )

    with pytest.raises(ToolExecutionError, match="has 2 pages"):
        _validate_artifact_output("tailor_resume", output)


def test_revision_feedback_can_reach_round_two_but_never_round_three() -> None:
    base_state = {
        "phase": "TAILOR",
        "status": "RUNNING",
        "top_3_job_ids": ["J1", "J2", "J3"],
        "fit_analyses": {"J1": {}, "J2": {}, "J3": {}},
        "tailoring_results": {"J2": {}, "J3": {}},
        "pending_revision_job_ids": ["J1"],
        "revision_round_job_ids": ["J1"],
        "revision_round": 1,
        "review_history": [
            {
                "actions_taken": {},
                "revision_rounds": [],
            }
        ],
        "current_tool_input": {"revision_feedback": "Make this shorter."},
        "tool_history": [],
        "errors": [],
    }
    first = TailorResumeOutput(
        job_id="J1",
        status="OK",
        output_tex_path="resume-r1.tex",
        output_pdf_path="resume-r1.pdf",
        page_count=1,
        revision_feedback_satisfied=False,
        revision_feedback_checks=["conciseness requested: not met"],
    )

    round_two = _apply_tool_output(base_state, "tailor_resume", first)
    assert round_two["revision_round"] == 2
    assert round_two["pending_revision_job_ids"] == ["J1"]
    assert round_two["phase"] == "TAILOR"

    second_state = {**base_state, **round_two}
    second = first.model_copy(
        update={
            "output_tex_path": "resume-r2.tex",
            "output_pdf_path": "resume-r2.pdf",
        }
    )
    stopped = _apply_tool_output(second_state, "tailor_resume", second)
    assert stopped["status"] == "FAILED_REVIEW"
    assert stopped["pending_revision_job_ids"] == []
    rounds = stopped["review_history"][0]["revision_rounds"]
    assert [entry["revision_round"] for entry in rounds] == [1, 2]


def test_rejections_are_revised_after_the_only_review_pause(
    tmp_path: Path, monkeypatch
) -> None:
    """One review payload can request revisions without a second interrupt."""

    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "outputs"))

    def write_pdf(path: Path) -> None:
        writer = PdfWriter()
        writer.add_blank_page(width=612, height=792)
        with path.open("wb") as handle:
            writer.write(handle)

    def compile_resume(text: str, tex_path: Path, pdf_path: Path):
        tex_path.write_text(text, encoding="utf-8")
        write_pdf(pdf_path)
        return 1, [], text

    def compile_letter(letter, job, tex_path: Path, pdf_path: Path):
        tex_path.write_text(cover_module._render_tex(letter, job, 0), encoding="utf-8")
        write_pdf(pdf_path)
        return 1, []

    monkeypatch.setattr(tailoring_module, "_compile_one_page", compile_resume)
    monkeypatch.setattr(cover_module, "_compile_one_page", compile_letter)
    monkeypatch.setattr(
        cover_module,
        "_read_approved_resume",
        lambda path: (
            "Avery Morgan\nHouston, TX | avery@example.com | github.com/avery",
            [],
        ),
    )
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
    first = invoke_new_run(app, state)
    payload = first["__interrupt__"][0].value
    feedback = {
        job_id: {"decision": "reject", "comment": "Make this shorter."}
        for job_id in payload["resumes"]
    }
    result = resume_run(app, "thread-fail-review", feedback)

    assert result["status"] == "COMPLETED"
    assert not result.get("__interrupt__")
    assert len(result["review_history"]) == 1
    assert len(result["cover_letter_results"]) == 3
