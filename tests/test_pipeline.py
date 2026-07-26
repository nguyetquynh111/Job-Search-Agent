"""Consolidated tests for this domain."""

from __future__ import annotations

from pathlib import Path
from pypdf import PdfReader
from pypdf import PdfWriter
from src.agent import (
    ToolExecutionError,
    _validate_artifact_output,
    create_sqlite_checkpointer,
)
from src.agent import (
    build_agent_graph,
    create_memory_checkpointer,
    invoke_new_run,
    resume_run,
)
from src.agent import MANDATORY_JOB_FILES, write_and_validate_outputs
from src.agent import create_initial_state
from src.review.human_review import build_review_payload
from src.tools.cover_letter import cover_letter as cover_module
from src.tools.cover_letter.cover_letter import run_cover_letter_tool
from src.tools.filtering_scoring.filtering import run_filtering_tool
from src.tools.filtering_scoring.scoring import run_scoring_tool
from src.tools.fit_analysis.fit_analysis import run_fit_analysis_tool
from src.tools.resume_tailoring import resume_tailoring as tailoring_module
from src.tools.resume_tailoring.resume_tailoring import TailorResumeOutput
from src.tools.resume_tailoring.resume_tailoring import run_resume_tailoring_tool
from src.tracing.langfuse import TraceManager
import ast
import pytest
import shutil

# --- test_agent_graph.py ---
"""Graph and artifact validation behavior."""


def test_sqlite_checkpointer_creates_missing_parent_and_database(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "missing" / "checkpoints.sqlite"
    _, context = create_sqlite_checkpointer(db_path)
    try:
        assert db_path.is_file()
    finally:
        context.__exit__(None, None, None)


def test_orchestrator_rejects_placeholder_artifact_paths() -> None:
    output = TailorResumeOutput(
        job_id="J001",
        status="OK",
        output_tex_path="outputs/J001/resume.tex",
        output_pdf_path="outputs/J001/resume.pdf",
        page_count=1,
    )

    with pytest.raises(ToolExecutionError, match="does not exist"):
        _validate_artifact_output("run_resume_tailoring_tool", output)


def test_orchestrator_rejects_a_real_two_page_pdf(tmp_path: Path) -> None:
    tex_path = tmp_path / "resume.tex"
    pdf_path = tmp_path / "resume.pdf"
    tex_path.write_text("\\documentclass{article}", encoding="utf-8")
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
        _validate_artifact_output("run_resume_tailoring_tool", output)


# --- test_agent_orchestration_contract.py ---
"""The orchestrator is a direct, readable five-tool pipeline."""


TOOL_CALLS = [
    "run_filtering_tool",
    "run_scoring_tool",
    "run_fit_analysis_tool",
    "run_resume_tailoring_tool",
    "run_cover_letter_tool",
]


def test_orchestrator_calls_tools_in_required_order() -> None:
    source = Path("src/agent.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [
        (node.lineno, node.func.id)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {*TOOL_CALLS, "interrupt"}
    ]
    calls = [name for _, name in sorted(calls)]

    first_tool_calls = []
    for name in calls:
        if name in TOOL_CALLS and name not in first_tool_calls:
            first_tool_calls.append(name)

    assert first_tool_calls == TOOL_CALLS
    assert calls.count("interrupt") == 1
    assert calls.index("run_resume_tailoring_tool") < calls.index("interrupt")
    assert calls.index("interrupt") < calls.index("run_cover_letter_tool")


def test_orchestrator_has_no_registry_or_dynamic_tool_loading() -> None:
    source = Path("src/agent.py").read_text(encoding="utf-8")

    assert "registry" not in source.casefold()
    assert "import_module" not in source
    assert "getattr(" not in source


# --- test_agent_phase_policy.py ---
"""Public tool-package contracts."""


def test_each_tool_package_exposes_one_run_callable() -> None:
    callables = [
        run_filtering_tool,
        run_scoring_tool,
        run_fit_analysis_tool,
        run_resume_tailoring_tool,
        run_cover_letter_tool,
    ]

    assert all(callable(tool) for tool in callables)
    assert [tool.__name__ for tool in callables] == [
        "run_filtering_tool",
        "run_scoring_tool",
        "run_fit_analysis_tool",
        "run_resume_tailoring_tool",
        "run_cover_letter_tool",
    ]


# --- test_agent_review_interrupt.py ---
"""Human-review payload contracts."""


def test_one_payload_contains_all_three_resumes() -> None:
    state = {
        "top_3_job_ids": ["J1", "J2", "J3"],
        "jobs": [
            {"job_id": "J1", "title": "One", "company": "A"},
            {"job_id": "J2", "title": "Two", "company": "B"},
            {"job_id": "J3", "title": "Three", "company": "C"},
        ],
        "fit_analyses": {job_id: {"job_id": job_id} for job_id in ["J1", "J2", "J3"]},
        "tailoring_results": {
            job_id: {
                "output_pdf_path": f"/tmp/{job_id}.pdf",
                "change_log": [],
            }
            for job_id in ["J1", "J2", "J3"]
        },
        "revision_round": 0,
    }

    payload = build_review_payload(state)

    assert payload.review_round == 1
    assert set(payload.resumes) == {"J1", "J2", "J3"}


# --- test_integration_end_to_end.py ---
"""Full workflow tests."""


def test_graph_wiring_across_review_memory_revision_and_letters(
    tmp_path: Path, monkeypatch
) -> None:
    """Exercise workflow wiring with in-memory PDF fixtures."""

    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "outputs"))

    def write_pdf(path: Path) -> None:
        writer = PdfWriter()
        writer.add_blank_page(width=612, height=792)
        with path.open("wb") as handle:
            writer.write(handle)

    def compile_resume(
        text: str,
        tex_path: Path,
        pdf_path: Path,
        *,
        tracer=None,
        trace_metadata=None,
    ):
        assert tracer is not None
        with tracer.span(
            "resume_tailoring.compile_pdf",
            trace_metadata,
            input={"tex_file": tex_path.name},
        ):
            tex_path.write_text(text, encoding="utf-8")
            write_pdf(pdf_path)
        with tracer.span(
            "resume_tailoring.validate_page_count",
            trace_metadata,
            input={"pdf_file": pdf_path.name},
        ):
            pass
        return 1, [], text

    def compile_letter(
        letter,
        job,
        tex_path: Path,
        pdf_path: Path,
        *,
        tracer=None,
        trace_metadata=None,
    ):
        assert tracer is not None
        with tracer.span(
            "cover_letter.compile_pdf",
            trace_metadata,
            input={"tex_file": tex_path.name},
        ):
            tex_path.write_text(
                cover_module._render_tex(letter, job, 0), encoding="utf-8"
            )
            write_pdf(pdf_path)
        with tracer.span(
            "cover_letter.validate_page_count",
            trace_metadata,
            input={"pdf_file": pdf_path.name},
        ):
            pass
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
    tracer = TraceManager(enabled=False)
    app = build_agent_graph(
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
        "comment": "Add LangGraph. I have used it in previous projects.",
    }

    final = resume_run(app, "thread-e2e", first_feedback)
    assert final["status"] == "COMPLETED", "; ".join(
        f"{round_entry['revision_round']}:{action['job_id']}:"
        f"{action['feedback_satisfied']}:{action['feedback_checks']}"
        for round_entry in final["review_history"][0]["revision_rounds"]
        for action in round_entry["actions"]
    )
    assert final["revision_round"] == 1
    assert not final.get("__interrupt__")
    assert any(fact["canonical_value"] == "LangGraph" for fact in final["memory_facts"])
    assert set(final["review_history"][0]["actions_taken"]) == {rejected_job_id}

    tool_names = [item["tool"] for item in final["tool_history"]]
    assert tool_names.count("run_filtering_tool") == 1
    assert tool_names.count("run_scoring_tool") == 1
    assert tool_names.count("run_fit_analysis_tool") == 3
    assert tool_names.count("run_resume_tailoring_tool") == 4
    assert tool_names.count("run_cover_letter_tool") == 3

    assert final["status"] == "COMPLETED"
    assert final["phase"] == "COMPLETE"
    assert len(final["cover_letter_results"]) == 3
    assert not final["errors"]
    assert final["trace_id"] == "trace-run-e2e"
    assert [event.name for event in tracer.events].count("job_search_agent_run") == 1
    resume_compile_events = [
        event for event in tracer.events if event.name == "resume_tailoring.compile_pdf"
    ]
    assert len(resume_compile_events) >= 4
    cover_compile_events = [
        event for event in tracer.events if event.name == "cover_letter.compile_pdf"
    ]
    assert len(cover_compile_events) >= 3
    assert len(final["fit_analysis_artifacts"]) == 3
    assert all(
        Path(path).is_file()
        for paths in final["fit_analysis_artifacts"].values()
        for path in paths.values()
    )
    ambiguous_tool_names = {
        "filter_jobs",
        "score_jobs",
        "analyze_fit",
        "tailor_resume",
        "generate_cover_letter",
    }
    names = [event.name for event in tracer.events]
    assert not any(
        names.count(name) > 1 and not name.endswith(("_job_1", "_job_2", "_job_3"))
        for name in ambiguous_tool_names
    )
    assert {event.trace_id for event in tracer.events} == {"trace-run-e2e"}
    assert [event.name for event in tracer.events].count("human_review_pause") == 1


# --- test_integration_output_contract.py ---
"""Canonical Top-3 output writing and validation."""


def _integration_output_write_one_page_pdf(path: Path) -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    with path.open("wb") as handle:
        writer.write(handle)


def test_output_writer_produces_and_validates_three_complete_job_folders(
    tmp_path: Path, monkeypatch
) -> None:
    output_dir = tmp_path / "outputs"
    monkeypatch.setenv("OUTPUT_DIR", str(output_dir))
    job_ids = ["J1", "J2", "J3"]
    tailoring = {}
    letters = {}
    fit_artifacts = {}

    for job_id in job_ids:
        job_dir = output_dir / job_id
        job_dir.mkdir(parents=True)
        _integration_output_write_one_page_pdf(job_dir / "resume_before.pdf")
        _integration_output_write_one_page_pdf(job_dir / "approved.pdf")
        _integration_output_write_one_page_pdf(job_dir / "letter.pdf")
        fit_path = job_dir / "fit_analysis.md"
        fit_path.write_text(f"# Fit analysis for {job_id}\n", encoding="utf-8")
        (job_dir / "resume.aux").write_text("temporary", encoding="utf-8")
        tailoring[job_id] = {"output_pdf_path": str(job_dir / "approved.pdf")}
        letters[job_id] = {"output_pdf_path": str(job_dir / "letter.pdf")}
        fit_artifacts[job_id] = {"markdown_path": str(fit_path)}

    state = {
        "top_3_job_ids": job_ids,
        "jobs": [
            {
                "job_id": job_id,
                "title": "ML Engineer",
                "company": f"Company {job_id}",
                "description": "Build ML systems.",
            }
            for job_id in job_ids
        ],
        "tailoring_results": tailoring,
        "cover_letter_results": letters,
        "fit_analysis_artifacts": fit_artifacts,
    }

    manifest = write_and_validate_outputs(state)

    assert manifest["job_ids"] == job_ids
    assert {path.name for path in output_dir.iterdir() if path.is_dir()} == set(job_ids)
    for job_id in job_ids:
        job_dir = output_dir / job_id
        assert all((job_dir / name).is_file() for name in MANDATORY_JOB_FILES)
        assert not (job_dir / "resume.aux").exists()


# --- test_integration_real_pdflatex_end_to_end.py ---
"""Workflow integration test using real LaTeX and PDF extraction."""


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
    checkpointer, context = create_sqlite_checkpointer(
        output_dir / "checkpoints.sqlite"
    )
    try:
        app = build_agent_graph(
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
        thread_id = state.get("thread_id")
        assert thread_id is not None
        final = resume_run(app, thread_id, feedback)
    finally:
        context.__exit__(None, None, None)

    assert final["status"] == "COMPLETED"
    assert final["revision_round"] == 1
    assert len(final["review_history"]) == 1
    assert [event.name for event in tracer.events].count("human_review_pause") == 1
    assert any(fact["canonical_value"] == "LangGraph" for fact in final["memory_facts"])
    for job_id in final["top_3_job_ids"]:
        resume = final["tailoring_results"][job_id]
        letter = final["cover_letter_results"][job_id]
        assert len(PdfReader(resume["output_pdf_path"]).pages) == 1
        assert len(PdfReader(letter["output_pdf_path"]).pages) == 1
        assert Path(resume["output_tex_path"]).is_file()
        assert Path(letter["output_tex_path"]).is_file()
