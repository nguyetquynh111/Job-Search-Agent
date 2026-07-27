"""Consolidated tests for this domain."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import threading
import time
from pypdf import PdfReader
from pypdf import PdfWriter
from src.agent import (
    ToolExecutionError,
    create_sqlite_checkpointer,
    validate_artifact_output,
)
from src.agent import (
    build_agent_graph,
    create_memory_checkpointer,
    invoke_new_run,
    resume_run,
)
from src.agent import (
    CandidatePreferences,
    Job,
    MANDATORY_JOB_FILES,
    write_and_validate_outputs,
)
from src.agent import create_initial_state
from src.review.human_review import build_review_payload
from src.tools.cover_letter import cover_letter as cover_module
from src.tools.cover_letter.cover_letter import run_cover_letter_tool
from src.tools.filtering_scoring.filtering import run_filtering_tool
from src.tools.filtering_scoring.scoring import ScoredJob, run_scoring_tool
from src.tools.fit_analysis.fit_analysis import run_fit_analysis_tool
from src.tools.resume_tailoring import resume_tailoring as tailoring_module
from src.tools.resume_tailoring.resume_tailoring import TailorResumeOutput
from src.tools.resume_tailoring.resume_tailoring import run_resume_tailoring_tool
from src.tools.registry import (
    ToolArgumentError,
    UnknownToolError,
    get_registered_tools,
    get_tool,
    get_tool_definitions,
    invoke_tool,
)
from src.agent.tool_selection import ModelToolCall, validate_model_tool_call
from src.tools.filtering_scoring.filtering import FilterJobsOutput
from src.tools.filtering_scoring.scoring import ScoreJobsOutput
from src.tools.fit_analysis.contracts import FitAnalysisOutput
from src.tracing.langfuse import TraceManager
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
        validate_artifact_output("run_resume_tailoring_tool", output)


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
        validate_artifact_output("run_resume_tailoring_tool", output)


# --- test_agent_orchestration_contract.py ---
"""The orchestrator dispatches structured tool calls through the registry."""


REQUIRED_TOOL_NAMES = [
    "filtering",
    "scoring",
    "fit_analysis",
    "resume_tailoring",
    "cover_letter",
]


class StateChoosingToolSelectionModel:
    model_name = "state-choosing-test-llm"

    def __init__(self, *, first_call: ModelToolCall | None = None) -> None:
        self.first_call = first_call
        self.calls: list[dict[str, object]] = []

    def select_tool(
        self,
        *,
        available_tools,
        tool_definitions,
        state_summary,
        tracer,
        previous_validation_results=(),
    ) -> ModelToolCall:
        del tool_definitions
        self.calls.append(
            {
                "available_tools": list(available_tools),
                "state_summary": dict(state_summary),
                "previous_validation_results": list(previous_validation_results),
            }
        )
        if self.first_call is not None:
            call = self.first_call
            self.first_call = None
        else:
            call = self._choose_from_state(state_summary)
        tracer.record_generation(
            {
                "purpose": "orchestration_tool_selection",
                "available_tools": list(available_tools),
            },
            name="orchestration.model_decision",
            model=self.model_name,
            messages={"state_summary": state_summary},
            response=call.model_dump(mode="json"),
            model_parameters={"temperature": 0, "test_double": True},
        )
        return call

    def _choose_from_state(self, state_summary) -> ModelToolCall:
        if state_summary["filtered_job_count"] == 0:
            return ModelToolCall(
                name="filtering",
                rationale="No filtered jobs exist yet.",
            )
        if state_summary["ranked_job_count"] == 0:
            return ModelToolCall(
                name="scoring",
                rationale="Filtered jobs are ready for deterministic scoring.",
            )
        if state_summary["fit_analysis_remaining_job_ids"]:
            job_id = state_summary["fit_analysis_remaining_job_ids"][0]
            return ModelToolCall(
                name="fit_analysis",
                arguments={"job_id": job_id},
                rationale="A top job still needs fit analysis.",
            )
        if state_summary["tailoring_remaining_job_ids"]:
            job_id = state_summary["tailoring_remaining_job_ids"][0]
            return ModelToolCall(
                name="resume_tailoring",
                arguments={"job_id": job_id},
                rationale="Fit analyses are complete; draft this resume.",
            )
        if state_summary["pending_revision_job_ids"]:
            job_id = state_summary["pending_revision_job_ids"][0]
            return ModelToolCall(
                name="resume_tailoring",
                arguments={"job_id": job_id},
                rationale="Human feedback requires a revision.",
            )
        if state_summary["cover_letter_remaining_job_ids"]:
            job_id = state_summary["cover_letter_remaining_job_ids"][0]
            return ModelToolCall(
                name="cover_letter",
                arguments={"job_id": job_id},
                rationale="Approved resumes are ready for cover letters.",
            )
        raise AssertionError(f"no valid test choice for state: {state_summary}")


def test_tool_registry_contains_required_tools_once() -> None:
    tools = get_registered_tools()

    assert [tool.name for tool in tools] == REQUIRED_TOOL_NAMES
    assert len({tool.name for tool in tools}) == len(REQUIRED_TOOL_NAMES)
    assert get_tool("scoring").callable_name == "run_scoring_tool"


def test_legacy_public_imports_and_core_imports_stay_ui_free() -> None:
    script = """
import sys
import src.agent.controller as controller
import src.agent.graph as graph
import src.agent.state as state
import src.schemas.jobs as jobs
import src.tools.registry as registry

assert controller.AgentController
assert graph.build_agent_graph
assert state.AgentState
assert jobs.Job
assert registry.get_registered_tools
assert "streamlit" not in sys.modules
print("public imports: OK")
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "public imports: OK"


def test_registry_validates_structured_dispatch_and_errors() -> None:
    job = Job(
        job_id="J-test",
        title="Machine Learning Engineer",
        company="Example AI",
        location="Remote, US",
        remote=True,
        description="Build ML systems.",
        required_skills=["Python"],
    )
    result = invoke_tool(
        "filtering",
        {
            "jobs": [job.model_dump()],
            "preferences": CandidatePreferences(remote_only=True).model_dump(),
        },
        context={"tracer": TraceManager(enabled=False)},
    )

    assert isinstance(result, FilterJobsOutput)
    assert [accepted.job_id for accepted in result.accepted_jobs] == ["J-test"]
    with pytest.raises(UnknownToolError, match="Unknown tool"):
        invoke_tool("unknown", {})
    with pytest.raises(ToolArgumentError, match="Invalid arguments"):
        invoke_tool("filtering", {"jobs": []})


def test_llm_style_tool_call_uses_bound_definitions_and_dispatches() -> None:
    definitions = get_tool_definitions()

    class FakeModel:
        def choose_tool(self, tool_definitions):
            assert [tool["name"] for tool in tool_definitions] == REQUIRED_TOOL_NAMES
            return {
                "name": "filtering",
                "arguments": {
                    "jobs": [
                        {
                            "job_id": "J-fake",
                            "title": "ML Engineer",
                            "company": "Example",
                            "description": "Build ML.",
                        }
                    ],
                    "preferences": {},
                },
            }

    tool_call = FakeModel().choose_tool(definitions)
    result = invoke_tool(tool_call["name"], tool_call["arguments"])

    assert isinstance(result, FilterJobsOutput)
    assert result.accepted_jobs[0].job_id == "J-fake"


def test_model_tool_call_shape_rejects_unknown_and_malformed_arguments() -> None:
    with pytest.raises(ToolExecutionError, match="unknown tool"):
        validate_model_tool_call(
            ModelToolCall(name="not_registered", arguments={}),
        )
    with pytest.raises(ToolExecutionError, match="may only include"):
        validate_model_tool_call(
            ModelToolCall(
                name="filtering",
                arguments={"model_score": 100},
            )
        )


def test_model_cannot_inject_deterministic_scoring_values() -> None:
    with pytest.raises(ToolExecutionError, match="may only include"):
        validate_model_tool_call(
            ModelToolCall(
                name="scoring",
                arguments={
                    "jobs": [],
                    "candidate_profile": {},
                    "ranked_jobs": [{"score": 100}],
                },
            )
        )


def test_different_repository_states_drive_different_model_tool_choices() -> None:
    selector = StateChoosingToolSelectionModel()

    filtering_call = selector._choose_from_state(
        {
            "filtered_job_count": 0,
            "ranked_job_count": 0,
            "fit_analysis_remaining_job_ids": [],
            "tailoring_remaining_job_ids": [],
            "pending_revision_job_ids": [],
            "cover_letter_remaining_job_ids": [],
        }
    )
    scoring_call = selector._choose_from_state(
        {
            "filtered_job_count": 5,
            "ranked_job_count": 0,
            "fit_analysis_remaining_job_ids": [],
            "tailoring_remaining_job_ids": [],
            "pending_revision_job_ids": [],
            "cover_letter_remaining_job_ids": [],
        }
    )

    assert filtering_call.name == "filtering"
    assert scoring_call.name == "scoring"


def test_graph_orchestration_dispatches_pre_review_tools_through_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.agent.graph as graph_module

    calls: list[str] = []
    fit_lock = threading.Lock()
    active_fit_calls = 0
    peak_fit_calls = 0

    def fake_invoke_tool(name, arguments, context=None):
        nonlocal active_fit_calls, peak_fit_calls
        calls.append(name)
        if name == "filtering":
            return FilterJobsOutput(accepted_jobs=arguments.jobs, rejected_jobs=[])
        if name == "scoring":
            ranked_jobs = [
                ScoredJob(
                    job=job,
                    score=100 - index,
                    score_breakdown={},
                    rationale="fake deterministic ranking",
                    evidence_ids=[],
                )
                for index, job in enumerate(arguments.jobs)
            ]
            return ScoreJobsOutput(
                ranked_jobs=ranked_jobs,
                top_3_job_ids=[job.job_id for job in arguments.jobs[:3]],
            )
        if name == "fit_analysis":
            with fit_lock:
                active_fit_calls += 1
                peak_fit_calls = max(peak_fit_calls, active_fit_calls)
            try:
                time.sleep(0.05)
                return FitAnalysisOutput(job_id=arguments.job.job_id)
            finally:
                with fit_lock:
                    active_fit_calls -= 1
        if name == "resume_tailoring":
            return TailorResumeOutput(
                job_id=arguments.job.job_id,
                status="OK",
                output_tex_path=f"/tmp/{arguments.job.job_id}.tex",
                output_pdf_path=f"/tmp/{arguments.job.job_id}.pdf",
                page_count=1,
            )
        raise AssertionError(f"unexpected tool: {name}")

    monkeypatch.setattr(graph_module, "invoke_tool", fake_invoke_tool)
    monkeypatch.setattr(graph_module, "validate_artifact_output", lambda *_args: None)

    selector = StateChoosingToolSelectionModel()
    tracer = TraceManager(enabled=False)
    app = build_agent_graph(
        checkpointer=create_memory_checkpointer(),
        tracer=tracer,
        tool_selection_model=selector,
    )
    result = invoke_new_run(app, create_initial_state(thread_id="thread-registry"))

    assert result["__interrupt__"]
    assert all(len(call["available_tools"]) >= 2 for call in selector.calls)
    assert calls == [
        "filtering",
        "scoring",
        "fit_analysis",
        "fit_analysis",
        "fit_analysis",
        "resume_tailoring",
        "resume_tailoring",
        "resume_tailoring",
    ]
    assert peak_fit_calls == 3
    decision_spans = [
        event for event in tracer.events if event.name.startswith("Decision:")
    ]
    assert decision_spans
    assert all(
        len(event.input["available_tools"]) == len(REQUIRED_TOOL_NAMES)
        for event in decision_spans
    )
    assert all(event.output["selected_tool"] for event in decision_spans)
    assert all(event.output["reason"] for event in decision_spans)
    generations = [
        event for event in tracer.events if event.name == "Workflow Decision LLM"
    ]
    decision_span_ids = {event.observation_id for event in decision_spans}
    assert generations
    assert all(event.model == selector.model_name for event in generations)
    assert all(event.parent_observation_id in decision_span_ids for event in generations)


def test_invalid_tool_choice_is_blocked_by_validator_not_hidden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.agent.graph as graph_module

    calls: list[str] = []

    def fake_invoke_tool(name, arguments, context=None):
        calls.append(name)
        if name == "filtering":
            return FilterJobsOutput(accepted_jobs=arguments.jobs, rejected_jobs=[])
        if name == "scoring":
            ranked_jobs = [
                ScoredJob(
                    job=job,
                    score=100 - index,
                    score_breakdown={},
                    rationale="fake deterministic ranking",
                    evidence_ids=[],
                )
                for index, job in enumerate(arguments.jobs)
            ]
            return ScoreJobsOutput(
                ranked_jobs=ranked_jobs,
                top_3_job_ids=[job.job_id for job in arguments.jobs[:3]],
            )
        if name == "fit_analysis":
            return FitAnalysisOutput(job_id=arguments.job.job_id)
        if name == "resume_tailoring":
            return TailorResumeOutput(
                job_id=arguments.job.job_id,
                status="OK",
                output_tex_path=f"/tmp/{arguments.job.job_id}.tex",
                output_pdf_path=f"/tmp/{arguments.job.job_id}.pdf",
                page_count=1,
            )
        raise AssertionError(f"unexpected tool dispatch: {name}")

    monkeypatch.setattr(graph_module, "invoke_tool", fake_invoke_tool)
    monkeypatch.setattr(graph_module, "validate_artifact_output", lambda *_args: None)

    selector = StateChoosingToolSelectionModel(
        first_call=ModelToolCall(
            name="cover_letter",
            arguments={"job_id": "J001"},
            rationale="Try a cover letter too early.",
        )
    )
    tracer = TraceManager(enabled=False)
    app = build_agent_graph(
        checkpointer=create_memory_checkpointer(),
        tracer=tracer,
        tool_selection_model=selector,
    )
    result = invoke_new_run(app, create_initial_state(thread_id="thread-invalid"))

    assert result["__interrupt__"]
    assert calls[0] == "filtering"
    assert "cover_letter" not in calls
    first_decision = next(
        event for event in tracer.events if event.name.startswith("Decision:")
    )
    assert len(first_decision.input["available_tools"]) == len(REQUIRED_TOOL_NAMES)
    assert first_decision.output["selected_tool"] == "cover_letter"
    assert first_decision.output["validation"]["valid"] is False
    assert (
        "human review approval gate"
        in first_decision.output["validation"]["message"]
    )
    assert selector.calls[1]["previous_validation_results"][0]["valid"] is False


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
        tex_path.write_text(text, encoding="utf-8")
        write_pdf(pdf_path)
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
        tex_path.write_text(
            cover_module._render_tex(letter, job, 0), encoding="utf-8"
        )
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
    memory_file.write_text(
        """
[
  {
    "fact_id": "mem-old-experience",
    "fact_type": "experience",
    "canonical_value": "2 years of experience in data engineering",
    "provenance": {
      "source": "human_review",
      "review_round": 1,
      "original_statement": "I have 2 years of experience in data engineering.",
      "related_job_id": "J000"
    },
    "created_at": "2026-01-01T00:00:00+00:00",
    "active": true
  }
]
""".strip(),
        encoding="utf-8",
    )
    tracer = TraceManager(enabled=False)
    selector = StateChoosingToolSelectionModel()
    app = build_agent_graph(
        checkpointer=create_memory_checkpointer(),
        tracer=tracer,
        tool_selection_model=selector,
    )
    state = create_initial_state(
        thread_id="thread-e2e",
        run_id="run-e2e",
        memory_file=str(memory_file),
    )

    first = invoke_new_run(app, state)
    first_payload = first["__interrupt__"][0].value
    rejected_job_id = list(first_payload["resumes"])[1]
    revision_submission = {
        "decisions": {
            job_id: {
                "decision": "reject" if job_id == rejected_job_id else "approve",
                "comment": (
                    "Add LangGraph. I have used it in previous projects."
                    if job_id == rejected_job_id
                    else ""
                ),
            }
            for job_id in first_payload["resumes"]
        }
    }

    final = resume_run(app, "thread-e2e", revision_submission)
    assert final["status"] == "COMPLETED", "; ".join(
        f"{round_entry['revision_round']}:{action['job_id']}:"
        f"{action['feedback_satisfied']}:{action['feedback_checks']}"
        for round_entry in final["review_history"][0]["revision_rounds"]
        for action in round_entry["actions"]
    )
    assert final["revision_round"] == 1
    assert not final.get("__interrupt__")
    assert any(
        fact["canonical_value"] == "LangGraph"
        for fact in final["memory_facts"]
    )
    assert set(final["review_history"][0]["actions_taken"]) == {rejected_job_id}
    assert any(
        fact["canonical_value"] == "LangGraph"
        for fact in final["review_history"][0]["memory_writes"]
    )
    revision_rounds = final["review_history"][0]["revision_rounds"]
    assert revision_rounds[0]["review_round"] == 1
    assert revision_rounds[0]["revision_round"] == 1
    assert revision_rounds[0]["feedback_received_by_job"] == {
        rejected_job_id: "Add LangGraph. I have used it in previous projects."
    }
    revision_action = revision_rounds[0]["actions"][0]
    assert revision_action["job_id"] == rejected_job_id
    assert revision_action["feedback_received"] == (
        "Add LangGraph. I have used it in previous projects."
    )
    assert revision_action["changes_accepted"]
    assert revision_action["evidence_used"]
    assert isinstance(revision_action["feedback_satisfied"], bool)

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
    assert [event.name for event in tracer.events].count("Job Search Agent Run") == 1
    assert [event.name for event in tracer.events].count("Memory Update Tool") == 1
    assert final["output_manifest"]["job_ids"] == final["top_3_job_ids"]
    assert final["output_manifest"]["run_id"] == "run-e2e"
    assert Path(final["output_manifest"]["output_root"]).name == "run-e2e"
    internal_names = {
        "orchestration.dispatch_validated_call",
        "tool_registry.dispatch",
        "resume_tailoring.compile_pdf",
        "resume_tailoring.validate_page_count",
        "cover_letter.compile_pdf",
        "cover_letter.validate_page_count",
        "memory.conflict_handling",
        "memory.propagation_plan",
    }
    assert not internal_names.intersection(
        event.name for event in tracer.events
    )
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
    assert [event.name for event in tracer.events].count("Human Review") == 1


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
        (job_dir / "approved.tex").write_text("resume source", encoding="utf-8")
        (job_dir / "letter.tex").write_text("letter source", encoding="utf-8")
        fit_path = job_dir / "fit_analysis.md"
        fit_json_path = job_dir / "fit_analysis.json"
        fit_path.write_text(f"# Fit analysis for {job_id}\n", encoding="utf-8")
        fit_json_path.write_text(
            '{"fit_analysis": "production-like fixture"}',
            encoding="utf-8",
        )
        (job_dir / "resume.aux").write_text("temporary", encoding="utf-8")
        tailoring[job_id] = {
            "output_pdf_path": str(job_dir / "approved.pdf"),
            "output_tex_path": str(job_dir / "approved.tex"),
            "change_log": [],
        }
        letters[job_id] = {
            "output_pdf_path": str(job_dir / "letter.pdf"),
            "output_tex_path": str(job_dir / "letter.tex"),
        }
        fit_artifacts[job_id] = {
            "markdown_path": str(fit_path),
            "json_path": str(fit_json_path),
        }

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
        "review_history": [
            {
                "review_round": 1,
                "decisions": {job_id: {"decision": "approve"} for job_id in job_ids},
                "memory_propagation": [
                    {
                        "job_id": "J1",
                        "output_tex_path": str(output_dir / "J1" / "resume_draft.tex"),
                        "output_pdf_path": str(output_dir / "J1" / "resume_draft.pdf"),
                    }
                ],
            }
        ],
        "trace_events": [
            {
                "name": "resume_tailoring.compile_pdf",
                "output": {
                    "output_tex_path": str(output_dir / "J1" / "resume_draft.tex"),
                    "output_pdf_path": str(output_dir / "J1" / "resume_draft.pdf"),
                },
            }
        ],
    }

    manifest = write_and_validate_outputs(state)

    assert manifest["job_ids"] == job_ids
    assert {path.name for path in output_dir.iterdir() if path.is_dir()} == set(job_ids)
    for job_id in job_ids:
        job_dir = output_dir / job_id
        assert all((job_dir / name).is_file() for name in MANDATORY_JOB_FILES)
        assert not (job_dir / "resume.aux").exists()
    revision_text = (output_dir / "J1" / "revision_history.json").read_text(
        encoding="utf-8"
    )
    trace_text = (output_dir / "trace_events.json").read_text(encoding="utf-8")
    assert "resume_draft" not in revision_text
    assert "output_tex_path" not in revision_text
    assert "resume_draft" not in trace_text
    assert "output_tex_path" not in trace_text


def test_output_writer_isolates_artifacts_by_run_id(
    tmp_path: Path, monkeypatch
) -> None:
    output_dir = tmp_path / "outputs"
    monkeypatch.setenv("OUTPUT_DIR", str(output_dir))
    job_ids = ["J1", "J2", "J3"]

    def state_for(run_id: str) -> dict[str, object]:
        tailoring = {}
        letters = {}
        fit_artifacts = {}
        for job_id in job_ids:
            job_dir = output_dir / run_id / job_id
            job_dir.mkdir(parents=True)
            _integration_output_write_one_page_pdf(job_dir / "resume_before.pdf")
            _integration_output_write_one_page_pdf(job_dir / "approved.pdf")
            _integration_output_write_one_page_pdf(job_dir / "letter.pdf")
            (job_dir / "approved.tex").write_text(f"resume {run_id}", encoding="utf-8")
            (job_dir / "letter.tex").write_text(f"letter {run_id}", encoding="utf-8")
            fit_path = job_dir / "fit_analysis.md"
            fit_json_path = job_dir / "fit_analysis.json"
            fit_path.write_text(f"# {run_id} {job_id}\n", encoding="utf-8")
            fit_json_path.write_text(
                f'{{"run_id": "{run_id}", "job_id": "{job_id}"}}',
                encoding="utf-8",
            )
            tailoring[job_id] = {
                "output_pdf_path": str(job_dir / "approved.pdf"),
                "output_tex_path": str(job_dir / "approved.tex"),
                "change_log": [],
            }
            letters[job_id] = {
                "output_pdf_path": str(job_dir / "letter.pdf"),
                "output_tex_path": str(job_dir / "letter.tex"),
            }
            fit_artifacts[job_id] = {
                "markdown_path": str(fit_path),
                "json_path": str(fit_json_path),
            }
        memory_file = output_dir / run_id / "memory.json"
        memory_file.write_text("[]", encoding="utf-8")
        return {
            "run_id": run_id,
            "memory_file": str(memory_file),
            "trace_id": f"trace-{run_id}",
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
            "trace_events": [{"name": run_id}],
        }

    first = write_and_validate_outputs(state_for("run-a"))
    second = write_and_validate_outputs(state_for("run-b"))

    assert Path(first["output_root"]) == output_dir / "run-a"
    assert Path(second["output_root"]) == output_dir / "run-b"
    assert (output_dir / "run-a" / "run_manifest.json").is_file()
    assert (output_dir / "run-b" / "run_manifest.json").is_file()
    assert (output_dir / "run-a" / "J1" / "resume_after.pdf").is_file()
    assert (output_dir / "run-b" / "J1" / "resume_after.pdf").is_file()
    assert not (output_dir / "run-a" / "J1" / "approved.tex").exists()
    assert not (output_dir / "run-b" / "J1" / "approved.tex").exists()


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
        selector = StateChoosingToolSelectionModel()
        app = build_agent_graph(
            checkpointer=checkpointer,
            tracer=tracer,
            tool_selection_model=selector,
        )
        state = create_initial_state(
            thread_id="thread-real-pdflatex",
            run_id="run-real-pdflatex",
            memory_file=str(memory_file),
        )

        waiting = invoke_new_run(app, state)
        assert waiting["__interrupt__"][0].value["resumes"]
        thread_id = state.get("thread_id")
        assert thread_id is not None
        final = resume_run(
            app,
            thread_id,
            {
                "decisions": {
                    job_id: {
                        "decision": "reject" if job_id == "J028" else "approve",
                        "comment": (
                            "Add LangGraph. I have used it in previous projects."
                            if job_id == "J028"
                            else ""
                        ),
                    }
                    for job_id in waiting["__interrupt__"][0].value["resumes"]
                }
            },
        )
    finally:
        context.__exit__(None, None, None)

    assert final["status"] == "COMPLETED"
    assert final["revision_round"] == 1
    assert final["review_history"][0]["rejected_job_ids"] == ["J028"]
    assert [event.name for event in tracer.events].count("Human Review") == 1
    assert any(
        fact["canonical_value"] == "LangGraph"
        for fact in final["memory_facts"]
    )
    for job_id in final["top_3_job_ids"]:
        resume = final["tailoring_results"][job_id]
        letter = final["cover_letter_results"][job_id]
        assert len(PdfReader(resume["output_pdf_path"]).pages) == 1
        assert len(PdfReader(letter["output_pdf_path"]).pages) == 1
        assert Path(resume["output_pdf_path"]).name == "resume_after.pdf"
        assert Path(letter["output_pdf_path"]).name == "cover_letter.pdf"
        assert "output_tex_path" not in resume
        assert "output_tex_path" not in letter
