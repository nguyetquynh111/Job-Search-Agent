"""Full workflow tests."""

from __future__ import annotations

import json
from pathlib import Path

from pypdf import PdfWriter

from src.agent.graph import (
    build_agent_graph,
    create_memory_checkpointer,
    invoke_new_run,
    resume_run,
)
from src.agent.state import create_initial_state
from src.observability.trace_manager import TraceManager
from src.tools.registry import load_tool_registry
from src.tools import generate_cover_letter as cover_module
from src.tools import tailor_resume as tailoring_module


def test_graph_wiring_across_review_memory_revision_and_letters(
    tmp_path: Path, monkeypatch
) -> None:
    """Fast wiring test; the separate real-pdflatex smoke test proves artifacts."""

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
    revision_decisions = [
        item
        for item in final["agent_decisions"]
        if item["selected_tool"] == "tailor_resume"
        and item["arguments"].get("revision_feedback")
    ]
    assert {item["arguments"]["job"]["job_id"] for item in revision_decisions} == {
        "J017",
        "J028",
    }
    assert any(
        evidence["evidence_id"].startswith("mem-")
        for item in revision_decisions
        for evidence in item["arguments"]["candidate_evidence"]
    )
    assert all(
        item["arguments"]["source_resume_tex_path"] != final["resume_path"]
        for item in revision_decisions
    )
    assert set(final["review_history"][0]["actions_taken"]) == {"J017", "J028"}

    assert final["status"] == "COMPLETED"
    assert final["phase"] == "COMPLETE"
    assert len(final["cover_letter_results"]) == 3
    assert not final["errors"]
    assert final["trace_id"] == "trace-run-e2e"
    assert [event.name for event in tracer.events].count("job_search_agent_run") == 1
    review_event = next(
        event for event in tracer.events if event.name == "human_review"
    )
    memory_event = next(
        event for event in tracer.events if event.name == "memory_write"
    )
    assert memory_event.parent_observation_id == review_event.observation_id
    assert memory_event.metadata["facts"][0]["fact"] == "LangGraph"
    assert memory_event.metadata["facts"][0]["provenance"]["source"] == "human_review"
    assert memory_event.metadata["facts"][0]["source_reviewer_feedback"] == (
        "Add LangGraph. I have used it in previous projects."
    )
    assert memory_event.metadata["facts"][0]["timestamp"]
    assert memory_event.metadata["affected_downstream_artifacts"] == [
        "J017/resume",
        "J028/resume",
    ]
    revision_event = next(
        event for event in tracer.events if event.name == "review_revision"
    )
    assert revision_event.parent_observation_id == review_event.observation_id
    revised_tool_events = [
        event
        for event in tracer.events
        if event.name.startswith("tailor_resume_job_")
        and event.parent_observation_id == revision_event.observation_id
    ]
    assert len(revised_tool_events) == 2
    cover_tool_events = [
        event
        for event in tracer.events
        if event.name.startswith("generate_cover_letter_job_")
    ]
    assert len(cover_tool_events) == 3
    assert all(
        event.parent_observation_id == review_event.observation_id
        for event in cover_tool_events
    )
    assert all("payload" in event.input for event in revised_tool_events)
    assert all("result" in event.output for event in revised_tool_events)
    assert all("configuration" in event.metadata for event in revised_tool_events)
    revised_ids = {event.observation_id for event in revised_tool_events}
    assert any(
        event.name == "resume_tailoring.compile_pdf"
        and event.parent_observation_id in revised_ids
        for event in tracer.events
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
    decision_event = next(
        event
        for event in tracer.events
        if event.name == "agent_controller"
        and event.metadata.get("selected_tool") == "analyze_fit"
    )
    assert decision_event.metadata["available_tools"]
    assert decision_event.metadata["target_job_id"] in final["top_3_job_ids"]
    assert decision_event.metadata["decision_reason"]
    assert "current_workflow_state" in decision_event.input
    fit_artifact_events = [
        event
        for event in tracer.events
        if event.name == "fit_analysis.write_artifacts"
    ]
    assert len(fit_artifact_events) >= 3
    assert all(
        event.output["markdown_exists"] and event.output["json_exists"]
        for event in fit_artifact_events
    )
    refreshed_fit_tools = [
        event
        for event in tracer.events
        if event.name.startswith("fit_analysis_job_")
        and event.parent_observation_id == revision_event.observation_id
    ]
    assert refreshed_fit_tools
    refreshed_fit_ids = {event.observation_id for event in refreshed_fit_tools}
    assert any(
        event.parent_observation_id in refreshed_fit_ids
        for event in fit_artifact_events
    )
    refreshed_jobs = {event.metadata["job_id"] for event in refreshed_fit_tools}
    for job_id in refreshed_jobs:
        payload = json.loads(
            Path(
                final["fit_analysis_artifacts"][job_id]["json_path"]
            ).read_text(encoding="utf-8")
        )
        evidence_ids = {
            evidence_id
            for field in (
                "aligned_skills",
                "evidenced_missing_skills",
                "genuine_gaps",
            )
            for claim in payload[field]
            for evidence_id in claim["evidence_ids"]
        }
        assert any(evidence_id.startswith("mem-") for evidence_id in evidence_ids)
    assert {event.trace_id for event in tracer.events} == {"trace-run-e2e"}
    assert [event.name for event in tracer.events].count("human_review_pause") == 1
