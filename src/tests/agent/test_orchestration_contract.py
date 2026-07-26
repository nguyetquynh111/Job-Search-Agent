"""Single-agent workflow tests independent of unfinished business tools."""

from __future__ import annotations

from pathlib import Path

import langchain_openai
import pytest
from pypdf import PdfWriter

from src.agent import graph as graph_module
from src.agent.controller import AgentIntent, SingleAgentController
from src.agent.graph import (
    build_agent_graph,
    create_memory_checkpointer,
    create_sqlite_checkpointer,
    invoke_new_run,
    resume_run,
)
from src.agent.state import create_initial_state
from src.data_loader import load_candidate_profile, load_jobs_csv, load_portfolio
from src.observability.trace_manager import TraceManager
from src.schemas.common import ChangeLogEntry
from src.schemas.cover_letter import (
    GenerateCoverLetterInput,
    GenerateCoverLetterOutput,
)
from src.schemas.filtering import FilterJobsInput, FilterJobsOutput
from src.schemas.fit_analysis import AnalyzeFitInput, FitAnalysisOutput
from src.schemas.scoring import ScoreJobsInput, ScoreJobsOutput, ScoredJob
from src.schemas.tailoring import TailorResumeInput, TailorResumeOutput
from src.tools.registry import ToolSpec


def _contract_registry(output_dir: Path | None = None) -> dict[str, ToolSpec]:
    """Return predictable contract-valid tools for orchestration testing."""

    artifact_root = output_dir or Path("outputs")

    def artifact_paths(job_id: str, stem: str) -> tuple[Path, Path]:
        directory = artifact_root / job_id
        directory.mkdir(parents=True, exist_ok=True)
        tex_path = directory / f"{stem}.tex"
        pdf_path = directory / f"{stem}.pdf"
        tex_path.write_text(
            "\\documentclass{article}\\begin{document}test\\end{document}"
        )
        writer = PdfWriter()
        writer.add_blank_page(width=612, height=792)
        with pdf_path.open("wb") as handle:
            writer.write(handle)
        return tex_path, pdf_path

    def filter_jobs(value: FilterJobsInput) -> FilterJobsOutput:
        return FilterJobsOutput(accepted_jobs=value.jobs, rejected_jobs=[])

    def score_jobs(value: ScoreJobsInput) -> ScoreJobsOutput:
        ranked = [
            ScoredJob(job=job, score=100 - index, rationale="contract test")
            for index, job in enumerate(value.jobs)
        ]
        return ScoreJobsOutput(
            ranked_jobs=ranked,
            top_3_job_ids=[item.job.job_id for item in ranked[:3]],
        )

    def analyze_fit(value: AnalyzeFitInput) -> FitAnalysisOutput:
        return FitAnalysisOutput(job_id=value.job.job_id)

    def tailor_resume(value: TailorResumeInput) -> TailorResumeOutput:
        tex_path, pdf_path = artifact_paths(value.job.job_id, "resume")
        return TailorResumeOutput(
            job_id=value.job.job_id,
            status="OK",
            output_tex_path=str(tex_path),
            output_pdf_path=str(pdf_path),
            page_count=1,
            change_log=[
                ChangeLogEntry(
                    change_id=f"change-{value.job.job_id}",
                    section="skills",
                    description=value.revision_feedback or "initial tailoring",
                    before_text="Python",
                    after_text="Python, evidenced skill",
                    reason="Contract-test evidence-backed skill highlighting.",
                    evidence_ids=[
                        item.evidence_id for item in value.candidate_evidence[-2:]
                    ],
                )
            ],
            revision_feedback_satisfied=True if value.revision_feedback else None,
            revision_feedback_checks=(
                ["contract revision explicitly satisfied"]
                if value.revision_feedback
                else []
            ),
        )

    def generate_cover_letter(
        value: GenerateCoverLetterInput,
    ) -> GenerateCoverLetterOutput:
        tex_path, pdf_path = artifact_paths(value.job.job_id, "cover-letter")
        return GenerateCoverLetterOutput(
            job_id=value.job.job_id,
            output_tex_path=str(tex_path),
            output_pdf_path=str(pdf_path),
            page_count=1,
        )

    contracts = {
        "filter_jobs": (
            filter_jobs,
            FilterJobsInput,
            FilterJobsOutput,
        ),
        "score_jobs": (score_jobs, ScoreJobsInput, ScoreJobsOutput),
        "analyze_fit": (analyze_fit, AnalyzeFitInput, FitAnalysisOutput),
        "tailor_resume": (
            tailor_resume,
            TailorResumeInput,
            TailorResumeOutput,
        ),
        "generate_cover_letter": (
            generate_cover_letter,
            GenerateCoverLetterInput,
            GenerateCoverLetterOutput,
        ),
    }
    return {
        name: ToolSpec(
            name=name,
            func=func,
            input_model=input_model,
            output_model=output_model,
            description=f"Contract test tool: {name}",
        )
        for name, (func, input_model, output_model) in contracts.items()
    }


def test_memory_fact_is_used_same_run_only_for_relevant_top_three(
    tmp_path: Path,
) -> None:
    """One learned fact becomes evidence in every Top-3 revision immediately."""

    memory_file = tmp_path / "memory.json"
    app = build_agent_graph(
        tools=_contract_registry(tmp_path / "outputs"),
        checkpointer=create_memory_checkpointer(),
        tracer=TraceManager(enabled=False),
    )
    state = create_initial_state(
        thread_id="thread-contract",
        run_id="run-contract",
        memory_file=str(memory_file),
    )

    first = invoke_new_run(app, state)
    assert first["__interrupt__"]
    rejected = first["top_3_job_ids"][0]
    feedback = {
        job_id: {"decision": "approve", "comment": ""}
        for job_id in first["top_3_job_ids"]
    }
    feedback[rejected] = {
        "decision": "reject",
        "comment": "Add LangGraph. I have used it in previous projects.",
    }

    revised = resume_run(app, state["thread_id"], feedback)
    revision_calls = [
        decision
        for decision in revised["agent_decisions"]
        if decision["selected_tool"] == "tailor_resume"
        and decision["arguments"].get("revision_feedback")
    ]

    assert revised["status"] == "COMPLETED"
    assert not revised.get("__interrupt__")
    expected = set(revised["top_3_job_ids"][:2])
    assert {call["arguments"]["job"]["job_id"] for call in revision_calls} == expected
    assert all(
        call["arguments"]["source_resume_tex_path"] != revised["resume_path"]
        for call in revision_calls
    )
    assert all(
        any(
            item["evidence_id"].startswith("mem-")
            for item in call["arguments"]["candidate_evidence"]
        )
        for call in revision_calls
    )
    assert set(revised["review_history"][0]["actions_taken"]) == expected

    assert len(revised["cover_letter_results"]) == 3


def test_llm_owns_tool_intent_but_not_candidate_evidence(
    tmp_path: Path, monkeypatch
) -> None:
    """The model selects a tool while Python builds its validated arguments."""

    captured_messages: list = []

    class FakeStructuredModel:
        def invoke(self, messages):
            captured_messages.extend(messages)
            return AgentIntent(
                phase="FILTER",
                selected_tool="filter_jobs",
                decision_summary="Filtering is the next required evidence-safe step.",
            )

    class FakeChatOpenAI:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        def with_structured_output(self, schema):
            assert schema is AgentIntent
            return FakeStructuredModel()

    monkeypatch.setattr(langchain_openai, "ChatOpenAI", FakeChatOpenAI)
    jobs = load_jobs_csv("data/jobs.csv")
    profile = load_candidate_profile("data/preferences.yaml")
    controller = SingleAgentController(
        _contract_registry(tmp_path / "outputs"),
        model_name="contract-test-model",
        enable_llm=True,
    )

    decision = controller.decide(
        {
            "phase": "FILTER",
            "jobs": [job.model_dump() for job in jobs],
            "candidate_profile": profile.model_dump(),
        }
    )

    assert decision.selected_tool == "filter_jobs"
    assert decision.decision_source == "llm"
    assert {action["tool_name"] for action in decision.available_actions} == {
        "filter_jobs"
    }
    assert len(decision.arguments["jobs"]) == len(jobs)
    assert "job description" not in captured_messages[-1][1].casefold()


def test_llm_can_choose_tool_and_target_when_multiple_actions_are_valid(
    tmp_path: Path, monkeypatch
) -> None:
    """Prove the controller does not force the deterministic first action."""

    captured_messages: list = []
    jobs = load_jobs_csv("data/jobs.csv")[:3]
    profile = load_candidate_profile("data/preferences.yaml")
    chosen_job_id = jobs[0].job_id
    model_intent = [
        AgentIntent(
            phase="FIT_ANALYSIS",
            selected_tool="tailor_resume",
            target_job_id=chosen_job_id,
            decision_summary="Tailor the analyzed job before analyzing another.",
        )
    ]

    class FakeStructuredModel:
        def invoke(self, messages):
            captured_messages.extend(messages)
            return model_intent[0]

    class FakeChatOpenAI:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        def with_structured_output(self, schema):
            assert schema is AgentIntent
            return FakeStructuredModel()

    monkeypatch.setattr(langchain_openai, "ChatOpenAI", FakeChatOpenAI)
    tracer = TraceManager(enabled=False)
    tracer.start_run("run-llm-choice", "thread-llm-choice")
    controller = SingleAgentController(
        _contract_registry(tmp_path / "outputs"),
        model_name="contract-test-model",
        enable_llm=True,
        tracer=tracer,
    )
    state = {
        "phase": "FIT_ANALYSIS",
        "jobs": [job.model_dump() for job in jobs],
        "ranked_jobs": [{"job": job.model_dump()} for job in jobs],
        "top_3_job_ids": [job.job_id for job in jobs],
        "fit_analyses": {
            chosen_job_id: FitAnalysisOutput(job_id=chosen_job_id).model_dump()
        },
        "tailoring_results": {},
        "pending_revision_job_ids": [],
        "candidate_profile": profile.model_dump(),
        "portfolio": load_portfolio("data/portfolio.txt").model_dump(),
        "memory_facts": [],
        "resume_path": "data/resume.tex",
    }

    decision = controller.decide(state)

    assert decision.selected_tool == "tailor_resume"
    assert decision.target_job_id == chosen_job_id
    assert decision.decision_source == "llm"
    assert {action["tool_name"] for action in decision.available_actions} == {
        "analyze_fit",
        "tailor_resume",
    }
    assert decision.unresolved_requirements
    prompt = captured_messages[-1][1]
    assert "analyze_fit" in prompt
    assert "tailor_resume" in prompt
    generations = [
        event for event in tracer.events if event.observation_type == "GENERATION"
    ]
    assert len(generations) == 1
    assert generations[0].output["selected_tool"] == "tailor_resume"
    assert generations[0].output["target_job_id"] == chosen_job_id

    other_job_id = jobs[1].job_id
    model_intent[0] = AgentIntent(
        phase="FIT_ANALYSIS",
        selected_tool="analyze_fit",
        target_job_id=other_job_id,
        decision_summary="Analyze another eligible Top-3 job first.",
    )
    alternative = controller.decide(state)

    assert alternative.selected_tool == "analyze_fit"
    assert alternative.target_job_id == other_job_id
    assert alternative.decision_source == "llm"


def test_invalid_model_action_is_blocked_by_prerequisite_guard(
    tmp_path: Path, monkeypatch
) -> None:
    jobs = load_jobs_csv("data/jobs.csv")[:3]
    profile = load_candidate_profile("data/preferences.yaml")

    class FakeStructuredModel:
        def invoke(self, messages):
            return AgentIntent(
                phase="FIT_ANALYSIS",
                selected_tool="tailor_resume",
                target_job_id=jobs[0].job_id,
                decision_summary="Attempt to tailor before fit analysis.",
            )

    class FakeChatOpenAI:
        def __init__(self, **kwargs) -> None:
            pass

        def with_structured_output(self, schema):
            return FakeStructuredModel()

    monkeypatch.setattr(langchain_openai, "ChatOpenAI", FakeChatOpenAI)
    controller = SingleAgentController(
        _contract_registry(tmp_path / "outputs"),
        model_name="contract-test-model",
        enable_llm=True,
    )
    state = {
        "phase": "FIT_ANALYSIS",
        "jobs": [job.model_dump() for job in jobs],
        "ranked_jobs": [{"job": job.model_dump()} for job in jobs],
        "top_3_job_ids": [job.job_id for job in jobs],
        "fit_analyses": {},
        "tailoring_results": {},
        "candidate_profile": profile.model_dump(),
        "portfolio": load_portfolio("data/portfolio.txt").model_dump(),
        "memory_facts": [],
        "resume_path": "data/resume.tex",
    }

    with pytest.raises(Exception, match="unavailable action"):
        controller._llm_decide(state)


def test_workflow_stops_with_actionable_error_when_fewer_than_three_jobs_survive(
    tmp_path: Path,
) -> None:
    registry = _contract_registry(tmp_path / "outputs")

    def keep_only_two(value: FilterJobsInput) -> FilterJobsOutput:
        return FilterJobsOutput(
            accepted_jobs=value.jobs[:2],
            rejected_jobs=[],
        )

    original = registry["filter_jobs"]
    registry["filter_jobs"] = ToolSpec(
        name=original.name,
        func=keep_only_two,
        input_model=original.input_model,
        output_model=original.output_model,
        description=original.description,
    )
    app = build_agent_graph(
        tools=registry,
        checkpointer=create_memory_checkpointer(),
        tracer=TraceManager(enabled=False),
    )
    state = create_initial_state(
        thread_id="thread-too-few",
        run_id="run-too-few",
        memory_file=str(tmp_path / "memory.json"),
    )

    result = invoke_new_run(app, state)

    assert result["status"] == "FAILED"
    assert result["phase"] == "ERROR"
    assert result["top_3_job_ids"] == []
    assert "accepted only 2" in result["errors"][-1]["message"]
    assert "Add more job inputs" in result["errors"][-1]["message"]


def test_review_interrupt_resumes_after_process_restart_with_sqlite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "outputs"))
    db_path = tmp_path / "outputs" / "checkpoints.sqlite"
    memory_file = tmp_path / "outputs" / "memory.json"
    thread_id = "thread-sqlite-restart"
    registry = _contract_registry(tmp_path / "outputs")

    first_checkpointer, first_context = create_sqlite_checkpointer(db_path)
    try:
        first_app = build_agent_graph(
            tools=registry,
            checkpointer=first_checkpointer,
            tracer=TraceManager(enabled=False),
        )
        waiting = invoke_new_run(
            first_app,
            create_initial_state(
                thread_id=thread_id,
                run_id="run-sqlite-restart",
                memory_file=str(memory_file),
            ),
        )
        payload = waiting["__interrupt__"][0].value
        original_trace_id = waiting["trace_id"]
        original_review_parent = waiting["review_trace_parent_id"]
    finally:
        first_context.__exit__(None, None, None)

    second_tracer = TraceManager(enabled=False)
    second_checkpointer, second_context = create_sqlite_checkpointer(db_path)
    try:
        second_app = build_agent_graph(
            tools=registry,
            checkpointer=second_checkpointer,
            tracer=second_tracer,
        )
        feedback = {
            job_id: {"decision": "approve", "comment": ""}
            for job_id in payload["resumes"]
        }
        rejected = next(iter(payload["resumes"]))
        feedback[rejected] = {
            "decision": "reject",
            "comment": "Make the two experience bullets shorter.",
        }
        final = resume_run(second_app, thread_id, feedback)
    finally:
        second_context.__exit__(None, None, None)

    assert final["status"] == "COMPLETED"
    assert final["trace_id"] == original_trace_id
    assert final["review_trace_parent_id"] == original_review_parent
    assert len(final["review_history"]) == 1
    assert not final.get("__interrupt__")
    feedback_event = next(
        event
        for event in second_tracer.events
        if event.name == "human_review_feedback"
    )
    assert feedback_event.parent_observation_id == original_review_parent


def test_graph_executes_model_selected_eligible_tool_without_substitution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "outputs"))
    jobs = load_jobs_csv("data/jobs.csv")[:3]
    profile = load_candidate_profile("data/preferences.yaml")
    portfolio = load_portfolio("data/portfolio.txt")
    selected_job_id = jobs[0].job_id

    class ModelFirstController:
        def __init__(self, registry, tracer=None) -> None:
            self.policy = SingleAgentController(
                registry,
                model_name="model-selected-test",
                enable_llm=False,
                tracer=tracer,
            )
            self.model_name = "model-selected-test"
            self.first = True

        def decide(self, state):
            if self.first:
                self.first = False
                actions = self.policy._allowed_actions(state)
                decision = self.policy._decision_for_action(
                    state, "tailor_resume", selected_job_id
                )
                return decision.model_copy(
                    update={
                        "decision_source": "llm",
                        "decision_summary": "Tailor the eligible analyzed job now.",
                        "available_actions": actions,
                    }
                )
            return self.policy.decide(state)

    monkeypatch.setattr(graph_module, "SingleAgentController", ModelFirstController)
    state = create_initial_state(
        thread_id="thread-model-execution",
        run_id="run-model-execution",
        memory_file=str(tmp_path / "memory.json"),
    )
    state.update(
        {
            "phase": "FIT_ANALYSIS",
            "status": "RUNNING",
            "jobs": [job.model_dump() for job in jobs],
            "candidate_profile": profile.model_dump(),
            "portfolio": portfolio.model_dump(),
            "resume_path": "data/resume.tex",
            "ranked_jobs": [
                {"job": job.model_dump(), "score": 90 - index, "rationale": "test"}
                for index, job in enumerate(jobs)
            ],
            "top_3_job_ids": [job.job_id for job in jobs],
            "fit_analyses": {
                selected_job_id: FitAnalysisOutput(
                    job_id=selected_job_id
                ).model_dump()
            },
        }
    )
    app = build_agent_graph(
        tools=_contract_registry(tmp_path / "outputs"),
        checkpointer=create_memory_checkpointer(),
        tracer=TraceManager(enabled=False),
    )

    waiting = invoke_new_run(app, state)

    assert waiting["agent_decisions"][0]["decision_source"] == "llm"
    assert waiting["agent_decisions"][0]["selected_tool"] == "tailor_resume"
    assert waiting["agent_decisions"][0]["target_job_id"] == selected_job_id
    assert waiting["tool_history"][0]["tool"] == "tailor_resume"
    assert (
        waiting["tool_history"][0]["output"]["job_id"] == selected_job_id
    )
