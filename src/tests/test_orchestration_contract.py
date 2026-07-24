"""Single-agent workflow tests independent of unfinished business tools."""

from __future__ import annotations

from pathlib import Path

import langchain_openai

from src.agent.controller import AgentIntent, SingleAgentController
from src.agent.graph import (
    build_agent_graph,
    create_memory_checkpointer,
    invoke_new_run,
    resume_run,
)
from src.agent.state import create_initial_state
from src.data_loader import load_candidate_profile, load_jobs_csv
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


def _contract_registry() -> dict[str, ToolSpec]:
    """Return predictable contract-valid tools for orchestration testing."""

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
        return TailorResumeOutput(
            job_id=value.job.job_id,
            status="OK",
            output_tex_path=f"outputs/{value.job.job_id}/resume.tex",
            output_pdf_path=f"outputs/{value.job.job_id}/resume.pdf",
            page_count=1,
            change_log=[
                ChangeLogEntry(
                    change_id=f"change-{value.job.job_id}",
                    section="skills",
                    description=value.revision_feedback or "initial tailoring",
                    evidence_ids=[
                        item.evidence_id for item in value.candidate_evidence[-2:]
                    ],
                )
            ],
        )

    def generate_cover_letter(
        value: GenerateCoverLetterInput,
    ) -> GenerateCoverLetterOutput:
        return GenerateCoverLetterOutput(
            job_id=value.job.job_id,
            output_tex_path=f"outputs/{value.job.job_id}/cover-letter.tex",
            output_pdf_path=f"outputs/{value.job.job_id}/cover-letter.pdf",
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


def test_memory_fact_is_revised_across_all_top_three(
    tmp_path: Path,
) -> None:
    """One learned fact becomes evidence in every Top-3 revision immediately."""

    memory_file = tmp_path / "memory.json"
    app = build_agent_graph(
        tools=_contract_registry(),
        checkpointer=create_memory_checkpointer(),
        tracer=TraceManager(enabled=False),
    )
    state = create_initial_state(
        thread_id="thread-contract",
        run_id="run-contract",
        memory_file=str(memory_file),
    )

    first = invoke_new_run(app, state)
    first_payload = first["__interrupt__"][0].value
    rejected = first["top_3_job_ids"][0]
    feedback = {
        job_id: {"decision": "approve", "comment": ""}
        for job_id in first["top_3_job_ids"]
    }
    feedback[rejected] = {
        "decision": "reject",
        "comment": "Add GraphQL. I have used it in previous projects.",
    }

    revised = resume_run(app, state["thread_id"], feedback)
    revision_calls = [
        decision
        for decision in revised["agent_decisions"]
        if decision["selected_tool"] == "tailor_resume"
        and decision["arguments"].get("revision_feedback")
    ]

    assert revised["status"] == "WAITING_FOR_REVIEW"
    assert {call["arguments"]["job"]["job_id"] for call in revision_calls} == set(
        revised["top_3_job_ids"]
    )
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
    assert set(revised["review_history"][0]["actions_taken"]) == set(
        revised["top_3_job_ids"]
    )

    approvals = {
        job_id: {"decision": "approve", "comment": ""}
        for job_id in revised["top_3_job_ids"]
    }
    completed = resume_run(app, state["thread_id"], approvals)

    assert completed["status"] == "COMPLETED"
    assert len(completed["cover_letter_results"]) == 3


def test_llm_owns_tool_intent_but_not_candidate_evidence(monkeypatch) -> None:
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
        _contract_registry(),
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
    assert len(decision.arguments["jobs"]) == len(jobs)
    assert "job description" not in captured_messages[-1][1].casefold()
