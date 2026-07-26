"""Explicit five-tool job-search workflow with one human-review pause.

You are the only LLM agent in this workflow.
"""

from __future__ import annotations

import logging
from contextlib import AbstractContextManager
from enum import StrEnum
from pathlib import Path
from typing import Any, TypedDict
from uuid import uuid4

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel

from app.configuration import (
    DEFAULT_JOBS_PATH,
    DEFAULT_PORTFOLIO_PATH,
    DEFAULT_PROFILE_PATH,
    DEFAULT_RESUME_PATH,
    MAX_REVISION_ROUNDS,
    get_config,
    validate_runtime_requirements,
)
from src.domain import (
    CandidatePreferences,
    CandidateProfile,
    ChangeLogEntry,
    EvidenceClaim,
    EvidenceItem,
    ExperienceRequirement,
    Job,
    Portfolio,
    PortfolioProject,
    ProjectSwap,
    RejectedJob,
    ResumeData,
    StrictBaseModel,
    company_comparison_key,
    normalize_string_list,
    parse_experience_requirement,
)
from src.utils.evidence_validation import (
    evidence_supports_keyword,
    evidence_supports_project,
    evidence_supports_skill,
    evidence_supports_statement,
    job_evidence_supports_skill,
)
from src.utils.input_loading import (
    load_resume_evidence,
    load_candidate_profile,
    load_jobs_csv,
    load_portfolio,
    load_resume_data,
    load_text_path,
)
from src.utils.job_evidence import (
    build_job_evidence,
    job_evidence_id,
    job_skill_evidence_id,
)
from src.utils.latex import escape_latex, pdf_page_count, pdflatex_command, run_pdflatex
from src.utils.output_validation import MANDATORY_JOB_FILES, write_and_validate_outputs
from src.utils.skill_matching import canonicalize, category_members, skill_in_text
from src.review.human_review import build_review_payload, normalize_review_feedback
from src.review.memory import (
    JSONMemoryStore,
    MemoryFact,
    extract_memory_facts,
    memory_fact_to_evidence,
    validate_memory_facts,
)
from src.tools.cover_letter.cover_letter import (
    GenerateCoverLetterInput,
    run_cover_letter_tool,
)
from src.tools.filtering_scoring.filtering import FilterJobsInput, run_filtering_tool
from src.tools.filtering_scoring.scoring import ScoreJobsInput, run_scoring_tool
from src.tools.fit_analysis.fit_analysis import (
    AnalyzeFitInput,
    FitAnalysisOutput,
    build_source_labels,
    run_fit_analysis_tool,
    write_fit_analysis,
)
from src.tools.resume_tailoring.resume_tailoring import (
    TailorResumeInput,
    run_resume_tailoring_tool,
)
from src.tracing.langfuse import TraceManager

logger = logging.getLogger(__name__)

__all__ = [
    "AgentState",
    "CandidatePreferences",
    "CandidateProfile",
    "ChangeLogEntry",
    "DEFAULT_JOBS_PATH",
    "DEFAULT_PORTFOLIO_PATH",
    "DEFAULT_PROFILE_PATH",
    "DEFAULT_RESUME_PATH",
    "EvidenceClaim",
    "EvidenceItem",
    "ExperienceRequirement",
    "Job",
    "MANDATORY_JOB_FILES",
    "MAX_REVISION_ROUNDS",
    "Phase",
    "Portfolio",
    "PortfolioProject",
    "ProjectSwap",
    "RejectedJob",
    "ResumeData",
    "RunStatus",
    "StrictBaseModel",
    "ToolExecutionError",
    "_validate_artifact_output",
    "build_agent_graph",
    "build_job_evidence",
    "canonicalize",
    "category_members",
    "company_comparison_key",
    "create_initial_state",
    "create_memory_checkpointer",
    "create_sqlite_checkpointer",
    "escape_latex",
    "evidence_supports_keyword",
    "evidence_supports_project",
    "evidence_supports_skill",
    "evidence_supports_statement",
    "get_config",
    "invoke_new_run",
    "job_evidence_id",
    "job_evidence_supports_skill",
    "job_skill_evidence_id",
    "load_candidate_profile",
    "load_jobs_csv",
    "load_portfolio",
    "load_resume_data",
    "load_resume_evidence",
    "parse_experience_requirement",
    "pdflatex_command",
    "pdf_page_count",
    "resume_run",
    "run_pdflatex",
    "skill_in_text",
    "validate_runtime_requirements",
    "write_and_validate_outputs",
]


def _state_value(state: AgentState, key: str) -> Any:
    """Return a required graph-state value for the current workflow phase."""

    if key not in state:
        raise ToolExecutionError(f"Missing required state value: {key}")
    return state[key]


class Phase(StrEnum):
    """Workflow phases."""

    INITIALIZE = "INITIALIZE"
    FILTER = "FILTER"
    SCORE = "SCORE"
    FIT_ANALYSIS = "FIT_ANALYSIS"
    TAILOR = "TAILOR"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    COVER_LETTERS = "COVER_LETTERS"
    COMPLETE = "COMPLETE"
    ERROR = "ERROR"


class RunStatus(StrEnum):
    """Workflow status values."""

    CREATED = "CREATED"
    RUNNING = "RUNNING"
    WAITING_FOR_REVIEW = "WAITING_FOR_REVIEW"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    FAILED_REVIEW = "FAILED_REVIEW"


class AgentState(TypedDict, total=False):
    """Shared state persisted by LangGraph's checkpointer."""

    run_id: str
    thread_id: str
    phase: str
    status: str

    input_paths: dict[str, str]
    candidate_profile: dict[str, Any]
    jobs: list[dict[str, Any]]
    resume_path: str
    portfolio: dict[str, Any]
    memory_file: str
    memory_facts: list[dict[str, Any]]
    new_memory_fact_ids: list[str]
    memory_validation_failures: list[str]

    filtered_jobs: list[dict[str, Any]]
    rejected_jobs: list[dict[str, Any]]
    ranked_jobs: list[dict[str, Any]]
    top_3_job_ids: list[str]

    fit_analyses: dict[str, dict[str, Any]]
    fit_analysis_artifacts: dict[str, dict[str, str]]
    fit_analysis_refresh_job_ids: list[str]
    tailoring_results: dict[str, dict[str, Any]]
    review_decisions: dict[str, dict[str, Any]]
    approved_job_ids: list[str]

    revision_round: int
    pending_revision_job_ids: list[str]
    revision_round_job_ids: list[str]
    cover_letter_results: dict[str, dict[str, Any]]

    current_tool: str | None
    current_tool_input: dict[str, Any]
    tool_history: list[dict[str, Any]]
    agent_decisions: list[dict[str, Any]]
    review_history: list[dict[str, Any]]
    interrupt_payload: dict[str, Any]
    review_feedback: dict[str, Any]
    errors: list[dict[str, Any]]

    trace_id: str | None
    trace_url: str | None
    review_trace_parent_id: str | None
    revision_trace_parent_id: str | None
    langfuse_status: str


def create_initial_state(
    jobs_path: str = "data/jobs.csv",
    candidate_profile_path: str = "data/preferences.yaml",
    resume_path: str = "data/resume.tex",
    portfolio_path: str = "data/portfolio.txt",
    memory_file: str | None = None,
    run_id: str | None = None,
    thread_id: str | None = None,
) -> AgentState:
    """Create a new graph run state from input paths."""

    resolved_run_id = run_id or f"run-{uuid4().hex[:12]}"
    resolved_thread_id = thread_id or f"thread-{uuid4().hex[:12]}"
    resolved_memory_file = memory_file or str(get_config().memory_file)
    return AgentState(
        run_id=resolved_run_id,
        thread_id=resolved_thread_id,
        phase=Phase.INITIALIZE.value,
        status=RunStatus.CREATED.value,
        input_paths={
            "jobs_path": jobs_path,
            "candidate_profile_path": candidate_profile_path,
            "resume_path": resume_path,
            "portfolio_path": portfolio_path,
            "memory_file": resolved_memory_file,
        },
        memory_file=resolved_memory_file,
        memory_facts=[],
        new_memory_fact_ids=[],
        memory_validation_failures=[],
        filtered_jobs=[],
        rejected_jobs=[],
        ranked_jobs=[],
        top_3_job_ids=[],
        fit_analyses={},
        fit_analysis_artifacts={},
        fit_analysis_refresh_job_ids=[],
        tailoring_results={},
        review_decisions={},
        approved_job_ids=[],
        revision_round=0,
        pending_revision_job_ids=[],
        revision_round_job_ids=[],
        cover_letter_results={},
        current_tool=None,
        current_tool_input={},
        tool_history=[],
        agent_decisions=[],
        review_history=[],
        errors=[],
        trace_id=None,
        trace_url=None,
        review_trace_parent_id=None,
        revision_trace_parent_id=None,
        langfuse_status="not initialized",
    )


class ToolExecutionError(RuntimeError):
    """Raised when a tool returns an invalid or incomplete artifact."""


def build_agent_graph(
    *,
    checkpointer: Any | None = None,
    tracer: TraceManager | None = None,
) -> Any:
    """Build the explicit workflow; tools are imported directly, never registered."""

    trace_manager = tracer or TraceManager(enabled=False)
    graph = StateGraph(AgentState)

    def initialize(state: AgentState) -> AgentState:
        """Load inputs and memory once before the tool sequence starts."""

        if state.get("candidate_profile") and state.get("jobs"):
            return {}
        paths = state.get("input_paths", {})
        jobs = load_jobs_csv(paths.get("jobs_path", "data/jobs.csv"))
        profile = load_candidate_profile(
            paths.get("candidate_profile_path", "data/preferences.yaml")
        )
        resume_path = load_text_path(
            paths.get("resume_path", "data/resume.tex"), "Resume"
        )
        resume = load_resume_data(resume_path)
        portfolio = load_portfolio(paths.get("portfolio_path", "data/portfolio.txt"))
        profile = profile.model_copy(
            update={
                "resume_content": resume.plain_text,
                "skills": normalize_string_list([*profile.skills, *resume.skills]),
                "education": normalize_string_list(
                    [*profile.education, *resume.education]
                ),
                "experience": normalize_string_list(
                    [*profile.experience, *resume.experience]
                ),
                "resume_projects": normalize_string_list(
                    [*profile.resume_projects, *resume.projects]
                ),
                "resume_evidence": [
                    *profile.resume_evidence,
                    *resume.evidence_items,
                ],
            }
        )
        memory_file = paths.get(
            "memory_file",
            state.get("memory_file", str(get_config().memory_file)),
        )
        memory_facts = JSONMemoryStore(memory_file).load()
        trace_id = trace_manager.start_run(
            run_id=_state_value(state, "run_id"),
            session_id=_state_value(state, "thread_id"),
            metadata={"status": RunStatus.RUNNING.value},
            input={"job_count": len(jobs)},
        )
        trace_manager.register_personal_data(profile.name, profile.email)
        return {
            "phase": Phase.FILTER.value,
            "status": RunStatus.RUNNING.value,
            "jobs": [job.model_dump() for job in jobs],
            "candidate_profile": profile.model_dump(),
            "resume_path": resume_path,
            "portfolio": portfolio.model_dump(),
            "memory_file": memory_file,
            "memory_facts": [fact.model_dump() for fact in memory_facts],
            "trace_id": trace_id,
            "trace_url": trace_manager.trace_url,
            "langfuse_status": trace_manager.status_message,
        }

    def run_pre_review_tools(state: AgentState) -> AgentState:
        """Run filtering, scoring, fit analysis, and resume tailoring in order."""

        profile = CandidateProfile.model_validate(
            _state_value(state, "candidate_profile")
        )
        jobs = [Job.model_validate(job) for job in _state_value(state, "jobs")]
        portfolio = Portfolio.model_validate(_state_value(state, "portfolio"))
        candidate_evidence = _candidate_evidence(state, profile, portfolio)
        history = list(state.get("tool_history", []))

        filtering_input = FilterJobsInput(
            jobs=jobs,
            preferences=profile.preferences,
        )
        filtering = run_filtering_tool(filtering_input, tracer=trace_manager)
        history.append(_history("run_filtering_tool", filtering))
        if len(filtering.accepted_jobs) < 3:
            raise ToolExecutionError(
                "Filtering must accept at least three jobs before scoring."
            )

        scoring_input = ScoreJobsInput(
            jobs=filtering.accepted_jobs,
            candidate_profile=profile,
            resume_evidence=profile.resume_evidence,
            master_skill_evidence=profile.master_skill_evidence,
            portfolio_evidence=[
                *portfolio.evidence_items,
                *profile.portfolio_evidence,
            ],
            memory_evidence=_memory_evidence(state),
        )
        scoring = run_scoring_tool(scoring_input, tracer=trace_manager)
        history.append(_history("run_scoring_tool", scoring))
        if len(scoring.top_3_job_ids) != 3:
            raise ToolExecutionError("Scoring must select exactly three jobs.")

        jobs_by_id = {job.job_id: job for job in jobs}
        analyses: dict[str, dict[str, Any]] = {}
        analysis_artifacts: dict[str, dict[str, str]] = {}
        for job_id in scoring.top_3_job_ids:
            job = jobs_by_id[job_id]
            fit_input = AnalyzeFitInput(
                job=job,
                candidate_profile=profile,
                evidence_items=candidate_evidence,
                job_evidence=build_job_evidence(job),
                current_resume_projects=profile.resume_projects,
                portfolio_projects=portfolio.projects,
            )
            fit = run_fit_analysis_tool(fit_input, tracer=trace_manager)
            analyses[job_id] = fit.model_dump()
            history.append(_history("run_fit_analysis_tool", fit))
            markdown_path, json_path = write_fit_analysis(
                fit,
                job,
                source_labels=build_source_labels(fit_input),
            )
            analysis_artifacts[job_id] = {
                "markdown_path": str(markdown_path),
                "json_path": str(json_path),
            }

        tailoring: dict[str, dict[str, Any]] = {}
        for job_id in scoring.top_3_job_ids:
            job = jobs_by_id[job_id]
            resume_input = TailorResumeInput(
                job=job,
                fit_analysis=FitAnalysisOutput.model_validate(analyses[job_id]),
                source_resume_tex_path=_state_value(state, "resume_path"),
                candidate_evidence=candidate_evidence,
                job_evidence=build_job_evidence(job),
            )
            resume = run_resume_tailoring_tool(resume_input, tracer=trace_manager)
            _validate_artifact_output("run_resume_tailoring_tool", resume)
            tailoring[job_id] = resume.model_dump()
            history.append(_history("run_resume_tailoring_tool", resume))

        return {
            "phase": Phase.HUMAN_REVIEW.value,
            "status": RunStatus.WAITING_FOR_REVIEW.value,
            "filtered_jobs": [job.model_dump() for job in filtering.accepted_jobs],
            "rejected_jobs": [
                rejected.model_dump() for rejected in filtering.rejected_jobs
            ],
            "ranked_jobs": [job.model_dump() for job in scoring.ranked_jobs],
            "top_3_job_ids": scoring.top_3_job_ids,
            "fit_analyses": analyses,
            "fit_analysis_artifacts": analysis_artifacts,
            "tailoring_results": tailoring,
            "tool_history": history,
        }

    def human_review(state: AgentState) -> AgentState:
        """Pause exactly once after all three resume drafts are ready."""

        if state.get("review_history"):
            raise ToolExecutionError("The workflow permits only one review pause.")
        payload = build_review_payload(dict(state))
        if not any(
            event.name == "human_review_pause" for event in trace_manager.events
        ):
            span_id = trace_manager.start_span(
                "human_review_pause",
                input={"job_ids": list(payload.resumes)},
            )
            trace_manager.end_span(span_id)
        review_feedback = interrupt(payload.model_dump())
        return {
            "interrupt_payload": payload.model_dump(),
            "review_feedback": review_feedback,
            "status": RunStatus.RUNNING.value,
        }

    def run_post_review_tools(state: AgentState) -> AgentState:
        """Apply reviewed revisions, then generate cover letters."""

        top_job_ids = list(_state_value(state, "top_3_job_ids"))
        feedback = normalize_review_feedback(
            state.get("review_feedback", {}),
            expected_job_ids=top_job_ids,
        )
        decisions = {
            job_id: decision.model_dump()
            for job_id, decision in feedback.decisions.items()
        }
        rejected = [
            job_id
            for job_id, decision in feedback.decisions.items()
            if decision.decision == "reject"
        ]
        memory_facts, new_fact_ids, memory_failures = _store_review_memory(
            state, decisions
        )
        state_with_memory: AgentState = {
            **state,
            "memory_facts": memory_facts,
        }
        profile = CandidateProfile.model_validate(
            _state_value(state, "candidate_profile")
        )
        portfolio = Portfolio.model_validate(_state_value(state, "portfolio"))
        candidate_evidence = _candidate_evidence(state_with_memory, profile, portfolio)
        jobs = {
            item["job_id"]: Job.model_validate(item)
            for item in _state_value(state, "jobs")
        }
        analyses = dict(_state_value(state, "fit_analyses"))
        tailoring = dict(_state_value(state, "tailoring_results"))
        history = list(state.get("tool_history", []))
        actions: dict[str, dict[str, Any]] = {}

        for job_id in rejected:
            feedback_text = decisions[job_id]["comment"]
            latest = tailoring[job_id]
            revision_round = 0
            for revision_round in range(1, MAX_REVISION_ROUNDS + 1):
                resume_input = TailorResumeInput(
                    job=jobs[job_id],
                    fit_analysis=FitAnalysisOutput.model_validate(analyses[job_id]),
                    source_resume_tex_path=latest["output_tex_path"],
                    candidate_evidence=candidate_evidence,
                    job_evidence=build_job_evidence(jobs[job_id]),
                    revision_feedback=feedback_text,
                )
                revised = run_resume_tailoring_tool(
                    resume_input,
                    tracer=trace_manager,
                )
                _validate_artifact_output("run_resume_tailoring_tool", revised)
                latest = revised.model_dump()
                history.append(_history("run_resume_tailoring_tool", revised))
                if revised.revision_feedback_satisfied is True:
                    break
            tailoring[job_id] = latest
            actions[job_id] = {
                "revision_round": revision_round,
                "status": latest["status"],
                "change_log": latest["change_log"],
                "output_tex_path": latest["output_tex_path"],
                "output_pdf_path": latest["output_pdf_path"],
            }
            if latest.get("revision_feedback_satisfied") is not True:
                return {
                    "phase": Phase.HUMAN_REVIEW.value,
                    "status": RunStatus.FAILED_REVIEW.value,
                    "tailoring_results": tailoring,
                    "tool_history": history,
                    "errors": [
                        *state.get("errors", []),
                        {
                            "phase": Phase.HUMAN_REVIEW.value,
                            "type": "FailedReview",
                            "message": (
                                f"Feedback for {job_id} remained unsatisfied after "
                                f"{MAX_REVISION_ROUNDS} revision rounds."
                            ),
                        },
                    ],
                }

        cover_letters: dict[str, dict[str, Any]] = {}
        for job_id in top_job_ids:
            cover_input = GenerateCoverLetterInput(
                job=jobs[job_id],
                approved_resume_path=tailoring[job_id]["output_pdf_path"],
                candidate_evidence=candidate_evidence,
                job_evidence=build_job_evidence(jobs[job_id]),
            )
            letter = run_cover_letter_tool(cover_input, tracer=trace_manager)
            _validate_artifact_output("run_cover_letter_tool", letter)
            cover_letters[job_id] = letter.model_dump()
            history.append(_history("run_cover_letter_tool", letter))

        review_history = [
            {
                "review_round": 1,
                "decisions": decisions,
                "rejected_job_ids": rejected,
                "memory_writes": [
                    fact
                    for fact in memory_facts
                    if fact["fact_id"] in set(new_fact_ids)
                ],
                "actions_taken": actions,
            }
        ]
        trace_manager.update_run(
            metadata={"status": RunStatus.COMPLETED.value},
            output={"top_3_job_ids": top_job_ids},
        )
        trace_manager.flush()
        return {
            "phase": Phase.COMPLETE.value,
            "status": RunStatus.COMPLETED.value,
            "review_decisions": decisions,
            "approved_job_ids": top_job_ids,
            "revision_round": max(
                (action["revision_round"] for action in actions.values()),
                default=0,
            ),
            "tailoring_results": tailoring,
            "cover_letter_results": cover_letters,
            "memory_facts": memory_facts,
            "new_memory_fact_ids": new_fact_ids,
            "memory_validation_failures": memory_failures,
            "review_history": review_history,
            "tool_history": history,
            "trace_url": trace_manager.trace_url,
            "langfuse_status": trace_manager.status_message,
        }

    graph.add_node("initialize", initialize)
    graph.add_node("run_pre_review_tools", run_pre_review_tools)
    graph.add_node("human_review", human_review)
    graph.add_node("run_post_review_tools", run_post_review_tools)
    graph.add_edge(START, "initialize")
    graph.add_edge("initialize", "run_pre_review_tools")
    graph.add_edge("run_pre_review_tools", "human_review")
    graph.add_edge("human_review", "run_post_review_tools")
    graph.add_edge("run_post_review_tools", END)
    return graph.compile(checkpointer=checkpointer)


def create_sqlite_checkpointer(
    db_path: str | Path,
) -> tuple[Any, AbstractContextManager[Any]]:
    """Create a SQLite checkpointer and keep its context manager alive."""

    from langgraph.checkpoint.sqlite import SqliteSaver

    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    context = SqliteSaver.from_conn_string(str(path))
    return context.__enter__(), context


def create_memory_checkpointer() -> Any:
    """Create an in-memory checkpointer."""

    from langgraph.checkpoint.memory import InMemorySaver

    return InMemorySaver()


def invoke_new_run(
    app: Any,
    state: AgentState | None = None,
    thread_id: str | None = None,
) -> dict[str, Any]:
    """Invoke a new graph run until the human-review pause."""

    run_state = state or create_initial_state(thread_id=thread_id)
    config = {"configurable": {"thread_id": _state_value(run_state, "thread_id")}}
    return app.invoke(run_state, config=config)


def resume_run(app: Any, thread_id: str, feedback: dict[str, Any]) -> dict[str, Any]:
    """Resume the graph once with all review decisions."""

    config = {"configurable": {"thread_id": thread_id}}
    return app.invoke(Command(resume=feedback), config=config)


def _candidate_evidence(
    state: AgentState,
    profile: CandidateProfile,
    portfolio: Portfolio,
) -> list[EvidenceItem]:
    return [
        *profile.resume_evidence,
        *profile.master_skill_evidence,
        *portfolio.evidence_items,
        *profile.portfolio_evidence,
        *_memory_evidence(state),
    ]


def _memory_evidence(state: AgentState) -> list[EvidenceItem]:
    return [
        EvidenceItem.model_validate(memory_fact_to_evidence(fact))
        for raw in state.get("memory_facts", [])
        if (fact := MemoryFact.model_validate(raw)).active
    ]


def _store_review_memory(
    state: AgentState,
    decisions: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    comments = {
        job_id: decision["comment"]
        for job_id, decision in decisions.items()
        if decision["comment"].strip()
    }
    extracted = extract_memory_facts(comments, review_round=1)
    valid, failures = validate_memory_facts(extracted, comments)
    store = JSONMemoryStore(_state_value(state, "memory_file"))
    before = store.load()
    before_keys = {fact.deduplication_key for fact in before if fact.active}
    updated = store.append_many(valid)
    new_ids = [
        fact.fact_id
        for fact in updated
        if fact.active and fact.deduplication_key not in before_keys
    ]
    return [fact.model_dump() for fact in updated], new_ids, failures


def _history(tool_name: str, output: BaseModel) -> dict[str, Any]:
    payload = output.model_dump()
    return {
        "tool": tool_name,
        "status": payload.get("status", "OK"),
        "output": payload,
    }


def _validate_artifact_output(tool_name: str, output: BaseModel) -> None:
    """Reject missing, failed, unreadable, or non-one-page tool artifacts."""

    if tool_name not in {
        "run_resume_tailoring_tool",
        "run_cover_letter_tool",
    }:
        return
    payload = output.model_dump()
    errors = payload.get("errors", [])
    if errors:
        raise ToolExecutionError(
            f"{tool_name} reported errors: {'; '.join(map(str, errors))}"
        )
    if payload.get("status") not in {None, "OK"}:
        raise ToolExecutionError(
            f"{tool_name} returned status {payload.get('status')!r}."
        )
    pdf_path = payload.get("output_pdf_path")
    tex_path = payload.get("output_tex_path")
    if not pdf_path or not Path(pdf_path).is_file():
        raise ToolExecutionError(f"{tool_name} PDF does not exist: {pdf_path}")
    if not tex_path or not Path(tex_path).is_file():
        raise ToolExecutionError(f"{tool_name} TeX file does not exist: {tex_path}")
    actual_page_count = pdf_page_count(Path(pdf_path))
    if actual_page_count != 1:
        raise ToolExecutionError(
            f"{tool_name} output PDF has {actual_page_count} pages; expected 1."
        )
