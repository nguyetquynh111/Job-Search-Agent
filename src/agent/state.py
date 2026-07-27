"""LangGraph state and initial-state construction for the job-search agent."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, TypedDict
from uuid import uuid4

from src.agent.errors import ToolExecutionError
from src.config import get_config


class Phase(StrEnum):
    """Workflow phases."""

    INITIALIZE = "INITIALIZE"
    FILTER = "FILTER"
    SCORE = "SCORE"
    FIT_ANALYSIS = "FIT_ANALYSIS"
    TAILOR = "TAILOR"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    REVISION = "REVISION"
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
    reviewer_feedback: str
    generated_memory: str
    errors: list[dict[str, Any]]

    trace_id: str | None
    trace_url: str | None
    trace_public: bool
    trace_ingest_confirmed: bool
    observation_count: int
    trace_export_error: str | None
    trace_debug_status: str | None
    output_manifest: dict[str, Any]
    review_trace_parent_id: str | None
    revision_trace_parent_id: str | None
    langfuse_status: str


def state_value(state: AgentState, key: str) -> Any:
    """Return a required graph-state value for the current workflow phase."""

    if key not in state:
        raise ToolExecutionError(f"Missing required state value: {key}")
    return state[key]


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
    # Candidate memory is deliberately shared across runs. Run artifacts remain
    # isolated, but facts learned during review must be available at next startup.
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
        reviewer_feedback="",
        generated_memory="",
        errors=[],
        trace_id=None,
        trace_url=None,
        trace_public=False,
        trace_ingest_confirmed=False,
        observation_count=0,
        trace_export_error=None,
        trace_debug_status=None,
        output_manifest={},
        review_trace_parent_id=None,
        revision_trace_parent_id=None,
        langfuse_status="not initialized",
    )


__all__ = ["AgentState", "Phase", "RunStatus", "create_initial_state", "state_value"]
