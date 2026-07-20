"""LangGraph shared state definitions."""

from __future__ import annotations

from typing import Any, TypedDict
from uuid import uuid4

from src.agent.phase_policy import Phase, RunStatus
from src.config import get_config


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

    filtered_jobs: list[dict[str, Any]]
    rejected_jobs: list[dict[str, Any]]
    ranked_jobs: list[dict[str, Any]]
    top_3_job_ids: list[str]

    fit_analyses: dict[str, dict[str, Any]]
    tailoring_results: dict[str, dict[str, Any]]
    review_decisions: dict[str, dict[str, Any]]
    approved_job_ids: list[str]

    revision_round: int
    pending_revision_job_ids: list[str]
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
    langfuse_status: str


def create_initial_state(
    jobs_path: str = "data/demo_jobs.csv",
    candidate_profile_path: str = "data/demo_preferences.yaml",
    resume_path: str = "data/demo_resume.tex",
    portfolio_path: str = "data/demo_portfolio.txt",
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
        filtered_jobs=[],
        rejected_jobs=[],
        ranked_jobs=[],
        top_3_job_ids=[],
        fit_analyses={},
        tailoring_results={},
        review_decisions={},
        approved_job_ids=[],
        revision_round=0,
        pending_revision_job_ids=[],
        cover_letter_results={},
        current_tool=None,
        current_tool_input={},
        tool_history=[],
        agent_decisions=[],
        review_history=[],
        errors=[],
        trace_id=None,
        trace_url=None,
        langfuse_status="not initialized",
    )
