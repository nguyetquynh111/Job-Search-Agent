"""Workflow phase and tool policy enforcement."""

from __future__ import annotations

from enum import StrEnum

MAX_REVISION_ROUNDS = 2


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


PHASE_TOOL_POLICY: dict[str, list[str]] = {
    Phase.FILTER.value: ["filter_jobs"],
    Phase.SCORE.value: ["score_jobs"],
    Phase.FIT_ANALYSIS.value: ["analyze_fit"],
    Phase.TAILOR.value: ["tailor_resume"],
    Phase.HUMAN_REVIEW.value: [],
    Phase.COVER_LETTERS.value: ["generate_cover_letter"],
}


class WorkflowPolicyError(RuntimeError):
    """Raised when a workflow phase or ordering rule is violated."""


def assert_tool_allowed(phase: str, tool_name: str) -> None:
    """Raise if a tool is not model-visible in the current phase."""

    allowed = PHASE_TOOL_POLICY.get(phase, [])
    if tool_name not in allowed:
        raise WorkflowPolicyError(
            f"Tool {tool_name!r} is not allowed in phase {phase!r}. Allowed: {allowed}"
        )


def assert_scoring_complete(ranked_jobs: list[dict], top_3_job_ids: list[str]) -> None:
    """Require scoring and top-three selection before fit analysis."""

    if not ranked_jobs:
        raise WorkflowPolicyError("Fit analysis cannot run before score_jobs.")
    if len(top_3_job_ids) != 3:
        raise WorkflowPolicyError("Fit analysis requires exactly three selected jobs.")


def assert_top_three_ready(top_3_job_ids: list[str]) -> None:
    """Require top-three IDs before downstream per-job work."""

    if len(top_3_job_ids) != 3:
        raise WorkflowPolicyError("Exactly three jobs must be selected before this phase.")


def assert_review_ready(top_3_job_ids: list[str], tailoring_results: dict[str, dict]) -> None:
    """Require tailored resumes for all selected jobs before review."""

    assert_top_three_ready(top_3_job_ids)
    missing = [job_id for job_id in top_3_job_ids if job_id not in tailoring_results]
    if missing:
        raise WorkflowPolicyError(
            f"Human review cannot start until all three resumes exist. Missing: {missing}"
        )


def assert_can_revise(revision_round: int) -> None:
    """Require revision rounds to remain within the configured maximum."""

    if revision_round >= MAX_REVISION_ROUNDS:
        raise WorkflowPolicyError(
            f"Maximum revision rounds exceeded: {MAX_REVISION_ROUNDS}"
        )


def assert_cover_letters_allowed(
    top_3_job_ids: list[str], approved_job_ids: list[str]
) -> None:
    """Require all selected resumes to be approved before cover letters."""

    assert_top_three_ready(top_3_job_ids)
    missing = sorted(set(top_3_job_ids) - set(approved_job_ids))
    if missing:
        raise WorkflowPolicyError(
            f"Cover letters cannot run before all three resumes are approved. Missing: {missing}"
        )
