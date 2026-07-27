"""Human-review payloads and feedback normalization."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from src.config import MAX_REVISION_ROUNDS
from src.domain import StrictBaseModel


class RevisionLimitError(RuntimeError):
    """Raised when automatic revisions exceed the configured limit."""


def assert_can_revise(revision_round: int) -> None:
    """Reject attempts to exceed the two-round revision limit."""

    if revision_round >= MAX_REVISION_ROUNDS:
        raise RevisionLimitError(
            f"Maximum revision rounds exceeded: {MAX_REVISION_ROUNDS}"
        )


def next_revision_round(revision_round: int) -> int:
    """Return the next valid revision round."""

    assert_can_revise(revision_round)
    return revision_round + 1


class ReviewDecision(StrictBaseModel):
    """Decision submitted by the reviewer for one resume."""

    decision: Literal["approve", "reject"]
    comment: str = ""


class CombinedReviewSubmission(StrictBaseModel):
    """The single human decision submitted for the complete Top 3 batch."""

    action: Literal["request_revision", "approve"]
    reviewer_feedback: str = ""
    generated_memory: str = ""
    affected_job_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_submission(self) -> "CombinedReviewSubmission":
        self.reviewer_feedback = " ".join(self.reviewer_feedback.split())
        self.generated_memory = " ".join(self.generated_memory.split())
        self.affected_job_ids = list(dict.fromkeys(self.affected_job_ids))
        if self.action == "request_revision":
            if not self.reviewer_feedback:
                raise ValueError("Reviewer feedback is required to request a revision.")
            if not self.affected_job_ids:
                raise ValueError("Select at least one resume to revise.")
        if self.action == "approve" and not (
            self.reviewer_feedback or self.generated_memory
        ):
            raise ValueError(
                "Reviewer feedback or generated memory is required before approval."
            )
        return self


class ReviewFeedback(StrictBaseModel):
    """Review decisions for all resumes in the current interrupt payload."""

    decisions: dict[str, ReviewDecision]

    @model_validator(mode="after")
    def require_all_comments_for_rejections(self) -> "ReviewFeedback":
        """Require actionable comments when a resume is rejected."""

        missing = [
            job_id
            for job_id, decision in self.decisions.items()
            if decision.decision == "reject" and not decision.comment.strip()
        ]
        if missing:
            raise ValueError(f"Rejected resumes require comments: {missing}")
        return self


class ReviewResumePayload(StrictBaseModel):
    """One resume entry in a LangGraph interrupt payload."""

    job_title: str
    company: str
    score: float | None = None
    fit_analysis: dict
    change_log: list[dict[str, Any]] = Field(default_factory=list)
    project_swap: dict[str, Any] | None = None
    resume_pdf_path: str


class ReviewInterruptPayload(StrictBaseModel):
    """Payload sent to Streamlit by the human review interrupt."""

    review_round: int
    max_revision_rounds: int
    revision_round: int = 0
    is_initial_review: bool = True
    reviewer_feedback: str = ""
    generated_memory: str = ""
    resumes: dict[str, ReviewResumePayload]


class ReviewHistoryEntry(StrictBaseModel):
    """Persisted review record for one round."""

    review_round: int
    decisions: dict[str, ReviewDecision]
    rejected_job_ids: list[str] = Field(default_factory=list)
    memory_writes: list[dict[str, Any]] = Field(default_factory=list)
    actions_taken: dict[str, dict[str, Any]] = Field(default_factory=dict)


class ReviewSubmissionError(RuntimeError):
    """Raised when submitted review feedback is incomplete or invalid."""


def _assert_review_ready(
    top_3_job_ids: list[str],
    tailoring_results: dict[str, dict],
) -> None:
    if len(top_3_job_ids) != 3:
        raise ReviewSubmissionError(
            "Human review requires exactly three selected jobs."
        )
    missing = [job_id for job_id in top_3_job_ids if job_id not in tailoring_results]
    if missing:
        raise ReviewSubmissionError(
            f"Human review requires all three resume drafts. Missing: {missing}"
        )


def build_review_payload(state: dict) -> ReviewInterruptPayload:
    """Create one interrupt payload containing all three tailored resumes."""

    top_3_job_ids = list(state.get("top_3_job_ids", []))
    tailoring_results = dict(state.get("tailoring_results", {}))
    _assert_review_ready(top_3_job_ids, tailoring_results)
    jobs = {job["job_id"]: job for job in state.get("jobs", [])}
    scores = {
        str(item.get("job", {}).get("job_id")): item.get("score")
        for item in state.get("ranked_jobs", [])
        if isinstance(item, dict) and isinstance(item.get("job"), dict)
    }
    fit_analyses = dict(state.get("fit_analyses", {}))
    resumes: dict[str, ReviewResumePayload] = {}
    for job_id in top_3_job_ids:
        job = jobs[job_id]
        tailoring = tailoring_results[job_id]
        resumes[job_id] = ReviewResumePayload(
            job_title=job["title"],
            company=job["company"],
            score=scores.get(job_id),
            fit_analysis=fit_analyses.get(job_id, {}),
            change_log=list(tailoring.get("change_log", [])),
            project_swap=fit_analyses.get(job_id, {}).get("project_swap"),
            resume_pdf_path=tailoring["output_pdf_path"],
        )
    revision_round = int(state.get("revision_round", 0))
    return ReviewInterruptPayload(
        review_round=revision_round + 1,
        max_revision_rounds=MAX_REVISION_ROUNDS,
        revision_round=revision_round,
        is_initial_review=revision_round == 0,
        reviewer_feedback=str(state.get("reviewer_feedback", "")),
        generated_memory=str(state.get("generated_memory", "")),
        resumes=resumes,
    )


def normalize_review_feedback(
    raw_feedback: dict, expected_job_ids: list[str]
) -> ReviewFeedback:
    """Validate review feedback for every resume in the payload."""

    payload = (
        raw_feedback if "decisions" in raw_feedback else {"decisions": raw_feedback}
    )
    feedback = ReviewFeedback.model_validate(payload)
    expected = set(expected_job_ids)
    received = set(feedback.decisions)
    if expected != received:
        missing = sorted(expected - received)
        extra = sorted(received - expected)
        raise ReviewSubmissionError(
            f"Review must include decisions for all selected jobs. Missing={missing}, extra={extra}"
        )
    return feedback


def normalize_combined_review_submission(
    raw_submission: dict[str, Any],
    expected_job_ids: list[str],
) -> CombinedReviewSubmission:
    """Validate one combined review action against the selected Top 3."""

    submission = CombinedReviewSubmission.model_validate(raw_submission)
    expected = set(expected_job_ids)
    extra = set(submission.affected_job_ids) - expected
    if extra:
        raise ReviewSubmissionError(
            f"Revision references jobs outside the Top 3: {sorted(extra)}"
        )
    return submission
