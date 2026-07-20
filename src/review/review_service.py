"""Human-review payload construction and validation."""

from __future__ import annotations

from src.agent.phase_policy import MAX_REVISION_ROUNDS, assert_review_ready
from src.schemas.review import ReviewFeedback, ReviewInterruptPayload, ReviewResumePayload


class ReviewSubmissionError(RuntimeError):
    """Raised when submitted review feedback is incomplete or invalid."""


def build_review_payload(state: dict) -> ReviewInterruptPayload:
    """Create one interrupt payload containing all three tailored resumes."""

    top_3_job_ids = list(state.get("top_3_job_ids", []))
    tailoring_results = dict(state.get("tailoring_results", {}))
    assert_review_ready(top_3_job_ids, tailoring_results)
    jobs = {job["job_id"]: job for job in state.get("jobs", [])}
    fit_analyses = dict(state.get("fit_analyses", {}))
    resumes: dict[str, ReviewResumePayload] = {}
    for job_id in top_3_job_ids:
        job = jobs[job_id]
        tailoring = tailoring_results[job_id]
        resumes[job_id] = ReviewResumePayload(
            job_title=job["title"],
            company=job["company"],
            fit_analysis=fit_analyses.get(job_id, {}),
            change_log=tailoring.get("change_log", []),
            resume_pdf_path=tailoring["output_pdf_path"],
        )
    return ReviewInterruptPayload(
        review_round=int(state.get("revision_round", 0)) + 1,
        max_revision_rounds=MAX_REVISION_ROUNDS,
        resumes=resumes,
    )


def normalize_review_feedback(
    raw_feedback: dict, expected_job_ids: list[str]
) -> ReviewFeedback:
    """Validate review feedback for every resume in the payload."""

    payload = raw_feedback if "decisions" in raw_feedback else {"decisions": raw_feedback}
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
