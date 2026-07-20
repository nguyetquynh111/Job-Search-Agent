"""Human-review schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from src.schemas.common import StrictBaseModel


class ReviewDecision(StrictBaseModel):
    """Decision submitted by the reviewer for one resume."""

    decision: Literal["approve", "reject"]
    comment: str = ""


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
    fit_analysis: dict
    change_log: list[dict]
    resume_pdf_path: str


class ReviewInterruptPayload(StrictBaseModel):
    """Payload sent to Streamlit by the human review interrupt."""

    review_round: int
    max_revision_rounds: int
    resumes: dict[str, ReviewResumePayload]


class ReviewHistoryEntry(StrictBaseModel):
    """Persisted review record for one round."""

    review_round: int
    decisions: dict[str, ReviewDecision]
    rejected_job_ids: list[str] = Field(default_factory=list)
