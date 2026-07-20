"""Scoring tool contract."""

from __future__ import annotations

from pydantic import Field, model_validator

from src.schemas.common import CandidateProfile, EvidenceItem, StrictBaseModel
from src.schemas.jobs import Job


class ScoreJobsInput(StrictBaseModel):
    """Input for score_jobs."""

    jobs: list[Job]
    candidate_profile: CandidateProfile
    resume_evidence: list[EvidenceItem] = Field(default_factory=list)
    portfolio_evidence: list[EvidenceItem] = Field(default_factory=list)
    memory_evidence: list[EvidenceItem] = Field(default_factory=list)


class ScoredJob(StrictBaseModel):
    """A job and deterministic score returned by the scoring tool."""

    job: Job
    score: float = Field(ge=0, le=100)
    rationale: str
    evidence_ids: list[str] = Field(default_factory=list)


class ScoreJobsOutput(StrictBaseModel):
    """Output from score_jobs."""

    ranked_jobs: list[ScoredJob]
    top_3_job_ids: list[str]

    @model_validator(mode="after")
    def validate_top_three(self) -> "ScoreJobsOutput":
        """Require top_3_job_ids to reference ranked jobs and contain at most three IDs."""

        ranked_ids = {job.job.job_id for job in self.ranked_jobs}
        if len(self.top_3_job_ids) > 3:
            raise ValueError("top_3_job_ids cannot contain more than three jobs")
        missing = [job_id for job_id in self.top_3_job_ids if job_id not in ranked_ids]
        if missing:
            raise ValueError(f"top_3_job_ids reference unknown ranked jobs: {missing}")
        return self
