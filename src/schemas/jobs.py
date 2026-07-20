"""Job schemas."""

from __future__ import annotations

from pydantic import Field

from src.schemas.common import StrictBaseModel


class Job(StrictBaseModel):
    """Normalized job posting."""

    job_id: str
    title: str
    company: str
    location: str = ""
    remote: bool = False
    description: str
    requirements: list[str] = Field(default_factory=list)
    salary_min: int | None = None
    salary_max: int | None = None
    source: str | None = None


class RejectedJob(StrictBaseModel):
    """Filtered-out job with one or more reasons."""

    job: Job
    reasons: list[str] = Field(min_length=1)
