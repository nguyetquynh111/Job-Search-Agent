"""Filtering tool contract."""

from __future__ import annotations

from src.schemas.common import CandidatePreferences, StrictBaseModel
from src.schemas.jobs import Job, RejectedJob


class FilterJobsInput(StrictBaseModel):
    """Input for filter_jobs."""

    jobs: list[Job]
    preferences: CandidatePreferences


class FilterJobsOutput(StrictBaseModel):
    """Output from filter_jobs."""

    accepted_jobs: list[Job]
    rejected_jobs: list[RejectedJob]
