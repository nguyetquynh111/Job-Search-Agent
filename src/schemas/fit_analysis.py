"""Fit-analysis tool contract."""

from __future__ import annotations

from pydantic import Field

from src.schemas.common import (
    CandidateProfile,
    EvidenceClaim,
    EvidenceItem,
    PortfolioProject,
    ProjectSwap,
    StrictBaseModel,
)
from src.schemas.jobs import Job


class AnalyzeFitInput(StrictBaseModel):
    """Input for analyze_fit."""

    job: Job
    candidate_profile: CandidateProfile
    evidence_items: list[EvidenceItem] = Field(default_factory=list)
    current_resume_projects: list[str] = Field(default_factory=list)
    portfolio_projects: list[PortfolioProject] = Field(default_factory=list)


class FitAnalysisOutput(StrictBaseModel):
    """Evidence-backed fit analysis for a single job."""

    job_id: str
    relevant_experience: list[EvidenceClaim] = Field(default_factory=list)
    seniority: list[EvidenceClaim] = Field(default_factory=list)
    education: list[EvidenceClaim] = Field(default_factory=list)
    aligned_skills: list[EvidenceClaim] = Field(default_factory=list)
    evidenced_missing_skills: list[EvidenceClaim] = Field(default_factory=list)
    genuine_gaps: list[EvidenceClaim] = Field(default_factory=list)
    project_analysis: list[EvidenceClaim] = Field(default_factory=list)
    project_swap: ProjectSwap | None = None
