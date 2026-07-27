"""Fit-analysis tool contract."""

from __future__ import annotations

from typing import Any

from pydantic import Field, field_validator

from src.domain import (
    CandidateProfile,
    EvidenceClaim,
    EvidenceItem,
    Job,
    PortfolioProject,
    ProjectSwap,
    StrictBaseModel,
)


class AnalyzeFitInput(StrictBaseModel):
    """Input for analyze_fit."""

    job: Job
    candidate_profile: CandidateProfile
    evidence_items: list[EvidenceItem] = Field(default_factory=list)
    job_evidence: list[EvidenceItem] = Field(default_factory=list)
    current_resume_projects: list[str] = Field(default_factory=list)
    portfolio_projects: list[PortfolioProject] = Field(default_factory=list)


# Project findings intentionally share the established evidence-claim schema.
ProjectAnalysisItem = EvidenceClaim


class FitAnalysisOutput(StrictBaseModel):
    """Evidence-backed fit analysis for a single job."""

    job_id: str
    relevant_experience: list[EvidenceClaim] = Field(default_factory=list)
    seniority: list[EvidenceClaim] = Field(default_factory=list)
    education: list[EvidenceClaim] = Field(default_factory=list)
    aligned_skills: list[EvidenceClaim] = Field(default_factory=list)
    evidenced_missing_skills: list[EvidenceClaim] = Field(default_factory=list)
    genuine_gaps: list[EvidenceClaim] = Field(default_factory=list)
    project_analysis: list[ProjectAnalysisItem] = Field(default_factory=list)
    project_swap: ProjectSwap | None = None
    validation_failures: list[str] = Field(default_factory=list)

    @field_validator("project_analysis", mode="before")
    @classmethod
    def normalize_project_analysis(cls, value: Any) -> Any:
        """Accept the common single-item LLM shape without weakening the contract."""

        if value is None:
            return []
        if isinstance(value, dict):
            return [value]
        return value
