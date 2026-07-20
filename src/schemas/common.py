"""Common schema primitives shared across tools and graph state."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class StrictBaseModel(BaseModel):
    """Base model that rejects unexpected fields at integration boundaries."""

    model_config = ConfigDict(extra="forbid")


class EvidenceItem(StrictBaseModel):
    """A source-backed evidence item that tools may cite by ID."""

    evidence_id: str
    source: str
    text: str
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CandidatePreferences(StrictBaseModel):
    """Candidate job preferences used by the filtering tool."""

    target_titles: list[str] = Field(default_factory=list)
    locations: list[str] = Field(default_factory=list)
    remote: bool = True
    job_types: list[str] = Field(default_factory=lambda: ["full-time"])
    min_salary: int | None = None
    excluded_keywords: list[str] = Field(default_factory=list)


class CandidateProfile(StrictBaseModel):
    """Candidate profile loaded from YAML or JSON input."""

    candidate_id: str
    name: str
    email: str | None = None
    preferences: CandidatePreferences = Field(default_factory=CandidatePreferences)
    skills: list[str] = Field(default_factory=list)
    education: list[str] = Field(default_factory=list)
    experience: list[str] = Field(default_factory=list)
    resume_evidence: list[EvidenceItem] = Field(default_factory=list)
    portfolio_evidence: list[EvidenceItem] = Field(default_factory=list)


class PortfolioProject(StrictBaseModel):
    """Portfolio project available for resume tailoring decisions."""

    project_id: str
    name: str
    description: str
    technologies: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)


class Portfolio(StrictBaseModel):
    """Candidate portfolio file."""

    projects: list[PortfolioProject] = Field(default_factory=list)
    evidence_items: list[EvidenceItem] = Field(default_factory=list)


class EvidenceClaim(StrictBaseModel):
    """Evidence-supported statement produced by fit analysis."""

    claim: str
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.8, ge=0, le=1)
    notes: str | None = None


class ProjectSwap(StrictBaseModel):
    """A recommended portfolio/resume project substitution."""

    remove_project: str | None = None
    add_project: str
    rationale: str
    evidence_ids: list[str] = Field(default_factory=list)


class ChangeLogEntry(StrictBaseModel):
    """Resume or cover-letter artifact change log entry."""

    change_id: str
    section: str
    description: str
    evidence_ids: list[str] = Field(default_factory=list)
