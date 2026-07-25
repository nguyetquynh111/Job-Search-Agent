"""Builders for fit-analysis tests (not collected as a test module)."""

from __future__ import annotations

from src.schemas.common import (
    CandidateProfile,
    CandidatePreferences,
    EvidenceItem,
    PortfolioProject,
)
from src.schemas.fit_analysis import AnalyzeFitInput
from src.schemas.jobs import Job


def make_evidence(
    evidence_id: str, source: str, text: str = "", tags: list[str] | None = None
) -> EvidenceItem:
    """Build an EvidenceItem for tests."""

    return EvidenceItem(
        evidence_id=evidence_id, source=source, text=text or evidence_id, tags=tags or []
    )


def make_job(
    job_id: str = "J1",
    required_skills: list[str] | None = None,
    years: int | float | None = None,
    title: str = "AI Engineer",
    description: str = "Build ML systems.",
) -> Job:
    """Build a Job for tests."""

    return Job(
        job_id=job_id,
        title=title,
        company="Acme",
        description=description,
        required_skills=required_skills or [],
        years_experience_required=years,
    )


def make_profile(
    skills: list[str] | None = None,
    years: int | float | None = 4,
    resume_projects: list[str] | None = None,
) -> CandidateProfile:
    """Build a CandidateProfile for tests."""

    return CandidateProfile(
        candidate_id="cand-1",
        name="Test Candidate",
        skills=skills or [],
        resume_projects=resume_projects or [],
        preferences=CandidatePreferences(years_of_experience=years),
    )


def make_portfolio_project(
    project_id: str,
    name: str,
    technologies: list[str] | None = None,
    domains: list[str] | None = None,
) -> PortfolioProject:
    """Build a PortfolioProject for tests."""

    return PortfolioProject(
        project_id=project_id,
        name=name,
        description=f"{name} description",
        technologies=technologies or [],
        domains=domains or [],
        evidence_ids=[f"portfolio-{project_id}"],
    )


def make_input(
    job: Job,
    profile: CandidateProfile,
    evidence_items: list[EvidenceItem] | None = None,
    current_resume_projects: list[str] | None = None,
    portfolio_projects: list[PortfolioProject] | None = None,
) -> AnalyzeFitInput:
    """Build an AnalyzeFitInput for tests."""

    return AnalyzeFitInput(
        job=job,
        candidate_profile=profile,
        evidence_items=evidence_items or [],
        current_resume_projects=current_resume_projects or [],
        portfolio_projects=portfolio_projects or [],
    )
