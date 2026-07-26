"""Contracts for the resume-tailoring tool."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import Field

from src.domain import ChangeLogEntry, EvidenceItem, Job, StrictBaseModel
from src.tools.fit_analysis.fit_analysis import FitAnalysisOutput
from src.tools.resume_tailoring.latex_structure import TextRange


class TailorResumeInput(StrictBaseModel):
    """Input for tailor_resume."""

    job: Job
    fit_analysis: FitAnalysisOutput
    source_resume_tex_path: str
    candidate_evidence: list[EvidenceItem] = Field(default_factory=list)
    job_evidence: list[EvidenceItem] = Field(default_factory=list)
    revision_feedback: str | None = None
    run_id: str | None = None


class TailorResumeOutput(StrictBaseModel):
    """Output from tailor_resume."""

    job_id: str
    status: str
    output_tex_path: str
    output_pdf_path: str
    page_count: int = Field(ge=0)
    change_log: list[ChangeLogEntry] = Field(default_factory=list)
    revision_feedback_satisfied: bool | None = None
    revision_feedback_checks: list[str] = Field(default_factory=list)
    validation_failures: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class TailoringError(RuntimeError):
    """Raised when a resume cannot be tailored without breaking its contract."""


@dataclass(frozen=True)
class PortfolioRecord:
    """Portfolio fields needed to render a real project into the resume."""

    project_id: str
    name: str
    period: str
    technologies: list[str]
    domain: str
    summary: str
    evidence_id: str


@dataclass(frozen=True)
class SourceEdit:
    """A single authorized replacement in the original uploaded source."""

    target: TextRange
    replacement: str
    category: str
    location: str


__all__ = [
    "PortfolioRecord",
    "SourceEdit",
    "TailorResumeInput",
    "TailorResumeOutput",
    "TailoringError",
]
