"""Resume-tailoring tool contract."""

from __future__ import annotations

from pydantic import Field

from src.schemas.common import ChangeLogEntry, EvidenceItem, StrictBaseModel
from src.schemas.fit_analysis import FitAnalysisOutput
from src.schemas.jobs import Job


class TailorResumeInput(StrictBaseModel):
    """Input for tailor_resume."""

    job: Job
    fit_analysis: FitAnalysisOutput
    source_resume_tex_path: str
    candidate_evidence: list[EvidenceItem] = Field(default_factory=list)
    job_evidence: list[EvidenceItem] = Field(default_factory=list)
    revision_feedback: str | None = None


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
