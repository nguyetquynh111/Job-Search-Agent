"""Cover-letter generation tool contract."""

from __future__ import annotations

from pydantic import Field

from src.schemas.common import EvidenceItem, StrictBaseModel
from src.schemas.jobs import Job


class GenerateCoverLetterInput(StrictBaseModel):
    """Input for generate_cover_letter."""

    job: Job
    approved_resume_path: str
    candidate_evidence: list[EvidenceItem] = Field(default_factory=list)


class GenerateCoverLetterOutput(StrictBaseModel):
    """Output from generate_cover_letter."""

    job_id: str
    output_tex_path: str
    output_pdf_path: str
    page_count: int = Field(ge=1)
    evidence_used: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
