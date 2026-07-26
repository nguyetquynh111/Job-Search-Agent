"""TEMPORARY STUB -- NOT the real Resume Tailoring tool.

WHY THIS EXISTS
---------------
``src.tools.registry.load_tool_registry`` requires all five tools to import
before the app (and the Streamlit UI) will start. The real ``tailor_resume`` is
not implemented yet, so this placeholder lets the app boot and lets the
**filtering + scoring + Top-3** stages be demoed live in the browser.

This does NO real work: it edits no LaTeX, runs no ``pdflatex``, and produces no
PDF. It only returns a contract-valid ``TailorResumeOutput`` so the workflow can
proceed past the tailoring phase.

DO NOT SUBMIT THIS AS THE FINISHED TOOL. Replace it with the real implementation
(see ``src/tools/resume_tailoring/``).
"""

from __future__ import annotations

import logging

from src.schemas.common import ChangeLogEntry
from src.schemas.tailoring import TailorResumeInput, TailorResumeOutput

logger = logging.getLogger(__name__)


def tailor_resume(inp: TailorResumeInput) -> TailorResumeOutput:
    """STUB: return a placeholder tailoring result without touching any files."""

    logger.warning(
        "tailor_resume STUB invoked for %s -- no real resume was produced.",
        inp.job.job_id,
    )
    return TailorResumeOutput(
        job_id=inp.job.job_id,
        status="OK",
        output_tex_path=f"outputs/{inp.job.job_id}/resume.tex",
        output_pdf_path=f"outputs/{inp.job.job_id}/resume.pdf",
        page_count=1,
        change_log=[
            ChangeLogEntry(
                change_id=f"stub-{inp.job.job_id}",
                section="summary",
                description=(
                    "[STUB] Placeholder tailoring — real LaTeX editing not yet "
                    "implemented."
                ),
                evidence_ids=[],
            )
        ],
        errors=[],
    )
