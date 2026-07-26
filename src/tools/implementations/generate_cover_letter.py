"""TEMPORARY STUB -- NOT the real Cover Letter tool.

WHY THIS EXISTS
---------------
A real ``generate_cover_letter`` implementation already exists, but it currently
lives in a mis-named folder -- ``src/tools/ implementations/`` (note the leading
space) -- so ``src.tools.registry`` (which imports from
``src.tools.implementations``) cannot load it. Until that file is moved into this
package and its ``pdflatex`` dependency is available, this placeholder lets the
app boot so the **filtering + scoring + Top-3** stages can be demoed live.

This does NO real work: it runs no ``pdflatex`` and produces no PDF. It only
returns a contract-valid ``GenerateCoverLetterOutput``.

DO NOT SUBMIT THIS AS THE FINISHED TOOL. Move the real implementation from
``src/tools/ implementations/generate_cover_letter.py`` into this package
(and install a LaTeX distribution so it can compile).
"""

from __future__ import annotations

import logging

from src.schemas.cover_letter import GenerateCoverLetterInput, GenerateCoverLetterOutput

logger = logging.getLogger(__name__)


def generate_cover_letter(
    inp: GenerateCoverLetterInput,
) -> GenerateCoverLetterOutput:
    """STUB: return a placeholder cover-letter result without compiling a PDF."""

    logger.warning(
        "generate_cover_letter STUB invoked for %s -- no real letter was produced.",
        inp.job.job_id,
    )
    return GenerateCoverLetterOutput(
        job_id=inp.job.job_id,
        output_tex_path=f"outputs/{inp.job.job_id}/cover-letter.tex",
        output_pdf_path=f"outputs/{inp.job.job_id}/cover-letter.pdf",
        page_count=1,
        evidence_used=[],
        errors=[],
    )
