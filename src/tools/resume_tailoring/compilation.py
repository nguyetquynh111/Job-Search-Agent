"""LaTeX output paths and one-page PDF compilation for resume tailoring."""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

from src.tracing.langfuse import TraceManager
from src.utils.latex import pdf_page_count, run_pdflatex

MAX_COMPILE_ATTEMPTS = 3


def output_paths(output_dir: Path, source_path: Path) -> tuple[Path, Path]:
    """Use a new file for revisions so the previous approved source is untouched."""

    try:
        source_is_generated = source_path.resolve().parent == output_dir.resolve()
    except OSError:
        source_is_generated = False
    if not source_is_generated:
        stem = "resume_draft"
    else:
        match = re.fullmatch(r"resume-revision-(\d+)", source_path.stem)
        revision = int(match.group(1)) + 1 if match else 1
        stem = f"resume-revision-{revision}"
    return output_dir / f"{stem}.tex", output_dir / f"{stem}.pdf"


def compile_one_page(
    source: str,
    tex_path: Path,
    pdf_path: Path,
    *,
    tracer: TraceManager | None = None,
    trace_metadata: dict[str, object] | None = None,
    revision: Callable[[str, int], str] | None = None,
    latex_runner: Callable[[Path], list[str]] | None = None,
) -> tuple[int | None, list[str], str]:
    """Compile LaTeX and validate the generated resume is exactly one page."""

    errors: list[str] = []
    working = source
    page_count: int | None = None
    for attempt in range(MAX_COMPILE_ATTEMPTS):
        tex_path.write_text(working, encoding="utf-8")
        runner = latex_runner or (
            lambda path: run_pdflatex(path, artifact_label="tailored resume")
        )
        run_errors = runner(tex_path)
        if run_errors:
            return None, run_errors, working
        pdf_created = pdf_path.is_file()
        if not pdf_created:
            return (
                None,
                ["pdflatex completed without producing the expected PDF."],
                working,
            )
        try:
            page_count = pdf_page_count(pdf_path)
        except Exception as exc:  # noqa: BLE001 - report invalid generated PDFs
            return None, [f"Generated resume PDF is invalid: {exc}"], working
        if page_count == 1:
            return 1, [], working
        errors.append(
            f"Resume compiled to {page_count} pages; exactly one is required."
        )
        if revision is not None and attempt + 1 < MAX_COMPILE_ATTEMPTS:
            revised = revision(working, attempt + 1)
            if revised == working:
                break
            working = revised
    errors.append(
        "A compliant one-page resume could not be produced within the configured "
        "revision limit. Only newly edited content was shortened; layout, margins, "
        "font size, spacing, and unrelated source were preserved."
    )
    return page_count, errors, working


__all__ = ["MAX_COMPILE_ATTEMPTS", "compile_one_page", "output_paths"]
