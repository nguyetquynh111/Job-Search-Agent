"""Canonical output writing and final artifact validation."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from pydantic import Field

from app.configuration import AppConfig, get_config
from src.domain import StrictBaseModel
from src.utils.latex import pdf_page_count


class OutputManifest(StrictBaseModel):
    """Validated canonical artifacts produced for the selected Top 3."""

    job_ids: list[str] = Field(min_length=3, max_length=3)
    job_directories: list[str] = Field(min_length=3, max_length=3)
    mandatory_files_per_job: list[str]
    resume_count: int = 3
    cover_letter_count: int = 3
    pdf_count: int = 9


MANDATORY_JOB_FILES = (
    "job_details.json",
    "resume_before.pdf",
    "resume_after.pdf",
    "cover_letter.pdf",
    "fit_analysis.md",
)
TEMPORARY_LATEX_SUFFIXES = (
    ".aux",
    ".fdb_latexmk",
    ".fls",
    ".log",
    ".out",
    ".synctex.gz",
    ".toc",
)


class OutputValidationError(RuntimeError):
    """Raised when final Top-3 output artifacts are incomplete or invalid."""


def write_and_validate_outputs(
    state: dict[str, Any],
    *,
    config: AppConfig | None = None,
) -> dict[str, Any]:
    """Write canonical Top-3 artifacts and validate the complete output contract."""

    active = config or get_config()
    top_job_ids = list(state.get("top_3_job_ids", []))
    if len(top_job_ids) != 3 or len(set(top_job_ids)) != 3:
        raise OutputValidationError(
            "Final output writing requires exactly three distinct Top-3 job IDs."
        )

    jobs = {
        item["job_id"]: item
        for item in state.get("jobs", [])
        if item.get("job_id") in top_job_ids
    }
    tailoring = state.get("tailoring_results", {})
    letters = state.get("cover_letter_results", {})
    artifacts = state.get("fit_analysis_artifacts", {})
    produced_directories: list[str] = []

    for job_id in top_job_ids:
        if job_id not in jobs:
            raise OutputValidationError(f"Missing job details for {job_id}.")
        if job_id not in tailoring:
            raise OutputValidationError(f"Missing final resume result for {job_id}.")
        if job_id not in letters:
            raise OutputValidationError(f"Missing cover-letter result for {job_id}.")

        job_dir = active.output_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        produced_directories.append(str(job_dir))
        (job_dir / "job_details.json").write_text(
            json.dumps(jobs[job_id], indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        _copy_artifact(
            tailoring[job_id].get("output_pdf_path"),
            job_dir / "resume_after.pdf",
            label=f"final resume for {job_id}",
        )
        _copy_artifact(
            letters[job_id].get("output_pdf_path"),
            job_dir / "cover_letter.pdf",
            label=f"cover letter for {job_id}",
        )

        fit_path = Path(
            artifacts.get(job_id, {}).get("markdown_path", job_dir / "fit_analysis.md")
        )
        if fit_path.resolve() != (job_dir / "fit_analysis.md").resolve():
            _copy_artifact(
                str(fit_path),
                job_dir / "fit_analysis.md",
                label=f"fit analysis for {job_id}",
            )

        _remove_temporary_latex_files(job_dir)
        _validate_job_directory(job_dir)

    if len(produced_directories) != 3:
        raise OutputValidationError(
            f"Expected three selected job directories, produced {len(produced_directories)}."
        )
    return OutputManifest(
        job_ids=top_job_ids,
        job_directories=produced_directories,
        mandatory_files_per_job=list(MANDATORY_JOB_FILES),
    ).model_dump()


def _copy_artifact(source: str | None, destination: Path, *, label: str) -> None:
    if not source:
        raise OutputValidationError(f"Missing source path for {label}.")
    source_path = Path(source)
    if not source_path.is_file():
        raise OutputValidationError(f"Missing {label}: {source_path}")
    try:
        same_file = source_path.resolve() == destination.resolve()
    except OSError:
        same_file = False
    if not same_file:
        shutil.copy2(source_path, destination)


def _remove_temporary_latex_files(job_dir: Path) -> None:
    for path in job_dir.iterdir():
        if path.is_file() and any(
            path.name.endswith(suffix) for suffix in TEMPORARY_LATEX_SUFFIXES
        ):
            path.unlink()


def _validate_job_directory(job_dir: Path) -> None:
    missing = [name for name in MANDATORY_JOB_FILES if not (job_dir / name).is_file()]
    if missing:
        raise OutputValidationError(
            f"{job_dir.name} is missing mandatory files: {', '.join(missing)}"
        )
    for filename in ("resume_before.pdf", "resume_after.pdf", "cover_letter.pdf"):
        page_count = pdf_page_count(job_dir / filename)
        if page_count != 1:
            raise OutputValidationError(
                f"{job_dir.name}/{filename} must be exactly one page; found {page_count}."
            )
    if not (job_dir / "fit_analysis.md").read_text(encoding="utf-8").strip():
        raise OutputValidationError(f"{job_dir.name}/fit_analysis.md is empty.")
    leftovers = [
        path.name
        for path in job_dir.iterdir()
        if path.is_file()
        and any(path.name.endswith(suffix) for suffix in TEMPORARY_LATEX_SUFFIXES)
    ]
    if leftovers:
        raise OutputValidationError(
            f"{job_dir.name} contains temporary LaTeX files: {sorted(leftovers)}"
        )
