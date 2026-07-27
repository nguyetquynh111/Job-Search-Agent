"""Canonical output writing and final artifact validation."""

from __future__ import annotations

import json
import shutil
import re
from pathlib import Path
from typing import Any

from pydantic import Field

from src.config import AppConfig, get_config
from src.domain import StrictBaseModel
from src.utils.latex import pdf_page_count
from src.utils.paths import run_output_dir


class OutputManifest(StrictBaseModel):
    """Validated canonical artifacts produced for the selected Top 3."""

    run_id: str | None = None
    output_root: str
    job_ids: list[str] = Field(min_length=3, max_length=3)
    job_directories: list[str] = Field(min_length=3, max_length=3)
    mandatory_files_per_job: list[str]
    resume_count: int = 3
    cover_letter_count: int = 3
    pdf_count: int = 9
    trace_id: str | None = None
    trace_url: str | None = None
    trace_public: bool = False
    trace_ingest_confirmed: bool = False
    observation_count: int = 0
    trace_export_error: str | None = None
    trace_debug_status: str | None = None
    langfuse_host: str | None = None
    langfuse_sdk_version: str | None = None


MANDATORY_JOB_FILES = (
    "job_details.json",
    "resume_before.pdf",
    "resume_after.pdf",
    "cover_letter.pdf",
    "fit_analysis.json",
    "fit_analysis.md",
    "change_log.json",
    "human_review_decision.json",
    "revision_history.json",
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
PUBLIC_TEX_PATH_KEYS = {
    "output_tex_path",
    "tex_path",
    "tex_file",
}
STALE_PUBLIC_ARTIFACT_NAMES = {
    "resume_draft.pdf": "resume_after.pdf",
    "resume_draft.tex": "resume_after.pdf",
    "approved.pdf": "resume_after.pdf",
    "approved.tex": "resume_after.pdf",
    "letter.pdf": "cover_letter.pdf",
    "letter.tex": "cover_letter.pdf",
    "cover_letter.tex": "cover_letter.pdf",
}


class OutputValidationError(RuntimeError):
    """Raised when final Top-3 output artifacts are incomplete or invalid."""


def write_and_validate_outputs(
    state: dict[str, Any],
    *,
    config: AppConfig | None = None,
    persist_run_files: bool = True,
) -> dict[str, Any]:
    """Write canonical Top-3 artifacts and validate the complete output contract."""

    active = config or get_config()
    run_id = state.get("run_id")
    root_dir = run_output_dir(run_id, config=active) if run_id else active.output_dir
    root_dir.mkdir(parents=True, exist_ok=True)
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

        job_dir = root_dir / job_id
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
        tailoring[job_id]["output_pdf_path"] = str(job_dir / "resume_after.pdf")
        tailoring[job_id].pop("output_tex_path", None)
        _copy_artifact(
            letters[job_id].get("output_pdf_path"),
            job_dir / "cover_letter.pdf",
            label=f"cover letter for {job_id}",
        )
        letters[job_id]["output_pdf_path"] = str(job_dir / "cover_letter.pdf")
        letters[job_id].pop("output_tex_path", None)

        fit_path = Path(
            artifacts.get(job_id, {}).get("markdown_path", job_dir / "fit_analysis.md")
        )
        if fit_path.resolve() != (job_dir / "fit_analysis.md").resolve():
            _copy_artifact(
                str(fit_path),
                job_dir / "fit_analysis.md",
                label=f"fit analysis for {job_id}",
            )
        artifacts.setdefault(job_id, {})["markdown_path"] = str(
            job_dir / "fit_analysis.md"
        )
        fit_json_path = artifacts.get(job_id, {}).get("json_path")
        if fit_json_path:
            _copy_artifact(
                fit_json_path,
                job_dir / "fit_analysis.json",
                label=f"fit analysis JSON for {job_id}",
            )
        artifacts.setdefault(job_id, {})["json_path"] = str(job_dir / "fit_analysis.json")
        _write_job_metadata_files(state, job_id, job_dir)

        _remove_temporary_latex_files(job_dir)
        _remove_non_submission_files(job_dir)
        _validate_job_directory(job_dir)

    if len(produced_directories) != 3:
        raise OutputValidationError(
            f"Expected three selected job directories, produced {len(produced_directories)}."
        )
    manifest = OutputManifest(
        run_id=run_id,
        output_root=_public_path(root_dir),
        job_ids=top_job_ids,
        job_directories=[_public_path(Path(path)) for path in produced_directories],
        mandatory_files_per_job=list(MANDATORY_JOB_FILES),
        trace_id=state.get("trace_id"),
        trace_url=(
            state.get("trace_url")
            if state.get("trace_ingest_confirmed") is True
            else None
        ),
        trace_public=state.get("trace_public") is True,
        trace_ingest_confirmed=state.get("trace_ingest_confirmed") is True,
        observation_count=int(state.get("observation_count") or 0),
        trace_export_error=state.get("trace_export_error"),
        trace_debug_status=state.get("trace_debug_status"),
        langfuse_host=state.get("langfuse_host"),
        langfuse_sdk_version=state.get("langfuse_sdk_version"),
    ).model_dump()
    if persist_run_files:
        write_run_files(state, manifest)
    return manifest


def write_run_files(state: dict[str, Any], manifest: dict[str, Any]) -> None:
    """Persist run-level files only after tracing has been flushed and checked."""

    root_dir = Path(str(manifest["output_root"]))
    public_manifest = sanitize_public_artifact_references(manifest)
    (root_dir / "run_manifest.json").write_text(
        json.dumps(public_manifest, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    (root_dir / "public_trace_url.txt").write_text(
        str(
            manifest.get("trace_url")
            if manifest.get("trace_ingest_confirmed") is True
            else ""
        ),
        encoding="utf-8",
    )
    trace_events = sanitize_public_artifact_references(state.get("trace_events", []))
    (root_dir / "trace_events.json").write_text(
        json.dumps(trace_events, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    for filename, key in (
        ("rejected_jobs.json", "rejected_jobs"),
        ("ranked_jobs.json", "ranked_jobs"),
    ):
        (root_dir / filename).write_text(
            json.dumps(
                sanitize_public_artifact_references(state.get(key, [])),
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
            encoding="utf-8",
        )
    memory_file = state.get("memory_file")
    if memory_file:
        memory_path = Path(memory_file)
        if memory_path.is_file():
            _copy_artifact(str(memory_path), root_dir / "memory.json", label="memory")
        elif not (root_dir / "memory.json").exists():
            (root_dir / "memory.json").write_text("[]", encoding="utf-8")
    elif not (root_dir / "memory.json").exists():
        (root_dir / "memory.json").write_text("[]", encoding="utf-8")


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


def _remove_non_submission_files(job_dir: Path) -> None:
    allowed = set(MANDATORY_JOB_FILES)
    for path in job_dir.iterdir():
        if path.is_file() and path.name not in allowed:
            path.unlink()


def _write_job_metadata_files(
    state: dict[str, Any], job_id: str, job_dir: Path
) -> None:
    tailoring = state.get("tailoring_results", {}).get(job_id, {})
    review_decision = state.get("review_decisions", {}).get(job_id, {})
    review_history = [
        entry
        for entry in state.get("review_history", [])
        if job_id in entry.get("decisions", {})
        or job_id in entry.get("actions_taken", {})
        or job_id in entry.get("rejected_job_ids", [])
    ]
    (job_dir / "change_log.json").write_text(
        json.dumps(
            sanitize_public_artifact_references(tailoring.get("change_log", [])),
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )
    (job_dir / "human_review_decision.json").write_text(
        json.dumps(
            sanitize_public_artifact_references(review_decision),
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )
    (job_dir / "revision_history.json").write_text(
        json.dumps(
            sanitize_public_artifact_references(review_history),
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )


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


def sanitize_public_artifact_references(value: Any) -> Any:
    """Remove stale/private path references from public submission artifacts."""

    if isinstance(value, dict):
        safe: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if key_text in PUBLIC_TEX_PATH_KEYS:
                continue
            safe[key_text] = sanitize_public_artifact_references(item)
        return safe
    if isinstance(value, list):
        return [sanitize_public_artifact_references(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_public_artifact_references(item) for item in value]
    if isinstance(value, str):
        return _sanitize_public_artifact_string(value)
    return value


def _sanitize_public_artifact_string(value: str) -> str:
    sanitized = value
    repo_root = Path.cwd().resolve()
    repo_prefix = str(repo_root) + "/"
    sanitized = sanitized.replace(repo_prefix, "")
    for stale, canonical in STALE_PUBLIC_ARTIFACT_NAMES.items():
        if stale in {"letter.pdf", "letter.tex"}:
            sanitized = re.sub(rf"(?<!cover_){re.escape(stale)}", canonical, sanitized)
        else:
            sanitized = sanitized.replace(stale, canonical)
    sanitized = re.sub(
        r"outputs/([^/]+)/source_resume/resume\.pdf",
        r"outputs/\1/resume_before.pdf",
        sanitized,
    )
    return sanitized


def _public_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)
