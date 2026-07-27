"""Lightweight validation and previews for repository input files."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import streamlit as st

from src.utils.input_loading import (
    load_candidate_profile,
    load_jobs_csv,
    load_portfolio,
    load_resume_data,
)
from src.utils.latex import pdf_page_count, run_pdflatex


@dataclass(frozen=True)
class ResumePreview:
    """Parsed resume metadata and its compiled visual preview."""

    resume: Any
    pdf_path: Path
    page_count: int


@dataclass(frozen=True)
class InputReviewItem:
    """One parsed input, or a safe validation error for the upload screen."""

    title: str
    filename: str
    value: Any | None = None
    error: str | None = None
    uploaded: bool = False

    @property
    def valid(self) -> bool:
        return self.value is not None and self.error is None


@dataclass(frozen=True)
class InputReview:
    """Parsed values used by the review panel and Start button."""

    items: dict[str, InputReviewItem]

    @property
    def errors(self) -> list[str]:
        return [
            item.error
            for item in self.items.values()
            if item.error is not None
        ]

    @property
    def valid(self) -> bool:
        return len(self.items) == 4 and all(
            item.valid for item in self.items.values()
        )

    @property
    def has_uploads(self) -> bool:
        return any(item.uploaded for item in self.items.values())

    @property
    def missing(self) -> list[str]:
        return [
            item.filename
            for item in self.items.values()
            if not item.uploaded
        ]


def load_review_item(title: str, filename: str, path: Path) -> InputReviewItem:
    """Parse an input with the same loader used by the agent workflow."""

    if not path.is_file():
        return InputReviewItem(
            title=title,
            filename=filename,
            error=f"{title}: upload {filename} to continue.",
            uploaded=True,
        )

    try:
        if filename == "resume.tex":
            value = _load_resume_preview(path)
        elif filename == "jobs.csv":
            value = load_jobs_csv(path)
        elif filename == "portfolio.txt":
            value = load_portfolio(path)
        elif filename == "preferences.yaml":
            value = load_candidate_profile(path)
        else:
            return InputReviewItem(
                title=title,
                filename=filename,
                error=f"{title}: preview is not configured for {filename}.",
                uploaded=True,
            )
    except Exception as exc:  # UI boundary: loaders may wrap third-party errors.
        return InputReviewItem(
            title=title,
            filename=filename,
            error=f"{title}: {exc}",
            uploaded=True,
        )

    return InputReviewItem(
        title=title,
        filename=filename,
        value=value,
        uploaded=True,
    )


def render_input_review(review: InputReview) -> None:
    """Render a compact, optional review of all parsed candidate inputs."""

    if not review.has_uploads:
        return

    valid_count = sum(item.valid for item in review.items.values())
    label = f"Review uploaded inputs · {valid_count}/{len(review.items)} ready"
    with st.expander(label, expanded=bool(review.errors)):
        tabs = st.tabs(["Resume", "Jobs", "Portfolio", "Preferences"])
        renderers = (
            _render_resume,
            _render_jobs,
            _render_portfolio,
            _render_preferences,
        )
        filenames = (
            "resume.tex",
            "jobs.csv",
            "portfolio.txt",
            "preferences.yaml",
        )
        for tab, filename, renderer in zip(
            tabs, filenames, renderers, strict=True
        ):
            with tab:
                item = review.items[filename]
                if item.error:
                    st.error(item.error)
                elif not item.uploaded:
                    st.info(f"Upload {filename} to preview this input.")
                else:
                    renderer(item.value)


def _render_resume(resume: Any) -> None:
    parsed = resume.resume
    summary = st.columns(4)
    values = (
        ("Pages", resume.page_count),
        ("Experience", len(parsed.experience)),
        ("Projects", len(parsed.projects)),
        ("Skills", len(parsed.skills)),
    )
    for column, (label, value) in zip(summary, values, strict=True):
        column.metric(label, value)
    try:
        st.pdf(resume.pdf_path, height=760, key="uploaded-resume-preview")
    except Exception as exc:
        st.error(f"PDF viewer unavailable: {exc}")


def _render_jobs(jobs: list[Any]) -> None:
    st.caption(f"{len(jobs)} jobs loaded · showing the fields used for review")
    rows = [
        {
            "Job ID": job.job_id,
            "Title": job.title,
            "Company": job.company,
            "Location": job.location,
            "Remote": job.remote,
            "Experience": job.years_experience_required,
            "Required skills": ", ".join(job.required_skills),
        }
        for job in jobs
    ]
    st.dataframe(
        rows,
        hide_index=True,
        width="stretch",
        height=min(420, 38 + max(1, min(len(rows), 10)) * 35),
    )


def _render_portfolio(portfolio: Any) -> None:
    projects = portfolio.projects
    technologies = {
        technology
        for project in projects
        for technology in project.technologies
    }
    left, right = st.columns(2)
    left.metric("Projects", len(projects))
    right.metric("Technologies", len(technologies))
    rows = [
        {
            "Project": project.name,
            "Role": project.role or "—",
            "Period": project.period or "—",
            "Technologies": ", ".join(project.technologies),
            "Summary": project.description,
        }
        for project in projects
    ]
    st.dataframe(
        rows,
        hide_index=True,
        width="stretch",
        height=min(420, 38 + max(1, min(len(rows), 8)) * 48),
    )


def _render_preferences(profile: Any) -> None:
    preferences = profile.preferences
    rows = [
        {
            "Preference": "Target roles",
            "Value": _display_list(preferences.target_job_titles),
        },
        {
            "Preference": "Locations",
            "Value": _display_list(preferences.preferred_locations),
        },
        {
            "Preference": "Remote only",
            "Value": "Yes" if preferences.remote_only else "No",
        },
        {
            "Preference": "Job types",
            "Value": _display_list(preferences.job_types),
        },
        {
            "Preference": "Minimum salary",
            "Value": (
                f"${preferences.min_salary:,}"
                if preferences.min_salary is not None
                else "Not specified"
            ),
        },
        {
            "Preference": "Excluded companies",
            "Value": _display_list(preferences.excluded_companies),
        },
        {
            "Preference": "Excluded keywords",
            "Value": _display_list(preferences.excluded_keywords),
        },
        {
            "Preference": "Master skills",
            "Value": _display_list(profile.master_skills),
        },
    ]
    st.dataframe(
        rows,
        hide_index=True,
        width="stretch",
        column_config={
            "Preference": st.column_config.TextColumn(width="small"),
            "Value": st.column_config.TextColumn(width="large"),
        },
    )


def _display_list(values: list[str]) -> str:
    return ", ".join(values) if values else "None"


def _load_resume_preview(path: Path) -> ResumePreview:
    """Compile the exact uploaded LaTeX source into a cached PDF preview."""

    source = path.read_bytes()
    digest = hashlib.sha256(source).hexdigest()[:16]
    preview_dir = (
        path.parent.parent
        / "tmp"
        / "pdfs"
        / "uploaded-resume-preview"
        / digest
    )
    preview_dir.mkdir(parents=True, exist_ok=True)
    tex_path = preview_dir / "resume.tex"
    pdf_path = preview_dir / "resume.pdf"

    if not pdf_path.is_file():
        tex_path.write_bytes(source)
        errors = run_pdflatex(
            tex_path,
            artifact_label="uploaded resume preview",
        )
        for suffix in (".aux", ".log", ".out"):
            tex_path.with_suffix(suffix).unlink(missing_ok=True)
        if errors:
            raise RuntimeError(errors[0])
        if not pdf_path.is_file():
            raise RuntimeError(
                "pdflatex completed without producing the resume preview PDF."
            )

    try:
        pages = pdf_page_count(pdf_path)
    except Exception as exc:
        raise RuntimeError(f"Generated resume preview PDF is invalid: {exc}") from exc

    return ResumePreview(
        resume=load_resume_data(path),
        pdf_path=pdf_path,
        page_count=pages,
    )
