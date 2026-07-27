"""Resume previews, diffs, change evidence, and validation indicators."""

from __future__ import annotations

import difflib
import importlib.util
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

import streamlit as st
from pypdf import PdfReader

from app.services.run_state import JobArtifacts, NOT_AVAILABLE, RunSnapshot


@dataclass(frozen=True)
class ValidationCheck:
    label: str
    status: str
    detail: str


def render_resume_changes(snapshot: RunSnapshot) -> None:
    st.markdown(
        """
        <div class="section-heading">
          <div><div class="eyebrow">Stage 05</div><h2>Resume tailoring evidence</h2></div>
          <span class="source-tag">tailor_resume</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "A check is marked passed only when the selected run contains supporting artifact evidence."
    )
    if not snapshot.top_3_job_ids:
        st.info(NOT_AVAILABLE)
        return
    tabs = st.tabs(
        [
            f"{job_id} · {snapshot.job(job_id).get('company', 'Company')}"
            for job_id in snapshot.top_3_job_ids
        ]
    )
    for tab, job_id in zip(tabs, snapshot.top_3_job_ids, strict=True):
        with tab:
            record = snapshot.artifact(job_id)
            _render_resume_header(snapshot, job_id, record)
            _render_validation_grid(validation_checks(snapshot, job_id))
            preview_tab, diff_tab, log_tab = st.tabs(
                ["PDF comparison", "Textual diff", "Change log & evidence"]
            )
            with preview_tab:
                before_column, after_column = st.columns(2, gap="large")
                with before_column:
                    st.markdown("#### Resume before")
                    render_pdf_preview(
                        record.path("resume_before"),
                        key=f"before-{snapshot.run_id}-{job_id}",
                    )
                with after_column:
                    label = "Resume draft" if record.after_is_draft else "Resume after"
                    st.markdown(f"#### {label}")
                    render_pdf_preview(
                        record.path("resume_after"),
                        key=f"after-{snapshot.run_id}-{job_id}",
                    )
            with diff_tab:
                diff = textual_diff(record)
                if diff:
                    st.code(diff, language="diff", line_numbers=True)
                    st.caption(
                        "Derived directly from the available TeX sources, or from extracted PDF text when TeX is absent."
                    )
                else:
                    st.info(NOT_AVAILABLE)
            with log_tab:
                _render_change_log(record.change_log, snapshot)
                analysis = snapshot.fit_analyses.get(job_id) or record.fit_analysis
                _render_project_swap(analysis)


def validation_checks(
    snapshot: RunSnapshot,
    job_id: str,
) -> list[ValidationCheck]:
    """Derive visible checks strictly from actual PDFs and validated change logs."""

    record = snapshot.artifact(job_id)
    changes = record.change_log
    validated = [
        item
        for item in changes
        if "validated" in str(item.get("validation_result", "")).casefold()
    ]
    summary = [item for item in validated if item.get("section") == "summary"]
    experience = [item for item in validated if item.get("section") == "experience"]
    skill_changes = [item for item in changes if item.get("section") == "skills"]
    project_changes = [item for item in validated if item.get("section") == "projects"]
    tailoring_failures = record.tailoring_result.get("validation_failures", [])

    checks = [
        _evidence_check(
            "Summary rewritten",
            bool(summary),
            bool(changes),
            "Validated summary change found."
            if summary
            else "No validated summary change is available.",
        ),
        _evidence_check(
            "Exactly two experience bullets modified",
            len(experience) == 2,
            bool(changes),
            f"{len(experience)} validated experience-bullet changes found."
            if changes
            else "Change log unavailable.",
        ),
    ]
    if skill_changes:
        valid_skills = all(
            item.get("evidence_ids")
            and "validated" in str(item.get("validation_result", "")).casefold()
            for item in skill_changes
        )
        checks.append(
            ValidationCheck(
                "Skills added only with evidence",
                "PASS" if valid_skills else "FAIL",
                f"{len(skill_changes)} skill change(s) inspected.",
            )
        )
    else:
        checks.append(
            ValidationCheck(
                "Skills added only with evidence",
                "UNKNOWN",
                "No skill change log is available.",
            )
        )

    analysis = snapshot.fit_analyses.get(job_id) or record.fit_analysis
    swap = analysis.get("project_swap") if isinstance(analysis, dict) else None
    if swap:
        portfolio_ids = [
            evidence_id
            for evidence_id in swap.get("evidence_ids", [])
            if snapshot.evidence_lookup.get(evidence_id, {}).get("source")
            == "portfolio"
        ]
        passed = bool(project_changes and portfolio_ids)
        checks.append(
            ValidationCheck(
                "Swapped project exists in portfolio",
                "PASS" if passed else "UNKNOWN",
                (
                    "Validated project edit cites " + ", ".join(portfolio_ids) + "."
                    if passed
                    else "A project recommendation exists, but applied-edit evidence is unavailable."
                ),
            )
        )
    else:
        checks.append(
            ValidationCheck(
                "Swapped project exists in portfolio",
                "NOT_APPLICABLE",
                "No project swap was recommended in this fit analysis.",
            )
        )

    no_invention_passed = bool(changes) and len(validated) == len(changes) and not tailoring_failures
    checks.append(
        ValidationCheck(
            "No invented employer, title, date, skill, or project",
            "PASS" if no_invention_passed else "UNKNOWN",
            (
                "Every recorded edit says source scope and evidence were validated."
                if no_invention_passed
                else "A complete validated change log is unavailable."
            ),
        )
    )
    page_count = record.page_counts.get("resume_after")
    checks.append(
        ValidationCheck(
            "Resume PDF is exactly one page",
            (
                "PASS"
                if page_count == 1
                else "FAIL"
                if isinstance(page_count, int)
                else "UNKNOWN"
            ),
            (
                f"PDF parser found {page_count} page{'s' if page_count != 1 else ''}."
                if isinstance(page_count, int)
                else "Resume-after PDF unavailable or unreadable."
            ),
        )
    )
    return checks


def render_pdf_preview(
    path: Path | None,
    *,
    key: str,
    show_download: bool = True,
    show_path: bool = True,
    download_label: str = "Download PDF",
) -> None:
    """Preview a real PDF when the optional viewer is installed."""

    if path is None or not path.is_file():
        st.info(NOT_AVAILABLE)
        return
    if importlib.util.find_spec("streamlit_pdf") is not None:
        try:
            st.pdf(path, height=610, key=key)
        except Exception:
            st.caption("Preview unavailable.")
    else:
        st.caption("Preview unavailable.")
    if show_download:
        st.download_button(
            download_label,
            data=path.read_bytes(),
            file_name=path.name,
            mime="application/pdf",
            key=f"download-{key}",
            width="stretch",
        )
    if show_path:
        st.caption(str(path))


def textual_diff(record: JobArtifacts, max_lines: int = 360) -> str:
    """Return a display-only unified diff from actual source artifacts."""

    before = _artifact_text(
        record.files.get("resume_before_tex"), record.path("resume_before")
    )
    after = _artifact_text(
        record.files.get("resume_after_tex"), record.path("resume_after")
    )
    if not before or not after:
        return ""
    lines = list(
        difflib.unified_diff(
            before.splitlines(),
            after.splitlines(),
            fromfile="resume_before",
            tofile="resume_after",
            lineterm="",
        )
    )
    if len(lines) > max_lines:
        lines = [*lines[:max_lines], f"... diff truncated after {max_lines} lines ..."]
    return "\n".join(lines)


def _artifact_text(source: Path | None, pdf: Path | None) -> str:
    if source and source.is_file():
        try:
            return source.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            pass
    if pdf and pdf.is_file():
        try:
            return "\n".join(page.extract_text() or "" for page in PdfReader(str(pdf)).pages)
        except Exception:
            return ""
    return ""


def _render_resume_header(
    snapshot: RunSnapshot, job_id: str, record: JobArtifacts
) -> None:
    job = snapshot.job(job_id)
    after_label = "Draft artifact" if record.after_is_draft else "Final artifact"
    st.markdown(
        f"""
        <div class="job-context">
          <div><span class="id-tag">{escape(job_id)}</span>
            <h3>{escape(str(job.get("title") or "Title unavailable"))}</h3>
          </div>
          <div class="job-context__meta">
            <strong>{escape(str(job.get("company") or "Company unavailable"))}</strong>
            <span>{escape(after_label)}</span>
            <span>Page count: {record.page_counts.get("resume_after") if record.page_counts.get("resume_after") is not None else "unavailable"}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_validation_grid(checks: list[ValidationCheck]) -> None:
    columns = st.columns(3)
    for index, check in enumerate(checks):
        tone = {
            "PASS": "pass",
            "FAIL": "fail",
            "UNKNOWN": "unknown",
            "NOT_APPLICABLE": "na",
        }[check.status]
        with columns[index % 3]:
            st.markdown(
                f"""
                <div class="validation-card validation-card--{tone}">
                  <div class="validation-card__status">{escape(check.status.replace("_", " "))}</div>
                  <div class="validation-card__label">{escape(check.label)}</div>
                  <p>{escape(check.detail)}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_change_log(
    changes: list[dict[str, Any]], snapshot: RunSnapshot
) -> None:
    if not changes:
        st.info(NOT_AVAILABLE)
        return
    for index, change in enumerate(changes, start=1):
        label = (
            f"{index:02d} · {change.get('section', 'section').title()} · "
            f"{change.get('description', 'Recorded change')}"
        )
        with st.expander(label):
            before, after = st.columns(2)
            before.markdown("**Before**")
            before.code(str(change.get("before_text") or NOT_AVAILABLE), language="text")
            after.markdown("**After**")
            after.code(str(change.get("after_text") or NOT_AVAILABLE), language="text")
            st.markdown(f"**Reason**  \n{change.get('reason') or NOT_AVAILABLE}")
            evidence_ids = change.get("evidence_ids", [])
            st.markdown(
                "**Evidence**  \n"
                + (
                    ", ".join(f"`{item}`" for item in evidence_ids)
                    if evidence_ids
                    else NOT_AVAILABLE
                )
            )
            for evidence_id in evidence_ids:
                evidence = snapshot.evidence_lookup.get(str(evidence_id))
                if evidence:
                    st.caption(
                        f"{evidence_id} · {evidence.get('source')} · {evidence.get('text')}"
                    )
            st.markdown(
                f"**Validator result**  \n{change.get('validation_result') or NOT_AVAILABLE}"
            )


def _render_project_swap(analysis: dict[str, Any]) -> None:
    swap = analysis.get("project_swap") if isinstance(analysis, dict) else None
    st.markdown("#### Project swap")
    if not isinstance(swap, dict):
        st.info("No project swap was recorded.")
        return
    st.markdown(
        f"**Remove:** {swap.get('remove_project') or NOT_AVAILABLE}  \n"
        f"**Add:** {swap.get('add_project') or NOT_AVAILABLE}  \n"
        f"**Reason:** {swap.get('rationale') or NOT_AVAILABLE}"
    )


def _evidence_check(
    label: str,
    passed: bool,
    evidence_exists: bool,
    detail: str,
) -> ValidationCheck:
    return ValidationCheck(
        label,
        "PASS" if passed else "FAIL" if evidence_exists else "UNKNOWN",
        detail,
    )
