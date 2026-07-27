"""Final artifact inventory and evidence-based completion checks."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path

import streamlit as st

from app.services.run_state import NOT_AVAILABLE, RunSnapshot


@dataclass(frozen=True)
class FinalCheck:
    label: str
    status: str
    detail: str


def render_deliverables(snapshot: RunSnapshot) -> None:
    st.markdown(
        """
        <div class="section-heading">
          <div><div class="eyebrow">Final stage</div><h2>Final outputs</h2></div>
          <span class="source-tag">Canonical artifacts</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    checks = final_checks(snapshot)
    columns = st.columns(4)
    for index, check in enumerate(checks):
        with columns[index % 4]:
            _check_card(check)
    st.markdown("### Top 3 deliverables")
    if not snapshot.top_3_job_ids:
        st.info(NOT_AVAILABLE)
        return
    for job_id in snapshot.top_3_job_ids:
        _deliverable_card(snapshot, job_id)


def final_checks(snapshot: RunSnapshot) -> list[FinalCheck]:
    """Use canonical run evidence; incomplete evidence remains unavailable."""

    artifacts = [snapshot.artifact(job_id) for job_id in snapshot.top_3_job_ids]
    automatic_top = (
        len(snapshot.top_3_job_ids) == 3
        and snapshot.top_selection_source
        in {"live_state", "manifest", "ranked_artifact"}
    )
    fit_count = sum(bool(item.fit_analysis) for item in artifacts)
    final_resume_count = sum(
        item.path("resume_after") is not None and not item.after_is_draft
        for item in artifacts
    )
    one_page_count = sum(
        item.page_counts.get("resume_after") == 1 and not item.after_is_draft
        for item in artifacts
    )
    cover_count = sum(item.path("cover_letter") is not None for item in artifacts)
    pause = snapshot.human_pause_used
    memory_needed = bool(snapshot.new_memory_fact_ids)
    trace_available = bool(snapshot.trace_events or snapshot.trace_url)
    return [
        _binary(
            "Three Top 3 jobs selected",
            automatic_top,
            "Automatic selection metadata present."
            if automatic_top
            else "Automatic selection metadata unavailable.",
        ),
        _count_check("Three fit analyses produced", fit_count, 3),
        _count_check("Three final resumes approved", final_resume_count, 3),
        _count_check("Three one-page resume PDFs", one_page_count, 3),
        _count_check("Three cover letters generated", cover_count, 3),
        FinalCheck(
            "Exactly one human pause completed",
            "PASS" if pause == 1 and snapshot.status == "COMPLETED" else "UNKNOWN",
            (
                "One review record and completed run are present."
                if pause == 1 and snapshot.status == "COMPLETED"
                else "Completed review evidence unavailable."
            ),
        ),
        FinalCheck(
            "Memory update shown when applicable",
            (
                "PASS"
                if memory_needed
                and all(
                    any(
                        str(fact.get("fact_id")) == fact_id
                        for fact in snapshot.memory_facts
                    )
                    for fact_id in snapshot.new_memory_fact_ids
                )
                else "NOT_APPLICABLE"
                if not memory_needed
                else "UNKNOWN"
            ),
            (
                f"{len(snapshot.new_memory_fact_ids)} run-attributed memory fact(s) present."
                if memory_needed
                else "No new memory write is attributed to this run."
            ),
        ),
        FinalCheck(
            "End-to-end trace available",
            "PASS" if trace_available else "UNKNOWN",
            snapshot.trace_status,
        ),
    ]


def _deliverable_card(snapshot: RunSnapshot, job_id: str) -> None:
    job = snapshot.job(job_id)
    record = snapshot.artifact(job_id)
    available = {
        "Job details": record.path("job_details"),
        "Resume before": record.path("resume_before"),
        "Resume after": (
            record.path("resume_after") if not record.after_is_draft else None
        ),
        "Resume draft": (
            record.path("resume_after") if record.after_is_draft else None
        ),
        "Fit analysis": record.path("fit_analysis_markdown")
        or record.path("fit_analysis_json"),
        "Cover letter": record.path("cover_letter"),
        "Change log": record.path("change_log"),
    }
    with st.container(border=True):
        header, meta = st.columns([0.74, 0.26])
        with header:
            st.markdown(
                f"""
                <div class="deliverable-title">
                  <span class="id-tag">{escape(job_id)}</span>
                  <div><h3>{escape(str(job.get("title") or "Title unavailable"))}</h3>
                  <p>{escape(str(job.get("company") or "Company unavailable"))}</p></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with meta:
            st.caption(str(job.get("location") or "Location unavailable"))
            if job.get("url"):
                st.link_button("Open job posting", str(job["url"]), width="stretch")
        columns = st.columns(4)
        for index, (label, path) in enumerate(available.items()):
            with columns[index % 4]:
                _artifact_control(snapshot, job_id, label, path)


def _artifact_control(
    snapshot: RunSnapshot,
    job_id: str,
    label: str,
    path: Path | None,
) -> None:
    if path is None or not path.is_file():
        st.markdown(
            f"""
            <div class="artifact-control artifact-control--missing">
              <span>{escape(label)}</span><strong>Unavailable</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return
    st.markdown(
        f"""
        <div class="artifact-control artifact-control--ready">
          <span>{escape(label)}</span><strong>{escape(path.name)}</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )
    mime = (
        "application/pdf"
        if path.suffix.casefold() == ".pdf"
        else "application/json"
        if path.suffix.casefold() == ".json"
        else "text/markdown"
    )
    st.download_button(
        "Open / download",
        data=path.read_bytes(),
        file_name=path.name,
        mime=mime,
        key=f"artifact-{snapshot.run_id}-{job_id}-{label}",
        width="stretch",
    )


def _binary(label: str, passed: bool, detail: str) -> FinalCheck:
    return FinalCheck(label, "PASS" if passed else "UNKNOWN", detail)


def _count_check(label: str, found: int, expected: int) -> FinalCheck:
    return FinalCheck(
        label,
        "PASS" if found == expected else "UNKNOWN",
        f"{found} of {expected} evidenced.",
    )


def _check_card(check: FinalCheck) -> None:
    tone = {
        "PASS": "pass",
        "UNKNOWN": "unknown",
        "NOT_APPLICABLE": "na",
    }.get(check.status, "unknown")
    st.markdown(
        f"""
        <div class="final-check final-check--{tone}">
          <span>{escape(check.status.replace("_", " "))}</span>
          <strong>{escape(check.label)}</strong>
          <p>{escape(check.detail)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
