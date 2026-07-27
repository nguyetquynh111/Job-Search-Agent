"""Filtering evidence view."""

from __future__ import annotations

from html import escape
from typing import Any

import streamlit as st

from app.services.run_state import NOT_AVAILABLE, RunSnapshot


def render_filtering(snapshot: RunSnapshot) -> None:
    st.markdown(
        """
        <div class="section-heading">
          <div><div class="eyebrow">Stage 02</div><h2>Filtering evidence</h2></div>
          <span class="source-tag">filter_jobs</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "Accepted and rejected jobs are shown only from structured run state or durable output artifacts."
    )
    _render_preferences(snapshot.preferences)
    accepted_tab, rejected_tab = st.tabs(
        [
            f"Accepted ({snapshot.accepted_count if snapshot.accepted_count is not None else '—'})",
            f"Rejected ({snapshot.rejected_count if snapshot.rejected_count is not None else '—'})",
        ]
    )
    with accepted_tab:
        if not snapshot.evidence_available.get("filtering"):
            st.info(NOT_AVAILABLE)
        elif not snapshot.filtered_jobs and not snapshot.ranked_jobs:
            st.info("The filtering artifact records no accepted jobs.")
        else:
            accepted = snapshot.filtered_jobs or [
                item.get("job", item) for item in snapshot.ranked_jobs
            ]
            for job in accepted:
                if isinstance(job, dict):
                    _job_row(job, accepted=True)
    with rejected_tab:
        if not snapshot.evidence_available.get("rejections"):
            st.info(NOT_AVAILABLE)
        elif not snapshot.rejected_jobs:
            st.success("The filtering artifact records no rejected jobs.")
        else:
            for item in snapshot.rejected_jobs:
                _rejected_row(item)


def _render_preferences(preferences: dict[str, Any]) -> None:
    with st.expander("Filtering preferences and rules", expanded=False):
        if not preferences:
            st.info(NOT_AVAILABLE)
            return
        labels = {
            "target_job_titles": "Target titles",
            "preferred_locations": "Preferred locations",
            "remote_only": "Remote only",
            "years_of_experience": "Candidate experience",
            "excluded_companies": "Excluded companies",
            "job_types": "Job types",
            "min_salary": "Minimum salary",
            "excluded_keywords": "Excluded keywords",
        }
        columns = st.columns(2)
        for index, (key, label) in enumerate(labels.items()):
            with columns[index % 2]:
                value = preferences.get(key)
                if key == "years_of_experience" and value is not None:
                    value = f"{value} years"
                elif key == "min_salary" and value is not None:
                    value = f"${int(value):,}"
                elif isinstance(value, list):
                    value = ", ".join(map(str, value)) or "None configured"
                st.markdown(f"**{label}**  \n{value if value not in (None, '') else 'Not configured'}")
        st.caption(
            "Rule order: excluded company → remote-only → location → experience → "
            "excluded keyword → target title."
        )


def _job_row(job: dict[str, Any], *, accepted: bool) -> None:
    title = str(job.get("title") or "Untitled role")
    company = str(job.get("company") or "Company unavailable")
    job_id = str(job.get("job_id") or "—")
    location = str(job.get("location") or "Location unavailable")
    remote = "Remote eligible" if job.get("remote") else "Location-based"
    tone = "accepted" if accepted else "neutral"
    st.markdown(
        f"""
        <article class="job-row job-row--{tone}">
          <div>
            <div class="job-row__title">{escape(title)}</div>
            <div class="job-row__meta">{escape(company)} · {escape(location)} · {escape(remote)}</div>
          </div>
          <span class="id-tag">{escape(job_id)}</span>
        </article>
        """,
        unsafe_allow_html=True,
    )


def _rejected_row(item: dict[str, Any]) -> None:
    job = item.get("job", item)
    reasons = item.get("reasons", [])
    if not isinstance(job, dict):
        return
    title = str(job.get("title") or "Untitled role")
    company = str(job.get("company") or "Company unavailable")
    job_id = str(job.get("job_id") or "—")
    with st.expander(f"{title} · {company}  [{job_id}]"):
        if reasons:
            for reason in reasons:
                category = rejection_category(str(reason))
                st.markdown(
                    f"""
                    <div class="reason-row">
                      <span class="reason-row__category">{escape(category)}</span>
                      <span>{escape(str(reason))}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("A rejected-job record exists, but no reason was logged.")


def rejection_category(reason: str) -> str:
    """Classify real reason text for display; the original reason remains intact."""

    normalized = reason.casefold()
    if "location" in normalized or "preferred" in normalized:
        return "Location mismatch"
    if "experience" in normalized or "years" in normalized:
        return "Experience mismatch"
    if "company" in normalized:
        return "Excluded company"
    if "remote" in normalized:
        return "Remote-only mismatch"
    if "title" in normalized:
        return "Target-title mismatch"
    if "keyword" in normalized:
        return "Excluded keyword"
    return "Rule mismatch"

