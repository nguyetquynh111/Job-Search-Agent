"""Evidence-backed Top 3 fit-analysis view."""

from __future__ import annotations

from html import escape
from typing import Any

import streamlit as st

from app.services.run_state import NOT_AVAILABLE, RunSnapshot


def render_fit_analysis(snapshot: RunSnapshot) -> None:
    st.markdown(
        """
        <div class="section-heading">
          <div><div class="eyebrow">Stage 04</div><h2>Top 3 fit analyses</h2></div>
          <span class="source-tag">analyze_fit</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if not snapshot.top_3_job_ids:
        st.info(NOT_AVAILABLE)
        return
    tabs = st.tabs([_tab_label(snapshot, job_id) for job_id in snapshot.top_3_job_ids])
    for tab, job_id in zip(tabs, snapshot.top_3_job_ids, strict=True):
        with tab:
            analysis = snapshot.fit_analyses.get(job_id) or snapshot.artifact(
                job_id
            ).fit_analysis
            if not analysis:
                st.info(NOT_AVAILABLE)
                continue
            _render_job_context(snapshot.job(job_id))
            left, right = st.columns([1.15, 0.85], gap="large")
            with left:
                _claim_section(
                    "Relevant experience",
                    analysis.get("relevant_experience", []),
                    snapshot,
                    "experience",
                )
                _claim_section(
                    "Seniority",
                    analysis.get("seniority", []),
                    snapshot,
                    "seniority",
                )
                _claim_section(
                    "Education",
                    analysis.get("education", []),
                    snapshot,
                    "education",
                )
            with right:
                _skill_matrix(analysis, snapshot)
                _project_section(analysis, snapshot)
            failures = analysis.get("validation_failures", [])
            if failures:
                with st.expander(
                    f"Fit-analysis validator notes ({len(failures)})", expanded=False
                ):
                    for failure in failures:
                        st.markdown(f"- {failure}")


def _render_job_context(job: dict[str, Any]) -> None:
    st.markdown(
        f"""
        <div class="job-context">
          <div><span class="id-tag">{escape(str(job.get("job_id", "—")))}</span>
          <h3>{escape(str(job.get("title") or "Title unavailable"))}</h3></div>
          <div class="job-context__meta">
            <strong>{escape(str(job.get("company") or "Company unavailable"))}</strong>
            <span>{escape(str(job.get("industry_domain") or "Domain unavailable"))}</span>
            <span>{escape(str(job.get("location") or "Location unavailable"))}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _claim_section(
    title: str,
    claims: list[dict[str, Any]],
    snapshot: RunSnapshot,
    tone: str,
) -> None:
    st.markdown(f'<h4 class="subsection-title">{escape(title)}</h4>', unsafe_allow_html=True)
    if not claims:
        st.caption("No claims recorded in this fit analysis.")
        return
    for claim in claims:
        _claim_card(claim, snapshot, tone)


def _skill_matrix(analysis: dict[str, Any], snapshot: RunSnapshot) -> None:
    st.markdown('<h4 class="subsection-title">Core skills</h4>', unsafe_allow_html=True)
    groups = [
        ("Aligned", analysis.get("aligned_skills", []), "aligned"),
        (
            "Missing on resume · evidenced elsewhere",
            analysis.get("evidenced_missing_skills", []),
            "evidenced",
        ),
        ("Genuine gaps", analysis.get("genuine_gaps", []), "gap"),
    ]
    for label, claims, tone in groups:
        st.markdown(
            f'<div class="skill-group-label skill-group-label--{tone}">{escape(label)} · {len(claims)}</div>',
            unsafe_allow_html=True,
        )
        if not claims:
            st.caption("None recorded.")
        for claim in claims:
            _claim_card(claim, snapshot, tone, compact=True)


def _project_section(analysis: dict[str, Any], snapshot: RunSnapshot) -> None:
    st.markdown('<h4 class="subsection-title">Projects</h4>', unsafe_allow_html=True)
    for claim in analysis.get("project_analysis", []):
        _claim_card(claim, snapshot, "project", compact=True)
    swap = analysis.get("project_swap")
    if not isinstance(swap, dict):
        st.caption("No project swap was recommended.")
        return
    st.markdown(
        f"""
        <div class="swap-card">
          <div class="swap-card__label">Suggested portfolio swap</div>
          <div class="swap-card__flow">
            <span>{escape(str(swap.get("remove_project") or "Current weak project unavailable"))}</span>
            <span class="swap-card__arrow">→</span>
            <strong>{escape(str(swap.get("add_project") or "Replacement unavailable"))}</strong>
          </div>
          <p>{escape(str(swap.get("rationale") or NOT_AVAILABLE))}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _evidence_details(swap.get("evidence_ids", []), snapshot)


def _claim_card(
    claim: dict[str, Any],
    snapshot: RunSnapshot,
    tone: str,
    *,
    compact: bool = False,
) -> None:
    if not isinstance(claim, dict):
        return
    confidence = claim.get("confidence")
    confidence_text = (
        f"{float(confidence):.0%} confidence"
        if isinstance(confidence, (float, int))
        else "Confidence unavailable"
    )
    compact_class = " claim-card--compact" if compact else ""
    st.markdown(
        f"""
        <div class="claim-card claim-card--{escape(tone)}{compact_class}">
          <p>{escape(str(claim.get("claim") or NOT_AVAILABLE))}</p>
          <div class="claim-card__meta">
            <span>{escape(confidence_text)}</span>
            <span>{escape(str(claim.get("notes") or ""))}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _evidence_details(claim.get("evidence_ids", []), snapshot)


def _evidence_details(evidence_ids: Any, snapshot: RunSnapshot) -> None:
    ids = [str(item) for item in evidence_ids] if isinstance(evidence_ids, list) else []
    if not ids:
        st.caption("Source evidence not recorded.")
        return
    with st.expander("Source evidence · " + ", ".join(ids), expanded=False):
        for evidence_id in ids:
            item = snapshot.evidence_lookup.get(evidence_id)
            if not item:
                st.markdown(f"`{evidence_id}` — source text unavailable")
                continue
            st.markdown(
                f"**{evidence_id}** · {item.get('source', 'source unavailable')}  \n"
                f"{item.get('text', NOT_AVAILABLE)}"
            )


def _tab_label(snapshot: RunSnapshot, job_id: str) -> str:
    job = snapshot.job(job_id)
    return f"{job_id} · {job.get('company', 'Company')}"

