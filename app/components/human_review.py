"""The workflow's single combined human-review gate."""

from __future__ import annotations

from html import escape
from typing import Any

import streamlit as st

from app.components.resume_diff import render_pdf_preview
from app.services.run_state import NOT_AVAILABLE, RunSnapshot


def review_controls_visible(snapshot: RunSnapshot) -> bool:
    """Controls exist only for a live graph at its actual interrupt."""

    return snapshot.is_waiting_for_review


def render_human_review(snapshot: RunSnapshot) -> dict[str, Any] | None:
    """Render the Top 3 change logs and collect one decision per resume."""

    st.markdown(
        '<h2 class="stage-title">Waiting for Human Review</h2>',
        unsafe_allow_html=True,
    )
    if not review_controls_visible(snapshot):
        st.info(NOT_AVAILABLE)
        return None

    resumes = snapshot.interrupt_payload.get("resumes", {})
    if not isinstance(resumes, dict) or not resumes:
        st.info(NOT_AVAILABLE)
        return None

    job_ids = list(resumes)
    tabs = st.tabs(
        [
            str(payload.get("job_title") or snapshot.job(job_id).get("title") or job_id)
            for job_id, payload in resumes.items()
        ]
    )
    decisions: dict[str, dict[str, str]] = {}
    invalid_rejections: list[str] = []
    for tab, job_id in zip(tabs, job_ids, strict=True):
        with tab:
            payload = resumes[job_id]
            title = (
                payload.get("job_title") or snapshot.job(job_id).get("title") or job_id
            )
            st.markdown(f"**Review and feedback · {escape(str(title))}**")
            decision = st.radio(
                "Decision",
                ("approve", "reject"),
                horizontal=True,
                key=f"review-decision-{snapshot.run_id}-{job_id}",
                label_visibility="collapsed",
            )
            comment = st.text_area(
                "Feedback for this CV",
                help=(
                    "Saved to memory with this CV's job ID. Candidate facts and "
                    "skills are only treated as evidence when explicitly stated."
                ),
                placeholder=(
                    "Example: Make the summary more concise. I have also used "
                    "GraphQL in production."
                ),
                key=f"review-comment-{snapshot.run_id}-{job_id}",
            ).strip()
            st.caption(
                "Feedback is stored separately for this CV. A comment is required "
                "when rejecting it."
            )
            if decision == "reject" and not comment:
                invalid_rejections.append(job_id)
            decisions[job_id] = {"decision": decision, "comment": comment}
            st.divider()
            _review_resume(snapshot, job_id, payload)

    if invalid_rejections:
        st.caption(
            "Rejected resumes require comments: " + ", ".join(invalid_rejections)
        )
    submit = st.button(
        "Submit All 3 CV Reviews & Continue",
        type="primary",
        width="stretch",
        disabled=bool(invalid_rejections),
    )
    return {"decisions": decisions} if submit else None


def _review_resume(
    snapshot: RunSnapshot,
    job_id: str,
    payload: dict[str, Any],
) -> None:
    title = payload.get("job_title") or snapshot.job(job_id).get("title") or job_id
    company = payload.get("company") or snapshot.job(job_id).get("company") or ""
    score = payload.get("score")
    score_text = f"{float(score):.1f}" if isinstance(score, (int, float)) else "—"
    st.markdown(
        f"<h3>{escape(str(title))}</h3>"
        f"<p>{escape(str(company))} · Score {escape(score_text)}</p>",
        unsafe_allow_html=True,
    )

    fit = payload.get("fit_analysis", {})
    st.markdown("**Fit analysis**")
    _fit_analysis(fit if isinstance(fit, dict) else {})

    st.markdown("**Project swap recommendation**")
    swap = payload.get("project_swap")
    if not isinstance(swap, dict):
        swap = fit.get("project_swap") if isinstance(fit, dict) else None
    if isinstance(swap, dict):
        removed = swap.get("remove_project") or "Current project"
        added = swap.get("add_project") or "Replacement project"
        st.markdown(f"{removed} → {added}")
        if swap.get("rationale"):
            st.caption(str(swap["rationale"]))
        _render_citations(swap.get("evidence_ids", []))
    else:
        st.caption("No project swap recommended.")

    st.markdown("**Tailoring change log**")
    changes = payload.get("change_log", [])
    if isinstance(changes, list) and changes:
        for change in changes:
            if not isinstance(change, dict):
                continue
            st.markdown(
                f"- **{escape(str(change.get('section') or 'change'))}:** "
                f"{escape(str(change.get('description') or change.get('reason') or 'Updated'))}"
            )
            _render_citations(change.get("evidence_ids", []))
    else:
        st.caption("No tailoring changes recorded.")

    st.markdown("**Resume preview**")
    render_pdf_preview(
        snapshot.artifact(job_id).path("resume_after"),
        key=f"review-{snapshot.run_id}-{job_id}",
        show_download=True,
        show_path=False,
        download_label="Download Resume PDF",
    )


def _fit_analysis(fit: dict[str, Any]) -> None:
    rendered = False
    for key in ("relevant_experience", "aligned_skills", "genuine_gaps"):
        for claim in fit.get(key, [])[:4]:
            text = claim.get("claim") if isinstance(claim, dict) else str(claim)
            if text:
                st.markdown(f"- {text}")
                if isinstance(claim, dict):
                    _render_citations(claim.get("evidence_ids", []))
                rendered = True
    if not rendered:
        st.caption(NOT_AVAILABLE)


def _render_citations(evidence_ids: Any) -> None:
    citations = [str(item) for item in evidence_ids or [] if str(item).strip()]
    if citations:
        st.caption("Citations: " + ", ".join(citations))
