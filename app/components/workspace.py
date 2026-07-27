"""Minimal, state-driven workspace for one Job Search Agent run."""

from __future__ import annotations

from collections import Counter
from html import escape
from pathlib import Path
from typing import Any

import streamlit as st

from app.components.filtering import rejection_category
from app.components.human_review import render_human_review
from app.services.run_state import NOT_AVAILABLE, RunSnapshot

WORKFLOW = [
    ("Analyze", "analyze"),
    ("Filter", "filter"),
    ("Rank", "rank"),
    ("Fit", "fit"),
    ("Tailor", "tailor"),
    ("Review", "review"),
    ("Final", "final"),
]


def render_workflow_strip(snapshot: RunSnapshot, *, inputs_ready: bool) -> None:
    """Render a compact stage strip without speculative percentages."""

    del inputs_ready
    states = _workflow_states(snapshot)
    nodes = []
    for label, key in WORKFLOW:
        state = states[key]
        nodes.append(
            f'<div class="mini-step mini-step--{escape(state)}">'
            f"<span></span><b>{escape(label)}</b></div>"
        )
    st.markdown(
        '<div class="mini-workflow">' + "<i></i>".join(nodes) + "</div>",
        unsafe_allow_html=True,
    )


def render_workspace(snapshot: RunSnapshot) -> dict[str, Any] | None:
    """Show only the current task, real outputs, and collapsed evidence."""

    if snapshot.is_waiting_for_review:
        decisions = render_human_review(snapshot)
        _render_evidence(snapshot)
        return decisions

    if snapshot.status == "COMPLETED":
        _render_results(snapshot)
        _render_evidence(snapshot)
        _render_session_controls(snapshot)
        return None

    _render_current_stage(snapshot)
    _render_evidence(snapshot)
    _render_session_controls(snapshot)
    return None


def _render_current_stage(snapshot: RunSnapshot) -> None:
    key = _current_workflow_key(snapshot)
    title = dict((key, label) for label, key in WORKFLOW).get(key, "Run")
    st.markdown(f'<h2 class="stage-title">{escape(title)}</h2>', unsafe_allow_html=True)

    if snapshot.phase == "REVISION" and snapshot.status == "RUNNING":
        st.info("Revising Resume...")
    elif key == "analyze":
        _render_analysis(snapshot)
    elif key == "filter":
        _render_filter_summary(snapshot)
    elif key == "rank":
        _render_ranking_summary(snapshot)
    elif key == "fit":
        _render_fit_summary(snapshot)
    elif key == "tailor":
        _render_tailor_summary(snapshot)
    elif key == "review":
        _render_read_only_review(snapshot)
    else:
        _render_finalizing(snapshot)


def _render_analysis(snapshot: RunSnapshot) -> None:
    if not snapshot.jobs_loaded:
        st.info("Analyzing jobs…" if not snapshot.read_only else NOT_AVAILABLE)
        return
    _metric_row(
        [
            ("Jobs", snapshot.jobs_loaded),
            ("Locations", _unique_count(snapshot.jobs_by_id.values(), "location")),
            ("Domains", _domain_count(snapshot.jobs_by_id.values())),
        ]
    )


def _render_filter_summary(snapshot: RunSnapshot) -> None:
    if not snapshot.evidence_available.get("filtering"):
        st.info("Filtering jobs…" if not snapshot.read_only else NOT_AVAILABLE)
        return
    _metric_row(
        [
            ("Eligible", snapshot.accepted_count or 0),
            ("Rejected", snapshot.rejected_count or 0),
        ]
    )
    categories = Counter()
    for item in snapshot.rejected_jobs:
        for reason in item.get("reasons", []) if isinstance(item, dict) else []:
            categories[rejection_category(str(reason))] += 1
    if categories:
        rows = "".join(
            f'<div class="compact-row"><span>{escape(name)}</span><b>{count}</b></div>'
            for name, count in categories.most_common(4)
        )
        st.markdown(f'<div class="compact-list">{rows}</div>', unsafe_allow_html=True)


def _render_ranking_summary(snapshot: RunSnapshot) -> None:
    if not snapshot.ranked_jobs:
        st.info("Scoring jobs…" if not snapshot.read_only else NOT_AVAILABLE)
        return
    st.caption("Scores calculated by deterministic Python code")
    _ranking_rows(snapshot, limit=8)


def _render_fit_summary(snapshot: RunSnapshot) -> None:
    if not snapshot.top_3_job_ids:
        st.info("Selecting Top 3…" if not snapshot.read_only else NOT_AVAILABLE)
        return
    for job_id in snapshot.top_3_job_ids:
        analysis = (
            snapshot.fit_analyses.get(job_id) or snapshot.artifact(job_id).fit_analysis
        )
        job = snapshot.job(job_id)
        with st.expander(_job_label(job, job_id), expanded=False):
            if not analysis:
                st.caption(
                    "Analyzing fit…" if not snapshot.read_only else NOT_AVAILABLE
                )
                continue
            _fit_content(analysis)


def _render_tailor_summary(snapshot: RunSnapshot) -> None:
    if not snapshot.top_3_job_ids:
        st.info("Tailoring resumes…" if not snapshot.read_only else NOT_AVAILABLE)
        return
    columns = st.columns(3, gap="medium")
    for column, job_id in zip(columns, snapshot.top_3_job_ids[:3], strict=False):
        with column:
            record = snapshot.artifact(job_id)
            job = snapshot.job(job_id)
            path = record.path("resume_after")
            with st.container(border=True):
                st.markdown(
                    f"**{escape(str(job.get('title') or job_id))}**",
                    unsafe_allow_html=True,
                )
                pages = record.page_counts.get("resume_after")
                st.caption(
                    f"{pages} page"
                    if pages == 1
                    else "Draft generated"
                    if path
                    else "Tailoring…"
                    if not snapshot.read_only
                    else NOT_AVAILABLE
                )
                _download(path, "Resume", f"tailor-{snapshot.run_id}-{job_id}")


def _render_read_only_review(snapshot: RunSnapshot) -> None:
    if snapshot.review_decisions:
        for job_id in snapshot.top_3_job_ids:
            job = snapshot.job(job_id)
            decision = snapshot.review_decisions.get(job_id, {})
            st.markdown(
                f'<div class="compact-row"><span>{escape(str(job.get("title") or job_id))}</span>'
                f"<b>{escape(str(decision.get('decision') or 'Recorded').title())}</b></div>",
                unsafe_allow_html=True,
            )
    else:
        st.info(NOT_AVAILABLE)


def _render_finalizing(snapshot: RunSnapshot) -> None:
    generated = sum(
        snapshot.artifact(job_id).path("cover_letter") is not None
        for job_id in snapshot.top_3_job_ids
    )
    st.info(f"Generating final files… {generated}/3 cover letters")


def _render_results(snapshot: RunSnapshot) -> None:
    st.markdown(
        '<h2 class="stage-title">Application packages</h2>', unsafe_allow_html=True
    )
    if not snapshot.top_3_job_ids:
        st.info(NOT_AVAILABLE)
        return

    scores = _score_lookup(snapshot.ranked_jobs)
    columns = st.columns(3, gap="medium")
    for column, job_id in zip(columns, snapshot.top_3_job_ids[:3], strict=False):
        with column:
            job = snapshot.job(job_id)
            record = snapshot.artifact(job_id)
            with st.container(border=True):
                score = scores.get(job_id)
                score_text = f"{score:.1f}" if score is not None else "—"
                st.markdown(
                    f"""
                    <div class="package-head">
                      <div><strong>{escape(str(job.get("title") or job_id))}</strong>
                      <span>{escape(str(job.get("company") or ""))}</span></div>
                      <b>{score_text}</b>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                _download(
                    record.path("resume_after"),
                    "Resume PDF",
                    f"resume-{snapshot.run_id}-{job_id}",
                )
                _download(
                    record.path("cover_letter"),
                    "Cover letter PDF",
                    f"letter-{snapshot.run_id}-{job_id}",
                )
                analysis = snapshot.fit_analyses.get(job_id) or record.fit_analysis
                with st.expander("Fit analysis"):
                    if analysis:
                        _fit_content(analysis)
                    else:
                        st.caption(NOT_AVAILABLE)
                with st.expander("Memory applied"):
                    _memory_evidence(snapshot)


def _fit_content(analysis: dict[str, Any]) -> None:
    sections = [
        ("Relevant experience", analysis.get("relevant_experience", [])),
        ("Aligned skills", analysis.get("aligned_skills", [])),
        ("Gaps", analysis.get("genuine_gaps", [])),
    ]
    for label, claims in sections:
        if not claims:
            continue
        st.markdown(f"**{label}**")
        for claim in claims[:4]:
            text = claim.get("claim") if isinstance(claim, dict) else str(claim)
            if text:
                st.markdown(f"- {text}")
    swap = analysis.get("project_swap")
    if isinstance(swap, dict):
        removed = swap.get("remove_project") or "Current project"
        added = swap.get("add_project") or "Replacement project"
        st.markdown(f"**Project swap:** {removed} → {added}")
        if swap.get("rationale"):
            st.caption(str(swap["rationale"]))


def _render_evidence(snapshot: RunSnapshot) -> None:
    has_any = any(
        [
            snapshot.evidence_available.get("filtering"),
            snapshot.ranked_jobs,
            snapshot.trace_events,
            snapshot.trace_url,
            snapshot.memory_facts,
            snapshot.review_history,
        ]
    )
    if not has_any:
        return

    with st.expander("Run Evidence"):
        _score_evidence(snapshot)
    with st.expander("Agent Trace"):
        _trace_evidence(snapshot)
    with st.expander("Filtering Log"):
        _filter_evidence(snapshot)
    with st.expander("Human Review & Rework"):
        _review_evidence(snapshot)
    with st.expander("Memory Update"):
        _memory_evidence(snapshot)


def _filter_evidence(snapshot: RunSnapshot) -> None:
    if not snapshot.evidence_available.get("filtering"):
        st.caption(NOT_AVAILABLE)
        return
    _metric_row(
        [
            ("Eligible", snapshot.accepted_count or 0),
            ("Rejected", snapshot.rejected_count or 0),
        ]
    )
    for item in snapshot.rejected_jobs:
        if not isinstance(item, dict):
            continue
        job = item.get("job", item)
        title = job.get("title") if isinstance(job, dict) else None
        reasons = item.get("reasons", [])
        st.markdown(f"**{title or item.get('job_id') or 'Rejected job'}**")
        st.caption("; ".join(str(reason) for reason in reasons) or NOT_AVAILABLE)


def _score_evidence(snapshot: RunSnapshot) -> None:
    if not snapshot.ranked_jobs:
        st.caption(NOT_AVAILABLE)
        return
    if snapshot.scoring_weights:
        formula = " + ".join(
            f"{weight:g}% {name.replace('_', ' ')}"
            for name, weight in snapshot.scoring_weights.items()
        )
        st.caption(formula)
    _ranking_rows(snapshot, limit=None)


def _memory_evidence(snapshot: RunSnapshot) -> None:
    if not snapshot.memory_facts:
        st.caption("No memory update recorded")
        return
    new_ids = set(snapshot.new_memory_fact_ids)
    facts = [
        fact
        for fact in snapshot.memory_facts
        if not new_ids or str(fact.get("fact_id", "")) in new_ids
    ]
    for fact in facts:
        text = (
            fact.get("canonical_value")
            or fact.get("fact")
            or fact.get("value")
            or fact.get("text")
        )
        if text:
            provenance = fact.get("provenance") or {}
            job_id = (
                provenance.get("related_job_id")
                if isinstance(provenance, dict)
                else None
            )
            label = f"**{job_id}:** " if job_id else ""
            st.markdown(f"- {label}{text}")


def _review_evidence(snapshot: RunSnapshot) -> None:
    if not snapshot.review_history:
        st.caption("Waiting for the first submitted review decision")
        return
    for entry in snapshot.review_history:
        action = str(entry.get("action") or "review").replace("_", " ").title()
        review_round = entry.get("review_round")
        affected = list(
            entry.get("affected_job_ids") or entry.get("rejected_job_ids") or []
        )
        st.markdown(f"**Round {review_round or '—'} · {action}**")
        feedback = str(entry.get("reviewer_feedback") or "").strip()
        if feedback:
            st.caption(f"Reviewer comment: “{feedback}”")
        decisions = entry.get("decisions") or {}
        if isinstance(decisions, dict):
            for job_id, decision in decisions.items():
                if not isinstance(decision, dict):
                    continue
                comment = str(decision.get("comment") or "").strip()
                if comment:
                    st.markdown(f"- **{job_id} feedback:** {comment}")
        if affected:
            st.markdown("- Rejected for rework: " + ", ".join(map(str, affected)))
        actions = entry.get("actions_taken") or {}
        if isinstance(actions, dict) and actions:
            st.markdown("- Revised resumes: " + ", ".join(map(str, actions.keys())))
        memory_writes = entry.get("memory_writes") or []
        for fact in memory_writes:
            if not isinstance(fact, dict):
                continue
            value = fact.get("canonical_value") or fact.get("value")
            if value:
                st.markdown(f"- Memory written: {value}")


def _trace_evidence(snapshot: RunSnapshot) -> None:
    if snapshot.trace_ingest_confirmed and snapshot.trace_url:
        st.link_button("Open trace", snapshot.trace_url)
    if not snapshot.trace_events:
        if not (snapshot.trace_ingest_confirmed and snapshot.trace_url):
            st.caption(NOT_AVAILABLE)
        return
    for event in snapshot.trace_events[-12:]:
        if not isinstance(event, dict):
            continue
        name = event.get("name") or event.get("event") or "Trace event"
        st.markdown(f"- {name}")


def _render_session_controls(snapshot: RunSnapshot) -> None:
    if snapshot.read_only:
        return
    _, right = st.columns([0.84, 0.16])
    with right:
        if st.button("Clear run", width="stretch"):
            st.session_state.pop("live_run_id", None)
            st.rerun()


def _ranking_rows(snapshot: RunSnapshot, *, limit: int | None) -> None:
    top_ids = set(snapshot.top_3_job_ids)
    items = snapshot.ranked_jobs if limit is None else snapshot.ranked_jobs[:limit]
    rows = []
    for rank, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        job = item.get("job", item)
        if not isinstance(job, dict):
            continue
        job_id = str(job.get("job_id") or "")
        score = item.get("score")
        score_text = f"{float(score):.1f}" if isinstance(score, (int, float)) else "—"
        selected = "<em>Selected</em>" if job_id in top_ids else ""
        rows.append(
            f'<div class="rank-line">'
            f"<span>{rank}</span>"
            f"<div><strong>{escape(str(job.get('title') or 'Untitled role'))}</strong>"
            f"<small>{escape(str(job.get('company') or ''))}</small></div>"
            f"{selected}<b>{score_text}</b>"
            "</div>"
        )
    st.markdown(
        '<div class="rank-list">' + "".join(rows) + "</div>", unsafe_allow_html=True
    )


def _download(path: Path | None, label: str, key: str) -> None:
    if path is None or not path.is_file():
        st.button(label, disabled=True, width="stretch", key=f"missing-{key}")
        return
    mime = (
        "application/pdf"
        if path.suffix.lower() == ".pdf"
        else "application/octet-stream"
    )
    st.download_button(
        label,
        data=path.read_bytes(),
        file_name=path.name,
        mime=mime,
        width="stretch",
        key=key,
    )


def _metric_row(items: list[tuple[str, Any]]) -> None:
    columns = st.columns(len(items), gap="small")
    for column, (label, value) in zip(columns, items, strict=True):
        with column:
            st.metric(label, value if value not in (None, "") else "—")


def _workflow_states(snapshot: RunSnapshot) -> dict[str, str]:
    records = [snapshot.artifact(job_id) for job_id in snapshot.top_3_job_ids]
    fit_count = sum(
        bool(
            snapshot.fit_analyses.get(job_id) or snapshot.artifact(job_id).fit_analysis
        )
        for job_id in snapshot.top_3_job_ids
    )
    tailored_count = sum(record.path("resume_after") is not None for record in records)
    final_count = sum(record.path("cover_letter") is not None for record in records)
    completed = {
        "analyze": bool(snapshot.jobs_loaded),
        "filter": bool(snapshot.evidence_available.get("filtering")),
        "rank": bool(
            snapshot.evidence_available.get("ranking")
            and len(snapshot.top_3_job_ids) == 3
        ),
        "fit": fit_count == 3,
        "tailor": tailored_count == 3,
        "review": snapshot.human_pause_used == 1 and not snapshot.is_waiting_for_review,
        "final": snapshot.status == "COMPLETED" and final_count == 3,
    }
    current = _current_workflow_key(snapshot, completed)
    return {
        key: "complete" if completed[key] else "active" if key == current else "pending"
        for _, key in WORKFLOW
    }


def _current_workflow_key(
    snapshot: RunSnapshot,
    completed: dict[str, bool] | None = None,
) -> str:
    completed = completed or {key: False for _, key in WORKFLOW}
    if snapshot.is_waiting_for_review:
        return "review"
    phase_map = {
        "INITIALIZE": "analyze",
        "ANALYZE": "analyze",
        "FILTER": "filter",
        "SCORE": "rank",
        "FIT_ANALYSIS": "fit",
        "TAILOR": "tailor",
        "HUMAN_REVIEW": "review",
        "REVISION": "review",
        "COVER_LETTERS": "final",
        "COMPLETE": "final",
    }
    preferred = phase_map.get(snapshot.phase)
    if preferred and not completed.get(preferred, False):
        return preferred
    for _, key in WORKFLOW:
        if not completed.get(key, False):
            return key
    return "final"


def _score_lookup(ranked_jobs: list[dict[str, Any]]) -> dict[str, float]:
    result: dict[str, float] = {}
    for item in ranked_jobs:
        if not isinstance(item, dict):
            continue
        job = item.get("job", item)
        if not isinstance(job, dict) or not job.get("job_id"):
            continue
        score = item.get("score")
        if isinstance(score, (int, float)):
            result[str(job["job_id"])] = float(score)
    return result


def _unique_count(jobs: Any, key: str) -> int:
    return len(
        {str(job.get(key)) for job in jobs if isinstance(job, dict) and job.get(key)}
    )


def _domain_count(jobs: Any) -> int:
    values = set()
    for job in jobs:
        if not isinstance(job, dict):
            continue
        value = job.get("industry_domain") or job.get("industry") or job.get("domain")
        if value:
            values.add(str(value))
    return len(values)


def _job_label(job: dict[str, Any], job_id: str) -> str:
    title = job.get("title") or job_id
    company = job.get("company")
    return f"{title} · {company}" if company else str(title)
