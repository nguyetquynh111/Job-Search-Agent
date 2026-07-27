"""Global run header, pipeline status, and architecture evidence."""

from __future__ import annotations

from html import escape
from typing import Any

import streamlit as st

from app.services.run_state import NOT_AVAILABLE, RunSnapshot

PIPELINE_STAGES = [
    ("Input", "INITIALIZE"),
    ("Filter", "FILTER"),
    ("Score", "SCORE"),
    ("Fit", "FIT_ANALYSIS"),
    ("Tailor", "TAILOR"),
    ("Human Review", "HUMAN_REVIEW"),
    ("Revision", "REVISION"),
    ("Cover Letters", "COVER_LETTERS"),
]

PHASE_INDEX = {phase: index for index, (_, phase) in enumerate(PIPELINE_STAGES)}
PHASE_LABELS = {
    "INITIALIZE": "Candidate inputs",
    "FILTER": "Job filtering",
    "SCORE": "Deterministic scoring",
    "FIT_ANALYSIS": "Top 3 fit analysis",
    "TAILOR": "Resume tailoring",
    "HUMAN_REVIEW": "Human review",
    "REVISION": "Revision and memory propagation",
    "COVER_LETTERS": "Cover letters",
    "COMPLETE": "Final outputs",
    "ERROR": "Failed",
}


def pipeline_states(snapshot: RunSnapshot) -> list[dict[str, str]]:
    """Calculate display states without advancing or simulating the workflow."""

    if snapshot.status == "COMPLETED" or snapshot.phase == "COMPLETE":
        return [
            {"label": label, "phase": phase, "state": "Completed"}
            for label, phase in PIPELINE_STAGES
        ]
    current = PHASE_INDEX.get(snapshot.phase, 0)
    result: list[dict[str, str]] = []
    for index, (label, phase) in enumerate(PIPELINE_STAGES):
        if index < current:
            state = "Completed"
        elif index > current:
            state = "Pending"
        elif snapshot.status in {"FAILED", "FAILED_REVIEW"}:
            state = "Failed"
        elif phase == "HUMAN_REVIEW" and snapshot.status == "WAITING_FOR_REVIEW":
            state = "Waiting for human review"
        else:
            state = "Running"
        result.append({"label": label, "phase": phase, "state": state})
    return result


def render_global_header(snapshot: RunSnapshot) -> None:
    """Render identity, status, metrics, and the horizontal stage indicator."""

    status_tone = _status_tone(snapshot.status)
    st.markdown(
        f"""
        <section class="hero">
          <div>
            <div class="eyebrow">Agent Run Control Center</div>
            <h1>Job Search Agent</h1>
            <p>Evidence-backed orchestration from candidate inputs to final application artifacts.</p>
          </div>
          <div class="hero__run">
            <span class="status status--{status_tone}">{escape(_status_label(snapshot.status))}</span>
            <span class="hero__run-id">{escape(snapshot.run_id)}</span>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    metrics: list[tuple[str, Any, str]] = [
        ("Current stage", PHASE_LABELS.get(snapshot.phase, snapshot.phase), "Repository state"),
        ("Jobs loaded", snapshot.jobs_loaded, "Run state or current repository input"),
        ("Accepted", snapshot.accepted_count, "Filtering artifact"),
        ("Rejected", snapshot.rejected_count, "Filtering artifact"),
        (
            "Human pauses",
            f"{snapshot.human_pause_used}/1"
            if snapshot.human_pause_used is not None
            else NOT_AVAILABLE,
            "Single permitted interrupt",
        ),
        (
            "Memory",
            f"{len(snapshot.memory_facts)} facts"
            if snapshot.evidence_available.get("memory")
            else NOT_AVAILABLE,
            "Durable memory artifact",
        ),
        ("Trace", snapshot.trace_status, "Public link or local events"),
    ]
    columns = st.columns([1.25, 0.8, 0.8, 0.8, 0.9, 0.9, 1.25])
    for column, (label, value, help_text) in zip(columns, metrics, strict=True):
        with column:
            st.metric(label, _compact_metric(value), help=help_text)
    render_horizontal_pipeline(snapshot)


def render_horizontal_pipeline(snapshot: RunSnapshot) -> None:
    items = []
    for item in pipeline_states(snapshot):
        modifier = _state_modifier(item["state"])
        items.append(
            f"""
            <div class="pipeline-step pipeline-step--{modifier}">
              <span class="pipeline-step__dot"></span>
              <span>{escape(item["label"])}</span>
            </div>
            """
        )
    st.markdown(
        '<div class="pipeline-strip">' + "".join(items) + "</div>",
        unsafe_allow_html=True,
    )


def render_sidebar_pipeline(snapshot: RunSnapshot) -> None:
    """Render all stage states in the left navigation rail."""

    st.markdown('<div class="sidebar-label">Pipeline state</div>', unsafe_allow_html=True)
    rows = []
    for item in pipeline_states(snapshot):
        modifier = _state_modifier(item["state"])
        rows.append(
            f"""
            <div class="sidebar-stage sidebar-stage--{modifier}">
              <span class="sidebar-stage__indicator"></span>
              <span class="sidebar-stage__name">{escape(item["label"])}</span>
              <span class="sidebar-stage__state">{escape(_short_state(item["state"]))}</span>
            </div>
            """
        )
    st.markdown("".join(rows), unsafe_allow_html=True)


def render_architecture_evidence(snapshot: RunSnapshot) -> None:
    """Show code-contract claims separately from selected-run evidence."""

    st.markdown(
        """
        <div class="section-heading">
          <div>
            <div class="eyebrow">Repository contract</div>
            <h2>One agent. Deterministic controls.</h2>
          </div>
          <span class="source-tag">src/agent/graph.py</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    claims = [
        (
            "Single LLM agent",
            "The graph module declares one agent and records model-selected tool calls.",
            bool(snapshot.agent_decisions)
            or any(event.get("observation_type") == "GENERATION" for event in snapshot.trace_events),
        ),
        (
            "Deterministic ranking",
            "Filtering and weighted scoring run in Python tools, not in the LLM.",
            bool(snapshot.scoring_weights),
        ),
        (
            "Automatic Top 3",
            "The scoring tool selects the first three ranked job IDs; the UI has no reorder control.",
            snapshot.top_selection_source
            in {"live_state", "manifest", "ranked_artifact"},
        ),
        (
            "One human pause",
            "LangGraph interrupts once after all three tailored drafts are ready.",
            bool(snapshot.interrupt_payload or snapshot.review_history),
        ),
        (
            "Two revision rounds maximum",
            "The existing review contract enforces MAX_REVISION_ROUNDS = 2.",
            True,
        ),
        (
            "Letters after approval",
            "The graph rejects cover-letter dispatch before review decisions exist.",
            bool(snapshot.cover_letter_results and snapshot.review_decisions),
        ),
    ]
    columns = st.columns(3)
    for index, (title, description, run_evidence) in enumerate(claims):
        with columns[index % 3]:
            label = "Run evidence present" if run_evidence else "Code contract only"
            tone = "verified" if run_evidence else "contract"
            st.markdown(
                f"""
                <div class="contract-card">
                  <div class="contract-card__top">
                    <span class="contract-card__title">{escape(title)}</span>
                    <span class="evidence-dot evidence-dot--{tone}"></span>
                  </div>
                  <p>{escape(description)}</p>
                  <span class="micro-label">{escape(label)}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _compact_metric(value: Any) -> str:
    if value is None:
        return "—"
    if value == NOT_AVAILABLE:
        return "Unavailable"
    return str(value)


def _status_label(status: str) -> str:
    return {
        "WAITING_FOR_REVIEW": "Waiting for human review",
        "INCOMPLETE_ARTIFACTS": "Incomplete artifact set",
        "COMPLETED": "Run completed",
        "RUNNING": "Run in progress",
        "CREATED": "Run created",
        "FAILED": "Run failed",
        "FAILED_REVIEW": "Review failed",
    }.get(status, status.replace("_", " ").title())


def _status_tone(status: str) -> str:
    if status == "COMPLETED":
        return "success"
    if status in {"FAILED", "FAILED_REVIEW"}:
        return "danger"
    if status == "WAITING_FOR_REVIEW":
        return "warning"
    return "neutral"


def _state_modifier(state: str) -> str:
    return {
        "Completed": "complete",
        "Running": "running",
        "Failed": "failed",
        "Waiting for human review": "waiting",
    }.get(state, "pending")


def _short_state(state: str) -> str:
    return "Waiting" if state == "Waiting for human review" else state

