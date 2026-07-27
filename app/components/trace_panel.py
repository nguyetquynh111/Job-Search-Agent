"""Trace tree and structured observability evidence."""

from __future__ import annotations

from html import escape
from typing import Any

import streamlit as st

from app.services.run_state import NOT_AVAILABLE, RunSnapshot


def render_trace_panel(snapshot: RunSnapshot) -> None:
    st.markdown(
        """
        <div class="section-heading">
          <div><div class="eyebrow">End-to-end evidence</div><h2>Trace</h2></div>
          <span class="source-tag">agent.run</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    identity, status, events = st.columns([1.1, 1.1, 0.8])
    identity.metric("Trace ID", snapshot.trace_id or "Unavailable")
    status.metric("Trace status", snapshot.trace_status)
    events.metric("Recorded events", len(snapshot.trace_events))
    if snapshot.trace_ingest_confirmed and snapshot.trace_url:
        st.link_button(
            "Open public trace",
            snapshot.trace_url,
            type="primary",
            width="content",
        )
    else:
        st.caption("Public trace link: " + NOT_AVAILABLE)
    if snapshot.trace_export_error:
        st.error(snapshot.trace_export_error)
        if snapshot.trace_debug_status:
            st.caption("Trace diagnostics: " + snapshot.trace_debug_status)
    if not snapshot.trace_events:
        st.info(NOT_AVAILABLE)
        _render_expected_shape()
        return
    st.markdown("### Recorded span tree")
    depths = _event_depths(snapshot.trace_events)
    for index, event in enumerate(snapshot.trace_events):
        _render_event(event, depths[index], index)
    if snapshot.agent_decisions:
        st.markdown("### Agent tool-selection records")
        for decision in snapshot.agent_decisions:
            title = (
                f"{decision.get('phase', 'Phase unavailable')} → "
                f"{decision.get('selected_tool', 'Tool unavailable')}"
            )
            with st.expander(title):
                st.json(decision, expanded=1)
    st.caption(
        "Only structured prompts, model metadata, tool decisions, inputs, outputs, and "
        "application rationale recorded by the repository are shown. Hidden chain-of-thought is not displayed."
    )


def _render_event(event: dict[str, Any], depth: int, index: int) -> None:
    status = str(event.get("status") or event.get("metadata", {}).get("status") or "OK")
    tone = "error" if status == "ERROR" else "ok"
    name = str(event.get("name") or "Unnamed observation")
    kind = str(event.get("observation_type") or "SPAN")
    duration = event.get("duration_ms")
    duration_text = (
        f"{float(duration):.0f} ms" if isinstance(duration, (int, float)) else "—"
    )
    indent = min(depth, 5) * 22
    st.markdown(
        f"""
        <div class="trace-row trace-row--{tone}" style="margin-left:{indent}px">
          <span class="trace-row__rail"></span>
          <div><strong>{escape(name)}</strong><span>{escape(kind)}</span></div>
          <span>{escape(duration_text)}</span>
          <span class="status status--{'danger' if tone == 'error' else 'success'}">{escape(status)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander(f"Inspect {name}", expanded=False):
        metadata = event.get("metadata") or {}
        model = event.get("model")
        parameters = event.get("model_parameters") or {}
        usage = event.get("usage") or {}
        if model:
            st.markdown(f"**Model**  \n{model}")
        if parameters:
            st.markdown("**Model / prompt configuration**")
            st.json(parameters, expanded=1)
        if usage:
            st.markdown("**Usage**")
            st.json(usage, expanded=1)
        st.markdown("**Input**")
        _json_or_unavailable(event.get("input"))
        st.markdown("**Output**")
        _json_or_unavailable(event.get("output"))
        st.markdown("**Attributes**")
        _json_or_unavailable(metadata)


def _event_depths(events: list[dict[str, Any]]) -> list[int]:
    by_id = {
        str(event.get("observation_id")): event
        for event in events
        if event.get("observation_id")
    }
    depths: list[int] = []
    for event in events:
        depth = 0
        parent = event.get("parent_observation_id")
        seen: set[str] = set()
        while parent and str(parent) in by_id and str(parent) not in seen:
            seen.add(str(parent))
            depth += 1
            parent = by_id[str(parent)].get("parent_observation_id")
        depths.append(depth)
    return depths


def _json_or_unavailable(value: Any) -> None:
    if value in (None, "", [], {}):
        st.caption(NOT_AVAILABLE)
    else:
        st.json(value, expanded=1)


def _render_expected_shape() -> None:
    with st.expander("Workflow trace shape defined by the repository", expanded=False):
        st.code(
            """agent.run
├── initialize.load_inputs
├── filtering_tool
├── scoring_tool
├── fit_analysis
│   ├── job_1
│   ├── job_2
│   └── job_3
├── resume_tailoring
│   ├── job_1
│   ├── job_2
│   └── job_3
├── human_review_pause
│   └── memory.write
└── cover_letter_generation""",
            language="text",
        )
        st.caption(
            "This is the repository's expected workflow shape, not an invented trace for the selected run."
        )
