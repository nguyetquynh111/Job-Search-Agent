"""Execution status page."""

from __future__ import annotations

import streamlit as st

from src.ui.components import (
    format_phase,
    format_status,
    group_rejected_jobs,
    render_agent_decisions,
    render_errors,
    render_page_header,
    render_ranked_job_card,
    render_sidebar,
    render_tool_activity,
)
from src.ui.graph_resource import configured_graph_bundle
from src.ui.session import ensure_session_defaults

ensure_session_defaults(st.session_state)
bundle = None
runtime_error: Exception | None = None
try:
    bundle = configured_graph_bundle()
except Exception as exc:
    runtime_error = exc

state = render_sidebar(
    bundle.app if bundle is not None else None,
    bundle.tracer if bundle is not None else None,
)

render_page_header(
    "Step 2 of 4",
    "Run progress",
    "Follow each decision from job filtering through resume generation.",
)
render_errors(state)

if runtime_error is not None:
    st.warning(f"The search workflow is not available in this environment yet: {runtime_error}")

if not state:
    st.markdown(
        '<div class="empty-state">No run has started yet. Set up your input files to begin.</div>',
        unsafe_allow_html=True,
    )
    if st.button("Set up search", type="primary"):
        st.switch_page("views/1_Input.py")
    st.stop()

phase_order = [
    "INITIALIZE",
    "FILTER",
    "SCORE",
    "FIT_ANALYSIS",
    "TAILOR",
    "HUMAN_REVIEW",
    "COVER_LETTERS",
    "COMPLETE",
]
phase = state.get("phase", "INITIALIZE")
progress = max(0, phase_order.index(phase) if phase in phase_order else 0) / (len(phase_order) - 1)
st.progress(progress)
st.caption(f"Current phase: {format_phase(phase)}")

jobs = state.get("jobs", [])
ranked = state.get("ranked_jobs", [])
metric_columns = st.columns(4)
metric_columns[0].metric("Status", format_status(state.get("status")))
metric_columns[1].metric("Jobs loaded", len(jobs))
metric_columns[2].metric("Jobs ranked", len(ranked))
metric_columns[3].metric("Review round", state.get("revision_round", 0))

decisions = state.get("agent_decisions", [])
history = state.get("tool_history", [])
rejected = state.get("rejected_jobs", [])

ranked_tab, decisions_tab, activity_tab, filtered_tab = st.tabs(
    ["Ranked jobs", "Agent decisions", "Tool activity", "Filtered out"]
)

with ranked_tab:
    if ranked:
        st.caption(
            "Scores are calculated by deterministic code. Component explanations "
            "below come directly from the scoring tool."
        )
        for ranked_item in ranked:
            render_ranked_job_card(
                ranked_item,
                state.get("top_3_job_ids", []),
            )
    else:
        st.caption("Ranked jobs will appear here as the run progresses.")

with decisions_tab:
    st.caption(
        "These are concise controller decisions about what should happen next. "
        "Deterministic tool results are shown separately."
    )
    render_agent_decisions(decisions)

with activity_tab:
    st.caption(
        "Structured execution records show tool inputs and outputs without "
        "presenting them as LLM reasoning."
    )
    render_tool_activity(history, state.get("errors", []))

with filtered_tab:
    if rejected:
        grouped = group_rejected_jobs(rejected)
        st.subheader("Rejection reasons")
        summary_columns = st.columns(min(4, max(1, len(grouped))))
        for index, (reason, jobs_for_reason) in enumerate(grouped.items()):
            summary_columns[index % len(summary_columns)].metric(
                reason,
                len(jobs_for_reason),
            )
        st.caption(
            "Reasons are preserved exactly as returned by the Filtering Tool. "
            "A role may appear in more than one group."
        )
        for reason, jobs_for_reason in grouped.items():
            with st.expander(f"{reason} · {len(jobs_for_reason)}"):
                for rejected_item in jobs_for_reason:
                    job = rejected_item.get("job", {})
                    st.markdown(
                        f"**{job.get('title', 'Role')}** at "
                        f"{job.get('company', 'Company')} · "
                        f"`{job.get('job_id', '—')}`"
                    )
                    st.caption(" · ".join(rejected_item.get("reasons", [])))
    else:
        st.caption("No jobs have been filtered out.")
