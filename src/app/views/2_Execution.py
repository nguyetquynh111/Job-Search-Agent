"""Execution status page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.ui.components import (
    format_phase,
    format_status,
    render_errors,
    render_page_header,
    render_sidebar,
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
        frame = pd.DataFrame(
            [
                {
                    "Job ID": item["job"]["job_id"],
                    "Role": item["job"]["title"],
                    "Company": item["job"]["company"],
                    "Match score": item["score"],
                    "Selected": item["job"]["job_id"] in state.get("top_3_job_ids", []),
                }
                for item in ranked
            ]
        )
        st.dataframe(frame, use_container_width=True, hide_index=True)
    else:
        st.caption("Ranked jobs will appear here as the run progresses.")

with decisions_tab:
    if decisions:
        st.dataframe(
            [
                {
                    "Phase": format_phase(item.get("phase")),
                    "Tool": str(item.get("selected_tool", "")).replace("_", " "),
                    "Summary": item.get("decision_summary"),
                }
                for item in decisions
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("Agent decisions will appear here as the run progresses.")

with activity_tab:
    if history:
        st.dataframe(
            [
                {
                    "Phase": format_phase(item.get("phase")),
                    "Tool": str(item.get("tool", "")).replace("_", " "),
                    "Status": "Complete",
                }
                for item in history
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("Tool activity will appear here as the run progresses.")

with filtered_tab:
    if rejected:
        st.dataframe(
            [
                {
                    "Job ID": item["job"]["job_id"],
                    "Role": item["job"]["title"],
                    "Company": item["job"]["company"],
                    "Reason": "; ".join(item["reasons"]),
                }
                for item in rejected
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("No jobs have been filtered out.")
