"""Human review page."""

from __future__ import annotations

import streamlit as st

from src.ui.components import (
    download_pdf_button,
    render_errors,
    render_page_header,
    render_sidebar,
)
from src.ui.graph_resource import configured_graph_bundle
from src.ui.session import ensure_session_defaults, resume_graph_run

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
    "Step 3 of 4",
    "Review tailored resumes",
    "Approve each draft or request a focused revision. Cover letters are created only after every resume is approved.",
)
render_errors(state)

if runtime_error is not None:
    st.warning(f"The search workflow is not available in this environment yet: {runtime_error}")

payload = st.session_state.get("interrupt_payload") or state.get("interrupt_payload")
if not payload:
    st.markdown(
        '<div class="empty-state">There are no drafts waiting for review.</div>',
        unsafe_allow_html=True,
    )
    if st.button("View run progress"):
        st.switch_page("views/2_Execution.py")
    st.stop()

resumes = payload.get("resumes", {})
revision_round = int(payload.get("revision_round", 0))
if payload.get("is_initial_review", revision_round == 0):
    st.caption(
        "Initial review · "
        f"up to {payload.get('max_revision_rounds', 2)} revision rounds"
    )
else:
    st.caption(
        f"Revision round {revision_round} of "
        f"{payload.get('max_revision_rounds', 2)}"
    )
tabs = st.tabs(
    [
        f"{resume['company']} · {resume['job_title']}"
        for resume in resumes.values()
    ]
)
decisions: dict[str, dict[str, str]] = {}
for tab, (job_id, resume) in zip(tabs, resumes.items(), strict=True):
    with tab:
        st.subheader(f"{resume['job_title']} at {resume['company']}")
        detail_column, decision_column = st.columns([1.55, 1], gap="large")
        with detail_column:
            fit = resume.get("fit_analysis", {})
            with st.expander("Fit analysis", expanded=True):
                st.json(fit, expanded=False)
            project_swap = fit.get("project_swap")
            if project_swap:
                with st.expander("Recommended project swap", expanded=True):
                    st.json(project_swap, expanded=True)
            with st.expander("Resume changes", expanded=True):
                changes = resume.get("change_log", [])
                if changes:
                    st.dataframe(changes, use_container_width=True, hide_index=True)
                else:
                    st.caption("No changes were recorded for this draft.")

        with decision_column:
            with st.container(border=True):
                st.subheader("Your decision")
                download_pdf_button(resume["resume_pdf_path"], "Download resume")
                decision = st.selectbox(
                    "Review decision",
                    options=["", "approve", "reject"],
                    format_func={
                        "": "Choose an option",
                        "approve": "Approve",
                        "reject": "Request changes",
                    }.get,
                    key=f"decision_{job_id}_{payload.get('review_round')}",
                )
                comment = st.text_area(
                    "Feedback",
                    help="Feedback is required when you request changes.",
                    placeholder="Describe the specific change you want…",
                    key=f"comment_{job_id}_{payload.get('review_round')}",
                    height=140,
                )
        decisions[job_id] = {"decision": decision, "comment": comment}

st.divider()
if st.button("Submit all decisions", type="primary", use_container_width=True):
    missing = [job_id for job_id, item in decisions.items() if not item["decision"]]
    if missing:
        st.error(f"Choose a decision for: {', '.join(missing)}.")
    elif any(
        item["decision"] == "reject" and not item["comment"].strip()
        for item in decisions.values()
    ):
        st.error("Add feedback for every resume that needs changes.")
    else:
        feedback = {
            job_id: {"decision": item["decision"], "comment": item["comment"]}
            for job_id, item in decisions.items()
        }
        try:
            result = resume_graph_run(bundle.app, st.session_state, feedback)
            if result.get("__interrupt__"):
                st.toast("Feedback submitted. Revised resumes are ready.")
            elif result.get("status") == "COMPLETED":
                st.toast("All resumes approved. Final files are ready.")
            else:
                st.info("Your decisions were submitted.")
            st.rerun()
        except Exception as exc:
            st.error(f"Unable to submit your decisions: {exc}")
