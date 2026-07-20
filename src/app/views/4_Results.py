"""Final results page."""

from __future__ import annotations

import streamlit as st

from src.ui.components import (
    download_pdf_button,
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
    "Step 4 of 4",
    "Application package",
    "Download the approved resume and cover letter prepared for each role.",
)
render_errors(state)

if runtime_error is not None:
    st.warning(f"The search workflow is not available in this environment yet: {runtime_error}")

if state.get("status") == "FAILED_REVIEW":
    st.error("The review limit was reached, so cover letters were not generated.")

tailoring = state.get("tailoring_results", {})
cover_letters = state.get("cover_letter_results", {})
jobs = {job["job_id"]: job for job in state.get("jobs", [])}

approved_ids = state.get("approved_job_ids", [])
if approved_ids:
    st.subheader("Ready to download")
    for job_id in approved_ids:
        resume = tailoring.get(job_id)
        letter = cover_letters.get(job_id)
        if not resume and not letter:
            continue
        job = jobs.get(job_id, {})
        with st.container(border=True):
            st.subheader(f"{job.get('title', 'Application')} at {job.get('company', job_id)}")
            st.caption(f"Job ID: {job_id}")
            resume_column, letter_column = st.columns(2)
            with resume_column:
                if resume:
                    download_pdf_button(resume["output_pdf_path"], "Download resume")
                    st.caption(resume["output_tex_path"])
                else:
                    st.caption("The approved resume is not available.")
            with letter_column:
                if letter:
                    download_pdf_button(letter["output_pdf_path"], "Download cover letter")
                    st.caption(letter["output_tex_path"])
                else:
                    st.caption("The cover letter is not available yet.")
else:
    st.markdown(
        '<div class="empty-state">Final files will appear here after every resume is approved.</div>',
        unsafe_allow_html=True,
    )
    if st.button("Open review", type="primary"):
        st.switch_page("views/3_Review.py")

with st.expander("Run details"):
    st.write("Review history")
    review_history = state.get("review_history", [])
    if review_history:
        st.json(review_history, expanded=False)
    else:
        st.caption("No review decisions are stored.")

    trace_url = state.get("trace_url")
    if trace_url:
        st.link_button("Open run trace", trace_url)
    else:
        st.caption(state.get("langfuse_status", "Tracing is not initialized."))

    st.caption(f"Output folder: outputs")
