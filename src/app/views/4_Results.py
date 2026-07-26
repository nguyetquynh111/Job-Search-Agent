"""Final results page."""

from __future__ import annotations

import streamlit as st

from src.ui.components import (
    build_outputs_zip,
    download_pdf_button,
    observability_summary,
    render_artifact_errors,
    render_errors,
    render_page_header,
    render_pdf_preview,
    render_sidebar,
    render_status_pill,
    review_status_for_job,
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
top_job_ids = list(state.get("top_3_job_ids", []))
source_resume_path = state.get("resume_path") or state.get("input_paths", {}).get(
    "resume_path",
    "",
)

st.subheader("Run observability")
summary = observability_summary(state, bundle.tracer if bundle is not None else None)
trace_url = state.get("trace_url")
trace_column, metrics_column = st.columns([1, 2.3], gap="large")
with trace_column:
    with st.container(border=True):
        st.caption("Trace status")
        st.markdown(f"**{summary['trace_status']}**")
        if trace_url:
            st.link_button(
                "Open public run trace",
                trace_url,
                type="primary",
                use_container_width=True,
            )
        else:
            st.caption("A trace link was not returned for this run.")
with metrics_column:
    metric_columns = st.columns(4)
    metric_columns[0].metric(
        "LLM calls",
        summary["llm_calls"] if summary["llm_calls"] is not None else "Unavailable",
    )
    metric_columns[1].metric("Tool calls", summary["tool_calls"])
    metric_columns[2].metric("Memory writes", summary["memory_writes"])
    metric_columns[3].metric("Human review", summary["human_review_status"])
    st.caption(f"Run ID: {summary['run_id'] or 'Unavailable'}")

approved_ids = state.get("approved_job_ids", [])
if approved_ids:
    st.subheader("Application packages")
    zip_payload = build_outputs_zip(
        jobs,
        tailoring,
        cover_letters,
        approved_ids,
    )
    if zip_payload:
        st.download_button(
            "Download all outputs as ZIP",
            data=zip_payload,
            file_name=f"{state.get('run_id') or 'job-search'}-outputs.zip",
            mime="application/zip",
            type="primary",
        )

    result_job_ids = list(approved_ids)
    selector_key = "results_selected_job_id"
    if st.session_state.get(selector_key) not in result_job_ids:
        st.session_state[selector_key] = result_job_ids[0]
    selected_job_id = st.selectbox(
        "Selected application",
        options=result_job_ids,
        format_func=lambda value: (
            f"{jobs.get(value, {}).get('company', value)} · "
            f"{jobs.get(value, {}).get('title', 'Application')}"
        ),
        key=selector_key,
    )
    resume = tailoring.get(selected_job_id, {})
    letter = cover_letters.get(selected_job_id, {})
    job = jobs.get(selected_job_id, {})
    with st.container(border=True):
        heading_column, status_column = st.columns([3, 1])
        with heading_column:
            st.subheader(
                f"{job.get('title', 'Application')} at "
                f"{job.get('company', selected_job_id)}"
            )
        with status_column:
            status_label, status_tone = review_status_for_job(
                state,
                selected_job_id,
            )
            render_status_pill(status_label, status_tone)

        before_tab, resume_tab, letter_tab = st.tabs(
            ["Before tailoring", "Final resume", "Final cover letter"]
        )
        with before_tab:
            render_pdf_preview(source_resume_path, "Resume before tailoring")
        with resume_tab:
            if resume:
                render_pdf_preview(
                    resume.get("output_pdf_path", ""),
                    "Approved tailored resume",
                )
                download_pdf_button(
                    resume.get("output_pdf_path", ""),
                    "Download resume",
                )
                render_artifact_errors(resume.get("errors"), "resume")
            else:
                st.info("The approved resume is not available.")
        with letter_tab:
            if letter:
                render_pdf_preview(
                    letter.get("output_pdf_path", ""),
                    "Final cover letter",
                )
                download_pdf_button(
                    letter.get("output_pdf_path", ""),
                    "Download cover letter",
                )
                render_artifact_errors(letter.get("errors"), "cover letter")
            else:
                st.info(
                    "No final cover letter exists because this role has not "
                    "completed the review workflow."
                )
else:
    st.markdown(
        '<div class="empty-state">Final files will appear here after every resume is approved.</div>',
        unsafe_allow_html=True,
    )
    if st.button("Open review", type="primary"):
        st.switch_page("views/3_Review.py")

missing_output_ids = [
    job_id
    for job_id in top_job_ids
    if job_id not in cover_letters or job_id not in tailoring
]
if missing_output_ids:
    st.subheader("Roles without final output")
    review_history = state.get("review_history", [])
    latest_decisions = (
        review_history[-1].get("decisions", {}) if review_history else {}
    )
    for job_id in missing_output_ids:
        job = jobs.get(job_id, {})
        decision = latest_decisions.get(job_id, {})
        if decision.get("decision") == "reject":
            reason = decision.get("comment") or "Revision was requested."
        elif state.get("status") == "FAILED_REVIEW":
            reason = "The maximum review revision count was reached."
        else:
            reason = "This role has not completed the review and generation stages."
        with st.container(border=True):
            st.markdown(
                f"**{job.get('title', 'Application')} at "
                f"{job.get('company', job_id)}**"
            )
            render_status_pill("No final output", "warning")
            st.caption(reason)

with st.expander("Technical details"):
    st.write("Review history")
    review_history = state.get("review_history", [])
    if review_history:
        st.json(review_history, expanded=False)
    else:
        st.caption("No review decisions are stored.")
    st.write("Internal identifiers")
    st.json(
        {
            "run_id": state.get("run_id"),
            "thread_id": state.get("thread_id"),
            "trace_id": state.get("trace_id"),
            "trace_url": trace_url,
        },
        expanded=False,
    )
    st.write("Artifact paths")
    st.json(
        {
            job_id: {
                "resume_tex": tailoring.get(job_id, {}).get("output_tex_path"),
                "resume_pdf": tailoring.get(job_id, {}).get("output_pdf_path"),
                "cover_letter_tex": cover_letters.get(job_id, {}).get(
                    "output_tex_path"
                ),
                "cover_letter_pdf": cover_letters.get(job_id, {}).get(
                    "output_pdf_path"
                ),
            }
            for job_id in top_job_ids
        },
        expanded=False,
    )
