"""Human review page."""

from __future__ import annotations

import streamlit as st

from src.ui.components import (
    build_evidence_lookup,
    download_pdf_button,
    render_errors,
    render_fit_analysis,
    render_page_header,
    render_pdf_preview,
    render_resume_diffs,
    render_review_gate_banner,
    render_sidebar,
    render_status_pill,
    review_status_for_job,
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
render_review_gate_banner()
render_errors(state)

if runtime_error is not None:
    st.warning(f"The search workflow is not available in this environment yet: {runtime_error}")

payload = st.session_state.get("interrupt_payload") or state.get("interrupt_payload")
top_job_ids = list(state.get("top_3_job_ids", []))
jobs = {job["job_id"]: job for job in state.get("jobs", [])}
tailoring = state.get("tailoring_results", {})
fit_analyses = state.get("fit_analyses", {})

if payload:
    resumes = payload.get("resumes", {})
else:
    resumes = {
        job_id: {
            "job_title": jobs.get(job_id, {}).get("title", "Application"),
            "company": jobs.get(job_id, {}).get("company", job_id),
            "fit_analysis": fit_analyses.get(job_id, {}),
            "change_log": tailoring.get(job_id, {}).get("change_log", []),
            "resume_pdf_path": tailoring.get(job_id, {}).get("output_pdf_path", ""),
        }
        for job_id in top_job_ids
        if job_id in tailoring
    }

if not resumes:
    st.markdown(
        '<div class="empty-state">There are no drafts waiting for review.</div>',
        unsafe_allow_html=True,
    )
    if st.button("View run progress"):
        st.switch_page("views/2_Execution.py")
    st.stop()

revision_round = int((payload or {}).get("revision_round", state.get("revision_round", 0)))
max_revision_rounds = int((payload or {}).get("max_revision_rounds", 2))
review_complete = bool(state.get("review_history")) or state.get("status") in {
    "COMPLETED",
    "FAILED_REVIEW",
}
controls_locked = (
    not payload
    or state.get("status") != "WAITING_FOR_REVIEW"
    or review_complete
    or revision_round >= max_revision_rounds
)

if payload and payload.get("is_initial_review", revision_round == 0):
    st.caption(
        "Initial review · "
        f"up to {max_revision_rounds} revision rounds"
    )
elif payload:
    st.caption(f"Revision round {revision_round} of {max_revision_rounds}")
else:
    st.success(
        "The human-review gate is complete. Decisions are locked and the workflow "
        "has continued without another pause."
    )

status_columns = st.columns(len(resumes))
for status_column, (job_id, resume) in zip(
    status_columns,
    resumes.items(),
    strict=True,
):
    with status_column:
        with st.container(border=True):
            st.caption(resume["company"])
            st.markdown(f"**{resume['job_title']}**")
            status_label, status_tone = review_status_for_job(
                state,
                job_id,
                has_active_payload=bool(payload),
                max_revision_rounds=max_revision_rounds,
            )
            render_status_pill(status_label, status_tone)

job_ids = list(resumes)
selector_key = "review_selected_job_id"
if st.session_state.get(selector_key) not in job_ids:
    st.session_state[selector_key] = job_ids[0]
selected_job_id = st.selectbox(
    "Selected application",
    options=job_ids,
    format_func=lambda value: (
        f"{resumes[value]['company']} · {resumes[value]['job_title']}"
    ),
    key=selector_key,
)
resume = resumes[selected_job_id]
fit = resume.get("fit_analysis", {})
evidence_lookup = build_evidence_lookup(state)
tailored_result = tailoring.get(selected_job_id, {})
source_resume_path = state.get("resume_path") or state.get("input_paths", {}).get(
    "resume_path",
    "",
)

st.subheader(f"{resume['job_title']} at {resume['company']}")
preview_tab, fit_tab, changes_tab = st.tabs(
    ["Document preview", "Fit analysis", "Resume changes"]
)

with preview_tab:
    before_column, after_column = st.columns(2, gap="large")
    with before_column:
        render_pdf_preview(source_resume_path, "Resume before tailoring")
    with after_column:
        render_pdf_preview(
            resume.get("resume_pdf_path", ""),
            "Resume after tailoring",
        )
        download_pdf_button(
            resume.get("resume_pdf_path", ""),
            "Download tailored resume",
        )

with fit_tab:
    render_fit_analysis(fit, evidence_lookup)

with changes_tab:
    render_resume_diffs(
        resume.get("change_log", []),
        source_resume_path,
        tailored_result.get("output_tex_path", ""),
        fit,
        evidence_lookup,
    )

run_key = str(state.get("run_id") or st.session_state.get("current_run_id") or "run")
review_round_key = int((payload or {}).get("review_round", 1))
st.subheader("Decisions for all selected roles")
decision_columns = st.columns(len(job_ids), gap="medium")
for decision_column, job_id in zip(decision_columns, job_ids, strict=True):
    decision_key = f"decision_{run_key}_{job_id}_{review_round_key}"
    comment_key = f"comment_{run_key}_{job_id}_{review_round_key}"
    if decision_key not in st.session_state:
        saved_decision = state.get("review_decisions", {}).get(job_id, {})
        st.session_state[decision_key] = saved_decision.get("decision", "")
    if comment_key not in st.session_state:
        saved_decision = state.get("review_decisions", {}).get(job_id, {})
        st.session_state[comment_key] = saved_decision.get("comment", "")
    with decision_column:
        with st.container(border=True):
            st.markdown(f"**{resumes[job_id]['company']}**")
            st.caption(resumes[job_id]["job_title"])
            st.selectbox(
                "Review decision",
                options=["", "approve", "reject"],
                format_func={
                    "": "Choose an option",
                    "approve": "Approve",
                    "reject": "Request changes",
                }.get,
                key=decision_key,
                disabled=controls_locked,
            )
            st.text_area(
                "Feedback",
                help="Feedback is required when you request changes.",
                placeholder="Describe the specific change you want…",
                key=comment_key,
                height=110,
                disabled=controls_locked,
            )
if controls_locked:
    st.caption("Review controls are locked because the single review gate has closed.")

st.divider()
if st.button(
    "Submit all decisions",
    type="primary",
    use_container_width=True,
    disabled=controls_locked,
):
    decisions = {
        job_id: {
            "decision": st.session_state.get(
                f"decision_{run_key}_{job_id}_{review_round_key}",
                "",
            ),
            "comment": st.session_state.get(
                f"comment_{run_key}_{job_id}_{review_round_key}",
                "",
            ),
        }
        for job_id in job_ids
    }
    missing = [job_id for job_id, item in decisions.items() if not item["decision"]]
    if missing:
        st.error(
            "Review every selected application before submitting. "
            f"Missing: {', '.join(missing)}."
        )
    elif any(
        item["decision"] == "reject" and not item["comment"].strip()
        for item in decisions.values()
    ):
        st.error("Add feedback for every resume that needs changes.")
    else:
        try:
            result = resume_graph_run(bundle.app, st.session_state, decisions)
            if result.get("status") == "COMPLETED":
                st.toast("Review complete. Final files are ready.")
            else:
                st.info(
                    "Review submitted. Requested revisions will continue "
                    "automatically without another approval gate."
                )
            st.rerun()
        except Exception as exc:
            st.error(f"Unable to submit your decisions: {exc}")
