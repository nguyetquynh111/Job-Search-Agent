"""Complete Streamlit page renderers."""

from __future__ import annotations


import streamlit as st

from app.components import (
    build_evidence_lookup,
    build_outputs_zip,
    download_pdf_button,
    format_phase,
    format_status,
    group_rejected_jobs,
    observability_summary,
    PHASE_ORDER,
    render_agent_decisions,
    render_artifact_errors,
    render_errors,
    render_fit_analysis,
    render_friendly_empty_state,
    render_input_file_guide,
    render_page_header,
    render_pdf_preview,
    render_ranked_job_card,
    render_resume_diffs,
    render_review_gate_banner,
    render_sidebar,
    render_status_pill,
    render_tool_activity,
    render_upload_validation,
    render_workflow_overview,
    review_status_for_job,
    validate_uploaded_file,
)
from app.runtime import (
    configured_graph_bundle,
    ensure_session_defaults,
    resume_graph_run,
    save_uploaded_inputs,
    start_graph_run,
)


def _review_decision_label(value: str) -> str:
    labels = {
        "": "Choose an option",
        "approve": "Approve",
        "reject": "Request changes",
    }
    return labels.get(value, value)


def render_input_page() -> None:
    ensure_session_defaults(st.session_state)
    bundle = None
    runtime_error: Exception | None = None
    try:
        bundle = configured_graph_bundle()
    except Exception as exc:
        # Keep uploads available even when workflow dependencies are missing.
        runtime_error = exc

    state = render_sidebar(
        bundle.app if bundle is not None else None,
        bundle.tracer if bundle is not None else None,
    )

    render_page_header(
        "Step 1 of 4",
        "Set up your job search",
        (
            "Start with the four files the agent needs. The app will rank your "
            "roles, prepare tailored drafts, then stop for your review before "
            "making final documents."
        ),
    )
    render_workflow_overview("INITIALIZE")
    render_errors(state)

    form_column, guide_column = st.columns([1.75, 1], gap="large")
    with form_column:
        with st.container(border=True):
            st.subheader("Input files")
            st.caption("Each file is checked before the search button turns on.")

            jobs_upload = st.file_uploader(
                "Job listings (.csv)",
                type=["csv"],
                help=(
                    "Use one row per job. The app expects job IDs, titles, "
                    "companies, descriptions, and requirements."
                ),
                key="jobs_upload",
            )
            jobs_valid, jobs_message = validate_uploaded_file(jobs_upload, {".csv"})
            render_upload_validation(jobs_valid, jobs_message)

            preferences_upload = st.file_uploader(
                "Preferences (.yaml)",
                type=["yaml"],
                help=(
                    "Define target titles, locations, remote preference, salary, "
                    "and exclusions."
                ),
                key="preferences_upload",
            )
            preferences_valid, preferences_message = validate_uploaded_file(
                preferences_upload,
                {".yaml"},
            )
            render_upload_validation(preferences_valid, preferences_message)

            left, right = st.columns(2)
            with left:
                resume_upload = st.file_uploader(
                    "Resume (.tex)",
                    type=["tex"],
                    help=(
                        "Compilable LaTeX with structurally identifiable summary, "
                        "experience, skills, and projects sections."
                    ),
                    key="resume_upload",
                )
                resume_valid, resume_message = validate_uploaded_file(
                    resume_upload,
                    {".tex"},
                )
                render_upload_validation(resume_valid, resume_message)
            with right:
                portfolio_upload = st.file_uploader(
                    "Portfolio (.txt)",
                    type=["txt"],
                    help="Plain-text file only. Separate projects with a blank line.",
                    key="portfolio_upload",
                )
                portfolio_valid, portfolio_message = validate_uploaded_file(
                    portfolio_upload,
                    {".txt"},
                )
                render_upload_validation(portfolio_valid, portfolio_message)

            all_files_valid = all(
                [jobs_valid, preferences_valid, resume_valid, portfolio_valid]
            )
            start_submitted = st.button(
                "Start search",
                type="primary",
                use_container_width=True,
                disabled=not all_files_valid,
            )
            if not all_files_valid:
                st.caption("Add all four valid files to enable the search.")

    with guide_column:
        with st.container(border=True):
            st.subheader("What happens next")
            render_input_file_guide()

        with st.expander("File requirements"):
            st.markdown(
                """
                - **Job listings:** CSV with `job_id`, `title`, `company`, `location`,
                  `remote`, `description`, and `requirements` columns.
                - **Resume:** a compilable `.tex` file with identifiable summary,
                  experience, skills, and projects sections; standard `\\item` and
                  common resume-template macros are supported.
                - **Portfolio:** a `.txt` file; separate projects with a blank line and
                  optionally add `Technologies: Python, SQL`.
                - **Preferences:** a `.yaml` file using keys such as
                  `target_titles`, `locations`, `remote`, `min_salary`, and
                  `excluded_keywords`.
                """
            )

        if state:
            with st.container(border=True):
                st.subheader("Saved run")
                st.caption("You can continue from the latest saved checkpoint.")
                if st.session_state.get("waiting_for_review"):
                    if st.button(
                        "Open review", type="primary", use_container_width=True
                    ):
                        st.session_state.__setitem__("active_page", "review")
                elif state.get("status") == "COMPLETED":
                    if st.button(
                        "Open results", type="primary", use_container_width=True
                    ):
                        st.session_state.__setitem__("active_page", "results")
                elif st.button("View run progress", use_container_width=True):
                    st.session_state.__setitem__("active_page", "execution")

    if start_submitted:
        uploads = {
            "jobs_path": jobs_upload,
            "preferences_path": preferences_upload,
            "resume_path": resume_upload,
            "portfolio_path": portfolio_upload,
        }
        labels = {
            "jobs_path": "Job listings",
            "preferences_path": "Preferences",
            "resume_path": "Resume",
            "portfolio_path": "Portfolio",
        }
        missing = [
            labels[key]
            for key, uploaded_file in uploads.items()
            if uploaded_file is None
        ]
        if missing:
            st.error(f"Upload these required files: {', '.join(missing)}.")
        else:
            try:
                save_uploaded_inputs(st.session_state, uploads)
                if bundle is None:
                    st.success("All four input files were uploaded successfully.")
                    detail = (
                        str(runtime_error) if runtime_error else "Unknown runtime error"
                    )
                    st.warning(
                        "The files are saved, but the search workflow is not available "
                        f"in this environment yet: {detail}"
                    )
                    st.stop()
                with st.spinner("Analyzing jobs and preparing resume drafts…"):
                    result = start_graph_run(bundle.app, st.session_state)
                if st.session_state.get("waiting_for_review") or result.get(
                    "__interrupt__"
                ):
                    st.session_state.__setitem__("active_page", "review")
                elif result.get("status") == "COMPLETED":
                    st.session_state.__setitem__("active_page", "results")
                else:
                    st.session_state.__setitem__("active_page", "execution")
            except Exception as exc:
                st.error(f"Unable to start the search: {exc}")


def render_execution_page() -> None:
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
        (
            "Watch the search move from filtering to ranking, fit analysis, "
            "resume drafting, review, and final files."
        ),
    )
    render_workflow_overview(state.get("phase"))
    render_errors(state)

    if runtime_error is not None:
        st.warning(
            f"The search workflow is not available in this environment yet: {runtime_error}"
        )

    if not state:
        render_friendly_empty_state(
            "No search has started yet",
            "Upload the four input files on the setup page, then start a run.",
        )
        if st.button("Set up search", type="primary"):
            st.session_state.__setitem__("active_page", "input")
        st.stop()

    phase = state.get("phase", "INITIALIZE")
    progress = max(0, PHASE_ORDER.index(phase) if phase in PHASE_ORDER else 0) / (
        len(PHASE_ORDER) - 1
    )
    st.progress(progress)
    st.caption(f"Current step: {format_phase(phase)}")

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
        ["Best matches", "Decisions", "Activity log", "Filtered out"]
    )

    with ranked_tab:
        if ranked:
            st.caption(
                "Scores come from deterministic matching rules. The notes show "
                "which parts of the job matched your evidence."
            )
            for ranked_item in ranked:
                render_ranked_job_card(
                    ranked_item,
                    state.get("top_3_job_ids", []),
                )
        else:
            st.caption("Best matches will appear here as soon as scoring finishes.")

    with decisions_tab:
        st.caption(
            "A short record of the workflow choices made during this run."
        )
        render_agent_decisions(decisions)

    with activity_tab:
        st.caption(
            "A readable log of the tools used by the workflow."
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
                "A role can appear in more than one group when multiple rules "
                "filtered it out."
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


def render_review_page() -> None:
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
        (
            "Compare each draft against your original resume, check why changes "
            "were made, then approve it or request a specific revision."
        ),
    )
    render_workflow_overview(state.get("phase") or "HUMAN_REVIEW")
    render_review_gate_banner()
    render_errors(state)

    if runtime_error is not None:
        st.warning(
            f"The search workflow is not available in this environment yet: {runtime_error}"
        )

    payload = st.session_state.get("interrupt_payload") or state.get(
        "interrupt_payload"
    )
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
        render_friendly_empty_state(
            "No drafts are waiting for review",
            "Draft resumes will appear here after the ranking and tailoring steps finish.",
        )
        if st.button("View run progress"):
            st.session_state.__setitem__("active_page", "execution")
        st.stop()

    revision_round = int(
        (payload or {}).get("revision_round", state.get("revision_round", 0))
    )
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
            f"Initial review. Up to {max_revision_rounds} revision rounds are "
            "available."
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
    if selected_job_id is None:
        st.stop()
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
        ["Preview", "Why this role", "What changed"]
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

    run_key = str(
        state.get("run_id") or st.session_state.get("current_run_id") or "run"
    )
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
                    format_func=_review_decision_label,
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
        st.caption(
            "Review controls are locked because this review step has already closed."
        )

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
                if bundle is None:
                    st.error("The workflow is not available in this environment yet.")
                    st.stop()
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


def render_results_page() -> None:
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
        "Download the final resume and cover letter for each approved role.",
    )
    render_workflow_overview(state.get("phase") or "COMPLETE")
    render_errors(state)

    if runtime_error is not None:
        st.warning(
            f"The search workflow is not available in this environment yet: {runtime_error}"
        )

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

    st.subheader("Run summary")
    summary = observability_summary(
        state, bundle.tracer if bundle is not None else None
    )
    trace_url = state.get("trace_url")
    trace_column, metrics_column = st.columns([1, 2.3], gap="large")
    with trace_column:
        with st.container(border=True):
            st.caption("Trace status")
            st.markdown(f"**{summary['trace_status']}**")
            if trace_url:
                st.link_button(
                    "Open run trace",
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
                "Download all final files",
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
        if selected_job_id is None:
            st.stop()
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
                ["Original resume", "Final resume", "Final cover letter"]
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
        render_friendly_empty_state(
            "No final files yet",
            (
                "Approve the tailored resumes first. The app will generate cover "
                "letters after review is complete."
            ),
        )
        if st.button("Open review", type="primary"):
            st.session_state.__setitem__("active_page", "review")

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
