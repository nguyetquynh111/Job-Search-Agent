"""Input configuration page."""

from __future__ import annotations

import streamlit as st

from src.ui.components import render_errors, render_page_header, render_sidebar
from src.ui.graph_resource import configured_graph_bundle
from src.ui.session import (
    ensure_session_defaults,
    save_uploaded_inputs,
    start_graph_run,
)

ensure_session_defaults(st.session_state)
bundle = None
runtime_error: Exception | None = None
try:
    bundle = configured_graph_bundle()
except Exception as exc:
    # Uploading inputs is useful on its own and must not be taken down by an
    # unavailable graph runtime or a tool that is still under development.
    runtime_error = exc

state = render_sidebar(
    bundle.app if bundle is not None else None,
    bundle.tracer if bundle is not None else None,
)

render_page_header(
    "Step 1 of 4",
    "Set up your job search",
    "Upload your job listings, resume, portfolio, and preferences. We will rank the best matches and pause for your approval before creating final documents.",
)
render_errors(state)

form_column, guide_column = st.columns([1.75, 1], gap="large")
with form_column:
    with st.container(border=True):
        st.subheader("Input files")
        st.caption("Upload all four files to start a new search.")

        with st.form("input_uploads_form"):
            jobs_upload = st.file_uploader(
                "Job listings",
                type=["csv"],
                help="CSV only. Include one job per row.",
                key="jobs_upload",
            )
            preferences_upload = st.file_uploader(
                "Preferences",
                type=["yaml"],
                help="YAML only. Define target titles, locations, remote preference, salary, and exclusions.",
                key="preferences_upload",
            )

            left, right = st.columns(2)
            with left:
                resume_upload = st.file_uploader(
                    "Resume",
                    type=["tex"],
                    help="LaTeX source file only.",
                    key="resume_upload",
                )
            with right:
                portfolio_upload = st.file_uploader(
                    "Portfolio",
                    type=["txt"],
                    help="Plain-text file only. Separate projects with a blank line.",
                    key="portfolio_upload",
                )

            start_submitted = st.form_submit_button(
                "Start search",
                type="primary",
                use_container_width=True,
            )

with guide_column:
    with st.container(border=True):
        st.subheader("What happens next")
        st.markdown(
            """
            <div class="workflow-step">
                <div class="workflow-step__number">1</div>
                <div><div class="workflow-step__title">Rank the opportunities</div>
                <div class="workflow-step__copy">Filter the job list and identify the three strongest matches.</div></div>
            </div>
            <div class="workflow-step">
                <div class="workflow-step__number">2</div>
                <div><div class="workflow-step__title">Tailor your materials</div>
                <div class="workflow-step__copy">Use evidence from your resume and portfolio for each selected role.</div></div>
            </div>
            <div class="workflow-step">
                <div class="workflow-step__number">3</div>
                <div><div class="workflow-step__title">Review before finalizing</div>
                <div class="workflow-step__copy">Approve drafts or request focused revisions before cover letters are generated.</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("File requirements"):
        st.markdown(
            """
            - **Job listings:** CSV with `job_id`, `title`, `company`, `location`,
              `remote`, `description`, and `requirements` columns.
            - **Resume:** a `.tex` LaTeX source file.
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
            st.caption("A previous run is available from the navigation.")
            if st.session_state.get("waiting_for_review"):
                if st.button("Open review", type="primary", use_container_width=True):
                    st.switch_page("views/3_Review.py")
            elif state.get("status") == "COMPLETED":
                if st.button("Open results", type="primary", use_container_width=True):
                    st.switch_page("views/4_Results.py")
            elif st.button("View run progress", use_container_width=True):
                st.switch_page("views/2_Execution.py")

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
    missing = [labels[key] for key, uploaded_file in uploads.items() if uploaded_file is None]
    if missing:
        st.error(f"Upload these required files: {', '.join(missing)}.")
    else:
        try:
            save_uploaded_inputs(st.session_state, uploads)
            if bundle is None:
                st.success("All four input files were uploaded successfully.")
                detail = str(runtime_error) if runtime_error else "Unknown runtime error"
                st.warning(
                    "The files are saved, but the search workflow is not available "
                    f"in this environment yet: {detail}"
                )
                st.stop()
            with st.spinner("Analyzing jobs and preparing resume drafts…"):
                result = start_graph_run(bundle.app, st.session_state)
            if st.session_state.get("waiting_for_review") or result.get("__interrupt__"):
                st.switch_page("views/3_Review.py")
            elif result.get("status") == "COMPLETED":
                st.switch_page("views/4_Results.py")
            else:
                st.switch_page("views/2_Execution.py")
        except Exception as exc:
            st.error(f"Unable to start the search: {exc}")
