"""Thin Streamlit entry point for the Job Search Agent."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    project_root = str(Path(__file__).resolve().parents[1])
    if sys.path[0] != project_root:
        try:
            sys.path.remove(project_root)
        except ValueError:
            pass
        sys.path.insert(0, project_root)

import streamlit as st
from dotenv import load_dotenv

from app.components import (
    build_evidence_lookup,
    build_outputs_zip,
    build_resume_change_views,
    group_rejected_jobs,
    observability_summary,
    render_sidebar_brand,
    render_sidebar_navigation,
    review_status_for_job,
    score_components_from_rationale,
    validate_uploaded_file,
)
from app.runtime import (
    ensure_session_defaults,
    reset_demo_data,
    save_uploaded_inputs,
    store_graph_result,
)
from app.styles import apply_app_styles
from app.views import (
    render_execution_page,
    render_input_page,
    render_results_page,
    render_review_page,
)

__all__ = [
    "build_evidence_lookup",
    "build_outputs_zip",
    "build_resume_change_views",
    "group_rejected_jobs",
    "ensure_session_defaults",
    "main",
    "observability_summary",
    "reset_demo_data",
    "review_status_for_job",
    "save_uploaded_inputs",
    "score_components_from_rationale",
    "store_graph_result",
    "validate_uploaded_file",
]


def main() -> None:
    """Configure Streamlit navigation and dispatch to page renderers."""

    load_dotenv()
    st.set_page_config(
        page_title="Job search agent",
        page_icon="compass",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    ensure_session_defaults(st.session_state)
    apply_app_styles()

    pages = [
        st.Page(
            render_input_page,
            title="1 · Set up search",
            icon=":material/tune:",
            default=True,
        ),
        st.Page(
            render_execution_page,
            title="2 · Run progress",
            icon=":material/timeline:",
        ),
        st.Page(
            render_review_page,
            title="3 · Review drafts",
            icon=":material/rate_review:",
        ),
        st.Page(
            render_results_page,
            title="4 · Results",
            icon=":material/folder_open:",
        ),
    ]
    page = st.navigation(pages, position="hidden")
    render_sidebar_brand()
    render_sidebar_navigation(pages)
    page.run()


if __name__ == "__main__":
    main()
