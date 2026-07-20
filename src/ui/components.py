"""Shared Streamlit components."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

import streamlit as st

from src.ui.session import (
    ensure_session_defaults,
    get_checkpoint_state,
    reset_demo_data,
    store_graph_result,
)

PHASE_LABELS = {
    "INITIALIZE": "Preparing inputs",
    "FILTER": "Filtering jobs",
    "SCORE": "Ranking matches",
    "FIT_ANALYSIS": "Analyzing fit",
    "TAILOR": "Tailoring resumes",
    "HUMAN_REVIEW": "Reviewing drafts",
    "COVER_LETTERS": "Writing cover letters",
    "COMPLETE": "Complete",
}

STATUS_LABELS = {
    "NOT_STARTED": "Not started",
    "RUNNING": "In progress",
    "WAITING_FOR_REVIEW": "Waiting for review",
    "COMPLETED": "Complete",
    "FAILED_REVIEW": "Review incomplete",
    "FAILED": "Needs attention",
}


def apply_app_styles() -> None:
    """Apply the visual system shared by every application page."""

    st.markdown(
        """
        <style>
            :root {
                --ink: #122033;
                --muted: #617087;
                --line: #dfe6ef;
                --panel: rgba(255, 255, 255, 0.92);
                --brand: #3156d9;
                --brand-dark: #213da8;
                --mint: #18a47b;
            }

            .stApp {
                background:
                    radial-gradient(circle at 88% 3%, rgba(49, 86, 217, 0.09), transparent 26rem),
                    linear-gradient(180deg, #f8faff 0%, #f4f7fb 100%);
                color: var(--ink);
            }

            [data-testid="stHeader"] {
                background: transparent;
            }

            [data-testid="stAppViewContainer"] > .main .block-container {
                max-width: 1180px;
                padding-top: 3.25rem;
                padding-bottom: 5rem;
            }

            [data-testid="stSidebar"] {
                background: #0f1b2d;
                border-right: 0;
            }

            [data-testid="stSidebar"] * {
                color: #dbe5f2;
            }

            [data-testid="stSidebar"] [data-testid="stSidebarNav"] a {
                border-radius: 0.7rem;
                margin-bottom: 0.2rem;
            }

            [data-testid="stSidebar"] [data-testid="stSidebarNav"] a:hover,
            [data-testid="stSidebar"] [data-testid="stSidebarNav"] a[aria-current="page"] {
                background: rgba(255, 255, 255, 0.09);
            }

            [data-testid="stSidebar"] [data-testid="stPageLink"] a {
                border-radius: 0.7rem;
                font-weight: 520;
                margin-bottom: 0.16rem;
                padding: 0.48rem 0.65rem;
                width: 100%;
            }

            [data-testid="stSidebar"] [data-testid="stPageLink"] a:hover,
            [data-testid="stSidebar"] [data-testid="stPageLink"] a[aria-current="page"] {
                background: rgba(255, 255, 255, 0.09);
            }

            [data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {
                background: rgba(255, 255, 255, 0.055);
                border-color: rgba(255, 255, 255, 0.10);
            }

            [data-testid="stSidebar"] hr {
                border-color: rgba(255, 255, 255, 0.12);
            }

            .app-brand {
                display: flex;
                align-items: center;
                gap: 0.75rem;
                margin: 0.15rem 0 1.25rem;
            }

            .app-brand__mark {
                width: 2.25rem;
                height: 2.25rem;
                display: grid;
                place-items: center;
                border-radius: 0.7rem;
                color: white !important;
                background: linear-gradient(145deg, #5274ed, #3156d9);
                box-shadow: 0 8px 22px rgba(49, 86, 217, 0.35);
                font-weight: 750;
            }

            .app-brand__name {
                color: white !important;
                font-size: 1rem;
                font-weight: 680;
                letter-spacing: -0.01em;
            }

            .app-brand__tagline {
                color: #91a1b8 !important;
                font-size: 0.72rem;
                margin-top: 0.08rem;
            }

            .page-header {
                margin-bottom: 1.75rem;
                max-width: 760px;
            }

            .page-header__eyebrow {
                color: var(--brand);
                font-size: 0.77rem;
                font-weight: 720;
                letter-spacing: 0.025em;
                margin-bottom: 0.55rem;
            }

            .page-header h1 {
                color: var(--ink);
                font-size: clamp(2.15rem, 4vw, 3.15rem);
                letter-spacing: -0.045em;
                line-height: 1.05;
                margin: 0;
            }

            .page-header p {
                color: var(--muted);
                font-size: 1.04rem;
                line-height: 1.65;
                margin: 0.8rem 0 0;
            }

            h2, h3 {
                color: var(--ink);
                letter-spacing: -0.025em;
            }

            [data-testid="stVerticalBlockBorderWrapper"] {
                background: var(--panel);
                border: 1px solid var(--line);
                border-radius: 1rem;
                box-shadow: 0 10px 35px rgba(27, 45, 74, 0.045);
            }

            [data-testid="stMetric"] {
                background: var(--panel);
                border: 1px solid var(--line);
                border-radius: 0.9rem;
                padding: 0.9rem 1rem;
            }

            [data-testid="stMetricValue"] {
                color: var(--ink);
                font-size: 1.45rem;
            }

            [data-testid="stTextInput"] input,
            [data-testid="stTextArea"] textarea,
            [data-baseweb="select"] > div {
                background: #fbfcfe;
                border-color: #d6dfeb;
                border-radius: 0.65rem;
            }

            [data-testid="stFileUploaderDropzone"] {
                background: #f8faff;
                border-color: #cdd8e7;
                border-radius: 0.75rem;
                padding: 0.8rem;
            }

            [data-testid^="stBaseButton-primary"] {
                background: linear-gradient(135deg, var(--brand), var(--brand-dark));
                border: 0;
                box-shadow: 0 8px 20px rgba(49, 86, 217, 0.2);
            }

            [data-testid="stBaseButton-secondary"] {
                border-color: #ced8e6;
            }

            [data-testid="stDataFrame"] {
                border: 1px solid var(--line);
                border-radius: 0.8rem;
                overflow: hidden;
            }

            .workflow-step {
                display: grid;
                grid-template-columns: 2rem 1fr;
                gap: 0.75rem;
                padding: 0.7rem 0;
            }

            .workflow-step__number {
                align-items: center;
                background: #edf1ff;
                border-radius: 50%;
                color: var(--brand);
                display: flex;
                font-size: 0.78rem;
                font-weight: 750;
                height: 2rem;
                justify-content: center;
            }

            .workflow-step__title {
                color: var(--ink);
                font-size: 0.93rem;
                font-weight: 680;
                margin-top: 0.05rem;
            }

            .workflow-step__copy {
                color: var(--muted);
                font-size: 0.82rem;
                line-height: 1.45;
                margin-top: 0.15rem;
            }

            .sidebar-kicker {
                color: #91a1b8 !important;
                font-size: 0.7rem;
                font-weight: 700;
                letter-spacing: 0.025em;
                margin: 1rem 0 0.55rem;
            }

            .sidebar-status {
                align-items: center;
                display: flex;
                gap: 0.5rem;
                font-size: 0.9rem;
                font-weight: 620;
            }

            .sidebar-status__dot {
                background: #55d6ad;
                border-radius: 50%;
                box-shadow: 0 0 0 4px rgba(85, 214, 173, 0.12);
                height: 0.5rem;
                width: 0.5rem;
            }

            [data-testid="stSidebar"] button[kind="secondary"] {
                background: transparent;
                border-color: rgba(255, 255, 255, 0.14);
            }

            [data-testid="stSidebar"] button[kind="secondary"]:hover {
                background: rgba(255, 255, 255, 0.07);
                border-color: rgba(255, 255, 255, 0.24);
            }

            [data-testid="stSidebar"] button:disabled {
                opacity: 0.42;
            }

            .empty-state {
                background: rgba(255, 255, 255, 0.74);
                border: 1px dashed #cbd6e5;
                border-radius: 1rem;
                color: var(--muted);
                padding: 2rem;
                text-align: center;
            }

            @media (max-width: 780px) {
                [data-testid="stAppViewContainer"] > .main .block-container {
                    padding-top: 2rem;
                }

                .page-header h1 {
                    font-size: 2.15rem;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_brand() -> None:
    """Render the product identity above the page navigation."""

    with st.sidebar:
        st.markdown(
            """
            <div class="app-brand">
                <div class="app-brand__mark">J</div>
                <div>
                    <div class="app-brand__name">Job search agent</div>
                    <div class="app-brand__tagline">Application workspace</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_sidebar_navigation(pages: list[Any]) -> None:
    """Render branded page navigation in the sidebar."""

    with st.sidebar:
        for page in pages:
            st.page_link(page, use_container_width=True)


def render_page_header(eyebrow: str, title: str, description: str) -> None:
    """Render a consistent page title and supporting description."""

    st.markdown(
        f"""
        <div class="page-header">
            <div class="page-header__eyebrow">{escape(eyebrow)}</div>
            <h1>{escape(title)}</h1>
            <p>{escape(description)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_phase(value: str | None) -> str:
    """Return a readable workflow phase label."""

    if not value:
        return "Not started"
    return PHASE_LABELS.get(value, value.replace("_", " ").capitalize())


def format_status(value: str | None) -> str:
    """Return a readable workflow status label."""

    if not value:
        return "Not started"
    return STATUS_LABELS.get(value, value.replace("_", " ").capitalize())


def render_sidebar(app: Any, tracer: Any | None = None) -> dict[str, Any]:
    """Render shared sidebar status and controls."""

    ensure_session_defaults(st.session_state)
    checkpoint_state = get_checkpoint_state(app, st.session_state.get("current_thread_id"))
    state = checkpoint_state or st.session_state.get("last_result") or {}
    observability_status = (
        state.get("langfuse_status")
        or getattr(tracer, "status_message", None)
        or "Observability: Local no-op tracing"
    )
    observability_status = observability_status.removeprefix("Observability: ")
    with st.sidebar:
        st.markdown('<div class="sidebar-kicker">Current run</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="sidebar-status">
                <span class="sidebar-status__dot"></span>
                <span>{escape(format_status(state.get("status")))}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(format_phase(state.get("phase")))

        with st.expander("Run details"):
            st.caption(
                f"Run ID: {state.get('run_id') or st.session_state.get('current_run_id') or '—'}"
            )
            st.caption(
                f"Thread ID: {state.get('thread_id') or st.session_state.get('current_thread_id') or '—'}"
            )
            st.caption(f"Revision round: {state.get('revision_round', 0)}")
            st.caption(f"Tracing: {observability_status}")
            if state.get("trace_id"):
                st.caption(f"Trace ID: {state['trace_id']}")

        st.divider()
        if st.button(
            "Refresh run status",
            use_container_width=True,
            disabled=not st.session_state.get("current_thread_id"),
        ):
            refreshed = get_checkpoint_state(app, st.session_state.get("current_thread_id"))
            if refreshed:
                store_graph_result(st.session_state, refreshed)
                st.toast("Run status refreshed.")
                st.rerun()
            else:
                st.warning("There is no saved run to refresh.")

        if st.button("Reset workspace", use_container_width=True):
            reset_demo_data(st.session_state)
            st.toast("Workspace reset.")
            st.rerun()
    return state


def render_errors(state: dict[str, Any]) -> None:
    """Render readable workflow errors without stack traces."""

    errors = state.get("errors", [])
    if not errors:
        return
    st.error("The workflow needs attention")
    for error in errors[-3:]:
        st.write(f"{error.get('type', 'Error')}: {error.get('message', '')}")


def download_pdf_button(path_value: str, label: str) -> None:
    """Render a download button when an artifact path exists."""

    path = Path(path_value)
    if path.exists():
        st.download_button(
            label,
            data=path.read_bytes(),
            file_name=path.name,
            mime="application/pdf",
            use_container_width=True,
        )
    else:
        st.warning(f"File not found: {path}")
