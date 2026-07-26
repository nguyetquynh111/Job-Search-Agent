"""Shared Streamlit components."""

from __future__ import annotations

import base64
from collections import defaultdict
from html import escape
from io import BytesIO
from pathlib import Path
import re
import tempfile
from typing import Any
import zipfile

import streamlit as st
import streamlit.components.v1 as st_components

from src.schemas.jobs import Job
from src.tools.job_evidence import build_job_evidence
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

            .review-gate {
                align-items: center;
                background: linear-gradient(135deg, #eef2ff, #f8faff);
                border: 1px solid #cbd6ff;
                border-left: 5px solid var(--brand);
                border-radius: 0.95rem;
                display: flex;
                gap: 0.85rem;
                margin: -0.25rem 0 1.5rem;
                padding: 1rem 1.1rem;
            }

            .review-gate__icon {
                align-items: center;
                background: var(--brand);
                border-radius: 50%;
                color: white;
                display: flex;
                flex: 0 0 auto;
                font-size: 0.9rem;
                font-weight: 760;
                height: 2rem;
                justify-content: center;
                width: 2rem;
            }

            .review-gate__title {
                color: var(--ink);
                font-size: 0.98rem;
                font-weight: 740;
            }

            .review-gate__copy {
                color: var(--muted);
                font-size: 0.82rem;
                margin-top: 0.14rem;
            }

            .status-pill,
            .evidence-badge,
            .source-badge {
                border-radius: 999px;
                display: inline-flex;
                font-size: 0.72rem;
                font-weight: 700;
                line-height: 1;
                margin: 0.1rem 0.22rem 0.1rem 0;
                padding: 0.35rem 0.55rem;
            }

            .status-pill--success {
                background: #e6f8f1;
                color: #087255;
            }

            .status-pill--warning {
                background: #fff2d9;
                color: #8a5a00;
            }

            .status-pill--neutral {
                background: #edf1f6;
                color: #526276;
            }

            .status-pill--brand {
                background: #e9edff;
                color: #2947b8;
            }

            .evidence-badge,
            .source-badge {
                background: #edf1ff;
                color: #3156b8;
            }

            .change-card,
            .decision-card,
            .tool-card,
            .fit-card,
            .score-card,
            .swap-card {
                background: rgba(255, 255, 255, 0.94);
                border: 1px solid var(--line);
                border-radius: 0.9rem;
                margin-bottom: 0.8rem;
                padding: 1rem;
            }

            .change-card__header,
            .tool-card__header,
            .score-card__header {
                align-items: center;
                display: flex;
                gap: 0.65rem;
                justify-content: space-between;
                margin-bottom: 0.7rem;
            }

            .change-card__title,
            .tool-card__title,
            .score-card__title,
            .fit-card__title {
                color: var(--ink);
                font-size: 0.93rem;
                font-weight: 720;
            }

            .diff-grid {
                display: grid;
                gap: 0.7rem;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                margin: 0.65rem 0;
            }

            .diff-block {
                border-radius: 0.7rem;
                min-height: 4.5rem;
                padding: 0.8rem;
            }

            .diff-block--removed {
                background: #fff1f1;
                border: 1px solid #f2caca;
            }

            .diff-block--added {
                background: #eaf8f2;
                border: 1px solid #bfe7d8;
            }

            .diff-block__label,
            .card-label {
                color: var(--muted);
                font-size: 0.68rem;
                font-weight: 760;
                letter-spacing: 0.04em;
                margin-bottom: 0.38rem;
                text-transform: uppercase;
            }

            .diff-block__text,
            .card-copy {
                color: var(--ink);
                font-size: 0.84rem;
                line-height: 1.5;
                overflow-wrap: anywhere;
                white-space: pre-wrap;
            }

            .card-meta {
                color: var(--muted);
                font-size: 0.8rem;
                line-height: 1.5;
                margin-top: 0.55rem;
            }

            .skill-groups {
                display: grid;
                gap: 0.8rem;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                margin: 0.85rem 0 1.1rem;
            }

            .skill-group {
                border-radius: 0.8rem;
                min-height: 8rem;
                padding: 0.85rem;
            }

            .skill-group--aligned {
                background: #eaf8f2;
                border: 1px solid #bfe7d8;
            }

            .skill-group--evidenced {
                background: #eef2ff;
                border: 1px solid #cfd8ff;
            }

            .skill-group--gap {
                background: #fff4e5;
                border: 1px solid #f2d5a8;
            }

            .skill-group__title {
                color: var(--ink);
                font-size: 0.82rem;
                font-weight: 730;
                margin-bottom: 0.55rem;
            }

            .skill-group__item {
                color: var(--ink);
                font-size: 0.8rem;
                line-height: 1.45;
                margin-bottom: 0.45rem;
            }

            .score-components {
                display: grid;
                gap: 0.55rem;
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }

            .score-component {
                background: #f7f9fc;
                border-radius: 0.65rem;
                padding: 0.65rem;
            }

            .score-total {
                color: var(--brand);
                font-size: 1.35rem;
                font-weight: 760;
            }

            .deterministic-note {
                align-items: center;
                color: #42536a;
                display: flex;
                font-size: 0.78rem;
                gap: 0.4rem;
                margin-bottom: 0.65rem;
            }

            .upload-validation {
                font-size: 0.78rem;
                margin: -0.45rem 0 0.75rem;
            }

            .upload-validation--ok {
                color: #087255;
            }

            .upload-validation--error {
                color: #a33a3a;
            }

            .preview-shell {
                background: #edf1f6;
                border: 1px solid var(--line);
                border-radius: 0.8rem;
                overflow: hidden;
            }

            .technical-note {
                color: var(--muted);
                font-size: 0.78rem;
                line-height: 1.45;
            }

            .reset-zone {
                border-top: 1px solid rgba(255, 255, 255, 0.12);
                margin-top: 0.8rem;
                padding-top: 0.8rem;
            }

            @media (max-width: 780px) {
                [data-testid="stAppViewContainer"] > .main .block-container {
                    padding-top: 2rem;
                }

                .page-header h1 {
                    font-size: 2.15rem;
                }

                .diff-grid,
                .skill-groups,
                .score-components {
                    grid-template-columns: 1fr;
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
    checkpoint_state = get_checkpoint_state(
        app, st.session_state.get("current_thread_id")
    )
    state = checkpoint_state or st.session_state.get("last_result") or {}
    observability_status = (
        state.get("langfuse_status")
        or getattr(tracer, "status_message", None)
        or "Observability: Local no-op tracing"
    )
    observability_status = observability_status.removeprefix("Observability: ")
    with st.sidebar:
        st.markdown(
            '<div class="sidebar-kicker">Current run</div>', unsafe_allow_html=True
        )
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
            key="refresh_run_status",
        ):
            refreshed = get_checkpoint_state(
                app, st.session_state.get("current_thread_id")
            )
            if refreshed:
                store_graph_result(st.session_state, refreshed)
                st.toast("Run status refreshed.")
                st.rerun()
            else:
                st.warning("There is no saved run to refresh.")

        st.markdown('<div class="reset-zone"></div>', unsafe_allow_html=True)
        if not st.session_state.get("confirm_reset_workspace", False):
            if st.button(
                "Reset workspace…",
                use_container_width=True,
                key="request_reset_workspace",
            ):
                st.session_state["confirm_reset_workspace"] = True
                st.rerun()
        else:
            st.warning("Reset the run, uploads, memory, and generated files?")
            confirm_column, cancel_column = st.columns(2)
            with confirm_column:
                if st.button(
                    "Confirm reset",
                    use_container_width=True,
                    key="confirm_reset_workspace_button",
                ):
                    reset_demo_data(st.session_state)
                    st.toast("Workspace reset.")
                    st.rerun()
            with cancel_column:
                if st.button(
                    "Cancel",
                    use_container_width=True,
                    key="cancel_reset_workspace",
                ):
                    st.session_state["confirm_reset_workspace"] = False
                    st.rerun()
    return state


def render_errors(state: dict[str, Any]) -> None:
    """Render readable workflow errors without stack traces."""

    errors = state.get("errors", [])
    if not errors:
        return
    st.error("The workflow needs attention")
    for error in errors[-3:]:
        message = str(error.get("message", ""))
        if _is_latex_error(message):
            st.write(
                "A document could not be compiled. Check the LaTeX source and try again."
            )
        else:
            st.write(_concise_error_message(message))
    with st.expander("Technical error details"):
        st.json(errors, expanded=False)


def download_pdf_button(path_value: str, label: str) -> None:
    """Render a download button when an artifact path exists."""

    path = Path(path_value)
    if path.is_file():
        st.download_button(
            label,
            data=path.read_bytes(),
            file_name=path.name,
            mime="application/pdf",
            use_container_width=True,
        )
    else:
        st.info("This PDF is not available yet.")


def _is_latex_error(message: str) -> bool:
    """Return whether a workflow error is a LaTeX compilation failure."""

    normalized = message.casefold()
    return any(
        token in normalized
        for token in ("latex", "pdflatex", ".tex", "one-page", "one page")
    )


def _concise_error_message(message: str, limit: int = 220) -> str:
    """Return a short, single-line error suitable for the primary UI."""

    single_line = " ".join(message.split())
    if not single_line:
        return "An unexpected workflow error occurred."
    return single_line if len(single_line) <= limit else single_line[: limit - 1] + "…"


def validate_uploaded_file(
    uploaded_file: Any,
    allowed_extensions: set[str],
) -> tuple[bool, str]:
    """Validate the UI-level properties available before a run is submitted."""

    if uploaded_file is None:
        return False, "Required file"
    name = str(getattr(uploaded_file, "name", ""))
    suffix = Path(name).suffix.casefold()
    if suffix not in {value.casefold() for value in allowed_extensions}:
        expected = ", ".join(sorted(allowed_extensions))
        return False, f"Use {expected}"
    try:
        payload = uploaded_file.getvalue()
    except Exception:
        return False, "Could not read this file"
    if not payload:
        return False, "File is empty"
    content_error = _validate_upload_payload(payload, suffix)
    if content_error:
        return False, content_error
    size_kb = max(1, round(len(payload) / 1024))
    return True, f"Ready · {size_kb:,} KB"


@st.cache_data(show_spinner=False)
def _validate_upload_payload(payload: bytes, suffix: str) -> str | None:
    """Reuse backend loaders to validate an upload before enabling a run."""

    loaders: dict[str, tuple[str, str]] = {
        ".csv": ("load_jobs_csv", "jobs.csv"),
        ".yaml": ("load_candidate_profile", "preferences.yaml"),
        ".tex": ("load_resume_data", "resume.tex"),
        ".txt": ("load_portfolio", "portfolio.txt"),
    }
    loader_spec = loaders.get(suffix)
    if loader_spec is None:
        return None
    loader_name, filename = loader_spec
    try:
        from src import data_loader

        with tempfile.TemporaryDirectory(
            prefix="job-search-upload-check-"
        ) as directory:
            path = Path(directory) / filename
            path.write_bytes(payload)
            loader = getattr(data_loader, loader_name)
            loader(path)
    except Exception as exc:
        return _concise_error_message(str(exc), limit=120)
    return None


def render_upload_validation(is_valid: bool, message: str) -> None:
    """Show an inline upload validation result."""

    modifier = "ok" if is_valid else "error"
    icon = "✓" if is_valid else "•"
    st.markdown(
        (
            f'<div class="upload-validation upload-validation--{modifier}">'
            f"{icon} {escape(message)}</div>"
        ),
        unsafe_allow_html=True,
    )


def render_review_gate_banner() -> None:
    """Make the workflow's only human interrupt explicit."""

    st.markdown(
        """
        <div class="review-gate">
            <div class="review-gate__icon">1×</div>
            <div>
                <div class="review-gate__title">
                    Human Review Gate — This is the only pause in the workflow.
                </div>
                <div class="review-gate__copy">
                    Submit one decision for every selected role. Requested revisions
                    continue automatically and never create another approval gate.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def review_status_for_job(
    state: dict[str, Any],
    job_id: str,
    *,
    has_active_payload: bool = False,
    max_revision_rounds: int = 2,
) -> tuple[str, str]:
    """Return a review label and visual status for one selected job."""

    cover_letters = state.get("cover_letter_results", {})
    if job_id in cover_letters or (
        state.get("status") == "COMPLETED"
        and job_id in state.get("approved_job_ids", [])
    ):
        return "Finalized", "success"

    decisions = state.get("review_decisions", {})
    if not decisions:
        history = state.get("review_history", [])
        if history:
            decisions = history[-1].get("decisions", {})
    decision = decisions.get(job_id, {})
    decision_value = decision.get("decision")

    if decision_value == "reject":
        round_number = max(
            1,
            min(
                int(state.get("revision_round", 0) or 1),
                max_revision_rounds,
            ),
        )
        return (
            f"Rejected — revision round {round_number} of {max_revision_rounds}",
            "warning",
        )
    if job_id in state.get("approved_job_ids", []) or decision_value == "approve":
        return "Approved", "success"
    if has_active_payload or state.get("status") == "WAITING_FOR_REVIEW":
        return "Awaiting review", "brand"
    return "Awaiting review", "neutral"


def render_status_pill(label: str, tone: str = "neutral") -> None:
    """Render one consistent status badge."""

    st.markdown(
        f'<span class="status-pill status-pill--{escape(tone)}">{escape(label)}</span>',
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def _compile_tex_preview(
    source: bytes, filename: str
) -> tuple[bytes | None, str | None]:
    """Compile a source resume into an isolated, cached preview PDF."""

    safe_name = Path(filename).name or "resume.tex"
    if Path(safe_name).suffix.casefold() != ".tex":
        safe_name = f"{Path(safe_name).stem or 'resume'}.tex"
    try:
        from src.tools.common import run_pdflatex

        with tempfile.TemporaryDirectory(prefix="job-search-preview-") as directory:
            tex_path = Path(directory) / safe_name
            tex_path.write_bytes(source)
            errors = run_pdflatex(tex_path, artifact_label="source resume preview")
            pdf_path = tex_path.with_suffix(".pdf")
            if errors:
                return None, "\n".join(str(error) for error in errors)
            if not pdf_path.is_file():
                return None, "The preview compiler did not produce a PDF."
            return pdf_path.read_bytes(), None
    except Exception as exc:
        return None, _concise_error_message(str(exc))


def _load_preview_pdf(path_value: str | Path) -> tuple[bytes | None, str | None]:
    """Load a PDF or compile a TeX source for previewing."""

    if not path_value:
        return None, "No document path was provided."
    path = Path(path_value)
    if not path.is_file():
        return None, "The document is not available yet."
    try:
        if path.suffix.casefold() == ".pdf":
            payload = path.read_bytes()
            if not payload.startswith(b"%PDF"):
                return None, "The file is not a valid PDF."
            return payload, None
        if path.suffix.casefold() == ".tex":
            return _compile_tex_preview(path.read_bytes(), path.name)
        return None, "Only PDF and LaTeX source files can be previewed."
    except OSError as exc:
        return None, _concise_error_message(str(exc))


def render_pdf_preview(
    path_value: str | Path,
    label: str,
    *,
    height: int = 620,
) -> bool:
    """Embed a PDF preview and return whether it rendered."""

    pdf_bytes, error = _load_preview_pdf(path_value)
    st.markdown(f"**{label}**")
    if pdf_bytes is None:
        st.info(
            f"{label} cannot be previewed. "
            f"{_concise_error_message(error or 'The PDF does not exist or could not be rendered.')}"
        )
        if error and _is_latex_error(error):
            with st.expander(f"Technical details · {label} preview logs"):
                st.code(error)
        return False
    encoded = base64.b64encode(pdf_bytes).decode("ascii")
    safe_label = escape(label, quote=True)
    st_components.html(
        f"""
        <div style="height:{height}px;border:1px solid #dfe6ef;border-radius:12px;
                    overflow:hidden;background:#edf1f6;">
          <embed title="{safe_label}" src="data:application/pdf;base64,{encoded}"
                 type="application/pdf" width="100%" height="100%" />
        </div>
        """,
        height=height + 4,
        scrolling=False,
    )
    return True


def build_evidence_lookup(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Build a UI-only lookup over evidence already present in graph state."""

    lookup: dict[str, dict[str, Any]] = {}
    profile = state.get("candidate_profile", {}) or {}
    portfolio = state.get("portfolio", {}) or {}
    collections = [
        profile.get("resume_evidence", []),
        profile.get("master_skill_evidence", []),
        profile.get("portfolio_evidence", []),
        portfolio.get("evidence_items", []),
    ]
    for items in collections:
        for item in items or []:
            evidence_id = item.get("evidence_id")
            if evidence_id:
                lookup[str(evidence_id)] = dict(item)
    for fact in state.get("memory_facts", []) or []:
        fact_id = fact.get("fact_id")
        if fact_id:
            lookup[str(fact_id)] = {
                "evidence_id": fact_id,
                "source": "memory",
                "text": (
                    f"{fact.get('fact_type', 'fact')}: "
                    f"{fact.get('canonical_value', '')}"
                ),
                "metadata": {"provenance": fact.get("provenance", {})},
            }
    for job_payload in state.get("jobs", []) or []:
        try:
            job = Job.model_validate(job_payload)
        except (TypeError, ValueError):
            continue
        for item in build_job_evidence(job):
            lookup[item.evidence_id] = item.model_dump()
    return lookup


def evidence_source_label(source: str | None) -> str:
    """Map backend evidence sources to the four user-facing source badges."""

    normalized = str(source or "").strip().casefold()
    return {
        "resume": "Resume",
        "portfolio": "Portfolio",
        "master_skills": "Master Skills",
        "memory": "Memory",
        "job_posting": "Job Posting",
        "company_details": "Company Details",
    }.get(normalized, "Evidence")


def _evidence_badges(
    evidence_ids: list[str],
    evidence_lookup: dict[str, dict[str, Any]],
) -> str:
    """Return safe HTML badges for a list of evidence references."""

    if not evidence_ids:
        return '<span class="source-badge">No evidence reference</span>'
    badges: list[str] = []
    for evidence_id in evidence_ids:
        item = evidence_lookup.get(str(evidence_id), {})
        label = evidence_source_label(item.get("source"))
        badges.append(
            f'<span class="evidence-badge" title="{escape(str(evidence_id), quote=True)}">'
            f"{escape(label)}</span>"
        )
    return "".join(badges)


def score_components_from_rationale(rationale: str) -> dict[str, str]:
    """Extract only component explanations explicitly returned by score_jobs."""

    text = str(rationale or "").strip()
    if "--" in text:
        text = text.split("--", 1)[1]
    components: dict[str, str] = {}
    labels = {
        "skills": "Skill match",
        "skill": "Skill match",
        "experience": "Experience alignment",
        "domain": "Industry/domain alignment",
        "industry": "Industry/domain alignment",
        "location": "Location alignment",
    }
    for part in text.split(";"):
        key, separator, value = part.partition(":")
        normalized = key.strip().casefold()
        if separator and normalized in labels and value.strip():
            components[labels[normalized]] = value.strip().rstrip(".")
    return components


def render_ranked_job_card(
    ranked_item: dict[str, Any],
    selected_job_ids: list[str],
) -> None:
    """Render an honest deterministic score card using available backend fields."""

    job = ranked_item.get("job", {})
    total = ranked_item.get("score")
    components = score_components_from_rationale(ranked_item.get("rationale", ""))
    selected = job.get("job_id") in selected_job_ids
    selected_badge = (
        '<span class="status-pill status-pill--success">Top 3</span>'
        if selected
        else ""
    )
    component_html = "".join(
        (
            '<div class="score-component">'
            f'<div class="card-label">{escape(label)}</div>'
            f'<div class="card-copy">{escape(value)}</div></div>'
        )
        for label, value in components.items()
    )
    note = (
        "The backend returns component explanations but not numeric component "
        "subscores, so no additional numbers are inferred."
        if components
        else "The backend returned only a total score for this role."
    )
    st.markdown(
        f"""
        <div class="score-card">
            <div class="score-card__header">
                <div>
                    <div class="score-card__title">
                        {escape(str(job.get("title", "Role")))}
                        <span style="color:#617087;font-weight:500">
                            at {escape(str(job.get("company", "Company")))}
                        </span>
                    </div>
                    <div class="card-meta">Job ID: {escape(str(job.get("job_id", "—")))}</div>
                </div>
                <div style="text-align:right">
                    {selected_badge}
                    <div class="score-total">{escape(str(total if total is not None else "—"))}</div>
                </div>
            </div>
            <div class="deterministic-note">◆ Deterministic code score · not generated by the LLM</div>
            <div class="score-components">{component_html}</div>
            <div class="card-meta">{escape(note)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def group_rejected_jobs(
    rejected_jobs: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Group rejected jobs by each exact reason returned by the filtering tool."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rejected in rejected_jobs:
        reasons = rejected.get("reasons", []) or ["Reason not provided"]
        for reason in reasons:
            grouped[str(reason)].append(rejected)
    return dict(grouped)


def _tool_input_summary(tool_name: str, payload: dict[str, Any]) -> str:
    """Summarize a stored tool input without presenting it as reasoning."""

    job = payload.get("job", {}) if isinstance(payload, dict) else {}
    if tool_name in {"filter_jobs", "score_jobs"}:
        jobs = payload.get("jobs", []) if isinstance(payload, dict) else []
        return f"{len(jobs)} job records"
    if tool_name == "analyze_fit":
        return (
            f"{job.get('job_id', 'one job')} · "
            f"{len(payload.get('evidence_items', []))} evidence items"
        )
    if tool_name == "tailor_resume":
        revision = (
            "revision requested"
            if payload.get("revision_feedback")
            else "initial draft"
        )
        return (
            f"{job.get('job_id', 'one job')} · {revision} · "
            f"{len(payload.get('candidate_evidence', []))} evidence items"
        )
    if tool_name == "generate_cover_letter":
        return (
            f"{job.get('job_id', 'one job')} · "
            f"{len(payload.get('candidate_evidence', []))} evidence items"
        )
    return "Structured workflow input"


def _tool_output_summary(tool_name: str, payload: dict[str, Any]) -> str:
    """Summarize a stored tool output."""

    if tool_name == "filter_jobs":
        return (
            f"{len(payload.get('accepted_jobs', []))} eligible · "
            f"{len(payload.get('rejected_jobs', []))} filtered out"
        )
    if tool_name == "score_jobs":
        return (
            f"{len(payload.get('ranked_jobs', []))} ranked · "
            f"{len(payload.get('top_3_job_ids', []))} selected"
        )
    if tool_name == "analyze_fit":
        return (
            f"{payload.get('job_id', 'Job')} analyzed · "
            f"{len(payload.get('genuine_gaps', []))} genuine gaps"
        )
    if tool_name == "tailor_resume":
        return (
            f"{payload.get('job_id', 'Job')} · "
            f"{len(payload.get('change_log', []))} changes · "
            f"{payload.get('page_count', 0)} page"
        )
    if tool_name == "generate_cover_letter":
        return (
            f"{payload.get('job_id', 'Job')} · "
            f"{payload.get('page_count', 0)} page · "
            f"{len(payload.get('evidence_used', []))} evidence references"
        )
    return "Structured workflow output"


def render_agent_decisions(decisions: list[dict[str, Any]]) -> None:
    """Render controller choices separately from deterministic tool outputs."""

    if not decisions:
        st.caption("Agent decisions will appear here as the run progresses.")
        return
    for decision in decisions:
        source = (
            "LLM controller"
            if decision.get("decision_source") == "llm"
            else "Offline policy"
        )
        st.markdown(
            f"""
            <div class="decision-card">
                <div class="card-label">
                    {escape(format_phase(decision.get("phase")))} · {escape(source)}
                </div>
                <div class="change-card__title">
                    {escape(str(decision.get("selected_tool", "")).replace("_", " ").title())}
                </div>
                <div class="card-copy">
                    {escape(str(decision.get("decision_summary", "No summary provided.")))}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_tool_activity(
    history: list[dict[str, Any]],
    errors: list[dict[str, Any]] | None = None,
) -> None:
    """Render structured execution cards independently from agent decisions."""

    if not history and not errors:
        st.caption("Tool activity will appear here as the run progresses.")
        return
    for index, item in enumerate(history, start=1):
        tool_name = str(item.get("tool", "tool"))
        output = item.get("output", {}) or {}
        input_payload = item.get("input", {}) or {}
        output_errors = output.get("errors", []) if isinstance(output, dict) else []
        status = "Needs attention" if output_errors else "Complete"
        tone = "warning" if output_errors else "success"
        duration = item.get("duration_ms")
        duration_html = (
            f'<div class="card-meta">Duration: {float(duration):,.0f} ms</div>'
            if isinstance(duration, (int, float))
            else ""
        )
        error_html = (
            f'<div class="card-meta"><strong>Error:</strong> '
            f"{escape(_concise_error_message(str(output_errors[-1])))}</div>"
            if output_errors
            else ""
        )
        st.markdown(
            f"""
            <div class="tool-card">
                <div class="tool-card__header">
                    <div>
                        <div class="card-label">
                            {escape(format_phase(item.get("phase")))} · Call {index}
                        </div>
                        <div class="tool-card__title">
                            {escape(tool_name.replace("_", " ").title())}
                        </div>
                    </div>
                    <span class="status-pill status-pill--{tone}">{escape(status)}</span>
                </div>
                <div class="card-meta"><strong>Input:</strong>
                    {escape(_tool_input_summary(tool_name, input_payload))}
                </div>
                <div class="card-meta"><strong>Output:</strong>
                    {escape(_tool_output_summary(tool_name, output))}
                </div>
                {duration_html}
                {error_html}
            </div>
            """,
            unsafe_allow_html=True,
        )
    for error in errors or []:
        st.markdown(
            f"""
            <div class="tool-card">
                <div class="tool-card__header">
                    <div class="tool-card__title">
                        {escape(str(error.get("tool_name") or "Workflow step").replace("_", " ").title())}
                    </div>
                    <span class="status-pill status-pill--warning">Error</span>
                </div>
                <div class="card-meta">
                    {escape(_concise_error_message(str(error.get("message", ""))))}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _claim_text(claim: dict[str, Any]) -> str:
    return str(claim.get("claim") or claim.get("notes") or "Evidence not provided")


def render_fit_analysis(
    fit: dict[str, Any],
    evidence_lookup: dict[str, dict[str, Any]],
) -> None:
    """Render fit analysis as evidence-backed groups instead of raw JSON."""

    skill_groups = [
        ("Aligned skills", "aligned", fit.get("aligned_skills", [])),
        (
            "Missing on resume · evidenced elsewhere",
            "evidenced",
            fit.get("evidenced_missing_skills", []),
        ),
        ("Genuine gaps", "gap", fit.get("genuine_gaps", [])),
    ]
    group_html: list[str] = []
    for title, modifier, claims in skill_groups:
        items = "".join(
            (
                '<div class="skill-group__item">'
                f"{escape(_claim_text(claim))}<br>"
                f"{_evidence_badges(claim.get('evidence_ids', []), evidence_lookup)}"
                "</div>"
            )
            for claim in claims
        )
        if not items:
            items = '<div class="skill-group__item">None identified</div>'
        group_html.append(
            f'<div class="skill-group skill-group--{modifier}">'
            f'<div class="skill-group__title">{escape(title)}</div>{items}</div>'
        )
    st.markdown(
        '<div class="skill-groups">' + "".join(group_html) + "</div>",
        unsafe_allow_html=True,
    )

    dimensions = [
        ("Relevant experience", "relevant_experience"),
        ("Seniority", "seniority"),
        ("Education", "education"),
        ("Core skills", "aligned_skills"),
        ("Projects", "project_analysis"),
    ]
    for label, key in dimensions:
        claims = fit.get(key, []) or []
        with st.expander(label, expanded=key == "relevant_experience"):
            if not claims:
                st.caption("No supporting evidence was returned for this dimension.")
            for claim in claims:
                st.markdown(f"**{_claim_text(claim)}**")
                evidence_ids = claim.get("evidence_ids", [])
                st.markdown(
                    _evidence_badges(evidence_ids, evidence_lookup),
                    unsafe_allow_html=True,
                )
                for evidence_id in evidence_ids:
                    item = evidence_lookup.get(str(evidence_id))
                    if item and item.get("text"):
                        st.caption(str(item["text"]))
                if claim.get("notes"):
                    st.caption(str(claim["notes"]))

    render_project_swap(fit.get("project_swap"), evidence_lookup)
    with st.expander("Technical details · raw fit-analysis JSON"):
        st.json(fit, expanded=False)


def _labeled_evidence_value(text: str, label: str) -> str | None:
    match = re.search(rf"(?m)^{re.escape(label)}:\s*(.+?)\s*$", text or "")
    return match.group(1).strip() if match else None


def render_project_swap(
    project_swap: dict[str, Any] | None,
    evidence_lookup: dict[str, dict[str, Any]],
) -> None:
    """Render the project swap as a dedicated before/after comparison."""

    st.markdown("#### Project swap recommendation")
    if not project_swap:
        st.success(
            "The current resume projects are already the strongest available "
            "matches for this role."
        )
        return
    evidence_ids = project_swap.get("evidence_ids", [])
    evidence_items = [
        evidence_lookup.get(str(evidence_id), {}) for evidence_id in evidence_ids
    ]
    technologies = next(
        (
            _labeled_evidence_value(str(item.get("text", "")), "TECH_STACK")
            for item in evidence_items
            if _labeled_evidence_value(str(item.get("text", "")), "TECH_STACK")
        ),
        None,
    )
    domain = next(
        (
            _labeled_evidence_value(str(item.get("text", "")), "DOMAIN")
            for item in evidence_items
            if _labeled_evidence_value(str(item.get("text", "")), "DOMAIN")
        ),
        None,
    )
    rationale = str(project_swap.get("rationale") or "No rationale was returned.")
    st.markdown(
        f"""
        <div class="swap-card">
            <div class="diff-grid">
                <div class="diff-block diff-block--removed">
                    <div class="diff-block__label">Removed project</div>
                    <div class="diff-block__text">
                        {escape(str(project_swap.get("remove_project") or "Not specified"))}
                    </div>
                </div>
                <div class="diff-block diff-block--added">
                    <div class="diff-block__label">Added portfolio project</div>
                    <div class="diff-block__text">
                        {escape(str(project_swap.get("add_project") or "Not specified"))}
                    </div>
                </div>
            </div>
            <div class="card-meta"><strong>Recommendation rationale:</strong>
                {escape(rationale)}</div>
            <div class="card-meta"><strong>Reason for removal:</strong>
                Not provided separately by the backend.</div>
            <div class="card-meta"><strong>Reason for addition:</strong>
                {escape(rationale)}</div>
            <div class="card-meta"><strong>Matching technologies:</strong>
                {escape(technologies or "Not provided")}</div>
            <div class="card-meta"><strong>Matching domain or industry:</strong>
                {escape(domain or "Not provided")}</div>
            <div class="card-meta"><strong>Evidence reference:</strong><br>
                {_evidence_badges(evidence_ids, evidence_lookup)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _balanced_latex_argument(source: str, command_index: int) -> str:
    opening = source.find("{", command_index)
    if opening < 0:
        return ""
    depth = 0
    escaped = False
    for index in range(opening, len(source)):
        char = source[index]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[opening + 1 : index]
    return ""


def _marker_value(source: str, marker: str, command: str | None = None) -> str:
    marker_text = f"% AGENT-EDIT-TARGET: {marker}"
    marker_index = source.find(marker_text)
    if marker_index < 0:
        return ""
    if command:
        command_index = source.find(f"\\{command}", marker_index + len(marker_text))
        return _balanced_latex_argument(source, command_index)
    tail = source[marker_index + len(marker_text) :]
    for line in tail.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("%"):
            return stripped
    return ""


def _skills_value(source: str) -> str:
    marker = "% AGENT-EDIT-TARGET: skills"
    start = source.find(marker)
    if start < 0:
        return ""
    end = source.find("\\end{itemize}", start)
    block = source[start + len(marker) : end if end >= 0 else None]
    lines = [
        _latex_to_display(line)
        for line in block.splitlines()
        if "\\small\\item" in line
    ]
    return "\n".join(line for line in lines if line)


def _latex_to_display(value: str) -> str:
    """Convert a small resume fragment into readable text for a visual diff."""

    text = str(value or "")
    replacements = {
        r"\&": "&",
        r"\%": "%",
        r"\_": "_",
        r"\$": "$",
        "~": " ",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    previous = None
    while previous != text:
        previous = text
        text = re.sub(r"\\[A-Za-z*]+\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\[A-Za-z*]+", "", text)
    text = text.replace("{", "").replace("}", "")
    return " ".join(text.split())


def _read_text_file(path_value: str | Path) -> str:
    if not path_value:
        return ""
    path = Path(path_value)
    try:
        return path.read_text(encoding="utf-8") if path.is_file() else ""
    except OSError:
        return ""


def build_resume_change_views(
    change_log: list[dict[str, Any]],
    source_tex_path: str | Path,
    tailored_tex_path: str | Path,
    fit_analysis: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build visual before/after records from the actual resume artifacts."""

    before_source = _read_text_file(source_tex_path)
    after_source = _read_text_file(tailored_tex_path)
    views: list[dict[str, Any]] = []
    experience_index = 0
    labels = {
        "summary": "Professional summary",
        "experience": "Experience bullet",
        "skills": "Skills",
        "projects": "Project swap",
    }
    for change in change_log:
        section = str(change.get("section", "")).casefold()
        if section not in labels:
            continue
        logged_before = str(change.get("before_text") or "")
        logged_after = str(change.get("after_text") or "")
        if section == "summary":
            before = logged_before or _marker_value(before_source, "summary")
            after = logged_after or _marker_value(after_source, "summary")
            category = labels[section]
        elif section == "experience":
            experience_index += 1
            marker = f"experience-bullet-{experience_index}"
            before = logged_before or _marker_value(
                before_source, marker, "resumeItem"
            )
            after = logged_after or _marker_value(
                after_source, marker, "resumeItem"
            )
            category = f"Experience bullet {experience_index} of 2"
        elif section == "skills":
            before = logged_before or _skills_value(before_source)
            after = logged_after or _skills_value(after_source)
            category = labels[section]
        else:
            swap = fit_analysis.get("project_swap") or {}
            before = logged_before or str(swap.get("remove_project") or "")
            after = logged_after or str(swap.get("add_project") or "")
            category = labels[section]
        views.append(
            {
                "category": category,
                "before": (
                    _latex_to_display(before)
                    or "Before text is not available in the artifact."
                ),
                "after": (
                    _latex_to_display(after)
                    or "After text is not available in the artifact."
                ),
                "evidence_ids": list(change.get("evidence_ids", [])),
                "reason": str(
                    change.get("reason")
                    or change.get("description")
                    or "Reason not provided."
                ),
            }
        )
    return views


def render_resume_diffs(
    change_log: list[dict[str, Any]],
    source_tex_path: str | Path,
    tailored_tex_path: str | Path,
    fit_analysis: dict[str, Any],
    evidence_lookup: dict[str, dict[str, Any]],
) -> None:
    """Render resume changes as visual diffs with evidence provenance."""

    views = build_resume_change_views(
        change_log,
        source_tex_path,
        tailored_tex_path,
        fit_analysis,
    )
    if not views:
        st.caption("No supported resume content changes were recorded for this draft.")
    for view in views:
        st.markdown(
            f"""
            <div class="change-card">
                <div class="change-card__header">
                    <div class="change-card__title">{escape(view["category"])}</div>
                </div>
                <div class="diff-grid">
                    <div class="diff-block diff-block--removed">
                        <div class="diff-block__label">Before</div>
                        <div class="diff-block__text">{escape(view["before"])}</div>
                    </div>
                    <div class="diff-block diff-block--added">
                        <div class="diff-block__label">After</div>
                        <div class="diff-block__text">{escape(view["after"])}</div>
                    </div>
                </div>
                <div class="card-meta"><strong>Evidence source</strong><br>
                    {_evidence_badges(view["evidence_ids"], evidence_lookup)}</div>
                <div class="card-meta"><strong>Reason:</strong>
                    {escape(view["reason"])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with st.expander("Technical details · raw change-log JSON"):
        st.json(change_log, expanded=False)


def render_artifact_errors(errors: list[str] | None, artifact_label: str) -> None:
    """Present concise artifact failures while retaining full diagnostic logs."""

    if not errors:
        return
    last_error = str(errors[-1])
    if _is_latex_error(last_error):
        st.error(
            f"The {artifact_label} could not be compiled. "
            "Check the LaTeX source or runtime installation."
        )
    else:
        st.error(
            f"The {artifact_label} is unavailable: {_concise_error_message(last_error)}"
        )
    with st.expander(f"Technical details · {artifact_label} logs"):
        for error in errors:
            st.code(str(error))


def build_outputs_zip(
    jobs: dict[str, dict[str, Any]],
    tailoring: dict[str, dict[str, Any]],
    cover_letters: dict[str, dict[str, Any]],
    job_ids: list[str],
) -> bytes | None:
    """Return a ZIP containing available final PDFs and source files."""

    buffer = BytesIO()
    file_count = 0
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for job_id in job_ids:
            job = jobs.get(job_id, {})
            role_slug = re.sub(
                r"[^a-z0-9]+",
                "-",
                str(job.get("title") or job_id).casefold(),
            ).strip("-")
            folder = f"{job_id}-{role_slug or 'application'}"
            artifacts = [
                ("resume.pdf", tailoring.get(job_id, {}).get("output_pdf_path")),
                ("resume.tex", tailoring.get(job_id, {}).get("output_tex_path")),
                (
                    "cover-letter.pdf",
                    cover_letters.get(job_id, {}).get("output_pdf_path"),
                ),
                (
                    "cover-letter.tex",
                    cover_letters.get(job_id, {}).get("output_tex_path"),
                ),
            ]
            for filename, path_value in artifacts:
                path = Path(path_value) if path_value else None
                if path and path.is_file():
                    archive.write(path, arcname=f"{folder}/{filename}")
                    file_count += 1
    return buffer.getvalue() if file_count else None


def observability_summary(
    state: dict[str, Any],
    tracer: Any | None,
) -> dict[str, Any]:
    """Summarize only observability metrics available in state or local events."""

    trace_id = state.get("trace_id")
    events = []
    for event in getattr(tracer, "events", []) or []:
        if not trace_id or getattr(event, "trace_id", None) == trace_id:
            events.append(event)
    llm_calls: int | None = None
    if events:
        llm_calls = sum(
            getattr(event, "observation_type", "") == "GENERATION" for event in events
        )
    memory_writes = sum(
        len(entry.get("memory_writes", []) or [])
        for entry in state.get("review_history", []) or []
    )
    if state.get("status") == "WAITING_FOR_REVIEW":
        human_review_status = "Paused for review"
    elif state.get("review_history"):
        human_review_status = "Complete"
    elif state.get("phase") in {"COVER_LETTERS", "COMPLETE"}:
        human_review_status = "Complete"
    else:
        human_review_status = "Not reached"
    return {
        "trace_status": state.get("langfuse_status") or "Not initialized",
        "run_id": state.get("run_id"),
        "llm_calls": llm_calls,
        "tool_calls": len(state.get("tool_history", []) or []),
        "memory_writes": memory_writes,
        "human_review_status": human_review_status,
    }
