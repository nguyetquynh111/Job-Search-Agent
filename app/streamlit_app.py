"""Minimal Streamlit interface for the existing Job Search Agent."""

# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st

from app.components.input_review import (
    InputReview,
    InputReviewItem,
    load_review_item,
    render_input_review,
)
from app.components.workspace import render_workflow_strip, render_workspace
from app.services.agent_adapter import (
    AgentAdapterError,
    LiveRunHandle,
    get_live_run,
    live_readiness,
    snapshot_for_handle,
    start_live_run,
    submit_human_review,
)
from app.services.artifact_loader import (
    ArtifactLoadError,
    discover_existing_runs,
    load_existing_run,
)
from app.services.run_state import RunSnapshot

INPUTS = [
    ("Resume", "resume.tex", ["tex"]),
    ("Jobs dataset", "jobs.csv", ["csv"]),
    ("Portfolio", "portfolio.txt", ["txt"]),
    ("Preferences", "preferences.yaml", ["yaml", "yml"]),
]


def main() -> None:
    st.set_page_config(
        page_title="Job Search Agent",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    _apply_styles()
    _render_header()

    saved_run_id = st.session_state.get("opened_run_id")
    if saved_run_id:
        snapshot = _load_saved_run(str(saved_run_id))
        render_workflow_strip(snapshot, inputs_ready=True)
        _render_run_notices(snapshot)
        render_workspace(snapshot)
        return

    live_run_id = st.session_state.get("live_run_id")
    if live_run_id:
        handle = get_live_run(str(live_run_id))
        if handle is None:
            st.error("This live worker is no longer available. Start a new run.")
            st.session_state.pop("live_run_id", None)
            return
        if handle.worker_running:
            _render_polling_live_run(handle.run_id)
        else:
            _render_live_run(handle)
        return

    review = _render_uploads()
    render_input_review(review)
    _render_start_button(review)


def _render_header() -> None:
    title_col, action_col = st.columns([0.84, 0.16], vertical_alignment="center")
    with title_col:
        st.title("Job Search Agent")
    with action_col:
        if st.session_state.get("opened_run_id"):
            if st.button("Back", width="stretch"):
                st.session_state.pop("opened_run_id", None)
                st.rerun()
        else:
            _render_saved_run_picker()


def _render_saved_run_picker() -> None:
    runs = discover_existing_runs(REPO_ROOT)
    with st.popover("Open run", width="stretch"):
        if not runs:
            st.caption("No saved runs")
            return
        selected = st.selectbox(
            "Saved run",
            runs,
            label_visibility="collapsed",
            key="saved_run_picker",
        )
        if st.button("Open", type="primary", width="stretch"):
            st.session_state["opened_run_id"] = selected
            st.rerun()


def _load_saved_run(run_id: str) -> RunSnapshot:
    try:
        return load_existing_run(run_id, REPO_ROOT)
    except ArtifactLoadError as exc:
        st.error(str(exc))
        return _empty_snapshot(run_id)


def _render_uploads() -> InputReview:
    items: dict[str, InputReviewItem] = {}
    rows = [INPUTS[:2], INPUTS[2:]]
    for row in rows:
        columns = st.columns(2, gap="medium")
        for column, (title, filename, file_types) in zip(columns, row, strict=True):
            with column:
                items[filename] = _render_upload_card(title, filename, file_types)
    return InputReview(items)


def _render_upload_card(
    title: str,
    filename: str,
    file_types: list[str],
) -> InputReviewItem:
    path = REPO_ROOT / "data" / filename
    with st.container(border=True):
        st.markdown(f"**{title}**")
        uploaded = st.file_uploader(
            title,
            type=file_types,
            key=f"input-upload-{filename}",
            label_visibility="collapsed",
        )
        if uploaded is not None:
            if _upload_extension_allowed(uploaded.name, file_types):
                _set_upload_error(filename, None)
                _persist_upload(uploaded.getvalue(), path)
            else:
                _set_upload_error(filename, uploaded.name)
                allowed = ", ".join(f".{suffix}" for suffix in file_types)
                st.error(
                    f"Rejected `{uploaded.name}`. Only these file types are allowed: "
                    f"{allowed}"
                )
                return InputReviewItem(
                    title=title,
                    filename=filename,
                    error=f"{title}: unsupported file type for {uploaded.name}.",
                    uploaded=True,
                )
        else:
            _set_upload_error(filename, None)
            return InputReviewItem(title=title, filename=filename)
        item = load_review_item(title, filename, path)
        if item.valid:
            source = uploaded.name if uploaded is not None else filename
            st.caption(f"✓ Ready · {source}")
        elif item.error:
            st.caption("Needs attention")
        return item


def _upload_extension_allowed(name: str, allowed_extensions: list[str]) -> bool:
    """Accept only the extensions explicitly configured for this input."""

    suffix = Path(name).suffix.casefold().removeprefix(".")
    allowed = {
        extension.casefold().removeprefix(".")
        for extension in allowed_extensions
    }
    return bool(suffix) and suffix in allowed


def _set_upload_error(expected_filename: str, rejected_name: str | None) -> None:
    errors = dict(st.session_state.get("invalid_uploads", {}))
    if rejected_name:
        errors[expected_filename] = rejected_name
    else:
        errors.pop(expected_filename, None)
    st.session_state["invalid_uploads"] = errors


def _render_start_button(review: InputReview) -> None:
    readiness = live_readiness(REPO_ROOT)
    invalid_uploads = dict(st.session_state.get("invalid_uploads", {}))
    ready = bool(readiness.ready and not invalid_uploads and review.valid)
    left, right = st.columns([0.76, 0.24], vertical_alignment="bottom")
    with left:
        missing_runtime = [
            *readiness.missing_environment,
            *([] if readiness.pdflatex_available else ["pdflatex"]),
        ]
        if invalid_uploads:
            rejected = ", ".join(str(name) for name in invalid_uploads.values())
            st.error(f"Remove or replace rejected files before starting: {rejected}")
        elif review.errors:
            st.error(
                "Fix invalid input data before starting. Open "
                "“Review uploaded inputs” for details."
            )
        elif review.missing:
            st.caption("Upload all four input files to start a new run.")
        if _repository_inputs_ready() and missing_runtime:
            st.caption("Missing: " + ", ".join(missing_runtime))
    with right:
        if st.button(
            "Start run",
            type="primary",
            width="stretch",
            disabled=not ready,
        ):
            try:
                handle = start_live_run(REPO_ROOT)
                st.session_state["live_run_id"] = handle.run_id
                st.rerun()
            except AgentAdapterError as exc:
                st.error(str(exc))


@st.fragment(run_every="1s")
def _render_polling_live_run(run_id: str) -> None:
    """Poll the background worker without blocking the Streamlit request."""

    handle = get_live_run(run_id)
    if handle is None:
        st.error("This live worker is no longer available.")
        return
    _render_live_run(handle)
    if not handle.worker_running:
        st.rerun()


def _render_live_run(handle: LiveRunHandle) -> None:
    snapshot = snapshot_for_handle(handle)
    render_workflow_strip(snapshot, inputs_ready=True)
    _render_run_notices(snapshot)
    submission = render_workspace(snapshot)
    if submission:
        _resume_after_review(handle.run_id, submission)


def _resume_after_review(
    run_id: str,
    submission: dict[str, object],
) -> None:
    handle = get_live_run(run_id)
    if handle is None:
        st.error("Live run unavailable.")
        return
    try:
        submit_human_review(handle, submission)
        st.rerun()
    except AgentAdapterError as exc:
        st.error(str(exc))


def _persist_upload(data: bytes, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.read_bytes() == data:
        return
    temporary = destination.with_suffix(destination.suffix + ".uploading")
    temporary.write_bytes(data)
    temporary.replace(destination)


def _repository_inputs_ready() -> bool:
    return all((REPO_ROOT / "data" / filename).is_file() for _, filename, _ in INPUTS)


def _render_run_notices(snapshot: RunSnapshot) -> None:
    if snapshot.errors:
        with st.expander("Errors", expanded=True):
            st.json(snapshot.errors, expanded=1)
    for warning in snapshot.warnings:
        st.warning(warning)


def _empty_snapshot(run_id: str) -> RunSnapshot:
    return RunSnapshot(
        mode="Existing Run",
        repo_root=REPO_ROOT,
        run_id=run_id,
        run_dir=None,
        status="CREATED",
        phase="INITIALIZE",
        read_only=True,
    )


def _apply_styles() -> None:
    css_path = APP_DIR / "styles" / "minimal.css"
    try:
        css = css_path.read_text(encoding="utf-8")
    except OSError:
        return
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
