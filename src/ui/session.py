"""Streamlit session-state helpers.

The graph checkpoint remains the authoritative workflow state. These helpers only
store UI/session values and the last interrupt payload.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from pathlib import Path
import shutil
import tempfile
from typing import Any
from uuid import uuid4

from src.config import get_config
from src.memory.store import JSONMemoryStore

DEFAULT_INPUT_PATHS = {
    "jobs_path": "data/demo_jobs.csv",
    "preferences_path": "data/demo_preferences.yaml",
    "resume_path": "data/demo_resume.tex",
    "portfolio_path": "data/demo_portfolio.txt",
}

UPLOAD_FILENAMES = {
    "jobs_path": "job-listings.csv",
    "preferences_path": "preferences.yaml",
    "resume_path": "resume.tex",
    "portfolio_path": "portfolio.txt",
}

UPLOAD_EXTENSIONS = {
    "jobs_path": {".csv"},
    "preferences_path": {".yaml"},
    "resume_path": {".tex"},
    "portfolio_path": {".txt"},
}


def ensure_session_defaults(session: MutableMapping[str, Any]) -> None:
    """Initialize UI-only session keys without creating a graph run."""

    memory_file = get_config().memory_file
    JSONMemoryStore(memory_file).load()
    session.setdefault("current_thread_id", None)
    session.setdefault("current_run_id", None)
    session.setdefault("waiting_for_review", False)
    session.setdefault("interrupt_payload", None)
    session.setdefault("last_result", None)
    input_paths = session.setdefault("input_paths", dict(DEFAULT_INPUT_PATHS))
    input_paths["memory_file"] = str(memory_file)


def start_graph_run(app: Any, session: MutableMapping[str, Any]) -> dict[str, Any]:
    """Start a new graph run from current UI input paths."""

    # Keep the upload UI importable even when optional workflow dependencies are
    # not installed yet. The graph is only needed after the user starts a run.
    from src.agent.graph import invoke_new_run
    from src.agent.state import create_initial_state

    paths = session.get("input_paths", DEFAULT_INPUT_PATHS)
    memory_file = get_config().memory_file
    JSONMemoryStore(memory_file).load()
    state = create_initial_state(
        jobs_path=paths.get("jobs_path", DEFAULT_INPUT_PATHS["jobs_path"]),
        candidate_profile_path=paths.get(
            "preferences_path", DEFAULT_INPUT_PATHS["preferences_path"]
        ),
        resume_path=paths.get("resume_path", DEFAULT_INPUT_PATHS["resume_path"]),
        portfolio_path=paths.get("portfolio_path", DEFAULT_INPUT_PATHS["portfolio_path"]),
        memory_file=str(memory_file),
    )
    result = invoke_new_run(app, state)
    store_graph_result(session, result)
    return result


def save_uploaded_inputs(
    session: MutableMapping[str, Any], uploads: dict[str, Any]
) -> dict[str, str]:
    """Persist uploaded files for the lifetime of the Streamlit session."""

    upload_id = session.setdefault("upload_id", uuid4().hex)
    upload_dir = Path(tempfile.gettempdir()) / "job-search-agent" / str(upload_id)
    upload_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, str] = {}
    for key, filename in UPLOAD_FILENAMES.items():
        uploaded_file = uploads.get(key)
        if uploaded_file is None:
            raise ValueError(f"Missing upload: {key}")
        suffix = Path(uploaded_file.name).suffix.lower()
        if suffix not in UPLOAD_EXTENSIONS[key]:
            allowed = ", ".join(sorted(UPLOAD_EXTENSIONS[key]))
            raise ValueError(f"{uploaded_file.name} must use one of: {allowed}")
        data = uploaded_file.getvalue()
        if not data:
            raise ValueError(f"Uploaded file is empty: {uploaded_file.name}")
        target = upload_dir / filename
        target.write_bytes(data)
        paths[key] = str(target)

    memory_file = get_config().memory_file
    JSONMemoryStore(memory_file).load()
    paths["memory_file"] = str(memory_file)
    session["input_paths"] = paths
    session["upload_dir"] = str(upload_dir)
    return paths


def resume_graph_run(
    app: Any, session: MutableMapping[str, Any], feedback: dict[str, Any]
) -> dict[str, Any]:
    """Resume the current graph run with human-review feedback."""

    thread_id = session.get("current_thread_id")
    if not thread_id:
        raise RuntimeError("No current thread_id is available to resume.")
    from src.agent.graph import resume_run

    result = resume_run(app, thread_id, feedback)
    store_graph_result(session, result)
    return result


def store_graph_result(session: MutableMapping[str, Any], result: dict[str, Any]) -> None:
    """Store UI-visible values from a graph result."""

    session["last_result"] = result
    if result.get("run_id"):
        session["current_run_id"] = result["run_id"]
    if result.get("thread_id"):
        session["current_thread_id"] = result["thread_id"]
    interrupts = result.get("__interrupt__", [])
    if interrupts:
        session["waiting_for_review"] = True
        session["interrupt_payload"] = interrupts[0].value
    elif (
        result.get("status") == "WAITING_FOR_REVIEW"
        and result.get("interrupt_payload")
    ):
        session["waiting_for_review"] = True
        session["interrupt_payload"] = result["interrupt_payload"]
    else:
        session["waiting_for_review"] = False
        session["interrupt_payload"] = None


def get_checkpoint_state(app: Any, thread_id: str | None) -> dict[str, Any]:
    """Read the authoritative state from LangGraph's checkpointer."""

    if app is None or not thread_id:
        return {}
    snapshot = app.get_state({"configurable": {"thread_id": thread_id}})
    return dict(snapshot.values or {})


def reset_demo_data(session: MutableMapping[str, Any]) -> None:
    """Reset UI state, memory file, and generated outputs."""

    config = get_config()
    JSONMemoryStore(config.memory_file).reset()
    output_dir = config.output_dir
    checkpoint_path = config.checkpoint_db.resolve()
    protected_runtime_files = {
        config.memory_file.resolve(),
        checkpoint_path,
        Path(f"{checkpoint_path}-journal"),
        Path(f"{checkpoint_path}-shm"),
        Path(f"{checkpoint_path}-wal"),
    }
    if output_dir.exists():
        for path in output_dir.iterdir():
            if path.name == ".gitkeep" or path.resolve() in protected_runtime_files:
                continue
            if path.is_file():
                path.unlink()
    upload_dir = session.get("upload_dir")
    if upload_dir:
        shutil.rmtree(upload_dir, ignore_errors=True)
    for key in [
        "current_thread_id",
        "current_run_id",
        "waiting_for_review",
        "interrupt_payload",
        "last_result",
        "input_paths",
        "upload_id",
        "upload_dir",
        "jobs_upload",
        "preferences_upload",
        "resume_upload",
        "portfolio_upload",
    ]:
        session.pop(key, None)
    ensure_session_defaults(session)
