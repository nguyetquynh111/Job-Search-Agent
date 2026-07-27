"""Background execution adapter for the repository's LangGraph workflow."""

from __future__ import annotations

import copy
import os
import shutil
import threading
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.services.artifact_loader import repository_root, snapshot_from_live_state
from app.services.run_state import RunSnapshot


class AgentAdapterError(RuntimeError):
    """Safe, user-facing live integration error."""


@dataclass
class LiveReadiness:
    ready: bool
    missing_environment: list[str]
    missing_files: list[str]
    pdflatex_available: bool


@dataclass
class LiveRunHandle:
    """Thread-safe resources and latest state for one background run."""

    run_id: str
    thread_id: str
    repo_root: Path
    result: dict[str, Any]
    app: Any | None = None
    tracer: Any | None = None
    review_submitted: bool = False
    worker: threading.Thread | None = field(default=None, repr=False)
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    @property
    def worker_running(self) -> bool:
        worker = self.worker
        return bool(worker and worker.is_alive())


_RUNS: dict[str, LiveRunHandle] = {}
_RUNS_LOCK = threading.RLock()


def live_readiness(repo_root: Path | None = None) -> LiveReadiness:
    """Check names/presence only; never return secret environment values."""

    root = (repo_root or repository_root()).resolve()
    _load_repository_dotenv(root)
    required_environment = (
        "LLM_MODEL",
        "DEEPINFRA_API_KEY",
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
    )
    missing_environment = [
        name for name in required_environment if not os.getenv(name, "").strip()
    ]
    required_files = (
        "data/jobs.csv",
        "data/preferences.yaml",
        "data/resume.tex",
        "data/portfolio.txt",
    )
    missing_files = [name for name in required_files if not (root / name).is_file()]
    latex = shutil.which("pdflatex") is not None
    return LiveReadiness(
        ready=not missing_environment and not missing_files and latex,
        missing_environment=missing_environment,
        missing_files=missing_files,
        pdflatex_available=latex,
    )


def start_live_run(repo_root: Path | None = None) -> LiveRunHandle:
    """Create a run immediately and execute it on a background worker."""

    root = (repo_root or repository_root()).resolve()
    readiness = live_readiness(root)
    if not readiness.ready:
        missing = [
            *readiness.missing_environment,
            *readiness.missing_files,
            *([] if readiness.pdflatex_available else ["pdflatex"]),
        ]
        raise AgentAdapterError(
            "Live Run is not ready. Missing: " + ", ".join(missing)
        )

    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
    run_id = f"ui-live-{stamp}"
    thread_id = f"thread-{run_id}"
    handle = LiveRunHandle(
        run_id=run_id,
        thread_id=thread_id,
        repo_root=root,
        result={
            "run_id": run_id,
            "thread_id": thread_id,
            "phase": "INITIALIZE",
            "status": "CREATED",
            "errors": [],
        },
    )
    worker = threading.Thread(
        target=_run_new_worker,
        args=(handle, root),
        name=f"job-search-{run_id}",
        daemon=True,
    )
    handle.worker = worker
    with _RUNS_LOCK:
        _RUNS[run_id] = handle
    worker.start()
    return handle


def get_live_run(run_id: str) -> LiveRunHandle | None:
    """Return an in-process live run by ID for UI polling."""

    with _RUNS_LOCK:
        return _RUNS.get(run_id)


def submit_human_review(
    handle: LiveRunHandle,
    submission: dict[str, Any],
) -> None:
    """Validate review input and resume the graph on a background worker."""

    with handle.lock:
        if handle.worker_running:
            raise AgentAdapterError("The agent is still running.")
        if handle.review_submitted:
            raise AgentAdapterError("This review action was already submitted.")

    expected = set(
        (snapshot_for_handle(handle).interrupt_payload.get("resumes") or {}).keys()
    )
    if not expected:
        raise AgentAdapterError("The live graph is not waiting for human review.")
    decisions = submission.get("decisions")
    if not isinstance(decisions, dict):
        action = str(submission.get("action", ""))
        affected = set(submission.get("affected_job_ids", []))
        feedback = str(submission.get("reviewer_feedback", "")).strip()
        if (
            action not in {"approve", "request_revision"}
            or affected - expected
            or (action == "request_revision" and (not affected or not feedback))
        ):
            raise AgentAdapterError("Submit one decision for every Top 3 resume.")
        decisions = {
            job_id: {
                "decision": (
                    "reject"
                    if action == "request_revision" and job_id in affected
                    else "approve"
                ),
                "comment": feedback if job_id in affected else "",
            }
            for job_id in expected
        }
        submission = {"decisions": decisions}
    if set(decisions) != expected:
        raise AgentAdapterError("Submit one decision for every Top 3 resume.")
    invalid = [
        job_id
        for job_id, decision in decisions.items()
        if not isinstance(decision, dict)
        or decision.get("decision") not in {"approve", "reject"}
        or (
            decision.get("decision") == "reject"
            and not str(decision.get("comment", "")).strip()
        )
    ]
    if invalid:
        raise AgentAdapterError(
            "Rejected resumes require comments; invalid decisions: "
            + ", ".join(sorted(invalid))
        )

    with handle.lock:
        if handle.app is None:
            raise AgentAdapterError("The live graph is unavailable.")
        handle.review_submitted = True
        provisional = copy.deepcopy(handle.result)
        provisional["status"] = "RUNNING"
        provisional["phase"] = (
            "REVISION"
            if any(
                item.get("decision") == "reject" for item in decisions.values()
            )
            else "COVER_LETTERS"
        )
        handle.result = provisional
        worker = threading.Thread(
            target=_run_review_worker,
            args=(handle, copy.deepcopy(submission)),
            name=f"job-search-review-{handle.run_id}",
            daemon=True,
        )
        handle.worker = worker
    worker.start()


def snapshot_for_handle(handle: LiveRunHandle) -> RunSnapshot:
    """Return a consistent snapshot while the worker continues in background."""

    with handle.lock:
        state = copy.deepcopy(handle.result)
        tracer = handle.tracer
        root = handle.repo_root
    refresh_trace_url = getattr(tracer, "refresh_trace_url", None)
    if callable(refresh_trace_url):
        refresh_trace_url()
    with handle.lock:
        events = [
            asdict(event) if hasattr(event, "__dataclass_fields__") else dict(event)
            for event in list(getattr(tracer, "events", []))
        ]
        trace_url = getattr(tracer, "trace_url", None)
        if tracer is not None:
            state.update(
                {
                    "trace_id": getattr(tracer, "trace_id", state.get("trace_id")),
                    "trace_public": bool(getattr(tracer, "trace_public", False)),
                    "trace_ingest_confirmed": bool(
                        getattr(tracer, "trace_ingest_confirmed", False)
                    ),
                    "observation_count": int(
                        getattr(tracer, "observation_count", 0)
                    ),
                    "trace_export_error": getattr(
                        tracer, "trace_export_error", None
                    ),
                    "trace_debug_status": getattr(
                        tracer, "trace_debug_status", None
                    ),
                }
            )
    return snapshot_from_live_state(
        state,
        repo_root=root,
        trace_events=events,
        trace_url=trace_url,
    )


def _run_new_worker(handle: LiveRunHandle, root: Path) -> None:
    _set_running(handle, phase="INITIALIZE")
    tracer: Any | None = None
    try:
        from src.agent import (
            DeepInfraToolSelectionModel,
            build_agent_graph,
            create_initial_state,
            create_memory_checkpointer,
            invoke_new_run,
        )
        from src.tracing.langfuse import TraceManager

        tracer = TraceManager()
        selector = DeepInfraToolSelectionModel(
            configured_model_name=os.environ["LLM_MODEL"]
        )
        graph = build_agent_graph(
            checkpointer=create_memory_checkpointer(),
            tracer=tracer,
            tool_selection_model=selector,
            progress_callback=lambda state: _store_progress(handle, state),
        )
        state = create_initial_state(
            jobs_path=str(root / "data" / "jobs.csv"),
            candidate_profile_path=str(root / "data" / "preferences.yaml"),
            resume_path=str(root / "data" / "resume.tex"),
            portfolio_path=str(root / "data" / "portfolio.txt"),
            memory_file=str(root / "outputs" / "memory.json"),
            run_id=handle.run_id,
            thread_id=handle.thread_id,
        )
        with handle.lock:
            handle.app = graph
            handle.tracer = tracer
        _store_progress(handle, state)
        result = invoke_new_run(graph, state)
        _store_progress(handle, result)
    except Exception as exc:
        _finalize_failed_trace(tracer, exc)
        _store_failure(handle, "The agent could not reach human review", exc)


def _run_review_worker(
    handle: LiveRunHandle,
    submission: dict[str, Any],
) -> None:
    try:
        from src.agent import resume_run

        result = resume_run(handle.app, handle.thread_id, submission)
        _store_progress(handle, result)
    except Exception as exc:
        _finalize_failed_trace(handle.tracer, exc)
        _store_failure(handle, "The agent could not resume", exc)


def _finalize_failed_trace(tracer: Any | None, exc: Exception) -> None:
    """End and flush a failed run without allowing tracing to mask the failure."""

    if tracer is None:
        return
    try:
        tracer.record_error(exc, {"run_status": "FAILED"})
        tracer.update_run(
            metadata={"status": "FAILED"},
            output={"error_type": exc.__class__.__name__, "error": str(exc)},
        )
    finally:
        tracer.end_root_observation()
        tracer.flush()


def _set_running(handle: LiveRunHandle, *, phase: str) -> None:
    with handle.lock:
        state = copy.deepcopy(handle.result)
        state["status"] = "RUNNING"
        state["phase"] = phase
        handle.result = state


def _store_progress(handle: LiveRunHandle, state: Mapping[str, Any]) -> None:
    with handle.lock:
        handle.result = copy.deepcopy(dict(state))


def _store_failure(handle: LiveRunHandle, message: str, exc: Exception) -> None:
    with handle.lock:
        state = copy.deepcopy(handle.result)
        state["status"] = "FAILED"
        state["phase"] = state.get("phase") or "ERROR"
        errors = list(state.get("errors", []))
        errors.append(
            {
                "phase": state["phase"],
                "type": exc.__class__.__name__,
                "message": f"{message}: {exc}",
            }
        )
        state["errors"] = errors
        handle.result = state


def _load_repository_dotenv(root: Path) -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(root / ".env", override=False)
    except Exception:
        return
