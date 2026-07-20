"""Langfuse observability tests."""

from __future__ import annotations

import logging
import sys
import types
from pathlib import Path
from typing import Any

from src.agent.graph import build_agent_graph, create_memory_checkpointer, invoke_new_run
from src.agent.state import create_initial_state
from src.observability.langfuse_client import (
    DEFAULT_LANGFUSE_HOST,
    STATUS_CONNECTED,
    STATUS_NOOP,
    STATUS_UNAVAILABLE,
)
from src.observability.trace_manager import ROOT_TRACE_NAME, TraceManager
from src.tools.registry import load_tool_registry
from src.ui.session import ensure_session_defaults, store_graph_result


class FakeLangfuseClient:
    """In-memory Langfuse v2-compatible client."""

    def __init__(self) -> None:
        self.trace_calls: list[dict[str, Any]] = []
        self.span_calls: list[dict[str, Any]] = []
        self.generation_calls: list[dict[str, Any]] = []
        self.event_calls: list[dict[str, Any]] = []
        self.flush_count = 0

    def auth_check(self) -> bool:
        return True

    def trace(self, **kwargs: Any) -> object:
        self.trace_calls.append(kwargs)
        return types.SimpleNamespace(url=None)

    def span(self, **kwargs: Any) -> object:
        self.span_calls.append(kwargs)
        return types.SimpleNamespace()

    def generation(self, **kwargs: Any) -> object:
        self.generation_calls.append(kwargs)
        return types.SimpleNamespace()

    def event(self, **kwargs: Any) -> object:
        self.event_calls.append(kwargs)
        return types.SimpleNamespace()

    def flush(self) -> None:
        self.flush_count += 1


class FailingLangfuseClient(FakeLangfuseClient):
    """Client that raises on remote tracing calls."""

    def trace(self, **kwargs: Any) -> object:
        raise RuntimeError("remote unavailable")

    def span(self, **kwargs: Any) -> object:
        raise RuntimeError("remote unavailable")


def test_valid_langfuse_configuration_selects_real_tracer(monkeypatch) -> None:
    """Valid env and auth create an enabled Langfuse tracer."""

    instances: list[Any] = []

    class FakeLangfuse(FakeLangfuseClient):
        def __init__(self, public_key: str, secret_key: str, host: str) -> None:
            super().__init__()
            self.public_key = public_key
            self.secret_key = secret_key
            self.host = host
            instances.append(self)

    module = types.ModuleType("langfuse")
    module.Langfuse = FakeLangfuse
    monkeypatch.setitem(sys.modules, "langfuse", module)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "public-test-value")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "secret-test-value")
    monkeypatch.setenv("LANGFUSE_HOST", DEFAULT_LANGFUSE_HOST)

    tracer = TraceManager()

    assert tracer.enabled is True
    assert tracer.status_message == STATUS_CONNECTED
    assert instances[0].host == DEFAULT_LANGFUSE_HOST


def test_missing_langfuse_credentials_select_noop_tracer(caplog) -> None:
    """Missing keys keep observability local."""

    caplog.set_level(logging.INFO)

    tracer = TraceManager()

    assert tracer.enabled is False
    assert tracer.status_message == STATUS_NOOP
    assert "Langfuse disabled because configuration is missing" in caplog.text


def test_langfuse_initialization_failure_selects_noop_tracer(
    monkeypatch, caplog
) -> None:
    """SDK initialization errors fall back without exposing exception details."""

    class FailingLangfuse:
        def __init__(self, public_key: str, secret_key: str, host: str) -> None:
            raise RuntimeError("boom")

    module = types.ModuleType("langfuse")
    module.Langfuse = FailingLangfuse
    monkeypatch.setitem(sys.modules, "langfuse", module)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "public-test-value")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "secret-test-value")
    monkeypatch.setenv("LANGFUSE_HOST", DEFAULT_LANGFUSE_HOST)
    caplog.set_level(logging.WARNING)

    tracer = TraceManager()

    assert tracer.enabled is False
    assert tracer.status_message == STATUS_UNAVAILABLE
    assert "Langfuse initialization failed; using no-op tracing" in caplog.text


def test_tracing_failure_does_not_terminate_agent_workflow(
    tmp_path: Path, monkeypatch
) -> None:
    """Remote tracing failures do not stop the graph from reaching review."""

    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "outputs"))
    memory_file = tmp_path / "memory.json"
    memory_file.write_text("[]", encoding="utf-8")
    tracer = TraceManager(client=FailingLangfuseClient(), enabled=True)
    app = build_agent_graph(
        tools=load_tool_registry(),
        checkpointer=create_memory_checkpointer(),
        tracer=tracer,
    )
    state = create_initial_state(
        thread_id="thread-tracing-failure",
        run_id="run-tracing-failure",
        memory_file=str(memory_file),
    )

    result = invoke_new_run(app, state)

    assert result["status"] == "WAITING_FOR_REVIEW"
    assert result.get("__interrupt__")
    assert result["langfuse_status"] == STATUS_UNAVAILABLE


def test_same_trace_id_is_propagated_through_nested_stages() -> None:
    """Root, spans, and generations use one trace ID."""

    client = FakeLangfuseClient()
    tracer = TraceManager(client=client, enabled=True)

    trace_id = tracer.start_run("run-propagation", "thread-propagation")
    with tracer.span("outer_stage"):
        with tracer.span("inner_stage"):
            pass
    tracer.record_generation({"tool_name": "filter_jobs", "decision_summary": "safe"})

    assert trace_id == "trace-run-propagation"
    assert client.trace_calls == [
        {
            "id": "trace-run-propagation",
            "name": ROOT_TRACE_NAME,
            "session_id": "thread-propagation",
            "metadata": {
                "run_id": "run-propagation",
                "session_id": "thread-propagation",
            },
        }
    ]
    assert {call["trace_id"] for call in client.span_calls} == {trace_id}
    assert {call["trace_id"] for call in client.generation_calls} == {trace_id}
    assert {event.trace_id for event in tracer.events} == {trace_id}


def test_streamlit_rerun_state_does_not_duplicate_root_trace() -> None:
    """A rerun that only rehydrates session state does not start another trace."""

    client = FakeLangfuseClient()
    tracer = TraceManager(client=client, enabled=True)
    trace_id = tracer.start_run("run-rerun", "thread-rerun")
    session: dict[str, Any] = {}
    ensure_session_defaults(session)
    store_graph_result(
        session,
        {
            "run_id": "run-rerun",
            "thread_id": "thread-rerun",
            "trace_id": trace_id,
            "status": "WAITING_FOR_REVIEW",
        },
    )

    ensure_session_defaults(session)
    tracer.start_run("run-rerun", "thread-rerun")

    assert session["current_run_id"] == "run-rerun"
    assert session["current_thread_id"] == "thread-rerun"
    assert len(client.trace_calls) == 1


def test_sensitive_credentials_do_not_appear_in_logs_or_status(
    monkeypatch, caplog
) -> None:
    """Secrets are never rendered in logs or status messages."""

    public_key = "public-test-value"
    secret_key = "secret-test-value"

    class FailingLangfuse:
        def __init__(self, public_key: str, secret_key: str, host: str) -> None:
            raise RuntimeError(f"bad credentials: {secret_key}")

    module = types.ModuleType("langfuse")
    module.Langfuse = FailingLangfuse
    monkeypatch.setitem(sys.modules, "langfuse", module)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", public_key)
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", secret_key)
    monkeypatch.setenv("LANGFUSE_HOST", DEFAULT_LANGFUSE_HOST)
    caplog.set_level(logging.WARNING)

    tracer = TraceManager()

    assert public_key not in caplog.text
    assert secret_key not in caplog.text
    assert public_key not in tracer.status_message
    assert secret_key not in tracer.status_message
