"""Langfuse observability tests."""

from __future__ import annotations

import logging
import sys
import types
from pathlib import Path
from typing import Any

import pytest
from pypdf import PdfWriter

from src.agent.graph import (
    build_agent_graph,
    create_memory_checkpointer,
    invoke_new_run,
)
from src.agent.state import create_initial_state
from src.observability.langfuse_client import (
    DEFAULT_LANGFUSE_HOST,
    STATUS_CONNECTED,
    STATUS_NOOP,
    STATUS_UNAVAILABLE,
)
from src.observability.trace_manager import ROOT_TRACE_NAME, TraceManager
from src.tools import tailor_resume as tailoring_module
from src.tools.registry import load_tool_registry
from src.ui.session import ensure_session_defaults, store_graph_result


class FakeLangfuseClient:
    """In-memory Langfuse v2-compatible client."""

    def __init__(self) -> None:
        self.trace_calls: list[dict[str, Any]] = []
        self.span_calls: list[dict[str, Any]] = []
        self.generation_calls: list[dict[str, Any]] = []
        self.event_calls: list[dict[str, Any]] = []
        self.trace_updates: list[dict[str, Any]] = []
        self.flush_count = 0

    def auth_check(self) -> bool:
        return True

    def trace(self, **kwargs: Any) -> object:
        self.trace_calls.append(kwargs)
        return types.SimpleNamespace(
            get_trace_url=lambda: "https://example.test/trace",
            update=lambda **update: self.trace_updates.append(update),
        )

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


def test_missing_langfuse_credentials_fail_fast() -> None:
    """Default tracing requires connected Langfuse credentials."""

    with pytest.raises(RuntimeError) as exc_info:
        TraceManager()

    message = str(exc_info.value)
    assert "Langfuse credentials are required" in message
    assert "LANGFUSE_PUBLIC_KEY" in message
    assert "LANGFUSE_SECRET_KEY" in message


def test_langfuse_initialization_failure_fails_fast(
    monkeypatch, caplog
) -> None:
    """SDK initialization errors stop startup without exposing details."""

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

    with pytest.raises(RuntimeError, match="connected tracing is required"):
        TraceManager()

    assert "Langfuse initialization failed" in caplog.text


def test_tracing_failure_does_not_terminate_agent_workflow(
    tmp_path: Path, monkeypatch
) -> None:
    """Remote tracing failures do not stop the graph from reaching review."""

    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "outputs"))

    def compile_resume(text: str, tex_path: Path, pdf_path: Path):
        tex_path.write_text(text, encoding="utf-8")
        writer = PdfWriter()
        writer.add_blank_page(width=612, height=792)
        with pdf_path.open("wb") as handle:
            writer.write(handle)
        return 1, [], text

    monkeypatch.setattr(tailoring_module, "_compile_one_page", compile_resume)
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
    """Root, spans, and generations use one trace ID and correct parents."""

    client = FakeLangfuseClient()
    tracer = TraceManager(client=client, enabled=True)

    trace_id = tracer.start_run("run-propagation", "thread-propagation")
    with tracer.span("outer_stage"):
        with tracer.span("inner_stage"):
            tracer.record_generation(
                {"tool_name": "filter_jobs", "decision_summary": "safe"},
                name="test_generation",
                model="test-model",
                messages=[("human", "safe prompt")],
                response="safe response",
                usage={"input_tokens": 2, "output_tokens": 2},
                model_parameters={"temperature": 0},
            )

    assert trace_id == "trace-run-propagation"
    assert len(client.trace_calls) == 1
    assert client.trace_calls[0]["id"] == trace_id
    assert client.trace_calls[0]["name"] == ROOT_TRACE_NAME
    assert client.trace_calls[0]["session_id"] == "thread-propagation"
    assert client.trace_calls[0]["public"] is True
    assert {call["trace_id"] for call in client.span_calls} == {trace_id}
    assert {call["trace_id"] for call in client.generation_calls} == {trace_id}
    assert {event.trace_id for event in tracer.events} == {trace_id}
    outer = next(call for call in client.span_calls if call["name"] == "outer_stage")
    inner = next(call for call in client.span_calls if call["name"] == "inner_stage")
    generation = client.generation_calls[0]
    assert outer["parent_observation_id"] is None
    assert inner["parent_observation_id"] == outer["id"]
    assert generation["parent_observation_id"] == inner["id"]
    assert generation["model"] == "test-model"
    assert generation["input"] == [["human", "safe prompt"]]
    assert generation["output"] == "safe response"
    assert generation["usage_details"] == {
        "input_tokens": 2,
        "output_tokens": 2,
    }
    assert generation["model_parameters"]["temperature"] == 0
    local_generation = next(
        event
        for event in tracer.events
        if event.observation_type == "GENERATION"
    )
    assert local_generation.model == "test-model"
    assert local_generation.model_parameters == {"temperature": 0}
    assert local_generation.usage == {
        "input_tokens": 2,
        "output_tokens": 2,
    }


def test_resume_after_interrupt_can_remain_nested_under_human_review() -> None:
    """An ended review parent can link work resumed in a later graph invocation."""

    client = FakeLangfuseClient()
    tracer = TraceManager(client=client, enabled=True)
    tracer.start_run("run-resume-parent", "thread-resume-parent")
    with tracer.span("human_review") as review_id:
        with tracer.span("human_review_pause"):
            pass

    with tracer.span(
        "memory_write",
        parent_observation_id=review_id,
        input={"review_comments": {"J1": "I know GraphQL."}},
    ):
        pass
    with tracer.span(
        "tailor_resume_job_1",
        parent_observation_id=review_id,
    ) as tool_id:
        with tracer.span("resume_tailoring.compile_pdf"):
            pass

    memory = next(call for call in client.span_calls if call["name"] == "memory_write")
    tool = next(
        call for call in client.span_calls if call["name"] == "tailor_resume_job_1"
    )
    compile_span = next(
        call
        for call in client.span_calls
        if call["name"] == "resume_tailoring.compile_pdf"
    )
    assert memory["parent_observation_id"] == review_id
    assert tool["parent_observation_id"] == review_id
    assert compile_span["parent_observation_id"] == tool_id


def test_span_errors_are_captured_without_losing_parentage() -> None:
    """Failed work records both an ERROR span and a nested error event."""

    client = FakeLangfuseClient()
    tracer = TraceManager(client=client, enabled=True)
    tracer.start_run("run-error", "thread-error")

    with pytest.raises(ValueError, match="candidate@example.com"):
        with tracer.span("failing_stage", input={"operation": "validate"}):
            raise ValueError("candidate@example.com could not be validated")

    failed_span = client.span_calls[0]
    error_event = client.event_calls[0]
    assert failed_span["level"] == "ERROR"
    assert failed_span["status_message"] == "ValueError"
    assert "[REDACTED_EMAIL]" in failed_span["output"]["error"]
    assert error_event["parent_observation_id"] == failed_span["id"]
    assert error_event["metadata"]["error_type"] == "ValueError"


def test_disabled_tracing_is_a_noop_for_execution() -> None:
    """The same tracing API remains safe with no client or credentials."""

    tracer = TraceManager(client=None, enabled=False)

    trace_id = tracer.start_run("run-disabled", "thread-disabled")
    with tracer.span("work", input={"value": 1}) as span_id:
        tracer.record_generation(
            name="offline_generation",
            model="offline",
            messages=[("human", "hello")],
            response="world",
        )
        tracer.update_span(span_id, output={"value": 2})
    tracer.update_run(output={"status": "complete"})
    tracer.flush()

    assert trace_id == "trace-run-disabled"
    assert tracer.enabled is False
    assert [event.name for event in tracer.events] == [
        ROOT_TRACE_NAME,
        "offline_generation",
        "work",
    ]


def test_trace_payloads_redact_secrets_and_registered_personal_data() -> None:
    """Remote payloads mask direct identifiers without hiding token counts."""

    client = FakeLangfuseClient()
    tracer = TraceManager(client=client, enabled=True)
    tracer.start_run("run-private", "thread-private")
    tracer.register_personal_data(
        "Avery Morgan",
        "Houston, TX",
        "github.com/avery",
    )
    tracer.record_generation(
        name="private_generation",
        model="test-model",
        messages=[
            (
                "human",
                "Avery Morgan in Houston, TX uses avery@example.com, "
                "713-555-0148, and github.com/avery.",
            )
        ],
        response="Contact Avery Morgan",
        usage={"input_tokens": 12, "output_tokens": 3},
        metadata={"api_key": "sk-lf-secret-value"},
    )

    generation = client.generation_calls[0]
    serialized = repr(generation)
    assert "Avery Morgan" not in serialized
    assert "Houston, TX" not in serialized
    assert "avery@example.com" not in serialized
    assert "713-555-0148" not in serialized
    assert "github.com/avery" not in serialized
    assert "sk-lf-secret-value" not in serialized
    assert generation["usage_details"] == {
        "input_tokens": 12,
        "output_tokens": 3,
    }


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


def test_new_process_continues_checkpointed_remote_root_trace() -> None:
    client = FakeLangfuseClient()
    first = TraceManager(client=client, enabled=True)
    trace_id = first.start_run("run-restart", "thread-restart")
    with first.span("human_review") as review_id:
        pass

    resumed = TraceManager(client=client, enabled=True)
    resumed.continue_run(
        run_id="run-restart",
        session_id="thread-restart",
        trace_id=trace_id,
        trace_url="https://example.test/trace",
    )
    with resumed.span(
        "memory_write",
        parent_observation_id=review_id,
        input={"provenance": "human_review"},
    ):
        pass

    assert {call["id"] for call in client.trace_calls} == {trace_id}
    resumed_span = client.span_calls[-1]
    assert resumed_span["trace_id"] == trace_id
    assert resumed_span["parent_observation_id"] == review_id
    assert resumed.trace_url == "https://example.test/trace"


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

    with pytest.raises(RuntimeError) as exc_info:
        TraceManager()

    assert public_key not in caplog.text
    assert secret_key not in caplog.text
    assert public_key not in str(exc_info.value)
    assert secret_key not in str(exc_info.value)
