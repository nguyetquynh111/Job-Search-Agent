"""Trace abstraction used by graph nodes and tools."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Iterator
from uuid import uuid4

from src.observability.langfuse_client import (
    STATUS_CONNECTED,
    STATUS_NOOP,
    STATUS_UNAVAILABLE,
    create_langfuse_client,
)

logger = logging.getLogger(__name__)

ROOT_TRACE_NAME = "job_search_agent_run"
MAX_METADATA_DEPTH = 4
MAX_METADATA_ITEMS = 20
MAX_METADATA_STRING_LENGTH = 240
SENSITIVE_KEY_PARTS = {
    "api_key",
    "authorization",
    "credential",
    "deepinfra",
    "key",
    "password",
    "secret",
    "token",
}


@dataclass
class TraceEvent:
    """Local trace event recorded even when Langfuse is disabled."""

    name: str
    metadata: dict[str, Any] = field(default_factory=dict)
    status: str = "OK"
    duration_ms: float | None = None
    trace_id: str | None = None


@dataclass
class _ActiveSpan:
    """In-process span state until a span is ended."""

    name: str
    metadata: dict[str, Any]
    started_perf: float
    started_at: datetime


class TraceManager:
    """Facade around Langfuse v2 with a no-op/local fallback mode."""

    def __init__(self, client: Any | None = None, enabled: bool | None = None) -> None:
        if enabled is None:
            resolved_client, status = create_langfuse_client()
            self.client = client if client is not None else resolved_client
            self.enabled = bool(self.client and status.enabled)
            self.status_message = status.message
            self.mode = status.mode
            self.sdk_version = status.sdk_version
        else:
            self.client = client
            self.enabled = bool(enabled and client is not None)
            self.status_message = STATUS_CONNECTED if self.enabled else STATUS_NOOP
            self.mode = "langfuse" if self.enabled else "noop"
            self.sdk_version = None
        self.run_id: str | None = None
        self.session_id: str | None = None
        self.trace_id: str | None = None
        self.trace_url: str | None = None
        self.events: list[TraceEvent] = []
        self._root_recorded = False
        self._active_spans: dict[str, _ActiveSpan] = {}

    def start_run(
        self,
        run_id: str,
        session_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Start or reuse the root trace for one submitted workflow."""

        if self.run_id == run_id and self.trace_id and self._root_recorded:
            return self.trace_id
        if self.run_id != run_id:
            self._active_spans.clear()
            self._root_recorded = False
            self.trace_url = None

        self.run_id = run_id
        self.session_id = session_id
        self.trace_id = f"trace-{run_id}"
        payload = safe_metadata(
            {
                "run_id": run_id,
                "session_id": session_id,
                **(metadata or {}),
            }
        )
        if self.enabled and self.client is not None:
            try:
                trace = self.client.trace(
                    id=self.trace_id,
                    name=ROOT_TRACE_NAME,
                    session_id=session_id,
                    metadata=payload,
                )
                self.trace_url = getattr(trace, "url", None)
            except Exception:
                self._disable_remote_tracing(
                    "Langfuse root trace failed; using no-op tracing"
                )
        self.events.append(
            TraceEvent(name=ROOT_TRACE_NAME, metadata=payload, trace_id=self.trace_id)
        )
        self._root_recorded = True
        return self.trace_id

    def start_root_trace(
        self, run_id: str, thread_id: str, metadata: dict[str, Any] | None = None
    ) -> str:
        """Backward-compatible alias for starting the workflow trace."""

        return self.start_run(run_id=run_id, session_id=thread_id, metadata=metadata)

    def start_span(self, name: str, metadata: dict[str, Any] | None = None) -> str:
        """Start a local span and return its observation ID."""

        span_id = uuid4().hex
        span_metadata = metadata if metadata is not None else {}
        span_metadata.setdefault("run_id", self.run_id)
        span_metadata.setdefault("session_id", self.session_id)
        span_metadata.setdefault("trace_id", self.trace_id)
        self._active_spans[span_id] = _ActiveSpan(
            name=name,
            metadata=span_metadata,
            started_perf=time.perf_counter(),
            started_at=datetime.now(UTC),
        )
        return span_id

    def end_span(
        self,
        span_id: str,
        metadata: dict[str, Any] | None = None,
        status: str = "OK",
        error_type: str | None = None,
    ) -> None:
        """End a span locally and send it to Langfuse when available."""

        active = self._active_spans.pop(span_id, None)
        if active is None:
            return
        ended_at = datetime.now(UTC)
        duration_ms = (time.perf_counter() - active.started_perf) * 1000
        event_metadata = safe_metadata({**active.metadata, **(metadata or {})})
        if error_type:
            event_metadata["error_type"] = error_type
        event_metadata["duration_ms"] = round(duration_ms, 3)
        event = TraceEvent(
            name=active.name,
            metadata=event_metadata,
            status=status,
            duration_ms=duration_ms,
            trace_id=self.trace_id,
        )
        self.events.append(event)
        if self.enabled and self.client is not None and self.trace_id:
            try:
                self.client.span(
                    id=span_id,
                    trace_id=self.trace_id,
                    name=active.name,
                    start_time=active.started_at,
                    end_time=ended_at,
                    metadata=event_metadata,
                    level="ERROR" if status == "ERROR" else "DEFAULT",
                    status_message=error_type,
                )
            except Exception:
                self._disable_remote_tracing("Langfuse span failed; using no-op tracing")

    @contextmanager
    def span(self, name: str, metadata: dict[str, Any] | None = None) -> Iterator[None]:
        """Record a nested observation duration and error status safely."""

        span_id = self.start_span(name, metadata)
        try:
            yield
        except Exception as exc:
            self.end_span(span_id, status="ERROR", error_type=exc.__class__.__name__)
            self.record_error(exc, {"span_name": name})
            raise
        else:
            self.end_span(span_id)

    def record_generation(self, metadata: dict[str, Any]) -> None:
        """Record an LLM/controller generation summary without prompts or outputs."""

        payload = safe_metadata(
            {
                "run_id": self.run_id,
                "session_id": self.session_id,
                "trace_id": self.trace_id,
                **metadata,
            }
        )
        self.events.append(
            TraceEvent(name="agent_controller", metadata=payload, trace_id=self.trace_id)
        )
        if self.enabled and self.client is not None and self.trace_id:
            try:
                model_name = payload.get("model_name")
                self.client.generation(
                    trace_id=self.trace_id,
                    name="agent_controller",
                    start_time=datetime.now(UTC),
                    end_time=datetime.now(UTC),
                    metadata=payload,
                    model=model_name if isinstance(model_name, str) else None,
                )
            except Exception:
                self._disable_remote_tracing(
                    "Langfuse generation failed; using no-op tracing"
                )

    def record_error(
        self, error: BaseException | str, metadata: dict[str, Any] | None = None
    ) -> None:
        """Record a safe workflow error event without raising observability errors."""

        error_type = error if isinstance(error, str) else error.__class__.__name__
        payload = safe_metadata(
            {
                "run_id": self.run_id,
                "session_id": self.session_id,
                "trace_id": self.trace_id,
                "error_type": str(error_type),
                **(metadata or {}),
            }
        )
        self.events.append(
            TraceEvent(
                name="workflow_error",
                metadata=payload,
                status="ERROR",
                trace_id=self.trace_id,
            )
        )
        if self.enabled and self.client is not None and self.trace_id:
            try:
                self.client.event(
                    trace_id=self.trace_id,
                    name="workflow_error",
                    start_time=datetime.now(UTC),
                    metadata=payload,
                    level="ERROR",
                    status_message=str(error_type),
                )
            except Exception:
                self._disable_remote_tracing(
                    "Langfuse error event failed; using no-op tracing"
                )

    def flush(self) -> None:
        """Flush pending Langfuse events if the client supports it."""

        if self.enabled and self.client is not None and hasattr(self.client, "flush"):
            try:
                self.client.flush()
            except Exception:
                self._disable_remote_tracing("Langfuse flush failed; using no-op tracing")

    def _disable_remote_tracing(self, message: str) -> None:
        self.enabled = False
        self.mode = "unavailable"
        self.status_message = STATUS_UNAVAILABLE
        logger.warning(message)


def safe_metadata(value: Any, depth: int = 0) -> Any:
    """Return metadata that is bounded and strips sensitive key/value pairs."""

    if depth > MAX_METADATA_DEPTH:
        return "<truncated>"
    if isinstance(value, dict):
        safe: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= MAX_METADATA_ITEMS:
                safe["remaining_items"] = len(value) - MAX_METADATA_ITEMS
                break
            key_text = str(key)
            if _is_sensitive_key(key_text):
                safe[key_text] = "<redacted>"
            else:
                safe[key_text] = safe_metadata(item, depth + 1)
        return safe
    if isinstance(value, list):
        safe_items = [
            safe_metadata(item, depth + 1) for item in value[:MAX_METADATA_ITEMS]
        ]
        if len(value) > MAX_METADATA_ITEMS:
            safe_items.append({"remaining_items": len(value) - MAX_METADATA_ITEMS})
        return safe_items
    if isinstance(value, tuple):
        return safe_metadata(list(value), depth)
    if isinstance(value, str):
        if len(value) > MAX_METADATA_STRING_LENGTH:
            return f"{value[:MAX_METADATA_STRING_LENGTH]}..."
        return value
    if isinstance(value, int | float | bool) or value is None:
        return value
    return str(value)


def _is_sensitive_key(key: str) -> bool:
    lowered = key.lower()
    return any(part in lowered for part in SENSITIVE_KEY_PARTS)
