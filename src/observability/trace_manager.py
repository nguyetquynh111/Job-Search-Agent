"""Langfuse v2 tracing with explicit offline mode and privacy sanitization."""

from __future__ import annotations

import logging
import re
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
MAX_METADATA_DEPTH = 6
MAX_METADATA_ITEMS = 50
MAX_METADATA_STRING_LENGTH = 8000
SENSITIVE_KEY_PARTS = {
    "api_key",
    "authorization",
    "credential",
    "deepinfra_api",
    "password",
    "private_key",
    "public_key",
    "secret",
    "token",
}
_EMAIL_RE = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_PHONE_RE = re.compile(
    r"(?<!\d)(?:\+?1[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]\d{3}[\s.-]\d{4}(?!\d)"
)
_SSN_RE = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")
_SECRET_VALUE_RE = re.compile(r"(?<![A-Za-z0-9])(?:sk|pk)-(?:lf-)?[A-Za-z0-9_-]{8,}")


@dataclass
class TraceEvent:
    """Local observation retained even when remote Langfuse is unavailable."""

    name: str
    metadata: dict[str, Any] = field(default_factory=dict)
    status: str = "OK"
    duration_ms: float | None = None
    trace_id: str | None = None
    observation_id: str | None = None
    parent_observation_id: str | None = None
    observation_type: str = "SPAN"
    input: Any = None
    output: Any = None
    model: str | None = None
    model_parameters: dict[str, Any] = field(default_factory=dict)
    usage: dict[str, Any] = field(default_factory=dict)


@dataclass
class _ActiveSpan:
    """In-process span state until the observation is ended."""

    name: str
    metadata: dict[str, Any]
    input: Any
    output: Any
    parent_observation_id: str | None
    started_perf: float
    started_at: datetime


class TraceManager:
    """Facade around the repository-compatible Langfuse v2 client."""

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
        self._root_client: Any | None = None
        self._active_spans: dict[str, _ActiveSpan] = {}
        self._span_stack: list[str] = []
        self._personal_values: set[str] = set()

    @property
    def current_observation_id(self) -> str | None:
        """Return the active parent observation, if any."""

        return self._span_stack[-1] if self._span_stack else None

    def register_personal_data(self, *values: str | None) -> None:
        """Register exact candidate values that must be masked in remote payloads."""

        for value in values:
            normalized = " ".join(str(value or "").split())
            if len(normalized) >= 3:
                self._personal_values.add(normalized)

    def sanitize(self, value: Any) -> Any:
        """Return a bounded, secret- and personal-data-safe trace value."""

        return safe_metadata(value, personal_values=self._personal_values)

    def start_run(
        self,
        run_id: str,
        session_id: str,
        metadata: dict[str, Any] | None = None,
        *,
        input: Any = None,
    ) -> str:
        """Create exactly one root trace for a submitted workflow run."""

        if self.run_id == run_id and self.trace_id and self._root_recorded:
            return self.trace_id
        if self.run_id != run_id:
            self._active_spans.clear()
            self._span_stack.clear()
            self._personal_values.clear()
            self._root_recorded = False
            self._root_client = None
            self.trace_url = None

        self.run_id = run_id
        self.session_id = session_id
        self.trace_id = f"trace-{run_id}"
        payload = self.sanitize(
            {
                "run_id": run_id,
                "session_id": session_id,
                "sdk_version": self.sdk_version,
                **(metadata or {}),
            }
        )
        safe_input = self.sanitize(input)
        if self.enabled and self.client is not None:
            try:
                self._root_client = self.client.trace(
                    id=self.trace_id,
                    name=ROOT_TRACE_NAME,
                    session_id=session_id,
                    input=safe_input,
                    metadata=payload,
                    public=True,
                )
                if hasattr(self._root_client, "get_trace_url"):
                    self.trace_url = self._root_client.get_trace_url()
                else:
                    self.trace_url = getattr(self._root_client, "url", None)
            except Exception:
                self._disable_remote_tracing(
                    "Langfuse root trace failed; using no-op tracing"
                )
        self.events.append(
            TraceEvent(
                name=ROOT_TRACE_NAME,
                metadata=payload,
                trace_id=self.trace_id,
                observation_type="TRACE",
                input=safe_input,
            )
        )
        self._root_recorded = True
        return self.trace_id

    def start_root_trace(
        self, run_id: str, thread_id: str, metadata: dict[str, Any] | None = None
    ) -> str:
        """Backward-compatible alias for starting the workflow trace."""

        return self.start_run(run_id=run_id, session_id=thread_id, metadata=metadata)

    def continue_run(
        self,
        *,
        run_id: str,
        session_id: str,
        trace_id: str,
        trace_url: str | None = None,
    ) -> None:
        """Reattach a new process to an existing checkpointed root trace."""

        if (
            self.run_id == run_id
            and self.trace_id == trace_id
            and self._root_recorded
        ):
            if trace_url:
                self.trace_url = trace_url
            return
        self.run_id = run_id
        self.session_id = session_id
        self.trace_id = trace_id
        self.trace_url = trace_url
        self._root_recorded = True
        self._active_spans.clear()
        self._span_stack.clear()
        if self.enabled and self.client is not None:
            try:
                # Langfuse v2 upserts by ID, so resumed observations remain on
                # the original remote root rather than creating another trace.
                self._root_client = self.client.trace(
                    id=trace_id,
                    name=ROOT_TRACE_NAME,
                    session_id=session_id,
                    metadata={"run_id": run_id, "resumed_from_checkpoint": True},
                    public=True,
                )
                if not self.trace_url and hasattr(
                    self._root_client, "get_trace_url"
                ):
                    self.trace_url = self._root_client.get_trace_url()
            except Exception:
                self._disable_remote_tracing(
                    "Langfuse trace continuation failed; using no-op tracing"
                )

    def update_run(
        self,
        *,
        metadata: dict[str, Any] | None = None,
        output: Any = None,
    ) -> None:
        """Update root-trace status/output without creating another trace."""

        safe_meta = self.sanitize(metadata or {})
        safe_output = self.sanitize(output)
        root_event = next(
            (event for event in self.events if event.observation_type == "TRACE"),
            None,
        )
        if root_event is not None:
            root_event.metadata.update(safe_meta)
            if output is not None:
                root_event.output = safe_output
            if safe_meta.get("status"):
                root_event.status = str(safe_meta["status"])
        if self.enabled and self._root_client is not None:
            try:
                self._root_client.update(metadata=safe_meta, output=safe_output)
            except Exception:
                self._disable_remote_tracing(
                    "Langfuse root update failed; using no-op tracing"
                )

    def start_span(
        self,
        name: str,
        metadata: dict[str, Any] | None = None,
        *,
        input: Any = None,
        parent_observation_id: str | None = None,
    ) -> str:
        """Start a child span under the active observation or root trace."""

        span_id = uuid4().hex
        span_metadata = metadata if metadata is not None else {}
        span_metadata.setdefault("run_id", self.run_id)
        span_metadata.setdefault("session_id", self.session_id)
        span_metadata.setdefault("trace_id", self.trace_id)
        parent_id = (
            parent_observation_id
            if parent_observation_id is not None
            else self.current_observation_id
        )
        self._active_spans[span_id] = _ActiveSpan(
            name=name,
            metadata=span_metadata,
            input=input,
            output=None,
            parent_observation_id=parent_id,
            started_perf=time.perf_counter(),
            started_at=datetime.now(UTC),
        )
        self._span_stack.append(span_id)
        return span_id

    def update_span(
        self,
        span_id: str,
        *,
        metadata: dict[str, Any] | None = None,
        output: Any = None,
    ) -> None:
        """Attach output or metadata before a span ends."""

        active = self._active_spans.get(span_id)
        if active is None:
            return
        if metadata:
            active.metadata.update(metadata)
        if output is not None:
            active.output = output

    def end_span(
        self,
        span_id: str,
        metadata: dict[str, Any] | None = None,
        status: str = "OK",
        error_type: str | None = None,
        *,
        output: Any = None,
    ) -> None:
        """End a span locally and send it with its correct parent to Langfuse."""

        active = self._active_spans.pop(span_id, None)
        self._remove_from_stack(span_id)
        if active is None:
            return
        ended_at = datetime.now(UTC)
        duration_ms = (time.perf_counter() - active.started_perf) * 1000
        event_metadata = self.sanitize({**active.metadata, **(metadata or {})})
        event_metadata["status"] = status
        if error_type:
            event_metadata["error_type"] = error_type
        event_metadata["duration_ms"] = round(duration_ms, 3)
        safe_input = self.sanitize(active.input)
        safe_output = self.sanitize(output if output is not None else active.output)
        event = TraceEvent(
            name=active.name,
            metadata=event_metadata,
            status=status,
            duration_ms=duration_ms,
            trace_id=self.trace_id,
            observation_id=span_id,
            parent_observation_id=active.parent_observation_id,
            observation_type="SPAN",
            input=safe_input,
            output=safe_output,
        )
        self.events.append(event)
        if self.enabled and self.client is not None and self.trace_id:
            try:
                self.client.span(
                    id=span_id,
                    trace_id=self.trace_id,
                    parent_observation_id=active.parent_observation_id,
                    name=active.name,
                    start_time=active.started_at,
                    end_time=ended_at,
                    input=safe_input,
                    output=safe_output,
                    metadata=event_metadata,
                    level="ERROR" if status == "ERROR" else "DEFAULT",
                    status_message=error_type,
                )
            except Exception:
                self._disable_remote_tracing(
                    "Langfuse span failed; using no-op tracing"
                )

    @contextmanager
    def span(
        self,
        name: str,
        metadata: dict[str, Any] | None = None,
        *,
        input: Any = None,
        parent_observation_id: str | None = None,
    ) -> Iterator[str]:
        """Record a nested span while preserving no-op behavior."""

        span_id = self.start_span(
            name,
            metadata,
            input=input,
            parent_observation_id=parent_observation_id,
        )
        try:
            yield span_id
        except Exception as exc:
            # Record the event before closing its failed parent span.
            self.record_error(exc, {"span_name": name})
            self.end_span(
                span_id,
                status="ERROR",
                error_type=exc.__class__.__name__,
                output={"error": str(exc)},
            )
            raise
        else:
            self.end_span(span_id)

    def record_generation(
        self,
        metadata: dict[str, Any] | None = None,
        *,
        name: str = "agent_controller",
        model: str | None = None,
        messages: Any = None,
        response: Any = None,
        usage: dict[str, Any] | None = None,
        model_parameters: dict[str, Any] | None = None,
        status: str = "OK",
        error_type: str | None = None,
        parent_observation_id: str | None = None,
    ) -> None:
        """Record one actual LLM call with prompt, response, usage, and config."""

        observation_id = uuid4().hex
        parent_id = (
            parent_observation_id
            if parent_observation_id is not None
            else self.current_observation_id
        )
        payload = self.sanitize(
            {
                "run_id": self.run_id,
                "session_id": self.session_id,
                "trace_id": self.trace_id,
                **(metadata or {}),
            }
        )
        if error_type:
            payload["error_type"] = error_type
        payload["status"] = status
        safe_messages = self.sanitize(messages)
        safe_response = self.sanitize(response)
        safe_usage = self.sanitize(usage or {})
        safe_parameters = self.sanitize(model_parameters or {})
        self.events.append(
            TraceEvent(
                name=name,
                metadata=payload,
                status=status,
                trace_id=self.trace_id,
                observation_id=observation_id,
                parent_observation_id=parent_id,
                observation_type="GENERATION",
                input=safe_messages,
                output=safe_response,
                model=model,
                model_parameters=safe_parameters,
                usage=safe_usage,
            )
        )
        if self.enabled and self.client is not None and self.trace_id:
            try:
                self.client.generation(
                    id=observation_id,
                    trace_id=self.trace_id,
                    parent_observation_id=parent_id,
                    name=name,
                    start_time=datetime.now(UTC),
                    end_time=datetime.now(UTC),
                    input=safe_messages,
                    output=safe_response,
                    metadata=payload,
                    model=model,
                    model_parameters=safe_parameters,
                    usage_details=safe_usage,
                    level="ERROR" if status == "ERROR" else "DEFAULT",
                    status_message=error_type,
                )
            except Exception:
                self._disable_remote_tracing(
                    "Langfuse generation failed; using no-op tracing"
                )

    def record_error(
        self, error: BaseException | str, metadata: dict[str, Any] | None = None
    ) -> None:
        """Record a sanitized error event under the active observation."""

        error_type = error if isinstance(error, str) else error.__class__.__name__
        error_message = str(error)
        observation_id = uuid4().hex
        parent_id = self.current_observation_id
        payload = self.sanitize(
            {
                "run_id": self.run_id,
                "session_id": self.session_id,
                "trace_id": self.trace_id,
                "error_type": str(error_type),
                "error_message": error_message,
                **(metadata or {}),
            }
        )
        event = TraceEvent(
            name="workflow_error",
            metadata=payload,
            status="ERROR",
            trace_id=self.trace_id,
            observation_id=observation_id,
            parent_observation_id=parent_id,
            observation_type="EVENT",
            output={
                "error_type": str(error_type),
                "error_message": self.sanitize(error_message),
            },
        )
        self.events.append(event)
        if self.enabled and self.client is not None and self.trace_id:
            try:
                self.client.event(
                    id=observation_id,
                    trace_id=self.trace_id,
                    parent_observation_id=parent_id,
                    name="workflow_error",
                    start_time=datetime.now(UTC),
                    metadata=payload,
                    output=event.output,
                    level="ERROR",
                    status_message=str(error_type),
                )
            except Exception:
                self._disable_remote_tracing(
                    "Langfuse error event failed; using no-op tracing"
                )

    def flush(self) -> None:
        """Flush pending Langfuse events when remote tracing is active."""

        if self.enabled and self.client is not None and hasattr(self.client, "flush"):
            try:
                self.client.flush()
            except Exception:
                self._disable_remote_tracing(
                    "Langfuse flush failed; using no-op tracing"
                )

    def _remove_from_stack(self, observation_id: str) -> None:
        if self._span_stack and self._span_stack[-1] == observation_id:
            self._span_stack.pop()
            return
        if observation_id in self._span_stack:
            self._span_stack.remove(observation_id)

    def _disable_remote_tracing(self, message: str) -> None:
        self.enabled = False
        self.mode = "unavailable"
        self.status_message = STATUS_UNAVAILABLE
        logger.warning(message)


def safe_metadata(
    value: Any,
    depth: int = 0,
    *,
    personal_values: set[str] | None = None,
) -> Any:
    """Bound and sanitize arbitrary trace data before it leaves the process."""

    if depth > MAX_METADATA_DEPTH:
        return "<truncated>"
    personal_values = personal_values or set()
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
                safe[key_text] = safe_metadata(
                    item,
                    depth + 1,
                    personal_values=personal_values,
                )
        return safe
    if isinstance(value, (list, tuple, set)):
        values = list(value)
        safe_items = [
            safe_metadata(
                item,
                depth + 1,
                personal_values=personal_values,
            )
            for item in values[:MAX_METADATA_ITEMS]
        ]
        if len(values) > MAX_METADATA_ITEMS:
            safe_items.append({"remaining_items": len(values) - MAX_METADATA_ITEMS})
        return safe_items
    if isinstance(value, str):
        sanitized = _sanitize_string(value, personal_values)
        if len(sanitized) > MAX_METADATA_STRING_LENGTH:
            return f"{sanitized[:MAX_METADATA_STRING_LENGTH]}..."
        return sanitized
    if isinstance(value, int | float | bool) or value is None:
        return value
    if hasattr(value, "model_dump"):
        return safe_metadata(
            value.model_dump(),
            depth + 1,
            personal_values=personal_values,
        )
    return _sanitize_string(str(value), personal_values)


def _sanitize_string(value: str, personal_values: set[str]) -> str:
    sanitized = _EMAIL_RE.sub("[REDACTED_EMAIL]", value)
    sanitized = _PHONE_RE.sub("[REDACTED_PHONE]", sanitized)
    sanitized = _SSN_RE.sub("[REDACTED_SSN]", sanitized)
    sanitized = _SECRET_VALUE_RE.sub("[REDACTED_SECRET]", sanitized)
    for personal in sorted(personal_values, key=len, reverse=True):
        sanitized = re.sub(
            re.escape(personal),
            "[REDACTED_PERSON]",
            sanitized,
            flags=re.IGNORECASE,
        )
    return sanitized


def _is_sensitive_key(key: str) -> bool:
    lowered = key.lower()
    if lowered in {
        "completion_tokens",
        "input_tokens",
        "output_tokens",
        "prompt_tokens",
        "total_tokens",
    }:
        return False
    return any(part in lowered for part in SENSITIVE_KEY_PARTS)
