"""Langfuse tracing client, spans, and decorators."""

from __future__ import annotations

import importlib.metadata
import json
import logging
import os
import re
import time
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import wraps
from typing import Any, Iterator, TypeVar
from urllib.parse import urlparse
from uuid import uuid4

from packaging.version import InvalidVersion, Version

logger = logging.getLogger(__name__)

DEFAULT_LANGFUSE_HOST = "https://us.cloud.langfuse.com"
MINIMUM_LANGFUSE_VERSION = Version("4.7.0")
STATUS_CONNECTED = "Observability: Langfuse connected"
STATUS_UNAVAILABLE = "Observability: Langfuse unavailable"
STATUS_NOOP = "Observability: Local no-op tracing"


@dataclass(frozen=True)
class LangfuseConfig:
    """Langfuse configuration sourced from environment variables."""

    public_key: str
    secret_key: str
    host: str


@dataclass(frozen=True)
class LangfuseClientStatus:
    """Langfuse client availability status."""

    enabled: bool
    message: str
    mode: str
    sdk_version: str | None = None


def load_langfuse_config() -> LangfuseConfig:
    """Load Langfuse configuration from environment variables."""

    public_key = (os.getenv("LANGFUSE_PUBLIC_KEY") or "").strip()
    secret_key = (os.getenv("LANGFUSE_SECRET_KEY") or "").strip()
    host = (
        os.getenv("LANGFUSE_HOST")
        or os.getenv("LANGFUSE_BASE_URL")
        or DEFAULT_LANGFUSE_HOST
    ).strip()
    missing = [
        name
        for name, value in (
            ("LANGFUSE_PUBLIC_KEY", public_key),
            ("LANGFUSE_SECRET_KEY", secret_key),
        )
        if not value
    ]
    if missing:
        raise RuntimeError("Langfuse credentials are required: " + ", ".join(missing))
    if not _valid_host(host):
        raise ValueError("Invalid LANGFUSE_HOST")
    return LangfuseConfig(public_key=public_key, secret_key=secret_key, host=host)


def create_langfuse_client() -> tuple[Any | None, LangfuseClientStatus]:
    """Create and authenticate the required Langfuse v4 client."""

    config = load_langfuse_config()
    try:
        from langfuse import Langfuse

        sdk_version = _langfuse_version()
        if sdk_version is None or Version(sdk_version) < MINIMUM_LANGFUSE_VERSION:
            raise RuntimeError("Langfuse Python SDK 4.7.0 or newer is required")
        client = Langfuse(
            public_key=config.public_key,
            secret_key=config.secret_key,
            host=config.host,
        )
        if hasattr(client, "auth_check") and not client.auth_check():
            raise RuntimeError("Langfuse authentication failed")
        logger.info(
            "Langfuse startup diagnostics: host=%s sdk_version=%s api=observations-v4",
            config.host,
            sdk_version,
        )
        return client, LangfuseClientStatus(
            enabled=True,
            message=STATUS_CONNECTED,
            mode="langfuse",
            sdk_version=sdk_version,
        )
    except (Exception, InvalidVersion):
        logger.error(
            "Langfuse initialization failed (host=%s sdk_version=%s)",
            config.host,
            _langfuse_version() or "unavailable",
        )
        raise RuntimeError(
            "Langfuse initialization failed; connected tracing is required"
        ) from None


def _valid_host(host: str) -> bool:
    parsed = urlparse(host)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _trace_url_from_client(client: Any, trace_id: str) -> str | None:
    """Return the v4 SDK's canonical project-scoped URL for a confirmed trace."""

    get_trace_url = getattr(client, "get_trace_url", None)
    if not callable(get_trace_url):
        return None
    try:
        value = get_trace_url(trace_id=trace_id)
    except Exception:
        return None
    return str(value) if _valid_trace_url(value, trace_id) else None


def _valid_trace_url(value: Any, trace_id: str) -> bool:
    """Accept only canonical project-scoped URLs for the requested trace."""

    if not isinstance(value, str):
        return False
    parsed = urlparse(value)
    parts = [part for part in parsed.path.split("/") if part]
    return (
        parsed.scheme in {"http", "https"}
        and bool(parsed.netloc)
        and len(parts) >= 4
        and parts[-4] == "project"
        and parts[-2] == "traces"
        and parts[-1] == trace_id
    )


def _langfuse_version() -> str | None:
    try:
        return importlib.metadata.version("langfuse")
    except importlib.metadata.PackageNotFoundError:
        return None


logger = logging.getLogger(__name__)

ROOT_TRACE_NAME = "Job Search Agent Run"
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


def _remote_model_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    """Return the flat model-parameter shape accepted by Langfuse v4."""

    normalized: dict[str, Any] = {}
    for key, value in parameters.items():
        if isinstance(value, dict):
            normalized[key] = json.dumps(
                value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            )
        else:
            normalized[key] = value
    return normalized


def _system_prompt_from_messages(messages: Any) -> Any:
    """Return the system message content without changing the message payload."""

    if not isinstance(messages, list):
        return None
    prompts: list[Any] = []
    for message in messages:
        if isinstance(message, (list, tuple)) and len(message) >= 2:
            if str(message[0]).casefold() == "system":
                prompts.append(message[1])
        elif isinstance(message, dict):
            role = message.get("role") or message.get("type")
            if str(role).casefold() == "system":
                prompts.append(message.get("content"))
    if not prompts:
        return None
    return prompts[0] if len(prompts) == 1 else prompts


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
    remote_observation: Any | None = None


class TraceManager:
    """Langfuse v4 observation facade with local, sanitized event retention."""

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
            self.sdk_version = _langfuse_version()
        self.host = (
            os.getenv("LANGFUSE_HOST")
            or os.getenv("LANGFUSE_BASE_URL")
            or DEFAULT_LANGFUSE_HOST
        ).strip()
        self.run_id: str | None = None
        self.session_id: str | None = None
        self.trace_id: str | None = None
        self.trace_url: str | None = None
        self.trace_public = False
        self.trace_ingest_confirmed = False
        self.observation_count = 0
        self.observation_names: list[str] = []
        self.trace_export_error: str | None = None
        self.trace_debug_status = "not started"
        self._last_trace_url_check = 0.0
        self.events: list[TraceEvent] = []
        self._root_recorded = False
        self._root_client: Any | None = None
        self._root_observation_id: str | None = None
        self._root_ended = False
        self._active_spans: dict[str, _ActiveSpan] = {}
        # Each worker needs its own nesting stack when independent job tools run
        # concurrently. Active spans remain shared because their UUID keys are
        # unique, while ContextVar prevents one worker becoming another's parent.
        self._span_stack: ContextVar[tuple[str, ...]] = ContextVar(
            f"trace_span_stack_{id(self)}",
            default=(),
        )
        self._personal_values: set[str] = set()

    @property
    def current_observation_id(self) -> str | None:
        """Return the active parent observation, if any."""

        stack = self._span_stack.get()
        return stack[-1] if stack else None

    @property
    def _uses_observation_api(self) -> bool:
        """Return whether the connected client uses Langfuse's v4 OTEL API."""

        return bool(
            self.client is not None
            and callable(getattr(self.client, "start_as_current_observation", None))
            and callable(getattr(self.client, "start_observation", None))
            and getattr(getattr(self.client, "api", None), "observations", None)
            is not None
        )

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
            self._span_stack.set(())
            self._personal_values.clear()
            self._root_recorded = False
            self._root_client = None
            self._root_observation_id = None
            self._root_ended = False
            self.trace_url = None
            self.trace_public = False
            self.trace_ingest_confirmed = False
            self.observation_count = 0
            self.observation_names = []
            self.trace_export_error = None
            self.trace_debug_status = "root observation pending"
            self._last_trace_url_check = 0.0

        self.run_id = run_id
        self.session_id = session_id
        self.trace_id = None
        payload = self.sanitize(
            {
                "run_id": run_id,
                "session_id": session_id,
                "sdk_version": self.sdk_version,
                **(metadata or {}),
            }
        )
        safe_input = self.sanitize(input)
        if self.enabled and self.client is not None and not self._uses_observation_api:
            self._disable_remote_tracing(
                "Langfuse v4 observation API unavailable; using no-op tracing"
            )
        if self.enabled and self.client is not None:
            try:
                with self.client.start_as_current_observation(
                    name="agent-run",
                    as_type="span",
                    input=safe_input,
                    metadata=payload,
                    end_on_exit=False,
                ) as root_client:
                    self.trace_id = str(
                        getattr(root_client, "trace_id", None)
                        or self.client.get_current_trace_id()
                    )
                self._root_client = root_client
                self._root_observation_id = str(root_client.id)
                root_client.set_trace_as_public()
                self.trace_public = True
                self.trace_debug_status = "root observation open"
            except Exception as exc:
                self._record_export_error("root observation creation", exc)
                self._disable_remote_tracing(
                    "Langfuse root trace failed; using no-op tracing"
                )
        if not self.trace_id:
            # Local-only IDs are never presented as remotely ingested trace IDs.
            self.trace_id = uuid4().hex
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

        if self.run_id == run_id and self.trace_id == trace_id and self._root_recorded:
            if self.trace_ingest_confirmed and _valid_trace_url(trace_url, trace_id):
                self.trace_url = trace_url
            return
        self.run_id = run_id
        self.session_id = session_id
        self.trace_id = trace_id
        self.trace_url = None
        self.trace_public = False
        self.trace_ingest_confirmed = False
        self.observation_count = 0
        self.observation_names = []
        self.trace_export_error = None
        self.trace_debug_status = "resume observation pending"
        self._root_recorded = True
        self._active_spans.clear()
        self._span_stack.set(())
        self._root_observation_id = None
        self._root_ended = False
        if self.enabled and self.client is not None and not self._uses_observation_api:
            self._disable_remote_tracing(
                "Langfuse v4 observation API unavailable; using no-op tracing"
            )
        if self.enabled and self.client is not None:
            try:
                root_client = self.client.start_observation(
                    trace_context={"trace_id": trace_id},
                    name="Workflow Resume",
                    as_type="span",
                    metadata={
                        "run_id": run_id,
                        "resumed_from_checkpoint": True,
                    },
                )
                root_client.set_trace_as_public()
                self.trace_public = True
                self._root_observation_id = str(root_client.id)
                self._root_client = root_client
                self.trace_debug_status = "resume observation open"
            except Exception as exc:
                self._record_export_error("trace continuation", exc)
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
                self._root_client.update(
                    metadata=safe_meta,
                    output=safe_output,
                )
                self._root_client.set_trace_as_public()
                self.trace_public = True
            except Exception as exc:
                self._record_export_error("root observation update", exc)
                self._disable_remote_tracing(
                    "Langfuse root update failed; using no-op tracing"
                )
            finally:
                self.end_root_observation()

    def start_span(
        self,
        name: str,
        metadata: dict[str, Any] | None = None,
        *,
        input: Any = None,
        parent_observation_id: str | None = None,
    ) -> str:
        """Start a child span under the active observation or root trace."""

        span_metadata = metadata if metadata is not None else {}
        span_metadata.setdefault("run_id", self.run_id)
        span_metadata.setdefault("session_id", self.session_id)
        span_metadata.setdefault("trace_id", self.trace_id)
        parent_id = (
            parent_observation_id
            if parent_observation_id is not None
            else self.current_observation_id
        )
        remote_observation = None
        if self.enabled and self._uses_observation_api and self.trace_id:
            try:
                remote_observation = self.client.start_observation(
                    trace_context={
                        "trace_id": self.trace_id,
                        "parent_span_id": parent_id or self._root_observation_id,
                    },
                    name=name,
                    as_type="span",
                    input=self.sanitize(input),
                    metadata=self.sanitize(span_metadata),
                )
                span_id = remote_observation.id
            except Exception as exc:
                self._record_export_error(f"span start ({name})", exc)
                self._disable_remote_tracing(
                    "Langfuse span start failed; using no-op tracing"
                )
                span_id = uuid4().hex
        else:
            span_id = uuid4().hex
        self._active_spans[span_id] = _ActiveSpan(
            name=name,
            metadata=span_metadata,
            input=input,
            output=None,
            parent_observation_id=parent_id,
            started_perf=time.perf_counter(),
            started_at=datetime.now(UTC),
            remote_observation=remote_observation,
        )
        self._span_stack.set((*self._span_stack.get(), span_id))
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
        if (
            self.enabled
            and self._uses_observation_api
            and active.remote_observation is not None
        ):
            try:
                active.remote_observation.update(
                    output=safe_output,
                    metadata=event_metadata,
                    level="ERROR" if status == "ERROR" else "DEFAULT",
                    status_message=error_type,
                )
                active.remote_observation.end()
            except Exception as exc:
                self._record_export_error(f"span end ({active.name})", exc)
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
        start_time: datetime | None = None,
    ) -> None:
        """Record one actual LLM call with prompt, response, usage, and config."""

        name = {
            "orchestration.model_decision": "Workflow Decision LLM",
            "fit_analysis_llm": "Fit Analysis LLM",
        }.get(name, name)
        ended_at = datetime.now(UTC)
        started_at = start_time or ended_at
        duration_ms = max(
            0.0, (ended_at - started_at).total_seconds() * 1000
        )
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
        system_prompt = _system_prompt_from_messages(safe_messages)
        if system_prompt is not None:
            payload["system_prompt"] = system_prompt
        payload["messages"] = safe_messages
        payload["response"] = safe_response
        payload["model"] = model
        payload["parameters"] = safe_parameters
        payload["token_usage"] = safe_usage
        payload["duration_ms"] = round(duration_ms, 3)
        remote_parameters = _remote_model_parameters(safe_parameters)
        remote_generation = None
        if self.enabled and self._uses_observation_api and self.trace_id:
            try:
                remote_generation = self.client.start_observation(
                    trace_context={
                        "trace_id": self.trace_id,
                        "parent_span_id": parent_id or self._root_observation_id,
                    },
                    name=name,
                    as_type="generation",
                    input=safe_messages,
                    output=safe_response,
                    metadata=payload,
                    model=model,
                    model_parameters=remote_parameters,
                    usage_details=safe_usage,
                    level="ERROR" if status == "ERROR" else "DEFAULT",
                    status_message=error_type,
                    completion_start_time=started_at,
                )
                observation_id = remote_generation.id
            except Exception as exc:
                self._record_export_error(f"generation start ({name})", exc)
                self._disable_remote_tracing(
                    "Langfuse generation failed; using no-op tracing"
                )
        self.events.append(
            TraceEvent(
                name=name,
                metadata=payload,
                status=status,
                duration_ms=duration_ms,
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
        if remote_generation is not None:
            try:
                remote_generation.end()
            except Exception as exc:
                self._record_export_error(f"generation end ({name})", exc)
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
        if self.enabled and self._uses_observation_api and self.trace_id:
            try:
                remote_error = self.client.start_observation(
                    trace_context={
                        "trace_id": self.trace_id,
                        "parent_span_id": parent_id or self._root_observation_id,
                    },
                    name="workflow_error",
                    as_type="span",
                    input=None,
                    output=event.output,
                    metadata=payload,
                    level="ERROR",
                    status_message=str(error_type),
                )
                event.observation_id = remote_error.id
                remote_error.end()
            except Exception as exc:
                self._record_export_error("error observation", exc)
                self._disable_remote_tracing(
                    "Langfuse error event failed; using no-op tracing"
                )

    def end_root_observation(self) -> None:
        """End the root observation exactly once, including after update failures."""

        if self._root_client is None or self._root_ended:
            return
        try:
            self._root_client.end()
            self.trace_debug_status = "root observation ended"
        except Exception as exc:
            self._record_export_error("root observation end", exc)
            self._disable_remote_tracing(
                "Langfuse root end failed; using no-op tracing"
            )
        finally:
            self._root_ended = True

    def flush(self) -> None:
        """Flush pending OTEL observations, retaining sanitized exporter errors."""

        if self.client is not None and hasattr(self.client, "flush"):
            try:
                self.client.flush()
                if not self.trace_export_error:
                    self.trace_debug_status = "OTEL flush completed"
            except Exception as exc:
                self._record_export_error("OTEL flush", exc)
                self._disable_remote_tracing(
                    "Langfuse flush failed; using no-op tracing"
                )

    def confirm_ingestion(
        self,
        *,
        timeout_seconds: float = 20.0,
        poll_interval_seconds: float = 1.0,
    ) -> bool:
        """Confirm v4 ingestion by polling observations for the actual OTEL trace ID."""

        self.trace_ingest_confirmed = False
        self.observation_count = 0
        self.observation_names = []
        self.trace_url = None
        if self.client is None or not self._uses_observation_api or not self.trace_id:
            self.trace_export_error = (
                self.trace_export_error
                or "Langfuse v4 Observations API is unavailable."
            )
            self.trace_debug_status = "ingestion confirmation unavailable"
            return False

        deadline = time.monotonic() + max(0.0, timeout_seconds)
        last_exception: Exception | None = None
        while True:
            try:
                response = self.client.api.observations.get_many(
                    trace_id=self.trace_id,
                    limit=100,
                )
                observations = list(getattr(response, "data", None) or [])
                self.observation_count = len(observations)
                self.observation_names = [
                    str(getattr(item, "name", None) or item.get("name") or "")
                    if isinstance(item, dict)
                    else str(getattr(item, "name", "") or "")
                    for item in observations
                ]
                if observations:
                    self.trace_ingest_confirmed = True
                    self.trace_export_error = None
                    self.trace_debug_status = (
                        f"v4 ingestion confirmed ({len(observations)} observations)"
                    )
                    self.trace_url = _trace_url_from_client(
                        self.client,
                        self.trace_id,
                    )
                    return True
            except Exception as exc:
                last_exception = exc
            if time.monotonic() >= deadline:
                break
            time.sleep(min(max(0.01, poll_interval_seconds), max(0.0, deadline - time.monotonic())))

        if last_exception is not None:
            self._record_export_error("v4 observations query", last_exception)
        else:
            self.trace_export_error = (
                "Tracing export error: no Langfuse v4 observations appeared "
                f"within {timeout_seconds:g} seconds."
            )
        self.trace_debug_status = (
            f"v4 ingestion unconfirmed (trace_id={self.trace_id}, "
            f"observation_count={self.observation_count})"
        )
        return False

    def refresh_trace_url(self, *, force: bool = False) -> str | None:
        """Refresh a link without blocking; links require v4 ingestion confirmation."""

        if self.trace_url or not self.enabled or not self.trace_id:
            return self.trace_url
        now = time.monotonic()
        if not force and now - self._last_trace_url_check < 5:
            return None
        self._last_trace_url_check = now
        self.confirm_ingestion(timeout_seconds=0)
        return self.trace_url

    def manifest_trace_fields(self) -> dict[str, Any]:
        """Return the explicit trace-delivery state persisted with a run."""

        return {
            "trace_id": self.trace_id,
            "trace_url": self.trace_url if self.trace_ingest_confirmed else None,
            "trace_public": self.trace_public,
            "trace_ingest_confirmed": self.trace_ingest_confirmed,
            "observation_count": self.observation_count,
            "trace_export_error": self.trace_export_error,
            "trace_debug_status": self.trace_debug_status,
            "langfuse_host": self.host if self.client is not None else None,
            "langfuse_sdk_version": self.sdk_version,
        }

    def _remove_from_stack(self, observation_id: str) -> None:
        stack = self._span_stack.get()
        if stack and stack[-1] == observation_id:
            self._span_stack.set(stack[:-1])
            return
        if observation_id in stack:
            self._span_stack.set(tuple(item for item in stack if item != observation_id))

    def _disable_remote_tracing(self, message: str) -> None:
        self.enabled = False
        self.mode = "unavailable"
        self.status_message = STATUS_UNAVAILABLE
        self.trace_url = None
        logger.warning(message)

    def _record_export_error(self, operation: str, error: BaseException) -> None:
        safe_error = self.sanitize(str(error))
        self.trace_export_error = (
            f"Tracing export error during {operation}: "
            f"{error.__class__.__name__}: {safe_error}"
        )
        self.trace_debug_status = f"{operation} failed"


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
    sanitized = _sanitize_public_path_tokens(sanitized)
    for personal in sorted(personal_values, key=len, reverse=True):
        sanitized = re.sub(
            re.escape(personal),
            "[REDACTED_PERSON]",
            sanitized,
            flags=re.IGNORECASE,
        )
    return sanitized


def _sanitize_public_path_tokens(value: str) -> str:
    sanitized = value.replace(str(os.getcwd()) + "/", "")
    replacements = {
        "resume_draft.pdf": "resume_after.pdf",
        "resume_draft.tex": "resume_after.pdf",
        "approved.pdf": "resume_after.pdf",
        "approved.tex": "resume_after.pdf",
        "letter.pdf": "cover_letter.pdf",
        "letter.tex": "cover_letter.pdf",
        "cover_letter.tex": "cover_letter.pdf",
    }
    for stale, canonical in replacements.items():
        sanitized = sanitized.replace(stale, canonical)
    sanitized = re.sub(
        r"outputs/([^/]+)/source_resume/resume\.pdf",
        r"outputs/\1/resume_before.pdf",
        sanitized,
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


F = TypeVar("F", bound=Callable[..., Any])


def traced_tool(name: str, tracer: TraceManager) -> Callable[[F], F]:
    """Wrap a tool function in a TraceManager span."""

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with tracer.span(name, {"tool_name": name}):
                return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator
