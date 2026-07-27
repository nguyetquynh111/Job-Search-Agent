"""Model-driven structured tool selection for the workflow graph."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from pydantic import Field, ValidationError
from pydantic.types import SecretStr

from src.agent.errors import ToolExecutionError
from src.config import get_config
from src.domain import StrictBaseModel
from src.tools.registry import get_tool_definitions
from src.tracing.langfuse import TraceManager


class ModelToolCall(StrictBaseModel):
    """One structured model-selected tool call."""

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    rationale: str = ""


class ToolSelectionModel(Protocol):
    """Protocol implemented by the live tool-selecting model."""

    model_name: str

    def select_tool(
        self,
        *,
        available_tools: Sequence[str],
        tool_definitions: Sequence[dict[str, Any]],
        state_summary: Mapping[str, Any],
        tracer: TraceManager,
        previous_validation_results: Sequence[Mapping[str, Any]] = (),
    ) -> ModelToolCall:
        """Return exactly one structured tool call."""
        ...


@dataclass
class DeepInfraToolSelectionModel:
    """Live OpenAI-compatible tool selector using configured DeepInfra chat."""

    model_name: str = "deepinfra-tool-selector"
    configured_model_name: str | None = None

    def select_tool(
        self,
        *,
        available_tools: Sequence[str],
        tool_definitions: Sequence[dict[str, Any]],
        state_summary: Mapping[str, Any],
        tracer: TraceManager,
        previous_validation_results: Sequence[Mapping[str, Any]] = (),
    ) -> ModelToolCall:
        from langchain_openai import ChatOpenAI

        config = get_config()
        if not config.llm_model and not self.configured_model_name:
            raise ToolExecutionError("Live tool selection requires LLM_MODEL.")
        if not config.deepinfra_api_key:
            raise ToolExecutionError("Live tool selection requires DEEPINFRA_API_KEY.")
        model_name = self.configured_model_name or config.llm_model
        public_tools = _selection_tool_definitions(tool_definitions, available_tools)
        llm = ChatOpenAI(
            model=model_name,
            api_key=SecretStr(config.deepinfra_api_key),
            base_url=config.deepinfra_base_url,
            temperature=0,
            timeout=_request_timeout_seconds(),
            max_retries=1,
        ).bind_tools(public_tools)
        messages = [
            (
                "system",
                "You are the single job-search workflow agent. Inspect the "
                "repository state and choose exactly one callable tool for the "
                "next useful action. All registered tools are visible; workflow "
                "order is enforced by validators after your choice. For "
                "job-specific tools, include the target job_id. Return a concise "
                "rationale in the rationale argument. Do not invent scores, "
                "artifact paths, evidence, or resume content.",
            ),
            (
                "human",
                json.dumps(
                    {
                        "state_summary": state_summary,
                        "available_tools": list(available_tools),
                        "previous_validation_results": list(
                            previous_validation_results
                        ),
                    },
                    ensure_ascii=False,
                    default=str,
                ),
            ),
        ]
        generation_started_at = datetime.now(UTC)
        try:
            response = llm.invoke(messages)
        except Exception as exc:
            tracer.record_generation(
                {"purpose": "orchestration_tool_selection"},
                name="Workflow Decision LLM",
                model=model_name,
                messages=messages,
                response={"error_type": exc.__class__.__name__},
                status="ERROR",
                error_type=exc.__class__.__name__,
                model_parameters={
                    "temperature": 0,
                    "base_url": config.deepinfra_base_url,
                    "available_tools": list(available_tools),
                },
                start_time=generation_started_at,
            )
            raise
        content = str(getattr(response, "content", "") or "")
        tool_calls = list(getattr(response, "tool_calls", []) or [])
        tracer.record_generation(
            {
                "purpose": "orchestration_tool_selection",
                "available_tools": list(available_tools),
            },
            name="Workflow Decision LLM",
            model=model_name,
            messages=messages,
            response={"content": content, "tool_calls": _jsonable(tool_calls)},
            usage=_usage(response),
            model_parameters={"temperature": 0, "base_url": config.deepinfra_base_url},
            start_time=generation_started_at,
        )
        if len(tool_calls) != 1:
            if content.strip():
                return _parse_selection_json(content)
            raise ToolExecutionError(
                f"Model must select exactly one tool; received {len(tool_calls)}."
            )
        call = _parse_langchain_tool_call(tool_calls[0])
        rationale = str(call.arguments.pop("rationale", "") or content)
        return ModelToolCall(
            name=call.name,
            arguments=call.arguments,
            rationale=rationale.strip(),
        )


def default_tool_selection_model() -> ToolSelectionModel:
    """Return the live model-backed selector used by the workflow agent."""

    return DeepInfraToolSelectionModel()


def validate_model_tool_call(
    call: ModelToolCall | Mapping[str, Any],
    *,
    available_tools: Sequence[str] | None = None,
) -> ModelToolCall:
    """Reject unknown tools and malformed selector arguments.

    Workflow order is intentionally not checked here; graph guardrails validate
    the selected call against state after the model has made a real choice.
    """

    try:
        parsed = ModelToolCall.model_validate(call)
    except ValidationError as exc:
        raise ToolExecutionError(f"Malformed model tool call: {exc}") from exc
    known_tool_names = {definition["name"] for definition in get_tool_definitions()}
    if parsed.name not in known_tool_names:
        raise ToolExecutionError(f"Model selected unknown tool: {parsed.name}")
    if available_tools is not None and parsed.name not in set(available_tools):
        raise ToolExecutionError(
            f"Model selected unavailable tool {parsed.name!r}; "
            f"available={list(available_tools)}."
        )
    _validate_selector_arguments(parsed)
    return parsed


def _validate_selector_arguments(call: ModelToolCall) -> None:
    if not isinstance(call.arguments, dict):
        raise ToolExecutionError(
            f"Arguments for selected tool {call.name} must be a JSON object."
        )
    extra_keys = set(call.arguments) - {"job_id", "rationale"}
    if extra_keys:
        raise ToolExecutionError(
            f"Selector arguments for {call.name} may only include job_id and "
            f"rationale; received {sorted(extra_keys)}."
        )


def _selection_tool_definitions(
    tool_definitions: Sequence[dict[str, Any]],
    available_tools: Sequence[str],
) -> list[dict[str, Any]]:
    available = set(available_tools)
    definitions = []
    for definition in tool_definitions:
        if definition["name"] not in available:
            continue
        definitions.append(
            {
                "type": "function",
                "function": {
                    "name": definition["name"],
                    "description": (
                        definition["description"]
                        + " The workflow assembles full typed arguments from "
                        "validated repository state after selection."
                    ),
                    "parameters": _selector_schema_for(definition["name"]),
                },
            }
        )
    return definitions


def _selector_schema_for(tool_name: str) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "rationale": {
            "type": "string",
            "description": "Brief reason this is the right next tool.",
        }
    }
    required = ["rationale"]
    if tool_name in {"fit_analysis", "resume_tailoring", "cover_letter"}:
        properties["job_id"] = {
            "type": "string",
            "description": "The target job_id from the current top-three set.",
        }
        required.append("job_id")
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": required,
    }


def _parse_langchain_tool_call(raw_call: Any) -> ModelToolCall:
    if isinstance(raw_call, Mapping):
        name = raw_call.get("name") or raw_call.get("function", {}).get("name")
        args = raw_call.get("args")
        if args is None:
            function_args = raw_call.get("function", {}).get("arguments", {})
            args = (
                json.loads(function_args)
                if isinstance(function_args, str)
                else function_args
            )
        return ModelToolCall(name=str(name or ""), arguments=dict(args or {}))
    name = getattr(raw_call, "name", "")
    args = getattr(raw_call, "args", {})
    return ModelToolCall(name=str(name), arguments=dict(args or {}))


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    return value


def _usage(message: Any) -> dict[str, int]:
    usage = getattr(message, "usage_metadata", None) or {}
    normalized: dict[str, int] = {}
    for source, target in (
        ("input_tokens", "input_tokens"),
        ("output_tokens", "output_tokens"),
        ("total_tokens", "total_tokens"),
    ):
        value = usage.get(source)
        if isinstance(value, int):
            normalized[target] = value
    return normalized


def _request_timeout_seconds() -> float:
    raw = os.getenv("LLM_REQUEST_TIMEOUT_SECONDS", "45")
    try:
        return max(5.0, float(raw))
    except ValueError:
        return 45.0


def _parse_selection_json(content: str) -> ModelToolCall:
    """Parse compact JSON fallback when a chat model returns text."""

    text = content.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ToolExecutionError(
            "Model did not return valid JSON tool selection."
        ) from exc
    if not isinstance(payload, dict):
        raise ToolExecutionError("Model tool selection JSON must be an object.")
    arguments = dict(payload.get("arguments") or {})
    if payload.get("job_id"):
        arguments.setdefault("job_id", payload["job_id"])
    return ModelToolCall(
        name=str(payload.get("name", "")),
        arguments=arguments,
        rationale=str(payload.get("rationale", "")),
    )


__all__ = [
    "DeepInfraToolSelectionModel",
    "ModelToolCall",
    "ToolSelectionModel",
    "default_tool_selection_model",
    "validate_model_tool_call",
]
