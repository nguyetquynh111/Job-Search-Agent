"""Model-driven structured tool selection for the workflow graph."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import Field, ValidationError
from pydantic.types import SecretStr

from src.agent.errors import ToolExecutionError
from src.config import get_config
from src.domain import StrictBaseModel
from src.tools.registry import get_tool, get_tool_definitions
from src.tracing.langfuse import TraceManager


class ModelToolCall(StrictBaseModel):
    """One structured model-selected tool call."""

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    rationale: str = ""


class ToolSelectionModel(Protocol):
    """Protocol implemented by live and deterministic tool-selecting models."""

    model_name: str

    def select_tool(
        self,
        *,
        phase: str,
        allowed_tools: Sequence[str],
        prepared_arguments: Mapping[str, Any],
        tool_definitions: Sequence[dict[str, Any]],
        state_summary: Mapping[str, Any],
        tracer: TraceManager,
        previous_errors: Sequence[str] = (),
    ) -> ModelToolCall:
        """Return exactly one structured tool call."""
        ...


@dataclass
class DeterministicToolSelectionModel:
    """Offline selector for tests and local deterministic runs."""

    model_name: str = "deterministic-tool-selector"
    scripted_calls: list[ModelToolCall | Mapping[str, Any]] | None = None

    def select_tool(
        self,
        *,
        phase: str,
        allowed_tools: Sequence[str],
        prepared_arguments: Mapping[str, Any],
        tool_definitions: Sequence[dict[str, Any]],
        state_summary: Mapping[str, Any],
        tracer: TraceManager,
        previous_errors: Sequence[str] = (),
    ) -> ModelToolCall:
        del tool_definitions, state_summary
        if self.scripted_calls:
            raw_call = self.scripted_calls.pop(0)
            call = ModelToolCall.model_validate(raw_call)
        else:
            if len(allowed_tools) != 1:
                raise ToolExecutionError(
                    "Deterministic tool selection requires exactly one legal tool."
                )
            name = allowed_tools[0]
            call = ModelToolCall(
                name=name,
                arguments=_jsonable(prepared_arguments[name]),
                rationale="Selected the only legal next registered tool.",
            )
        tracer.record_generation(
            {
                "purpose": "orchestration_tool_selection",
                "phase": phase,
                "allowed_tools": list(allowed_tools),
                "previous_errors": list(previous_errors),
            },
            name="orchestration.model_decision",
            model=self.model_name,
            messages={"prepared_tool_names": list(prepared_arguments)},
            response=call.model_dump(mode="json"),
            usage={},
            model_parameters={"temperature": 0, "mock": True},
        )
        return call


@dataclass
class DeepInfraToolSelectionModel:
    """Live OpenAI-compatible tool selector using configured DeepInfra chat."""

    model_name: str = "deepinfra-tool-selector"
    configured_model_name: str | None = None

    def select_tool(
        self,
        *,
        phase: str,
        allowed_tools: Sequence[str],
        prepared_arguments: Mapping[str, Any],
        tool_definitions: Sequence[dict[str, Any]],
        state_summary: Mapping[str, Any],
        tracer: TraceManager,
        previous_errors: Sequence[str] = (),
    ) -> ModelToolCall:
        from langchain_openai import ChatOpenAI

        config = get_config()
        if not config.llm_model or not config.deepinfra_api_key:
            raise ToolExecutionError(
                "Live tool selection requires LLM_MODEL and DEEPINFRA_API_KEY."
            )
        model_name = self.configured_model_name or config.llm_model
        tool_schemas = [
            {
                "type": "function",
                "function": {
                    "name": definition["name"],
                    "description": definition["description"],
                    "parameters": definition["parameters"],
                },
            }
            for definition in tool_definitions
            if definition["name"] in set(allowed_tools)
        ]
        llm = ChatOpenAI(
            model=model_name,
            api_key=SecretStr(config.deepinfra_api_key),
            base_url=config.deepinfra_base_url,
            temperature=0,
        ).bind_tools(tool_schemas)
        messages = [
            (
                "system",
                "You are the job-search workflow controller. Select exactly one "
                "registered tool by emitting one tool call. Use only the supplied "
                "arguments for that tool. Do not invent scores or artifact paths.",
            ),
            (
                "human",
                json.dumps(
                    {
                        "phase": phase,
                        "allowed_tools": list(allowed_tools),
                        "state_summary": state_summary,
                        "prepared_arguments": {
                            name: _jsonable(prepared_arguments[name])
                            for name in allowed_tools
                        },
                        "previous_errors": list(previous_errors),
                    },
                    ensure_ascii=False,
                    default=str,
                ),
            ),
        ]
        try:
            response = llm.invoke(messages)
        except Exception as exc:
            tracer.record_generation(
                {"purpose": "orchestration_tool_selection", "phase": phase},
                name="orchestration.model_decision",
                model=model_name,
                messages=messages,
                response={"error_type": exc.__class__.__name__},
                status="ERROR",
                error_type=exc.__class__.__name__,
                model_parameters={
                    "temperature": 0,
                    "base_url": config.deepinfra_base_url,
                },
            )
            raise
        raw_calls = getattr(response, "tool_calls", None) or []
        tracer.record_generation(
            {
                "purpose": "orchestration_tool_selection",
                "phase": phase,
                "allowed_tools": list(allowed_tools),
            },
            name="orchestration.model_decision",
            model=model_name,
            messages=messages,
            response={
                "content": getattr(response, "content", ""),
                "tool_calls": _jsonable(raw_calls),
            },
            usage=_usage(response),
            model_parameters={"temperature": 0, "base_url": config.deepinfra_base_url},
        )
        if len(raw_calls) != 1:
            raise ToolExecutionError(
                f"Model must emit exactly one tool call; received {len(raw_calls)}."
            )
        return _parse_langchain_tool_call(raw_calls[0])


def default_tool_selection_model() -> ToolSelectionModel:
    """Use live selection when configured, otherwise an offline deterministic selector."""

    config = get_config()
    if config.llm_model and config.deepinfra_api_key:
        return DeepInfraToolSelectionModel()
    return DeterministicToolSelectionModel()


def select_validated_tool_call(
    model: ToolSelectionModel,
    *,
    phase: str,
    expected_tool: str,
    prepared_arguments: Mapping[str, Any],
    state_summary: Mapping[str, Any],
    tracer: TraceManager,
    max_invalid_calls: int = 2,
) -> ModelToolCall:
    """Ask the model for a legal call and validate it before execution."""

    allowed_tools = [expected_tool]
    definitions = get_tool_definitions()
    errors: list[str] = []
    for _attempt in range(max_invalid_calls + 1):
        call = model.select_tool(
            phase=phase,
            allowed_tools=allowed_tools,
            prepared_arguments=prepared_arguments,
            tool_definitions=definitions,
            state_summary=state_summary,
            tracer=tracer,
            previous_errors=errors,
        )
        try:
            validate_model_tool_call(call, expected_tool=expected_tool)
            return call
        except ToolExecutionError as exc:
            errors.append(str(exc))
    raise ToolExecutionError(
        "Model produced repeated invalid tool calls: " + "; ".join(errors)
    )


def validate_model_tool_call(
    call: ModelToolCall | Mapping[str, Any],
    *,
    expected_tool: str,
) -> ModelToolCall:
    """Reject unknown tools, illegal ordering, and malformed arguments."""

    try:
        parsed = ModelToolCall.model_validate(call)
    except ValidationError as exc:
        raise ToolExecutionError(f"Malformed model tool call: {exc}") from exc
    if parsed.name != expected_tool:
        known_tool_names = {definition["name"] for definition in get_tool_definitions()}
        if parsed.name not in known_tool_names:
            raise ToolExecutionError(f"Model selected unknown tool: {parsed.name}")
        raise ToolExecutionError(
            f"Illegal tool order: expected {expected_tool}, received {parsed.name}."
        )
    try:
        get_tool(parsed.name).input_model.model_validate(parsed.arguments)
    except ValidationError as exc:
        raise ToolExecutionError(
            f"Malformed arguments for selected tool {parsed.name}: {exc}"
        ) from exc
    return parsed


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


__all__ = [
    "DeepInfraToolSelectionModel",
    "DeterministicToolSelectionModel",
    "ModelToolCall",
    "ToolSelectionModel",
    "default_tool_selection_model",
    "select_validated_tool_call",
    "validate_model_tool_call",
]
