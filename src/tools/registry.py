"""Public registry and executor for job-search agent tools."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from src.tools.cover_letter.cover_letter import (
    GenerateCoverLetterInput,
    GenerateCoverLetterOutput,
    run_cover_letter_tool,
)
from src.tools.filtering_scoring.filtering import (
    FilterJobsInput,
    FilterJobsOutput,
    run_filtering_tool,
)
from src.tools.filtering_scoring.scoring import (
    ScoreJobsInput,
    ScoreJobsOutput,
    run_scoring_tool,
)
from src.tools.fit_analysis.contracts import AnalyzeFitInput, FitAnalysisOutput
from src.tools.fit_analysis.fit_analysis import run_fit_analysis_tool
from src.tools.resume_tailoring.contracts import TailorResumeInput, TailorResumeOutput
from src.tools.resume_tailoring.resume_tailoring import (
    run_resume_tailoring_tool,
)
from src.tracing.langfuse import TraceManager


class ToolRegistryError(RuntimeError):
    """Raised when tool dispatch cannot complete."""


class UnknownToolError(ToolRegistryError):
    """Raised when a requested tool name is not registered."""


class ToolArgumentError(ToolRegistryError):
    """Raised when structured tool arguments do not match a contract."""


@dataclass(frozen=True)
class ToolSpec:
    """Stable public contract for one callable agent tool."""

    name: str
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    callable_name: str
    handler: Callable[..., BaseModel]

    @property
    def input_schema(self) -> dict[str, Any]:
        """Return the JSON schema supplied to tool-selecting models."""

        return self.input_model.model_json_schema()

    def invoke(
        self,
        arguments: BaseModel | Mapping[str, Any],
        *,
        tracer: TraceManager | None = None,
    ) -> BaseModel:
        """Validate arguments, execute the tool, and validate its result."""

        try:
            if isinstance(arguments, self.input_model):
                parsed = arguments
            else:
                parsed = self.input_model.model_validate(arguments)
        except ValidationError as exc:
            raise ToolArgumentError(
                f"Invalid arguments for tool {self.name!r}: {exc}"
            ) from exc

        result = self.handler(parsed, tracer=tracer)
        try:
            return self.output_model.model_validate(result)
        except ValidationError as exc:
            raise ToolRegistryError(
                f"Tool {self.name!r} returned invalid output: {exc}"
            ) from exc


_TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="filtering",
        description="Filter jobs against candidate preferences.",
        input_model=FilterJobsInput,
        output_model=FilterJobsOutput,
        callable_name="run_filtering_tool",
        handler=run_filtering_tool,
    ),
    ToolSpec(
        name="scoring",
        description="Deterministically score, rank, and select the Top 3 jobs.",
        input_model=ScoreJobsInput,
        output_model=ScoreJobsOutput,
        callable_name="run_scoring_tool",
        handler=run_scoring_tool,
    ),
    ToolSpec(
        name="fit_analysis",
        description="Analyze one job fit using evidence-bound claims.",
        input_model=AnalyzeFitInput,
        output_model=FitAnalysisOutput,
        callable_name="run_fit_analysis_tool",
        handler=run_fit_analysis_tool,
    ),
    ToolSpec(
        name="resume_tailoring",
        description="Tailor and compile a one-page resume for one job.",
        input_model=TailorResumeInput,
        output_model=TailorResumeOutput,
        callable_name="run_resume_tailoring_tool",
        handler=run_resume_tailoring_tool,
    ),
    ToolSpec(
        name="cover_letter",
        description="Generate and compile a one-page cover letter for one job.",
        input_model=GenerateCoverLetterInput,
        output_model=GenerateCoverLetterOutput,
        callable_name="run_cover_letter_tool",
        handler=run_cover_letter_tool,
    ),
)
_REGISTRY: dict[str, ToolSpec] = {spec.name: spec for spec in _TOOL_SPECS}


def get_tool_registry() -> dict[str, ToolSpec]:
    """Return registered tools keyed by canonical name."""

    return dict(_REGISTRY)


def get_registered_tools() -> list[ToolSpec]:
    """Return registered tools in deterministic execution order."""

    return list(_TOOL_SPECS)


def get_tool(name: str) -> ToolSpec:
    """Return one registered tool by canonical name."""

    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise UnknownToolError(f"Unknown tool: {name}") from exc


def get_tool_definitions() -> list[dict[str, Any]]:
    """Return structured tool definitions suitable for LLM tool binding."""

    return [
        {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
        }
        for tool in _TOOL_SPECS
    ]


def invoke_tool(
    name: str,
    arguments: BaseModel | Mapping[str, Any],
    context: Mapping[str, Any] | None = None,
) -> BaseModel:
    """Dispatch a structured tool call through the public registry."""

    tool = get_tool(name)
    tracer = None
    if context is not None:
        candidate = context.get("tracer")
        if isinstance(candidate, TraceManager):
            tracer = candidate

    # Registry dispatch is an implementation detail. The agent graph owns the
    # single assignment-level tool observation and passes the tracer through
    # solely so actual LLM generations can remain attached to that observation.
    return tool.invoke(arguments, tracer=tracer)


def _argument_job_id(arguments: BaseModel | Mapping[str, Any]) -> str | None:
    if isinstance(arguments, BaseModel):
        job = getattr(arguments, "job", None)
    else:
        job = arguments.get("job")
    if isinstance(job, BaseModel):
        value = getattr(job, "job_id", None)
    elif isinstance(job, Mapping):
        value = job.get("job_id")
    else:
        value = None
    return str(value) if value else None


def _serialize_arguments(arguments: BaseModel | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(arguments, BaseModel):
        return arguments.model_dump(mode="json")
    return dict(arguments)


def _result_summary(result: BaseModel) -> dict[str, Any]:
    payload = result.model_dump()
    summary: dict[str, Any] = {
        "output_type": type(result).__name__,
        "status": payload.get("status", "OK"),
    }
    for key in (
        "job_id",
        "top_3_job_ids",
        "accepted_jobs",
        "rejected_jobs",
        "ranked_jobs",
        "errors",
    ):
        if key not in payload:
            continue
        value = payload[key]
        if isinstance(value, list):
            summary[key] = len(value) if key.endswith("_jobs") else value[:10]
        else:
            summary[key] = value
    return summary


__all__ = [
    "ToolArgumentError",
    "ToolRegistryError",
    "ToolSpec",
    "UnknownToolError",
    "get_registered_tools",
    "get_tool",
    "get_tool_definitions",
    "get_tool_registry",
    "invoke_tool",
]
