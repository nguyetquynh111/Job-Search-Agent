"""Central model-visible tool registry."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from src.schemas.cover_letter import GenerateCoverLetterInput, GenerateCoverLetterOutput
from src.schemas.filtering import FilterJobsInput, FilterJobsOutput
from src.schemas.fit_analysis import AnalyzeFitInput, FitAnalysisOutput
from src.schemas.scoring import ScoreJobsInput, ScoreJobsOutput
from src.schemas.tailoring import TailorResumeInput, TailorResumeOutput

ToolFunction = Callable[[BaseModel], BaseModel]


class MissingRealToolError(ImportError):
    """Raised when a real tool module is missing."""


@dataclass(frozen=True)
class ToolSpec:
    """A registered model-visible tool."""

    name: str
    func: ToolFunction
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    description: str


TOOL_CONTRACTS: dict[str, tuple[type[BaseModel], type[BaseModel], str]] = {
    "filter_jobs": (
        FilterJobsInput,
        FilterJobsOutput,
        "Filter jobs according to candidate preferences and log rejection reasons.",
    ),
    "score_jobs": (
        ScoreJobsInput,
        ScoreJobsOutput,
        "Score accepted jobs deterministically using profile and evidence.",
    ),
    "analyze_fit": (
        AnalyzeFitInput,
        FitAnalysisOutput,
        "Analyze candidate fit for one selected job using evidence.",
    ),
    "tailor_resume": (
        TailorResumeInput,
        TailorResumeOutput,
        "Tailor one resume for one selected job.",
    ),
    "generate_cover_letter": (
        GenerateCoverLetterInput,
        GenerateCoverLetterOutput,
        "Generate a cover letter for one approved resume and job.",
    ),
}


def load_tool_registry() -> dict[str, ToolSpec]:
    """Load the five model-visible tools from real implementations."""

    source_package = "src.tools.implementations"
    registry: dict[str, ToolSpec] = {}
    for name, (input_model, output_model, description) in TOOL_CONTRACTS.items():
        try:
            module = importlib.import_module(f"{source_package}.{name}")
        except ModuleNotFoundError as exc:
            raise MissingRealToolError(
                f"Real tool implementation missing: src/tools/implementations/{name}.py "
                f"with function {name}({input_model.__name__}) -> {output_model.__name__}"
            ) from exc
        func = getattr(module, name, None)
        if func is None:
            raise MissingRealToolError(
                f"Tool function {name} not found in {source_package}.{name}"
            )
        registry[name] = ToolSpec(
            name=name,
            func=func,
            input_model=input_model,
            output_model=output_model,
            description=description,
        )
    return registry


def as_langchain_tools(registry: dict[str, ToolSpec]) -> list[Any]:
    """Convert registry entries to LangChain StructuredTool objects when available."""

    try:
        from langchain_core.tools import StructuredTool
    except Exception:
        return []
    tools = []
    for spec in registry.values():
        tools.append(
            StructuredTool.from_function(
                func=lambda **kwargs: kwargs,
                name=spec.name,
                description=spec.description,
                args_schema=spec.input_model,
            )
        )
    return tools
