"""Minimal DeepInfra (OpenAI-compatible) chat client for fit analysis.

Private plumbing for this tool only, mirroring the client construction and env
usage in ``src/agent/controller.py``. Intentionally one call function with no
provider registry, abstraction layers, retry framework, or caching. If this grows
past ~40 lines, promote it to shared infrastructure instead of expanding here.
"""

from __future__ import annotations

import logging
from typing import Any

from src.config import get_config
from src.observability.trace_manager import TraceManager

logger = logging.getLogger(__name__)


def model_configured() -> bool:
    """Return whether an LLM model and API key are configured."""

    config = get_config()
    return bool(config.llm_model and config.deepinfra_api_key)


def complete(
    system: str,
    user: str,
    *,
    tracer: TraceManager | None = None,
    metadata: dict[str, Any] | None = None,
    generation_name: str = "fit_analysis_llm",
) -> str:
    """Call the configured DeepInfra chat model and return the raw text response."""

    from langchain_openai import ChatOpenAI

    config = get_config()
    llm = ChatOpenAI(
        model=config.llm_model,
        api_key=config.deepinfra_api_key,
        base_url=config.deepinfra_base_url,
        temperature=0,
    )
    messages = [("system", system), ("human", user)]
    try:
        response = llm.invoke(messages)
        content = response.content
        text = content if isinstance(content, str) else str(content)
        if tracer is not None:
            tracer.record_generation(
                {
                    "provider": "deepinfra",
                    "purpose": "fit_analysis",
                    **(metadata or {}),
                },
                name=generation_name,
                model=config.llm_model,
                messages=messages,
                response=text,
                usage=_usage(response),
                model_parameters={
                    "temperature": 0,
                    "base_url": config.deepinfra_base_url,
                },
            )
        return text
    except Exception as exc:
        if tracer is not None:
            tracer.record_generation(
                {
                    "provider": "deepinfra",
                    "purpose": "fit_analysis",
                    **(metadata or {}),
                },
                name=generation_name,
                model=config.llm_model,
                messages=messages,
                response={"error_type": exc.__class__.__name__},
                usage={},
                model_parameters={
                    "temperature": 0,
                    "base_url": config.deepinfra_base_url,
                },
                status="ERROR",
                error_type=exc.__class__.__name__,
            )
        raise


def _usage(message: Any) -> dict[str, int]:
    usage = getattr(message, "usage_metadata", None)
    if not usage:
        metadata = getattr(message, "response_metadata", {}) or {}
        usage = metadata.get("token_usage") or metadata.get("usage")
    if not isinstance(usage, dict):
        return {}
    normalized: dict[str, int] = {}
    for source, target in (
        ("prompt_tokens", "input_tokens"),
        ("completion_tokens", "output_tokens"),
        ("total_tokens", "total_tokens"),
        ("input_tokens", "input_tokens"),
        ("output_tokens", "output_tokens"),
    ):
        value = usage.get(source)
        if isinstance(value, int):
            normalized[target] = value
    return normalized
