"""Minimal DeepInfra (OpenAI-compatible) chat client for fit analysis.

Private plumbing for this tool only, mirroring the client construction and env
usage in ``src/agent/controller.py``. Intentionally one call function with no
provider registry, abstraction layers, retry framework, or caching. If this grows
past ~40 lines, promote it to shared infrastructure instead of expanding here.
"""

from __future__ import annotations

import logging

from src.config import get_config

logger = logging.getLogger(__name__)


def model_configured() -> bool:
    """Return whether an LLM model and API key are configured."""

    config = get_config()
    return bool(config.llm_model and config.deepinfra_api_key)


def complete(system: str, user: str) -> str:
    """Call the configured DeepInfra chat model and return the raw text response."""

    from langchain_openai import ChatOpenAI

    config = get_config()
    llm = ChatOpenAI(
        model=config.llm_model,
        api_key=config.deepinfra_api_key,
        base_url=config.deepinfra_base_url,
        temperature=0,
    )
    response = llm.invoke([("system", system), ("human", user)])
    content = response.content
    return content if isinstance(content, str) else str(content)
