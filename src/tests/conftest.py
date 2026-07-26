"""Shared pytest configuration."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def disable_llm_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep tests offline unless a test explicitly configures DeepInfra."""

    from src.agent import graph as graph_module
    from src.agent.controller import SingleAgentController

    monkeypatch.delenv("DEEPINFRA_API_KEY", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_HOST", raising=False)

    def deterministic_test_controller(*args, **kwargs):
        kwargs.setdefault("enable_llm", False)
        return SingleAgentController(*args, **kwargs)

    monkeypatch.setattr(
        graph_module,
        "SingleAgentController",
        deterministic_test_controller,
    )
