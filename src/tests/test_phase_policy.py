"""Phase policy tests."""

from __future__ import annotations

import pytest

from src.agent.controller import SingleAgentController
from src.config import DEEPINFRA_BASE_URL
from src.agent.phase_policy import (
    MAX_REVISION_ROUNDS,
    PHASE_TOOL_POLICY,
    Phase,
    WorkflowPolicyError,
    assert_can_revise,
    assert_cover_letters_allowed,
    assert_review_ready,
    assert_tool_allowed,
)
from src.tools.registry import load_tool_registry


def test_only_allowed_tools_can_run_in_each_phase() -> None:
    """Only configured tools can run in each phase."""

    for phase, allowed_tools in PHASE_TOOL_POLICY.items():
        for tool in allowed_tools:
            assert_tool_allowed(phase, tool)
        blocked = "generate_cover_letter" if phase != Phase.COVER_LETTERS.value else "score_jobs"
        if blocked not in allowed_tools:
            with pytest.raises(WorkflowPolicyError):
                assert_tool_allowed(phase, blocked)


def test_scoring_cannot_be_bypassed() -> None:
    """The controller rejects fit analysis before score_jobs output exists."""

    controller = SingleAgentController(load_tool_registry(), enable_llm=False)
    with pytest.raises(Exception, match="score_jobs"):
        controller.decide({"phase": Phase.FIT_ANALYSIS.value})


def test_controller_uses_deepinfra_env(monkeypatch) -> None:
    """DeepInfra credentials enable LLM decisions."""

    monkeypatch.delenv("DEEPINFRA_BASE_URL", raising=False)
    monkeypatch.setenv("DEEPINFRA_API_KEY", "test-token")
    monkeypatch.setenv("LLM_MODEL", "deepseek-ai/DeepSeek-V3")

    controller = SingleAgentController(load_tool_registry())

    assert controller.enable_llm is True
    assert controller.deepinfra_api_key == "test-token"
    assert controller.deepinfra_base_url == DEEPINFRA_BASE_URL


def test_top_3_are_selected_before_fit_analysis() -> None:
    """Fit analysis requires exactly three top job IDs."""

    with pytest.raises(WorkflowPolicyError):
        assert_review_ready(["J001", "J002"], {"J001": {}, "J002": {}})


def test_human_review_requires_all_three_tailoring_results() -> None:
    """Review cannot start with missing tailored resumes."""

    with pytest.raises(WorkflowPolicyError, match="Missing"):
        assert_review_ready(
            ["J001", "J002", "J003"],
            {"J001": {}, "J002": {}},
        )


def test_revision_round_cannot_exceed_two() -> None:
    """The revision guard enforces the configured maximum."""

    assert_can_revise(MAX_REVISION_ROUNDS - 1)
    with pytest.raises(WorkflowPolicyError):
        assert_can_revise(MAX_REVISION_ROUNDS)


def test_cover_letters_cannot_run_before_all_three_approvals() -> None:
    """Cover-letter phase requires all top-three resumes to be approved."""

    with pytest.raises(WorkflowPolicyError):
        assert_cover_letters_allowed(["J001", "J002", "J003"], ["J001", "J002"])
