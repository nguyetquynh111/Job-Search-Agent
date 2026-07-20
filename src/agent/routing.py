"""Deterministic graph routing helpers."""

from __future__ import annotations

from src.agent.phase_policy import Phase, RunStatus
from src.agent.state import AgentState


def route_after_initialize(state: AgentState) -> str:
    """Route from initialize into the controller or error handler."""

    if state.get("status") == RunStatus.FAILED.value:
        return "error"
    return "agent_controller"


def route_after_agent_controller(state: AgentState) -> str:
    """Route from the controller to tool execution or error handling."""

    if state.get("status") == RunStatus.FAILED.value:
        return "error"
    return "execute_tool"


def route_after_execute_tool(state: AgentState) -> str:
    """Route after one tool execution."""

    if state.get("status") == RunStatus.FAILED.value:
        return "error"
    if state.get("phase") == Phase.HUMAN_REVIEW.value:
        return "prepare_review"
    if state.get("phase") == Phase.COMPLETE.value:
        return "complete"
    return "agent_controller"


def route_after_prepare_review(state: AgentState) -> str:
    """Route from review preparation to the interrupt or error handling."""

    if state.get("status") == RunStatus.FAILED.value:
        return "error"
    return "human_review_interrupt"


def route_after_feedback_node(state: AgentState) -> str:
    """Route after feedback or memory nodes."""

    if state.get("status") == RunStatus.FAILED.value:
        return "error"
    return "continue"


def route_after_revision_controller(state: AgentState) -> str:
    """Route after human feedback and memory updates."""

    if state.get("status") in {RunStatus.FAILED.value, RunStatus.FAILED_REVIEW.value}:
        return "error"
    if state.get("phase") == Phase.COVER_LETTERS.value:
        return "finalize_resumes"
    return "agent_controller"
