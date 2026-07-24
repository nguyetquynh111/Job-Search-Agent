"""LangGraph workflow for the single-agent Job Search Agent."""

from __future__ import annotations

import logging
import sqlite3
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, ValidationError

from src.agent.controller import (
    CONTROLLER_SYSTEM_PROMPT_VERSION,
    SingleAgentController,
)
from src.agent.phase_policy import (
    MAX_REVISION_ROUNDS,
    Phase,
    RunStatus,
    WorkflowPolicyError,
    assert_can_revise,
    assert_cover_letters_allowed,
    assert_tool_allowed,
)
from src.agent.routing import (
    route_after_agent_controller,
    route_after_execute_tool,
    route_after_feedback_node,
    route_after_initialize,
    route_after_prepare_review,
    route_after_revision_controller,
)
from src.agent.state import AgentState, create_initial_state
from src.config import get_config
from src.data_loader import (
    InputLoadError,
    load_candidate_profile,
    load_jobs_csv,
    load_portfolio,
    load_resume_data,
    load_text_path,
)
from src.memory.extractor import extract_memory_facts
from src.memory.store import JSONMemoryStore, MemoryStoreError
from src.observability.trace_manager import TraceManager
from src.review.review_service import (
    ReviewSubmissionError,
    build_review_payload,
    normalize_review_feedback,
)
from src.schemas.common import normalize_string_list
from src.tools.registry import ToolSpec, load_tool_registry

logger = logging.getLogger(__name__)


def build_agent_graph(
    tools: dict[str, ToolSpec] | None = None,
    checkpointer: Any | None = None,
    tracer: TraceManager | None = None,
) -> Any:
    """Build the LangGraph workflow with one LLM agent/controller."""

    tool_registry = tools or load_tool_registry()
    trace_manager = tracer or TraceManager(enabled=False)
    controller = SingleAgentController(tool_registry)

    def initialize(state: AgentState) -> AgentState:
        _sync_trace_manager(state, trace_manager)
        if state.get("candidate_profile") and state.get("jobs"):
            return {}
        try:
            run_id = state["run_id"]
            thread_id = state["thread_id"]
            paths = state.get("input_paths", {})
            trace_id = trace_manager.start_run(
                run_id=run_id,
                session_id=thread_id,
                metadata={"status": "STARTED"},
            )
            load_jobs_metadata = {
                "run_id": run_id,
                "session_id": thread_id,
                "status": "STARTED",
            }
            with trace_manager.span("load_job_data", load_jobs_metadata):
                jobs = load_jobs_csv(paths.get("jobs_path", "data/jobs.csv"))
                load_jobs_metadata.update({"status": "OK", "result_count": len(jobs)})
            load_preferences_metadata = {
                "run_id": run_id,
                "session_id": thread_id,
                "status": "STARTED",
            }
            with trace_manager.span("load_preferences", load_preferences_metadata):
                profile = load_candidate_profile(
                    paths.get("preferences_path")
                    or paths.get(
                        "candidate_profile_path", "data/candidate_profile.yaml"
                    )
                )
                load_preferences_metadata.update(
                    {
                        "status": "OK",
                        "target_title_count": len(profile.preferences.target_titles),
                    }
                )
            load_resume_metadata = {
                "run_id": run_id,
                "session_id": thread_id,
                "status": "STARTED",
            }
            with trace_manager.span("load_resume", load_resume_metadata):
                resume_path = load_text_path(
                    paths.get("resume_path", "data/resume.tex"), "Resume"
                )
                resume_data = load_resume_data(resume_path)
                profile = profile.model_copy(
                    update={
                        "resume_content": resume_data.plain_text,
                        "skills": normalize_string_list(
                            [*profile.skills, *resume_data.skills]
                        ),
                        "education": normalize_string_list(
                            [*profile.education, *resume_data.education]
                        ),
                        "experience": normalize_string_list(
                            [*profile.experience, *resume_data.experience]
                        ),
                        "resume_projects": normalize_string_list(
                            [*profile.resume_projects, *resume_data.projects]
                        ),
                        "resume_evidence": [
                            *profile.resume_evidence,
                            *resume_data.evidence_items,
                        ],
                    }
                )
                load_resume_metadata.update(
                    {
                        "status": "OK",
                        "evidence_count": len(resume_data.evidence_items),
                        "resume_project_count": len(resume_data.projects),
                    }
                )
            load_portfolio_metadata = {
                "run_id": run_id,
                "session_id": thread_id,
                "status": "STARTED",
            }
            with trace_manager.span("load_portfolio", load_portfolio_metadata):
                portfolio = load_portfolio(
                    paths.get("portfolio_path", "data/portfolio.yaml")
                )
                load_portfolio_metadata.update(
                    {"status": "OK", "project_count": len(portfolio.projects)}
                )
            memory_file = paths.get(
                "memory_file",
                state.get("memory_file", str(get_config().memory_file)),
            )
            load_memory_metadata = {
                "run_id": run_id,
                "session_id": thread_id,
                "status": "STARTED",
            }
            with trace_manager.span("load_memory", load_memory_metadata):
                memory_facts = JSONMemoryStore(memory_file).load()
                load_memory_metadata.update(
                    {"status": "OK", "result_count": len(memory_facts)}
                )
            return {
                "phase": Phase.FILTER.value,
                "status": RunStatus.RUNNING.value,
                "jobs": [job.model_dump() for job in jobs],
                "candidate_profile": profile.model_dump(),
                "resume_path": resume_path,
                "portfolio": portfolio.model_dump(),
                "memory_file": memory_file,
                "memory_facts": [fact.model_dump() for fact in memory_facts],
                "trace_id": trace_id,
                "trace_url": trace_manager.trace_url,
                "langfuse_status": trace_manager.status_message,
            }
        except (InputLoadError, MemoryStoreError, ValidationError, Exception) as exc:
            trace_manager.record_error(
                exc,
                {
                    "run_id": state.get("run_id"),
                    "session_id": state.get("thread_id"),
                    "phase": Phase.INITIALIZE.value,
                },
            )
            logger.exception("Initialization failed")
            return _append_error(
                state,
                phase=Phase.INITIALIZE.value,
                message=str(exc),
                error_type=exc.__class__.__name__,
            )

    def agent_controller(state: AgentState) -> AgentState:
        _sync_trace_manager(state, trace_manager)
        try:
            with trace_manager.span("agent_controller", {"phase": state.get("phase")}):
                decision = controller.decide(state)
            metadata = {
                "run_id": state.get("run_id"),
                "session_id": state.get("thread_id"),
                "phase": state.get("phase"),
                "tool_name": decision.selected_tool,
                "model_name": controller.model_name,
                "system_prompt_version": CONTROLLER_SYSTEM_PROMPT_VERSION,
                "decision_summary": decision.decision_summary,
                "decision_source": decision.decision_source,
                "evidence_ids": decision.evidence_ids[:10],
                "evidence_count": len(decision.evidence_ids),
            }
            trace_manager.record_generation(metadata)
            return {
                "current_tool": decision.selected_tool,
                "current_tool_input": decision.arguments,
                "agent_decisions": [
                    *state.get("agent_decisions", []),
                    decision.model_dump(),
                ],
                **_trace_state(trace_manager),
            }
        except Exception as exc:
            trace_manager.record_error(
                exc,
                {
                    "run_id": state.get("run_id"),
                    "session_id": state.get("thread_id"),
                    "phase": state.get("phase", Phase.ERROR.value),
                },
            )
            logger.exception("Agent controller failed")
            return _append_error(
                state,
                phase=state.get("phase", Phase.ERROR.value),
                message=str(exc),
                error_type=exc.__class__.__name__,
            )

    def execute_tool(state: AgentState) -> AgentState:
        _sync_trace_manager(state, trace_manager)
        tool_name = state.get("current_tool")
        if not tool_name:
            return _append_error(
                state,
                phase=state.get("phase", Phase.ERROR.value),
                message="No current_tool selected by controller.",
                error_type="ToolExecutionError",
            )
        if tool_name not in tool_registry:
            return _append_error(
                state,
                phase=state.get("phase", Phase.ERROR.value),
                message=f"Tool {tool_name} is not registered.",
                error_type="ToolExecutionError",
            )
        spec = tool_registry[tool_name]
        phase = state.get("phase", Phase.ERROR.value)
        metadata: dict[str, Any] = {
            "run_id": state.get("run_id"),
            "session_id": state.get("thread_id"),
            "tool_name": tool_name,
            "phase": phase,
            "status": "STARTED",
        }
        try:
            assert_tool_allowed(phase, tool_name)
            input_model = spec.input_model.model_validate(
                state.get("current_tool_input", {})
            )
            metadata.update(_tool_input_metadata(tool_name, input_model))
            if tool_name == "tailor_resume":
                metadata["revision_round"] = int(state.get("revision_round", 0))
            span_name = _span_name_for_tool(tool_name, state, input_model)
            with trace_manager.span(span_name, metadata):
                raw_output = spec.func(input_model)
                output_model = spec.output_model.model_validate(
                    raw_output.model_dump()
                    if isinstance(raw_output, BaseModel)
                    else raw_output
                )
                metadata.update(
                    {
                        **_tool_output_metadata(tool_name, output_model),
                        "status": "OK",
                    }
                )
            if tool_name == "score_jobs":
                with trace_manager.span(
                    "select_top_3",
                    {
                        "run_id": state.get("run_id"),
                        "session_id": state.get("thread_id"),
                        "top_3_job_ids": output_model.model_dump().get(
                            "top_3_job_ids", []
                        ),
                        "phase": phase,
                        "status": "OK",
                    },
                ):
                    pass
            updates = _apply_tool_output(state, tool_name, output_model)
            updates.update(_trace_state(trace_manager))
            return updates
        except Exception as exc:
            metadata.update({"status": "ERROR", "error_type": exc.__class__.__name__})
            trace_manager.record_error(exc, metadata)
            logger.exception("Tool execution failed: %s", tool_name)
            return _append_error(
                state,
                phase=phase,
                message=str(exc),
                error_type=exc.__class__.__name__,
                tool_name=tool_name,
            )

    def prepare_review(state: AgentState) -> AgentState:
        _sync_trace_manager(state, trace_manager)
        try:
            review_metadata = {
                "run_id": state.get("run_id"),
                "session_id": state.get("thread_id"),
                "phase": Phase.HUMAN_REVIEW.value,
                "review_round": int(state.get("revision_round", 0)) + 1,
                "status": "STARTED",
            }
            with trace_manager.span("human_review", review_metadata):
                payload = build_review_payload(state)
                review_metadata.update(
                    {"status": "OK", "result_count": len(payload.resumes)}
                )
            trace_manager.flush()
            return {
                "interrupt_payload": payload.model_dump(),
                "status": RunStatus.WAITING_FOR_REVIEW.value,
                "phase": Phase.HUMAN_REVIEW.value,
                "new_memory_fact_ids": [],
                **_trace_state(trace_manager),
            }
        except Exception as exc:
            trace_manager.record_error(
                exc,
                {
                    "run_id": state.get("run_id"),
                    "session_id": state.get("thread_id"),
                    "phase": Phase.HUMAN_REVIEW.value,
                },
            )
            logger.exception("Review preparation failed")
            return _append_error(
                state,
                phase=Phase.HUMAN_REVIEW.value,
                message=str(exc),
                error_type=exc.__class__.__name__,
            )

    def human_review_interrupt(state: AgentState) -> AgentState:
        feedback = interrupt(state["interrupt_payload"])
        return {
            "review_feedback": feedback,
            "status": RunStatus.RUNNING.value,
        }

    def process_feedback(state: AgentState) -> AgentState:
        _sync_trace_manager(state, trace_manager)
        try:
            top_3_job_ids = state.get("top_3_job_ids", [])
            feedback = normalize_review_feedback(
                state.get("review_feedback", {}),
                expected_job_ids=top_3_job_ids,
            )
            rejected_job_ids = [
                job_id
                for job_id, decision in feedback.decisions.items()
                if decision.decision == "reject"
            ]
            approved_job_ids = [
                job_id
                for job_id, decision in feedback.decisions.items()
                if decision.decision == "approve"
            ]
            review_round = int(state.get("revision_round", 0)) + 1
            history_entry = {
                "review_round": review_round,
                "decisions": {
                    job_id: decision.model_dump()
                    for job_id, decision in feedback.decisions.items()
                },
                "rejected_job_ids": rejected_job_ids,
                "memory_writes": [],
                "actions_taken": {},
            }
            return {
                "review_decisions": {
                    job_id: decision.model_dump()
                    for job_id, decision in feedback.decisions.items()
                },
                "approved_job_ids": approved_job_ids,
                "pending_revision_job_ids": rejected_job_ids,
                "review_history": [*state.get("review_history", []), history_entry],
                **_trace_state(trace_manager),
            }
        except (ReviewSubmissionError, ValidationError, Exception) as exc:
            trace_manager.record_error(
                exc,
                {
                    "run_id": state.get("run_id"),
                    "session_id": state.get("thread_id"),
                    "phase": Phase.HUMAN_REVIEW.value,
                },
            )
            logger.exception("Review feedback failed")
            return _append_error(
                state,
                phase=Phase.HUMAN_REVIEW.value,
                message=str(exc),
                error_type=exc.__class__.__name__,
            )

    def update_memory(state: AgentState) -> AgentState:
        _sync_trace_manager(state, trace_manager)
        try:
            decisions = state.get("review_decisions", {})
            comments_by_job = {
                job_id: decision.get("comment", "")
                for job_id, decision in decisions.items()
                if decision.get("comment", "").strip()
            }
            review_round = int(state.get("revision_round", 0)) + 1
            facts = extract_memory_facts(comments_by_job, review_round)
            metadata = {
                "run_id": state.get("run_id"),
                "session_id": state.get("thread_id"),
                "review_round": review_round,
                "fact_count": len(facts),
                "source": "human_review",
                "status": "STARTED",
            }
            with trace_manager.span("persist_memory", metadata):
                store = JSONMemoryStore(
                    state.get("memory_file", str(get_config().memory_file))
                )
                before = store.load()
                before_keys = {
                    fact.deduplication_key for fact in before if fact.active
                }
                updated = store.append_many(facts)
                new_facts = [
                    fact
                    for fact in updated
                    if fact.active and fact.deduplication_key not in before_keys
                ]
                metadata["fact_count"] = len(new_facts)
                metadata["memory_fact_ids"] = [
                    fact.fact_id for fact in new_facts
                ]
                metadata["status"] = "OK"
            history = [*state.get("review_history", [])]
            if history:
                history[-1] = {
                    **history[-1],
                    "memory_writes": [
                        fact.model_dump() for fact in new_facts
                    ],
                }
            return {
                "memory_facts": [fact.model_dump() for fact in updated],
                "new_memory_fact_ids": [fact.fact_id for fact in new_facts],
                "review_history": history,
                **_trace_state(trace_manager),
            }
        except Exception as exc:
            trace_manager.record_error(
                exc,
                {
                    "run_id": state.get("run_id"),
                    "session_id": state.get("thread_id"),
                    "phase": Phase.HUMAN_REVIEW.value,
                },
            )
            logger.exception("Memory update failed")
            return _append_error(
                state,
                phase=Phase.HUMAN_REVIEW.value,
                message=str(exc),
                error_type=exc.__class__.__name__,
            )

    def revision_controller(state: AgentState) -> AgentState:
        _sync_trace_manager(state, trace_manager)
        try:
            rejected = list(state.get("pending_revision_job_ids", []))
            learned_fact_ids = list(state.get("new_memory_fact_ids", []))
            affected_job_ids = (
                list(state.get("top_3_job_ids", []))
                if learned_fact_ids
                else rejected
            )
            if not affected_job_ids:
                assert_cover_letters_allowed(
                    state.get("top_3_job_ids", []), state.get("approved_job_ids", [])
                )
                return {
                    "phase": Phase.COVER_LETTERS.value,
                    "status": RunStatus.RUNNING.value,
                    **_trace_state(trace_manager),
                }
            revision_round = int(state.get("revision_round", 0))
            if revision_round >= MAX_REVISION_ROUNDS:
                return {
                    "status": RunStatus.FAILED_REVIEW.value,
                    "phase": Phase.HUMAN_REVIEW.value,
                    "errors": [
                        *state.get("errors", []),
                        {
                            "phase": Phase.HUMAN_REVIEW.value,
                            "message": (
                                "Resume changes are still required after the "
                                "maximum revision rounds."
                            ),
                            "type": "FailedReview",
                        },
                    ],
                    **_trace_state(trace_manager),
                }
            assert_can_revise(revision_round)
            next_round = revision_round + 1
            with trace_manager.span(
                "revise_output",
                {
                    "run_id": state.get("run_id"),
                    "session_id": state.get("thread_id"),
                    "review_round": next_round,
                    "result_count": len(affected_job_ids),
                    "rejected_job_ids": rejected,
                    "affected_job_ids": affected_job_ids,
                    "new_memory_fact_ids": learned_fact_ids,
                    "status": "OK",
                },
            ):
                return {
                    "revision_round": next_round,
                    "pending_revision_job_ids": affected_job_ids,
                    "approved_job_ids": [
                        job_id
                        for job_id in state.get("approved_job_ids", [])
                        if job_id not in affected_job_ids
                    ],
                    "phase": Phase.TAILOR.value,
                    "status": RunStatus.RUNNING.value,
                    **_trace_state(trace_manager),
                }
        except Exception as exc:
            trace_manager.record_error(
                exc,
                {
                    "run_id": state.get("run_id"),
                    "session_id": state.get("thread_id"),
                    "phase": Phase.HUMAN_REVIEW.value,
                },
            )
            logger.exception("Revision controller failed")
            return _append_error(
                state,
                phase=Phase.HUMAN_REVIEW.value,
                message=str(exc),
                error_type=exc.__class__.__name__,
            )

    def finalize_resumes(state: AgentState) -> AgentState:
        _sync_trace_manager(state, trace_manager)
        try:
            assert_cover_letters_allowed(
                state.get("top_3_job_ids", []), state.get("approved_job_ids", [])
            )
            with trace_manager.span(
                "finalize_resumes",
                {
                    "run_id": state.get("run_id"),
                    "session_id": state.get("thread_id"),
                    "approved_job_ids": state.get("approved_job_ids", []),
                    "result_count": len(state.get("approved_job_ids", [])),
                    "status": "OK",
                },
            ):
                return {
                    "phase": Phase.COVER_LETTERS.value,
                    "status": RunStatus.RUNNING.value,
                    **_trace_state(trace_manager),
                }
        except WorkflowPolicyError as exc:
            trace_manager.record_error(
                exc,
                {
                    "run_id": state.get("run_id"),
                    "session_id": state.get("thread_id"),
                    "phase": Phase.HUMAN_REVIEW.value,
                },
            )
            return _append_error(
                state,
                phase=Phase.HUMAN_REVIEW.value,
                message=str(exc),
                error_type=exc.__class__.__name__,
            )

    def generate_cover_letters(state: AgentState) -> AgentState:
        return {
            "phase": Phase.COVER_LETTERS.value,
            "status": RunStatus.RUNNING.value,
            **_trace_state(trace_manager),
        }

    def complete(state: AgentState) -> AgentState:
        _sync_trace_manager(state, trace_manager)
        trace_manager.flush()
        return {
            "phase": Phase.COMPLETE.value,
            "status": RunStatus.COMPLETED.value,
            "trace_id": trace_manager.trace_id or state.get("trace_id"),
            "trace_url": trace_manager.trace_url or state.get("trace_url"),
            "langfuse_status": trace_manager.status_message,
        }

    def error(state: AgentState) -> AgentState:
        _sync_trace_manager(state, trace_manager)
        trace_manager.flush()
        status = state.get("status")
        if status == RunStatus.FAILED_REVIEW.value:
            return {
                "phase": Phase.HUMAN_REVIEW.value,
                "status": RunStatus.FAILED_REVIEW.value,
                **_trace_state(trace_manager),
            }
        return {
            "phase": Phase.ERROR.value,
            "status": RunStatus.FAILED.value,
            **_trace_state(trace_manager),
        }

    graph = StateGraph(AgentState)
    graph.add_node("initialize", initialize)
    graph.add_node("agent_controller", agent_controller)
    graph.add_node("execute_tool", execute_tool)
    graph.add_node("prepare_review", prepare_review)
    graph.add_node("human_review_interrupt", human_review_interrupt)
    graph.add_node("process_feedback", process_feedback)
    graph.add_node("update_memory", update_memory)
    graph.add_node("revision_controller", revision_controller)
    graph.add_node("finalize_resumes", finalize_resumes)
    graph.add_node("generate_cover_letters", generate_cover_letters)
    graph.add_node("complete", complete)
    graph.add_node("error", error)

    graph.add_edge(START, "initialize")
    graph.add_conditional_edges(
        "initialize",
        route_after_initialize,
        {"agent_controller": "agent_controller", "error": "error"},
    )
    graph.add_conditional_edges(
        "agent_controller",
        route_after_agent_controller,
        {"execute_tool": "execute_tool", "error": "error"},
    )
    graph.add_conditional_edges(
        "execute_tool",
        route_after_execute_tool,
        {
            "agent_controller": "agent_controller",
            "prepare_review": "prepare_review",
            "complete": "complete",
            "error": "error",
        },
    )
    graph.add_conditional_edges(
        "prepare_review",
        route_after_prepare_review,
        {"human_review_interrupt": "human_review_interrupt", "error": "error"},
    )
    graph.add_edge("human_review_interrupt", "process_feedback")
    graph.add_conditional_edges(
        "process_feedback",
        route_after_feedback_node,
        {"continue": "update_memory", "error": "error"},
    )
    graph.add_conditional_edges(
        "update_memory",
        route_after_feedback_node,
        {"continue": "revision_controller", "error": "error"},
    )
    graph.add_conditional_edges(
        "revision_controller",
        route_after_revision_controller,
        {
            "agent_controller": "agent_controller",
            "finalize_resumes": "finalize_resumes",
            "error": "error",
        },
    )
    graph.add_edge("finalize_resumes", "generate_cover_letters")
    graph.add_edge("generate_cover_letters", "agent_controller")
    graph.add_edge("complete", END)
    graph.add_edge("error", END)
    return graph.compile(checkpointer=checkpointer)


def create_sqlite_checkpointer(
    db_path: str | Path,
) -> tuple[Any, AbstractContextManager[Any]]:
    """Create a SQLite checkpointer and keep its context manager alive."""

    from langgraph.checkpoint.sqlite import SqliteSaver

    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    context = SqliteSaver.from_conn_string(str(path))
    return context.__enter__(), context


def create_memory_checkpointer() -> Any:
    """Create an in-memory checkpointer for tests."""

    from langgraph.checkpoint.memory import InMemorySaver

    return InMemorySaver()


def invoke_new_run(
    app: Any,
    state: AgentState | None = None,
    thread_id: str | None = None,
) -> dict[str, Any]:
    """Invoke a new graph run until completion or human interrupt."""

    run_state = state or create_initial_state(thread_id=thread_id)
    config = {"configurable": {"thread_id": run_state["thread_id"]}}
    return app.invoke(run_state, config=config)


def resume_run(app: Any, thread_id: str, feedback: dict[str, Any]) -> dict[str, Any]:
    """Resume a graph run from human review feedback."""

    config = {"configurable": {"thread_id": thread_id}}
    return app.invoke(Command(resume=feedback), config=config)


def _apply_tool_output(
    state: AgentState, tool_name: str, output_model: BaseModel
) -> AgentState:
    output = output_model.model_dump()
    history_entry = {
        "tool": tool_name,
        "phase": state.get("phase"),
        "input": _redact_large_values(state.get("current_tool_input", {})),
        "output": _redact_large_values(output),
    }
    updates: AgentState = {
        "tool_history": [*state.get("tool_history", []), history_entry],
        "current_tool": None,
        "current_tool_input": {},
    }
    top_3_job_ids = state.get("top_3_job_ids", [])
    if tool_name == "filter_jobs":
        updates.update(
            {
                "filtered_jobs": output["accepted_jobs"],
                "rejected_jobs": output["rejected_jobs"],
                "phase": Phase.SCORE.value,
            }
        )
    elif tool_name == "score_jobs":
        updates.update(
            {
                "ranked_jobs": output["ranked_jobs"],
                "top_3_job_ids": output["top_3_job_ids"],
                "phase": Phase.FIT_ANALYSIS.value,
            }
        )
    elif tool_name == "analyze_fit":
        analyses = {**state.get("fit_analyses", {}), output["job_id"]: output}
        next_phase = (
            Phase.TAILOR.value
            if all(job_id in analyses for job_id in top_3_job_ids)
            else Phase.FIT_ANALYSIS.value
        )
        updates.update({"fit_analyses": analyses, "phase": next_phase})
    elif tool_name == "tailor_resume":
        tailoring = {**state.get("tailoring_results", {}), output["job_id"]: output}
        was_revision = output["job_id"] in state.get(
            "pending_revision_job_ids", []
        )
        pending = [
            job_id
            for job_id in state.get("pending_revision_job_ids", [])
            if job_id != output["job_id"]
        ]
        all_tailored = all(job_id in tailoring for job_id in top_3_job_ids)
        next_phase = (
            Phase.HUMAN_REVIEW.value
            if all_tailored and not pending
            else Phase.TAILOR.value
        )
        updates.update(
            {
                "tailoring_results": tailoring,
                "pending_revision_job_ids": pending,
                "phase": next_phase,
            }
        )
        if was_revision:
            history = [*state.get("review_history", [])]
            if history:
                actions = {
                    **history[-1].get("actions_taken", {}),
                    output["job_id"]: {
                        "revision_round": int(state.get("revision_round", 0)),
                        "status": output.get("status"),
                        "change_log": output.get("change_log", []),
                        "output_tex_path": output.get("output_tex_path"),
                        "output_pdf_path": output.get("output_pdf_path"),
                        "memory_fact_ids_available": list(
                            state.get("new_memory_fact_ids", [])
                        ),
                    },
                }
                history[-1] = {**history[-1], "actions_taken": actions}
                updates["review_history"] = history
    elif tool_name == "generate_cover_letter":
        cover_letters = {
            **state.get("cover_letter_results", {}),
            output["job_id"]: output,
        }
        next_phase = (
            Phase.COMPLETE.value
            if all(job_id in cover_letters for job_id in top_3_job_ids)
            else Phase.COVER_LETTERS.value
        )
        updates.update({"cover_letter_results": cover_letters, "phase": next_phase})
    return updates


def _append_error(
    state: AgentState,
    phase: str,
    message: str,
    error_type: str,
    tool_name: str | None = None,
) -> AgentState:
    error = {
        "phase": phase,
        "message": message,
        "type": error_type,
        "tool": tool_name,
    }
    return {
        "phase": (
            Phase.ERROR.value
            if state.get("status") != RunStatus.FAILED_REVIEW.value
            else phase
        ),
        "status": RunStatus.FAILED.value,
        "errors": [*state.get("errors", []), error],
    }


def _sync_trace_manager(state: AgentState, tracer: TraceManager) -> None:
    run_id = state.get("run_id")
    if run_id and tracer.run_id != run_id:
        tracer.run_id = run_id
        tracer.trace_id = None
        tracer.trace_url = None
        tracer._root_recorded = False
    trace_id = state.get("trace_id")
    if trace_id:
        tracer.trace_id = trace_id
        tracer._root_recorded = True
    thread_id = state.get("thread_id")
    if thread_id:
        tracer.session_id = thread_id
    trace_url = state.get("trace_url")
    if trace_url:
        tracer.trace_url = trace_url


def _trace_state(tracer: TraceManager) -> AgentState:
    return {
        "trace_id": tracer.trace_id,
        "trace_url": tracer.trace_url,
        "langfuse_status": tracer.status_message,
    }


def _redact_large_values(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {key: _redact_large_values(value) for key, value in payload.items()}
    if isinstance(payload, list):
        if len(payload) > 5:
            return [_redact_large_values(value) for value in payload[:5]] + ["..."]
        return [_redact_large_values(value) for value in payload]
    if isinstance(payload, str) and len(payload) > 800:
        return payload[:800] + "..."
    return payload


def _tool_input_metadata(tool_name: str, input_model: BaseModel) -> dict[str, Any]:
    payload = input_model.model_dump()
    metadata: dict[str, Any] = {}
    if tool_name == "filter_jobs":
        metadata["input_job_count"] = len(payload.get("jobs", []))
    elif tool_name == "score_jobs":
        metadata["input_job_count"] = len(payload.get("jobs", []))
        metadata["resume_evidence_count"] = len(payload.get("resume_evidence", []))
        metadata["master_skill_evidence_count"] = len(
            payload.get("master_skill_evidence", [])
        )
        metadata["portfolio_evidence_count"] = len(
            payload.get("portfolio_evidence", [])
        )
        metadata["memory_evidence_count"] = len(payload.get("memory_evidence", []))
    elif tool_name in {"analyze_fit", "tailor_resume", "generate_cover_letter"}:
        metadata.update(_job_metadata(payload.get("job", {})))
        if tool_name == "tailor_resume":
            metadata["review_round"] = 1 if payload.get("revision_feedback") else 0
            metadata["revision_requested"] = bool(payload.get("revision_feedback"))
            metadata["evidence_count"] = len(payload.get("candidate_evidence", []))
        elif tool_name == "generate_cover_letter":
            metadata["evidence_count"] = len(payload.get("candidate_evidence", []))
        else:
            metadata["evidence_count"] = len(payload.get("evidence_items", []))
    return metadata


def _tool_output_metadata(tool_name: str, output_model: BaseModel) -> dict[str, Any]:
    payload = output_model.model_dump()
    if tool_name == "filter_jobs":
        accepted = payload.get("accepted_jobs", [])
        rejected = payload.get("rejected_jobs", [])
        return {
            "result_count": len(accepted),
            "rejected_count": len(rejected),
            "accepted_job_ids": [job.get("job_id") for job in accepted[:10]],
        }
    if tool_name == "score_jobs":
        ranked = payload.get("ranked_jobs", [])
        return {
            "result_count": len(ranked),
            "top_3_job_ids": payload.get("top_3_job_ids", []),
            "top_matches": [
                {
                    **_job_metadata(item.get("job", {})),
                    "match_score": item.get("score"),
                }
                for item in ranked[:3]
            ],
        }
    if tool_name in {"analyze_fit", "tailor_resume", "generate_cover_letter"}:
        metadata = {
            "job_id": payload.get("job_id"),
            "status": payload.get("status", "OK"),
            "error_count": len(payload.get("errors", [])),
            "evidence_ids": payload.get("evidence_used", [])[:10],
            "evidence_count": len(payload.get("evidence_used", [])),
        }
        page_count = payload.get("page_count")
        if isinstance(page_count, int):
            metadata["page_count"] = page_count
        if tool_name == "tailor_resume":
            metadata["change_count"] = len(payload.get("change_log", []))
        return metadata
    return {}


def _job_metadata(job: dict[str, Any]) -> dict[str, Any]:
    return {
        "job_id": job.get("job_id"),
        "job_title": job.get("title"),
        "company": job.get("company"),
    }


def _span_name_for_tool(
    tool_name: str, state: AgentState, input_model: BaseModel
) -> str:
    if tool_name not in {
        "analyze_fit",
        "tailor_resume",
        "generate_cover_letter",
    }:
        return tool_name
    payload = input_model.model_dump()
    job = payload.get("job", {})
    job_id = job.get("job_id") or payload.get("job_id")
    top_3 = state.get("top_3_job_ids", [])
    index = top_3.index(job_id) + 1 if job_id in top_3 else 1
    prefix = {
        "analyze_fit": "fit_analysis",
        "tailor_resume": "tailor_resume",
        "generate_cover_letter": "generate_cover_letter",
    }[tool_name]
    return f"{prefix}_job_{index}"
