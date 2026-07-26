"""LangGraph workflow for the single-agent Job Search Agent."""

from __future__ import annotations

import inspect
import logging
import re
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
    PHASE_TOOL_POLICY,
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
from src.memory.extractor import extract_memory_facts, validate_memory_facts
from src.memory.models import MemoryFact
from src.memory.store import JSONMemoryStore, MemoryStoreError
from src.observability.trace_manager import TraceManager
from src.review.review_service import (
    ReviewSubmissionError,
    build_review_payload,
    normalize_review_feedback,
)
from src.schemas.common import normalize_string_list
from src.schemas.jobs import Job
from src.tools.evidence_validation import job_evidence_supports_skill
from src.tools.fit_analysis.render import build_source_labels, write_fit_analysis
from src.tools.job_evidence import build_job_evidence
from src.tools.registry import ToolSpec, load_tool_registry

logger = logging.getLogger(__name__)


class ToolExecutionError(RuntimeError):
    """Raised when a tool reports success without producing a valid artifact."""


def build_agent_graph(
    tools: dict[str, ToolSpec] | None = None,
    checkpointer: Any | None = None,
    tracer: TraceManager | None = None,
) -> Any:
    """Build the LangGraph workflow with one LLM agent/controller."""

    tool_registry = tools or load_tool_registry()
    trace_manager = tracer or TraceManager(enabled=False)
    controller = SingleAgentController(tool_registry, tracer=trace_manager)

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
                input={
                    "input_files": {
                        key: Path(value).name
                        for key, value in paths.items()
                        if isinstance(value, str)
                    }
                },
            )
            load_jobs_metadata = {
                "run_id": run_id,
                "session_id": thread_id,
                "status": "STARTED",
            }
            with trace_manager.span(
                "load_job_data",
                load_jobs_metadata,
                input={"file": Path(paths.get("jobs_path", "data/jobs.csv")).name},
            ) as load_jobs_span:
                jobs = load_jobs_csv(paths.get("jobs_path", "data/jobs.csv"))
                load_jobs_metadata.update({"status": "OK", "result_count": len(jobs)})
                trace_manager.update_span(
                    load_jobs_span,
                    output={
                        "job_count": len(jobs),
                        "job_ids": [job.job_id for job in jobs],
                    },
                )
            load_preferences_metadata = {
                "run_id": run_id,
                "session_id": thread_id,
                "status": "STARTED",
            }
            with trace_manager.span(
                "load_preferences",
                load_preferences_metadata,
                input={
                    "file": Path(
                        paths.get("preferences_path")
                        or paths.get(
                            "candidate_profile_path", "data/candidate_profile.yaml"
                        )
                    ).name
                },
            ) as load_preferences_span:
                profile = load_candidate_profile(
                    paths.get("preferences_path")
                    or paths.get(
                        "candidate_profile_path", "data/candidate_profile.yaml"
                    )
                )
                load_preferences_metadata.update(
                    {
                        "status": "OK",
                        "candidate_id": profile.candidate_id,
                        "target_title_count": len(profile.preferences.target_titles),
                    }
                )
                trace_manager.register_personal_data(profile.name, profile.email)
                trace_manager.update_span(
                    load_preferences_span,
                    output={
                        "candidate_id": profile.candidate_id,
                        "master_skill_count": len(profile.master_skills),
                        "target_title_count": len(profile.preferences.target_titles),
                    },
                )
            load_resume_metadata = {
                "run_id": run_id,
                "session_id": thread_id,
                "status": "STARTED",
            }
            with trace_manager.span(
                "load_resume",
                {**load_resume_metadata, "candidate_id": profile.candidate_id},
                input={"file": Path(paths.get("resume_path", "data/resume.tex")).name},
            ) as load_resume_span:
                resume_path = load_text_path(
                    paths.get("resume_path", "data/resume.tex"), "Resume"
                )
                resume_data = load_resume_data(resume_path)
                _register_resume_personal_data(
                    trace_manager,
                    Path(resume_path).read_text(encoding="utf-8"),
                )
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
                trace_manager.update_span(
                    load_resume_span,
                    metadata={
                        "status": "OK",
                        "candidate_id": profile.candidate_id,
                        "evidence_count": len(resume_data.evidence_items),
                    },
                    output={
                        "evidence_count": len(resume_data.evidence_items),
                        "resume_project_count": len(resume_data.projects),
                        "experience_count": len(resume_data.experience),
                    },
                )
            load_portfolio_metadata = {
                "run_id": run_id,
                "session_id": thread_id,
                "status": "STARTED",
            }
            with trace_manager.span(
                "load_portfolio",
                {**load_portfolio_metadata, "candidate_id": profile.candidate_id},
                input={
                    "file": Path(
                        paths.get("portfolio_path", "data/portfolio.yaml")
                    ).name
                },
            ) as load_portfolio_span:
                portfolio = load_portfolio(
                    paths.get("portfolio_path", "data/portfolio.yaml")
                )
                load_portfolio_metadata.update(
                    {"status": "OK", "project_count": len(portfolio.projects)}
                )
                trace_manager.update_span(
                    load_portfolio_span,
                    metadata={
                        "status": "OK",
                        "candidate_id": profile.candidate_id,
                        "project_count": len(portfolio.projects),
                    },
                    output={
                        "project_count": len(portfolio.projects),
                        "project_ids": [
                            project.project_id for project in portfolio.projects
                        ],
                    },
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
            with trace_manager.span(
                "load_memory",
                {**load_memory_metadata, "candidate_id": profile.candidate_id},
                input={"file": Path(memory_file).name},
            ) as load_memory_span:
                memory_facts = JSONMemoryStore(memory_file).load()
                load_memory_metadata.update(
                    {"status": "OK", "result_count": len(memory_facts)}
                )
                trace_manager.update_span(
                    load_memory_span,
                    metadata={
                        "status": "OK",
                        "candidate_id": profile.candidate_id,
                        "result_count": len(memory_facts),
                    },
                    output={
                        "fact_count": len(memory_facts),
                        "fact_ids": [fact.fact_id for fact in memory_facts],
                    },
                )
            trace_manager.update_run(
                metadata={
                    "status": "RUNNING",
                    "candidate_id": profile.candidate_id,
                    "input_job_count": len(jobs),
                }
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
            controller_metadata = {
                **_state_metadata(state),
                "phase": state.get("phase"),
            }
            with trace_manager.span(
                "agent_controller",
                controller_metadata,
                input={
                    "phase": state.get("phase"),
                    "top_3_job_ids": state.get("top_3_job_ids", []),
                    "pending_revision_job_ids": state.get(
                        "pending_revision_job_ids", []
                    ),
                    "current_workflow_state": {
                        "fit_analysis_completed": sorted(
                            state.get("fit_analyses", {})
                        ),
                        "tailoring_completed": sorted(
                            state.get("tailoring_results", {})
                        ),
                        "approved_job_ids": state.get("approved_job_ids", []),
                        "cover_letters_completed": sorted(
                            state.get("cover_letter_results", {})
                        ),
                        "revision_round": int(state.get("revision_round", 0)),
                    },
                },
                parent_observation_id=_workflow_trace_parent(state),
            ) as controller_span:
                decision = controller.decide(state)
                trace_manager.update_span(
                    controller_span,
                    output={
                        "available_actions": decision.available_actions,
                        "selected_tool": decision.selected_tool,
                        "target_job_id": decision.target_job_id,
                        "decision_source": decision.decision_source,
                        "decision_summary": decision.decision_summary,
                        "unresolved_requirements": decision.unresolved_requirements,
                    },
                    metadata={
                        "available_tools": sorted(
                            {
                                action["tool_name"]
                                for action in decision.available_actions
                            }
                        ),
                        "available_action_count": len(decision.available_actions),
                        "selected_tool": decision.selected_tool,
                        "target_job_id": decision.target_job_id,
                        "model_name": controller.model_name,
                        "system_prompt_version": CONTROLLER_SYSTEM_PROMPT_VERSION,
                        "decision_source": decision.decision_source,
                        "decision_reason": decision.decision_summary,
                        "evidence_ids": decision.evidence_ids,
                    },
                )
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
            **_state_metadata(state),
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
            metadata["configuration"] = _tool_trace_input(
                tool_name, input_model
            )["configuration"]
            if tool_name == "tailor_resume":
                metadata["revision_round"] = int(state.get("revision_round", 0))
            span_name = _span_name_for_tool(tool_name, state, input_model)
            with trace_manager.span(
                span_name,
                metadata,
                input=_tool_trace_input(tool_name, input_model),
                parent_observation_id=_workflow_trace_parent(state),
            ) as tool_span:
                raw_output = _invoke_tool(spec.func, input_model, trace_manager)
                output_model = spec.output_model.model_validate(
                    raw_output.model_dump()
                    if isinstance(raw_output, BaseModel)
                    else raw_output
                )
                if (
                    tool_name == "filter_jobs"
                    and len(output_model.accepted_jobs) < 3
                ):
                    raise ToolExecutionError(
                        "Exactly three Top jobs are required, but filtering accepted "
                        f"only {len(output_model.accepted_jobs)}. Add more job inputs "
                        "or relax the filtering preferences."
                    )
                fit_artifact_paths: dict[str, str] = {}
                if tool_name == "analyze_fit":
                    with trace_manager.span(
                        "fit_analysis.write_artifacts",
                        {
                            **_state_metadata(state),
                            "tool_name": tool_name,
                            "job_id": output_model.job_id,
                        },
                        input={
                            "formats": ["markdown", "json"],
                            "validated_analysis": True,
                        },
                    ) as artifact_span:
                        md_path, json_path = write_fit_analysis(
                            output_model,
                            input_model.job,
                            source_labels=build_source_labels(input_model),
                        )
                        fit_artifact_paths = {
                            "markdown_path": str(md_path),
                            "json_path": str(json_path),
                        }
                        trace_manager.update_span(
                            artifact_span,
                            metadata=fit_artifact_paths,
                            output={
                                **fit_artifact_paths,
                                "markdown_exists": md_path.is_file(),
                                "json_exists": json_path.is_file(),
                            },
                        )
                if tool_name in {"tailor_resume", "generate_cover_letter"}:
                    validation_input = _tool_output_metadata(tool_name, output_model)
                    with trace_manager.span(
                        "artifact_output_validation",
                        {
                            **_state_metadata(state),
                            "tool_name": tool_name,
                            "job_id": output_model.model_dump().get("job_id"),
                        },
                        input=validation_input,
                    ) as validation_span:
                        _validate_artifact_output(tool_name, output_model)
                        trace_manager.update_span(
                            validation_span,
                            output={"valid": True, **validation_input},
                        )
                metadata.update(
                    {
                        **_tool_output_metadata(tool_name, output_model),
                        "status": "OK",
                    }
                )
                trace_manager.update_span(
                    tool_span,
                    metadata=(
                        {"artifact_paths": fit_artifact_paths}
                        if fit_artifact_paths
                        else None
                    ),
                    output={
                        "result": output_model.model_dump(),
                        "summary": _tool_output_metadata(
                            tool_name, output_model
                        ),
                        "artifact_paths": fit_artifact_paths,
                    },
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
                    input={"ranked_job_count": len(output_model.ranked_jobs)},
                ) as top_span:
                    trace_manager.update_span(
                        top_span,
                        output={
                            "top_3_job_ids": output_model.top_3_job_ids,
                            "top_scores": [
                                item.score for item in output_model.ranked_jobs[:3]
                            ],
                        },
                    )
            updates = _apply_tool_output(state, tool_name, output_model)
            if fit_artifact_paths:
                updates["fit_analysis_artifacts"] = {
                    **state.get("fit_analysis_artifacts", {}),
                    output_model.job_id: fit_artifact_paths,
                }
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
            if state.get("review_history"):
                raise WorkflowPolicyError(
                    "The workflow permits exactly one human-review interrupt."
                )
            review_metadata = {
                "run_id": state.get("run_id"),
                "session_id": state.get("thread_id"),
                "phase": Phase.HUMAN_REVIEW.value,
                "review_round": int(state.get("revision_round", 0)) + 1,
                "status": "STARTED",
            }
            with trace_manager.span(
                "human_review",
                review_metadata,
                input={
                    "top_3_job_ids": state.get("top_3_job_ids", []),
                    "review_round": review_metadata["review_round"],
                },
            ) as review_parent:
                with trace_manager.span(
                    "human_review_pause",
                    review_metadata,
                    input={
                        "top_3_job_ids": state.get("top_3_job_ids", []),
                        "resume_paths": {
                            job_id: result.get("output_pdf_path")
                            for job_id, result in state.get(
                                "tailoring_results", {}
                            ).items()
                        },
                    },
                ) as review_span:
                    payload = build_review_payload(state)
                    review_metadata.update(
                        {"status": "OK", "result_count": len(payload.resumes)}
                    )
                    trace_manager.update_span(
                        review_span,
                        output=payload.model_dump(),
                    )
                trace_manager.update_span(
                    review_parent,
                    output={
                        "status": RunStatus.WAITING_FOR_REVIEW.value,
                        "resume_count": len(payload.resumes),
                        "job_ids": list(payload.resumes),
                    },
                )
            trace_manager.flush()
            return {
                "interrupt_payload": payload.model_dump(),
                "status": RunStatus.WAITING_FOR_REVIEW.value,
                "phase": Phase.HUMAN_REVIEW.value,
                "new_memory_fact_ids": [],
                "review_trace_parent_id": review_parent,
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
            review_round = int(state.get("revision_round", 0)) + 1
            with trace_manager.span(
                "human_review_feedback",
                {
                    **_state_metadata(state),
                    "phase": Phase.HUMAN_REVIEW.value,
                    "review_round": review_round,
                },
                input={"decisions": state.get("review_feedback", {})},
                parent_observation_id=state.get("review_trace_parent_id"),
            ) as feedback_span:
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
                trace_manager.update_span(
                    feedback_span,
                    output={
                        "decisions": {
                            job_id: decision.model_dump()
                            for job_id, decision in feedback.decisions.items()
                        },
                        "approved_job_ids": approved_job_ids,
                        "rejected_job_ids": rejected_job_ids,
                    },
                    metadata={
                        "approved_count": len(approved_job_ids),
                        "rejected_count": len(rejected_job_ids),
                        "status": "OK",
                    },
                )
            history_entry = {
                "review_round": review_round,
                "decisions": {
                    job_id: decision.model_dump()
                    for job_id, decision in feedback.decisions.items()
                },
                "rejected_job_ids": rejected_job_ids,
                "memory_writes": [],
                "actions_taken": {},
                "revision_rounds": [],
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
            extracted = extract_memory_facts(comments_by_job, review_round)
            facts, validation_failures = validate_memory_facts(
                extracted, comments_by_job
            )
            metadata = {
                **_state_metadata(state),
                "review_round": review_round,
                "fact_count": len(facts),
                "source": "human_review",
                "validation_failures": validation_failures,
                "status": "STARTED",
            }
            with trace_manager.span(
                "memory_write",
                metadata,
                input={"review_comments": comments_by_job},
                parent_observation_id=state.get("review_trace_parent_id"),
            ) as memory_span:
                store = JSONMemoryStore(
                    state.get("memory_file", str(get_config().memory_file))
                )
                before = store.load()
                before_keys = {fact.deduplication_key for fact in before if fact.active}
                updated = store.append_many(facts)
                new_facts = [
                    fact
                    for fact in updated
                    if fact.active and fact.deduplication_key not in before_keys
                ]
                metadata["fact_count"] = len(new_facts)
                metadata["memory_fact_ids"] = [fact.fact_id for fact in new_facts]
                affected_job_ids = _affected_jobs_for_memory(state, new_facts)
                fact_payloads = [
                    {
                        "fact": fact.canonical_value,
                        "fact_type": fact.fact_type,
                        "fact_id": fact.fact_id,
                        "provenance": fact.provenance.model_dump(),
                        "source_reviewer_feedback": fact.provenance.original_statement,
                        "timestamp": fact.created_at,
                        "affected_downstream_artifacts": [
                            f"{job_id}/resume" for job_id in affected_job_ids
                        ],
                    }
                    for fact in new_facts
                ]
                metadata.update(
                    {
                        "facts": fact_payloads,
                        "affected_job_ids": affected_job_ids,
                        "affected_downstream_artifacts": [
                            f"{job_id}/resume" for job_id in affected_job_ids
                        ],
                    }
                )
                metadata["status"] = "OK"
                trace_manager.update_span(
                    memory_span,
                    metadata=metadata,
                    output={
                        "new_facts": [fact.model_dump() for fact in new_facts],
                        "fact_count": len(new_facts),
                        "validation_failures": validation_failures,
                        "affected_job_ids": affected_job_ids,
                        "affected_downstream_artifacts": [
                            f"{job_id}/resume" for job_id in affected_job_ids
                        ],
                    },
                )
            history = [*state.get("review_history", [])]
            if history:
                history[-1] = {
                    **history[-1],
                    "memory_writes": [fact.model_dump() for fact in new_facts],
                }
            return {
                "memory_facts": [fact.model_dump() for fact in updated],
                "new_memory_fact_ids": [fact.fact_id for fact in new_facts],
                "memory_validation_failures": validation_failures,
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
            learned_facts = [
                MemoryFact.model_validate(item)
                for item in state.get("memory_facts", [])
                if item.get("fact_id") in set(learned_fact_ids)
            ]
            affected_job_ids = _affected_jobs_for_memory(
                state, learned_facts, always_include=rejected
            )
            analysis_refresh_job_ids = _affected_jobs_for_memory(
                state, learned_facts
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
                "review_revision",
                {
                    **_state_metadata(state),
                    "review_round": next_round,
                    "result_count": len(affected_job_ids),
                    "rejected_job_ids": rejected,
                    "affected_job_ids": affected_job_ids,
                    "fit_analysis_refresh_job_ids": analysis_refresh_job_ids,
                    "new_memory_fact_ids": learned_fact_ids,
                    "status": "OK",
                },
                input={
                    "review_decisions": state.get("review_decisions", {}),
                    "new_memory_fact_ids": learned_fact_ids,
                },
                parent_observation_id=state.get("review_trace_parent_id"),
            ) as revision_span:
                trace_manager.update_span(
                    revision_span,
                    output={
                        "affected_job_ids": affected_job_ids,
                        "fit_analysis_refresh_job_ids": analysis_refresh_job_ids,
                        "next_phase": (
                            Phase.FIT_ANALYSIS.value
                            if analysis_refresh_job_ids
                            else Phase.TAILOR.value
                        ),
                    },
                )
                return {
                    "revision_round": next_round,
                    "pending_revision_job_ids": affected_job_ids,
                    "fit_analysis_refresh_job_ids": analysis_refresh_job_ids,
                    "revision_round_job_ids": affected_job_ids,
                    "revision_trace_parent_id": revision_span,
                    "approved_job_ids": [
                        job_id
                        for job_id in state.get("approved_job_ids", [])
                        if job_id not in affected_job_ids
                    ],
                    "phase": (
                        Phase.FIT_ANALYSIS.value
                        if analysis_refresh_job_ids
                        else Phase.TAILOR.value
                    ),
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
                input={
                    "top_3_job_ids": state.get("top_3_job_ids", []),
                    "approved_job_ids": state.get("approved_job_ids", []),
                },
                parent_observation_id=state.get("review_trace_parent_id"),
            ) as finalize_span:
                trace_manager.update_span(
                    finalize_span,
                    output={
                        "approved": True,
                        "next_phase": Phase.COVER_LETTERS.value,
                    },
                )
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
        try:
            with trace_manager.span(
                "final_output_validation",
                {
                    **_state_metadata(state),
                    "phase": Phase.COMPLETE.value,
                },
                input={
                    "top_3_job_ids": state.get("top_3_job_ids", []),
                    "resume_job_ids": list(state.get("tailoring_results", {})),
                    "cover_letter_job_ids": list(state.get("cover_letter_results", {})),
                },
                parent_observation_id=state.get("review_trace_parent_id"),
            ) as validation_span:
                validation = _validate_final_outputs(state, tool_registry)
                trace_manager.update_span(
                    validation_span,
                    output={"valid": True, **validation},
                    metadata={"status": "OK"},
                )
            trace_manager.update_run(
                metadata={
                    "status": RunStatus.COMPLETED.value,
                    "candidate_id": _candidate_id(state),
                },
                output=validation,
            )
            trace_manager.flush()
            return {
                "phase": Phase.COMPLETE.value,
                "status": RunStatus.COMPLETED.value,
                "trace_id": trace_manager.trace_id or state.get("trace_id"),
                "trace_url": trace_manager.trace_url or state.get("trace_url"),
                "langfuse_status": trace_manager.status_message,
            }
        except Exception as exc:
            trace_manager.record_error(
                exc,
                {**_state_metadata(state), "phase": Phase.COMPLETE.value},
            )
            trace_manager.update_run(
                metadata={
                    "status": RunStatus.FAILED.value,
                    "error_type": exc.__class__.__name__,
                },
                output={"valid": False, "error_type": exc.__class__.__name__},
            )
            trace_manager.flush()
            return _append_error(
                state,
                phase=Phase.COMPLETE.value,
                message=str(exc),
                error_type=exc.__class__.__name__,
            )

    def error(state: AgentState) -> AgentState:
        _sync_trace_manager(state, trace_manager)
        trace_manager.update_run(
            metadata={
                "status": state.get("status", RunStatus.FAILED.value),
                "error_count": len(state.get("errors", [])),
            },
            output={
                "phase": state.get("phase"),
                "errors": state.get("errors", []),
            },
        )
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
        refresh = [
            job_id
            for job_id in state.get("fit_analysis_refresh_job_ids", [])
            if job_id != output["job_id"]
        ]
        if refresh:
            next_phase = Phase.FIT_ANALYSIS.value
        elif state.get("pending_revision_job_ids", []):
            next_phase = Phase.TAILOR.value
        else:
            next_phase = (
                Phase.TAILOR.value
                if all(job_id in analyses for job_id in top_3_job_ids)
                else Phase.FIT_ANALYSIS.value
            )
        updates.update(
            {
                "fit_analyses": analyses,
                "fit_analysis_refresh_job_ids": refresh,
                "phase": next_phase,
            }
        )
    elif tool_name == "tailor_resume":
        tailoring = {**state.get("tailoring_results", {}), output["job_id"]: output}
        was_revision = output["job_id"] in state.get("pending_revision_job_ids", [])
        pending = [
            job_id
            for job_id in state.get("pending_revision_job_ids", [])
            if job_id != output["job_id"]
        ]
        all_tailored = all(job_id in tailoring for job_id in top_3_job_ids)
        revision_batch_complete = was_revision and all_tailored and not pending
        current_round = int(state.get("revision_round", 0))
        round_targets = list(
            state.get("revision_round_job_ids", [])
            or state.get("pending_revision_job_ids", [])
        )
        unsatisfied = (
            [
                job_id
                for job_id in round_targets
                if tailoring.get(job_id, {}).get("revision_feedback_satisfied")
                is not True
            ]
            if revision_batch_complete
            else []
        )
        if (
            revision_batch_complete
            and unsatisfied
            and current_round < MAX_REVISION_ROUNDS
        ):
            next_phase = Phase.TAILOR.value
            pending = unsatisfied
        elif revision_batch_complete and unsatisfied:
            next_phase = Phase.HUMAN_REVIEW.value
        elif revision_batch_complete:
            next_phase = Phase.COVER_LETTERS.value
        elif all_tailored and not pending:
            next_phase = Phase.HUMAN_REVIEW.value
        else:
            missing_analyses = [
                job_id
                for job_id in top_3_job_ids
                if job_id not in state.get("fit_analyses", {})
            ]
            next_phase = (
                Phase.FIT_ANALYSIS.value
                if missing_analyses
                else Phase.TAILOR.value
            )
        updates.update(
            {
                "tailoring_results": tailoring,
                "pending_revision_job_ids": pending,
                "phase": next_phase,
            }
        )
        if (
            revision_batch_complete
            and unsatisfied
            and current_round < MAX_REVISION_ROUNDS
        ):
            updates["revision_round"] = current_round + 1
            updates["revision_round_job_ids"] = unsatisfied
        elif revision_batch_complete and unsatisfied:
            updates["revision_round_job_ids"] = []
            updates["status"] = RunStatus.FAILED_REVIEW.value
            updates["errors"] = [
                *state.get("errors", []),
                {
                    "phase": Phase.HUMAN_REVIEW.value,
                    "message": (
                        "Reviewer feedback remains unsatisfied after the maximum "
                        f"{MAX_REVISION_ROUNDS} revision rounds: {unsatisfied}."
                    ),
                    "type": "FailedReview",
                },
            ]
        elif revision_batch_complete:
            # Successful revisions satisfy the original review.
            updates["approved_job_ids"] = list(top_3_job_ids)
            updates["revision_round_job_ids"] = []
            updates["revision_trace_parent_id"] = None
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
                revision_rounds = [
                    {
                        **entry,
                        "actions": [*entry.get("actions", [])],
                    }
                    for entry in history[-1].get("revision_rounds", [])
                ]
                round_entry = next(
                    (
                        entry
                        for entry in revision_rounds
                        if entry.get("revision_round") == current_round
                    ),
                    None,
                )
                action = {
                    "job_id": output["job_id"],
                    "feedback_received": state.get("current_tool_input", {}).get(
                        "revision_feedback"
                    ),
                    "status": output.get("status"),
                    "feedback_satisfied": output.get(
                        "revision_feedback_satisfied"
                    ),
                    "feedback_checks": output.get(
                        "revision_feedback_checks", []
                    ),
                    "actions_taken": output.get("change_log", []),
                    "output_tex_path": output.get("output_tex_path"),
                    "output_pdf_path": output.get("output_pdf_path"),
                    "memory_fact_ids_available": list(
                        state.get("new_memory_fact_ids", [])
                    ),
                }
                if round_entry is None:
                    revision_rounds.append(
                        {
                            "revision_round": current_round,
                            "actions": [action],
                        }
                    )
                else:
                    round_entry["actions"].append(action)
                history[-1] = {
                    **history[-1],
                    "actions_taken": actions,
                    "revision_rounds": revision_rounds,
                }
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


def _candidate_id(state: AgentState) -> str | None:
    profile = state.get("candidate_profile", {})
    return profile.get("candidate_id") if isinstance(profile, dict) else None


def _affected_jobs_for_memory(
    state: AgentState,
    facts: list[MemoryFact],
    *,
    always_include: list[str] | None = None,
) -> list[str]:
    """Return rejected jobs plus other Top-3 jobs where a new fact is relevant."""

    ordered = list(state.get("top_3_job_ids", []))
    affected = set(always_include or [])
    jobs = {
        item["job_id"]: Job.model_validate(item)
        for item in state.get("jobs", [])
        if item.get("job_id") in ordered
    }
    for fact in facts:
        if fact.fact_type.casefold() != "skill":
            continue
        for job_id, job in jobs.items():
            if any(
                job_evidence_supports_skill(item, fact.canonical_value)
                for item in build_job_evidence(job)
            ):
                affected.add(job_id)
    return [job_id for job_id in ordered if job_id in affected]


def _register_resume_personal_data(tracer: TraceManager, source: str) -> None:
    """Register direct identifiers found in the LaTeX contact header."""

    header = source.partition("%----------HEADER----------")[2].partition(
        "%----------SUMMARY----------"
    )[0]
    name_match = re.search(r"\{\\Huge\s+\\scshape\s+([^{}]+)\}", header)
    if name_match:
        tracer.register_personal_data(name_match.group(1).strip())
    for value in re.findall(r"\\href\{([^{}]+)\}", header):
        tracer.register_personal_data(value.removeprefix("mailto:"))
    for value in re.findall(r"\b[A-Z][A-Za-z .'-]+,\s*[A-Z]{2}\b", header):
        tracer.register_personal_data(value.strip())


def _state_metadata(state: AgentState) -> dict[str, Any]:
    """Common identifiers attached to every workflow observation."""

    return {
        "run_id": state.get("run_id"),
        "session_id": state.get("thread_id"),
        "candidate_id": _candidate_id(state),
        "review_round": int(state.get("revision_round", 0)),
    }


def _workflow_trace_parent(state: AgentState) -> str | None:
    """Keep resumed revision work nested under its stable revision parent."""

    return (
        state.get("revision_trace_parent_id")
        or state.get("review_trace_parent_id")
    )


def _invoke_tool(func: Any, input_model: BaseModel, tracer: TraceManager) -> Any:
    """Pass the active tracer only to production tools that accept it."""

    parameters = inspect.signature(func).parameters
    if "tracer" in parameters:
        return func(input_model, tracer=tracer)
    return func(input_model)


def _tool_trace_input(tool_name: str, input_model: BaseModel) -> dict[str, Any]:
    """Return the complete structured input plus non-secret execution config."""

    metadata = _tool_input_metadata(tool_name, input_model)
    payload = input_model.model_dump()
    if tool_name in {"analyze_fit", "tailor_resume", "generate_cover_letter"}:
        metadata.update(_job_metadata(payload.get("job", {})))
    if tool_name == "filter_jobs":
        metadata["preference_flags"] = {
            "remote_only": payload.get("preferences", {}).get("remote_only"),
            "preferred_location_count": len(
                payload.get("preferences", {}).get("preferred_locations", [])
            ),
            "excluded_company_count": len(
                payload.get("preferences", {}).get("excluded_companies", [])
            ),
        }
    return {
        "tool_name": tool_name,
        "input_schema": input_model.__class__.__name__,
        "payload": payload,
        "configuration": {
            "controller_prompt_version": CONTROLLER_SYSTEM_PROMPT_VERSION,
            "phase_policy": PHASE_TOOL_POLICY.get(
                {
                    "filter_jobs": Phase.FILTER.value,
                    "score_jobs": Phase.SCORE.value,
                    "analyze_fit": Phase.FIT_ANALYSIS.value,
                    "tailor_resume": Phase.TAILOR.value,
                    "generate_cover_letter": Phase.COVER_LETTERS.value,
                }.get(tool_name, ""),
                [],
            ),
            "pdflatex_required": tool_name
            in {"tailor_resume", "generate_cover_letter"},
            "exactly_one_page_required": tool_name
            in {"tailor_resume", "generate_cover_letter"},
        },
        "summary": metadata,
    }


def _validate_final_outputs(
    state: AgentState, registry: dict[str, ToolSpec]
) -> dict[str, Any]:
    """Validate all six final PDFs before marking the root trace complete."""

    top_3 = list(state.get("top_3_job_ids", []))
    resumes = state.get("tailoring_results", {})
    letters = state.get("cover_letter_results", {})
    if len(top_3) != 3:
        raise ToolExecutionError("Final validation requires exactly three jobs.")
    for job_id in top_3:
        if job_id not in resumes:
            raise ToolExecutionError(f"Missing tailored resume for {job_id}.")
        if job_id not in letters:
            raise ToolExecutionError(f"Missing cover letter for {job_id}.")
        resume_output = registry["tailor_resume"].output_model.model_validate(
            resumes[job_id]
        )
        letter_output = registry["generate_cover_letter"].output_model.model_validate(
            letters[job_id]
        )
        _validate_artifact_output("tailor_resume", resume_output)
        _validate_artifact_output("generate_cover_letter", letter_output)
    return {
        "job_ids": top_3,
        "resume_count": len(top_3),
        "cover_letter_count": len(top_3),
        "pdf_count": len(top_3) * 2,
    }


def _validate_artifact_output(tool_name: str, output_model: BaseModel) -> None:
    """Reject placeholder paths, failed compilation, and non-one-page PDFs."""

    if tool_name not in {"tailor_resume", "generate_cover_letter"}:
        return
    payload = output_model.model_dump()
    errors = payload.get("errors", [])
    if errors:
        raise ToolExecutionError(
            f"{tool_name} reported errors: {'; '.join(str(item) for item in errors)}"
        )
    if tool_name == "tailor_resume" and payload.get("status") != "OK":
        raise ToolExecutionError(
            f"{tool_name} returned status {payload.get('status')!r}."
        )
    if payload.get("page_count") != 1:
        raise ToolExecutionError(
            f"{tool_name} must produce exactly one page; "
            f"reported {payload.get('page_count')!r}."
        )
    tex_path = Path(str(payload.get("output_tex_path") or ""))
    pdf_path = Path(str(payload.get("output_pdf_path") or ""))
    if not tex_path.is_file():
        raise ToolExecutionError(f"{tool_name} TeX output does not exist: {tex_path}")
    if not pdf_path.is_file():
        raise ToolExecutionError(f"{tool_name} PDF output does not exist: {pdf_path}")
    try:
        from pypdf import PdfReader

        actual_pages = len(PdfReader(str(pdf_path)).pages)
    except Exception as exc:
        raise ToolExecutionError(
            f"{tool_name} produced an unreadable PDF: {exc}"
        ) from exc
    if actual_pages != 1:
        raise ToolExecutionError(
            f"{tool_name} PDF has {actual_pages} pages; exactly one is required."
        )


def _sync_trace_manager(state: AgentState, tracer: TraceManager) -> None:
    run_id = state.get("run_id")
    trace_id = state.get("trace_id")
    thread_id = state.get("thread_id")
    if run_id and trace_id and thread_id:
        tracer.continue_run(
            run_id=run_id,
            session_id=thread_id,
            trace_id=trace_id,
            trace_url=state.get("trace_url"),
        )
        return
    if run_id and tracer.run_id != run_id:
        tracer.run_id = run_id
        tracer.trace_id = None
        tracer.trace_url = None
        tracer._root_recorded = False
        tracer._root_client = None
        tracer._active_spans.clear()
        tracer._span_stack.clear()
        tracer._personal_values.clear()
    if trace_id:
        tracer.trace_id = trace_id
        tracer._root_recorded = True
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
