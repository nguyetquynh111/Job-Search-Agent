"""Explicit five-tool job-search workflow with one human-review pause.

You are the only LLM agent in this workflow.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel

from src.agent.errors import ToolExecutionError
from src.agent.state import AgentState, Phase, RunStatus, state_value
from src.agent.tool_selection import (
    ModelToolCall,
    ToolSelectionModel,
    default_tool_selection_model,
    validate_model_tool_call,
)
from src.config import (
    MAX_REVISION_ROUNDS,
    get_config,
)
from src.domain import (
    CandidateProfile,
    EvidenceItem,
    Job,
    Portfolio,
    normalize_string_list,
)
from src.utils.input_loading import (
    load_candidate_profile,
    load_jobs_csv,
    load_portfolio,
    load_resume_data,
    load_text_path,
)
from src.utils.job_evidence import build_job_evidence
from src.utils.latex import pdf_page_count
from src.utils.evidence_validation import job_evidence_supports_skill
from src.review.human_review import (
    build_review_payload,
    normalize_combined_review_submission,
    normalize_review_feedback,
)
from src.review.memory import (
    JSONMemoryStore,
    MemoryFact,
    extract_memory_facts,
    memory_fact_to_evidence,
    validate_memory_facts,
)
from src.tools.cover_letter.cover_letter import (
    GenerateCoverLetterInput,
    GenerateCoverLetterOutput,
)
from src.tools.filtering_scoring.filtering import FilterJobsInput, FilterJobsOutput
from src.tools.filtering_scoring.scoring import ScoreJobsInput, ScoreJobsOutput
from src.tools.fit_analysis.fit_analysis import (
    AnalyzeFitInput,
    FitAnalysisOutput,
    build_source_labels,
    write_fit_analysis,
)
from src.tools.resume_tailoring.contracts import (
    TailorResumeInput,
    TailorResumeOutput,
)
from src.tools.registry import (
    get_registered_tools,
    get_tool,
    get_tool_definitions,
    invoke_tool,
)
from src.tracing.langfuse import TraceManager
from src.utils.output_validation import (
    sanitize_public_artifact_references,
    write_and_validate_outputs,
    write_run_files,
)

logger = logging.getLogger(__name__)

__all__ = ["build_agent_graph", "validate_artifact_output"]


def build_agent_graph(
    *,
    checkpointer: Any | None = None,
    tracer: TraceManager | None = None,
    tool_selection_model: ToolSelectionModel | None = None,
    progress_callback: Callable[[AgentState], None] | None = None,
) -> Any:
    """Build the explicit workflow that dispatches required work via the registry."""

    trace_manager = tracer or TraceManager(enabled=False)
    selector = tool_selection_model or default_tool_selection_model()
    graph = StateGraph(AgentState)

    def initialize(state: AgentState) -> AgentState:
        """Load inputs and memory once before the tool sequence starts."""

        if state.get("candidate_profile") and state.get("jobs"):
            return {}
        paths = state.get("input_paths", {})
        trace_id = trace_manager.start_run(
            run_id=state_value(state, "run_id"),
            session_id=state_value(state, "thread_id"),
            metadata={
                "status": RunStatus.RUNNING.value,
                "available_tools": [tool.name for tool in get_registered_tools()],
            },
            input={"input_paths": paths},
        )
        dataset_span = trace_manager.start_span(
            "Analyze Dataset",
            metadata={"stage": "initialization"},
            input={"jobs_path": paths.get("jobs_path", "data/jobs.csv")},
        )
        try:
            jobs = load_jobs_csv(paths.get("jobs_path", "data/jobs.csv"))
        except Exception as exc:
            trace_manager.end_span(
                dataset_span,
                status="ERROR",
                error_type=exc.__class__.__name__,
                output={"error": str(exc)},
            )
            raise
        trace_manager.end_span(
            dataset_span,
            metadata={"job_count": len(jobs)},
            output={
                "job_count": len(jobs),
                "jobs": [job.model_dump(mode="json") for job in jobs],
            },
        )
        profile_span = trace_manager.start_span(
            "Candidate Profile",
            metadata={"stage": "initialization"},
            input={
                "candidate_profile_path": paths.get(
                    "candidate_profile_path", "data/preferences.yaml"
                ),
                "resume_path": paths.get("resume_path", "data/resume.tex"),
                "portfolio_path": paths.get("portfolio_path", "data/portfolio.txt"),
            },
        )
        try:
            profile = load_candidate_profile(
                paths.get("candidate_profile_path", "data/preferences.yaml")
            )
            resume_path = load_text_path(
                paths.get("resume_path", "data/resume.tex"), "Resume"
            )
            resume = load_resume_data(resume_path)
            portfolio = load_portfolio(
                paths.get("portfolio_path", "data/portfolio.txt")
            )
        except Exception as exc:
            trace_manager.end_span(
                profile_span,
                status="ERROR",
                error_type=exc.__class__.__name__,
                output={"error": str(exc)},
            )
            raise
        profile = profile.model_copy(
            update={
                "resume_content": resume.plain_text,
                "skills": normalize_string_list([*profile.skills, *resume.skills]),
                "education": normalize_string_list(
                    [*profile.education, *resume.education]
                ),
                "experience": normalize_string_list(
                    [*profile.experience, *resume.experience]
                ),
                "resume_projects": normalize_string_list(
                    [*profile.resume_projects, *resume.projects]
                ),
                "resume_evidence": [
                    *profile.resume_evidence,
                    *resume.evidence_items,
                ],
            }
        )
        memory_file = paths.get(
            "memory_file",
            state.get("memory_file", str(get_config().memory_file)),
        )
        memory_facts = JSONMemoryStore(memory_file).load()
        trace_manager.register_personal_data(profile.name, profile.email)
        trace_manager.end_span(
            profile_span,
            metadata={
                "portfolio_project_count": len(portfolio.projects),
                "resume_evidence_count": len(resume.evidence_items),
                "memory_fact_count": len(memory_facts),
            },
            output={
                "candidate_profile": profile.model_dump(mode="json"),
                "portfolio": portfolio.model_dump(mode="json"),
                "resume_path": resume_path,
                "memory_file": memory_file,
                "memory_fact_count": len(memory_facts),
            },
        )
        initialized: AgentState = {
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
        _publish_progress(progress_callback, {**state, **initialized})
        return initialized

    def run_pre_review_tools(state: AgentState) -> AgentState:
        """Let the agent iterate until the human-review gate is ready."""

        working: AgentState = {
            **state,
            "tool_history": list(state.get("tool_history", [])),
            "agent_decisions": list(state.get("agent_decisions", [])),
            "errors": list(state.get("errors", [])),
        }
        _run_agent_loop(
            selector,
            working,
            tracer=trace_manager,
            stop_condition=_ready_for_human_review,
            progress_callback=progress_callback,
        )

        return {
            "phase": Phase.HUMAN_REVIEW.value,
            "status": RunStatus.WAITING_FOR_REVIEW.value,
            "filtered_jobs": working.get("filtered_jobs", []),
            "rejected_jobs": working.get("rejected_jobs", []),
            "ranked_jobs": working.get("ranked_jobs", []),
            "top_3_job_ids": working.get("top_3_job_ids", []),
            "fit_analyses": working.get("fit_analyses", {}),
            "fit_analysis_artifacts": working.get("fit_analysis_artifacts", {}),
            "tailoring_results": working.get("tailoring_results", {}),
            "tool_history": working.get("tool_history", []),
            "agent_decisions": working.get("agent_decisions", []),
        }

    def human_review(state: AgentState) -> AgentState:
        """Pause at the one combined gate whenever the drafts need review."""

        payload = build_review_payload(dict(state))
        waiting = {
            **state,
            "phase": Phase.HUMAN_REVIEW.value,
            "status": RunStatus.WAITING_FOR_REVIEW.value,
            "interrupt_payload": payload.model_dump(),
        }
        # Execution stops at interrupt(), so sync queued observations before
        # the review UI exposes the public trace link.
        trace_manager.flush()
        _publish_progress(progress_callback, waiting)
        review_feedback = interrupt(payload.model_dump())
        span_id = trace_manager.start_span(
            "Human Review",
            input=payload.model_dump(),
            metadata={"review_round": payload.review_round},
        )
        trace_manager.end_span(
            span_id,
            output={
                "action": review_feedback.get("action"),
                "decisions": review_feedback.get("decisions", {}),
                "review_comment": review_feedback.get("reviewer_feedback", ""),
            },
        )
        return {
            "interrupt_payload": payload.model_dump(),
            "review_feedback": review_feedback,
            "status": RunStatus.RUNNING.value,
        }

    def process_review(state: AgentState) -> AgentState:
        """Process all three per-resume decisions from the workflow's only pause."""

        top_job_ids = list(state_value(state, "top_3_job_ids"))
        raw_feedback = dict(state.get("review_feedback", {}))
        if "decisions" in raw_feedback:
            feedback = normalize_review_feedback(raw_feedback, top_job_ids)
            decisions = {
                job_id: decision.model_dump()
                for job_id, decision in feedback.decisions.items()
            }
        else:
            # Keep the controller API compatible with older console clients while
            # converting their one batch action into the per-resume contract.
            submission = normalize_combined_review_submission(
                raw_feedback,
                top_job_ids,
            )
            affected = set(submission.affected_job_ids)
            decisions = {
                job_id: {
                    "decision": (
                        "reject"
                        if submission.action == "request_revision"
                        and job_id in affected
                        else "approve"
                    ),
                    "comment": (
                        submission.reviewer_feedback if job_id in affected else ""
                    ),
                }
                for job_id in top_job_ids
            }

        memory_facts, written_fact_ids, memory_failures = _store_review_memory(
            state,
            decisions,
            tracer=trace_manager,
        )
        propagation_actions = _plan_memory_propagation(
            state,
            decisions,
            memory_facts,
            written_fact_ids,
            tracer=trace_manager,
        )
        propagation_job_ids = [
            action["job_id"]
            for action in propagation_actions
            if action["status"] == "pending_tailoring"
        ]
        propagation_feedback = {
            action["job_id"]: action["feedback"]
            for action in propagation_actions
            if action.get("feedback")
        }
        rejected_job_ids = [
            job_id
            for job_id, decision in decisions.items()
            if decision["decision"] == "reject"
        ]
        needs_rework = bool(rejected_job_ids or propagation_job_ids)
        return {
            "phase": (
                Phase.REVISION.value if needs_rework else Phase.COVER_LETTERS.value
            ),
            "status": RunStatus.RUNNING.value,
            "review_decisions": decisions,
            "approved_job_ids": top_job_ids,
            "pending_revision_job_ids": rejected_job_ids,
            "revision_round_job_ids": rejected_job_ids,
            "memory_facts": memory_facts,
            "new_memory_fact_ids": list(
                dict.fromkeys(
                    [*state.get("new_memory_fact_ids", []), *written_fact_ids]
                )
            ),
            "memory_validation_failures": memory_failures,
            "memory_propagation_actions": propagation_actions,
            "memory_propagation_job_ids": propagation_job_ids,
            "memory_propagation_feedback_by_job": propagation_feedback,
        }

    def revise_resumes(state: AgentState) -> AgentState:
        """Apply feedback, retry at most twice, and never open another human gate."""

        working: AgentState = {
            **state,
            "phase": Phase.REVISION.value,
            "status": RunStatus.RUNNING.value,
            "tool_history": list(state.get("tool_history", [])),
            "agent_decisions": list(state.get("agent_decisions", [])),
            "review_actions": {},
            "revision_round_logs": {},
        }
        _publish_progress(progress_callback, working)
        _run_agent_loop(
            selector,
            working,
            tracer=trace_manager,
            stop_condition=lambda current: (
                not _pending_revision_job_ids(current)
                or current.get("status") == RunStatus.FAILED_REVIEW.value
            ),
            progress_callback=progress_callback,
        )
        if working.get("status") == RunStatus.FAILED_REVIEW.value:
            raise ToolExecutionError(
                "Review feedback could not be satisfied within the maximum "
                f"{MAX_REVISION_ROUNDS} revision rounds."
            )
        history = [
            *state.get("review_history", []),
            {
                "review_round": 1,
                "decisions": dict(state.get("review_decisions", {})),
                "rejected_job_ids": list(state.get("revision_round_job_ids", [])),
                "memory_writes": [
                    fact
                    for fact in state.get("memory_facts", [])
                    if fact.get("fact_id") in set(state.get("new_memory_fact_ids", []))
                ],
                "affected_job_ids": list(state.get("revision_round_job_ids", [])),
                "actions_taken": dict(working.get("review_actions", {})),
                "revision_rounds": list(
                    working.get("revision_round_logs", {}).values()
                ),
            },
        ]
        return {
            "phase": Phase.COVER_LETTERS.value,
            "status": RunStatus.RUNNING.value,
            "revision_round": max(
                [0, *working.get("revision_attempts_by_job", {}).values()]
            ),
            "tailoring_results": working.get("tailoring_results", {}),
            "tool_history": working.get("tool_history", []),
            "agent_decisions": working.get("agent_decisions", []),
            "review_history": history,
            "review_decisions": dict(state.get("review_decisions", {})),
            "approved_job_ids": list(state.get("top_3_job_ids", [])),
            "pending_revision_job_ids": [],
            "memory_propagation_job_ids": [],
        }

    def approve_and_finish(state: AgentState) -> AgentState:
        """Generate final PDFs and letters after the single review gate."""

        top_job_ids = list(state_value(state, "top_3_job_ids"))
        if _pending_revision_job_ids(state):
            raise ToolExecutionError(
                "Finalization cannot start while review rework is pending."
            )
        memory_facts = list(state.get("memory_facts", []))
        new_fact_ids = list(state.get("new_memory_fact_ids", []))
        working: AgentState = {
            **state,
            "phase": Phase.COVER_LETTERS.value,
            "status": RunStatus.RUNNING.value,
            "memory_facts": memory_facts,
            "new_memory_fact_ids": new_fact_ids,
            "tool_history": list(state.get("tool_history", [])),
            "agent_decisions": list(state.get("agent_decisions", [])),
        }
        _publish_progress(progress_callback, working)
        _run_agent_loop(
            selector,
            working,
            tracer=trace_manager,
            stop_condition=_post_review_done,
            progress_callback=progress_callback,
        )
        approval_record = {
            "review_round": 1,
            "action": "finalized_after_single_review",
            "decisions": dict(state.get("review_decisions", {})),
            "memory_writes": [
                fact
                for fact in memory_facts
                if fact.get("fact_id") in set(new_fact_ids)
            ],
        }
        review_history = [*state.get("review_history", []), approval_record]
        final_payload = {
            **state,
            "trace_id": trace_manager.trace_id,
            "trace_url": None,
            "trace_public": trace_manager.trace_public,
            "trace_ingest_confirmed": False,
            "observation_count": 0,
            "langfuse_status": trace_manager.status_message,
            "review_decisions": working.get("review_decisions", {}),
            "approved_job_ids": top_job_ids,
            "tailoring_results": working.get("tailoring_results", {}),
            "cover_letter_results": working.get("cover_letter_results", {}),
            "fit_analysis_artifacts": state.get("fit_analysis_artifacts", {}),
            "top_3_job_ids": top_job_ids,
            "jobs": state.get("jobs", []),
            "review_history": review_history,
            "agent_decisions": working.get("agent_decisions", []),
        }
        final_outputs_span = trace_manager.start_span(
            "Final Outputs",
            metadata={"stage": "finalization"},
            input={
                "top_3_job_ids": top_job_ids,
                "resume_results": working.get("tailoring_results", {}),
                "cover_letter_results": working.get("cover_letter_results", {}),
            },
        )
        try:
            try:
                output_manifest = write_and_validate_outputs(
                    final_payload,
                    persist_run_files=False,
                )
            except Exception as exc:
                trace_manager.end_span(
                    final_outputs_span,
                    status="ERROR",
                    error_type=exc.__class__.__name__,
                    output={"error": str(exc)},
                )
                raise
            final_output_summary = sanitize_public_artifact_references(
                {
                    "top_3_job_ids": top_job_ids,
                    "output_manifest": output_manifest,
                    "artifact_paths": _final_artifact_paths(
                        output_manifest,
                        working.get("tailoring_results", {}),
                        working.get("cover_letter_results", {}),
                    ),
                    "cover_letter_results": working.get("cover_letter_results", {}),
                    "review_history": review_history,
                    "memory_update": approval_record,
                }
            )
            trace_manager.end_span(
                final_outputs_span,
                output=final_output_summary,
            )
            trace_manager.update_run(
                metadata={"status": RunStatus.COMPLETED.value},
                output=final_output_summary,
            )
        finally:
            # The root must be ended and OTEL must be flushed even when artifact
            # validation or finalization raises. No completion manifest exists yet.
            trace_manager.end_root_observation()
            trace_manager.flush()

        trace_manager.confirm_ingestion(timeout_seconds=20)
        trace_fields = trace_manager.manifest_trace_fields()
        final_payload.update(trace_fields)
        final_payload["trace_events"] = [event.__dict__ for event in trace_manager.events]
        output_manifest.update(trace_fields)
        write_run_files(final_payload, output_manifest)
        return {
            "phase": Phase.COMPLETE.value,
            "status": RunStatus.COMPLETED.value,
            "review_decisions": working.get("review_decisions", {}),
            "approved_job_ids": top_job_ids,
            "revision_round": int(state.get("revision_round", 0)),
            "tailoring_results": working.get("tailoring_results", {}),
            "cover_letter_results": working.get("cover_letter_results", {}),
            "memory_facts": memory_facts,
            "new_memory_fact_ids": new_fact_ids,
            "memory_validation_failures": [],
            "review_history": review_history,
            "tool_history": working.get("tool_history", []),
            "agent_decisions": working.get("agent_decisions", []),
            "output_manifest": output_manifest,
            **trace_fields,
            "langfuse_status": trace_manager.status_message,
        }

    graph.add_node("initialize", initialize)
    graph.add_node("run_pre_review_tools", run_pre_review_tools)
    graph.add_node("human_review", human_review)
    graph.add_node("process_review", process_review)
    graph.add_node("revise_resumes", revise_resumes)
    graph.add_node("approve_and_finish", approve_and_finish)
    graph.add_edge(START, "initialize")
    graph.add_edge("initialize", "run_pre_review_tools")
    graph.add_edge("run_pre_review_tools", "human_review")
    graph.add_edge("human_review", "process_review")
    graph.add_conditional_edges(
        "process_review",
        lambda state: state.get("phase"),
        {
            Phase.REVISION.value: "revise_resumes",
            Phase.COVER_LETTERS.value: "approve_and_finish",
        },
    )
    graph.add_edge("revise_resumes", "approve_and_finish")
    graph.add_edge("approve_and_finish", END)
    return graph.compile(checkpointer=checkpointer)


def _run_agent_loop(
    selector: ToolSelectionModel,
    state: AgentState,
    *,
    tracer: TraceManager,
    stop_condition: Callable[[AgentState], bool],
    progress_callback: Callable[[AgentState], None] | None = None,
    max_iterations: int = 30,
    max_invalid_calls: int = 3,
) -> None:
    """Run the single agent's iterative decide-validate-dispatch loop."""

    invalid_results: list[dict[str, Any]] = []
    for _iteration in range(max_iterations):
        if stop_condition(state):
            return
        current_summary = _decision_state_summary(state)
        decision_phase = state.get("phase")
        decision_name = _decision_span_name(decision_phase)
        available_tools = _available_tool_names()
        span_id = tracer.start_span(
            decision_name,
            metadata={
                "phase": decision_phase,
                "workflow_stage": _workflow_stage_label(decision_phase),
            },
            input={
                "current_state": current_summary,
                "available_tools": _tool_definition_summary(),
                "previous_validation_results": invalid_results,
            },
        )
        call: ModelToolCall | None = None
        try:
            call = selector.select_tool(
                available_tools=available_tools,
                tool_definitions=get_tool_definitions(),
                state_summary=current_summary,
                tracer=tracer,
                previous_validation_results=invalid_results,
            )
            validation = _validate_and_prepare_call(call, state)
            if not validation["valid"]:
                decision = _decision_record(
                    call,
                    state,
                    validation,
                    phase=decision_phase,
                )
                state.setdefault("agent_decisions", []).append(decision)
                invalid_results.append(validation)
                tracer.end_span(
                    span_id,
                    output={
                        "decision": "retry_tool_selection",
                        "reason": _decision_reason(decision_phase, call.name),
                        "available_tools": available_tools,
                        "selected_tool": call.name,
                        "arguments": call.arguments,
                        "validation": validation,
                    },
                )
                if len(invalid_results) >= max_invalid_calls:
                    raise ToolExecutionError(
                        "Model produced repeated invalid tool choices: "
                        + "; ".join(item["message"] for item in invalid_results)
                    )
                continue
            tracer.end_span(
                span_id,
                output={
                    "decision": "call_tool",
                    "reason": _decision_reason(decision_phase, call.name),
                    "available_tools": available_tools,
                    "selected_tool": call.name,
                    "arguments": _jsonable(validation["registry_arguments"]),
                },
            )
            if call.name == "fit_analysis":
                batch = _dispatch_fit_analysis_batch(
                    call,
                    state,
                    tracer=tracer,
                )
                for batch_call, arguments, result in batch:
                    _apply_tool_result(state, batch_call, result, arguments)
                _publish_progress(progress_callback, state)
                result_summary = {
                    **validation,
                    "job_ids": [
                        _argument_job_id(arguments)
                        for _batch_call, arguments, _result in batch
                    ],
                    "tool_results": [
                        _tool_result_summary(result)
                        for _batch_call, _arguments, result in batch
                    ],
                }
            else:
                result = _dispatch_validated_call(
                    call,
                    validation["registry_arguments"],
                    tracer=tracer,
                )
                _apply_tool_result(
                    state, call, result, validation["registry_arguments"]
                )
                _publish_progress(progress_callback, state)
                result_summary = {
                    **validation,
                    "tool_result": _tool_result_summary(result),
                }
            decision = _decision_record(
                call,
                state,
                result_summary,
                phase=decision_phase,
            )
            state.setdefault("agent_decisions", []).append(decision)
            invalid_results = []
        except Exception as exc:
            tracer.end_span(
                span_id,
                status="ERROR",
                error_type=exc.__class__.__name__,
                output={
                    "decision": "tool_selection_failed",
                    "reason": _decision_reason(
                        decision_phase, call.name if call else None
                    ),
                    "available_tools": available_tools,
                    "selected_tool": call.name if call else None,
                    "arguments": call.arguments if call else {},
                    "validation": {
                        "valid": False,
                        "message": str(exc),
                        "error_type": exc.__class__.__name__,
                    },
                },
            )
            raise
    raise ToolExecutionError(
        f"Agent loop exceeded {max_iterations} iterations without reaching "
        f"the stop condition."
    )


def _publish_progress(
    callback: Callable[[AgentState], None] | None,
    state: AgentState,
) -> None:
    """Publish a best-effort state snapshot without affecting the workflow."""

    if callback is None:
        return
    try:
        callback(state)
    except Exception:
        logger.exception("Progress callback failed; workflow execution continues.")


def _decision_span_name(phase: str | None) -> str:
    return {
        Phase.FILTER.value: "Decision: Filtering",
        Phase.SCORE.value: "Decision: Scoring",
        Phase.FIT_ANALYSIS.value: "Decision: Fit Analysis",
        Phase.TAILOR.value: "Decision: Resume Tailoring",
        Phase.REVISION.value: "Decision: Resume Tailoring",
        Phase.COVER_LETTERS.value: "Decision: Cover Letter",
    }.get(str(phase), "Decision: Workflow")


def _workflow_stage_label(phase: str | None) -> str:
    return {
        Phase.FILTER.value: "Filtering",
        Phase.SCORE.value: "Scoring",
        Phase.FIT_ANALYSIS.value: "Fit Analysis",
        Phase.TAILOR.value: "Resume Tailoring",
        Phase.REVISION.value: "Resume Tailoring",
        Phase.COVER_LETTERS.value: "Cover Letter",
    }.get(str(phase), "Workflow")


def _tool_span_name(tool_name: str) -> str:
    return {
        "filtering": "Filtering Tool",
        "scoring": "Scoring Tool",
        "fit_analysis": "Fit Analysis Tool",
        "resume_tailoring": "Resume Tailoring Tool",
        "cover_letter": "Cover Letter Tool",
    }.get(tool_name, f"{tool_name.replace('_', ' ').title()} Tool")


def _decision_reason(phase: str | None, tool_name: str | None) -> str:
    stage = _workflow_stage_label(phase)
    selected = _tool_span_name(tool_name) if tool_name else "the next assignment tool"
    return (
        f"The workflow is currently in the {str(phase or 'WORKFLOW')} phase. "
        f"I need to call {selected} to complete the {stage} stage."
    )


def _dispatch_validated_call(
    call: ModelToolCall,
    arguments: BaseModel,
    *,
    tracer: TraceManager,
) -> BaseModel:
    tool_name = _tool_span_name(call.name)
    serialized_arguments = arguments.model_dump(mode="json")
    span_id = tracer.start_span(
        tool_name,
        metadata={
            "selected_tool": call.name,
            "job_id": _argument_job_id(arguments),
            "argument_type": type(arguments).__name__,
        },
        input=serialized_arguments,
    )
    try:
        result = invoke_tool(call.name, arguments, context={"tracer": tracer})
    except Exception as exc:
        tracer.end_span(
            span_id,
            status="ERROR",
            error_type=exc.__class__.__name__,
            output={"error": str(exc)},
        )
        raise
    tracer.end_span(
        span_id,
        metadata={
            "output_type": type(result).__name__,
            "status": getattr(result, "status", "OK"),
        },
        output=_assignment_tool_output(call.name, result),
    )
    return result


def _assignment_tool_output(tool_name: str, result: BaseModel) -> dict[str, Any]:
    """Present one result in assignment language while retaining the full contract."""

    payload = result.model_dump(mode="json")
    if tool_name == "filtering":
        rejected = payload.get("rejected_jobs", [])
        return {
            **payload,
            "remaining_jobs": payload.get("accepted_jobs", []),
            "rejection_reasons": [
                {
                    "job_id": item.get("job", {}).get("job_id"),
                    "reasons": item.get("reasons", []),
                }
                for item in rejected
            ],
        }
    if tool_name == "scoring":
        top_ids = set(payload.get("top_3_job_ids", []))
        return {
            **payload,
            "top_3": [
                item
                for item in payload.get("ranked_jobs", [])
                if item.get("job", {}).get("job_id") in top_ids
            ],
        }
    if tool_name in {"resume_tailoring", "cover_letter"}:
        source_path = payload.get("output_tex_path")
        generated_text: str | None = None
        if source_path:
            try:
                generated_text = Path(source_path).read_text(encoding="utf-8")
            except OSError:
                generated_text = None
        content_key = (
            "tailored_resume" if tool_name == "resume_tailoring" else "cover_letter"
        )
        return {
            **payload,
            content_key: generated_text,
            "pdf_path": payload.get("output_pdf_path"),
            "page_validation": {
                "page_count": payload.get("page_count"),
                "exactly_one_page": payload.get("page_count") == 1,
                "errors": [
                    *payload.get("validation_failures", []),
                    *payload.get("errors", []),
                ],
            },
        }
    return payload


def _dispatch_fit_analysis_batch(
    selected_call: ModelToolCall,
    state: AgentState,
    *,
    tracer: TraceManager,
) -> list[tuple[ModelToolCall, BaseModel, BaseModel]]:
    """Run all remaining Top-3 fit analyses concurrently, capped at three."""

    completed = set(state.get("fit_analyses", {}))
    remaining_job_ids = [
        job_id for job_id in state.get("top_3_job_ids", []) if job_id not in completed
    ]
    selected_job_id = _selected_job_id(selected_call)
    if selected_job_id not in remaining_job_ids:
        raise ToolExecutionError(
            f"fit_analysis is not pending for selected job {selected_job_id!r}."
        )

    calls_and_arguments: list[tuple[ModelToolCall, BaseModel]] = []
    for job_id in remaining_job_ids:
        call = (
            selected_call
            if job_id == selected_job_id
            else ModelToolCall(
                name="fit_analysis",
                arguments={"job_id": job_id},
                rationale="Run the remaining Top-3 fit analyses in the same batch.",
            )
        )
        validation = _validate_and_prepare_call(call, state)
        if not validation["valid"]:
            raise ToolExecutionError(str(validation["message"]))
        calls_and_arguments.append((call, validation["registry_arguments"]))

    span_id = tracer.start_span(
        "Fit Analysis Tool",
        metadata={
            "selected_tool": "fit_analysis",
            "job_count": len(calls_and_arguments),
        },
        input={
            "top_3_jobs": [
                arguments.model_dump(mode="json")
                for _call, arguments in calls_and_arguments
            ]
        },
    )
    try:
        with ThreadPoolExecutor(
            max_workers=min(3, len(calls_and_arguments)),
            thread_name_prefix="fit-analysis",
        ) as executor:
            futures = [
                executor.submit(
                    copy_context().run,
                    invoke_tool,
                    call.name,
                    arguments,
                    {"tracer": tracer},
                )
                for call, arguments in calls_and_arguments
            ]
            # Resolve in Top-3 order so state, history, and artifact ordering
            # remain deterministic even though expensive work finishes out of
            # order.
            results = [
                (call, arguments, future.result())
                for (call, arguments), future in zip(
                    calls_and_arguments,
                    futures,
                    strict=True,
                )
            ]
    except Exception as exc:
        tracer.end_span(
            span_id,
            status="ERROR",
            error_type=exc.__class__.__name__,
            output={"error": str(exc)},
        )
        raise
    tracer.end_span(
        span_id,
        metadata={"output_count": len(results)},
        output={
            "fit_analyses": [
                result.model_dump(mode="json") for _call, _arguments, result in results
            ],
            "project_swap_suggestions": [
                result.project_swap.model_dump(mode="json")
                if result.project_swap
                else None
                for _call, _arguments, result in results
            ],
        },
    )
    return results


def _validate_and_prepare_call(
    call: ModelToolCall | Mapping[str, Any],
    state: AgentState,
) -> dict[str, Any]:
    try:
        parsed = validate_model_tool_call(
            call,
            available_tools=_available_tool_names(),
        )
    except ToolExecutionError as exc:
        return _invalid_result(str(exc), tool_name=getattr(call, "name", None))

    try:
        arguments = _registry_arguments_for_call(parsed, state)
    except ToolExecutionError as exc:
        return _invalid_result(str(exc), tool_name=parsed.name)
    return {
        "valid": True,
        "message": "Accepted by workflow state validators.",
        "selected_tool": parsed.name,
        "job_id": _argument_job_id(arguments),
        "registry_arguments": arguments,
    }


def _registry_arguments_for_call(
    call: ModelToolCall,
    state: AgentState,
) -> BaseModel:
    tool_name = call.name
    profile = CandidateProfile.model_validate(state_value(state, "candidate_profile"))
    portfolio = Portfolio.model_validate(state_value(state, "portfolio"))
    jobs = [Job.model_validate(job) for job in state_value(state, "jobs")]
    jobs_by_id = {job.job_id: job for job in jobs}
    candidate_evidence = _candidate_evidence(state, profile, portfolio)
    top_job_ids = list(state.get("top_3_job_ids", []))

    if tool_name == "filtering":
        if state.get("filtered_jobs"):
            raise ToolExecutionError("filtering has already completed.")
        return FilterJobsInput(jobs=jobs, preferences=profile.preferences)

    if tool_name == "scoring":
        if not state.get("filtered_jobs"):
            raise ToolExecutionError("scoring requires accepted jobs from filtering.")
        if state.get("ranked_jobs"):
            raise ToolExecutionError("scoring has already completed.")
        filtered_jobs = [
            Job.model_validate(job) for job in state_value(state, "filtered_jobs")
        ]
        if len(filtered_jobs) < 3:
            raise ToolExecutionError(
                "Filtering must accept at least three jobs before scoring."
            )
        return ScoreJobsInput(
            jobs=filtered_jobs,
            candidate_profile=profile,
            resume_evidence=profile.resume_evidence,
            master_skill_evidence=profile.master_skill_evidence,
            portfolio_evidence=[*portfolio.evidence_items, *profile.portfolio_evidence],
            memory_evidence=_memory_evidence(state),
        )

    if tool_name == "fit_analysis":
        job_id = _selected_job_id(call)
        if not top_job_ids:
            raise ToolExecutionError(
                "fit_analysis requires top_3_job_ids from scoring."
            )
        if state.get("tailoring_results"):
            raise ToolExecutionError(
                "fit_analysis cannot run after resume tailoring has started."
            )
        if job_id not in top_job_ids:
            raise ToolExecutionError(
                f"fit_analysis job_id must be one of {top_job_ids}; received {job_id!r}."
            )
        if job_id in state.get("fit_analyses", {}):
            raise ToolExecutionError(f"fit_analysis already completed for {job_id}.")
        job = jobs_by_id[job_id]
        return AnalyzeFitInput(
            job=job,
            candidate_profile=profile,
            evidence_items=candidate_evidence,
            job_evidence=build_job_evidence(job),
            current_resume_projects=profile.resume_projects,
            portfolio_projects=portfolio.projects,
        )

    if tool_name == "resume_tailoring":
        job_id = _selected_job_id(call)
        if not top_job_ids:
            raise ToolExecutionError(
                "resume_tailoring requires top_3_job_ids from scoring."
            )
        if job_id not in top_job_ids:
            raise ToolExecutionError(
                f"resume_tailoring job_id must be one of {top_job_ids}; "
                f"received {job_id!r}."
            )
        fit_analyses = state.get("fit_analyses", {})
        missing_fit = [job_id for job_id in top_job_ids if job_id not in fit_analyses]
        if missing_fit:
            raise ToolExecutionError(
                "resume_tailoring requires fit_analysis for every top job first; "
                f"missing={missing_fit}."
            )
        review_decisions = state.get("review_decisions", {})
        rejected_job_ids = _rejected_job_ids(state)
        if not review_decisions:
            if job_id in state.get("tailoring_results", {}):
                raise ToolExecutionError(
                    f"resume_tailoring already completed for {job_id}."
                )
            return TailorResumeInput(
                job=jobs_by_id[job_id],
                fit_analysis=FitAnalysisOutput.model_validate(fit_analyses[job_id]),
                source_resume_tex_path=state_value(state, "resume_path"),
                candidate_evidence=candidate_evidence,
                job_evidence=build_job_evidence(jobs_by_id[job_id]),
                run_id=state_value(state, "run_id"),
            )
        if job_id not in rejected_job_ids:
            propagation_ids = _pending_memory_propagation_job_ids(state)
            if job_id not in propagation_ids:
                raise ToolExecutionError(
                    "After review, resume_tailoring is only valid for rejected "
                    "jobs needing revision or approved jobs with relevant new "
                    f"memory; rejected={rejected_job_ids}; "
                    f"memory_propagation={propagation_ids}."
                )
            feedback_text = str(
                state.get("memory_propagation_feedback_by_job", {}).get(job_id, "")
            )
            return TailorResumeInput(
                job=jobs_by_id[job_id],
                fit_analysis=FitAnalysisOutput.model_validate(fit_analyses[job_id]),
                source_resume_tex_path=state_value(state, "resume_path"),
                candidate_evidence=candidate_evidence,
                job_evidence=build_job_evidence(jobs_by_id[job_id]),
                revision_feedback=feedback_text,
                run_id=state_value(state, "run_id"),
            )
        attempts = _revision_attempts(state, job_id)
        latest = state.get("tailoring_results", {}).get(job_id, {})
        if latest.get("revision_feedback_satisfied") is True:
            raise ToolExecutionError(
                f"Revision feedback is already satisfied for {job_id}."
            )
        if attempts >= MAX_REVISION_ROUNDS:
            raise ToolExecutionError(
                f"Maximum revision rounds exceeded for {job_id}: {MAX_REVISION_ROUNDS}."
            )
        feedback_text = _combined_review_and_memory_feedback(state, job_id)
        return TailorResumeInput(
            job=jobs_by_id[job_id],
            fit_analysis=FitAnalysisOutput.model_validate(fit_analyses[job_id]),
            source_resume_tex_path=state_value(state, "resume_path"),
            candidate_evidence=candidate_evidence,
            job_evidence=build_job_evidence(jobs_by_id[job_id]),
            revision_feedback=feedback_text,
            run_id=state_value(state, "run_id"),
        )

    if tool_name == "cover_letter":
        job_id = _selected_job_id(call)
        if not state.get("review_decisions"):
            raise ToolExecutionError(
                "cover_letter cannot run before the human review approval gate."
            )
        if _pending_revision_job_ids(state):
            raise ToolExecutionError(
                "cover_letter cannot run while rejected resumes still need revision."
            )
        if job_id not in state.get("approved_job_ids", []):
            raise ToolExecutionError(
                f"cover_letter job_id must be approved; received {job_id!r}."
            )
        if job_id in state.get("cover_letter_results", {}):
            raise ToolExecutionError(f"cover_letter already completed for {job_id}.")
        tailoring = state_value(state, "tailoring_results")
        return GenerateCoverLetterInput(
            job=jobs_by_id[job_id],
            approved_resume_path=tailoring[job_id]["output_pdf_path"],
            candidate_evidence=candidate_evidence,
            job_evidence=build_job_evidence(jobs_by_id[job_id]),
            run_id=state_value(state, "run_id"),
        )

    raise ToolExecutionError(f"No state validator exists for tool {tool_name}.")


def _apply_tool_result(
    state: AgentState,
    call: ModelToolCall,
    result: BaseModel,
    arguments: BaseModel,
) -> None:
    tool = get_tool(call.name)
    state.setdefault("tool_history", []).append(_history(tool.callable_name, result))
    if call.name == "filtering":
        output = FilterJobsOutput.model_validate(result)
        state["filtered_jobs"] = [job.model_dump() for job in output.accepted_jobs]
        state["rejected_jobs"] = [
            rejected.model_dump() for rejected in output.rejected_jobs
        ]
        state["phase"] = Phase.SCORE.value
        if len(output.accepted_jobs) < 3:
            raise ToolExecutionError(
                "Filtering must accept at least three jobs before scoring."
            )
        return
    if call.name == "scoring":
        output = ScoreJobsOutput.model_validate(result)
        if len(output.top_3_job_ids) != 3:
            raise ToolExecutionError("Scoring must select exactly three jobs.")
        state["ranked_jobs"] = [job.model_dump() for job in output.ranked_jobs]
        state["top_3_job_ids"] = output.top_3_job_ids
        state["phase"] = Phase.FIT_ANALYSIS.value
        return
    if call.name == "fit_analysis":
        output = FitAnalysisOutput.model_validate(result)
        state.setdefault("fit_analyses", {})[output.job_id] = output.model_dump()
        markdown_path, json_path = write_fit_analysis(
            output,
            Job.model_validate(getattr(arguments, "job")),
            run_id=state_value(state, "run_id"),
            source_labels=build_source_labels(
                AnalyzeFitInput.model_validate(arguments)
            ),
        )
        state.setdefault("fit_analysis_artifacts", {})[output.job_id] = {
            "markdown_path": str(markdown_path),
            "json_path": str(json_path),
        }
        if _all_top_jobs_have(state, "fit_analyses"):
            state["phase"] = Phase.TAILOR.value
        return
    if call.name == "resume_tailoring":
        output = TailorResumeOutput.model_validate(result)
        validate_artifact_output(tool.callable_name, output)
        state.setdefault("tailoring_results", {})[output.job_id] = output.model_dump()
        if state.get("review_decisions") and output.job_id in _rejected_job_ids(state):
            _record_revision_result(state, output)
        elif output.job_id in state.get("memory_propagation_job_ids", []):
            _record_memory_propagation_result(state, output)
        elif _ready_for_human_review(state):
            state["phase"] = Phase.HUMAN_REVIEW.value
        return
    if call.name == "cover_letter":
        output = GenerateCoverLetterOutput.model_validate(result)
        validate_artifact_output(tool.callable_name, output)
        state.setdefault("cover_letter_results", {})[output.job_id] = (
            output.model_dump()
        )
        if _all_top_jobs_have(state, "cover_letter_results"):
            state["phase"] = Phase.COMPLETE.value
            state["status"] = RunStatus.COMPLETED.value


def _record_revision_result(state: AgentState, output: TailorResumeOutput) -> None:
    job_id = output.job_id
    attempts_by_job = dict(state.get("revision_attempts_by_job", {}))
    revision_round = int(attempts_by_job.get(job_id, 0)) + 1
    attempts_by_job[job_id] = revision_round
    state["revision_attempts_by_job"] = attempts_by_job
    feedback_text = str(
        state.get("review_decisions", {}).get(job_id, {}).get("comment", "")
    )
    round_logs = state.setdefault("revision_round_logs", {})
    round_log = round_logs.setdefault(
        revision_round,
        {
            "review_round": 1,
            "revision_round": revision_round,
            "feedback_received_by_job": {},
            "actions": [],
        },
    )
    round_log["feedback_received_by_job"][job_id] = feedback_text
    round_log["actions"].append(
        _revision_attempt_log(
            job_id=job_id,
            revision_round=revision_round,
            feedback_text=feedback_text,
            output=output,
        )
    )
    state.setdefault("review_actions", {})[job_id] = {
        "revision_round": revision_round,
        "status": output.status,
        "change_log": [change.model_dump() for change in output.change_log],
        "output_tex_path": output.output_tex_path,
        "output_pdf_path": output.output_pdf_path,
    }
    propagation_action = _memory_propagation_action(state, job_id)
    if propagation_action.get("status") == "handled_by_rejected_revision":
        propagation_action.update(
            {
                "status": (
                    "applied_via_rejected_revision"
                    if output.status == "OK"
                    else "failed_via_rejected_revision"
                ),
                "output_tex_path": output.output_tex_path,
                "output_pdf_path": output.output_pdf_path,
                "change_log": [change.model_dump() for change in output.change_log],
            }
        )
    if output.revision_feedback_satisfied is True:
        state["pending_revision_job_ids"] = [
            item for item in state.get("pending_revision_job_ids", []) if item != job_id
        ]
        if not _pending_revision_job_ids(state):
            state["phase"] = Phase.COVER_LETTERS.value
        return
    if revision_round >= MAX_REVISION_ROUNDS:
        state["status"] = RunStatus.FAILED_REVIEW.value
        state["phase"] = Phase.HUMAN_REVIEW.value
        state.setdefault("errors", []).append(
            {
                "phase": Phase.HUMAN_REVIEW.value,
                "type": "FailedReview",
                "message": (
                    f"Feedback for {job_id} remained unsatisfied after "
                    f"{MAX_REVISION_ROUNDS} revision rounds."
                ),
            }
        )


def _record_memory_propagation_result(
    state: AgentState, output: TailorResumeOutput
) -> None:
    job_id = output.job_id
    feedback_text = str(
        state.get("memory_propagation_feedback_by_job", {}).get(job_id, "")
    )
    action = _memory_propagation_action(state, job_id)
    action.update(
        {
            "status": "applied" if output.status == "OK" else "failed",
            "output_tex_path": output.output_tex_path,
            "output_pdf_path": output.output_pdf_path,
            "change_log": [change.model_dump() for change in output.change_log],
        }
    )
    round_logs = state.setdefault("revision_round_logs", {})
    round_log = round_logs.setdefault(
        "memory_propagation",
        {
            "review_round": 1,
            "revision_round": 0,
            "feedback_received_by_job": {},
            "actions": [],
            "kind": "memory_propagation",
        },
    )
    round_log["feedback_received_by_job"][job_id] = feedback_text
    round_log["actions"].append(
        {
            "review_round": 1,
            "revision_round": 0,
            "job_id": job_id,
            "feedback_received": feedback_text,
            "actions_taken": [
                "Reran resume_tailoring to propagate relevant same-run memory.",
                "No additional human review pause was requested.",
            ],
            "changes_accepted": [change.model_dump() for change in output.change_log],
            "changes_rejected_or_skipped": [
                *output.validation_failures,
                *output.errors,
            ],
            "evidence_used": sorted(
                {
                    evidence_id
                    for change in output.change_log
                    for evidence_id in change.evidence_ids
                }
            ),
            "status": output.status,
            "feedback_satisfied": output.revision_feedback_satisfied,
            "feedback_checks": output.revision_feedback_checks,
            "output_tex_path": output.output_tex_path,
            "output_pdf_path": output.output_pdf_path,
        }
    )
    state["memory_propagation_job_ids"] = [
        item for item in state.get("memory_propagation_job_ids", []) if item != job_id
    ]
    if not _pending_revision_job_ids(state):
        state["phase"] = Phase.COVER_LETTERS.value


def _ready_for_human_review(state: AgentState) -> bool:
    return bool(state.get("top_3_job_ids")) and _all_top_jobs_have(
        state, "tailoring_results"
    )


def _post_review_done(state: AgentState) -> bool:
    if state.get("status") == RunStatus.FAILED_REVIEW.value:
        return True
    return bool(state.get("review_decisions")) and _all_top_jobs_have(
        state, "cover_letter_results"
    )


def _decision_state_summary(state: AgentState) -> dict[str, Any]:
    top_job_ids = list(state.get("top_3_job_ids", []))
    fit_done = sorted(state.get("fit_analyses", {}))
    tailoring_done = sorted(state.get("tailoring_results", {}))
    cover_done = sorted(state.get("cover_letter_results", {}))
    return {
        "run_id": state.get("run_id"),
        "phase": state.get("phase"),
        "status": state.get("status"),
        "job_count": len(state.get("jobs", [])),
        "filtered_job_count": len(state.get("filtered_jobs", [])),
        "ranked_job_count": len(state.get("ranked_jobs", [])),
        "top_3_job_ids": top_job_ids,
        "fit_analysis_completed_job_ids": fit_done,
        "fit_analysis_remaining_job_ids": [
            job_id for job_id in top_job_ids if job_id not in fit_done
        ],
        "tailoring_completed_job_ids": tailoring_done,
        "tailoring_remaining_job_ids": [
            job_id for job_id in top_job_ids if job_id not in tailoring_done
        ],
        "review_decisions_present": bool(state.get("review_decisions")),
        "rejected_job_ids": _rejected_job_ids(state),
        "pending_revision_job_ids": _pending_revision_job_ids(state),
        "revision_attempts_by_job": dict(state.get("revision_attempts_by_job", {})),
        "max_revision_rounds": MAX_REVISION_ROUNDS,
        "cover_letter_completed_job_ids": cover_done,
        "cover_letter_remaining_job_ids": [
            job_id for job_id in top_job_ids if job_id not in cover_done
        ],
    }


def _available_tool_names() -> list[str]:
    return [tool.name for tool in get_registered_tools()]


def _tool_definition_summary() -> list[dict[str, str]]:
    return [
        {"name": tool.name, "description": tool.description}
        for tool in get_registered_tools()
    ]


def _invalid_result(message: str, *, tool_name: str | None = None) -> dict[str, Any]:
    return {
        "valid": False,
        "selected_tool": tool_name,
        "message": message,
    }


def _decision_record(
    call: ModelToolCall,
    state: AgentState,
    result: Mapping[str, Any],
    *,
    phase: str | None = None,
) -> dict[str, Any]:
    return {
        "phase": phase if phase is not None else state.get("phase"),
        "available_tools": _available_tool_names(),
        "selected_tool": call.name,
        "arguments": call.arguments,
        "job_id": result.get("job_id") or call.arguments.get("job_id"),
        "rationale": call.rationale,
        "result": _decision_result_summary(result),
    }


def _selected_job_id(call: ModelToolCall) -> str:
    value = call.arguments.get("job_id")
    if not value:
        raise ToolExecutionError(f"{call.name} requires a job_id selector argument.")
    return str(value)


def _all_top_jobs_have(state: AgentState, key: str) -> bool:
    top_job_ids = list(state.get("top_3_job_ids", []))
    values = state.get(key, {})
    return bool(top_job_ids) and all(job_id in values for job_id in top_job_ids)


def _rejected_job_ids(state: AgentState) -> list[str]:
    decisions = state.get("review_decisions", {})
    return [
        job_id
        for job_id, decision in decisions.items()
        if decision.get("decision") == "reject"
    ]


def _pending_revision_job_ids(state: AgentState) -> list[str]:
    pending: list[str] = []
    tailoring = state.get("tailoring_results", {})
    for job_id in _rejected_job_ids(state):
        latest = tailoring.get(job_id, {})
        attempts = _revision_attempts(state, job_id)
        if latest.get("revision_feedback_satisfied") is True:
            continue
        if attempts >= MAX_REVISION_ROUNDS:
            pending.append(job_id)
            continue
        pending.append(job_id)
    pending.extend(_pending_memory_propagation_job_ids(state))
    return pending


def _pending_memory_propagation_job_ids(state: AgentState) -> list[str]:
    tailoring = state.get("tailoring_results", {})
    pending: list[str] = []
    for job_id in state.get("memory_propagation_job_ids", []):
        action = _memory_propagation_action(state, job_id)
        if action.get("status") != "pending_tailoring":
            continue
        latest = tailoring.get(job_id, {})
        memory_ids = set(action.get("memory_fact_ids", []))
        if memory_ids and any(
            memory_ids & set(change.get("evidence_ids", []))
            for change in latest.get("change_log", [])
        ):
            action["status"] = "already_applied"
            continue
        pending.append(job_id)
    return pending


def _revision_attempts(state: AgentState, job_id: str) -> int:
    return int(state.get("revision_attempts_by_job", {}).get(job_id, 0))


def _tool_result_summary(result: BaseModel) -> dict[str, Any]:
    payload = result.model_dump()
    summary: dict[str, Any] = {
        "output_type": type(result).__name__,
        "status": payload.get("status", "OK"),
    }
    if payload.get("job_id"):
        summary["job_id"] = payload["job_id"]
    if "top_3_job_ids" in payload:
        summary["top_3_job_ids"] = payload["top_3_job_ids"]
    if "accepted_jobs" in payload:
        summary["accepted_job_count"] = len(payload["accepted_jobs"])
    if "ranked_jobs" in payload:
        summary["ranked_job_count"] = len(payload["ranked_jobs"])
    if "revision_feedback_satisfied" in payload:
        summary["revision_feedback_satisfied"] = payload["revision_feedback_satisfied"]
    return summary


def _decision_result_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    summary = {
        key: value for key, value in dict(result).items() if key != "registry_arguments"
    }
    arguments = result.get("registry_arguments")
    if isinstance(arguments, BaseModel):
        payload = arguments.model_dump(mode="json")
        summary["registry_argument_summary"] = {
            "argument_type": type(arguments).__name__,
            "argument_keys": sorted(payload),
            "job_id": _argument_job_id(arguments),
        }
    return _jsonable(summary)


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    return value


def _argument_job_id(arguments: BaseModel | dict[str, Any]) -> str | None:
    if isinstance(arguments, BaseModel):
        job = getattr(arguments, "job", None)
    else:
        job = arguments.get("job")
    if isinstance(job, BaseModel):
        value = getattr(job, "job_id", None)
    elif isinstance(job, dict):
        value = job.get("job_id")
    else:
        value = None
    return str(value) if value else None


def _candidate_evidence(
    state: AgentState,
    profile: CandidateProfile,
    portfolio: Portfolio,
) -> list[EvidenceItem]:
    return [
        *profile.resume_evidence,
        *profile.master_skill_evidence,
        *portfolio.evidence_items,
        *profile.portfolio_evidence,
        *_memory_evidence(state),
    ]


def _memory_evidence(state: AgentState) -> list[EvidenceItem]:
    return [
        EvidenceItem.model_validate(memory_fact_to_evidence(fact))
        for raw in state.get("memory_facts", [])
        if (fact := MemoryFact.model_validate(raw)).active
        and fact.fact_type != "resume_feedback"
    ]


def _store_review_memory(
    state: AgentState,
    decisions: dict[str, dict[str, Any]],
    *,
    tracer: TraceManager,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    span_id = tracer.start_span(
        "Memory Update Tool",
        metadata={"memory_file": state_value(state, "memory_file")},
        input={
            "review_comments": {
                job_id: decision.get("comment", "")
                for job_id, decision in decisions.items()
            },
            "reviewed_job_ids": sorted(decisions),
        },
    )
    comments = {
        job_id: decision["comment"]
        for job_id, decision in decisions.items()
        if decision["comment"].strip()
    }
    try:
        review_round = int(state.get("interrupt_payload", {}).get("review_round", 1))
        extracted = extract_memory_facts(comments, review_round=review_round)
        valid, failures = validate_memory_facts(extracted, comments)
        store = JSONMemoryStore(state_value(state, "memory_file"))
        before = store.load()
        before_keys = {fact.deduplication_key for fact in before if fact.active}
        updated = store.append_many(valid, run_id=state_value(state, "run_id"))
        new_ids = [
            fact.fact_id
            for fact in updated
            if fact.active and fact.deduplication_key not in before_keys
        ]
        conflict_events = list(
            {
                conflict.conflict_id: conflict.model_dump()
                for fact in updated
                for conflict in fact.conflicts
            }.values()
        )
    except Exception as exc:
        tracer.end_span(
            span_id,
            status="ERROR",
            error_type=exc.__class__.__name__,
            output={"error": str(exc)},
        )
        raise
    tracer.end_span(
        span_id,
        output={
            "comments_with_content": len(comments),
            "valid_fact_count": len(valid),
            "new_fact_ids": new_ids,
            "new_memory_fact": [
                fact.model_dump()
                for fact in updated
                if fact.active and fact.fact_id in set(new_ids)
            ],
            "memory_facts_written": [
                fact.model_dump()
                for fact in updated
                if fact.active and fact.fact_id in set(new_ids)
            ],
            "validation_failures": failures,
            "conflict_count": len(conflict_events),
            "conflicts": conflict_events,
            "active_values": [fact.canonical_value for fact in updated if fact.active],
            "memory_file_updated": state_value(state, "memory_file"),
        },
    )
    return [fact.model_dump() for fact in updated], new_ids, failures


def _plan_memory_propagation(
    state: AgentState,
    decisions: dict[str, dict[str, Any]],
    memory_facts: list[dict[str, Any]],
    new_fact_ids: list[str],
    *,
    tracer: TraceManager,
) -> list[dict[str, Any]]:
    try:
        facts = [
            MemoryFact.model_validate(raw)
            for raw in memory_facts
            if raw.get("fact_id") in set(new_fact_ids) and raw.get("active", True)
        ]
        jobs = {
            raw["job_id"]: Job.model_validate(raw)
            for raw in state.get("jobs", [])
            if raw.get("job_id") in set(state.get("top_3_job_ids", []))
        }
        actions: list[dict[str, Any]] = []
        for job_id in state.get("top_3_job_ids", []):
            relevant = _relevant_memory_facts_for_job(facts, jobs[job_id])
            feedback = _memory_feedback(relevant)
            if not relevant:
                status = "not_relevant"
            elif decisions.get(job_id, {}).get("decision") == "reject":
                status = "handled_by_rejected_revision"
            else:
                status = "pending_tailoring"
            actions.append(
                {
                    "job_id": job_id,
                    "status": status,
                    "memory_fact_ids": [fact.fact_id for fact in relevant],
                    "memory_facts": [
                        {
                            "fact_id": fact.fact_id,
                            "fact_type": fact.fact_type,
                            "canonical_value": fact.canonical_value,
                            "provenance": fact.provenance.model_dump(),
                        }
                        for fact in relevant
                    ],
                    "feedback": feedback,
                    "reason": _memory_propagation_reason(status),
                }
            )
        return actions
    except Exception:
        raise


def _relevant_memory_facts_for_job(
    facts: list[MemoryFact], job: Job
) -> list[MemoryFact]:
    job_evidence = build_job_evidence(job)
    relevant: list[MemoryFact] = []
    for fact in facts:
        if fact.fact_type != "skill":
            continue
        if any(
            job_evidence_supports_skill(item, fact.canonical_value)
            for item in job_evidence
        ):
            relevant.append(fact)
    return relevant


def _memory_feedback(facts: list[MemoryFact]) -> str:
    if not facts:
        return ""
    parts = [f"{fact.canonical_value} ({fact.fact_id})" for fact in facts]
    return "Apply newly learned, reviewer-stated fact(s): " + "; ".join(parts) + "."


def _memory_propagation_reason(status: str) -> str:
    return {
        "pending_tailoring": (
            "New reviewer memory matches this job's posted requirements and can be "
            "applied through the allowed tailoring sections."
        ),
        "handled_by_rejected_revision": (
            "The job already needs a rejected-resume revision; relevant memory is "
            "included in that revision feedback."
        ),
        "not_relevant": (
            "No newly written memory fact matches this job's posted requirements."
        ),
    }.get(status, status)


def _combined_review_and_memory_feedback(state: AgentState, job_id: str) -> str:
    pieces = [
        str(state.get("review_decisions", {}).get(job_id, {}).get("comment", "")),
        str(state.get("memory_propagation_feedback_by_job", {}).get(job_id, "")),
    ]
    return " ".join(piece.strip() for piece in pieces if piece.strip())


def _memory_propagation_action(state: AgentState, job_id: str) -> dict[str, Any]:
    actions = state.setdefault("memory_propagation_actions", [])
    for action in actions:
        if action.get("job_id") == job_id:
            return action
    action = {
        "job_id": job_id,
        "status": "not_planned",
        "memory_fact_ids": [],
        "memory_facts": [],
        "feedback": "",
        "reason": "No propagation action was planned for this job.",
    }
    actions.append(action)
    return action


def _final_artifact_paths(
    manifest: dict[str, Any],
    tailoring_results: Mapping[str, Any],
    cover_letter_results: Mapping[str, Any],
) -> dict[str, dict[str, str]]:
    paths: dict[str, dict[str, str]] = {}
    for job_id, job_dir in zip(
        manifest.get("job_ids", []),
        manifest.get("job_directories", []),
        strict=False,
    ):
        paths[job_id] = {
            "job_directory": str(job_dir),
            "resume_after_pdf": str(
                tailoring_results.get(job_id, {}).get("output_pdf_path", "")
            ),
            "cover_letter_pdf": str(
                cover_letter_results.get(job_id, {}).get("output_pdf_path", "")
            ),
        }
    return paths


def _revision_attempt_log(
    *,
    job_id: str,
    revision_round: int,
    feedback_text: str,
    output: TailorResumeOutput,
) -> dict[str, Any]:
    """Return the per-attempt audit record required by human review."""

    changes = [change.model_dump() for change in output.change_log]
    evidence_used = sorted(
        {
            evidence_id
            for change in output.change_log
            for evidence_id in change.evidence_ids
        }
    )
    rejected_or_skipped = [
        *output.validation_failures,
        *output.errors,
        *[
            check
            for check in output.revision_feedback_checks
            if "not met" in check.casefold()
        ],
    ]
    actions_taken = [
        "Reran resume_tailoring for the rejected resume.",
        (
            "Revision feedback satisfied; revised resume accepted."
            if output.revision_feedback_satisfied is True
            else "Revision feedback not yet satisfied."
        ),
    ]
    return {
        "review_round": 1,
        "revision_round": revision_round,
        "job_id": job_id,
        "feedback_received": feedback_text,
        "actions_taken": actions_taken,
        "changes_accepted": changes if output.status == "OK" else [],
        "changes_rejected_or_skipped": list(dict.fromkeys(rejected_or_skipped)),
        "evidence_used": evidence_used,
        "status": output.status,
        "feedback_satisfied": output.revision_feedback_satisfied,
        "feedback_checks": output.revision_feedback_checks,
        "output_tex_path": output.output_tex_path,
        "output_pdf_path": output.output_pdf_path,
    }


def _history(tool_name: str, output: BaseModel) -> dict[str, Any]:
    payload = output.model_dump()
    return {
        "tool": tool_name,
        "status": payload.get("status", "OK"),
        "output": payload,
    }


def validate_artifact_output(tool_name: str, output: BaseModel) -> None:
    """Reject missing, failed, unreadable, or non-one-page tool artifacts."""

    if tool_name not in {
        "run_resume_tailoring_tool",
        "run_cover_letter_tool",
    }:
        return
    payload = output.model_dump()
    errors = payload.get("errors", [])
    if errors:
        raise ToolExecutionError(
            f"{tool_name} reported errors: {'; '.join(map(str, errors))}"
        )
    if payload.get("status") not in {None, "OK"}:
        raise ToolExecutionError(
            f"{tool_name} returned status {payload.get('status')!r}."
        )
    pdf_path = payload.get("output_pdf_path")
    tex_path = payload.get("output_tex_path")
    if not pdf_path or not Path(pdf_path).is_file():
        raise ToolExecutionError(f"{tool_name} PDF does not exist: {pdf_path}")
    if not tex_path or not Path(tex_path).is_file():
        raise ToolExecutionError(f"{tool_name} TeX file does not exist: {tex_path}")
    actual_page_count = pdf_page_count(Path(pdf_path))
    if actual_page_count != 1:
        raise ToolExecutionError(
            f"{tool_name} output PDF has {actual_page_count} pages; expected 1."
        )
