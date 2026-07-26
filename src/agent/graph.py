"""Explicit five-tool job-search workflow with one human-review pause.

You are the only LLM agent in this workflow.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel

from src.agent.errors import ToolExecutionError
from src.agent.state import AgentState, Phase, RunStatus, state_value
from src.agent.tool_selection import (
    ToolSelectionModel,
    default_tool_selection_model,
    select_validated_tool_call,
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
from src.review.human_review import build_review_payload, normalize_review_feedback
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
from src.tools.registry import get_tool, invoke_tool
from src.tracing.langfuse import TraceManager
from src.tools.registry import get_registered_tools
from src.utils.output_validation import write_and_validate_outputs

logger = logging.getLogger(__name__)

__all__ = ["build_agent_graph", "validate_artifact_output"]


def build_agent_graph(
    *,
    checkpointer: Any | None = None,
    tracer: TraceManager | None = None,
    tool_selection_model: ToolSelectionModel | None = None,
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
        load_span = trace_manager.start_span(
            "initialize.load_inputs",
            input={"input_paths": paths},
        )
        try:
            jobs = load_jobs_csv(paths.get("jobs_path", "data/jobs.csv"))
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
                load_span,
                status="ERROR",
                error_type=exc.__class__.__name__,
                output={"error": str(exc)},
            )
            raise
        trace_manager.end_span(
            load_span,
            output={
                "job_count": len(jobs),
                "portfolio_project_count": len(portfolio.projects),
                "resume_evidence_count": len(resume.evidence_items),
            },
        )
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
        memory_span = trace_manager.start_span(
            "memory.read",
            input={"memory_file": memory_file},
        )
        memory_facts = JSONMemoryStore(memory_file).load()
        trace_manager.end_span(
            memory_span,
            output={"memory_fact_count": len(memory_facts)},
        )
        trace_manager.register_personal_data(profile.name, profile.email)
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

    def run_pre_review_tools(state: AgentState) -> AgentState:
        """Run filtering, scoring, fit analysis, and resume tailoring in order."""

        profile = CandidateProfile.model_validate(
            state_value(state, "candidate_profile")
        )
        jobs = [Job.model_validate(job) for job in state_value(state, "jobs")]
        portfolio = Portfolio.model_validate(state_value(state, "portfolio"))
        candidate_evidence = _candidate_evidence(state, profile, portfolio)
        history = list(state.get("tool_history", []))
        decisions = list(state.get("agent_decisions", []))
        run_id = state_value(state, "run_id")

        filtering_input = FilterJobsInput(
            jobs=jobs,
            preferences=profile.preferences,
        )
        filtering, decision = _select_and_invoke_required_tool(
            selector,
            state,
            "filtering",
            filtering_input,
            tracer=trace_manager,
            output_type=FilterJobsOutput,
        )
        decisions.append(decision)
        history.append(_history(get_tool("filtering").callable_name, filtering))
        if len(filtering.accepted_jobs) < 3:
            raise ToolExecutionError(
                "Filtering must accept at least three jobs before scoring."
            )

        scoring_input = ScoreJobsInput(
            jobs=filtering.accepted_jobs,
            candidate_profile=profile,
            resume_evidence=profile.resume_evidence,
            master_skill_evidence=profile.master_skill_evidence,
            portfolio_evidence=[
                *portfolio.evidence_items,
                *profile.portfolio_evidence,
            ],
            memory_evidence=_memory_evidence(state),
        )
        scoring, decision = _select_and_invoke_required_tool(
            selector,
            state,
            "scoring",
            scoring_input,
            tracer=trace_manager,
            output_type=ScoreJobsOutput,
        )
        decisions.append(decision)
        history.append(_history(get_tool("scoring").callable_name, scoring))
        if len(scoring.top_3_job_ids) != 3:
            raise ToolExecutionError("Scoring must select exactly three jobs.")

        jobs_by_id = {job.job_id: job for job in jobs}
        analyses: dict[str, dict[str, Any]] = {}
        analysis_artifacts: dict[str, dict[str, str]] = {}
        for job_id in scoring.top_3_job_ids:
            job = jobs_by_id[job_id]
            fit_input = AnalyzeFitInput(
                job=job,
                candidate_profile=profile,
                evidence_items=candidate_evidence,
                job_evidence=build_job_evidence(job),
                current_resume_projects=profile.resume_projects,
                portfolio_projects=portfolio.projects,
            )
            fit, decision = _select_and_invoke_required_tool(
                selector,
                state,
                "fit_analysis",
                fit_input,
                tracer=trace_manager,
                output_type=FitAnalysisOutput,
            )
            decisions.append(decision)
            analyses[job_id] = fit.model_dump()
            history.append(_history(get_tool("fit_analysis").callable_name, fit))
            markdown_path, json_path = write_fit_analysis(
                fit,
                job,
                run_id=run_id,
                source_labels=build_source_labels(fit_input),
            )
            analysis_artifacts[job_id] = {
                "markdown_path": str(markdown_path),
                "json_path": str(json_path),
            }

        tailoring: dict[str, dict[str, Any]] = {}
        for job_id in scoring.top_3_job_ids:
            job = jobs_by_id[job_id]
            resume_input = TailorResumeInput(
                job=job,
                fit_analysis=FitAnalysisOutput.model_validate(analyses[job_id]),
                source_resume_tex_path=state_value(state, "resume_path"),
                candidate_evidence=candidate_evidence,
                job_evidence=build_job_evidence(job),
                run_id=run_id,
            )
            resume, decision = _select_and_invoke_required_tool(
                selector,
                state,
                "resume_tailoring",
                resume_input,
                tracer=trace_manager,
                output_type=TailorResumeOutput,
            )
            decisions.append(decision)
            validate_artifact_output(get_tool("resume_tailoring").callable_name, resume)
            tailoring[job_id] = resume.model_dump()
            history.append(_history(get_tool("resume_tailoring").callable_name, resume))

        return {
            "phase": Phase.HUMAN_REVIEW.value,
            "status": RunStatus.WAITING_FOR_REVIEW.value,
            "filtered_jobs": [job.model_dump() for job in filtering.accepted_jobs],
            "rejected_jobs": [
                rejected.model_dump() for rejected in filtering.rejected_jobs
            ],
            "ranked_jobs": [job.model_dump() for job in scoring.ranked_jobs],
            "top_3_job_ids": scoring.top_3_job_ids,
            "fit_analyses": analyses,
            "fit_analysis_artifacts": analysis_artifacts,
            "tailoring_results": tailoring,
            "tool_history": history,
            "agent_decisions": decisions,
        }

    def human_review(state: AgentState) -> AgentState:
        """Pause exactly once after all three resume drafts are ready."""

        if state.get("review_history"):
            raise ToolExecutionError("The workflow permits only one review pause.")
        payload = build_review_payload(dict(state))
        if not any(
            event.name == "human_review_pause" for event in trace_manager.events
        ):
            span_id = trace_manager.start_span(
                "human_review_pause",
                input={"job_ids": list(payload.resumes)},
            )
            trace_manager.end_span(span_id)
        review_feedback = interrupt(payload.model_dump())
        return {
            "interrupt_payload": payload.model_dump(),
            "review_feedback": review_feedback,
            "status": RunStatus.RUNNING.value,
        }

    def run_post_review_tools(state: AgentState) -> AgentState:
        """Apply reviewed revisions, then generate cover letters."""

        top_job_ids = list(state_value(state, "top_3_job_ids"))
        feedback = normalize_review_feedback(
            state.get("review_feedback", {}),
            expected_job_ids=top_job_ids,
        )
        decisions = {
            job_id: decision.model_dump()
            for job_id, decision in feedback.decisions.items()
        }
        rejected = [
            job_id
            for job_id, decision in feedback.decisions.items()
            if decision.decision == "reject"
        ]
        memory_facts, new_fact_ids, memory_failures = _store_review_memory(
            state, decisions, tracer=trace_manager
        )
        state_with_memory: AgentState = {
            **state,
            "memory_facts": memory_facts,
        }
        profile = CandidateProfile.model_validate(
            state_value(state, "candidate_profile")
        )
        portfolio = Portfolio.model_validate(state_value(state, "portfolio"))
        candidate_evidence = _candidate_evidence(state_with_memory, profile, portfolio)
        run_id = state_value(state, "run_id")
        jobs = {
            item["job_id"]: Job.model_validate(item)
            for item in state_value(state, "jobs")
        }
        analyses = dict(state_value(state, "fit_analyses"))
        tailoring = dict(state_value(state, "tailoring_results"))
        history = list(state.get("tool_history", []))
        decisions_history = list(state.get("agent_decisions", []))
        actions: dict[str, dict[str, Any]] = {}
        revision_round_logs: dict[int, dict[str, Any]] = {}

        for job_id in rejected:
            feedback_text = decisions[job_id]["comment"]
            latest = tailoring[job_id]
            revision_round = 0
            for revision_round in range(1, MAX_REVISION_ROUNDS + 1):
                resume_input = TailorResumeInput(
                    job=jobs[job_id],
                    fit_analysis=FitAnalysisOutput.model_validate(analyses[job_id]),
                    source_resume_tex_path=state_value(state, "resume_path"),
                    candidate_evidence=candidate_evidence,
                    job_evidence=build_job_evidence(jobs[job_id]),
                    revision_feedback=feedback_text,
                    run_id=run_id,
                )
                revised, decision = _select_and_invoke_required_tool(
                    selector,
                    state,
                    "resume_tailoring",
                    resume_input,
                    tracer=trace_manager,
                    output_type=TailorResumeOutput,
                )
                decisions_history.append(decision)
                validate_artifact_output(
                    get_tool("resume_tailoring").callable_name, revised
                )
                latest = revised.model_dump()
                history.append(
                    _history(get_tool("resume_tailoring").callable_name, revised)
                )
                round_log = revision_round_logs.setdefault(
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
                        output=revised,
                    )
                )
                if revised.revision_feedback_satisfied is True:
                    break
            tailoring[job_id] = latest
            actions[job_id] = {
                "revision_round": revision_round,
                "status": latest["status"],
                "change_log": latest["change_log"],
                "output_tex_path": latest["output_tex_path"],
                "output_pdf_path": latest["output_pdf_path"],
            }
            if latest.get("revision_feedback_satisfied") is not True:
                review_history = [
                    {
                        "review_round": 1,
                        "decisions": decisions,
                        "rejected_job_ids": rejected,
                        "memory_writes": [
                            fact
                            for fact in memory_facts
                            if fact["fact_id"] in set(new_fact_ids)
                        ],
                        "actions_taken": actions,
                        "revision_rounds": list(revision_round_logs.values()),
                    }
                ]
                return {
                    "phase": Phase.HUMAN_REVIEW.value,
                    "status": RunStatus.FAILED_REVIEW.value,
                    "tailoring_results": tailoring,
                    "tool_history": history,
                    "agent_decisions": decisions_history,
                    "review_history": review_history,
                    "errors": [
                        *state.get("errors", []),
                        {
                            "phase": Phase.HUMAN_REVIEW.value,
                            "type": "FailedReview",
                            "message": (
                                f"Feedback for {job_id} remained unsatisfied after "
                                f"{MAX_REVISION_ROUNDS} revision rounds."
                            ),
                        },
                    ],
                }

        cover_letters: dict[str, dict[str, Any]] = {}
        for job_id in top_job_ids:
            cover_input = GenerateCoverLetterInput(
                job=jobs[job_id],
                approved_resume_path=tailoring[job_id]["output_pdf_path"],
                candidate_evidence=candidate_evidence,
                job_evidence=build_job_evidence(jobs[job_id]),
                run_id=run_id,
            )
            letter, decision = _select_and_invoke_required_tool(
                selector,
                state,
                "cover_letter",
                cover_input,
                tracer=trace_manager,
                output_type=GenerateCoverLetterOutput,
            )
            decisions_history.append(decision)
            validate_artifact_output(get_tool("cover_letter").callable_name, letter)
            cover_letters[job_id] = letter.model_dump()
            history.append(_history(get_tool("cover_letter").callable_name, letter))

        review_history = [
            {
                "review_round": 1,
                "decisions": decisions,
                "rejected_job_ids": rejected,
                "memory_writes": [
                    fact
                    for fact in memory_facts
                    if fact["fact_id"] in set(new_fact_ids)
                ],
                "actions_taken": actions,
                "revision_rounds": list(revision_round_logs.values()),
            }
        ]
        final_payload = {
            **state,
            "review_decisions": decisions,
            "approved_job_ids": top_job_ids,
            "tailoring_results": tailoring,
            "cover_letter_results": cover_letters,
            "fit_analysis_artifacts": state.get("fit_analysis_artifacts", {}),
            "top_3_job_ids": top_job_ids,
            "jobs": state.get("jobs", []),
            "review_history": review_history,
            "agent_decisions": decisions_history,
            "trace_events": [event.__dict__ for event in trace_manager.events],
        }
        output_manifest = write_and_validate_outputs(final_payload)
        trace_manager.update_run(
            metadata={"status": RunStatus.COMPLETED.value},
            output={
                "top_3_job_ids": top_job_ids,
                "output_manifest": output_manifest,
            },
        )
        trace_manager.flush()
        return {
            "phase": Phase.COMPLETE.value,
            "status": RunStatus.COMPLETED.value,
            "review_decisions": decisions,
            "approved_job_ids": top_job_ids,
            "revision_round": max(
                (action["revision_round"] for action in actions.values()),
                default=0,
            ),
            "tailoring_results": tailoring,
            "cover_letter_results": cover_letters,
            "memory_facts": memory_facts,
            "new_memory_fact_ids": new_fact_ids,
            "memory_validation_failures": memory_failures,
            "review_history": review_history,
            "tool_history": history,
            "agent_decisions": decisions_history,
            "output_manifest": output_manifest,
            "trace_url": trace_manager.trace_url,
            "langfuse_status": trace_manager.status_message,
        }

    graph.add_node("initialize", initialize)
    graph.add_node("run_pre_review_tools", run_pre_review_tools)
    graph.add_node("human_review", human_review)
    graph.add_node("run_post_review_tools", run_post_review_tools)
    graph.add_edge(START, "initialize")
    graph.add_edge("initialize", "run_pre_review_tools")
    graph.add_edge("run_pre_review_tools", "human_review")
    graph.add_edge("human_review", "run_post_review_tools")
    graph.add_edge("run_post_review_tools", END)
    return graph.compile(checkpointer=checkpointer)


def _invoke_required_tool(
    tool_name: str,
    arguments: BaseModel,
    *,
    tracer: TraceManager,
    output_type: type[BaseModel],
) -> Any:
    result = invoke_tool(tool_name, arguments, context={"tracer": tracer})
    if isinstance(result, output_type):
        return result
    return output_type.model_validate(result)


def _select_and_invoke_required_tool(
    selector: ToolSelectionModel,
    state: AgentState,
    tool_name: str,
    arguments: BaseModel,
    *,
    tracer: TraceManager,
    output_type: type[BaseModel],
) -> tuple[Any, dict[str, Any]]:
    call = select_validated_tool_call(
        selector,
        phase=state.get("phase", "UNKNOWN"),
        expected_tool=tool_name,
        prepared_arguments={tool_name: arguments},
        state_summary=_decision_state_summary(state),
        tracer=tracer,
    )
    span_id = tracer.start_span(
        "orchestration.dispatch_validated_call",
        {
            "selected_tool": call.name,
            "expected_tool": tool_name,
            "job_id": _argument_job_id(call.arguments),
        },
        input={
            "tool_name": call.name,
            "argument_keys": sorted(call.arguments),
        },
    )
    try:
        parsed_arguments = get_tool(call.name).input_model.model_validate(
            call.arguments
        )
        result = invoke_tool(call.name, parsed_arguments, context={"tracer": tracer})
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
        output={"output_type": type(result).__name__},
    )
    if not isinstance(result, output_type):
        result = output_type.model_validate(result)
    return result, {
        "phase": state.get("phase"),
        "selected_tool": call.name,
        "expected_tool": tool_name,
        "job_id": _argument_job_id(call.arguments),
        "rationale": call.rationale,
        "argument_keys": sorted(call.arguments),
    }


def _decision_state_summary(state: AgentState) -> dict[str, Any]:
    return {
        "run_id": state.get("run_id"),
        "phase": state.get("phase"),
        "status": state.get("status"),
        "filtered_job_count": len(state.get("filtered_jobs", [])),
        "ranked_job_count": len(state.get("ranked_jobs", [])),
        "top_3_job_ids": list(state.get("top_3_job_ids", [])),
        "review_round": state.get("revision_round", 0),
    }


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
    ]


def _store_review_memory(
    state: AgentState,
    decisions: dict[str, dict[str, Any]],
    *,
    tracer: TraceManager,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    span_id = tracer.start_span(
        "memory.write",
        input={"reviewed_job_ids": sorted(decisions)},
    )
    comments = {
        job_id: decision["comment"]
        for job_id, decision in decisions.items()
        if decision["comment"].strip()
    }
    try:
        extracted = extract_memory_facts(comments, review_round=1)
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
        conflict_events = [
            conflict.model_dump() for fact in updated for conflict in fact.conflicts
        ]
        conflict_span = tracer.start_span(
            "memory.conflict_handling",
            input={
                "candidate_fact_count": len(valid),
                "existing_fact_count": len(before),
            },
        )
        tracer.end_span(
            conflict_span,
            output={
                "conflict_count": len(conflict_events),
                "conflicts": conflict_events,
                "active_values": [
                    fact.canonical_value for fact in updated if fact.active
                ],
            },
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
            "validation_failures": failures,
            "conflict_count": len(conflict_events),
            "active_values": [fact.canonical_value for fact in updated if fact.active],
        },
    )
    return [fact.model_dump() for fact in updated], new_ids, failures


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
