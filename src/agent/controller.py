"""Single LLM-agent controller and offline fallback decisions."""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import Field

from src.agent.phase_policy import (
    Phase,
    assert_cover_letters_allowed,
    assert_scoring_complete,
    assert_tool_allowed,
)
from src.agent.state import AgentState
from src.config import get_config
from src.memory.models import MemoryFact, memory_fact_to_evidence
from src.observability.trace_manager import TraceManager
from src.schemas.common import (
    CandidateProfile,
    EvidenceItem,
    Portfolio,
    StrictBaseModel,
)
from src.schemas.cover_letter import GenerateCoverLetterInput
from src.schemas.filtering import FilterJobsInput
from src.schemas.fit_analysis import AnalyzeFitInput
from src.schemas.jobs import Job
from src.schemas.scoring import ScoreJobsInput
from src.schemas.tailoring import TailorResumeInput
from src.tools.job_evidence import build_job_evidence
from src.tools.registry import ToolSpec, as_langchain_tools

CONTROLLER_SYSTEM_PROMPT_VERSION = "controller-v1"
CONTROLLER_SYSTEM_PROMPT = """
You are the only LLM agent in the Job Search Agent workflow.
Use only the tools allowed in the current phase.
Never calculate job scores; score_jobs returns authoritative numeric scores.
Never bypass filtering or scoring.
Never generate cover letters before all three resumes are approved.
Never invent candidate facts.
Use evidence IDs when making factual decisions.
Return structured decisions with a concise decision_summary for tracing.
Do not expose private chain-of-thought.
""".strip()


class AgentDecision(StrictBaseModel):
    """Structured controller output."""

    phase: str
    selected_tool: str
    target_job_id: str | None = None
    arguments: dict[str, Any]
    decision_summary: str
    evidence_ids: list[str] = Field(default_factory=list)
    available_actions: list[dict[str, Any]] = Field(default_factory=list)
    unresolved_requirements: list[str] = Field(default_factory=list)
    decision_source: Literal["llm", "offline_policy"] = "offline_policy"


class AgentIntent(StrictBaseModel):
    """Small model-owned decision; Python builds the validated tool payload."""

    phase: str
    selected_tool: str
    target_job_id: str | None = None
    decision_summary: str


class AgentControllerError(RuntimeError):
    """Raised when the single controller cannot select a valid next tool."""


class SingleAgentController:
    """The only agent/controller that chooses model-visible tool calls."""

    def __init__(
        self,
        registry: dict[str, ToolSpec],
        model_name: str | None = None,
        enable_llm: bool | None = None,
        tracer: TraceManager | None = None,
    ) -> None:
        config = get_config()
        self.registry = registry
        self.model_name = model_name or config.llm_model or "offline-controller"
        self.deepinfra_api_key = config.deepinfra_api_key
        self.deepinfra_base_url = config.deepinfra_base_url
        self.available_tools = list(registry)
        self.langchain_tools = as_langchain_tools(registry)
        self.tracer = tracer
        self.enable_llm = (
            enable_llm
            if enable_llm is not None
            else bool(self.deepinfra_api_key and config.llm_model)
        )
        self._use_deterministic_policy = enable_llm is False

    def decide(self, state: AgentState) -> AgentDecision:
        """Select exactly one allowed tool for the current phase."""

        if self.enable_llm:
            try:
                return self._llm_decide(state)
            except Exception as exc:
                raise AgentControllerError("The LLM controller failed.") from exc
        if self._use_deterministic_policy:
            return self._offline_decide(state)
        raise AgentControllerError(
            "An LLM controller is required. Configure DEEPINFRA_API_KEY and LLM_MODEL."
        )

    def _llm_decide(self, state: AgentState) -> AgentDecision:
        """Let the LLM select the next tool without letting it rewrite evidence."""

        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=self.model_name,
            api_key=self.deepinfra_api_key,
            base_url=self.deepinfra_base_url,
            temperature=0,
        )
        try:
            structured = llm.with_structured_output(AgentIntent, include_raw=True)
            includes_raw_message = True
        except TypeError:
            # Some providers and test doubles do not support raw responses.
            structured = llm.with_structured_output(AgentIntent)
            includes_raw_message = False
        phase = state.get("phase", Phase.INITIALIZE.value)
        allowed_actions = self._allowed_actions(state)
        allowed_tools = list(dict.fromkeys(action["tool_name"] for action in allowed_actions))
        tool_descriptions = {
            name: self.registry[name].description for name in allowed_tools
        }
        snapshot = self._workflow_snapshot(state)
        messages = [
            ("system", CONTROLLER_SYSTEM_PROMPT),
            (
                "human",
                "Choose exactly one next tool from the allowed tools. Explain the "
                "decision briefly using observable workflow state and choose one "
                "listed target_job_id when the action has targets. Do not provide "
                "chain-of-thought or tool arguments.\n"
                f"Workflow state: {json.dumps(snapshot, sort_keys=True)}\n"
                f"Allowed tools: {json.dumps(tool_descriptions, sort_keys=True)}\n"
                f"Allowed actions: {json.dumps(allowed_actions, sort_keys=True)}",
            ),
        ]
        try:
            raw_result = structured.invoke(messages)
            if (
                includes_raw_message
                and isinstance(raw_result, dict)
                and "parsed" in raw_result
            ):
                intent = raw_result["parsed"]
                raw_message = raw_result.get("raw")
                parsing_error = raw_result.get("parsing_error")
                if parsing_error is not None:
                    raise AgentControllerError(
                        f"Structured controller output could not be parsed: "
                        f"{parsing_error.__class__.__name__}"
                    )
            else:
                intent = raw_result
                raw_message = None
            if not isinstance(intent, AgentIntent):
                intent = AgentIntent.model_validate(intent)
            if self.tracer is not None:
                self.tracer.record_generation(
                    {
                        "phase": phase,
                        "tool_names": allowed_tools,
                        "provider": "deepinfra",
                        "system_prompt_version": CONTROLLER_SYSTEM_PROMPT_VERSION,
                    },
                    name="agent_controller_llm",
                    model=self.model_name,
                    messages=messages,
                    response=intent.model_dump(),
                    usage=_message_usage(raw_message),
                    model_parameters={
                        "temperature": 0,
                        "base_url": self.deepinfra_base_url,
                        "structured_output": True,
                        "include_raw": includes_raw_message,
                    },
                )
        except Exception as exc:
            if self.tracer is not None:
                self.tracer.record_generation(
                    {
                        "phase": phase,
                        "tool_names": allowed_tools,
                        "provider": "deepinfra",
                        "system_prompt_version": CONTROLLER_SYSTEM_PROMPT_VERSION,
                    },
                    name="agent_controller_llm",
                    model=self.model_name,
                    messages=messages,
                    response={"error_type": exc.__class__.__name__},
                    usage={},
                    model_parameters={
                        "temperature": 0,
                        "base_url": self.deepinfra_base_url,
                        "structured_output": True,
                        "include_raw": includes_raw_message,
                    },
                    status="ERROR",
                    error_type=exc.__class__.__name__,
                )
            raise
        if intent.phase != phase:
            raise AgentControllerError(
                f"Model returned phase {intent.phase!r}; expected {phase!r}."
            )
        valid_action = next(
            (
                action
                for action in allowed_actions
                if action["tool_name"] == intent.selected_tool
                and (
                    (
                        not action["target_job_ids"]
                        and intent.target_job_id is None
                    )
                    or intent.target_job_id in action["target_job_ids"]
                )
            ),
            None,
        )
        if valid_action is None:
            raise AgentControllerError(
                f"Model selected an unavailable action: tool={intent.selected_tool!r}, "
                f"target_job_id={intent.target_job_id!r}."
            )
        assert_tool_allowed(phase, intent.selected_tool)

        # Validate the model's choice, then build its evidence payload.
        decision = self._decision_for_action(
            state, intent.selected_tool, intent.target_job_id
        )
        return decision.model_copy(
            update={
                "target_job_id": intent.target_job_id,
                "decision_summary": intent.decision_summary,
                "decision_source": "llm",
                "available_actions": allowed_actions,
                "unresolved_requirements": self._unresolved_requirements(state),
            }
        )

    def _offline_decide(self, state: AgentState) -> AgentDecision:
        """Explicitly enabled deterministic fallback for tests and development."""

        actions = self._allowed_actions(state)
        if not actions:
            raise AgentControllerError("No model-visible action satisfies prerequisites.")
        action = actions[0]
        target_ids = action["target_job_ids"]
        target_job_id = target_ids[0] if target_ids else None
        decision = self._decision_for_action(
            state, action["tool_name"], target_job_id
        )
        phase = state.get("phase", Phase.INITIALIZE.value)
        assert_tool_allowed(phase, decision.selected_tool)
        return decision.model_copy(
            update={
                "available_actions": actions,
                "unresolved_requirements": self._unresolved_requirements(state),
            }
        )

    def _decision_for_action(
        self, state: AgentState, tool_name: str, target_job_id: str | None
    ) -> AgentDecision:
        if tool_name == "filter_jobs":
            return self._filter_decision(state)
        if tool_name == "score_jobs":
            return self._score_decision(state)
        if tool_name == "analyze_fit":
            return self._fit_decision(state, target_job_id)
        if tool_name == "tailor_resume":
            return self._tailor_decision(state, target_job_id)
        if tool_name == "generate_cover_letter":
            return self._cover_letter_decision(state, target_job_id)
        raise AgentControllerError(f"Unsupported model-selected tool: {tool_name}")

    def _filter_decision(self, state: AgentState) -> AgentDecision:
        profile = CandidateProfile.model_validate(state["candidate_profile"])
        jobs = [Job.model_validate(job) for job in state.get("jobs", [])]
        payload = FilterJobsInput(jobs=jobs, preferences=profile.preferences)
        return AgentDecision(
            phase=Phase.FILTER.value,
            selected_tool="filter_jobs",
            arguments=payload.model_dump(),
            decision_summary="The controller selected filtering before scoring any jobs.",
            evidence_ids=[],
        )

    def _score_decision(self, state: AgentState) -> AgentDecision:
        profile = CandidateProfile.model_validate(state["candidate_profile"])
        jobs = [Job.model_validate(job) for job in state.get("filtered_jobs", [])]
        payload = ScoreJobsInput(
            jobs=jobs,
            candidate_profile=profile,
            resume_evidence=profile.resume_evidence,
            master_skill_evidence=profile.master_skill_evidence,
            portfolio_evidence=self._portfolio_evidence(state),
            memory_evidence=self._memory_evidence(state),
        )
        evidence_ids = [item.evidence_id for item in payload.resume_evidence[:3]]
        return AgentDecision(
            phase=Phase.SCORE.value,
            selected_tool="score_jobs",
            arguments=payload.model_dump(),
            decision_summary="The controller selected scoring for filtered jobs; numeric scores remain tool-owned.",
            evidence_ids=evidence_ids,
        )

    def _fit_decision(
        self, state: AgentState, target_job_id: str | None = None
    ) -> AgentDecision:
        assert_scoring_complete(
            state.get("ranked_jobs", []), state.get("top_3_job_ids", [])
        )
        next_job_id = target_job_id or self._next_missing(
            state["top_3_job_ids"], state.get("fit_analyses", {})
        )
        if not next_job_id:
            raise AgentControllerError("All selected jobs already have fit analyses.")
        if next_job_id in state.get("fit_analyses", {}):
            raise AgentControllerError(f"Fit analysis already exists for {next_job_id}.")
        if next_job_id not in state.get("top_3_job_ids", []):
            raise AgentControllerError(f"{next_job_id} is not a selected Top-3 job.")
        job = self._job_by_id(state, next_job_id)
        profile = CandidateProfile.model_validate(state["candidate_profile"])
        portfolio = Portfolio.model_validate(state["portfolio"])
        evidence_items = [
            *profile.resume_evidence,
            *profile.master_skill_evidence,
            *self._portfolio_evidence(state),
            *self._memory_evidence(state),
        ]
        payload = AnalyzeFitInput(
            job=job,
            candidate_profile=profile,
            evidence_items=evidence_items,
            job_evidence=build_job_evidence(job),
            current_resume_projects=profile.resume_projects,
            portfolio_projects=portfolio.projects,
        )
        return AgentDecision(
            phase=Phase.FIT_ANALYSIS.value,
            selected_tool="analyze_fit",
            target_job_id=job.job_id,
            arguments=payload.model_dump(),
            decision_summary=f"The controller selected fit analysis for {job.job_id}.",
            evidence_ids=[item.evidence_id for item in evidence_items[:5]],
        )

    def _tailor_decision(
        self, state: AgentState, target_job_id: str | None = None
    ) -> AgentDecision:
        pending = list(state.get("pending_revision_job_ids", []))
        if pending:
            next_job_id = target_job_id or pending[0]
            if next_job_id not in pending:
                raise AgentControllerError(
                    f"{next_job_id} is not pending resume revision."
                )
            revision_feedback = _decision_comment(state, next_job_id)
            summary = f"The controller selected resume revision for {next_job_id}."
        else:
            next_job_id = target_job_id or self._next_missing(
                state.get("top_3_job_ids", []), state.get("tailoring_results", {})
            )
            if next_job_id in state.get("tailoring_results", {}):
                raise AgentControllerError(f"Resume already tailored for {next_job_id}.")
            revision_feedback = None
            summary = (
                f"The controller selected initial resume tailoring for {next_job_id}."
            )
        if not next_job_id:
            raise AgentControllerError("No selected job is pending resume tailoring.")
        job = self._job_by_id(state, next_job_id)
        fit_analysis = state.get("fit_analyses", {}).get(next_job_id)
        if not fit_analysis:
            raise AgentControllerError(f"Missing fit analysis for {next_job_id}.")
        evidence_items = self._candidate_evidence(state)
        payload = TailorResumeInput(
            job=job,
            fit_analysis=fit_analysis,
            source_resume_tex_path=self._resume_source_path(state, next_job_id),
            candidate_evidence=evidence_items,
            job_evidence=build_job_evidence(job),
            revision_feedback=_revision_feedback(state, next_job_id, revision_feedback),
        )
        return AgentDecision(
            phase=Phase.TAILOR.value,
            selected_tool="tailor_resume",
            target_job_id=job.job_id,
            arguments=payload.model_dump(),
            decision_summary=summary,
            evidence_ids=[item.evidence_id for item in evidence_items[:5]],
        )

    def _cover_letter_decision(
        self, state: AgentState, target_job_id: str | None = None
    ) -> AgentDecision:
        assert_cover_letters_allowed(
            state.get("top_3_job_ids", []), state.get("approved_job_ids", [])
        )
        next_job_id = target_job_id or self._next_missing(
            state.get("top_3_job_ids", []), state.get("cover_letter_results", {})
        )
        if not next_job_id:
            raise AgentControllerError("All selected jobs already have cover letters.")
        if next_job_id in state.get("cover_letter_results", {}):
            raise AgentControllerError(f"Cover letter already exists for {next_job_id}.")
        if next_job_id not in state.get("top_3_job_ids", []):
            raise AgentControllerError(f"{next_job_id} is not a selected Top-3 job.")
        job = self._job_by_id(state, next_job_id)
        tailoring = state.get("tailoring_results", {})[next_job_id]
        evidence_items = self._candidate_evidence(state)
        payload = GenerateCoverLetterInput(
            job=job,
            approved_resume_path=tailoring["output_pdf_path"],
            candidate_evidence=evidence_items,
            job_evidence=build_job_evidence(job),
        )
        return AgentDecision(
            phase=Phase.COVER_LETTERS.value,
            selected_tool="generate_cover_letter",
            target_job_id=job.job_id,
            arguments=payload.model_dump(),
            decision_summary=f"The controller selected cover-letter generation for {next_job_id}.",
            evidence_ids=[item.evidence_id for item in evidence_items[:5]],
        )

    def _candidate_evidence(self, state: AgentState) -> list[EvidenceItem]:
        profile = CandidateProfile.model_validate(state["candidate_profile"])
        return [
            *profile.resume_evidence,
            *profile.master_skill_evidence,
            *self._portfolio_evidence(state),
            *self._memory_evidence(state),
        ]

    def _portfolio_evidence(self, state: AgentState) -> list[EvidenceItem]:
        portfolio = Portfolio.model_validate(state["portfolio"])
        profile = CandidateProfile.model_validate(state["candidate_profile"])
        return [*portfolio.evidence_items, *profile.portfolio_evidence]

    def _memory_evidence(self, state: AgentState) -> list[EvidenceItem]:
        facts = [
            MemoryFact.model_validate(item) for item in state.get("memory_facts", [])
        ]
        return [
            EvidenceItem.model_validate(memory_fact_to_evidence(fact))
            for fact in facts
            if fact.active
        ]

    def _job_by_id(self, state: AgentState, job_id: str) -> Job:
        jobs = {job["job_id"]: job for job in state.get("jobs", [])}
        if job_id not in jobs:
            raise AgentControllerError(f"Unknown job ID: {job_id}")
        return Job.model_validate(jobs[job_id])

    def _next_missing(
        self, ordered_ids: list[str], completed: dict[str, Any]
    ) -> str | None:
        for item_id in ordered_ids:
            if item_id not in completed:
                return item_id
        return None

    def _allowed_actions(self, state: AgentState) -> list[dict[str, Any]]:
        """Return prerequisite-valid tools and model-selectable job targets."""

        phase = state.get("phase", Phase.INITIALIZE.value)
        actions: list[dict[str, Any]] = []
        if phase == Phase.FILTER.value:
            actions.append({"tool_name": "filter_jobs", "target_job_ids": []})
        elif phase == Phase.SCORE.value:
            actions.append({"tool_name": "score_jobs", "target_job_ids": []})
        elif phase in {Phase.FIT_ANALYSIS.value, Phase.TAILOR.value}:
            top_3 = list(state.get("top_3_job_ids", []))
            assert_scoring_complete(state.get("ranked_jobs", []), top_3)
            pending = list(state.get("pending_revision_job_ids", []))
            if pending:
                actions.append(
                    {"tool_name": "tailor_resume", "target_job_ids": pending}
                )
            else:
                missing_fit = [
                    job_id
                    for job_id in top_3
                    if job_id not in state.get("fit_analyses", {})
                ]
                ready_to_tailor = [
                    job_id
                    for job_id in top_3
                    if job_id in state.get("fit_analyses", {})
                    and job_id not in state.get("tailoring_results", {})
                ]
                if missing_fit:
                    actions.append(
                        {"tool_name": "analyze_fit", "target_job_ids": missing_fit}
                    )
                if ready_to_tailor:
                    actions.append(
                        {
                            "tool_name": "tailor_resume",
                            "target_job_ids": ready_to_tailor,
                        }
                    )
        elif phase == Phase.COVER_LETTERS.value:
            assert_cover_letters_allowed(
                state.get("top_3_job_ids", []), state.get("approved_job_ids", [])
            )
            missing_letters = [
                job_id
                for job_id in state.get("top_3_job_ids", [])
                if job_id not in state.get("cover_letter_results", {})
            ]
            if missing_letters:
                actions.append(
                    {
                        "tool_name": "generate_cover_letter",
                        "target_job_ids": missing_letters,
                    }
                )
        return [
            action for action in actions if action["tool_name"] in self.available_tools
        ]

    def _resume_source_path(self, state: AgentState, job_id: str) -> str:
        """Use the prior tailored TeX as the source for a revision."""

        previous = state.get("tailoring_results", {}).get(job_id, {})
        if job_id in state.get("pending_revision_job_ids", []):
            return previous.get("output_tex_path") or state["resume_path"]
        return state["resume_path"]

    def _workflow_snapshot(self, state: AgentState) -> dict[str, Any]:
        """Return compact, non-sensitive progress context for tool selection."""

        top_3 = state.get("top_3_job_ids", [])
        return {
            "phase": state.get("phase"),
            "filtered_job_count": len(state.get("filtered_jobs", [])),
            "ranked_job_count": len(state.get("ranked_jobs", [])),
            "top_3_job_ids": top_3,
            "fit_analysis_completed": [
                job_id for job_id in top_3 if job_id in state.get("fit_analyses", {})
            ],
            "tailoring_completed": [
                job_id
                for job_id in top_3
                if job_id in state.get("tailoring_results", {})
            ],
            "pending_revision_job_ids": state.get("pending_revision_job_ids", []),
            "approved_job_ids": state.get("approved_job_ids", []),
            "cover_letters_completed": [
                job_id
                for job_id in top_3
                if job_id in state.get("cover_letter_results", {})
            ],
            "revision_round": state.get("revision_round", 0),
            "memory_fact_count": len(state.get("memory_facts", [])),
            "unresolved_requirements": self._unresolved_requirements(state),
            "constraints": {
                "single_agent": True,
                "single_review_pause": True,
                "maximum_revision_rounds": 2,
                "all_resumes_required_before_review": True,
                "all_resumes_approved_before_cover_letters": True,
            },
            "allowed_actions": self._allowed_actions(state),
        }

    def _unresolved_requirements(self, state: AgentState) -> list[str]:
        """Describe business work still open without selecting the next action."""

        phase = state.get("phase")
        top_3 = list(state.get("top_3_job_ids", []))
        unresolved: list[str] = []
        if phase == Phase.FILTER.value:
            unresolved.append("filter jobs using deterministic preference rules")
        if phase == Phase.SCORE.value:
            unresolved.append("score filtered jobs and select the deterministic Top 3")
        unresolved.extend(
            f"analyze fit for {job_id}"
            for job_id in top_3
            if job_id not in state.get("fit_analyses", {})
        )
        unresolved.extend(
            f"tailor resume for {job_id}"
            for job_id in top_3
            if job_id in state.get("fit_analyses", {})
            and job_id not in state.get("tailoring_results", {})
        )
        unresolved.extend(
            f"revise resume for {job_id}"
            for job_id in state.get("pending_revision_job_ids", [])
        )
        if phase == Phase.COVER_LETTERS.value:
            unresolved.extend(
                f"generate cover letter for {job_id}"
                for job_id in top_3
                if job_id not in state.get("cover_letter_results", {})
            )
        return list(dict.fromkeys(unresolved))


def _decision_comment(state: AgentState, job_id: str) -> str | None:
    decision = state.get("review_decisions", {}).get(job_id, {})
    comment = decision.get("comment", "")
    return comment.strip() or None


def _revision_feedback(
    state: AgentState, job_id: str, reviewer_comment: str | None
) -> str | None:
    """Combine direct feedback with newly learned, globally available evidence."""

    parts = [reviewer_comment] if reviewer_comment else []
    new_ids = set(state.get("new_memory_fact_ids", []))
    learned = [
        MemoryFact.model_validate(item)
        for item in state.get("memory_facts", [])
        if item.get("fact_id") in new_ids and item.get("active", True)
    ]
    if learned:
        evidence = ", ".join(
            f"{fact.canonical_value} ({fact.fact_id})" for fact in learned
        )
        parts.append(
            "Apply these newly learned candidate facts when relevant to this job, "
            f"using their memory IDs as evidence: {evidence}."
        )
    return "\n".join(parts) or None


def _message_usage(message: Any) -> dict[str, int]:
    """Normalize LangChain/OpenAI token usage when the provider returns it."""

    if message is None:
        return {}
    usage = getattr(message, "usage_metadata", None)
    if not usage:
        response_metadata = getattr(message, "response_metadata", {}) or {}
        usage = response_metadata.get("token_usage") or response_metadata.get("usage")
    if not isinstance(usage, dict):
        return {}
    aliases = {
        "prompt_tokens": "input_tokens",
        "completion_tokens": "output_tokens",
        "total_tokens": "total_tokens",
        "input_tokens": "input_tokens",
        "output_tokens": "output_tokens",
    }
    normalized: dict[str, int] = {}
    for source, target in aliases.items():
        value = usage.get(source)
        if isinstance(value, int):
            normalized[target] = value
    return normalized
