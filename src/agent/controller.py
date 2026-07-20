"""Single LLM-agent controller and offline fallback decisions."""

from __future__ import annotations

import logging
from typing import Any

from pydantic import Field

from src.agent.phase_policy import (
    PHASE_TOOL_POLICY,
    Phase,
    assert_cover_letters_allowed,
    assert_scoring_complete,
    assert_tool_allowed,
)
from src.agent.state import AgentState
from src.memory.models import MemoryFact, memory_fact_to_evidence
from src.schemas.common import CandidateProfile, EvidenceItem, Portfolio
from src.schemas.cover_letter import GenerateCoverLetterInput
from src.schemas.filtering import FilterJobsInput
from src.schemas.fit_analysis import AnalyzeFitInput
from src.schemas.jobs import Job
from src.schemas.scoring import ScoreJobsInput
from src.schemas.tailoring import TailorResumeInput
from src.schemas.common import StrictBaseModel
from src.config import DEEPINFRA_BASE_URL, get_config
from src.tools.registry import ToolSpec, as_langchain_tools

logger = logging.getLogger(__name__)

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
    arguments: dict[str, Any]
    decision_summary: str
    evidence_ids: list[str] = Field(default_factory=list)


class AgentControllerError(RuntimeError):
    """Raised when the single controller cannot select a valid next tool."""


class SingleAgentController:
    """The only agent/controller that chooses model-visible tool calls."""

    def __init__(
        self,
        registry: dict[str, ToolSpec],
        model_name: str | None = None,
        enable_llm: bool | None = None,
    ) -> None:
        config = get_config()
        self.registry = registry
        self.model_name = model_name or config.llm_model or "offline-controller"
        self.deepinfra_api_key = config.deepinfra_api_key
        self.deepinfra_base_url = config.deepinfra_base_url
        self.available_tools = list(registry)
        self.langchain_tools = as_langchain_tools(registry)
        self.enable_llm = (
            enable_llm
            if enable_llm is not None
            else bool(self.deepinfra_api_key and config.llm_model)
        )

    def decide(self, state: AgentState) -> AgentDecision:
        """Select exactly one allowed tool for the current phase."""

        if self.enable_llm:
            try:
                return self._llm_decide(state)
            except Exception as exc:
                logger.warning("LLM controller unavailable; using offline controller: %s", exc)
        return self._offline_decide(state)

    def _llm_decide(self, state: AgentState) -> AgentDecision:
        """Use a structured LLM call for controller decisions when configured."""

        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=self.model_name,
            api_key=self.deepinfra_api_key,
            base_url=self.deepinfra_base_url,
            temperature=0,
        )
        structured = llm.with_structured_output(AgentDecision)
        phase = state.get("phase", Phase.INITIALIZE.value)
        draft = self._offline_decide(state)
        prompt = (
            f"{CONTROLLER_SYSTEM_PROMPT}\n\n"
            f"Current phase: {phase}\n"
            f"Allowed tools: {self._allowed_tool_names(phase)}\n"
            f"Use this validated argument draft unless you can identify a schema issue: "
            f"{draft.model_dump()}"
        )
        decision = structured.invoke(prompt)
        assert_tool_allowed(phase, decision.selected_tool)
        return decision

    def _offline_decide(self, state: AgentState) -> AgentDecision:
        """Deterministic fallback used for local demos and tests."""

        phase = state.get("phase", Phase.INITIALIZE.value)
        if phase == Phase.FILTER.value:
            decision = self._filter_decision(state)
        elif phase == Phase.SCORE.value:
            decision = self._score_decision(state)
        elif phase == Phase.FIT_ANALYSIS.value:
            decision = self._fit_decision(state)
        elif phase == Phase.TAILOR.value:
            decision = self._tailor_decision(state)
        elif phase == Phase.COVER_LETTERS.value:
            decision = self._cover_letter_decision(state)
        else:
            raise AgentControllerError(f"No model-visible tool is available in phase {phase}.")
        assert_tool_allowed(phase, decision.selected_tool)
        return decision

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

    def _fit_decision(self, state: AgentState) -> AgentDecision:
        assert_scoring_complete(
            state.get("ranked_jobs", []), state.get("top_3_job_ids", [])
        )
        next_job_id = self._next_missing(
            state["top_3_job_ids"], state.get("fit_analyses", {})
        )
        if not next_job_id:
            raise AgentControllerError("All selected jobs already have fit analyses.")
        job = self._job_by_id(state, next_job_id)
        profile = CandidateProfile.model_validate(state["candidate_profile"])
        portfolio = Portfolio.model_validate(state["portfolio"])
        evidence_items = [
            *profile.resume_evidence,
            *self._portfolio_evidence(state),
            *self._memory_evidence(state),
        ]
        payload = AnalyzeFitInput(
            job=job,
            candidate_profile=profile,
            evidence_items=evidence_items,
            current_resume_projects=["Student Support RAG Assistant", "Forecasting Dashboard"],
            portfolio_projects=portfolio.projects,
        )
        return AgentDecision(
            phase=Phase.FIT_ANALYSIS.value,
            selected_tool="analyze_fit",
            arguments=payload.model_dump(),
            decision_summary=f"The controller selected fit analysis for {job.job_id}.",
            evidence_ids=[item.evidence_id for item in evidence_items[:5]],
        )

    def _tailor_decision(self, state: AgentState) -> AgentDecision:
        pending = list(state.get("pending_revision_job_ids", []))
        if pending:
            next_job_id = pending[0]
            revision_feedback = _decision_comment(state, next_job_id)
            summary = f"The controller selected resume revision for {next_job_id}."
        else:
            next_job_id = self._next_missing(
                state.get("top_3_job_ids", []), state.get("tailoring_results", {})
            )
            revision_feedback = None
            summary = f"The controller selected initial resume tailoring for {next_job_id}."
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
            source_resume_tex_path=state["resume_path"],
            candidate_evidence=evidence_items,
            revision_feedback=revision_feedback,
        )
        return AgentDecision(
            phase=Phase.TAILOR.value,
            selected_tool="tailor_resume",
            arguments=payload.model_dump(),
            decision_summary=summary,
            evidence_ids=[item.evidence_id for item in evidence_items[:5]],
        )

    def _cover_letter_decision(self, state: AgentState) -> AgentDecision:
        assert_cover_letters_allowed(
            state.get("top_3_job_ids", []), state.get("approved_job_ids", [])
        )
        next_job_id = self._next_missing(
            state.get("top_3_job_ids", []), state.get("cover_letter_results", {})
        )
        if not next_job_id:
            raise AgentControllerError("All selected jobs already have cover letters.")
        job = self._job_by_id(state, next_job_id)
        tailoring = state.get("tailoring_results", {})[next_job_id]
        evidence_items = self._candidate_evidence(state)
        payload = GenerateCoverLetterInput(
            job=job,
            approved_resume_path=tailoring["output_pdf_path"],
            candidate_evidence=evidence_items,
        )
        return AgentDecision(
            phase=Phase.COVER_LETTERS.value,
            selected_tool="generate_cover_letter",
            arguments=payload.model_dump(),
            decision_summary=f"The controller selected cover-letter generation for {next_job_id}.",
            evidence_ids=[item.evidence_id for item in evidence_items[:5]],
        )

    def _candidate_evidence(self, state: AgentState) -> list[EvidenceItem]:
        profile = CandidateProfile.model_validate(state["candidate_profile"])
        return [
            *profile.resume_evidence,
            *self._portfolio_evidence(state),
            *self._memory_evidence(state),
        ]

    def _portfolio_evidence(self, state: AgentState) -> list[EvidenceItem]:
        portfolio = Portfolio.model_validate(state["portfolio"])
        profile = CandidateProfile.model_validate(state["candidate_profile"])
        return [*portfolio.evidence_items, *profile.portfolio_evidence]

    def _memory_evidence(self, state: AgentState) -> list[EvidenceItem]:
        facts = [MemoryFact.model_validate(item) for item in state.get("memory_facts", [])]
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

    def _next_missing(self, ordered_ids: list[str], completed: dict[str, Any]) -> str | None:
        for item_id in ordered_ids:
            if item_id not in completed:
                return item_id
        return None

    def _allowed_tool_names(self, phase: str) -> list[str]:
        return [
            name
            for name in PHASE_TOOL_POLICY.get(phase, [])
            if name in self.available_tools
        ]


def _decision_comment(state: AgentState, job_id: str) -> str | None:
    decision = state.get("review_decisions", {}).get(job_id, {})
    comment = decision.get("comment", "")
    return comment.strip() or None
