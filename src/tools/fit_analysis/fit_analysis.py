"""Public entrypoint and compatibility facade for fit analysis."""

from __future__ import annotations

import logging

from src.config import get_config
from src.tracing.langfuse import TraceManager
from src.tools.fit_analysis.contracts import AnalyzeFitInput, FitAnalysisOutput
from src.tools.fit_analysis.evidence_index import (
    EvidenceIndex,
    build_evidence_index,
    source_kind,
)
import src.tools.fit_analysis.llm as llm
from src.tools.fit_analysis.llm import (
    SENIORITY_RATIONALE_SYSTEM,
    SWAP_RATIONALE_SYSTEM,
    SYSTEM_PROMPT,
    analyze,
    build_seniority_rationale_prompt,
    build_swap_rationale_prompt,
    build_user_prompt,
    complete,
    enrich_seniority,
    enrich_swap,
    model_configured,
    seniority_prose_ok,
)
from src.tools.fit_analysis.prepass import (
    PrePass,
    SkillClaim,
    build_fallback_output,
    build_project_section,
    build_skill_section,
    expand_canonical_skills,
    prepass_summary,
    run_prepass,
    seniority_verdict,
)
from src.tools.fit_analysis.render import (
    build_source_labels,
    render_fit_analysis,
    write_fit_analysis,
)
from src.tools.fit_analysis.rules import (
    MASTER_SKILLS,
    MEMORY,
    MATCH,
    MISMATCH,
    MISSING,
    PARTIAL,
    PORTFOLIO,
    RESUME,
    marker,
    narrative_confidence,
    parse,
    sanitize_text,
    skill_confidence,
    tag,
)
from src.tools.fit_analysis.swap import (
    CurrentVerdict,
    ProjectScore,
    SwapDecision,
    UBIQUITOUS_SKILLS,
    build_project_swap,
    choose_swap,
    match_terms,
    normalize_name,
    rank_projects,
    resolve_portfolio_project,
    score_for,
    score_project,
    validate_proposed_swap,
)
import src.tools.fit_analysis.validation as validation
from src.tools.fit_analysis.validation import merge_llm_proposal, post_validate
from src.utils.job_evidence import build_job_evidence

logger = logging.getLogger(__name__)


def analyze_fit(
    inp: AnalyzeFitInput,
    *,
    tracer: TraceManager | None = None,
    complete_fn: llm.CompleteFn | None = None,
) -> FitAnalysisOutput:
    """Analyze one job and return an evidence-bound fit report."""

    if not inp.job_evidence:
        inp = inp.model_copy(update={"job_evidence": build_job_evidence(inp.job)})
    active = tracer or TraceManager(enabled=False)
    span_id = active.start_span(
        "fit_analysis.analysis_pipeline",
        {
            "tool_name": "analyze_fit",
            "job_id": inp.job.job_id,
            "company": inp.job.company,
        },
        input={
            "job_id": inp.job.job_id,
            "company": inp.job.company,
            "required_skill_count": len(inp.job.required_skills),
            "evidence_count": len(inp.evidence_items),
            "portfolio_project_count": len(inp.portfolio_projects),
        },
    )
    try:
        active_complete = _traced_completion(active, inp, complete_fn)
        index = build_evidence_index(
            inp.evidence_items,
            vocabulary=set(expand_canonical_skills(inp.job.required_skills)),
        )
        prepass = run_prepass(inp, index)
        output, path_label, meta = llm.analyze(
            inp, prepass, index, complete_fn=active_complete
        )
        merge_repairs: list[str] = []
        if path_label == "llm":
            enrichment_complete = active_complete or llm.complete
            output, merge_repairs, model_chose_swap = validation.merge_llm_proposal(
                output, inp, prepass
            )
            updates: dict[str, object] = {}
            if not output.seniority:
                updates["seniority"] = llm.enrich_seniority(
                    prepass.seniority, enrichment_complete
                )
            if output.project_swap is not None and not model_chose_swap:
                updates["project_swap"] = llm.enrich_swap(
                    output.project_swap, prepass, inp, enrichment_complete
                )
            if updates:
                output = output.model_copy(update=updates)
        else:
            aligned, evidenced_missing, genuine_gaps = build_skill_section(prepass)
            project_analysis, project_swap = build_project_section(
                prepass.swap, prepass.job_context_evidence_id
            )
            output = output.model_copy(
                update={
                    "aligned_skills": aligned,
                    "evidenced_missing_skills": evidenced_missing,
                    "genuine_gaps": genuine_gaps,
                    "seniority": prepass.seniority,
                    "project_analysis": project_analysis,
                    "project_swap": project_swap,
                }
            )
        validated, repairs = validation.post_validate(output, inp)
        repairs = [*merge_repairs, *repairs]
    except Exception as exc:
        active.end_span(
            span_id,
            status="ERROR",
            error_type=exc.__class__.__name__,
            output={"error_type": exc.__class__.__name__},
        )
        logger.exception("Fit analysis failed for %s", inp.job.job_id)
        raise
    active.end_span(
        span_id,
        metadata={
            "path": path_label,
            "llm_used": path_label == "llm",
            "model": meta.get("model"),
            "system_prompt": meta.get("system_prompt_name"),
            "evidence_skill_count": len(index.by_skill),
            "evidence_index": index.summary(),
            "aligned_count": len(validated.aligned_skills),
            "evidenced_missing_count": len(validated.evidenced_missing_skills),
            "genuine_gap_count": len(validated.genuine_gaps),
            "repairs": repairs,
            "repair_count": len(repairs),
            "project_swap": bool(validated.project_swap),
        },
        output={
            "job_id": validated.job_id,
            "aligned_count": len(validated.aligned_skills),
            "evidenced_missing_count": len(validated.evidenced_missing_skills),
            "genuine_gap_count": len(validated.genuine_gaps),
            "project_swap": validated.project_swap.model_dump()
            if validated.project_swap
            else None,
            "path": path_label,
        },
    )
    logger.info(
        "Fit analysis complete for %s via %s path (%d repairs).",
        inp.job.job_id,
        path_label,
        len(repairs),
    )
    return validated


def _traced_completion(
    tracer: TraceManager,
    inp: AnalyzeFitInput,
    complete_fn: llm.CompleteFn | None,
) -> llm.CompleteFn | None:
    metadata = {
        "job_id": inp.job.job_id,
        "company": inp.job.company,
        "tool_name": "analyze_fit",
    }
    if complete_fn is None:
        if not llm.model_configured():
            return None

        def production(system: str, user: str) -> str:
            return llm.complete(system, user, tracer=tracer, metadata=metadata)

        return production

    def injected(system: str, user: str) -> str:
        messages = [("system", system), ("human", user)]
        try:
            response = complete_fn(system, user)
            tracer.record_generation(
                {"provider": "injected", "purpose": "fit_analysis", **metadata},
                name="fit_analysis_llm",
                model=get_config().llm_model or "test-model",
                messages=messages,
                response=response,
                usage={},
                model_parameters={"temperature": 0},
            )
            return response
        except Exception as exc:
            tracer.record_generation(
                {"provider": "injected", "purpose": "fit_analysis", **metadata},
                name="fit_analysis_llm",
                model=get_config().llm_model or "test-model",
                messages=messages,
                response={"error_type": exc.__class__.__name__},
                usage={},
                model_parameters={"temperature": 0},
                status="ERROR",
                error_type=exc.__class__.__name__,
            )
            raise

    return injected


run_fit_analysis_tool = analyze_fit
run_fit_analysis_tool.__name__ = "run_fit_analysis_tool"

__all__ = [
    "AnalyzeFitInput",
    "FitAnalysisOutput",
    "CurrentVerdict",
    "EvidenceIndex",
    "ProjectScore",
    "SkillClaim",
    "SwapDecision",
    "PrePass",
    "MATCH",
    "MASTER_SKILLS",
    "MEMORY",
    "MISMATCH",
    "MISSING",
    "PARTIAL",
    "PORTFOLIO",
    "RESUME",
    "SYSTEM_PROMPT",
    "SENIORITY_RATIONALE_SYSTEM",
    "SWAP_RATIONALE_SYSTEM",
    "UBIQUITOUS_SKILLS",
    "analyze",
    "analyze_fit",
    "build_evidence_index",
    "expand_canonical_skills",
    "build_fallback_output",
    "build_project_section",
    "build_project_swap",
    "build_seniority_rationale_prompt",
    "build_skill_section",
    "build_source_labels",
    "build_swap_rationale_prompt",
    "build_user_prompt",
    "choose_swap",
    "complete",
    "enrich_seniority",
    "enrich_swap",
    "marker",
    "match_terms",
    "merge_llm_proposal",
    "model_configured",
    "narrative_confidence",
    "normalize_name",
    "parse",
    "post_validate",
    "prepass_summary",
    "rank_projects",
    "render_fit_analysis",
    "resolve_portfolio_project",
    "run_fit_analysis_tool",
    "run_prepass",
    "sanitize_text",
    "score_for",
    "score_project",
    "seniority_prose_ok",
    "seniority_verdict",
    "skill_confidence",
    "source_kind",
    "tag",
    "validate_proposed_swap",
    "write_fit_analysis",
]
