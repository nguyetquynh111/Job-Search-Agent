"""Implementation of evidence-backed fit analysis for one job.

Entry point loaded by ``src.tools.registry``:

    analyze_fit(inp: AnalyzeFitInput) -> FitAnalysisOutput

Pipeline: sanitize + index evidence -> deterministic pre-pass (grounding and
fallback) -> LLM reasoning proposes the fit judgments -> deterministic guards
validate that proposal (``merge.py``) -> post-validation enforces grounding. With
no model configured, a failed call, or an unvalidatable proposal, the pre-pass
supplies the entire analysis. The run's single trace is owned by orchestration;
this tool only emits a nested span.
"""

from __future__ import annotations

import logging

from src.config import get_config
from src.observability.trace_manager import TraceManager
from src.schemas.fit_analysis import AnalyzeFitInput, FitAnalysisOutput
from src.tools.fit_analysis import (
    llm_client,
    merge,
    postvalidate,
    rationale,
    reasoning,
)
from src.tools.fit_analysis.evidence_index import build_evidence_index
from src.tools.fit_analysis.prepass import (
    _expand_canonicals,
    build_project_section,
    build_skill_section,
    run_prepass,
)
from src.tools.job_evidence import build_job_evidence

logger = logging.getLogger(__name__)


def analyze_fit(
    inp: AnalyzeFitInput,
    *,
    tracer: TraceManager | None = None,
    complete_fn: reasoning.CompleteFn | None = None,
) -> FitAnalysisOutput:
    """Analyze the candidate's fit for one job, returning an evidence-bound result.

    Pass ``tracer`` to nest the span under the run trace; the default local tracer
    keeps this callable standalone and never fails when Langfuse is unconfigured.
    ``complete_fn`` injects an LLM client (tests); production uses the configured one.
    """

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
        # Include job skills when scanning short resume entries.
        index = build_evidence_index(
            inp.evidence_items,
            vocabulary=set(_expand_canonicals(inp.job.required_skills)),
        )
        prepass = run_prepass(inp, index)
        output, path_label, meta = reasoning.analyze(
            inp, prepass, index, complete_fn=active_complete
        )
        # Keep valid model judgments and fill any gaps from the pre-pass.
        merge_repairs: list[str] = []
        if path_label == "llm":
            output, merge_repairs, model_chose_swap = merge.merge_llm_proposal(
                output, inp, prepass
            )
            updates: dict[str, object] = {}
            if not output.seniority:
                # Add wording to the pre-pass seniority result.
                updates["seniority"] = rationale.enrich_seniority(
                    prepass.seniority, active_complete
                )
            if output.project_swap is not None and not model_chose_swap:
                # Let the model phrase the already-selected swap.
                updates["project_swap"] = rationale.enrich_swap(
                    output.project_swap, prepass, inp, active_complete
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
        validated, repairs = postvalidate.post_validate(output, inp)
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
    complete_fn: reasoning.CompleteFn | None,
) -> reasoning.CompleteFn | None:
    metadata = {
        "job_id": inp.job.job_id,
        "company": inp.job.company,
        "tool_name": "analyze_fit",
    }
    if complete_fn is None:
        if not llm_client.model_configured():
            return None

        def production(system: str, user: str) -> str:
            return llm_client.complete(
                system,
                user,
                tracer=tracer,
                metadata=metadata,
            )

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
