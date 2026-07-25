"""The ``analyze_fit`` tool: evidence-backed fit analysis for one job.

Entry point loaded by ``src.tools.registry``:

    analyze_fit(inp: AnalyzeFitInput) -> FitAnalysisOutput

Pipeline: sanitize + index evidence -> deterministic pre-pass -> LLM reasoning
(or deterministic fallback when no model is configured) -> post-validation. The
run's single trace is owned by orchestration; this tool only emits a nested span.
"""

from __future__ import annotations

import logging

from src.observability.trace_manager import TraceManager
from src.schemas.fit_analysis import AnalyzeFitInput, FitAnalysisOutput
from src.tools.implementations.fit_analysis import (
    llm_client,
    postvalidate,
    rationale,
    reasoning,
)
from src.tools.implementations.fit_analysis.evidence_index import build_evidence_index
from src.tools.implementations.fit_analysis.prepass import (
    build_project_section,
    build_skill_section,
    run_prepass,
)

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

    active = tracer or TraceManager(enabled=False)
    span_id = active.start_span(
        "analyze_fit",
        {"tool_name": "analyze_fit", "job_id": inp.job.job_id, "company": inp.job.company},
    )
    try:
        index = build_evidence_index(inp.evidence_items)
        prepass = run_prepass(inp, index)
        output, path_label, meta = reasoning.analyze(inp, prepass, index, complete_fn=complete_fn)
        # Candidate facts are DECIDED deterministically so the model cannot rewrite
        # them and correctness is guaranteed on both paths: the three skill buckets
        # (an evidence lookup, incl. category->member resolution), projects (one
        # shared ranking, so the swap and project_analysis can never disagree), and
        # seniority (a year comparison). On the LLM path the model then writes only
        # the *prose* for the fixed seniority/swap outcomes, constrained to the
        # decision, with a template fallback. The LLM owns experience and education.
        aligned, evidenced_missing, genuine_gaps = build_skill_section(prepass)
        seniority = prepass.seniority
        project_analysis, project_swap = build_project_section(prepass)
        if path_label == "llm":
            active_complete = complete_fn or llm_client.complete
            seniority = rationale.enrich_seniority(seniority, active_complete)
            project_swap = rationale.enrich_swap(project_swap, prepass, inp, active_complete)
        output = output.model_copy(
            update={
                "aligned_skills": aligned,
                "evidenced_missing_skills": evidenced_missing,
                "genuine_gaps": genuine_gaps,
                "seniority": seniority,
                "project_analysis": project_analysis,
                "project_swap": project_swap,
            }
        )
        validated, repairs = postvalidate.post_validate(output, inp)
    except Exception as exc:
        active.end_span(span_id, status="ERROR", error_type=exc.__class__.__name__)
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
    )
    logger.info(
        "Fit analysis complete for %s via %s path (%d repairs).",
        inp.job.job_id,
        path_label,
        len(repairs),
    )
    return validated
