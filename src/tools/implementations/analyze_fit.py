"""The ``analyze_fit`` tool: evidence-backed fit analysis for one job.

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

from src.observability.trace_manager import TraceManager
from src.schemas.fit_analysis import AnalyzeFitInput, FitAnalysisOutput
from src.tools.implementations.fit_analysis import (
    llm_client,
    merge,
    postvalidate,
    rationale,
    reasoning,
)
from src.tools.implementations.fit_analysis.evidence_index import build_evidence_index
from src.tools.implementations.fit_analysis.prepass import (
    _expand_canonicals,
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
        # The job's own required skills join the scan vocabulary so a requirement
        # named only in a resume education/experience line (e.g. "M.S. Data
        # Science" grounding "data science") is found rather than called a gap.
        index = build_evidence_index(
            inp.evidence_items, vocabulary=set(_expand_canonicals(inp.job.required_skills))
        )
        prepass = run_prepass(inp, index)
        output, path_label, meta = reasoning.analyze(inp, prepass, index, complete_fn=complete_fn)
        # The MODEL makes the fit judgments; deterministic code guards them. On the
        # LLM path its buckets and swap are kept and then validated (every required
        # skill covered, the swap real and materially better) -- see merge.py. When
        # no model is configured, or the call fails, or its proposal cannot be
        # validated, the deterministic pre-pass supplies the whole analysis, so
        # today's behaviour is the floor and never the ceiling.
        merge_repairs: list[str] = []
        if path_label == "llm":
            output, merge_repairs, model_chose_swap = merge.merge_llm_proposal(
                output, inp, prepass
            )
            active_complete = complete_fn or llm_client.complete
            updates: dict[str, object] = {}
            if not output.seniority:
                # The model omitted seniority: take the deterministic finding and let
                # the model phrase it, rejecting prose that contradicts the verdict.
                updates["seniority"] = rationale.enrich_seniority(
                    prepass.seniority, active_complete
                )
            if output.project_swap is not None and not model_chose_swap:
                # The swap came from the ranking, so its rationale is templated prose;
                # the model rewrites it, constrained to the already-decided project.
                updates["project_swap"] = rationale.enrich_swap(
                    output.project_swap, prepass, inp, active_complete
                )
            if updates:
                output = output.model_copy(update=updates)
        else:
            aligned, evidenced_missing, genuine_gaps = build_skill_section(prepass)
            project_analysis, project_swap = build_project_section(prepass.swap)
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
