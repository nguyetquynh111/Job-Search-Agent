"""The ``score_jobs`` tool: deterministic scoring, ranking, and Top-3 selection.

Entry point loaded by ``src.tools.registry``:

    score_jobs(inp: ScoreJobsInput) -> ScoreJobsOutput

The number is computed by code, never by a model. Every job is
scored against the whole candidate profile -- resume, portfolio, master skills,
and memory -- folded into one index (see ``scoring/profile_index.py``). Scores,
weights, and rationale live in ``scoring/formula.py``. Jobs are ranked highest
to lowest with a stable tie-break (input order), and the Top-3 are selected
automatically. Evidence IDs on each result always exist in the input, satisfying
the contract. The run's single trace is owned by orchestration; this tool only
emits a nested span when a tracer is supplied.
"""

from __future__ import annotations

import logging

from src.observability.trace_manager import TraceManager
from src.schemas.scoring import ScoreJobsInput, ScoreJobsOutput, ScoredJob
from src.tools.scoring.formula import score_one_job
from src.tools.scoring.profile_index import build_candidate_index

logger = logging.getLogger(__name__)

TOP_N = 3


def score_jobs(
    inp: ScoreJobsInput,
    *,
    tracer: TraceManager | None = None,
) -> ScoreJobsOutput:
    """Score, rank, and select the Top-3 jobs against the full candidate profile.

    Pass ``tracer`` to nest a span under the run trace; the default disabled
    tracer keeps this callable standalone and never fails when Langfuse is
    unconfigured.
    """

    active = tracer or TraceManager(enabled=False)
    span_id = active.start_span(
        "score_jobs",
        {"tool_name": "score_jobs", "input_job_count": len(inp.jobs)},
        input={
            "job_count": len(inp.jobs),
            "resume_evidence_count": len(inp.resume_evidence),
            "portfolio_evidence_count": len(inp.portfolio_evidence),
            "master_skill_evidence_count": len(inp.master_skill_evidence),
            "memory_evidence_count": len(inp.memory_evidence),
        },
    )
    try:
        evidence = [
            *inp.resume_evidence,
            *inp.master_skill_evidence,
            *inp.portfolio_evidence,
            *inp.memory_evidence,
        ]
        valid_evidence_ids = {item.evidence_id for item in evidence}
        index = build_candidate_index(inp.candidate_profile, evidence)

        scored = []
        for job in inp.jobs:
            result = score_one_job(job, index)
            # Keep only evidence supplied with this request.
            evidence_ids = [
                evidence_id
                for evidence_id in result.evidence_ids
                if evidence_id in valid_evidence_ids
            ]
            scored.append(
                ScoredJob(
                    job=job,
                    score=result.score,
                    rationale=result.rationale,
                    evidence_ids=evidence_ids,
                )
            )

        # Preserve input order when scores tie.
        ranked = sorted(scored, key=lambda item: item.score, reverse=True)
        top_3_job_ids = [item.job.job_id for item in ranked[:TOP_N]]
        output = ScoreJobsOutput(ranked_jobs=ranked, top_3_job_ids=top_3_job_ids)
    except Exception as exc:
        active.end_span(
            span_id,
            status="ERROR",
            error_type=exc.__class__.__name__,
            output={"error_type": exc.__class__.__name__},
        )
        logger.exception("Scoring failed")
        raise
    active.end_span(
        span_id,
        metadata={
            "result_count": len(output.ranked_jobs),
            "top_3_job_ids": output.top_3_job_ids,
            "top_scores": [item.score for item in output.ranked_jobs[:TOP_N]],
        },
        output={
            "top_3_job_ids": output.top_3_job_ids,
            "ranked_jobs": [
                {
                    "job_id": item.job.job_id,
                    "company": item.job.company,
                    "score": item.score,
                }
                for item in output.ranked_jobs
            ],
        },
    )
    logger.info(
        "Scoring complete: ranked %d jobs; Top-3 = %s.",
        len(output.ranked_jobs),
        output.top_3_job_ids,
    )
    return output
