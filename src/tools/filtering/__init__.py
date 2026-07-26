"""The ``filter_jobs`` tool: deterministic, evidence-logged preference filtering.

Entry point loaded by ``src.tools.registry``:

    filter_jobs(inp: FilterJobsInput) -> FilterJobsOutput

Every input job lands in exactly one output list. Accepted jobs pass all rules;
rejected jobs carry one or more clear reasons (see
``src/tools/filtering/rules.py``). The tool is pure: it never
mutates a job, never calls an LLM, and always returns the same result for the
same input. The run's single trace is owned by orchestration; this tool only
emits a nested span when a tracer is supplied.
"""

from __future__ import annotations

import logging

from src.observability.trace_manager import TraceManager
from src.schemas.filtering import FilterJobsInput, FilterJobsOutput
from src.schemas.jobs import RejectedJob
from src.tools.filtering.rules import evaluate_job

logger = logging.getLogger(__name__)


def filter_jobs(
    inp: FilterJobsInput,
    *,
    tracer: TraceManager | None = None,
) -> FilterJobsOutput:
    """Split jobs into accepted/rejected against the candidate's preferences.

    Pass ``tracer`` to nest a span under the run trace; the default disabled
    tracer keeps this callable standalone and never fails when Langfuse is
    unconfigured.
    """

    active = tracer or TraceManager(enabled=False)
    span_id = active.start_span(
        "filter_jobs",
        {"tool_name": "filter_jobs", "input_job_count": len(inp.jobs)},
        input={
            "job_count": len(inp.jobs),
            "remote_only": inp.preferences.remote_only,
            "preferred_location_count": len(inp.preferences.preferred_locations),
            "excluded_company_count": len(inp.preferences.excluded_companies),
        },
    )
    try:
        accepted = []
        rejected = []
        for job in inp.jobs:
            reasons = evaluate_job(job, inp.preferences)
            if reasons:
                rejected.append(RejectedJob(job=job, reasons=reasons))
                logger.debug("Rejected %s: %s", job.job_id, "; ".join(reasons))
            else:
                accepted.append(job)
        output = FilterJobsOutput(accepted_jobs=accepted, rejected_jobs=rejected)
    except Exception as exc:
        active.end_span(
            span_id,
            status="ERROR",
            error_type=exc.__class__.__name__,
            output={"error_type": exc.__class__.__name__},
        )
        logger.exception("Filtering failed")
        raise
    active.end_span(
        span_id,
        metadata={
            "accepted_count": len(output.accepted_jobs),
            "rejected_count": len(output.rejected_jobs),
            "accepted_job_ids": [job.job_id for job in output.accepted_jobs][:10],
        },
        output={
            "accepted_job_ids": [job.job_id for job in output.accepted_jobs],
            "rejected_jobs": [
                {
                    "job_id": item.job.job_id,
                    "reasons": item.reasons,
                }
                for item in output.rejected_jobs
            ],
        },
    )
    logger.info(
        "Filtering complete: %d accepted, %d rejected of %d jobs.",
        len(output.accepted_jobs),
        len(output.rejected_jobs),
        len(inp.jobs),
    )
    return output
