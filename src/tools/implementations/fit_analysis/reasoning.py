"""Choose the LLM reasoning path or the deterministic fallback.

Both paths return a ``FitAnalysisOutput`` that is subsequently post-validated, so
downstream behaviour is identical regardless of which path ran. The chosen path
is reported to the caller so it can be logged and traced unmistakably.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable

from pydantic import ValidationError

from src.config import get_config
from src.schemas.fit_analysis import AnalyzeFitInput, FitAnalysisOutput
from src.tools.implementations.fit_analysis import llm_client, prompts
from src.tools.implementations.fit_analysis.evidence_index import EvidenceIndex
from src.tools.implementations.fit_analysis.prepass import (
    PrePass,
    build_fallback_output,
    prepass_summary,
)

logger = logging.getLogger(__name__)

CompleteFn = Callable[[str, str], str]

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def _strip_fences(text: str) -> str:
    """Remove Markdown code fences a model may wrap JSON in."""

    return _FENCE.sub("", text).strip()


def _parse_output(text: str) -> FitAnalysisOutput:
    """Parse and validate a model response into a FitAnalysisOutput."""

    data = json.loads(_strip_fences(text))
    return FitAnalysisOutput.model_validate(data)


def _run_llm(
    inp: AnalyzeFitInput, grounding: dict[str, object], complete_fn: CompleteFn
) -> FitAnalysisOutput:
    """Call the model for structured output, retrying once on a validation error."""

    system = prompts.SYSTEM_PROMPT
    user = prompts.build_user_prompt(inp, grounding)
    try:
        return _parse_output(complete_fn(system, user))
    except (ValidationError, json.JSONDecodeError, ValueError) as exc:
        logger.warning("Fit-analysis LLM output invalid; retrying once: %s", exc)
        retry_user = (
            f"{user}\n\nYour previous response could not be parsed into the required "
            f"schema. Error:\n{exc}\nReturn ONLY corrected JSON."
        )
        return _parse_output(complete_fn(system, retry_user))


def analyze(
    inp: AnalyzeFitInput,
    prepass: PrePass,
    index: EvidenceIndex,
    *,
    complete_fn: CompleteFn | None = None,
) -> tuple[FitAnalysisOutput, str, dict[str, object]]:
    """Return (output, path_label, meta) using the LLM when possible, else fallback.

    ``path_label`` is "llm" or "fallback". A stub ``complete_fn`` forces the LLM
    path in tests; with no model configured and no stub, the fallback runs.
    """

    grounding = prepass_summary(prepass, index)
    grounding["job_text"] = prepass.job_text

    active_complete = complete_fn
    if active_complete is None and llm_client.model_configured():
        active_complete = llm_client.complete

    meta: dict[str, object] = {"model": get_config().llm_model or None}
    if active_complete is not None:
        try:
            output = _run_llm(inp, grounding, active_complete)
            meta["system_prompt_name"] = "fit_analysis.SYSTEM_PROMPT"
            logger.info("Fit analysis for %s used the LLM path.", inp.job.job_id)
            return output, "llm", meta
        except Exception as exc:  # noqa: BLE001 - any LLM failure must degrade safely
            logger.warning(
                "Fit-analysis LLM path failed for %s; using deterministic fallback: %s",
                inp.job.job_id,
                exc,
            )
            meta["llm_error"] = exc.__class__.__name__

    output = build_fallback_output(inp, prepass)
    logger.info("Fit analysis for %s used the deterministic fallback path.", inp.job.job_id)
    return output, "fallback", meta
