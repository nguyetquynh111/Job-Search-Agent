"""LLM support for fit analysis.

This keeps prompt construction, the DeepInfra client, structured-output parsing,
and rationale-only rewrites together.  The deterministic pipeline still owns all
final validation.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from typing import Any

from pydantic import ValidationError
from pydantic.types import SecretStr

from src.config import get_config
from src.domain import EvidenceClaim, ProjectSwap
from src.tracing.langfuse import TraceManager
from src.tools.fit_analysis.contracts import AnalyzeFitInput, FitAnalysisOutput
from src.tools.fit_analysis.evidence_index import EvidenceIndex
from src.tools.fit_analysis.prepass import (
    PrePass,
    build_fallback_output,
    prepass_summary,
)
import src.tools.fit_analysis.rules as rules
from src.tools.fit_analysis.swap import normalize_name

logger = logging.getLogger(__name__)

CompleteFn = Callable[[str, str], str]

SYSTEM_PROMPT = """\
Perform the fit-analysis operation inside the single-agent job-search workflow. \
Explain how well ONE candidate fits ONE job, using ONLY the facts supplied to you. \
This operation may reason about fit, but it does not select or orchestrate tools.

Hard rules:
1. NEVER invent facts. Reason only over the job, candidate profile, portfolio, \
and evidence index provided in the user message. Do not retrieve anything.
2. Every claim you make must cite evidence_ids that appear in the supplied \
evidence index. Do not cite an ID that was not given to you. A skill with no \
supporting evidence has an empty evidence_ids list and belongs in genuine_gaps.
3. Split required skills into THREE disjoint buckets:
   - aligned_skills: required by the job AND already present on the resume.
   - evidenced_missing_skills: required by the job, NOT on the resume, BUT \
evidenced elsewhere (portfolio, master skills, or memory). These are the ONLY \
skills the resume tool may add later, so each MUST name the skill and cite the \
evidence ID that justifies it.
   - genuine_gaps: required by the job with NO supporting evidence anywhere.
   A skill must appear in at most one bucket. Absence from the resume alone is \
NOT a genuine gap if portfolio, master-skills, or memory evidence supports it.
4. Memory evidence (source "memory") is first-class: treat it exactly like \
portfolio or master-skills evidence.
5. You NEVER output a numeric match score; scoring is a separate tool that \
already ran. Do not rank the job.
6. For project_swap, you may only propose a project that exists in the supplied \
portfolio. remove_project must be an existing resume project name; add_project \
must be an existing portfolio PROJECT_NAME. If the current projects are already \
the best available, set project_swap to null and say so in project_analysis. \
(The tool finalizes the project section deterministically from a shared ranking; \
still provide your best project_analysis.)
7. In every claim's "notes", BEGIN with a verdict tag: "verdict=match", \
"verdict=partial", or "verdict=mismatch" (use "verdict=missing" for \
evidenced_missing_skills), followed by "; " and a short human note. Derive the \
verdict from the actual comparison — NEVER mark a shortfall (e.g. 4 years vs 5+ \
required) as a match; that is "verdict=partial".
8. For relevant_experience, write ONE CONTRASTIVE claim per role. Each claim MUST: \
name the role and company, state what that experience centered on, and judge how \
it aligns — or does not — with THIS posting's focus. Example claim value: \
"Senior AI Engineer at Nimbus AI Products: centered on production RAG and LLM \
chatbot delivery — strong overlap with the job's focus on generative AI, weaker on \
its recommendation-systems requirement." Use "verdict=partial" or \
"verdict=mismatch" when alignment is weak, not "verdict=match". Do NOT output a \
bare label like "Years of Experience".

Return ONLY a JSON object (no prose, no code fences) with exactly these keys:
job_id (string),
relevant_experience, seniority, education, aligned_skills, \
evidenced_missing_skills, genuine_gaps, project_analysis \
(each a list of {"claim": string, "evidence_ids": [string], \
"confidence": number 0..1, "notes": string or null}),
project_swap (null or {"remove_project": string or null, "add_project": string, \
"rationale": string, "evidence_ids": [string]}).
Format each skill claim as "<Skill>: <reason>".

EVERY one of those seven fields is a JSON ARRAY, even when it holds exactly one \
element — write "seniority": [{...}], NEVER "seniority": {...}. Only project_swap \
is an object or null.
"""

SENIORITY_RATIONALE_SYSTEM = """\
You rewrite a FIXED seniority finding as one natural sentence for a candidate. The \
verdict has already been decided deterministically from a years comparison; you \
MUST NOT change it or invent facts. If the verdict is "partial" or "mismatch", do \
NOT say the candidate meets, exceeds, or satisfies the requirement. Return ONLY the \
sentence, no preamble.
"""

SWAP_RATIONALE_SYSTEM = """\
You write ONE sentence explaining a project swap that has ALREADY been decided by \
the tool. You MUST recommend exactly the given add-project (never a different one), \
and your sentence MUST reference technology, domain, and industry. Do not invent \
facts beyond those provided. Return ONLY the sentence, no preamble.
"""

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)
_POSITIVE_WORDS = (
    "exceed",
    "surpass",
    "more than enough",
    "fully meet",
    "over-qualified",
    "overqualified",
    "well above",
    "satisfies the requirement",
    "meets the requirement",
)
_NEGATIVE_WORDS = (
    "falls short",
    "below the",
    "less than the",
    "under-qualified",
    "does not meet",
)
_MAX_RATIONALE_CHARS = 400


def model_configured() -> bool:
    """Return whether an LLM model and API key are configured."""

    config = get_config()
    return bool(config.llm_model and config.deepinfra_api_key)


def complete(
    system: str,
    user: str,
    *,
    tracer: TraceManager | None = None,
    metadata: dict[str, Any] | None = None,
    generation_name: str = "fit_analysis_llm",
) -> str:
    """Call the configured DeepInfra chat model and return the raw text response."""

    from langchain_openai import ChatOpenAI

    config = get_config()
    llm = ChatOpenAI(
        model=config.llm_model,
        api_key=SecretStr(config.deepinfra_api_key),
        base_url=config.deepinfra_base_url,
        temperature=0,
    )
    messages = [("system", system), ("human", user)]
    try:
        response = llm.invoke(messages)
        content = response.content
        text = content if isinstance(content, str) else str(content)
        if tracer is not None:
            tracer.record_generation(
                {
                    "provider": "deepinfra",
                    "purpose": "fit_analysis",
                    **(metadata or {}),
                },
                name=generation_name,
                model=config.llm_model,
                messages=messages,
                response=text,
                usage=_usage(response),
                model_parameters={
                    "temperature": 0,
                    "base_url": config.deepinfra_base_url,
                },
            )
        return text
    except Exception as exc:
        if tracer is not None:
            tracer.record_generation(
                {
                    "provider": "deepinfra",
                    "purpose": "fit_analysis",
                    **(metadata or {}),
                },
                name=generation_name,
                model=config.llm_model,
                messages=messages,
                response={"error_type": exc.__class__.__name__},
                usage={},
                model_parameters={
                    "temperature": 0,
                    "base_url": config.deepinfra_base_url,
                },
                status="ERROR",
                error_type=exc.__class__.__name__,
            )
        raise


def _usage(message: Any) -> dict[str, int]:
    usage = getattr(message, "usage_metadata", None)
    if not usage:
        metadata = getattr(message, "response_metadata", {}) or {}
        usage = metadata.get("token_usage") or metadata.get("usage")
    if not isinstance(usage, dict):
        return {}
    normalized: dict[str, int] = {}
    for source, target in (
        ("prompt_tokens", "input_tokens"),
        ("completion_tokens", "output_tokens"),
        ("total_tokens", "total_tokens"),
        ("input_tokens", "input_tokens"),
        ("output_tokens", "output_tokens"),
    ):
        value = usage.get(source)
        if isinstance(value, int):
            normalized[target] = value
    return normalized


def build_seniority_rationale_prompt(claim_text: str, verdict_value: str) -> str:
    """Prompt to rephrase the fixed seniority finding, preserving its verdict."""

    return (
        f"Deterministic seniority finding (verdict={verdict_value}; do not change its "
        f"meaning): {claim_text}\nRewrite it as one clear sentence for the candidate."
    )


def build_swap_rationale_prompt(
    remove: str, add: str, tech: str, domain: str, industry: str
) -> str:
    """Prompt to write the rationale for an already-decided project swap."""

    return (
        f"The tool decided to replace the resume project '{remove}' with the portfolio "
        f"project '{add}'. Technology overlap with the job: {tech}. Domain alignment: "
        f"{domain}. Industry fit: {industry}. Write one sentence recommending this swap "
        f"that cites technology, domain, and industry."
    )


def build_user_prompt(inp: AnalyzeFitInput, grounding: dict[str, object]) -> str:
    """Build the user prompt with the job, candidate, portfolio, and grounding."""

    profile = inp.candidate_profile
    payload = {
        "job": {
            "job_id": inp.job.job_id,
            "title": inp.job.title,
            "company": inp.job.company,
            "industry_domain": inp.job.industry_domain,
            "required_skills": inp.job.required_skills,
            "years_experience_required": inp.job.years_experience_required,
            "experience_requirement_text": inp.job.experience_requirement.raw_text,
            "description": grounding.get("job_text", inp.job.description),
        },
        "candidate": {
            "name": profile.name,
            "resume_skills": profile.skills,
            "master_skills": profile.master_skills,
            "education": profile.education,
            "experience": profile.experience,
            "current_resume_projects": inp.current_resume_projects,
            "years_of_experience": profile.preferences.years_of_experience,
        },
        "portfolio_projects": [
            {
                "project_id": project.project_id,
                "name": project.name,
                "technologies": project.technologies,
                "domains": project.domains,
                "resume_swap_value": project.resume_swap_value,
                "evidence_ids": project.evidence_ids,
            }
            for project in inp.portfolio_projects
        ],
        "evidence_index": grounding.get("evidence_index", {}),
        "deterministic_grounding": {
            "aligned_skills": grounding.get("aligned_skills", []),
            "evidenced_missing_skills": grounding.get("evidenced_missing_skills", []),
            "genuine_gaps": grounding.get("genuine_gaps", []),
            "suggested_swap": grounding.get("suggested_swap"),
        },
        "job_evidence": [item.model_dump() for item in inp.job_evidence],
        "valid_evidence_ids": sorted(
            {item.evidence_id for item in [*inp.evidence_items, *inp.job_evidence]}
        ),
    }
    return (
        "Analyze the candidate's fit for this job. Use only the following data; "
        "cite only evidence_ids listed in valid_evidence_ids.\n"
        + json.dumps(payload, sort_keys=True, ensure_ascii=False)
    )


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

    system = SYSTEM_PROMPT
    user = build_user_prompt(inp, grounding)
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
    if active_complete is None and model_configured():
        active_complete = complete

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
    logger.info(
        "Fit analysis for %s used the deterministic fallback path.", inp.job.job_id
    )
    return output, "fallback", meta


def _clean(text: str) -> str:
    """Trim a model response to a single-line rationale."""

    return " ".join(text.strip().split())


def enrich_seniority(
    claims: list[EvidenceClaim], complete_fn: CompleteFn
) -> list[EvidenceClaim]:
    """Rewrite the seniority claim's prose without changing its verdict."""

    if not claims:
        return claims
    claim = claims[0]
    value, _human = rules.parse(claim.notes)
    try:
        raw = complete_fn(
            SENIORITY_RATIONALE_SYSTEM,
            build_seniority_rationale_prompt(claim.claim, value or "unknown"),
        )
        prose = _clean(raw)
    except Exception as exc:  # noqa: BLE001 - any LLM failure keeps the template
        logger.warning("Seniority rationale LLM call failed; keeping template: %s", exc)
        return claims

    if not seniority_prose_ok(prose, value):
        logger.info(
            "Seniority rationale rejected (contradicts verdict); keeping template."
        )
        return claims
    return [claim.model_copy(update={"claim": prose})]


def seniority_prose_ok(prose: str, value: str | None) -> bool:
    """Reject prose that is empty, too long, or contradicts the fixed verdict."""

    if not prose or len(prose) > _MAX_RATIONALE_CHARS:
        return False
    lowered = prose.lower()
    if value in {rules.PARTIAL, rules.MISMATCH} and any(
        w in lowered for w in _POSITIVE_WORDS
    ):
        return False
    if value == rules.MATCH and any(w in lowered for w in _NEGATIVE_WORDS):
        return False
    return True


def enrich_swap(
    swap: ProjectSwap | None,
    prepass: PrePass,
    inp: AnalyzeFitInput,
    complete_fn: CompleteFn,
) -> ProjectSwap | None:
    """Rewrite the swap rationale prose, constrained to the decided add/remove pair."""

    if swap is None or prepass.swap is None:
        return swap
    add_score = next(
        (s for s in prepass.swap.rankings if s.project.name == swap.add_project), None
    )
    if add_score is None:
        return swap
    tech = ", ".join(add_score.matched_distinctive) or "shared general tooling only"
    domain = ", ".join(add_score.domain_matches) or "broader domain overlap"
    industry = (
        ", ".join(add_score.industry_matches) or "transferable industry experience"
    )
    try:
        raw = complete_fn(
            SWAP_RATIONALE_SYSTEM,
            build_swap_rationale_prompt(
                swap.remove_project or "", swap.add_project, tech, domain, industry
            ),
        )
        prose = _clean(raw)
    except Exception as exc:  # noqa: BLE001 - any LLM failure keeps the template
        logger.warning("Swap rationale LLM call failed; keeping template: %s", exc)
        return swap

    if not _swap_prose_ok(prose, swap, inp):
        logger.info(
            "Swap rationale rejected (drifts from decided project); keeping template."
        )
        return swap
    return swap.model_copy(update={"rationale": prose})


def _swap_prose_ok(prose: str, swap: ProjectSwap, inp: AnalyzeFitInput) -> bool:
    """Reject prose that omits the add project or names a different portfolio project."""

    if not prose or len(prose) > _MAX_RATIONALE_CHARS:
        return False
    normalized = normalize_name(prose)
    if normalize_name(swap.add_project) not in normalized:
        return False
    for project in inp.portfolio_projects:
        if project.name in {swap.add_project, swap.remove_project}:
            continue
        if normalize_name(project.name) in normalized:
            return False
    return True
