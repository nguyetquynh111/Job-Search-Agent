"""Merge an LLM proposal with deterministic guards: propose -> validate -> fall back.

The model owns the fit judgments. This module keeps the model's bucket assignment
and its project swap, then enforces the invariants the deterministic pass used to
enforce by overwriting them:

* **Completeness.** Every required skill must land in exactly one bucket. Any
  requirement the model omitted is filled in from the deterministic pre-pass, so an
  LLM proposal can never produce a *less* complete analysis than the fallback.
* **Swap safety.** The model's ``add_project`` is honoured only when it exists in
  the portfolio, is not already on the resume, and beats the outgoing project by the
  same ranking margin the deterministic path requires. Otherwise the ranked choice
  is used and the rejection is logged with its reason.
* **Seniority consistency.** The model's seniority prose is kept only when it does
  not contradict the deterministic years verdict.

Grounding (evidence IDs must exist, buckets disjoint, confidence by source kind) is
left to :mod:`postvalidate`, which runs on both paths.
"""

from __future__ import annotations

import logging

from src.schemas.common import EvidenceClaim
from src.schemas.fit_analysis import AnalyzeFitInput, FitAnalysisOutput
from src.tools.implementations.fit_analysis import rationale, swap as swap_module
from src.tools.implementations.fit_analysis.aliases import canonicalize
from src.tools.implementations.fit_analysis.prepass import (
    PrePass,
    _dedupe_canonical,
    _seniority_verdict,
    build_project_section,
    build_skill_section,
)

logger = logging.getLogger(__name__)

_BUCKET_FIELDS = ("aligned_skills", "evidenced_missing_skills", "genuine_gaps")


def _skill_key(claim: EvidenceClaim) -> str:
    """Return the canonical skill token a skill claim is about."""

    return canonicalize(claim.claim.split(":", 1)[0])


def _deterministic_by_skill(
    prepass: PrePass,
) -> dict[str, tuple[str, EvidenceClaim]]:
    """Map canonical skill -> (bucket field name, deterministic claim)."""

    aligned, evidenced_missing, gaps = build_skill_section(prepass)
    mapping: dict[str, tuple[str, EvidenceClaim]] = {}
    for field, entries, claims in (
        ("aligned_skills", prepass.aligned, aligned),
        ("evidenced_missing_skills", prepass.evidenced_missing, evidenced_missing),
        ("genuine_gaps", prepass.genuine_gaps, gaps),
    ):
        for entry, claim in zip(entries, claims):
            mapping[canonicalize(entry.skill)] = (field, claim)
    return mapping


def _merge_skill_buckets(
    llm_output: FitAnalysisOutput, inp: AnalyzeFitInput, prepass: PrePass, repairs: list[str]
) -> dict[str, list[EvidenceClaim]]:
    """Keep the model's buckets and fill in any required skill it left out."""

    buckets = {field: list(getattr(llm_output, field)) for field in _BUCKET_FIELDS}
    covered = {
        _skill_key(claim) for field in _BUCKET_FIELDS for claim in buckets[field]
    }
    deterministic = _deterministic_by_skill(prepass)

    for _original, canonical in _dedupe_canonical(inp.job.required_skills):
        if canonical in covered:
            continue
        fallback = deterministic.get(canonical)
        if fallback is None:
            continue
        field, claim = fallback
        buckets[field].append(claim)
        covered.add(canonical)
        repairs.append(f"filled omitted required skill '{canonical}' from the deterministic pass")
    return buckets


def _merge_seniority(
    llm_output: FitAnalysisOutput, inp: AnalyzeFitInput, prepass: PrePass, repairs: list[str]
) -> list[EvidenceClaim]:
    """Keep the model's seniority claim unless its prose contradicts the years verdict."""

    if not llm_output.seniority:
        return prepass.seniority

    required = inp.job.years_experience_required
    if required is None:
        required = inp.job.experience_requirement.minimum_years
    decided = _seniority_verdict(inp.candidate_profile.preferences.years_of_experience, required)

    claim = llm_output.seniority[0]
    if not rationale.seniority_prose_ok(claim.claim, decided):
        repairs.append(
            f"replaced LLM seniority prose (contradicts deterministic verdict '{decided}')"
        )
        return prepass.seniority
    return llm_output.seniority


def _merge_project_section(
    llm_output: FitAnalysisOutput, inp: AnalyzeFitInput, prepass: PrePass, repairs: list[str]
):
    """Honour the model's swap when it validates, else use the ranked choice.

    Returns ``(project_analysis, swap, model_chose_swap)``. ``model_chose_swap`` lets
    the caller tell whether the rationale is the model's own prose or a template that
    is still worth having the model phrase.
    """

    decision = prepass.swap
    if decision is None:
        return list(llm_output.project_analysis), None, False

    model_chose = False
    proposed = llm_output.project_swap
    if proposed is not None:
        add, remove_name, reason = swap_module.validate_proposed_swap(
            decision,
            proposed.remove_project,
            proposed.add_project,
            inp.current_resume_projects,
            inp.portfolio_projects,
        )
        if add is not None and remove_name is not None:
            decision = swap_module.choose_swap(decision, remove_name, add, proposed.rationale)
            model_chose = True
            logger.info(
                "Accepted the model's project swap: %r -> %r", remove_name, add.name
            )
        else:
            repairs.append(f"rejected the model's project swap: {reason}")
            logger.info("Rejected the model's project swap (%s); using the ranked choice.", reason)
    elif decision.swap is not None:
        repairs.append(
            "model proposed no swap but the ranking found a materially better project; "
            "used the ranked choice"
        )

    # Rebuilt from the (possibly re-marked) shared ranking so the removed project is
    # always the one reported as weak.
    claims, swap = build_project_section(decision)
    return claims, swap, model_chose


def merge_llm_proposal(
    llm_output: FitAnalysisOutput,
    inp: AnalyzeFitInput,
    prepass: PrePass,
) -> tuple[FitAnalysisOutput, list[str], bool]:
    """Return the model's analysis with deterministic guards applied.

    Yields ``(merged, repairs, model_chose_swap)``. ``relevant_experience`` and
    ``education`` are the model's own narrative and pass through untouched;
    :mod:`postvalidate` still strips any unknown evidence ID.
    """

    repairs: list[str] = []
    buckets = _merge_skill_buckets(llm_output, inp, prepass, repairs)
    project_analysis, project_swap, model_chose_swap = _merge_project_section(
        llm_output, inp, prepass, repairs
    )
    merged = llm_output.model_copy(
        update={
            **buckets,
            "seniority": _merge_seniority(llm_output, inp, prepass, repairs),
            "project_analysis": project_analysis,
            "project_swap": project_swap,
        }
    )
    if repairs:
        logger.info("Merged LLM proposal with %d deterministic guard(s): %s", len(repairs), repairs)
    return merged, repairs, model_chose_swap
