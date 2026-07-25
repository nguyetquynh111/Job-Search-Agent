"""LLM-written rationale prose for decisions the tool already made deterministically.

The tool decides seniority (a years comparison) and the project swap (a ranking).
On the LLM path only, the model is asked to write the *prose* for that fixed
outcome — it never chooses. If the prose contradicts the deterministic verdict or
drifts to a different project, the templated text is kept. The fallback path never
calls the model and always uses the templates.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from src.schemas.common import EvidenceClaim, ProjectSwap
from src.schemas.fit_analysis import AnalyzeFitInput
from src.tools.implementations.fit_analysis import prompts, verdict
from src.tools.implementations.fit_analysis.prepass import PrePass
from src.tools.implementations.fit_analysis.swap import normalize_name

logger = logging.getLogger(__name__)

CompleteFn = Callable[[str, str], str]

_POSITIVE_WORDS = (
    "exceed", "surpass", "more than enough", "fully meet", "over-qualified",
    "overqualified", "well above", "satisfies the requirement", "meets the requirement",
)
_NEGATIVE_WORDS = ("falls short", "below the", "less than the", "under-qualified", "does not meet")
_MAX_RATIONALE_CHARS = 400


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
    value, human = verdict.parse(claim.notes)
    try:
        raw = complete_fn(
            prompts.SENIORITY_RATIONALE_SYSTEM,
            prompts.build_seniority_rationale_prompt(claim.claim, value or "unknown"),
        )
        prose = _clean(raw)
    except Exception as exc:  # noqa: BLE001 - any LLM failure keeps the template
        logger.warning("Seniority rationale LLM call failed; keeping template: %s", exc)
        return claims

    if not _seniority_prose_ok(prose, value):
        logger.info("Seniority rationale rejected (contradicts verdict); keeping template.")
        return claims
    return [claim.model_copy(update={"claim": prose})]


def _seniority_prose_ok(prose: str, value: str | None) -> bool:
    """Reject prose that is empty, too long, or contradicts the fixed verdict."""

    if not prose or len(prose) > _MAX_RATIONALE_CHARS:
        return False
    lowered = prose.lower()
    if value in {verdict.PARTIAL, verdict.MISMATCH} and any(w in lowered for w in _POSITIVE_WORDS):
        return False
    if value == verdict.MATCH and any(w in lowered for w in _NEGATIVE_WORDS):
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
    industry = ", ".join(add_score.industry_matches) or "transferable industry experience"
    try:
        raw = complete_fn(
            prompts.SWAP_RATIONALE_SYSTEM,
            prompts.build_swap_rationale_prompt(
                swap.remove_project or "", swap.add_project, tech, domain, industry
            ),
        )
        prose = _clean(raw)
    except Exception as exc:  # noqa: BLE001 - any LLM failure keeps the template
        logger.warning("Swap rationale LLM call failed; keeping template: %s", exc)
        return swap

    if not _swap_prose_ok(prose, swap, inp):
        logger.info("Swap rationale rejected (drifts from decided project); keeping template.")
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
            return False  # drifted to a different project
    return True
