"""Post-validation applied to both the LLM and fallback outputs.

Enforces the tool's guarantees regardless of which path produced the analysis:
evidence IDs must exist, the three skill buckets are disjoint, swaps reference
real projects, the job IDs match, and confidence is assigned by the documented
source-kind rule rather than left arbitrary. Every repair is logged.
"""

from __future__ import annotations

import logging

from src.schemas.common import EvidenceClaim
from src.schemas.fit_analysis import AnalyzeFitInput, FitAnalysisOutput
from src.tools.implementations.fit_analysis import confidence, verdict
from src.tools.implementations.fit_analysis.aliases import canonicalize
from src.tools.implementations.fit_analysis.evidence_index import source_kind
from src.tools.implementations.fit_analysis.prepass import _seniority_verdict

logger = logging.getLogger(__name__)

# Skill-bucket priority for disjointness: aligned beats evidenced-missing beats gap.
_BUCKET_PRIORITY = {"aligned": 3, "evidenced_missing": 2, "genuine_gap": 1}


def _skill_key(claim: EvidenceClaim) -> str:
    """Return the canonical skill token a skill claim is about."""

    head = claim.claim.split(":", 1)[0]
    return canonicalize(head)


def _valid_ids(claim: EvidenceClaim, known_ids: set[str], repairs: list[str]) -> list[str]:
    """Return only the cited evidence IDs that actually exist, logging drops."""

    kept: list[str] = []
    for evidence_id in claim.evidence_ids:
        if evidence_id in known_ids:
            kept.append(evidence_id)
        else:
            repairs.append(f"dropped unknown evidence_id '{evidence_id}'")
    return kept


def post_validate(
    output: FitAnalysisOutput, inp: AnalyzeFitInput
) -> tuple[FitAnalysisOutput, list[str]]:
    """Return a repaired FitAnalysisOutput plus a list of repair descriptions."""

    repairs: list[str] = []
    known_ids = {item.evidence_id for item in inp.evidence_items}
    kind_of = {item.evidence_id: source_kind(item.source) for item in inp.evidence_items}

    # --- aligned (a pass marker REQUIRES resume evidence) ---
    # Mirrors the evidenced-missing demotion below so no bucket can assert an
    # ungrounded claim: "already on your resume" needs a resume-sourced citation,
    # evidence from elsewhere means the skill is evidenced-missing instead, and no
    # valid citation at all means the tool found no support for it.
    aligned: list[EvidenceClaim] = []
    promoted_to_missing: list[EvidenceClaim] = []
    demoted_from_aligned: list[EvidenceClaim] = []
    for claim in output.aligned_skills:
        ids = _valid_ids(claim, known_ids, repairs)
        sources = {kind_of[i] for i in ids}
        skill = claim.claim.split(":", 1)[0]
        if confidence.RESUME in sources:
            aligned.append(
                claim.model_copy(
                    update={
                        "evidence_ids": ids,
                        "confidence": confidence.skill_confidence(sources, on_resume=True),
                        "notes": _reverdict(claim.notes, verdict.MATCH),
                    }
                )
            )
        elif ids:
            where = ", ".join(sorted(sources))
            repairs.append(
                f"moved '{skill}' from aligned to evidenced_missing (no resume evidence)"
            )
            promoted_to_missing.append(
                claim.model_copy(
                    update={
                        "claim": (
                            f"{skill}: required by the job, not yet on your resume, "
                            f"but evidenced in {where}."
                        ),
                        "evidence_ids": ids,
                    }
                )
            )
        else:
            repairs.append(f"demoted '{skill}' from aligned to genuine_gaps (no valid evidence)")
            demoted_from_aligned.append(
                EvidenceClaim(
                    claim=f"{skill}: required by the job with no valid supporting evidence.",
                    evidence_ids=[],
                    confidence=confidence.skill_confidence(set(), on_resume=False),
                    notes=verdict.tag(
                        verdict.MISMATCH, "Demoted from aligned during post-validation."
                    ),
                )
            )

    # --- evidenced-missing (demote to gaps when no valid evidence remains) ---
    evidenced_missing: list[EvidenceClaim] = []
    demoted: list[EvidenceClaim] = []
    for claim in [*output.evidenced_missing_skills, *promoted_to_missing]:
        ids = _valid_ids(claim, known_ids, repairs)
        if not ids:
            skill = claim.claim.split(":", 1)[0]
            repairs.append(f"demoted '{skill}' to genuine_gaps (no valid evidence)")
            demoted.append(
                EvidenceClaim(
                    claim=f"{skill}: required by the job with no valid supporting evidence.",
                    evidence_ids=[],
                    confidence=confidence.skill_confidence(set(), on_resume=False),
                    notes=verdict.tag(
                        verdict.MISMATCH, "Demoted from evidenced-missing during post-validation."
                    ),
                )
            )
            continue
        sources = {kind_of[i] for i in ids}
        evidenced_missing.append(
            claim.model_copy(
                update={
                    "evidence_ids": ids,
                    "confidence": confidence.skill_confidence(sources, on_resume=False),
                    "notes": _reverdict(claim.notes, verdict.MISSING),
                }
            )
        )

    # --- genuine gaps ---
    genuine_gaps: list[EvidenceClaim] = []
    for claim in [*output.genuine_gaps, *demoted, *demoted_from_aligned]:
        ids = _valid_ids(claim, known_ids, repairs)
        genuine_gaps.append(
            claim.model_copy(
                update={
                    "evidence_ids": ids,
                    "confidence": confidence.skill_confidence(set(), on_resume=False),
                    "notes": _reverdict(claim.notes, verdict.MISMATCH),
                }
            )
        )

    aligned, evidenced_missing, genuine_gaps = _enforce_disjoint(
        aligned, evidenced_missing, genuine_gaps, repairs
    )

    required_min = inp.job.years_experience_required
    if required_min is None:
        required_min = inp.job.experience_requirement.minimum_years
    seniority_verdict = _seniority_verdict(
        inp.candidate_profile.preferences.years_of_experience, required_min
    )

    updated = output.model_copy(
        update={
            "job_id": inp.job.job_id,
            "relevant_experience": _fix_narrative(output.relevant_experience, known_ids, repairs),
            "seniority": _fix_narrative(
                output.seniority, known_ids, repairs, forced=seniority_verdict
            ),
            "education": _fix_narrative(output.education, known_ids, repairs),
            "aligned_skills": aligned,
            "evidenced_missing_skills": evidenced_missing,
            "genuine_gaps": genuine_gaps,
            "project_analysis": _fix_project_analysis(output.project_analysis, known_ids, repairs),
            "project_swap": _fix_swap(output, inp, repairs),
        }
    )
    if output.job_id != inp.job.job_id:
        repairs.append(f"forced job_id to '{inp.job.job_id}'")
    if repairs:
        logger.info("Fit-analysis post-validation repaired %d issue(s): %s", len(repairs), repairs)
    return updated, repairs


def _enforce_disjoint(
    aligned: list[EvidenceClaim],
    evidenced_missing: list[EvidenceClaim],
    genuine_gaps: list[EvidenceClaim],
    repairs: list[str],
) -> tuple[list[EvidenceClaim], list[EvidenceClaim], list[EvidenceClaim]]:
    """Keep each skill in only its highest-priority bucket."""

    best: dict[str, int] = {}
    for claims, name in (
        (aligned, "aligned"),
        (evidenced_missing, "evidenced_missing"),
        (genuine_gaps, "genuine_gap"),
    ):
        for claim in claims:
            key = _skill_key(claim)
            best[key] = max(best.get(key, 0), _BUCKET_PRIORITY[name])

    def _filter(claims: list[EvidenceClaim], name: str) -> list[EvidenceClaim]:
        kept: list[EvidenceClaim] = []
        for claim in claims:
            key = _skill_key(claim)
            if best.get(key) == _BUCKET_PRIORITY[name]:
                kept.append(claim)
            else:
                repairs.append(f"removed '{key}' from {name} (kept in higher-priority bucket)")
        return kept

    return (
        _filter(aligned, "aligned"),
        _filter(evidenced_missing, "evidenced_missing"),
        _filter(genuine_gaps, "genuine_gap"),
    )


def _reverdict(notes: str | None, value: str) -> str:
    """Force a claim's verdict to ``value`` while preserving its human note."""

    _, human = verdict.parse(notes)
    return verdict.tag(value, human)


def _fix_narrative(
    claims: list[EvidenceClaim],
    known_ids: set[str],
    repairs: list[str],
    *,
    forced: str | None = None,
    default: str = verdict.PARTIAL,
) -> list[EvidenceClaim]:
    """Drop unknown IDs, reassign confidence, and ensure a non-defaulting verdict.

    A forced verdict overrides the claim (used for seniority, computed
    authoritatively from years). Otherwise a claim's own valid verdict is kept and
    a missing/invalid one falls back to ``default`` — never to a positive marker.
    """

    fixed: list[EvidenceClaim] = []
    for claim in claims:
        ids = _valid_ids(claim, known_ids, repairs)
        existing, human = verdict.parse(claim.notes)
        value = forced or existing or default
        fixed.append(
            claim.model_copy(
                update={
                    "evidence_ids": ids,
                    "confidence": confidence.narrative_confidence(len(ids)),
                    "notes": verdict.tag(value, human),
                }
            )
        )
    return fixed


def _fix_project_analysis(
    claims: list[EvidenceClaim], known_ids: set[str], repairs: list[str]
) -> list[EvidenceClaim]:
    """Drop unknown IDs from project-analysis claims and clamp confidence."""

    fixed: list[EvidenceClaim] = []
    for claim in claims:
        ids = _valid_ids(claim, known_ids, repairs)
        clamped = min(1.0, max(0.0, claim.confidence))
        fixed.append(claim.model_copy(update={"evidence_ids": ids, "confidence": clamped}))
    return fixed


def _fix_swap(output: FitAnalysisOutput, inp: AnalyzeFitInput, repairs: list[str]):
    """Null a swap that references a non-existent project; drop unknown IDs."""

    swap = output.project_swap
    if swap is None:
        return None
    portfolio_names = {project.name for project in inp.portfolio_projects}
    if swap.add_project not in portfolio_names:
        repairs.append(f"nulled swap: add_project '{swap.add_project}' not in portfolio")
        return None
    if swap.remove_project is not None and swap.remove_project not in inp.current_resume_projects:
        repairs.append(
            f"nulled swap: remove_project '{swap.remove_project}' not a current resume project"
        )
        return None
    known_ids = {item.evidence_id for item in inp.evidence_items}
    ids = [i for i in swap.evidence_ids if i in known_ids]
    return swap.model_copy(update={"evidence_ids": ids})
