"""Post-validation applied to both the LLM and fallback outputs.

Enforces the tool's guarantees regardless of which path produced the analysis:
evidence IDs must exist, the three skill buckets are disjoint, swaps reference
real projects, the job IDs match, and confidence is assigned by the documented
source-kind rule rather than left arbitrary. Every repair is logged.
"""

from __future__ import annotations

import logging
import re

from src.schemas.common import EvidenceClaim, EvidenceItem
from src.schemas.fit_analysis import AnalyzeFitInput, FitAnalysisOutput
from src.tools.evidence_validation import (
    CANDIDATE_SOURCES,
    JOB_SOURCES,
    evidence_supports_project,
    supporting_ids_for_skill,
    supporting_ids_for_statement,
)
from src.tools.fit_analysis import confidence, verdict
from src.tools.fit_analysis.aliases import canonicalize, category_members, skill_in_text
from src.tools.fit_analysis.evidence_index import source_kind
from src.tools.fit_analysis.prepass import _seniority_verdict
from src.tools.job_evidence import (
    build_job_evidence,
    job_evidence_id,
    job_skill_evidence_id,
)

logger = logging.getLogger(__name__)

# Prefer the strongest supported bucket when claims overlap.
_BUCKET_PRIORITY = {"aligned": 3, "evidenced_missing": 2, "genuine_gap": 1}


def _skill_key(claim: EvidenceClaim) -> str:
    """Return the canonical skill token a skill claim is about."""

    head = claim.claim.split(":", 1)[0]
    return canonicalize(head)


def _valid_ids(
    claim: EvidenceClaim, known_ids: set[str], repairs: list[str]
) -> list[str]:
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
    job_evidence = inp.job_evidence or build_job_evidence(inp.job)
    evidence_by_id = {item.evidence_id: item for item in inp.evidence_items}
    for project in inp.portfolio_projects:
        for evidence_id in project.evidence_ids:
            evidence_by_id.setdefault(
                evidence_id,
                EvidenceItem(
                    evidence_id=evidence_id,
                    source="portfolio",
                    text=(
                        f"PROJECT_NAME: {project.name}\n"
                        f"SUMMARY: {project.description}\n"
                        f"TECH_STACK: {'; '.join(project.technologies)}"
                    ),
                    tags=[*project.technologies, *project.keywords, "project"],
                    metadata={
                        "project_id": project.project_id,
                        "project_name": project.name,
                    },
                ),
            )
    evidence_by_id.update({item.evidence_id: item for item in job_evidence})
    portfolio_ids = {
        evidence_id
        for project in inp.portfolio_projects
        for evidence_id in project.evidence_ids
    }
    candidate_ids = {item.evidence_id for item in inp.evidence_items} | portfolio_ids
    job_ids = {item.evidence_id for item in job_evidence}
    known_ids = candidate_ids | job_ids
    kind_of = {
        item.evidence_id: source_kind(item.source) for item in inp.evidence_items
    }
    kind_of.update({evidence_id: confidence.PORTFOLIO for evidence_id in portfolio_ids})

    # Aligned skills need resume evidence.
    aligned: list[EvidenceClaim] = []
    promoted_to_missing: list[EvidenceClaim] = []
    demoted_from_aligned: list[EvidenceClaim] = []
    for claim in output.aligned_skills:
        ids = _valid_ids(claim, known_ids, repairs)
        skill = claim.claim.split(":", 1)[0]
        requirement = _job_requirement(inp, skill)
        if requirement is None:
            repairs.append(
                f"rejected unsupported skill claim '{skill}': not a job requirement"
            )
            continue
        requirement_id = requirement[1]
        ids = list(dict.fromkeys([requirement_id, *ids]))
        support_ids = supporting_ids_for_skill(
            skill, ids, evidence_by_id, sources=CANDIDATE_SOURCES
        )
        rejected_ids = [
            evidence_id
            for evidence_id in ids
            if evidence_id in candidate_ids and evidence_id not in support_ids
        ]
        if rejected_ids:
            repairs.append(
                f"rejected semantically unrelated evidence for '{skill}': {rejected_ids}"
            )
        sources = {kind_of[i] for i in support_ids}
        if confidence.RESUME in sources:
            aligned.append(
                claim.model_copy(
                    update={
                        "evidence_ids": ids,
                        "confidence": confidence.skill_confidence(
                            sources, on_resume=True
                        ),
                        "notes": _reverdict(claim.notes, verdict.MATCH),
                    }
                )
            )
        elif support_ids:
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
            repairs.append(
                f"demoted '{skill}' from aligned to genuine_gaps (no valid evidence)"
            )
            demoted_from_aligned.append(
                EvidenceClaim(
                    claim=f"{skill}: required by the job with no valid supporting evidence.",
                    evidence_ids=[requirement_id],
                    confidence=confidence.skill_confidence(set(), on_resume=False),
                    notes=verdict.tag(
                        verdict.MISMATCH, "Demoted from aligned during post-validation."
                    ),
                )
            )

    # Move unsupported off-resume skills to gaps.
    evidenced_missing: list[EvidenceClaim] = []
    demoted: list[EvidenceClaim] = []
    for claim in [*output.evidenced_missing_skills, *promoted_to_missing]:
        ids = _valid_ids(claim, known_ids, repairs)
        skill = claim.claim.split(":", 1)[0]
        requirement = _job_requirement(inp, skill)
        if requirement is None:
            repairs.append(
                f"rejected unsupported keyword/skill '{skill}': not a job requirement"
            )
            continue
        requirement_id = requirement[1]
        ids = list(dict.fromkeys([requirement_id, *ids]))
        support_ids = supporting_ids_for_skill(
            skill, ids, evidence_by_id, sources=CANDIDATE_SOURCES
        )
        rejected_ids = [
            evidence_id
            for evidence_id in ids
            if evidence_id in candidate_ids and evidence_id not in support_ids
        ]
        if rejected_ids:
            repairs.append(
                f"rejected semantically unrelated evidence for '{skill}': {rejected_ids}"
            )
        if not support_ids:
            repairs.append(
                f"demoted '{skill}' to genuine_gaps (no semantically matching evidence)"
            )
            demoted.append(
                EvidenceClaim(
                    claim=f"{skill}: required by the job with no valid supporting evidence.",
                    evidence_ids=[requirement_id],
                    confidence=confidence.skill_confidence(set(), on_resume=False),
                    notes=verdict.tag(
                        verdict.MISMATCH,
                        "Demoted from evidenced-missing during post-validation.",
                    ),
                )
            )
            continue
        sources = {kind_of[i] for i in support_ids}
        evidenced_missing.append(
            claim.model_copy(
                update={
                    "evidence_ids": ids,
                    "confidence": confidence.skill_confidence(sources, on_resume=False),
                    "notes": _reverdict(claim.notes, verdict.MISSING),
                }
            )
        )

    # Gaps cite only the job requirement.
    genuine_gaps: list[EvidenceClaim] = []
    for claim in [*output.genuine_gaps, *demoted, *demoted_from_aligned]:
        skill = claim.claim.split(":", 1)[0]
        requirement = _job_requirement(inp, skill)
        if requirement is None:
            repairs.append(
                f"rejected unsupported gap/keyword '{skill}': not a job requirement"
            )
            continue
        requirement_id = requirement[1]
        _valid_ids(claim, known_ids, repairs)
        genuine_gaps.append(
            claim.model_copy(
                update={
                    # Candidate evidence would contradict a genuine gap.
                    "evidence_ids": [requirement_id],
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
    if not any(
        item.source == "resume"
        and "experience" in {tag.casefold() for tag in item.tags}
        for item in inp.evidence_items
    ):
        seniority_verdict = verdict.PARTIAL

    updated = output.model_copy(
        update={
            "job_id": inp.job.job_id,
            "relevant_experience": _fix_narrative(
                output.relevant_experience,
                known_ids,
                repairs,
                evidence_by_id=evidence_by_id,
                label="relevant experience",
                required_job_id=job_evidence_id(inp.job, "description"),
            ),
            "seniority": _fix_narrative(
                output.seniority,
                known_ids,
                repairs,
                evidence_by_id=evidence_by_id,
                label="seniority",
                forced=seniority_verdict,
                allow_profile_fact=True,
                required_job_id=(
                    job_evidence_id(inp.job, "experience")
                    if job_evidence_id(inp.job, "experience") in job_ids
                    else job_evidence_id(inp.job, "description")
                ),
            ),
            "education": _fix_narrative(
                output.education,
                known_ids,
                repairs,
                evidence_by_id=evidence_by_id,
                label="education",
                required_job_id=job_evidence_id(inp.job, "description"),
            ),
            "aligned_skills": aligned,
            "evidenced_missing_skills": evidenced_missing,
            "genuine_gaps": genuine_gaps,
            "project_analysis": _fix_project_analysis(
                output.project_analysis,
                known_ids,
                repairs,
                evidence_by_id,
                job_evidence_id(inp.job, "description"),
            ),
            "project_swap": _fix_swap(output, inp, repairs, job_evidence),
            "validation_failures": list(
                dict.fromkeys([*output.validation_failures, *repairs])
            ),
        }
    )
    if output.job_id != inp.job.job_id:
        repairs.append(f"forced job_id to '{inp.job.job_id}'")
    if repairs:
        logger.info(
            "Fit-analysis post-validation repaired %d issue(s): %s",
            len(repairs),
            repairs,
        )
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
                repairs.append(
                    f"removed '{key}' from {name} (kept in higher-priority bucket)"
                )
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
    evidence_by_id: dict[str, EvidenceItem],
    label: str,
    forced: str | None = None,
    default: str = verdict.PARTIAL,
    required_job_id: str = "",
    allow_profile_fact: bool = False,
) -> list[EvidenceClaim]:
    """Drop unknown IDs, reassign confidence, and ensure a non-defaulting verdict.

    A forced verdict overrides the claim (used for seniority, computed
    authoritatively from years). Otherwise a claim's own valid verdict is kept and
    a missing/invalid one falls back to ``default`` — never to a positive marker.
    """

    fixed: list[EvidenceClaim] = []
    for claim in claims:
        ids = _valid_ids(claim, known_ids, repairs)
        candidate_ids = [
            evidence_id
            for evidence_id in ids
            if evidence_id in evidence_by_id
            and evidence_by_id[evidence_id].source in CANDIDATE_SOURCES
        ]
        if allow_profile_fact:
            supported_candidate_ids = [
                evidence_id
                for evidence_id in candidate_ids
                if "experience"
                in {tag.casefold() for tag in evidence_by_id[evidence_id].tags}
            ]
        else:
            supported_candidate_ids = supporting_ids_for_statement(
                claim.claim,
                candidate_ids,
                evidence_by_id,
                sources=CANDIDATE_SOURCES,
            )
        safe_unverified = (
            allow_profile_fact
            and "could not be verified" in claim.claim.casefold()
        )
        if not supported_candidate_ids and not safe_unverified:
            repairs.append(
                f"rejected {label} claim with no semantically matching candidate evidence: "
                f"{claim.claim!r}"
            )
            continue
        ids = [
            evidence_id
            for evidence_id in ids
            if evidence_id not in candidate_ids
            or evidence_id in supported_candidate_ids
        ]
        ids = list(
            dict.fromkeys([*([required_job_id] if required_job_id else []), *ids])
        )
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
    claims: list[EvidenceClaim],
    known_ids: set[str],
    repairs: list[str],
    evidence_by_id: dict[str, EvidenceItem],
    required_job_id: str,
) -> list[EvidenceClaim]:
    """Drop unknown IDs from project-analysis claims and clamp confidence."""

    fixed: list[EvidenceClaim] = []
    for claim in claims:
        ids = _valid_ids(claim, known_ids, repairs)
        quoted_names = re.findall(r"'([^']+)'|\"([^\"]+)\"", claim.claim)
        project_names = [left or right for left, right in quoted_names]
        if project_names:
            portfolio_items = [
                evidence_by_id[evidence_id]
                for evidence_id in ids
                if evidence_id in evidence_by_id
                and evidence_by_id[evidence_id].source == "portfolio"
            ]
            if not any(
                evidence_supports_project(item, name)
                for name in project_names
                for item in portfolio_items
            ):
                repairs.append(
                    "rejected project-analysis claim with no exact portfolio project "
                    f"citation: {claim.claim!r}"
                )
                continue
        ids = list(dict.fromkeys([required_job_id, *ids]))
        clamped = min(1.0, max(0.0, claim.confidence))
        fixed.append(
            claim.model_copy(update={"evidence_ids": ids, "confidence": clamped})
        )
    return fixed


def _job_requirement(
    inp: AnalyzeFitInput, skill: str
) -> tuple[str, str] | None:
    """Resolve a claim only to an exact/aliased posted requirement."""

    canonical = canonicalize(skill)
    if not canonical:
        return None
    for required in inp.job.required_skills:
        required_canonical = canonicalize(required)
        members = {canonicalize(member) for member in category_members(required_canonical)}
        if (
            canonical == required_canonical
            or canonical in members
            or skill_in_text(canonical, required)
        ):
            return required, job_skill_evidence_id(inp.job, required)
    return None


def _fix_swap(
    output: FitAnalysisOutput,
    inp: AnalyzeFitInput,
    repairs: list[str],
    job_evidence,
):
    """Null a swap that references a non-existent project; drop unknown IDs."""

    swap = output.project_swap
    if swap is None:
        return None
    portfolio_names = {project.name for project in inp.portfolio_projects}
    if swap.add_project not in portfolio_names:
        repairs.append(
            f"nulled swap: add_project '{swap.add_project}' not in portfolio"
        )
        return None
    if (
        swap.remove_project is not None
        and swap.remove_project not in inp.current_resume_projects
    ):
        repairs.append(
            f"nulled swap: remove_project '{swap.remove_project}' not a current resume project"
        )
        return None
    candidate_ids = {item.evidence_id for item in inp.evidence_items} | {
        evidence_id
        for project in inp.portfolio_projects
        for evidence_id in project.evidence_ids
    }
    job_ids = {item.evidence_id for item in job_evidence}
    portfolio_ids = {
        evidence_id
        for project in inp.portfolio_projects
        if project.name == swap.add_project
        for evidence_id in project.evidence_ids
    }
    ids = [i for i in swap.evidence_ids if i in candidate_ids or i in job_ids]
    valid_portfolio_ids = [i for i in ids if i in portfolio_ids]
    if not valid_portfolio_ids:
        repairs.append(
            f"nulled swap: add_project '{swap.add_project}' lacks portfolio evidence"
        )
        return None
    ids = list(
        dict.fromkeys([job_evidence_id(inp.job, "description"), *valid_portfolio_ids])
    )
    return swap.model_copy(update={"evidence_ids": ids})
