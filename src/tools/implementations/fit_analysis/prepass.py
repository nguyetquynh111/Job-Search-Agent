"""Deterministic pre-pass: pure functions that need no LLM.

Computes the disjoint skill buckets, contrastive relevant experience, a
verdict-based seniority match, education, and the project section (analysis +
swap) from one shared ranking. It is both the grounding for the LLM prompt and
the complete offline fallback. Every claim carries a verdict in its notes so the
renderer derives markers from the verdict, never a default.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from src.schemas.common import EvidenceClaim, EvidenceItem, ProjectSwap
from src.schemas.fit_analysis import AnalyzeFitInput, FitAnalysisOutput
from src.tools.implementations.fit_analysis import confidence, verdict
from src.tools.implementations.fit_analysis.aliases import (
    canonicalize,
    category_members,
    skill_in_text,
)
from src.tools.implementations.fit_analysis.evidence_index import EvidenceIndex
from src.tools.implementations.fit_analysis.sanitize import sanitize_text
from src.tools.implementations.fit_analysis.swap import SwapDecision, build_project_swap

logger = logging.getLogger(__name__)

_SENIORITY_MARKERS = ("principal", "staff", "lead", "senior")
_DEGREE_MARKERS = (
    "phd", "ph.d", "doctora", "master", "m.s", "msc", "bachelor", "b.s", "bsc", "degree",
)


@dataclass
class SkillClaim:
    """An intermediate, evidence-bound skill finding before rendering to a claim."""

    skill: str
    evidence_ids: list[str]
    sources: set[str]
    on_resume: bool
    members: list[str] = field(default_factory=list)  # concrete tools for a category


@dataclass
class PrePass:
    """Everything the deterministic pass computed for one job."""

    job_id: str
    job_text: str
    aligned: list[SkillClaim] = field(default_factory=list)
    evidenced_missing: list[SkillClaim] = field(default_factory=list)
    genuine_gaps: list[SkillClaim] = field(default_factory=list)
    experience: list[EvidenceClaim] = field(default_factory=list)
    seniority: list[EvidenceClaim] = field(default_factory=list)
    education: list[EvidenceClaim] = field(default_factory=list)
    swap: SwapDecision | None = None


def _evidence_by_tag(items: list[EvidenceItem], tag: str) -> list[EvidenceItem]:
    """Return resume-sourced evidence items carrying a given section tag."""

    return [
        item
        for item in items
        if item.source == "resume" and tag in {t.lower() for t in item.tags}
    ]


def _dedupe_canonical(skills: list[str]) -> list[tuple[str, str]]:
    """Return (original, canonical) pairs, first occurrence wins."""

    seen: set[str] = set()
    pairs: list[tuple[str, str]] = []
    for skill in skills:
        canonical = canonicalize(skill)
        if canonical and canonical not in seen:
            seen.add(canonical)
            pairs.append((skill, canonical))
    return pairs


# Combined labels like "Docker/Kubernetes" or "GenAI/LLMs" name several skills at
# once; split them so each part can be resolved independently.
_SPLIT_RE = re.compile(r"\s*(?:/| and | & )\s*", re.IGNORECASE)


def _split_parts(skill: str) -> list[str]:
    """Return the whole skill plus any slash/'and'-separated parts."""

    parts = [skill]
    for part in _SPLIT_RE.split(skill):
        part = part.strip()
        if part and part != skill:
            parts.append(part)
    return parts


def _expand_canonicals(skills: list[str]) -> list[str]:
    """Return canonical tokens for required skills, including split-part tokens."""

    seen: set[str] = set()
    expanded: list[str] = []
    for skill in skills:
        for part in _split_parts(skill):
            canonical = canonicalize(part)
            if canonical and canonical not in seen:
                seen.add(canonical)
                expanded.append(canonical)
    return expanded


def _resolve_required(
    original: str, index: EvidenceIndex, resume_skill_set: set[str]
) -> SkillClaim:
    """Resolve a required skill, trying the whole token then its parts, and merge.

    A combined label is aligned if any part is on the resume, evidenced-missing if
    any part is evidenced elsewhere, and only a genuine gap if no part resolves.
    """

    seen: set[str] = set()
    claims: list[SkillClaim] = []
    for part in _split_parts(original):
        canonical = canonicalize(part)
        if canonical and canonical not in seen:
            seen.add(canonical)
            claims.append(_resolve_skill(original, canonical, index, resume_skill_set))

    on_resume = [c for c in claims if c.on_resume]
    evidenced = [c for c in claims if c.sources and not c.on_resume]
    chosen = on_resume or evidenced
    if not chosen:
        return SkillClaim(skill=original, evidence_ids=[], sources=set(), on_resume=False, members=[])

    ids: list[str] = []
    seen_ids: set[str] = set()
    sources: set[str] = set()
    members: list[str] = []
    for claim in chosen:
        for evidence_id in claim.evidence_ids:
            if evidence_id not in seen_ids:
                seen_ids.add(evidence_id)
                ids.append(evidence_id)
        sources |= claim.sources
        for member in claim.members:
            if member not in members:
                members.append(member)
    return SkillClaim(
        skill=original,
        evidence_ids=ids[:6],
        sources=sources,
        on_resume=bool(on_resume),
        members=members[:3],
    )


def _resolve_skill(
    original: str, canonical: str, index: EvidenceIndex, resume_skill_set: set[str]
) -> SkillClaim:
    """Resolve one required skill directly or via a category member (e.g. agents←AutoGen).

    A capability required by the job (``agents``, ``APIs``) is satisfied by evidence
    of a concrete member tool (``AutoGen``, ``FastAPI``). The concrete tool(s) are
    recorded so the claim can name what makes it defensible. Direct and member
    evidence are weighed together, and a resume-present support wins over an
    off-resume one (so ``deep learning`` is aligned via a resume PyTorch even if a
    portfolio domain tag also mentions it).
    """

    # Each support: (member_label_or_None, on_resume, sources, evidence_ids).
    supports: list[tuple[str | None, bool, set[str], list[str]]] = []
    direct_sources = index.sources_for(canonical)
    direct_on_resume = canonical in resume_skill_set or "resume" in direct_sources
    if direct_sources or direct_on_resume:
        supports.append((None, direct_on_resume, direct_sources, index.ordered_ids(canonical, limit=6)))
    for display in sorted(category_members(canonical)):
        member_canonical = canonicalize(display)
        m_sources = index.sources_for(member_canonical)
        m_on_resume = member_canonical in resume_skill_set or "resume" in m_sources
        if m_sources or m_on_resume:
            supports.append((display, m_on_resume, m_sources, index.ordered_ids(member_canonical, limit=3)))

    if not supports:
        return SkillClaim(skill=original, evidence_ids=[], sources=set(), on_resume=False, members=[])

    on_resume = any(support[1] for support in supports)
    # When aligned, cite only the resume-present supports; otherwise cite everything.
    chosen = [s for s in supports if s[1]] if on_resume else supports
    ids: list[str] = []
    seen: set[str] = set()
    for _, _, _, support_ids in chosen:
        for evidence_id in support_ids:
            if evidence_id not in seen:
                seen.add(evidence_id)
                ids.append(evidence_id)
    sources: set[str] = set().union(*(s[2] for s in chosen)) if chosen else set()
    members = [label for label, _, _, _ in chosen if label is not None][:3]
    return SkillClaim(
        skill=original, evidence_ids=ids[:6], sources=sources, on_resume=on_resume, members=members
    )


def _skill_buckets(
    inp: AnalyzeFitInput, index: EvidenceIndex
) -> tuple[list[SkillClaim], list[SkillClaim], list[SkillClaim]]:
    """Split required skills into aligned / evidenced-missing / genuine-gap."""

    resume_skill_set = {canonicalize(skill) for skill in inp.candidate_profile.skills}
    aligned: list[SkillClaim] = []
    evidenced_missing: list[SkillClaim] = []
    genuine_gaps: list[SkillClaim] = []

    for original, _canonical in _dedupe_canonical(inp.job.required_skills):
        claim = _resolve_required(original, index, resume_skill_set)
        if claim.on_resume:
            aligned.append(claim)
        elif claim.sources:
            evidenced_missing.append(claim)
        else:
            genuine_gaps.append(claim)
    return aligned, evidenced_missing, genuine_gaps


def _parse_experience(text: str) -> tuple[str, str, str]:
    """Split a resume experience line into (role, company, focus)."""

    parts = [segment.strip() for segment in text.split("|")]
    role = parts[0] if parts else text
    company = parts[2] if len(parts) >= 3 else ""
    focus = " ".join(parts[4:]).strip() if len(parts) >= 5 else " ".join(parts[1:]).strip()
    return role, company, focus


def _experience_matches(original: str, text: str) -> bool:
    """Return whether any part of a required skill appears in experience text."""

    return any(skill_in_text(canonicalize(part), text) for part in _split_parts(original))


def _relevant_experience(
    inp: AnalyzeFitInput, required_original: list[str]
) -> list[EvidenceClaim]:
    """Build contrastive experience claims: role, company, focus, alignment verdict."""

    items = _evidence_by_tag(inp.evidence_items, "experience")
    job_focus = ", ".join(required_original[:3]) or inp.job.title
    claims: list[EvidenceClaim] = []
    for item in items:
        role, company, focus = _parse_experience(item.text)
        matched = [original for original in required_original if _experience_matches(original, item.text)]
        where = f"{role} at {company}".strip(" at")
        if matched:
            claim_text = (
                f"{where}: centered on {focus[:200]} — aligns with the job's focus on "
                f"{', '.join(matched[:4])}."
            )
            note = verdict.tag(verdict.MATCH, f"Overlaps required skills: {', '.join(matched)}")
        else:
            claim_text = (
                f"{where}: centered on {focus[:200]} — limited overlap with the job's focus "
                f"on {job_focus}."
            )
            note = verdict.tag(verdict.PARTIAL, "No direct required-skill overlap.")
        claims.append(
            EvidenceClaim(
                claim=claim_text,
                evidence_ids=[item.evidence_id],
                confidence=confidence.narrative_confidence(1),
                notes=note,
            )
        )
    return claims


def _seniority_verdict(candidate_years, required_min) -> str:
    """Return the seniority verdict from candidate vs required years."""

    if required_min is None:
        return verdict.MATCH  # no barrier stated
    if candidate_years is None:
        return verdict.PARTIAL
    if candidate_years >= required_min:
        return verdict.MATCH
    if candidate_years >= 0.6 * required_min:
        return verdict.PARTIAL
    return verdict.MISMATCH


def _seniority(inp: AnalyzeFitInput) -> list[EvidenceClaim]:
    """Infer a seniority match verdict from titles and years, tolerating None."""

    items = _evidence_by_tag(inp.evidence_items, "experience")
    combined = " ".join(item.text for item in items).lower()
    candidate_level = next(
        (marker for marker in _SENIORITY_MARKERS if marker in combined), None
    )
    years = inp.candidate_profile.preferences.years_of_experience
    required = inp.job.years_experience_required
    if required is None:
        required = inp.job.experience_requirement.minimum_years

    if required is not None:
        job_expectation = f"{required}+ years expected"
    elif inp.job.experience_requirement.raw_text:
        job_expectation = inp.job.experience_requirement.raw_text
    else:
        job_expectation = "no explicit experience requirement stated"

    parts: list[str] = []
    if candidate_level:
        parts.append(f"candidate has held {candidate_level}-level roles")
    if years is not None:
        parts.append(f"~{years} years of experience")
    candidate_desc = "; ".join(parts) or "seniority inferred from resume history"

    result = _seniority_verdict(years, required)
    human = {
        verdict.MATCH: "Meets or exceeds the stated experience.",
        verdict.PARTIAL: "Close to but below the stated experience.",
        verdict.MISMATCH: "Falls short of the stated experience.",
    }[result]
    if required is None and inp.job.years_experience_required is None and inp.job.experience_requirement.minimum_years is None:
        human = "Job years not specified; not inferred."

    evidence_ids = [item.evidence_id for item in items[:2]]
    return [
        EvidenceClaim(
            claim=f"Seniority: {candidate_desc} vs job ({job_expectation}).",
            evidence_ids=evidence_ids,
            confidence=confidence.narrative_confidence(len(evidence_ids)),
            notes=verdict.tag(result, human),
        )
    ]


def _education(inp: AnalyzeFitInput, job_text: str) -> list[EvidenceClaim]:
    """Build education claims, matching resume degrees to any job degree language."""

    items = _evidence_by_tag(inp.evidence_items, "education")
    job_wants_degree = any(marker in job_text.lower() for marker in _DEGREE_MARKERS)
    claims: list[EvidenceClaim] = []
    for item in items:
        note = (
            "Job references a degree requirement; candidate holds a relevant degree."
            if job_wants_degree
            else "Job states no specific degree requirement."
        )
        claims.append(
            EvidenceClaim(
                claim=f"Education: {item.text}",
                evidence_ids=[item.evidence_id],
                confidence=confidence.narrative_confidence(1),
                notes=verdict.tag(verdict.MATCH, note),
            )
        )
    return claims


def run_prepass(inp: AnalyzeFitInput, index: EvidenceIndex) -> PrePass:
    """Run every deterministic computation for one job."""

    job_text = sanitize_text(
        f"{inp.job.title}. {inp.job.description} {inp.job.company_details}"
    )
    required_pairs = _dedupe_canonical(inp.job.required_skills)
    required_original = [original for original, _ in required_pairs]
    # Project scoring matches on expanded tokens so combined labels like
    # "Docker/Kubernetes" and category members are both considered.
    required_canonical = _expand_canonicals(inp.job.required_skills)

    aligned, evidenced_missing, genuine_gaps = _skill_buckets(inp, index)
    return PrePass(
        job_id=inp.job.job_id,
        job_text=job_text,
        aligned=aligned,
        evidenced_missing=evidenced_missing,
        genuine_gaps=genuine_gaps,
        experience=_relevant_experience(inp, required_original),
        seniority=_seniority(inp),
        education=_education(inp, job_text),
        swap=build_project_swap(
            inp.current_resume_projects,
            inp.portfolio_projects,
            inp.job,
            required_canonical,
            job_text,
        ),
    )


def _skill_claim(entry: SkillClaim, kind: str) -> EvidenceClaim:
    """Render a bucketed skill finding into an evidence-bound, verdict-tagged claim.

    When a capability is satisfied via a category member, the concrete tool(s) are
    named so the claim is defensible (e.g. ``agents ... via AutoGen``).
    """

    via = ", ".join(entry.members)
    if kind == "aligned":
        suffix = f" via {via}" if via else ""
        text = f"{entry.skill}: required by the job and present on your resume{suffix}."
        note = verdict.tag(verdict.MATCH)
    elif kind == "evidenced_missing":
        where = ", ".join(sorted(entry.sources)) or "supplied evidence"
        suffix = f" via {via}" if via else ""
        text = (
            f"{entry.skill}: required by the job, not yet on your resume, "
            f"but evidenced in {where}{suffix}."
        )
        human = (f"via={via}; " if via else "") + f"Safe to add during tailoring; evidenced by {where}."
        note = verdict.tag(verdict.MISSING, human)
    else:
        text = f"{entry.skill}: required by the job with no supporting evidence found."
        note = verdict.tag(verdict.MISMATCH, "No evidence in resume, portfolio, master skills, or memory.")
    return EvidenceClaim(
        claim=text,
        evidence_ids=list(entry.evidence_ids),
        confidence=confidence.skill_confidence(entry.sources, entry.on_resume),
        notes=note,
    )


def build_skill_section(
    prepass: PrePass,
) -> tuple[list[EvidenceClaim], list[EvidenceClaim], list[EvidenceClaim]]:
    """Build the three disjoint skill-bucket claim lists from the deterministic pass."""

    return (
        [_skill_claim(entry, "aligned") for entry in prepass.aligned],
        [_skill_claim(entry, "evidenced_missing") for entry in prepass.evidenced_missing],
        [_skill_claim(entry, "genuine_gap") for entry in prepass.genuine_gaps],
    )


def build_project_section(
    swap: SwapDecision | None,
) -> tuple[list[EvidenceClaim], ProjectSwap | None]:
    """Build project_analysis claims and the swap from the one shared ranking.

    Both come from the same :class:`SwapDecision`, so the project reported as
    removed is the one that decision marked weak and is never simultaneously
    reported as strongly aligned -- whether that project was chosen by the ranking
    or accepted from the model.
    """

    claims: list[EvidenceClaim] = []
    if swap is None:
        return claims, None

    for entry in swap.current_verdicts:
        if entry.project is None:
            claims.append(
                EvidenceClaim(
                    claim=f"Current project '{entry.name}' could not be matched to a portfolio entry.",
                    evidence_ids=[],
                    confidence=0.6,
                    notes=verdict.tag(verdict.PARTIAL, "No portfolio project with a matching name."),
                )
            )
            continue
        dims: list[str] = []
        if entry.distinctive:
            dims.append(f"tech: {', '.join(entry.distinctive)}")
        if entry.domain_matches:
            dims.append(f"domain: {', '.join(entry.domain_matches)}")
        if entry.industry_matches:
            dims.append(f"industry: {', '.join(entry.industry_matches)}")
        detail = "; ".join(dims) or "shared general tooling only"
        if entry.removed:
            claim_text = (
                f"Current project '{entry.name}' ({entry.project.project_id}) is the weakest "
                f"match for this job ({detail}); recommended for swap."
            )
        elif entry.verdict == verdict.MATCH:
            claim_text = (
                f"Current project '{entry.name}' ({entry.project.project_id}) aligns well "
                f"with this job ({detail})."
            )
        else:
            claim_text = (
                f"Current project '{entry.name}' ({entry.project.project_id}) has limited "
                f"alignment with this job ({detail})."
            )
        claims.append(
            EvidenceClaim(
                claim=claim_text,
                evidence_ids=list(entry.project.evidence_ids),
                confidence=0.8 if entry.verdict == verdict.MATCH else 0.7,
                notes=verdict.tag(entry.verdict),
            )
        )

    for note in swap.weak_slot_notes:
        claims.append(
            EvidenceClaim(claim=note, evidence_ids=[], confidence=0.7, notes=verdict.tag(verdict.PARTIAL))
        )
    if swap.swap is None:
        claims.append(
            EvidenceClaim(
                claim=(
                    "Current resume projects are already the strongest available match "
                    "for this job; no project swap is recommended."
                ),
                evidence_ids=[],
                confidence=0.75,
                notes=verdict.tag(verdict.MATCH),
            )
        )
    return claims, swap.swap


def build_fallback_output(inp: AnalyzeFitInput, prepass: PrePass) -> FitAnalysisOutput:
    """Assemble a complete FitAnalysisOutput from the deterministic pre-pass alone."""

    project_analysis, project_swap = build_project_section(prepass.swap)
    aligned, evidenced_missing, genuine_gaps = build_skill_section(prepass)
    return FitAnalysisOutput(
        job_id=inp.job.job_id,
        relevant_experience=prepass.experience,
        seniority=prepass.seniority,
        education=prepass.education,
        aligned_skills=aligned,
        evidenced_missing_skills=evidenced_missing,
        genuine_gaps=genuine_gaps,
        project_analysis=project_analysis,
        project_swap=project_swap,
    )


def prepass_summary(prepass: PrePass, index: EvidenceIndex) -> dict[str, object]:
    """Return a compact, JSON-safe grounding summary for the prompt and trace."""

    return {
        "aligned_skills": [entry.skill for entry in prepass.aligned],
        "evidenced_missing_skills": [
            {
                "skill": entry.skill,
                "evidence_ids": entry.evidence_ids,
                "sources": sorted(entry.sources),
                "via_members": entry.members,
            }
            for entry in prepass.evidenced_missing
        ],
        "genuine_gaps": [entry.skill for entry in prepass.genuine_gaps],
        "evidence_index": index.summary(),
        "suggested_swap": (
            {"remove_project": prepass.swap.swap.remove_project, "add_project": prepass.swap.swap.add_project}
            if prepass.swap and prepass.swap.swap
            else None
        ),
    }
