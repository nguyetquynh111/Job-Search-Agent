"""Builds a searchable view of the *whole* candidate profile for scoring.

The scoring formula must judge each job against everything the candidate can
prove: the resume, every portfolio project, the master skills list, and any
saved memory facts (Section 3.2). This module folds all of those sources into
one :class:`CandidateIndex` so the formula can ask two questions cheaply and
deterministically:

* "Does the candidate have skill X?" -- and, if so, which evidence IDs prove it.
* "Does the candidate work in domain Y?" -- matched against a curated domain
  vocabulary drawn from the portfolio, master skills, and target titles.

Skill comparison reuses the hand-curated alias/category maps from the fit
analysis tool so a requirement named "ML" is satisfied by resume skill
"machine learning", and a capability like "vector databases" is satisfied by a
concrete member such as "Pinecone". No fuzzy or embedding matching is used, so
the same profile always yields the same index.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.schemas.common import CandidateProfile, EvidenceItem
from src.tools.implementations.fit_analysis.aliases import (
    canonicalize,
    category_members,
    curated_vocabulary,
    skill_in_text,
)

# Canonical domain phrases the candidate may work in. Job ``industry_domain``
# text is matched against whichever of these appear anywhere in the profile.
_DOMAIN_VOCABULARY = (
    "computer vision",
    "machine learning",
    "deep learning",
    "natural language processing",
    "generative ai",
    "retrieval-augmented generation",
    "mlops",
    "data science",
    "data infrastructure",
    "healthcare",
    "medical imaging",
    "clinical decision support",
    "edge ai",
    "real-time",
    "recommendation",
    "ranking",
    "information extraction",
    "predictive analytics",
    "conversational ai",
    "agentic ai",
    "personalization",
    "experimentation",
    "customer intelligence",
    "robotics",
)


@dataclass
class CandidateIndex:
    """A canonicalized, evidence-linked view of the full candidate profile."""

    skills: set[str] = field(default_factory=set)
    skill_evidence: dict[str, list[str]] = field(default_factory=dict)
    domain_terms: set[str] = field(default_factory=set)
    years_of_experience: float | None = None

    def has_skill(self, canonical: str) -> bool:
        """Return whether the candidate has the given canonical skill directly."""

        return canonical in self.skills

    def satisfies(self, required_canonical: str) -> bool:
        """Direct match, or -- for a capability category -- any member skill."""

        if required_canonical in self.skills:
            return True
        members = category_members(required_canonical)
        return any(canonicalize(member) in self.skills for member in members)

    def evidence_for(self, required_canonical: str) -> list[str]:
        """Evidence IDs proving the required skill, directly or via a member."""

        if required_canonical in self.skill_evidence:
            return list(self.skill_evidence[required_canonical])
        ids: list[str] = []
        for member in category_members(required_canonical):
            ids.extend(self.skill_evidence.get(canonicalize(member), []))
        # Preserve order while de-duplicating.
        return list(dict.fromkeys(ids))


def build_candidate_index(
    profile: CandidateProfile,
    evidence: list[EvidenceItem],
) -> CandidateIndex:
    """Fold profile fields and every evidence source into one scoring index."""

    index = CandidateIndex(
        years_of_experience=(
            float(profile.preferences.years_of_experience)
            if profile.preferences.years_of_experience is not None
            else None
        )
    )

    # 1) Skills named directly on the profile (resume + master skills lists).
    for skill in [*profile.skills, *profile.master_skills]:
        index.skills.add(canonicalize(skill))

    # 2) Skills named as evidence tags (portfolio tech stacks carry these).
    for item in evidence:
        for tag in item.tags:
            canonical = canonicalize(tag)
            index.skills.add(canonical)
            _add_evidence(index, canonical, item.evidence_id)

    # 3) Skills mentioned in free-text evidence (resume lines, project blurbs).
    #    Bounded to the curated vocabulary plus the skills already known, so a
    #    resume bullet can supply an evidence ID for a skill listed elsewhere.
    scan_vocabulary = set(index.skills) | curated_vocabulary()
    for item in evidence:
        text = item.text
        if not text:
            continue
        for canonical in scan_vocabulary:
            if skill_in_text(canonical, text):
                index.skills.add(canonical)
                _add_evidence(index, canonical, item.evidence_id)

    # 4) Domain terms the candidate demonstrably works in.
    index.domain_terms = _collect_domain_terms(profile, evidence)
    return index


def _add_evidence(index: CandidateIndex, canonical: str, evidence_id: str) -> None:
    bucket = index.skill_evidence.setdefault(canonical, [])
    if evidence_id and evidence_id not in bucket:
        bucket.append(evidence_id)


def _collect_domain_terms(
    profile: CandidateProfile,
    evidence: list[EvidenceItem],
) -> set[str]:
    """Return the curated domain phrases present anywhere in the profile."""

    haystack_parts = [
        *profile.master_skills,
        *profile.skills,
        *profile.preferences.target_job_titles,
    ]
    for item in evidence:
        haystack_parts.append(item.text)
        haystack_parts.extend(item.tags)
    haystack = re.sub(r"\s+", " ", " ".join(haystack_parts).lower())

    found: set[str] = set()
    for phrase in _DOMAIN_VOCABULARY:
        if phrase in haystack:
            found.add(phrase)
    # Canonical skill names double as domain signals (e.g. NLP, RAG, GenAI).
    for skill in [*profile.master_skills, *profile.skills]:
        canonical = canonicalize(skill)
        if canonical in _DOMAIN_VOCABULARY:
            found.add(canonical)
    return found
