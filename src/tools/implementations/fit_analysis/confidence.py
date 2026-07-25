"""Deterministic confidence assignment shared by the fallback and LLM paths.

Confidence is a function of *which kinds of evidence* support a claim, never an
arbitrary model guess. The ordering encodes: corroboration by multiple
independent sources beats a single descriptive portfolio entry, which beats a
candidate-asserted master-skill or memory fact. Documented in the tool README.
"""

from __future__ import annotations

# Skill-claim source kinds we distinguish for scoring.
RESUME = "resume"
MASTER_SKILLS = "master_skills"
PORTFOLIO = "portfolio"
MEMORY = "memory"


def skill_confidence(source_kinds: set[str], on_resume: bool) -> float:
    """Return the confidence for a skill claim given its evidence source kinds."""

    non_resume = {kind for kind in source_kinds if kind != RESUME}
    if on_resume:
        return 0.9 if non_resume else 0.8
    if len(non_resume) >= 2:
        return 0.85
    if PORTFOLIO in non_resume:
        return 0.70
    if MASTER_SKILLS in non_resume:
        return 0.65
    if MEMORY in non_resume:
        return 0.60
    # No supporting evidence anywhere: a genuine gap, stated with confidence.
    return 0.75


def narrative_confidence(evidence_id_count: int) -> float:
    """Return confidence for experience/seniority/education claims."""

    return 0.9 if evidence_id_count >= 2 else 0.8
