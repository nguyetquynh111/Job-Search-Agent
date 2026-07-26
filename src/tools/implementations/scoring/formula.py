"""The deterministic scoring formula (signals, weights, and rationale text).

A model must never generate the number (Section 3.2): this is plain arithmetic
over a :class:`~src.tools.implementations.scoring.profile_index.CandidateIndex`.
Each job earns four weighted sub-scores in ``[0, 1]`` which combine into a final
``0..100`` score:

    +--------------------------+--------+---------------------------------------+
    | Signal                   | Weight | How it is measured                    |
    +--------------------------+--------+---------------------------------------+
    | Skill match              |   55   | fraction of the job's required skills  |
    |                          |        | the candidate can evidence            |
    | Experience alignment     |   25   | candidate years vs. the role minimum  |
    | Industry/domain alignment|   15   | overlap of job domain phrases with    |
    |                          |        | the candidate's demonstrated domains  |
    | Location alignment       |    5   | remote/preferred-location fit         |
    |                          |        | (optional; small by design)           |
    +--------------------------+--------+---------------------------------------+

Weights sum to 100 so a perfect match scores 100. Skill match dominates, then
experience, then domain; location is a light tie-breaker because the assignment
marks it optional for scoring.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.schemas.jobs import Job
from src.tools.implementations.filtering.rules import is_remote_eligible
from src.tools.implementations.fit_analysis.aliases import canonicalize
from src.tools.implementations.scoring.profile_index import CandidateIndex

SCORE_WEIGHTS: dict[str, float] = {
    "skill_match": 55.0,
    "experience_alignment": 25.0,
    "industry_domain_alignment": 15.0,
    "location_alignment": 5.0,
}

# Sub-scores used when a job simply does not supply a signal, so a missing field
# neither rewards nor punishes a job relative to its peers.
_NEUTRAL_EXPERIENCE = 0.75
_NEUTRAL_DOMAIN = 0.5
_MIN_EXPERIENCE_CREDIT = 0.2
_NO_DOMAIN_OVERLAP_FLOOR = 0.3
_LOCATION_MISMATCH = 0.4


@dataclass
class JobScore:
    """The score, rationale, and backing evidence for one job."""

    score: float
    rationale: str
    evidence_ids: list[str]


def _skill_match(job: Job, index: CandidateIndex) -> tuple[float, list[str], str]:
    """Fraction of required skills evidenced, with backing evidence IDs."""

    required = [canonicalize(skill) for skill in job.required_skills]
    required = list(dict.fromkeys(required))  # de-dup, keep order
    if not required:
        return 0.6, [], "no required skills listed (neutral skill credit)"
    matched: list[str] = []
    evidence_ids: list[str] = []
    for skill in required:
        if index.satisfies(skill):
            matched.append(skill)
            evidence_ids.extend(index.evidence_for(skill))
    fraction = len(matched) / len(required)
    evidence_ids = list(dict.fromkeys(evidence_ids))
    summary = f"{len(matched)}/{len(required)} required skills evidenced"
    return fraction, evidence_ids, summary


def _experience_alignment(job: Job, index: CandidateIndex) -> tuple[float, str]:
    """Candidate years vs. the role's minimum required years."""

    candidate = index.years_of_experience
    required = job.experience_requirement.minimum_years
    if required is None:
        required = job.years_experience_required
    if candidate is None or required is None:
        return _NEUTRAL_EXPERIENCE, "experience requirement unspecified (neutral)"
    if candidate >= required:
        return 1.0, f"{_fmt(candidate)}y meets the {_fmt(required)}y+ minimum"
    ratio = candidate / required if required else 1.0
    score = max(_MIN_EXPERIENCE_CREDIT, ratio)
    return score, f"{_fmt(candidate)}y is below the {_fmt(required)}y+ minimum"


def _domain_phrases(job: Job) -> list[str]:
    parts = re.split(r"[/,;]", job.industry_domain)
    return [re.sub(r"\s+", " ", part.strip().lower()) for part in parts if part.strip()]


def _industry_domain_alignment(
    job: Job, index: CandidateIndex
) -> tuple[float, str]:
    """Overlap between the job's domain phrases and the candidate's domains."""

    phrases = _domain_phrases(job)
    if not phrases:
        return _NEUTRAL_DOMAIN, "no domain listed (neutral)"
    matched = [
        phrase
        for phrase in phrases
        if any(term in phrase or phrase in term for term in index.domain_terms)
    ]
    if not matched:
        return _NO_DOMAIN_OVERLAP_FLOOR, "no domain overlap"
    fraction = len(matched) / len(phrases)
    # Reward any genuine overlap while keeping partial matches below a full one.
    score = _NO_DOMAIN_OVERLAP_FLOOR + (1 - _NO_DOMAIN_OVERLAP_FLOOR) * fraction
    return score, f"domain overlap on {', '.join(matched)}"


def _location_alignment(job: Job, index: CandidateIndex) -> tuple[float, str]:
    """Light optional signal: remote-eligible jobs align best."""

    if is_remote_eligible(job):
        return 1.0, "remote-eligible"
    return _LOCATION_MISMATCH, "onsite in a non-preferred location"


def score_one_job(job: Job, index: CandidateIndex) -> JobScore:
    """Compute the weighted 0..100 score, rationale, and evidence for one job."""

    skill_fraction, skill_evidence, skill_summary = _skill_match(job, index)
    experience_fraction, experience_summary = _experience_alignment(job, index)
    domain_fraction, domain_summary = _industry_domain_alignment(job, index)
    location_fraction, location_summary = _location_alignment(job, index)

    total = (
        skill_fraction * SCORE_WEIGHTS["skill_match"]
        + experience_fraction * SCORE_WEIGHTS["experience_alignment"]
        + domain_fraction * SCORE_WEIGHTS["industry_domain_alignment"]
        + location_fraction * SCORE_WEIGHTS["location_alignment"]
    )
    score = round(min(100.0, max(0.0, total)), 1)
    rationale = (
        f"Score {score}/100 -- skills: {skill_summary}; "
        f"experience: {experience_summary}; "
        f"domain: {domain_summary}; location: {location_summary}."
    )
    return JobScore(score=score, rationale=rationale, evidence_ids=skill_evidence)


def _fmt(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:g}"
