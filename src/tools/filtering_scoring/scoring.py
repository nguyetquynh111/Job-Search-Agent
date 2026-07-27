from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from pydantic import Field, model_validator

from src.domain import (
    CandidateProfile,
    EvidenceItem,
    Job,
    StrictBaseModel,
)
from src.tools.filtering_scoring.filtering import is_remote_eligible
from src.tracing.langfuse import TraceManager
from src.utils.skill_matching import (
    canonicalize,
    category_members,
    curated_vocabulary,
    skill_in_text,
)

logger = logging.getLogger(__name__)


class ScoreJobsInput(StrictBaseModel):
    """Input for score_jobs."""

    jobs: list[Job]
    candidate_profile: CandidateProfile
    resume_evidence: list[EvidenceItem] = Field(default_factory=list)
    master_skill_evidence: list[EvidenceItem] = Field(default_factory=list)
    portfolio_evidence: list[EvidenceItem] = Field(default_factory=list)
    memory_evidence: list[EvidenceItem] = Field(default_factory=list)


class ScoreComponent(StrictBaseModel):
    """One deterministic weighted component of a final job score."""

    alignment: float = Field(ge=0, le=1)
    weight: float = Field(ge=0, le=100)
    points: float = Field(ge=0, le=100)


class ScoredJob(StrictBaseModel):
    """A job and deterministic score returned by the scoring tool."""

    job: Job
    score: float = Field(ge=0, le=100)
    score_breakdown: dict[str, ScoreComponent] = Field(default_factory=dict)
    rationale: str
    evidence_ids: list[str] = Field(default_factory=list)


class ScoreJobsOutput(StrictBaseModel):
    """Output from score_jobs."""

    ranked_jobs: list[ScoredJob]
    top_3_job_ids: list[str]

    @model_validator(mode="after")
    def validate_top_three(self) -> "ScoreJobsOutput":
        """Require top_3_job_ids to reference ranked jobs and contain at most three IDs."""

        ranked_ids = {job.job.job_id for job in self.ranked_jobs}
        if len(self.top_3_job_ids) > 3:
            raise ValueError("top_3_job_ids cannot contain more than three jobs")
        missing = [job_id for job_id in self.top_3_job_ids if job_id not in ranked_ids]
        if missing:
            raise ValueError(f"top_3_job_ids reference unknown ranked jobs: {missing}")
        return self


# Domains matched against the candidate's full profile.
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
        # Remove duplicates without changing order.
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

    # Add skills listed directly in the profile.
    for skill in [*profile.skills, *profile.master_skills]:
        index.skills.add(canonicalize(skill))

    # Add skills carried by evidence tags.
    for item in evidence:
        for tag in item.tags:
            canonical = canonicalize(tag)
            index.skills.add(canonical)
            _add_evidence(index, canonical, item.evidence_id)

    # Scan free text against a small, known vocabulary.
    scan_vocabulary = set(index.skills) | curated_vocabulary()
    for item in evidence:
        text = item.text
        if not text:
            continue
        for canonical in scan_vocabulary:
            if skill_in_text(canonical, text):
                index.skills.add(canonical)
                _add_evidence(index, canonical, item.evidence_id)

    # Collect domains supported by the profile.
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
    # Some canonical skills also signal a domain.
    for skill in [*profile.master_skills, *profile.skills]:
        canonical = canonicalize(skill)
        if canonical in _DOMAIN_VOCABULARY:
            found.add(canonical)
    return found


SCORING_WEIGHTS: dict[str, float] = {
    "skill_match": 55.0,
    "experience_alignment": 25.0,
    "industry_domain_alignment": 15.0,
    "location_alignment": 5.0,
}

# Neutral scores keep missing data from helping or hurting a job.
_NEUTRAL_EXPERIENCE = 0.75
_NEUTRAL_DOMAIN = 0.5
_MIN_EXPERIENCE_CREDIT = 0.2
_NO_DOMAIN_OVERLAP_FLOOR = 0.3
_LOCATION_MISMATCH = 0.4


@dataclass
class JobScore:
    """The score, rationale, and backing evidence for one job."""

    score: float
    breakdown: dict[str, ScoreComponent]
    rationale: str
    evidence_ids: list[str]


def _skill_match(job: Job, index: CandidateIndex) -> tuple[float, list[str], str]:
    """Fraction of required skills evidenced, with backing evidence IDs."""

    required = [canonicalize(skill) for skill in job.required_skills]
    required = list(dict.fromkeys(required))  # Remove duplicates, keep order.
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


def _industry_domain_alignment(job: Job, index: CandidateIndex) -> tuple[float, str]:
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
    # Reward overlap without treating a partial match as complete.
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
        skill_fraction * SCORING_WEIGHTS["skill_match"]
        + experience_fraction * SCORING_WEIGHTS["experience_alignment"]
        + domain_fraction * SCORING_WEIGHTS["industry_domain_alignment"]
        + location_fraction * SCORING_WEIGHTS["location_alignment"]
    )
    score = round(min(100.0, max(0.0, total)), 1)
    breakdown = {
        name: ScoreComponent(
            alignment=round(fraction, 4),
            weight=SCORING_WEIGHTS[name],
            points=round(fraction * SCORING_WEIGHTS[name], 2),
        )
        for name, fraction in (
            ("skill_match", skill_fraction),
            ("experience_alignment", experience_fraction),
            ("industry_domain_alignment", domain_fraction),
            ("location_alignment", location_fraction),
        )
    }
    rationale = (
        f"Score {score}/100 -- skills: {skill_summary}; "
        f"experience: {experience_summary}; "
        f"domain: {domain_summary}; location: {location_summary}."
    )
    return JobScore(
        score=score,
        breakdown=breakdown,
        rationale=rationale,
        evidence_ids=skill_evidence,
    )


def _fmt(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


TOP_N = 3


def run_scoring_tool(
    inp: ScoreJobsInput,
    *,
    tracer: TraceManager | None = None,
) -> ScoreJobsOutput:
    """Score, rank, and select the Top-3 jobs against the full candidate profile.

    Pass ``tracer`` to nest a span under the run trace; the default disabled
    tracer keeps this callable standalone and never fails when Langfuse is
    unconfigured.
    """

    try:
        evidence = [
            *inp.resume_evidence,
            *inp.master_skill_evidence,
            *inp.portfolio_evidence,
            *inp.memory_evidence,
        ]
        valid_evidence_ids = {item.evidence_id for item in evidence}
        index = build_candidate_index(inp.candidate_profile, evidence)

        scored = []
        for job in inp.jobs:
            result = score_one_job(job, index)
            # Keep only evidence supplied with this request.
            evidence_ids = [
                evidence_id
                for evidence_id in result.evidence_ids
                if evidence_id in valid_evidence_ids
            ]
            scored.append(
                ScoredJob(
                    job=job,
                    score=result.score,
                    score_breakdown=result.breakdown,
                    rationale=result.rationale,
                    evidence_ids=evidence_ids,
                )
            )

        # Preserve input order when scores tie.
        ranked = sorted(scored, key=lambda item: item.score, reverse=True)
        top_3_job_ids = [item.job.job_id for item in ranked[:TOP_N]]
        output = ScoreJobsOutput(ranked_jobs=ranked, top_3_job_ids=top_3_job_ids)
    except Exception:
        logger.exception("Scoring failed")
        raise
    logger.info(
        "Scoring complete: ranked %d jobs; Top-3 = %s.",
        len(output.ranked_jobs),
        output.top_3_job_ids,
    )
    return output
