"""Deterministic filtering rules and their per-job rejection reasons.

Each rule is a pure function ``(job, preferences) -> list[str]`` that returns
zero rejection reasons (the job passes the rule) or one/more human-readable
reasons (the job fails it). A rule is *inert* whenever its backing preference is
not specified, so the same code runs for every candidate regardless of which
preferences they filled in. ``evaluate_job`` runs every rule and collects all
reasons, so a rejected job reports every way it failed -- not just the first.

The assignment's four required filters are :func:`company_exclusion_reasons`,
:func:`remote_only_reasons`, :func:`location_reasons`, and
:func:`experience_reasons`. Two preference-driven rules,
:func:`excluded_keyword_reasons` and :func:`target_title_reasons`, are also
applied because ``CandidatePreferences`` carries excluded keywords and target
job titles (Section 2.4). Nothing here mutates the job.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from src.schemas.common import CandidatePreferences, company_comparison_key
from src.schemas.jobs import Job

Rule = Callable[[Job, CandidatePreferences], list[str]]

# Location tokens that describe US-wide / remote postings rather than a city.
_NATIONWIDE_MARKERS = (
    "united states",
    "nationwide",
    "anywhere",
    "multiple",
    "u.s.",
    "usa",
)
# Words that carry no city/state meaning when matching a preferred location.
_GENERIC_LOCATION_TOKENS = {
    "remote",
    "us",
    "usa",
    "united",
    "states",
    "hybrid",
    "onsite",
    "on",
    "site",
    "or",
    "and",
    "area",
    "metro",
    "greater",
}
# Title words too generic to signal AI/ML relevance on their own.
_GENERIC_TITLE_TOKENS = {
    "engineer",
    "engineering",
    "senior",
    "junior",
    "staff",
    "principal",
    "lead",
    "intern",
    "internship",
    "co",
    "op",
    "sr",
    "jr",
    "associate",
    "specialist",
    "manager",
    "analyst",
    "consultant",
    "summer",
    "fall",
    "spring",
    "of",
    "the",
    "and",
    "for",
    "ii",
    "iii",
    "iv",
    "i",
    "project",
    "product",
    "software",
    "solutions",
}
# AI/ML role vocabulary that always signals relevance to the candidate's targets.
_AI_ML_TITLE_TERMS = {
    "ai",
    "ml",
    "machine learning",
    "deep learning",
    "computer vision",
    "cv",
    "nlp",
    "generative",
    "genai",
    "llm",
    "mlops",
    "data science",
    "data scientist",
    "artificial intelligence",
    "rag",
    "agentic",
    "neural",
}


def _norm(text: str | None) -> str:
    """Lower-case and collapse whitespace for stable comparisons."""

    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _contains_term(haystack: str, term: str) -> bool:
    """Whole-word (or whole-phrase) containment for a normalized term."""

    term = _norm(term)
    if not term:
        return False
    if " " in term:
        return term in haystack
    return re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", haystack) is not None


def _format_years(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def is_remote_eligible(job: Job) -> bool:
    """A job is remote-eligible when flagged remote or its location says remote."""

    if job.remote is True:
        return True
    return "remote" in _norm(job.location)


def _is_nationwide(location_norm: str) -> bool:
    return any(marker in location_norm for marker in _NATIONWIDE_MARKERS)


def _location_tokens(preferred_norm: str) -> list[str]:
    return [
        token
        for token in re.split(r"[^a-z0-9]+", preferred_norm)
        if token and token not in _GENERIC_LOCATION_TOKENS
    ]


# --- The four required filters -------------------------------------------------


def company_exclusion_reasons(job: Job, prefs: CandidatePreferences) -> list[str]:
    """Reject jobs whose company matches the exclusion list (case/space-insensitive)."""

    excluded = prefs.excluded_company_comparison_keys
    if not excluded:
        return []
    if company_comparison_key(job.company) in excluded:
        return [f"Company '{job.company}' is on the candidate's exclusion list."]
    return []


def remote_only_reasons(job: Job, prefs: CandidatePreferences) -> list[str]:
    """When remote-only is set, reject any role that is not remote-eligible."""

    if not prefs.remote_only:
        return []
    if is_remote_eligible(job):
        return []
    return [
        "Candidate accepts remote roles only, but "
        f"'{job.location or 'unspecified location'}' is not remote-eligible."
    ]


def location_reasons(job: Job, prefs: CandidatePreferences) -> list[str]:
    """Reject onsite roles outside the candidate's preferred locations.

    Remote-eligible roles always pass (they can be worked from any home base). A
    US-wide posting passes when the candidate lists any remote/US preference.
    Otherwise the job's location must share a city/state token with a preference.
    """

    preferred = prefs.preferred_locations
    if not preferred:
        return []
    if is_remote_eligible(job):
        return []
    location_norm = _norm(job.location)
    accepts_remote = any("remote" in _norm(entry) for entry in preferred)
    if accepts_remote and _is_nationwide(location_norm):
        return []
    for entry in preferred:
        for token in _location_tokens(_norm(entry)):
            if _contains_term(location_norm, token):
                return []
    return [
        f"Location '{job.location or 'unspecified'}' is outside the preferred "
        f"locations ({', '.join(preferred)})."
    ]


def experience_reasons(job: Job, prefs: CandidatePreferences) -> list[str]:
    """Reject roles whose minimum required experience exceeds the candidate's.

    Only under-qualification is filtered; there is no reliable upper bound in the
    dataset, so an over-qualified match is left for the scoring tool to weigh.
    """

    candidate_years = prefs.years_of_experience
    if candidate_years is None:
        return []
    required = job.experience_requirement.minimum_years
    if required is None:
        required = job.years_experience_required
    if required is None:
        return []
    if float(required) > float(candidate_years):
        return [
            f"Role requires about {_format_years(required)}+ years of experience; "
            f"candidate has {_format_years(candidate_years)}."
        ]
    return []


# --- Preference-driven filters (Section 2.4) -----------------------------------


def excluded_keyword_reasons(job: Job, prefs: CandidatePreferences) -> list[str]:
    """Reject jobs whose title or description contains an excluded keyword."""

    if not prefs.excluded_keywords:
        return []
    haystack = f"{_norm(job.title)} {_norm(job.description)}"
    reasons: list[str] = []
    for keyword in prefs.excluded_keywords:
        if _contains_term(haystack, keyword):
            reasons.append(f"Posting contains the excluded keyword '{keyword}'.")
    return reasons


def _target_title_terms(target_titles: list[str]) -> set[str]:
    """Significant title tokens from the targets plus the AI/ML role vocabulary."""

    terms: set[str] = set(_AI_ML_TITLE_TERMS)
    for title in target_titles:
        for token in re.split(r"[^a-z0-9]+", _norm(title)):
            if token and token not in _GENERIC_TITLE_TOKENS:
                terms.add(token)
    return terms


def target_title_reasons(job: Job, prefs: CandidatePreferences) -> list[str]:
    """Reject titles that share no AI/ML/target term with the candidate's goals."""

    targets = prefs.target_job_titles
    if not targets:
        return []
    terms = _target_title_terms(targets)
    title_norm = _norm(job.title)
    if any(_contains_term(title_norm, term) for term in terms):
        return []
    return [
        f"Title '{job.title}' does not align with the candidate's target titles "
        f"({', '.join(targets)})."
    ]


# Ordered so rejection reasons read in a stable, sensible sequence.
FILTER_RULES: tuple[Rule, ...] = (
    company_exclusion_reasons,
    remote_only_reasons,
    location_reasons,
    experience_reasons,
    excluded_keyword_reasons,
    target_title_reasons,
)


def evaluate_job(job: Job, prefs: CandidatePreferences) -> list[str]:
    """Return every rejection reason for one job (empty means the job is accepted)."""

    reasons: list[str] = []
    for rule in FILTER_RULES:
        reasons.extend(rule(job, prefs))
    return reasons
