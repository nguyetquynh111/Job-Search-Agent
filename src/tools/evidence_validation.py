"""Deterministic semantic checks for evidence-backed candidate claims.

The checks in this module deliberately prefer false negatives to invented
candidate facts.  IDs establish provenance; these helpers separately establish
that the cited record says the thing being claimed.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from src.schemas.common import EvidenceItem
from src.tools.skill_matching import (
    canonicalize,
    category_members,
    skill_in_text,
)

CANDIDATE_SOURCES = {"resume", "portfolio", "master_skills", "memory"}
JOB_SOURCES = {"job_posting", "company_details"}

_GENERIC_WORDS = {
    "align",
    "aligned",
    "candidate",
    "centered",
    "company",
    "demonstrated",
    "evidence",
    "evidenced",
    "experience",
    "focus",
    "job",
    "match",
    "project",
    "required",
    "resume",
    "role",
    "skill",
    "skills",
    "supported",
    "using",
    "with",
    "work",
}
_WORD_RE = re.compile(r"[a-z0-9][a-z0-9+#./-]*", re.IGNORECASE)
_NEGATION_TEMPLATE = (
    r"\b(?:no|not|never|without|lack(?:s|ed|ing)?|"
    r"do(?:es)?\s+not\s+(?:know|use|have)|"
    r"did\s+not\s+(?:know|use|have))\b[^.!?;]{{0,45}}{term}"
)


def evidence_supports_skill(item: EvidenceItem, skill: str) -> bool:
    """Return whether one record explicitly and non-negatively supports a skill."""

    canonical = canonicalize(skill)
    if not canonical or _term_is_negated(item.text, canonical):
        return False
    tag_canonicals = {canonicalize(tag) for tag in item.tags}
    if canonical in tag_canonicals or skill_in_text(canonical, item.text):
        return True

    # A named concrete tool can satisfy an explicitly curated broad category.
    members = {canonicalize(member) for member in category_members(canonical)}
    if not members:
        return False
    return any(
        member in tag_canonicals or skill_in_text(member, item.text)
        for member in members
    )


def evidence_supports_keyword(item: EvidenceItem, keyword: str) -> bool:
    """Return whether a keyword/phrase is explicit in text or structured tags."""

    canonical = canonicalize(keyword)
    if not canonical or _term_is_negated(item.text, canonical):
        return False
    if evidence_supports_skill(item, keyword):
        return True
    normalized = _normalize(keyword)
    return bool(
        normalized
        and (
            _bounded_phrase(normalized, _normalize(item.text))
            or any(normalized == _normalize(tag) for tag in item.tags)
        )
    )


def evidence_supports_project(item: EvidenceItem, project_name: str) -> bool:
    """Require an exact portfolio project identity, not a vaguely similar record."""

    if item.source != "portfolio":
        return False
    expected = _normalize(project_name)
    if not expected:
        return False
    names = {
        _normalize(str(item.metadata.get("project_name") or "")),
        _normalize(_labeled_value(item.text, "PROJECT_NAME") or ""),
    }
    return expected in names


def job_evidence_supports_skill(item: EvidenceItem, skill: str) -> bool:
    """Match an exact skill to a posted skill or its curated broad category."""

    if item.source not in JOB_SOURCES:
        return False
    if evidence_supports_skill(item, skill):
        return True
    canonical = canonicalize(skill)
    for tag in item.tags:
        members = {
            canonicalize(member) for member in category_members(canonicalize(tag))
        }
        if canonical in members:
            return True
    return False


def evidence_supports_statement(claim: str, item: EvidenceItem) -> bool:
    """Conservatively match a narrative claim to the cited source text.

    Exact numbers must be present.  In addition, at least two meaningful claim
    terms (or the only meaningful term for a short claim) must occur in the
    evidence.  This accepts harmless paraphrasing while rejecting unrelated or
    content-free citations.
    """

    claim_numbers = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", claim))
    evidence_numbers = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", item.text))
    if claim_numbers and not claim_numbers.issubset(evidence_numbers):
        return False

    claim_terms = _meaningful_terms(claim)
    evidence_text = _normalize(item.text)
    evidence_tags = " ".join(_normalize(tag) for tag in item.tags)
    matched = {
        term
        for term in claim_terms
        if _bounded_phrase(term, evidence_text)
        or _bounded_phrase(term, evidence_tags)
    }
    required = 1 if len(claim_terms) <= 2 else 2
    return len(matched) >= required


def supporting_ids_for_skill(
    skill: str,
    evidence_ids: Iterable[str],
    evidence_by_id: dict[str, EvidenceItem],
    *,
    sources: set[str] | None = None,
) -> list[str]:
    """Return only cited IDs whose records semantically support ``skill``."""

    supported: list[str] = []
    for evidence_id in evidence_ids:
        item = evidence_by_id.get(evidence_id)
        if item is None or (sources is not None and item.source not in sources):
            continue
        if evidence_supports_skill(item, skill):
            supported.append(evidence_id)
    return list(dict.fromkeys(supported))


def supporting_ids_for_statement(
    claim: str,
    evidence_ids: Iterable[str],
    evidence_by_id: dict[str, EvidenceItem],
    *,
    sources: set[str] | None = None,
) -> list[str]:
    """Return only citations that contain meaningful support for a statement."""

    supported: list[str] = []
    for evidence_id in evidence_ids:
        item = evidence_by_id.get(evidence_id)
        if item is None or (sources is not None and item.source not in sources):
            continue
        if evidence_supports_statement(claim, item):
            supported.append(evidence_id)
    return list(dict.fromkeys(supported))


def _meaningful_terms(value: str) -> set[str]:
    terms: set[str] = set()
    for token in _WORD_RE.findall(_normalize(value)):
        token = token.strip("./-")
        if len(token) < 3 or token in _GENERIC_WORDS or token.isdigit():
            continue
        terms.add(token)
    return terms


def _term_is_negated(text: str, canonical: str) -> bool:
    for form in {canonical, *category_members(canonical)}:
        escaped = re.escape(_normalize(str(form)))
        if re.search(
            _NEGATION_TEMPLATE.format(term=rf"(?<![a-z0-9]){escaped}(?![a-z0-9])"),
            _normalize(text),
        ):
            return True
    return False


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().casefold())


def _bounded_phrase(phrase: str, text: str) -> bool:
    return bool(
        re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text)
    )


def _labeled_value(text: str, label: str) -> str | None:
    match = re.search(rf"(?mi)^{re.escape(label)}:\s*(.+?)\s*$", text)
    return match.group(1).strip() if match else None
