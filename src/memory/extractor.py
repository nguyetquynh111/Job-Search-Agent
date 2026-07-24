"""Extract durable, reviewer-stated candidate facts from review comments."""

from __future__ import annotations

import re
from uuid import uuid4

from src.memory.models import MemoryFact, MemoryProvenance

_KNOWN_TECHNOLOGIES = {
    "graphql": "GraphQL",
    "kubernetes": "Kubernetes",
    "docker": "Docker",
    "aws": "AWS",
    "azure": "Azure",
    "gcp": "GCP",
    "react": "React",
    "typescript": "TypeScript",
    "langgraph": "LangGraph",
    "langchain": "LangChain",
    "fastapi": "FastAPI",
    "postgres": "Postgres",
    "postgresql": "Postgres",
}
_EDITING_PREFERENCES = (
    "shorter",
    "friendlier",
    "tone",
    "move this",
    "move the",
    "bullet upward",
    "format",
    "layout",
    "font",
)
_SKILL_PATTERNS = (
    re.compile(
        r"\bI\s+(?:also\s+)?(?:know|use|used|have used|work with|have worked with|"
        r"am proficient in|am skilled in|am experienced with)\s+"
        r"(?P<skills>[^.!?;]+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bmy\s+(?:technical\s+)?skills?\s+(?:include|are)\s+"
        r"(?P<skills>[^.!?;]+)",
        re.IGNORECASE,
    ),
)
_TRAILING_CONTEXT = re.compile(
    r"\s+(?:in|on|for|during)\s+(?:my\s+)?(?:previous|past|current|a|the)\b.*$",
    re.IGNORECASE,
)
_EXPERIENCE_PATTERN = re.compile(
    r"\bI\s+have\s+(?P<years>\d+(?:\.\d+)?)\+?\s+years?\s+of\s+"
    r"(?:professional\s+)?experience(?:\s+(?:in|with)\s+(?P<area>[^.!?;]+))?",
    re.IGNORECASE,
)
_CANDIDATE_FACT_PATTERNS = (
    re.compile(
        r"\bI\s+(?:hold|earned|completed)\s+[^.!?;]*"
        r"(?:degree|certification|certificate)\b[^.!?;]*",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bI\s+am\s+(?:legally\s+)?authorized\s+to\s+work\b[^.!?;]*",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bI\s+(?:do not|don't)\s+require\s+(?:visa\s+)?sponsorship\b[^.!?;]*",
        re.IGNORECASE,
    ),
)


def extract_memory_facts(
    comments_by_job: dict[str, str], review_round: int
) -> list[MemoryFact]:
    """Extract only explicit skills and candidate facts, preserving provenance.

    Review text is untrusted as resume evidence unless it contains a first-person
    assertion. Editing requests are deliberately ignored. The extraction is
    deterministic so the exact statement that justified each memory entry remains
    auditable.
    """

    facts: list[MemoryFact] = []
    seen: set[tuple[str, str]] = set()
    for job_id, comment in comments_by_job.items():
        normalized = " ".join(comment.split())
        if not normalized:
            continue

        extracted = [
            *_extract_known_technologies(normalized),
            *_extract_asserted_skills(normalized),
            *_extract_candidate_facts(normalized),
        ]
        for fact_type, value in extracted:
            key = (fact_type.casefold(), value.casefold())
            if key in seen:
                continue
            seen.add(key)
            facts.append(
                MemoryFact(
                    fact_id=f"mem-{uuid4().hex[:12]}",
                    fact_type=fact_type,
                    canonical_value=value,
                    provenance=MemoryProvenance(
                        source="human_review",
                        review_round=review_round,
                        original_statement=normalized,
                        related_job_id=job_id,
                    ),
                )
            )
    return facts


def _extract_known_technologies(comment: str) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for token, canonical in _KNOWN_TECHNOLOGIES.items():
        escaped = re.escape(token)
        direct_assertion = re.search(
            rf"\b(?:I|my)\b[^.!?;]{{0,100}}\b{escaped}\b",
            comment,
            re.IGNORECASE,
        )
        anaphoric_assertion = re.search(
            rf"\b(?:add|include)\s+{escaped}\b[^.!?]*[.!?]\s*"
            r"I\s+(?:know|use|used|have used|work with)\s+it\b",
            comment,
            re.IGNORECASE,
        )
        if direct_assertion or anaphoric_assertion:
            found.append(("skill", canonical))
    return found


def _extract_asserted_skills(comment: str) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for pattern in _SKILL_PATTERNS:
        for match in pattern.finditer(comment):
            raw = _TRAILING_CONTEXT.sub("", match.group("skills")).strip(" ,:-")
            for value in _split_skill_list(raw):
                if _looks_like_skill(value):
                    found.append(("skill", _canonicalize_skill(value)))
    return found


def _extract_candidate_facts(comment: str) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for match in _EXPERIENCE_PATTERN.finditer(comment):
        years = match.group("years")
        area = (match.group("area") or "").strip(" ,:-")
        value = f"{years} years of experience"
        if area:
            value = f"{value} in {area}"
        found.append(("experience", value))
    for pattern in _CANDIDATE_FACT_PATTERNS:
        for match in pattern.finditer(comment):
            value = match.group(0).strip(" ,:-")
            found.append(("candidate_fact", value))
    return found


def _split_skill_list(value: str) -> list[str]:
    value = re.sub(r"^(?:the\s+)?(?:tools?|technologies?)\s+", "", value, flags=re.I)
    parts = re.split(r"\s*,\s*|\s+(?:and|&)\s+", value)
    cleaned = [
        re.sub(r"^(?:and|&)\s+", "", part.strip(" ,:-"), flags=re.I)
        for part in parts
    ]
    return [part for part in cleaned if part]


def _looks_like_skill(value: str) -> bool:
    lowered = value.casefold()
    if not value or lowered in {"it", "them", "this", "that"}:
        return False
    if any(preference in lowered for preference in _EDITING_PREFERENCES):
        return False
    return len(value.split()) <= 5 and len(value) <= 60


def _canonicalize_skill(value: str) -> str:
    known = _KNOWN_TECHNOLOGIES.get(value.casefold())
    return known or value.strip()
