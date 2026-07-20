"""Simple review-comment memory extraction."""

from __future__ import annotations

import re
from itertools import count

from src.memory.models import MemoryFact, MemoryProvenance

_FACT_COUNTER = count(1)
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


def extract_memory_facts(
    comments_by_job: dict[str, str], review_round: int
) -> list[MemoryFact]:
    """Extract durable candidate facts from human-review comments."""

    facts: list[MemoryFact] = []
    for job_id, comment in comments_by_job.items():
        normalized = comment.strip()
        if not normalized or _looks_like_editing_preference(normalized):
            continue
        for fact_type, value in _extract_known_technologies(normalized):
            facts.append(
                MemoryFact(
                    fact_id=f"mem-{next(_FACT_COUNTER):04d}",
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
    lowered = comment.lower()
    if not re.search(r"\b(i|my|me)\b", lowered):
        return []
    found: list[tuple[str, str]] = []
    for token, canonical in _KNOWN_TECHNOLOGIES.items():
        if re.search(rf"\b{re.escape(token)}\b", lowered):
            found.append(("skill", canonical))
    return found


def _looks_like_editing_preference(comment: str) -> bool:
    lowered = comment.lower()
    return any(phrase in lowered for phrase in _EDITING_PREFERENCES)
