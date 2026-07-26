"""Human-review memory models, extraction, validation, and JSON storage."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import Field, ValidationError

from src.domain import StrictBaseModel

logger = logging.getLogger(__name__)


class MemoryProvenance(StrictBaseModel):
    """Where a candidate memory fact came from."""

    source: str
    review_round: int | None = None
    original_statement: str
    related_job_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryConflict(StrictBaseModel):
    """Recorded conflict and deterministic resolution between memory facts."""

    conflict_id: str
    old_fact: dict[str, Any]
    new_fact: dict[str, Any]
    source: str
    timestamp: str
    affected_job_id: str | None = None
    affected_run_id: str | None = None
    resolution: str
    active_value: str


class MemoryFact(StrictBaseModel):
    """Candidate memory fact persisted to JSON."""

    fact_id: str
    fact_type: str
    canonical_value: str
    provenance: MemoryProvenance
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    active: bool = True
    conflicts: list[MemoryConflict] = Field(default_factory=list)

    @property
    def deduplication_key(self) -> tuple[str, str]:
        """Return the case-insensitive identity used by the JSON store."""

        return (
            self.fact_type.strip().casefold(),
            " ".join(self.canonical_value.split()).casefold(),
        )


def memory_fact_to_evidence(fact: MemoryFact) -> dict[str, Any]:
    """Represent a memory fact as evidence for downstream tools."""

    return {
        "evidence_id": fact.fact_id,
        "source": "memory",
        "text": f"{fact.fact_type}: {fact.canonical_value}",
        "tags": [fact.fact_type, "memory"],
        "metadata": {
            "provenance": fact.provenance.model_dump(),
            "created_at": fact.created_at,
        },
    }


class MemoryStoreError(RuntimeError):
    """Raised when memory cannot be loaded or written safely."""


class JSONMemoryStore:
    """Persistent JSON-file candidate memory store."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> list[MemoryFact]:
        """Load active and inactive memory facts from JSON."""

        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("[]", encoding="utf-8")
            return []
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise MemoryStoreError(f"Memory file is corrupted: {self.path}") from exc
        if not isinstance(raw, list):
            raise MemoryStoreError(
                f"Memory file must contain a JSON array: {self.path}"
            )
        try:
            return [MemoryFact.model_validate(item) for item in raw]
        except ValidationError as exc:
            raise MemoryStoreError(
                f"Memory file contains invalid entries: {self.path}"
            ) from exc

    def append_many(
        self,
        facts: list[MemoryFact],
        *,
        run_id: str | None = None,
    ) -> list[MemoryFact]:
        """Append new facts and persist the full memory file."""

        if not facts:
            return self.load()
        current = self.load()
        known = {fact.deduplication_key for fact in current if fact.active}
        deduped: list[MemoryFact] = []
        for fact in facts:
            key = fact.deduplication_key
            if key in known:
                logger.info("Skipping duplicate memory fact: %s", fact.canonical_value)
                continue
            current, fact = _resolve_conflicts(current, fact, run_id=run_id)
            known.add(key)
            deduped.append(fact)
        updated = [*current, *deduped]
        self._write(updated)
        return updated

    def reset(self) -> None:
        """Reset memory to an empty JSON array."""

        self._write([])

    def _write(self, facts: list[MemoryFact]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = [fact.model_dump() for fact in facts]
        tmp_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp_path.replace(self.path)


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
            rf"\bI\s+(?:also\s+)?(?:know|use|used|have used|work with|"
            rf"have worked with|am proficient in|am skilled in|"
            rf"am experienced with)\b[^.!?;]{{0,80}}\b{escaped}\b",
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
        re.sub(r"^(?:and|&)\s+", "", part.strip(" ,:-"), flags=re.I) for part in parts
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


def _resolve_conflicts(
    current: list[MemoryFact],
    new_fact: MemoryFact,
    *,
    run_id: str | None,
) -> tuple[list[MemoryFact], MemoryFact]:
    conflict_group = _conflict_group(new_fact)
    if conflict_group is None:
        return current, new_fact
    resolved = list(current)
    for index, old_fact in enumerate(current):
        if not old_fact.active:
            continue
        if _conflict_group(old_fact) != conflict_group:
            continue
        if old_fact.canonical_value.casefold() == new_fact.canonical_value.casefold():
            continue
        timestamp = datetime.now(timezone.utc).isoformat()
        conflict = MemoryConflict(
            conflict_id=f"conflict-{uuid4().hex[:12]}",
            old_fact=_fact_snapshot(old_fact),
            new_fact=_fact_snapshot(new_fact),
            source=new_fact.provenance.source,
            timestamp=timestamp,
            affected_job_id=new_fact.provenance.related_job_id,
            affected_run_id=run_id,
            resolution=(
                "latest_validated_human_review_fact_wins; previous fact retained "
                "inactive for audit"
            ),
            active_value=new_fact.canonical_value,
        )
        resolved[index] = old_fact.model_copy(
            update={
                "active": False,
                "conflicts": [*old_fact.conflicts, conflict],
            }
        )
        new_fact = new_fact.model_copy(
            update={"active": True, "conflicts": [*new_fact.conflicts, conflict]}
        )
    return resolved, new_fact


def _conflict_group(fact: MemoryFact) -> str | None:
    fact_type = fact.fact_type.casefold()
    value = fact.canonical_value.casefold()
    if fact_type == "experience":
        return "experience"
    if fact_type == "candidate_fact" and any(
        token in value for token in ("authorized to work", "visa", "sponsorship")
    ):
        return "work_authorization"
    return None


def _fact_snapshot(fact: MemoryFact) -> dict[str, Any]:
    return {
        "fact_id": fact.fact_id,
        "fact_type": fact.fact_type,
        "canonical_value": fact.canonical_value,
        "source": fact.provenance.source,
        "created_at": fact.created_at,
        "related_job_id": fact.provenance.related_job_id,
    }


def validate_memory_facts(
    facts: list[MemoryFact], comments_by_job: dict[str, str]
) -> tuple[list[MemoryFact], list[str]]:
    """Revalidate extracted facts against the exact reviewer statement.

    Extraction is intentionally deterministic, but storage has its own guard so a
    future extractor change cannot silently turn an editing request into candidate
    evidence.
    """

    normalized_comments = {
        job_id: " ".join(comment.split()) for job_id, comment in comments_by_job.items()
    }
    valid: list[MemoryFact] = []
    failures: list[str] = []
    for fact in facts:
        job_id = fact.provenance.related_job_id
        statement = normalized_comments.get(job_id or "", "")
        if (
            fact.provenance.source != "human_review"
            or not statement
            or fact.provenance.original_statement != statement
        ):
            failures.append(
                f"rejected memory fact {fact.fact_id}: provenance does not match "
                "the submitted reviewer feedback"
            )
            continue
        reextracted = extract_memory_facts({job_id or "unknown": statement}, 0)
        supported = any(
            candidate.fact_type.casefold() == fact.fact_type.casefold()
            and candidate.canonical_value.casefold() == fact.canonical_value.casefold()
            for candidate in reextracted
        )
        if not supported:
            failures.append(
                f"rejected memory fact {fact.fact_id}: reviewer statement does not "
                f"explicitly assert {fact.canonical_value!r}"
            )
            continue
        valid.append(fact)
    return valid, failures
