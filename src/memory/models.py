"""Persistent memory models."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import Field

from src.schemas.common import StrictBaseModel


class MemoryProvenance(StrictBaseModel):
    """Where a candidate memory fact came from."""

    source: str
    review_round: int | None = None
    original_statement: str
    related_job_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


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
