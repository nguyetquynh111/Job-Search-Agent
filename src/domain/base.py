"""Shared validation primitives and evidence contracts."""

from __future__ import annotations

import json
import math
import re
import unicodedata
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


def normalize_string_list(value: Any) -> list[str]:
    """Parse common list representations and de-duplicate values in input order."""

    if value is None or _is_nan(value):
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                decoded = json.loads(stripped)
            except (json.JSONDecodeError, TypeError):
                decoded = None
            if isinstance(decoded, list):
                value = decoded
            else:
                value = re.split(r"[;,]", stripped)
        else:
            value = re.split(r"[;,]", stripped)
    elif not isinstance(value, (list, tuple, set)):
        value = [value]

    normalized: list[str] = []
    seen: set[str] = set()
    for item in value:
        if item is None or _is_nan(item):
            continue
        display_value = str(item).strip()
        if not display_value:
            continue
        comparison_key = display_value.casefold()
        if comparison_key in seen:
            continue
        seen.add(comparison_key)
        normalized.append(display_value)
    return normalized


def company_comparison_key(value: str) -> str:
    """Return a normalized key without changing the company's display name."""

    normalized = unicodedata.normalize("NFKC", value).casefold().strip()
    return " ".join(part for part in re.split(r"[\W_]+", normalized) if part)


def _is_nan(value: Any) -> bool:
    return isinstance(value, float) and math.isnan(value)


class StrictBaseModel(BaseModel):
    """Base model that rejects unexpected fields at integration boundaries."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class EvidenceItem(StrictBaseModel):
    """A source-backed evidence item that tools may cite by ID."""

    evidence_id: str
    source: str
    text: str
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("evidence_id", "source", "text", mode="before")
    @classmethod
    def trim_text(_cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value

    @field_validator("tags", mode="before")
    @classmethod
    def normalize_tags(_cls, value: Any) -> list[str]:
        return normalize_string_list(value)


class EvidenceClaim(StrictBaseModel):
    """Evidence-supported statement produced by fit analysis."""

    claim: str
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.8, ge=0, le=1)
    notes: str | None = None


class ProjectSwap(StrictBaseModel):
    """A recommended portfolio/resume project substitution."""

    remove_project: str | None = None
    add_project: str
    rationale: str
    evidence_ids: list[str] = Field(default_factory=list)


class ChangeLogEntry(StrictBaseModel):
    """Resume or cover-letter artifact change log entry."""

    change_id: str
    section: str
    description: str
    before_text: str
    after_text: str
    reason: str
    evidence_ids: list[str] = Field(min_length=1)
    source_location: str = ""
    validation_result: str = ""
