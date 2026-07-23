"""Common schema primitives shared across tools and graph state."""

from __future__ import annotations

import json
import math
import re
import unicodedata
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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
    def trim_text(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value

    @field_validator("tags", mode="before")
    @classmethod
    def normalize_tags(cls, value: Any) -> list[str]:
        return normalize_string_list(value)


class CandidatePreferences(StrictBaseModel):
    """Candidate job preferences used by the filtering tool."""

    target_job_titles: list[str] = Field(default_factory=list)
    target_titles: list[str] = Field(default_factory=list)
    preferred_locations: list[str] = Field(default_factory=list)
    locations: list[str] = Field(default_factory=list)
    remote_only: bool = False
    remote: bool = False
    years_of_experience: int | float | None = Field(default=None, ge=0)
    excluded_companies: list[str] = Field(default_factory=list)
    job_types: list[str] = Field(default_factory=lambda: ["full-time"])
    min_salary: int | None = None
    excluded_keywords: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_names(cls, value: Any) -> Any:
        """Accept both assignment names and the original public field names."""

        if not isinstance(value, dict):
            return value
        payload = dict(value)

        preferred_locations = payload.get("preferred_locations")
        if preferred_locations is None:
            preferred_locations = payload.get("locations")
        payload["preferred_locations"] = preferred_locations or []
        payload["locations"] = payload["preferred_locations"]

        target_job_titles = payload.get("target_job_titles")
        if target_job_titles is None:
            target_job_titles = payload.get("target_titles")
        payload["target_job_titles"] = target_job_titles or []
        payload["target_titles"] = payload["target_job_titles"]

        if "remote_only" in payload:
            remote_only = payload["remote_only"]
        else:
            remote_only = payload.get("remote", False)
        payload["remote_only"] = remote_only
        payload["remote"] = remote_only

        if "excluded_companies" not in payload:
            payload["excluded_companies"] = payload.get("companies_to_exclude", [])
        payload.pop("companies_to_exclude", None)
        return payload

    @field_validator(
        "target_job_titles",
        "target_titles",
        "preferred_locations",
        "locations",
        "excluded_companies",
        "job_types",
        "excluded_keywords",
        mode="before",
    )
    @classmethod
    def normalize_lists(cls, value: Any) -> list[str]:
        return normalize_string_list(value)

    @field_validator("years_of_experience", mode="before")
    @classmethod
    def normalize_years_of_experience(cls, value: Any) -> Any:
        if value is None or _is_nan(value):
            return None
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            return float(stripped) if "." in stripped else int(stripped)
        return value

    @property
    def excluded_company_comparison_keys(self) -> set[str]:
        """Company keys intended only for filtering comparisons."""

        return {company_comparison_key(name) for name in self.excluded_companies}


class ResumeData(StrictBaseModel):
    """Structured content extracted from the repository's LaTeX resume contract."""

    plain_text: str
    professional_summary: str | None = None
    skills: list[str] = Field(default_factory=list)
    education: list[str] = Field(default_factory=list)
    experience: list[str] = Field(default_factory=list)
    projects: list[str] = Field(default_factory=list)
    evidence_items: list[EvidenceItem] = Field(default_factory=list)

    @field_validator("skills", "education", "experience", "projects", mode="before")
    @classmethod
    def normalize_lists(cls, value: Any) -> list[str]:
        return normalize_string_list(value)


class PortfolioProject(StrictBaseModel):
    """Portfolio project available for resume tailoring decisions."""

    project_id: str
    name: str
    description: str
    technologies: list[str] = Field(default_factory=list)
    domains: list[str] = Field(default_factory=list)
    industries: list[str] = Field(default_factory=list)
    period: str | None = None
    role: str | None = None
    organization_alias: str | None = None
    resume_swap_value: str | None = None
    keywords: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)

    @field_validator("project_id", "name", "description", mode="before")
    @classmethod
    def trim_required_text(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value

    @field_validator(
        "period", "role", "organization_alias", "resume_swap_value", mode="before"
    )
    @classmethod
    def blank_optional_text(cls, value: Any) -> Any:
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return value

    @field_validator(
        "technologies",
        "domains",
        "industries",
        "keywords",
        "evidence_ids",
        mode="before",
    )
    @classmethod
    def normalize_lists(cls, value: Any) -> list[str]:
        return normalize_string_list(value)


class CandidateProfile(StrictBaseModel):
    """Candidate profile loaded from YAML or JSON input."""

    candidate_id: str
    name: str
    email: str | None = None
    persona: dict[str, Any] = Field(default_factory=dict)
    preferences: CandidatePreferences = Field(default_factory=CandidatePreferences)
    # ``skills`` remains resume-only. Master skills are intentionally separate.
    skills: list[str] = Field(default_factory=list)
    master_skills: list[str] = Field(default_factory=list)
    education: list[str] = Field(default_factory=list)
    experience: list[str] = Field(default_factory=list)
    resume_content: str | None = None
    resume_projects: list[str] = Field(default_factory=list)
    resume_evidence: list[EvidenceItem] = Field(default_factory=list)
    master_skill_evidence: list[EvidenceItem] = Field(default_factory=list)
    portfolio_evidence: list[EvidenceItem] = Field(default_factory=list)

    @field_validator(
        "skills",
        "master_skills",
        "education",
        "experience",
        "resume_projects",
        mode="before",
    )
    @classmethod
    def normalize_profile_lists(cls, value: Any) -> list[str]:
        return normalize_string_list(value)

    @field_validator("candidate_id", "name", mode="before")
    @classmethod
    def trim_identity(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value

    @field_validator("email", "resume_content", mode="before")
    @classmethod
    def blank_optional_text(cls, value: Any) -> Any:
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return value


class Portfolio(StrictBaseModel):
    """Candidate portfolio file."""

    projects: list[PortfolioProject] = Field(default_factory=list)
    evidence_items: list[EvidenceItem] = Field(default_factory=list)


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
    evidence_ids: list[str] = Field(default_factory=list)
