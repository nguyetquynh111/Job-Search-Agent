"""Job schemas."""

from __future__ import annotations

import re
from typing import Any

from pydantic import AnyHttpUrl, Field, TypeAdapter, field_validator, model_validator

from src.schemas.common import StrictBaseModel, normalize_string_list

_HTTP_URL_ADAPTER = TypeAdapter(AnyHttpUrl)
_NOT_SPECIFIED_PATTERNS = (
    "not specified",
    "not stated",
    "no explicit years",
    "no years stated",
    "professional years of experience are not required",
    "professional work experience is not required",
    "prior full-time professional experience is not required",
    "prior finance experience and professional work experience are not required",
)


class ExperienceRequirement(StrictBaseModel):
    """Normalized bounds plus the exact experience text supplied by the source."""

    minimum_years: float | None = Field(default=None, ge=0)
    maximum_years: float | None = Field(default=None, ge=0)
    raw_text: str | None = None

    @field_validator("raw_text", mode="before")
    @classmethod
    def blank_raw_text(cls, value: Any) -> Any:
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return value

    @model_validator(mode="after")
    def validate_bounds(self) -> "ExperienceRequirement":
        if (
            self.minimum_years is not None
            and self.maximum_years is not None
            and self.minimum_years > self.maximum_years
        ):
            raise ValueError("minimum_years cannot exceed maximum_years")
        return self


def parse_experience_requirement(value: Any) -> ExperienceRequirement:
    """Normalize numeric, exact, minimum, and range experience representations."""

    if value is None:
        return ExperienceRequirement()
    if isinstance(value, bool):
        return ExperienceRequirement(raw_text=str(value))
    if isinstance(value, (int, float)):
        if value != value:  # NaN
            return ExperienceRequirement()
        numeric = float(value)
        return ExperienceRequirement(
            minimum_years=numeric,
            maximum_years=numeric,
            raw_text=str(value),
        )

    raw_text = str(value).strip()
    if not raw_text:
        return ExperienceRequirement()
    lowered = raw_text.casefold()
    if any(pattern in lowered for pattern in _NOT_SPECIFIED_PATTERNS):
        return ExperienceRequirement(raw_text=raw_text)

    number = r"(\d+(?:\.\d+)?)"
    exact = re.fullmatch(rf"{number}\s*(?:years?|yrs?)?(?:\s+of experience)?", lowered)
    if exact:
        years = float(exact.group(1))
        return ExperienceRequirement(
            minimum_years=years,
            maximum_years=years,
            raw_text=raw_text,
        )

    minimum = re.fullmatch(
        rf"{number}\s*\+\s*(?:years?|yrs?)(?:\s+of experience)?", lowered
    ) or re.fullmatch(
        rf"(?:at least|minimum|min\.?|more than)\s+{number}\s*"
        rf"(?:years?|yrs?)(?:\s+of experience)?",
        lowered,
    )
    if minimum is None:
        qualified_minimum = re.fullmatch(
            rf"{number}\s*\+\s*(?:years?|yrs?)\b.*", lowered
        )
        year_mentions = re.findall(r"\d+(?:\.\d+)?\s*\+?\s*(?:years?|yrs?)\b", lowered)
        if qualified_minimum and len(year_mentions) == 1:
            minimum = qualified_minimum
    if minimum:
        return ExperienceRequirement(
            minimum_years=float(minimum.group(1)),
            raw_text=raw_text,
        )

    year_range = re.fullmatch(
        rf"{number}\s*(?:-|–|—|to)\s*{number}\s*"
        rf"(?:years?|yrs?)(?:\s+of experience)?",
        lowered,
    )
    if year_range:
        return ExperienceRequirement(
            minimum_years=float(year_range.group(1)),
            maximum_years=float(year_range.group(2)),
            raw_text=raw_text,
        )

    return ExperienceRequirement(raw_text=raw_text)


class Job(StrictBaseModel):
    """Normalized job posting."""

    job_id: str
    title: str
    company: str
    industry_domain: str = ""
    location: str = ""
    remote: bool | None = None
    description: str
    required_skills: list[str] = Field(default_factory=list)
    years_experience_required: int | float | None = Field(default=None, ge=0)
    experience_requirement: ExperienceRequirement = Field(
        default_factory=ExperienceRequirement
    )
    company_details: str = ""
    url: str = ""
    # Original fields remain available to existing callers and fixtures.
    requirements: list[str] = Field(default_factory=list)
    salary_min: int | None = None
    salary_max: int | None = None
    source: str | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_fields(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        payload = dict(value)

        if "required_skills" not in payload:
            payload["required_skills"] = payload.get("requirements", [])
        if "requirements" not in payload:
            payload["requirements"] = payload.get("required_skills", [])

        years_value = payload.get("years_experience_required")
        if years_value is None and "years_required" in payload:
            years_value = payload.pop("years_required")
            payload["years_experience_required"] = years_value
        if "experience_requirement" not in payload and years_value is not None:
            normalized = parse_experience_requirement(years_value)
            payload["experience_requirement"] = normalized.model_dump()
            payload["years_experience_required"] = normalized.minimum_years
        return payload

    @field_validator("job_id", "title", "company", "description", mode="before")
    @classmethod
    def trim_required_text(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value

    @field_validator("job_id", "title", "company", "description")
    @classmethod
    def reject_blank_required_text(cls, value: str, info: Any) -> str:
        if not value:
            raise ValueError(f"{info.field_name} must not be blank")
        return value

    @field_validator(
        "industry_domain", "location", "company_details", "url", mode="before"
    )
    @classmethod
    def normalize_optional_display_text(cls, value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()

    @field_validator("source", mode="before")
    @classmethod
    def normalize_source(cls, value: Any) -> Any:
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return value

    @field_validator("required_skills", "requirements", mode="before")
    @classmethod
    def normalize_skills(cls, value: Any) -> list[str]:
        return normalize_string_list(value)

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        if value:
            _HTTP_URL_ADAPTER.validate_python(value)
        return value

    @model_validator(mode="after")
    def synchronize_compatibility_fields(self) -> "Job":
        if not self.required_skills and self.requirements:
            self.required_skills = list(self.requirements)
        if not self.requirements and self.required_skills:
            self.requirements = list(self.required_skills)
        if (
            self.years_experience_required is None
            and self.experience_requirement.minimum_years is not None
        ):
            minimum = self.experience_requirement.minimum_years
            self.years_experience_required = (
                int(minimum) if minimum.is_integer() else minimum
            )
        return self


class RejectedJob(StrictBaseModel):
    """Filtered-out job with one or more reasons."""

    job: Job
    reasons: list[str] = Field(min_length=1)
