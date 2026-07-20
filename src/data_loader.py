"""Input loading and validation utilities."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import pandas as pd
import yaml

from src.schemas.common import (
    CandidatePreferences,
    CandidateProfile,
    EvidenceItem,
    Portfolio,
    PortfolioProject,
)
from src.schemas.jobs import Job

REQUIRED_JOB_COLUMNS = {
    "title",
    "company",
    "description",
}


class InputLoadError(RuntimeError):
    """Raised when a user-provided input file is missing or invalid."""


def load_jobs_csv(path: str | Path) -> list[Job]:
    """Load normalized job postings from a CSV file."""

    resolved = Path(path)
    if not resolved.exists():
        raise InputLoadError(f"Jobs CSV not found: {resolved}")
    frame = _normalize_job_columns(pd.read_csv(resolved))
    missing = REQUIRED_JOB_COLUMNS - set(frame.columns)
    if missing:
        raise InputLoadError(f"Jobs CSV missing required columns: {sorted(missing)}")
    jobs: list[Job] = []
    for index, row in enumerate(frame.to_dict(orient="records"), start=1):
        requirements_value = _optional_str(row.get("requirements")) or ""
        if isinstance(requirements_value, str):
            requirements_list = [
                item.strip()
                for item in re.split(r"[;,]", requirements_value)
                if item.strip()
            ]
        else:
            requirements_list = list(requirements_value)
        location = _optional_str(row.get("location")) or ""
        remote_value = row.get("remote")
        remote = (
            _parse_bool(remote_value)
            if remote_value is not None and not pd.isna(remote_value)
            else "remote" in location.lower()
        )
        jobs.append(
            Job(
                job_id=_optional_str(row.get("job_id")) or f"J{index:03d}",
                title=str(row["title"]),
                company=str(row["company"]),
                location=location,
                remote=remote,
                description=str(row["description"]),
                requirements=requirements_list,
                salary_min=_optional_int(row.get("salary_min")),
                salary_max=_optional_int(row.get("salary_max")),
                source=_optional_str(row.get("source")),
            )
        )
    return jobs


def _normalize_job_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Map common spreadsheet headings to the internal job schema."""

    aliases = {
        "job id": "job_id",
        "id": "job_id",
        "job title": "title",
        "role": "title",
        "company name": "company",
        "required skills": "requirements",
        "skills": "requirements",
        "salary minimum": "salary_min",
        "minimum salary": "salary_min",
        "salary maximum": "salary_max",
        "maximum salary": "salary_max",
        "url": "source",
        "job url": "source",
        "link": "source",
    }
    rename: dict[str, str] = {}
    for column in frame.columns:
        normalized = re.sub(r"[^a-z0-9]+", " ", str(column).lstrip("\ufeff").lower()).strip()
        if normalized.startswith("job description"):
            rename[column] = "description"
        elif normalized in aliases:
            rename[column] = aliases[normalized]
        else:
            snake_case = normalized.replace(" ", "_")
            if snake_case in {
                "job_id",
                "title",
                "company",
                "location",
                "remote",
                "description",
                "requirements",
                "salary_min",
                "salary_max",
                "source",
            }:
                rename[column] = snake_case
    return frame.rename(columns=rename)


def load_candidate_profile(path: str | Path) -> CandidateProfile:
    """Load either a legacy candidate profile or preferences-only YAML."""

    payload = _load_yaml(path, "Preferences")
    if "candidate_id" in payload and "name" in payload:
        return CandidateProfile.model_validate(payload)

    preferences_payload = payload.get("preferences", payload)
    preferences = CandidatePreferences.model_validate(preferences_payload)
    return CandidateProfile(
        candidate_id="uploaded-candidate",
        name="Candidate",
        preferences=preferences,
    )


def load_portfolio(path: str | Path) -> Portfolio:
    """Load a plain-text upload or a legacy YAML portfolio."""

    resolved = Path(path)
    if resolved.suffix.lower() == ".txt":
        return _load_text_portfolio(resolved)
    payload = _load_yaml(resolved, "Portfolio")
    return Portfolio.model_validate(payload)


def load_text_path(path: str | Path, label: str) -> str:
    """Validate that a text artifact path exists and return it as a string."""

    resolved = Path(path)
    if not resolved.exists():
        raise InputLoadError(f"{label} not found: {resolved}")
    return str(resolved)


def load_resume_evidence(path: str | Path) -> list[EvidenceItem]:
    """Convert an uploaded LaTeX resume into searchable candidate evidence."""

    resolved = Path(path)
    if not resolved.exists():
        raise InputLoadError(f"Resume not found: {resolved}")
    content = resolved.read_text(encoding="utf-8").strip()
    if not content:
        raise InputLoadError(f"Resume file is empty: {resolved}")
    plain_text = re.sub(r"%.*", " ", content)
    plain_text = re.sub(r"\\[A-Za-z@]+\*?(?:\[[^\]]*\])?", " ", plain_text)
    plain_text = re.sub(r"[{}]", " ", plain_text)
    plain_text = re.sub(r"\s+", " ", plain_text).strip()
    return [
        EvidenceItem(
            evidence_id="resume-upload-001",
            source="resume",
            text=plain_text,
            tags=[],
        )
    ]


def _load_text_portfolio(path: Path) -> Portfolio:
    """Parse blank-line-separated portfolio projects from a text file."""

    if not path.exists():
        raise InputLoadError(f"Portfolio file not found: {path}")
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        raise InputLoadError(f"Portfolio file is empty: {path}")

    blocks = [block.strip() for block in re.split(r"\n\s*\n", content) if block.strip()]
    projects: list[PortfolioProject] = []
    evidence_items: list[EvidenceItem] = []
    for index, block in enumerate(blocks, start=1):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        name = lines[0].lstrip("#- ").strip() or f"Portfolio project {index}"
        description_lines = [
            line for line in lines[1:] if not line.lower().startswith("technologies:")
        ]
        description = " ".join(description_lines) or name
        technology_line = next(
            (line for line in lines if line.lower().startswith("technologies:")), ""
        )
        technologies = [
            item.strip()
            for item in technology_line.partition(":")[2].split(",")
            if item.strip()
        ]
        evidence_id = f"portfolio-upload-{index:03d}"
        evidence_items.append(
            EvidenceItem(
                evidence_id=evidence_id,
                source="portfolio",
                text=block,
                tags=technologies,
            )
        )
        projects.append(
            PortfolioProject(
                project_id=f"UPLOAD-{index:03d}",
                name=name,
                description=description,
                technologies=technologies,
                evidence_ids=[evidence_id],
            )
        )
    return Portfolio(projects=projects, evidence_items=evidence_items)


def _load_yaml(path: str | Path, label: str) -> dict[str, Any]:
    resolved = Path(path)
    if not resolved.exists():
        raise InputLoadError(f"{label} file not found: {resolved}")
    with resolved.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise InputLoadError(f"{label} file must contain a mapping: {resolved}")
    return payload


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def _optional_int(value: Any) -> int | None:
    if pd.isna(value):
        return None
    if value in ("", None):
        return None
    return int(value)


def _optional_str(value: Any) -> str | None:
    if pd.isna(value):
        return None
    if value in ("", None):
        return None
    return str(value)
