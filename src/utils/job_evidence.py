"""Stable evidence records derived from one supplied job posting."""

from __future__ import annotations

import re

from src.domain import EvidenceItem, Job


def build_job_evidence(job: Job) -> list[EvidenceItem]:
    """Return citeable posting evidence without introducing external facts."""

    prefix = f"job-{_safe_id(job.job_id)}"
    items = [
        EvidenceItem(
            evidence_id=f"{prefix}-title",
            source="job_posting",
            text=f"Job title: {job.title} at {job.company}",
            tags=["job_title"],
        ),
        EvidenceItem(
            evidence_id=f"{prefix}-description",
            source="job_posting",
            text=job.description,
            tags=["job_description"],
        ),
    ]
    if job.company_details:
        items.append(
            EvidenceItem(
                evidence_id=f"{prefix}-company-details",
                source="company_details",
                text=job.company_details,
                tags=["company_details"],
            )
        )
    requirement = job.experience_requirement.raw_text
    if not requirement and job.years_experience_required is not None:
        requirement = f"{job.years_experience_required}+ years"
    if requirement:
        items.append(
            EvidenceItem(
                evidence_id=f"{prefix}-experience",
                source="job_posting",
                text=f"Experience requirement: {requirement}",
                tags=["job_experience_requirement"],
            )
        )
    for index, skill in enumerate(job.required_skills, start=1):
        items.append(
            EvidenceItem(
                evidence_id=f"{prefix}-skill-{index:03d}",
                source="job_posting",
                text=f"Required skill: {skill}",
                tags=["job_skill_requirement", skill],
                metadata={"required_skill": skill},
            )
        )
    return items


def job_skill_evidence_id(job: Job, skill: str) -> str:
    """Return the stable evidence ID for a required skill."""

    normalized = skill.strip().casefold()
    for index, required in enumerate(job.required_skills, start=1):
        if required.strip().casefold() == normalized:
            return f"job-{_safe_id(job.job_id)}-skill-{index:03d}"
    return f"job-{_safe_id(job.job_id)}-description"


def job_evidence_id(job: Job, kind: str) -> str:
    return f"job-{_safe_id(job.job_id)}-{kind}"


def _safe_id(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip()).strip("-") or "unknown"
