"""Input loading and validation utilities."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import pandas as pd
from pydantic import AnyHttpUrl, TypeAdapter, ValidationError
import yaml

from src.schemas.common import (
    CandidatePreferences,
    CandidateProfile,
    EvidenceItem,
    Portfolio,
    PortfolioProject,
    ResumeData,
    normalize_string_list,
)
from src.schemas.jobs import Job, parse_experience_requirement

REQUIRED_JOB_COLUMNS = {
    "title",
    "company",
    "description",
}
_HTTP_URL_ADAPTER = TypeAdapter(AnyHttpUrl)


class InputLoadError(RuntimeError):
    """Raised when a user-provided input file is missing or invalid."""


def load_jobs_csv(path: str | Path) -> list[Job]:
    """Load assignment-style and legacy job CSV rows into normalized jobs."""

    resolved = Path(path)
    if not resolved.exists():
        raise InputLoadError(f"Jobs CSV not found: {resolved}")
    try:
        frame = _normalize_job_columns(pd.read_csv(resolved))
    except Exception as exc:
        raise InputLoadError(f"Unable to read jobs CSV: {resolved}") from exc
    missing = REQUIRED_JOB_COLUMNS - set(frame.columns)
    if missing:
        raise InputLoadError(f"Jobs CSV missing required columns: {sorted(missing)}")

    jobs: list[Job] = []
    for index, row in enumerate(frame.to_dict(orient="records"), start=1):
        try:
            required_skills_value = _first_present(
                row.get("required_skills"), row.get("requirements")
            )
            required_skills = normalize_string_list(required_skills_value)
            requirements = normalize_string_list(
                _first_present(row.get("requirements"), required_skills)
            )

            experience_source = _first_present(
                row.get("years_experience_required"),
                row.get("experience_requirement"),
            )
            experience_requirement = parse_experience_requirement(experience_source)
            minimum_years = experience_requirement.minimum_years
            years_experience_required: int | float | None = None
            if minimum_years is not None:
                years_experience_required = (
                    int(minimum_years) if minimum_years.is_integer() else minimum_years
                )

            location = _optional_str(row.get("location")) or ""
            remote = _parse_remote_status(row.get("remote"), location)
            url = _optional_str(row.get("url")) or ""
            source = _optional_str(row.get("source"))

            # URL/link headers explicitly identify a source URL. Retain ``source``
            # as well for callers written against the legacy schema.
            if url and source is None:
                source = url
            # A legacy ``source`` value is promoted only when it validates as an
            # HTTP(S) URL; labels such as "LinkedIn" remain source-only.
            if not url and source and _is_http_url(source):
                url = source

            jobs.append(
                Job(
                    job_id=_optional_str(row.get("job_id")) or f"J{index:03d}",
                    title=_optional_str(row.get("title")) or "",
                    company=_optional_str(row.get("company")) or "",
                    industry_domain=_optional_str(row.get("industry_domain")) or "",
                    location=location,
                    remote=remote,
                    required_skills=required_skills,
                    years_experience_required=years_experience_required,
                    experience_requirement=experience_requirement,
                    description=_optional_str(row.get("description")) or "",
                    company_details=_optional_str(row.get("company_details")) or "",
                    url=url,
                    requirements=requirements,
                    salary_min=_optional_int(row.get("salary_min")),
                    salary_max=_optional_int(row.get("salary_max")),
                    source=source,
                )
            )
        except (TypeError, ValueError, ValidationError) as exc:
            raise InputLoadError(f"Invalid jobs CSV row {index}: {exc}") from exc
    return jobs


def _normalize_job_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Map common spreadsheet headings to the internal job schema."""

    aliases = {
        "job id": "job_id",
        "id": "job_id",
        "job title": "title",
        "role": "title",
        "company name": "company",
        "industry": "industry_domain",
        "industry domain": "industry_domain",
        "domain": "industry_domain",
        "required skills": "required_skills",
        "skills": "required_skills",
        "years of experience required": "years_experience_required",
        "years experience required": "years_experience_required",
        "experience required": "years_experience_required",
        "company details": "company_details",
        "company information": "company_details",
        "remote status": "remote",
        "salary minimum": "salary_min",
        "minimum salary": "salary_min",
        "salary maximum": "salary_max",
        "maximum salary": "salary_max",
        "url": "url",
        "job url": "url",
        "link": "url",
    }
    supported = {
        "job_id",
        "title",
        "company",
        "industry_domain",
        "location",
        "remote",
        "description",
        "required_skills",
        "years_experience_required",
        "experience_requirement",
        "company_details",
        "url",
        "requirements",
        "salary_min",
        "salary_max",
        "source",
    }
    rename: dict[str, str] = {}
    for column in frame.columns:
        normalized = re.sub(
            r"[^a-z0-9]+", " ", str(column).lstrip("\ufeff").lower()
        ).strip()
        if normalized.startswith("job description"):
            rename[column] = "description"
        elif normalized.startswith("company details"):
            rename[column] = "company_details"
        elif normalized in aliases:
            rename[column] = aliases[normalized]
        else:
            snake_case = normalized.replace(" ", "_")
            if snake_case in supported:
                rename[column] = snake_case
    return frame.rename(columns=rename)


def load_candidate_profile(path: str | Path) -> CandidateProfile:
    """Load a complete legacy profile or an assignment preferences YAML file."""

    resolved = Path(path)
    payload = _load_yaml(resolved, "Preferences")
    candidate_payload = payload.get("candidate", {})
    if not isinstance(candidate_payload, dict):
        raise InputLoadError("Preferences field 'candidate' must contain a mapping")

    raw_preferences = payload.get("preferences")
    if raw_preferences is None:
        preference_keys = set(CandidatePreferences.model_fields) | {
            "companies_to_exclude"
        }
        raw_preferences = {
            key: value for key, value in payload.items() if key in preference_keys
        }
    if not isinstance(raw_preferences, dict):
        raise InputLoadError("Preferences field 'preferences' must contain a mapping")
    preferences_payload = dict(raw_preferences)
    if "years_of_experience" not in preferences_payload:
        candidate_years = candidate_payload.get("years_of_experience")
        if candidate_years is not None:
            preferences_payload["years_of_experience"] = candidate_years

    try:
        preferences = CandidatePreferences.model_validate(preferences_payload)
        master_skills = normalize_string_list(
            _first_present(
                payload.get("master_skills"), candidate_payload.get("master_skills")
            )
        )
        provided_master_evidence = payload.get("master_skill_evidence", [])
        master_skill_evidence = list(provided_master_evidence or [])
        if master_skills and not master_skill_evidence:
            master_skill_evidence = [
                EvidenceItem(
                    evidence_id="master-skills-001",
                    source="master_skills",
                    text=f"Master skills: {', '.join(master_skills)}",
                    tags=master_skills,
                    metadata={"source_path": str(resolved)},
                )
            ]

        persona = payload.get("persona", candidate_payload)
        if not isinstance(persona, dict):
            raise InputLoadError("Candidate persona must contain a mapping")
        return CandidateProfile(
            candidate_id=_optional_str(
                _first_present(
                    payload.get("candidate_id"), candidate_payload.get("candidate_id")
                )
            )
            or "uploaded-candidate",
            name=_optional_str(
                _first_present(payload.get("name"), candidate_payload.get("name"))
            )
            or "",
            email=_optional_str(
                _first_present(payload.get("email"), candidate_payload.get("email"))
            ),
            persona=persona,
            preferences=preferences,
            skills=payload.get("skills", []),
            master_skills=master_skills,
            education=payload.get("education", []),
            experience=payload.get("experience", []),
            resume_content=payload.get("resume_content"),
            resume_projects=payload.get("resume_projects", []),
            resume_evidence=payload.get("resume_evidence", []),
            master_skill_evidence=master_skill_evidence,
            portfolio_evidence=payload.get("portfolio_evidence", []),
        )
    except (TypeError, ValueError, ValidationError) as exc:
        if isinstance(exc, InputLoadError):
            raise
        raise InputLoadError(f"Invalid preferences file {resolved}: {exc}") from exc


def load_portfolio(path: str | Path) -> Portfolio:
    """Load a plain-text upload or a legacy YAML portfolio."""

    resolved = Path(path)
    if resolved.suffix.lower() == ".txt":
        return _load_text_portfolio(resolved)
    payload = _load_yaml(resolved, "Portfolio")
    try:
        return Portfolio.model_validate(payload)
    except ValidationError as exc:
        raise InputLoadError(f"Invalid portfolio file {resolved}: {exc}") from exc


def load_text_path(path: str | Path, label: str) -> str:
    """Validate that a text artifact path exists and return it as a string."""

    resolved = Path(path)
    if not resolved.exists():
        raise InputLoadError(f"{label} not found: {resolved}")
    return str(resolved)


def load_resume_data(path: str | Path) -> ResumeData:
    """Extract structured sections from the repository's LaTeX resume template."""

    resolved = Path(path)
    if not resolved.exists():
        raise InputLoadError(f"Resume not found: {resolved}")
    content = resolved.read_text(encoding="utf-8").strip()
    if not content:
        raise InputLoadError(f"Resume file is empty: {resolved}")

    document_content = content.partition(r"\begin{document}")[2] or content
    document_content = document_content.partition(r"\end{document}")[0]
    plain_text = _latex_to_plain(document_content)
    summary_section = _latex_section(content, "Professional Summary")
    summary = _latex_to_plain(summary_section) or None

    education_section = _latex_section(content, "Education")
    education = [
        " | ".join(_latex_to_plain(argument) for argument in arguments)
        for arguments, _ in _extract_command_arguments(
            education_section, "resumeEntry", 4
        )
    ]

    experience_section = _latex_section(content, "Experience")
    experience_entries = _extract_command_arguments(
        experience_section, "resumeEntry", 4
    )
    experience: list[str] = []
    for entry_index, (arguments, offset) in enumerate(experience_entries):
        next_offset = (
            experience_entries[entry_index + 1][1]
            if entry_index + 1 < len(experience_entries)
            else len(experience_section)
        )
        entry_block = experience_section[offset:next_offset]
        bullets = [
            _latex_to_plain(items[0])
            for items, _ in _extract_command_arguments(entry_block, "resumeItem", 1)
        ]
        parts = [*(_latex_to_plain(argument) for argument in arguments), *bullets]
        experience.append(" | ".join(part for part in parts if part))

    projects_section = _latex_section(content, "Projects")
    project_entries = _extract_command_arguments(projects_section, "resumeEntry", 4)
    projects = [
        _latex_to_plain(arguments[0])
        for arguments, _ in project_entries
        if _latex_to_plain(arguments[0])
    ]

    skills_section = _latex_section(content, "Skills")
    skill_values: list[str] = []
    for arguments, _ in _extract_command_arguments(skills_section, "item", 1):
        plain_item = _latex_to_plain(arguments[0])
        _, separator, values = plain_item.partition(":")
        skill_values.extend(normalize_string_list(values if separator else plain_item))

    evidence_items = [
        EvidenceItem(
            evidence_id="resume-upload-001",
            source="resume",
            text=plain_text,
            tags=["resume"],
            metadata={"source_path": str(resolved)},
        )
    ]
    for section_name, values in (
        ("education", education),
        ("experience", experience),
        ("project", projects),
    ):
        for index, text in enumerate(values, start=1):
            evidence_items.append(
                EvidenceItem(
                    evidence_id=f"resume-{section_name}-{index:03d}",
                    source="resume",
                    text=text,
                    tags=[section_name],
                    metadata={"section": section_name, "source_path": str(resolved)},
                )
            )
    if skill_values:
        evidence_items.append(
            EvidenceItem(
                evidence_id="resume-skills-001",
                source="resume",
                text=f"Resume skills: {', '.join(skill_values)}",
                tags=["skills", *skill_values],
                metadata={"section": "skills", "source_path": str(resolved)},
            )
        )

    return ResumeData(
        plain_text=plain_text,
        professional_summary=summary,
        skills=skill_values,
        education=education,
        experience=experience,
        projects=projects,
        evidence_items=evidence_items,
    )


def load_resume_evidence(path: str | Path) -> list[EvidenceItem]:
    """Convert an uploaded LaTeX resume into searchable candidate evidence."""

    return load_resume_data(path).evidence_items


def _load_text_portfolio(path: Path) -> Portfolio:
    """Parse assignment-style labeled projects or legacy blank-line blocks."""

    if not path.exists():
        raise InputLoadError(f"Portfolio file not found: {path}")
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        raise InputLoadError(f"Portfolio file is empty: {path}")

    project_markers = list(re.finditer(r"(?m)^PROJECT_ID:\s*(.+?)\s*$", content))
    if project_markers:
        return _load_labeled_portfolio(content, project_markers)
    return _load_legacy_text_portfolio(content)


def _load_labeled_portfolio(
    content: str, project_markers: list[re.Match[str]]
) -> Portfolio:
    projects: list[PortfolioProject] = []
    evidence_items: list[EvidenceItem] = []
    for index, marker in enumerate(project_markers, start=1):
        end = (
            project_markers[index].start()
            if index < len(project_markers)
            else len(content)
        )
        block = content[marker.start() : end]
        block = re.split(r"(?m)^={10,}\s*$", block, maxsplit=1)[0].strip()
        project_id = marker.group(1).strip()
        name = _labeled_value(block, "PROJECT_NAME") or f"Portfolio project {index}"
        description = _labeled_value(block, "SUMMARY") or name
        technologies = normalize_string_list(_labeled_value(block, "TECH_STACK"))
        domains = normalize_string_list(_labeled_value(block, "DOMAIN"))
        industries = normalize_string_list(_labeled_value(block, "INDUSTRY"))
        keywords = normalize_string_list(_labeled_value(block, "KEYWORDS"))
        evidence_id = f"portfolio-{project_id}"
        evidence_items.append(
            EvidenceItem(
                evidence_id=evidence_id,
                source="portfolio",
                text=block,
                tags=[*technologies, *domains, *industries, *keywords],
                metadata={"project_id": project_id},
            )
        )
        projects.append(
            PortfolioProject(
                project_id=project_id,
                name=name,
                description=description,
                technologies=technologies,
                domains=domains,
                industries=industries,
                period=_labeled_value(block, "PERIOD"),
                role=_labeled_value(block, "ROLE"),
                organization_alias=_labeled_value(block, "ORGANIZATION_ALIAS"),
                resume_swap_value=_labeled_value(block, "RESUME_SWAP_VALUE"),
                keywords=keywords,
                evidence_ids=[evidence_id],
            )
        )
    return Portfolio(projects=projects, evidence_items=evidence_items)


def _load_legacy_text_portfolio(content: str) -> Portfolio:
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
        technologies = normalize_string_list(technology_line.partition(":")[2])
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


def _parse_remote_status(value: Any, location: str) -> bool | None:
    if not _is_blank(value):
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        normalized = str(value).strip().casefold()
        if normalized in {"true", "1", "yes", "y", "remote", "fully remote"}:
            return True
        if normalized in {
            "false",
            "0",
            "no",
            "n",
            "on-site",
            "onsite",
            "hybrid",
        }:
            return False
        return None

    normalized_location = location.casefold()
    if "remote" in normalized_location:
        return True
    if "on-site" in normalized_location or "onsite" in normalized_location:
        return False
    if "hybrid" in normalized_location:
        return False
    return None


def _optional_int(value: Any) -> int | None:
    if _is_blank(value):
        return None
    return int(float(value))


def _optional_str(value: Any) -> str | None:
    if _is_blank(value):
        return None
    stripped = str(value).strip()
    return stripped or None


def _first_present(*values: Any) -> Any:
    for value in values:
        if not _is_blank(value):
            return value
    return None


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, set, dict)):
        return not value
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(result) if isinstance(result, bool) else False


def _is_http_url(value: str) -> bool:
    try:
        _HTTP_URL_ADAPTER.validate_python(value)
    except ValidationError:
        return False
    return True


def _labeled_value(block: str, label: str) -> str | None:
    match = re.search(rf"(?m)^{re.escape(label)}:\s*(.*?)\s*$", block)
    return match.group(1).strip() if match and match.group(1).strip() else None


def _latex_section(content: str, section_name: str) -> str:
    match = re.search(
        rf"(?s)\\section\{{{re.escape(section_name)}\}}(.*?)"
        rf"(?=\\section\{{|\\end\{{document\}})",
        content,
    )
    return match.group(1) if match else ""


def _extract_command_arguments(
    content: str, command: str, argument_count: int
) -> list[tuple[list[str], int]]:
    results: list[tuple[list[str], int]] = []
    pattern = re.compile(rf"\\{re.escape(command)}\s*")
    for match in pattern.finditer(content):
        cursor = match.end()
        arguments: list[str] = []
        for _ in range(argument_count):
            while cursor < len(content) and content[cursor].isspace():
                cursor += 1
            if cursor >= len(content) or content[cursor] != "{":
                break
            argument, cursor = _balanced_brace_content(content, cursor)
            arguments.append(argument)
        if len(arguments) == argument_count:
            results.append((arguments, match.start()))
    return results


def _balanced_brace_content(content: str, start: int) -> tuple[str, int]:
    depth = 0
    for cursor in range(start, len(content)):
        character = content[cursor]
        if character == "{" and (cursor == 0 or content[cursor - 1] != "\\"):
            depth += 1
        elif character == "}" and (cursor == 0 or content[cursor - 1] != "\\"):
            depth -= 1
            if depth == 0:
                return content[start + 1 : cursor], cursor + 1
    return content[start + 1 :], len(content)


def _latex_to_plain(content: str) -> str:
    plain = re.sub(r"(?m)(?<!\\)%.*$", " ", content)
    plain = re.sub(r"\\href\{[^{}]*\}\{([^{}]*)\}", r"\1", plain)
    plain = re.sub(r"\\(?:begin|end)\{[^{}]*\}", " ", plain)
    plain = re.sub(r"\\[A-Za-z@]+\*?(?:\[[^\]]*\])?", " ", plain)
    plain = plain.replace(r"\%", "%").replace(r"\&", "&")
    plain = plain.replace("$|$", " | ")
    plain = re.sub(r"[{}$]", " ", plain)
    return re.sub(r"\s+", " ", plain).strip()
