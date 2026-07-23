"""Assignment schema and data-loading compatibility tests."""

from __future__ import annotations

import csv
import importlib
from pathlib import Path

import pytest

from src.data_loader import (
    load_candidate_profile,
    load_jobs_csv,
    load_portfolio,
    load_resume_data,
)
from src.memory.models import MemoryFact, MemoryProvenance
from src.schemas.common import CandidatePreferences, CandidateProfile
from src.schemas.jobs import Job, parse_experience_requirement


def _write_job_csv(path: Path, headers: list[str], values: list[object]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerow(values)


def test_complete_assignment_job_row_preserves_every_field(tmp_path: Path) -> None:
    path = tmp_path / "jobs.csv"
    _write_job_csv(
        path,
        [
            "Job Title",
            "Company",
            "Industry/Domain",
            "Location",
            "Required Skills",
            "Years of Experience Required",
            "Job Description (10-20 lines; include responsibilities and qualifications)",
            "Company Details (2-3 lines about the company)",
            "URL",
            "Remote Status",
        ],
        [
            " Machine Learning Engineer ",
            " Acme AI ",
            "Healthcare / Computer Vision",
            "Remote, US",
            "Python; PyTorch; Python",
            "3-5 years",
            "Build and validate medical imaging models.",
            "Acme develops clinical decision-support software.",
            "https://example.com/jobs/123",
            "true",
        ],
    )

    job = load_jobs_csv(path)[0]

    assert job.title == "Machine Learning Engineer"
    assert job.company == "Acme AI"
    assert job.industry_domain == "Healthcare / Computer Vision"
    assert job.location == "Remote, US"
    assert job.required_skills == ["Python", "PyTorch"]
    assert job.requirements == ["Python", "PyTorch"]
    assert job.years_experience_required == 3
    assert job.experience_requirement.minimum_years == 3
    assert job.experience_requirement.maximum_years == 5
    assert job.experience_requirement.raw_text == "3-5 years"
    assert job.description == "Build and validate medical imaging models."
    assert job.company_details.startswith("Acme develops")
    assert job.url == "https://example.com/jobs/123"
    assert job.source == job.url
    assert job.remote is True


def test_legacy_job_row_keeps_original_fields_without_fabricating_url(
    tmp_path: Path,
) -> None:
    path = tmp_path / "legacy.csv"
    _write_job_csv(
        path,
        [
            "job_id",
            "title",
            "company",
            "description",
            "requirements",
            "source",
        ],
        [
            "LEG-1",
            "Data Scientist",
            "Legacy Co",
            "Analyze data.",
            "SQL,Python",
            "Indeed",
        ],
    )

    job = load_jobs_csv(path)[0]

    assert job.job_id == "LEG-1"
    assert job.required_skills == ["SQL", "Python"]
    assert job.requirements == ["SQL", "Python"]
    assert job.source == "Indeed"
    assert job.url == ""
    assert job.remote is None


@pytest.mark.parametrize(
    ("skills_value", "expected"),
    [
        (["Python", " SQL ", "python"], ["Python", "SQL"]),
        ('["Python", "SQL", "Python"]', ["Python", "SQL"]),
        ("Python, SQL, Python", ["Python", "SQL"]),
        ("Python; SQL; Python", ["Python", "SQL"]),
        ("", []),
    ],
)
def test_required_skills_accept_common_representations(
    skills_value: object, expected: list[str]
) -> None:
    job = Job(
        job_id="J1",
        title="Engineer",
        company="Acme",
        description="Build systems.",
        required_skills=skills_value,
    )

    assert job.required_skills == expected
    assert job.requirements == expected


def test_job_loader_parses_json_skill_array(tmp_path: Path) -> None:
    path = tmp_path / "json-skills.csv"
    _write_job_csv(
        path,
        ["title", "company", "description", "required_skills"],
        ["Engineer", "Acme", "Build systems.", '["Python", "SQL", "Python"]'],
    )

    assert load_jobs_csv(path)[0].required_skills == ["Python", "SQL"]


@pytest.mark.parametrize(
    ("raw", "minimum", "maximum", "raw_text"),
    [
        ("3 years", 3.0, 3.0, "3 years"),
        ("3+ years", 3.0, None, "3+ years"),
        ("3-5 years", 3.0, 5.0, "3-5 years"),
        ("Not specified", None, None, "Not specified"),
        ("", None, None, None),
    ],
)
def test_experience_requirement_normalization(
    raw: str,
    minimum: float | None,
    maximum: float | None,
    raw_text: str | None,
) -> None:
    result = parse_experience_requirement(raw)

    assert result.minimum_years == minimum
    assert result.maximum_years == maximum
    assert result.raw_text == raw_text


def test_uncertain_experience_text_is_preserved_without_inventing_bounds() -> None:
    raw = "Master's + 10 years, or bachelor's + 12 years"

    result = parse_experience_requirement(raw)

    assert result.minimum_years is None
    assert result.maximum_years is None
    assert result.raw_text == raw


def test_assignment_preferences_load_with_deduplication_and_legacy_access(
    tmp_path: Path,
) -> None:
    path = tmp_path / "preferences.yaml"
    path.write_text(
        """
candidate:
  years_of_experience: 4.5
preferences:
  preferred_locations: [" Houston, TX ", "Houston, TX", "Remote, US"]
  remote_only: true
  companies_to_exclude: [" Acme, Inc. ", "acme, inc.", "Other Co"]
  target_job_titles: ["AI Engineer", "AI Engineer", "ML Engineer"]
  excluded_keywords: ["clearance"]
master_skills: ["Python", "PyTorch", "python"]
""".strip(),
        encoding="utf-8",
    )

    profile = load_candidate_profile(path)
    preferences = profile.preferences

    assert preferences.preferred_locations == ["Houston, TX", "Remote, US"]
    assert preferences.locations == preferences.preferred_locations
    assert preferences.remote_only is True
    assert preferences.remote is True
    assert preferences.years_of_experience == 4.5
    assert preferences.excluded_companies == ["Acme, Inc.", "Other Co"]
    assert preferences.excluded_keywords == ["clearance"]
    assert preferences.target_job_titles == ["AI Engineer", "ML Engineer"]
    assert preferences.target_titles == preferences.target_job_titles
    assert preferences.excluded_company_comparison_keys == {
        "acme inc",
        "other co",
    }


def test_legacy_preference_names_deserialize() -> None:
    preferences = CandidatePreferences.model_validate(
        {
            "locations": ["Chicago"],
            "remote": True,
            "target_titles": ["Data Scientist"],
            "excluded_keywords": ["senior-only"],
        }
    )

    assert preferences.preferred_locations == ["Chicago"]
    assert preferences.remote_only is True
    assert preferences.target_job_titles == ["Data Scientist"]
    assert preferences.excluded_companies == []


def test_master_skills_remain_distinct_from_resume_skills(tmp_path: Path) -> None:
    preferences_path = tmp_path / "preferences.yaml"
    preferences_path.write_text(
        "master_skills: [Python, GraphQL]\n",
        encoding="utf-8",
    )
    resume_path = tmp_path / "resume.tex"
    resume_path.write_text(
        r"""
\documentclass{article}
\begin{document}
\section{Skills}
\small\item{\textbf{AI/ML:} Python, PyTorch}
\section{Projects}
\resumeEntry{Vision Project}{2025}{PyTorch}{Healthcare}
\end{document}
""".strip(),
        encoding="utf-8",
    )

    profile = load_candidate_profile(preferences_path)
    resume = load_resume_data(resume_path)
    enriched = profile.model_copy(
        update={
            "skills": resume.skills,
            "resume_projects": resume.projects,
            "resume_evidence": resume.evidence_items,
        }
    )

    assert enriched.skills == ["Python", "PyTorch"]
    assert enriched.master_skills == ["Python", "GraphQL"]
    assert enriched.master_skill_evidence[0].source == "master_skills"
    assert enriched.resume_projects == ["Vision Project"]


def test_repository_inputs_expose_complete_profile_layers() -> None:
    profile = load_candidate_profile("data/preferences.yaml")
    portfolio = load_portfolio("data/portfolio.txt")
    resume = load_resume_data("data/resume.tex")

    assert profile.preferences.years_of_experience == 4
    assert profile.master_skills
    assert len(portfolio.projects) == 8
    assert portfolio.projects[0].project_id == "P01"
    assert portfolio.projects[0].technologies
    assert resume.education
    assert resume.experience
    assert resume.skills
    assert resume.projects == [
        "Cardiovascular Flow and Stenosis Analysis",
        "No-Code LLM Chatbot Builder",
    ]


def test_schema_and_memory_provenance_round_trip() -> None:
    job = Job(
        job_id="J1",
        title="ML Engineer",
        company="Acme",
        description="Build models.",
        required_skills=["Python", "Python"],
        years_experience_required="3+ years",
        url="https://example.com/job",
    )
    profile = CandidateProfile(
        candidate_id="candidate-1",
        name="Avery Morgan",
        preferences={"preferred_locations": ["Remote"], "remote_only": True},
        skills=["PyTorch"],
        master_skills=["Python"],
    )
    fact = MemoryFact(
        fact_id="mem-1",
        fact_type="skill",
        canonical_value="GraphQL",
        provenance=MemoryProvenance(
            source="human_review",
            review_round=2,
            original_statement="I know GraphQL.",
            related_job_id="J1",
        ),
    )

    assert Job.model_validate_json(job.model_dump_json()) == job
    assert CandidateProfile.model_validate_json(profile.model_dump_json()) == profile
    restored_fact = MemoryFact.model_validate_json(fact.model_dump_json())
    assert restored_fact.provenance.source == "human_review"
    assert restored_fact.provenance.review_round == 2
    assert restored_fact.provenance.original_statement == "I know GraphQL."


def test_related_schema_and_loading_modules_import() -> None:
    modules = [
        "src.schemas.jobs",
        "src.schemas.common",
        "src.schemas.filtering",
        "src.schemas.scoring",
        "src.schemas.fit_analysis",
        "src.schemas.tailoring",
        "src.schemas.cover_letter",
        "src.data_loader",
        "src.config",
        "src.agent.state",
        "src.memory.models",
        "src.memory.store",
        "src.review.review_service",
        "src.observability.trace_manager",
        "src.ui.graph_resource",
    ]

    for module in modules:
        assert importlib.import_module(module)
