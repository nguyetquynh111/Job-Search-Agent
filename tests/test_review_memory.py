"""Consolidated runtime/data/memory/tracing tests."""

from __future__ import annotations

from app.app import ensure_session_defaults, store_graph_result
from app.app import save_uploaded_inputs
from dataclasses import dataclass
from pathlib import Path
from pypdf import PdfWriter
from src.agent import (
    build_agent_graph,
    create_memory_checkpointer,
    invoke_new_run,
)
from src.agent import (
    load_candidate_profile,
    load_jobs_csv,
    load_portfolio,
    load_resume_data,
)
from src.agent import (
    load_resume_evidence,
)
from src.agent import CandidatePreferences, CandidateProfile
from src.agent import Job, parse_experience_requirement
from src.agent import create_initial_state
from src.agent import get_config, validate_runtime_requirements
from src.review.memory import JSONMemoryStore
from src.review.memory import MemoryFact, MemoryProvenance
from src.review.memory import extract_memory_facts, validate_memory_facts
from src.tools.resume_tailoring import resume_tailoring as tailoring_module
from src.tracing.langfuse import (
    DEFAULT_LANGFUSE_HOST,
    STATUS_CONNECTED,
    STATUS_UNAVAILABLE,
)
from src.tracing.langfuse import ROOT_TRACE_NAME, TraceManager
from tests import preflight
from tests.test_pipeline import StateChoosingToolSelectionModel
from typing import Any
import csv
import importlib
import logging
import pytest
import sys
import types

# --- test_config.py ---
"""Application configuration tests."""


def test_runtime_paths_are_derived_from_output_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "runtime"
    monkeypatch.setenv("OUTPUT_DIR", str(output_dir))

    config = get_config()

    assert config.output_dir == output_dir
    assert config.memory_file == output_dir / "memory.json"
    assert config.checkpoint_db == output_dir / "checkpoints.sqlite"


def test_runtime_preflight_requires_llm_pdflatex_and_langfuse(monkeypatch) -> None:
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("DEEPINFRA_API_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.setattr("app.configuration.shutil.which", lambda _executable: None)

    with pytest.raises(RuntimeError) as exc_info:
        validate_runtime_requirements()

    message = str(exc_info.value)
    assert "pdflatex" in message
    assert "LLM_MODEL" in message
    assert "LANGFUSE_PUBLIC_KEY" in message
    assert "LANGFUSE_SECRET_KEY" in message


def test_runtime_preflight_accepts_complete_configuration(monkeypatch) -> None:
    monkeypatch.setenv("LLM_MODEL", "example/model")
    monkeypatch.setenv("DEEPINFRA_API_KEY", "test-key")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "public-test-key")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "secret-test-key")
    monkeypatch.setattr(
        "app.configuration.shutil.which", lambda _executable: "/usr/bin/pdflatex"
    )

    validate_runtime_requirements()


# --- test_preflight.py ---
"""Production preflight behavior."""


def _runtime_preflight_fixtures(root: Path) -> None:
    for relative in preflight.REQUIRED_FIXTURES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture", encoding="utf-8")


def test_preflight_reports_credentials_and_incomplete_langfuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _runtime_preflight_fixtures(tmp_path)
    monkeypatch.setattr(preflight, "_check_latex", lambda errors: None)
    monkeypatch.setattr(
        preflight.importlib.util,
        "find_spec",
        lambda module: object(),
    )
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("DEEPINFRA_API_KEY", raising=False)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "public")
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    errors = preflight.run_preflight(
        repo_root=tmp_path,
        output_dir=tmp_path / "outputs",
    )

    assert any("LLM_MODEL and DEEPINFRA_API_KEY" in error for error in errors)
    assert any("LANGFUSE_SECRET_KEY" in error for error in errors)


def test_preflight_reports_unreadable_or_missing_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _runtime_preflight_fixtures(tmp_path)
    (tmp_path / "data/jobs.csv").unlink()
    monkeypatch.setattr(preflight, "_check_latex", lambda errors: None)
    monkeypatch.setattr(
        preflight.importlib.util,
        "find_spec",
        lambda module: object(),
    )
    monkeypatch.setenv("LLM_MODEL", "provider/model")
    monkeypatch.setenv("DEEPINFRA_API_KEY", "key")
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    errors = preflight.run_preflight(
        repo_root=tmp_path,
        output_dir=tmp_path / "outputs",
    )

    assert any("data/jobs.csv" in error for error in errors)
    assert any("LANGFUSE_PUBLIC_KEY" in error for error in errors)
    assert any("LANGFUSE_SECRET_KEY" in error for error in errors)


# --- test_data_loading_contracts.py ---
"""Assignment schema and data-loading compatibility tests."""


def _runtime_data_write_job_csv(
    path: Path, headers: list[str], values: list[object]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerow(values)


def test_complete_assignment_job_row_preserves_every_field(tmp_path: Path) -> None:
    path = tmp_path / "jobs.csv"
    _runtime_data_write_job_csv(
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
    _runtime_data_write_job_csv(
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
    job = Job.model_validate(
        {
            "job_id": "J1",
            "title": "Engineer",
            "company": "Acme",
            "description": "Build systems.",
            "required_skills": skills_value,
        }
    )

    assert job.required_skills == expected
    assert job.requirements == expected


def test_job_loader_parses_json_skill_array(tmp_path: Path) -> None:
    path = tmp_path / "json-skills.csv"
    _runtime_data_write_job_csv(
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
    job = Job.model_validate(
        {
            "job_id": "J1",
            "title": "ML Engineer",
            "company": "Acme",
            "description": "Build models.",
            "required_skills": ["Python", "Python"],
            "years_experience_required": "3+ years",
            "url": "https://example.com/job",
        }
    )
    profile = CandidateProfile.model_validate(
        {
            "candidate_id": "candidate-1",
            "name": "Avery Morgan",
            "preferences": {"preferred_locations": ["Remote"], "remote_only": True},
            "skills": ["PyTorch"],
            "master_skills": ["Python"],
        }
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
        "src.agent",
        "src.agent",
        "src.tools.filtering_scoring.filtering",
        "src.tools.filtering_scoring.scoring",
        "src.tools.fit_analysis.fit_analysis",
        "src.tools.resume_tailoring.resume_tailoring",
        "src.tools.cover_letter.cover_letter",
        "src.agent",
        "src.agent",
        "src.agent",
        "src.review.memory",
        "src.review.memory",
        "src.review.human_review",
        "src.tracing.langfuse",
        "app.app",
    ]

    for module in modules:
        assert importlib.import_module(module)


# --- test_data_uploaded_inputs.py ---
"""Tests for the four-file upload workflow."""


@dataclass
class FakeUpload:
    """Small UploadedFile stand-in for session helper tests."""

    name: str
    content: bytes

    def getvalue(self) -> bytes:
        return self.content


def test_preferences_only_yaml_builds_internal_profile(tmp_path: Path) -> None:
    preferences_path = tmp_path / "preferences.yaml"
    preferences_path.write_text(
        """
target_titles:
  - Data Scientist
locations:
  - Chicago
remote: true
min_salary: 90000
excluded_keywords:
  - senior-only
""".strip(),
        encoding="utf-8",
    )

    profile = load_candidate_profile(preferences_path)

    assert profile.preferences.target_titles == ["Data Scientist"]
    assert profile.preferences.locations == ["Chicago"]
    assert profile.skills == []


def test_job_csv_accepts_human_readable_headers(tmp_path: Path) -> None:
    jobs_path = tmp_path / "jobs.csv"
    jobs_path.write_text(
        """Job Title,Company,Location,Required Skills,Job Description,URL
Data Scientist,Acme,Remote,"Python; SQL",Build forecasting systems.,https://example.com/job
""",
        encoding="utf-8",
    )

    jobs = load_jobs_csv(jobs_path)

    assert jobs[0].job_id == "J001"
    assert jobs[0].title == "Data Scientist"
    assert jobs[0].requirements == ["Python", "SQL"]
    assert jobs[0].remote is True
    assert jobs[0].source == "https://example.com/job"


def test_text_portfolio_and_tex_resume_become_evidence(tmp_path: Path) -> None:
    portfolio_path = tmp_path / "portfolio.txt"
    portfolio_path.write_text(
        """Forecasting dashboard
Built demand forecasts for weekly planning.
Technologies: Python, SQL

Search assistant
Built cited answers over policy documents.
""",
        encoding="utf-8",
    )
    resume_path = tmp_path / "resume.tex"
    resume_path.write_text(
        r"\documentclass{article}\begin{document}Built Python pipelines.\end{document}",
        encoding="utf-8",
    )

    portfolio = load_portfolio(portfolio_path)
    resume_evidence = load_resume_evidence(resume_path)

    assert [project.name for project in portfolio.projects] == [
        "Forecasting dashboard",
        "Search assistant",
    ]
    assert portfolio.projects[0].technologies == ["Python", "SQL"]
    assert "Built Python pipelines" in resume_evidence[0].text


def test_uploads_use_output_memory_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("app.runtime.tempfile.gettempdir", lambda: str(tmp_path))
    output_dir = tmp_path / "outputs"
    memory_file = output_dir / "memory.json"
    monkeypatch.setenv("OUTPUT_DIR", str(output_dir))
    session: dict = {}
    uploads = {
        "jobs_path": FakeUpload("jobs.csv", b"job_id,title\nJ1,Engineer\n"),
        "preferences_path": FakeUpload("preferences.yaml", b"remote: true\n"),
        "resume_path": FakeUpload("resume.tex", b"\\documentclass{article}"),
        "portfolio_path": FakeUpload("portfolio.txt", b"Project\nDescription"),
    }

    paths = save_uploaded_inputs(session, uploads)

    assert Path(paths["jobs_path"]).read_bytes() == uploads["jobs_path"].content
    assert Path(paths["preferences_path"]).suffix == ".yaml"
    assert Path(paths["resume_path"]).suffix == ".tex"
    assert Path(paths["portfolio_path"]).suffix == ".txt"
    assert Path(paths["memory_file"]) == memory_file
    assert memory_file.read_text(encoding="utf-8") == "[]"
    assert "memory_file" not in uploads


def test_upload_rejects_wrong_file_type(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("app.runtime.tempfile.gettempdir", lambda: str(tmp_path))
    uploads = {
        "jobs_path": FakeUpload("jobs.txt", b"wrong type"),
        "preferences_path": FakeUpload("preferences.yaml", b"remote: true\n"),
        "resume_path": FakeUpload("resume.tex", b"resume"),
        "portfolio_path": FakeUpload("portfolio.txt", b"portfolio"),
    }

    with pytest.raises(ValueError, match="jobs.txt must use one of: .csv"):
        save_uploaded_inputs({}, uploads)


# --- test_memory_store.py ---
"""Memory store tests."""


def test_memory_persists_to_json(tmp_path: Path) -> None:
    """Memory facts are written and reloaded from JSON."""

    path = tmp_path / "memory.json"
    store = JSONMemoryStore(path)
    fact = MemoryFact(
        fact_id="mem-test",
        fact_type="skill",
        canonical_value="GraphQL",
        provenance=MemoryProvenance(
            source="human_review",
            review_round=1,
            original_statement="I have used GraphQL.",
            related_job_id="J002",
        ),
    )
    store.append_many([fact])

    loaded = store.load()
    assert len(loaded) == 1
    assert loaded[0].canonical_value == "GraphQL"


def test_memory_conflict_resolution_keeps_audit_and_active_latest(
    tmp_path: Path,
) -> None:
    path = tmp_path / "memory.json"
    store = JSONMemoryStore(path)
    old = MemoryFact(
        fact_id="mem-old",
        fact_type="experience",
        canonical_value="2 years of experience in data engineering",
        provenance=MemoryProvenance(
            source="human_review",
            review_round=1,
            original_statement="I have 2 years of experience in data engineering.",
            related_job_id="J001",
        ),
    )
    new = MemoryFact(
        fact_id="mem-new",
        fact_type="experience",
        canonical_value="5 years of experience in data engineering",
        provenance=MemoryProvenance(
            source="human_review",
            review_round=1,
            original_statement="I have 5 years of experience in data engineering.",
            related_job_id="J002",
        ),
    )

    store.append_many([old], run_id="run-old")
    loaded = store.append_many([new], run_id="run-new")

    assert [fact.active for fact in loaded] == [False, True]
    conflict = loaded[1].conflicts[0]
    assert conflict.old_fact["canonical_value"] == old.canonical_value
    assert conflict.new_fact["canonical_value"] == new.canonical_value
    assert conflict.affected_job_id == "J002"
    assert conflict.affected_run_id == "run-new"
    assert conflict.active_value == new.canonical_value


def test_memory_extractor_ignores_editing_preferences() -> None:
    """Extractor stores candidate facts, not editing preferences."""

    facts = extract_memory_facts(
        {
            "J001": "Make this shorter and use a friendlier tone.",
            "J002": "Add GraphQL. I have used it in previous projects.",
        },
        review_round=1,
    )

    assert [fact.canonical_value for fact in facts] == ["GraphQL"]


def test_memory_extractor_supports_new_skills_and_candidate_facts() -> None:
    """Self-asserted facts are preserved without a fixed technology whitelist."""

    facts = extract_memory_facts(
        {
            "J001": (
                "I have used Snowflake, dbt, and Airflow in previous projects. "
                "I have 5 years of experience in data engineering."
            )
        },
        review_round=2,
    )

    assert {(fact.fact_type, fact.canonical_value) for fact in facts} == {
        ("skill", "Snowflake"),
        ("skill", "dbt"),
        ("skill", "Airflow"),
        ("experience", "5 years of experience in data engineering"),
    }
    assert all(fact.provenance.review_round == 2 for fact in facts)
    assert all(fact.provenance.related_job_id == "J001" for fact in facts)


def test_memory_ids_do_not_repeat_across_extraction_calls() -> None:
    """A process restart/counter reset cannot create colliding evidence IDs."""

    first = extract_memory_facts({"J001": "I know Rust."}, review_round=1)
    second = extract_memory_facts({"J002": "I know Go."}, review_round=1)

    assert first[0].fact_id != second[0].fact_id
    assert first[0].fact_id.startswith("mem-")
    assert second[0].fact_id.startswith("mem-")


def test_negated_or_pure_editing_skill_request_is_not_stored() -> None:
    comments = {
        "J001": "Add Go to the resume, but I have never used Go and do not know it."
    }
    facts = extract_memory_facts(comments, review_round=1)
    valid, failures = validate_memory_facts(facts, comments)

    assert facts == []
    assert valid == []
    assert failures == []


# --- test_tracing.py ---
"""Langfuse tracing tests."""


class FakeLangfuseClient:
    """In-memory Langfuse v2-compatible client."""

    def __init__(self) -> None:
        self.trace_calls: list[dict[str, Any]] = []
        self.span_calls: list[dict[str, Any]] = []
        self.generation_calls: list[dict[str, Any]] = []
        self.event_calls: list[dict[str, Any]] = []
        self.trace_updates: list[dict[str, Any]] = []
        self.flush_count = 0

    def auth_check(self) -> bool:
        return True

    def trace(self, **kwargs: Any) -> object:
        self.trace_calls.append(kwargs)
        return types.SimpleNamespace(
            get_trace_url=lambda: "https://example.test/trace",
            update=lambda **update: self.trace_updates.append(update),
        )

    def span(self, **kwargs: Any) -> object:
        self.span_calls.append(kwargs)
        return types.SimpleNamespace()

    def generation(self, **kwargs: Any) -> object:
        self.generation_calls.append(kwargs)
        return types.SimpleNamespace()

    def event(self, **kwargs: Any) -> object:
        self.event_calls.append(kwargs)
        return types.SimpleNamespace()

    def flush(self) -> None:
        self.flush_count += 1


class FailingLangfuseClient(FakeLangfuseClient):
    """Client that raises on remote tracing calls."""

    def trace(self, **kwargs: Any) -> object:
        raise RuntimeError("remote unavailable")

    def span(self, **kwargs: Any) -> object:
        raise RuntimeError("remote unavailable")


def test_valid_langfuse_configuration_selects_real_tracer(monkeypatch) -> None:
    """Valid env and auth create an enabled Langfuse tracer."""

    instances: list[Any] = []

    class FakeLangfuse(FakeLangfuseClient):
        def __init__(self, public_key: str, secret_key: str, host: str) -> None:
            super().__init__()
            self.public_key = public_key
            self.secret_key = secret_key
            self.host = host
            instances.append(self)

    module = types.ModuleType("langfuse")
    setattr(module, "Langfuse", FakeLangfuse)
    monkeypatch.setitem(sys.modules, "langfuse", module)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "public-test-value")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "secret-test-value")
    monkeypatch.setenv("LANGFUSE_HOST", DEFAULT_LANGFUSE_HOST)

    tracer = TraceManager()

    assert tracer.enabled is True
    assert tracer.status_message == STATUS_CONNECTED
    assert instances[0].host == DEFAULT_LANGFUSE_HOST


def test_missing_langfuse_credentials_fail_fast() -> None:
    """Default tracing requires connected Langfuse credentials."""

    with pytest.raises(RuntimeError) as exc_info:
        TraceManager()

    message = str(exc_info.value)
    assert "Langfuse credentials are required" in message
    assert "LANGFUSE_PUBLIC_KEY" in message
    assert "LANGFUSE_SECRET_KEY" in message


def test_langfuse_initialization_failure_fails_fast(monkeypatch, caplog) -> None:
    """SDK initialization errors stop startup without exposing details."""

    class FailingLangfuse:
        def __init__(self, public_key: str, secret_key: str, host: str) -> None:
            raise RuntimeError("boom")

    module = types.ModuleType("langfuse")
    setattr(module, "Langfuse", FailingLangfuse)
    monkeypatch.setitem(sys.modules, "langfuse", module)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "public-test-value")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "secret-test-value")
    monkeypatch.setenv("LANGFUSE_HOST", DEFAULT_LANGFUSE_HOST)
    caplog.set_level(logging.WARNING)

    with pytest.raises(RuntimeError, match="connected tracing is required"):
        TraceManager()

    assert "Langfuse initialization failed" in caplog.text


def test_tracing_failure_does_not_terminate_agent_workflow(
    tmp_path: Path, monkeypatch
) -> None:
    """Remote tracing failures do not stop the graph from reaching review."""

    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "outputs"))

    def compile_resume(text: str, tex_path: Path, pdf_path: Path):
        tex_path.write_text(text, encoding="utf-8")
        writer = PdfWriter()
        writer.add_blank_page(width=612, height=792)
        with pdf_path.open("wb") as handle:
            writer.write(handle)
        return 1, [], text

    monkeypatch.setattr(tailoring_module, "_compile_one_page", compile_resume)
    memory_file = tmp_path / "memory.json"
    memory_file.write_text("[]", encoding="utf-8")
    tracer = TraceManager(client=FailingLangfuseClient(), enabled=True)
    app = build_agent_graph(
        checkpointer=create_memory_checkpointer(),
        tracer=tracer,
        tool_selection_model=StateChoosingToolSelectionModel(),
    )
    state = create_initial_state(
        thread_id="thread-tracing-failure",
        run_id="run-tracing-failure",
        memory_file=str(memory_file),
    )

    result = invoke_new_run(app, state)

    assert result["status"] == "WAITING_FOR_REVIEW"
    assert result.get("__interrupt__")
    assert result["langfuse_status"] == STATUS_UNAVAILABLE


def test_same_trace_id_is_propagated_through_nested_stages() -> None:
    """Root, spans, and generations use one trace ID and correct parents."""

    client = FakeLangfuseClient()
    tracer = TraceManager(client=client, enabled=True)

    trace_id = tracer.start_run("run-propagation", "thread-propagation")
    with tracer.span("outer_stage"):
        with tracer.span("inner_stage"):
            tracer.record_generation(
                {"tool_name": "filter_jobs", "decision_summary": "safe"},
                name="test_generation",
                model="test-model",
                messages=[("human", "safe prompt")],
                response="safe response",
                usage={"input_tokens": 2, "output_tokens": 2},
                model_parameters={"temperature": 0},
            )

    assert trace_id == "trace-run-propagation"
    assert len(client.trace_calls) == 1
    assert client.trace_calls[0]["id"] == trace_id
    assert client.trace_calls[0]["name"] == ROOT_TRACE_NAME
    assert client.trace_calls[0]["session_id"] == "thread-propagation"
    assert client.trace_calls[0]["public"] is True
    assert {call["trace_id"] for call in client.span_calls} == {trace_id}
    assert {call["trace_id"] for call in client.generation_calls} == {trace_id}
    assert {event.trace_id for event in tracer.events} == {trace_id}
    outer = next(call for call in client.span_calls if call["name"] == "outer_stage")
    inner = next(call for call in client.span_calls if call["name"] == "inner_stage")
    generation = client.generation_calls[0]
    assert outer["parent_observation_id"] is None
    assert inner["parent_observation_id"] == outer["id"]
    assert generation["parent_observation_id"] == inner["id"]
    assert generation["model"] == "test-model"
    assert generation["input"] == [["human", "safe prompt"]]
    assert generation["output"] == "safe response"
    assert generation["usage_details"] == {
        "input_tokens": 2,
        "output_tokens": 2,
    }
    assert generation["model_parameters"]["temperature"] == 0
    local_generation = next(
        event for event in tracer.events if event.observation_type == "GENERATION"
    )
    assert local_generation.model == "test-model"
    assert local_generation.model_parameters == {"temperature": 0}
    assert local_generation.usage == {
        "input_tokens": 2,
        "output_tokens": 2,
    }


def test_resume_after_interrupt_can_remain_nested_under_human_review() -> None:
    """An ended review parent can link work resumed in a later graph invocation."""

    client = FakeLangfuseClient()
    tracer = TraceManager(client=client, enabled=True)
    tracer.start_run("run-resume-parent", "thread-resume-parent")
    with tracer.span("human_review") as review_id:
        with tracer.span("human_review_pause"):
            pass

    with tracer.span(
        "memory_write",
        parent_observation_id=review_id,
        input={"review_comments": {"J1": "I know GraphQL."}},
    ):
        pass
    with tracer.span(
        "tailor_resume_job_1",
        parent_observation_id=review_id,
    ) as tool_id:
        with tracer.span("resume_tailoring.compile_pdf"):
            pass

    memory = next(call for call in client.span_calls if call["name"] == "memory_write")
    tool = next(
        call for call in client.span_calls if call["name"] == "tailor_resume_job_1"
    )
    compile_span = next(
        call
        for call in client.span_calls
        if call["name"] == "resume_tailoring.compile_pdf"
    )
    assert memory["parent_observation_id"] == review_id
    assert tool["parent_observation_id"] == review_id
    assert compile_span["parent_observation_id"] == tool_id


def test_span_errors_are_captured_without_losing_parentage() -> None:
    """Failed work records both an ERROR span and a nested error event."""

    client = FakeLangfuseClient()
    tracer = TraceManager(client=client, enabled=True)
    tracer.start_run("run-error", "thread-error")

    with pytest.raises(ValueError, match="candidate@example.com"):
        with tracer.span("failing_stage", input={"operation": "validate"}):
            raise ValueError("candidate@example.com could not be validated")

    failed_span = client.span_calls[0]
    error_event = client.event_calls[0]
    assert failed_span["level"] == "ERROR"
    assert failed_span["status_message"] == "ValueError"
    assert "[REDACTED_EMAIL]" in failed_span["output"]["error"]
    assert error_event["parent_observation_id"] == failed_span["id"]
    assert error_event["metadata"]["error_type"] == "ValueError"


def test_disabled_tracing_is_a_noop_for_execution() -> None:
    """The same tracing API remains safe with no client or credentials."""

    tracer = TraceManager(client=None, enabled=False)

    trace_id = tracer.start_run("run-disabled", "thread-disabled")
    with tracer.span("work", input={"value": 1}) as span_id:
        tracer.record_generation(
            name="offline_generation",
            model="offline",
            messages=[("human", "hello")],
            response="world",
        )
        tracer.update_span(span_id, output={"value": 2})
    tracer.update_run(output={"status": "complete"})
    tracer.flush()

    assert trace_id == "trace-run-disabled"
    assert tracer.enabled is False
    assert [event.name for event in tracer.events] == [
        ROOT_TRACE_NAME,
        "offline_generation",
        "work",
    ]


def test_trace_payloads_redact_secrets_and_registered_personal_data() -> None:
    """Remote payloads mask direct identifiers without hiding token counts."""

    client = FakeLangfuseClient()
    tracer = TraceManager(client=client, enabled=True)
    tracer.start_run("run-private", "thread-private")
    tracer.register_personal_data(
        "Avery Morgan",
        "Houston, TX",
        "github.com/avery",
    )
    tracer.record_generation(
        name="private_generation",
        model="test-model",
        messages=[
            (
                "human",
                "Avery Morgan in Houston, TX uses avery@example.com, "
                "713-555-0148, and github.com/avery.",
            )
        ],
        response="Contact Avery Morgan",
        usage={"input_tokens": 12, "output_tokens": 3},
        metadata={"api_key": "sk-lf-secret-value"},
    )

    generation = client.generation_calls[0]
    serialized = repr(generation)
    assert "Avery Morgan" not in serialized
    assert "Houston, TX" not in serialized
    assert "avery@example.com" not in serialized
    assert "713-555-0148" not in serialized
    assert "github.com/avery" not in serialized
    assert "sk-lf-secret-value" not in serialized
    assert generation["usage_details"] == {
        "input_tokens": 12,
        "output_tokens": 3,
    }


def test_streamlit_rerun_state_does_not_duplicate_root_trace() -> None:
    """A rerun that only rehydrates session state does not start another trace."""

    client = FakeLangfuseClient()
    tracer = TraceManager(client=client, enabled=True)
    trace_id = tracer.start_run("run-rerun", "thread-rerun")
    session: dict[str, Any] = {}
    ensure_session_defaults(session)
    store_graph_result(
        session,
        {
            "run_id": "run-rerun",
            "thread_id": "thread-rerun",
            "trace_id": trace_id,
            "status": "WAITING_FOR_REVIEW",
        },
    )

    ensure_session_defaults(session)
    tracer.start_run("run-rerun", "thread-rerun")

    assert session["current_run_id"] == "run-rerun"
    assert session["current_thread_id"] == "thread-rerun"
    assert len(client.trace_calls) == 1


def test_new_process_continues_checkpointed_remote_root_trace() -> None:
    client = FakeLangfuseClient()
    first = TraceManager(client=client, enabled=True)
    trace_id = first.start_run("run-restart", "thread-restart")
    with first.span("human_review") as review_id:
        pass

    resumed = TraceManager(client=client, enabled=True)
    resumed.continue_run(
        run_id="run-restart",
        session_id="thread-restart",
        trace_id=trace_id,
        trace_url="https://example.test/trace",
    )
    with resumed.span(
        "memory_write",
        parent_observation_id=review_id,
        input={"provenance": "human_review"},
    ):
        pass

    assert {call["id"] for call in client.trace_calls} == {trace_id}
    resumed_span = client.span_calls[-1]
    assert resumed_span["trace_id"] == trace_id
    assert resumed_span["parent_observation_id"] == review_id
    assert resumed.trace_url == "https://example.test/trace"


def test_sensitive_credentials_do_not_appear_in_logs_or_status(
    monkeypatch, caplog
) -> None:
    """Secrets are never rendered in logs or status messages."""

    public_key = "public-test-value"
    secret_key = "secret-test-value"

    class FailingLangfuse:
        def __init__(self, public_key: str, secret_key: str, host: str) -> None:
            raise RuntimeError(f"bad credentials: {secret_key}")

    module = types.ModuleType("langfuse")
    setattr(module, "Langfuse", FailingLangfuse)
    monkeypatch.setitem(sys.modules, "langfuse", module)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", public_key)
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", secret_key)
    monkeypatch.setenv("LANGFUSE_HOST", DEFAULT_LANGFUSE_HOST)
    caplog.set_level(logging.WARNING)

    with pytest.raises(RuntimeError) as exc_info:
        TraceManager()

    assert public_key not in caplog.text
    assert secret_key not in caplog.text
    assert public_key not in str(exc_info.value)
    assert secret_key not in str(exc_info.value)
