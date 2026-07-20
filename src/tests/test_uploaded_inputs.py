"""Tests for the four-file upload workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from src.data_loader import (
    load_candidate_profile,
    load_jobs_csv,
    load_portfolio,
    load_resume_evidence,
)
from src.ui.session import save_uploaded_inputs


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
    monkeypatch.setattr("src.ui.session.tempfile.gettempdir", lambda: str(tmp_path))
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
    monkeypatch.setattr("src.ui.session.tempfile.gettempdir", lambda: str(tmp_path))
    uploads = {
        "jobs_path": FakeUpload("jobs.txt", b"wrong type"),
        "preferences_path": FakeUpload("preferences.yaml", b"remote: true\n"),
        "resume_path": FakeUpload("resume.tex", b"resume"),
        "portfolio_path": FakeUpload("portfolio.txt", b"portfolio"),
    }

    with pytest.raises(ValueError, match="jobs.txt must use one of: .csv"):
        save_uploaded_inputs({}, uploads)
