"""Production cover-letter behavior."""

from __future__ import annotations

from pathlib import Path

from src.schemas.common import EvidenceItem
from src.schemas.cover_letter import GenerateCoverLetterInput
from src.schemas.jobs import Job
from src.tools import generate_cover_letter as cover


def _input(resume_path: str) -> GenerateCoverLetterInput:
    return GenerateCoverLetterInput(
        job=Job(
            job_id="J901",
            title="Machine Learning Engineer",
            company="Example AI",
            industry_domain="Machine Learning",
            location="Remote",
            description="Build ML systems.",
            required_skills=["Python"],
            company_details="Example AI builds production machine-learning systems.",
        ),
        approved_resume_path=resume_path,
        candidate_evidence=[
            EvidenceItem(
                evidence_id="resume-experience-001",
                source="resume",
                text="Built production Python services with measurable reliability gains.",
                tags=["experience", "Python"],
            )
        ],
    )


def test_cover_letter_requires_an_actual_approved_resume(tmp_path: Path) -> None:
    result = cover.generate_cover_letter(_input(str(tmp_path / "missing.pdf")))

    assert result.page_count == 0
    assert result.output_tex_path == ""
    assert result.output_pdf_path == ""
    assert result.errors


def test_cover_letter_uses_evidence_and_reports_real_paths(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setattr(
        cover,
        "_read_approved_resume",
        lambda path: (
            "Avery Morgan\nHouston, TX | avery@example.com | github.com/avery",
            [],
        ),
    )

    def fake_compile(letter, job, tex_path: Path, pdf_path: Path):
        tex_path.write_text(cover._render_tex(letter, job, 0), encoding="utf-8")
        pdf_path.write_bytes(b"%PDF-1.4 test")
        return 1, []

    monkeypatch.setattr(cover, "_compile_one_page", fake_compile)
    result = cover.generate_cover_letter(_input("approved.pdf"))

    assert result.page_count == 1
    assert Path(result.output_tex_path).is_file()
    assert Path(result.output_pdf_path).is_file()
    assert set(result.evidence_used) == {
        "job-J901-title",
        "job-J901-description",
        "job-J901-company-details",
        "job-J901-skill-001",
        "resume-experience-001",
    }
    assert not result.errors


def test_negated_skill_evidence_is_not_used_in_cover_letter() -> None:
    negated = EvidenceItem(
        evidence_id="resume-negated-go",
        source="resume",
        text="I have never used Go.",
        tags=["experience", "Go"],
    )

    skills, evidence_ids = cover._match_required_skills(
        ["Go"], {negated.evidence_id: negated}
    )

    assert skills == []
    assert evidence_ids == set()
