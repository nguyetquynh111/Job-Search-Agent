"""Production resume-tailoring behavior."""

from __future__ import annotations

from pathlib import Path

from src.schemas.common import ChangeLogEntry, EvidenceClaim, EvidenceItem, ProjectSwap
from src.schemas.fit_analysis import FitAnalysisOutput
from src.schemas.jobs import Job
from src.schemas.tailoring import TailorResumeInput
from src.tools import tailor_resume as tailoring
from src.tools.job_evidence import build_job_evidence


def _job() -> Job:
    return Job(
        job_id="J900",
        title="Machine Learning Engineer",
        company="Example AI",
        industry_domain="Machine Learning",
        location="Remote",
        description="Build production ML systems.",
        required_skills=["Python", "Kubernetes", "Go"],
    )


def _portfolio_text() -> str:
    return """PROJECT_ID: P02
PROJECT_NAME: AI CRM Platform
PERIOD: 2024-2025
ROLE: Senior AI Engineer
DOMAIN: Predictive Analytics; NLP
TECH_STACK: Python; FastAPI; Docker; Kubernetes; AWS SageMaker
SUMMARY: Engineered an AI-powered CRM platform for sales prioritization.
"""


def _input(source: Path) -> TailorResumeInput:
    evidence = [
        EvidenceItem(
            evidence_id="resume-experience-001",
            source="resume",
            text=(
                "Graduate Research Assistant | Built a visual-QC workflow over "
                "1.7M segmentation/tracking predictions; achieved 90.5% test "
                "accuracy and 0.764 F1 using out-of-fold evaluation and targeted "
                "error analysis with Python and computer vision."
            ),
            tags=["experience", "Python", "Computer Vision"],
        ),
        EvidenceItem(
            evidence_id="resume-experience-002",
            source="resume",
            text=(
                "Senior AI Engineer | Delivered production Python services with "
                "Docker and Kubernetes."
            ),
            tags=["experience", "Python", "Docker", "Kubernetes"],
        ),
        EvidenceItem(
            evidence_id="master-skill-kubernetes",
            source="master_skills",
            text="Skill: Kubernetes",
            tags=["Kubernetes"],
        ),
        EvidenceItem(
            evidence_id="portfolio-P02",
            source="portfolio",
            text=_portfolio_text(),
            tags=["Python", "Kubernetes", "Predictive Analytics"],
            metadata={"project_id": "P02"},
        ),
    ]
    fit = FitAnalysisOutput(
        job_id="J900",
        aligned_skills=[
            EvidenceClaim(
                claim="Python: already demonstrated.",
                evidence_ids=["resume-experience-001", "resume-experience-002"],
            )
        ],
        evidenced_missing_skills=[
            EvidenceClaim(
                claim="Kubernetes: safe to add.",
                evidence_ids=["master-skill-kubernetes"],
            )
        ],
        genuine_gaps=[EvidenceClaim(claim="Go: no supporting evidence.")],
        project_swap=ProjectSwap(
            remove_project="Cardiovascular Flow and Stenosis Analysis",
            add_project="AI CRM Platform",
            rationale="Closer production-ML match.",
            evidence_ids=["portfolio-P02"],
        ),
    )
    return TailorResumeInput(
        job=_job(),
        fit_analysis=fit,
        source_resume_tex_path=str(source),
        candidate_evidence=evidence,
    )


def test_tailoring_changes_only_two_targeted_experience_bullets(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source.tex"
    original = Path("data/resume.tex").read_text(encoding="utf-8")
    source.write_text(original, encoding="utf-8")
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "outputs"))

    def fake_compile(text: str, tex_path: Path, pdf_path: Path):
        tex_path.write_text(text, encoding="utf-8")
        pdf_path.write_bytes(b"%PDF-1.4 test")
        return 1, [], text

    monkeypatch.setattr(tailoring, "_compile_one_page", fake_compile)
    result = tailoring.tailor_resume(_input(source))

    assert result.status == "OK"
    assert result.page_count == 1
    assert source.read_text(encoding="utf-8") == original
    tailored = Path(result.output_tex_path).read_text(encoding="utf-8")
    assert "Kubernetes" in tailored
    assert "Go, " not in tailored
    assert "AI CRM Platform" in tailored
    assert "Cardiovascular Flow and Stenosis Analysis" not in tailored
    assert sum(entry.section == "experience" for entry in result.change_log) == 2
    known_evidence = {
        item.evidence_id
        for item in [*_input(source).candidate_evidence, *build_job_evidence(_job())]
    }
    assert {entry.section for entry in result.change_log} <= {
        "summary",
        "experience",
        "skills",
        "projects",
    }
    assert all(entry.before_text != entry.after_text for entry in result.change_log)
    assert all(entry.reason.strip() for entry in result.change_log)
    assert all(
        entry.evidence_ids and set(entry.evidence_ids) <= known_evidence
        for entry in result.change_log
    )
    for marker in tailoring.EXPERIENCE_TARGETS:
        assert tailoring._command_argument_after_marker(
            original, marker, "resumeItem"
        ) != tailoring._command_argument_after_marker(tailored, marker, "resumeItem")


def test_tailoring_reports_compile_failure_honestly(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source.tex"
    source.write_text(Path("data/resume.tex").read_text(encoding="utf-8"))
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setattr(
        tailoring,
        "_compile_one_page",
        lambda text, tex, pdf: (None, ["pdflatex unavailable"], text),
    )

    result = tailoring.tailor_resume(_input(source))

    assert result.status == "ERROR"
    assert result.page_count == 0
    assert result.output_pdf_path == ""
    assert result.errors == ["pdflatex unavailable"]


def test_project_swap_requires_real_portfolio_evidence(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source.tex"
    source.write_text(Path("data/resume.tex").read_text(encoding="utf-8"))
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "outputs"))
    payload = _input(source)
    payload = payload.model_copy(
        update={"candidate_evidence": payload.candidate_evidence[:-1]}
    )

    result = tailoring.tailor_resume(payload)

    assert result.status == "ERROR"
    assert any("project addition" in error for error in result.errors)
    assert result.validation_failures == result.errors


def test_direct_tailoring_rejects_go_claim_citing_python(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.tex"
    source.write_text(Path("data/resume.tex").read_text(encoding="utf-8"))
    payload = _input(source)
    payload = payload.model_copy(
        update={
            "fit_analysis": payload.fit_analysis.model_copy(
                update={
                    "evidenced_missing_skills": [
                        EvidenceClaim(
                            claim="Go: safe to add",
                            evidence_ids=["resume-experience-001"],
                        )
                    ]
                }
            )
        }
    )

    result = tailoring.tailor_resume(payload)

    assert result.status == "ERROR"
    assert any("Go" in failure for failure in result.validation_failures)
    assert not result.output_pdf_path


def test_unrelated_revision_change_does_not_satisfy_feedback() -> None:
    change = ChangeLogEntry(
        change_id="unrelated",
        section="summary",
        description="Reworded the summary.",
        before_text="Machine learning engineer.",
        after_text="Experienced machine learning engineer.",
        reason="Improve role alignment.",
        evidence_ids=["resume-python", "job-python"],
    )
    evidence = {
        "resume-python": EvidenceItem(
            evidence_id="resume-python",
            source="resume",
            text="Python engineer.",
            tags=["Python"],
        ),
        "job-python": EvidenceItem(
            evidence_id="job-python",
            source="job_posting",
            text="Required skill: Python",
            tags=["job_skill_requirement", "Python"],
        ),
    }

    satisfied, checks = tailoring._check_revision_feedback(
        "Emphasize stakeholder leadership.", [change], evidence
    )

    assert satisfied is False
    assert checks == ["feedback-specific content overlap: not met"]
