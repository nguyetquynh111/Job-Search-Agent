"""Production resume-tailoring behavior."""

from __future__ import annotations

from pathlib import Path

import pytest
from pypdf import PdfWriter

from src.observability.trace_manager import TraceManager
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


def test_skill_elsewhere_cannot_be_attached_to_unrelated_experience_bullet(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.tex"
    source.write_text(Path("data/resume.tex").read_text(encoding="utf-8"))
    payload = _input(source)
    elsewhere_go = EvidenceItem(
        evidence_id="master-go",
        source="master_skills",
        text="Skill: Go",
        tags=["Go"],
    )
    payload = payload.model_copy(
        update={
            "candidate_evidence": [*payload.candidate_evidence, elsewhere_go],
            "fit_analysis": payload.fit_analysis.model_copy(
                update={
                    "evidenced_missing_skills": [
                        EvidenceClaim(
                            claim="Go: evidenced elsewhere",
                            evidence_ids=["master-go"],
                        )
                    ]
                }
            ),
        }
    )
    evidence = {item.evidence_id: item for item in payload.candidate_evidence}

    rewritten, cited = tailoring._rewrite_experience_bullet(
        "Built a visual quality-control workflow in Python.",
        1,
        payload,
        evidence,
    )

    assert "Go" not in rewritten
    assert "master-go" not in cited


def test_after_text_with_new_unsupported_skill_is_rejected() -> None:
    job = _job()
    bullet = EvidenceItem(
        evidence_id="resume-python",
        source="resume",
        text="Built production services in Python.",
        tags=["experience", "Python"],
    )
    elsewhere_go = EvidenceItem(
        evidence_id="master-go",
        source="master_skills",
        text="Skill: Go",
        tags=["Go"],
    )
    payload = TailorResumeInput(
        job=job,
        fit_analysis=FitAnalysisOutput(
            job_id=job.job_id,
            evidenced_missing_skills=[
                EvidenceClaim(
                    claim="Go: evidenced elsewhere",
                    evidence_ids=["master-go"],
                )
            ],
        ),
        source_resume_tex_path="data/resume.tex",
        candidate_evidence=[bullet, elsewhere_go],
        job_evidence=build_job_evidence(job),
    )
    change = ChangeLogEntry(
        change_id="unsupported-go",
        section="experience",
        description="Rewrote experience bullet.",
        before_text="Built production services in Python.",
        after_text="Built production services in Python and Go.",
        reason="Target a posted requirement.",
        evidence_ids=[
            "resume-python",
            "master-go",
            "job-J900-skill-003",
        ],
    )
    evidence = {
        item.evidence_id: item
        for item in [*payload.candidate_evidence, *payload.job_evidence]
    }

    with pytest.raises(tailoring.TailoringError, match="unsupported wording"):
        tailoring._validate_change_log([change], evidence, payload)


def test_supported_skill_with_correct_bullet_evidence_remains_allowed() -> None:
    source = Path("data/resume.tex")
    payload = _input(source)
    before = "Delivered production Python services with Docker and Kubernetes."
    change = ChangeLogEntry(
        change_id="supported-kubernetes",
        section="experience",
        description="Rewrote experience bullet.",
        before_text=before,
        after_text=f"Applied Kubernetes through work that {before.lower()}",
        reason="Emphasize a supported posted requirement.",
        evidence_ids=[
            "resume-experience-002",
            "job-J900-skill-002",
        ],
    )
    evidence = {
        item.evidence_id: item
        for item in [*payload.candidate_evidence, *build_job_evidence(payload.job)]
    }

    tailoring._validate_change_log([change], evidence, payload)


def test_unsupported_project_name_in_experience_bullet_is_rejected() -> None:
    source = Path("data/resume.tex")
    payload = _input(source)
    change = ChangeLogEntry(
        change_id="unsupported-project",
        section="experience",
        description="Rewrote experience bullet.",
        before_text=(
            "Graduate Research Assistant | Built a visual-QC workflow over "
            "1.7M segmentation/tracking predictions; achieved 90.5% test "
            "accuracy and 0.764 F1 using out-of-fold evaluation and targeted "
            "error analysis with Python and computer vision."
        ),
        after_text=(
            "Built the AI CRM Platform while delivering a visual-QC workflow "
            "with Python."
        ),
        reason="Target the role.",
        evidence_ids=[
            "resume-experience-001",
            "portfolio-P02",
            "job-J900-description",
        ],
    )
    evidence = {
        item.evidence_id: item
        for item in [*payload.candidate_evidence, *build_job_evidence(payload.job)]
    }

    with pytest.raises(tailoring.TailoringError, match="unsupported wording"):
        tailoring._validate_change_log([change], evidence, payload)


def test_compilation_trace_records_command_result_and_page_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tex_path = tmp_path / "resume.tex"
    pdf_path = tmp_path / "resume.pdf"

    def successful_compile(path: Path) -> list[str]:
        writer = PdfWriter()
        writer.add_blank_page(width=612, height=792)
        with pdf_path.open("wb") as handle:
            writer.write(handle)
        return []

    monkeypatch.setattr(tailoring, "_run_pdflatex", successful_compile)
    tracer = TraceManager(enabled=False)
    tracer.start_run("run-compile-metadata", "thread-compile-metadata")

    pages, errors, _ = tailoring._compile_one_page(
        "\\documentclass{article}\\begin{document}ok\\end{document}",
        tex_path,
        pdf_path,
        tracer=tracer,
        trace_metadata={"job_id": "J900", "resume_id": "resume-J900"},
    )

    assert pages == 1
    assert errors == []
    compilation = next(
        event
        for event in tracer.events
        if event.name == "resume_tailoring.compile_pdf"
    )
    page_check = next(
        event
        for event in tracer.events
        if event.name == "resume_tailoring.validate_page_count"
    )
    assert compilation.input["command"][0] == "pdflatex"
    assert compilation.output["result"] == "success"
    assert compilation.metadata["job_id"] == "J900"
    assert compilation.metadata["resume_id"] == "resume-J900"
    assert page_check.output["page_count"] == 1
    assert page_check.output["exactly_one_page"] is True
