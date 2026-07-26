"""Resume tailoring tool contract tests."""

from __future__ import annotations

from pathlib import Path

from src.data_loader import (
    load_candidate_profile,
    load_jobs_csv,
    load_portfolio,
    load_resume_data,
)
from src.schemas.common import EvidenceClaim, ProjectSwap
from src.schemas.fit_analysis import FitAnalysisOutput
from src.schemas.tailoring import TailorResumeInput
from src.tools.implementations.tailor_resume import tailor_resume


def test_tailor_resume_edits_only_allowed_marked_content(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "outputs"))
    source_resume = Path("data/resume.tex")
    original_tex = source_resume.read_text(encoding="utf-8")

    job = load_jobs_csv("data/jobs.csv")[1]
    profile = load_candidate_profile("data/preferences.yaml")
    resume = load_resume_data(source_resume)
    portfolio = load_portfolio("data/portfolio.txt")
    evidence = [
        *resume.evidence_items,
        *profile.master_skill_evidence,
        *portfolio.evidence_items,
    ]
    fit_analysis = FitAnalysisOutput(
        job_id=job.job_id,
        relevant_experience=[
            EvidenceClaim(
                claim="Nimbus AI Products experience aligns with production GenAI.",
                evidence_ids=["resume-experience-002"],
            )
        ],
        aligned_skills=[
            EvidenceClaim(
                claim="Aligned: Python, Kubernetes, and Git.",
                evidence_ids=["resume-skills-001", "master-skills-001"],
            )
        ],
        evidenced_missing_skills=[
            EvidenceClaim(
                claim="Missing but evidenced elsewhere: LangChain/LangGraph.",
                evidence_ids=["portfolio-P03"],
            )
        ],
        genuine_gaps=[
            EvidenceClaim(claim="Genuine gap: Google ADK.", evidence_ids=[])
        ],
        project_swap=ProjectSwap(
            remove_project="Cardiovascular Flow and Stenosis Analysis",
            add_project="P03",
            rationale="The chatbot builder directly supports GenAI and agent workflows.",
            evidence_ids=["portfolio-P03"],
        ),
    )

    result = tailor_resume(
        TailorResumeInput(
            job=job,
            fit_analysis=fit_analysis,
            source_resume_tex_path=str(source_resume),
            candidate_evidence=evidence,
        )
    )

    tailored_tex = Path(result.output_tex_path).read_text(encoding="utf-8")
    assert source_resume.read_text(encoding="utf-8") == original_tex
    assert result.output_tex_path != str(source_resume)
    assert "Google ADK" not in tailored_tex
    assert "No-Code LLM Chatbot Builder" in tailored_tex
    assert "\\resumeItem{Built large-scale ML evaluation workflows" in tailored_tex
    assert "\\resumeItem{Delivered production AI applications" in tailored_tex
    assert len(
        [entry for entry in result.change_log if entry.section == "experience"]
    ) == 2
    assert any(
        "Before:" in entry.description and "After:" in entry.description
        for entry in result.change_log
    )
    assert any("portfolio-P03" in entry.evidence_ids for entry in result.change_log)


def test_tailor_resume_applies_a_real_project_swap(
    tmp_path: Path, monkeypatch
) -> None:
    """The requested swap-in project must not already be on the resume, so this
    exercises actual block replacement rather than the already-present no-op.
    """

    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "outputs"))
    source_resume = Path("data/resume.tex")
    original_tex = source_resume.read_text(encoding="utf-8")

    job = load_jobs_csv("data/jobs.csv")[1]
    profile = load_candidate_profile("data/preferences.yaml")
    resume = load_resume_data(source_resume)
    portfolio = load_portfolio("data/portfolio.txt")
    evidence = [
        *resume.evidence_items,
        *profile.master_skill_evidence,
        *portfolio.evidence_items,
    ]
    fit_analysis = FitAnalysisOutput(
        job_id=job.job_id,
        project_swap=ProjectSwap(
            remove_project="Cardiovascular Flow and Stenosis Analysis",
            add_project="P02",
            rationale=(
                "The AI CRM Platform's FastAPI and AWS SageMaker stack matches "
                "this role's production requirements better than the "
                "cardiovascular imaging project."
            ),
            evidence_ids=["portfolio-P02"],
        ),
    )

    result = tailor_resume(
        TailorResumeInput(
            job=job,
            fit_analysis=fit_analysis,
            source_resume_tex_path=str(source_resume),
            candidate_evidence=evidence,
        )
    )

    tailored_tex = Path(result.output_tex_path).read_text(encoding="utf-8")
    assert source_resume.read_text(encoding="utf-8") == original_tex
    assert "Cardiovascular Flow and Stenosis Analysis" not in tailored_tex
    assert "AI CRM Platform" in tailored_tex

    swap_entries = [entry for entry in result.change_log if entry.section == "projects"]
    assert len(swap_entries) == 1
    assert "Cardiovascular Flow and Stenosis Analysis" in swap_entries[0].description
    assert "AI CRM Platform" in swap_entries[0].description
    assert "portfolio-P02" in swap_entries[0].evidence_ids
