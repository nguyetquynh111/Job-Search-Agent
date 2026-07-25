"""Entry point: offline end-to-end and the 'no better swap available' branch."""

from __future__ import annotations

from src.schemas.fit_analysis import FitAnalysisOutput
from src.tests.fit_analysis_helpers import (
    make_evidence,
    make_input,
    make_job,
    make_portfolio_project,
    make_profile,
)
from src.tools.implementations import analyze_fit as entry
from src.tools.implementations.fit_analysis import llm_client


def test_analyze_fit_runs_offline(monkeypatch) -> None:
    monkeypatch.setattr(llm_client, "model_configured", lambda: False)
    job = make_job(job_id="J42", required_skills=["Python", "Rust"])
    profile = make_profile(skills=["Python"])
    items = [make_evidence("resume-skills-001", "resume", "Resume skills: Python", tags=["Python"])]
    result = entry.analyze_fit(make_input(job, profile, items))

    assert isinstance(result, FitAnalysisOutput)
    assert result.job_id == "J42"
    assert [c.claim.split(":")[0] for c in result.aligned_skills] == ["Python"]
    assert [c.claim.split(":")[0] for c in result.genuine_gaps] == ["Rust"]
    # Every cited evidence id must exist in the input.
    known = {i.evidence_id for i in items}
    for claim in result.aligned_skills:
        assert set(claim.evidence_ids) <= known


def test_no_better_swap_available_is_stated(monkeypatch) -> None:
    monkeypatch.setattr(llm_client, "model_configured", lambda: False)
    job = make_job(required_skills=["Python"])
    profile = make_profile(skills=["Python"], resume_projects=["Alpha", "Beta"])
    portfolio = [
        make_portfolio_project("P1", "Alpha", technologies=["Python"]),
        make_portfolio_project("P2", "Beta", technologies=["Python"]),
    ]
    items = [make_evidence("resume-skills-001", "resume", "Resume skills: Python", tags=["Python"])]
    result = entry.analyze_fit(
        make_input(
            job,
            profile,
            items,
            current_resume_projects=["Alpha", "Beta"],
            portfolio_projects=portfolio,
        )
    )
    assert result.project_swap is None
    assert any(
        "no project swap is recommended" in c.claim.lower() for c in result.project_analysis
    )


def test_swapped_out_project_is_never_reported_as_strong(monkeypatch) -> None:
    # 3b: project_analysis and the swap derive from one ranking, so the removed
    # project must be reported as the weak one, not "aligns well".
    from src.tools.implementations.fit_analysis import verdict

    monkeypatch.setattr(llm_client, "model_configured", lambda: False)
    job = make_job(
        required_skills=["PyTorch", "RAG"],
        description="Generative AI role using RAG and LLMs.",
    )
    profile = make_profile(skills=["Python"], resume_projects=["Vision Project", "Chatbot"])
    portfolio = [
        make_portfolio_project("P1", "Vision Project", technologies=["PyTorch"], domains=["Computer Vision"]),
        make_portfolio_project("P2", "Chatbot", technologies=["RAG"], domains=["Generative AI"]),
        make_portfolio_project("P3", "RAG Platform", technologies=["RAG", "PyTorch"], domains=["Generative AI"]),
    ]
    items = [make_evidence("resume-skills-001", "resume", "Resume skills: Python", tags=["Python"])]
    result = entry.analyze_fit(
        make_input(
            job, profile, items,
            current_resume_projects=["Vision Project", "Chatbot"],
            portfolio_projects=portfolio,
        )
    )
    if result.project_swap is not None:
        removed = result.project_swap.remove_project
        for claim in result.project_analysis:
            if claim.claim.startswith(f"Current project '{removed}'"):
                value, _ = verdict.parse(claim.notes)
                assert value == verdict.MISMATCH
                assert "aligns well" not in claim.claim
