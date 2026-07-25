"""Category->instance skill resolution and the swap minimum-improvement threshold."""

from __future__ import annotations

from src.tests.fit_analysis_helpers import (
    make_evidence,
    make_input,
    make_job,
    make_portfolio_project,
    make_profile,
)
from src.tools.implementations.fit_analysis.evidence_index import build_evidence_index
from src.tools.implementations.fit_analysis.prepass import run_prepass


def _prepass(job, profile, evidence, portfolio=None, current=None):
    inp = make_input(
        job, profile, evidence,
        current_resume_projects=current or [], portfolio_projects=portfolio or [],
    )
    return run_prepass(inp, build_evidence_index(inp.evidence_items))


def test_category_evidenced_by_portfolio_member_is_missing_not_gap() -> None:
    job = make_job(required_skills=["agents"])
    profile = make_profile(skills=["Python"])
    evidence = [
        make_evidence("resume-skills-001", "resume", "Resume skills: Python", tags=["Python"]),
        make_evidence("portfolio-P1", "portfolio", "chatbot", tags=["AutoGen"]),
    ]
    pp = _prepass(job, profile, evidence)
    assert [c.skill for c in pp.evidenced_missing] == ["agents"]
    assert pp.evidenced_missing[0].members == ["AutoGen"]  # concrete tool named
    assert pp.evidenced_missing[0].evidence_ids == ["portfolio-P1"]
    assert pp.genuine_gaps == []


def test_category_member_on_resume_is_aligned() -> None:
    job = make_job(required_skills=["APIs"])
    profile = make_profile(skills=["FastAPI"])
    evidence = [make_evidence("resume-skills-001", "resume", "Resume skills: FastAPI", tags=["FastAPI"])]
    pp = _prepass(job, profile, evidence)
    assert [c.skill for c in pp.aligned] == ["APIs"]
    assert "FastAPI" in pp.aligned[0].members


def test_category_without_evidenced_member_stays_gap() -> None:
    job = make_job(required_skills=["vector databases"])
    profile = make_profile(skills=["Python"])
    evidence = [make_evidence("resume-skills-001", "resume", "Resume skills: Python", tags=["Python"])]
    pp = _prepass(job, profile, evidence)
    assert [c.skill for c in pp.genuine_gaps] == ["vector databases"]
    assert pp.evidenced_missing == []


def test_slash_combined_label_resolves_via_parts() -> None:
    job = make_job(required_skills=["Docker/Kubernetes"])
    profile = make_profile(skills=["Docker"])
    evidence = [make_evidence("resume-skills-001", "resume", "Resume skills: Docker", tags=["Docker"])]
    pp = _prepass(job, profile, evidence)
    assert [c.skill for c in pp.aligned] == ["Docker/Kubernetes"]


def _swap_portfolio(external_tech, external_domains=None):
    return [
        make_portfolio_project("P1", "Current Strong", technologies=["PyTorch"]),
        make_portfolio_project("P2", "Current Weak", technologies=["Python"]),
        make_portfolio_project("P3", "External", technologies=external_tech, domains=external_domains or []),
    ]


def test_weak_swap_below_margin_is_suppressed() -> None:
    # Weakest incumbent (Python, 0.5) vs external (PyTorch, 2.0): margin 1.5 < 2.0 -> no swap.
    job = make_job(required_skills=["PyTorch", "Python"])
    profile = make_profile(skills=["Python"], resume_projects=["Current Strong", "Current Weak"])
    evidence = [make_evidence("resume-skills-001", "resume", "Resume skills: Python", tags=["Python"])]
    pp = _prepass(
        job, profile, evidence,
        portfolio=_swap_portfolio(["PyTorch"]),
        current=["Current Strong", "Current Weak"],
    )
    assert pp.swap.swap is None
    from src.tools.implementations.fit_analysis.prepass import build_project_section
    claims, swap = build_project_section(pp.swap)
    assert swap is None
    assert any("no project swap is recommended" in c.claim.lower() for c in claims)


def test_meaningful_swap_above_margin_is_recommended() -> None:
    # External adds a distinctive skill AND a domain (score ~4) vs weak incumbent (0.5).
    job = make_job(required_skills=["PyTorch", "Python"], description="Computer vision role.")
    profile = make_profile(skills=["Python"], resume_projects=["Current Strong", "Current Weak"])
    evidence = [make_evidence("resume-skills-001", "resume", "Resume skills: Python", tags=["Python"])]
    pp = _prepass(
        job, profile, evidence,
        portfolio=_swap_portfolio(["PyTorch"], external_domains=["Computer Vision"]),
        current=["Current Strong", "Current Weak"],
    )
    assert pp.swap.swap is not None
    assert pp.swap.swap.remove_project == "Current Weak"
    assert pp.swap.swap.add_project == "External"
