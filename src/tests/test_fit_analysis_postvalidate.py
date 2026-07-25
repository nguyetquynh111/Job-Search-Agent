"""Post-validation: evidence-ID repair, demotion, disjointness, swap rejection."""

from __future__ import annotations

from src.schemas.common import EvidenceClaim, ProjectSwap
from src.schemas.fit_analysis import FitAnalysisOutput
from src.tests.fit_analysis_helpers import (
    make_evidence,
    make_input,
    make_job,
    make_portfolio_project,
    make_profile,
)
from src.tools.implementations.fit_analysis.postvalidate import post_validate


def _base_input(portfolio=None, current=None, evidence=None):
    job = make_job(required_skills=["Python"])
    profile = make_profile(skills=["Python"], resume_projects=current or [])
    return make_input(
        job,
        profile,
        evidence_items=evidence or [make_evidence("ev-real", "resume", "Resume skills: Python")],
        current_resume_projects=current or [],
        portfolio_projects=portfolio or [],
    )


def test_unknown_evidence_id_is_dropped() -> None:
    inp = _base_input()
    output = FitAnalysisOutput(
        job_id="WRONG",
        aligned_skills=[
            EvidenceClaim(claim="Python: aligned", evidence_ids=["ev-real", "ev-ghost"])
        ],
    )
    result, repairs = post_validate(output, inp)
    assert result.aligned_skills[0].evidence_ids == ["ev-real"]
    assert any("ev-ghost" in r for r in repairs)
    assert result.job_id == "J1"  # forced to match input job


def test_evidenced_missing_without_valid_evidence_is_demoted() -> None:
    inp = _base_input()
    output = FitAnalysisOutput(
        job_id="J1",
        evidenced_missing_skills=[
            EvidenceClaim(claim="Kafka: evidenced", evidence_ids=["ev-ghost"])
        ],
    )
    result, repairs = post_validate(output, inp)
    assert result.evidenced_missing_skills == []
    assert [c.claim.split(":")[0] for c in result.genuine_gaps] == ["Kafka"]
    assert any("demoted" in r.lower() for r in repairs)


def test_buckets_are_made_disjoint_by_priority() -> None:
    inp = _base_input()
    output = FitAnalysisOutput(
        job_id="J1",
        aligned_skills=[EvidenceClaim(claim="Python: aligned", evidence_ids=["ev-real"])],
        evidenced_missing_skills=[
            EvidenceClaim(claim="Python: also here", evidence_ids=["ev-real"])
        ],
    )
    result, repairs = post_validate(output, inp)
    assert len(result.aligned_skills) == 1
    assert result.evidenced_missing_skills == []
    assert any("higher-priority" in r for r in repairs)


def test_hallucinated_project_swap_is_rejected() -> None:
    portfolio = [make_portfolio_project("P1", "Real Project", technologies=["Python"])]
    inp = _base_input(portfolio=portfolio, current=["Old Project"])
    output = FitAnalysisOutput(
        job_id="J1",
        project_swap=ProjectSwap(
            remove_project="Old Project",
            add_project="Imaginary Project",  # not in portfolio
            rationale="made up",
        ),
    )
    result, repairs = post_validate(output, inp)
    assert result.project_swap is None
    assert any("not in portfolio" in r for r in repairs)


def test_confidence_reflects_source_kinds() -> None:
    evidence = [
        make_evidence("resume-skills-001", "resume", "Resume skills: Python"),
        make_evidence("portfolio-P1", "portfolio", "b", tags=["Python"]),
    ]
    inp = _base_input(evidence=evidence)
    output = FitAnalysisOutput(
        job_id="J1",
        aligned_skills=[
            EvidenceClaim(claim="Python: aligned", evidence_ids=["resume-skills-001", "portfolio-P1"])
        ],
    )
    result, _ = post_validate(output, inp)
    # On resume + corroborated by another source -> 0.9 per the documented rule.
    assert result.aligned_skills[0].confidence == 0.9
