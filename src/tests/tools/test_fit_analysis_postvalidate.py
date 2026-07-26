"""Post-validation: evidence-ID repair, demotion, disjointness, swap rejection."""

from __future__ import annotations

from src.schemas.common import EvidenceClaim, ProjectSwap
from src.schemas.fit_analysis import FitAnalysisOutput
from src.tests.tools.fit_analysis_helpers import (
    make_evidence,
    make_input,
    make_job,
    make_portfolio_project,
    make_profile,
)
from src.tools.fit_analysis.postvalidate import post_validate


def _base_input(portfolio=None, current=None, evidence=None, required_skills=None):
    job = make_job(required_skills=required_skills or ["Python"])
    profile = make_profile(skills=["Python"], resume_projects=current or [])
    return make_input(
        job,
        profile,
        evidence_items=evidence
        or [make_evidence("ev-real", "resume", "Resume skills: Python")],
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
    assert result.aligned_skills[0].evidence_ids == [
        "job-J1-skill-001",
        "ev-real",
    ]
    assert any("ev-ghost" in r for r in repairs)
    assert result.job_id == "J1"  # Match the input job.


def test_evidenced_missing_without_valid_evidence_is_demoted() -> None:
    inp = _base_input(required_skills=["Kafka"])
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


def test_aligned_without_valid_evidence_is_demoted_to_gap() -> None:
    # A passing claim needs a real citation.
    inp = _base_input()
    output = FitAnalysisOutput(
        job_id="J1",
        aligned_skills=[
            EvidenceClaim(claim="Python: aligned", evidence_ids=["ev-ghost"])
        ],
    )
    result, repairs = post_validate(output, inp)
    assert result.aligned_skills == []
    assert [c.claim.split(":")[0] for c in result.genuine_gaps] == ["Python"]
    assert any("no valid evidence" in r for r in repairs)


def test_aligned_with_only_non_resume_evidence_becomes_evidenced_missing() -> None:
    # Portfolio-only proof belongs in the off-resume bucket.
    evidence = [
        make_evidence("resume-skills-001", "resume", "Resume skills: Python"),
        make_evidence(
            "portfolio-P1", "portfolio", "Kafka pipeline work", tags=["Kafka"]
        ),
    ]
    inp = _base_input(evidence=evidence, required_skills=["Kafka"])
    output = FitAnalysisOutput(
        job_id="J1",
        aligned_skills=[
            EvidenceClaim(claim="Kafka: aligned", evidence_ids=["portfolio-P1"])
        ],
    )
    result, repairs = post_validate(output, inp)
    assert result.aligned_skills == []
    assert [c.claim.split(":")[0] for c in result.evidenced_missing_skills] == ["Kafka"]
    assert "not yet on your resume" in result.evidenced_missing_skills[0].claim
    assert any("no resume evidence" in r for r in repairs)


def test_buckets_are_made_disjoint_by_priority() -> None:
    inp = _base_input()
    output = FitAnalysisOutput(
        job_id="J1",
        aligned_skills=[
            EvidenceClaim(claim="Python: aligned", evidence_ids=["ev-real"])
        ],
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
            add_project="Imaginary Project",  # Not in the portfolio.
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
            EvidenceClaim(
                claim="Python: aligned",
                evidence_ids=["resume-skills-001", "portfolio-P1"],
            )
        ],
    )
    result, _ = post_validate(output, inp)
    # Resume evidence plus another source earns 0.9 confidence.
    assert result.aligned_skills[0].confidence == 0.9


def test_go_claim_cannot_use_python_evidence() -> None:
    inp = _base_input(
        required_skills=["Go"],
        evidence=[
            make_evidence(
                "resume-python",
                "resume",
                "Built production services in Python.",
                tags=["experience", "Python"],
            )
        ],
    )
    output = FitAnalysisOutput(
        job_id="J1",
        aligned_skills=[
            EvidenceClaim(claim="Go: aligned", evidence_ids=["resume-python"])
        ],
    )

    result, failures = post_validate(output, inp)

    assert result.aligned_skills == []
    assert [claim.claim.split(":")[0] for claim in result.genuine_gaps] == ["Go"]
    assert any("semantically unrelated evidence" in item for item in failures)
    assert result.validation_failures == failures


def test_harmless_alias_is_accepted_but_unrelated_keyword_is_not() -> None:
    inp = _base_input(
        required_skills=["Kubernetes"],
        evidence=[
            make_evidence(
                "master-k8s",
                "master_skills",
                "Skill: k8s",
                tags=["k8s"],
            )
        ],
    )
    output = FitAnalysisOutput(
        job_id="J1",
        evidenced_missing_skills=[
            EvidenceClaim(
                claim="Kubernetes: evidenced",
                evidence_ids=["master-k8s"],
            ),
            EvidenceClaim(
                claim="Go: keyword",
                evidence_ids=["master-k8s"],
            ),
        ],
    )

    result, failures = post_validate(output, inp)

    assert [claim.claim.split(":")[0] for claim in result.evidenced_missing_skills] == [
        "Kubernetes"
    ]
    assert all(not claim.claim.startswith("Go:") for claim in result.genuine_gaps)
    assert any("unsupported keyword/skill 'Go'" in item for item in failures)
