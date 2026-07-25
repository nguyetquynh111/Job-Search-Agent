"""The model owns the judgments: its proposal survives, guards catch what it gets wrong.

These assert through ``analyze_fit`` rather than ``reasoning.analyze`` in isolation,
because the whole point is what survives the merge/post-validation pipeline.
"""

from __future__ import annotations

import json

from src.tests.fit_analysis_helpers import (
    make_evidence,
    make_input,
    make_job,
    make_portfolio_project,
    make_profile,
)
from src.tools.implementations import analyze_fit as entry
from src.tools.implementations.fit_analysis import verdict

_EMPTY = {
    "relevant_experience": [],
    "seniority": [],
    "education": [],
    "aligned_skills": [],
    "evidenced_missing_skills": [],
    "genuine_gaps": [],
    "project_analysis": [],
    "project_swap": None,
}


def _stub(**overrides):
    """Return a complete_fn serving one LLM proposal, ignoring rationale sub-calls."""

    payload = {"job_id": "J1", **_EMPTY, **overrides}

    def complete(system: str, user: str) -> str:
        if "ONE sentence" in system or "rewrite" in system.lower():
            return "unusable prose"
        return json.dumps(payload)

    return complete


def _claim(text: str, ids: list[str], note: str) -> dict:
    return {"claim": text, "evidence_ids": ids, "confidence": 0.8, "notes": note}


def _skill_names(claims) -> set[str]:
    return {c.claim.split(":")[0] for c in claims}


def _swap_fixture(strong_domains: list[str] | None = None):
    """Two resume projects plus two equally strong external candidates.

    ``strong_domains`` lifts the strong incumbent's score so that replacing *it*
    falls below the swap margin, which is what the rejection path needs.
    """

    job = make_job(required_skills=["PyTorch", "Python"], description="Computer vision role.")
    profile = make_profile(skills=["Python"], resume_projects=["Current Strong", "Current Weak"])
    portfolio = [
        make_portfolio_project(
            "P1", "Current Strong", technologies=["PyTorch"], domains=strong_domains or []
        ),
        make_portfolio_project("P2", "Current Weak", technologies=["Python"]),
        make_portfolio_project("P3", "External One", technologies=["PyTorch"], domains=["Computer Vision"]),
        make_portfolio_project("P4", "External Two", technologies=["PyTorch"], domains=["Computer Vision"]),
    ]
    items = [make_evidence("resume-skills-001", "resume", "Resume skills: Python", tags=["Python"])]
    return make_input(
        job, profile, items,
        current_resume_projects=["Current Strong", "Current Weak"],
        portfolio_projects=portfolio,
    )


def test_model_bucket_assignment_survives() -> None:
    # The deterministic pass would call GraphQL a genuine gap (no evidence indexed
    # for it); the model says it is evidenced by a real portfolio ID, and wins.
    job = make_job(required_skills=["Python", "GraphQL"])
    profile = make_profile(skills=["Python"])
    items = [
        make_evidence("resume-skills-001", "resume", "Resume skills: Python", tags=["Python"]),
        make_evidence("portfolio-P1", "portfolio", "GraphQL gateway work", tags=["service layer"]),
    ]
    result = entry.analyze_fit(
        make_input(job, profile, items),
        complete_fn=_stub(
            evidenced_missing_skills=[
                _claim("GraphQL: evidenced by the gateway project", ["portfolio-P1"], "verdict=missing; llm")
            ]
        ),
    )
    assert "GraphQL" in _skill_names(result.evidenced_missing_skills)
    assert "GraphQL" not in _skill_names(result.genuine_gaps)


def test_omitted_required_skill_is_backfilled_from_deterministic_pass() -> None:
    # A model that returns nothing must not yield a less complete analysis.
    job = make_job(required_skills=["Python", "Rust"])
    profile = make_profile(skills=["Python"])
    items = [make_evidence("resume-skills-001", "resume", "Resume skills: Python", tags=["Python"])]
    result = entry.analyze_fit(make_input(job, profile, items), complete_fn=_stub())

    covered = (
        _skill_names(result.aligned_skills)
        | _skill_names(result.evidenced_missing_skills)
        | _skill_names(result.genuine_gaps)
    )
    assert {"Python", "Rust"} <= covered
    assert "Python" in _skill_names(result.aligned_skills)
    assert "Rust" in _skill_names(result.genuine_gaps)


def test_model_swap_choice_is_honoured_when_it_validates() -> None:
    inp = _swap_fixture()
    result = entry.analyze_fit(
        inp,
        complete_fn=_stub(
            project_swap={
                "remove_project": "Current Weak",
                "add_project": "External Two",  # deterministic ranking would pick External One
                "rationale": "The model's own reasoning for External Two.",
                "evidence_ids": ["portfolio-P4"],
            }
        ),
    )
    assert result.project_swap is not None
    assert result.project_swap.add_project == "External Two"
    assert result.project_swap.remove_project == "Current Weak"
    assert result.project_swap.rationale == "The model's own reasoning for External Two."


def test_model_chosen_removal_is_reported_as_the_weak_slot() -> None:
    # The invariant must hold for a model-chosen swap too, not just the ranked one.
    inp = _swap_fixture()
    result = entry.analyze_fit(
        inp,
        complete_fn=_stub(
            project_swap={
                "remove_project": "Current Weak",
                "add_project": "External Two",
                "rationale": "model rationale",
                "evidence_ids": [],
            }
        ),
    )
    removed = result.project_swap.remove_project
    matching = [c for c in result.project_analysis if c.claim.startswith(f"Current project '{removed}'")]
    assert matching, "the removed project must appear in project_analysis"
    for claim in matching:
        value, _ = verdict.parse(claim.notes)
        assert value == verdict.MISMATCH
        assert "aligns well" not in claim.claim


def test_invented_swap_target_falls_back_to_ranked_choice() -> None:
    inp = _swap_fixture()
    result = entry.analyze_fit(
        inp,
        complete_fn=_stub(
            project_swap={
                "remove_project": "Current Weak",
                "add_project": "Project That Does Not Exist",
                "rationale": "fabricated",
                "evidence_ids": [],
            }
        ),
    )
    # Falls back to the deterministic ranking rather than emitting the invention.
    assert result.project_swap is not None
    assert result.project_swap.add_project == "External One"


def test_swap_below_margin_falls_back_to_ranked_choice() -> None:
    # The strong incumbent now scores as well as the external candidate, so swapping
    # it out gains nothing and the proposal must be rejected.
    inp = _swap_fixture(strong_domains=["Computer Vision"])
    result = entry.analyze_fit(
        inp,
        complete_fn=_stub(
            project_swap={
                "remove_project": "Current Strong",
                "add_project": "External Two",
                "rationale": "marginal",
                "evidence_ids": [],
            }
        ),
    )
    assert result.project_swap.remove_project == "Current Weak"  # ranked choice
    assert result.project_swap.add_project == "External One"


def test_seniority_prose_contradicting_the_years_verdict_is_replaced() -> None:
    job = make_job(required_skills=["Python"], years=10)
    profile = make_profile(skills=["Python"], years=2)  # a clear shortfall
    items = [make_evidence("resume-skills-001", "resume", "Resume skills: Python", tags=["Python"])]
    result = entry.analyze_fit(
        make_input(job, profile, items),
        complete_fn=_stub(
            seniority=[_claim("You exceed the requirement comfortably.", [], "verdict=match; llm")]
        ),
    )
    assert result.seniority, "a seniority claim must still be produced"
    claim = result.seniority[0]
    assert "exceed" not in claim.claim.lower()
    value, _ = verdict.parse(claim.notes)
    assert value in {verdict.PARTIAL, verdict.MISMATCH}


def test_model_narrative_sections_pass_through() -> None:
    job = make_job(required_skills=["Python"])
    profile = make_profile(skills=["Python"])
    items = [
        make_evidence("resume-skills-001", "resume", "Resume skills: Python", tags=["Python"]),
        make_evidence("resume-education-001", "resume", "M.S. Data Science", tags=["education"]),
    ]
    result = entry.analyze_fit(
        make_input(job, profile, items),
        complete_fn=_stub(
            relevant_experience=[
                _claim("Senior Engineer at Acme: centered on X — strong overlap.", ["resume-skills-001"], "verdict=match; llm")
            ],
            education=[_claim("Education: M.S. Data Science", ["resume-education-001"], "verdict=match; llm")],
        ),
    )
    assert result.relevant_experience[0].claim.startswith("Senior Engineer at Acme")
    assert result.education[0].claim == "Education: M.S. Data Science"
