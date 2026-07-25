"""Constrained LLM rationale: used when consistent, template kept when it drifts."""

from __future__ import annotations

from src.schemas.common import EvidenceClaim
from src.tests.fit_analysis_helpers import (
    make_evidence,
    make_input,
    make_job,
    make_portfolio_project,
    make_profile,
)
from src.tools.implementations.fit_analysis import rationale, verdict
from src.tools.implementations.fit_analysis.evidence_index import build_evidence_index
from src.tools.implementations.fit_analysis.prepass import run_prepass


def _seniority_claim() -> list[EvidenceClaim]:
    return [
        EvidenceClaim(
            claim="Seniority: ~4 years of experience vs job (5+ years expected).",
            evidence_ids=[],
            notes=verdict.tag(verdict.PARTIAL, "Close but below."),
        )
    ]


def test_seniority_rationale_used_when_consistent() -> None:
    out = rationale.enrich_seniority(
        _seniority_claim(),
        lambda s, u: "With about four years, the candidate is close to but below the "
        "five-plus years this role expects.",
    )
    assert out[0].claim.startswith("With about four years")


def test_seniority_rationale_rejected_on_contradiction() -> None:
    out = rationale.enrich_seniority(
        _seniority_claim(),
        lambda s, u: "The candidate clearly exceeds the requirement for this role.",
    )
    assert out[0].claim.startswith("Seniority:")  # template kept


def _swap_prepass_and_input():
    job = make_job(required_skills=["PyTorch", "RAG"], description="Generative AI with RAG.")
    profile = make_profile(skills=["Python"], resume_projects=["Vision Project", "Chatbot"])
    portfolio = [
        make_portfolio_project("P1", "Vision Project", technologies=["PyTorch"], domains=["Computer Vision"]),
        make_portfolio_project("P2", "Chatbot", technologies=["Python"]),
        make_portfolio_project("P3", "RAG Platform", technologies=["RAG", "PyTorch"], domains=["Generative AI"]),
        make_portfolio_project("P4", "Distractor Project", technologies=["Go"]),
    ]
    items = [make_evidence("resume-skills-001", "resume", "Resume skills: Python", tags=["Python"])]
    inp = make_input(
        job, profile, items,
        current_resume_projects=["Vision Project", "Chatbot"], portfolio_projects=portfolio,
    )
    return run_prepass(inp, build_evidence_index(items)), inp


def test_swap_rationale_used_when_naming_correct_project() -> None:
    prepass, inp = _swap_prepass_and_input()
    assert prepass.swap.swap is not None  # a swap was decided
    add = prepass.swap.swap.add_project
    out = rationale.enrich_swap(
        prepass.swap.swap, prepass, inp,
        lambda s, u: f"Swap in '{add}' for its RAG technology, generative-AI domain, and industry fit.",
    )
    assert out.rationale.startswith("Swap in")


def test_swap_rationale_rejected_when_it_names_a_different_project() -> None:
    prepass, inp = _swap_prepass_and_input()
    original = prepass.swap.swap.rationale
    out = rationale.enrich_swap(
        prepass.swap.swap, prepass, inp,
        lambda s, u: "Actually you should add the Distractor Project instead.",
    )
    assert out.rationale == original  # template kept; drift rejected
