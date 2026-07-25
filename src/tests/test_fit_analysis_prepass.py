"""Deterministic pre-pass: indexing, skill buckets, memory evidence, None years."""

from __future__ import annotations

from src.tests.fit_analysis_helpers import (
    make_evidence,
    make_input,
    make_job,
    make_profile,
)
from src.tools.implementations.fit_analysis.evidence_index import build_evidence_index
from src.tools.implementations.fit_analysis.prepass import run_prepass


def test_alias_and_evidence_indexing_canonicalizes_variants() -> None:
    items = [
        make_evidence("m1", "master_skills", "Master skills: k8s, ML", tags=["k8s", "ML"]),
        make_evidence("p1", "portfolio", "block", tags=["torch"]),
    ]
    index = build_evidence_index(items)
    assert index.has("kubernetes")
    assert index.has("machine learning")
    assert index.has("pytorch")
    assert index.sources_for("kubernetes") == {"master_skills"}
    assert index.sources_for("pytorch") == {"portfolio"}


def test_ordered_ids_prefers_resume_then_portfolio() -> None:
    items = [
        make_evidence("portfolio-P1", "portfolio", "b", tags=["Python"]),
        make_evidence("resume-skills-001", "resume", "Resume skills: Python", tags=["Python"]),
        make_evidence("master-skills-001", "master_skills", "Master: Python", tags=["Python"]),
    ]
    index = build_evidence_index(items)
    ordered = index.ordered_ids("python")
    assert ordered[0] == "resume-skills-001"
    assert ordered[1] == "master-skills-001"
    assert ordered[2] == "portfolio-P1"


def test_skill_buckets_are_split_correctly() -> None:
    job = make_job(required_skills=["Python", "PyTorch", "Rust"])
    profile = make_profile(skills=["Python"])  # only Python on the resume
    items = [
        make_evidence("resume-skills-001", "resume", "Resume skills: Python", tags=["Python"]),
        make_evidence("portfolio-P1", "portfolio", "b", tags=["PyTorch"]),
    ]
    prepass = run_prepass(make_input(job, profile, items), build_evidence_index(items))

    aligned = {c.skill for c in prepass.aligned}
    missing = {c.skill for c in prepass.evidenced_missing}
    gaps = {c.skill for c in prepass.genuine_gaps}
    assert aligned == {"Python"}
    assert missing == {"PyTorch"}  # evidenced in portfolio, not on resume
    assert gaps == {"Rust"}  # no evidence anywhere


def test_memory_only_evidenced_skill_is_evidenced_missing_not_gap() -> None:
    job = make_job(required_skills=["GraphQL"])
    profile = make_profile(skills=["Python"])
    items = [make_evidence("mem-1", "memory", "skill: GraphQL", tags=["skill", "memory"])]
    prepass = run_prepass(make_input(job, profile, items), build_evidence_index(items))

    assert [c.skill for c in prepass.evidenced_missing] == ["GraphQL"]
    assert prepass.genuine_gaps == []
    assert prepass.evidenced_missing[0].sources == {"memory"}


def test_seniority_verdict_reflects_year_shortfall() -> None:
    from src.tools.implementations.fit_analysis import verdict
    from src.tools.implementations.fit_analysis.prepass import _seniority_verdict

    assert _seniority_verdict(6, 5) == verdict.MATCH
    assert _seniority_verdict(4, 5) == verdict.PARTIAL  # close shortfall
    assert _seniority_verdict(1, 5) == verdict.MISMATCH  # large shortfall
    assert _seniority_verdict(None, 5) == verdict.PARTIAL  # unknown candidate years
    assert _seniority_verdict(4, None) == verdict.MATCH  # no requirement stated


def test_none_years_required_does_not_crash_and_is_not_inferred() -> None:
    job = make_job(required_skills=["Python"], years=None)
    profile = make_profile(skills=["Python"], years=4)
    items = [
        make_evidence(
            "resume-experience-001",
            "resume",
            "Senior AI Engineer at Acme",
            tags=["experience"],
        )
    ]
    prepass = run_prepass(make_input(job, profile, items), build_evidence_index(items))

    assert len(prepass.seniority) == 1
    claim = prepass.seniority[0]
    assert claim.notes is not None and "not specified" in claim.notes.lower()
