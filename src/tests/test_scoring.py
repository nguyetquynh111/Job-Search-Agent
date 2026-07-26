"""Contract-focused unit tests for the deterministic ``score_jobs`` tool."""

from __future__ import annotations

from src.schemas.common import CandidateProfile, EvidenceItem
from src.schemas.jobs import Job
from src.schemas.scoring import ScoreJobsInput
from src.tools.implementations.score_jobs import score_jobs


def _job(job_id: str, **overrides: object) -> Job:
    payload = {
        "job_id": job_id,
        "title": "Machine Learning Engineer",
        "company": "Acme AI",
        "industry_domain": "Software / Machine Learning",
        "location": "Austin, TX",
        "description": "Build ML systems.",
        "required_skills": ["Python", "PyTorch", "Computer Vision"],
        "years_experience_required": 3,
        "url": "",
    }
    payload.update(overrides)
    return Job.model_validate(payload)


def _profile(**overrides: object) -> CandidateProfile:
    payload = {
        "candidate_id": "c1",
        "name": "Test Candidate",
        "preferences": {"years_of_experience": 4},
        "skills": ["Python", "PyTorch"],
        "master_skills": ["Computer Vision", "NLP"],
    }
    payload.update(overrides)
    return CandidateProfile.model_validate(payload)


def _run(jobs: list[Job], profile: CandidateProfile, **evidence: object) -> object:
    return score_jobs(
        ScoreJobsInput(jobs=jobs, candidate_profile=profile, **evidence)
    )


def test_scores_are_bounded_and_ranking_is_descending() -> None:
    jobs = [
        _job("HIGH", required_skills=["Python", "PyTorch", "Computer Vision"]),
        _job("LOW", required_skills=["Rust", "Kubernetes", "Go"], industry_domain="Fintech"),
    ]
    out = _run(jobs, _profile())
    scores = [sj.score for sj in out.ranked_jobs]
    assert all(0 <= s <= 100 for s in scores)
    assert scores == sorted(scores, reverse=True)
    assert out.ranked_jobs[0].job.job_id == "HIGH"


def test_top_3_selected_automatically() -> None:
    jobs = [_job(f"J{i}", required_skills=["Python"]) for i in range(5)]
    out = _run(jobs, _profile())
    assert out.top_3_job_ids == [sj.job.job_id for sj in out.ranked_jobs[:3]]
    assert len(out.top_3_job_ids) == 3


def test_full_skill_and_experience_match_scores_high() -> None:
    # 3/3 skills, experience met, domain overlaps the candidate's CV work, remote.
    job = _job("A", industry_domain="Computer Vision", location="Remote", remote=True)
    out = _run([job], _profile())
    assert out.ranked_jobs[0].score >= 90


def test_scoring_is_deterministic() -> None:
    jobs = [_job("A"), _job("B", required_skills=["Rust"])]
    profile = _profile()
    first = _run(jobs, profile)
    second = _run(jobs, profile)
    assert [(s.job.job_id, s.score) for s in first.ranked_jobs] == [
        (s.job.job_id, s.score) for s in second.ranked_jobs
    ]
    assert first.top_3_job_ids == second.top_3_job_ids


def test_evidence_ids_always_exist_in_input() -> None:
    evidence = [
        EvidenceItem(evidence_id="port-1", source="portfolio", text="PyTorch project", tags=["PyTorch"]),
    ]
    out = _run([_job("A")], _profile(), portfolio_evidence=evidence)
    valid = {"port-1"}
    for sj in out.ranked_jobs:
        assert set(sj.evidence_ids).issubset(valid)
    # The PyTorch requirement should be backed by the portfolio evidence item.
    assert "port-1" in out.ranked_jobs[0].evidence_ids


def test_category_requirement_satisfied_by_member_skill() -> None:
    # "vector databases" (a capability) is satisfied by "Pinecone" (a member).
    job = _job("A", required_skills=["vector databases"], years_experience_required=1)
    profile = _profile(master_skills=["Pinecone"])
    out = _run([job], profile)
    assert "1/1 required skills evidenced" in out.ranked_jobs[0].rationale


def test_memory_fact_counts_as_evidence() -> None:
    job = _job("A", required_skills=["GraphQL"], years_experience_required=1, industry_domain="")
    without = _run([job], _profile())
    memory = [EvidenceItem(evidence_id="mem-1", source="memory", text="GraphQL", tags=["GraphQL"])]
    with_mem = _run([job], _profile(), memory_evidence=memory)
    assert with_mem.ranked_jobs[0].score > without.ranked_jobs[0].score
    assert "mem-1" in with_mem.ranked_jobs[0].evidence_ids


def test_ranking_is_stable_for_ties() -> None:
    # Identical jobs must keep their input order (stable sort).
    jobs = [_job("FIRST"), _job("SECOND"), _job("THIRD")]
    out = _run(jobs, _profile())
    assert [sj.job.job_id for sj in out.ranked_jobs] == ["FIRST", "SECOND", "THIRD"]
