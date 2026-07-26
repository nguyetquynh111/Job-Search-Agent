"""Contract-focused unit tests for the ``filter_jobs`` tool."""

from __future__ import annotations

from src.schemas.common import CandidatePreferences
from src.schemas.filtering import FilterJobsInput
from src.schemas.jobs import Job
from src.tools.implementations.filter_jobs import filter_jobs


def _job(job_id: str, **overrides: object) -> Job:
    payload = {
        "job_id": job_id,
        "title": "Machine Learning Engineer",
        "company": "Acme AI",
        "industry_domain": "Software / Machine Learning",
        "location": "Austin, TX",
        "description": "Build ML systems.",
        "required_skills": ["Python", "PyTorch"],
        "url": "",
    }
    payload.update(overrides)
    return Job.model_validate(payload)


def _run(jobs: list[Job], **pref_kwargs: object) -> FilterJobsInput:
    prefs = CandidatePreferences.model_validate(pref_kwargs)
    return filter_jobs(FilterJobsInput(jobs=jobs, preferences=prefs))


def test_every_job_appears_in_exactly_one_output_list() -> None:
    jobs = [_job("A", location="Austin, TX"), _job("B", location="Berlin, Germany")]
    out = _run(jobs, preferred_locations=["Austin, TX"])
    accepted = {job.job_id for job in out.accepted_jobs}
    rejected = {rj.job.job_id for rj in out.rejected_jobs}
    assert accepted | rejected == {"A", "B"}
    assert not (accepted & rejected)


def test_no_preferences_accepts_everything() -> None:
    jobs = [_job("A", location="Nowhere"), _job("B", company="Anything")]
    out = _run(jobs)  # empty preferences => every rule inert
    assert len(out.accepted_jobs) == 2
    assert not out.rejected_jobs


def test_location_filter_rejects_onsite_outside_preferred() -> None:
    out = _run([_job("A", location="Seattle, WA")], preferred_locations=["Houston, TX"])
    assert not out.accepted_jobs
    assert "Seattle" in out.rejected_jobs[0].reasons[0]


def test_location_filter_matches_city_token() -> None:
    out = _run(
        [_job("A", location="Houston, TX (Hybrid)")],
        preferred_locations=["Houston, TX"],
    )
    assert [job.job_id for job in out.accepted_jobs] == ["A"]


def test_remote_job_always_passes_location() -> None:
    out = _run(
        [_job("A", location="Fully Remote", remote=True)],
        preferred_locations=["Houston, TX"],
    )
    assert [job.job_id for job in out.accepted_jobs] == ["A"]


def test_nationwide_job_passes_when_candidate_accepts_remote() -> None:
    out = _run(
        [_job("A", location="United States")],
        preferred_locations=["Houston, TX", "Remote, US"],
    )
    assert [job.job_id for job in out.accepted_jobs] == ["A"]


def test_remote_only_rejects_onsite() -> None:
    out = _run([_job("A", location="Austin, TX", remote=False)], remote_only=True)
    assert not out.accepted_jobs
    assert "remote" in out.rejected_jobs[0].reasons[0].lower()


def test_experience_filter_rejects_underqualified() -> None:
    out = _run(
        [_job("A", years_experience_required=8)],
        years_of_experience=4,
    )
    assert not out.accepted_jobs
    assert "8+ years" in out.rejected_jobs[0].reasons[0]


def test_experience_filter_accepts_when_candidate_meets_minimum() -> None:
    out = _run(
        [_job("A", years_experience_required=3)],
        years_of_experience=4,
    )
    assert [job.job_id for job in out.accepted_jobs] == ["A"]


def test_company_exclusion_is_case_and_space_insensitive() -> None:
    out = _run(
        [_job("A", company="Booz Allen Hamilton")],
        excluded_companies=["booz-allen  hamilton"],
    )
    assert not out.accepted_jobs
    assert "exclusion list" in out.rejected_jobs[0].reasons[0]


def test_excluded_keyword_rejects_matching_posting() -> None:
    out = _run(
        [_job("A", title="Blockchain ML Engineer")],
        excluded_keywords=["blockchain"],
    )
    assert not out.accepted_jobs
    assert "blockchain" in out.rejected_jobs[0].reasons[0].lower()


def test_target_title_filter_rejects_unrelated_title() -> None:
    out = _run(
        [_job("A", title="Senior Database Administrator")],
        target_job_titles=["AI Engineer", "Machine Learning Engineer"],
    )
    assert not out.accepted_jobs
    assert "target titles" in out.rejected_jobs[0].reasons[0]


def test_target_title_filter_keeps_ai_ml_titles() -> None:
    out = _run(
        [_job("A", title="Computer Vision AI Engineer")],
        target_job_titles=["AI Engineer"],
    )
    assert [job.job_id for job in out.accepted_jobs] == ["A"]


def test_rejected_job_collects_all_failing_reasons() -> None:
    out = _run(
        [_job("A", location="Seattle, WA", years_experience_required=9)],
        preferred_locations=["Houston, TX"],
        years_of_experience=4,
    )
    reasons = out.rejected_jobs[0].reasons
    assert len(reasons) >= 2  # both location and experience failures logged


def test_filtering_is_deterministic_and_does_not_mutate_jobs() -> None:
    jobs = [_job("A", location="Seattle, WA"), _job("B", location="Houston, TX")]
    first = _run(jobs, preferred_locations=["Houston, TX"])
    second = _run(jobs, preferred_locations=["Houston, TX"])
    assert [j.job_id for j in first.accepted_jobs] == [
        j.job_id for j in second.accepted_jobs
    ]
    assert jobs[0].location == "Seattle, WA"  # inputs untouched
