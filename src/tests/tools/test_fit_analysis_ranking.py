"""Regression pins for project relevance ranking against the repository's real data.

The deterministic ranking is the fallback path, so it must be defensible on its own
even when the LLM path is unavailable.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.data_loader import load_jobs_csv, load_portfolio
from src.tools.fit_analysis.prepass import _expand_canonicals
from src.tools.fit_analysis.sanitize import sanitize_text
from src.tools.fit_analysis.swap import _matches, rank_projects

_ROOT = Path(__file__).resolve().parents[3]
_JOBS = _ROOT / "data" / "jobs.csv"
_PORTFOLIO = _ROOT / "data" / "portfolio.txt"

pytestmark = pytest.mark.skipif(
    not (_JOBS.exists() and _PORTFOLIO.exists()),
    reason="repository data files not present",
)


def _ranking(job_id: str):
    job = next(j for j in load_jobs_csv(str(_JOBS)) if j.job_id == job_id)
    portfolio = load_portfolio(str(_PORTFOLIO))
    job_text = sanitize_text(f"{job.title}. {job.description} {job.company_details}")
    return job, rank_projects(
        portfolio.projects, job, _expand_canonicals(job.required_skills), job_text
    )


def test_j030_prefers_the_recommendation_project_over_medical_imaging() -> None:
    """A streaming-personalization role must rank ranking work above ultrasound imaging.

    Whole-phrase domain matching used to score the portfolio's "Recommendation and
    Ranking" project at zero against a posting requiring "recommendation systems",
    while an unrelated medical project won on the literal string "Deep Learning".
    """

    job, ranking = _ranking("J030")
    assert "recommendation systems" in [s.lower() for s in job.required_skills]

    by_id = {item.project.project_id: item for item in ranking}
    assert by_id["P08"].score > by_id["P04"].score
    assert ranking[0].project.project_id == "P08"
    assert "Recommendation and Ranking" in by_id["P08"].domain_matches
    # Generic overlap alone does not count.
    assert by_id["P04"].domain_matches == []


def test_partial_phrase_overlap_needs_a_distinctive_word() -> None:
    # One distinctive word can carry a phrase.
    assert _matches(
        ["Recommendation and Ranking"], "we build recommendation systems"
    ) == ["Recommendation and Ranking"]
    # Filler words cannot.
    assert _matches(["Deep Learning"], "deep learning models") == []
    assert _matches(["Computer Vision"], "a computer in the office") == []
    assert _matches(["Computer Vision"], "vision transformers") == ["Computer Vision"]
    # Singular and plural forms match.
    assert _matches(["Decentralized Data Systems"], "a decentralized system") == [
        "Decentralized Data Systems"
    ]
