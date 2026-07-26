"""Renderer: verdict-derived markers, inline citations, swap/no-swap branches."""

from __future__ import annotations

from src.schemas.common import EvidenceClaim, ProjectSwap
from src.schemas.fit_analysis import FitAnalysisOutput
from src.tests.tools.fit_analysis_helpers import make_job
from src.tools.fit_analysis import verdict
from src.tools.fit_analysis.render import render_fit_analysis


def _sample() -> FitAnalysisOutput:
    return FitAnalysisOutput(
        job_id="J1",
        aligned_skills=[
            EvidenceClaim(
                claim="Python: aligned",
                evidence_ids=["e1"],
                notes=verdict.tag(verdict.MATCH),
            )
        ],
        evidenced_missing_skills=[
            EvidenceClaim(
                claim="deep learning: evidenced",
                evidence_ids=["portfolio-P04"],
                notes=verdict.tag(verdict.MISSING),
            )
        ],
        genuine_gaps=[
            EvidenceClaim(claim="Rust: gap", notes=verdict.tag(verdict.MISMATCH))
        ],
    )


def test_render_contains_all_sections() -> None:
    text = render_fit_analysis(_sample(), make_job())
    assert "Tell me why this job is a good fit for me." in text
    for section in (
        "Relevant Experience",
        "Seniority",
        "Education",
        "Core Skills",
        "Projects",
    ):
        assert f"## {section}" in text
    assert "✅ Python: aligned" in text
    assert "❌ Rust" in text  # Gaps render with the skill name.


def test_missing_group_marker_is_configurable() -> None:
    output = _sample()
    assert "❌ deep learning" in render_fit_analysis(
        output, make_job(), missing_marker="❌"
    )
    assert "➕ deep learning" in render_fit_analysis(
        output, make_job(), missing_marker="➕"
    )


def test_missing_skill_shows_human_readable_source_inline() -> None:
    labels = {"portfolio-P04": 'used in "IVUS Analysis"'}
    text = render_fit_analysis(_sample(), make_job(), source_labels=labels)
    assert 'deep learning (used in "IVUS Analysis")' in text
    assert (
        "_[evidence: portfolio-P04]_" in text
    )  # IDs follow the readable citation.


def test_partial_renders_as_cross_not_positive() -> None:
    # A seniority shortfall renders as a failure.
    output = FitAnalysisOutput(
        job_id="J1",
        seniority=[
            EvidenceClaim(
                claim="Seniority: 4y vs 5+", notes=verdict.tag(verdict.PARTIAL)
            )
        ],
    )
    text = render_fit_analysis(output, make_job())
    assert "❌ Seniority: 4y vs 5+" in text
    assert "✅" not in text  # A partial result cannot pass.
    assert "⚠️" not in text  # The report uses no warning marker.


def test_render_states_no_swap_branch_explicitly() -> None:
    text = render_fit_analysis(
        FitAnalysisOutput(job_id="J1", project_swap=None), make_job()
    )
    assert "No swap recommended" in text


def test_render_shows_recommended_swap() -> None:
    output = FitAnalysisOutput(
        job_id="J1",
        project_swap=ProjectSwap(
            remove_project="Old",
            add_project="New",
            rationale="stronger",
            evidence_ids=["portfolio-P2"],
        ),
    )
    text = render_fit_analysis(output, make_job())
    assert 'replace "Old" with "New"' in text
    assert "stronger" in text
