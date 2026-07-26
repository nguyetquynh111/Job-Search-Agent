"""Public facade for the resume-tailoring tool.

The implementation lives in :mod:`src.tools.resume_tailoring.engine` so this
module can remain a small, stable API surface for callers and tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.tools.resume_tailoring import engine as _engine
from src.tools.resume_tailoring.contracts import (
    PortfolioRecord,
    SourceEdit,
    TailorResumeInput,
    TailorResumeOutput,
    TailoringError,
)
from src.tools.resume_tailoring.latex_structure import (
    LatexStructureError,
    parse_resume_structure,
)
from src.tracing.langfuse import TraceManager

_engine_compile_one_page = _engine._compile_one_page
_failure = _engine._failure
_build_summary = _engine._build_summary
_select_achievement = _engine._select_achievement
_evidence_sentence = _engine._evidence_sentence
_limit_sentence = _engine._limit_sentence
_summary_text = _engine._summary_text
_rewrite_experience_bullet = _engine._rewrite_experience_bullet
_plan_evidenced_skill_edit = _engine._plan_evidenced_skill_edit
_plan_project_swap = _engine._plan_project_swap
_render_project_like_existing = _engine._render_project_like_existing
_apply_source_edits = _engine._apply_source_edits
_select_experience_bullets = _engine._select_experience_bullets
_token_overlap = _engine._token_overlap
_revise_edited_content = _engine._revise_edited_content
_compact_prose = _engine._compact_prose
_refresh_change_log_after_text = _engine._refresh_change_log_after_text
_portfolio_record = _engine._portfolio_record
_assert_exactly_two_experience_bullets_changed = (
    _engine._assert_exactly_two_experience_bullets_changed
)
_assert_only_allowed_modifications = _engine._assert_only_allowed_modifications
_validate_change_log = _engine._validate_change_log
_unsupported_introduced_experience_terms = (
    _engine._unsupported_introduced_experience_terms
)
_word_tokens = _engine._word_tokens
_check_revision_feedback = _engine._check_revision_feedback
_validate_fit_analysis_evidence = _engine._validate_fit_analysis_evidence
_job_skill_matches = _engine._job_skill_matches
_skills_named_by_change = _engine._skills_named_by_change
_feedback_skill_requests = _engine._feedback_skill_requests
_meaningful_feedback_terms = _engine._meaningful_feedback_terms
_balanced_brace_content = _engine._balanced_brace_content
_run_pdflatex = _engine._run_pdflatex
run_pdflatex = _engine.run_pdflatex
_labeled_value = _engine._labeled_value
_split_labeled_list = _engine._split_labeled_list
_join_words = _engine._join_words
_latex_to_plain = _engine._latex_to_plain
_lower_first = _engine._lower_first


def _sync_patchable_engine_hooks() -> None:
    """Mirror facade-level monkeypatches onto the implementation module."""

    _engine._compile_one_page = _compile_one_page
    _engine._run_pdflatex = _run_pdflatex


def _compile_one_page(
    source: str,
    tex_path: Path,
    pdf_path: Path,
    *,
    tracer: TraceManager | None = None,
    trace_metadata: dict[str, object] | None = None,
    revision: Any = None,
) -> tuple[int | None, list[str], str]:
    """Compile through the engine while preserving the public monkeypatch seam."""

    _engine._run_pdflatex = _run_pdflatex
    return _engine_compile_one_page(
        source,
        tex_path,
        pdf_path,
        tracer=tracer,
        trace_metadata=trace_metadata,
        revision=revision,
    )


def run_resume_tailoring_tool(
    inp: TailorResumeInput,
    *,
    tracer: TraceManager | None = None,
) -> TailorResumeOutput:
    """Tailor one LaTeX resume and compile the verified one-page output."""

    _sync_patchable_engine_hooks()
    return _engine.run_resume_tailoring_tool(inp, tracer=tracer)


__all__ = [
    "PortfolioRecord",
    "SourceEdit",
    "TailorResumeInput",
    "TailorResumeOutput",
    "TailoringError",
    "LatexStructureError",
    "parse_resume_structure",
    "run_pdflatex",
    "run_resume_tailoring_tool",
]
