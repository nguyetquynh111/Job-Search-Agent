"""Canonical module paths for model-visible tools."""

TOOL_MODULES: dict[str, str] = {
    "filter_jobs": "src.tools.filtering.tool",
    "score_jobs": "src.tools.scoring.tool",
    "analyze_fit": "src.tools.fit_analysis.tool",
    "tailor_resume": "src.tools.resume_tailoring.tool",
    "generate_cover_letter": "src.tools.cover_letter.tool",
}
