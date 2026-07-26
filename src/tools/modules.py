"""Canonical module paths for model-visible tools."""

TOOL_MODULES: dict[str, str] = {
    "filter_jobs": "src.tools.filtering",
    "score_jobs": "src.tools.scoring",
    "analyze_fit": "src.tools.fit_analysis",
    "tailor_resume": "src.tools.resume_tailoring",
    "generate_cover_letter": "src.tools.cover_letter",
}
