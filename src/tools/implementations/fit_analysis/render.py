"""Human-readable fit-analysis rendering.

``render_fit_analysis`` is a pure function returning the assignment-required
report. Markers are derived from each claim's verdict (never defaulted to ✅), and
missing-but-evidenced skills name their human-readable source inline. Evidence IDs
are kept, printed after the readable citation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.config import AppConfig, get_config
from src.schemas.common import EvidenceClaim
from src.schemas.fit_analysis import AnalyzeFitInput, FitAnalysisOutput
from src.schemas.jobs import Job
from src.tools.implementations.fit_analysis import verdict

logger = logging.getLogger(__name__)


def build_source_labels(inp: AnalyzeFitInput) -> dict[str, str]:
    """Map evidence IDs to human-readable source labels for inline citation."""

    labels: dict[str, str] = {}
    for project in inp.portfolio_projects:
        for evidence_id in project.evidence_ids:
            labels[evidence_id] = f'used in "{project.name}"'
    for item in inp.evidence_items:
        if item.evidence_id in labels:
            continue
        kind = item.source.strip().lower()
        if kind == "master_skills":
            labels[item.evidence_id] = "master skills list"
        elif kind == "memory":
            labels[item.evidence_id] = "stated during review"
        elif kind == "portfolio":
            labels[item.evidence_id] = "portfolio"
        elif kind == "resume":
            tags = {t.lower() for t in item.tags}
            if "skills" in tags:
                labels[item.evidence_id] = "resume skills"
            elif "experience" in tags:
                labels[item.evidence_id] = "resume experience"
            elif "education" in tags:
                labels[item.evidence_id] = "resume education"
            else:
                labels[item.evidence_id] = "resume"
        else:
            labels[item.evidence_id] = kind or "evidence"
    return labels


def _readable_sources(claim: EvidenceClaim, labels: dict[str, str] | None) -> list[str]:
    """Return de-duplicated human labels for a claim's evidence IDs."""

    if not labels:
        return []
    ordered: list[str] = []
    for evidence_id in claim.evidence_ids:
        label = labels.get(evidence_id)
        if label and label not in ordered:
            ordered.append(label)
    return ordered


def _citation(claim: EvidenceClaim, labels: dict[str, str] | None) -> str:
    """Render the trailing '(readable sources) [evidence: ids]' citation."""

    readable = _readable_sources(claim, labels)
    parts = []
    if readable:
        parts.append(f"({'; '.join(readable)})")
    if claim.evidence_ids:
        parts.append(f"_[evidence: {', '.join(claim.evidence_ids)}]_")
    return ("  " + "  ".join(parts)) if parts else ""


def _narrative_lines(
    claims: list[EvidenceClaim], labels: dict[str, str] | None, missing_marker: str = "❌"
) -> list[str]:
    """Render full-text claims with a verdict-derived marker."""

    if not claims:
        return ["_None._"]
    lines = []
    for claim in claims:
        value, human = verdict.parse(claim.notes)
        marker = verdict.marker(value, missing_marker)
        lines.append(f"{marker} {claim.claim}{_citation(claim, labels)}")
    return lines


def _skill_lines(
    claims: list[EvidenceClaim], marker: str, labels: dict[str, str] | None, inline_source: bool
) -> list[str]:
    """Render skill claims; optionally as 'skill (readable source)' with IDs after."""

    if not claims:
        return ["_None._"]
    lines = []
    for claim in claims:
        if inline_source:
            skill = claim.claim.split(":", 1)[0]
            readable = _readable_sources(claim, labels)
            via = _via_members(claim)
            inner = "; ".join(readable)
            if via:
                inner = f"{inner} via {via}" if inner else f"via {via}"
            source = f" ({inner})" if inner else ""
            ids = f"  _[evidence: {', '.join(claim.evidence_ids)}]_" if claim.evidence_ids else ""
            lines.append(f"{marker} {skill}{source}{ids}")
        else:
            lines.append(f"{marker} {claim.claim}{_citation(claim, labels)}")
    return lines


def _via_members(claim: EvidenceClaim) -> str:
    """Return the concrete category members named in a claim's notes (or '')."""

    _, human = verdict.parse(claim.notes)
    if human and human.startswith("via="):
        return human[len("via=") :].split(";", 1)[0].strip()
    return ""


def render_fit_analysis(
    output: FitAnalysisOutput,
    job: Job | None = None,
    *,
    source_labels: dict[str, str] | None = None,
    missing_marker: str = "❌",
) -> str:
    """Return the 'why this job is a good fit' report as Markdown (pure function).

    ``missing_marker`` sets the marker for the missing-but-evidenced group ("❌" to
    match the assignment's two-❌-groups convention, or "➕" to distinguish visually).
    """

    labels = source_labels
    parts: list[str] = ["# Tell me why this job is a good fit for me.", ""]
    if job is not None:
        parts.append(f"**Job:** {job.title} at {job.company}  \n**Job ID:** {output.job_id}")
    else:
        parts.append(f"**Job ID:** {output.job_id}")
    parts.append("")

    parts.append("## Relevant Experience")
    parts += _narrative_lines(output.relevant_experience, labels)
    parts.append("")

    parts.append("## Seniority")
    parts += _narrative_lines(output.seniority, labels)
    parts.append("")

    parts.append("## Education")
    parts += _narrative_lines(output.education, labels)
    parts.append("")

    parts.append("## Core Skills")
    parts.append("")
    parts.append("**Aligned (already on your resume):**")
    parts += _skill_lines(output.aligned_skills, "✅", labels, inline_source=False)
    parts.append("")
    parts.append("**Missing but evidenced (safe to add during tailoring):**")
    parts += _skill_lines(output.evidenced_missing_skills, missing_marker, labels, inline_source=True)
    parts.append("")
    parts.append("**Genuine gaps (no supporting evidence):**")
    parts += _skill_lines(output.genuine_gaps, "❌", labels, inline_source=True)
    parts.append("")

    parts.append("## Projects")
    parts += _narrative_lines(output.project_analysis, labels)
    parts.append("")
    if output.project_swap is not None:
        swap = output.project_swap
        ids = f"  _[evidence: {', '.join(swap.evidence_ids)}]_" if swap.evidence_ids else ""
        parts.append(
            f"**Recommended swap:** replace \"{swap.remove_project}\" with "
            f"\"{swap.add_project}\" — {swap.rationale}{ids}"
        )
    else:
        parts.append(
            "**No swap recommended:** the current resume projects are the strongest "
            "available match for this job."
        )
    parts.append("")
    return "\n".join(parts)


def write_fit_analysis(
    output: FitAnalysisOutput,
    job: Job | None = None,
    config: AppConfig | None = None,
    *,
    source_labels: dict[str, str] | None = None,
    missing_marker: str = "❌",
) -> tuple[Path, Path]:
    """Write ``fit_analysis.md`` and ``fit_analysis.json`` to ``<output_dir>/<job_id>/``.

    The per-job directory is shared with other tools: it is created if absent and
    never cleared. Only these two files are written.
    """

    config = config or get_config()
    job_dir = config.output_dir / output.job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    md_path = job_dir / "fit_analysis.md"
    json_path = job_dir / "fit_analysis.json"
    md_path.write_text(
        render_fit_analysis(output, job, source_labels=source_labels, missing_marker=missing_marker),
        encoding="utf-8",
    )
    json_path.write_text(
        json.dumps(output.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Wrote fit analysis to %s and %s", md_path, json_path)
    return md_path, json_path
