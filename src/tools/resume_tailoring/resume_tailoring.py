from __future__ import annotations

import inspect
import logging
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import Field

from app.configuration import get_config
from src.domain import ChangeLogEntry, EvidenceItem, Job, StrictBaseModel
from src.tools.fit_analysis.fit_analysis import FitAnalysisOutput
from src.tools.resume_tailoring.latex_structure import (
    LatexItem,
    LatexStructureError,
    ProjectEntry,
    ResumeStructure,
    TextRange,
    balanced_brace_content,
    parse_resume_structure,
)
from src.utils.evidence_validation import CANDIDATE_SOURCES, JOB_SOURCES
from src.utils.evidence_validation import (
    evidence_supports_keyword,
    evidence_supports_project,
    evidence_supports_skill,
    evidence_supports_statement,
    job_evidence_supports_skill,
)
from src.utils.job_evidence import (
    build_job_evidence,
    job_evidence_id,
    job_skill_evidence_id,
)
from src.utils.latex import escape_latex, pdf_page_count, pdflatex_command, run_pdflatex
from src.utils.skill_matching import (
    canonicalize,
    category_members,
    skill_in_text,
)
from src.tracing.langfuse import TraceManager

logger = logging.getLogger(__name__)

_escape_latex = escape_latex


class TailorResumeInput(StrictBaseModel):
    """Input for tailor_resume."""

    job: Job
    fit_analysis: FitAnalysisOutput
    source_resume_tex_path: str
    candidate_evidence: list[EvidenceItem] = Field(default_factory=list)
    job_evidence: list[EvidenceItem] = Field(default_factory=list)
    revision_feedback: str | None = None


class TailorResumeOutput(StrictBaseModel):
    """Output from tailor_resume."""

    job_id: str
    status: str
    output_tex_path: str
    output_pdf_path: str
    page_count: int = Field(ge=0)
    change_log: list[ChangeLogEntry] = Field(default_factory=list)
    revision_feedback_satisfied: bool | None = None
    revision_feedback_checks: list[str] = Field(default_factory=list)
    validation_failures: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


MAX_COMPILE_ATTEMPTS = 3


class TailoringError(RuntimeError):
    """Raised when a resume cannot be tailored without breaking its contract."""


@dataclass(frozen=True)
class PortfolioRecord:
    """Portfolio fields needed to render a real project into the resume."""

    project_id: str
    name: str
    period: str
    technologies: list[str]
    domain: str
    summary: str
    evidence_id: str


@dataclass(frozen=True)
class SourceEdit:
    """A single authorized replacement in the original uploaded source."""

    target: TextRange
    replacement: str
    category: str
    location: str


def run_resume_tailoring_tool(
    inp: TailorResumeInput,
    *,
    tracer: TraceManager | None = None,
) -> TailorResumeOutput:
    """Tailor one LaTeX resume and compile it only when it remains one page."""

    if not inp.job_evidence:
        inp = inp.model_copy(update={"job_evidence": build_job_evidence(inp.job)})
    active = tracer or TraceManager(enabled=False)
    trace_metadata = {
        "tool_name": "tailor_resume",
        "job_id": inp.job.job_id,
        "resume_id": f"resume-{inp.job.job_id}",
        "company": inp.job.company,
        "review_round": 1 if inp.revision_feedback else 0,
    }
    if inp.fit_analysis.job_id != inp.job.job_id:
        return _failure(
            inp.job.job_id,
            [
                f"Fit-analysis job ID {inp.fit_analysis.job_id!r} does not match the job."
            ],
        )

    source_path = Path(inp.source_resume_tex_path)
    if not source_path.is_file():
        return _failure(inp.job.job_id, [f"Source resume TeX not found: {source_path}"])
    try:
        source = source_path.read_text(encoding="utf-8")
    except OSError as exc:
        return _failure(inp.job.job_id, [f"Could not read source resume: {exc}"])
    if not source.strip():
        return _failure(inp.job.job_id, ["Source resume TeX is empty."])

    evidence = {item.evidence_id: item for item in inp.candidate_evidence}
    all_evidence = {
        item.evidence_id: item for item in [*inp.candidate_evidence, *inp.job_evidence]
    }
    validation_failures = _validate_fit_analysis_evidence(inp, all_evidence)
    if validation_failures:
        return _failure(
            inp.job.job_id,
            validation_failures,
            validation_failures=validation_failures,
        )
    try:
        output_dir = get_config().output_dir / inp.job.job_id
        output_dir.mkdir(parents=True, exist_ok=True)
        tex_path, pdf_path = _output_paths(output_dir, source_path)
        before_pdf_path = output_dir / "resume_before.pdf"
        if not before_pdf_path.is_file():
            before_tex_path = output_dir / "resume_before.tex"
            before_kwargs: dict[str, Any] = {}
            before_parameters = inspect.signature(_compile_one_page).parameters
            if "tracer" in before_parameters:
                before_kwargs.update(
                    {
                        "tracer": active,
                        "trace_metadata": {
                            **trace_metadata,
                            "artifact_role": "resume_before",
                        },
                    }
                )
            before_pages, before_errors, _ = _compile_one_page(
                source,
                before_tex_path,
                before_pdf_path,
                **before_kwargs,
            )
            if before_pages != 1 or before_errors or not before_pdf_path.is_file():
                return _failure(
                    inp.job.job_id,
                    before_errors
                    or ["Could not compile the original resume to exactly one page."],
                )

        with active.span(
            "resume_tailoring.edit_content",
            trace_metadata,
            input={
                "job_id": inp.job.job_id,
                "fit_analysis_job_id": inp.fit_analysis.job_id,
                "evidence_count": len(evidence),
                "revision_requested": bool(inp.revision_feedback),
            },
        ) as editing_span:
            structure = parse_resume_structure(
                source,
                require_projects=inp.fit_analysis.project_swap is not None,
            )
            edits: list[SourceEdit] = []
            changes: list[ChangeLogEntry] = []

            old_summary = structure.summary_range.text(source)
            summary, summary_ids = _build_summary(inp, evidence)
            summary_ids = list(
                dict.fromkeys([job_evidence_id(inp.job, "title"), *summary_ids])
            )
            if old_summary.strip() != summary.strip():
                edits.append(
                    SourceEdit(
                        structure.summary_range,
                        summary,
                        "summary",
                        f"{structure.sections['summary'].title.title()} section",
                    )
                )
                changes.append(
                    ChangeLogEntry(
                        change_id=f"{inp.job.job_id}-summary",
                        section="summary",
                        description=f"Rewrote the summary for {inp.job.title}.",
                        before_text=old_summary,
                        after_text=summary,
                        reason=(
                            "Align the professional summary with the target role using "
                            "only job-posting and candidate evidence."
                        ),
                        evidence_ids=summary_ids,
                        source_location=(
                            f"{structure.sections['summary'].title.title()} section"
                        ),
                        validation_result="pending structural and evidence validation",
                    )
                )
            elif not inp.revision_feedback:
                raise TailoringError(
                    "The proposed Professional Summary is identical to the existing "
                    "summary; a real rewrite is required."
                )

            selected_bullets = _select_experience_bullets(structure, inp, evidence)
            selected_ordinals = [item.ordinal for item, _ in selected_bullets]
            for index, (bullet, bullet_evidence) in enumerate(
                selected_bullets, start=1
            ):
                old_bullet = bullet.content(source)
                new_bullet, evidence_ids = _rewrite_experience_bullet(
                    old_bullet,
                    index,
                    inp,
                    evidence,
                    evidence_item=bullet_evidence,
                )
                edits.append(
                    SourceEdit(
                        bullet.content_range,
                        new_bullet,
                        "experience",
                        f"Experience bullet {bullet.ordinal}",
                    )
                )
                changes.append(
                    ChangeLogEntry(
                        change_id=f"{inp.job.job_id}-experience-{index}",
                        section="experience",
                        description=(
                            f"Rewrote existing experience bullet {bullet.ordinal}."
                        ),
                        before_text=old_bullet,
                        after_text=new_bullet,
                        reason=(
                            "Emphasize the existing experience most relevant to "
                            "evidenced job requirements without changing its facts."
                        ),
                        evidence_ids=evidence_ids,
                        source_location=f"Experience bullet {bullet.ordinal}",
                        validation_result="pending structural and evidence validation",
                    )
                )

            (
                skill_edit,
                added_skills,
                skill_ids,
                skill_evidence,
            ) = _plan_evidenced_skill_edit(source, structure, inp, evidence)
            if skill_edit is not None:
                edits.append(skill_edit)
                incremental_before = skill_edit.target.text(source)
                for skill_index, skill in enumerate(added_skills, start=1):
                    separator = (
                        "" if incremental_before.rstrip().endswith((",", ";")) else ","
                    )
                    incremental_after = (
                        incremental_before.rstrip()
                        + separator
                        + " "
                        + _escape_latex(skill)
                    )
                    changes.append(
                        ChangeLogEntry(
                            change_id=(f"{inp.job.job_id}-skills-{skill_index}"),
                            section="skills",
                            description=f"Added evidenced skill: {skill}.",
                            before_text=incremental_before,
                            after_text=incremental_after,
                            reason=(
                                "Surface a job-relevant skill already supported by "
                                "candidate evidence."
                            ),
                            evidence_ids=skill_evidence[skill],
                            source_location=skill_edit.location,
                            validation_result=(
                                "pending structural and evidence validation"
                            ),
                        )
                    )
                    incremental_before = incremental_after

            if inp.fit_analysis.project_swap is not None:
                project_edit, record = _plan_project_swap(
                    source, structure, inp, evidence
                )
                if project_edit is not None:
                    edits.append(project_edit)
                    changes.append(
                        ChangeLogEntry(
                            change_id=f"{inp.job.job_id}-project-swap",
                            section="projects",
                            description=(
                                f'Replaced "{inp.fit_analysis.project_swap.remove_project}" '
                                f'with "{record.name}".'
                            ),
                            before_text=project_edit.target.text(source),
                            after_text=project_edit.replacement,
                            reason=inp.fit_analysis.project_swap.rationale,
                            evidence_ids=list(
                                dict.fromkeys(
                                    [
                                        job_evidence_id(inp.job, "description"),
                                        record.evidence_id,
                                    ]
                                )
                            ),
                            source_location=project_edit.location,
                            validation_result=(
                                "pending structural and portfolio validation"
                            ),
                        )
                    )
            updated = _apply_source_edits(source, edits)
            _assert_exactly_two_experience_bullets_changed(source, updated)
            project_swap_applied = any(
                change.section == "projects" for change in changes
            )
            _assert_only_allowed_modifications(
                source,
                updated,
                edits=edits,
            )
            _validate_change_log(changes, all_evidence, inp)
            changes = [
                change.model_copy(
                    update={
                        "validation_result": (
                            "validated: source scope preserved and evidence confirmed"
                        )
                    }
                )
                for change in changes
            ]
            active.update_span(
                editing_span,
                output={
                    "change_count": len(changes),
                    "sections": [change.section for change in changes],
                    "added_skills": added_skills,
                    "project_swap_applied": project_swap_applied,
                },
                metadata={"status": "OK"},
            )
        compile_kwargs = {}
        compile_parameters = inspect.signature(_compile_one_page).parameters
        if "tracer" in compile_parameters:
            compile_kwargs.update(
                {
                    "tracer": active,
                    "trace_metadata": trace_metadata,
                }
            )
        if "revision" in compile_parameters:
            compile_kwargs["revision"] = lambda working, attempt: (
                _revise_edited_content(
                    working,
                    source,
                    selected_ordinals,
                    attempt,
                )
            )
        page_count, compile_errors, _compiled_source = _compile_one_page(
            updated,
            tex_path,
            pdf_path,
            **compile_kwargs,
        )
        if _compiled_source != updated:
            _assert_exactly_two_experience_bullets_changed(source, _compiled_source)
            changes = _refresh_change_log_after_text(
                changes,
                _compiled_source,
                selected_ordinals,
            )
            _validate_change_log(changes, all_evidence, inp)
        success = page_count == 1 and pdf_path.is_file() and not compile_errors
        feedback_satisfied, feedback_checks = _check_revision_feedback(
            inp.revision_feedback, changes, all_evidence
        )
        return TailorResumeOutput(
            job_id=inp.job.job_id,
            status="OK" if success else "ERROR",
            output_tex_path=str(tex_path),
            output_pdf_path=str(pdf_path) if success else "",
            page_count=page_count or 0,
            change_log=changes,
            revision_feedback_satisfied=feedback_satisfied,
            revision_feedback_checks=feedback_checks,
            validation_failures=[],
            errors=compile_errors,
        )
    except (TailoringError, LatexStructureError) as exc:
        return _failure(inp.job.job_id, [str(exc)])
    except Exception as exc:  # noqa: BLE001 - return an honest tool failure
        return _failure(inp.job.job_id, [f"Unexpected tailoring failure: {exc}"])


def _failure(
    job_id: str,
    errors: list[str],
    *,
    validation_failures: list[str] | None = None,
) -> TailorResumeOutput:
    return TailorResumeOutput(
        job_id=job_id,
        status="ERROR",
        output_tex_path="",
        output_pdf_path="",
        page_count=0,
        change_log=[],
        revision_feedback_satisfied=None,
        revision_feedback_checks=[],
        validation_failures=validation_failures or [],
        errors=errors,
    )


def _output_paths(output_dir: Path, source_path: Path) -> tuple[Path, Path]:
    """Use a new file for revisions so the previous approved source is untouched."""

    try:
        source_is_generated = source_path.resolve().parent == output_dir.resolve()
    except OSError:
        source_is_generated = False
    if not source_is_generated:
        stem = "resume_draft"
    else:
        match = re.fullmatch(r"resume-revision-(\d+)", source_path.stem)
        revision = int(match.group(1)) + 1 if match else 1
        stem = f"resume-revision-{revision}"
    return output_dir / f"{stem}.tex", output_dir / f"{stem}.pdf"


def _build_summary(
    inp: TailorResumeInput, evidence: dict[str, EvidenceItem]
) -> tuple[str, list[str]]:
    claims = [
        *inp.fit_analysis.aligned_skills,
        *inp.fit_analysis.evidenced_missing_skills,
    ]
    supported: list[tuple[str, list[str]]] = []
    for claim in claims:
        skill = claim.claim.split(":", 1)[0].strip()
        valid_ids = [item for item in claim.evidence_ids if item in evidence]
        if (
            skill
            and valid_ids
            and canonicalize(skill)
            not in {canonicalize(value) for value, _ in supported}
        ):
            supported.append((skill, valid_ids))
    skills = [value for value, _ in supported[:5]]
    evidence_ids = list(dict.fromkeys(item for _, ids in supported[:5] for item in ids))

    if skills:
        skill_text = _join_words(skills)
        summary = (
            f"Candidate targeting {_escape_latex(inp.job.title)} roles with "
            f"evidence-backed experience in {_escape_latex(skill_text)}."
        )
    else:
        summary = (
            f"Candidate targeting {_escape_latex(inp.job.title)} roles with "
            "experience supported by the accompanying resume and portfolio."
        )

    achievement = _select_achievement(evidence.values(), skills)
    if achievement is None:
        achievement = next(
            (
                item
                for item in evidence.values()
                if item.source in {"resume", "portfolio", "memory"}
            ),
            None,
        )
    if achievement is not None:
        sentence = _evidence_sentence(achievement.text)
        if sentence:
            summary += f" {_escape_latex(sentence)}"
            evidence_ids.append(achievement.evidence_id)
    return summary, list(dict.fromkeys(evidence_ids))


def _select_achievement(
    items: Iterable[EvidenceItem], skills: list[str]
) -> EvidenceItem | None:
    ranked: list[tuple[int, EvidenceItem]] = []
    for item in items:
        if not isinstance(item, EvidenceItem):
            continue
        if item.source not in {"resume", "portfolio", "memory"}:
            continue
        item_tags = {tag.casefold() for tag in item.tags}
        if item.source == "resume" and not {"experience", "project"} & item_tags:
            continue
        score = sum(skill_in_text(canonicalize(skill), item.text) for skill in skills)
        score += 2 if re.search(r"\d", item.text) else 0
        score += 1 if {"experience", "project"} & item_tags else 0
        ranked.append((score, item))
    ranked.sort(key=lambda pair: pair[0], reverse=True)
    return ranked[0][1] if ranked and ranked[0][0] > 0 else None


def _evidence_sentence(text: str) -> str:
    summary = _labeled_value(text, "SUMMARY")
    if summary:
        return _limit_sentence(summary)
    if " | " in text:
        return _limit_sentence(text.rsplit(" | ", 1)[-1])
    first = re.split(r"(?<=[.!?])\s+", " ".join(text.split()))[0]
    return _limit_sentence(first)


def _limit_sentence(text: str, limit: int = 190) -> str:
    cleaned = " ".join(text.split()).strip(" .")
    if len(cleaned) > limit:
        cleaned = cleaned[:limit].rsplit(" ", 1)[0].rstrip(" ,;:")
    return f"{cleaned}." if cleaned else ""


def _summary_text(source: str) -> str:
    """Return summary prose located from the document's section structure."""

    structure = parse_resume_structure(source)
    return structure.summary_range.text(source).strip()


def _rewrite_experience_bullet(
    old_latex: str,
    index: int,
    inp: TailorResumeInput,
    evidence: dict[str, EvidenceItem],
    *,
    evidence_item: EvidenceItem | None = None,
) -> tuple[str, list[str]]:
    preferred_id = f"resume-experience-{index:03d}"
    item = evidence_item or evidence.get(preferred_id)
    if item is None:
        candidates = [
            candidate
            for candidate in evidence.values()
            if candidate.source == "resume"
            and "experience" in {tag.casefold() for tag in candidate.tags}
        ]
        item = candidates[index - 1] if len(candidates) >= index else None

    claims = [
        *inp.fit_analysis.aligned_skills,
        *inp.fit_analysis.evidenced_missing_skills,
    ]
    matched_skills: list[str] = []
    evidence_ids: list[str] = []
    if item is not None:
        evidence_ids.append(item.evidence_id)
        for claim in claims:
            skill = claim.claim.split(":", 1)[0].strip()
            # Evidence elsewhere in the candidate profile cannot justify
            # attaching a skill to this specific experience bullet.
            if evidence_supports_skill(item, skill):
                if skill and canonicalize(skill) not in {
                    canonicalize(value) for value in matched_skills
                }:
                    matched_skills.append(skill)

    concise = bool(
        inp.revision_feedback
        and re.search(
            r"\b(shorter|concise|trim|brief)\b",
            inp.revision_feedback,
            re.IGNORECASE,
        )
    )
    factual_body = old_latex.strip()
    already_tailored = re.match(
        r"^(?:Applied .+? through work that|Demonstrated .+? through work that|"
        r"Highlighted relevant work that)\s+(.+)$",
        factual_body,
    )
    if already_tailored:
        factual_body = already_tailored.group(1)
    if concise:
        factual_body = (
            re.split(r";|\\,", factual_body, maxsplit=1)[0].rstrip(" .") + "."
        )
    factual_body = _lower_first(factual_body)
    if concise:
        label = (
            _escape_latex(matched_skills[0])
            if matched_skills
            else "relevant experience"
        )
        prefix = f"Applied {label}: "
        budget = max(24, len(old_latex.strip()) - len(prefix) - 2)
        if len(factual_body) > budget:
            factual_body = factual_body[:budget].rsplit(" ", 1)[0].rstrip(" ,;:")
            factual_body = factual_body.rstrip(".") + "."
    elif matched_skills:
        verb = "Demonstrated" if already_tailored else "Applied"
        prefix = f"{verb} {_escape_latex(_join_words(matched_skills[:3]))} through work that "
    else:
        prefix = "Highlighted relevant work that "
    rewritten = prefix + factual_body
    if rewritten.strip() == old_latex.strip():
        rewritten = "Delivered " + _lower_first(old_latex.strip())
    if item is None:
        raise TailoringError(
            f"Experience bullet {index} has no resume evidence and cannot be rewritten."
        )
    requirement_ids = [
        job_skill_evidence_id(inp.job, skill) for skill in matched_skills[:3]
    ]
    return rewritten, list(
        dict.fromkeys(
            [
                job_evidence_id(inp.job, "title"),
                *requirement_ids,
                *evidence_ids,
            ]
        )
    )


def _plan_evidenced_skill_edit(
    source: str,
    structure: ResumeStructure,
    inp: TailorResumeInput,
    evidence: dict[str, EvidenceItem],
) -> tuple[
    SourceEdit | None,
    list[str],
    list[str],
    dict[str, list[str]],
]:
    candidates: list[tuple[str, list[str]]] = []
    for claim in inp.fit_analysis.evidenced_missing_skills:
        skill = claim.claim.split(":", 1)[0].strip()
        valid_ids = [item for item in claim.evidence_ids if item in evidence]
        if skill and valid_ids:
            candidates.append(
                (
                    skill,
                    [job_skill_evidence_id(inp.job, skill), *valid_ids],
                )
            )

    feedback = inp.revision_feedback or ""
    for item in evidence.values():
        if item.source != "memory" or "skill" not in {
            tag.casefold() for tag in item.tags
        }:
            continue
        value = item.text.partition(":")[2].strip()
        if not value:
            continue
        if item.evidence_id in feedback or value.casefold() in feedback.casefold():
            requirement_id = next(
                (
                    job_item.evidence_id
                    for job_item in inp.job_evidence
                    if job_evidence_supports_skill(job_item, value)
                ),
                job_evidence_id(inp.job, "description"),
            )
            candidates.append(
                (
                    value,
                    [requirement_id, item.evidence_id],
                )
            )

    skills_section = structure.sections["skills"].content_range.text(source)
    existing_plain = _latex_to_plain(skills_section)
    additions: list[str] = []
    evidence_ids: list[str] = []
    evidence_by_skill: dict[str, list[str]] = {}
    seen = {
        canonicalize(match)
        for match in re.split(r"[,;\n]", existing_plain)
        if match.strip()
    }
    for skill, ids in candidates:
        canonical = canonicalize(skill)
        if (
            not canonical
            or canonical in seen
            or skill_in_text(canonical, existing_plain)
        ):
            continue
        seen.add(canonical)
        additions.append(skill)
        evidence_ids.extend(ids)
        evidence_by_skill[skill] = list(dict.fromkeys(ids))

    if not additions:
        return None, [], [], {}

    target = max(
        structure.skill_items,
        key=lambda item: sum(
            skill_in_text(canonicalize(skill), item.content(source))
            for skill in additions
        ),
    )
    old_content = target.content(source)
    separator = "" if old_content.rstrip().endswith((",", ";")) else ","
    replacement = (
        old_content.rstrip()
        + separator
        + " "
        + ", ".join(_escape_latex(value) for value in additions)
    )
    return (
        SourceEdit(
            target.content_range,
            replacement,
            "skills",
            f"Skills item {target.ordinal}",
        ),
        additions,
        list(dict.fromkeys(evidence_ids)),
        evidence_by_skill,
    )


def _plan_project_swap(
    source: str,
    structure: ResumeStructure,
    inp: TailorResumeInput,
    evidence: dict[str, EvidenceItem],
) -> tuple[SourceEdit | None, PortfolioRecord]:
    swap = inp.fit_analysis.project_swap
    if swap is None:
        raise TailoringError("Project swap was unexpectedly absent.")
    valid_items = [
        evidence[evidence_id]
        for evidence_id in swap.evidence_ids
        if evidence_id in evidence and evidence[evidence_id].source == "portfolio"
    ]
    item = next(
        (
            candidate
            for candidate in valid_items
            if swap.add_project.casefold() in candidate.text.casefold()
        ),
        None,
    )
    if item is None:
        raise TailoringError(
            f'Project "{swap.add_project}" has no matching portfolio evidence.'
        )
    record = _portfolio_record(item, expected_name=swap.add_project)
    if any(
        entry.name.casefold() == record.name.casefold()
        for entry in structure.project_entries
    ):
        return None, record
    if not swap.remove_project:
        raise TailoringError(
            "Project swap does not identify a resume project to remove."
        )

    matching_entries = [
        entry
        for entry in structure.project_entries
        if entry.name.casefold() == swap.remove_project.casefold()
    ]
    if len(matching_entries) != 1:
        raise TailoringError(
            f'Resume project "{swap.remove_project}" was not found uniquely in the '
            "Projects section."
        )
    entry = matching_entries[0]
    replacement = _render_project_like_existing(source, entry, record)
    return (
        SourceEdit(
            entry.block_range,
            replacement,
            "projects",
            f'Projects entry "{entry.name}"',
        ),
        record,
    )


def _render_project_like_existing(
    source: str, entry: ProjectEntry, record: PortfolioRecord
) -> str:
    """Fill an existing project entry while retaining its exact LaTeX skeleton."""

    if len(entry.bullets) != 1:
        raise TailoringError(
            "The selected project entry must have one descriptive bullet so its "
            "format can be preserved safely."
        )
    block = entry.block_range.text(source)
    if entry.command == "resumeEntry":
        if len(entry.field_ranges) != 4:
            raise TailoringError(
                "The selected project macro does not expose four existing header "
                "fields, so it cannot be populated safely."
            )
        replacements = [
            (entry.field_ranges[0], _escape_latex(record.name)),
            (entry.field_ranges[1], _escape_latex(record.period)),
            (
                entry.field_ranges[2],
                _escape_latex(", ".join(record.technologies[:6])),
            ),
            (entry.field_ranges[3], _escape_latex(record.domain)),
            (entry.bullets[0].content_range, _escape_latex(record.summary)),
        ]
    else:
        if len(entry.field_ranges) != 1:
            raise TailoringError("The standard project item has no unique title field.")
        replacements = [
            (entry.field_ranges[0], _escape_latex(record.name)),
            (entry.bullets[0].content_range, _escape_latex(record.summary)),
        ]
        header_end = entry.bullets[0].block_range.start
        header = source[entry.field_ranges[0].end : header_end]
        technology_match = re.search(r"\\(?:emph|textit)\s*\{", header)
        if technology_match:
            opening = entry.field_ranges[0].end + technology_match.end() - 1
            _, argument_end = _balanced_brace_content(source, opening)
            replacements.append(
                (
                    TextRange(opening + 1, argument_end - 1),
                    _escape_latex(", ".join(record.technologies[:6])),
                )
            )
        period_match = re.search(
            r"\b(?:19|20)\d{2}(?:\s*--\s*(?:Present|(?:19|20)\d{2}))?\b",
            header,
            re.IGNORECASE,
        )
        if period_match:
            replacements.append(
                (
                    TextRange(
                        entry.field_ranges[0].end + period_match.start(),
                        entry.field_ranges[0].end + period_match.end(),
                    ),
                    _escape_latex(record.period),
                )
            )
    relative = [
        SourceEdit(
            TextRange(
                target.start - entry.block_range.start,
                target.end - entry.block_range.start,
            ),
            value,
            "projects",
            entry.name,
        )
        for target, value in replacements
    ]
    return _apply_source_edits(block, relative)


def _apply_source_edits(source: str, edits: list[SourceEdit]) -> str:
    """Apply non-overlapping source ranges without touching any other byte."""

    ordered = sorted(edits, key=lambda edit: edit.target.start)
    cursor = 0
    parts: list[str] = []
    for edit in ordered:
        if (
            edit.target.start < cursor
            or edit.target.start < 0
            or edit.target.end < edit.target.start
            or edit.target.end > len(source)
        ):
            raise TailoringError(
                f"Authorized {edit.category} edit ranges overlap or are invalid."
            )
        parts.extend((source[cursor : edit.target.start], edit.replacement))
        cursor = edit.target.end
    parts.append(source[cursor:])
    return "".join(parts)


def _select_experience_bullets(
    structure: ResumeStructure,
    inp: TailorResumeInput,
    evidence: dict[str, EvidenceItem],
) -> list[tuple[LatexItem, EvidenceItem]]:
    """Choose two structurally valid bullets with matching resume evidence."""

    experience_evidence = [
        item
        for item in evidence.values()
        if item.source == "resume"
        and "experience" in {tag.casefold() for tag in item.tags}
    ]
    if len(experience_evidence) < 2:
        raise TailoringError(
            "At least two resume experience evidence records are required to "
            "rewrite exactly two existing bullets."
        )
    job_skills = [
        claim.claim.split(":", 1)[0].strip()
        for claim in [
            *inp.fit_analysis.aligned_skills,
            *inp.fit_analysis.evidenced_missing_skills,
        ]
    ]
    ranked: list[tuple[float, int, LatexItem, EvidenceItem]] = []
    for bullet in structure.experience_bullets:
        text = bullet.content(structure.source)
        best: tuple[float, EvidenceItem] | None = None
        for item in experience_evidence:
            overlap = _token_overlap(text, item.text)
            statement_match = evidence_supports_statement(text, item)
            if overlap < 0.22 and not statement_match:
                continue
            relevance = sum(
                evidence_supports_skill(item, skill)
                or skill_in_text(canonicalize(skill), text)
                for skill in job_skills
            )
            score = overlap * 10 + relevance * 2 + (3 if statement_match else 0)
            if best is None or score > best[0]:
                best = (score, item)
        if best is not None:
            ranked.append((best[0], -bullet.ordinal, bullet, best[1]))
    ranked.sort(key=lambda value: (value[0], value[1]), reverse=True)

    selected: list[tuple[LatexItem, EvidenceItem]] = []
    used_evidence: set[str] = set()
    for _, _, bullet, item in ranked:
        if item.evidence_id in used_evidence:
            alternative = max(
                (
                    candidate
                    for candidate in experience_evidence
                    if candidate.evidence_id not in used_evidence
                    and (
                        _token_overlap(bullet.content(structure.source), candidate.text)
                        >= 0.22
                        or evidence_supports_statement(
                            bullet.content(structure.source), candidate
                        )
                    )
                ),
                key=lambda candidate: _token_overlap(
                    bullet.content(structure.source), candidate.text
                ),
                default=None,
            )
            if alternative is None:
                continue
            item = alternative
        selected.append((bullet, item))
        used_evidence.add(item.evidence_id)
        if len(selected) == 2:
            break
    if len(selected) != 2:
        raise TailoringError(
            "Could not safely match two distinct existing experience bullets to "
            "their resume evidence."
        )
    return sorted(selected, key=lambda value: value[0].ordinal)


def _token_overlap(left: str, right: str) -> float:
    left_tokens = {
        token for token in _word_tokens(_latex_to_plain(left)) if len(token) > 2
    }
    right_tokens = {
        token for token in _word_tokens(_latex_to_plain(right)) if len(token) > 2
    }
    if not left_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens)


def _revise_edited_content(
    working: str,
    original: str,
    selected_ordinals: list[int],
    attempt: int,
) -> str:
    """Shorten only newly edited prose after a multi-page compilation."""

    current = parse_resume_structure(working)
    baseline = parse_resume_structure(original)
    limit = 150 if attempt == 1 else 105
    edits: list[SourceEdit] = []
    summary = current.summary_range.text(working)
    compact_summary = _compact_prose(summary, limit + 30)
    if compact_summary != summary:
        edits.append(
            SourceEdit(
                current.summary_range,
                compact_summary,
                "summary",
                "Summary overflow revision",
            )
        )
    for ordinal in selected_ordinals:
        current_item = current.experience_bullets[ordinal - 1]
        original_item = baseline.experience_bullets[ordinal - 1]
        current_text = current_item.content(working)
        original_text = original_item.content(original)
        skill_match = re.match(
            r"^(?:Applied|Demonstrated)\s+(.+?)\s+through work that\s+",
            current_text,
        )
        label = skill_match.group(1) if skill_match else "Relevant experience"
        factual = re.split(r";", original_text, maxsplit=1)[0].strip()
        replacement = f"{label}: {_lower_first(factual)}"
        replacement = _compact_prose(replacement, limit)
        if replacement == original_text.strip():
            replacement = f"Relevant: {_lower_first(replacement)}"
        edits.append(
            SourceEdit(
                current_item.content_range,
                replacement,
                "experience",
                f"Experience bullet {ordinal} overflow revision",
            )
        )
    return _apply_source_edits(working, edits)


def _compact_prose(value: str, limit: int) -> str:
    compact = " ".join(value.split())
    sentences = re.split(r"(?<=[.!?])\s+", compact)
    if len(sentences) > 1:
        compact = sentences[0]
    if len(compact) > limit:
        compact = compact[:limit].rsplit(" ", 1)[0].rstrip(" ,;:")
        if value.rstrip().endswith("."):
            compact += "."
    return compact


def _refresh_change_log_after_text(
    changes: list[ChangeLogEntry],
    final: str,
    selected_ordinals: list[int],
) -> list[ChangeLogEntry]:
    """Ensure the log describes the source that actually compiled."""

    final_structure = parse_resume_structure(final)
    experience_index = 0
    refreshed: list[ChangeLogEntry] = []
    for change in changes:
        after_text = change.after_text
        if change.section == "summary":
            after_text = final_structure.summary_range.text(final)
        elif change.section == "experience":
            ordinal = selected_ordinals[experience_index]
            after_text = final_structure.experience_bullets[ordinal - 1].content(final)
            experience_index += 1
        refreshed.append(change.model_copy(update={"after_text": after_text}))
    return refreshed


def _portfolio_record(item: EvidenceItem, expected_name: str) -> PortfolioRecord:
    name = _labeled_value(item.text, "PROJECT_NAME") or ""
    if name.casefold() != expected_name.casefold():
        raise TailoringError(
            f'Portfolio evidence {item.evidence_id} does not describe "{expected_name}".'
        )
    technologies = _split_labeled_list(_labeled_value(item.text, "TECH_STACK"))
    summary = _labeled_value(item.text, "SUMMARY") or ""
    if not technologies or not summary:
        raise TailoringError(
            f"Portfolio evidence {item.evidence_id} lacks a tech stack or summary."
        )
    return PortfolioRecord(
        project_id=_labeled_value(item.text, "PROJECT_ID")
        or str(item.metadata.get("project_id") or item.evidence_id),
        name=name,
        period=_labeled_value(item.text, "PERIOD") or "Date not specified",
        technologies=technologies,
        domain=(_labeled_value(item.text, "DOMAIN") or "Portfolio Project").split(";")[
            0
        ],
        summary=summary,
        evidence_id=item.evidence_id,
    )


def _assert_exactly_two_experience_bullets_changed(before: str, after: str) -> None:
    before_items = parse_resume_structure(before).experience_bullets
    after_items = parse_resume_structure(after).experience_bullets
    if len(before_items) != len(after_items):
        raise TailoringError(
            "Tailoring added or removed an experience bullet; bullet count must "
            "remain unchanged."
        )
    changed = sum(
        old.content(before) != new.content(after)
        for old, new in zip(before_items, after_items, strict=True)
    )
    if changed != 2:
        raise TailoringError(
            "Tailoring must replace exactly two existing experience bullets; "
            f"changed {changed}."
        )


def _assert_only_allowed_modifications(
    before: str, after: str, *, edits: list[SourceEdit]
) -> None:
    """Prove byte-for-byte that all source differences came from planned ranges."""

    if _apply_source_edits(before, edits) != after:
        raise TailoringError(
            "Resume changed outside summary, exactly two experience bullets, "
            "evidenced skills, or an approved project swap."
        )


def _validate_change_log(
    changes: list[ChangeLogEntry],
    evidence: dict[str, EvidenceItem],
    inp: TailorResumeInput,
) -> None:
    """Reject incomplete or ungrounded artifact changes before compilation."""

    allowed_sections = {"summary", "experience", "skills", "projects"}
    candidate_sources = {"resume", "portfolio", "master_skills", "memory"}
    job_sources = {"job_posting", "company_details"}
    for change in changes:
        if change.section not in allowed_sections:
            raise TailoringError(
                f"Unauthorized resume change section: {change.section!r}."
            )
        if change.before_text.strip() == change.after_text.strip():
            raise TailoringError(
                f"Change {change.change_id!r} does not contain a real before/after edit."
            )
        if not change.reason.strip():
            raise TailoringError(f"Change {change.change_id!r} has no reason.")
        unknown = [item for item in change.evidence_ids if item not in evidence]
        if unknown:
            raise TailoringError(
                f"Change {change.change_id!r} cites unknown evidence: {unknown}."
            )
        sources = {evidence[item].source for item in change.evidence_ids}
        if not sources & candidate_sources:
            raise TailoringError(
                f"Change {change.change_id!r} lacks candidate evidence."
            )
        if not sources & job_sources:
            raise TailoringError(
                f"Change {change.change_id!r} lacks job-posting evidence."
            )
        if change.section == "projects" and "portfolio" not in sources:
            raise TailoringError(
                f"Project change {change.change_id!r} lacks portfolio evidence."
            )
        candidate_items = [
            evidence[evidence_id]
            for evidence_id in change.evidence_ids
            if evidence[evidence_id].source in candidate_sources
        ]
        job_items = [
            evidence[evidence_id]
            for evidence_id in change.evidence_ids
            if evidence[evidence_id].source in job_sources
        ]
        if change.section == "experience":
            supporting_bullets = [
                item
                for item in candidate_items
                if item.source == "resume"
                and "experience" in {tag.casefold() for tag in item.tags}
                and evidence_supports_statement(change.before_text, item)
            ]
            if not supporting_bullets:
                raise TailoringError(
                    f"Experience change {change.change_id!r} is not supported by "
                    "the cited resume statement."
                )
            unsupported = _unsupported_introduced_experience_terms(
                change.before_text,
                change.after_text,
                supporting_bullets,
                inp,
                evidence,
            )
            if unsupported:
                raise TailoringError(
                    f"Experience change {change.change_id!r} introduces unsupported "
                    f"wording for its specific bullet: {unsupported}."
                )
        elif change.section == "skills":
            for skill in _skills_named_by_change(change):
                if not any(
                    evidence_supports_skill(item, skill) for item in candidate_items
                ):
                    raise TailoringError(
                        f"Skill change {change.change_id!r} cites no evidence for "
                        f"{skill!r}."
                    )
                if not any(
                    job_evidence_supports_skill(item, skill) for item in job_items
                ):
                    raise TailoringError(
                        f"Skill change {change.change_id!r} adds {skill!r}, which is "
                        "not supported by the cited job requirement."
                    )
        elif change.section == "projects":
            added_name_match = re.search(r'with "([^"]+)"', change.description)
            added_name = (
                added_name_match.group(1) if added_name_match else change.after_text
            )
            if not any(
                evidence_supports_project(item, added_name) for item in candidate_items
            ):
                raise TailoringError(
                    f"Project change {change.change_id!r} does not cite the exact "
                    f"portfolio project {added_name!r}."
                )
        elif change.section == "summary":
            claimed_skills = [
                claim.claim.split(":", 1)[0].strip()
                for claim in [
                    *inp.fit_analysis.aligned_skills,
                    *inp.fit_analysis.evidenced_missing_skills,
                ]
                if skill_in_text(
                    canonicalize(claim.claim.split(":", 1)[0]),
                    change.after_text,
                )
            ]
            for skill in claimed_skills:
                if not any(
                    evidence_supports_skill(item, skill) for item in candidate_items
                ):
                    raise TailoringError(
                        f"Summary change {change.change_id!r} cites no semantic "
                        f"support for {skill!r}."
                    )


def _unsupported_introduced_experience_terms(
    before_text: str,
    after_text: str,
    bullet_evidence: list[EvidenceItem],
    inp: TailorResumeInput,
    all_evidence: dict[str, EvidenceItem],
) -> list[str]:
    """Return new factual wording not supported by the cited resume bullet.

    The comparison validates the complete proposed bullet, while allowing the
    small set of connective words used by the deterministic rewrite template.
    Known skill aliases are checked as phrases first so harmless normalization
    such as ``k8s`` -> ``Kubernetes`` remains valid.
    """

    before = _latex_to_plain(before_text).casefold()
    after = _latex_to_plain(after_text).casefold()
    known_skills = list(
        dict.fromkeys(
            [
                *inp.job.required_skills,
                *[
                    claim.claim.split(":", 1)[0].strip()
                    for claim in [
                        *inp.fit_analysis.aligned_skills,
                        *inp.fit_analysis.evidenced_missing_skills,
                        *inp.fit_analysis.genuine_gaps,
                    ]
                ],
            ]
        )
    )
    unsupported: list[str] = []
    covered_tokens: set[str] = set()
    for skill in known_skills:
        canonical = canonicalize(skill)
        if not canonical or not skill_in_text(canonical, after):
            continue
        if skill_in_text(canonical, before):
            covered_tokens.update(_word_tokens(skill))
            continue
        if any(evidence_supports_skill(item, skill) for item in bullet_evidence):
            covered_tokens.update(_word_tokens(skill))
        else:
            unsupported.append(skill)

    project_names = list(
        dict.fromkeys(
            name
            for item in all_evidence.values()
            if item.source == "portfolio"
            for name in [
                str(item.metadata.get("project_name") or "").strip(),
                (_labeled_value(item.text, "PROJECT_NAME") or "").strip(),
            ]
            if name
        )
    )
    for project_name in project_names:
        normalized = project_name.casefold()
        if normalized in after and normalized not in before:
            # Portfolio evidence may prove a project exists, but it cannot turn
            # an unrelated resume experience bullet into project experience.
            unsupported.append(project_name)

    generic = {
        "and",
        "applied",
        "delivered",
        "demonstrated",
        "experience",
        "highlighted",
        "relevant",
        "that",
        "through",
        "work",
    }
    before_tokens = _word_tokens(before)
    for token in sorted(_word_tokens(after) - before_tokens):
        if len(token) < 3 or token in generic or token in covered_tokens:
            continue
        if not any(evidence_supports_keyword(item, token) for item in bullet_evidence):
            unsupported.append(token)
    return list(dict.fromkeys(unsupported))


def _word_tokens(value: str) -> set[str]:
    return {
        token.strip("./-")
        for token in re.findall(r"[a-z0-9][a-z0-9+#./-]*", value.casefold())
        if token.strip("./-")
    }


def _check_revision_feedback(
    feedback: str | None,
    changes: list[ChangeLogEntry],
    evidence: dict[str, EvidenceItem],
) -> tuple[bool | None, list[str]]:
    """Return deterministic checks used to decide whether another round is needed."""

    if not feedback:
        return None, []
    checks: list[str] = []
    satisfied = True
    if not changes:
        return False, ["revision made no evidence-backed content changes"]
    if re.search(r"\b(shorter|concise|trim|brief)\b", feedback, re.IGNORECASE):
        experience = [change for change in changes if change.section == "experience"]
        concise = bool(experience) and sum(
            len(change.after_text) for change in experience
        ) < sum(len(change.before_text) for change in experience)
        checks.append(f"conciseness requested: {'met' if concise else 'not met'}")
        satisfied = satisfied and concise
    if re.search(r"\b(project|swap)\b", feedback, re.IGNORECASE):
        project_changed = any(change.section == "projects" for change in changes)
        checks.append(
            f"project change requested: {'met' if project_changed else 'not met'}"
        )
        satisfied = satisfied and project_changed
    requested_skills = _feedback_skill_requests(feedback, evidence)
    for skill in requested_skills:
        relevant_changes = [
            change
            for change in changes
            if evidence_supports_keyword(
                EvidenceItem(
                    evidence_id="revision-after",
                    source="resume",
                    text=change.after_text,
                ),
                skill,
            )
            and not evidence_supports_keyword(
                EvidenceItem(
                    evidence_id="revision-before",
                    source="resume",
                    text=change.before_text,
                ),
                skill,
            )
        ]
        grounded = any(
            any(
                evidence_supports_skill(evidence[evidence_id], skill)
                for evidence_id in change.evidence_ids
                if evidence_id in evidence
                and evidence[evidence_id].source in CANDIDATE_SOURCES
            )
            and any(
                job_evidence_supports_skill(evidence[evidence_id], skill)
                for evidence_id in change.evidence_ids
                if evidence_id in evidence
                and evidence[evidence_id].source in JOB_SOURCES
            )
            for change in relevant_changes
        )
        met = bool(relevant_changes) and grounded
        checks.append(
            f"requested skill/keyword {skill!r}: {'met' if met else 'not met'}"
        )
        satisfied = satisfied and met

    section_requests = {
        "summary": r"\bsummary\b",
        "experience": r"\b(?:experience|bullet)\b",
        "skills": r"\bskills?\b",
    }
    for section, pattern in section_requests.items():
        if re.search(pattern, feedback, re.IGNORECASE):
            met = any(change.section == section for change in changes)
            checks.append(f"{section} change requested: {'met' if met else 'not met'}")
            satisfied = satisfied and met

    if not checks:
        introduced = _meaningful_feedback_terms(feedback)
        changed_text = " ".join(
            f"{change.description} {change.after_text} {change.reason}"
            for change in changes
        ).casefold()
        matched = sorted(
            term
            for term in introduced
            if re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", changed_text)
        )
        met = bool(introduced) and bool(matched)
        checks.append(
            "feedback-specific content overlap: "
            + ("met (" + ", ".join(matched) + ")" if met else "not met")
        )
        satisfied = satisfied and met
    return satisfied, checks


def _validate_fit_analysis_evidence(
    inp: TailorResumeInput, evidence: dict[str, EvidenceItem]
) -> list[str]:
    """Reject a direct tailoring call that bypasses fit post-validation."""

    failures: list[str] = []
    required = inp.job.required_skills
    for bucket, claims, require_resume in (
        ("aligned", inp.fit_analysis.aligned_skills, True),
        ("evidenced_missing", inp.fit_analysis.evidenced_missing_skills, False),
    ):
        for claim in claims:
            skill = claim.claim.split(":", 1)[0].strip()
            if not any(_job_skill_matches(skill, value) for value in required):
                failures.append(
                    f"{bucket} claim {skill!r} is not an exact/aliased job requirement"
                )
                continue
            candidates = [
                evidence[evidence_id]
                for evidence_id in claim.evidence_ids
                if evidence_id in evidence
                and evidence[evidence_id].source in CANDIDATE_SOURCES
                and evidence_supports_skill(evidence[evidence_id], skill)
            ]
            if not candidates:
                failures.append(
                    f"{bucket} claim {skill!r} has no semantically matching "
                    "candidate citation"
                )
            elif require_resume and not any(
                item.source == "resume" for item in candidates
            ):
                failures.append(
                    f"aligned claim {skill!r} has no matching resume citation"
                )

    swap = inp.fit_analysis.project_swap
    if swap is not None:
        matching = [
            evidence[evidence_id]
            for evidence_id in swap.evidence_ids
            if evidence_id in evidence
            and evidence_supports_project(evidence[evidence_id], swap.add_project)
        ]
        if not matching:
            failures.append(
                f"project addition {swap.add_project!r} has no exact portfolio citation"
            )
    return failures


def _job_skill_matches(skill: str, required: str) -> bool:
    canonical = canonicalize(skill)
    required_canonical = canonicalize(required)
    return (
        canonical == required_canonical
        or canonical
        in {canonicalize(member) for member in category_members(required_canonical)}
        or skill_in_text(canonical, required)
        or evidence_supports_skill(
            EvidenceItem(
                evidence_id="job-requirement",
                source="job_posting",
                text=f"Required skill: {required}",
                tags=[required],
            ),
            skill,
        )
    )


def _skills_named_by_change(change: ChangeLogEntry) -> list[str]:
    plural_prefix = "Added evidenced skills:"
    singular_prefix = "Added evidenced skill:"
    description = change.description.casefold()
    if (
        plural_prefix.casefold() not in description
        and singular_prefix.casefold() not in description
    ):
        return []
    _, _, values = change.description.partition(":")
    return [
        value.strip().rstrip(".")
        for value in values.split(",")
        if value.strip().rstrip(".")
    ]


def _feedback_skill_requests(
    feedback: str, evidence: dict[str, EvidenceItem]
) -> list[str]:
    requested: list[str] = []
    for item in evidence.values():
        if item.source != "memory":
            continue
        value = item.text.partition(":")[2].strip()
        if value and (
            item.evidence_id in feedback or value.casefold() in feedback.casefold()
        ):
            requested.append(value)
    for match in re.finditer(
        r"\b(?:add|include|highlight|mention|surface)\s+"
        r"(?P<value>[A-Za-z0-9+#/-]+(?:\s+[A-Za-z0-9+#/-]+){0,3})",
        feedback,
        re.IGNORECASE,
    ):
        value = re.split(
            r"\b(?:to|in|on|because|using|when|where)\b",
            match.group("value"),
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0].strip(" .,:;")
        if value:
            requested.append(value)
    return list(dict.fromkeys(requested))


def _meaningful_feedback_terms(feedback: str) -> set[str]:
    stop = {
        "change",
        "feedback",
        "make",
        "more",
        "please",
        "resume",
        "revision",
        "this",
        "that",
        "the",
        "with",
    }
    return {
        token
        for token in re.findall(r"[a-z0-9+#./-]+", feedback.casefold())
        if len(token) >= 4 and token not in stop
    }


def _balanced_brace_content(source: str, opening: int) -> tuple[str, int]:
    try:
        return balanced_brace_content(source, opening)
    except LatexStructureError as exc:
        raise TailoringError(str(exc)) from exc


def _compile_one_page(
    source: str,
    tex_path: Path,
    pdf_path: Path,
    *,
    tracer: TraceManager | None = None,
    trace_metadata: dict[str, object] | None = None,
    revision: Callable[[str, int], str] | None = None,
) -> tuple[int | None, list[str], str]:
    active = tracer or TraceManager(enabled=False)
    metadata = dict(trace_metadata or {})
    errors: list[str] = []
    working = source
    page_count: int | None = None
    for attempt in range(MAX_COMPILE_ATTEMPTS):
        tex_path.write_text(working, encoding="utf-8")
        compile_span = active.start_span(
            "resume_tailoring.compile_pdf",
            {**metadata, "attempt": attempt + 1},
            input={
                "engine": "pdflatex",
                "command": pdflatex_command(tex_path),
                "tex_file": tex_path.name,
                "source_length": len(working),
            },
        )
        run_errors = _run_pdflatex(tex_path)
        if run_errors:
            active.end_span(
                compile_span,
                status="ERROR",
                error_type="LatexCompilationError",
                output={
                    "result": "error",
                    "errors": run_errors,
                    "pdf_created": pdf_path.is_file(),
                },
            )
            return None, run_errors, working
        pdf_created = pdf_path.is_file()
        active.end_span(
            compile_span,
            status="OK" if pdf_created else "ERROR",
            error_type=None if pdf_created else "MissingPdf",
            output={
                "result": "success" if pdf_created else "missing_pdf",
                "pdf_created": pdf_created,
                "pdf_file": pdf_path.name,
            },
        )
        if not pdf_created:
            return (
                None,
                ["pdflatex completed without producing the expected PDF."],
                working,
            )
        verify_span = active.start_span(
            "resume_tailoring.validate_page_count",
            {**metadata, "attempt": attempt + 1},
            input={"pdf_file": pdf_path.name, "required_page_count": 1},
        )
        try:
            page_count = pdf_page_count(pdf_path)
        except Exception as exc:  # noqa: BLE001 - report invalid generated PDFs
            active.end_span(
                verify_span,
                status="ERROR",
                error_type=exc.__class__.__name__,
                output={"valid_pdf": False},
            )
            return None, [f"Generated resume PDF is invalid: {exc}"], working
        active.end_span(
            verify_span,
            status="OK" if page_count == 1 else "ERROR",
            error_type=None if page_count == 1 else "PageCountMismatch",
            output={
                "valid_pdf": True,
                "page_count": page_count,
                "exactly_one_page": page_count == 1,
            },
        )
        if page_count == 1:
            return 1, [], working
        errors.append(
            f"Resume compiled to {page_count} pages; exactly one is required."
        )
        if revision is not None and attempt + 1 < MAX_COMPILE_ATTEMPTS:
            revised = revision(working, attempt + 1)
            if revised == working:
                break
            working = revised
    errors.append(
        "A compliant one-page resume could not be produced within the configured "
        "revision limit. Only newly edited content was shortened; layout, margins, "
        "font size, spacing, and unrelated source were preserved."
    )
    return page_count, errors, working


def _run_pdflatex(tex_path: Path) -> list[str]:
    return run_pdflatex(tex_path, artifact_label="resume")


def _labeled_value(text: str, label: str) -> str | None:
    match = re.search(rf"(?m)^{re.escape(label)}:\s*(.+?)\s*$", text)
    return match.group(1).strip() if match else None


def _split_labeled_list(value: str | None) -> list[str]:
    return [item.strip() for item in re.split(r"[;,]", value or "") if item.strip()]


def _join_words(values: list[str]) -> str:
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return ", ".join(values[:-1]) + f", and {values[-1]}"


def _latex_to_plain(text: str) -> str:
    plain = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?", " ", text)
    plain = plain.replace(r"\&", "&").replace(r"\%", "%").replace(r"\_", "_")
    plain = re.sub(r"[{}]", " ", plain)
    return re.sub(r"\s+", " ", plain)


def _lower_first(text: str) -> str:
    return text[:1].lower() + text[1:] if text else text
