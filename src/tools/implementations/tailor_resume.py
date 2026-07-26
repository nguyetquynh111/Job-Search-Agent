"""Implementation of the ``tailor_resume`` model-visible tool.

The repository resume template is the contract: this tool edits only the
explicit AGENT markers in the LaTeX source, compiles the result, and reports a
before/after change log with evidence IDs.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from src.config import get_config
from src.schemas.common import ChangeLogEntry, EvidenceClaim, EvidenceItem
from src.schemas.tailoring import TailorResumeInput, TailorResumeOutput

COMPILE_TIMEOUT_SECONDS = 60
MAX_SUMMARY_WORDS = 37
MAX_BULLET_WORDS = 24
MAX_SKILLS_TO_SURFACE = 4

_LATEX_SPECIAL_CHARS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


@dataclass(frozen=True)
class ProjectEvidence:
    project_id: str
    name: str
    period: str
    domain: str
    technologies: list[str]
    summary: str
    contribution: str
    evidence_id: str


@dataclass(frozen=True)
class SupportedSkill:
    display: str
    evidence_id: str


def tailor_resume(inp: TailorResumeInput) -> TailorResumeOutput:
    """Tailor one LaTeX resume for one job while preserving the base source."""

    errors: list[str] = []
    if inp.fit_analysis.job_id != inp.job.job_id:
        errors.append(
            f"Fit analysis job_id {inp.fit_analysis.job_id} does not match job {inp.job.job_id}."
        )

    source_path = Path(inp.source_resume_tex_path)
    if not source_path.exists():
        return TailorResumeOutput(
            job_id=inp.job.job_id,
            status="ERROR",
            output_tex_path="",
            output_pdf_path="",
            page_count=1,
            change_log=[],
            errors=[f"Source resume not found: {source_path}"],
        )

    output_dir = get_config().output_dir / inp.job.job_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_tex_path = output_dir / "resume.tex"
    output_pdf_path = output_dir / "resume.pdf"

    source_tex = source_path.read_text(encoding="utf-8")
    evidence_pool = {item.evidence_id: item for item in inp.candidate_evidence}
    portfolio_projects = _portfolio_projects(evidence_pool.values())

    tailored_tex = source_tex
    change_log: list[ChangeLogEntry] = []

    tailored_tex, summary_change = _replace_summary(tailored_tex, inp, evidence_pool)
    change_log.append(summary_change)

    bullet_changes: list[ChangeLogEntry] = []
    for marker, bullet_number in (
        ("experience-bullet-1", 1),
        ("experience-bullet-2", 2),
    ):
        tailored_tex, change = _replace_experience_bullet(
            tailored_tex, marker, bullet_number, inp, evidence_pool
        )
        bullet_changes.append(change)
    change_log.extend(bullet_changes)

    tailored_tex, skill_changes = _surface_skills(tailored_tex, inp, evidence_pool)
    change_log.extend(skill_changes)

    tailored_tex, swap_change = _maybe_swap_project(
        tailored_tex, inp, portfolio_projects
    )
    if swap_change:
        change_log.append(swap_change)

    output_tex_path.write_text(tailored_tex, encoding="utf-8")

    page_count, compile_errors = _compile_and_count_pages(output_tex_path, output_pdf_path)
    errors.extend(compile_errors)

    if page_count and page_count > 1:
        tailored_tex = _shorten_for_one_page(tailored_tex)
        output_tex_path.write_text(tailored_tex, encoding="utf-8")
        page_count, retry_errors = _compile_and_count_pages(
            output_tex_path, output_pdf_path
        )
        errors.extend(retry_errors)

    if page_count != 1:
        errors.append(
            f"One-page rule not satisfied for {inp.job.job_id}; detected page_count={page_count or 'unknown'}."
        )

    return TailorResumeOutput(
        job_id=inp.job.job_id,
        status="OK" if page_count == 1 and not errors else "ERROR",
        output_tex_path=str(output_tex_path),
        output_pdf_path=str(output_pdf_path),
        page_count=page_count or 1,
        change_log=change_log,
        errors=errors,
    )


def _replace_summary(
    tex: str,
    inp: TailorResumeInput,
    evidence_pool: dict[str, EvidenceItem],
) -> tuple[str, ChangeLogEntry]:
    old_summary = _line_after_marker(tex, "summary")
    skills = _supported_skills(inp, evidence_pool, limit=3)
    metrics = _best_metrics(evidence_pool.values(), limit=2)
    skill_text = _join([skill.display for skill in skills])
    summary_bits = [
        "AI/ML engineer with 4+ years building production systems",
        f"for {inp.job.industry_domain}" if inp.job.industry_domain else "",
        f"with {skill_text}" if skill_text else "",
        f"and outcomes including {_join(metrics)}" if metrics else "",
    ]
    new_summary = _limit_words(" ".join(bit for bit in summary_bits if bit), MAX_SUMMARY_WORDS)
    new_summary = _ensure_sentence(new_summary)
    tex = _replace_line_after_marker(tex, "summary", new_summary)
    evidence_ids = _claim_evidence_ids(
        [
            *inp.fit_analysis.aligned_skills,
            *inp.fit_analysis.evidenced_missing_skills,
            *inp.fit_analysis.relevant_experience,
        ],
        evidence_pool,
    )[:5]
    return tex, ChangeLogEntry(
        change_id=f"{inp.job.job_id}-summary",
        section="professional_summary",
        description=(
            f"Before: {old_summary} | After: {new_summary} | Reason: align the "
            f"summary to {inp.job.title} at {inp.job.company} using evidenced skills."
        ),
        evidence_ids=evidence_ids,
    )


def _replace_experience_bullet(
    tex: str,
    marker: str,
    bullet_number: int,
    inp: TailorResumeInput,
    evidence_pool: dict[str, EvidenceItem],
) -> tuple[str, ChangeLogEntry]:
    old_bullet = _resume_item_after_marker(tex, marker)
    skills = _supported_skills(inp, evidence_pool, limit=3)
    skill_text = _join([skill.display for skill in skills])
    if bullet_number == 1:
        new_bullet = (
            "Built large-scale ML evaluation workflows with targeted error analysis"
            + (f" for {skill_text}" if skill_text else "")
            + "; validated model quality across 1.7M prediction records."
        )
    else:
        new_bullet = (
            "Delivered production AI applications"
            + (f" using {skill_text}" if skill_text else "")
            + "; deployed APIs and model workflows with Docker, Kubernetes, and AWS SageMaker."
        )
    new_bullet = _limit_words(new_bullet, MAX_BULLET_WORDS)
    tex = _replace_resume_item_after_marker(tex, marker, new_bullet)
    evidence_ids = _claim_evidence_ids(
        [
            *inp.fit_analysis.relevant_experience,
            *inp.fit_analysis.aligned_skills,
            *inp.fit_analysis.evidenced_missing_skills,
        ],
        evidence_pool,
    )[:5]
    return tex, ChangeLogEntry(
        change_id=f"{inp.job.job_id}-{marker}",
        section="experience",
        description=(
            f"Before: {old_bullet} | After: {new_bullet} | Reason: modify exactly "
            f"experience bullet {bullet_number} to emphasize job-relevant, evidenced work."
        ),
        evidence_ids=evidence_ids,
    )


def _surface_skills(
    tex: str,
    inp: TailorResumeInput,
    evidence_pool: dict[str, EvidenceItem],
) -> tuple[str, list[ChangeLogEntry]]:
    changes: list[ChangeLogEntry] = []
    skills = _supported_skills(inp, evidence_pool, limit=MAX_SKILLS_TO_SURFACE)
    if inp.revision_feedback:
        skills = _prioritize_feedback_skills(skills, inp.revision_feedback)

    for skill in skills:
        if _latex_skill_present(tex, skill.display):
            updated = _bold_first_skill_occurrence(tex, skill.display)
            if updated != tex:
                tex = updated
                changes.append(
                    ChangeLogEntry(
                        change_id=f"{inp.job.job_id}-skill-{_slug(skill.display)}",
                        section="skills",
                        description=(
                            f"Before: {skill.display} | After: \\textbf{{{skill.display}}} | "
                            "Reason: highlight an already-present skill named in the job requirements."
                        ),
                        evidence_ids=[skill.evidence_id],
                    )
                )
        else:
            tex = _append_skill_to_category(tex, skill.display, _skill_category(skill.display))
            changes.append(
                ChangeLogEntry(
                    change_id=f"{inp.job.job_id}-skill-{_slug(skill.display)}",
                    section="skills",
                    description=(
                        f"Before: skill not listed | After: added {skill.display} | "
                        "Reason: job-relevant skill is evidenced in the candidate profile."
                    ),
                    evidence_ids=[skill.evidence_id],
                )
            )
    return tex, changes


def _maybe_swap_project(
    tex: str,
    inp: TailorResumeInput,
    portfolio_projects: dict[str, ProjectEvidence],
) -> tuple[str, ChangeLogEntry | None]:
    swap = inp.fit_analysis.project_swap
    if not swap:
        return tex, None
    add_project = _find_project(swap.add_project, portfolio_projects)
    if not add_project:
        return tex, ChangeLogEntry(
            change_id=f"{inp.job.job_id}-project-swap-not-applied",
            section="projects",
            description=(
                f"Before: current projects unchanged | After: no swap applied | "
                f"Reason: recommended project '{swap.add_project}' was not found in portfolio evidence."
            ),
            evidence_ids=swap.evidence_ids,
        )
    if _project_already_present(tex, add_project):
        return tex, None

    target_block, remove_name = _project_block_to_replace(tex, swap.remove_project)
    if not target_block:
        return tex, ChangeLogEntry(
            change_id=f"{inp.job.job_id}-project-swap-not-applied",
            section="projects",
            description=(
                "Before: current projects unchanged | After: no swap applied | "
                "Reason: no marked project block was available for replacement."
            ),
            evidence_ids=[add_project.evidence_id, *swap.evidence_ids],
        )

    replacement = _render_project_block(add_project)
    tex = tex.replace(target_block, replacement, 1)
    return tex, ChangeLogEntry(
        change_id=f"{inp.job.job_id}-project-swap",
        section="projects",
        description=(
            f"Before: {remove_name or 'marked resume project'} | After: {add_project.name} | "
            f"Reason: {swap.rationale}"
        ),
        evidence_ids=_dedupe([add_project.evidence_id, *swap.evidence_ids]),
    )


def _compile_and_count_pages(tex_path: Path, pdf_path: Path) -> tuple[int | None, list[str]]:
    # tectonic is preferred: it is a single self-contained binary (no system
    # LaTeX distribution required) and fetches any packages it needs, so it
    # works the same way across every teammate's OS. pdflatex is the fallback
    # for machines with a full local TeX Live/MacTeX/MiKTeX install.
    tectonic = shutil.which("tectonic")
    if tectonic:
        return _run_compiler(
            [tectonic, tex_path.name],
            engine="tectonic",
            tex_path=tex_path,
            pdf_path=pdf_path,
        )

    pdflatex = shutil.which("pdflatex")
    if pdflatex:
        return _run_compiler(
            [pdflatex, "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
            engine="pdflatex",
            tex_path=tex_path,
            pdf_path=pdf_path,
        )

    return None, [
        "Neither tectonic nor pdflatex is installed or on PATH; resume PDF was not compiled."
    ]


def _run_compiler(
    command: list[str], engine: str, tex_path: Path, pdf_path: Path
) -> tuple[int | None, list[str]]:
    try:
        completed = subprocess.run(
            command,
            cwd=str(tex_path.parent),
            check=False,
            capture_output=True,
            text=True,
            timeout=COMPILE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return None, [f"{engine} timed out after {COMPILE_TIMEOUT_SECONDS} seconds."]

    if completed.returncode != 0:
        tail = "\n".join((completed.stdout + completed.stderr).splitlines()[-12:])
        return None, [f"{engine} failed for {tex_path}: {tail}"]

    generated_pdf = tex_path.with_suffix(".pdf")
    if generated_pdf != pdf_path and generated_pdf.exists():
        generated_pdf.replace(pdf_path)
    if not pdf_path.exists():
        return None, [f"Expected PDF was not created: {pdf_path}"]

    return _count_pdf_pages(pdf_path)


def _count_pdf_pages(pdf_path: Path) -> tuple[int | None, list[str]]:
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(pdf_path))
        return len(reader.pages), []
    except ModuleNotFoundError:
        pdfinfo = shutil.which("pdfinfo")
        if not pdfinfo:
            return None, ["Neither pypdf nor pdfinfo is available for PDF page counting."]
        completed = subprocess.run(
            [pdfinfo, str(pdf_path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=COMPILE_TIMEOUT_SECONDS,
        )
        if completed.returncode != 0:
            return None, [f"pdfinfo failed for {pdf_path}: {completed.stderr.strip()}"]
        match = re.search(r"(?m)^Pages:\s+(\d+)\s*$", completed.stdout)
        if not match:
            return None, [f"Could not find page count in pdfinfo output for {pdf_path}."]
        return int(match.group(1)), []
    except Exception as exc:  # noqa: BLE001 - report as tool output, do not crash graph
        return None, [f"Could not read compiled PDF page count: {exc}"]


def _shorten_for_one_page(tex: str) -> str:
    tex = re.sub(r"\\documentclass\[letterpaper,11pt\]", r"\\documentclass[letterpaper,10pt]", tex)
    tex = re.sub(r"\\titlespacing\*\{\\section\}\{0pt\}\{8pt\}\{4pt\}", r"\\titlespacing*{\\section}{0pt}{6pt}{3pt}", tex)
    return tex


def _supported_skills(
    inp: TailorResumeInput,
    evidence_pool: dict[str, EvidenceItem],
    limit: int,
) -> list[SupportedSkill]:
    requested = _skills_from_claims(
        [*inp.fit_analysis.aligned_skills, *inp.fit_analysis.evidenced_missing_skills],
        inp.job.required_skills,
    )
    if inp.revision_feedback:
        requested = [
            *_skills_from_feedback(inp.revision_feedback, inp.job.required_skills),
            *requested,
        ]
    supported: list[SupportedSkill] = []
    seen: set[str] = set()
    for skill in requested:
        evidence = _best_skill_evidence(skill, evidence_pool.values())
        if not evidence:
            continue
        key = _skill_key(skill)
        if key in seen:
            continue
        seen.add(key)
        supported.append(SupportedSkill(display=skill, evidence_id=evidence.evidence_id))
        if len(supported) >= limit:
            break
    return supported


def _skills_from_claims(claims: list[EvidenceClaim], required_skills: list[str]) -> list[str]:
    found: list[str] = []
    for skill in required_skills:
        needle = skill.casefold()
        if any(needle and needle in claim.claim.casefold() for claim in claims):
            found.append(skill)
    if found:
        return found
    for claim in claims:
        for token in re.findall(r"[A-Za-z][A-Za-z0-9+.#/-]*(?:\s+[A-Za-z][A-Za-z0-9+.#/-]*)?", claim.claim):
            if len(token) > 2 and token[:1].isupper():
                found.append(token)
    return _dedupe(found)


def _skills_from_feedback(feedback: str, required_skills: list[str]) -> list[str]:
    found = [
        skill
        for skill in required_skills
        if re.search(rf"(?i)(?<![A-Za-z0-9]){re.escape(skill)}(?![A-Za-z0-9])", feedback)
    ]
    if found:
        return found
    add_match = re.search(r"(?i)\badd\s+([A-Za-z][A-Za-z0-9+.#/-]*)", feedback)
    return [add_match.group(1)] if add_match else []


def _best_skill_evidence(
    skill: str, evidence_items: list[EvidenceItem]
) -> EvidenceItem | None:
    aliases = _skill_aliases(skill)
    source_priority = {"resume": 0, "master_skills": 1, "portfolio": 2, "memory": 3}
    matches: list[EvidenceItem] = []
    for item in evidence_items:
        text = f"{item.text} {' '.join(item.tags)}".casefold()
        if any(alias in text for alias in aliases):
            matches.append(item)
    if not matches:
        return None
    return sorted(matches, key=lambda item: source_priority.get(item.source, 9))[0]


def _claim_evidence_ids(
    claims: list[EvidenceClaim], evidence_pool: dict[str, EvidenceItem]
) -> list[str]:
    ids = [
        evidence_id
        for claim in claims
        for evidence_id in claim.evidence_ids
        if evidence_id in evidence_pool
    ]
    if ids:
        return _dedupe(ids)
    return list(evidence_pool)[:3]


def _best_metrics(evidence_items: list[EvidenceItem], limit: int) -> list[str]:
    metrics: list[str] = []
    for item in evidence_items:
        for match in re.findall(
            r"(?:\d+(?:\.\d+)?M|\d+(?:\.\d+)?%|0\.\d+\s*F1|80%\s+matching success|95%\s+parser accuracy)",
            item.text,
        ):
            metrics.append(match)
            if len(metrics) >= limit:
                return metrics
    return metrics


def _portfolio_projects(evidence_items: list[EvidenceItem]) -> dict[str, ProjectEvidence]:
    projects: dict[str, ProjectEvidence] = {}
    for item in evidence_items:
        if item.source != "portfolio":
            continue
        project_id = _labeled_value(item.text, "PROJECT_ID") or item.metadata.get("project_id")
        name = _labeled_value(item.text, "PROJECT_NAME")
        if not project_id or not name:
            continue
        contribution = _first_bullet_after_label(item.text, "CORE_CONTRIBUTIONS")
        projects[project_id.casefold()] = ProjectEvidence(
            project_id=project_id,
            name=name,
            period=_labeled_value(item.text, "PERIOD") or "",
            domain=(_labeled_value(item.text, "DOMAIN") or "").split(";")[0].strip(),
            technologies=_split_labeled_list(_labeled_value(item.text, "TECH_STACK")),
            summary=_labeled_value(item.text, "SUMMARY") or "",
            contribution=contribution,
            evidence_id=item.evidence_id,
        )
    return projects


def _find_project(
    requested: str, portfolio_projects: dict[str, ProjectEvidence]
) -> ProjectEvidence | None:
    key = requested.casefold().strip()
    if key in portfolio_projects:
        return portfolio_projects[key]
    for project in portfolio_projects.values():
        if key == project.name.casefold() or key in project.name.casefold():
            return project
    return None


def _project_block_to_replace(tex: str, requested_remove: str | None) -> tuple[str, str]:
    pattern = re.compile(
        r"(?ms)^  % AGENT-SWAP-TARGET: project-\d+ \| PORTFOLIO-ID: (?P<id>\S+)\n"
        r"  \\resumeEntry\{(?P<name>[^{}]+)\}.*?"
        r"  \\resumeItemListEnd\n"
    )
    matches = list(pattern.finditer(tex))
    if not matches:
        return "", ""
    if requested_remove:
        needle = requested_remove.casefold()
        for match in matches:
            if needle in match.group("name").casefold() or needle == match.group("id").casefold():
                return match.group(0), match.group("name")
        return "", ""
    match = matches[0]
    return match.group(0), match.group("name")


def _project_already_present(tex: str, project: ProjectEvidence) -> bool:
    projects_section = tex.partition(r"\section{Projects}")[2].partition(r"\section{Skills}")[0]
    return (
        project.project_id.casefold() in projects_section.casefold()
        or project.name.casefold() in projects_section.casefold()
    )


def _render_project_block(project: ProjectEvidence) -> str:
    technologies = ", ".join(project.technologies[:6])
    bullet = project.contribution or project.summary
    return (
        f"  % AGENT-SWAP-TARGET: project-1 | PORTFOLIO-ID: {project.project_id}\n"
        f"  \\resumeEntry{{{_escape_latex(project.name)}}}{{{_escape_latex(project.period)}}}\n"
        f"    {{{_escape_latex(technologies)}}}{{{_escape_latex(project.domain)}}}\n"
        "  \\resumeItemListStart\n"
        f"    \\resumeItem{{{_escape_latex(_limit_words(bullet, 24))}}}\n"
        "  \\resumeItemListEnd\n"
    )


def _line_after_marker(tex: str, marker: str) -> str:
    match = re.search(
        rf"(?m)^% AGENT-EDIT-TARGET: {re.escape(marker)}\n(?P<line>.+)$", tex
    )
    return match.group("line").strip() if match else ""


def _replace_line_after_marker(tex: str, marker: str, new_line: str) -> str:
    return re.sub(
        rf"(?m)(^% AGENT-EDIT-TARGET: {re.escape(marker)}\n).+$",
        rf"\g<1>{_escape_latex(new_line)}",
        tex,
        count=1,
    )


def _resume_item_after_marker(tex: str, marker: str) -> str:
    match = re.search(
        rf"(?s)% AGENT-EDIT-TARGET: {re.escape(marker)}\s*\\resumeItem\{{(?P<item>.*?)\}}",
        tex,
    )
    return match.group("item").strip() if match else ""


def _replace_resume_item_after_marker(tex: str, marker: str, new_item: str) -> str:
    return re.sub(
        rf"(?s)(% AGENT-EDIT-TARGET: {re.escape(marker)}\s*\\resumeItem\{{)(.*?)(\}})",
        rf"\g<1>{_escape_latex(new_item)}\g<3>",
        tex,
        count=1,
    )


def _append_skill_to_category(tex: str, skill: str, category: str) -> str:
    escaped_skill = _escape_latex(skill)
    pattern = re.compile(
        rf"(\\small\\item\{{\\textbf\{{{re.escape(category)}:\}}\s*)(?P<skills>.*?)(\}})",
        re.DOTALL,
    )
    match = pattern.search(tex)
    if not match:
        return tex
    existing = match.group("skills").rstrip()
    separator = "" if existing.endswith(",") else ","
    replacement = f"{match.group(1)}{existing}{separator} {escaped_skill}{match.group(3)}"
    return tex[: match.start()] + replacement + tex[match.end() :]


def _bold_first_skill_occurrence(tex: str, skill: str) -> str:
    escaped = _escape_latex(skill)
    if rf"\textbf{{{escaped}}}" in tex:
        return tex
    skills_start = tex.find("% AGENT-EDIT-TARGET: skills")
    if skills_start < 0:
        return tex
    pattern = re.compile(rf"(?<![A-Za-z0-9]){re.escape(escaped)}(?![A-Za-z0-9])")
    return tex[:skills_start] + pattern.sub(rf"\\textbf{{{escaped}}}", tex[skills_start:], count=1)


def _latex_skill_present(tex: str, skill: str) -> bool:
    skills_section = tex.partition("% AGENT-EDIT-TARGET: skills")[2]
    return bool(
        re.search(
            rf"(?i)(?<![A-Za-z0-9]){re.escape(_escape_latex(skill))}(?![A-Za-z0-9])",
            skills_section,
        )
    )


def _prioritize_feedback_skills(
    skills: list[SupportedSkill], feedback: str
) -> list[SupportedSkill]:
    lowered = feedback.casefold()
    return sorted(skills, key=lambda skill: 0 if skill.display.casefold() in lowered else 1)


def _skill_category(skill: str) -> str:
    lowered = skill.casefold()
    if any(token in lowered for token in ("sql", "mongo", "elastic", "neo4j", "rust", "java", "graph")):
        return "Data \\& Languages"
    if any(token in lowered for token in ("docker", "kubernetes", "aws", "cloud", "api", "fastapi", "mlflow", "airflow")):
        return "Production"
    return "AI/ML"


def _skill_aliases(skill: str) -> set[str]:
    lowered = skill.casefold()
    aliases = {lowered}
    if lowered == "machine learning":
        aliases.add("ml")
    if lowered == "ml":
        aliases.add("machine learning")
    if lowered == "generative ai":
        aliases.add("llm")
    if lowered == "llm":
        aliases.add("large language model")
    return aliases


def _skill_key(skill: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", skill.casefold())


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-") or "skill"


def _labeled_value(text: str, label: str) -> str | None:
    match = re.search(rf"(?m)^{re.escape(label)}:\s*(.*?)\s*$", text)
    return match.group(1).strip() if match and match.group(1).strip() else None


def _first_bullet_after_label(text: str, label: str) -> str:
    match = re.search(rf"(?ms)^{re.escape(label)}:\s*\n(?P<body>.*?)(?=^[A-Z_]+:|\Z)", text)
    if not match:
        return ""
    bullet = re.search(r"(?m)^-\s*(.+?)\s*$", match.group("body"))
    return bullet.group(1).strip() if bullet else ""


def _split_labeled_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in re.split(r"[;,]", value) if part.strip()]


def _escape_latex(value: str) -> str:
    return "".join(_LATEX_SPECIAL_CHARS.get(character, character) for character in value)


def _limit_words(value: str, max_words: int) -> str:
    words = value.split()
    return " ".join(words[:max_words]).rstrip(",;") if len(words) > max_words else value


def _ensure_sentence(value: str) -> str:
    return value if value.endswith((".", "!", "?")) else value + "."


def _join(items: list[str]) -> str:
    clean = [item for item in items if item]
    if len(clean) <= 1:
        return clean[0] if clean else ""
    return ", ".join(clean[:-1]) + f", and {clean[-1]}"


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        key = item.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result
