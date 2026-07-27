from __future__ import annotations

import inspect
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import Field

from src.config import get_config
from src.domain import (
    EvidenceItem,
    Job,
    StrictBaseModel,
)
from src.tracing.langfuse import TraceManager
from src.utils.evidence_validation import evidence_supports_skill
from src.utils.job_evidence import (
    build_job_evidence,
    job_evidence_id,
    job_skill_evidence_id,
)
from src.utils.latex import escape_latex, pdf_page_count, run_pdflatex
from src.utils.paths import job_output_dir
from src.utils.skill_matching import (
    canonicalize,
)

logger = logging.getLogger(__name__)


class GenerateCoverLetterInput(StrictBaseModel):
    """Input for generate_cover_letter."""

    job: Job
    approved_resume_path: str
    candidate_evidence: list[EvidenceItem] = Field(default_factory=list)
    job_evidence: list[EvidenceItem] = Field(default_factory=list)
    run_id: str | None = None


class GenerateCoverLetterOutput(StrictBaseModel):
    """Output from generate_cover_letter."""

    job_id: str
    output_tex_path: str
    output_pdf_path: str
    page_count: int = Field(ge=0)
    evidence_used: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


_escape_latex = escape_latex


# Tunable limits

MAX_ALIGNED_SKILLS = 6
MAX_COMPILE_ATTEMPTS = 4

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE_RE = re.compile(r"(?:\+?\d[\d\-.\s]{7,}\d)")
_URL_RE = re.compile(r"(?:https?://|github\.com/|linkedin\.com/)[^\s|]+")
_LOCATION_RE = re.compile(r"\b([A-Z][a-zA-Z.]+(?:\s[A-Z][a-zA-Z.]+)*,\s*[A-Z]{2})\b")

# Internal data


@dataclass
class ContactInfo:
    name: str = "Candidate"
    location: str = ""
    email: str = ""
    phone: str = ""
    link: str = ""


@dataclass
class LetterContent:
    contact: ContactInfo
    greeting: str
    opening: str
    body_paragraphs: list[str]
    skills_line: str
    closing: str
    evidence_used: set[str] = field(default_factory=set)


# Public API


def run_cover_letter_tool(
    value: GenerateCoverLetterInput,
    *,
    tracer: TraceManager | None = None,
) -> GenerateCoverLetterOutput:
    """Generate a one-page, evidence-grounded cover letter PDF for one job."""

    if not value.job_evidence:
        value = value.model_copy(update={"job_evidence": build_job_evidence(value.job)})
    active = tracer or TraceManager(enabled=False)
    trace_metadata = {
        "tool_name": "generate_cover_letter",
        "job_id": value.job.job_id,
        "resume_id": f"resume-{value.job.job_id}",
        "company": value.job.company,
    }
    errors: list[str] = []
    evidence_pool = {item.evidence_id: item for item in value.candidate_evidence}
    job_evidence_pool = {item.evidence_id: item for item in value.job_evidence}

    resume_text, resume_errors = _read_approved_resume(value.approved_resume_path)
    if resume_errors:
        return GenerateCoverLetterOutput(
            job_id=value.job.job_id,
            output_tex_path="",
            output_pdf_path="",
            page_count=0,
            evidence_used=[],
            errors=resume_errors,
        )
    with active.span(
        "cover_letter.generate_content",
        trace_metadata,
        input={
            "job_id": value.job.job_id,
            "evidence_count": len(evidence_pool),
            "required_skill_count": len(value.job.required_skills),
        },
    ) as content_span:
        contact = _extract_contact(resume_text)
        letter = _build_letter(value.job, evidence_pool, contact, job_evidence_pool)
        active.update_span(
            content_span,
            output={
                "evidence_ids": sorted(letter.evidence_used),
                "paragraph_count": len(letter.body_paragraphs),
            },
            metadata={"status": "OK"},
        )

    config = get_config()
    job_dir = job_output_dir(value.job.job_id, run_id=value.run_id, config=config)
    job_dir.mkdir(parents=True, exist_ok=True)
    tex_path = job_dir / "cover_letter.tex"
    pdf_path = job_dir / "cover_letter.pdf"

    compile_kwargs = {}
    compile_parameters = inspect.signature(_compile_one_page).parameters
    if "tracer" in compile_parameters:
        compile_kwargs = {
            "tracer": active,
            "trace_metadata": trace_metadata,
        }
    page_count, compile_errors = _compile_one_page(
        letter,
        value.job,
        tex_path,
        pdf_path,
        **compile_kwargs,
    )
    errors.extend(compile_errors)

    return GenerateCoverLetterOutput(
        job_id=value.job.job_id,
        output_tex_path=str(tex_path),
        output_pdf_path=str(pdf_path) if page_count == 1 else "",
        page_count=page_count or 0,
        evidence_used=sorted(letter.evidence_used),
        errors=errors,
    )


# Resume contact details


def _read_approved_resume(resume_path: str) -> tuple[str, list[str]]:
    """Read a genuine, one-page approved resume before drafting the letter."""

    path = Path(resume_path)
    if not path.exists():
        return "", [f"Approved resume PDF not found: {resume_path}"]
    if path.suffix.casefold() != ".pdf":
        return "", [f"Approved resume must be a PDF: {resume_path}"]
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        page_count = len(reader.pages)
        if page_count != 1:
            return "", [
                f"Approved resume must be exactly one page; found {page_count}."
            ]
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        if not text.strip():
            return "", [f"Approved resume contains no extractable text: {resume_path}"]
        return text, []
    except Exception as exc:  # noqa: BLE001 - return a contract-valid failure
        return "", [f"Could not read approved resume PDF: {exc}"]


def _extract_contact(resume_text: str) -> ContactInfo:
    lines = [line.strip() for line in resume_text.splitlines() if line.strip()]
    name = lines[0] if lines else "Candidate"
    # Search each line so a match cannot spill into the candidate's name.
    header_lines = lines[1:6] if len(lines) > 1 else []

    email = phone = url = location = ""
    for line in header_lines:
        if not email and (match := _EMAIL_RE.search(line)):
            email = match.group(0)
        if not phone and (match := _PHONE_RE.search(line)):
            phone = match.group(0).strip()
        if not url and (match := _URL_RE.search(line)):
            url = match.group(0)
        if not location and (match := _LOCATION_RE.search(line)):
            location = match.group(1)

    return ContactInfo(name=name, location=location, email=email, phone=phone, link=url)


# Evidence-backed letter content


def _build_letter(
    job: Job,
    evidence_pool: dict[str, EvidenceItem],
    contact: ContactInfo,
    job_evidence_pool: dict[str, EvidenceItem] | None = None,
) -> LetterContent:
    if job_evidence_pool is None:
        job_evidence_pool = {item.evidence_id: item for item in build_job_evidence(job)}
    used: set[str] = {
        job_evidence_id(job, "title"),
        job_evidence_id(job, "description"),
    }

    aligned_skills, skill_evidence_ids = _match_required_skills(
        job.required_skills, evidence_pool
    )
    used.update(skill_evidence_ids)
    used.update(job_skill_evidence_id(job, skill) for skill in aligned_skills)

    achievements = _select_achievement_evidence(evidence_pool, aligned_skills)
    used.update(item.evidence_id for item in achievements)

    hook = _company_hook(job)
    if hook:
        used.add(job_evidence_id(job, "company-details"))

    greeting = f"Dear {job.company} Hiring Team,"

    opening = (
        f"I am writing to apply for the {job.title} position at {job.company}. {hook}"
        if hook
        else f"I am writing to apply for the {job.title} position at {job.company}."
    )

    body_paragraphs: list[str] = []
    if achievements:
        sentences = []
        for item in achievements[:2]:
            sentences.append(f"{_clean_sentence(item.text)}.")
        body_paragraphs.append(
            "In my recent work, "
            + " ".join(sentences)
            + " This experience maps directly onto the "
            f"responsibilities described for the {job.title} role."
        )
    else:
        body_paragraphs.append(
            f"I am interested in the responsibilities described for the {job.title} "
            "role and would welcome the opportunity to discuss my candidacy."
        )

    if aligned_skills:
        body_paragraphs.append(
            f"My skill set includes {_join_list(aligned_skills)}, each of which is called out explicitly "
            f"in the {job.title} requirements at {job.company}."
        )

    skills_line = (
        f"Relevant skills: {_join_list(aligned_skills)}."
        if aligned_skills
        else "I welcome the chance to discuss how my broader background fits this role."
    )

    closing = (
        "Thank you for your time and consideration. I would welcome the opportunity to discuss how "
        f"I can contribute to {job.company}.\\\\[6pt]\nSincerely,\\\\\n{_escape_latex(contact.name)}"
    )

    known_ids = set(evidence_pool) | set(job_evidence_pool)
    unknown_ids = used - known_ids
    if unknown_ids:
        raise ValueError(
            f"Cover letter cites evidence not present in its input: {sorted(unknown_ids)}"
        )

    return LetterContent(
        contact=contact,
        greeting=greeting,
        opening=opening,
        body_paragraphs=body_paragraphs,
        skills_line=skills_line,
        closing=closing,
        evidence_used=used,
    )


def _match_required_skills(
    required_skills: list[str], evidence_pool: dict[str, EvidenceItem]
) -> tuple[list[str], set[str]]:
    """Return required skills evidenced in the candidate's profile, and the citing IDs.

    Preference order mirrors the assignment's evidence hierarchy: resume,
    then master-skills, then portfolio, then memory (facts learned during
    the human-review pause).
    """

    source_priority = {"resume": 0, "master_skills": 1, "portfolio": 2, "memory": 3}
    aligned: list[str] = []
    citing_ids: set[str] = set()

    for skill in required_skills:
        needle = canonicalize(skill)
        if not needle:
            continue
        candidates = [
            item
            for item in evidence_pool.values()
            if evidence_supports_skill(item, skill)
        ]
        if not candidates:
            continue
        candidates.sort(key=lambda item: source_priority.get(item.source, 9))
        best = candidates[0]
        aligned.append(skill)
        citing_ids.add(best.evidence_id)
        if len(aligned) >= MAX_ALIGNED_SKILLS:
            break

    return aligned, citing_ids


def _select_achievement_evidence(
    evidence_pool: dict[str, EvidenceItem], aligned_skills: list[str]
) -> list[EvidenceItem]:
    """Pick up to two concrete, skill-overlapping achievement bullets to cite."""

    achievement_tags = {"experience", "project", "portfolio"}
    needles = [skill.casefold() for skill in aligned_skills]

    scored: list[tuple[int, EvidenceItem]] = []
    for item in evidence_pool.values():
        if item.source not in {"resume", "portfolio", "memory"}:
            continue
        if (
            not (achievement_tags & {tag.casefold() for tag in item.tags})
            and item.source != "portfolio"
        ):
            continue
        overlap = sum(
            1 for needle in needles if needle and needle in item.text.casefold()
        )
        # Prefer direct skill overlap, but always retain grounded experience as
        # a fallback so every letter maps real candidate work to the role.
        scored.append((overlap, item))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    # Use at most one experience and one project example.
    picked: list[EvidenceItem] = []
    seen_sources: set[str] = set()
    for _, item in scored:
        bucket = (
            "portfolio"
            if item.source == "portfolio" or "project" in item.tags
            else "experience"
        )
        if bucket in seen_sources:
            continue
        picked.append(item)
        seen_sources.add(bucket)
        if len(picked) == 2:
            break
    return picked


def _company_hook(job: Job) -> str:
    details = job.company_details.strip()
    if not details:
        return ""
    first_sentence = re.split(r"(?<=[.!?])\s+", details)[0].strip()
    return first_sentence


def _clean_sentence(text: str) -> str:
    summary_match = re.search(r"(?m)^SUMMARY:\s*(.+)$", text)
    if summary_match:
        text = summary_match.group(1)
    elif " | " in text:
        text = text.rsplit(" | ", 1)[-1]
    cleaned = " ".join(text.split())
    cleaned = cleaned.rstrip(".")
    return cleaned[0].upper() + cleaned[1:] if cleaned else cleaned


def _join_list(items: list[str]) -> str:
    if len(items) <= 1:
        return items[0] if items else ""
    return ", ".join(items[:-1]) + f", and {items[-1]}"


# One-page PDF rendering

_TEMPLATE = r"""\documentclass[letterpaper,%(fontsize)spt]{article}

\usepackage[empty]{fullpage}
\usepackage[hidelinks]{hyperref}
\usepackage[english]{babel}

\pagestyle{empty}
\raggedbottom
\raggedright

\addtolength{\oddsidemargin}{-0.5in}
\addtolength{\evensidemargin}{-0.5in}
\addtolength{\textwidth}{1in}
\addtolength{\topmargin}{%(topmargin)s}
\addtolength{\textheight}{%(textheight)s}

\begin{document}
\begin{center}
  {\Large \scshape %(name)s} \\ \vspace{2pt}
  %(header_line)s
\end{center}
\vspace{10pt}

\noindent %(date)s

\vspace{10pt}
\noindent %(greeting)s

\vspace{8pt}
\noindent %(opening)s

%(body)s

\vspace{8pt}
\noindent %(skills_line)s

\vspace{10pt}
\noindent %(closing)s

\end{document}
"""


def _render_tex(letter: LetterContent, job: Job, shrink_level: int) -> str:
    header_parts = [
        part
        for part in (
            letter.contact.location,
            letter.contact.phone,
            letter.contact.email,
            letter.contact.link,
        )
        if part
    ]
    header_line = " $|$ ".join(_escape_latex(part) for part in header_parts)

    body = "\n".join(
        f"\n\\vspace{{6pt}}\n\\noindent {_escape_latex(paragraph)}"
        for paragraph in letter.body_paragraphs
    )

    from datetime import date

    return _TEMPLATE % {
        "fontsize": 11 if shrink_level == 0 else 10,
        "topmargin": "-0.7in" if shrink_level else "-0.6in",
        "textheight": "1.4in" if shrink_level else "1.2in",
        "name": _escape_latex(letter.contact.name),
        "header_line": header_line,
        "date": date.today().strftime("%B %d, %Y"),
        "greeting": _escape_latex(letter.greeting),
        "opening": _escape_latex(letter.opening),
        "body": body,
        "skills_line": _escape_latex(letter.skills_line),
        "closing": letter.closing,  # The candidate's name is already escaped.
    }


def _compile_one_page(
    letter: LetterContent,
    job: Job,
    tex_path: Path,
    pdf_path: Path,
    *,
    tracer: TraceManager | None = None,
    trace_metadata: dict[str, object] | None = None,
) -> tuple[int | None, list[str]]:
    errors: list[str] = []
    working_letter = letter
    page_count: int | None = None

    for attempt in range(MAX_COMPILE_ATTEMPTS):
        shrink_level = attempt
        if attempt >= 2 and len(working_letter.body_paragraphs) > 1:
            # Keep the strongest paragraph when space is tight.
            working_letter = LetterContent(
                contact=working_letter.contact,
                greeting=working_letter.greeting,
                opening=working_letter.opening,
                body_paragraphs=working_letter.body_paragraphs[:1],
                skills_line=working_letter.skills_line,
                closing=working_letter.closing,
                evidence_used=working_letter.evidence_used,
            )

        tex_source = _render_tex(working_letter, job, shrink_level)
        tex_path.write_text(tex_source, encoding="utf-8")

        run_errors = _run_pdflatex(tex_path)
        if run_errors:
            errors.extend(run_errors)
            return None, errors
        pdf_created = pdf_path.is_file()

        if not pdf_created:
            errors.append(f"pdflatex reported success but {pdf_path} was not produced.")
            return None, errors

        try:
            page_count = pdf_page_count(pdf_path)
        except Exception as exc:  # noqa: BLE001 - report invalid generated PDFs
            errors.append(f"Generated cover-letter PDF is invalid: {exc}")
            return None, errors
        if page_count == 1:
            letter.evidence_used = working_letter.evidence_used
            return 1, errors

        errors.append(
            f"Attempt {attempt + 1}: cover letter compiled to {page_count} pages; retrying with tighter content."
        )

    errors.append("Could not fit the cover letter onto one page after all retries.")
    return page_count, errors  # Return the last measured page count.


def _run_pdflatex(tex_path: Path) -> list[str]:
    return run_pdflatex(tex_path, artifact_label="cover letter")
