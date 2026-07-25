"""Implementation of the ``generate_cover_letter`` model-visible tool.

Contract: ``GenerateCoverLetterInput`` -> ``GenerateCoverLetterOutput``
(see ``src/schemas/cover_letter.py`` and ``src/tools/cover_letter/``).

Design notes
------------
* No fact used in the letter is invented. Every skill, achievement, or
  claim written into the body must be traceable to an ``EvidenceItem`` in
  ``value.candidate_evidence`` (resume, master skills, portfolio, memory).
  Only IDs that exist in that list are ever reported in ``evidence_used``,
  satisfying the contract rule "Every evidence ID must exist in the input".
* The job description / company details are never treated as evidence
  *about the candidate* -- they are only used to decide which evidence is
  relevant and to build the company-specific hook.
* ``approved_resume_path`` (the tailored, human-approved PDF for this job)
  is read only to recover contact-header details (name, location, email,
  phone, links) so the letter's header matches the resume that goes out
  with it. Nothing pulled from it is cited as evidence, since it is not
  part of ``candidate_evidence`` per the contract.
* Compilation follows the same LaTeX -> pdflatex -> PDF pipeline used for
  the resume, and enforces a one-page PDF the same way the resume
  tailoring tool must (trim, then shrink, then re-check).
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader

from src.config import get_config
from src.schemas.common import EvidenceItem
from src.schemas.cover_letter import GenerateCoverLetterInput, GenerateCoverLetterOutput
from src.schemas.jobs import Job

# --------------------------------------------------------------------------
# Tunables
# --------------------------------------------------------------------------

MAX_ALIGNED_SKILLS = 6
MAX_COMPILE_ATTEMPTS = 4
PDFLATEX_TIMEOUT_SECONDS = 60

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE_RE = re.compile(r"(?:\+?\d[\d\-.\s]{7,}\d)")
_URL_RE = re.compile(r"(?:https?://|github\.com/|linkedin\.com/)[^\s|]+")
_LOCATION_RE = re.compile(r"\b([A-Z][a-zA-Z.]+(?:\s[A-Z][a-zA-Z.]+)*,\s*[A-Z]{2})\b")

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


# --------------------------------------------------------------------------
# Small internal data holders
# --------------------------------------------------------------------------


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


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------


def generate_cover_letter(value: GenerateCoverLetterInput) -> GenerateCoverLetterOutput:
    """Generate a one-page, evidence-grounded cover letter PDF for one job."""

    errors: list[str] = []
    evidence_pool = {item.evidence_id: item for item in value.candidate_evidence}

    resume_text, contact_errors = _read_resume_text(value.approved_resume_path)
    errors.extend(contact_errors)
    contact = _extract_contact(resume_text)

    letter = _build_letter(value.job, evidence_pool, contact)

    config = get_config()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    tex_path = config.output_dir / f"{value.job.job_id}-cover-letter.tex"
    pdf_path = config.output_dir / f"{value.job.job_id}-cover-letter.pdf"

    page_count, compile_errors = _compile_one_page(letter, value.job, tex_path, pdf_path)
    errors.extend(compile_errors)

    return GenerateCoverLetterOutput(
        job_id=value.job.job_id,
        output_tex_path=str(tex_path),
        output_pdf_path=str(pdf_path) if page_count else "",
        page_count=page_count or 1,
        evidence_used=sorted(letter.evidence_used),
        errors=errors,
    )


# --------------------------------------------------------------------------
# Step 1: recover header details from the approved resume PDF
# --------------------------------------------------------------------------


def _read_resume_text(resume_path: str) -> tuple[str, list[str]]:
    path = Path(resume_path)
    if not path.exists():
        return "", [f"Approved resume PDF not found at {resume_path}; using placeholder header."]
    try:
        reader = PdfReader(str(path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return text, []
    except Exception as exc:  # noqa: BLE001 - degrade gracefully, never crash the run
        return "", [f"Could not read approved resume PDF ({exc}); using placeholder header."]


def _extract_contact(resume_text: str) -> ContactInfo:
    lines = [line.strip() for line in resume_text.splitlines() if line.strip()]
    name = lines[0] if lines else "Candidate"
    # Search line-by-line (not the whole blob) so patterns never span a
    # newline and accidentally swallow the name into the address line.
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


# --------------------------------------------------------------------------
# Step 2: build the letter content strictly from evidence
# --------------------------------------------------------------------------


def _build_letter(
    job: Job, evidence_pool: dict[str, EvidenceItem], contact: ContactInfo
) -> LetterContent:
    used: set[str] = set()

    aligned_skills, skill_evidence_ids = _match_required_skills(job.required_skills, evidence_pool)
    used.update(skill_evidence_ids)

    achievements = _select_achievement_evidence(evidence_pool, aligned_skills)
    used.update(item.evidence_id for item in achievements)

    hook = _company_hook(job)

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
            sentences.append(f"{_clean_sentence(item.text)} ({item.evidence_id}).")
        body_paragraphs.append(
            "In my recent work, " + " ".join(sentences) + " This experience maps directly onto the "
            f"responsibilities described for the {job.title} role."
        )
    else:
        body_paragraphs.append(
            f"My background aligns with the core responsibilities of the {job.title} role, and I am "
            "eager to bring that experience to your team."
        )
        errors_note = f"no-evidence-overlap:{job.job_id}"
        # No fabricated specifics are added when there is no overlapping evidence.
        _ = errors_note

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
        needle = skill.strip().casefold()
        if not needle:
            continue
        candidates = [
            item
            for item in evidence_pool.values()
            if needle in item.text.casefold() or any(needle == tag.casefold() for tag in item.tags)
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
        if not (achievement_tags & {tag.casefold() for tag in item.tags}) and item.source != "portfolio":
            continue
        overlap = sum(1 for needle in needles if needle and needle in item.text.casefold())
        if overlap:
            scored.append((overlap, item))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    # Prefer variety: at most one resume-experience item and one portfolio/project item.
    picked: list[EvidenceItem] = []
    seen_sources: set[str] = set()
    for _, item in scored:
        bucket = "portfolio" if item.source == "portfolio" or "project" in item.tags else "experience"
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
    cleaned = " ".join(text.split())
    cleaned = cleaned.rstrip(".")
    return cleaned[0].upper() + cleaned[1:] if cleaned else cleaned


def _join_list(items: list[str]) -> str:
    if len(items) <= 1:
        return items[0] if items else ""
    return ", ".join(items[:-1]) + f", and {items[-1]}"


# --------------------------------------------------------------------------
# Step 3: render LaTeX and compile a one-page PDF
# --------------------------------------------------------------------------

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
        f"\n\\vspace{{6pt}}\n\\noindent {_escape_latex(paragraph)}" for paragraph in letter.body_paragraphs
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
        "closing": letter.closing,  # already LaTeX-safe (built with escaped name)
    }


def _escape_latex(text: str) -> str:
    return "".join(_LATEX_SPECIAL_CHARS.get(ch, ch) for ch in text)


def _compile_one_page(
    letter: LetterContent, job: Job, tex_path: Path, pdf_path: Path
) -> tuple[int | None, list[str]]:
    errors: list[str] = []
    working_letter = letter

    for attempt in range(MAX_COMPILE_ATTEMPTS):
        shrink_level = attempt
        if attempt >= 2 and len(working_letter.body_paragraphs) > 1:
            # Trim to the single strongest, evidence-backed paragraph.
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

        if not pdf_path.exists():
            errors.append(f"pdflatex reported success but {pdf_path} was not produced.")
            return None, errors

        page_count = len(PdfReader(str(pdf_path)).pages)
        if page_count == 1:
            letter.evidence_used = working_letter.evidence_used
            return 1, errors

        errors.append(
            f"Attempt {attempt + 1}: cover letter compiled to {page_count} pages; retrying with tighter content."
        )

    errors.append("Could not fit the cover letter onto one page after all retries.")
    return page_count, errors  # last known page count, still reported honestly


def _run_pdflatex(tex_path: Path) -> list[str]:
    try:
        result = subprocess.run(
            [
                "pdflatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                "-output-directory",
                str(tex_path.parent),
                str(tex_path),
            ],
            capture_output=True,
            text=True,
            timeout=PDFLATEX_TIMEOUT_SECONDS,
        )
    except FileNotFoundError:
        return ["pdflatex is not installed or not on PATH. Install a LaTeX distribution to compile PDFs."]
    except subprocess.TimeoutExpired:
        return ["pdflatex timed out while compiling the cover letter."]

    if result.returncode != 0:
        tail = "\n".join(result.stdout.splitlines()[-15:])
        return [f"pdflatex failed for {tex_path.name}: {tail}"]
    return []
