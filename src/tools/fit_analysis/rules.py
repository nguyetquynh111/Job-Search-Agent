"""Deterministic rules shared by fit-analysis stages.

``FitAnalysisOutput`` is frozen, so a claim's verdict is carried in the
``EvidenceClaim.notes`` field as a ``verdict=<value>`` prefix. The renderer reads
it to choose a marker and never defaults an unknown verdict to a positive (✅)
marker.

Confidence and text sanitizing live here too because they are tiny pure rules
used by multiple stages, not separate responsibilities.
"""

from __future__ import annotations

import re

MATCH = "match"
PARTIAL = "partial"
MISMATCH = "mismatch"
MISSING = "missing"  # Supported elsewhere, but missing from the resume.

_VALID = {MATCH, PARTIAL, MISMATCH, MISSING}


def tag(value: str, note: str | None = None) -> str:
    """Encode a verdict (and optional human note) into a notes string."""

    return f"verdict={value}" + (f"; {note}" if note else "")


def parse(notes: str | None) -> tuple[str | None, str | None]:
    """Return (verdict, human_note) parsed from a notes string."""

    if not notes:
        return None, None
    text = notes.strip()
    if not text.startswith("verdict="):
        return None, text
    body = text[len("verdict=") :]
    value, _, human = body.partition(";")
    value = value.strip()
    human = human.strip() or None
    return (value if value in _VALID else None), human


def marker(value: str | None, missing_marker: str = "❌") -> str:
    """Return the render marker for a verdict.

    The rendered report uses only ✅ and ❌ (per the assignment example, which marks
    a partial alignment with ❌ and distinguishes it by label). ``match`` renders ✅;
    ``partial``/``mismatch``/unknown render ❌ (never a positive default); ``missing``
    uses ``missing_marker``. The finer verdict is preserved in the JSON ``notes``.
    """

    if value == MISSING:
        return missing_marker
    return "✅" if value == MATCH else "❌"


# Evidence sources used in confidence scores.
RESUME = "resume"
MASTER_SKILLS = "master_skills"
PORTFOLIO = "portfolio"
MEMORY = "memory"


def skill_confidence(source_kinds: set[str], on_resume: bool) -> float:
    """Return the confidence for a skill claim given its evidence source kinds."""

    non_resume = {kind for kind in source_kinds if kind != RESUME}
    if on_resume:
        return 0.9 if non_resume else 0.8
    if len(non_resume) >= 2:
        return 0.85
    if PORTFOLIO in non_resume:
        return 0.70
    if MASTER_SKILLS in non_resume:
        return 0.65
    if MEMORY in non_resume:
        return 0.60
    # A confirmed lack of evidence is still a confident finding.
    return 0.75


def narrative_confidence(evidence_id_count: int) -> float:
    """Return confidence for experience/seniority/education claims."""

    return 0.9 if evidence_id_count >= 2 else 0.8


_NOISE = re.compile(r"[�•▪●■ ‣⁃]+")
_MOJIBAKE = re.compile(r"(?:â€¢|Ã¢|â¢)+")


def sanitize_text(text: str | None) -> str:
    """Return stable ASCII-ish text for deterministic matching."""

    if not text:
        return ""
    cleaned = _MOJIBAKE.sub(" ", text)
    cleaned = _NOISE.sub(" ", cleaned)
    cleaned = cleaned.replace("\u2013", "-").replace("\u2014", "-")
    cleaned = cleaned.replace("\u2018", "'").replace("\u2019", "'")
    cleaned = cleaned.replace("\u201c", '"').replace("\u201d", '"')
    return re.sub(r"\s+", " ", cleaned).strip()
