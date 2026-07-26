"""Claim verdicts and their render markers.

``FitAnalysisOutput`` is frozen, so a claim's verdict is carried in the
``EvidenceClaim.notes`` field as a ``verdict=<value>`` prefix. The renderer reads
it to choose a marker and never defaults an unknown verdict to a positive (✅)
marker.
"""

from __future__ import annotations

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
