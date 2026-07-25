"""Defensive text normalization for job postings.

``data/jobs.csv`` contains bullet characters that decode as replacement
characters (mojibake). We sanitize inside this tool rather than editing the
shared ``data_loader`` (an upstream fix is suggested in the tool README).
"""

from __future__ import annotations

import re

# Replacement char, common bullet glyphs, and non-breaking space.
_NOISE = re.compile(r"[�•▪●■ ‣⁃]+")
# Mojibake sequences that appear when UTF-8 bullets are mis-decoded as latin-1.
_MOJIBAKE = re.compile(r"(?:â€¢|Ã¢|â¢)+")


def sanitize_text(text: str | None) -> str:
    """Strip mojibake/bullet noise and collapse whitespace in job text."""

    if not text:
        return ""
    cleaned = _MOJIBAKE.sub(" ", text)
    cleaned = _NOISE.sub(" ", cleaned)
    cleaned = cleaned.replace("\\n", " ").replace("\r", " ").replace("\n", " ")
    return re.sub(r"\s+", " ", cleaned).strip()
