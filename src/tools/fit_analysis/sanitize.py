"""Defensive text normalization for job postings.

``data/jobs.csv`` contains bullet characters that decode as replacement
characters (mojibake). We sanitize inside this tool rather than editing the
shared ``data_loader`` so the behavior stays local to fit analysis.
"""

from __future__ import annotations

import re

# Normalize common broken characters and bullets.
_NOISE = re.compile(r"[�•▪●■ ‣⁃]+")
# Repair bullets decoded with the wrong encoding.
_MOJIBAKE = re.compile(r"(?:â€¢|Ã¢|â¢)+")


def sanitize_text(text: str | None) -> str:
    """Strip mojibake/bullet noise and collapse whitespace in job text."""

    if not text:
        return ""
    cleaned = _MOJIBAKE.sub(" ", text)
    cleaned = _NOISE.sub(" ", cleaned)
    cleaned = cleaned.replace("\\n", " ").replace("\r", " ").replace("\n", " ")
    return re.sub(r"\s+", " ", cleaned).strip()
