"""Build a canonical skill -> evidence index from the supplied evidence items.

The index is the single source of truth for "is this skill evidenced, by which
IDs, and from what kind of source". It powers the disjoint skill buckets and the
deterministic confidence rule. Memory evidence is treated exactly like portfolio
or master-skills evidence.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from src.schemas.common import EvidenceItem
from src.tools.implementations.fit_analysis.aliases import canonicalize

logger = logging.getLogger(__name__)

# Map an EvidenceItem.source to the source kind used for confidence scoring.
_SOURCE_KINDS = {
    "resume": "resume",
    "master_skills": "master_skills",
    "portfolio": "portfolio",
    "memory": "memory",
}

# Source priority for ordering/capping cited evidence IDs (lower = stronger proof).
_SOURCE_PRIORITY = {"resume": 0, "master_skills": 1, "portfolio": 2, "memory": 3, "other": 4}

# Colon-parsing of evidence text is only safe for short, skill-style items (e.g.
# memory facts "skill: PyTorch"); full-text resume/portfolio blobs use tags only.
_MAX_COLON_PARSE_CHARS = 120

# Tags that describe the evidence container rather than a concrete skill.
_GENERIC_TAGS = {
    "skills",
    "skill",
    "resume",
    "memory",
    "portfolio",
    "education",
    "experience",
    "project",
    "projects",
    "master_skills",
    "candidate_fact",
}


def source_kind(source: str) -> str:
    """Return the confidence source kind for an evidence source label."""

    return _SOURCE_KINDS.get(source.strip().lower(), "other")


@dataclass
class EvidenceIndex:
    """Canonical skill lookup over the supplied evidence items."""

    by_skill: dict[str, list[tuple[str, str]]] = field(default_factory=dict)
    all_ids: set[str] = field(default_factory=set)

    def ids_for(self, canonical: str) -> list[str]:
        """Return evidence IDs supporting a canonical skill, de-duplicated in order."""

        seen: set[str] = set()
        ordered: list[str] = []
        for evidence_id, _ in self.by_skill.get(canonical, []):
            if evidence_id not in seen:
                seen.add(evidence_id)
                ordered.append(evidence_id)
        return ordered

    def ordered_ids(self, canonical: str, limit: int | None = None) -> list[str]:
        """Return supporting IDs ordered by source strength, optionally capped.

        Strongest proof first (resume, then master skills, portfolio, memory) so a
        claim cites its best evidence rather than an arbitrary long list.
        """

        seen: set[str] = set()
        ranked: list[tuple[int, int, str]] = []
        for position, (evidence_id, kind) in enumerate(self.by_skill.get(canonical, [])):
            if evidence_id in seen:
                continue
            seen.add(evidence_id)
            ranked.append((_SOURCE_PRIORITY.get(kind, 4), position, evidence_id))
        ranked.sort()
        ids = [evidence_id for _, _, evidence_id in ranked]
        return ids[:limit] if limit is not None else ids

    def sources_for(self, canonical: str) -> set[str]:
        """Return the set of source kinds supporting a canonical skill."""

        return {kind for _, kind in self.by_skill.get(canonical, [])}

    def has(self, canonical: str) -> bool:
        """Return whether any evidence supports a canonical skill."""

        return canonical in self.by_skill

    def summary(self) -> dict[str, list[str]]:
        """Return a compact skill -> source-kinds view for prompts and traces."""

        return {
            skill: sorted({kind for _, kind in entries})
            for skill, entries in sorted(self.by_skill.items())
        }


def _candidate_tokens(item: EvidenceItem) -> set[str]:
    """Extract candidate skill tokens from an evidence item's tags and text."""

    tokens: set[str] = set()
    for tag in item.tags:
        if tag.strip().lower() not in _GENERIC_TAGS:
            tokens.add(tag)
    # Only short, skill-style items (e.g. memory facts "skill: PyTorch") are parsed
    # by their "label: value" suffix. Full-text resume/portfolio blobs use tags only,
    # so their entire body is not shredded into spurious skill tokens.
    if ":" in item.text and (
        item.source.strip().lower() == "memory" or len(item.text) <= _MAX_COLON_PARSE_CHARS
    ):
        _, _, tail = item.text.partition(":")
        for piece in re.split(r"[;,]", tail):
            piece = piece.strip()
            if piece and len(piece.split()) <= 5:
                tokens.add(piece)
    return tokens


def build_evidence_index(evidence_items: list[EvidenceItem]) -> EvidenceIndex:
    """Index every evidence item by the canonical skills it supports."""

    index = EvidenceIndex()
    for item in evidence_items:
        index.all_ids.add(item.evidence_id)
        kind = source_kind(item.source)
        for token in _candidate_tokens(item):
            canonical = canonicalize(token)
            if not canonical:
                continue
            index.by_skill.setdefault(canonical, []).append((item.evidence_id, kind))
    logger.debug(
        "Evidence index built: %d skills across %d evidence items",
        len(index.by_skill),
        len(index.all_ids),
    )
    return index
