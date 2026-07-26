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

from src.domain import EvidenceItem
from src.utils.skill_matching import (
    canonicalize,
    curated_vocabulary,
    skill_in_text,
)

logger = logging.getLogger(__name__)

# Normalize source labels for confidence scoring.
_SOURCE_KINDS = {
    "resume": "resume",
    "master_skills": "master_skills",
    "portfolio": "portfolio",
    "memory": "memory",
}

# Lower values mean stronger evidence.
_SOURCE_PRIORITY = {
    "resume": 0,
    "master_skills": 1,
    "portfolio": 2,
    "memory": 3,
    "other": 4,
}

# Parse labels like "skill: PyTorch" only in short evidence.
_MAX_COLON_PARSE_CHARS = 120

# Scan short resume entries, but never the full uploaded resume.
_SCANNABLE_TAGS = {"education", "experience", "project", "projects"}

# These tags describe containers, not skills.
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
        for position, (evidence_id, kind) in enumerate(
            self.by_skill.get(canonical, [])
        ):
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
    # Parse label/value text only when it is short enough to be specific.
    if ":" in item.text and (
        item.source.strip().lower() == "memory"
        or len(item.text) <= _MAX_COLON_PARSE_CHARS
    ):
        _, _, tail = item.text.partition(":")
        for piece in re.split(r"[;,]", tail):
            piece = piece.strip()
            if piece and len(piece.split()) <= 5:
                tokens.add(piece)
    return tokens


def _scanned_tokens(item: EvidenceItem, vocabulary: set[str]) -> set[str]:
    """Return vocabulary skills mentioned in a section item's free text.

    Only items carrying a :data:`_SCANNABLE_TAGS` section tag are scanned, and only
    against the supplied bounded vocabulary, so an education or experience line can
    ground a skill claim without shredding full documents into spurious tokens.
    """

    tags = {tag.strip().lower() for tag in item.tags}
    if not tags & _SCANNABLE_TAGS:
        return set()
    return {
        canonical for canonical in vocabulary if skill_in_text(canonical, item.text)
    }


def build_evidence_index(
    evidence_items: list[EvidenceItem], vocabulary: set[str] | None = None
) -> EvidenceIndex:
    """Index every evidence item by the canonical skills it supports.

    ``vocabulary`` extends the curated alias/category tokens that free-text section
    items are scanned for -- callers pass the job's required skills so a requirement
    named only in a resume education or experience line is still found.
    """

    scan_vocabulary = {canonicalize(token) for token in vocabulary or set()}
    scan_vocabulary |= curated_vocabulary()
    scan_vocabulary.discard("")

    index = EvidenceIndex()
    for item in evidence_items:
        index.all_ids.add(item.evidence_id)
        kind = source_kind(item.source)
        tokens = _candidate_tokens(item) | _scanned_tokens(item, scan_vocabulary)
        for token in tokens:
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
