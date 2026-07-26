"""Backward-compatible imports for shared exact skill matching."""

from src.tools.skill_matching import (
    CANONICAL_ALIASES,
    CATEGORY_SKILLS,
    canonicalize,
    category_members,
    curated_vocabulary,
    skill_in_text,
    variants,
)

__all__ = [
    "CANONICAL_ALIASES",
    "CATEGORY_SKILLS",
    "canonicalize",
    "category_members",
    "curated_vocabulary",
    "skill_in_text",
    "variants",
]
