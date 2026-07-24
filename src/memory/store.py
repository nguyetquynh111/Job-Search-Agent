"""JSON-backed candidate memory store."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import ValidationError

from src.memory.models import MemoryFact

logger = logging.getLogger(__name__)


class MemoryStoreError(RuntimeError):
    """Raised when memory cannot be loaded or written safely."""


class JSONMemoryStore:
    """Persistent JSON-file candidate memory store."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> list[MemoryFact]:
        """Load active and inactive memory facts from JSON."""

        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("[]", encoding="utf-8")
            return []
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise MemoryStoreError(f"Memory file is corrupted: {self.path}") from exc
        if not isinstance(raw, list):
            raise MemoryStoreError(f"Memory file must contain a JSON array: {self.path}")
        try:
            return [MemoryFact.model_validate(item) for item in raw]
        except ValidationError as exc:
            raise MemoryStoreError(f"Memory file contains invalid entries: {self.path}") from exc

    def append_many(self, facts: list[MemoryFact]) -> list[MemoryFact]:
        """Append new facts and persist the full memory file."""

        if not facts:
            return self.load()
        current = self.load()
        known = {
            fact.deduplication_key
            for fact in current
            if fact.active
        }
        deduped: list[MemoryFact] = []
        for fact in facts:
            key = fact.deduplication_key
            if key in known:
                logger.info("Skipping duplicate memory fact: %s", fact.canonical_value)
                continue
            known.add(key)
            deduped.append(fact)
        updated = [*current, *deduped]
        self._write(updated)
        return updated

    def reset(self) -> None:
        """Reset memory to an empty JSON array."""

        self._write([])

    def _write(self, facts: list[MemoryFact]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = [fact.model_dump() for fact in facts]
        tmp_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp_path.replace(self.path)
