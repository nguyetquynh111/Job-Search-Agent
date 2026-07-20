"""Memory store tests."""

from __future__ import annotations

from pathlib import Path

from src.memory.extractor import extract_memory_facts
from src.memory.models import MemoryFact, MemoryProvenance
from src.memory.store import JSONMemoryStore


def test_memory_persists_to_json(tmp_path: Path) -> None:
    """Memory facts are written and reloaded from JSON."""

    path = tmp_path / "memory.json"
    store = JSONMemoryStore(path)
    fact = MemoryFact(
        fact_id="mem-test",
        fact_type="skill",
        canonical_value="GraphQL",
        provenance=MemoryProvenance(
            source="human_review",
            review_round=1,
            original_statement="I have used GraphQL.",
            related_job_id="J002",
        ),
    )
    store.append_many([fact])

    loaded = store.load()
    assert len(loaded) == 1
    assert loaded[0].canonical_value == "GraphQL"


def test_memory_extractor_ignores_editing_preferences() -> None:
    """Extractor stores candidate facts, not editing preferences."""

    facts = extract_memory_facts(
        {
            "J001": "Make this shorter and use a friendlier tone.",
            "J002": "Add GraphQL. I have used it in previous projects.",
        },
        review_round=1,
    )

    assert [fact.canonical_value for fact in facts] == ["GraphQL"]
