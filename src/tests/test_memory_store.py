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


def test_memory_extractor_supports_new_skills_and_candidate_facts() -> None:
    """Self-asserted facts are preserved without a fixed technology whitelist."""

    facts = extract_memory_facts(
        {
            "J001": (
                "I have used Snowflake, dbt, and Airflow in previous projects. "
                "I have 5 years of experience in data engineering."
            )
        },
        review_round=2,
    )

    assert {(fact.fact_type, fact.canonical_value) for fact in facts} == {
        ("skill", "Snowflake"),
        ("skill", "dbt"),
        ("skill", "Airflow"),
        ("experience", "5 years of experience in data engineering"),
    }
    assert all(fact.provenance.review_round == 2 for fact in facts)
    assert all(fact.provenance.related_job_id == "J001" for fact in facts)


def test_memory_ids_do_not_repeat_across_extraction_calls() -> None:
    """A process restart/counter reset cannot create colliding evidence IDs."""

    first = extract_memory_facts({"J001": "I know Rust."}, review_round=1)
    second = extract_memory_facts({"J002": "I know Go."}, review_round=1)

    assert first[0].fact_id != second[0].fact_id
    assert first[0].fact_id.startswith("mem-")
    assert second[0].fact_id.startswith("mem-")
