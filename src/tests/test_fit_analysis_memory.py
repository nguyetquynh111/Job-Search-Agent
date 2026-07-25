"""End-to-end proof that memory facts act as first-class evidence."""

from __future__ import annotations

from pathlib import Path

from src.memory.models import memory_fact_to_evidence
from src.memory.store import JSONMemoryStore
from src.schemas.common import EvidenceItem
from src.tests.fit_analysis_helpers import make_input, make_job, make_profile
from src.tools.implementations import analyze_fit as entry
from src.tools.implementations.fit_analysis import llm_client
from src.tools.implementations.fit_analysis.render import build_source_labels, render_fit_analysis

_FIXTURE = Path(__file__).parent / "fixtures" / "memory_sample.json"


def _memory_evidence() -> list[EvidenceItem]:
    facts = JSONMemoryStore(_FIXTURE).load()  # real MemoryFact shape, never the live file
    return [EvidenceItem.model_validate(memory_fact_to_evidence(f)) for f in facts if f.active]


def test_memory_only_skill_is_evidenced_missing_and_cites_review(monkeypatch) -> None:
    monkeypatch.setattr(llm_client, "model_configured", lambda: False)
    memory = _memory_evidence()
    # GraphQL is required by the job, absent from the resume, evidenced only by memory.
    job = make_job(required_skills=["GraphQL", "Python"])
    profile = make_profile(skills=["Python"])
    resume = EvidenceItem(
        evidence_id="resume-skills-001", source="resume", text="Resume skills: Python", tags=["Python"]
    )
    result = entry.analyze_fit(make_input(job, profile, [resume, *memory]))

    missing = {c.claim.split(":")[0] for c in result.evidenced_missing_skills}
    assert "GraphQL" in missing
    graphql = next(c for c in result.evidenced_missing_skills if c.claim.startswith("GraphQL"))
    assert graphql.evidence_ids == ["mem-graphql01"]

    text = render_fit_analysis(
        result, job, source_labels=build_source_labels(make_input(job, profile, [resume, *memory]))
    )
    assert 'GraphQL (stated during review)' in text
