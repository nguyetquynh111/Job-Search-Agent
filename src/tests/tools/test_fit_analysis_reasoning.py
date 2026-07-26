"""Reasoning path selection: LLM stub, retry-on-invalid, and fallback."""

from __future__ import annotations

import json

from src.tests.tools.fit_analysis_helpers import (
    make_evidence,
    make_input,
    make_job,
    make_profile,
)
from src.tools.fit_analysis import llm_client, reasoning
from src.tools.fit_analysis.evidence_index import build_evidence_index
from src.tools.fit_analysis.prepass import run_prepass

_VALID_JSON = json.dumps(
    {
        "job_id": "J1",
        "relevant_experience": [],
        "seniority": [],
        "education": [],
        "aligned_skills": [
            {
                "claim": "Python: aligned",
                "evidence_ids": ["resume-skills-001"],
                "confidence": 0.8,
                "notes": None,
            }
        ],
        "evidenced_missing_skills": [],
        "genuine_gaps": [],
        "project_analysis": [],
        "project_swap": None,
    }
)


def _fixture():
    job = make_job(required_skills=["Python"])
    profile = make_profile(skills=["Python"])
    items = [
        make_evidence(
            "resume-skills-001", "resume", "Resume skills: Python", tags=["Python"]
        )
    ]
    inp = make_input(job, profile, items)
    index = build_evidence_index(items)
    return inp, run_prepass(inp, index), index


def test_stub_llm_produces_llm_path() -> None:
    inp, prepass, index = _fixture()
    output, path, _ = reasoning.analyze(
        inp, prepass, index, complete_fn=lambda s, u: _VALID_JSON
    )
    assert path == "llm"
    assert output.aligned_skills[0].claim.startswith("Python")


def test_invalid_then_valid_triggers_single_retry() -> None:
    inp, prepass, index = _fixture()
    calls = {"n": 0}

    def flaky(system: str, user: str) -> str:
        calls["n"] += 1
        return "not json" if calls["n"] == 1 else _VALID_JSON

    output, path, _ = reasoning.analyze(inp, prepass, index, complete_fn=flaky)
    assert calls["n"] == 2
    assert path == "llm"
    assert output.job_id == "J1"


def test_llm_exception_falls_back_deterministically() -> None:
    inp, prepass, index = _fixture()

    def boom(system: str, user: str) -> str:
        raise RuntimeError("network down")

    output, path, meta = reasoning.analyze(inp, prepass, index, complete_fn=boom)
    assert path == "fallback"
    assert meta.get("llm_error") == "RuntimeError"
    assert [c.claim.split(":")[0] for c in output.aligned_skills] == ["Python"]


def test_no_model_configured_uses_fallback_offline(monkeypatch) -> None:
    monkeypatch.setattr(llm_client, "model_configured", lambda: False)
    inp, prepass, index = _fixture()
    output, path, _ = reasoning.analyze(inp, prepass, index)
    assert path == "fallback"
    assert output.job_id == "J1"
