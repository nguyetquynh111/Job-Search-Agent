"""Consolidated fit-analysis tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from src.agent import (
    CandidateProfile,
    CandidatePreferences,
    EvidenceItem,
    PortfolioProject,
)
from src.agent import EvidenceClaim, ProjectSwap
from src.agent import Job
from src.agent import build_job_evidence
from src.agent import load_candidate_profile, load_jobs_csv, load_portfolio
from src.agent import load_resume_data, normalize_string_list
from src.review.memory import JSONMemoryStore
from src.review.memory import memory_fact_to_evidence
from src.tools.fit_analysis import fit_analysis as entry
from src.tools.fit_analysis import fit_analysis as rationale
from src.tools.fit_analysis import fit_analysis as reasoning
from src.tools.fit_analysis import fit_analysis as verdict
from src.tools.fit_analysis import llm as llm_client
from src.tools.fit_analysis.fit_analysis import AnalyzeFitInput
from src.tools.fit_analysis.fit_analysis import FitAnalysisOutput
from src.tools.fit_analysis.fit_analysis import ProjectAnalysisItem
from src.tools.fit_analysis.fit_analysis import build_evidence_index
from src.tools.fit_analysis.fit_analysis import build_source_labels, render_fit_analysis
from src.tools.fit_analysis.fit_analysis import expand_canonical_skills
from src.tools.fit_analysis.fit_analysis import match_terms, rank_projects
from src.tools.fit_analysis.fit_analysis import post_validate
from src.tools.fit_analysis.fit_analysis import run_prepass
from src.tools.fit_analysis.fit_analysis import sanitize_text
from src.tools.fit_analysis.fit_analysis import seniority_verdict
import json
import pytest

# --- fit_analysis_helpers.py ---
"""Builders for fit-analysis tests (not collected as a test module)."""


def make_evidence(
    evidence_id: str, source: str, text: str = "", tags: list[str] | None = None
) -> EvidenceItem:
    """Build an EvidenceItem for tests."""

    return EvidenceItem(
        evidence_id=evidence_id,
        source=source,
        text=text or evidence_id,
        tags=tags or [],
    )


def make_job(
    job_id: str = "J1",
    required_skills: list[str] | None = None,
    years: int | float | None = None,
    title: str = "AI Engineer",
    description: str = "Build ML systems.",
) -> Job:
    """Build a Job for tests."""

    return Job(
        job_id=job_id,
        title=title,
        company="Acme",
        description=description,
        required_skills=required_skills or [],
        years_experience_required=years,
    )


def make_profile(
    skills: list[str] | None = None,
    years: int | float | None = 4,
    resume_projects: list[str] | None = None,
) -> CandidateProfile:
    """Build a CandidateProfile for tests."""

    return CandidateProfile(
        candidate_id="cand-1",
        name="Test Candidate",
        skills=skills or [],
        resume_projects=resume_projects or [],
        preferences=CandidatePreferences(years_of_experience=years),
    )


def make_portfolio_project(
    project_id: str,
    name: str,
    technologies: list[str] | None = None,
    domains: list[str] | None = None,
) -> PortfolioProject:
    """Build a PortfolioProject for tests."""

    return PortfolioProject(
        project_id=project_id,
        name=name,
        description=f"{name} description",
        technologies=technologies or [],
        domains=domains or [],
        evidence_ids=[f"portfolio-{project_id}"],
    )


def make_input(
    job: Job,
    profile: CandidateProfile,
    evidence_items: list[EvidenceItem] | None = None,
    current_resume_projects: list[str] | None = None,
    portfolio_projects: list[PortfolioProject] | None = None,
) -> AnalyzeFitInput:
    """Build an AnalyzeFitInput for tests."""

    return AnalyzeFitInput(
        job=job,
        candidate_profile=profile,
        evidence_items=evidence_items or [],
        current_resume_projects=current_resume_projects or [],
        portfolio_projects=portfolio_projects or [],
    )


def production_like_input(job_id: str = "J017") -> AnalyzeFitInput:
    jobs = {job.job_id: job for job in load_jobs_csv("data/jobs.csv")}
    profile = load_candidate_profile("data/preferences.yaml")
    resume = load_resume_data("data/resume.tex")
    portfolio = load_portfolio("data/portfolio.txt")
    profile = profile.model_copy(
        update={
            "resume_content": resume.plain_text,
            "skills": normalize_string_list([*profile.skills, *resume.skills]),
            "education": normalize_string_list(
                [*profile.education, *resume.education]
            ),
            "experience": normalize_string_list(
                [*profile.experience, *resume.experience]
            ),
            "resume_projects": normalize_string_list(
                [*profile.resume_projects, *resume.projects]
            ),
            "resume_evidence": [*profile.resume_evidence, *resume.evidence_items],
        }
    )
    job = jobs[job_id]
    return AnalyzeFitInput(
        job=job,
        candidate_profile=profile,
        evidence_items=[
            *profile.resume_evidence,
            *profile.master_skill_evidence,
            *portfolio.evidence_items,
            *profile.portfolio_evidence,
        ],
        job_evidence=build_job_evidence(job),
        current_resume_projects=profile.resume_projects,
        portfolio_projects=portfolio.projects,
    )


def test_production_like_llm_omission_gets_complete_fit_sections() -> None:
    inp = production_like_input("J017")

    def empty_but_valid(_system: str, _user: str) -> str:
        return json.dumps(
            {
                "job_id": inp.job.job_id,
                "relevant_experience": [],
                "seniority": [],
                "education": [],
                "aligned_skills": [],
                "evidenced_missing_skills": [],
                "genuine_gaps": [],
                "project_analysis": [],
                "project_swap": None,
            }
        )

    output = entry.analyze_fit(inp, complete_fn=empty_but_valid)

    assert output.relevant_experience
    assert output.seniority
    assert output.education
    assert output.aligned_skills or output.evidenced_missing_skills or output.genuine_gaps
    assert output.project_analysis
    for claims in (
        output.relevant_experience,
        output.seniority,
        output.education,
        output.project_analysis,
    ):
        assert all(claim.evidence_ids for claim in claims)


# --- test_tool_fit_analysis_category.py ---
"""Category->instance skill resolution and the swap minimum-improvement threshold."""


def _fit_category_prepass(job, profile, evidence, portfolio=None, current=None):
    inp = make_input(
        job,
        profile,
        evidence,
        current_resume_projects=current or [],
        portfolio_projects=portfolio or [],
    )
    return run_prepass(inp, build_evidence_index(inp.evidence_items))


def test_category_evidenced_by_portfolio_member_is_missing_not_gap() -> None:
    job = make_job(required_skills=["agents"])
    profile = make_profile(skills=["Python"])
    evidence = [
        make_evidence(
            "resume-skills-001", "resume", "Resume skills: Python", tags=["Python"]
        ),
        make_evidence("portfolio-P1", "portfolio", "chatbot", tags=["AutoGen"]),
    ]
    pp = _fit_category_prepass(job, profile, evidence)
    assert [c.skill for c in pp.evidenced_missing] == ["agents"]
    assert pp.evidenced_missing[0].members == ["AutoGen"]  # Name the concrete tool.
    assert pp.evidenced_missing[0].evidence_ids == ["portfolio-P1"]
    assert pp.genuine_gaps == []


def test_category_member_on_resume_is_aligned() -> None:
    job = make_job(required_skills=["APIs"])
    profile = make_profile(skills=["FastAPI"])
    evidence = [
        make_evidence(
            "resume-skills-001", "resume", "Resume skills: FastAPI", tags=["FastAPI"]
        )
    ]
    pp = _fit_category_prepass(job, profile, evidence)
    assert [c.skill for c in pp.aligned] == ["APIs"]
    assert "FastAPI" in pp.aligned[0].members


def test_category_without_evidenced_member_stays_gap() -> None:
    job = make_job(required_skills=["vector databases"])
    profile = make_profile(skills=["Python"])
    evidence = [
        make_evidence(
            "resume-skills-001", "resume", "Resume skills: Python", tags=["Python"]
        )
    ]
    pp = _fit_category_prepass(job, profile, evidence)
    assert [c.skill for c in pp.genuine_gaps] == ["vector databases"]
    assert pp.evidenced_missing == []


def test_slash_combined_label_resolves_via_parts() -> None:
    job = make_job(required_skills=["Docker/Kubernetes"])
    profile = make_profile(skills=["Docker"])
    evidence = [
        make_evidence(
            "resume-skills-001", "resume", "Resume skills: Docker", tags=["Docker"]
        )
    ]
    pp = _fit_category_prepass(job, profile, evidence)
    assert [c.skill for c in pp.aligned] == ["Docker/Kubernetes"]


def _fit_category_swap_portfolio(external_tech, external_domains=None):
    return [
        make_portfolio_project("P1", "Current Strong", technologies=["PyTorch"]),
        make_portfolio_project("P2", "Current Weak", technologies=["Python"]),
        make_portfolio_project(
            "P3", "External", technologies=external_tech, domains=external_domains or []
        ),
    ]


def test_weak_swap_below_margin_is_suppressed() -> None:
    # A 1.5-point gain is too small for a swap.
    job = make_job(required_skills=["PyTorch", "Python"])
    profile = make_profile(
        skills=["Python"], resume_projects=["Current Strong", "Current Weak"]
    )
    evidence = [
        make_evidence(
            "resume-skills-001", "resume", "Resume skills: Python", tags=["Python"]
        )
    ]
    pp = _fit_category_prepass(
        job,
        profile,
        evidence,
        portfolio=_fit_category_swap_portfolio(["PyTorch"]),
        current=["Current Strong", "Current Weak"],
    )
    assert pp.swap is not None
    assert pp.swap.swap is None
    from src.tools.fit_analysis.fit_analysis import build_project_section

    claims, swap = build_project_section(pp.swap)
    assert swap is None
    assert any("no project swap is recommended" in c.claim.lower() for c in claims)


def test_meaningful_swap_above_margin_is_recommended() -> None:
    # A new skill and domain make the external project clearly stronger.
    job = make_job(
        required_skills=["PyTorch", "Python"], description="Computer vision role."
    )
    profile = make_profile(
        skills=["Python"], resume_projects=["Current Strong", "Current Weak"]
    )
    evidence = [
        make_evidence(
            "resume-skills-001", "resume", "Resume skills: Python", tags=["Python"]
        )
    ]
    pp = _fit_category_prepass(
        job,
        profile,
        evidence,
        portfolio=_fit_category_swap_portfolio(
            ["PyTorch"], external_domains=["Computer Vision"]
        ),
        current=["Current Strong", "Current Weak"],
    )
    assert pp.swap is not None
    selected_swap = pp.swap.swap
    assert selected_swap is not None
    assert selected_swap.remove_project == "Current Weak"
    assert selected_swap.add_project == "External"


# --- test_tool_fit_analysis_entrypoint.py ---
"""Entry point: offline end-to-end and the 'no better swap available' branch."""


def test_analyze_fit_runs_offline(monkeypatch) -> None:
    monkeypatch.setattr(llm_client, "model_configured", lambda: False)
    job = make_job(job_id="J42", required_skills=["Python", "Rust"])
    profile = make_profile(skills=["Python"])
    items = [
        make_evidence(
            "resume-skills-001", "resume", "Resume skills: Python", tags=["Python"]
        )
    ]
    result = entry.run_fit_analysis_tool(make_input(job, profile, items))

    assert isinstance(result, FitAnalysisOutput)
    assert result.job_id == "J42"
    assert [c.claim.split(":")[0] for c in result.aligned_skills] == ["Python"]
    assert [c.claim.split(":")[0] for c in result.genuine_gaps] == ["Rust"]
    # Each claim cites the candidate evidence and job requirement.
    known = {i.evidence_id for i in items} | {"job-J42-skill-001"}
    for claim in result.aligned_skills:
        assert set(claim.evidence_ids) <= known
        assert "job-J42-skill-001" in claim.evidence_ids
    every_claim = [
        *result.relevant_experience,
        *result.seniority,
        *result.education,
        *result.aligned_skills,
        *result.evidenced_missing_skills,
        *result.genuine_gaps,
        *result.project_analysis,
    ]
    assert every_claim
    assert all(
        any(evidence_id.startswith("job-J42-") for evidence_id in claim.evidence_ids)
        for claim in every_claim
    )


def test_no_pass_marker_is_rendered_without_evidence(monkeypatch) -> None:
    # A skill without citable evidence cannot pass.
    monkeypatch.setattr(llm_client, "model_configured", lambda: False)
    job = make_job(required_skills=["Python", "SQL"])
    profile = make_profile(skills=["Python", "SQL"])
    result = entry.run_fit_analysis_tool(make_input(job, profile, evidence_items=[]))

    assert [c for c in result.aligned_skills if not c.evidence_ids] == []
    for claim in [*result.aligned_skills, *result.evidenced_missing_skills]:
        assert claim.evidence_ids, "every non-gap claim must cite real evidence"


def test_no_better_swap_available_is_stated(monkeypatch) -> None:
    monkeypatch.setattr(llm_client, "model_configured", lambda: False)
    job = make_job(required_skills=["Python"])
    profile = make_profile(skills=["Python"], resume_projects=["Alpha", "Beta"])
    portfolio = [
        make_portfolio_project("P1", "Alpha", technologies=["Python"]),
        make_portfolio_project("P2", "Beta", technologies=["Python"]),
    ]
    items = [
        make_evidence(
            "resume-skills-001", "resume", "Resume skills: Python", tags=["Python"]
        )
    ]
    result = entry.run_fit_analysis_tool(
        make_input(
            job,
            profile,
            items,
            current_resume_projects=["Alpha", "Beta"],
            portfolio_projects=portfolio,
        )
    )
    assert result.project_swap is None
    assert any(
        "no project swap is recommended" in c.claim.lower()
        for c in result.project_analysis
    )
    conclusion = next(
        claim
        for claim in result.project_analysis
        if "no project swap is recommended" in claim.claim.lower()
    )
    assert {"portfolio-P1", "portfolio-P2"} <= set(conclusion.evidence_ids)


def test_swapped_out_project_is_never_reported_as_strong(monkeypatch) -> None:
    # The removed project must also be the weak one in the analysis.
    from src.tools.fit_analysis import fit_analysis as verdict

    monkeypatch.setattr(llm_client, "model_configured", lambda: False)
    job = make_job(
        required_skills=["PyTorch", "RAG"],
        description="Generative AI role using RAG and LLMs.",
    )
    profile = make_profile(
        skills=["Python"], resume_projects=["Vision Project", "Chatbot"]
    )
    portfolio = [
        make_portfolio_project(
            "P1",
            "Vision Project",
            technologies=["PyTorch"],
            domains=["Computer Vision"],
        ),
        make_portfolio_project(
            "P2", "Chatbot", technologies=["RAG"], domains=["Generative AI"]
        ),
        make_portfolio_project(
            "P3",
            "RAG Platform",
            technologies=["RAG", "PyTorch"],
            domains=["Generative AI"],
        ),
    ]
    items = [
        make_evidence(
            "resume-skills-001", "resume", "Resume skills: Python", tags=["Python"]
        )
    ]
    result = entry.run_fit_analysis_tool(
        make_input(
            job,
            profile,
            items,
            current_resume_projects=["Vision Project", "Chatbot"],
            portfolio_projects=portfolio,
        )
    )
    if result.project_swap is not None:
        removed = result.project_swap.remove_project
        for claim in result.project_analysis:
            if claim.claim.startswith(f"Current project '{removed}'"):
                value, _ = verdict.parse(claim.notes)
                assert value == verdict.MISMATCH
                assert "aligns well" not in claim.claim


# --- test_tool_fit_analysis_memory.py ---
"""End-to-end proof that memory facts act as first-class evidence."""


_FIXTURE = Path(__file__).parent / "fixtures" / "memory_sample.json"


def _fit_memory_memory_evidence() -> list[EvidenceItem]:
    facts = JSONMemoryStore(
        _FIXTURE
    ).load()  # Use real memory objects without touching the live file.
    return [
        EvidenceItem.model_validate(memory_fact_to_evidence(f))
        for f in facts
        if f.active
    ]


def test_memory_only_skill_is_evidenced_missing_and_cites_review(monkeypatch) -> None:
    monkeypatch.setattr(llm_client, "model_configured", lambda: False)
    memory = _fit_memory_memory_evidence()
    # Only memory supports the missing GraphQL skill.
    job = make_job(required_skills=["GraphQL", "Python"])
    profile = make_profile(skills=["Python"])
    resume = EvidenceItem(
        evidence_id="resume-skills-001",
        source="resume",
        text="Resume skills: Python",
        tags=["Python"],
    )
    result = entry.run_fit_analysis_tool(make_input(job, profile, [resume, *memory]))

    missing = {c.claim.split(":")[0] for c in result.evidenced_missing_skills}
    assert "GraphQL" in missing
    graphql = next(
        c for c in result.evidenced_missing_skills if c.claim.startswith("GraphQL")
    )
    assert graphql.evidence_ids == ["job-J1-skill-001", "mem-graphql01"]

    text = render_fit_analysis(
        result,
        job,
        source_labels=build_source_labels(make_input(job, profile, [resume, *memory])),
    )
    assert "GraphQL (job posting; stated during review)" in text


# --- test_tool_fit_analysis_merge.py ---
"""The model owns the judgments: its proposal survives, guards catch what it gets wrong.

These assert through ``analyze_fit`` rather than ``reasoning.analyze`` in isolation,
because the whole point is what survives the merge/post-validation pipeline.
"""


_EMPTY = {
    "relevant_experience": [],
    "seniority": [],
    "education": [],
    "aligned_skills": [],
    "evidenced_missing_skills": [],
    "genuine_gaps": [],
    "project_analysis": [],
    "project_swap": None,
}


def _fit_merge_stub(**overrides):
    """Return a complete_fn serving one LLM proposal, ignoring rationale sub-calls."""

    payload = {"job_id": "J1", **_EMPTY, **overrides}

    def complete(system: str, user: str) -> str:
        if "ONE sentence" in system or "rewrite" in system.lower():
            return "unusable prose"
        return json.dumps(payload)

    return complete


def _fit_merge_claim(text: str, ids: list[str], note: str) -> dict:
    return {"claim": text, "evidence_ids": ids, "confidence": 0.8, "notes": note}


def _fit_merge_skill_names(claims) -> set[str]:
    return {c.claim.split(":")[0] for c in claims}


def _fit_merge_swap_fixture(strong_domains: list[str] | None = None):
    """Two resume projects plus two equally strong external candidates.

    ``strong_domains`` lifts the strong incumbent's score so that replacing *it*
    falls below the swap margin, which is what the rejection path needs.
    """

    job = make_job(
        required_skills=["PyTorch", "Python"], description="Computer vision role."
    )
    profile = make_profile(
        skills=["Python"], resume_projects=["Current Strong", "Current Weak"]
    )
    portfolio = [
        make_portfolio_project(
            "P1",
            "Current Strong",
            technologies=["PyTorch"],
            domains=strong_domains or [],
        ),
        make_portfolio_project("P2", "Current Weak", technologies=["Python"]),
        make_portfolio_project(
            "P3", "External One", technologies=["PyTorch"], domains=["Computer Vision"]
        ),
        make_portfolio_project(
            "P4", "External Two", technologies=["PyTorch"], domains=["Computer Vision"]
        ),
    ]
    items = [
        make_evidence(
            "resume-skills-001", "resume", "Resume skills: Python", tags=["Python"]
        )
    ]
    return make_input(
        job,
        profile,
        items,
        current_resume_projects=["Current Strong", "Current Weak"],
        portfolio_projects=portfolio,
    )


def test_model_bucket_assignment_survives() -> None:
    # Real portfolio evidence can correct the pre-pass.
    job = make_job(required_skills=["Python", "GraphQL"])
    profile = make_profile(skills=["Python"])
    items = [
        make_evidence(
            "resume-skills-001", "resume", "Resume skills: Python", tags=["Python"]
        ),
        make_evidence(
            "portfolio-P1", "portfolio", "GraphQL gateway work", tags=["service layer"]
        ),
    ]
    result = entry.run_fit_analysis_tool(
        make_input(job, profile, items),
        complete_fn=_fit_merge_stub(
            evidenced_missing_skills=[
                _fit_merge_claim(
                    "GraphQL: evidenced by the gateway project",
                    ["portfolio-P1"],
                    "verdict=missing; llm",
                )
            ]
        ),
    )
    assert "GraphQL" in _fit_merge_skill_names(result.evidenced_missing_skills)
    assert "GraphQL" not in _fit_merge_skill_names(result.genuine_gaps)


def test_omitted_required_skill_is_backfilled_from_deterministic_pass() -> None:
    # An empty model response still produces a complete analysis.
    job = make_job(required_skills=["Python", "Rust"])
    profile = make_profile(skills=["Python"])
    items = [
        make_evidence(
            "resume-experience-001",
            "resume",
            "Junior Engineer building Python services.",
            tags=["experience", "Python"],
        )
    ]
    result = entry.run_fit_analysis_tool(
        make_input(job, profile, items), complete_fn=_fit_merge_stub()
    )

    covered = (
        _fit_merge_skill_names(result.aligned_skills)
        | _fit_merge_skill_names(result.evidenced_missing_skills)
        | _fit_merge_skill_names(result.genuine_gaps)
    )
    assert {"Python", "Rust"} <= covered
    assert "Python" in _fit_merge_skill_names(result.aligned_skills)
    assert "Rust" in _fit_merge_skill_names(result.genuine_gaps)


def test_model_swap_choice_is_honoured_when_it_validates() -> None:
    inp = _fit_merge_swap_fixture()
    result = entry.run_fit_analysis_tool(
        inp,
        complete_fn=_fit_merge_stub(
            project_swap={
                "remove_project": "Current Weak",
                "add_project": "External Two",  # The ranking prefers External One.
                "rationale": "The model's own reasoning for External Two.",
                "evidence_ids": ["portfolio-P4"],
            }
        ),
    )
    assert result.project_swap is not None
    assert result.project_swap.add_project == "External Two"
    assert result.project_swap.remove_project == "Current Weak"
    assert (
        result.project_swap.rationale == "The model's own reasoning for External Two."
    )


def test_model_chosen_removal_is_reported_as_the_weak_slot() -> None:
    # Model-selected swaps still use the shared ranking.
    inp = _fit_merge_swap_fixture()
    result = entry.run_fit_analysis_tool(
        inp,
        complete_fn=_fit_merge_stub(
            project_swap={
                "remove_project": "Current Weak",
                "add_project": "External Two",
                "rationale": "model rationale",
                "evidence_ids": [],
            }
        ),
    )
    assert result.project_swap is not None
    removed = result.project_swap.remove_project
    matching = [
        c
        for c in result.project_analysis
        if c.claim.startswith(f"Current project '{removed}'")
    ]
    assert matching, "the removed project must appear in project_analysis"
    for claim in matching:
        value, _ = verdict.parse(claim.notes)
        assert value == verdict.MISMATCH
        assert "aligns well" not in claim.claim


def test_invented_swap_target_falls_back_to_ranked_choice() -> None:
    inp = _fit_merge_swap_fixture()
    result = entry.run_fit_analysis_tool(
        inp,
        complete_fn=_fit_merge_stub(
            project_swap={
                "remove_project": "Current Weak",
                "add_project": "Project That Does Not Exist",
                "rationale": "fabricated",
                "evidence_ids": [],
            }
        ),
    )
    # Invented projects fall back to the ranked choice.
    assert result.project_swap is not None
    assert result.project_swap.add_project == "External One"


def test_swap_below_margin_falls_back_to_ranked_choice() -> None:
    # Reject a swap that offers no improvement.
    inp = _fit_merge_swap_fixture(strong_domains=["Computer Vision"])
    result = entry.run_fit_analysis_tool(
        inp,
        complete_fn=_fit_merge_stub(
            project_swap={
                "remove_project": "Current Strong",
                "add_project": "External Two",
                "rationale": "marginal",
                "evidence_ids": [],
            }
        ),
    )
    assert result.project_swap is not None
    assert result.project_swap.remove_project == "Current Weak"  # Ranked choice.
    assert result.project_swap.add_project == "External One"


def test_seniority_prose_contradicting_the_years_verdict_is_replaced() -> None:
    job = make_job(required_skills=["Python"], years=10)
    profile = make_profile(skills=["Python"], years=2)  # Clear shortfall.
    items = [
        make_evidence(
            "resume-experience-001",
            "resume",
            "Junior Engineer building Python services.",
            tags=["experience", "Python"],
        )
    ]
    result = entry.run_fit_analysis_tool(
        make_input(job, profile, items),
        complete_fn=_fit_merge_stub(
            seniority=[
                _fit_merge_claim(
                    "You exceed the requirement comfortably.", [], "verdict=match; llm"
                )
            ]
        ),
    )
    assert result.seniority, "a seniority claim must still be produced"
    claim = result.seniority[0]
    assert "exceed" not in claim.claim.lower()
    value, _ = verdict.parse(claim.notes)
    assert value in {verdict.PARTIAL, verdict.MISMATCH}


def test_model_narrative_sections_pass_through() -> None:
    job = make_job(required_skills=["Python"])
    profile = make_profile(skills=["Python"])
    items = [
        make_evidence(
            "resume-experience-001",
            "resume",
            "Senior Engineer at Acme centered on X with strong Python overlap.",
            tags=["experience", "Python"],
        ),
        make_evidence(
            "resume-education-001", "resume", "M.S. Data Science", tags=["education"]
        ),
    ]
    result = entry.run_fit_analysis_tool(
        make_input(job, profile, items),
        complete_fn=_fit_merge_stub(
            relevant_experience=[
                _fit_merge_claim(
                    "Senior Engineer at Acme: centered on X — strong overlap.",
                    ["resume-experience-001"],
                    "verdict=match; llm",
                )
            ],
            education=[
                _fit_merge_claim(
                    "Education: M.S. Data Science",
                    ["resume-education-001"],
                    "verdict=match; llm",
                )
            ],
        ),
    )
    assert result.relevant_experience[0].claim.startswith("Senior Engineer at Acme")
    assert result.education[0].claim == "Education: M.S. Data Science"


# --- test_tool_fit_analysis_postvalidate.py ---
"""Post-validation: evidence-ID repair, demotion, disjointness, swap rejection."""


def _fit_post_base_input(
    portfolio=None, current=None, evidence=None, required_skills=None
):
    job = make_job(required_skills=required_skills or ["Python"])
    profile = make_profile(skills=["Python"], resume_projects=current or [])
    return make_input(
        job,
        profile,
        evidence_items=evidence
        or [make_evidence("ev-real", "resume", "Resume skills: Python")],
        current_resume_projects=current or [],
        portfolio_projects=portfolio or [],
    )


def test_unknown_evidence_id_is_dropped() -> None:
    inp = _fit_post_base_input()
    output = FitAnalysisOutput(
        job_id="WRONG",
        aligned_skills=[
            EvidenceClaim(claim="Python: aligned", evidence_ids=["ev-real", "ev-ghost"])
        ],
    )
    result, repairs = post_validate(output, inp)
    assert result.aligned_skills[0].evidence_ids == [
        "job-J1-skill-001",
        "ev-real",
    ]
    assert any("ev-ghost" in r for r in repairs)
    assert result.job_id == "J1"  # Match the input job.


def test_evidenced_missing_without_valid_evidence_is_demoted() -> None:
    inp = _fit_post_base_input(required_skills=["Kafka"])
    output = FitAnalysisOutput(
        job_id="J1",
        evidenced_missing_skills=[
            EvidenceClaim(claim="Kafka: evidenced", evidence_ids=["ev-ghost"])
        ],
    )
    result, repairs = post_validate(output, inp)
    assert result.evidenced_missing_skills == []
    assert [c.claim.split(":")[0] for c in result.genuine_gaps] == ["Kafka"]
    assert any("demoted" in r.lower() for r in repairs)


def test_aligned_without_valid_evidence_is_demoted_to_gap() -> None:
    # A passing claim needs a real citation.
    inp = _fit_post_base_input()
    output = FitAnalysisOutput(
        job_id="J1",
        aligned_skills=[
            EvidenceClaim(claim="Python: aligned", evidence_ids=["ev-ghost"])
        ],
    )
    result, repairs = post_validate(output, inp)
    assert result.aligned_skills == []
    assert [c.claim.split(":")[0] for c in result.genuine_gaps] == ["Python"]
    assert any("no valid evidence" in r for r in repairs)


def test_aligned_with_only_non_resume_evidence_becomes_evidenced_missing() -> None:
    # Portfolio-only proof belongs in the off-resume bucket.
    evidence = [
        make_evidence("resume-skills-001", "resume", "Resume skills: Python"),
        make_evidence(
            "portfolio-P1", "portfolio", "Kafka pipeline work", tags=["Kafka"]
        ),
    ]
    inp = _fit_post_base_input(evidence=evidence, required_skills=["Kafka"])
    output = FitAnalysisOutput(
        job_id="J1",
        aligned_skills=[
            EvidenceClaim(claim="Kafka: aligned", evidence_ids=["portfolio-P1"])
        ],
    )
    result, repairs = post_validate(output, inp)
    assert result.aligned_skills == []
    assert [c.claim.split(":")[0] for c in result.evidenced_missing_skills] == ["Kafka"]
    assert "not yet on your resume" in result.evidenced_missing_skills[0].claim
    assert any("no resume evidence" in r for r in repairs)


def test_buckets_are_made_disjoint_by_priority() -> None:
    inp = _fit_post_base_input()
    output = FitAnalysisOutput(
        job_id="J1",
        aligned_skills=[
            EvidenceClaim(claim="Python: aligned", evidence_ids=["ev-real"])
        ],
        evidenced_missing_skills=[
            EvidenceClaim(claim="Python: also here", evidence_ids=["ev-real"])
        ],
    )
    result, repairs = post_validate(output, inp)
    assert len(result.aligned_skills) == 1
    assert result.evidenced_missing_skills == []
    assert any("higher-priority" in r for r in repairs)


def test_hallucinated_project_swap_is_rejected() -> None:
    portfolio = [make_portfolio_project("P1", "Real Project", technologies=["Python"])]
    inp = _fit_post_base_input(portfolio=portfolio, current=["Old Project"])
    output = FitAnalysisOutput(
        job_id="J1",
        project_swap=ProjectSwap(
            remove_project="Old Project",
            add_project="Imaginary Project",  # Not in the portfolio.
            rationale="made up",
        ),
    )
    result, repairs = post_validate(output, inp)
    assert result.project_swap is None
    assert any("not in portfolio" in r for r in repairs)


def test_confidence_reflects_source_kinds() -> None:
    evidence = [
        make_evidence("resume-skills-001", "resume", "Resume skills: Python"),
        make_evidence("portfolio-P1", "portfolio", "b", tags=["Python"]),
    ]
    inp = _fit_post_base_input(evidence=evidence)
    output = FitAnalysisOutput(
        job_id="J1",
        aligned_skills=[
            EvidenceClaim(
                claim="Python: aligned",
                evidence_ids=["resume-skills-001", "portfolio-P1"],
            )
        ],
    )
    result, _ = post_validate(output, inp)
    # Resume evidence plus another source earns 0.9 verdict.
    assert result.aligned_skills[0].confidence == 0.9


def test_go_claim_cannot_use_python_evidence() -> None:
    inp = _fit_post_base_input(
        required_skills=["Go"],
        evidence=[
            make_evidence(
                "resume-python",
                "resume",
                "Built production services in Python.",
                tags=["experience", "Python"],
            )
        ],
    )
    output = FitAnalysisOutput(
        job_id="J1",
        aligned_skills=[
            EvidenceClaim(claim="Go: aligned", evidence_ids=["resume-python"])
        ],
    )

    result, failures = post_validate(output, inp)

    assert result.aligned_skills == []
    assert [claim.claim.split(":")[0] for claim in result.genuine_gaps] == ["Go"]
    assert any("semantically unrelated evidence" in item for item in failures)
    assert result.validation_failures == failures


def test_harmless_alias_is_accepted_but_unrelated_keyword_is_not() -> None:
    inp = _fit_post_base_input(
        required_skills=["Kubernetes"],
        evidence=[
            make_evidence(
                "master-k8s",
                "master_skills",
                "Skill: k8s",
                tags=["k8s"],
            )
        ],
    )
    output = FitAnalysisOutput(
        job_id="J1",
        evidenced_missing_skills=[
            EvidenceClaim(
                claim="Kubernetes: evidenced",
                evidence_ids=["master-k8s"],
            ),
            EvidenceClaim(
                claim="Go: keyword",
                evidence_ids=["master-k8s"],
            ),
        ],
    )

    result, failures = post_validate(output, inp)

    assert [claim.claim.split(":")[0] for claim in result.evidenced_missing_skills] == [
        "Kubernetes"
    ]
    assert all(not claim.claim.startswith("Go:") for claim in result.genuine_gaps)
    assert any("unsupported keyword/skill 'Go'" in item for item in failures)


def test_semantically_unrelated_known_ids_are_removed_from_skill_claim() -> None:
    inp = _fit_post_base_input(
        required_skills=["Python"],
        evidence=[
            make_evidence(
                "resume-python",
                "resume",
                "Built production services in Python.",
                tags=["experience", "Python"],
            ),
            make_evidence(
                "resume-unrelated",
                "resume",
                "Led stakeholder planning and roadmap reviews.",
                tags=["experience", "leadership"],
            ),
        ],
    )
    output = FitAnalysisOutput(
        job_id="J1",
        aligned_skills=[
            EvidenceClaim(
                claim="Python: aligned",
                evidence_ids=["resume-python", "resume-unrelated"],
            )
        ],
    )

    result, repairs = post_validate(output, inp)

    assert result.aligned_skills[0].evidence_ids == [
        "job-J1-skill-001",
        "resume-python",
    ]
    assert any("resume-unrelated" in repair for repair in repairs)


def test_negated_skill_evidence_cannot_support_experience() -> None:
    inp = _fit_post_base_input(
        required_skills=["Go"],
        evidence=[
            make_evidence(
                "resume-negated-go",
                "resume",
                "I have never used Go in production.",
                tags=["experience"],
            )
        ],
    )
    output = FitAnalysisOutput(
        job_id="J1",
        aligned_skills=[
            EvidenceClaim(
                claim="Go: aligned",
                evidence_ids=["resume-negated-go"],
            )
        ],
    )

    result, repairs = post_validate(output, inp)

    assert result.aligned_skills == []
    assert [claim.claim.split(":")[0] for claim in result.genuine_gaps] == ["Go"]
    assert any("semantically unrelated evidence" in repair for repair in repairs)


def test_project_claim_keeps_only_the_compared_portfolio_record() -> None:
    portfolio = [
        make_portfolio_project("P1", "Relevant Project", technologies=["Python"]),
        make_portfolio_project("P2", "Unrelated Project", technologies=["Rust"]),
    ]
    inp = _fit_post_base_input(
        portfolio=portfolio,
        current=["Relevant Project"],
    )
    output = FitAnalysisOutput(
        job_id="J1",
        project_analysis=[
            EvidenceClaim(
                claim="Current project 'Relevant Project' aligns with this job.",
                evidence_ids=["portfolio-P1", "portfolio-P2"],
            )
        ],
    )

    result, repairs = post_validate(output, inp)

    assert result.project_analysis[0].evidence_ids == [
        "job-J1-description",
        "portfolio-P1",
    ]
    assert any("portfolio-P2" in repair for repair in repairs)


# --- test_tool_fit_analysis_prepass.py ---
"""Deterministic pre-pass: indexing, skill buckets, memory evidence, None years."""


def test_alias_and_evidence_indexing_canonicalizes_variants() -> None:
    items = [
        make_evidence(
            "m1", "master_skills", "Master skills: k8s, ML", tags=["k8s", "ML"]
        ),
        make_evidence("p1", "portfolio", "block", tags=["torch"]),
    ]
    index = build_evidence_index(items)
    assert index.has("kubernetes")
    assert index.has("machine learning")
    assert index.has("pytorch")
    assert index.sources_for("kubernetes") == {"master_skills"}
    assert index.sources_for("pytorch") == {"portfolio"}


def test_ordered_ids_prefers_resume_then_portfolio() -> None:
    items = [
        make_evidence("portfolio-P1", "portfolio", "b", tags=["Python"]),
        make_evidence(
            "resume-skills-001", "resume", "Resume skills: Python", tags=["Python"]
        ),
        make_evidence(
            "master-skills-001", "master_skills", "Master: Python", tags=["Python"]
        ),
    ]
    index = build_evidence_index(items)
    ordered = index.ordered_ids("python")
    assert ordered[0] == "resume-skills-001"
    assert ordered[1] == "master-skills-001"
    assert ordered[2] == "portfolio-P1"


_EDUCATION_LINE = (
    "Pacifica Institute of Science | Houston, TX | M.S. Data Science (3.64/4.0) | "
    "B.S. Math & Computer Science (3.44/4.0) | 2021--2023 | 2017--2021"
)


def test_education_line_can_ground_a_skill_claim() -> None:
    # Scan the entry text because its tag does not name its skills.
    items = [
        make_evidence(
            "resume-education-002", "resume", _EDUCATION_LINE, tags=["education"]
        )
    ]
    index = build_evidence_index(items, vocabulary={"data science"})

    assert index.has("data science")
    assert index.ids_for("data science") == ["resume-education-002"]
    assert index.sources_for("data science") == {"resume"}


def test_education_evidenced_skill_is_not_reported_as_a_gap() -> None:
    job = make_job(required_skills=["data science"])
    profile = make_profile(skills=[])
    items = [
        make_evidence(
            "resume-education-002", "resume", _EDUCATION_LINE, tags=["education"]
        )
    ]
    inp = make_input(job, profile, items)
    prepass = run_prepass(inp, build_evidence_index(items, vocabulary={"data science"}))

    assert [c.skill for c in prepass.genuine_gaps] == []
    grounded = [*prepass.aligned, *prepass.evidenced_missing]
    assert [c.skill for c in grounded] == ["data science"]
    assert grounded[0].evidence_ids == ["resume-education-002"]


def test_whole_resume_blob_is_not_text_scanned() -> None:
    # Scan short entries, not the full resume.
    blob = make_evidence(
        "resume-upload-001",
        "resume",
        "Avery Morgan. Summary: data science and kubernetes and pytorch everywhere. "
        * 20,
        tags=["resume"],
    )
    index = build_evidence_index([blob], vocabulary={"data science"})
    assert not index.has("data science")
    assert not index.has("kubernetes")


def test_skill_buckets_are_split_correctly() -> None:
    job = make_job(required_skills=["Python", "PyTorch", "Rust"])
    profile = make_profile(skills=["Python"])  # Only Python is on the resume.
    items = [
        make_evidence(
            "resume-skills-001", "resume", "Resume skills: Python", tags=["Python"]
        ),
        make_evidence("portfolio-P1", "portfolio", "b", tags=["PyTorch"]),
    ]
    prepass = run_prepass(make_input(job, profile, items), build_evidence_index(items))

    aligned = {c.skill for c in prepass.aligned}
    missing = {c.skill for c in prepass.evidenced_missing}
    gaps = {c.skill for c in prepass.genuine_gaps}
    assert aligned == {"Python"}
    assert missing == {"PyTorch"}  # Only the portfolio supports PyTorch.
    assert gaps == {"Rust"}  # Nothing supports Rust.


def test_memory_only_evidenced_skill_is_evidenced_missing_not_gap() -> None:
    job = make_job(required_skills=["GraphQL"])
    profile = make_profile(skills=["Python"])
    items = [
        make_evidence("mem-1", "memory", "skill: GraphQL", tags=["skill", "memory"])
    ]
    prepass = run_prepass(make_input(job, profile, items), build_evidence_index(items))

    assert [c.skill for c in prepass.evidenced_missing] == ["GraphQL"]
    assert prepass.genuine_gaps == []
    assert prepass.evidenced_missing[0].sources == {"memory"}


def test_seniority_verdict_reflects_year_shortfall() -> None:
    from src.tools.fit_analysis import fit_analysis as verdict

    assert seniority_verdict(6, 5) == verdict.MATCH
    assert seniority_verdict(4, 5) == verdict.PARTIAL  # Small shortfall.
    assert seniority_verdict(1, 5) == verdict.MISMATCH  # Large shortfall.
    assert seniority_verdict(None, 5) == verdict.PARTIAL  # Candidate years unknown.
    assert seniority_verdict(4, None) == verdict.MATCH  # No stated requirement.


def test_none_years_required_does_not_crash_and_is_not_inferred() -> None:
    job = make_job(required_skills=["Python"], years=None)
    profile = make_profile(skills=["Python"], years=4)
    items = [
        make_evidence(
            "resume-experience-001",
            "resume",
            "Senior AI Engineer at Acme",
            tags=["experience"],
        )
    ]
    prepass = run_prepass(make_input(job, profile, items), build_evidence_index(items))

    assert len(prepass.seniority) == 1
    claim = prepass.seniority[0]
    assert claim.notes is not None and "not specified" in claim.notes.lower()


# --- test_tool_fit_analysis_ranking.py ---
"""Regression pins for project relevance ranking against the repository's real data.

The deterministic ranking is the fallback path, so it must be defensible on its own
even when the LLM path is unavailable.
"""


_ROOT = Path(__file__).resolve().parents[1]
_JOBS = _ROOT / "data" / "jobs.csv"
_PORTFOLIO = _ROOT / "data" / "portfolio.txt"

pytestmark = pytest.mark.skipif(
    not (_JOBS.exists() and _PORTFOLIO.exists()),
    reason="repository data files not present",
)


def _fit_ranking_ranking(job_id: str):
    job = next(j for j in load_jobs_csv(str(_JOBS)) if j.job_id == job_id)
    portfolio = load_portfolio(str(_PORTFOLIO))
    job_text = sanitize_text(f"{job.title}. {job.description} {job.company_details}")
    return job, rank_projects(
        portfolio.projects, job, expand_canonical_skills(job.required_skills), job_text
    )


def test_j030_prefers_the_recommendation_project_over_medical_imaging() -> None:
    """A streaming-personalization role must rank ranking work above ultrasound imaging.

    Whole-phrase domain matching used to score the portfolio's "Recommendation and
    Ranking" project at zero against a posting requiring "recommendation systems",
    while an unrelated medical project won on the literal string "Deep Learning".
    """

    job, ranking = _fit_ranking_ranking("J030")
    assert "recommendation systems" in [s.lower() for s in job.required_skills]

    by_id = {item.project.project_id: item for item in ranking}
    assert by_id["P08"].score > by_id["P04"].score
    assert ranking[0].project.project_id == "P08"
    assert "Recommendation and Ranking" in by_id["P08"].domain_matches
    # Generic overlap alone does not count.
    assert by_id["P04"].domain_matches == []


def test_partial_phrase_overlap_needs_a_distinctive_word() -> None:
    # One distinctive word can carry a phrase.
    assert match_terms(
        ["Recommendation and Ranking"], "we build recommendation systems"
    ) == ["Recommendation and Ranking"]
    # Filler words cannot.
    assert match_terms(["Deep Learning"], "deep learning models") == []
    assert match_terms(["Computer Vision"], "a computer in the office") == []
    assert match_terms(["Computer Vision"], "vision transformers") == [
        "Computer Vision"
    ]
    # Singular and plural forms match.
    assert match_terms(["Decentralized Data Systems"], "a decentralized system") == [
        "Decentralized Data Systems"
    ]


# --- test_tool_fit_analysis_rationale.py ---
"""Constrained LLM rationale: used when consistent, template kept when it drifts."""


def _fit_rationale_seniority_claim() -> list[EvidenceClaim]:
    return [
        EvidenceClaim(
            claim="Seniority: ~4 years of experience vs job (5+ years expected).",
            evidence_ids=[],
            notes=verdict.tag(verdict.PARTIAL, "Close but below."),
        )
    ]


def test_seniority_rationale_used_when_consistent() -> None:
    out = rationale.enrich_seniority(
        _fit_rationale_seniority_claim(),
        lambda _system, _user: (
            "With about four years, the candidate is close to but below the "
            "five-plus years this role expects."
        ),
    )
    assert out[0].claim.startswith("With about four years")


def test_seniority_rationale_rejected_on_contradiction() -> None:
    out = rationale.enrich_seniority(
        _fit_rationale_seniority_claim(),
        lambda _system, _user: (
            "The candidate clearly exceeds the requirement for this role."
        ),
    )
    assert out[0].claim.startswith("Seniority:")  # Keep the template.


def _fit_rationale_swap_prepass_and_input():
    job = make_job(
        required_skills=["PyTorch", "RAG"], description="Generative AI with RAG."
    )
    profile = make_profile(
        skills=["Python"], resume_projects=["Vision Project", "Chatbot"]
    )
    portfolio = [
        make_portfolio_project(
            "P1",
            "Vision Project",
            technologies=["PyTorch"],
            domains=["Computer Vision"],
        ),
        make_portfolio_project("P2", "Chatbot", technologies=["Python"]),
        make_portfolio_project(
            "P3",
            "RAG Platform",
            technologies=["RAG", "PyTorch"],
            domains=["Generative AI"],
        ),
        make_portfolio_project("P4", "Distractor Project", technologies=["Go"]),
    ]
    items = [
        make_evidence(
            "resume-skills-001", "resume", "Resume skills: Python", tags=["Python"]
        )
    ]
    inp = make_input(
        job,
        profile,
        items,
        current_resume_projects=["Vision Project", "Chatbot"],
        portfolio_projects=portfolio,
    )
    return run_prepass(inp, build_evidence_index(items)), inp


def test_swap_rationale_used_when_naming_correct_project() -> None:
    prepass, inp = _fit_rationale_swap_prepass_and_input()
    assert prepass.swap is not None
    selected_swap = prepass.swap.swap
    assert selected_swap is not None  # A swap was selected.
    add = selected_swap.add_project
    out = rationale.enrich_swap(
        selected_swap,
        prepass,
        inp,
        lambda _system, _user: (
            f"Swap in '{add}' for its RAG technology, generative-AI domain, and industry fit."
        ),
    )
    assert out is not None
    assert out.rationale.startswith("Swap in")


def test_swap_rationale_rejected_when_it_names_a_different_project() -> None:
    prepass, inp = _fit_rationale_swap_prepass_and_input()
    assert prepass.swap is not None
    selected_swap = prepass.swap.swap
    assert selected_swap is not None
    original = selected_swap.rationale
    out = rationale.enrich_swap(
        selected_swap,
        prepass,
        inp,
        lambda _system, _user: (
            "Actually you should add the Distractor Project instead."
        ),
    )
    assert out is not None
    assert out.rationale == original  # Reject drift and keep the template.


# --- test_tool_fit_analysis_reasoning.py ---
"""Reasoning path selection: LLM stub, retry-on-invalid, and fallback."""


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


def _fit_reasoning_fixture():
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


_PROJECT_ANALYSIS_ITEM = {
    "claim": "Current project 'X' has limited alignment.",
    "evidence_ids": ["portfolio-P1"],
    "confidence": 0.7,
    "notes": "verdict=partial; limited overlap",
}


def test_project_analysis_accepts_valid_list() -> None:
    output = FitAnalysisOutput(
        job_id="J1", project_analysis=[_PROJECT_ANALYSIS_ITEM]
    )

    assert len(output.project_analysis) == 1
    assert isinstance(output.project_analysis[0], ProjectAnalysisItem)


def test_project_analysis_normalizes_single_dictionary() -> None:
    output = FitAnalysisOutput(
        job_id="J1", project_analysis=_PROJECT_ANALYSIS_ITEM  # type: ignore[arg-type]
    )

    assert len(output.project_analysis) == 1
    assert output.project_analysis[0].claim == _PROJECT_ANALYSIS_ITEM["claim"]


def test_project_analysis_normalizes_none_to_empty_list() -> None:
    output = FitAnalysisOutput(
        job_id="J1", project_analysis=None  # type: ignore[arg-type]
    )

    assert output.project_analysis == []


def test_project_analysis_rejects_invalid_scalar() -> None:
    with pytest.raises(ValueError):
        FitAnalysisOutput(
            job_id="J1", project_analysis="not a list"  # type: ignore[arg-type]
        )


def test_stub_llm_produces_llm_path() -> None:
    inp, prepass, index = _fit_reasoning_fixture()
    output, path, _ = reasoning.analyze(
        inp, prepass, index, complete_fn=lambda _system, _user: _VALID_JSON
    )
    assert path == "llm"
    assert output.aligned_skills[0].claim.startswith("Python")


def test_live_completion_requests_deepinfra_json_mode(monkeypatch) -> None:
    calls: dict[str, object] = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs) -> None:
            calls["init"] = kwargs

        def bind(self, **kwargs):
            calls["bind"] = kwargs
            return self

        def invoke(self, messages):
            calls["messages"] = messages
            return SimpleNamespace(content='{"job_id":"J1"}', usage_metadata={})

    monkeypatch.setattr("langchain_openai.ChatOpenAI", FakeChatOpenAI)
    monkeypatch.setattr(
        llm_client,
        "get_config",
        lambda: SimpleNamespace(
            llm_model="test-model",
            deepinfra_api_key="test-key",
            deepinfra_base_url="https://example.invalid/v1/openai",
        ),
    )

    result = llm_client.complete("system", "user")

    assert result == '{"job_id":"J1"}'
    assert calls["bind"] == {"response_format": {"type": "json_object"}}


def test_invalid_then_valid_triggers_single_retry() -> None:
    inp, prepass, index = _fit_reasoning_fixture()
    calls = {"n": 0}

    def flaky(system: str, user: str) -> str:
        calls["n"] += 1
        return "not json" if calls["n"] == 1 else _VALID_JSON

    output, path, _ = reasoning.analyze(inp, prepass, index, complete_fn=flaky)
    assert calls["n"] == 2
    assert path == "llm"
    assert output.job_id == "J1"


def test_retry_prompt_includes_validation_error_and_array_requirement() -> None:
    inp, prepass, index = _fit_reasoning_fixture()
    prompts: list[str] = []
    invalid = json.dumps(
        {
            **json.loads(_VALID_JSON),
            "project_analysis": "not a list",
        }
    )

    def invalid_then_valid(_system: str, user: str) -> str:
        prompts.append(user)
        return invalid if len(prompts) == 1 else _VALID_JSON

    output, path, _ = reasoning.analyze(
        inp, prepass, index, complete_fn=invalid_then_valid
    )

    assert path == "llm"
    assert output.job_id == "J1"
    assert len(prompts) == 2
    assert prompts[1] != prompts[0]
    assert "Validation error:" in prompts[1]
    assert "Input should be a valid list" in prompts[1]
    assert "project_analysis" in prompts[1]
    assert "MUST ALWAYS be a JSON array" in prompts[1]
    assert "Correct only the JSON shape" in prompts[1]
    assert "Return JSON only" in prompts[1]


def test_llm_exception_falls_back_deterministically() -> None:
    inp, prepass, index = _fit_reasoning_fixture()

    def boom(system: str, user: str) -> str:
        raise RuntimeError("network down")

    output, path, meta = reasoning.analyze(inp, prepass, index, complete_fn=boom)
    assert path == "fallback"
    assert meta.get("llm_error") == "RuntimeError"
    assert [c.claim.split(":")[0] for c in output.aligned_skills] == ["Python"]


def test_no_model_configured_uses_fallback_offline(monkeypatch) -> None:
    monkeypatch.setattr(llm_client, "model_configured", lambda: False)
    inp, prepass, index = _fit_reasoning_fixture()
    output, path, _ = reasoning.analyze(inp, prepass, index)
    assert path == "fallback"
    assert output.job_id == "J1"


# --- test_tool_fit_analysis_render.py ---
"""Renderer: verdict-derived markers, inline citations, swap/no-swap branches."""


def _fit_render_sample() -> FitAnalysisOutput:
    return FitAnalysisOutput(
        job_id="J1",
        aligned_skills=[
            EvidenceClaim(
                claim="Python: aligned",
                evidence_ids=["e1"],
                notes=verdict.tag(verdict.MATCH),
            )
        ],
        evidenced_missing_skills=[
            EvidenceClaim(
                claim="deep learning: evidenced",
                evidence_ids=["portfolio-P04"],
                notes=verdict.tag(verdict.MISSING),
            )
        ],
        genuine_gaps=[
            EvidenceClaim(claim="Rust: gap", notes=verdict.tag(verdict.MISMATCH))
        ],
    )


def test_render_contains_all_sections() -> None:
    text = render_fit_analysis(_fit_render_sample(), make_job())
    assert "Tell me why this job is a good fit for me." in text
    for section in (
        "Relevant Experience",
        "Seniority",
        "Education",
        "Core Skills",
        "Projects",
    ):
        assert f"## {section}" in text
    assert "✅ Python: aligned" in text
    assert "❌ Rust" in text  # Gaps render with the skill name.


def test_missing_group_marker_is_configurable() -> None:
    output = _fit_render_sample()
    assert "❌ deep learning" in render_fit_analysis(
        output, make_job(), missing_marker="❌"
    )
    assert "➕ deep learning" in render_fit_analysis(
        output, make_job(), missing_marker="➕"
    )


def test_missing_skill_shows_human_readable_source_inline() -> None:
    labels = {"portfolio-P04": 'used in "IVUS Analysis"'}
    text = render_fit_analysis(_fit_render_sample(), make_job(), source_labels=labels)
    assert 'deep learning (used in "IVUS Analysis")' in text
    assert "_[evidence: portfolio-P04]_" in text  # IDs follow the readable citation.


def test_partial_renders_as_cross_not_positive() -> None:
    # A seniority shortfall renders as a failure.
    output = FitAnalysisOutput(
        job_id="J1",
        seniority=[
            EvidenceClaim(
                claim="Seniority: 4y vs 5+", notes=verdict.tag(verdict.PARTIAL)
            )
        ],
    )
    text = render_fit_analysis(output, make_job())
    assert "❌ Seniority: 4y vs 5+" in text
    assert "✅" not in text  # A partial result cannot pass.
    assert "⚠️" not in text  # The report uses no warning marker.


def test_render_states_no_swap_branch_explicitly() -> None:
    text = render_fit_analysis(
        FitAnalysisOutput(job_id="J1", project_swap=None), make_job()
    )
    assert "No swap recommended" in text


def test_render_shows_recommended_swap() -> None:
    output = FitAnalysisOutput(
        job_id="J1",
        project_swap=ProjectSwap(
            remove_project="Old",
            add_project="New",
            rationale="stronger",
            evidence_ids=["portfolio-P2"],
        ),
    )
    text = render_fit_analysis(output, make_job())
    assert 'replace "Old" with "New"' in text
    assert "stronger" in text
