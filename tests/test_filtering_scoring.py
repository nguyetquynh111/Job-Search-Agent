"""Consolidated tests for this domain."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from pypdf import PdfWriter
from src.agent import CandidatePreferences
from src.agent import CandidateProfile, EvidenceItem
from src.agent import ChangeLogEntry, EvidenceClaim, ProjectSwap
from src.agent import Job
from src.agent import build_job_evidence
from src.tools.cover_letter import cover_letter as cover
from src.tools.cover_letter.cover_letter import GenerateCoverLetterInput
from src.tools.filtering_scoring.filtering import FilterJobsInput, FilterJobsOutput
from src.tools.filtering_scoring.filtering import run_filtering_tool
from src.tools.filtering_scoring.scoring import ScoreJobsInput, ScoreJobsOutput
from src.tools.filtering_scoring.scoring import run_scoring_tool
from src.tools.fit_analysis.fit_analysis import FitAnalysisOutput
from src.tools.resume_tailoring import resume_tailoring as tailoring
from src.tools.resume_tailoring.resume_tailoring import (
    LatexStructureError,
    parse_resume_structure,
)
from src.tools.resume_tailoring.resume_tailoring import TailorResumeInput
from src.tracing.langfuse import TraceManager
import pytest

# --- test_tool_filtering.py ---
"""Contract-focused unit tests for the ``filter_jobs`` tool."""


def _tool_filtering_job(job_id: str, **overrides: object) -> Job:
    payload = {
        "job_id": job_id,
        "title": "Machine Learning Engineer",
        "company": "Acme AI",
        "industry_domain": "Software / Machine Learning",
        "location": "Austin, TX",
        "description": "Build ML systems.",
        "required_skills": ["Python", "PyTorch"],
        "url": "",
    }
    payload.update(overrides)
    return Job.model_validate(payload)


def _tool_filtering_run(jobs: list[Job], **pref_kwargs: object) -> FilterJobsOutput:
    prefs = CandidatePreferences.model_validate(pref_kwargs)
    return run_filtering_tool(FilterJobsInput(jobs=jobs, preferences=prefs))


def test_every_job_appears_in_exactly_one_output_list() -> None:
    jobs = [
        _tool_filtering_job("A", location="Austin, TX"),
        _tool_filtering_job("B", location="Berlin, Germany"),
    ]
    out = _tool_filtering_run(jobs, preferred_locations=["Austin, TX"])
    accepted = {job.job_id for job in out.accepted_jobs}
    rejected = {rj.job.job_id for rj in out.rejected_jobs}
    assert accepted | rejected == {"A", "B"}
    assert not (accepted & rejected)


def test_no_preferences_accepts_everything() -> None:
    jobs = [
        _tool_filtering_job("A", location="Nowhere"),
        _tool_filtering_job("B", company="Anything"),
    ]
    out = _tool_filtering_run(jobs)  # Empty preferences disable every rule.
    assert len(out.accepted_jobs) == 2
    assert not out.rejected_jobs


def test_location_filter_rejects_onsite_outside_preferred() -> None:
    out = _tool_filtering_run(
        [_tool_filtering_job("A", location="Seattle, WA")],
        preferred_locations=["Houston, TX"],
    )
    assert not out.accepted_jobs
    assert "Seattle" in out.rejected_jobs[0].reasons[0]


def test_location_filter_matches_city_token() -> None:
    out = _tool_filtering_run(
        [_tool_filtering_job("A", location="Houston, TX (Hybrid)")],
        preferred_locations=["Houston, TX"],
    )
    assert [job.job_id for job in out.accepted_jobs] == ["A"]


def test_remote_job_always_passes_location() -> None:
    out = _tool_filtering_run(
        [_tool_filtering_job("A", location="Fully Remote", remote=True)],
        preferred_locations=["Houston, TX"],
    )
    assert [job.job_id for job in out.accepted_jobs] == ["A"]


def test_nationwide_job_passes_when_candidate_accepts_remote() -> None:
    out = _tool_filtering_run(
        [_tool_filtering_job("A", location="United States")],
        preferred_locations=["Houston, TX", "Remote, US"],
    )
    assert [job.job_id for job in out.accepted_jobs] == ["A"]


def test_remote_only_rejects_onsite() -> None:
    out = _tool_filtering_run(
        [_tool_filtering_job("A", location="Austin, TX", remote=False)],
        remote_only=True,
    )
    assert not out.accepted_jobs
    assert "remote" in out.rejected_jobs[0].reasons[0].lower()


def test_experience_filter_rejects_underqualified() -> None:
    out = _tool_filtering_run(
        [_tool_filtering_job("A", years_experience_required=8)],
        years_of_experience=4,
    )
    assert not out.accepted_jobs
    assert "8+ years" in out.rejected_jobs[0].reasons[0]


def test_experience_filter_accepts_when_candidate_meets_minimum() -> None:
    out = _tool_filtering_run(
        [_tool_filtering_job("A", years_experience_required=3)],
        years_of_experience=4,
    )
    assert [job.job_id for job in out.accepted_jobs] == ["A"]


def test_company_exclusion_is_case_and_space_insensitive() -> None:
    out = _tool_filtering_run(
        [_tool_filtering_job("A", company="Booz Allen Hamilton")],
        excluded_companies=["booz-allen  hamilton"],
    )
    assert not out.accepted_jobs
    assert "exclusion list" in out.rejected_jobs[0].reasons[0]


def test_excluded_keyword_rejects_matching_posting() -> None:
    out = _tool_filtering_run(
        [_tool_filtering_job("A", title="Blockchain ML Engineer")],
        excluded_keywords=["blockchain"],
    )
    assert not out.accepted_jobs
    assert "blockchain" in out.rejected_jobs[0].reasons[0].lower()


def test_target_title_filter_rejects_unrelated_title() -> None:
    out = _tool_filtering_run(
        [_tool_filtering_job("A", title="Senior Database Administrator")],
        target_job_titles=["AI Engineer", "Machine Learning Engineer"],
    )
    assert not out.accepted_jobs
    assert "target titles" in out.rejected_jobs[0].reasons[0]


def test_target_title_filter_keeps_ai_ml_titles() -> None:
    out = _tool_filtering_run(
        [_tool_filtering_job("A", title="Computer Vision AI Engineer")],
        target_job_titles=["AI Engineer"],
    )
    assert [job.job_id for job in out.accepted_jobs] == ["A"]


def test_rejected_job_collects_all_failing_reasons() -> None:
    out = _tool_filtering_run(
        [_tool_filtering_job("A", location="Seattle, WA", years_experience_required=9)],
        preferred_locations=["Houston, TX"],
        years_of_experience=4,
    )
    reasons = out.rejected_jobs[0].reasons
    assert len(reasons) >= 2  # Both failures are reported.


def test_filtering_is_deterministic_and_does_not_mutate_jobs() -> None:
    jobs = [
        _tool_filtering_job("A", location="Seattle, WA"),
        _tool_filtering_job("B", location="Houston, TX"),
    ]
    first = _tool_filtering_run(jobs, preferred_locations=["Houston, TX"])
    second = _tool_filtering_run(jobs, preferred_locations=["Houston, TX"])
    assert [j.job_id for j in first.accepted_jobs] == [
        j.job_id for j in second.accepted_jobs
    ]
    assert jobs[0].location == "Seattle, WA"  # Inputs stay unchanged.


# --- test_tool_scoring.py ---
"""Contract-focused unit tests for the deterministic ``score_jobs`` tool."""


def _tool_scoring_job(job_id: str, **overrides: object) -> Job:
    payload = {
        "job_id": job_id,
        "title": "Machine Learning Engineer",
        "company": "Acme AI",
        "industry_domain": "Software / Machine Learning",
        "location": "Austin, TX",
        "description": "Build ML systems.",
        "required_skills": ["Python", "PyTorch", "Computer Vision"],
        "years_experience_required": 3,
        "url": "",
    }
    payload.update(overrides)
    return Job.model_validate(payload)


def _tool_scoring_profile(**overrides: object) -> CandidateProfile:
    payload = {
        "candidate_id": "c1",
        "name": "Test Candidate",
        "preferences": {"years_of_experience": 4},
        "skills": ["Python", "PyTorch"],
        "master_skills": ["Computer Vision", "NLP"],
    }
    payload.update(overrides)
    return CandidateProfile.model_validate(payload)


def _tool_scoring_run(
    jobs: list[Job], profile: CandidateProfile, **evidence: Any
) -> ScoreJobsOutput:
    return run_scoring_tool(
        ScoreJobsInput(jobs=jobs, candidate_profile=profile, **evidence)
    )


def test_scores_are_bounded_and_ranking_is_descending() -> None:
    jobs = [
        _tool_scoring_job(
            "HIGH", required_skills=["Python", "PyTorch", "Computer Vision"]
        ),
        _tool_scoring_job(
            "LOW",
            required_skills=["Rust", "Kubernetes", "Go"],
            industry_domain="Fintech",
        ),
    ]
    out = _tool_scoring_run(jobs, _tool_scoring_profile())
    scores = [sj.score for sj in out.ranked_jobs]
    assert all(0 <= s <= 100 for s in scores)
    assert scores == sorted(scores, reverse=True)
    assert out.ranked_jobs[0].job.job_id == "HIGH"


def test_top_3_selected_automatically() -> None:
    jobs = [_tool_scoring_job(f"J{i}", required_skills=["Python"]) for i in range(5)]
    out = _tool_scoring_run(jobs, _tool_scoring_profile())
    assert out.top_3_job_ids == [sj.job.job_id for sj in out.ranked_jobs[:3]]
    assert len(out.top_3_job_ids) == 3


def test_full_skill_and_experience_match_scores_high() -> None:
    # Full skill, experience, domain, and location match.
    job = _tool_scoring_job(
        "A", industry_domain="Computer Vision", location="Remote", remote=True
    )
    out = _tool_scoring_run([job], _tool_scoring_profile())
    assert out.ranked_jobs[0].score >= 90


def test_scoring_is_deterministic() -> None:
    jobs = [_tool_scoring_job("A"), _tool_scoring_job("B", required_skills=["Rust"])]
    profile = _tool_scoring_profile()
    first = _tool_scoring_run(jobs, profile)
    second = _tool_scoring_run(jobs, profile)
    assert [(s.job.job_id, s.score) for s in first.ranked_jobs] == [
        (s.job.job_id, s.score) for s in second.ranked_jobs
    ]
    assert first.top_3_job_ids == second.top_3_job_ids


def test_evidence_ids_always_exist_in_input() -> None:
    evidence = [
        EvidenceItem(
            evidence_id="port-1",
            source="portfolio",
            text="PyTorch project",
            tags=["PyTorch"],
        ),
    ]
    out = _tool_scoring_run(
        [_tool_scoring_job("A")], _tool_scoring_profile(), portfolio_evidence=evidence
    )
    valid = {"port-1"}
    for sj in out.ranked_jobs:
        assert set(sj.evidence_ids).issubset(valid)
    # Portfolio evidence supports the PyTorch match.
    assert "port-1" in out.ranked_jobs[0].evidence_ids


def test_category_requirement_satisfied_by_member_skill() -> None:
    # Pinecone satisfies the broader vector database skill.
    job = _tool_scoring_job(
        "A", required_skills=["vector databases"], years_experience_required=1
    )
    profile = _tool_scoring_profile(master_skills=["Pinecone"])
    out = _tool_scoring_run([job], profile)
    assert "1/1 required skills evidenced" in out.ranked_jobs[0].rationale


def test_memory_fact_counts_as_evidence() -> None:
    job = _tool_scoring_job(
        "A",
        required_skills=["GraphQL"],
        years_experience_required=1,
        industry_domain="",
    )
    without = _tool_scoring_run([job], _tool_scoring_profile())
    memory = [
        EvidenceItem(
            evidence_id="mem-1", source="memory", text="GraphQL", tags=["GraphQL"]
        )
    ]
    with_mem = _tool_scoring_run([job], _tool_scoring_profile(), memory_evidence=memory)
    assert with_mem.ranked_jobs[0].score > without.ranked_jobs[0].score
    assert "mem-1" in with_mem.ranked_jobs[0].evidence_ids


def test_ranking_is_stable_for_ties() -> None:
    # Tied jobs keep their input order.
    jobs = [
        _tool_scoring_job("FIRST"),
        _tool_scoring_job("SECOND"),
        _tool_scoring_job("THIRD"),
    ]
    out = _tool_scoring_run(jobs, _tool_scoring_profile())
    assert [sj.job.job_id for sj in out.ranked_jobs] == ["FIRST", "SECOND", "THIRD"]


def test_documented_weighted_formula_has_an_exact_result() -> None:
    """Pin the arithmetic so a future LLM or heuristic cannot replace it."""

    job = _tool_scoring_job(
        "EXACT",
        required_skills=["Python", "Rust"],
        years_experience_required=4,
        industry_domain="Computer Vision / Fintech",
        location="Chicago, IL",
        remote=False,
    )
    profile = _tool_scoring_profile(
        skills=["Python"],
        master_skills=["Computer Vision"],
        preferences={"years_of_experience": 2},
    )

    result = _tool_scoring_run([job], profile).ranked_jobs[0]

    # .5*55 skills + .5*25 experience + .65*15 domain + .4*5 location
    assert result.score == 51.8


# --- test_tool_cover_letter.py ---
"""Production cover-letter behavior."""


def _tool_cover_letter_input(resume_path: str) -> GenerateCoverLetterInput:
    return GenerateCoverLetterInput(
        job=Job(
            job_id="J901",
            title="Machine Learning Engineer",
            company="Example AI",
            industry_domain="Machine Learning",
            location="Remote",
            description="Build ML systems.",
            required_skills=["Python"],
            company_details="Example AI builds production machine-learning systems.",
        ),
        approved_resume_path=resume_path,
        candidate_evidence=[
            EvidenceItem(
                evidence_id="resume-experience-001",
                source="resume",
                text="Built production Python services with measurable reliability gains.",
                tags=["experience", "Python"],
            )
        ],
    )


def test_cover_letter_requires_an_actual_approved_resume(tmp_path: Path) -> None:
    result = cover.run_cover_letter_tool(
        _tool_cover_letter_input(str(tmp_path / "missing.pdf"))
    )

    assert result.page_count == 0
    assert result.output_tex_path == ""
    assert result.output_pdf_path == ""
    assert result.errors


def test_cover_letter_uses_evidence_and_reports_real_paths(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setattr(
        cover,
        "_read_approved_resume",
        lambda path: (
            "Avery Morgan\nHouston, TX | avery@example.com | github.com/avery",
            [],
        ),
    )

    def fake_compile(letter, job, tex_path: Path, pdf_path: Path):
        tex_path.write_text(cover._render_tex(letter, job, 0), encoding="utf-8")
        pdf_path.write_bytes(b"%PDF-1.4 test")
        return 1, []

    monkeypatch.setattr(cover, "_compile_one_page", fake_compile)
    result = cover.run_cover_letter_tool(_tool_cover_letter_input("approved.pdf"))

    assert result.page_count == 1
    assert Path(result.output_tex_path).is_file()
    assert Path(result.output_pdf_path).is_file()
    assert set(result.evidence_used) == {
        "job-J901-title",
        "job-J901-description",
        "job-J901-company-details",
        "job-J901-skill-001",
        "resume-experience-001",
    }
    assert not result.errors


def test_negated_skill_evidence_is_not_used_in_cover_letter() -> None:
    negated = EvidenceItem(
        evidence_id="resume-negated-go",
        source="resume",
        text="I have never used Go.",
        tags=["experience", "Go"],
    )

    skills, evidence_ids = cover._match_required_skills(
        ["Go"], {negated.evidence_id: negated}
    )

    assert skills == []
    assert evidence_ids == set()


# --- test_tool_resume_latex_structure.py ---
"""Structural resume targeting without template-specific annotations."""


def _input(source: Path) -> TailorResumeInput:
    return _tool_resume_tailoring_input(source)


def _tool_resume_latex_ordinary_resume(*, second_summary: bool = False) -> str:
    duplicate = (
        "\\section{Professional Summary}\nDuplicate.\n" if second_summary else ""
    )
    return rf"""
\documentclass{{article}}
\begin{{document}}
\section{{Summary}}
Engineer building reliable Python systems.
{duplicate}
\section{{Professional Experience}}
\begin{{itemize}}
  \item Built Python forecasting services for weekly planning.
  \item Deployed containerized APIs with Docker and Kubernetes.
  \item Improved data quality checks for production pipelines.
\end{{itemize}}
\section{{Technical Skills}}
\begin{{itemize}}
  \item \textbf{{Languages:}} Python, SQL
\end{{itemize}}
\section{{Selected Projects}}
\begin{{itemize}}
  \item \textbf{{Forecasting Dashboard}} — Python
  \begin{{itemize}}
    \item Built weekly demand forecasts.
  \end{{itemize}}
\end{{itemize}}
\end{{document}}
""".strip()


def test_parser_accepts_alternate_sections_and_standard_items() -> None:
    source = _tool_resume_latex_ordinary_resume()

    structure = parse_resume_structure(source, require_projects=True)

    assert structure.sections["summary"].title == "summary"
    assert structure.sections["experience"].title == "professional experience"
    assert structure.sections["skills"].title == "technical skills"
    assert structure.sections["projects"].title == "selected projects"
    assert [item.command for item in structure.experience_bullets] == [
        "item",
        "item",
        "item",
    ]
    assert [item.content(source) for item in structure.experience_bullets][
        1
    ].startswith("Deployed containerized")
    assert structure.project_entries[0].name == "Forecasting Dashboard"


def test_parser_accepts_resume_item_macros_without_annotations() -> None:
    source = Path("data/resume.tex").read_text(encoding="utf-8")

    structure = parse_resume_structure(source, require_projects=True)

    assert len(structure.experience_bullets) == 4
    assert all(item.command == "resumeItem" for item in structure.experience_bullets)
    assert len(structure.project_entries) == 2


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (
            _tool_resume_latex_ordinary_resume().replace(
                "\\section{Summary}", "\\section{Profile}"
            ),
            "summary",
        ),
        (
            _tool_resume_latex_ordinary_resume(second_summary=True),
            "Multiple plausible summary",
        ),
    ],
)
def test_parser_rejects_missing_or_ambiguous_required_sections(
    source: str, message: str
) -> None:
    with pytest.raises(LatexStructureError, match=message):
        parse_resume_structure(source)


def test_evidenced_skill_is_added_to_existing_skills_item(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = Path("data/resume.tex").read_text(encoding="utf-8")
    original = original.replace(", Kubernetes", "")
    source = tmp_path / "source.tex"
    source.write_text(original, encoding="utf-8")
    payload = _input(source)
    payload = payload.model_copy(
        update={
            "fit_analysis": payload.fit_analysis.model_copy(
                update={"project_swap": None}
            )
        }
    )
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "outputs"))

    def compile_once(text: str, tex_path: Path, pdf_path: Path):
        tex_path.write_text(text, encoding="utf-8")
        pdf_path.write_bytes(b"%PDF-1.4 test")
        return 1, [], text

    monkeypatch.setattr(tailoring, "_compile_one_page", compile_once)

    result = tailoring.run_resume_tailoring_tool(payload)

    assert result.status == "OK"
    assert sum(change.section == "skills" for change in result.change_log) == 1
    skill_change = next(
        change for change in result.change_log if change.section == "skills"
    )
    assert "Kubernetes" not in skill_change.before_text
    assert "Kubernetes" in skill_change.after_text
    assert skill_change.evidence_ids
    assert skill_change.source_location.startswith("Skills item")
    assert skill_change.validation_result.startswith("validated")


def test_project_swap_preserves_existing_entry_skeleton() -> None:
    source = Path("data/resume.tex").read_text(encoding="utf-8")
    structure = parse_resume_structure(source, require_projects=True)
    payload = _input(Path("data/resume.tex"))
    evidence = {item.evidence_id: item for item in payload.candidate_evidence}

    edit, record = tailoring._plan_project_swap(source, structure, payload, evidence)

    assert edit is not None
    before = edit.target.text(source)
    after = edit.replacement
    assert record.name == "AI CRM Platform"
    for command in (
        "\\resumeEntry",
        "\\resumeItemListStart",
        "\\resumeItem",
        "\\resumeItemListEnd",
    ):
        assert before.count(command) == after.count(command)
    assert "Cardiovascular Flow and Stenosis Analysis" not in after
    assert "AI CRM Platform" in after


def test_overflow_revision_changes_content_not_layout() -> None:
    original = Path("data/resume.tex").read_text(encoding="utf-8")
    current = original.replace(
        "AI/ML engineer and Ph.D. researcher",
        "Candidate targeting machine learning roles. "
        "AI/ML engineer and Ph.D. researcher",
    )
    structure = parse_resume_structure(current)
    edits = []
    for ordinal in (1, 2):
        item = structure.experience_bullets[ordinal - 1]
        edits.append(
            tailoring.SourceEdit(
                item.content_range,
                "Highlighted relevant work that " + item.content(current),
                "experience",
                f"Experience bullet {ordinal}",
            )
        )
    current = tailoring._apply_source_edits(current, edits)

    revised = tailoring._revise_edited_content(current, original, [1, 2], 1)

    assert "\\documentclass[letterpaper,11pt]{article}" in revised
    assert "\\addtolength{\\textwidth}{1in}" in revised
    assert "\\titlespacing*{\\section}{0pt}{8pt}{4pt}" in revised
    tailoring._assert_exactly_two_experience_bullets_changed(original, revised)


def test_compile_retries_overflow_with_content_revision_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tex_path = tmp_path / "resume.tex"
    pdf_path = tmp_path / "resume.pdf"
    calls = 0

    def compile_pdf(_: Path) -> list[str]:
        nonlocal calls
        calls += 1
        writer = PdfWriter()
        for _page_index in range(2 if calls == 1 else 1):
            writer.add_blank_page(width=612, height=792)
        with pdf_path.open("wb") as handle:
            writer.write(handle)
        return []

    source = (
        "\\documentclass[11pt]{article}\n"
        "\\begin{document}\nlong newly edited summary\n\\end{document}\n"
    )
    monkeypatch.setattr(tailoring, "_run_pdflatex", compile_pdf)

    pages, errors, compiled = tailoring._compile_one_page(
        source,
        tex_path,
        pdf_path,
        revision=lambda working, _: working.replace(
            "long newly edited summary", "short summary"
        ),
    )

    assert pages == 1
    assert errors == []
    assert calls == 2
    assert "short summary" in compiled
    assert compiled.startswith("\\documentclass[11pt]{article}")


# --- test_tool_resume_tailoring.py ---
"""Production resume-tailoring behavior."""


def _tool_resume_tailoring_job() -> Job:
    return Job(
        job_id="J900",
        title="Machine Learning Engineer",
        company="Example AI",
        industry_domain="Machine Learning",
        location="Remote",
        description="Build production ML systems.",
        required_skills=["Python", "Kubernetes", "Go"],
    )


def _tool_resume_tailoring_portfolio_text() -> str:
    return """PROJECT_ID: P02
PROJECT_NAME: AI CRM Platform
PERIOD: 2024-2025
ROLE: Senior AI Engineer
DOMAIN: Predictive Analytics; NLP
TECH_STACK: Python; FastAPI; Docker; Kubernetes; AWS SageMaker
SUMMARY: Engineered an AI-powered CRM platform for sales prioritization.
"""


def _tool_resume_tailoring_input(source: Path) -> TailorResumeInput:
    evidence = [
        EvidenceItem(
            evidence_id="resume-experience-001",
            source="resume",
            text=(
                "Graduate Research Assistant | Built a visual-QC workflow over "
                "1.7M segmentation/tracking predictions; achieved 90.5% test "
                "accuracy and 0.764 F1 using out-of-fold evaluation and targeted "
                "error analysis with Python and computer vision."
            ),
            tags=["experience", "Python", "Computer Vision"],
        ),
        EvidenceItem(
            evidence_id="resume-experience-002",
            source="resume",
            text=(
                "Senior AI Engineer | Delivered production Python services with "
                "Docker and Kubernetes."
            ),
            tags=["experience", "Python", "Docker", "Kubernetes"],
        ),
        EvidenceItem(
            evidence_id="master-skill-kubernetes",
            source="master_skills",
            text="Skill: Kubernetes",
            tags=["Kubernetes"],
        ),
        EvidenceItem(
            evidence_id="portfolio-P02",
            source="portfolio",
            text=_tool_resume_tailoring_portfolio_text(),
            tags=["Python", "Kubernetes", "Predictive Analytics"],
            metadata={"project_id": "P02"},
        ),
    ]
    fit = FitAnalysisOutput(
        job_id="J900",
        aligned_skills=[
            EvidenceClaim(
                claim="Python: already demonstrated.",
                evidence_ids=["resume-experience-001", "resume-experience-002"],
            )
        ],
        evidenced_missing_skills=[
            EvidenceClaim(
                claim="Kubernetes: safe to add.",
                evidence_ids=["master-skill-kubernetes"],
            )
        ],
        genuine_gaps=[EvidenceClaim(claim="Go: no supporting evidence.")],
        project_swap=ProjectSwap(
            remove_project="Cardiovascular Flow and Stenosis Analysis",
            add_project="AI CRM Platform",
            rationale="Closer production-ML match.",
            evidence_ids=["portfolio-P02"],
        ),
    )
    return TailorResumeInput(
        job=_tool_resume_tailoring_job(),
        fit_analysis=fit,
        source_resume_tex_path=str(source),
        candidate_evidence=evidence,
    )


def test_tailoring_changes_only_two_targeted_experience_bullets(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source.tex"
    original = Path("data/resume.tex").read_text(encoding="utf-8")
    source.write_text(original, encoding="utf-8")
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "outputs"))

    def fake_compile(text: str, tex_path: Path, pdf_path: Path):
        tex_path.write_text(text, encoding="utf-8")
        pdf_path.write_bytes(b"%PDF-1.4 test")
        return 1, [], text

    monkeypatch.setattr(tailoring, "_compile_one_page", fake_compile)
    result = tailoring.run_resume_tailoring_tool(_tool_resume_tailoring_input(source))

    assert result.status == "OK"
    assert result.page_count == 1
    assert source.read_text(encoding="utf-8") == original
    tailored = Path(result.output_tex_path).read_text(encoding="utf-8")
    assert "Kubernetes" in tailored
    assert "Go, " not in tailored
    assert "AI CRM Platform" in tailored
    assert "Cardiovascular Flow and Stenosis Analysis" not in tailored
    assert sum(entry.section == "experience" for entry in result.change_log) == 2
    known_evidence = {
        item.evidence_id
        for item in [
            *_tool_resume_tailoring_input(source).candidate_evidence,
            *build_job_evidence(_tool_resume_tailoring_job()),
        ]
    }
    assert {entry.section for entry in result.change_log} <= {
        "summary",
        "experience",
        "skills",
        "projects",
    }
    assert all(entry.before_text != entry.after_text for entry in result.change_log)
    assert all(entry.reason.strip() for entry in result.change_log)
    assert all(
        entry.evidence_ids and set(entry.evidence_ids) <= known_evidence
        for entry in result.change_log
    )
    before_bullets = parse_resume_structure(original).experience_bullets
    after_bullets = parse_resume_structure(tailored).experience_bullets
    assert len(before_bullets) == len(after_bullets)
    assert (
        sum(
            before.content(original) != after.content(tailored)
            for before, after in zip(before_bullets, after_bullets, strict=True)
        )
        == 2
    )


def test_review_preference_keeps_cardiovascular_as_first_project(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source.tex"
    source.write_text(Path("data/resume.tex").read_text(encoding="utf-8"))
    payload = _tool_resume_tailoring_input(source)
    cardiovascular = EvidenceItem(
        evidence_id="portfolio-cardiovascular",
        source="portfolio",
        text=(
            "PROJECT_NAME: Cardiovascular Flow and Stenosis Analysis\n"
            "SUMMARY: Analyzed cardiovascular flow and stenosis."
        ),
        tags=["Medical Computer Vision"],
        metadata={
            "project_name": "Cardiovascular Flow and Stenosis Analysis",
        },
    )
    payload = payload.model_copy(
        update={
            "candidate_evidence": [*payload.candidate_evidence, cardiovascular],
            "revision_feedback": (
                "Candidate prefers Cardiovascular Flow and Stenosis Analysis "
                "to be the first project."
            ),
        }
    )
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "outputs"))

    def fake_compile(text: str, tex_path: Path, pdf_path: Path):
        tex_path.write_text(text, encoding="utf-8")
        pdf_path.write_bytes(b"%PDF-1.4 test")
        return 1, [], text

    monkeypatch.setattr(tailoring, "_compile_one_page", fake_compile)

    result = tailoring.run_resume_tailoring_tool(payload)

    assert result.status == "OK"
    assert result.revision_feedback_satisfied is True
    assert any("first project" in check and "met" in check for check in result.revision_feedback_checks)
    tailored = Path(result.output_tex_path).read_text(encoding="utf-8")
    projects = parse_resume_structure(tailored, require_projects=True).project_entries
    assert projects[0].name == "Cardiovascular Flow and Stenosis Analysis"
    assert "AI CRM Platform" not in tailored


def test_review_should_be_1st_project_reorders_requested_project(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source.tex"
    source.write_text(Path("data/resume.tex").read_text(encoding="utf-8"))
    payload = _tool_resume_tailoring_input(source)
    chatbot = EvidenceItem(
        evidence_id="portfolio-chatbot",
        source="portfolio",
        text=(
            "PROJECT_NAME: No-Code LLM Chatbot Builder\n"
            "SUMMARY: Built an agentic RAG chatbot platform."
        ),
        tags=["Agentic AI", "RAG"],
        metadata={"project_name": "No-Code LLM Chatbot Builder"},
    )
    payload = payload.model_copy(
        update={
            "candidate_evidence": [*payload.candidate_evidence, chatbot],
            "revision_feedback": (
                "Not enough agentic skills. "
                "No-Code LLM Chatbot Builder should be the 1st project"
            ),
        }
    )
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "outputs"))

    def fake_compile(text: str, tex_path: Path, pdf_path: Path):
        tex_path.write_text(text, encoding="utf-8")
        pdf_path.write_bytes(b"%PDF-1.4 test")
        return 1, [], text

    monkeypatch.setattr(tailoring, "_compile_one_page", fake_compile)

    result = tailoring.run_resume_tailoring_tool(payload)

    assert result.status == "OK"
    assert any(
        'first project "No-Code LLM Chatbot Builder": met' == check
        for check in result.revision_feedback_checks
    )
    tailored = Path(result.output_tex_path).read_text(encoding="utf-8")
    projects = parse_resume_structure(tailored, require_projects=True).project_entries
    assert projects[0].name == "No-Code LLM Chatbot Builder"
    assert any(change.change_id.endswith("-project-order") for change in result.change_log)
    assert not any(change.change_id.endswith("-project-swap") for change in result.change_log)


def test_tailoring_reports_compile_failure_honestly(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source.tex"
    source.write_text(Path("data/resume.tex").read_text(encoding="utf-8"))
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setattr(
        tailoring,
        "_compile_one_page",
        lambda text, _tex, _pdf: (None, ["pdflatex unavailable"], text),
    )

    result = tailoring.run_resume_tailoring_tool(_tool_resume_tailoring_input(source))

    assert result.status == "ERROR"
    assert result.page_count == 0
    assert result.output_pdf_path == ""
    assert result.errors == ["pdflatex unavailable"]


def test_project_swap_requires_real_portfolio_evidence(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source.tex"
    source.write_text(Path("data/resume.tex").read_text(encoding="utf-8"))
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "outputs"))
    payload = _tool_resume_tailoring_input(source)
    payload = payload.model_copy(
        update={"candidate_evidence": payload.candidate_evidence[:-1]}
    )

    result = tailoring.run_resume_tailoring_tool(payload)

    assert result.status == "ERROR"
    assert any("project addition" in error for error in result.errors)
    assert result.validation_failures == result.errors


def test_direct_tailoring_rejects_go_claim_citing_python(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.tex"
    source.write_text(Path("data/resume.tex").read_text(encoding="utf-8"))
    payload = _tool_resume_tailoring_input(source)
    payload = payload.model_copy(
        update={
            "fit_analysis": payload.fit_analysis.model_copy(
                update={
                    "evidenced_missing_skills": [
                        EvidenceClaim(
                            claim="Go: safe to add",
                            evidence_ids=["resume-experience-001"],
                        )
                    ]
                }
            )
        }
    )

    result = tailoring.run_resume_tailoring_tool(payload)

    assert result.status == "ERROR"
    assert any("Go" in failure for failure in result.validation_failures)
    assert not result.output_pdf_path


def test_unrelated_revision_change_does_not_satisfy_feedback() -> None:
    change = ChangeLogEntry(
        change_id="unrelated",
        section="summary",
        description="Reworded the summary.",
        before_text="Machine learning engineer.",
        after_text="Experienced machine learning engineer.",
        reason="Improve role alignment.",
        evidence_ids=["resume-python", "job-python"],
    )
    evidence = {
        "resume-python": EvidenceItem(
            evidence_id="resume-python",
            source="resume",
            text="Python engineer.",
            tags=["Python"],
        ),
        "job-python": EvidenceItem(
            evidence_id="job-python",
            source="job_posting",
            text="Required skill: Python",
            tags=["job_skill_requirement", "Python"],
        ),
    }

    satisfied, checks = tailoring._check_revision_feedback(
        "Emphasize stakeholder leadership.", [change], evidence
    )

    assert satisfied is False
    assert checks == ["feedback-specific content overlap: not met"]


def test_skill_elsewhere_cannot_be_attached_to_unrelated_experience_bullet(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.tex"
    source.write_text(Path("data/resume.tex").read_text(encoding="utf-8"))
    payload = _tool_resume_tailoring_input(source)
    elsewhere_go = EvidenceItem(
        evidence_id="master-go",
        source="master_skills",
        text="Skill: Go",
        tags=["Go"],
    )
    payload = payload.model_copy(
        update={
            "candidate_evidence": [*payload.candidate_evidence, elsewhere_go],
            "fit_analysis": payload.fit_analysis.model_copy(
                update={
                    "evidenced_missing_skills": [
                        EvidenceClaim(
                            claim="Go: evidenced elsewhere",
                            evidence_ids=["master-go"],
                        )
                    ]
                }
            ),
        }
    )
    evidence = {item.evidence_id: item for item in payload.candidate_evidence}

    rewritten, cited = tailoring._rewrite_experience_bullet(
        "Built a visual quality-control workflow in Python.",
        1,
        payload,
        evidence,
    )

    assert "Go" not in rewritten
    assert "master-go" not in cited


def test_after_text_with_new_unsupported_skill_is_rejected() -> None:
    job = _tool_resume_tailoring_job()
    bullet = EvidenceItem(
        evidence_id="resume-python",
        source="resume",
        text="Built production services in Python.",
        tags=["experience", "Python"],
    )
    elsewhere_go = EvidenceItem(
        evidence_id="master-go",
        source="master_skills",
        text="Skill: Go",
        tags=["Go"],
    )
    payload = TailorResumeInput(
        job=job,
        fit_analysis=FitAnalysisOutput(
            job_id=job.job_id,
            evidenced_missing_skills=[
                EvidenceClaim(
                    claim="Go: evidenced elsewhere",
                    evidence_ids=["master-go"],
                )
            ],
        ),
        source_resume_tex_path="data/resume.tex",
        candidate_evidence=[bullet, elsewhere_go],
        job_evidence=build_job_evidence(job),
    )
    change = ChangeLogEntry(
        change_id="unsupported-go",
        section="experience",
        description="Rewrote experience bullet.",
        before_text="Built production services in Python.",
        after_text="Built production services in Python and Go.",
        reason="Target a posted requirement.",
        evidence_ids=[
            "resume-python",
            "master-go",
            "job-J900-skill-003",
        ],
    )
    evidence = {
        item.evidence_id: item
        for item in [*payload.candidate_evidence, *payload.job_evidence]
    }

    with pytest.raises(tailoring.TailoringError, match="unsupported wording"):
        tailoring._validate_change_log([change], evidence, payload)


def test_supported_skill_with_correct_bullet_evidence_remains_allowed() -> None:
    source = Path("data/resume.tex")
    payload = _tool_resume_tailoring_input(source)
    before = "Delivered production Python services with Docker and Kubernetes."
    change = ChangeLogEntry(
        change_id="supported-kubernetes",
        section="experience",
        description="Rewrote experience bullet.",
        before_text=before,
        after_text=f"Applied Kubernetes through work that {before.lower()}",
        reason="Emphasize a supported posted requirement.",
        evidence_ids=[
            "resume-experience-002",
            "job-J900-skill-002",
        ],
    )
    evidence = {
        item.evidence_id: item
        for item in [*payload.candidate_evidence, *build_job_evidence(payload.job)]
    }

    tailoring._validate_change_log([change], evidence, payload)


def test_unsupported_project_name_in_experience_bullet_is_rejected() -> None:
    source = Path("data/resume.tex")
    payload = _tool_resume_tailoring_input(source)
    change = ChangeLogEntry(
        change_id="unsupported-project",
        section="experience",
        description="Rewrote experience bullet.",
        before_text=(
            "Graduate Research Assistant | Built a visual-QC workflow over "
            "1.7M segmentation/tracking predictions; achieved 90.5% test "
            "accuracy and 0.764 F1 using out-of-fold evaluation and targeted "
            "error analysis with Python and computer vision."
        ),
        after_text=(
            "Built the AI CRM Platform while delivering a visual-QC workflow "
            "with Python."
        ),
        reason="Target the role.",
        evidence_ids=[
            "resume-experience-001",
            "portfolio-P02",
            "job-J900-description",
        ],
    )
    evidence = {
        item.evidence_id: item
        for item in [*payload.candidate_evidence, *build_job_evidence(payload.job)]
    }

    with pytest.raises(tailoring.TailoringError, match="unsupported wording"):
        tailoring._validate_change_log([change], evidence, payload)


def test_compilation_helpers_do_not_create_internal_trace_spans(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tex_path = tmp_path / "resume.tex"
    pdf_path = tmp_path / "resume.pdf"

    def successful_compile(path: Path) -> list[str]:
        writer = PdfWriter()
        writer.add_blank_page(width=612, height=792)
        with pdf_path.open("wb") as handle:
            writer.write(handle)
        return []

    monkeypatch.setattr(tailoring, "_run_pdflatex", successful_compile)
    tracer = TraceManager(enabled=False)
    tracer.start_run("run-compile-metadata", "thread-compile-metadata")

    pages, errors, _ = tailoring._compile_one_page(
        "\\documentclass{article}\\begin{document}ok\\end{document}",
        tex_path,
        pdf_path,
        tracer=tracer,
        trace_metadata={"job_id": "J900", "resume_id": "resume-J900"},
    )

    assert pages == 1
    assert errors == []
    assert [event.name for event in tracer.events] == ["Job Search Agent Run"]
