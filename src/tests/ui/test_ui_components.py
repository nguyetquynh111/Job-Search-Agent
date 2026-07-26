"""Pure helper coverage for the Streamlit presentation layer."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import zipfile
from io import BytesIO

from src.ui.components import (
    build_evidence_lookup,
    build_outputs_zip,
    build_resume_change_views,
    group_rejected_jobs,
    observability_summary,
    review_status_for_job,
    score_components_from_rationale,
    validate_uploaded_file,
)


class _Upload:
    def __init__(self, name: str, value: bytes) -> None:
        self.name = name
        self._value = value

    def getvalue(self) -> bytes:
        return self._value


def test_upload_validation_requires_expected_nonempty_file() -> None:
    assert validate_uploaded_file(None, {".csv"}) == (False, "Required file")
    assert validate_uploaded_file(_Upload("jobs.txt", b"x"), {".csv"}) == (
        False,
        "Use .csv",
    )
    assert validate_uploaded_file(_Upload("jobs.csv", b""), {".csv"}) == (
        False,
        "File is empty",
    )
    jobs_fixture = Path(__file__).parents[3] / "data" / "jobs.csv"
    valid, message = validate_uploaded_file(
        _Upload("jobs.csv", jobs_fixture.read_bytes()),
        {".csv"},
    )
    assert valid is True
    assert message.startswith("Ready")


def test_upload_validation_reuses_backend_loaders_for_all_inputs() -> None:
    data_dir = Path(__file__).parents[3] / "data"
    fixtures = [
        ("jobs.csv", data_dir / "jobs.csv", {".csv"}),
        ("preferences.yaml", data_dir / "preferences.yaml", {".yaml"}),
        ("resume.tex", data_dir / "resume.tex", {".tex"}),
        ("portfolio.txt", data_dir / "portfolio.txt", {".txt"}),
    ]

    for upload_name, path, extensions in fixtures:
        valid, message = validate_uploaded_file(
            _Upload(upload_name, path.read_bytes()),
            extensions,
        )
        assert valid is True, message


def test_score_components_use_only_returned_rationale() -> None:
    components = score_components_from_rationale(
        "Score 91/100 -- skills: 4/5 required skills evidenced; "
        "experience: 6y meets the 5y+ minimum; "
        "domain: domain overlap on healthcare; location: remote-eligible."
    )

    assert components == {
        "Skill match": "4/5 required skills evidenced",
        "Experience alignment": "6y meets the 5y+ minimum",
        "Industry/domain alignment": "domain overlap on healthcare",
        "Location alignment": "remote-eligible",
    }


def test_rejected_jobs_are_grouped_by_exact_backend_reason() -> None:
    rejected = [
        {"job": {"job_id": "A"}, "reasons": ["Location mismatch", "Too senior"]},
        {"job": {"job_id": "B"}, "reasons": ["Location mismatch"]},
    ]

    grouped = group_rejected_jobs(rejected)

    assert list(grouped) == ["Location mismatch", "Too senior"]
    assert [item["job"]["job_id"] for item in grouped["Location mismatch"]] == [
        "A",
        "B",
    ]


def test_change_views_extract_supported_resume_sections(tmp_path: Path) -> None:
    before = tmp_path / "before.tex"
    after = tmp_path / "after.tex"
    before.write_text(
        """
% AGENT-EDIT-TARGET: summary
Old summary.
% AGENT-EDIT-TARGET: experience-bullet-1
\\resumeItem{Old first bullet.}
% AGENT-EDIT-TARGET: experience-bullet-2
\\resumeItem{Old second bullet.}
% AGENT-EDIT-TARGET: skills
\\begin{itemize}
  \\small\\item{Python}
\\end{itemize}
""",
        encoding="utf-8",
    )
    after.write_text(
        """
% AGENT-EDIT-TARGET: summary
New summary.
% AGENT-EDIT-TARGET: experience-bullet-1
\\resumeItem{New first bullet.}
% AGENT-EDIT-TARGET: experience-bullet-2
\\resumeItem{New second bullet.}
% AGENT-EDIT-TARGET: skills
\\begin{itemize}
  \\small\\item{Python, SQL}
\\end{itemize}
""",
        encoding="utf-8",
    )
    changes = [
        {"section": "summary", "description": "summary", "evidence_ids": ["r1"]},
        {"section": "experience", "description": "first", "evidence_ids": ["r2"]},
        {"section": "experience", "description": "second", "evidence_ids": ["r3"]},
        {"section": "skills", "description": "skills", "evidence_ids": ["m1"]},
        {"section": "formatting", "description": "layout", "evidence_ids": []},
    ]

    views = build_resume_change_views(changes, before, after, {})

    assert [view["category"] for view in views] == [
        "Professional summary",
        "Experience bullet 1 of 2",
        "Experience bullet 2 of 2",
        "Skills",
    ]
    assert views[0]["before"] == "Old summary."
    assert views[0]["after"] == "New summary."
    assert views[1]["before"] == "Old first bullet."
    assert views[1]["after"] == "New first bullet."


def test_change_views_prefer_exact_logged_before_after_and_reason(
    tmp_path: Path,
) -> None:
    views = build_resume_change_views(
        [
            {
                "section": "summary",
                "description": "legacy description",
                "before_text": "Exact old text",
                "after_text": "Exact new text",
                "reason": "Evidence-backed targeting.",
                "evidence_ids": ["job-J1-title", "resume-summary"],
            }
        ],
        tmp_path / "missing-before.tex",
        tmp_path / "missing-after.tex",
        {},
    )

    assert views[0]["before"] == "Exact old text"
    assert views[0]["after"] == "Exact new text"
    assert views[0]["reason"] == "Evidence-backed targeting."


def test_evidence_lookup_includes_job_posting_records() -> None:
    lookup = build_evidence_lookup(
        {
            "jobs": [
                {
                    "job_id": "J1",
                    "title": "ML Engineer",
                    "company": "Acme",
                    "location": "Remote",
                    "description": "Build ML systems.",
                    "required_skills": ["Python"],
                }
            ]
        }
    )

    assert lookup["job-J1-title"]["source"] == "job_posting"
    assert lookup["job-J1-skill-001"]["text"] == "Required skill: Python"


def test_review_status_supports_locked_and_revision_states() -> None:
    assert review_status_for_job({}, "A", has_active_payload=True) == (
        "Awaiting review",
        "brand",
    )
    rejected = {
        "revision_round": 2,
        "review_decisions": {"A": {"decision": "reject", "comment": "Change it"}},
    }
    assert review_status_for_job(rejected, "A") == (
        "Rejected — revision round 2 of 2",
        "warning",
    )
    finalized = {
        "status": "COMPLETED",
        "approved_job_ids": ["A"],
        "cover_letter_results": {"A": {}},
    }
    assert review_status_for_job(finalized, "A") == ("Finalized", "success")


def test_observability_summary_does_not_fabricate_llm_count() -> None:
    state = {
        "trace_id": "trace-1",
        "tool_history": [{}, {}],
        "review_history": [{"memory_writes": [{}, {}]}],
    }
    without_events = observability_summary(state, tracer=None)
    assert without_events["llm_calls"] is None
    assert without_events["tool_calls"] == 2
    assert without_events["memory_writes"] == 2

    tracer = SimpleNamespace(
        events=[
            SimpleNamespace(trace_id="trace-1", observation_type="GENERATION"),
            SimpleNamespace(trace_id="trace-1", observation_type="SPAN"),
        ]
    )
    assert observability_summary(state, tracer)["llm_calls"] == 1


def test_zip_contains_only_existing_final_artifacts(tmp_path: Path) -> None:
    resume = tmp_path / "resume.pdf"
    letter = tmp_path / "letter.pdf"
    resume.write_bytes(b"%PDF-resume")
    letter.write_bytes(b"%PDF-letter")

    payload = build_outputs_zip(
        {"J1": {"title": "ML Engineer"}},
        {"J1": {"output_pdf_path": str(resume)}},
        {"J1": {"output_pdf_path": str(letter)}},
        ["J1"],
    )

    assert payload is not None
    with zipfile.ZipFile(BytesIO(payload)) as archive:
        assert archive.namelist() == [
            "J1-ml-engineer/resume.pdf",
            "J1-ml-engineer/cover-letter.pdf",
        ]
