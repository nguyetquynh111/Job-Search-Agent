"""Consolidated tests for this domain."""

from __future__ import annotations

from app.app import (
    build_evidence_lookup,
    build_outputs_zip,
    build_resume_change_views,
    group_rejected_jobs,
    observability_summary,
    review_status_for_job,
    score_components_from_rationale,
    validate_uploaded_file,
)
from app.app import ensure_session_defaults, reset_demo_data
from io import BytesIO
from pathlib import Path
from streamlit.testing.v1 import AppTest
from types import SimpleNamespace
import zipfile

# --- test_ui_components.py ---
"""Pure helper coverage for the Streamlit presentation layer."""


class _ui_components_Upload:
    def __init__(self, name: str, value: bytes) -> None:
        self.name = name
        self._value = value

    def getvalue(self) -> bytes:
        return self._value


def test_upload_validation_requires_expected_nonempty_file() -> None:
    assert validate_uploaded_file(None, {".csv"}) == (False, "Required file")
    assert validate_uploaded_file(
        _ui_components_Upload("jobs.txt", b"x"), {".csv"}
    ) == (
        False,
        "Use .csv",
    )
    assert validate_uploaded_file(_ui_components_Upload("jobs.csv", b""), {".csv"}) == (
        False,
        "File is empty",
    )
    jobs_fixture = Path(__file__).parents[1] / "data" / "jobs.csv"
    valid, message = validate_uploaded_file(
        _ui_components_Upload("jobs.csv", jobs_fixture.read_bytes()),
        {".csv"},
    )
    assert valid is True
    assert message.startswith("Ready")


def test_upload_validation_reuses_backend_loaders_for_all_inputs() -> None:
    data_dir = Path(__file__).parents[1] / "data"
    fixtures = [
        ("jobs.csv", data_dir / "jobs.csv", {".csv"}),
        ("preferences.yaml", data_dir / "preferences.yaml", {".yaml"}),
        ("resume.tex", data_dir / "resume.tex", {".tex"}),
        ("portfolio.txt", data_dir / "portfolio.txt", {".txt"}),
    ]

    for upload_name, path, extensions in fixtures:
        valid, message = validate_uploaded_file(
            _ui_components_Upload(upload_name, path.read_bytes()),
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
\\section{Summary}
Old summary.
\\section{Professional Experience}
\\begin{itemize}
\\resumeItem{Old first bullet.}
\\resumeItem{Old second bullet.}
\\end{itemize}
\\section{Technical Skills}
\\begin{itemize}
  \\small\\item{Python}
\\end{itemize}
\\section{Selected Projects}
\\begin{itemize}
  \\item \\textbf{Example Project}
\\end{itemize}
""",
        encoding="utf-8",
    )
    after.write_text(
        """
\\section{Summary}
New summary.
\\section{Professional Experience}
\\begin{itemize}
\\resumeItem{New first bullet.}
\\resumeItem{New second bullet.}
\\end{itemize}
\\section{Technical Skills}
\\begin{itemize}
  \\small\\item{Python, SQL}
\\end{itemize}
\\section{Selected Projects}
\\begin{itemize}
  \\item \\textbf{Example Project}
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


# --- test_ui_streamlit_session.py ---
"""Streamlit session helper tests."""


def test_streamlit_reruns_do_not_create_new_graph_run_accidentally(
    tmp_path: Path, monkeypatch
) -> None:
    """Initializing UI state does not create run IDs or thread IDs."""

    output_dir = tmp_path / "runtime"
    memory_file = output_dir / "memory.json"
    monkeypatch.setenv("OUTPUT_DIR", str(output_dir))
    session: dict = {}
    ensure_session_defaults(session)
    first = dict(session)
    ensure_session_defaults(session)

    assert session == first
    assert session["current_run_id"] is None
    assert session["current_thread_id"] is None
    assert memory_file.read_text(encoding="utf-8") == "[]"


def test_reset_demo_data_preserves_checkpoint_files(
    tmp_path: Path, monkeypatch
) -> None:
    """Resetting generated artifacts must not remove the SQLite checkpoint."""

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OUTPUT_DIR", "outputs")
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    checkpoint_files = [
        output_dir / "checkpoints.sqlite",
        output_dir / "checkpoints.sqlite-journal",
        output_dir / "checkpoints.sqlite-shm",
        output_dir / "checkpoints.sqlite-wal",
    ]
    for path in checkpoint_files:
        path.write_text("checkpoint", encoding="utf-8")
    generated_file = output_dir / "generated-resume.pdf"
    generated_file.write_text("artifact", encoding="utf-8")
    memory_file = output_dir / "memory.json"
    memory_file.write_text('[{"stale": true}]', encoding="utf-8")
    session = {"input_paths": {"memory_file": str(memory_file)}}

    reset_demo_data(session)

    assert all(path.exists() for path in checkpoint_files)
    assert memory_file.read_text(encoding="utf-8") == "[]"
    assert not generated_file.exists()


# --- test_ui_upload_page.py ---
"""Import test for the upload page in a partially installed environment."""


def test_upload_page_renders_when_graph_runtime_is_unavailable(
    tmp_path: Path, monkeypatch
) -> None:
    """Optional workflow dependencies must not take down the input form."""

    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "outputs"))
    page = Path(__file__).parents[1] / "app" / "app.py"

    app = AppTest.from_file(str(page), default_timeout=15).run()

    assert not app.exception
    assert [item.value for item in app.subheader[:2]] == [
        "Input files",
        "What happens next",
    ]
    assert len(app.get("file_uploader")) == 4
    assert "Start search" in [button.label for button in app.button]
