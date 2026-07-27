"""Validation coverage for the Streamlit control center."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace

from streamlit.testing.v1 import AppTest

import app.services.agent_adapter as agent_adapter
from app.components import workspace
from app.components.human_review import review_controls_visible
from app.components.ranking import render_ranking
from app.services.artifact_loader import (
    discover_existing_runs,
    load_existing_run,
    repository_root,
)
from app.services.run_state import RunSnapshot

def test_app_imports() -> None:
    import app.streamlit_app as streamlit_app

    assert callable(streamlit_app.main)


def test_upload_extension_validation_rejects_wrong_suffixes() -> None:
    from app.streamlit_app import _upload_extension_allowed

    assert _upload_extension_allowed("resume.TEX", ["tex"])
    assert _upload_extension_allowed("preferences.yml", ["yaml", "yml"])
    assert not _upload_extension_allowed("resume.pdf", ["tex"])
    assert not _upload_extension_allowed("jobs.csv.exe", ["csv"])
    assert not _upload_extension_allowed("portfolio", ["txt"])


def test_input_review_uses_workflow_loaders() -> None:
    from app.components.input_review import InputReview, load_review_item

    root = Path(__file__).resolve().parents[2]
    items = {
        filename: load_review_item(title, filename, root / "data" / filename)
        for title, filename in (
            ("Resume", "resume.tex"),
            ("Jobs dataset", "jobs.csv"),
            ("Portfolio", "portfolio.txt"),
            ("Preferences", "preferences.yaml"),
        )
    }
    review = InputReview(items)

    assert review.valid
    assert not review.errors
    assert items["resume.tex"].value.resume.plain_text
    assert items["resume.tex"].value.pdf_path.is_file()
    assert items["resume.tex"].value.page_count >= 1
    assert items["jobs.csv"].value
    assert items["portfolio.txt"].value.projects
    assert items["preferences.yaml"].value.preferences.target_job_titles


def test_input_review_reports_parse_errors(tmp_path: Path) -> None:
    from app.components.input_review import load_review_item

    invalid_jobs = tmp_path / "jobs.csv"
    invalid_jobs.write_text("company\\nExample Co\\n", encoding="utf-8")

    item = load_review_item("Jobs dataset", "jobs.csv", invalid_jobs)

    assert not item.valid
    assert item.error is not None
    assert "missing required columns" in item.error


def test_repository_files_are_not_implicitly_uploaded() -> None:
    from app.components.input_review import InputReview, InputReviewItem

    review = InputReview(
        {
            filename: InputReviewItem(title, filename)
            for title, filename in (
                ("Resume", "resume.tex"),
                ("Jobs dataset", "jobs.csv"),
                ("Portfolio", "portfolio.txt"),
                ("Preferences", "preferences.yaml"),
            )
        }
    )

    assert not review.has_uploads
    assert not review.valid
    assert review.missing == [
        "resume.tex",
        "jobs.csv",
        "portfolio.txt",
        "preferences.yaml",
    ]


def test_resume_review_renders_pdf_instead_of_extracted_text(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from app.components import input_review

    metrics: list[tuple[str, int]] = []
    previews: list[tuple[Path, int, str]] = []

    class MetricColumn:
        def metric(self, label: str, value: int) -> None:
            metrics.append((label, value))

    monkeypatch.setattr(
        input_review.st,
        "columns",
        lambda count: [MetricColumn() for _ in range(count)],
    )
    monkeypatch.setattr(
        input_review.st,
        "pdf",
        lambda path, height, key: previews.append((path, height, key)),
    )
    monkeypatch.setattr(input_review.st, "error", lambda _message: None)
    preview = input_review.ResumePreview(
        resume=SimpleNamespace(
            experience=["Experience"],
            projects=["Project"],
            skills=["Python"],
        ),
        pdf_path=tmp_path / "resume.pdf",
        page_count=1,
    )

    input_review._render_resume(preview)  # noqa: SLF001

    assert metrics[0] == ("Pages", 1)
    assert previews == [
        (preview.pdf_path, 760, "uploaded-resume-preview")
    ]


def test_repository_paths_resolve() -> None:
    root = repository_root()
    assert root == Path(__file__).resolve().parents[2]
    assert (root / "src" / "agent" / "graph.py").is_file()
    assert (root / "data" / "jobs.csv").is_file()


def test_missing_optional_artifacts_do_not_crash(tmp_path: Path) -> None:
    run_dir = tmp_path / "outputs" / "run-empty"
    run_dir.mkdir(parents=True)
    snapshot = load_existing_run("run-empty", tmp_path)
    assert snapshot.run_id == "run-empty"
    assert snapshot.status == "INCOMPLETE_ARTIFACTS"
    assert snapshot.top_3_job_ids == []
    assert snapshot.trace_events == []


def test_existing_run_loading_uses_automatic_selection_when_available() -> None:
    runs = discover_existing_runs()
    assert runs, "Repository should contain at least one existing output run."
    snapshot = load_existing_run(runs[0])
    assert snapshot.read_only is True
    assert snapshot.run_dir is not None
    assert snapshot.run_dir.name == runs[0]


def test_top_three_view_has_no_manual_reselection_control() -> None:
    source = Path(render_ranking.__code__.co_filename).read_text(encoding="utf-8")
    assert "selectbox(" not in source
    assert "multiselect(" not in source
    assert "data_editor(" not in source


def test_ranking_rows_render_as_html_instead_of_markdown_code(monkeypatch) -> None:
    rendered: list[tuple[str, bool]] = []
    snapshot = RunSnapshot(
        mode="Existing Run",
        repo_root=repository_root(),
        run_id="ranking-html-test",
        run_dir=None,
        status="COMPLETE",
        phase="COMPLETE",
        read_only=True,
        top_3_job_ids=["J001", "J002"],
        ranked_jobs=[
            {
                "job": {
                    "job_id": f"J00{index}",
                    "title": f"Role {index}",
                    "company": f"Company {index}",
                },
                "score": 90 - index,
            }
            for index in range(1, 4)
        ],
    )

    monkeypatch.setattr(
        workspace.st,
        "markdown",
        lambda body, unsafe_allow_html=False: rendered.append(
            (body, unsafe_allow_html)
        ),
    )

    workspace._ranking_rows(snapshot, limit=None)  # noqa: SLF001

    assert len(rendered) == 1
    html, unsafe_allow_html = rendered[0]
    assert unsafe_allow_html is True
    assert "\n" not in html
    assert html.count('<div class="rank-line">') == 3
    assert html.count("<em>Selected</em>") == 2


def test_review_controls_only_appear_at_real_live_interrupt() -> None:
    waiting = RunSnapshot(
        mode="Live Run",
        repo_root=repository_root(),
        run_id="run-test",
        run_dir=None,
        phase="HUMAN_REVIEW",
        status="WAITING_FOR_REVIEW",
        read_only=False,
        interrupt_payload={"resumes": {"J001": {}}},
    )
    readonly = RunSnapshot(
        mode="Live Run",
        repo_root=repository_root(),
        run_id="run-test",
        run_dir=None,
        phase="HUMAN_REVIEW",
        status="WAITING_FOR_REVIEW",
        read_only=True,
        interrupt_payload={"resumes": {"J001": {}}},
    )
    not_waiting = RunSnapshot(
        mode="Live Run",
        repo_root=repository_root(),
        run_id="run-test",
        run_dir=None,
        phase="TAILOR",
        status="RUNNING",
        read_only=False,
        interrupt_payload={},
    )
    assert review_controls_visible(waiting)
    assert not review_controls_visible(readonly)
    assert not review_controls_visible(not_waiting)


def test_live_review_renders_one_batched_three_resume_form() -> None:
    harness = Path(__file__).with_name("review_harness.py")
    app = AppTest.from_file(harness, default_timeout=20).run()
    assert not app.exception
    assert len(app.tabs) == 3
    assert len(app.radio) == 3
    assert [area.label for area in app.text_area] == ["Feedback for this CV"] * 3
    assert {button.label for button in app.button} >= {
        "Submit All 3 CV Reviews & Continue",
    }
    controls = {button.label: button for button in app.button}
    assert not controls["Submit All 3 CV Reviews & Continue"].disabled

    app.radio[1].set_value("reject").run()
    controls = {button.label: button for button in app.button}
    assert controls["Submit All 3 CV Reviews & Continue"].disabled

    app.text_area[1].input(
        "Candidate prefers Cardiovascular Flow and Stenosis Analysis "
        "to be the first project."
    ).run()
    controls = {button.label: button for button in app.button}
    assert not controls["Submit All 3 CV Reviews & Continue"].disabled
    assert app.text_area[0].value == ""
    assert app.text_area[2].value == ""


def test_start_live_run_returns_before_background_worker_finishes(
    monkeypatch,
    tmp_path: Path,
) -> None:
    started = threading.Event()
    release = threading.Event()

    monkeypatch.setattr(
        agent_adapter,
        "live_readiness",
        lambda _root: agent_adapter.LiveReadiness(True, [], [], True),
    )

    def fake_worker(handle, _root) -> None:
        started.set()
        release.wait(timeout=2)
        agent_adapter._store_progress(  # noqa: SLF001 - worker contract test
            handle,
            {
                "run_id": handle.run_id,
                "thread_id": handle.thread_id,
                "phase": "FILTER",
                "status": "RUNNING",
                "jobs": [],
            },
        )

    monkeypatch.setattr(agent_adapter, "_run_new_worker", fake_worker)
    before = time.perf_counter()
    handle = agent_adapter.start_live_run(tmp_path)
    elapsed = time.perf_counter() - before

    try:
        assert elapsed < 0.25
        assert started.wait(timeout=1)
        assert handle.worker_running
        assert agent_adapter.get_live_run(handle.run_id) is handle
        assert agent_adapter.snapshot_for_handle(handle).phase == "INITIALIZE"
    finally:
        release.set()
        if handle.worker:
            handle.worker.join(timeout=2)


def test_review_submission_also_runs_in_background(
    monkeypatch,
    tmp_path: Path,
) -> None:
    release = threading.Event()
    job_ids = ["J001", "J002", "J003"]
    handle = agent_adapter.LiveRunHandle(
        run_id="ui-live-review-test",
        thread_id="thread-ui-live-review-test",
        repo_root=tmp_path,
        app=object(),
        result={
            "run_id": "ui-live-review-test",
            "thread_id": "thread-ui-live-review-test",
            "phase": "HUMAN_REVIEW",
            "status": "WAITING_FOR_REVIEW",
            "top_3_job_ids": job_ids,
            "interrupt_payload": {
                "resumes": {job_id: {} for job_id in job_ids},
            },
        },
    )

    monkeypatch.setattr(
        agent_adapter,
        "_run_review_worker",
        lambda _handle, _decisions: release.wait(timeout=2),
    )
    submission = {
        "decisions": {
            job_id: {"decision": "approve", "comment": ""}
            for job_id in job_ids
        },
    }
    before = time.perf_counter()
    agent_adapter.submit_human_review(handle, submission)
    elapsed = time.perf_counter() - before

    try:
        assert elapsed < 0.25
        assert handle.worker_running
        assert agent_adapter.snapshot_for_handle(handle).phase == "COVER_LETTERS"
    finally:
        release.set()
        if handle.worker:
            handle.worker.join(timeout=2)
