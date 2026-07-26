"""Production E2E runner guardrails."""

from __future__ import annotations

from pathlib import Path
import subprocess

from pypdf import PdfWriter

import main as run_production_e2e
from src.review.memory import JSONMemoryStore
from src.utils.output_validation import sanitize_public_artifact_references


def test_production_runner_reports_exact_missing_live_environment(
    monkeypatch,
) -> None:
    for name in (
        "LLM_MODEL",
        "DEEPINFRA_API_KEY",
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
    ):
        monkeypatch.delenv(name, raising=False)

    assert run_production_e2e._missing_live_environment() == [
        "LLM_MODEL",
        "DEEPINFRA_API_KEY",
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
    ]


def test_production_runner_seeds_real_conflicting_memory(tmp_path: Path) -> None:
    memory_file = tmp_path / "memory.json"

    run_production_e2e._seed_conflicting_memory(memory_file)
    facts = JSONMemoryStore(memory_file).load()

    assert len(facts) == 1
    assert facts[0].fact_id == "mem-seeded-old-experience"
    assert facts[0].fact_type == "experience"
    assert facts[0].canonical_value == "2 years of experience in data engineering"
    assert facts[0].active is True


def test_original_resume_compile_writes_to_output_dir_not_data(
    monkeypatch,
    tmp_path: Path,
) -> None:
    commands: list[list[str]] = []

    def fake_run(command, **_kwargs):
        commands.append([str(part) for part in command])
        output_dir = Path(command[command.index("-output-directory") + 1])
        writer = PdfWriter()
        writer.add_blank_page(width=612, height=792)
        with (output_dir / "resume.pdf").open("wb") as handle:
            writer.write(handle)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(run_production_e2e.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(run_production_e2e.subprocess, "run", fake_run)

    compile_dir = tmp_path / "outputs" / "run-1" / "source_resume"
    pdf_path = run_production_e2e._compile_original_resume(compile_dir)

    assert pdf_path == compile_dir / "resume.pdf"
    assert commands[0][commands[0].index("-output-directory") + 1] == str(compile_dir)
    assert commands[0][-1].endswith("data/resume.tex")


def test_production_runner_removes_non_submission_output_artifacts(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "outputs"
    run_id = "custom-run-id"
    final = output_dir / run_id
    complete_sibling = output_dir / "run-complete"
    incomplete_sibling = output_dir / "run-incomplete"
    final.mkdir(parents=True)
    complete_sibling.mkdir(parents=True)
    incomplete_sibling.mkdir(parents=True)
    (final / "checkpoints.sqlite").write_text("checkpoint", encoding="utf-8")
    (output_dir / "memory.json").write_text("duplicate", encoding="utf-8")
    (complete_sibling / "run_manifest.json").write_text("{}", encoding="utf-8")
    (incomplete_sibling / "memory.json").write_text("stale", encoding="utf-8")

    run_production_e2e._remove_non_submission_output_artifacts(
        output_dir, run_id
    )

    assert not (final / "checkpoints.sqlite").exists()
    assert not (output_dir / "memory.json").exists()
    assert complete_sibling.is_dir()
    assert not incomplete_sibling.exists()


def test_public_artifact_sanitizer_does_not_corrupt_cover_letter_name() -> None:
    payload = {
        "files": [
            "outputs/run-1/J001/cover_letter.pdf",
            "outputs/run-1/J001/letter.pdf",
        ]
    }

    sanitized = sanitize_public_artifact_references(payload)

    assert sanitized["files"][0] == "outputs/run-1/J001/cover_letter.pdf"
    assert sanitized["files"][1] == "outputs/run-1/J001/cover_letter.pdf"
