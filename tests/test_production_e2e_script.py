"""Production E2E runner guardrails."""

from __future__ import annotations

from pathlib import Path
import subprocess

from pypdf import PdfWriter

from scripts import run_production_e2e
from src.review.memory import JSONMemoryStore


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
