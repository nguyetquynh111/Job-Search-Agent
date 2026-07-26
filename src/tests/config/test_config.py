"""Application configuration tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import get_config, validate_runtime_requirements


def test_runtime_paths_are_derived_from_output_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "runtime"
    monkeypatch.setenv("OUTPUT_DIR", str(output_dir))

    config = get_config()

    assert config.output_dir == output_dir
    assert config.memory_file == output_dir / "memory.json"
    assert config.checkpoint_db == output_dir / "checkpoints.sqlite"


def test_runtime_preflight_requires_llm_and_pdflatex(monkeypatch) -> None:
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("DEEPINFRA_API_KEY", raising=False)
    monkeypatch.setattr("src.config.shutil.which", lambda executable: None)

    with pytest.raises(RuntimeError) as exc_info:
        validate_runtime_requirements()

    message = str(exc_info.value)
    assert "pdflatex" in message
    assert "LLM_MODEL" in message


def test_runtime_preflight_accepts_complete_configuration(monkeypatch) -> None:
    monkeypatch.setenv("LLM_MODEL", "example/model")
    monkeypatch.setenv("DEEPINFRA_API_KEY", "test-key")
    monkeypatch.setattr(
        "src.config.shutil.which", lambda executable: "/usr/bin/pdflatex"
    )

    validate_runtime_requirements()
