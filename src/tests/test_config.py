"""Application configuration tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import get_config


def test_runtime_paths_are_derived_from_output_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "runtime"
    monkeypatch.setenv("OUTPUT_DIR", str(output_dir))

    config = get_config()

    assert config.output_dir == output_dir
    assert config.memory_file == output_dir / "memory.json"
    assert config.checkpoint_db == output_dir / "checkpoints.sqlite"
