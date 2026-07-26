"""Run-scoped output path helpers."""

from __future__ import annotations

from pathlib import Path

from src.config import AppConfig, get_config


def run_output_dir(run_id: str, config: AppConfig | None = None) -> Path:
    """Return the artifact root for one workflow run."""

    active = config or get_config()
    return active.output_dir / run_id


def job_output_dir(
    job_id: str,
    *,
    run_id: str | None = None,
    config: AppConfig | None = None,
) -> Path:
    """Return a job artifact directory, optionally scoped by run ID."""

    active = config or get_config()
    root = active.output_dir / run_id if run_id else active.output_dir
    return root / job_id


def memory_path_for_run(run_id: str, config: AppConfig | None = None) -> Path:
    """Return the memory JSON path for one workflow run."""

    return run_output_dir(run_id, config=config) / "memory.json"


__all__ = ["job_output_dir", "memory_path_for_run", "run_output_dir"]
