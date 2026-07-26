"""Production preflight behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts import preflight


def _fixtures(root: Path) -> None:
    for relative in preflight.REQUIRED_FIXTURES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture", encoding="utf-8")


def test_preflight_reports_credentials_and_incomplete_langfuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fixtures(tmp_path)
    monkeypatch.setattr(preflight, "_check_latex", lambda errors: None)
    monkeypatch.setattr(
        preflight.importlib.util,
        "find_spec",
        lambda module: object(),
    )
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("DEEPINFRA_API_KEY", raising=False)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "public")
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    errors = preflight.run_preflight(
        repo_root=tmp_path,
        output_dir=tmp_path / "outputs",
        require_langfuse=True,
    )

    assert any("LLM_MODEL and DEEPINFRA_API_KEY" in error for error in errors)
    assert any("LANGFUSE_SECRET_KEY" in error for error in errors)


def test_preflight_reports_unreadable_or_missing_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fixtures(tmp_path)
    (tmp_path / "data/jobs.csv").unlink()
    monkeypatch.setattr(preflight, "_check_latex", lambda errors: None)
    monkeypatch.setattr(
        preflight.importlib.util,
        "find_spec",
        lambda module: object(),
    )
    monkeypatch.setenv("LLM_MODEL", "provider/model")
    monkeypatch.setenv("DEEPINFRA_API_KEY", "key")
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    errors = preflight.run_preflight(
        repo_root=tmp_path,
        output_dir=tmp_path / "outputs",
        require_langfuse=False,
    )

    assert any("data/jobs.csv" in error for error in errors)
