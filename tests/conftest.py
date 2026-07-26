"""Shared pytest configuration."""

from __future__ import annotations

import sys
import shutil
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(autouse=True)
def disable_llm_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep tests offline unless a test explicitly configures DeepInfra."""

    monkeypatch.delenv("DEEPINFRA_API_KEY", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_HOST", raising=False)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Remove pytest/Python caches after successful test runs."""

    if exitstatus != 0:
        return
    _remove_test_caches()


def _remove_test_caches() -> None:
    cache_dir = PROJECT_ROOT / ".pytest_cache"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)

    for pycache_dir in PROJECT_ROOT.rglob("__pycache__"):
        if ".git" in pycache_dir.parts or ".venv" in pycache_dir.parts:
            continue
        shutil.rmtree(pycache_dir)
