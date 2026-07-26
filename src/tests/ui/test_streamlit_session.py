"""Streamlit session helper tests."""

from __future__ import annotations

from pathlib import Path

from src.ui.session import ensure_session_defaults, reset_demo_data


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
