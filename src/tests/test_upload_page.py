"""Smoke test for the upload page in a partially installed environment."""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_upload_page_renders_when_graph_runtime_is_unavailable(
    tmp_path: Path, monkeypatch
) -> None:
    """Optional workflow dependencies must not take down the input form."""

    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "outputs"))
    page = Path(__file__).parents[1] / "app" / "views" / "1_Input.py"

    app = AppTest.from_file(str(page), default_timeout=15).run()

    assert not app.exception
    assert [item.value for item in app.subheader[:2]] == [
        "Input files",
        "What happens next",
    ]
    assert len(app.get("file_uploader")) == 4
    assert "Start search" in [button.label for button in app.button]
