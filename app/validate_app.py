"""Run lightweight app-local validation without starting a live agent."""

# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.components.human_review import review_controls_visible
from app.services.artifact_loader import discover_existing_runs, load_existing_run


def main() -> int:
    checks: list[tuple[str, bool, str]] = []
    try:
        import app.streamlit_app as streamlit_app

        checks.append(("App import", callable(streamlit_app.main), "main() is callable"))
    except Exception as exc:
        checks.append(("App import", False, str(exc)))

    runs = discover_existing_runs(REPO_ROOT)
    checks.append(("Existing-run discovery", bool(runs), f"{len(runs)} run(s) found"))
    if runs:
        try:
            snapshot = load_existing_run(runs[0], REPO_ROOT)
            checks.extend(
                [
                    (
                        "Repository path",
                        snapshot.repo_root == REPO_ROOT,
                        str(snapshot.repo_root),
                    ),
                    (
                        "Read-only loading",
                        snapshot.read_only,
                        snapshot.run_id,
                    ),
                    (
                        "Review control guard",
                        not review_controls_visible(snapshot),
                        "Existing Run cannot expose live review actions",
                    ),
                ]
            )
        except Exception as exc:
            checks.append(("Existing-run loading", False, str(exc)))

    failed = False
    for label, passed, detail in checks:
        failed = failed or not passed
        print(f"{'PASS' if passed else 'FAIL'}  {label}: {detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
