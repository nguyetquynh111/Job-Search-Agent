#!/usr/bin/env python3
"""Fail-fast production preflight for the documented workflow."""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REQUIRED_MODULES = {
    "langchain_core": "langchain-core",
    "langchain_openai": "langchain-openai",
    "langgraph": "langgraph",
    "langfuse": "langfuse",
    "pandas": "pandas",
    "pydantic": "pydantic",
    "pypdf": "pypdf",
    "dotenv": "python-dotenv",
    "yaml": "PyYAML",
    "streamlit": "streamlit",
}
REQUIRED_FIXTURES = (
    "data/jobs.csv",
    "data/preferences.yaml",
    "data/resume.tex",
    "data/portfolio.txt",
)
LATEX_PACKAGES = (
    "fullpage.sty",
    "titlesec.sty",
    "enumitem.sty",
    "hyperref.sty",
    "babel.sty",
    "english.ldf",
    "tabularx.sty",
)


def _load_dotenv(repo_root: Path) -> None:
    """Read simple KEY=VALUE entries without requiring python-dotenv first."""

    path = repo_root / ".env"
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def _check_latex(errors: list[str]) -> None:
    pdflatex = shutil.which("pdflatex")
    if not pdflatex:
        errors.append("pdflatex is not installed or not on PATH")
        return
    kpsewhich = shutil.which("kpsewhich")
    if not kpsewhich:
        errors.append("kpsewhich is unavailable; the TeX distribution is incomplete")
        return
    missing = [
        package
        for package in LATEX_PACKAGES
        if not subprocess.run(
            [kpsewhich, package],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
    ]
    if missing:
        errors.append("missing required LaTeX packages: " + ", ".join(missing))
        return
    with tempfile.TemporaryDirectory(prefix="job-agent-preflight-") as directory:
        root = Path(directory)
        tex_path = root / "preflight.tex"
        tex_path.write_text(
            "\\documentclass{article}\n"
            "\\usepackage{fullpage,titlesec,enumitem,hyperref,tabularx}\n"
            "\\usepackage[english]{babel}\n"
            "\\begin{document}preflight\\end{document}\n",
            encoding="utf-8",
        )
        result = subprocess.run(
            [
                pdflatex,
                "-interaction=nonstopmode",
                "-halt-on-error",
                "-output-directory",
                directory,
                str(tex_path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if result.returncode != 0 or not (root / "preflight.pdf").is_file():
            tail = "\n".join((result.stdout or result.stderr).splitlines()[-8:])
            errors.append(f"pdflatex package compilation check failed: {tail}")


def run_preflight(
    *,
    repo_root: Path,
    output_dir: Path,
) -> list[str]:
    """Return every discovered production-blocking error."""

    errors: list[str] = []
    _load_dotenv(repo_root)
    if sys.version_info[:2] != (3, 12):
        errors.append(
            "Python 3.12 is required; found "
            f"{sys.version_info.major}.{sys.version_info.minor}"
        )
    missing_modules = [
        distribution
        for module, distribution in REQUIRED_MODULES.items()
        if importlib.util.find_spec(module) is None
    ]
    if missing_modules:
        errors.append(
            "missing Python dependencies: " + ", ".join(sorted(missing_modules))
        )
    _check_latex(errors)
    if not os.getenv("LLM_MODEL") or not os.getenv("DEEPINFRA_API_KEY"):
        errors.append("LLM_MODEL and DEEPINFRA_API_KEY are required")
    langfuse_values = {
        "LANGFUSE_PUBLIC_KEY": os.getenv("LANGFUSE_PUBLIC_KEY", "").strip(),
        "LANGFUSE_SECRET_KEY": os.getenv("LANGFUSE_SECRET_KEY", "").strip(),
        "LANGFUSE_HOST": (
            os.getenv("LANGFUSE_HOST")
            or os.getenv("LANGFUSE_BASE_URL")
            or "https://us.cloud.langfuse.com"
        ).strip(),
    }
    missing = [key for key, value in langfuse_values.items() if not value]
    if missing:
        errors.append("connected Langfuse tracing requires: " + ", ".join(missing))
    for relative in REQUIRED_FIXTURES:
        path = repo_root / relative
        if not path.is_file() or not os.access(path, os.R_OK):
            errors.append(
                f"required input fixture is missing or unreadable: {relative}"
            )
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        probe = output_dir / ".preflight-write-probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        errors.append(f"output path is not writable ({output_dir}): {exc}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=os.getenv("OUTPUT_DIR", "outputs"))
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    errors = run_preflight(
        repo_root=repo_root,
        output_dir=(repo_root / args.output_dir).resolve()
        if not Path(args.output_dir).is_absolute()
        else Path(args.output_dir),
    )
    if errors:
        print("Production preflight FAILED:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("Production preflight passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
