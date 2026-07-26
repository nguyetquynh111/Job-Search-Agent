"""LaTeX escaping, compilation, and PDF page counting boundaries."""

from __future__ import annotations

import subprocess
from pathlib import Path

from pypdf import PdfReader

DEFAULT_PDFLATEX_TIMEOUT_SECONDS = 60

_SPECIAL_CHARACTERS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def escape_latex(text: str) -> str:
    """Escape plain text for use inside a LaTeX command argument."""

    return "".join(_SPECIAL_CHARACTERS.get(character, character) for character in text)


def pdf_page_count(path: Path) -> int:
    """Return the page count of a readable PDF."""

    return len(PdfReader(str(path)).pages)


def pdflatex_command(tex_path: Path) -> list[str]:
    """Return the exact production compilation command for one TeX source."""

    return [
        "pdflatex",
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-output-directory",
        str(tex_path.parent),
        str(tex_path),
    ]


def run_pdflatex(
    tex_path: Path,
    *,
    artifact_label: str,
    timeout_seconds: int = DEFAULT_PDFLATEX_TIMEOUT_SECONDS,
) -> list[str]:
    """Compile one TeX source and return user-facing errors instead of raising."""

    try:
        result = subprocess.run(
            pdflatex_command(tex_path),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except FileNotFoundError:
        return [
            "pdflatex is not installed or not on PATH. "
            "Install a LaTeX distribution to compile PDFs."
        ]
    except subprocess.TimeoutExpired:
        return [f"pdflatex timed out while compiling the {artifact_label}."]

    if result.returncode == 0:
        return []
    output = result.stdout or result.stderr
    tail = "\n".join(output.splitlines()[-15:])
    return [f"pdflatex failed for {tex_path.name}: {tail}"]
