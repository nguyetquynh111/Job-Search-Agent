"""Structural resume targeting without template-specific annotations."""

from __future__ import annotations

from pathlib import Path

import pytest
from pypdf import PdfWriter

from src.tests.tools.test_resume_tailoring import _input
from src.tools import tailor_resume as tailoring
from src.tools.resume_tailoring.latex_structure import (
    LatexStructureError,
    parse_resume_structure,
)


def _ordinary_resume(*, second_summary: bool = False) -> str:
    duplicate = "\\section{Professional Summary}\nDuplicate.\n" if second_summary else ""
    return rf"""
\documentclass{{article}}
\begin{{document}}
\section{{Summary}}
Engineer building reliable Python systems.
{duplicate}
\section{{Professional Experience}}
\begin{{itemize}}
  \item Built Python forecasting services for weekly planning.
  \item Deployed containerized APIs with Docker and Kubernetes.
  \item Improved data quality checks for production pipelines.
\end{{itemize}}
\section{{Technical Skills}}
\begin{{itemize}}
  \item \textbf{{Languages:}} Python, SQL
\end{{itemize}}
\section{{Selected Projects}}
\begin{{itemize}}
  \item \textbf{{Forecasting Dashboard}} — Python
  \begin{{itemize}}
    \item Built weekly demand forecasts.
  \end{{itemize}}
\end{{itemize}}
\end{{document}}
""".strip()


def test_parser_accepts_alternate_sections_and_standard_items() -> None:
    source = _ordinary_resume()

    structure = parse_resume_structure(source, require_projects=True)

    assert structure.sections["summary"].title == "summary"
    assert structure.sections["experience"].title == "professional experience"
    assert structure.sections["skills"].title == "technical skills"
    assert structure.sections["projects"].title == "selected projects"
    assert [item.command for item in structure.experience_bullets] == [
        "item",
        "item",
        "item",
    ]
    assert [item.content(source) for item in structure.experience_bullets][1].startswith(
        "Deployed containerized"
    )
    assert structure.project_entries[0].name == "Forecasting Dashboard"


def test_parser_accepts_resume_item_macros_without_annotations() -> None:
    source = Path("data/resume.tex").read_text(encoding="utf-8")

    structure = parse_resume_structure(source, require_projects=True)

    assert len(structure.experience_bullets) == 4
    assert all(item.command == "resumeItem" for item in structure.experience_bullets)
    assert len(structure.project_entries) == 2


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (
            _ordinary_resume().replace("\\section{Summary}", "\\section{Profile}"),
            "summary",
        ),
        (_ordinary_resume(second_summary=True), "Multiple plausible summary"),
    ],
)
def test_parser_rejects_missing_or_ambiguous_required_sections(
    source: str, message: str
) -> None:
    with pytest.raises(LatexStructureError, match=message):
        parse_resume_structure(source)


def test_evidenced_skill_is_added_to_existing_skills_item(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = Path("data/resume.tex").read_text(encoding="utf-8")
    original = original.replace(", Kubernetes", "")
    source = tmp_path / "source.tex"
    source.write_text(original, encoding="utf-8")
    payload = _input(source)
    payload = payload.model_copy(
        update={
            "fit_analysis": payload.fit_analysis.model_copy(
                update={"project_swap": None}
            )
        }
    )
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "outputs"))

    def compile_once(text: str, tex_path: Path, pdf_path: Path):
        tex_path.write_text(text, encoding="utf-8")
        pdf_path.write_bytes(b"%PDF-1.4 test")
        return 1, [], text

    monkeypatch.setattr(tailoring, "_compile_one_page", compile_once)

    result = tailoring.tailor_resume(payload)

    assert result.status == "OK"
    assert sum(change.section == "skills" for change in result.change_log) == 1
    skill_change = next(
        change for change in result.change_log if change.section == "skills"
    )
    assert "Kubernetes" not in skill_change.before_text
    assert "Kubernetes" in skill_change.after_text
    assert skill_change.evidence_ids
    assert skill_change.source_location.startswith("Skills item")
    assert skill_change.validation_result.startswith("validated")


def test_project_swap_preserves_existing_entry_skeleton() -> None:
    source = Path("data/resume.tex").read_text(encoding="utf-8")
    structure = parse_resume_structure(source, require_projects=True)
    payload = _input(Path("data/resume.tex"))
    evidence = {item.evidence_id: item for item in payload.candidate_evidence}

    edit, record = tailoring._plan_project_swap(
        source, structure, payload, evidence
    )

    assert edit is not None
    before = edit.target.text(source)
    after = edit.replacement
    assert record.name == "AI CRM Platform"
    for command in (
        "\\resumeEntry",
        "\\resumeItemListStart",
        "\\resumeItem",
        "\\resumeItemListEnd",
    ):
        assert before.count(command) == after.count(command)
    assert "Cardiovascular Flow and Stenosis Analysis" not in after
    assert "AI CRM Platform" in after


def test_overflow_revision_changes_content_not_layout() -> None:
    original = Path("data/resume.tex").read_text(encoding="utf-8")
    current = original.replace(
        "AI/ML engineer and Ph.D. researcher",
        "Candidate targeting machine learning roles. "
        "AI/ML engineer and Ph.D. researcher",
    )
    structure = parse_resume_structure(current)
    edits = []
    for ordinal in (1, 2):
        item = structure.experience_bullets[ordinal - 1]
        edits.append(
            tailoring.SourceEdit(
                item.content_range,
                "Highlighted relevant work that " + item.content(current),
                "experience",
                f"Experience bullet {ordinal}",
            )
        )
    current = tailoring._apply_source_edits(current, edits)

    revised = tailoring._revise_edited_content(current, original, [1, 2], 1)

    assert "\\documentclass[letterpaper,11pt]{article}" in revised
    assert "\\addtolength{\\textwidth}{1in}" in revised
    assert "\\titlespacing*{\\section}{0pt}{8pt}{4pt}" in revised
    tailoring._assert_exactly_two_experience_bullets_changed(original, revised)


def test_compile_retries_overflow_with_content_revision_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tex_path = tmp_path / "resume.tex"
    pdf_path = tmp_path / "resume.pdf"
    calls = 0

    def compile_pdf(_: Path) -> list[str]:
        nonlocal calls
        calls += 1
        writer = PdfWriter()
        for _ in range(2 if calls == 1 else 1):
            writer.add_blank_page(width=612, height=792)
        with pdf_path.open("wb") as handle:
            writer.write(handle)
        return []

    source = (
        "\\documentclass[11pt]{article}\n"
        "\\begin{document}\nlong newly edited summary\n\\end{document}\n"
    )
    monkeypatch.setattr(tailoring, "_run_pdflatex", compile_pdf)

    pages, errors, compiled = tailoring._compile_one_page(
        source,
        tex_path,
        pdf_path,
        revision=lambda working, _: working.replace(
            "long newly edited summary", "short summary"
        ),
    )

    assert pages == 1
    assert errors == []
    assert calls == 2
    assert "short summary" in compiled
    assert compiled.startswith("\\documentclass[11pt]{article}")
