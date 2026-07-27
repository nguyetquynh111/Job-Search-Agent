"""Repository layout invariants."""

from pathlib import Path

_IGNORED_DIRECTORIES = {".git", ".venv", "node_modules", "venv"}


def test_source_tree_contains_no_markdown_files() -> None:
    markdown_files = sorted(
        path for path in Path("src").rglob("*.md") if path.name != "prompt.md"
    )

    assert markdown_files == []


def test_python_filenames_do_not_start_with_underscore() -> None:
    violations = sorted(
        path
        for path in Path(".").rglob("_*.py")
        if path.name != "__init__.py"
        and not _IGNORED_DIRECTORIES.intersection(path.parts)
    )

    assert violations == []


def test_legacy_shared_tools_directory_is_absent() -> None:
    assert not Path("src/tools/_shared").exists()


def test_tests_are_grouped_into_a_short_domain_tree() -> None:
    assert not Path("src/tests").exists()
    directories = {
        path.name
        for path in Path("tests").iterdir()
        if path.is_dir() and path.name != "__pycache__"
    }
    assert directories == {"fixtures"}


def test_tools_contains_required_packages_and_registry_boundary() -> None:
    source_files = {
        path.name
        for path in Path("src/tools").glob("*.py")
        if path.name != "__init__.py"
    }

    assert source_files == {"registry.py"}

    tool_directories = {
        path.name
        for path in Path("src/tools").iterdir()
        if path.is_dir() and path.name != "__pycache__"
    }

    assert tool_directories == {
        "filtering_scoring",
        "fit_analysis",
        "resume_tailoring",
        "cover_letter",
    }


def test_legacy_responsibility_locations_are_absent() -> None:
    assert not Path("src/observability").exists()
    assert not Path("src/memory/models.py").exists()
    assert not Path("src/shared").exists()
    assert not Path("src/data_loader.py").exists()
    assert Path("src/agent").is_dir()
    assert Path("src/agent/controller.py").is_file()
    assert Path("src/agent/graph.py").is_file()
    assert Path("src/agent/state.py").is_file()
    assert Path("app/streamlit_app.py").is_file()


def test_shared_helpers_live_under_utils() -> None:
    assert Path("src/utils").is_dir()
    assert not Path("src/evidence_validation.py").exists()
    assert not Path("src/input_loading.py").exists()
    assert not Path("src/job_evidence.py").exists()
    assert not Path("src/latex.py").exists()
    assert not Path("src/output_validation.py").exists()
    assert not Path("src/skill_matching.py").exists()


def test_single_agent_source_has_no_worker_or_coordinator_abstractions() -> None:
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(Path("src").rglob("*.py"))
        if "tests" not in path.parts
    ).casefold()

    assert "class supervisor" not in source
    assert "class workeragent" not in source
    assert "class coordinator" not in source
    assert source.count("you are the only llm agent") == 1
