"""Repository layout invariants."""

from pathlib import Path

_IGNORED_DIRECTORIES = {".git", ".venv", "node_modules", "venv"}


def test_source_tree_contains_no_markdown_files() -> None:
    markdown_files = sorted(Path("src").rglob("*.md"))

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


def test_tests_are_grouped_by_domain() -> None:
    assert sorted(Path("src/tests").glob("test_*.py")) == []
