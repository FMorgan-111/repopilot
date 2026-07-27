"""Shared fail-closed admission policy for Python test paths."""

from __future__ import annotations

import re
from pathlib import Path

from .evaluator_safety import contains_evaluator_only
from .repo_paths import canonical_repo_path

_TEST_NAME_RE = re.compile(r"(?:test_.+|.+_test)\.py\Z", re.IGNORECASE)
_FORBIDDEN_TEST_TREE_PARTS = frozenset(
    {
        ".cache",
        ".circleci",
        ".github",
        ".gitlab",
        ".mypy_cache",
        ".nox",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        "__pycache__",
        "build",
        "ci",
        "config",
        "configs",
        "dependencies",
        "dependency",
        "dist",
        "generated",
        "htmlcov",
        "node_modules",
        "site-packages",
        "third_party",
        "vendor",
        "venv",
        "workflow",
        "workflows",
    }
)


def path_has_symlink(root: Path, relative: str) -> bool:
    current = root
    for part in Path(relative).parts:
        current = current / part
        if current.is_symlink():
            return True
        if not current.exists():
            break
    return False


def is_allowed_test_path(repo_root: Path, file_path: str) -> bool:
    """Return whether a canonical path fits an established safe test layout."""
    try:
        root = repo_root.resolve(strict=True)
        relative = canonical_repo_path(file_path)
    except (OSError, ValueError):
        return False
    if (
        not root.is_dir()
        or contains_evaluator_only(relative)
        or not _TEST_NAME_RE.fullmatch(Path(relative).name)
    ):
        return False
    if path_has_symlink(root, relative):
        return False
    destination = (root / relative).resolve(strict=False)
    if not destination.is_relative_to(root):
        return False
    if destination.exists() and not destination.is_file():
        return False

    parts = Path(relative).parts
    if any(part.casefold() in _FORBIDDEN_TEST_TREE_PARTS for part in parts[:-1]):
        return False
    test_root_index = next(
        (
            index
            for index, part in enumerate(parts[:-1])
            if part.casefold() in {"test", "tests"}
        ),
        None,
    )
    if test_root_index is not None:
        established = root.joinpath(*parts[: test_root_index + 1])
        return established.is_dir() and not established.is_symlink()

    parent = destination.parent
    if not parent.is_dir() or parent.is_symlink():
        return False
    return destination.is_file() or any(
        item != destination
        and item.is_file()
        and not item.is_symlink()
        and _TEST_NAME_RE.fullmatch(item.name)
        for item in parent.iterdir()
    )


__all__ = ["is_allowed_test_path", "path_has_symlink"]
