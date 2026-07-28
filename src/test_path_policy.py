"""Shared fail-closed admission policy for Python test paths."""

from __future__ import annotations

import os
import re
import stat
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
    """Fail-closed bool wrapper for callers that cannot classify I/O faults."""
    try:
        return inspect_allowed_test_path(repo_root, file_path)
    except (OSError, RuntimeError):
        return False


def _lstat_or_none(path: Path) -> os.stat_result | None:
    try:
        return os.lstat(path)
    except FileNotFoundError:
        return None


def inspect_allowed_test_path(repo_root: Path, file_path: str) -> bool:
    """Inspect test-path admission while preserving filesystem failures."""
    try:
        root = Path(repo_root).resolve(strict=True)
        relative = canonical_repo_path(file_path)
    except ValueError:
        return False
    root_info = os.lstat(root)
    if (
        not stat.S_ISDIR(root_info.st_mode)
        or contains_evaluator_only(relative)
        or not _TEST_NAME_RE.fullmatch(Path(relative).name)
    ):
        return False

    parts = Path(relative).parts
    current = root
    destination_info: os.stat_result | None = None
    for index, part in enumerate(parts):
        current = current / part
        info = _lstat_or_none(current)
        if info is None:
            break
        if stat.S_ISLNK(info.st_mode):
            return False
        if index < len(parts) - 1 and not stat.S_ISDIR(info.st_mode):
            return False
        if index == len(parts) - 1:
            destination_info = info
    if destination_info is not None and not stat.S_ISREG(destination_info.st_mode):
        return False

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
        established_info = _lstat_or_none(established)
        return established_info is not None and stat.S_ISDIR(established_info.st_mode)

    destination = root / relative
    parent_info = _lstat_or_none(destination.parent)
    if parent_info is None or not stat.S_ISDIR(parent_info.st_mode):
        return False
    if destination_info is not None:
        return True
    with os.scandir(destination.parent) as entries:
        return any(
            entry.name != destination.name
            and entry.is_file(follow_symlinks=False)
            and _TEST_NAME_RE.fullmatch(entry.name)
            for entry in entries
        )


__all__ = [
    "inspect_allowed_test_path",
    "is_allowed_test_path",
    "path_has_symlink",
]
