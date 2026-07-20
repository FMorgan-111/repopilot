"""Canonical repository-path validation shared by safety boundaries."""

from __future__ import annotations

import unicodedata
from pathlib import Path


def canonical_repo_path(value: object, *, allow_dot: bool = False) -> str:
    """Return an unchanged NFC relative path or fail closed."""
    if not isinstance(value, str) or not value:
        raise ValueError("repository path must be a non-empty string")
    if value == "." and allow_dot:
        return value
    if value != unicodedata.normalize("NFC", value):
        raise ValueError("repository path is not NFC-normalized")
    if "\\" in value or Path(value).is_absolute():
        raise ValueError("repository path is not canonical and relative")
    if any(unicodedata.category(character).startswith("C") for character in value):
        raise ValueError("repository path contains a control character")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("repository path contains an unsafe component")
    if len(value.encode("utf-8")) > 4096:
        raise ValueError("repository path is too long")
    return value


def canonical_repo_path_key(value: object, *, allow_dot: bool = False) -> str:
    return canonical_repo_path(value, allow_dot=allow_dot).casefold()
