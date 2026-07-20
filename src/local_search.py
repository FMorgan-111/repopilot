"""Bounded source search over an exact local Git checkout."""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path

_DOC_SUFFIXES = (".md", ".rst", ".txt", ".rdoc", ".adoc")


def search_local_checkout(
    repo_path: str,
    terms: Sequence[str],
    max_files: int = 4000,
    max_file_bytes: int = 256_000,
) -> list[dict[str, str]]:
    """Return the best tracked source files containing any search term."""
    root = Path(repo_path).resolve()
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        capture_output=True,
        check=True,
    )
    lowered_terms = [term.lower() for term in terms if term]
    candidates: list[tuple[int, dict[str, str]]] = []

    for raw_path in result.stdout.split(b"\0")[:max_files]:
        if not raw_path:
            continue
        path = raw_path.decode("utf-8", errors="replace")
        lowered_path = path.lower()
        if (
            lowered_path.endswith(_DOC_SUFFIXES)
            or lowered_path.startswith("docs/")
            or "/docs/" in lowered_path
        ):
            continue
        file_path = (root / path).resolve()
        if (
            root not in file_path.parents
            or not file_path.is_file()
            or file_path.stat().st_size > max_file_bytes
        ):
            continue
        try:
            content = file_path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        haystack = f"{lowered_path}\n{content.lower()}"
        score = sum(haystack.count(term) for term in lowered_terms)
        if score:
            candidates.append(
                (
                    score,
                    {"path": path, "content": content, "sha": ""},
                )
            )

    candidates.sort(key=lambda pair: (-pair[0], pair[1]["path"]))
    return [item for _, item in candidates[:20]]
