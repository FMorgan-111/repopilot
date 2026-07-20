"""Bounded source search over an exact local Git checkout."""

from __future__ import annotations

import ast
import re
import subprocess
from collections.abc import Sequence
from pathlib import Path

_DOC_SUFFIXES = (".md", ".rst", ".txt", ".rdoc", ".adoc")
_MAX_TOOL_CONTENT = 8_000


def _tracked_text_files(
    repo_path: str,
    *,
    relative_to: str = "",
    max_files: int = 4_000,
    max_file_bytes: int = 256_000,
) -> list[tuple[str, str]]:
    root = Path(repo_path).resolve()
    scope = (root / relative_to).resolve() if relative_to else root
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        capture_output=True,
        check=True,
    )
    files: list[tuple[str, str]] = []
    for raw_path in result.stdout.split(b"\0")[:max_files]:
        if not raw_path:
            continue
        path = raw_path.decode("utf-8", errors="replace")
        file_path = (root / path).resolve()
        if (
            not file_path.is_relative_to(root)
            or not file_path.is_relative_to(scope)
            or not file_path.is_file()
            or file_path.stat().st_size > max_file_bytes
        ):
            continue
        try:
            files.append((path, file_path.read_text(encoding="utf-8")))
        except (UnicodeDecodeError, OSError):
            continue
    return files


def _bounded_lines(matches: list[str]) -> str:
    return "\n".join(matches)[:_MAX_TOOL_CONTENT]


def search_local_symbol(repo_path: str, symbol: str, path: str = "") -> str:
    pattern = re.compile(rf"\b{re.escape(symbol)}\b")
    matches: list[str] = []
    for file_path, content in _tracked_text_files(repo_path, relative_to=path):
        for line_number, line in enumerate(content.splitlines(), start=1):
            if pattern.search(line):
                matches.append(f"{file_path}:{line_number}: {line.strip()}")
    return _bounded_lines(matches)


def search_local_text(repo_path: str, text: str, path: str = "") -> str:
    matches: list[str] = []
    for file_path, content in _tracked_text_files(repo_path, relative_to=path):
        for line_number, line in enumerate(content.splitlines(), start=1):
            if text in line:
                matches.append(f"{file_path}:{line_number}: {line.strip()}")
    return _bounded_lines(matches)


def read_local_range(
    repo_path: str,
    path: str,
    start_line: int,
    end_line: int,
) -> str:
    root = Path(repo_path).resolve()
    file_path = (root / path).resolve()
    content = file_path.read_text(encoding="utf-8").splitlines()
    selected = [
        f"{number}: {content[number - 1]}"
        for number in range(start_line, min(end_line, len(content)) + 1)
    ]
    return _bounded_lines(selected)


def read_local_symbol(repo_path: str, path: str, symbol: str) -> str:
    root = Path(repo_path).resolve()
    file_path = (root / path).resolve()
    content = file_path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return _read_non_python_symbol(content, symbol)
    parts = symbol.split(".")
    candidates: list[ast.AST] = list(tree.body)
    target: ast.AST | None = None
    for part in parts:
        target = next(
            (
                node
                for node in candidates
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == part
            ),
            None,
        )
        if target is None:
            return ""
        candidates = list(getattr(target, "body", []))
    lines = content.splitlines()
    start = int(getattr(target, "lineno", 1))
    end = int(getattr(target, "end_lineno", start))
    return _bounded_lines(lines[start - 1 : end])


def _read_non_python_symbol(content: str, symbol: str) -> str:
    """Return one bounded declaration block for languages without a Python AST."""
    name = symbol.rsplit(".", 1)[-1]
    declaration = re.compile(
        rf"\b(?:class|interface|struct|enum|function|fn|func|def)\s+{re.escape(name)}\b"
        rf"|\b{re.escape(name)}\s*\([^\n]*\)\s*(?:\{{|=>)"
    )
    lines = content.splitlines()
    start = next((index for index, line in enumerate(lines) if declaration.search(line)), None)
    if start is None:
        return ""
    selected: list[str] = []
    brace_depth = 0
    saw_brace = False
    base_indent = len(lines[start]) - len(lines[start].lstrip())
    for line in lines[start : start + 400]:
        if selected and not saw_brace:
            indent = len(line) - len(line.lstrip())
            if line.strip() and indent <= base_indent:
                break
        selected.append(line)
        brace_depth += line.count("{") - line.count("}")
        saw_brace = saw_brace or "{" in line
        if saw_brace and brace_depth <= 0:
            break
    return _bounded_lines(selected)


def find_local_references(repo_path: str, symbol: str, path: str = "") -> str:
    return search_local_symbol(repo_path, symbol, path)


def list_local_related_tests(repo_path: str, path: str) -> list[str]:
    source_stem = Path(path).stem.lower()
    related: list[str] = []
    for file_path, content in _tracked_text_files(repo_path):
        lowered_path = file_path.lower()
        is_test = (
            lowered_path.startswith("tests/")
            or "/tests/" in lowered_path
            or Path(lowered_path).name.startswith("test_")
            or ".test." in lowered_path
            or ".spec." in lowered_path
        )
        if is_test and (source_stem in lowered_path or source_stem in content.lower()):
            related.append(file_path)
    return related[:100]


def search_local_checkout(
    repo_path: str,
    terms: Sequence[str],
    max_files: int = 4000,
    max_file_bytes: int = 256_000,
) -> list[dict[str, str]]:
    """Return the best tracked source files containing any search term."""
    lowered_terms = [term.lower() for term in terms if term]
    candidates: list[tuple[int, dict[str, str]]] = []

    for path, content in _tracked_text_files(
        repo_path,
        max_files=max_files,
        max_file_bytes=max_file_bytes,
    ):
        lowered_path = path.lower()
        if (
            lowered_path.endswith(_DOC_SUFFIXES)
            or lowered_path.startswith("docs/")
            or "/docs/" in lowered_path
        ):
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
