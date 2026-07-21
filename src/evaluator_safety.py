"""Shared removal boundary for evaluator-only metadata and values."""

from __future__ import annotations

import re

_EVALUATOR_ONLY_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9])(?:gold[_ -]?patch|test[_ -]?patch|"
    r"fail[_ -]?to[_ -]?pass|pass[_ -]?to[_ -]?pass|instance[_ -]?id|"
    r"evaluator|grading|oracle)(?![A-Za-z0-9])"
)


def contains_evaluator_only(value: object) -> bool:
    """Return whether a value contains an evaluator-only field marker."""
    return _EVALUATOR_ONLY_RE.search(str(value or "")) is not None


def sanitize_evaluator_text(value: object) -> str:
    """Remove complete lines and indented blocks containing evaluator metadata.

    Substituting just a field name is unsafe because its value would remain.  A
    marker therefore removes its whole line; a marker used as a block header
    also removes following lines until the next blank line.
    """
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    safe_lines: list[str] = []
    skipping_block = False
    for line in text.split("\n"):
        if skipping_block:
            if line.strip():
                continue
            skipping_block = False
            if safe_lines and safe_lines[-1] != "":
                safe_lines.append("")
            continue
        marker = _EVALUATOR_ONLY_RE.search(line)
        if marker is None:
            safe_lines.append(line)
            continue
        suffix = line[marker.end() :].strip()
        if not suffix or suffix in {":", "=", "-"}:
            skipping_block = True
    return "\n".join(safe_lines).strip()


def safe_prediction_patch(value: object) -> str:
    """Reject an entire patch if it contains evaluator-only metadata."""
    text = str(value or "")
    return "" if contains_evaluator_only(text) else text


__all__ = [
    "contains_evaluator_only",
    "safe_prediction_patch",
    "sanitize_evaluator_text",
]
