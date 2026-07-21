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
    """Truncate at the first line bearing evaluator-only metadata."""
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    safe_lines: list[str] = []
    for line in text.split("\n"):
        if _EVALUATOR_ONLY_RE.search(line) is not None:
            break
        safe_lines.append(line)
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
