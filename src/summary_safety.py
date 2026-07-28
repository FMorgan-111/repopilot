"""Context-free safety boundary for persisted rolling summaries."""

from __future__ import annotations

import re
from collections.abc import Iterable

from .model_provider import redact_secrets

_EVALUATOR_FIELD_RE = re.compile(
    r"(?i)\b(?:gold[_ -]?patch|test[_ -]?patch|FAIL_TO_PASS|PASS_TO_PASS)\b"
)
_PATCH_FIELD_RE = re.compile(
    r'''(?ix)(?:\{\s*)?(?<![\w])(?:"patch"|'patch'|patch)\s*[:=]'''
)
_RAW_HTTP_MARKER_RE = re.compile(
    r"(?i)(?:\braw[\s_-]+HTTP\b|HTTP/\d(?:\.\d)?\b|"
    r"(?:GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS)\s+https?://|"
    r"(?:raw[\s_-]+)?HTTP[\s_-]+(?:request|response|payload|body|headers?)\b|"
    r"(?:request|response)[\s_-]+(?:payload|body|headers?)\s*:)"
)
_GENERATED_TEST_PATH_RE = re.compile(
    r"(?i)\b(?:tests?|test)/[^\s\"']*generated[^\s\"']*"
)


def sanitize_summary_text(
    value: object,
    limit: int = 200,
    *,
    denied_literals: Iterable[str] = (),
) -> str:
    """Redact, stop at forbidden content, normalize, and cap Unicode text."""
    text = redact_secrets(str(value or ""))
    boundary_offsets = [
        marker.start()
        for marker in (
            _EVALUATOR_FIELD_RE.search(text),
            _PATCH_FIELD_RE.search(text),
            _RAW_HTTP_MARKER_RE.search(text),
            _GENERATED_TEST_PATH_RE.search(text),
        )
        if marker is not None
    ]
    boundary_offsets.extend(
        offset
        for literal in denied_literals
        if literal and (offset := text.find(literal)) >= 0
    )
    if boundary_offsets:
        text = text[: min(boundary_offsets)]
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    return normalized[: max(0, limit)]


def sanitize_model_context(
    value: object,
    limit: int,
    *,
    denied_literals: Iterable[str] = (),
) -> str:
    """Expose the shared bounded/redacted boundary for model-facing context."""
    return sanitize_summary_text(
        value,
        limit,
        denied_literals=denied_literals,
    )


__all__ = ["sanitize_model_context", "sanitize_summary_text"]
