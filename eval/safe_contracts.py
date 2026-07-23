"""Shared trust boundaries for persisted and rendered evaluation results.

This module intentionally uses only the standard library so the harness,
taxonomy CLI, and report CLI can all import the same validators without an
``eval``/``src`` import cycle.
"""

from __future__ import annotations

import math
import re
import sys
import unicodedata
from pathlib import Path, PurePosixPath
from typing import Any

if not __package__:  # Support documented direct eval script entry points.
    repo_root = str(Path(__file__).resolve().parents[1])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

from src.exception_safety import (
    MODEL_GATEWAY_ERROR_CLASSES,
)
from src.exception_safety import (
    normalize_exception_class as _normalize_exception_class,
)

normalize_exception_class = _normalize_exception_class

VERIFIED_COVERAGE_STATUSES = frozenset(
    {"existing_verified", "generated_verified"}
)

_PROOF_KEYS = frozenset(
    {"source", "status", "test_files", "fixed_runs", "base_runs"}
)
_RUN_KEYS = frozenset(
    {"outcome", "failing_test_ids", "assertion_fingerprint"}
)
_SOURCE_STATUS = {
    "existing": "existing_verified",
    "generated": "generated_verified",
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_TEST_FILE_RE = re.compile(r"(?:test_[^/]+|[^/]+_test)\.py", re.IGNORECASE)
_EVALUATOR_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9])(?:gold[_ -]?patch|test[_ -]?patch|"
    r"fail[_ -]?to[_ -]?pass|pass[_ -]?to[_ -]?pass)"
    r"(?![A-Za-z0-9])"
)
_BEARER_RE = re.compile(r"(?i)(\bBearer\s+)[^\s,;]+")
_QUERY_SECRET_RE = re.compile(
    r"(?i)([?&](?:api[-_]?key|access[-_]?token|token|key|authorization)=)"
    r"[^&#\s;]+"
)
_QUOTED_FIELD_SECRET_RE = re.compile(
    r"(?i)(\b(?:authorization|proxy-authorization|x-api-key|api[-_]?key|"
    r"access[-_]?token|token)\b\s*[:=]\s*|"
    r"[\"'](?:authorization|proxy-authorization|x-api-key|api[-_]?key|"
    r"access[-_]?token|token)[\"']\s*[:=]\s*)"
    r"([\"'])(?:Bearer\s+)?[^\"']*\2"
)
_ASSIGNED_SECRET_RE = re.compile(
    r"(?i)((?<![?&])\b(?:authorization|proxy-authorization|x-api-key|api[-_]?key|"
    r"access[-_]?token|token)\b\s*[:=]\s*)"
    r"(?:Bearer\s+)?(?![\"'])[^\s,;]+"
)
_ENV_ASSIGNMENT_RE = re.compile(
    r"(?i)(?<![?&A-Za-z0-9_])"
    r"(?P<prefix>(?P<name>[A-Za-z_][A-Za-z0-9_]*)[ \t]*=[ \t]*)"
    r"(?P<value>\"(?:\\.|[^\"\\\r\n])*\"|"
    r"'(?:\\.|[^'\\\r\n])*'|[^\s,;#]+)"
)
_SENSITIVE_ENV_SUFFIXES = (
    "ACCESS_TOKEN",
    "AUTHORIZATION",
    "API_KEY",
    "APIKEY",
    "TOKEN",
)
_SK_RE = re.compile(r"(?<![A-Za-z0-9_-])sk-[A-Za-z0-9_-]{8,}")
_ARCHIVE_PAYLOAD_RE = re.compile(r"(?:PK\\x03\\x04|PK\x03\x04)")
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
_SENSITIVE_PATH_PARTS = frozenset(
    {
        ".aws",
        ".azure",
        ".config",
        ".env",
        ".gcloud",
        ".git",
        ".gnupg",
        ".hg",
        ".kube",
        ".ssh",
        ".svn",
        "credentials",
        "secrets",
    }
)
MODEL_INVOCATION_NODES = frozenset(
    {
        "plan",
        "plan_fix",
        "reflect",
        "reflect_on_failure",
        "test_generation",
        "test_generator",
        "outcome_summary",
    }
)
_GENERATE_PLAN_GATEWAY_RE = re.compile(
    r"(?i)failed to generate fix plan:\s*(?:"
    + "|".join(sorted(MODEL_GATEWAY_ERROR_CLASSES))
    + r")\b"
)


def sanitize_output_text(value: Any, limit: int = 2_000) -> str:
    """Return bounded scalar text with secrets/evaluator payloads removed."""
    if value is None:
        return ""
    if not isinstance(value, (str, int, float, bool)):
        return ""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    text = _ENV_ASSIGNMENT_RE.sub(_redact_env_assignment, text)
    text = _QUERY_SECRET_RE.sub(r"\1[REDACTED]", text)
    text = _QUOTED_FIELD_SECRET_RE.sub(
        lambda match: (
            f"{match.group(1)}{match.group(2)}[REDACTED]{match.group(2)}"
        ),
        text,
    )
    text = _ASSIGNED_SECRET_RE.sub(r"\1[REDACTED]", text)
    text = _BEARER_RE.sub(r"\1[REDACTED]", text)
    text = _SK_RE.sub("[REDACTED]", text)
    boundaries = [
        match.start()
        for match in (_EVALUATOR_RE.search(text), _ARCHIVE_PAYLOAD_RE.search(text))
        if match is not None
    ]
    if boundaries:
        text = text[: min(boundaries)]
    return text.strip()[: max(0, limit)]


def _redact_env_assignment(match: re.Match[str]) -> str:
    name = match.group("name")
    if not name.upper().endswith(_SENSITIVE_ENV_SUFFIXES):
        return match.group(0)
    value = match.group("value")
    quote = value[0] if value[:1] in {"'", '"'} else ""
    return f"{match.group('prefix')}{quote}[REDACTED]{quote}"


def safe_int(value: Any, default: int = 0) -> int:
    """Normalize an integer field without treating booleans as counters."""
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    """Normalize a finite floating point field."""
    if isinstance(value, bool):
        return default
    try:
        normalized = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return normalized if math.isfinite(normalized) else default


def normalize_official_resolved(value: Any) -> bool | None:
    """Only an actual JSON boolean is an official scorer result."""
    return value if type(value) is bool else None


def _canonical_test_path(value: Any) -> str | None:
    if not isinstance(value, str) or not value or len(value.encode("utf-8")) > 4096:
        return None
    if value != unicodedata.normalize("NFC", value) or "\\" in value:
        return None
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        return None
    path = PurePosixPath(value)
    parts = path.parts
    if (
        path.is_absolute()
        or not parts
        or value.endswith("/")
        or "//" in value
        or any(part in {"", ".", ".."} for part in parts)
        or path.as_posix() != value
    ):
        return None
    lowered = tuple(part.casefold() for part in parts)
    if any(
        part in _FORBIDDEN_TEST_TREE_PARTS or part in _SENSITIVE_PATH_PARTS
        for part in lowered[:-1]
    ):
        return None
    filename = parts[-1]
    if (
        not _TEST_FILE_RE.fullmatch(filename)
        or _EVALUATOR_RE.search(value)
        or filename.casefold().startswith(".env")
    ):
        return None
    return value


def _validate_run(
    value: Any,
    *,
    expected_outcome: str,
    test_files: frozenset[str],
) -> dict[str, Any] | None:
    if not isinstance(value, dict) or frozenset(value) != _RUN_KEYS:
        return None
    outcome = value.get("outcome")
    failing_ids = value.get("failing_test_ids")
    fingerprint = value.get("assertion_fingerprint")
    if outcome != expected_outcome or not isinstance(failing_ids, list):
        return None
    if not isinstance(fingerprint, str):
        return None
    if expected_outcome == "pass":
        if failing_ids != [] or fingerprint != "":
            return None
        return {
            "outcome": "pass",
            "failing_test_ids": [],
            "assertion_fingerprint": "",
        }
    if not 1 <= len(failing_ids) <= 100 or not _SHA256_RE.fullmatch(fingerprint):
        return None
    safe_ids: list[str] = []
    for test_id in failing_ids:
        if (
            not isinstance(test_id, str)
            or not 1 <= len(test_id) <= 500
            or any(
                ord(character) < 32 or ord(character) == 127
                for character in test_id
            )
            or sanitize_output_text(test_id, 500) != test_id
        ):
            return None
        path, separator, selector = test_id.partition("::")
        if not separator or not selector or path not in test_files:
            return None
        safe_ids.append(test_id)
    if len(safe_ids) != len(set(safe_ids)):
        return None
    return {
        "outcome": "assertion_failure",
        "failing_test_ids": safe_ids,
        "assertion_fingerprint": fingerprint,
    }


def validate_safe_coverage_proof(value: Any) -> dict[str, Any] | None:
    """Validate the complete public differential-coverage proof projection."""
    if not isinstance(value, dict) or frozenset(value) != _PROOF_KEYS:
        return None
    source = value.get("source")
    status = value.get("status")
    if _SOURCE_STATUS.get(source) != status:
        return None
    raw_test_files = value.get("test_files")
    if not isinstance(raw_test_files, list) or not 1 <= len(raw_test_files) <= 16:
        return None
    test_files: list[str] = []
    for raw_path in raw_test_files:
        canonical = _canonical_test_path(raw_path)
        if canonical is None:
            return None
        test_files.append(canonical)
    if len(test_files) != len(set(test_files)):
        return None
    test_file_set = frozenset(test_files)
    fixed_values = value.get("fixed_runs")
    base_values = value.get("base_runs")
    if (
        not isinstance(fixed_values, list)
        or not isinstance(base_values, list)
        or len(fixed_values) != 2
        or len(base_values) != 2
    ):
        return None
    fixed_runs = [
        _validate_run(item, expected_outcome="pass", test_files=test_file_set)
        for item in fixed_values
    ]
    base_runs = [
        _validate_run(
            item,
            expected_outcome="assertion_failure",
            test_files=test_file_set,
        )
        for item in base_values
    ]
    if any(item is None for item in (*fixed_runs, *base_runs)):
        return None
    if base_runs[0] != base_runs[1]:
        return None
    return {
        "source": source,
        "status": status,
        "test_files": test_files,
        "fixed_runs": fixed_runs,
        "base_runs": base_runs,
    }


def has_verified_coverage_proof(status: Any, proof: Any) -> bool:
    validated = validate_safe_coverage_proof(proof)
    return bool(
        status in VERIFIED_COVERAGE_STATUSES
        and validated is not None
        and validated["status"] == status
    )


def has_verified_agent_success(sample: Any) -> bool:
    if not isinstance(sample, dict):
        return False
    claimed = sample.get("agent_success", sample.get("success", False))
    return bool(
        claimed is True
        and has_verified_coverage_proof(
            sample.get("coverage_status"), sample.get("coverage_proof")
        )
    )


def is_model_gateway_error_text(value: Any) -> bool:
    return bool(
        isinstance(value, str) and _GENERATE_PLAN_GATEWAY_RE.search(value)
    )


def has_structured_model_gateway_failure(sample: Any) -> bool:
    """Recognize only allowlisted, bounded model failure evidence."""
    if not isinstance(sample, dict):
        return False
    payload = sample.get("agent_payload")
    payload = payload if isinstance(payload, dict) else {}
    sequences = (
        sample.get("model_invocations"),
        payload.get("model_invocations"),
        payload.get("model_history"),
    )
    for sequence in sequences:
        if not isinstance(sequence, list):
            continue
        for invocation in sequence[:1_000]:
            if not isinstance(invocation, dict):
                continue
            if (
                invocation.get("node") in MODEL_INVOCATION_NODES
                and invocation.get("status") == "error"
                and invocation.get("error_class") in MODEL_GATEWAY_ERROR_CLASSES
            ):
                return True
    return False


def has_model_gateway_failure_code(sample: Any) -> bool:
    """Recognize an explicit terminal gateway code without inspecting history."""
    return bool(
        isinstance(sample, dict)
        and any(
            sample.get(key) == "model_gateway_infra"
            for key in ("failure_code", "model_failure_code")
        )
    )
