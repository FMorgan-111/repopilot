"""Bounded, redacted evidence from exceptions hidden by phase timeouts."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Literal

from .async_safety import CancellationDrainError
from .exception_safety import normalize_exception_class
from .model_provider import redact_secrets

_MAX_CAUSE_DEPTH = 8
_MAX_OPERATION = 120
_MAX_EXCEPTION_CLASS = 120
_MAX_ERROR_SUMMARY = 300
_MAX_EVIDENCE_SUMMARY = 480

TimeoutFailureKind = Literal[
    "pr_cleanup", "pr_transaction", "generic_drain"
]


@dataclass(frozen=True)
class TimeoutCleanupEvidence:
    """Safe details proving that timeout cancellation draining also failed."""

    failure_kind: TimeoutFailureKind
    cause_type: str
    cleanup_error_type: str
    cleanup_error: str
    operation: str = ""
    pr_number: int | None = None

    def summary(self) -> str:
        pr_number = _rendered_pr_number(
            self.failure_kind, self.pr_number
        )
        if self.failure_kind == "pr_cleanup":
            target = _pr_target(pr_number)
            message = f"PR cancellation cleanup failed for {target}"
        elif self.failure_kind == "pr_transaction":
            target = _pr_target(pr_number)
            message = f"PR cancellation transaction failed for {target}"
        else:
            message = f"Cancellation cleanup failed during {self.operation}"
        cleanup_error_type = _safe_exception_class(
            self.cleanup_error_type
        )
        rendered = (
            f"{message} ({cleanup_error_type}: {self.cleanup_error})"
        )
        return _safe_text(rendered, _MAX_EVIDENCE_SUMMARY)

    def diagnostic_details(self) -> dict[str, object]:
        details: dict[str, object] = {
            "timeout_cleanup_kind": self.failure_kind,
            "timeout_cause_type": _safe_exception_class(self.cause_type),
            "cleanup_error_type": _safe_exception_class(
                self.cleanup_error_type
            ),
            "cleanup_error": self.cleanup_error,
        }
        if self.failure_kind == "generic_drain":
            details["cleanup_operation"] = self.operation
        pr_number = _rendered_pr_number(
            self.failure_kind, self.pr_number
        )
        if pr_number is not None:
            details["cleanup_pr_number"] = pr_number
        return details


def _pr_target(pr_number: int | None) -> str:
    if pr_number is not None:
        return f"pull request {pr_number}"
    return "an unknown pull request"


def _safe_text(value: str, limit: int) -> str:
    redacted = redact_secrets(value)
    normalized = " ".join(redacted.split())
    return normalized[:limit]


def _safe_exception_class(value: object) -> str:
    normalized = normalize_exception_class(value)
    if normalized and len(normalized) <= _MAX_EXCEPTION_CLASS:
        return normalized
    return "BaseException"


def _safe_exception_type(error: BaseException) -> str:
    return _safe_exception_class(error)


def _safe_error_summary(error: BaseException) -> str:
    try:
        message = str(error)
    except Exception:
        message = ""
    normalized = _safe_text(message, _MAX_ERROR_SUMMARY)
    return normalized or _safe_exception_type(error)


def _safe_operation(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return _safe_text(value, _MAX_OPERATION)


def _positive_pr_number(value: object) -> int | None:
    if type(value) is int and value > 0:
        return value
    return None


def _rendered_pr_number(
    failure_kind: TimeoutFailureKind, value: object
) -> int | None:
    if failure_kind not in {"pr_cleanup", "pr_transaction"}:
        return None
    return _positive_pr_number(value)


def _evidence_for_drain(
    drain: CancellationDrainError,
) -> TimeoutCleanupEvidence | None:
    attributes = vars(drain)
    cause_type = _safe_exception_type(drain)
    drain_type = type(drain).__name__
    if drain_type == "PRCancellationCleanupError":
        failure_kind: TimeoutFailureKind = "pr_cleanup"
        failure = attributes.get("cleanup_error")
    elif drain_type == "PRCancellationTransactionError":
        failure_kind = "pr_transaction"
        failure = attributes.get("transaction_error")
    else:
        failure_kind = "generic_drain"
        failure = attributes.get("cleanup_error")
    if not isinstance(failure, BaseException):
        return None
    return TimeoutCleanupEvidence(
        failure_kind=failure_kind,
        cause_type=cause_type,
        cleanup_error_type=_safe_exception_type(failure),
        cleanup_error=_safe_error_summary(failure),
        operation=(
            _safe_operation(attributes.get("operation"))
            if failure_kind == "generic_drain"
            else ""
        ),
        pr_number=(
            _positive_pr_number(attributes.get("pr_number"))
            if failure_kind != "generic_drain"
            else None
        ),
    )


def extract_timeout_cleanup_evidence(
    error: BaseException,
) -> TimeoutCleanupEvidence | None:
    """Find bounded cancellation-drain evidence in an exception graph."""
    candidates: deque[BaseException] = deque([error])
    seen: set[int] = set()
    fallback: TimeoutCleanupEvidence | None = None
    while candidates and len(seen) < _MAX_CAUSE_DEPTH:
        candidate = candidates.popleft()
        identity = id(candidate)
        if identity in seen:
            continue
        seen.add(identity)

        drain = (
            candidate
            if isinstance(candidate, CancellationDrainError)
            else None
        )
        cleanup_error = (
            vars(drain).get("cleanup_error") if drain is not None else None
        )
        children = (
            candidate.__cause__,
            candidate.__context__,
            cleanup_error,
        )
        for index, child in enumerate(children):
            if not isinstance(child, BaseException):
                continue
            child_identity = id(child)
            if child_identity in seen or any(
                child is earlier for earlier in children[:index]
            ):
                continue
            candidates.append(child)

        if drain is None:
            continue
        evidence = _evidence_for_drain(drain)
        if evidence is None:
            continue
        if evidence.failure_kind != "generic_drain":
            return evidence
        if fallback is None:
            fallback = evidence
    return fallback
