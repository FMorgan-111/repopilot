"""Bounded, redacted evidence from exceptions hidden by phase timeouts."""

from __future__ import annotations

from dataclasses import dataclass

from .model_provider import redact_secrets

_MAX_CAUSE_DEPTH = 8
_MAX_ERROR_SUMMARY = 300


@dataclass(frozen=True)
class TimeoutCleanupEvidence:
    """Safe details proving that timeout cancellation cleanup also failed."""

    cause_type: str
    cleanup_error_type: str
    cleanup_error: str
    pr_number: int | None

    def summary(self) -> str:
        target = (
            f"pull request {self.pr_number}"
            if self.pr_number is not None
            else "an unknown pull request"
        )
        return (
            f"PR cancellation cleanup failed for {target} "
            f"({self.cleanup_error_type}: {self.cleanup_error})"
        )

    def diagnostic_details(self) -> dict[str, object]:
        details: dict[str, object] = {
            "timeout_cause_type": self.cause_type,
            "cleanup_error_type": self.cleanup_error_type,
            "cleanup_error": self.cleanup_error,
        }
        if self.pr_number is not None:
            details["cleanup_pr_number"] = self.pr_number
        return details


def _safe_error_summary(error: BaseException) -> str:
    redacted = redact_secrets(str(error))
    normalized = " ".join(redacted.split())
    return (normalized or type(error).__name__)[:_MAX_ERROR_SUMMARY]


def extract_timeout_cleanup_evidence(
    error: BaseException,
) -> TimeoutCleanupEvidence | None:
    """Find RepoPilot PR-cleanup evidence in a bounded exception cause chain."""
    current: BaseException | None = error
    seen: set[int] = set()
    for _depth in range(_MAX_CAUSE_DEPTH):
        if current is None or id(current) in seen:
            return None
        seen.add(id(current))
        attributes = vars(current)
        cleanup_error = attributes.get("cleanup_error")
        if (
            type(current).__name__ == "PRCancellationCleanupError"
            and isinstance(cleanup_error, BaseException)
        ):
            raw_number = attributes.get("pr_number")
            number = (
                raw_number
                if type(raw_number) is int and raw_number > 0
                else None
            )
            return TimeoutCleanupEvidence(
                cause_type=type(current).__name__,
                cleanup_error_type=type(cleanup_error).__name__,
                cleanup_error=_safe_error_summary(cleanup_error),
                pr_number=number,
            )
        current = current.__cause__ or current.__context__
    return None
