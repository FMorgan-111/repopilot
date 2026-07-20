"""Bounded, allowlisted context passed to the escalation model."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
)

from .http_client import LLMResponseError
from .model_provider import redact_secrets
from .state import AgentState, FixAttempt, ModelInvocation

ISSUE_TITLE_LIMIT = 500
ISSUE_BODY_LIMIT = 4_000
EVIDENCE_LIMIT = 12
EVIDENCE_CONTENT_LIMIT = 6_000
EVIDENCE_SUMMARY_LIMIT = 500
EVIDENCE_TOTAL_LIMIT = 24_000
HISTORY_ITEM_LIMIT = 1_000
HISTORY_LIST_LIMIT = 8
REQUIRED_BEHAVIOR_LIMIT = 2_500
REPOSITORY_LIMIT = 300
BASE_COMMIT_LIMIT = 500
EVIDENCE_ID_LIMIT = 80
EVIDENCE_TOOL_LIMIT = 80
EVIDENCE_PATH_LIMIT = 500
EVIDENCE_SYMBOL_LIMIT = 300
EVIDENCE_FINGERPRINT_LIMIT = 128
ESCALATION_PACKET_RENDER_LIMIT = 70_000
REMAINING_TOKEN_BUDGET_LIMIT = 1_000_000_000_000
REMAINING_EXECUTION_ATTEMPTS_LIMIT = 1_000_000

_EVALUATOR_FIELD_RE = re.compile(
    r"(?i)\b(?:gold[_ -]?patch|test[_ -]?patch|FAIL_TO_PASS|PASS_TO_PASS)\b"
)
_RAW_HTTP_MARKER_RE = re.compile(
    r"(?i)(?:HTTP/\d(?:\.\d)?\b|"
    r"(?:GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS)\s+https?://|"
    r"(?:raw[\s_-]+)?HTTP[\s_-]+(?:request|response|payload|body|headers?)\b|"
    r"(?:request|response)[\s_-]+(?:payload|body|headers?)\s*:)"
)


def _safe_text(
    value: object,
    limit: int,
    *,
    denied_literals: Iterable[str] = (),
) -> str:
    """Redact and stop at the first forbidden boundary, then apply a hard cap."""
    text = redact_secrets(str(value or ""))
    boundary_offsets = [
        marker.start()
        for marker in (
            _EVALUATOR_FIELD_RE.search(text),
            _RAW_HTTP_MARKER_RE.search(text),
        )
        if marker is not None
    ]
    denied = tuple(item for item in denied_literals if item)
    boundary_offsets.extend(
        offset
        for literal in denied
        if (offset := text.find(literal)) >= 0
    )
    if boundary_offsets:
        text = text[: min(boundary_offsets)]
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    return normalized[: max(0, limit)]


def _bounded_items(
    values: Iterable[object], *, denied_literals: Iterable[str] = ()
) -> tuple[str, ...]:
    items: list[str] = []
    if isinstance(values, (str, bytes, bytearray)):
        values = (values,)
    for value in values:
        safe = _safe_text(
            value,
            HISTORY_ITEM_LIMIT,
            denied_literals=denied_literals,
        )
        if safe and safe not in items:
            items.append(safe)
        if len(items) >= HISTORY_LIST_LIMIT:
            break
    return tuple(items)


def _bounded_nonnegative_int(value: object, maximum: int) -> int:
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return 0
    return min(max(0, parsed), maximum)


class EscalationEvidence(BaseModel):
    """Deeply immutable evidence representation safe for model escalation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: str = Field(max_length=EVIDENCE_ID_LIMIT)
    tool: str = Field(max_length=EVIDENCE_TOOL_LIMIT)
    file_path: str | None = Field(default=None, max_length=EVIDENCE_PATH_LIMIT)
    symbol: str | None = Field(default=None, max_length=EVIDENCE_SYMBOL_LIMIT)
    summary: str = Field(max_length=EVIDENCE_SUMMARY_LIMIT)
    content: str = Field(max_length=EVIDENCE_CONTENT_LIMIT)
    fingerprint: str = Field(max_length=EVIDENCE_FINGERPRINT_LIMIT)

    @field_validator(
        "evidence_id",
        "tool",
        "file_path",
        "symbol",
        "summary",
        "content",
        "fingerprint",
        mode="before",
    )
    @classmethod
    def _bound_text_fields(cls, value: object, info: Any) -> str | None:
        if value is None and info.field_name in {"file_path", "symbol"}:
            return None
        limits = {
            "evidence_id": EVIDENCE_ID_LIMIT,
            "tool": EVIDENCE_TOOL_LIMIT,
            "file_path": EVIDENCE_PATH_LIMIT,
            "symbol": EVIDENCE_SYMBOL_LIMIT,
            "summary": EVIDENCE_SUMMARY_LIMIT,
            "content": EVIDENCE_CONTENT_LIMIT,
            "fingerprint": EVIDENCE_FINGERPRINT_LIMIT,
        }
        return _safe_text(value, limits[info.field_name])


def _as_escalation_evidence(
    value: object,
    *,
    denied_literals: Iterable[str] = (),
) -> EscalationEvidence | None:
    if isinstance(value, BaseModel):
        value = value.model_dump()
    if not isinstance(value, dict):
        return None
    denied = tuple(denied_literals)
    try:
        return EscalationEvidence(
            evidence_id=_safe_text(
                value.get("evidence_id"), EVIDENCE_ID_LIMIT, denied_literals=denied
            ),
            tool=_safe_text(
                value.get("tool"), EVIDENCE_TOOL_LIMIT, denied_literals=denied
            ),
            file_path=(
                _safe_text(
                    value.get("file_path"),
                    EVIDENCE_PATH_LIMIT,
                    denied_literals=denied,
                )
                if value.get("file_path") is not None
                else None
            ),
            symbol=(
                _safe_text(
                    value.get("symbol"),
                    EVIDENCE_SYMBOL_LIMIT,
                    denied_literals=denied,
                )
                if value.get("symbol") is not None
                else None
            ),
            summary=_safe_text(
                value.get("summary"),
                EVIDENCE_SUMMARY_LIMIT,
                denied_literals=denied,
            ),
            content=_safe_text(
                value.get("content"),
                EVIDENCE_CONTENT_LIMIT,
                denied_literals=denied,
            ),
            fingerprint=_safe_text(
                value.get("fingerprint"),
                EVIDENCE_FINGERPRINT_LIMIT,
                denied_literals=denied,
            ),
        )
    except ValidationError:
        return None


def _bounded_evidence_values(
    values: Iterable[object],
    *,
    denied_literals: Iterable[str] = (),
) -> tuple[EscalationEvidence, ...]:
    selected: list[EscalationEvidence] = []
    rendered_size = 0
    if isinstance(values, (str, bytes, bytearray)):
        return ()
    for item in values:
        if len(selected) >= EVIDENCE_LIMIT:
            break
        safe_item = _as_escalation_evidence(
            item,
            denied_literals=denied_literals,
        )
        if safe_item is None:
            continue
        encoded = json.dumps(
            safe_item.model_dump(),
            ensure_ascii=False,
            sort_keys=True,
        )
        if rendered_size + len(encoded) > EVIDENCE_TOTAL_LIMIT:
            continue
        rendered_size += len(encoded)
        selected.append(safe_item)
    return tuple(selected)


class EscalationPacket(BaseModel):
    """The complete and only state payload exposed after model escalation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    issue_title: str = Field(max_length=ISSUE_TITLE_LIMIT)
    issue_body: str = Field(max_length=ISSUE_BODY_LIMIT)
    repository: str = Field(max_length=REPOSITORY_LIMIT)
    base_commit: str = Field(max_length=BASE_COMMIT_LIMIT)
    evidence: tuple[EscalationEvidence, ...] = Field(
        default=(), max_length=EVIDENCE_LIMIT
    )
    failed_edit_signatures: tuple[str, ...] = Field(
        default=(), max_length=HISTORY_LIST_LIMIT
    )
    patch_errors: tuple[str, ...] = Field(default=(), max_length=HISTORY_LIST_LIMIT)
    test_error_summaries: tuple[str, ...] = Field(
        default=(), max_length=HISTORY_LIST_LIMIT
    )
    rejected_approaches: tuple[str, ...] = Field(
        default=(), max_length=HISTORY_LIST_LIMIT
    )
    required_behavior: str = Field(max_length=REQUIRED_BEHAVIOR_LIMIT)
    remaining_token_budget: int = Field(
        ge=0, le=REMAINING_TOKEN_BUDGET_LIMIT
    )
    remaining_execution_attempts: int = Field(
        ge=0, le=REMAINING_EXECUTION_ATTEMPTS_LIMIT
    )

    @field_validator(
        "issue_title",
        "issue_body",
        "repository",
        "base_commit",
        "required_behavior",
        mode="before",
    )
    @classmethod
    def _bound_scalar_text(cls, value: object, info: Any) -> str:
        limits = {
            "issue_title": ISSUE_TITLE_LIMIT,
            "issue_body": ISSUE_BODY_LIMIT,
            "repository": REPOSITORY_LIMIT,
            "base_commit": BASE_COMMIT_LIMIT,
            "required_behavior": REQUIRED_BEHAVIOR_LIMIT,
        }
        return _safe_text(value, limits[info.field_name])

    @field_validator("evidence", mode="before")
    @classmethod
    def _bound_nested_evidence(cls, value: object) -> tuple[EscalationEvidence, ...]:
        if value is None:
            return ()
        try:
            return _bounded_evidence_values(value)  # type: ignore[arg-type]
        except TypeError:
            return ()

    @field_validator(
        "failed_edit_signatures",
        "patch_errors",
        "test_error_summaries",
        "rejected_approaches",
        mode="before",
    )
    @classmethod
    def _bound_history_items(cls, value: object) -> tuple[str, ...]:
        if value is None:
            return ()
        try:
            return _bounded_items(value)  # type: ignore[arg-type]
        except TypeError:
            return ()

    @field_validator("remaining_token_budget", mode="before")
    @classmethod
    def _bound_remaining_token_budget(cls, value: object) -> int:
        return _bounded_nonnegative_int(value, REMAINING_TOKEN_BUDGET_LIMIT)

    @field_validator("remaining_execution_attempts", mode="before")
    @classmethod
    def _bound_remaining_execution_attempts(cls, value: object) -> int:
        return _bounded_nonnegative_int(
            value,
            REMAINING_EXECUTION_ATTEMPTS_LIMIT,
        )


def _generated_test_paths(state: AgentState) -> set[str]:
    return {approval.path for approval in state.generated_test_approvals}


def _bounded_evidence(state: AgentState) -> tuple[EscalationEvidence, ...]:
    generated_paths = _generated_test_paths(state)
    eligible = (
        item for item in state.evidence if item.file_path not in generated_paths
    )
    return _bounded_evidence_values(eligible, denied_literals=generated_paths)


def _attempt_signature(attempt: FixAttempt) -> str:
    payload = {
        "patch": hashlib.sha256(attempt.patch_content.encode("utf-8")).hexdigest(),
        "edits": [
            {
                "file": edit.file_path,
                "node": edit.node_target,
                "search": hashlib.sha256(edit.search.encode("utf-8")).hexdigest(),
                "replace": hashlib.sha256(edit.replace.encode("utf-8")).hexdigest(),
            }
            for edit in attempt.patch_edits
        ],
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _rejected_approaches(state: AgentState) -> tuple[str, ...]:
    values: list[str] = []
    for frame in state.frame_history:
        # At packet construction time every prior plan frame belongs to an
        # already-failed loop; its compact summary is the safest useful record
        # of the approach that Opus should not repeat.
        if frame.stage == "plan" and frame.summary:
            values.append(frame.summary)
        for hypothesis in frame.hypotheses:
            if hypothesis.id != frame.selected_hypothesis_id or hypothesis.why_not_selected:
                explanation = hypothesis.claim
                if hypothesis.why_not_selected:
                    explanation = f"{explanation}: {hypothesis.why_not_selected}"
                values.append(explanation)
    return _bounded_items(values, denied_literals=_generated_test_paths(state))


def build_escalation_packet(state: AgentState) -> EscalationPacket:
    """Copy only approved agent state fields into an independently bounded packet."""
    failed_attempts = [attempt for attempt in state.fix_attempts if not attempt.success]
    patch_failures = [
        attempt.error_log
        for attempt in failed_attempts
        if attempt.failure_kind == "patch_apply_failed"
        or attempt.test_result == "patch_apply_failed"
    ]
    test_failures = [
        attempt.error_log
        for attempt in failed_attempts
        if attempt.error_log
        and attempt.failure_kind != "patch_apply_failed"
        and attempt.test_result != "patch_apply_failed"
    ]
    generated_paths = _generated_test_paths(state)
    safe_body = _safe_text(
        state.issue_body,
        ISSUE_BODY_LIMIT,
        denied_literals=generated_paths,
    )
    safe_title = _safe_text(
        state.issue_title,
        ISSUE_TITLE_LIMIT,
        denied_literals=generated_paths,
    )
    return EscalationPacket(
        issue_title=safe_title,
        issue_body=safe_body,
        repository=_safe_text(
            f"{state.owner}/{state.repo}".strip("/"), REPOSITORY_LIMIT
        ),
        # Exact equality is intentional: the packet must identify the prepared base.
        base_commit=state.repo_ref,
        evidence=_bounded_evidence(state),
        failed_edit_signatures=[
            _attempt_signature(attempt)
            for attempt in failed_attempts[-HISTORY_LIST_LIMIT:]
        ],
        patch_errors=_bounded_items(
            patch_failures,
            denied_literals=generated_paths,
        ),
        test_error_summaries=_bounded_items(
            test_failures,
            denied_literals=generated_paths,
        ),
        rejected_approaches=_rejected_approaches(state),
        required_behavior=_safe_text(
            f"{safe_title}: {safe_body}" if safe_body else safe_title,
            REQUIRED_BEHAVIOR_LIMIT,
        ),
        remaining_token_budget=max(0, state.token_budget - state.token_usage),
        remaining_execution_attempts=max(0, state.max_retries - state.retry_count),
    )


def render_escalation_packet(packet: EscalationPacket) -> str:
    """Rebuild from untrusted attributes and enforce the final total bound."""
    safe_packet = EscalationPacket(
        issue_title=getattr(packet, "issue_title", ""),
        issue_body=getattr(packet, "issue_body", ""),
        repository=getattr(packet, "repository", ""),
        base_commit=getattr(packet, "base_commit", ""),
        evidence=getattr(packet, "evidence", ()),
        failed_edit_signatures=getattr(packet, "failed_edit_signatures", ()),
        patch_errors=getattr(packet, "patch_errors", ()),
        test_error_summaries=getattr(packet, "test_error_summaries", ()),
        rejected_approaches=getattr(packet, "rejected_approaches", ()),
        required_behavior=getattr(packet, "required_behavior", ""),
        remaining_token_budget=getattr(packet, "remaining_token_budget", 0),
        remaining_execution_attempts=getattr(
            packet,
            "remaining_execution_attempts",
            0,
        ),
    )
    payload = safe_packet.model_dump(mode="json")

    def redact_value(value: object) -> object:
        if isinstance(value, str):
            return redact_secrets(value)
        if isinstance(value, list):
            return [redact_value(item) for item in value]
        if isinstance(value, dict):
            return {key: redact_value(item) for key, item in value.items()}
        return value

    payload = redact_value(payload)

    def render(value: object) -> str:
        return json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    rendered = render(payload)
    sequence_trim_order = (
        "rejected_approaches",
        "test_error_summaries",
        "patch_errors",
        "failed_edit_signatures",
        "evidence",
    )
    while len(rendered) > ESCALATION_PACKET_RENDER_LIMIT:
        trimmed = False
        for field_name in sequence_trim_order:
            values = payload[field_name]  # type: ignore[index]
            if values:
                values.pop()
                trimmed = True
                break
        if not trimmed:
            text_fields = (
                "issue_body",
                "required_behavior",
                "issue_title",
                "repository",
                "base_commit",
            )
            field_name = max(
                text_fields,
                key=lambda name: len(payload[name]),  # type: ignore[index, arg-type]
            )
            value = payload[field_name]  # type: ignore[index]
            if not value:
                raise ValueError("EscalationPacket cannot fit its rendered bound")
            payload[field_name] = value[: len(value) // 2]  # type: ignore[index]
        rendered = render(payload)
    return rendered


def immediate_model_policy_reason(exc: BaseException) -> str:
    """Classify only retry-exhausted response failures approved by ModelPolicy."""
    if isinstance(exc, LLMResponseError) and "empty chat completion" in str(exc).lower():
        return "empty_completion_after_retries"
    if isinstance(exc, (ValidationError, ValueError)):
        return "invalid_structured_response_after_retries"
    return ""


def record_model_invocation(
    state: AgentState,
    *,
    model: str,
    provider: Literal["primary", "escalation"],
    node: str,
    elapsed_seconds: float,
    input_tokens: int,
    output_tokens: int,
    status: Literal["ok", "invalid_response", "error"],
    error: BaseException | None = None,
) -> None:
    """Persist a bounded invocation record without exception messages."""
    state.model_history.append(
        ModelInvocation(
            model=model,
            provider=provider,
            node=node,
            elapsed_seconds=max(0.0, elapsed_seconds),
            input_tokens=max(0, input_tokens),
            output_tokens=max(0, output_tokens),
            status=status,
            error_class=error or "",
        )
    )
