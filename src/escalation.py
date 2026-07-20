"""Bounded, allowlisted context passed to the escalation model."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .http_client import LLMResponseError
from .model_provider import redact_secrets
from .state import AgentState, Evidence, FixAttempt, ModelInvocation

ISSUE_TITLE_LIMIT = 500
ISSUE_BODY_LIMIT = 4_000
EVIDENCE_LIMIT = 12
EVIDENCE_CONTENT_LIMIT = 6_000
EVIDENCE_SUMMARY_LIMIT = 500
EVIDENCE_TOTAL_LIMIT = 24_000
HISTORY_ITEM_LIMIT = 1_000
HISTORY_LIST_LIMIT = 8
REQUIRED_BEHAVIOR_LIMIT = 2_500

_EVALUATOR_FIELD_RE = re.compile(
    r"(?i)\b(?:gold[_ -]?patch|test[_ -]?patch|FAIL_TO_PASS|PASS_TO_PASS)\b"
)
_RAW_HTTP_LINE_RE = re.compile(
    r"(?i)^\s*(?:HTTP/\d|(?:GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS)\s+https?://|"
    r"(?:raw\s+)?HTTP\s+(?:request|response|payload|body|headers?)\b|"
    r"(?:request|response)\s+(?:payload|body|headers?)\s*:)"
)


class EscalationPacket(BaseModel):
    """The complete and only state payload exposed after model escalation."""

    model_config = ConfigDict(extra="forbid")

    issue_title: str
    issue_body: str
    repository: str
    base_commit: str
    evidence: list[Evidence] = Field(default_factory=list)
    failed_edit_signatures: list[str] = Field(default_factory=list)
    patch_errors: list[str] = Field(default_factory=list)
    test_error_summaries: list[str] = Field(default_factory=list)
    rejected_approaches: list[str] = Field(default_factory=list)
    required_behavior: str
    remaining_token_budget: int
    remaining_execution_attempts: int


def _safe_text(
    value: object,
    limit: int,
    *,
    denied_literals: Iterable[str] = (),
) -> str:
    """Redact credentials/evaluator payloads and apply one independent bound."""
    text = redact_secrets(str(value or ""))
    marker = _EVALUATOR_FIELD_RE.search(text)
    if marker is not None:
        text = text[: marker.start()]
    denied = tuple(item for item in denied_literals if item)
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    normalized = "\n".join(
        line
        for line in lines
        if not _RAW_HTTP_LINE_RE.search(line)
        and not any(item in line for item in denied)
    ).strip()
    return normalized[: max(0, limit)]


def _generated_test_paths(state: AgentState) -> set[str]:
    return {approval.path for approval in state.generated_test_approvals}


def _bounded_evidence(state: AgentState) -> list[Evidence]:
    generated_paths = _generated_test_paths(state)
    selected: list[Evidence] = []
    rendered_size = 0
    for item in state.evidence:
        if len(selected) >= EVIDENCE_LIMIT or item.file_path in generated_paths:
            continue
        safe_item = Evidence(
            evidence_id=_safe_text(item.evidence_id, 80),
            tool=_safe_text(item.tool, 80),
            file_path=(
                _safe_text(item.file_path, 500) if item.file_path is not None else None
            ),
            symbol=(
                _safe_text(item.symbol, 300) if item.symbol is not None else None
            ),
            summary=_safe_text(
                item.summary,
                EVIDENCE_SUMMARY_LIMIT,
                denied_literals=generated_paths,
            ),
            content=_safe_text(
                item.content,
                EVIDENCE_CONTENT_LIMIT,
                denied_literals=generated_paths,
            ),
            fingerprint=_safe_text(item.fingerprint, 128),
        )
        encoded = json.dumps(
            safe_item.model_dump(),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        if rendered_size + len(encoded) > EVIDENCE_TOTAL_LIMIT:
            continue
        rendered_size += len(encoded)
        selected.append(safe_item)
    return selected


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


def _bounded_items(
    values: Iterable[object], *, denied_literals: Iterable[str] = ()
) -> list[str]:
    items: list[str] = []
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
    return items


def _rejected_approaches(state: AgentState) -> list[str]:
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
        repository=_safe_text(f"{state.owner}/{state.repo}".strip("/"), 300),
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
    """Render a deterministic packet and redact once more at the final boundary."""
    rendered = json.dumps(
        packet.model_dump(),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    # Redact decoded string values rather than the JSON source: the generic
    # credential regex intentionally consumes punctuation and must never be
    # allowed to consume JSON's structural quote characters.
    payload = json.loads(rendered)

    def redact_value(value: object) -> object:
        if isinstance(value, str):
            return redact_secrets(value)
        if isinstance(value, list):
            return [redact_value(item) for item in value]
        if isinstance(value, dict):
            return {key: redact_value(item) for key, item in value.items()}
        return value

    return json.dumps(
        redact_value(payload),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


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
