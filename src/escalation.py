"""Bounded, allowlisted context passed to the escalation model."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Sequence
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
)

from .evidence import EvidenceStore
from .http_client import LLMResponseError, is_retryable_llm_error
from .model_provider import redact_secrets
from .state import (
    AgentState,
    Evidence,
    FixAttempt,
    ModelInvocation,
    _issue_search_terms,
)
from .summary_safety import sanitize_model_context

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
REPAIR_SOURCE_TOOL = "planner_relevant_file"
REPAIR_SOURCE_FILE_LIMIT = 3
REPAIR_STATE_EVIDENCE_LIMIT = 30
REPAIR_TOOL_EVIDENCE_RESERVE = 1

_EVALUATOR_FIELD_RE = re.compile(
    r"(?i)\b(?:gold[_ -]?patch|test[_ -]?patch|FAIL_TO_PASS|PASS_TO_PASS)\b"
)
_RAW_HTTP_MARKER_RE = re.compile(
    r"(?i)(?:\braw[\s_-]+HTTP\b|HTTP/\d(?:\.\d)?\b|"
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
    """Compatibility wrapper around the public model-context sanitizer."""
    return sanitize_model_context(
        value,
        limit,
        denied_literals=denied_literals,
    )


def relevance_window(content: str, terms: Sequence[str], limit: int) -> str:
    """Center a bounded source window on the densest issue-term match."""
    if len(content) <= limit:
        return content
    lines = content.split("\n")
    lowered = [line.lower() for line in lines]
    lowered_terms = [term.lower() for term in terms if term.strip()]

    best_idx, best_score = -1, 0
    for index, line in enumerate(lowered):
        score = sum(1 for term in lowered_terms if term in line)
        if score > best_score:
            best_score, best_idx = score, index
    if best_idx < 0:
        return f"{content[:limit].rstrip()}..."

    lo = hi = best_idx
    size = len(lines[best_idx])
    while True:
        moved = False
        if lo > 0 and size + len(lines[lo - 1]) + 1 < limit:
            lo -= 1
            size += len(lines[lo]) + 1
            moved = True
        if hi < len(lines) - 1 and size + len(lines[hi + 1]) + 1 < limit:
            hi += 1
            size += len(lines[hi]) + 1
            moved = True
        if not moved:
            break

    window = "\n".join(lines[lo : hi + 1])
    if lo > 0:
        window = f"... [{lo} lines above truncated] ...\n{window}"
    if hi < len(lines) - 1:
        window = f"{window}\n... [{len(lines) - 1 - hi} lines below truncated] ..."
    return window


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


def _bounded_evidence(
    state: AgentState,
    evidence_ids: tuple[str, ...] | None = None,
) -> tuple[EscalationEvidence, ...]:
    generated_paths = _generated_test_paths(state)
    source = (
        state.evidence
        if evidence_ids is None
        else EvidenceStore(state).select(list(evidence_ids))
    )
    eligible = (
        item for item in source if item.file_path not in generated_paths
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


def build_escalation_packet(
    state: AgentState,
    *,
    evidence_ids: tuple[str, ...] | None = None,
) -> EscalationPacket:
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
        evidence=_bounded_evidence(state, evidence_ids),
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


def _hydrated_repair_source_evidence(state: AgentState) -> tuple[Evidence, ...]:
    """Build stable safe source items from only the hydrated top three files."""
    generated_paths = _generated_test_paths(state)
    terms = _issue_search_terms(state.issue_title, state.issue_body)
    temporary = state.model_copy(deep=True)
    temporary.evidence = []
    store = EvidenceStore(
        temporary,
        max_items=REPAIR_SOURCE_FILE_LIMIT,
        max_content_chars=EVIDENCE_CONTENT_LIMIT,
        max_summary_chars=EVIDENCE_SUMMARY_LIMIT,
    )
    source_evidence: list[Evidence] = []
    for file in state.relevant_files[:REPAIR_SOURCE_FILE_LIMIT]:
        if file.path in generated_paths:
            continue
        added = store.add(
            tool=REPAIR_SOURCE_TOOL,
            summary=f"Hydrated exact-checkout source window for {file.path}",
            content=_safe_text(
                relevance_window(file.content, terms, EVIDENCE_CONTENT_LIMIT),
                EVIDENCE_CONTENT_LIMIT,
                denied_literals=generated_paths,
            ),
            file_path=file.path,
        )
        if added.evidence is not None:
            source_evidence.append(added.evidence)
    return tuple(source_evidence)


def _initial_packet_repair_source(
    state: AgentState,
    packet: EscalationPacket,
) -> tuple[EscalationEvidence, ...]:
    """Retain bounded supplied source only when no hydrated files exist."""
    if state.relevant_files:
        return ()
    generated_paths = _generated_test_paths(state)
    source: list[EscalationEvidence] = []
    for item in packet.evidence:
        if item.tool != REPAIR_SOURCE_TOOL or item.file_path in generated_paths:
            continue
        safe_item = _as_escalation_evidence(
            item,
            denied_literals=generated_paths,
        )
        if safe_item is not None:
            source.append(safe_item)
        if len(source) >= REPAIR_SOURCE_FILE_LIMIT:
            break
    return tuple(source)


def _reserve_repair_tool_evidence_capacity(
    state: AgentState,
    source_evidence: tuple[Evidence, ...],
    *,
    protected_evidence_ids: tuple[str, ...],
) -> None:
    """Compact persisted evidence while reserving one real router slot."""
    source_ids = {item.evidence_id for item in source_evidence}
    source_fingerprints = {item.fingerprint for item in source_evidence}
    preserved = [
        item
        for item in state.evidence
        if item.evidence_id not in source_ids
        and item.fingerprint not in source_fingerprints
        and (not state.relevant_files or item.tool != REPAIR_SOURCE_TOOL)
    ]
    protected_ids = set(protected_evidence_ids)
    protected = [
        item for item in preserved if item.evidence_id in protected_ids
    ]
    unprotected = [
        item for item in preserved if item.evidence_id not in protected_ids
    ]
    available = max(
        0,
        REPAIR_STATE_EVIDENCE_LIMIT
        - REPAIR_TOOL_EVIDENCE_RESERVE
        - len(source_evidence),
    )
    state.evidence = [*protected, *unprotected][:available] + list(source_evidence)


def prepare_repair_plan_packet(
    state: AgentState,
    packet: EscalationPacket | None = None,
    *,
    evidence_ids: tuple[str, ...] | None = None,
) -> EscalationPacket:
    """Hydrate and order bounded evidence for every RepairPlan prompt."""
    initial_packet = packet or build_escalation_packet(state)
    hydrated_source = _hydrated_repair_source_evidence(state)
    supplied_source = _initial_packet_repair_source(state, initial_packet)
    source_evidence: tuple[Evidence | EscalationEvidence, ...] = (
        hydrated_source or supplied_source
    )
    _reserve_repair_tool_evidence_capacity(
        state,
        hydrated_source,
        protected_evidence_ids=evidence_ids or (),
    )

    if evidence_ids is None:
        leading = source_evidence
        remaining = (
            item
            for item in initial_packet.evidence
            if item.tool != REPAIR_SOURCE_TOOL
        )
    else:
        explicit = build_escalation_packet(
            state,
            evidence_ids=evidence_ids,
        ).evidence
        leading = tuple(
            item for item in explicit if item.tool != REPAIR_SOURCE_TOOL
        )
        remaining = iter(source_evidence)

    merged: list[Evidence | EscalationEvidence] = []
    seen_ids: set[str] = set()
    seen_fingerprints: set[str] = set()
    for item in (*leading, *remaining):
        if (
            item.evidence_id in seen_ids
            or item.fingerprint in seen_fingerprints
        ):
            continue
        seen_ids.add(item.evidence_id)
        seen_fingerprints.add(item.fingerprint)
        merged.append(item)

    payload = initial_packet.model_dump(mode="python")
    payload["evidence"] = _bounded_evidence_values(
        merged,
        denied_literals=_generated_test_paths(state),
    )
    return EscalationPacket.model_validate(payload)


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
    if is_retryable_llm_error(exc):
        return "primary_gateway_unavailable_after_retries"
    if (
        isinstance(exc, LLMResponseError)
        and "empty chat completion" in str(exc).lower()
    ):
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
    status: Literal["ok", "invalid_response", "error", "cancelled"],
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
