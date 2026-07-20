"""Deterministic, one-way model escalation policy."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator

from .model_provider import escalation_is_configured, get_model_config
from .state import (
    APPROVED_ESCALATION_REASONS,
    APPROVED_NO_PROGRESS_KINDS,
    APPROVED_ROUTING_NODES,
    AgentState,
    NoProgressEvent,
    sanitize_escalation_reason,
)

_IMMEDIATE_REASONS = frozenset(
    {
        "empty_completion_after_retries",
        "invalid_structured_response_after_retries",
    }
)

# These names cover the six policy signals in the design while retaining short
# names for callers that already classified a plan, context, edit, or test event.
_PLAN_SIGNATURE_KINDS = frozenset(
    {
        "nonexistent_search_block",
        "nonexistent_search_blocks",
        "repeated_patch_signature",
        "repeated_edit",
        "repeated_unlocatable_edit",
        "unchanged_hypothesis",
        "unchanged_plan",
        "no_evidence_or_applicable_patch",
        "plan",
        "edit",
        "repeated_plan",
    }
)
_CONTEXT_SIGNATURE_KINDS = frozenset(
    {"unchanged_context", "context", "repeated_context"}
)
_TEST_SIGNATURE_KINDS = frozenset(
    {
        "unchanged_test_failure",
        "test_failure",
        "repeated_test",
        "repeated_test_failure",
    }
)


class EscalationDecision(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    escalate: bool
    reason: str = ""

    @field_validator("reason", mode="before")
    @classmethod
    def _keep_approved_reason(cls, value: Any) -> str:
        return sanitize_escalation_reason(value)


def _canonical_fingerprint(value: Any) -> str:
    canonical = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def record_progress(state: AgentState) -> None:
    """Reset the consecutive no-progress counter after material progress."""
    state.no_progress_rounds = 0
    state.last_plan_signature = ""
    state.last_context_fingerprint = ""
    state.last_test_failure_signature = ""


def _signature_field(kind: str) -> str:
    if kind in _PLAN_SIGNATURE_KINDS:
        return "last_plan_signature"
    if kind in _CONTEXT_SIGNATURE_KINDS:
        return "last_context_fingerprint"
    if kind in _TEST_SIGNATURE_KINDS:
        return "last_test_failure_signature"
    return ""


def record_no_progress(
    state: AgentState,
    *,
    kind: str,
    fingerprint: str,
    node: str,
) -> None:
    """Record a safe, canonical no-progress event and update its signature."""
    if kind not in APPROVED_NO_PROGRESS_KINDS or node not in APPROVED_ROUTING_NODES:
        record_progress(state)
        return

    signature_field = _signature_field(kind)
    if not signature_field:
        record_progress(state)
        return

    canonical_fingerprint = _canonical_fingerprint(fingerprint)
    previous_signature = getattr(state, signature_field)
    previous_event = state.no_progress_history[-1] if state.no_progress_history else None
    is_consecutive_match = (
        bool(previous_signature)
        and previous_signature == canonical_fingerprint
        and previous_event is not None
        and previous_event.fingerprint == canonical_fingerprint
        and _signature_field(previous_event.kind) == signature_field
    )
    state.no_progress_rounds = (
        state.no_progress_rounds + 1 if is_consecutive_match else 1
    )
    setattr(state, signature_field, canonical_fingerprint)
    state.no_progress_history.append(
        NoProgressEvent(
            kind=kind,
            fingerprint=canonical_fingerprint,
            node=node,
        )
    )



def should_escalate(
    state: AgentState, *, immediate_reason: str = ""
) -> EscalationDecision:
    """Return the deterministic escalation decision for the current state."""
    if state.escalated or state.active_provider == "escalation":
        return EscalationDecision(escalate=False)
    if not escalation_is_configured():
        return EscalationDecision(escalate=False)

    if immediate_reason in _IMMEDIATE_REASONS:
        return EscalationDecision(escalate=True, reason=immediate_reason)
    if state.token_usage >= primary_budget_limit(state):
        return EscalationDecision(
            escalate=True,
            reason="primary_budget_reserve",
        )
    if state.no_progress_rounds >= 2:
        reason = "repeated_no_progress"
        if state.no_progress_history:
            latest_kind = state.no_progress_history[-1].kind
            if latest_kind in APPROVED_NO_PROGRESS_KINDS:
                reason = latest_kind
        return EscalationDecision(escalate=True, reason=reason)
    return EscalationDecision(escalate=False)


def apply_escalation(
    state: AgentState,
    decision: EscalationDecision,
) -> None:
    """Apply an escalation once without ever switching back to primary."""
    if (
        not decision.escalate
        or not decision.reason
        or decision.reason not in APPROVED_ESCALATION_REASONS
        or state.escalated
    ):
        return
    if state.active_provider == "escalation" or not escalation_is_configured():
        return

    previous_model = state.active_model
    escalation_model = get_model_config("escalation").model
    state.active_provider = "escalation"
    state.active_model = escalation_model
    state.escalated = True
    state.escalation_reason = decision.reason
    state.node_diagnostics.append(
        {
            "event": "model_escalated",
            "from": previous_model,
            "to": escalation_model,
            "reason": decision.reason,
            "round": state.no_progress_rounds,
        }
    )


def primary_budget_limit(state: AgentState) -> int:
    """Return primary capacity after reserving 40k tokens for escalation."""
    return min(55_000, max(0, state.token_budget - 40_000))
