"""Deterministic, one-way model escalation policy."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel

from .model_provider import escalation_is_configured, get_model_config
from .state import AgentState, NoProgressEvent

_IMMEDIATE_REASONS = frozenset(
    {
        "empty_completion_after_retries",
        "invalid_structured_response_after_retries",
    }
)

# These names cover the six policy signals in the design while retaining short
# names for callers that already classified a plan, context, edit, or test event.
_APPROVED_NO_PROGRESS_KINDS = frozenset(
    {
        "nonexistent_search_block",
        "nonexistent_search_blocks",
        "repeated_patch_signature",
        "repeated_edit",
        "repeated_unlocatable_edit",
        "unchanged_context",
        "unchanged_hypothesis",
        "unchanged_plan",
        "unchanged_test_failure",
        "no_evidence_or_applicable_patch",
        "plan",
        "context",
        "edit",
        "test_failure",
    }
)

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
    }
)
_CONTEXT_SIGNATURE_KINDS = frozenset({"unchanged_context", "context"})
_TEST_SIGNATURE_KINDS = frozenset({"unchanged_test_failure", "test_failure"})


class EscalationDecision(BaseModel):
    escalate: bool
    reason: str = ""


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


def record_no_progress(
    state: AgentState,
    *,
    kind: str,
    fingerprint: str,
    node: str,
) -> None:
    """Record a safe, canonical no-progress event and update its signature."""
    canonical_fingerprint = _canonical_fingerprint(fingerprint)
    state.no_progress_history.append(
        NoProgressEvent(
            kind=kind,
            fingerprint=canonical_fingerprint,
            node=node,
        )
    )

    if kind not in _APPROVED_NO_PROGRESS_KINDS:
        state.no_progress_rounds = 0
        return

    state.no_progress_rounds += 1
    if kind in _PLAN_SIGNATURE_KINDS:
        state.last_plan_signature = canonical_fingerprint
    elif kind in _CONTEXT_SIGNATURE_KINDS:
        state.last_context_fingerprint = canonical_fingerprint
    elif kind in _TEST_SIGNATURE_KINDS:
        state.last_test_failure_signature = canonical_fingerprint


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
            if latest_kind in _APPROVED_NO_PROGRESS_KINDS:
                reason = latest_kind
        return EscalationDecision(escalate=True, reason=reason)
    return EscalationDecision(escalate=False)


def apply_escalation(
    state: AgentState,
    decision: EscalationDecision,
) -> None:
    """Apply an escalation once without ever switching back to primary."""
    if not decision.escalate or state.escalated:
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
