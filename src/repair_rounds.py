"""Monotonic, exactly-once accounting for semantic repair transactions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .model_policy import apply_escalation, should_escalate
from .state import AgentState, Phase, _record_node_diagnostic

RepairProvider = Literal["primary", "escalation"]

_REPAIR_LEDGER_FIELDS = frozenset(
    {
        "primary_failed_repair_rounds",
        "repair_round_sequence",
        "current_repair_round_id",
        "current_repair_provider",
        "current_repair_model",
        "authorized_repair_round_id",
        "authorized_repair_provider",
        "authorized_repair_model",
        "last_counted_repair_round_id",
        "repair_correction_context",
    }
)


@dataclass(frozen=True)
class RepairRetryDecision:
    counted: bool
    retry_allowed: bool
    round_id: int


def validate_repair_round_state(state: AgentState) -> None:
    """Reject impossible/tampered ledger relationships before routing."""
    if not 0 <= state.retry_count <= state.max_retries:
        raise ValueError("repair retry count is outside its configured budget")
    legacy_migration_candidate = (
        state.repair_round_sequence == 0
        and state.current_repair_round_id == 0
        and state.last_counted_repair_round_id == 0
        and state.retry_count > 0
    )
    if legacy_migration_candidate:
        has_explicit_ledger_field = bool(
            state.model_fields_set & _REPAIR_LEDGER_FIELDS
        )
        has_nondefault_ledger_value = any(
            (
                state.primary_failed_repair_rounds != 0,
                state.current_repair_provider is not None,
                bool(state.current_repair_model),
                state.authorized_repair_round_id != 0,
                state.authorized_repair_provider is not None,
                bool(state.authorized_repair_model),
                bool(state.repair_correction_context),
            )
        )
        if has_explicit_ledger_field or has_nondefault_ledger_value:
            raise ValueError("partial repair ledger cannot be migrated")
        state.repair_round_sequence = state.retry_count
        state.last_counted_repair_round_id = state.retry_count

    transaction_cap = state.max_retries + 1
    if not (
        0
        <= state.last_counted_repair_round_id
        <= state.repair_round_sequence
        <= transaction_cap
    ):
        raise ValueError("repair round sequence exceeds the transaction cap")
    if state.last_counted_repair_round_id - state.retry_count not in {0, 1}:
        raise ValueError("repair retry count does not match the counted ledger")
    if state.repair_round_sequence - state.last_counted_repair_round_id not in {
        0,
        1,
    }:
        raise ValueError("repair round sequence is not monotonic")
    if state.current_repair_round_id not in {0, state.repair_round_sequence}:
        raise ValueError("current repair round does not match the sequence")
    if state.current_repair_round_id > transaction_cap:
        raise ValueError("current repair round exceeds the transaction cap")
    if state.primary_failed_repair_rounds > state.last_counted_repair_round_id:
        raise ValueError("primary repair failures exceed counted repair rounds")
    if state.current_repair_round_id == 0 and (
        state.current_repair_provider is not None or state.current_repair_model
    ):
        raise ValueError("current repair attribution is orphaned")
    if state.current_repair_round_id > 0 and (
        state.current_repair_provider is None or not state.current_repair_model
    ):
        raise ValueError("current repair round lacks runtime author attribution")
    if state.authorized_repair_round_id == 0 and (
        state.authorized_repair_provider is not None
        or state.authorized_repair_model
    ):
        raise ValueError("authorized repair attribution is orphaned")
    if state.authorized_repair_round_id > 0:
        if (
            state.authorized_repair_provider is None
            or not state.authorized_repair_model
        ):
            raise ValueError("authorized repair round lacks runtime attribution")
        if state.authorized_repair_round_id > state.repair_round_sequence:
            raise ValueError("authorized repair round exceeds the sequence")
        if state.tool_patch_approval is None:
            raise ValueError("authorized repair round lacks PatchGate approval")


def begin_repair_round(state: AgentState) -> int:
    validate_repair_round_state(state)
    if (
        state.current_repair_round_id > 0
        and state.current_repair_round_id > state.last_counted_repair_round_id
    ):
        return state.current_repair_round_id
    if (
        max(state.repair_round_sequence, state.last_counted_repair_round_id)
        >= state.max_retries + 1
    ):
        raise ValueError("repair retry budget is exhausted")
    state.repair_round_sequence = max(
        state.repair_round_sequence,
        state.last_counted_repair_round_id,
    )
    state.repair_round_sequence += 1
    state.current_repair_round_id = state.repair_round_sequence
    bind_repair_round_author(state)
    return state.current_repair_round_id


def bind_repair_round_author(state: AgentState) -> None:
    if state.current_repair_round_id <= 0:
        raise ValueError("repair round must begin before author binding")
    state.current_repair_provider = state.active_provider
    state.current_repair_model = state.active_model


def freeze_authorized_repair_round(state: AgentState) -> None:
    if state.current_repair_round_id <= 0:
        raise ValueError("authorized patch requires an active repair round")
    if state.current_repair_provider is None or not state.current_repair_model:
        raise ValueError("authorized patch requires runtime author binding")
    if (
        state.tool_patch_approval is None
        or not state.patch_content
        or not state.patch_edits
    ):
        raise ValueError("authorized repair attribution requires PatchGate output")
    proposed = (
        state.current_repair_round_id,
        state.current_repair_provider,
        state.current_repair_model,
    )
    existing = (
        state.authorized_repair_round_id,
        state.authorized_repair_provider,
        state.authorized_repair_model,
    )
    if state.authorized_repair_round_id > 0:
        if existing != proposed:
            raise ValueError("authorized repair attribution is already frozen")
        return
    state.authorized_repair_round_id = state.current_repair_round_id
    state.authorized_repair_provider = state.current_repair_provider
    state.authorized_repair_model = state.current_repair_model


def clear_authorized_repair_round(state: AgentState) -> None:
    state.authorized_repair_round_id = 0
    state.authorized_repair_provider = None
    state.authorized_repair_model = ""


def retire_patch_authorization(
    state: AgentState,
    *,
    preserve_authorized_attribution: bool = False,
) -> None:
    state.patch_content = ""
    state.patch_edits = []
    state.active_repair_plan = None
    state.tool_patch_approval = None
    state.generated_test_approvals = []
    if not preserve_authorized_attribution:
        clear_authorized_repair_round(state)


def record_failed_repair_round(
    state: AgentState,
    *,
    round_id: int,
    provider: RepairProvider,
    model: str,
    failure_reason: str,
    retry_phase: Phase,
    immediate_reason: str = "",
) -> RepairRetryDecision:
    if round_id <= 0:
        raise ValueError("repair round ID must be positive")
    if round_id <= state.last_counted_repair_round_id:
        return RepairRetryDecision(
            counted=False,
            retry_allowed=state.current_phase not in {Phase.FAILURE, Phase.FAILED},
            round_id=round_id,
        )

    validate_repair_round_state(state)
    if (
        round_id != state.current_repair_round_id
        or round_id > state.repair_round_sequence
    ):
        raise ValueError("repair failure does not match the open repair round")
    if (
        provider != state.current_repair_provider
        or model != state.current_repair_model
    ):
        raise ValueError("repair failure author attribution does not match binding")

    state.last_counted_repair_round_id = round_id
    if provider == "primary":
        state.primary_failed_repair_rounds += 1

    retry_allowed = state.retry_count < state.max_retries
    if retry_allowed:
        state.retry_count += 1
        state.current_repair_round_id = 0
        state.current_repair_provider = None
        state.current_repair_model = ""
        state.failure_reason = str(failure_reason or "")[:8_000]
        decision = should_escalate(state, immediate_reason=immediate_reason)
        apply_escalation(state, decision)
        state.current_phase = retry_phase
    else:
        state.failure_reason = f"Maximum retries reached: {state.max_retries}."
        state.current_phase = Phase.FAILURE

    _record_node_diagnostic(
        state,
        node=retry_phase.value.lower(),
        event="repair_round_failed",
        status="error",
        elapsed_seconds=0.0,
        provider=provider,
        model=model,
        repair_round_id=round_id,
        retry_count=state.retry_count,
        primary_failed_repair_rounds=state.primary_failed_repair_rounds,
        failure_reason=str(failure_reason or "")[:64],
    )
    return RepairRetryDecision(
        counted=True,
        retry_allowed=retry_allowed,
        round_id=round_id,
    )
