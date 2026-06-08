"""VERIFY phase: Parse test output and route to COMMIT, retry PLAN, or FAILED."""

from __future__ import annotations

from typing import Any

from ..state import (
    AgentState,
    Phase,
    _as_state,
    _is_budget_exceeded,
    _same_failure_seen_twice,
)


async def verify_fix(state: AgentState | dict[str, Any]) -> AgentState:
    """Parse test output and route to COMMIT, retry PLAN, or FAILED."""
    state = _as_state(state)
    if not state.fix_attempts:
        state.failure_reason = "No fix attempt was recorded."
        state.current_phase = Phase.FAILURE
        return state

    latest = state.fix_attempts[-1]
    if latest.success:
        state.current_phase = Phase.COMMIT
        return state

    if _same_failure_seen_twice(state):
        state.failure_reason = "Same patch produced the same failure twice."
        state.current_phase = Phase.FAILURE
        return state

    if state.retry_count >= state.max_retries:
        state.failure_reason = f"Maximum retries reached: {state.max_retries}."
        state.current_phase = Phase.FAILURE
        return state

    if _is_budget_exceeded(state):
        state.failure_reason = "Token budget exceeded during verification."
        state.current_phase = Phase.FAILURE
        return state

    state.retry_count += 1
    state.current_phase = Phase.REFLECT
    return state
