"""REFLECT phase: Ask the LLM to analyze WHY the previous fix attempt failed."""

from __future__ import annotations

import json
from typing import Any

from ..state import (
    AgentState,
    DecisionFrame,
    Phase,
    _as_state,
    _estimate_tokens,
    _extract_json_object,
    _is_budget_exceeded,
    _record_decision_frame,
    _remember,
)
from ..llm import llm_call


async def reflect_on_failure(state: AgentState | dict[str, Any]) -> AgentState:
    """Ask the LLM to analyze WHY the previous fix attempt failed."""
    state = _as_state(state)
    if _is_budget_exceeded(state):
        state.failure_reason = "Token budget exceeded before reflection."
        state.current_phase = Phase.FAILURE
        return state

    attempts = state.fix_attempts
    if not attempts:
        state.failure_reason = "No fix attempt to reflect on."
        state.current_phase = Phase.PLAN
        return state

    latest = attempts[-1]
    test_output = latest.error_log[:3000] if latest.error_log else "(no output)"
    patch_snippet = latest.patch_content[:2000] if latest.patch_content else "(no patch)"

    previous_summary = ""
    if len(attempts) > 1:
        previous_summary = "\n\n".join(
            f"Previous attempt {idx + 1}: success={a.success}, test_result={a.test_result}"
            for idx, a in enumerate(attempts[:-1])
        ) or "(none)"

    system = (
        "You are RepoPilot's reflection node. Analyze WHY the fix failed. "
        "Be specific. Return JSON with keys: root_cause (string), "
        "what_went_wrong (string), suggested_fix_approach (string), "
        "files_that_also_need_changes (array of strings), hypotheses (array of "
        "objects with id, claim, evidence, score), selected_hypothesis_id "
        "(string), evidence (array of strings), next_checks (array of strings), "
        "risk (low|medium|high|unknown), confidence (number 0.0 to 1.0)."
    )
    user = (
        f"Issue Title: {state.issue_title}\n\n"
        f"Issue Body (first 2000 chars):\n{state.issue_body[:2000]}\n\n"
        f"Patch Applied:\n{patch_snippet}\n\n"
        f"Test Output:\n{test_output}\n\n"
        f"Previous Attempts Summary:\n{previous_summary}"
    )

    try:
        response = _extract_json_object(await llm_call(system, user))
        state.reflection_notes = json.dumps(response)
        state.token_usage += _estimate_tokens(system, user, state.reflection_notes)
        _remember(state, "assistant", f"Reflection: {state.reflection_notes[:2000]}")
        frame = DecisionFrame(
            stage="reflect",
            summary=response.get("root_cause", ""),
            hypotheses=response.get("hypotheses", []),
            selected_hypothesis_id=response.get("selected_hypothesis_id"),
            evidence=response.get("evidence", []),
            next_checks=response.get("next_checks", []),
            recommended_action="plan",
            confidence=response.get("confidence", 0.0),
            risk=response.get("risk", "unknown"),
            parent_frame_id=state.decision_frame.frame_id if state.decision_frame else None,
            trace_notes=json.dumps(
                {
                    "what_went_wrong": response.get("what_went_wrong", ""),
                    "suggested_fix_approach": response.get("suggested_fix_approach", ""),
                    "files_that_also_need_changes": response.get(
                        "files_that_also_need_changes", []
                    ),
                }
            ),
        )
        _record_decision_frame(state, frame)
    except Exception as exc:
        state.reflection_notes = f"Reflection failed: {exc}"
        state.token_usage += _estimate_tokens(system, user)
        _remember(state, "assistant", f"Reflection error: {exc}")

    state.current_phase = Phase.PLAN
    return state
