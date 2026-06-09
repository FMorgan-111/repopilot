"""PLAN phase: Ask the LLM for a concrete patch-oriented plan."""

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


async def plan_fix(state: AgentState | dict[str, Any]) -> AgentState:
    """Ask the LLM for a concrete patch-oriented plan."""
    import sys
    state = _as_state(state)
    if _is_budget_exceeded(state):
        state.failure_reason = "Token budget exceeded before planning."
        state.current_phase = Phase.FAILURE
        return state

    previous_failures = "\n\n".join(
        f"Attempt {idx + 1}: {attempt.test_result}\n{attempt.error_log[:2000]}"
        for idx, attempt in enumerate(state.fix_attempts)
    )
    reflection_context = ""
    if state.reflection_notes:
        reflection_context = f"\n\nREFLECTION ANALYSIS:\n{state.reflection_notes}"

    files_context = "\n\n".join(
        f"FILE: {file.path}\nRELEVANCE: {file.relevance_score} - {file.reason}\n"
        f"CONTENT:\n{file.content[:2000]}"
        for file in state.relevant_files[:2]
    )
    system = (
        "You are RepoPilot's planning node. Return ONLY JSON with keys: "
        "plan (markdown string), patch (unified diff string), files (array of paths), "
        "test_command (string), hypotheses (array of objects with id, claim, evidence, "
        "score), selected_hypothesis_id (string), evidence (array of strings), "
        "next_checks (array of strings), risk (low|medium|high|unknown), "
        "confidence (number 0.0 to 1.0). The patch should be apply-able with git apply."
    )
    user = (
        f"Issue URL: {state.issue_url}\n"
        f"Title: {state.issue_title}\n\nBody:\n{state.issue_body[:4000]}\n\n"
        f"Relevant files:\n{files_context}\n\nPrevious failures:\n{previous_failures}"
        f"{reflection_context}"
    )

    try:
        print(f"  [plan] Calling LLM for fix plan...", file=sys.stderr, flush=True)
        response = _extract_json_object(await llm_call(system, user))
    except Exception as exc:
        state.failure_reason = f"Failed to generate fix plan: {exc}"
        state.current_phase = Phase.FAILURE
        return state

    state.fix_plan = response.get("plan", "")
    state.patch_content = response.get("patch", "")
    state.test_command = response.get("test_command", "")
    print(f"  [plan] Plan received ({len(state.fix_plan)} chars, patch={len(state.patch_content)} chars)", file=sys.stderr, flush=True)
    state.token_usage += _estimate_tokens(system, user, json.dumps(response))
    _remember(state, "assistant", state.fix_plan[:2000])
    state.current_phase = Phase.EXECUTE if state.patch_content else Phase.FAILURE
    frame = DecisionFrame(
        stage="plan",
        summary=state.fix_plan,
        hypotheses=response.get("hypotheses", []),
        selected_hypothesis_id=response.get("selected_hypothesis_id"),
        evidence=response.get("evidence", []),
        next_checks=response.get("next_checks", []),
        recommended_action="execute" if state.patch_content else "stop",
        confidence=response.get("confidence", 0.0),
        risk=response.get("risk", "unknown"),
        parent_frame_id=state.decision_frame.frame_id if state.decision_frame else None,
        trace_notes=json.dumps({"files": response.get("files", [])}),
    )
    _record_decision_frame(state, frame)
    if not state.patch_content:
        state.failure_reason = "Planner did not produce a patch."
    return state
