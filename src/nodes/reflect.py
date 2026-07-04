"""REFLECT phase: Ask the LLM to analyze WHY the previous fix attempt failed."""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from typing import Any

from ..llm import llm_call
from ..schemas import ReflectDecision
from .plan import _prior_failed_edits_context
from ..state import (
    AgentState,
    Phase,
    _as_state,
    _describe_exception,
    _estimate_tokens,
    _extract_json_object,
    _human_answer_context,
    _is_budget_exceeded,
    _record_decision_frame,
    _record_frame_health_warning,
    _record_node_diagnostic,
    _remember,
)


def _normalize_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        normalized: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise TypeError("Expected a sequence of strings")
            normalized.append(item)
        return normalized
    raise TypeError("Expected None, a string, or a sequence of strings")


def _normalize_reflect_decision(response: dict[str, Any]) -> ReflectDecision:
    files_that_also_need_changes = _normalize_string_list(
        response.get("files_that_also_need_changes")
    )
    if "decision_frame" in response:
        return ReflectDecision.model_validate(
            {
                **response,
                "files_that_also_need_changes": files_that_also_need_changes,
            }
        )
    return ReflectDecision.model_validate(
        {
            "root_cause": response.get("root_cause", ""),
            "what_went_wrong": response.get("what_went_wrong", ""),
            "suggested_fix_approach": response.get("suggested_fix_approach", ""),
            "files_that_also_need_changes": files_that_also_need_changes,
            "decision_frame": {
                "stage": "reflect",
                "summary": response.get("root_cause", ""),
                "hypotheses": response.get("hypotheses", []),
                "selected_hypothesis_id": response.get("selected_hypothesis_id"),
                "evidence": response.get("evidence", []),
                "next_checks": response.get("next_checks", []),
                "recommended_action": "plan",
                "confidence": response.get("confidence", 0.0),
                "risk": response.get("risk", "unknown"),
                "trace_notes": json.dumps(
                    {
                        "what_went_wrong": response.get("what_went_wrong", ""),
                        "suggested_fix_approach": response.get(
                            "suggested_fix_approach", ""
                        ),
                        "files_that_also_need_changes": files_that_also_need_changes,
                    }
                ),
            },
        }
    )


def _is_patch_apply_failure(attempt: Any) -> bool:
    return (
        getattr(attempt, "failure_kind", "") == "patch_apply_failed"
        or getattr(attempt, "test_result", "") == "patch_apply_failed"
    )


def _truncate_prompt_text(value: str, limit: int = 500) -> str:
    value = value.strip()
    if len(value) <= limit:
        return value
    return f"{value[:limit].rstrip()}..."


def _selected_hypothesis_context(state: AgentState) -> str:
    frames = []
    if state.decision_frame is not None:
        frames.append(state.decision_frame)
    for frame in reversed(state.frame_history):
        if state.decision_frame is not None and frame is state.decision_frame:
            continue
        if state.decision_frame is not None and frame.frame_id == state.decision_frame.frame_id:
            continue
        frames.append(frame)
        if len(frames) >= 2:
            break

    if not frames:
        return ""

    lines = []
    for frame in frames[:2]:
        selected_id = frame.selected_hypothesis_id or "(none)"
        lines.append(
            "- "
            f"{frame.stage} frame {frame.frame_id or '(unrecorded)'}: "
            f"summary={_truncate_prompt_text(frame.summary, 300)}; "
            f"selected_hypothesis_id={selected_id}"
        )
        selected = next(
            (hypothesis for hypothesis in frame.hypotheses if hypothesis.id == selected_id),
            None,
        )
        if selected is not None:
            lines.append(
                "  "
                f"selected hypothesis {selected.id}: "
                f"{_truncate_prompt_text(selected.claim, 300)}"
            )
            if selected.evidence:
                evidence = "; ".join(selected.evidence[:2])
                lines.append(f"  evidence: {_truncate_prompt_text(evidence, 300)}")
        elif frame.evidence:
            evidence = "; ".join(frame.evidence[:2])
            lines.append(f"  frame evidence: {_truncate_prompt_text(evidence, 300)}")
    return "\n".join(lines)


def _patch_apply_failure_prompt(state: AgentState, patch_apply_error: str) -> str:
    selected_context = _selected_hypothesis_context(state)
    if not selected_context:
        selected_context = "(no prior selected hypothesis recorded)"
    return (
        "Patch Apply Failure Instructions:\n"
        "- Tests did not run; the patch failed before tests ran and only patch formatting/path/hunk context failed.\n"
        "- If the error says patch preflight check failed, git rejected the old unified diff during `git apply --check`; switch to exact search/replace patch_edits before consuming semantic retries.\n"
        "- Treat the next LLM action as exact patch_edits repair, not root-cause exploration.\n"
        "- Keep the selected root-cause hypothesis unless the apply error proves "
        "the target file or hunk context is impossible.\n"
        "- Repair the previous patch's file paths and search blocks before changing semantics.\n"
        "- The replacement plan should use patch_edits entries with file, search, replace, and optional replace_all.\n"
        "- Search blocks must be copied exactly from the target file and should match uniquely unless replace_all is true.\n\n"
        "Previous patch apply error:\n"
        f"{_truncate_prompt_text(patch_apply_error, 500)}\n\n"
        "Previous selected hypothesis context:\n"
        f"{selected_context}"
    )


def _patch_edit_snippet(attempt: Any, limit: int = 2000) -> str:
    patch_edits = getattr(attempt, "patch_edits", [])
    if not patch_edits:
        return ""

    blocks = ["Patch Edits Applied:"]
    for idx, edit in enumerate(patch_edits, start=1):
        blocks.extend(
            [
                f"Edit {idx}:",
                f"file: {edit.file_path}",
                f"replace_all: {edit.replace_all}",
                "search:",
                edit.search,
                "replace:",
                edit.replace,
            ]
        )
    return _truncate_prompt_text("\n".join(blocks), limit)


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
    patch_snippet = (
        _patch_edit_snippet(latest)
        or latest.patch_content[:2000]
        or "(no patch)"
    )
    patch_apply_failure = _is_patch_apply_failure(latest)

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
        "files_that_also_need_changes (array of strings), decision_frame (object). "
        "decision_frame must include: stage='reflect', summary, hypotheses "
        "(array of objects with id, claim, evidence, score), selected_hypothesis_id, "
        "evidence, next_checks, recommended_action='plan', "
        "risk (low|medium|high|unknown), confidence (number 0.0 to 1.0)."
    )
    if patch_apply_failure:
        system = (
            f"{system} When the latest failure is patch_apply_failed, treat the"
            " analysis as search/replace repair guidance: tests did not run, the patch failed before tests ran,"
            " only patch formatting/path/hunk context failed, and the selected"
            " hypothesis should remain anchored unless the apply error proves the"
            " target file or hunk context is impossible."
        )
    failure_context = f"Test Output:\n{test_output}"
    if patch_apply_failure:
        failure_context = (
            "Patch Apply Error (tests did not run; the patch failed before tests ran):\n"
            f"{test_output}\n\n"
            "Only patch formatting/path/hunk context failed."
        )
    user = (
        f"Issue Title: {state.issue_title}\n\n"
        f"Issue Body (first 2000 chars):\n{state.issue_body[:2000]}\n\n"
        f"Patch Applied:\n{patch_snippet}\n\n"
        f"{failure_context}\n\n"
        f"Previous Attempts Summary:\n{previous_summary}"
    )
    resumed_answer_context = _human_answer_context(state)
    if resumed_answer_context:
        user = f"{user}\n\n{resumed_answer_context}"
    diversity_context = _prior_failed_edits_context(state)
    if diversity_context:
        user = (
            f"{user}\n\n{diversity_context}\n\n"
            "Your suggested_fix_approach MUST propose a DIFFERENT edit than the "
            "already-failed ones above — target a different location, hunk, or "
            "root cause rather than repeating a failed search/replace."
        )
    if patch_apply_failure:
        user = f"{user}\n\n{_patch_apply_failure_prompt(state, test_output)}"
    prompt_tokens_estimate = _estimate_tokens(system, user)

    t0 = time.monotonic()
    try:
        response = _extract_json_object(await llm_call(system, user))
        response_text = json.dumps(response)
        _record_node_diagnostic(
            state,
            node="reflect_on_failure",
            event="llm_call",
            status="success",
            elapsed_seconds=time.monotonic() - t0,
            prompt_tokens_estimate=prompt_tokens_estimate,
            response_tokens_estimate=_estimate_tokens(response_text),
        )
        has_explicit_frame = "decision_frame" in response
        decision = _normalize_reflect_decision(response)
        state.reflection_notes = response_text
        state.token_usage += _estimate_tokens(system, user, state.reflection_notes)
        _remember(state, "assistant", f"Reflection: {state.reflection_notes[:2000]}")
        frame = decision.decision_frame
        frame.parent_frame_id = state.decision_frame.frame_id if state.decision_frame else None
        if not frame.trace_notes:
            frame.trace_notes = json.dumps(
                {
                    "what_went_wrong": decision.what_went_wrong,
                    "suggested_fix_approach": decision.suggested_fix_approach,
                    "files_that_also_need_changes": decision.files_that_also_need_changes,
                }
            )
        _record_decision_frame(state, frame)
        if not has_explicit_frame:
            _record_frame_health_warning(
                state,
                node="reflect_on_failure",
                expected_stage="reflect",
                frame=frame,
                reason="missing_explicit_decision_frame",
            )
    except Exception as exc:
        _record_node_diagnostic(
            state,
            node="reflect_on_failure",
            event="llm_call",
            status="error",
            elapsed_seconds=time.monotonic() - t0,
            error=exc,
            prompt_tokens_estimate=prompt_tokens_estimate,
        )
        state.reflection_notes = f"Reflection failed: {_describe_exception(exc)}"
        state.token_usage += _estimate_tokens(system, user)
        _remember(state, "assistant", f"Reflection error: {_describe_exception(exc)}")

    state.current_phase = Phase.PLAN
    return state
