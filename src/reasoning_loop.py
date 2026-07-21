"""Shared bounded mechanics for PLAN and REFLECT reasoning rounds."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from .evidence import EvidenceStore
from .model_policy import (
    apply_escalation,
    record_no_progress,
    record_progress,
    should_escalate,
)
from .state import AgentState, _record_node_diagnostic
from .tool_policy import ToolIntent
from .tool_router import ToolRouteResult

MAX_TOOL_REQUESTS_PER_ROUND = 8
NEW_EVIDENCE_SECTION = "NEW APPROVED EVIDENCE FROM THE PREVIOUS TOOL REQUEST"
_OUTCOME_FIELDS = {
    "plan": frozenset(
        {
            "kind",
            "plan",
            "patch",
            "patch_edits",
            "files",
            "test_command",
            "decision_frame",
        }
    ),
    "reflect": frozenset(
        {
            "kind",
            "root_cause",
            "what_went_wrong",
            "suggested_fix_approach",
            "files_that_also_need_changes",
            "decision_frame",
        }
    ),
}


@dataclass(frozen=True)
class ToolStep:
    handled: bool = False
    evidence_ids: tuple[str, ...] = ()
    stop_reason: str = ""


def validate_reasoning_response(
    response: dict[str, Any],
    *,
    outcome_kind: str,
) -> str:
    """Validate the explicit union, adapting only untagged legacy outcomes."""
    if outcome_kind not in {"plan", "reflect"}:
        raise ValueError("structured response has an unsupported outcome kind")

    explicit_kind = str(response.get("kind") or "").strip().lower()
    if not explicit_kind:
        if response.get("tool_intent") is not None:
            response_tool_intent(response)
            return "tool"
        if response.get("stop_reason") and len(response) == 1:
            return "stop"
        return outcome_kind

    if explicit_kind not in {"tool", "stop", outcome_kind}:
        raise ValueError("structured response has an unknown or wrong variant")
    if explicit_kind == "tool":
        response_tool_intent(response)
        unexpected = set(response) - {"kind", "tool_intent"}
        if unexpected:
            raise ValueError("structured response mixed tool and outcome variants")
        return explicit_kind
    if explicit_kind == "stop":
        unexpected = set(response) - {"kind", "stop_reason"}
        if unexpected:
            raise ValueError("structured response mixed stop and outcome variants")
        return explicit_kind
    if response.get("tool_intent") is not None or response.get("stop_reason"):
        raise ValueError("structured response mixed outcome with another variant")
    unexpected = set(response) - _OUTCOME_FIELDS[outcome_kind]
    if unexpected:
        raise ValueError("structured response mixed or added outcome fields")
    return explicit_kind


def response_tool_intent(response: dict[str, Any]) -> ToolIntent | None:
    """Return one validated tool intent without accepting mixed outcome payloads."""
    kind = str(response.get("kind") or "").strip().lower()
    raw_intent = response.get("tool_intent")
    if kind and kind not in {"tool", "plan", "reflect", "stop"}:
        raise ValueError("structured response has an unknown variant")
    if kind != "tool" and raw_intent is None:
        return None
    if kind and kind != "tool" and raw_intent is not None:
        raise ValueError("structured response mixed tool and outcome variants")
    if raw_intent is None:
        raise ValueError("tool response omitted tool_intent")
    outcome_fields = (
        "plan",
        "patch",
        "patch_edits",
        "repair_plan",
        "verified_edits",
        "root_cause",
        "stop_reason",
    )
    if any(response.get(field) for field in outcome_fields):
        raise ValueError("structured response mixed tool and outcome variants")
    return ToolIntent.model_validate(raw_intent)


def is_stop_response(response: dict[str, Any]) -> bool:
    return str(response.get("kind") or "").strip().lower() == "stop"


def prompt_with_new_evidence(
    base_prompt: str,
    state: AgentState,
    evidence_ids: tuple[str, ...],
) -> str:
    """Render only IDs produced by the immediately preceding tool request."""
    selected = EvidenceStore(state).select(list(evidence_ids))
    rendered = EvidenceStore.render_for_prompt(selected)
    ids = ", ".join(item.evidence_id for item in selected)
    return (
        f"{base_prompt}\n\n{NEW_EVIDENCE_SECTION}\n"
        f"Evidence IDs: {ids or '(none)'}\n{rendered or '(no new evidence)'}"
    )


def record_opus_no_progress(
    state: AgentState,
    *,
    node: str,
    fingerprint: Any,
) -> bool:
    """Record an Opus-local streak and report whether its two-round cap is hit."""
    opus_round = state.opus_no_progress_rounds.get(node, 0) + 1
    state.opus_no_progress_rounds[node] = opus_round
    record_no_progress(
        state,
        kind="no_evidence_or_applicable_patch",
        node=node,
        fingerprint=fingerprint,
    )
    _record_node_diagnostic(
        state,
        node=node,
        event="opus_no_progress",
        status="error",
        elapsed_seconds=0.0,
        round=opus_round,
    )
    state.no_progress_rounds = opus_round
    return opus_round >= 2


async def route_reasoning_tool(
    state: AgentState,
    response: dict[str, Any],
    *,
    node: str,
    calls_this_round: int,
    router: Callable[..., Awaitable[ToolRouteResult]],
) -> ToolStep:
    """Authorize one model-selected tool and update deterministic progress state."""
    intent = response_tool_intent(response)
    if intent is None:
        return ToolStep()
    if calls_this_round >= MAX_TOOL_REQUESTS_PER_ROUND:
        _record_node_diagnostic(
            state,
            node=node,
            event="tool_round_limit",
            status="error",
            elapsed_seconds=0.0,
            calls_this_round=calls_this_round,
        )
        return ToolStep(handled=True, stop_reason="tool_round_limit")

    result = await router(
        state,
        intent,
        calls_this_round=calls_this_round,
    )
    _record_node_diagnostic(
        state,
        node=node,
        event="tool_intent",
        status=result.status,
        elapsed_seconds=0.0,
        action=intent.action,
        evidence_id=result.evidence_id,
        calls_this_round=calls_this_round + 1,
    )
    if result.made_progress and result.evidence_id:
        record_progress(state)
        return ToolStep(handled=True, evidence_ids=(result.evidence_id,))

    fingerprint = json.dumps(
        {"action": intent.action, "args": intent.args},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    if state.active_provider == "escalation":
        if record_opus_no_progress(state, node=node, fingerprint=fingerprint):
            return ToolStep(handled=True, stop_reason="opus_no_progress_limit")
    else:
        record_no_progress(
            state,
            kind="unchanged_context",
            node=node,
            fingerprint=fingerprint,
        )
        apply_escalation(state, should_escalate(state))
    if result.control_action == "finish_investigation":
        return ToolStep(handled=True, stop_reason="model_stop")
    return ToolStep(handled=True)
