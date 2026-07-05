"""RepoPilot v2 agent graph runner, router, and fallback classes."""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from typing import Any

from .state import AgentState, NodeFn, Phase, _as_state, _record_node_diagnostic

try:  # pragma: no cover - exercised only when langgraph is installed.
    from langgraph.graph import END, StateGraph
except ImportError:  # pragma: no cover - fallback is covered by tests.
    END = "__end__"
    StateGraph = None

# ── per-phase timeouts (seconds) ──────────────────────────────────────────
# plan_fix / reflect_on_failure must cover the full LLM retry budget (wall-clock
# + backoff = 320s) plus margin. Reasoning models (gpt-5.5) push single calls to
# 100-123s, and the retry path can stack; 360s clears it. understand also makes
# an LLM call so it must cover the retry window too.
PHASE_TIMEOUTS: dict[str, float] = {
    "understand_issue": 360.0,
    "locate_code": 180.0,
    "plan_fix": 360.0,
    "execute_fix": 600.0,
    "verify_fix": 15.0,
    "reflect_on_failure": 360.0,
    "commit_fix": 600.0,
    "handle_failure": 60.0,
}

logger = logging.getLogger("repopilot.graph")

_RECOMMENDED_PHASES: dict[str, Phase] = {
    "collect_more_context": Phase.LOCATE,
    "plan": Phase.PLAN,
    "execute": Phase.EXECUTE,
    "reflect": Phase.REFLECT,
    "stop": Phase.FAILURE,
    "ask_user": Phase.WAITING_FOR_USER,
}


class FallbackCompiledGraph:
    """Minimal async graph runner matching the LangGraph node contract."""

    def __init__(self, nodes: dict[str, NodeFn], start: str):
        self.nodes = nodes
        self.start = start
        self._progress_fn = _default_progress

    async def ainvoke(self, state: AgentState | dict[str, Any]) -> AgentState:
        current = self.start
        working = _as_state(state)
        guard = 0
        while current != END:
            guard += 1
            if guard > 64:
                working.failure_reason = "State graph guard limit reached."
                working.current_phase = Phase.FAILED
                self._progress_fn(
                    current, "ABORT", "guard limit (64) reached"
                )
                return working

            self._progress_fn(current, "START")

            node = self.nodes[current]
            timeout = PHASE_TIMEOUTS.get(current, 60.0)
            t0 = time.monotonic()
            try:
                working = _as_state(
                    await asyncio.wait_for(node(working), timeout=timeout)
                )
            except asyncio.TimeoutError:
                elapsed = time.monotonic() - t0
                working.failure_reason = (
                    f"Phase {current} timed out after {timeout}s (elapsed {elapsed:.1f}s)"
                )
                working.current_phase = Phase.FAILURE
                _record_node_diagnostic(
                    working,
                    node=current,
                    event="phase",
                    status="timeout",
                    elapsed_seconds=elapsed,
                    error=asyncio.TimeoutError(),
                    phase_timeout_seconds=timeout,
                )
                self._progress_fn(
                    current, "TIMEOUT",
                    f"{timeout}s limit exceeded"
                )
                return working
            except Exception as exc:
                elapsed = time.monotonic() - t0
                working.failure_reason = (
                    f"Phase {current} crashed: {exc}"
                )
                working.current_phase = Phase.FAILURE
                _record_node_diagnostic(
                    working,
                    node=current,
                    event="phase",
                    status="error",
                    elapsed_seconds=elapsed,
                    error=exc,
                    phase_timeout_seconds=timeout,
                )
                self._progress_fn(
                    current, "ERROR",
                    f"{type(exc).__name__}: {exc}"
                )
                return working

            elapsed = time.monotonic() - t0
            next_phase = route_from_state(working)
            self._progress_fn(
                current,
                "DONE",
                f"→ {next_phase} ({elapsed:.1f}s)",
            )
            current = next_phase
        return working


def _default_progress(node: str, event: str, detail: str = "") -> None:
    """Minimal progress printer — writes to stderr so stdout stays clean."""
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {node:24s} {event:8s}"
    if detail:
        line += f"  {detail}"
    print(line, file=sys.stderr, flush=True)


class FallbackStateGraph:
    def __init__(self) -> None:
        self.nodes: dict[str, NodeFn] = {}
        self.start = ""

    def add_node(self, name: str, fn: NodeFn) -> None:
        self.nodes[name] = fn

    def set_entry_point(self, name: str) -> None:
        self.start = name

    def compile(self) -> FallbackCompiledGraph:
        return FallbackCompiledGraph(self.nodes, self.start)


def route_from_state(state: AgentState | dict[str, Any]) -> str:
    working = _as_state(state)
    recommended = _consume_decision_recommended_phase(working)
    selected_phase = (
        recommended["phase"]
        if recommended and "phase" in recommended
        else working.current_phase
    )
    route = _route_for_phase(selected_phase)
    _record_route_decision(working, selected_phase, route, recommended)
    return route


def _route_for_phase(phase: Phase) -> str:
    if phase == Phase.UNDERSTAND:
        return "understand_issue"
    if phase == Phase.LOCATE:
        return "locate_code"
    if phase == Phase.PLAN:
        return "plan_fix"
    if phase == Phase.REFLECT:
        return "reflect_on_failure"
    if phase == Phase.EXECUTE:
        return "execute_fix"
    if phase == Phase.VERIFY:
        return "verify_fix"
    if phase == Phase.COMMIT:
        return "commit_fix"
    if phase == Phase.WAITING_FOR_USER:
        return END
    if phase == Phase.FAILURE:
        return "handle_failure"
    return END


def _consume_decision_recommended_phase(state: AgentState) -> dict[str, Any] | None:
    frame = state.decision_frame
    if frame is None:
        return {"fallback_reason": "no_decision_frame"}
    if not frame.frame_id:
        return {
            "frame_id": "",
            "recommended_action": frame.recommended_action,
            "fallback_reason": "no_frame_id",
        }
    if not state.frame_history or state.frame_history[-1].frame_id != frame.frame_id:
        return {
            "frame_id": frame.frame_id,
            "recommended_action": frame.recommended_action,
            "fallback_reason": "stale_frame",
        }
    if frame.frame_id == state.decision_route_checked_frame_id:
        return {
            "frame_id": frame.frame_id,
            "recommended_action": frame.recommended_action,
            "fallback_reason": "already_consumed",
        }

    expected_phase = _RECOMMENDED_PHASES.get(frame.recommended_action)
    if expected_phase is None:
        return {
            "frame_id": frame.frame_id,
            "recommended_action": frame.recommended_action,
            "fallback_reason": "unsupported_recommended_action",
        }

    original_phase = state.current_phase
    state.decision_route_checked_frame_id = frame.frame_id
    if expected_phase == Phase.WAITING_FOR_USER:
        if expected_phase != original_phase:
            _record_decision_warning(state, expected_phase, original_phase)
        _prepare_human_input_request(state)
        return {
            "source": "decision_frame",
            "phase": expected_phase,
            "frame_id": frame.frame_id,
            "recommended_action": frame.recommended_action,
        }

    if expected_phase == original_phase:
        return {
            "source": "decision_frame",
            "phase": expected_phase,
            "frame_id": frame.frame_id,
            "recommended_action": frame.recommended_action,
        }

    _record_decision_warning(state, expected_phase, original_phase)
    return {
        "source": "decision_frame",
        "phase": expected_phase,
        "frame_id": frame.frame_id,
        "recommended_action": frame.recommended_action,
    }


def _record_decision_warning(
    state: AgentState,
    expected_phase: Phase,
    actual_phase: Phase,
) -> None:
    frame = state.decision_frame
    if frame is None:
        return
    message = (
        f"DecisionFrame recommended_action '{frame.recommended_action}' "
        f"expected phase {expected_phase.value} but current_phase is "
        f"{actual_phase.value}"
    )
    warning = {
        "frame_id": frame.frame_id,
        "stage": frame.stage,
        "recommended_action": frame.recommended_action,
        "expected_phase": expected_phase.value,
        "actual_phase": actual_phase.value,
        "message": message,
    }
    state.decision_warnings.append(warning)
    logger.warning(message)


def _prepare_human_input_request(state: AgentState) -> None:
    frame = state.decision_frame
    if frame is None:
        return
    question = frame.next_checks[0].strip() if frame.next_checks else ""
    if not question:
        question = frame.summary
    state.pending_human_input = True
    state.current_phase = Phase.WAITING_FOR_USER
    state.human_input_request = {
        "frame_id": frame.frame_id,
        "stage": frame.stage,
        "question": question,
        "summary": frame.summary,
        "risk": frame.risk,
        "confidence": frame.confidence,
    }


def _record_route_decision(
    state: AgentState,
    selected_phase: Phase,
    route: str,
    recommended: dict[str, Any] | None,
) -> None:
    decision = {
        "source": "decision_frame" if recommended and recommended.get("phase") else "current_phase",
        "current_phase": state.current_phase.value,
        "selected_phase": selected_phase.value,
        "route": route,
    }
    if recommended:
        if "frame_id" in recommended:
            decision["frame_id"] = recommended["frame_id"]
        if "recommended_action" in recommended:
            decision["recommended_action"] = recommended["recommended_action"]
        if "fallback_reason" in recommended:
            decision["fallback_reason"] = recommended["fallback_reason"]
    state.route_decisions.append(decision)


async def run_graph(graph: Any, state: AgentState) -> AgentState:
    result = await graph.ainvoke(state)
    return _as_state(result)
