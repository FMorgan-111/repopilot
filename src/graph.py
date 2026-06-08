"""RepoPilot v2 agent graph runner, router, and fallback classes."""

from __future__ import annotations

import asyncio
import sys
import time
from typing import Any

from .state import AgentState, NodeFn, _as_state, Phase

try:  # pragma: no cover - exercised only when langgraph is installed.
    from langgraph.graph import END, StateGraph
except ImportError:  # pragma: no cover - fallback is covered by tests.
    END = "__end__"
    StateGraph = None

# ── per-phase timeouts (seconds) ──────────────────────────────────────────
PHASE_TIMEOUTS: dict[str, float] = {
    "understand_issue": 15.0,
    "locate_code": 30.0,
    "plan_fix": 40.0,
    "execute_fix": 120.0,
    "verify_fix": 15.0,
    "reflect_on_failure": 35.0,
    "commit_fix": 30.0,
    "handle_failure": 5.0,
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
    current = _as_state(state).current_phase
    if current == Phase.UNDERSTAND:
        return "understand_issue"
    if current == Phase.LOCATE:
        return "locate_code"
    if current == Phase.PLAN:
        return "plan_fix"
    if current == Phase.REFLECT:
        return "reflect_on_failure"
    if current == Phase.EXECUTE:
        return "execute_fix"
    if current == Phase.VERIFY:
        return "verify_fix"
    if current == Phase.COMMIT:
        return "commit_fix"
    if current == Phase.FAILURE:
        return "handle_failure"
    return END


async def run_graph(graph: Any, state: AgentState) -> AgentState:
    result = await graph.ainvoke(state)
    return _as_state(result)
