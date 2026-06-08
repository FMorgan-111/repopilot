"""RepoPilot v2 agent graph runner, router, and fallback classes."""

from __future__ import annotations

from typing import Any

from .state import AgentState, NodeFn, _as_state, Phase

try:  # pragma: no cover - exercised only when langgraph is installed.
    from langgraph.graph import END, StateGraph
except ImportError:  # pragma: no cover - fallback is covered by tests.
    END = "__end__"
    StateGraph = None


class FallbackCompiledGraph:
    """Minimal async graph runner matching the LangGraph node contract."""

    def __init__(self, nodes: dict[str, NodeFn], start: str):
        self.nodes = nodes
        self.start = start

    async def ainvoke(self, state: AgentState | dict[str, Any]) -> AgentState:
        current = self.start
        working = _as_state(state)
        guard = 0
        while current != END:
            guard += 1
            if guard > 64:
                working.failure_reason = "State graph guard limit reached."
                working.current_phase = Phase.FAILED
                return working
            node = self.nodes[current]
            working = _as_state(await node(working))
            current = route_from_state(working)
        return working


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
