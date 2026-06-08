"""RepoPilot v2 agent: graph-based issue fixing loop.

This module is a thin re-export wrapper. The implementation lives in:
  src/state.py        — models, enums, and helper functions
  src/nodes/          — individual phase implementations
  src/graph.py        — graph runner, router, and fallback classes
"""

from __future__ import annotations

import asyncio
from typing import Any

from .state import (
    AgentState,
    ConversationTurn,
    FileInfo,
    FinalReport,
    FixAttempt,
    NodeFn,
    Phase,
    ToolCall,
    _as_state,
    _estimate_tokens,
    _extract_json_object,
    _is_budget_exceeded,
    _issue_search_terms,
    _primary_patch_file,
    _rank_reason,
    _record_tool,
    _remember,
)
from .nodes.understand import understand_issue
from .nodes.locate import locate_code
from .nodes.plan import plan_fix
from .nodes.execute import execute_fix, git_clone, apply_patch, run_pytest
from .nodes.verify import verify_fix
from .nodes.reflect import reflect_on_failure
from .nodes.commit import commit_fix, push_files, create_pr
from .nodes.failure import handle_failure
from .graph import (
    END,
    FallbackCompiledGraph,
    FallbackStateGraph,
    route_from_state,
    run_graph,
    StateGraph,
)
from .tracer import Tracer


def _wrap_node(name: str, fn: Any) -> Any:
    """Wrap a node function with progress output and timeout."""
    import sys
    import time as _time
    from .graph import PHASE_TIMEOUTS

    timeout = PHASE_TIMEOUTS.get(name, 60.0)

    async def wrapped(state):
        t0 = _time.monotonic()
        print(f"[{_time.strftime('%H:%M:%S')}] {name:24s} START", file=sys.stderr, flush=True)
        try:
            result = await asyncio.wait_for(fn(state), timeout=timeout)
        except asyncio.TimeoutError:
            elapsed = _time.monotonic() - t0
            print(f"[{_time.strftime('%H:%M:%S')}] {name:24s} TIMEOUT ({elapsed:.1f}s)", file=sys.stderr, flush=True)
            s = _as_state(state)
            s.failure_reason = f"Phase {name} timed out after {timeout}s"
            s.current_phase = Phase.FAILURE
            return s
        except Exception as exc:
            elapsed = _time.monotonic() - t0
            print(f"[{_time.strftime('%H:%M:%S')}] {name:24s} ERROR {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            s = _as_state(state)
            s.failure_reason = f"Phase {name} crashed: {exc}"
            s.current_phase = Phase.FAILURE
            return s
        elapsed = _time.monotonic() - t0
        next_phase = _as_state(result).current_phase.value if hasattr(_as_state(result), 'current_phase') else '?'
        print(f"[{_time.strftime('%H:%M:%S')}] {name:24s} DONE → {next_phase} ({elapsed:.1f}s)", file=sys.stderr, flush=True)
        return result

    return wrapped


def build_agent_graph() -> Any:
    """Build the RepoPilot v2 state graph.

    Defined here (not in graph.py) so that monkeypatching the node-function
    attributes on this module (as tests do) flows through to the graph.
    """
    # Wrap all nodes with progress output + timeouts
    _w = _wrap_node

    if StateGraph is None:
        graph = FallbackStateGraph()
        for name, fn in {
            "understand_issue": _w("understand_issue", understand_issue),
            "locate_code": _w("locate_code", locate_code),
            "plan_fix": _w("plan_fix", plan_fix),
            "reflect_on_failure": _w("reflect_on_failure", reflect_on_failure),
            "execute_fix": _w("execute_fix", execute_fix),
            "verify_fix": _w("verify_fix", verify_fix),
            "commit_fix": _w("commit_fix", commit_fix),
            "handle_failure": _w("handle_failure", handle_failure),
        }.items():
            graph.add_node(name, fn)
        graph.set_entry_point("understand_issue")
        return graph.compile()

    graph = StateGraph(AgentState)
    graph.add_node("understand_issue", _w("understand_issue", understand_issue))
    graph.add_node("locate_code", _w("locate_code", locate_code))
    graph.add_node("plan_fix", _w("plan_fix", plan_fix))
    graph.add_node("execute_fix", _w("execute_fix", execute_fix))
    graph.add_node("verify_fix", _w("verify_fix", verify_fix))
    graph.add_node("reflect_on_failure", _w("reflect_on_failure", reflect_on_failure))
    graph.add_node("commit_fix", _w("commit_fix", commit_fix))
    graph.add_node("handle_failure", _w("handle_failure", handle_failure))
    for node in [
        "understand_issue",
        "locate_code",
        "plan_fix",
        "reflect_on_failure",
        "execute_fix",
        "verify_fix",
        "commit_fix",
        "handle_failure",
    ]:
        graph.add_conditional_edges(
            node,
            route_from_state,
            {
                "understand_issue": "understand_issue",
                "locate_code": "locate_code",
                "plan_fix": "plan_fix",
                "reflect_on_failure": "reflect_on_failure",
                "execute_fix": "execute_fix",
                "verify_fix": "verify_fix",
                "commit_fix": "commit_fix",
                "handle_failure": "handle_failure",
                END: END,
            },
        )
    graph.set_entry_point("understand_issue")
    return graph.compile()


def final_report_from_state(state: AgentState, turns_taken: int) -> FinalReport:
    return FinalReport(
        issue_url=state.issue_url,
        fix_applied=state.current_phase == Phase.DONE,
        pr_url=state.pr_url,
        test_results=state.fix_attempts[-1].test_result if state.fix_attempts else "",
        turns_taken=turns_taken,
        token_used=state.token_usage,
    )


async def agent_v2(issue_url: str, max_retries: int = 3, token_budget: int = 50000) -> dict:
    """Run the full RepoPilot v2 graph with progress output and trace saving."""
    import sys
    import time as _time
    t_start = _time.monotonic()
    print(f"[agent_v2] Starting for {issue_url}", file=sys.stderr, flush=True)

    tracer = Tracer()
    state = AgentState(
        issue_url=issue_url,
        max_retries=max_retries,
        token_budget=token_budget,
        trace_id=tracer.trace_id,
    )
    print(f"[agent_v2] Building agent graph...", file=sys.stderr, flush=True)
    graph = build_agent_graph()
    print(f"[agent_v2] Running graph (trace={tracer.trace_id})...", file=sys.stderr, flush=True)

    try:
        final_state = await run_graph(graph, state)
    except Exception as exc:
        elapsed = _time.monotonic() - t_start
        print(f"[agent_v2] Graph crashed after {elapsed:.1f}s: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        tracer.log(
            "agent_v2_crash",
            {"issue_url": issue_url},
            {"error": f"{type(exc).__name__}: {exc}"},
            error=str(exc),
        )
        # Save partial trace
        _save_trace(tracer, "examples/traces/case_1.json")
        return {
            "error": f"Graph crashed: {type(exc).__name__}: {exc}",
            "trace_id": tracer.trace_id,
            "done": True,
            "success": False,
            "final_phase": "CRASHED",
        }

    elapsed = _time.monotonic() - t_start
    report = final_report_from_state(final_state, len(final_state.tool_calls))
    tracer.log(
        "agent_v2_done",
        {"issue_url": issue_url},
        {"phase": final_state.current_phase.value, "pr_url": final_state.pr_url},
        error=final_state.failure_reason or None,
    )
    print(f"[agent_v2] Done in {elapsed:.1f}s → {final_state.current_phase.value}", file=sys.stderr, flush=True)

    payload = report.model_dump()
    payload.update(
        {
            "done": final_state.current_phase in {Phase.DONE, Phase.FAILED},
            "success": final_state.current_phase == Phase.DONE,
            "final_phase": final_state.current_phase.value,
            "trace_id": tracer.trace_id,
            "relevant_files": [file.model_dump() for file in final_state.relevant_files],
            "fix_attempts": [attempt.model_dump() for attempt in final_state.fix_attempts],
            "error": final_state.failure_reason or None,
        }
    )

    # Save trace to file
    _save_trace(tracer, "examples/traces/case_1.json")
    return payload


def _save_trace(tracer: Tracer, path: str) -> None:
    """Save trace steps to a JSON file."""
    import json
    from pathlib import Path
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(tracer.steps, indent=2, default=str), encoding="utf-8")
        import sys
        print(f"[agent_v2] Trace saved to {p.resolve()}", file=sys.stderr, flush=True)
    except Exception as exc:
        import sys
        print(f"[agent_v2] Failed to save trace: {exc}", file=sys.stderr, flush=True)


async def intelligent_analyze_issue(
    issue_url: str, max_retries: int = 3, token_budget: int = 50000
) -> dict:
    """Backward-compatible alias for the previous experimental endpoint."""
    return await agent_v2(issue_url, max_retries=max_retries, token_budget=token_budget)


if __name__ == "__main__":  # pragma: no cover
    print(
        asyncio.run(
            agent_v2(
                "https://github.com/example/repo/issues/1",
                max_retries=1,
                token_budget=10000,
            )
        )
    )
