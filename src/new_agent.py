"""RepoPilot v2 agent: graph-based issue fixing loop.

This module is a thin re-export wrapper. The implementation lives in:
  src/state.py        — models, enums, and helper functions
  src/nodes/          — individual phase implementations
  src/graph.py        — graph runner, router, and fallback classes
"""

from __future__ import annotations

import asyncio
from typing import Any

from .graph import (
    END,
    FallbackCompiledGraph,
    FallbackStateGraph,
    StateGraph,
    route_from_state,
    run_graph,
)
from .nodes.commit import commit_fix, create_pr, push_files
from .nodes.execute import apply_patch, execute_fix, git_clone, run_pytest
from .nodes.failure import handle_failure
from .nodes.locate import locate_code
from .nodes.plan import plan_fix
from .nodes.reflect import reflect_on_failure
from .nodes.understand import understand_issue
from .nodes.verify import verify_fix
from .run_store import load_run, save_run
from .state import (
    AgentState,
    ConversationTurn,
    DecisionFrame,
    FileInfo,
    FinalReport,
    FixAttempt,
    Hypothesis,
    NodeFn,
    PatchEdit,
    Phase,
    ToolCall,
    _as_state,
    _estimate_tokens,
    _extract_json_object,
    _is_budget_exceeded,
    _issue_search_terms,
    _primary_patch_file,
    _rank_reason,
    _record_decision_frame,
    _record_node_diagnostic,
    _record_tool,
    _remember,
)
from .tracer import Tracer

__all__ = [
    "END",
    "AgentState",
    "ConversationTurn",
    "DecisionFrame",
    "FallbackCompiledGraph",
    "FallbackStateGraph",
    "FileInfo",
    "FinalReport",
    "FixAttempt",
    "Hypothesis",
    "NodeFn",
    "PatchEdit",
    "Phase",
    "StateGraph",
    "ToolCall",
    "Tracer",
    "_as_state",
    "_estimate_tokens",
    "_extract_json_object",
    "_is_budget_exceeded",
    "_issue_search_terms",
    "_primary_patch_file",
    "_rank_reason",
    "_record_decision_frame",
    "_record_node_diagnostic",
    "_record_tool",
    "_remember",
    "agent_payload_from_state",
    "agent_v2",
    "apply_patch",
    "build_agent_graph",
    "commit_fix",
    "create_pr",
    "execute_fix",
    "final_report_from_state",
    "git_clone",
    "handle_failure",
    "intelligent_analyze_issue",
    "locate_code",
    "plan_fix",
    "push_files",
    "reflect_on_failure",
    "resume_agent_v2",
    "route_from_state",
    "run_graph",
    "run_pytest",
    "understand_issue",
    "verify_fix",
]


def _wrap_node(name: str, fn: Any, *, record_route_decision: bool = False) -> Any:
    """Wrap a node function with progress output and timeout."""
    import sys
    import time as _time

    from .graph import PHASE_TIMEOUTS

    timeout = PHASE_TIMEOUTS.get(name, 60.0)

    def route_detail(state: AgentState) -> str:
        if record_route_decision:
            return route_from_state(state)
        return state.current_phase.value

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
            _record_node_diagnostic(
                s,
                node=name,
                event="phase",
                status="timeout",
                elapsed_seconds=elapsed,
                error=asyncio.TimeoutError(),
                phase_timeout_seconds=timeout,
            )
            if record_route_decision:
                route_from_state(s)
            return s
        except Exception as exc:
            elapsed = _time.monotonic() - t0
            print(f"[{_time.strftime('%H:%M:%S')}] {name:24s} ERROR {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            s = _as_state(state)
            s.failure_reason = f"Phase {name} crashed: {exc}"
            s.current_phase = Phase.FAILURE
            _record_node_diagnostic(
                s,
                node=name,
                event="phase",
                status="error",
                elapsed_seconds=elapsed,
                error=exc,
                phase_timeout_seconds=timeout,
            )
            if record_route_decision:
                route_from_state(s)
            return s
        elapsed = _time.monotonic() - t0
        result_state = _as_state(result)
        next_phase = route_detail(result_state)
        print(f"[{_time.strftime('%H:%M:%S')}] {name:24s} DONE → {next_phase} ({elapsed:.1f}s)", file=sys.stderr, flush=True)
        return result_state

    return wrapped


def build_agent_graph(start_phase: Phase = Phase.UNDERSTAND) -> Any:
    """Build the RepoPilot v2 state graph.

    Defined here (not in graph.py) so that monkeypatching the node-function
    attributes on this module (as tests do) flows through to the graph.
    """
    # Wrap all nodes with progress output + timeouts
    _w = _wrap_node
    entry_point = _entry_point_for_phase(start_phase)

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
        graph.set_entry_point(entry_point)
        return graph.compile()

    async def route_from_recorded_decision(state: AgentState | dict[str, Any]) -> str:
        state = _as_state(state)
        if state.route_decisions:
            return state.route_decisions[-1]["route"]
        return route_from_state(state)

    graph = StateGraph(AgentState)
    graph.add_node("understand_issue", _w("understand_issue", understand_issue, record_route_decision=True))
    graph.add_node("locate_code", _w("locate_code", locate_code, record_route_decision=True))
    graph.add_node("plan_fix", _w("plan_fix", plan_fix, record_route_decision=True))
    graph.add_node("execute_fix", _w("execute_fix", execute_fix, record_route_decision=True))
    graph.add_node("verify_fix", _w("verify_fix", verify_fix, record_route_decision=True))
    graph.add_node("reflect_on_failure", _w("reflect_on_failure", reflect_on_failure, record_route_decision=True))
    graph.add_node("commit_fix", _w("commit_fix", commit_fix, record_route_decision=True))
    graph.add_node("handle_failure", _w("handle_failure", handle_failure, record_route_decision=True))
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
            route_from_recorded_decision,
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
    graph.set_entry_point(entry_point)
    return graph.compile()


def _entry_point_for_phase(phase: Phase) -> str:
    route = {
        Phase.UNDERSTAND: "understand_issue",
        Phase.LOCATE: "locate_code",
        Phase.PLAN: "plan_fix",
        Phase.REFLECT: "reflect_on_failure",
        Phase.EXECUTE: "execute_fix",
        Phase.VERIFY: "verify_fix",
        Phase.COMMIT: "commit_fix",
        Phase.FAILURE: "handle_failure",
    }.get(phase)
    if route is None:
        raise ValueError(f"Cannot start graph from phase {phase.value}.")
    return route


def final_report_from_state(state: AgentState, turns_taken: int) -> FinalReport:
    return FinalReport(
        issue_url=state.issue_url,
        fix_applied=state.current_phase == Phase.DONE,
        pr_url=state.pr_url,
        test_results=state.fix_attempts[-1].test_result if state.fix_attempts else "",
        turns_taken=turns_taken,
        token_used=state.token_usage,
    )


def agent_payload_from_state(state: AgentState, turns_taken: int) -> dict[str, Any]:
    report = final_report_from_state(state, turns_taken)
    payload = report.model_dump()
    payload.update(
        {
            "done": state.current_phase in {Phase.DONE, Phase.FAILED},
            "success": state.current_phase == Phase.DONE,
            "waiting_for_user": state.current_phase == Phase.WAITING_FOR_USER,
            "final_phase": state.current_phase.value,
            "trace_id": state.trace_id,
            "relevant_files": [file.model_dump() for file in state.relevant_files],
            "fix_attempts": [attempt.model_dump() for attempt in state.fix_attempts],
            "decision_frame": (
                state.decision_frame.model_dump() if state.decision_frame else None
            ),
            "frame_history": [frame.model_dump() for frame in state.frame_history],
            "decision_warnings": state.decision_warnings,
            "route_decisions": state.route_decisions,
            "node_diagnostics": state.node_diagnostics,
            "human_input_request": state.human_input_request,
            "error": state.failure_reason or None,
        }
    )
    payload["run_id"] = state.trace_id
    return payload


def _best_effort_save_run(state: AgentState) -> None:
    import sys

    try:
        save_run(state)
    except OSError as exc:
        print(
            f"[agent_v2] Failed to save run {state.trace_id}: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
    except PermissionError as exc:
        print(
            f"[agent_v2] Failed to save run {state.trace_id}: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )


async def agent_v2(
    issue_url: str,
    max_retries: int = 3,
    token_budget: int = 50000,
    save_final_run: bool = False,
    skip_commit: bool = False,
    seed: dict | None = None,
) -> dict:
    """Run the full RepoPilot v2 graph with progress output and trace saving.

    When `seed` is provided (offline eval), pre-populate the issue text and
    relevant files and start at PLAN — skipping UNDERSTAND/LOCATE so a flaky
    GitHub code search cannot starve the run before the patch stage."""
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
        skip_commit=skip_commit,
    )
    start_phase = Phase.UNDERSTAND
    if seed:
        state.owner = seed.get("owner", "")
        state.repo = seed.get("repo", "")
        state.issue_number = seed.get("issue_number", 0)
        state.issue_title = seed.get("issue_title", "")
        state.issue_body = seed.get("issue_body", "")
        state.relevant_files = [FileInfo(**f) for f in seed.get("relevant_files", [])]
        state.current_phase = Phase.PLAN
        start_phase = Phase.PLAN
        print(
            f"[agent_v2] Seeded {len(state.relevant_files)} file(s) for "
            f"{state.owner}/{state.repo}; starting at PLAN",
            file=sys.stderr,
            flush=True,
        )
    print("[agent_v2] Building agent graph...", file=sys.stderr, flush=True)
    graph = build_agent_graph(start_phase=start_phase)
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
        _save_trace(tracer, f"examples/traces/trace_{tracer.trace_id}.json", state)
        return {
            "error": f"Graph crashed: {type(exc).__name__}: {exc}",
            "trace_id": tracer.trace_id,
            "run_id": tracer.trace_id,
            "done": True,
            "success": False,
            "waiting_for_user": False,
            "final_phase": "CRASHED",
            "human_input_request": {},
            "node_diagnostics": state.node_diagnostics,
        }

    elapsed = _time.monotonic() - t_start
    tracer.log(
        "agent_v2_done",
        {"issue_url": issue_url},
        {"phase": final_state.current_phase.value, "pr_url": final_state.pr_url},
        error=final_state.failure_reason or None,
    )
    # Per-attempt failure classification — the raw signal for diagnosing WHY a
    # run failed (apply vs test vs path), one line per fix attempt.
    for i, att in enumerate(final_state.fix_attempts, start=1):
        kind = att.failure_kind or ("success" if att.success else att.test_result or "?")
        err = (att.error_log or "").replace("\n", " ")[:160]
        print(
            f"  [classify] attempt {i}: kind={kind} err={err}",
            file=sys.stderr,
            flush=True,
        )
    print(f"[agent_v2] Done in {elapsed:.1f}s → {final_state.current_phase.value}", file=sys.stderr, flush=True)

    payload = agent_payload_from_state(final_state, len(final_state.tool_calls))

    if save_final_run or final_state.current_phase == Phase.WAITING_FOR_USER:
        _best_effort_save_run(final_state)

    # Save trace to file
    _save_trace(tracer, f"examples/traces/trace_{tracer.trace_id}.json", final_state)
    return payload


async def resume_agent_v2(run_id: str, human_answer: str) -> dict:
    """Resume a paused RepoPilot v2 run with a human answer."""
    state = load_run(run_id)
    if (
        state.current_phase != Phase.WAITING_FOR_USER
        or not state.pending_human_input
        or not state.human_input_request
    ):
        payload = agent_payload_from_state(state, len(state.tool_calls))
        payload["success"] = False
        payload["error"] = f"Run {run_id} is not waiting for user input."
        return payload

    _remember(
        state,
        "user",
        f"Human answer for paused run {run_id}:\n{human_answer}",
    )
    state.pending_human_input = False
    state.human_input_request = {}
    state.current_phase = Phase.PLAN
    if state.decision_frame and not state.decision_route_checked_frame_id:
        state.decision_route_checked_frame_id = state.decision_frame.frame_id

    graph = build_agent_graph(start_phase=Phase.PLAN)
    final_state = await run_graph(graph, state)
    payload = agent_payload_from_state(final_state, len(final_state.tool_calls))

    if final_state.current_phase == Phase.WAITING_FOR_USER:
        _best_effort_save_run(final_state)

    tracer = Tracer()
    tracer.trace_id = state.trace_id
    _save_trace(tracer, f"examples/traces/trace_{tracer.trace_id}.json", final_state)
    return payload


def _save_trace(tracer: Tracer, path: str, state: AgentState | None = None) -> None:
    """Save trace steps and decision frames to a JSON file."""
    import json
    from pathlib import Path
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "trace_id": tracer.trace_id,
            "steps": tracer.steps,
            "decision_frame": (
                state.decision_frame.model_dump()
                if state and state.decision_frame
                else None
            ),
            "frame_history": (
                [frame.model_dump() for frame in state.frame_history]
                if state
                else []
            ),
            "decision_warnings": state.decision_warnings if state else [],
            "route_decisions": state.route_decisions if state else [],
            "node_diagnostics": state.node_diagnostics if state else [],
            "pending_human_input": state.pending_human_input if state else False,
            "human_input_request": state.human_input_request if state else {},
        }
        p.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
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
