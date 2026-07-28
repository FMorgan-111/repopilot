"""RepoPilot v2 agent: graph-based issue fixing loop.

This module is a thin re-export wrapper. The implementation lives in:
  src/state.py        — models, enums, and helper functions
  src/nodes/          — individual phase implementations
  src/graph.py        — graph runner, router, and fallback classes
"""

from __future__ import annotations

import asyncio
import json
import re
import subprocess
from typing import Any

from .async_safety import CancellationDrainError, drain_task, wait_for_phase
from .coverage_gate import (
    LiveCoverageBinding,
    validate_live_coverage_binding,
    validate_terminal_coverage_binding,
)
from .evaluator_safety import safe_prediction_patch
from .graph import (
    END,
    FallbackCompiledGraph,
    FallbackStateGraph,
    StateGraph,
    capture_graph_states,
    route_from_state,
    run_graph,
)
from .model_provider import get_model_config
from .nodes.commit import commit_fix, create_pr, push_files
from .nodes.coverage import ensure_coverage
from .nodes.execute import apply_patch, execute_fix, git_clone, git_diff, run_pytest
from .nodes.failure import handle_failure
from .nodes.locate import locate_code
from .nodes.plan import plan_fix
from .nodes.reflect import reflect_on_failure
from .nodes.understand import understand_issue
from .nodes.verify import _attempt_binding, _trusted_authorized_binding, verify_fix
from .patch_gate import revalidate_approved_patch
from .run_store import claim_run_for_resume, load_run, save_run
from .safe_subprocess import tool_sandbox_config_from_env
from .state import (
    DEFAULT_AGENT_V2_MAX_RETRIES,
    DEFAULT_AGENT_V2_TOKEN_BUDGET,
    AgentState,
    ConversationTurn,
    CoverageProof,
    DecisionFrame,
    FileInfo,
    FinalReport,
    FixAttempt,
    Hypothesis,
    ModelInvocation,
    NodeFn,
    NoProgressEvent,
    PatchEdit,
    Phase,
    TestRunFingerprint,
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
    sanitize_node_diagnostics,
)
from .summary_safety import sanitize_summary_text
from .tracer import Tracer

__all__ = [
    "END",
    "AgentState",
    "ConversationTurn",
    "CoverageProof",
    "DecisionFrame",
    "FallbackCompiledGraph",
    "FallbackStateGraph",
    "FileInfo",
    "FinalReport",
    "FixAttempt",
    "Hypothesis",
    "ModelInvocation",
    "NodeFn",
    "NoProgressEvent",
    "PatchEdit",
    "Phase",
    "StateGraph",
    "ToolCall",
    "TestRunFingerprint",
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
    "ensure_coverage",
    "final_report_from_state",
    "git_clone",
    "git_diff",
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
            result = await wait_for_phase(fn(state), timeout=timeout)
        except asyncio.TimeoutError as exc:
            elapsed = _time.monotonic() - t0
            print(f"[{_time.strftime('%H:%M:%S')}] {name:24s} TIMEOUT ({elapsed:.1f}s)", file=sys.stderr, flush=True)
            s = _as_state(state)
            from .timeout_diagnostics import extract_timeout_cleanup_evidence

            cleanup_evidence = extract_timeout_cleanup_evidence(exc)
            failure_reason = f"Phase {name} timed out after {timeout}s"
            if cleanup_evidence is not None:
                failure_reason += f"; {cleanup_evidence.summary()}"
            s.failure_reason = failure_reason
            s.current_phase = Phase.FAILURE
            _record_node_diagnostic(
                s,
                node=name,
                event="phase",
                status="timeout",
                elapsed_seconds=elapsed,
                error=asyncio.TimeoutError(),
                phase_timeout_seconds=timeout,
                **(
                    cleanup_evidence.diagnostic_details()
                    if cleanup_evidence is not None
                    else {}
                ),
            )
            if record_route_decision:
                route_from_state(s)
            return s
        except CancellationDrainError:
            raise
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
            "understand_issue": understand_issue,
            "locate_code": locate_code,
            "plan_fix": plan_fix,
            "reflect_on_failure": reflect_on_failure,
            "execute_fix": execute_fix,
            "verify_fix": verify_fix,
            "ensure_coverage": ensure_coverage,
            "commit_fix": commit_fix,
            "handle_failure": handle_failure,
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
    graph.add_node("ensure_coverage", _w("ensure_coverage", ensure_coverage, record_route_decision=True))
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
        "ensure_coverage",
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
                "ensure_coverage": "ensure_coverage",
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
        Phase.COVERAGE: "ensure_coverage",
        Phase.COMMIT: "commit_fix",
        Phase.FAILURE: "handle_failure",
    }.get(phase)
    if route is None:
        raise ValueError(f"Cannot start graph from phase {phase.value}.")
    return route


def _terminal_binding_for_report(
    state: AgentState,
) -> LiveCoverageBinding | None:
    if state.current_phase != Phase.DONE:
        return None
    validation_state = state.model_copy(deep=True)
    try:
        return validate_terminal_coverage_binding(validation_state)
    except (OSError, RuntimeError, ValueError):
        return None


def _final_report_and_binding(
    state: AgentState, turns_taken: int
) -> tuple[FinalReport, LiveCoverageBinding | None]:
    live_binding = _terminal_binding_for_report(state)
    report = FinalReport(
        issue_url=state.issue_url,
        fix_applied=live_binding is not None,
        pr_url=state.pr_url,
        test_results=state.fix_attempts[-1].test_result if state.fix_attempts else "",
        turns_taken=turns_taken,
        token_used=state.token_usage + state.summary_token_usage,
    )
    return report, live_binding


def final_report_from_state(state: AgentState, turns_taken: int) -> FinalReport:
    report, _ = _final_report_and_binding(state, turns_taken)
    return report


def _validated_patch_for_report(state: AgentState) -> str:
    preflight_state = state.model_copy(deep=True)
    try:
        revalidate_approved_patch(preflight_state)
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError):
        pass
    else:
        return safe_prediction_patch(preflight_state.patch_content)

    live_state = state.model_copy(deep=True)
    attempt = _trusted_attempt_for_patch(live_state, live_state.patch_content)
    if attempt is None:
        return ""
    try:
        binding = validate_live_coverage_binding(live_state)
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError):
        return ""
    return safe_prediction_patch(binding.patch_content)


def _trusted_attempt_for_patch(
    state: AgentState, patch: str
) -> FixAttempt | None:
    for attempt in reversed(state.fix_attempts):
        if attempt.patch_content != patch:
            continue
        try:
            if _attempt_binding(attempt) != _trusted_authorized_binding(state, attempt):
                continue
        except ValueError:
            continue
        completed, _ = _attempt_test_completion(attempt)
        if not completed:
            continue
        return attempt
    return None


def _attempt_test_completion(attempt: FixAttempt) -> tuple[bool, bool | None]:
    if attempt.test_result == "execution_error":
        completed = attempt.failure_kind == "infra_error" and attempt.success is False
        return completed, None

    try:
        result = json.loads(attempt.test_result)
    except (TypeError, ValueError):
        return False, None
    if (
        not isinstance(result, dict)
        or set(result) != {"command", "returncode", "success"}
        or not isinstance(result["command"], str)
        or not result["command"].strip()
        or type(result["returncode"]) is not int
        or type(result["success"]) is not bool
    ):
        return False, None

    success = result["success"]
    expected_failure_kind = "" if success else "test_failed"
    if (
        success is not (result["returncode"] == 0)
        or success is not attempt.success
        or attempt.failure_kind != expected_failure_kind
    ):
        return False, None
    return True, success


def _tests_passed_for_patch(state: AgentState, patch: str) -> bool | None:
    if not patch:
        return None
    attempt = _trusted_attempt_for_patch(state, patch)
    if attempt is None:
        return None
    _, tests_passed = _attempt_test_completion(attempt)
    return tests_passed


def agent_payload_from_state(state: AgentState, turns_taken: int) -> dict[str, Any]:
    report, _ = _final_report_and_binding(state, turns_taken)
    terminal_success = report.fix_applied
    model_patch = _validated_patch_for_report(state)
    tests_passed = _tests_passed_for_patch(state, model_patch)
    payload = report.model_dump()
    payload.update(
        {
            "done": state.current_phase in {Phase.DONE, Phase.FAILED},
            "success": terminal_success,
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
            "node_diagnostics": sanitize_node_diagnostics(state.node_diagnostics),
            "active_model": state.active_model,
            "active_provider": state.active_provider,
            "escalated": state.escalated,
            "escalation_reason": state.escalation_reason,
            "no_progress_rounds": state.no_progress_rounds,
            "last_plan_signature": state.last_plan_signature,
            "last_context_fingerprint": state.last_context_fingerprint,
            "last_test_failure_signature": state.last_test_failure_signature,
            "model_history": [
                invocation.model_dump(mode="json")
                for invocation in state.model_history
            ],
            "test_generation_attempts": state.test_generation_attempts,
            "no_progress_history": [
                event.model_dump(mode="json")
                for event in state.no_progress_history
            ],
            "human_input_request": state.human_input_request,
            "coverage_status": state.coverage_status,
            "coverage_test_files": state.coverage_test_files,
            "coverage_test_command": state.coverage_test_command,
            "coverage_failure_reason": state.coverage_failure_reason,
            "coverage_proof": (
                state.coverage_proof.model_dump(mode="json")
                if state.coverage_proof
                else None
            ),
            "patch_generated": bool(model_patch),
            "tests_passed": tests_passed,
            "model_patch": model_patch,
            "error": state.failure_reason or None,
        }
    )
    payload["run_id"] = state.trace_id
    return payload


def _best_effort_save_run(state: AgentState) -> None:
    import sys

    try:
        save_run(state)
    except CancellationDrainError:
        raise
    except Exception as exc:
        error_summary = sanitize_summary_text(
            f"{type(exc).__name__}: {exc}",
            300,
        ) or type(exc).__name__
        print(
            f"[agent_v2] Failed to save run {state.trace_id}: {error_summary}",
            file=sys.stderr,
            flush=True,
        )


async def agent_v2(
    issue_url: str,
    max_retries: int = DEFAULT_AGENT_V2_MAX_RETRIES,
    token_budget: int = DEFAULT_AGENT_V2_TOKEN_BUDGET,
    save_final_run: bool = False,
    skip_commit: bool = False,
    seed: dict | None = None,
    patch_only: bool = True,
) -> dict:
    """Run the full RepoPilot v2 graph with progress output and trace saving.

    When `seed` is provided (offline eval), pre-populate the issue text. An
    issue-only seed starts at LOCATE; an oracle seed with hydrated files starts
    at PLAN. Both skip the live GitHub issue request in UNDERSTAND."""
    import sys
    import time as _time
    t_start = _time.monotonic()
    print(f"[agent_v2] Starting for {issue_url}", file=sys.stderr, flush=True)

    tracer = Tracer()
    seeded_trace_id = str((seed or {}).get("trace_id", ""))
    if re.fullmatch(r"[0-9a-f]{12}", seeded_trace_id):
        tracer.trace_id = seeded_trace_id
    state = AgentState(
        issue_url=issue_url,
        max_retries=max_retries,
        token_budget=token_budget,
        trace_id=tracer.trace_id,
        skip_commit=skip_commit,
        patch_only=patch_only,
        active_model=get_model_config("primary").model,
        active_provider="primary",
        tool_sandbox_config=tool_sandbox_config_from_env(),
    )
    start_phase = Phase.UNDERSTAND
    if seed:
        state.owner = seed.get("owner", "")
        state.repo = seed.get("repo", "")
        state.issue_number = seed.get("issue_number", 0)
        state.issue_title = seed.get("issue_title", "")
        state.issue_body = seed.get("issue_body", "")
        state.repo_ref = seed.get("repo_ref", "")
        state.repo_path = seed.get("repo_path", "")
        state.relevant_files = [FileInfo(**f) for f in seed.get("relevant_files", [])]
        start_phase = Phase.PLAN if state.relevant_files else Phase.LOCATE
        state.current_phase = start_phase
        print(
            f"[agent_v2] Seeded {len(state.relevant_files)} file(s) for "
            f"{state.owner}/{state.repo}; starting at {start_phase.value}",
            file=sys.stderr,
            flush=True,
        )
    print("[agent_v2] Building agent graph...", file=sys.stderr, flush=True)
    graph = build_agent_graph(start_phase=start_phase)
    print(f"[agent_v2] Running graph (trace={tracer.trace_id})...", file=sys.stderr, flush=True)
    latest_graph_state = state

    def observe_graph_state(candidate: AgentState) -> None:
        nonlocal latest_graph_state
        latest_graph_state = candidate

    try:
        with capture_graph_states(observe_graph_state):
            final_state = await run_graph(graph, state)
    except CancellationDrainError:
        raise
    except Exception as exc:
        elapsed = _time.monotonic() - t_start
        crash_error = sanitize_summary_text(
            f"Graph crashed: {type(exc).__name__}: {exc}",
            500,
        ) or "Graph crashed."
        print(
            f"[agent_v2] {crash_error} after {elapsed:.1f}s",
            file=sys.stderr,
            flush=True,
        )
        tracer.log(
            "agent_v2_crash",
            {"issue_url": issue_url},
            {"error": crash_error},
            error=crash_error,
        )
        crash_state = latest_graph_state.model_copy(deep=True)
        crash_state.current_phase = Phase.FAILED
        crash_state.pending_human_input = False
        crash_state.human_input_request = {}
        crash_state.failure_reason = crash_error
        payload = agent_payload_from_state(
            crash_state,
            len(crash_state.tool_calls),
        )
        payload.update(
            {
                "done": True,
                "success": False,
                "fix_applied": False,
                "waiting_for_user": False,
                "final_phase": "CRASHED",
            }
        )
        if save_final_run:
            _best_effort_save_run(crash_state)
        _save_trace(
            tracer,
            f"examples/traces/trace_{tracer.trace_id}.json",
            crash_state,
        )
        return payload

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


async def resume_agent_v2(
    run_id: str,
    human_answer: str,
    *,
    state: AgentState | None = None,
) -> dict:
    """Resume a paused RepoPilot v2 run with a human answer."""
    if state is None:
        state = await _run_store_call(load_run, run_id)
    state = await _run_store_call(claim_run_for_resume, run_id, state)
    claimed_state = state.model_copy(deep=True)

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
    final_state.resume_in_progress = False
    # This write is the commit point, including when the graph pauses again.
    # Cancellation drains it to one durable outcome before propagating.
    await _run_store_call(
        save_run,
        final_state,
        rollback_state=claimed_state,
    )
    payload = agent_payload_from_state(final_state, len(final_state.tool_calls))

    tracer = Tracer()
    tracer.trace_id = state.trace_id
    _save_trace(tracer, f"examples/traces/trace_{tracer.trace_id}.json", final_state)
    return payload


async def _run_store_call(operation, /, *args, **kwargs):
    """Run blocking store I/O off-loop and drain it before cancellation wins."""
    worker = asyncio.create_task(asyncio.to_thread(operation, *args, **kwargs))
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError as original_cancel:
        outcome = await drain_task(worker)
        if outcome.error is not None:
            raise CancellationDrainError(
                "run store call", original_cancel, outcome.error
            ) from outcome.error
        raise original_cancel


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
            "coverage_status": state.coverage_status if state else "pending",
            "coverage_proof": (
                state.coverage_proof.model_dump(mode="json")
                if state and state.coverage_proof
                else None
            ),
        }
        p.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        import sys
        print(f"[agent_v2] Trace saved to {p.resolve()}", file=sys.stderr, flush=True)
    except Exception as exc:
        import sys
        print(f"[agent_v2] Failed to save trace: {exc}", file=sys.stderr, flush=True)


async def intelligent_analyze_issue(
    issue_url: str,
    max_retries: int = DEFAULT_AGENT_V2_MAX_RETRIES,
    token_budget: int = DEFAULT_AGENT_V2_TOKEN_BUDGET,
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
