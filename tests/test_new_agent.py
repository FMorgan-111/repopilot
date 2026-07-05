import asyncio
import functools
import subprocess

import src.nodes.execute as execute_node
import src.run_store as run_store
from src import graph, http_client, new_agent


async def test_agent_v2_state_machine_transitions_to_done(monkeypatch):
    visited = []

    async def understand(state):
        visited.append(state.current_phase.value)
        state.issue_title = "Login crash"
        state.current_phase = new_agent.Phase.LOCATE
        return state

    async def locate(state):
        visited.append(state.current_phase.value)
        state.current_phase = new_agent.Phase.PLAN
        return state

    async def plan(state):
        visited.append(state.current_phase.value)
        state.current_phase = new_agent.Phase.EXECUTE
        return state

    async def execute(state):
        visited.append(state.current_phase.value)
        state.current_phase = new_agent.Phase.VERIFY
        return state

    async def verify(state):
        visited.append(state.current_phase.value)
        state.current_phase = new_agent.Phase.COMMIT
        return state

    async def commit(state):
        visited.append(state.current_phase.value)
        state.pr_url = "https://github.com/acme/widget/pull/42"
        state.current_phase = new_agent.Phase.DONE
        return state

    async def failure(state):
        visited.append(state.current_phase.value)
        state.current_phase = new_agent.Phase.FAILED
        return state

    monkeypatch.setattr(new_agent, "understand_issue", understand)
    monkeypatch.setattr(new_agent, "locate_code", locate)
    monkeypatch.setattr(new_agent, "plan_fix", plan)
    monkeypatch.setattr(new_agent, "execute_fix", execute)
    monkeypatch.setattr(new_agent, "verify_fix", verify)
    monkeypatch.setattr(new_agent, "commit_fix", commit)
    monkeypatch.setattr(new_agent, "handle_failure", failure)

    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        max_retries=2,
        token_budget=5000,
    )

    final_state = await new_agent.run_graph(new_agent.build_agent_graph(), state)

    assert visited == ["UNDERSTAND", "LOCATE", "PLAN", "EXECUTE", "VERIFY", "COMMIT"]
    assert final_state.current_phase == new_agent.Phase.DONE
    assert final_state.pr_url == "https://github.com/acme/widget/pull/42"
    assert [decision["route"] for decision in final_state.route_decisions] == [
        "locate_code",
        "plan_fix",
        "execute_fix",
        "verify_fix",
        "commit_fix",
        "__end__",
    ]


def test_langgraph_conditional_router_uses_native_async_callable():
    if new_agent.StateGraph is None:
        return

    graph = new_agent.build_agent_graph()
    branch = graph.builder.branches["understand_issue"]
    spec = next(iter(branch.values()))

    assert not (
        isinstance(spec.path.afunc, functools.partial)
        and spec.path.afunc.func.__name__ == "run_in_executor"
    )


def test_llm_phase_timeouts_cover_retry_window():
    llm_retry_window = http_client.llm_retry_budget_seconds()
    planner_margin = 30.0

    assert graph.PHASE_TIMEOUTS["understand_issue"] >= llm_retry_window
    assert graph.PHASE_TIMEOUTS["locate_code"] >= 180.0
    assert graph.PHASE_TIMEOUTS["plan_fix"] >= llm_retry_window + planner_margin
    assert graph.PHASE_TIMEOUTS["execute_fix"] >= 600.0
    assert graph.PHASE_TIMEOUTS["reflect_on_failure"] >= llm_retry_window + planner_margin
    assert graph.PHASE_TIMEOUTS["commit_fix"] >= 600.0
    assert graph.PHASE_TIMEOUTS["handle_failure"] >= 60.0


async def test_fallback_graph_records_phase_timeout_diagnostic(monkeypatch):
    async def slow_node(state):
        await asyncio.sleep(1)
        return state

    monkeypatch.setitem(graph.PHASE_TIMEOUTS, "plan_fix", 0.01)
    compiled = graph.FallbackCompiledGraph({"plan_fix": slow_node}, "plan_fix")
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        current_phase=new_agent.Phase.PLAN,
    )

    final_state = await compiled.ainvoke(state)

    assert final_state.current_phase == new_agent.Phase.FAILURE
    assert final_state.node_diagnostics[-1]["node"] == "plan_fix"
    assert final_state.node_diagnostics[-1]["event"] == "phase"
    assert final_state.node_diagnostics[-1]["status"] == "timeout"
    assert final_state.node_diagnostics[-1]["error_type"] == "TimeoutError"
    assert final_state.node_diagnostics[-1]["phase_timeout_seconds"] == 0.01


async def test_phase_timeout_preserves_existing_node_diagnostics(monkeypatch):
    async def slow_node(state):
        state.node_diagnostics.append(
            {
                "node": "plan_fix",
                "event": "prompt_built",
                "status": "success",
                "prompt_tokens_estimate": 3456,
                "relevant_file_count": 2,
            }
        )
        await asyncio.sleep(1)
        return state

    monkeypatch.setitem(graph.PHASE_TIMEOUTS, "plan_fix", 0.01)
    compiled = graph.FallbackCompiledGraph({"plan_fix": slow_node}, "plan_fix")
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        current_phase=new_agent.Phase.PLAN,
    )

    final_state = await compiled.ainvoke(state)

    assert final_state.node_diagnostics[-2]["event"] == "prompt_built"
    assert final_state.node_diagnostics[-2]["prompt_tokens_estimate"] == 3456
    assert final_state.node_diagnostics[-1]["event"] == "phase"
    assert final_state.node_diagnostics[-1]["status"] == "timeout"


async def test_agent_v2_crash_payload_exposes_human_input_defaults(monkeypatch):
    saved_traces = []

    async def crash_graph(graph, state):
        raise RuntimeError("boom")

    def save_trace(tracer, path, state=None):
        saved_traces.append({"path": path, "trace_id": tracer.trace_id, "state": state})

    monkeypatch.setattr(new_agent, "run_graph", crash_graph)
    monkeypatch.setattr(new_agent, "_save_trace", save_trace)

    payload = await new_agent.agent_v2("https://github.com/acme/widget/issues/7")

    assert payload["done"] is True
    assert payload["success"] is False
    assert payload["waiting_for_user"] is False
    assert payload["final_phase"] == "CRASHED"
    assert payload["human_input_request"] == {}
    assert saved_traces[0]["state"].issue_url == "https://github.com/acme/widget/issues/7"


async def test_agent_v2_saves_waiting_for_user_run(monkeypatch, tmp_path):
    async def fake_run_graph(graph, state):
        state.current_phase = new_agent.Phase.WAITING_FOR_USER
        state.pending_human_input = True
        state.human_input_request = {
            "frame_id": "df_0001",
            "stage": "plan",
            "question": "Confirm whether breaking changes are allowed.",
            "summary": "Need user approval before patching.",
            "risk": "high",
            "confidence": 0.88,
        }
        state.route_decisions.append(
            {
                "source": "decision_frame",
                "current_phase": "PLAN",
                "selected_phase": "WAITING_FOR_USER",
                "route": new_agent.END,
                "frame_id": "df_0001",
                "recommended_action": "ask_user",
            }
        )
        state.frame_history.append(
            new_agent.DecisionFrame(
                frame_id="df_0001",
                stage="plan",
                summary="Need user approval before patching.",
                recommended_action="ask_user",
                confidence=0.88,
                risk="high",
            )
        )
        return state

    monkeypatch.setattr(new_agent, "run_graph", fake_run_graph)
    monkeypatch.setattr(new_agent, "_save_trace", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        new_agent,
        "save_run",
        lambda state: run_store.save_run(state, root_dir=tmp_path / ".repopilot"),
    )

    payload = await new_agent.agent_v2("https://github.com/acme/widget/issues/7")

    assert payload["run_id"] == payload["trace_id"]
    assert payload["waiting_for_user"] is True
    assert (tmp_path / ".repopilot" / "runs" / f"{payload['trace_id']}.json").exists()


async def test_agent_v2_ignores_final_run_persistence_errors(monkeypatch):
    async def fake_run_graph(graph, state):
        state.current_phase = new_agent.Phase.DONE
        state.pr_url = "https://github.com/acme/widget/pull/42"
        return state

    def failing_save_run(state):
        raise OSError("read-only file system")

    monkeypatch.setattr(new_agent, "run_graph", fake_run_graph)
    monkeypatch.setattr(new_agent, "_save_trace", lambda *args, **kwargs: None)
    monkeypatch.setattr(new_agent, "save_run", failing_save_run)

    payload = await new_agent.agent_v2(
        "https://github.com/acme/widget/issues/7",
        save_final_run=True,
    )

    assert payload["success"] is True
    assert payload["final_phase"] == "DONE"
    assert payload["run_id"] == payload["trace_id"]


async def test_agent_v2_saves_final_run_when_requested(monkeypatch, tmp_path):
    async def fake_run_graph(graph, state):
        state.current_phase = new_agent.Phase.DONE
        state.pr_url = "https://github.com/acme/widget/pull/42"
        return state

    monkeypatch.setattr(new_agent, "run_graph", fake_run_graph)
    monkeypatch.setattr(new_agent, "_save_trace", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        new_agent,
        "save_run",
        lambda state: run_store.save_run(state, root_dir=tmp_path / ".repopilot"),
    )

    payload = await new_agent.agent_v2(
        "https://github.com/acme/widget/issues/7",
        save_final_run=True,
    )

    assert payload["success"] is True
    assert payload["run_id"] == payload["trace_id"]
    assert (tmp_path / ".repopilot" / "runs" / f"{payload['trace_id']}.json").exists()


async def test_agent_v2_starts_graph_at_understand(monkeypatch):
    captured_start_phases = []

    class FakeGraph:
        pass

    def fake_build_agent_graph(start_phase=new_agent.Phase.UNDERSTAND):
        captured_start_phases.append(start_phase)
        return FakeGraph()

    async def fake_run_graph(graph, state):
        state.current_phase = new_agent.Phase.DONE
        return state

    monkeypatch.setattr(new_agent, "build_agent_graph", fake_build_agent_graph)
    monkeypatch.setattr(new_agent, "run_graph", fake_run_graph)
    monkeypatch.setattr(new_agent, "_save_trace", lambda *args, **kwargs: None)

    await new_agent.agent_v2("https://github.com/acme/widget/issues/7")

    assert captured_start_phases == [new_agent.Phase.UNDERSTAND]


async def test_resume_agent_v2_rejects_non_paused_run(monkeypatch):
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        trace_id="abc123def456",
        current_phase=new_agent.Phase.DONE,
    )

    monkeypatch.setattr(new_agent, "load_run", lambda run_id: state)

    payload = await new_agent.resume_agent_v2(
        "abc123def456",
        "Breaking changes are not allowed.",
    )

    assert payload["success"] is False
    assert payload["waiting_for_user"] is False
    assert payload["final_phase"] == "DONE"
    assert payload["error"] == "Run abc123def456 is not waiting for user input."


async def test_resume_agent_v2_injects_answer_and_resumes_from_plan(monkeypatch):
    captured_states = []
    frame = new_agent.DecisionFrame(
        frame_id="df_0001",
        stage="plan",
        summary="Need user approval before patching.",
        recommended_action="ask_user",
        next_checks=["Confirm whether breaking changes are allowed."],
        confidence=0.88,
        risk="high",
    )
    paused_state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        trace_id="abc123def456",
        current_phase=new_agent.Phase.WAITING_FOR_USER,
        pending_human_input=True,
        human_input_request={
            "frame_id": "df_0001",
            "stage": "plan",
            "question": "Confirm whether breaking changes are allowed.",
            "summary": "Need user approval before patching.",
            "risk": "high",
            "confidence": 0.88,
        },
        decision_frame=frame,
        frame_history=[frame],
        decision_route_checked_frame_id="df_0001",
    )

    async def fake_run_graph(graph, state):
        captured_states.append(state.model_copy(deep=True))
        state.current_phase = new_agent.Phase.DONE
        return state

    monkeypatch.setattr(new_agent, "load_run", lambda run_id: paused_state)
    monkeypatch.setattr(new_agent, "run_graph", fake_run_graph)
    monkeypatch.setattr(new_agent, "_save_trace", lambda *args, **kwargs: None)

    payload = await new_agent.resume_agent_v2(
        "abc123def456",
        "Breaking changes are not allowed.",
    )

    resumed_state = captured_states[0]
    assert resumed_state.current_phase == new_agent.Phase.PLAN
    assert resumed_state.pending_human_input is False
    assert resumed_state.human_input_request == {}
    assert resumed_state.decision_route_checked_frame_id == "df_0001"
    assert resumed_state.conversation_history[-1] == new_agent.ConversationTurn(
        role="user",
        content=(
            "Human answer for paused run abc123def456:\n"
            "Breaking changes are not allowed."
        ),
    )
    assert payload["success"] is True
    assert payload["run_id"] == "abc123def456"
    assert payload["final_phase"] == "DONE"


async def test_resume_agent_v2_starts_graph_at_plan(monkeypatch):
    visited = []
    frame = new_agent.DecisionFrame(
        frame_id="df_0001",
        stage="plan",
        summary="Need user approval before patching.",
        recommended_action="ask_user",
    )
    paused_state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        trace_id="abc123def456",
        current_phase=new_agent.Phase.WAITING_FOR_USER,
        pending_human_input=True,
        human_input_request={"question": "Confirm the API behavior."},
        decision_frame=frame,
        frame_history=[frame],
        decision_route_checked_frame_id="df_0001",
    )

    async def understand(state):
        visited.append("understand_issue")
        state.current_phase = new_agent.Phase.LOCATE
        return state

    async def plan(state):
        visited.append("plan_fix")
        state.current_phase = new_agent.Phase.DONE
        return state

    monkeypatch.setattr(new_agent, "load_run", lambda run_id: paused_state)
    monkeypatch.setattr(new_agent, "_save_trace", lambda *args, **kwargs: None)
    monkeypatch.setattr(new_agent, "understand_issue", understand)
    monkeypatch.setattr(new_agent, "plan_fix", plan)

    payload = await new_agent.resume_agent_v2(
        "abc123def456",
        "Use the existing API behavior.",
    )

    assert visited == ["plan_fix"]
    assert payload["final_phase"] == "DONE"


async def test_resume_agent_v2_saves_run_when_it_pauses_again(monkeypatch, tmp_path):
    frame = new_agent.DecisionFrame(
        frame_id="df_0001",
        stage="plan",
        summary="Need user approval before patching.",
        recommended_action="ask_user",
    )
    paused_state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        trace_id="abc123def456",
        current_phase=new_agent.Phase.WAITING_FOR_USER,
        pending_human_input=True,
        human_input_request={"question": "Confirm the API behavior."},
        decision_frame=frame,
        frame_history=[frame],
        decision_route_checked_frame_id="df_0001",
    )

    async def fake_run_graph(graph, state):
        state.current_phase = new_agent.Phase.WAITING_FOR_USER
        state.pending_human_input = True
        state.human_input_request = {
            "frame_id": "df_0002",
            "stage": "plan",
            "question": "Confirm whether to update the public API.",
            "summary": "Need another product decision.",
            "risk": "high",
            "confidence": 0.7,
        }
        return state

    monkeypatch.setattr(new_agent, "load_run", lambda run_id: paused_state)
    monkeypatch.setattr(new_agent, "run_graph", fake_run_graph)
    monkeypatch.setattr(new_agent, "_save_trace", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        new_agent,
        "save_run",
        lambda state: run_store.save_run(state, root_dir=tmp_path / ".repopilot"),
    )

    payload = await new_agent.resume_agent_v2(
        "abc123def456",
        "Use the existing API behavior.",
    )

    assert payload["waiting_for_user"] is True
    assert (tmp_path / ".repopilot" / "runs" / "abc123def456.json").exists()


async def test_verify_fix_replans_failed_attempt_once():
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        max_retries=2,
    )
    state.current_phase = new_agent.Phase.VERIFY
    state.fix_attempts.append(
        new_agent.FixAttempt(
            patch_content="diff --git a/src/auth.py b/src/auth.py",
            file_path="src/auth.py",
            test_result="failed",
            error_log="assert False",
            success=False,
        )
    )

    next_state = await new_agent.verify_fix(state)

    assert next_state.current_phase == new_agent.Phase.REFLECT
    assert next_state.retry_count == 1


async def test_execute_fix_marks_patch_apply_failure_kind(monkeypatch):
    async def fake_apply_patch(repo_path, patch_content):
        return execute_node.PatchApplyResult(
            applied=False,
            output="error: corrupt patch at line 3",
            patch_content=patch_content,
        )

    monkeypatch.setattr(execute_node, "apply_patch_with_repair", fake_apply_patch)
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        repo_path="/tmp/repopilot-test-repo",
        patch_content="malformed diff",
    )

    next_state = await execute_node.execute_fix(state)

    assert next_state.current_phase == new_agent.Phase.VERIFY
    assert len(next_state.fix_attempts) == 1
    assert next_state.fix_attempts[-1].test_result == "patch_apply_failed"
    assert next_state.fix_attempts[-1].failure_kind == "patch_apply_failed"
    assert next_state.fix_attempts[-1].error_log == "error: corrupt patch at line 3"


async def test_execute_fix_marks_test_failure_kind(monkeypatch):
    async def fake_apply_patch(repo_path, patch_content):
        return execute_node.PatchApplyResult(
            applied=True,
            output="",
            patch_content=patch_content,
        )

    async def fake_run_pytest(repo_path, command=None):
        return {
            "command": "pytest tests/test_auth.py -q",
            "returncode": 1,
            "stdout": "FAILED tests/test_auth.py",
            "stderr": "",
            "success": False,
        }

    monkeypatch.setattr(execute_node, "apply_patch_with_repair", fake_apply_patch)
    monkeypatch.setattr(execute_node, "run_pytest", fake_run_pytest)
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        repo_path="/tmp/repopilot-test-repo",
        patch_content="diff --git a/src/auth.py b/src/auth.py",
        test_command="pytest tests/test_auth.py -q",
    )

    next_state = await execute_node.execute_fix(state)

    assert next_state.current_phase == new_agent.Phase.VERIFY
    assert len(next_state.fix_attempts) == 1
    assert next_state.fix_attempts[-1].success is False
    assert next_state.fix_attempts[-1].failure_kind == "test_failed"
    assert "FAILED tests/test_auth.py" in next_state.fix_attempts[-1].error_log


async def test_execute_fix_marks_execution_error_failure_kind(monkeypatch):
    async def fake_apply_patch(repo_path, patch_content):
        raise RuntimeError("git apply crashed")

    monkeypatch.setattr(execute_node, "apply_patch_with_repair", fake_apply_patch)
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        repo_path="/tmp/repopilot-test-repo",
        patch_content="diff --git a/src/auth.py b/src/auth.py",
    )

    next_state = await execute_node.execute_fix(state)

    assert next_state.current_phase == new_agent.Phase.VERIFY
    assert len(next_state.fix_attempts) == 1
    assert next_state.fix_attempts[-1].test_result == "execution_error"
    assert next_state.fix_attempts[-1].failure_kind == "execution_error"
    assert next_state.fix_attempts[-1].error_log == "git apply crashed"


async def test_run_git_async_kills_process_on_timeout():
    import time

    t0 = time.monotonic()
    # A sleep far longer than the timeout must be killed promptly, not waited
    # out — this is what lets the phase timeout reclaim a hung clone.
    try:
        await execute_node._run_git_async(["sleep", "30"], timeout=0.3)
    except asyncio.TimeoutError:
        pass
    else:
        raise AssertionError("expected TimeoutError")
    assert time.monotonic() - t0 < 5


async def test_worktree_is_healthy_detects_empty_and_valid(monkeypatch, tmp_path):
    # Empty/broken work tree (no HEAD, no files) → unhealthy; a real one with
    # HEAD + files → healthy. This is the guard that stops reusing a broken
    # clone (which made every patch fail "target file was not found").
    work = tmp_path / "wt"
    (work / ".git").mkdir(parents=True)

    async def fake_run_git(args, timeout, cwd=None):
        if "rev-parse" in args:
            return execute_node._ProcResult(0, "abc123\n", "")  # HEAD resolves
        if "ls-files" in args:
            return execute_node._ProcResult(0, "a.py\nb.py\n", "")  # has files
        return execute_node._ProcResult(0, "", "")

    monkeypatch.setattr(execute_node, "_run_git_async", fake_run_git)
    assert await execute_node._worktree_is_healthy(str(work)) is True

    async def fake_empty(args, timeout, cwd=None):
        if "rev-parse" in args:
            return execute_node._ProcResult(128, "", "fatal: Needed a single revision")
        return execute_node._ProcResult(0, "", "")

    monkeypatch.setattr(execute_node, "_run_git_async", fake_empty)
    assert await execute_node._worktree_is_healthy(str(work)) is False


async def test_worktree_is_healthy_false_when_head_ok_but_no_files(monkeypatch, tmp_path):
    work = tmp_path / "wt2"
    (work / ".git").mkdir(parents=True)

    async def fake_no_files(args, timeout, cwd=None):
        if "rev-parse" in args:
            return execute_node._ProcResult(0, "abc\n", "")
        if "ls-files" in args:
            return execute_node._ProcResult(0, "", "")  # HEAD ok but empty checkout
        return execute_node._ProcResult(0, "", "")

    monkeypatch.setattr(execute_node, "_run_git_async", fake_no_files)
    assert await execute_node._worktree_is_healthy(str(work)) is False


async def test_git_clone_uses_cached_repo_without_remote_clone(monkeypatch, tmp_path):
    repopilot_home = tmp_path / ".repopilot"
    cache_path = repopilot_home / "repos" / "acme-widget"
    (cache_path / ".git").mkdir(parents=True)
    monkeypatch.setenv("REPOPILOT_HOME", str(repopilot_home))
    commands = []

    async def fake_run_git(args, timeout, cwd=None):
        commands.append(args)
        return execute_node._ProcResult(0, "", "")

    monkeypatch.setattr(execute_node, "_run_git_async", fake_run_git)
    # Health check has its own test; here we exercise the clone path only.
    async def _healthy(_w):
        return True
    monkeypatch.setattr(execute_node, "_worktree_is_healthy", _healthy)
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        owner="acme",
        repo="widget",
    )

    repo_path = await execute_node.git_clone(state)

    assert commands == [
        [
            "git",
            "clone",
            "--local",
            "--no-hardlinks",
            str(cache_path),
            repo_path,
        ]
    ]
    assert all(
        "github.com" not in part for command in commands for part in command
    )


async def test_git_clone_populates_cache_then_clones_worktree(monkeypatch, tmp_path):
    repopilot_home = tmp_path / ".repopilot"
    cache_path = repopilot_home / "repos" / "acme-widget"
    monkeypatch.setenv("REPOPILOT_HOME", str(repopilot_home))
    monkeypatch.setenv("GITHUB_TOKEN", "gho_cachetoken")
    commands = []

    async def fake_run_git(args, timeout, cwd=None):
        commands.append(args)
        if args[:2] == ["git", "clone"] and args[-1] == str(cache_path):
            (cache_path / ".git").mkdir(parents=True)
        return execute_node._ProcResult(0, "", "")

    monkeypatch.setattr(execute_node, "_run_git_async", fake_run_git)
    async def _healthy(_w):
        return True
    monkeypatch.setattr(execute_node, "_worktree_is_healthy", _healthy)
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        owner="acme",
        repo="widget",
    )

    repo_path = await execute_node.git_clone(state)

    assert commands[0][:2] == ["git", "clone"]
    assert commands[0][-1] == str(cache_path)
    assert any(
        "https://x-access-token:gho_cachetoken@github.com/acme/widget.git" == part
        for part in commands[0]
    )
    assert commands[1] == [
        "git",
        "-C",
        str(cache_path),
        "remote",
        "set-url",
        "origin",
        "https://github.com/acme/widget.git",
    ]
    assert commands[2] == [
        "git",
        "clone",
        "--local",
        "--no-hardlinks",
        str(cache_path),
        repo_path,
    ]


async def test_git_clone_failed_cache_population_redacts_token(monkeypatch, tmp_path):
    repopilot_home = tmp_path / ".repopilot"
    cache_path = repopilot_home / "repos" / "acme-widget"
    monkeypatch.setenv("REPOPILOT_HOME", str(repopilot_home))
    monkeypatch.setenv("GITHUB_TOKEN", "gho_cachetoken")
    commands = []

    async def fake_run_git(args, timeout, cwd=None):
        commands.append(args)
        return execute_node._ProcResult(
            128,
            "",
            (
                "fatal: unable to access "
                "'https://x-access-token:gho_cachetoken@github.com/acme/widget.git/'"
            ),
        )

    monkeypatch.setattr(execute_node, "_run_git_async", fake_run_git)
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        owner="acme",
        repo="widget",
    )

    try:
        await execute_node.git_clone(state)
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("git_clone should fail when all cache population fails")

    assert len(commands) == 3
    assert "gho_cachetoken" not in message
    assert "https://x-access-token:<redacted>@github.com/acme/widget.git" in message
    assert not cache_path.exists()


async def test_execute_fix_redacts_github_token_from_execution_error(monkeypatch):
    token = "gho_secret123"
    tokenized_url = f"https://x-access-token:{token}@github.com/acme/widget.git"

    async def fake_git_clone(state):
        raise subprocess.TimeoutExpired(
            cmd=[
                "git",
                "clone",
                "--depth",
                "1",
                tokenized_url,
                "/tmp/repopilot-acme-widget",
            ],
            timeout=180,
        )

    monkeypatch.setattr(execute_node, "git_clone", fake_git_clone)
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        owner="acme",
        repo="widget",
        patch_content="diff --git a/src/auth.py b/src/auth.py",
    )

    next_state = await execute_node.execute_fix(state)

    assert next_state.current_phase == new_agent.Phase.VERIFY
    assert len(next_state.fix_attempts) == 1
    assert next_state.fix_attempts[-1].test_result == "execution_error"
    assert next_state.fix_attempts[-1].failure_kind == "infra_error"
    assert token not in next_state.fix_attempts[-1].error_log
    assert tokenized_url not in next_state.fix_attempts[-1].error_log
    assert "https://x-access-token:<redacted>@github.com/acme/widget.git" in (
        next_state.fix_attempts[-1].error_log
    )


async def test_execute_fix_marks_clone_network_failure_as_infra_error(monkeypatch):
    async def fake_git_clone(state):
        raise RuntimeError(
            "fatal: unable to access 'https://github.com/acme/widget.git/': "
            "Failed to connect to github.com port 443"
        )

    monkeypatch.setattr(execute_node, "git_clone", fake_git_clone)
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        owner="acme",
        repo="widget",
        patch_content="diff --git a/src/auth.py b/src/auth.py",
    )

    next_state = await execute_node.execute_fix(state)

    assert next_state.current_phase == new_agent.Phase.VERIFY
    assert len(next_state.fix_attempts) == 1
    assert next_state.fix_attempts[-1].test_result == "execution_error"
    assert next_state.fix_attempts[-1].failure_kind == "infra_error"
    assert "Failed to connect to github.com port 443" in (
        next_state.fix_attempts[-1].error_log
    )


async def test_execute_fix_redacts_github_token_from_patch_apply_failure(monkeypatch):
    token = "gho_patchsecret"

    async def fake_apply_patch(repo_path, patch_content):
        return execute_node.PatchApplyResult(
            applied=False,
            output=(
                "fatal: unable to access "
                f"'https://x-access-token:{token}@github.com/acme/widget.git/'"
            ),
            patch_content=patch_content,
        )

    monkeypatch.setattr(execute_node, "apply_patch_with_repair", fake_apply_patch)
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        repo_path="/tmp/repopilot-test-repo",
        patch_content="malformed diff",
    )

    next_state = await execute_node.execute_fix(state)

    assert token not in next_state.fix_attempts[-1].error_log
    assert "https://x-access-token:<redacted>@github.com/acme/widget.git" in (
        next_state.fix_attempts[-1].error_log
    )


async def test_execute_fix_redacts_github_token_from_test_output(monkeypatch):
    token = "gho_testsecret"

    async def fake_apply_patch(repo_path, patch_content):
        return execute_node.PatchApplyResult(
            applied=True,
            output="",
            patch_content=patch_content,
        )

    async def fake_run_pytest(repo_path, command=None):
        return {
            "command": "pytest tests/test_auth.py -q",
            "returncode": 1,
            "stdout": (
                "failed cloning "
                f"https://x-access-token:{token}@github.com/acme/widget.git"
            ),
            "stderr": "",
            "success": False,
        }

    monkeypatch.setattr(execute_node, "apply_patch_with_repair", fake_apply_patch)
    monkeypatch.setattr(execute_node, "run_pytest", fake_run_pytest)
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        repo_path="/tmp/repopilot-test-repo",
        patch_content="diff --git a/src/auth.py b/src/auth.py",
        test_command="pytest tests/test_auth.py -q",
    )

    next_state = await execute_node.execute_fix(state)

    assert token not in next_state.fix_attempts[-1].error_log
    assert "https://x-access-token:<redacted>@github.com/acme/widget.git" in (
        next_state.fix_attempts[-1].error_log
    )


async def test_verify_fix_first_patch_apply_failure_does_not_increment_retry_count():
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        max_retries=1,
        retry_count=0,
        current_phase=new_agent.Phase.VERIFY,
        fix_attempts=[
            new_agent.FixAttempt(
                patch_content="malformed diff",
                test_result="patch_apply_failed",
                failure_kind="patch_apply_failed",
                error_log="error: No valid patches in input",
                success=False,
            )
        ],
    )

    next_state = await new_agent.verify_fix(state)

    assert next_state.current_phase == new_agent.Phase.REFLECT
    assert next_state.retry_count == 0


async def test_verify_fix_legacy_patch_apply_failure_does_not_increment_retry_count():
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        max_retries=1,
        retry_count=0,
        current_phase=new_agent.Phase.VERIFY,
        fix_attempts=[
            new_agent.FixAttempt(
                patch_content="malformed diff",
                test_result="patch_apply_failed",
                error_log="error: No valid patches in input",
                success=False,
            )
        ],
    )

    next_state = await new_agent.verify_fix(state)

    assert next_state.current_phase == new_agent.Phase.REFLECT
    assert next_state.retry_count == 0


async def test_verify_fix_second_consecutive_patch_apply_failure_consumes_retry():
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        max_retries=1,
        retry_count=0,
        current_phase=new_agent.Phase.VERIFY,
        fix_attempts=[
            new_agent.FixAttempt(
                patch_content="malformed diff",
                test_result="patch_apply_failed",
                failure_kind="patch_apply_failed",
                error_log="error: No valid patches in input",
                success=False,
            ),
            new_agent.FixAttempt(
                patch_content="still malformed diff",
                test_result="patch_apply_failed",
                failure_kind="patch_apply_failed",
                error_log="error: corrupt patch at line 3",
                success=False,
            ),
        ],
    )

    next_state = await new_agent.verify_fix(state)

    assert next_state.current_phase == new_agent.Phase.REFLECT
    assert next_state.retry_count == 1


async def test_verify_fix_same_patch_apply_failure_twice_routes_to_failure():
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        max_retries=2,
        retry_count=0,
        current_phase=new_agent.Phase.VERIFY,
        fix_attempts=[
            new_agent.FixAttempt(
                patch_content="malformed diff",
                test_result="patch_apply_failed",
                failure_kind="patch_apply_failed",
                error_log="error: corrupt patch at line 3",
                success=False,
            ),
            new_agent.FixAttempt(
                patch_content="malformed diff",
                test_result="patch_apply_failed",
                failure_kind="patch_apply_failed",
                error_log="error: corrupt patch at line 3",
                success=False,
            ),
        ],
    )

    next_state = await new_agent.verify_fix(state)

    assert next_state.current_phase == new_agent.Phase.FAILURE
    assert next_state.failure_reason == "Same patch produced the same failure twice."
    assert next_state.retry_count == 0


async def test_verify_fix_test_failure_still_increments_retry_count():
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        max_retries=1,
        retry_count=0,
        current_phase=new_agent.Phase.VERIFY,
        fix_attempts=[
            new_agent.FixAttempt(
                patch_content="diff --git a/src/auth.py b/src/auth.py",
                file_path="src/auth.py",
                test_result="failed",
                failure_kind="test_failed",
                error_log="assert False",
                success=False,
            )
        ],
    )

    next_state = await new_agent.verify_fix(state)

    assert next_state.current_phase == new_agent.Phase.REFLECT
    assert next_state.retry_count == 1


async def test_verify_fix_infra_error_routes_to_failure_without_retry():
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        max_retries=1,
        retry_count=0,
        current_phase=new_agent.Phase.VERIFY,
        fix_attempts=[
            new_agent.FixAttempt(
                patch_content="diff --git a/src/auth.py b/src/auth.py",
                file_path="src/auth.py",
                test_result="execution_error",
                failure_kind="infra_error",
                error_log="fatal: unable to access github.com",
                success=False,
            )
        ],
    )

    next_state = await new_agent.verify_fix(state)

    assert next_state.current_phase == new_agent.Phase.FAILURE
    assert next_state.retry_count == 0
    assert next_state.failure_reason == (
        "Infrastructure error during execution: fatal: unable to access github.com"
    )
