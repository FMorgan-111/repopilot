import functools

import src.run_store as run_store
from src import new_agent


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
