import json
import logging

import pytest

from src import new_agent
from src.nodes import plan as plan_node
from src.nodes import reflect as reflect_node


async def test_plan_fix_records_plan_decision_frame(monkeypatch):
    calls = []

    async def fake_llm_call(system, user):
        calls.append({"system": system, "user": user})
        return json.dumps(
            {
                "plan": "Patch auth submit handling.",
                "patch": "diff --git a/src/auth.py b/src/auth.py\n--- a/src/auth.py\n+++ b/src/auth.py\n",
                "files": ["src/auth.py"],
                "test_command": "pytest tests/test_auth.py -q",
                "decision_frame": {
                    "stage": "plan",
                    "summary": "Patch auth submit handling.",
                    "recommended_action": "execute",
                    "hypotheses": [
                        {
                            "id": "H1",
                            "claim": "Auth submit path mishandles missing user input.",
                            "evidence": ["Issue mentions a crash after submit."],
                            "score": 0.84,
                        }
                    ],
                    "selected_hypothesis_id": "H1",
                    "next_checks": ["Run the auth regression test."],
                    "risk": "medium",
                    "confidence": 0.84,
                },
            }
        )

    monkeypatch.setattr(plan_node, "llm_call", fake_llm_call)
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        issue_title="Login crash",
        issue_body="Crashes after submit.",
    )

    next_state = await plan_node.plan_fix(state)

    assert next_state.current_phase == new_agent.Phase.EXECUTE
    assert next_state.decision_frame is not None
    assert next_state.decision_frame.stage == "plan"
    assert next_state.decision_frame.recommended_action == "execute"
    assert next_state.decision_frame.selected_hypothesis_id == "H1"
    assert next_state.frame_history[-1] == next_state.decision_frame
    for key in [
        "decision_frame",
        "stage",
        "recommended_action",
        "ask_user",
        "collect_more_context",
    ]:
        assert key in calls[0]["system"]


async def test_plan_fix_no_patch_execute_recommendation_routes_to_failure(monkeypatch):
    async def fake_llm_call(system, user):
        return json.dumps(
            {
                "plan": "Patch is required but missing.",
                "patch": "",
                "files": [],
                "test_command": "",
                "decision_frame": {
                    "stage": "plan",
                    "summary": "Patch is required but missing.",
                    "recommended_action": "execute",
                    "next_checks": ["Run the missing regression test."],
                    "risk": "medium",
                    "confidence": 0.51,
                },
            }
        )

    monkeypatch.setattr(plan_node, "llm_call", fake_llm_call)
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        issue_title="Login crash",
        issue_body="Crashes after submit.",
        current_phase=new_agent.Phase.PLAN,
    )

    planned_state = await plan_node.plan_fix(state)
    route = new_agent.route_from_state(planned_state)

    assert planned_state.current_phase == new_agent.Phase.FAILURE
    assert planned_state.failure_reason == "Planner did not produce a patch."
    assert planned_state.decision_frame.recommended_action == "stop"
    assert route == "handle_failure"


async def test_plan_fix_collect_more_context_without_patch_routes_to_locate(monkeypatch):
    async def fake_llm_call(system, user):
        return json.dumps(
            {
                "plan": "Need to inspect routing middleware before patching.",
                "patch": "",
                "files": [],
                "test_command": "",
                "decision_frame": {
                    "stage": "plan",
                    "summary": "Need to inspect routing middleware before patching.",
                    "recommended_action": "collect_more_context",
                    "next_checks": ["Search for the request router middleware."],
                    "risk": "unknown",
                    "confidence": 0.55,
                },
            }
        )

    monkeypatch.setattr(plan_node, "llm_call", fake_llm_call)
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        issue_title="Login crash",
        issue_body="Crashes after submit.",
        current_phase=new_agent.Phase.PLAN,
    )

    planned_state = await plan_node.plan_fix(state)
    route = new_agent.route_from_state(planned_state)

    assert route == "locate_code"
    assert planned_state.current_phase == new_agent.Phase.PLAN
    assert planned_state.failure_reason == ""
    assert planned_state.decision_frame.recommended_action == "collect_more_context"
    assert planned_state.decision_warnings[0]["expected_phase"] == "LOCATE"
    assert planned_state.decision_warnings[0]["actual_phase"] == "PLAN"


async def test_plan_fix_ask_user_preserves_plan_phase_for_router(monkeypatch):
    async def fake_llm_call(system, user):
        return json.dumps(
            {
                "plan": "Need product confirmation before patching.",
                "patch": "",
                "files": [],
                "test_command": "",
                "decision_frame": {
                    "stage": "plan",
                    "summary": "Need product confirmation before patching.",
                    "recommended_action": "ask_user",
                    "next_checks": [
                        "Confirm whether a breaking API response change is allowed."
                    ],
                    "risk": "high",
                    "confidence": 0.61,
                },
            }
        )

    monkeypatch.setattr(plan_node, "llm_call", fake_llm_call)
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        issue_title="Login crash",
        issue_body="Crashes after submit.",
        current_phase=new_agent.Phase.PLAN,
    )

    planned_state = await plan_node.plan_fix(state)
    route = new_agent.route_from_state(planned_state)

    assert route == new_agent.END
    assert planned_state.current_phase == new_agent.Phase.WAITING_FOR_USER
    assert planned_state.failure_reason == ""
    assert planned_state.decision_warnings[0]["recommended_action"] == "ask_user"
    assert planned_state.decision_warnings[0]["expected_phase"] == "WAITING_FOR_USER"
    assert planned_state.decision_warnings[0]["actual_phase"] == "PLAN"


async def test_reflect_on_failure_records_reflect_decision_frame(monkeypatch):
    calls = []

    async def fake_llm_call(system, user):
        calls.append({"system": system, "user": user})
        return json.dumps(
            {
                "root_cause": "The patch changed the wrong branch.",
                "what_went_wrong": "It ignored the failing None case.",
                "suggested_fix_approach": "Patch the None guard before submit.",
                "files_that_also_need_changes": ["src/auth.py"],
                "decision_frame": {
                    "stage": "reflect",
                    "summary": "The patch changed the wrong branch.",
                    "recommended_action": "plan",
                    "hypotheses": [
                        {
                            "id": "H1",
                            "claim": "Previous patch targeted the wrong condition.",
                            "evidence": ["Test output still fails on None input."],
                            "score": 0.9,
                        }
                    ],
                    "selected_hypothesis_id": "H1",
                    "next_checks": ["Re-run the failing auth test."],
                    "risk": "low",
                    "confidence": 0.9,
                },
            }
        )

    monkeypatch.setattr(reflect_node, "llm_call", fake_llm_call)
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        issue_title="Login crash",
        issue_body="Crashes after submit.",
    )
    state.fix_attempts.append(
        new_agent.FixAttempt(
            patch_content="diff --git a/src/auth.py b/src/auth.py",
            file_path="src/auth.py",
            test_result="failed",
            error_log="assert user is not None",
            success=False,
        )
    )

    next_state = await reflect_node.reflect_on_failure(state)

    assert next_state.current_phase == new_agent.Phase.PLAN
    assert next_state.decision_frame is not None
    assert next_state.decision_frame.stage == "reflect"
    assert next_state.decision_frame.recommended_action == "plan"
    assert next_state.decision_frame.selected_hypothesis_id == "H1"
    assert next_state.frame_history[-1] == next_state.decision_frame
    for key in [
        "decision_frame",
        "stage",
        "recommended_action",
    ]:
        assert key in calls[0]["system"]


def test_agent_payload_exposes_decision_frame_history():
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
    )
    frame = new_agent.DecisionFrame(
        stage="plan",
        summary="Patch auth submit handling.",
        recommended_action="execute",
        confidence=0.82,
        risk="medium",
    )
    new_agent._record_decision_frame(state, frame)

    payload = new_agent.agent_payload_from_state(state, turns_taken=0)

    assert payload["decision_frame"]["stage"] == "plan"
    assert payload["decision_frame"]["recommended_action"] == "execute"
    assert payload["frame_history"][0]["frame_id"] == "df_0001"


def test_save_trace_writes_frame_history(tmp_path):
    tracer = new_agent.Tracer()
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        trace_id=tracer.trace_id,
    )
    frame = new_agent.DecisionFrame(
        stage="plan",
        summary="Patch auth submit handling.",
        recommended_action="execute",
        confidence=0.82,
        risk="medium",
    )
    new_agent._record_decision_frame(state, frame)
    tracer.log("agent_v2_done", {"issue_url": state.issue_url}, {"phase": "DONE"})

    trace_path = tmp_path / "trace.json"
    new_agent._save_trace(tracer, trace_path, state)

    data = json.loads(trace_path.read_text(encoding="utf-8"))
    assert data["trace_id"] == tracer.trace_id
    assert data["steps"][0]["step"] == "agent_v2_done"
    assert data["frame_history"][0]["frame_id"] == "df_0001"
    assert data["frame_history"][0]["stage"] == "plan"
    assert data["frame_history"][0]["recommended_action"] == "execute"


def test_save_trace_writes_decision_warnings(tmp_path, caplog):
    caplog.set_level(logging.WARNING, logger="repopilot.graph")
    tracer = new_agent.Tracer()
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        current_phase=new_agent.Phase.PLAN,
        trace_id=tracer.trace_id,
    )
    frame = new_agent.DecisionFrame(
        stage="plan",
        summary="Ready to execute the patch.",
        recommended_action="execute",
        confidence=0.82,
        risk="medium",
    )
    new_agent._record_decision_frame(state, frame)
    new_agent.route_from_state(state)

    trace_path = tmp_path / "trace.json"
    new_agent._save_trace(tracer, trace_path, state)

    data = json.loads(trace_path.read_text(encoding="utf-8"))
    assert data["decision_warnings"][0]["frame_id"] == "df_0001"
    assert data["decision_warnings"][0]["recommended_action"] == "execute"
    assert data["decision_warnings"][0]["actual_phase"] == "PLAN"


def test_save_trace_writes_route_decisions(tmp_path):
    tracer = new_agent.Tracer()
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        current_phase=new_agent.Phase.PLAN,
        trace_id=tracer.trace_id,
    )

    route = new_agent.route_from_state(state)
    trace_path = tmp_path / "trace.json"
    new_agent._save_trace(tracer, trace_path, state)

    data = json.loads(trace_path.read_text(encoding="utf-8"))
    assert route == "plan_fix"
    assert data["route_decisions"] == [
        {
            "source": "current_phase",
            "current_phase": "PLAN",
            "selected_phase": "PLAN",
            "route": "plan_fix",
            "fallback_reason": "no_decision_frame",
        }
    ]


def test_save_trace_writes_human_input_request(tmp_path):
    tracer = new_agent.Tracer()
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        trace_id=tracer.trace_id,
        current_phase=new_agent.Phase.WAITING_FOR_USER,
        pending_human_input=True,
        human_input_request={
            "frame_id": "df_0001",
            "stage": "plan",
            "question": "Confirm whether a breaking API response change is allowed.",
            "summary": "Need product decision about API compatibility.",
            "risk": "high",
            "confidence": 0.61,
        },
    )

    trace_path = tmp_path / "trace.json"
    new_agent._save_trace(tracer, trace_path, state)

    data = json.loads(trace_path.read_text(encoding="utf-8"))
    assert data["pending_human_input"] is True
    assert data["human_input_request"]["question"] == (
        "Confirm whether a breaking API response change is allowed."
    )


def test_route_from_state_records_recommended_action_mismatch_warning(caplog):
    caplog.set_level(logging.WARNING, logger="repopilot.graph")
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        current_phase=new_agent.Phase.PLAN,
    )
    frame = new_agent.DecisionFrame(
        stage="plan",
        summary="Ready to execute the patch.",
        recommended_action="execute",
        confidence=0.82,
        risk="medium",
    )
    new_agent._record_decision_frame(state, frame)

    route = new_agent.route_from_state(state)

    assert route == "execute_fix"
    assert state.decision_warnings == [
        {
            "frame_id": "df_0001",
            "stage": "plan",
            "recommended_action": "execute",
            "expected_phase": "EXECUTE",
            "actual_phase": "PLAN",
            "message": (
                "DecisionFrame recommended_action 'execute' expected phase "
                "EXECUTE but current_phase is PLAN"
            ),
        }
    ]
    assert state.decision_route_checked_frame_id == "df_0001"
    assert "recommended_action 'execute' expected phase EXECUTE" in caplog.text
    assert state.route_decisions[-1] == {
        "source": "decision_frame",
        "current_phase": "PLAN",
        "selected_phase": "EXECUTE",
        "route": "execute_fix",
        "frame_id": "df_0001",
        "recommended_action": "execute",
    }


def test_route_from_state_skips_warning_for_aligned_recommended_action(caplog):
    caplog.set_level(logging.WARNING, logger="repopilot.graph")
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        current_phase=new_agent.Phase.EXECUTE,
    )
    frame = new_agent.DecisionFrame(
        stage="plan",
        summary="Ready to execute the patch.",
        recommended_action="execute",
        confidence=0.82,
        risk="medium",
    )
    new_agent._record_decision_frame(state, frame)

    route = new_agent.route_from_state(state)

    assert route == "execute_fix"
    assert state.decision_warnings == []
    assert state.decision_route_checked_frame_id == "df_0001"
    assert caplog.text == ""
    assert state.route_decisions[-1]["source"] == "decision_frame"
    assert state.route_decisions[-1]["selected_phase"] == "EXECUTE"
    assert state.route_decisions[-1]["route"] == "execute_fix"


def test_route_from_state_consumes_each_decision_frame_once(caplog):
    caplog.set_level(logging.WARNING, logger="repopilot.graph")
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        current_phase=new_agent.Phase.PLAN,
    )
    frame = new_agent.DecisionFrame(
        stage="plan",
        summary="Ready to execute the patch.",
        recommended_action="execute",
        confidence=0.82,
        risk="medium",
    )
    new_agent._record_decision_frame(state, frame)

    first_route = new_agent.route_from_state(state)
    second_route = new_agent.route_from_state(state)

    assert first_route == "execute_fix"
    assert second_route == "plan_fix"
    assert len(state.decision_warnings) == 1
    assert caplog.text.count("recommended_action 'execute' expected phase EXECUTE") == 1
    assert state.route_decisions[0]["source"] == "decision_frame"
    assert state.route_decisions[1] == {
        "source": "current_phase",
        "current_phase": "PLAN",
        "selected_phase": "PLAN",
        "route": "plan_fix",
        "frame_id": "df_0001",
        "recommended_action": "execute",
        "fallback_reason": "already_consumed",
    }


@pytest.mark.parametrize(
    ("recommended_action", "current_phase", "expected_route", "expected_phase"),
    [
        ("plan", new_agent.Phase.REFLECT, "plan_fix", "PLAN"),
        ("reflect", new_agent.Phase.VERIFY, "reflect_on_failure", "REFLECT"),
        ("stop", new_agent.Phase.PLAN, "handle_failure", "FAILURE"),
    ],
)
def test_route_from_state_consumes_supported_recommended_actions(
    caplog,
    recommended_action,
    current_phase,
    expected_route,
    expected_phase,
):
    caplog.set_level(logging.WARNING, logger="repopilot.graph")
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        current_phase=current_phase,
    )
    frame = new_agent.DecisionFrame(
        stage="reflect" if recommended_action == "plan" else "plan",
        summary=f"Recommend {recommended_action}.",
        recommended_action=recommended_action,
        confidence=0.82,
        risk="medium",
    )
    new_agent._record_decision_frame(state, frame)

    route = new_agent.route_from_state(state)

    assert route == expected_route
    assert state.decision_warnings[0]["recommended_action"] == recommended_action
    assert state.decision_warnings[0]["expected_phase"] == expected_phase
    assert state.decision_route_checked_frame_id == "df_0001"


def test_route_from_state_consumes_collect_more_context_recommendation(caplog):
    caplog.set_level(logging.WARNING, logger="repopilot.graph")
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        current_phase=new_agent.Phase.PLAN,
    )
    frame = new_agent.DecisionFrame(
        stage="plan",
        summary="Need broader code context before patching.",
        recommended_action="collect_more_context",
        confidence=0.52,
        risk="unknown",
    )
    new_agent._record_decision_frame(state, frame)

    route = new_agent.route_from_state(state)

    assert route == "locate_code"
    assert state.decision_route_checked_frame_id == "df_0001"
    assert state.decision_warnings[0]["recommended_action"] == "collect_more_context"
    assert state.decision_warnings[0]["expected_phase"] == "LOCATE"
    assert state.route_decisions[-1] == {
        "source": "decision_frame",
        "current_phase": "PLAN",
        "selected_phase": "LOCATE",
        "route": "locate_code",
        "frame_id": "df_0001",
        "recommended_action": "collect_more_context",
    }


def test_route_from_state_consumes_ask_user_as_human_input_pause(caplog):
    caplog.set_level(logging.WARNING, logger="repopilot.graph")
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        current_phase=new_agent.Phase.PLAN,
    )
    frame = new_agent.DecisionFrame(
        stage="plan",
        summary="Need product decision about API compatibility.",
        recommended_action="ask_user",
        next_checks=["Confirm whether a breaking API response change is allowed."],
        confidence=0.61,
        risk="high",
    )
    new_agent._record_decision_frame(state, frame)

    route = new_agent.route_from_state(state)

    assert route == new_agent.END
    assert state.current_phase == new_agent.Phase.WAITING_FOR_USER
    assert state.decision_route_checked_frame_id == "df_0001"
    assert state.pending_human_input is True
    assert state.human_input_request == {
        "frame_id": "df_0001",
        "stage": "plan",
        "question": "Confirm whether a breaking API response change is allowed.",
        "summary": "Need product decision about API compatibility.",
        "risk": "high",
        "confidence": 0.61,
    }
    assert state.decision_warnings[0]["recommended_action"] == "ask_user"
    assert state.decision_warnings[0]["expected_phase"] == "WAITING_FOR_USER"
    assert state.decision_warnings[0]["actual_phase"] == "PLAN"
    assert state.route_decisions[-1] == {
        "source": "decision_frame",
        "current_phase": "WAITING_FOR_USER",
        "selected_phase": "WAITING_FOR_USER",
        "route": new_agent.END,
        "frame_id": "df_0001",
        "recommended_action": "ask_user",
    }


def test_route_from_state_uses_summary_as_human_input_question_when_no_next_checks():
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        current_phase=new_agent.Phase.PLAN,
    )
    frame = new_agent.DecisionFrame(
        stage="plan",
        summary="Need deployment environment details.",
        recommended_action="ask_user",
        next_checks=[],
        confidence=0.44,
        risk="unknown",
    )
    new_agent._record_decision_frame(state, frame)

    route = new_agent.route_from_state(state)

    assert route == new_agent.END
    assert state.human_input_request["question"] == "Need deployment environment details."


def test_route_from_state_uses_summary_as_human_input_question_when_first_check_blank():
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        current_phase=new_agent.Phase.PLAN,
    )
    frame = new_agent.DecisionFrame(
        stage="plan",
        summary="Need deployment environment details.",
        recommended_action="ask_user",
        next_checks=[""],
        confidence=0.44,
        risk="unknown",
    )
    new_agent._record_decision_frame(state, frame)

    route = new_agent.route_from_state(state)

    assert route == new_agent.END
    assert state.human_input_request["question"] == "Need deployment environment details."


def test_route_from_state_falls_back_for_unsupported_recommended_action(caplog):
    caplog.set_level(logging.WARNING, logger="repopilot.graph")
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        current_phase=new_agent.Phase.PLAN,
    )
    frame = new_agent.DecisionFrame.model_construct(
        frame_id="",
        stage="plan",
        summary="Need an unknown future action before routing.",
        hypotheses=[],
        selected_hypothesis_id=None,
        evidence=[],
        next_checks=[],
        recommended_action="future_action",
        confidence=0.42,
        risk="unknown",
        parent_frame_id=None,
        trace_notes="",
    )
    new_agent._record_decision_frame(state, frame)

    route = new_agent.route_from_state(state)

    assert route == "plan_fix"
    assert state.decision_warnings == []
    assert state.decision_route_checked_frame_id == ""
    assert caplog.text == ""
    assert state.route_decisions[-1] == {
        "source": "current_phase",
        "current_phase": "PLAN",
        "selected_phase": "PLAN",
        "route": "plan_fix",
        "frame_id": "df_0001",
        "recommended_action": "future_action",
        "fallback_reason": "unsupported_recommended_action",
    }


def test_route_from_state_falls_back_when_decision_frame_has_no_id(caplog):
    caplog.set_level(logging.WARNING, logger="repopilot.graph")
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        current_phase=new_agent.Phase.PLAN,
        decision_frame=new_agent.DecisionFrame(
            stage="plan",
            summary="Ready to execute the patch.",
            recommended_action="execute",
            confidence=0.82,
            risk="medium",
        ),
    )

    route = new_agent.route_from_state(state)

    assert route == "plan_fix"
    assert state.decision_warnings == []
    assert state.decision_route_checked_frame_id == ""
    assert caplog.text == ""
    assert state.route_decisions[-1]["fallback_reason"] == "no_frame_id"


def test_route_from_state_falls_back_for_stale_decision_frame(caplog):
    caplog.set_level(logging.WARNING, logger="repopilot.graph")
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        current_phase=new_agent.Phase.PLAN,
    )
    old_frame = new_agent.DecisionFrame(
        stage="plan",
        summary="Old execute recommendation.",
        recommended_action="execute",
        confidence=0.82,
        risk="medium",
    )
    new_frame = new_agent.DecisionFrame(
        stage="reflect",
        summary="Newer reflect recommendation.",
        recommended_action="plan",
        confidence=0.76,
        risk="low",
    )
    new_agent._record_decision_frame(state, old_frame)
    new_agent._record_decision_frame(state, new_frame)
    state.decision_frame = old_frame

    route = new_agent.route_from_state(state)

    assert route == "plan_fix"
    assert state.decision_warnings == []
    assert state.decision_route_checked_frame_id == ""
    assert caplog.text == ""
    assert state.route_decisions[-1]["fallback_reason"] == "stale_frame"


def test_agent_payload_exposes_route_decisions():
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        current_phase=new_agent.Phase.PLAN,
    )
    new_agent.route_from_state(state)

    payload = new_agent.agent_payload_from_state(state, turns_taken=0)

    assert payload["route_decisions"][0]["source"] == "current_phase"
    assert payload["route_decisions"][0]["route"] == "plan_fix"


def test_agent_payload_exposes_human_input_pause():
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        current_phase=new_agent.Phase.WAITING_FOR_USER,
        pending_human_input=True,
        human_input_request={
            "frame_id": "df_0001",
            "stage": "plan",
            "question": "Confirm whether a breaking API response change is allowed.",
            "summary": "Need product decision about API compatibility.",
            "risk": "high",
            "confidence": 0.61,
        },
    )

    payload = new_agent.agent_payload_from_state(state, turns_taken=0)

    assert payload["done"] is False
    assert payload["success"] is False
    assert payload["waiting_for_user"] is True
    assert payload["final_phase"] == "WAITING_FOR_USER"
    assert payload["human_input_request"]["question"] == (
        "Confirm whether a breaking API response change is allowed."
    )
