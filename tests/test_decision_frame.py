import json

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
        "hypotheses",
        "selected_hypothesis_id",
        "next_checks",
        "risk",
        "confidence",
    ]:
        assert key in calls[0]["system"]


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
        "hypotheses",
        "selected_hypothesis_id",
        "next_checks",
        "risk",
        "confidence",
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
