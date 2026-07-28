import asyncio
import json
import logging
import subprocess

import pytest

from src import new_agent
from src.async_safety import CancellationDrainError
from src.nodes import plan as plan_node
from src.nodes import reflect as reflect_node


def _exact_plan_response(*, decision_frame=None):
    return {
        "kind": "plan",
        "plan": "Return the new sentinel.",
        "patch": "",
        "patch_edits": [
            {
                "file": "src/widget.py",
                "search": "return 'old-sentinel'",
                "replace": "return 'new-sentinel'",
            }
        ],
        "files": ["src/widget.py"],
        "test_command": "pytest tests/test_widget.py -q",
        "decision_frame": decision_frame
        or {
            "stage": "plan",
            "summary": "Update the sentinel.",
            "recommended_action": "execute",
            "risk": "low",
            "confidence": 0.9,
        },
    }


async def test_plan_fix_records_plan_decision_frame(exact_repair_state, monkeypatch):
    calls = []

    async def fake_llm_call(system, user, model=None, *, provider="primary", **_kwargs):
        calls.append({"system": system, "user": user})
        return {
            "patch_edits": [
                {
                    "file_path": "src/widget.py",
                    "search": "return 'old-sentinel'",
                    "replace": "return 'new-sentinel'",
                }
            ]
        }

    monkeypatch.setattr(plan_node, "llm_call", fake_llm_call)
    next_state = await plan_node.plan_fix(exact_repair_state)

    assert next_state.current_phase == new_agent.Phase.EXECUTE
    assert next_state.decision_frame is not None
    assert next_state.decision_frame.stage == "plan"
    assert next_state.decision_frame.recommended_action == "execute"
    assert next_state.decision_frame.selected_hypothesis_id is None
    assert next_state.frame_history[-1] == next_state.decision_frame
    [invocation] = next_state.model_history
    assert next_state.token_usage == (
        invocation.input_tokens + invocation.output_tokens
    )
    for field in ("patch_edits", "file_path", "search", "replace"):
        assert field in calls[0]["system"]
    for field in (
        "decision_frame",
        "recommended_action",
        "confidence",
        "risk",
        "test_command",
    ):
        assert field not in calls[0]["system"]


async def test_plan_tool_cancellation_drain_is_not_model_error(monkeypatch):
    cancellation = asyncio.CancelledError("cancel plan tool")
    cleanup_error = RuntimeError("plan tool cleanup failed")
    sentinel = CancellationDrainError("plan tool", cancellation, cleanup_error)

    async def model_selects_tool(*_args, **_kwargs):
        return {
            "kind": "tool",
            "tool_intent": {
                "action": "search_text",
                "args": {"text": "submit"},
                "reason": "find the failing path",
                "expected_evidence": "the submit implementation",
            },
        }

    async def cancelled_tool(*_args, **_kwargs):
        raise sentinel

    monkeypatch.setattr(plan_node, "llm_call", model_selects_tool)
    monkeypatch.setattr(plan_node, "route_tool_intent", cancelled_tool)
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        issue_title="Login crash",
        issue_body="Crashes after submit.",
        current_phase=new_agent.Phase.PLAN,
    )

    with pytest.raises(CancellationDrainError) as raised:
        await plan_node.plan_fix(state)

    assert raised.value is sentinel
    assert not any(item.status == "error" for item in state.model_history)


async def test_reflect_summarizes_only_after_recording_valid_frame(monkeypatch):
    async def fake_llm_call(system, user, model=None, *, provider="primary", **_kwargs):
        return {
            "root_cause": "The guard ran after submit.",
            "what_went_wrong": "The unsafe call still executed.",
            "suggested_fix_approach": "Move the guard before submit.",
            "files_that_also_need_changes": [],
            "decision_frame": {
                "stage": "reflect",
                "summary": "Move the guard before submit.",
                "recommended_action": "plan",
                "risk": "low",
                "confidence": 0.9,
            },
        }

    calls = []

    async def fake_summarize(state, **kwargs):
        assert state.current_phase == new_agent.Phase.REFLECT
        assert state.decision_frame is not None
        assert state.decision_frame.stage == "reflect"
        assert state.frame_history[-1] == state.decision_frame
        calls.append(state.decision_frame.frame_id)
        return "attempt outcome"

    monkeypatch.setattr(reflect_node, "llm_call", fake_llm_call)
    monkeypatch.setattr(reflect_node, "summarize_attempt_outcome", fake_summarize)
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        issue_title="Login crash",
        issue_body="Crashes after submit.",
        current_phase=new_agent.Phase.REFLECT,
        fix_attempts=[
            new_agent.FixAttempt(
                test_result="failed",
                failure_kind="assertion_failure",
                error_log="assert submit was not called",
            )
        ],
    )

    next_state = await reflect_node.reflect_on_failure(state)

    assert calls == [next_state.decision_frame.frame_id]
    assert next_state.attempt_outcome_summary == "attempt outcome"
    assert next_state.current_phase == new_agent.Phase.PLAN
    [invocation] = next_state.model_history
    assert invocation.node == "reflect_on_failure"
    assert next_state.token_usage == (
        invocation.input_tokens + invocation.output_tokens
    )


async def test_reflect_error_replaces_outcome_summary_exactly_once(monkeypatch):
    monkeypatch.delenv("REPOPILOT_ESCALATION_ENABLED", raising=False)
    monkeypatch.delenv("LLM_ESCALATION_API_KEY", raising=False)

    async def fake_llm_call(system, user, model=None, *, provider="primary", **_kwargs):
        raise RuntimeError("reflection provider unavailable")

    calls = []

    async def fake_summarize(state, **kwargs):
        assert state.current_phase == new_agent.Phase.REFLECT
        assert state.reflection_notes == "Reflection failed: RuntimeError"
        calls.append(state.attempt_outcome_summary)
        return "replacement outcome after reflection error"

    monkeypatch.setattr(reflect_node, "llm_call", fake_llm_call)
    monkeypatch.setattr(reflect_node, "summarize_attempt_outcome", fake_summarize)
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        current_phase=new_agent.Phase.REFLECT,
        attempt_outcome_summary="stale outcome",
        fix_attempts=[
            new_agent.FixAttempt(
                test_result="failed",
                failure_kind="assertion_failure",
                error_log="assert submit was not called",
            )
        ],
    )

    next_state = await reflect_node.reflect_on_failure(state)

    assert calls == ["stale outcome"]
    assert next_state.attempt_outcome_summary == (
        "replacement outcome after reflection error"
    )
    assert next_state.current_phase == new_agent.Phase.PLAN


async def test_reflect_tool_cancellation_drain_is_not_model_error(monkeypatch):
    cancellation = asyncio.CancelledError("cancel reflect tool")
    cleanup_error = RuntimeError("reflect tool cleanup failed")
    sentinel = CancellationDrainError("reflect tool", cancellation, cleanup_error)

    async def model_selects_tool(*_args, **_kwargs):
        return {
            "kind": "tool",
            "tool_intent": {
                "action": "search_text",
                "args": {"text": "submit"},
                "reason": "find the failed path",
                "expected_evidence": "the submit implementation",
            },
        }

    async def cancelled_tool(*_args, **_kwargs):
        raise sentinel

    monkeypatch.setattr(reflect_node, "llm_call", model_selects_tool)
    monkeypatch.setattr(reflect_node, "route_tool_intent", cancelled_tool)
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        issue_title="Login crash",
        issue_body="Crashes after submit.",
        current_phase=new_agent.Phase.REFLECT,
        fix_attempts=[
            new_agent.FixAttempt(
                test_result="failed",
                failure_kind="assertion_failure",
                error_log="assert submit was not called",
            )
        ],
    )

    with pytest.raises(CancellationDrainError) as raised:
        await reflect_node.reflect_on_failure(state)

    assert raised.value is sentinel
    assert not any(item.status == "error" for item in state.model_history)


async def test_plan_fix_records_search_replace_patch_edits(
    exact_repair_state, monkeypatch
):
    calls = []

    async def fake_llm_call(system, user, model=None, *, provider="primary", **_kwargs):
        calls.append({"system": system, "user": user})
        return _exact_plan_response()

    monkeypatch.setattr(plan_node, "llm_call", fake_llm_call)
    next_state = await plan_node.plan_fix(exact_repair_state)

    assert next_state.current_phase == new_agent.Phase.EXECUTE
    assert next_state.patch_content.startswith("diff --git a/src/widget.py")
    assert next_state.patch_edits[0].file_path == "src/widget.py"
    assert next_state.patch_edits[0].search == "return 'old-sentinel'"
    assert "patch_edits" in calls[0]["system"]
    assert "search" in calls[0]["system"]
    assert "replace" in calls[0]["system"]


async def test_patch_only_plan_stops_after_preflight_without_mutating_checkout(
    exact_repair_state, monkeypatch
):
    async def fake_llm_call(*_args, **_kwargs):
        return _exact_plan_response()

    exact_repair_state.patch_only = True
    monkeypatch.setattr(plan_node, "llm_call", fake_llm_call)

    next_state = await plan_node.plan_fix(exact_repair_state)

    assert next_state.current_phase == new_agent.Phase.DONE
    assert next_state.tool_patch_approval is not None
    assert next_state.patch_content.startswith("diff --git a/src/widget.py")
    assert new_agent.route_from_state(next_state) == new_agent.END
    checkout_diff = subprocess.run(
        ["git", "-C", next_state.repo_path, "diff", "--exit-code"],
        check=False,
        capture_output=True,
    )
    assert checkout_diff.returncode == 0


async def test_plan_fix_records_llm_diagnostic_on_timeout(monkeypatch):
    async def fake_llm_call(system, user, model=None, *, provider="primary", **_kwargs):
        raise asyncio.TimeoutError()

    monkeypatch.setattr(plan_node, "llm_call", fake_llm_call)
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        issue_title="Login crash",
        issue_body="Crashes after submit.",
        current_phase=new_agent.Phase.PLAN,
    )

    next_state = await plan_node.plan_fix(state)

    assert next_state.current_phase == new_agent.Phase.FAILURE
    assert next_state.failure_reason == "The planning model call failed."
    diagnostic = next(
        item for item in next_state.node_diagnostics if item.get("event") == "llm_call"
    )
    assert diagnostic["node"] == "plan_fix"
    assert diagnostic["status"] == "error"
    assert diagnostic["error_type"] == "TimeoutError"
    assert diagnostic["prompt_tokens_estimate"] > 0


async def test_plan_fix_records_successful_llm_diagnostic(
    exact_repair_state, monkeypatch
):
    async def fake_llm_call(system, user, model=None, *, provider="primary", **_kwargs):
        return _exact_plan_response()

    monkeypatch.setattr(plan_node, "llm_call", fake_llm_call)
    next_state = await plan_node.plan_fix(exact_repair_state)

    assert next_state.current_phase == new_agent.Phase.EXECUTE
    assert next_state.node_diagnostics[-1]["node"] == "plan_fix"
    assert next_state.node_diagnostics[-1]["event"] == "llm_call"
    assert next_state.node_diagnostics[-1]["status"] == "success"
    assert next_state.node_diagnostics[-1]["prompt_tokens_estimate"] > 0
    assert next_state.node_diagnostics[-1]["response_tokens_estimate"] > 0


async def test_invalid_structured_plan_counts_one_failed_transaction(
    exact_repair_state, monkeypatch
):
    calls = []

    async def fake_llm_call(system, user, model=None, *, provider="primary", **_kwargs):
        calls.append((system, user, model, provider))
        return {
            "plan": "invalid frame stage",
            "decision_frame": {
                "stage": "reflect",
                "summary": "This cannot validate as a plan.",
            },
        }

    monkeypatch.setattr(plan_node, "llm_call", fake_llm_call)

    next_state = await plan_node.plan_fix(exact_repair_state)

    assert len(calls) == 1
    assert next_state.current_phase == new_agent.Phase.PLAN
    assert next_state.retry_count == 1
    assert next_state.primary_failed_repair_rounds == 1
    assert next_state.last_counted_repair_round_id == 1
    assert next_state.current_repair_round_id == 0
    assert next_state.model_history[-1].status == "ok"
    assert next_state.node_diagnostics[-1]["event"] == "repair_round_failed"


async def test_invalid_plan_response_never_persists_response_bearing_error(monkeypatch):
    hostile_values = (
        "sk-validation-secret-sentinel",
        "FAIL_TO_PASS evaluator-validation-sentinel",
        "HTTP response payload: raw-validation-http-sentinel",
    )
    monkeypatch.delenv("REPOPILOT_ESCALATION_ENABLED", raising=False)
    monkeypatch.delenv("LLM_ESCALATION_API_KEY", raising=False)

    async def fake_llm_call(system, user, model=None, *, provider="primary", **_kwargs):
        return {
            "plan": " | ".join(hostile_values),
            "decision_frame": {
                "stage": "reflect",
                "summary": " | ".join(hostile_values),
            },
        }

    monkeypatch.setattr(plan_node, "llm_call", fake_llm_call)
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        issue_title="Login crash",
        issue_body="Crashes after submit.",
        current_phase=new_agent.Phase.PLAN,
        max_retries=0,
    )

    next_state = await plan_node.plan_fix(state)
    dumped = next_state.model_dump_json()

    assert next_state.current_phase == new_agent.Phase.FAILURE
    assert next_state.failure_reason == "Maximum retries reached: 0."
    assert next_state.model_history[-1].status == "ok"
    assert next_state.model_history[-1].error_class == ""
    assert next_state.node_diagnostics[-1]["event"] == "repair_round_failed"
    assert next_state.node_diagnostics[-1]["provider"] == "primary"
    assert next_state.token_usage == sum(
        item.input_tokens + item.output_tokens
        for item in next_state.model_history
        if item.node != "outcome_summary"
    )
    for forbidden in hostile_values:
        assert forbidden not in dumped


async def test_provider_error_in_reflect_never_persists_provider_message(monkeypatch):
    from src.http_client import LLMResponseError

    hostile_values = (
        "sk-provider-secret-sentinel",
        "PASS_TO_PASS evaluator-provider-sentinel",
        "raw HTTP response body: raw-provider-http-sentinel",
    )
    monkeypatch.delenv("REPOPILOT_ESCALATION_ENABLED", raising=False)
    monkeypatch.delenv("LLM_ESCALATION_API_KEY", raising=False)

    async def fake_llm_call(system, user, **kwargs):
        raise LLMResponseError(" | ".join(hostile_values))

    monkeypatch.setattr(reflect_node, "llm_call", fake_llm_call)
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        issue_title="Login crash",
        issue_body="Crashes after submit.",
        current_phase=new_agent.Phase.REFLECT,
        fix_attempts=[
            new_agent.FixAttempt(
                test_result="failed",
                error_log="AssertionError: safe existing failure",
            )
        ],
    )

    next_state = await reflect_node.reflect_on_failure(state)
    dumped = next_state.model_dump_json()

    assert next_state.current_phase == new_agent.Phase.PLAN
    assert next_state.reflection_notes == "Reflection failed: LLMResponseError"
    assert next_state.model_history[-2].node == "reflect_on_failure"
    assert next_state.model_history[-2].error_class == "LLMResponseError"
    assert next_state.model_history[-1].node == "outcome_summary"
    assert next_state.model_history[-1].error_class == "LLMResponseError"
    assert next_state.node_diagnostics[-1]["error_type"] == "LLMResponseError"
    assert "policy_reason" not in next_state.node_diagnostics[-1]
    assert "error" not in next_state.node_diagnostics[-1]
    assert next_state.conversation_history[-1].content == (
        "Reflection error: LLMResponseError"
    )
    assert next_state.token_usage == sum(
        item.input_tokens + item.output_tokens
        for item in next_state.model_history
        if item.node != "outcome_summary"
    )
    assert next_state.summary_token_usage == sum(
        item.input_tokens + item.output_tokens
        for item in next_state.model_history
        if item.node == "outcome_summary"
    )
    for forbidden in hostile_values:
        assert forbidden not in dumped


async def test_invalid_reflect_response_never_persists_response_values(monkeypatch):
    hostile_values = (
        "sk-reflect-validation-secret",
        "gold_patch evaluator-reflect-sentinel",
        "HTTP/1.1 503 raw-reflect-http-sentinel",
    )
    monkeypatch.delenv("REPOPILOT_ESCALATION_ENABLED", raising=False)
    monkeypatch.delenv("LLM_ESCALATION_API_KEY", raising=False)

    async def fake_llm_call(system, user, model=None, *, provider="primary", **_kwargs):
        hostile = " | ".join(hostile_values)
        return {
            "root_cause": hostile,
            "what_went_wrong": hostile,
            "suggested_fix_approach": hostile,
            "decision_frame": {
                "stage": "plan",
                "summary": hostile,
            },
        }

    monkeypatch.setattr(reflect_node, "llm_call", fake_llm_call)
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        issue_title="Login crash",
        issue_body="Crashes after submit.",
        current_phase=new_agent.Phase.REFLECT,
        fix_attempts=[
            new_agent.FixAttempt(
                test_result="failed",
                error_log="AssertionError: safe existing failure",
            )
        ],
    )

    next_state = await reflect_node.reflect_on_failure(state)
    dumped = next_state.model_dump_json()

    assert next_state.reflection_notes == "Reflection failed: ValidationError"
    assert next_state.node_diagnostics[-1]["error_type"] == "ValidationError"
    assert next_state.node_diagnostics[-1]["policy_reason"] == (
        "invalid_structured_response_after_retries"
    )
    assert "error" not in next_state.node_diagnostics[-1]
    for forbidden in hostile_values:
        assert forbidden not in dumped


async def test_plan_fix_records_prompt_built_diagnostic(monkeypatch):
    async def fake_llm_call(system, user, model=None, *, provider="primary", **_kwargs):
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
        current_phase=new_agent.Phase.PLAN,
    )

    next_state = await plan_node.plan_fix(state)

    prompt_diag = next(
        item for item in next_state.node_diagnostics if item["event"] == "prompt_built"
    )
    assert prompt_diag["node"] == "plan_fix"
    assert prompt_diag["prompt_tokens_estimate"] > 0
    assert prompt_diag["previous_failure_count"] == 0


async def test_plan_fix_prompt_uses_compact_file_context(monkeypatch):
    captured = {}

    async def fake_llm_call(system, user, model=None, *, provider="primary", **_kwargs):
        captured["user"] = user
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
                    "risk": "medium",
                    "confidence": 0.84,
                },
            }
        )

    monkeypatch.setattr(plan_node, "llm_call", fake_llm_call)
    over_limit = plan_node.PLAN_FILE_CONTENT_LIMIT + 3000
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        issue_title="Login crash",
        issue_body="x" * 8000,
        current_phase=new_agent.Phase.PLAN,
        relevant_files=[
            new_agent.FileInfo(
                path="src/auth.py",
                relevance_score=0.9,
                reason="auth path",
                content="a" * over_limit,
            ),
            new_agent.FileInfo(
                path="src/session.py",
                relevance_score=0.8,
                reason="session path",
                content="b" * over_limit,
            ),
        ],
    )

    await plan_node.plan_fix(state)

    # Each file is truncated at the per-file limit (plus the "..." suffix),
    # never shown in full — bounded context, but enough to see real code.
    assert "a" * (plan_node.PLAN_FILE_CONTENT_LIMIT + 1) not in captured["user"]
    assert "b" * (plan_node.PLAN_FILE_CONTENT_LIMIT + 1) not in captured["user"]
    # The issue body is likewise capped.
    assert "x" * (plan_node.PLAN_ISSUE_BODY_LIMIT + 1) not in captured["user"]


async def test_plan_fix_prompt_includes_function_bodies_not_just_imports(monkeypatch):
    captured = {}

    async def fake_llm_call(system, user, model=None, *, provider="primary", **_kwargs):
        captured["user"] = user
        return json.dumps(
            {
                "plan": "Patch env identity.",
                "patch": "diff --git a/src/tox/tox_env/api.py b/src/tox/tox_env/api.py\n",
                "files": ["src/tox/tox_env/api.py"],
                "test_command": "pytest -q",
                "decision_frame": {
                    "stage": "plan",
                    "summary": "Patch env identity.",
                    "recommended_action": "execute",
                    "risk": "medium",
                    "confidence": 0.8,
                },
            }
        )

    monkeypatch.setattr(plan_node, "llm_call", fake_llm_call)
    # Mimic a real file: imports up top, the relevant logic far below the old
    # 1200-char cutoff. The planner must see the method, not just the imports.
    file_body = (
        "import os\n" * 150  # ~1500 chars of imports, past the old 1200 cap
        + "\n\ndef env_dir(self):\n"
        + "    return self._conf.name  # <- the line the fix targets\n"
    )
    assert len(file_body.split("def env_dir")[0]) > 1200  # logic is past old cap
    state = new_agent.AgentState(
        issue_url="https://github.com/tox-dev/tox/issues/3075",
        issue_title="env reuse",
        issue_body="tip-black reuses black env",
        current_phase=new_agent.Phase.PLAN,
        relevant_files=[
            new_agent.FileInfo(
                path="src/tox/tox_env/api.py",
                relevance_score=0.95,
                reason="requested by planner next_checks",
                content=file_body,
            ),
        ],
    )

    await plan_node.plan_fix(state)

    assert "def env_dir(self):" in captured["user"]
    assert "the line the fix targets" in captured["user"]


async def test_plan_fix_no_patch_execute_recommendation_routes_to_failure(monkeypatch):
    async def fake_llm_call(system, user, model=None, *, provider="primary", **_kwargs):
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

    assert planned_state.current_phase == new_agent.Phase.PLAN
    assert planned_state.failure_reason == "patch_authorization_rejected"
    assert planned_state.retry_count == 1
    assert planned_state.decision_frame.recommended_action == "plan"
    assert route == "plan_fix"


async def test_plan_fix_after_patch_apply_failure_includes_hypothesis_anchor(
    exact_repair_state,
    monkeypatch,
):
    calls = []

    async def fake_llm_call(system, user, model=None, *, provider="primary", **_kwargs):
        calls.append({"system": system, "user": user})
        return _exact_plan_response(
            decision_frame={
                "stage": "plan",
                "summary": "Repair the envpython patch.",
                "recommended_action": "execute",
                "hypotheses": [
                    {
                        "id": "H1",
                        "claim": "envpython chooses the wrong interpreter.",
                        "evidence": ["Previous plan selected envpython."],
                        "score": 0.82,
                    }
                ],
                "selected_hypothesis_id": "H1",
                "risk": "medium",
                "confidence": 0.82,
            }
        )

    monkeypatch.setattr(plan_node, "llm_call", fake_llm_call)
    state = exact_repair_state
    state.reflection_notes = '{"suggested_fix_approach": "Repair the malformed edit."}'
    previous_plan = new_agent.DecisionFrame(
        stage="plan",
        summary="Patch envpython environment resolution.",
        hypotheses=[
            new_agent.Hypothesis(
                id="H1",
                claim="envpython chooses the wrong interpreter for the active env.",
                evidence=["Issue points to envpython path selection."],
                score=0.82,
            )
        ],
        selected_hypothesis_id="H1",
        recommended_action="execute",
        risk="medium",
        confidence=0.82,
    )
    new_agent._record_decision_frame(state, previous_plan)
    reflect_frame = new_agent.DecisionFrame(
        stage="reflect",
        summary="The previous unified diff was malformed.",
        hypotheses=[
            new_agent.Hypothesis(
                id="H1",
                claim="envpython remains the root-cause hypothesis.",
                evidence=["The patch failed before tests ran."],
                score=0.8,
            )
        ],
        selected_hypothesis_id="H1",
        recommended_action="plan",
        risk="medium",
        confidence=0.8,
    )
    new_agent._record_decision_frame(state, reflect_frame)
    state.fix_attempts.append(
        new_agent.FixAttempt(
            patch_content="diff --git a/src/envpython.py b/src/envpython.py\n@@ broken\n",
            file_path="src/envpython.py",
            test_result="patch_apply_failed",
            failure_kind="patch_apply_failed",
            error_log="error: corrupt patch at line 2",
            success=False,
        )
    )

    await plan_node.plan_fix(state)

    prompt = calls[0]["user"]
    assert "Hypothesis Continuity Instructions" in prompt
    assert "the proposal failed before tests" in prompt.lower()
    assert "envpython chooses the wrong interpreter for the active env" in prompt
    assert "envpython chooses the wrong interpreter" in prompt


async def test_plan_fix_after_patch_apply_failure_restores_drifted_hypothesis(
    exact_repair_state,
    monkeypatch,
):
    async def fake_llm_call(system, user, model=None, *, provider="primary", **_kwargs):
        return _exact_plan_response(
            decision_frame={
                "stage": "plan",
                "summary": "Patch env hashing instead.",
                "recommended_action": "execute",
                "hypotheses": [
                    {
                        "id": "H2",
                        "claim": "The env uniqueness hash is unstable.",
                        "evidence": ["Unrelated alternate hypothesis."],
                        "score": 0.7,
                    }
                ],
                "selected_hypothesis_id": "H2",
                "risk": "medium",
                "confidence": 0.7,
            }
        )

    monkeypatch.setattr(plan_node, "llm_call", fake_llm_call)
    state = exact_repair_state
    state.reflection_notes = '{"what_went_wrong": "The structured edit was malformed."}'
    previous_plan = new_agent.DecisionFrame(
        stage="plan",
        summary="Patch envpython environment resolution.",
        hypotheses=[
            new_agent.Hypothesis(
                id="H1",
                claim="envpython chooses the wrong interpreter for the active env.",
                evidence=["Issue points to envpython path selection."],
                score=0.82,
            )
        ],
        selected_hypothesis_id="H1",
        recommended_action="execute",
        risk="medium",
        confidence=0.82,
    )
    new_agent._record_decision_frame(state, previous_plan)
    state.fix_attempts.append(
        new_agent.FixAttempt(
            patch_content="diff --git a/src/envpython.py b/src/envpython.py\n@@ broken\n",
            file_path="src/envpython.py",
            test_result="patch_apply_failed",
            failure_kind="patch_apply_failed",
            error_log="error: corrupt patch at line 2",
            success=False,
        )
    )

    next_state = await plan_node.plan_fix(state)

    assert next_state.decision_frame.selected_hypothesis_id == "H1"
    assert next_state.decision_frame.hypotheses[0].id == "H1"
    assert (
        next_state.decision_frame.hypotheses[0].claim
        == "envpython chooses the wrong interpreter for the active env."
    )
    assert next_state.decision_warnings[-1]["warning_type"] == (
        "hypothesis_consistency"
    )
    assert next_state.decision_warnings[-1]["reason"] == (
        "preserved_selected_hypothesis_after_patch_apply_failure"
    )
    assert next_state.decision_warnings[-1]["llm_selected_hypothesis_id"] == ""


async def test_plan_fix_collect_more_context_without_patch_routes_to_locate(
    monkeypatch,
):
    async def fake_llm_call(system, user, model=None, *, provider="primary", **_kwargs):
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
    assert planned_state.current_phase == new_agent.Phase.LOCATE
    assert planned_state.failure_reason == ""
    assert planned_state.decision_frame.recommended_action == "collect_more_context"
    assert planned_state.decision_warnings == []


async def test_plan_fix_ask_user_enters_waiting_phase_for_router(monkeypatch):
    async def fake_llm_call(system, user, model=None, *, provider="primary", **_kwargs):
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
    assert planned_state.decision_warnings == []


async def test_reflect_on_failure_records_reflect_decision_frame(monkeypatch):
    calls = []

    async def fake_llm_call(system, user, model=None, *, provider="primary", **_kwargs):
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


async def test_reflect_on_patch_apply_failure_preserves_selected_hypothesis_context(
    monkeypatch,
):
    calls = []

    async def fake_llm_request(
        messages, model=None, temperature=0.2, *, provider="primary"
    ):
        calls.append({"messages": messages, "model": model})
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "root_cause": "The patch failed before tests ran.",
                                "what_went_wrong": "The unified diff was malformed.",
                                "suggested_fix_approach": (
                                    "Regenerate an apply-able unified diff."
                                ),
                                "files_that_also_need_changes": ["src/envpython.py"],
                                "decision_frame": {
                                    "stage": "reflect",
                                    "summary": "Repair the patch format.",
                                    "recommended_action": "plan",
                                    "hypotheses": [
                                        {
                                            "id": "H1",
                                            "claim": "envpython chooses the wrong interpreter.",
                                            "evidence": [
                                                "Previous selected hypothesis."
                                            ],
                                            "score": 0.8,
                                        }
                                    ],
                                    "selected_hypothesis_id": "H1",
                                    "next_checks": [
                                        "Generate an apply-able unified diff."
                                    ],
                                    "risk": "medium",
                                    "confidence": 0.8,
                                },
                            }
                        )
                    }
                }
            ]
        }

    monkeypatch.setattr("src.llm.llm_request", fake_llm_request)
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/8",
        issue_title="envpython picks the wrong environment",
        issue_body="The envpython helper resolves python from the wrong env.",
        current_phase=new_agent.Phase.REFLECT,
    )
    plan_frame = new_agent.DecisionFrame(
        stage="plan",
        summary="Patch envpython environment resolution.",
        hypotheses=[
            new_agent.Hypothesis(
                id="H1",
                claim="envpython chooses the wrong interpreter for the active env.",
                evidence=["Issue points to envpython path selection."],
                score=0.82,
            ),
            new_agent.Hypothesis(
                id="H2",
                claim="The env uniqueness hash is unstable.",
                evidence=["Possible unrelated collision."],
                score=0.22,
            ),
        ],
        selected_hypothesis_id="H1",
        evidence=["envpython is the selected root-cause area."],
        next_checks=["Patch envpython path lookup."],
        recommended_action="execute",
        risk="medium",
        confidence=0.82,
    )
    new_agent._record_decision_frame(state, plan_frame)
    state.fix_attempts.append(
        new_agent.FixAttempt(
            patch_content=(
                "diff --git a/src/envpython.py b/src/envpython.py\n"
                "--- a/src/envpython.py\n"
                "+++ b/src/envpython.py\n"
                "@@ malformed hunk\n"
            ),
            file_path="src/envpython.py",
            test_result="patch_apply_failed",
            error_log="error: corrupt patch at line 4",
            success=False,
        )
    )

    await reflect_node.reflect_on_failure(state)

    prompt = "\n\n".join(message["content"] for message in calls[0]["messages"])
    prompt_lower = prompt.lower()
    assert "failed to apply" in prompt_lower or "patch apply" in prompt_lower
    assert (
        "keep" in prompt_lower and "root-cause hypothesis" in prompt_lower
    ) or "selected hypothesis" in prompt_lower
    assert "unified diff" in prompt_lower
    assert "formatting" in prompt_lower and (
        "context repair" in prompt_lower or "hunk context" in prompt_lower
    )
    assert "the patch failed before tests ran" in prompt_lower
    assert "envpython chooses the wrong interpreter" in prompt
    assert "H1" in prompt


async def test_reflect_records_frame_health_warning_for_legacy_output(monkeypatch):
    async def fake_llm_call(system, user, model=None, *, provider="primary", **_kwargs):
        return json.dumps(
            {
                "root_cause": "The patch changed the wrong branch.",
                "what_went_wrong": "It ignored the failing None case.",
                "suggested_fix_approach": "Patch the None guard before submit.",
                "files_that_also_need_changes": ["src/auth.py"],
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
    assert next_state.decision_frame.stage == "reflect"
    assert next_state.decision_warnings[-1] == {
        "warning_type": "frame_health",
        "node": "reflect_on_failure",
        "frame_id": "df_0001",
        "expected_stage": "reflect",
        "actual_stage": "reflect",
        "reason": "missing_explicit_decision_frame",
    }


async def test_reflect_accepts_legacy_single_file_string(monkeypatch):
    async def fake_llm_call(system, user, model=None, *, provider="primary", **_kwargs):
        return json.dumps(
            {
                "root_cause": "The patch changed the wrong branch.",
                "what_went_wrong": "It ignored the failing None case.",
                "suggested_fix_approach": "Patch the None guard before submit.",
                "files_that_also_need_changes": "src/auth.py",
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
    assert json.loads(next_state.decision_frame.trace_notes)[
        "files_that_also_need_changes"
    ] == ["src/auth.py"]


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


def test_agent_payload_exposes_node_diagnostics():
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        node_diagnostics=[
            {
                "node": "plan_fix",
                "event": "phase",
                "status": "timeout",
                "elapsed_seconds": 90.0,
                "error_type": "TimeoutError",
                "error": "TimeoutError",
                "phase_timeout_seconds": 90.0,
            }
        ],
    )

    payload = new_agent.agent_payload_from_state(state, turns_taken=0)

    assert payload["node_diagnostics"] == [
        {
            "node": "plan_fix",
            "event": "phase",
            "status": "timeout",
            "elapsed_seconds": 90.0,
            "error_type": "TimeoutError",
            "error": "TimeoutError",
            "phase_timeout_seconds": 90.0,
        }
    ]


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


def test_save_trace_writes_node_diagnostics(tmp_path):
    tracer = new_agent.Tracer()
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        trace_id=tracer.trace_id,
        node_diagnostics=[
            {
                "node": "plan_fix",
                "event": "phase",
                "status": "timeout",
                "elapsed_seconds": 90.0,
                "error_type": "TimeoutError",
                "error": "TimeoutError",
                "phase_timeout_seconds": 90.0,
            }
        ],
    )

    trace_path = tmp_path / "trace.json"
    new_agent._save_trace(tracer, trace_path, state)

    data = json.loads(trace_path.read_text(encoding="utf-8"))
    assert data["node_diagnostics"] == [
        {
            "node": "plan_fix",
            "event": "phase",
            "status": "timeout",
            "elapsed_seconds": 90.0,
            "error_type": "TimeoutError",
            "error": "TimeoutError",
            "phase_timeout_seconds": 90.0,
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
    assert (
        state.human_input_request["question"] == "Need deployment environment details."
    )


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
    assert (
        state.human_input_request["question"] == "Need deployment environment details."
    )


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
