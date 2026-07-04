"""Tests for convergence fixes: patch diversification (solution 1) and
final-attempt force-execute (solution 2) in plan_fix / reflect_on_failure."""

import json

from src import new_agent
from src.nodes import plan as plan_node
from src.nodes import reflect as reflect_node
from src.state import PatchEdit


def _failed_attempt(file_path="app/router.py", search="def handle():",
                    replace="def handle(x):"):
    return new_agent.FixAttempt(
        patch_edits=[PatchEdit(file_path=file_path, search=search, replace=replace)],
        test_result="failed",
        success=False,
    )


def _base_state(**kw):
    return new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        issue_title="Login crash",
        issue_body="Crashes after submit.",
        current_phase=new_agent.Phase.PLAN,
        **kw,
    )


def _execute_response(file_path, search, replace, summary="Apply the fix."):
    return json.dumps(
        {
            "plan": summary,
            "patch": "",
            "patch_edits": [{"file": file_path, "search": search, "replace": replace}],
            "files": [file_path],
            "test_command": "pytest",
            "decision_frame": {
                "stage": "plan",
                "summary": summary,
                "recommended_action": "execute",
                "risk": "low",
                "confidence": 0.7,
            },
        }
    )


def _collect_more_context_response(summary="Need more context."):
    return json.dumps(
        {
            "plan": summary,
            "patch": "",
            "files": [],
            "test_command": "",
            "decision_frame": {
                "stage": "plan",
                "summary": summary,
                "recommended_action": "collect_more_context",
                "risk": "unknown",
                "confidence": 0.5,
            },
        }
    )


# ---- pure helpers -----------------------------------------------------------

def test_prior_failed_edits_context_empty_without_failures():
    assert plan_node._prior_failed_edits_context(_base_state()) == ""


def test_prior_failed_edits_context_lists_failed_edits():
    state = _base_state(fix_attempts=[_failed_attempt()])
    ctx = plan_node._prior_failed_edits_context(state)
    assert "ALREADY-TRIED EDITS" in ctx
    assert "app/router.py" in ctx


def test_prior_failed_edits_context_ignores_successful_attempts():
    ok = _failed_attempt()
    ok.success = True
    assert plan_node._prior_failed_edits_context(_base_state(fix_attempts=[ok])) == ""


def test_planned_edits_repeat_failure_detects_repeat():
    state = _base_state(fix_attempts=[_failed_attempt()])
    state.patch_edits = [
        PatchEdit(file_path="app/router.py", search="def handle():", replace="X")
    ]
    assert plan_node._planned_edits_repeat_failure(state) is True


def test_planned_edits_repeat_failure_false_when_diversified():
    state = _base_state(fix_attempts=[_failed_attempt()])
    state.patch_edits = [
        PatchEdit(file_path="app/other.py", search="def other():", replace="Y")
    ]
    assert plan_node._planned_edits_repeat_failure(state) is False


def test_is_final_attempt():
    assert plan_node._is_final_attempt(_base_state(retry_count=3, max_retries=3)) is True
    assert plan_node._is_final_attempt(_base_state(retry_count=2, max_retries=3)) is False


# ---- solution 1: diversification -------------------------------------------

async def test_plan_fix_records_warning_when_patch_repeats_failure(monkeypatch):
    async def fake_llm_call(system, user):
        return _execute_response("app/router.py", "def handle():", "def handle(x):")

    monkeypatch.setattr(plan_node, "llm_call", fake_llm_call)
    state = _base_state(retry_count=1, max_retries=3,
                        fix_attempts=[_failed_attempt()])

    next_state = await plan_node.plan_fix(state)

    assert next_state.current_phase == new_agent.Phase.EXECUTE
    assert any(
        w.get("warning") == "repeated_failed_patch"
        for w in next_state.decision_warnings
    )


async def test_plan_fix_no_warning_when_patch_diversified(monkeypatch):
    async def fake_llm_call(system, user):
        return _execute_response("app/other.py", "def other():", "def other(x):")

    monkeypatch.setattr(plan_node, "llm_call", fake_llm_call)
    state = _base_state(retry_count=1, max_retries=3,
                        fix_attempts=[_failed_attempt()])

    next_state = await plan_node.plan_fix(state)

    assert next_state.current_phase == new_agent.Phase.EXECUTE
    assert not any(
        w.get("warning") == "repeated_failed_patch"
        for w in next_state.decision_warnings
    )


async def test_plan_prompt_includes_failed_edits(monkeypatch):
    captured = {}

    async def fake_llm_call(system, user):
        captured["user"] = user
        return _execute_response("app/other.py", "def other():", "Y")

    monkeypatch.setattr(plan_node, "llm_call", fake_llm_call)
    state = _base_state(retry_count=1, max_retries=3,
                        fix_attempts=[_failed_attempt()])

    await plan_node.plan_fix(state)

    assert "ALREADY-TRIED EDITS" in captured["user"]
    assert "app/router.py" in captured["user"]


# ---- solution 2: final-attempt force execute -------------------------------

async def test_plan_fix_final_attempt_blocks_collect_more_context(monkeypatch):
    async def fake_llm_call(system, user):
        return _collect_more_context_response()

    monkeypatch.setattr(plan_node, "llm_call", fake_llm_call)
    state = _base_state(retry_count=3, max_retries=3)

    next_state = await plan_node.plan_fix(state)

    assert next_state.current_phase == new_agent.Phase.FAILURE
    assert "Final attempt" in next_state.failure_reason
    # must NOT have looped as a context round
    assert next_state.context_collection_count == 0


async def test_plan_fix_non_final_collect_more_context_still_loops(monkeypatch):
    async def fake_llm_call(system, user):
        return _collect_more_context_response()

    monkeypatch.setattr(plan_node, "llm_call", fake_llm_call)
    state = _base_state(retry_count=0, max_retries=3)

    next_state = await plan_node.plan_fix(state)

    assert next_state.current_phase == new_agent.Phase.PLAN
    assert next_state.context_collection_count == 1


async def test_plan_prompt_includes_final_attempt_instruction(monkeypatch):
    captured = {}

    async def fake_llm_call(system, user):
        captured["system"] = system
        return _execute_response("app/x.py", "a", "b")

    monkeypatch.setattr(plan_node, "llm_call", fake_llm_call)
    state = _base_state(retry_count=3, max_retries=3)

    await plan_node.plan_fix(state)

    assert "FINAL planning attempt" in captured["system"]


# ---- reflect wiring ---------------------------------------------------------

async def test_reflect_prompt_includes_failed_edits(monkeypatch):
    captured = {}

    async def fake_llm_call(system, user):
        captured["user"] = user
        return json.dumps(
            {
                "root_cause": "wrong file",
                "what_went_wrong": "edit missed",
                "suggested_fix_approach": "try elsewhere",
                "files_that_also_need_changes": [],
                "decision_frame": {
                    "stage": "reflect",
                    "summary": "wrong file",
                    "recommended_action": "plan",
                    "risk": "medium",
                    "confidence": 0.4,
                },
            }
        )

    monkeypatch.setattr(reflect_node, "llm_call", fake_llm_call)
    state = _base_state(retry_count=1, max_retries=3,
                        fix_attempts=[_failed_attempt()])

    await reflect_node.reflect_on_failure(state)

    assert "ALREADY-TRIED EDITS" in captured["user"]
    assert "DIFFERENT edit" in captured["user"]
