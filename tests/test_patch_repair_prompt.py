import json

from src import new_agent
from src.nodes import plan as plan_node
from src.nodes import reflect as reflect_node


async def test_reflect_patch_apply_failure_prompt_prefers_search_replace_repair(monkeypatch):
    calls = []

    async def fake_llm_call(system, user):
        calls.append({"system": system, "user": user})
        return json.dumps(
            {
                "root_cause": "The patch failed before tests ran.",
                "what_went_wrong": "The old patch format was malformed.",
                "suggested_fix_approach": "Regenerate the change as patch_edits.",
                "files_that_also_need_changes": [],
                "decision_frame": {
                    "stage": "reflect",
                    "summary": "Repair the patch format.",
                    "recommended_action": "plan",
                    "hypotheses": [
                        {
                            "id": "H1",
                            "claim": "envpython chooses the wrong interpreter.",
                            "evidence": ["Previous selected hypothesis."],
                            "score": 0.8,
                        }
                    ],
                    "selected_hypothesis_id": "H1",
                    "risk": "medium",
                    "confidence": 0.8,
                },
            }
        )

    monkeypatch.setattr(reflect_node, "llm_call", fake_llm_call)
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
            )
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

    prompt = f"{calls[0]['system']}\n\n{calls[0]['user']}"
    assert "tests did not run" in prompt.lower()
    assert "search/replace" in prompt.lower()
    assert "patch_edits" in prompt
    assert "search" in prompt
    assert "replace" in prompt
    assert "index 1234567..abcdefg" not in prompt
    assert "# ... apply similar" not in prompt
    assert "Previous patch apply error" in prompt
    assert "corrupt patch at line 4" in prompt
    assert "preflight" in prompt.lower()
    assert "H1" in prompt
    assert "envpython chooses the wrong interpreter" in prompt


async def test_reflect_search_replace_failure_prompt_includes_failed_edits(monkeypatch):
    calls = []

    async def fake_llm_call(system, user):
        calls.append({"system": system, "user": user})
        return json.dumps(
            {
                "root_cause": "The patch failed before tests ran.",
                "what_went_wrong": "The search block did not match the file.",
                "suggested_fix_approach": "Use a search block copied from the file.",
                "files_that_also_need_changes": [],
                "decision_frame": {
                    "stage": "reflect",
                    "summary": "Repair the patch_edits search block.",
                    "recommended_action": "plan",
                    "risk": "medium",
                    "confidence": 0.8,
                },
            }
        )

    monkeypatch.setattr(reflect_node, "llm_call", fake_llm_call)
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/8",
        issue_title="envpython picks the wrong environment",
        issue_body="The envpython helper resolves python from the wrong env.",
        current_phase=new_agent.Phase.REFLECT,
        fix_attempts=[
            new_agent.FixAttempt(
                patch_edits=[
                    new_agent.PatchEdit(
                        file_path="src/envpython.py",
                        search="old envpython lookup\n",
                        replace="new envpython lookup\n",
                    )
                ],
                test_result="patch_apply_failed",
                failure_kind="patch_apply_failed",
                error_log=(
                    "Search/replace edit failed: edit 1 search block was not found "
                    "in src/envpython.py."
                ),
                success=False,
            )
        ],
    )

    await reflect_node.reflect_on_failure(state)

    prompt = calls[0]["user"]
    assert "Patch Edits Applied" in prompt
    assert "file: src/envpython.py" in prompt
    assert "search:" in prompt
    assert "old envpython lookup" in prompt
    assert "replace:" in prompt
    assert "new envpython lookup" in prompt


async def test_plan_patch_apply_failure_prompt_requires_search_replace_repair(monkeypatch):
    calls = []

    async def fake_llm_call(system, user):
        calls.append({"system": system, "user": user})
        return json.dumps(
            {
                "plan": "Repair the envpython patch before changing semantics.",
                "patch": "",
                "patch_edits": [
                    {
                        "file": "src/envpython.py",
                        "search": "old envpython lookup\n",
                        "replace": "new envpython lookup\n",
                    }
                ],
                "files": ["src/envpython.py"],
                "test_command": "pytest tests/test_envpython.py -q",
                "decision_frame": {
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
                },
            }
        )

    monkeypatch.setattr(plan_node, "llm_call", fake_llm_call)
    state = new_agent.AgentState(
        issue_url="https://github.com/tox-dev/tox/issues/3075",
        issue_title="envpython picks the wrong environment",
        issue_body="The envpython helper resolves python from the wrong env.",
        current_phase=new_agent.Phase.PLAN,
        reflection_notes='{"suggested_fix_approach": "Repair the malformed diff."}',
    )
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
            error_log="Patch preflight check failed:\nerror: corrupt patch at line 2",
            success=False,
        )
    )

    await plan_node.plan_fix(state)

    prompt = calls[0]["user"]
    assert "Hypothesis Continuity Instructions" in prompt
    assert "tests did not run" in prompt.lower()
    assert "exact patch_edits repair" in prompt.lower()
    assert "repair the previous patch's file paths and search blocks before changing semantics" in prompt.lower()
    assert "patch_edits" in calls[0]["system"]
    assert "search" in calls[0]["system"]
    assert "replace" in calls[0]["system"]
    assert "index 1234567..abcdefg" not in prompt
    assert "# ... apply similar" not in prompt
    assert "Previous patch apply error" in prompt
    assert "preflight" in prompt.lower()
    assert "corrupt patch at line 2" in prompt
    assert "H1" in prompt
    assert "envpython chooses the wrong interpreter" in prompt


async def test_plan_fix_includes_human_answer_context(monkeypatch):
    calls = []

    async def fake_llm_call(system, user):
        calls.append({"system": system, "user": user})
        return json.dumps(
            {
                "plan": "Use the human answer to refine the patch.",
                "patch": "diff --git a/src/envpython.py b/src/envpython.py\n",
                "files": ["src/envpython.py"],
                "test_command": "pytest tests/test_envpython.py -q",
                "decision_frame": {
                    "stage": "plan",
                    "summary": "Use the human answer to refine the patch.",
                    "recommended_action": "execute",
                    "risk": "medium",
                    "confidence": 0.8,
                },
            }
        )

    monkeypatch.setattr(plan_node, "llm_call", fake_llm_call)
    state = new_agent.AgentState(
        issue_url="https://github.com/tox-dev/tox/issues/3075",
        issue_title="envpython picks the wrong environment",
        issue_body="The envpython helper resolves python from the wrong env.",
        current_phase=new_agent.Phase.PLAN,
    )
    state.conversation_history.append(
        new_agent.ConversationTurn(
            role="user",
            content=(
                "Human answer for paused run abc123def456:\n"
                "Breaking changes are allowed."
            ),
        )
    )

    await plan_node.plan_fix(state)

    prompt = calls[0]["user"]
    assert "Human answer since resume" in prompt
    assert "Breaking changes are allowed." in prompt


async def test_reflect_on_failure_includes_human_answer_context(monkeypatch):
    calls = []

    async def fake_llm_call(system, user):
        calls.append({"system": system, "user": user})
        return json.dumps(
            {
                "root_cause": "The patch failed before tests ran.",
                "what_went_wrong": "The unified diff was malformed.",
                "suggested_fix_approach": "Repair the diff syntax.",
                "files_that_also_need_changes": [],
                "decision_frame": {
                    "stage": "reflect",
                    "summary": "Repair the patch format.",
                    "recommended_action": "plan",
                    "risk": "medium",
                    "confidence": 0.8,
                },
            }
        )

    monkeypatch.setattr(reflect_node, "llm_call", fake_llm_call)
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/8",
        issue_title="envpython picks the wrong environment",
        issue_body="The envpython helper resolves python from the wrong env.",
        current_phase=new_agent.Phase.REFLECT,
    )
    state.conversation_history.append(
        new_agent.ConversationTurn(
            role="user",
            content=(
                "Human answer for paused run abc123def456:\n"
                "Breaking changes are allowed."
            ),
        )
    )
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

    prompt = f"{calls[0]['system']}\n\n{calls[0]['user']}"
    assert "Human answer since resume" in prompt
    assert "Breaking changes are allowed." in prompt
