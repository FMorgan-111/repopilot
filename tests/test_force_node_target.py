"""After a patch fails to APPLY (search text never matched), the planner is
pushed to re-express the fix via node_target instead of retrying search."""

import json

from src import new_agent
from src.nodes import plan as plan_node
from src.state import AgentState, FileInfo, PatchEdit, Phase

PY = (
    "import os\n\n\n"
    "class Handler:\n"
    "    def process(self, x):\n"
    "        return x + 1\n"
)


def _state(**kw):
    return AgentState(
        issue_url="https://github.com/a/b/issues/1",
        issue_title="bug",
        issue_body="broken",
        current_phase=Phase.PLAN,
        relevant_files=[FileInfo(path="app/h.py", content=PY, relevance_score=0.9)],
        **kw,
    )


def _apply_failed_attempt(file_path="app/h.py", search="    def process(self, x):\n        return x + 1"):
    return new_agent.FixAttempt(
        patch_edits=[PatchEdit(file_path=file_path, search=search, replace="X")],
        test_result="patch_apply_failed",
        failure_kind="patch_apply_failed",
        success=False,
    )


# ---- helpers ---------------------------------------------------------------

def test_enclosing_node_names_finds_method():
    names = plan_node._enclosing_node_names(PY, "        return x + 1")
    assert "Handler.process" in names


def test_enclosing_node_names_empty_for_missing_anchor():
    assert plan_node._enclosing_node_names(PY, "not in file at all") == []


def test_force_node_target_instructions_names_node():
    state = _state(fix_attempts=[_apply_failed_attempt()])
    text = plan_node._force_node_target_instructions(state)
    assert "node_target" in text
    assert "app/h.py:Handler.process" in text
    assert "FAILED TO APPLY" in text


def test_force_node_target_skips_non_python():
    state = _state(fix_attempts=[_apply_failed_attempt(file_path="README.md")])
    state.relevant_files = [FileInfo(path="README.md", content="text", relevance_score=1.0)]
    assert plan_node._force_node_target_instructions(state) == ""


def test_force_node_target_empty_without_apply_failure():
    ok = _apply_failed_attempt()
    ok.failure_kind = "test_failed"
    ok.test_result = "failed"
    state = _state(fix_attempts=[ok])
    assert plan_node._force_node_target_instructions(state) == ""


# ---- prompt wiring ---------------------------------------------------------

async def test_plan_prompt_forces_node_target_after_apply_failure(monkeypatch):
    captured = {}

    async def fake_llm_call(system, user):
        captured["system"] = system
        return json.dumps(
            {
                "plan": "retarget",
                "patch": "",
                "patch_edits": [
                    {"file": "app/h.py", "node_target": "Handler.process",
                     "replace": "    def process(self, x):\n        return x + 2"}
                ],
                "files": ["app/h.py"],
                "test_command": "pytest",
                "decision_frame": {
                    "stage": "plan", "summary": "retarget",
                    "recommended_action": "execute", "risk": "low", "confidence": 0.7,
                },
            }
        )

    monkeypatch.setattr(plan_node, "llm_call", fake_llm_call)
    state = _state(retry_count=1, max_retries=3, fix_attempts=[_apply_failed_attempt()])

    await plan_node.plan_fix(state)

    assert "node_target" in captured["system"]
    assert "FAILED TO APPLY" in captured["system"]
    # The old "retry search/replace" instruction must NOT be present.
    assert "repair the previous patch as exact patch_edits with correct" not in captured["system"]


async def test_plan_prompt_keeps_search_repair_for_non_python(monkeypatch):
    captured = {}

    async def fake_llm_call(system, user):
        captured["system"] = system
        return json.dumps(
            {
                "plan": "fix", "patch": "",
                "patch_edits": [{"file": "cfg.ini", "search": "a=1", "replace": "a=2"}],
                "files": ["cfg.ini"], "test_command": "pytest",
                "decision_frame": {
                    "stage": "plan", "summary": "fix",
                    "recommended_action": "execute", "risk": "low", "confidence": 0.7,
                },
            }
        )

    monkeypatch.setattr(plan_node, "llm_call", fake_llm_call)
    state = _state(
        retry_count=1, max_retries=3,
        fix_attempts=[_apply_failed_attempt(file_path="cfg.ini", search="a=1")],
    )
    state.relevant_files = [FileInfo(path="cfg.ini", content="a=1\n", relevance_score=1.0)]

    await plan_node.plan_fix(state)

    # Falls back to the search-repair instruction for non-Python files.
    assert "repair the previous patch as exact patch_edits with correct" in captured["system"]
