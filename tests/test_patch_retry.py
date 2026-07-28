import subprocess

import src.new_agent as new_agent
from src.nodes import plan as plan_node
from src.repair_rounds import begin_repair_round, bind_repair_round_author
from src.state import RepairPlan, VerifiedEdit, VerifiedEditBatch


def _patch_gate_state(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
    source = "def value():\n    return 1\n"
    (repo / "a.py").write_text(source, encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "a.py"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "base"], check=True)
    ref = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    state = new_agent.AgentState(
        issue_url="https://github.com/a/b/issues/1",
        issue_title="value is wrong",
        issue_body="value should return two",
        repo_path=str(repo),
        repo_ref=ref,
        active_provider="escalation",
        active_model="claude-opus-4-8:stable",
        escalated=True,
        escalation_reason="repeated_no_progress",
        retry_count=1,
    )
    plan = RepairPlan(
        root_cause="value returns one",
        target_files=["a.py"],
        target_symbols=[],
        required_behavior="return two",
        regression_test_strategy="run focused test",
    )
    return state, plan, source


def _verified(search, replace="return 2"):
    return VerifiedEditBatch(
        edits=[
            VerifiedEdit(
                file_path="a.py", search=search, replace=replace, intent="fix value"
            )
        ]
    )


async def test_primary_plan_replacement_atomically_retires_old_gate_approval(
    tmp_path, monkeypatch
):
    state, repair_plan, _source = _patch_gate_state(tmp_path)
    state.active_repair_plan = repair_plan
    from src.patch_gate import validate_patch_batch

    begin_repair_round(state)
    bind_repair_round_author(state)
    assert validate_patch_batch(state, repair_plan, _verified("return 1")).accepted
    assert state.tool_patch_approval is not None
    old_fingerprint = state.tool_patch_approval.patch_gate_fingerprint
    old_round_id = state.authorized_repair_round_id
    state.current_repair_round_id = 0
    state.current_repair_provider = None
    state.current_repair_model = ""
    state.active_provider = "primary"
    state.active_model = "gemini-3.5-flash:stable"

    async def primary_plan(*args, **kwargs):
        return __import__("json").dumps(
            {
                "plan": "replace through primary planning",
                "patch": "",
                "patch_edits": [
                    {"file": "a.py", "search": "return 1", "replace": "return 3"}
                ],
                "files": ["a.py"],
                "test_command": "pytest -q",
                "decision_frame": {
                    "stage": "plan",
                    "summary": "new primary plan",
                    "recommended_action": "execute",
                    "risk": "low",
                    "confidence": 0.8,
                },
            }
        )

    monkeypatch.setattr(plan_node, "llm_call", primary_plan)
    result = await plan_node.plan_fix(state)

    assert result.current_phase == new_agent.Phase.EXECUTE
    assert result.tool_patch_approval is not None
    assert result.tool_patch_approval.patch_gate_fingerprint != old_fingerprint
    assert result.active_repair_plan is not None
    assert result.patch_edits[0].replace == "return 3"
    assert result.patch_edits[0].exact_only is True
    assert result.authorized_repair_round_id == old_round_id + 1
    assert result.authorized_repair_provider == "primary"
    assert result.authorized_repair_model == "gemini-3.5-flash:stable"


async def test_unattributed_patch_failure_is_integrity_failure_when_budget_exhausted():
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        max_retries=1,
        retry_count=1,
        current_phase=new_agent.Phase.VERIFY,
        fix_attempts=[
            new_agent.FixAttempt(
                patch_content="malformed diff 1",
                test_result="patch_apply_failed",
                failure_kind="patch_apply_failed",
                error_log="error: first patch apply failed",
                success=False,
            ),
            new_agent.FixAttempt(
                patch_content="malformed diff 2",
                test_result="patch_apply_failed",
                failure_kind="patch_apply_failed",
                error_log="error: second patch apply failed",
                success=False,
            ),
        ],
    )

    next_state = await new_agent.verify_fix(state)

    assert next_state.current_phase == new_agent.Phase.FAILURE
    assert next_state.retry_count == 1
    assert "state-integrity" in next_state.failure_reason.lower()


async def test_unattributed_preflight_failures_do_not_consume_semantic_retry():
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        max_retries=1,
        retry_count=0,
        current_phase=new_agent.Phase.VERIFY,
        fix_attempts=[
            new_agent.FixAttempt(
                patch_content="malformed diff 1",
                test_result="patch_apply_failed",
                failure_kind="patch_apply_failed",
                error_log="Patch preflight check failed:\nerror: No valid patches in input",
                success=False,
            ),
            new_agent.FixAttempt(
                patch_content="malformed diff 2",
                test_result="patch_apply_failed",
                failure_kind="patch_apply_failed",
                error_log="Patch preflight check failed:\nerror: corrupt patch at line 3",
                success=False,
            ),
        ],
    )

    next_state = await new_agent.verify_fix(state)

    assert next_state.current_phase == new_agent.Phase.FAILURE
    assert next_state.retry_count == 0
    assert "state-integrity" in next_state.failure_reason.lower()


async def test_unattributed_search_replace_failures_do_not_consume_retry():
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        max_retries=1,
        retry_count=0,
        current_phase=new_agent.Phase.VERIFY,
        fix_attempts=[
            new_agent.FixAttempt(
                patch_edits=[
                    new_agent.PatchEdit(
                        file_path="src/auth.py",
                        search="missing old block\n",
                        replace="new block\n",
                    )
                ],
                test_result="patch_apply_failed",
                failure_kind="patch_apply_failed",
                error_log=(
                    "Search/replace edit failed: edit 1 search block was not found "
                    "in src/auth.py."
                ),
                success=False,
            ),
            new_agent.FixAttempt(
                patch_edits=[
                    new_agent.PatchEdit(
                        file_path="src/auth.py",
                        search="still missing old block\n",
                        replace="new block\n",
                    )
                ],
                test_result="patch_apply_failed",
                failure_kind="patch_apply_failed",
                error_log=(
                    "Search/replace edit failed: edit 1 search block was not found "
                    "in src/auth.py."
                ),
                success=False,
            ),
        ],
    )

    next_state = await new_agent.verify_fix(state)

    assert next_state.current_phase == new_agent.Phase.FAILURE
    assert next_state.retry_count == 0
    assert "state-integrity" in next_state.failure_reason.lower()


async def test_unattributed_preflight_history_is_state_integrity_failure():
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        max_retries=1,
        retry_count=0,
        current_phase=new_agent.Phase.VERIFY,
        fix_attempts=[
            new_agent.FixAttempt(
                patch_content="malformed diff 1",
                test_result="patch_apply_failed",
                failure_kind="patch_apply_failed",
                error_log="Patch preflight check failed:\nerror: No valid patches in input",
                success=False,
            ),
            new_agent.FixAttempt(
                patch_content="malformed diff 2",
                test_result="patch_apply_failed",
                failure_kind="patch_apply_failed",
                error_log="Patch preflight check failed:\nerror: corrupt patch at line 3",
                success=False,
            ),
            new_agent.FixAttempt(
                patch_content="malformed diff 3",
                test_result="patch_apply_failed",
                failure_kind="patch_apply_failed",
                error_log="Patch preflight check failed:\nerror: corrupt patch at line 5",
                success=False,
            ),
        ],
    )

    next_state = await new_agent.verify_fix(state)

    assert next_state.current_phase == new_agent.Phase.FAILURE
    assert next_state.retry_count == 0
    assert "state-integrity" in next_state.failure_reason.lower()


async def test_unattributed_search_replace_history_is_state_integrity_failure():
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        max_retries=1,
        retry_count=0,
        current_phase=new_agent.Phase.VERIFY,
        fix_attempts=[
            new_agent.FixAttempt(
                patch_edits=[
                    new_agent.PatchEdit(
                        file_path="src/auth.py",
                        search=f"missing old block {idx}\n",
                        replace="new block\n",
                    )
                ],
                test_result="patch_apply_failed",
                failure_kind="patch_apply_failed",
                error_log=(
                    "Search/replace edit failed: edit 1 search block was not found "
                    "in src/auth.py."
                ),
                success=False,
            )
            for idx in range(3)
        ],
    )

    next_state = await new_agent.verify_fix(state)

    assert next_state.current_phase == new_agent.Phase.FAILURE
    assert next_state.retry_count == 0
    assert "state-integrity" in next_state.failure_reason.lower()
