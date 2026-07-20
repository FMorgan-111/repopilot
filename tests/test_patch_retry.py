import src.new_agent as new_agent
import subprocess
from types import SimpleNamespace

from src import repair_flow
from src.nodes import plan as plan_node
from src.state import FileInfo, RepairPlan, VerifiedEdit, VerifiedEditBatch


def _patch_gate_state(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
    source = "def value():\n    return 1\n"
    (repo / "a.py").write_text(source, encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "a.py"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "base"], check=True)
    ref = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
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
        edits=[VerifiedEdit(file_path="a.py", search=search, replace=replace, intent="fix value")]
    )


async def test_patch_gate_uses_two_local_corrections_without_consuming_retry(tmp_path, monkeypatch):
    state, repair_plan, source = _patch_gate_state(tmp_path)
    seen = []

    async def generate(*args, **kwargs):
        assert kwargs["validate_edits"] is False
        return repair_plan, _verified("not in file")

    responses = [_verified("still absent"), _verified("return 1")]

    async def correct(current_state, plan, batch, issues):
        seen.append((current_state.active_model, issues[0].code, issues[0].correction_context))
        return responses.pop(0)

    monkeypatch.setattr(plan_node, "generate_opus_repair", generate)
    monkeypatch.setattr(plan_node, "request_verified_edit_correction", correct)

    result = await plan_node.plan_fix(state)

    assert result.current_phase == new_agent.Phase.EXECUTE
    assert result.retry_count == 1
    assert result.patch_correction_count == 0
    assert len(seen) == 2
    assert all(item[0] == "claude-opus-4-8:stable" for item in seen)
    assert seen[0][1] == "search_missing"
    assert source[:100] in seen[0][2]
    assert result.tool_patch_approval is not None


async def test_third_invalid_patch_routes_to_reflect_without_third_correction(tmp_path, monkeypatch):
    state, repair_plan, _source = _patch_gate_state(tmp_path)
    calls = 0

    async def generate(*args, **kwargs):
        return repair_plan, _verified("missing zero")

    async def correct(*args):
        nonlocal calls
        calls += 1
        return _verified(f"missing {calls}")

    monkeypatch.setattr(plan_node, "generate_opus_repair", generate)
    monkeypatch.setattr(plan_node, "request_verified_edit_correction", correct)

    result = await plan_node.plan_fix(state)

    assert calls == 2
    assert result.current_phase == new_agent.Phase.REFLECT
    assert result.patch_correction_count == 2
    assert result.retry_count == 1
    assert not result.patch_edits


async def test_cancelling_pathless_batch_gets_exactly_two_local_corrections(
    tmp_path, monkeypatch
):
    state, repair_plan, _source = _patch_gate_state(tmp_path)
    cancelling = VerifiedEditBatch(
        edits=[
            VerifiedEdit(
                file_path="a.py", search="return 1", replace="return 2", intent="change"
            ),
            VerifiedEdit(
                file_path="a.py", search="return 2", replace="return 1", intent="cancel"
            ),
        ]
    )
    calls = []

    async def generate(*args, **kwargs):
        return repair_plan, cancelling

    async def correct(_state, _plan, _batch, issues):
        calls.append(issues)
        return cancelling

    monkeypatch.setattr(plan_node, "generate_opus_repair", generate)
    monkeypatch.setattr(plan_node, "request_verified_edit_correction", correct)

    result = await plan_node.plan_fix(state)

    assert len(calls) == 2
    assert all(issues[0].file_path == "" for issues in calls)
    assert all(len(issues[0].correction_context) <= 1_200 for issues in calls)
    assert result.current_phase == new_agent.Phase.REFLECT
    assert result.patch_correction_count == 2


async def test_pathless_issue_renders_bounded_correction_payload(monkeypatch, tmp_path):
    state, repair_plan, _source = _patch_gate_state(tmp_path)
    previous = _verified("return 1")
    captured = {}

    async def fake_call_schema(_state, **kwargs):
        captured.update(kwargs)
        return previous

    monkeypatch.setattr(repair_flow, "_call_schema", fake_call_schema)
    result = await repair_flow.request_verified_edit_correction(
        state,
        repair_plan,
        previous,
        [
            SimpleNamespace(
                code="empty_patch",
                file_path="",
                message="cancelled batch",
                correction_context="x" * 5_000,
            )
        ],
    )

    payload = __import__("json").loads(captured["user"])
    assert result is previous
    assert payload["patch_gate_issues"][0]["file_path"] == ""
    assert len(payload["patch_gate_issues"][0]["real_code_window"]) == 1_200


async def test_valid_gate_approval_skips_stale_relevant_file_hallucination_check(
    tmp_path, monkeypatch
):
    state, repair_plan, _source = _patch_gate_state(tmp_path)
    state.relevant_files = [
        FileInfo(path="a.py", content="stale content\n", reason="old snapshot")
    ]

    async def generate(*args, **kwargs):
        return repair_plan, _verified("return 1")

    monkeypatch.setattr(plan_node, "generate_opus_repair", generate)
    result = await plan_node.plan_fix(state)

    assert result.current_phase == new_agent.Phase.EXECUTE
    assert result.hallucinated_search_block_count == 0
    assert result.tool_patch_approval is not None


async def test_final_gate_rejection_clears_active_plan_and_approval(tmp_path, monkeypatch):
    state, repair_plan, _source = _patch_gate_state(tmp_path)

    async def generate(*args, **kwargs):
        return repair_plan, _verified("missing zero")

    async def correct(*args):
        return _verified("still missing")

    monkeypatch.setattr(plan_node, "generate_opus_repair", generate)
    monkeypatch.setattr(plan_node, "request_verified_edit_correction", correct)
    result = await plan_node.plan_fix(state)

    assert result.current_phase == new_agent.Phase.REFLECT
    assert result.tool_patch_approval is None
    assert result.active_repair_plan is None


async def test_primary_plan_replacement_atomically_retires_old_gate_approval(
    tmp_path, monkeypatch
):
    state, repair_plan, _source = _patch_gate_state(tmp_path)
    state.active_repair_plan = repair_plan
    from src.patch_gate import validate_patch_batch

    assert validate_patch_batch(state, repair_plan, _verified("return 1")).accepted
    assert state.tool_patch_approval is not None
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
    assert result.tool_patch_approval is None
    assert result.active_repair_plan is None
    assert result.patch_edits and not result.patch_edits[0].exact_only


async def test_patch_apply_failure_routes_to_failure_when_budget_exhausted():
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
    assert next_state.failure_reason == "Maximum retries reached: 1."


async def test_consecutive_preflight_failures_do_not_consume_semantic_retry():
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

    assert next_state.current_phase == new_agent.Phase.REFLECT
    assert next_state.retry_count == 0


async def test_consecutive_search_replace_failures_do_not_consume_semantic_retry():
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

    assert next_state.current_phase == new_agent.Phase.REFLECT
    assert next_state.retry_count == 0


async def test_preflight_repair_failures_are_bounded_without_semantic_retry():
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
    assert next_state.failure_reason == (
        "Patch repair budget exhausted after 3 failures."
    )


async def test_search_replace_repair_failures_are_bounded_without_semantic_retry():
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
    assert next_state.failure_reason == (
        "Patch repair budget exhausted after 3 failures."
    )
