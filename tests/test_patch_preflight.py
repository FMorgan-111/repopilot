import subprocess

import src.nodes.execute as execute_node
from src import new_agent
from src.patch_repair import repair_unified_diff


async def test_execute_fix_applies_search_replace_patch_edits_without_git_apply(
    monkeypatch, tmp_path
):
    app_path = tmp_path / "src" / "auth.py"
    app_path.parent.mkdir()
    app_path.write_text(
        "def verify(retry_count, max_retries):\n"
        "    if retry_count > max_retries:\n"
        "        return FAILURE\n"
        "    return OK\n",
        encoding="utf-8",
    )

    async def fail_apply_patch_with_repair(repo_path, patch_content):
        raise AssertionError("search/replace edits should not invoke git apply")

    async def fake_run_pytest(repo_path, command=None):
        return {
            "command": command or "pytest -q",
            "returncode": 0,
            "stdout": "",
            "stderr": "",
            "success": True,
        }

    monkeypatch.setattr(
        execute_node, "apply_patch_with_repair", fail_apply_patch_with_repair
    )
    monkeypatch.setattr(execute_node, "run_pytest", fake_run_pytest)
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        repo_path=str(tmp_path),
        patch_edits=[
            new_agent.PatchEdit(
                file_path="src/auth.py",
                search=(
                    "    if retry_count > max_retries:\n"
                    "        return FAILURE\n"
                ),
                replace=(
                    "    if retry_count >= max_retries:\n"
                    "        return FAILURE\n"
                ),
            )
        ],
        test_command="pytest tests/test_auth.py -q",
    )

    next_state = await execute_node.execute_fix(state)

    assert "if retry_count >= max_retries" in app_path.read_text(encoding="utf-8")
    assert next_state.current_phase == new_agent.Phase.VERIFY
    assert next_state.fix_attempts[-1].success is True
    assert next_state.fix_attempts[-1].patch_edits[0].file_path == "src/auth.py"
    assert any(call.tool_name == "apply_patch_edits" for call in next_state.tool_calls)


async def test_execute_fix_search_replace_failure_does_not_modify_file(
    monkeypatch, tmp_path
):
    app_path = tmp_path / "src" / "auth.py"
    app_path.parent.mkdir()
    original = "def verify():\n    return OK\n"
    app_path.write_text(original, encoding="utf-8")

    async def fail_apply_patch_with_repair(repo_path, patch_content):
        raise AssertionError("search/replace edits should not invoke git apply")

    monkeypatch.setattr(
        execute_node, "apply_patch_with_repair", fail_apply_patch_with_repair
    )
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        repo_path=str(tmp_path),
        patch_edits=[
            new_agent.PatchEdit(
                file_path="src/auth.py",
                search="missing text\n",
                replace="replacement text\n",
            )
        ],
    )

    next_state = await execute_node.execute_fix(state)

    assert app_path.read_text(encoding="utf-8") == original
    assert next_state.current_phase == new_agent.Phase.VERIFY
    assert next_state.fix_attempts[-1].success is False
    assert next_state.fix_attempts[-1].test_result == "patch_apply_failed"
    assert next_state.fix_attempts[-1].failure_kind == "patch_apply_failed"
    assert "Search/replace edit failed" in next_state.fix_attempts[-1].error_log
    assert "search block was not found" in next_state.fix_attempts[-1].error_log


async def test_apply_patch_checks_before_applying(monkeypatch, tmp_path):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs.get("input")))
        if cmd == ["git", "apply", "--check", "-"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd == ["git", "apply", "-"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="applied", stderr="")
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(execute_node.subprocess, "run", fake_run)

    applied, output = await execute_node.apply_patch(str(tmp_path), "patch-content")

    assert applied is True
    assert output == "applied"
    assert [cmd for cmd, _ in calls] == [
        ["git", "apply", "--check", "-"],
        ["git", "apply", "-"],
    ]
    assert [payload for _, payload in calls] == ["patch-content", "patch-content"]


async def test_execute_fix_preflight_failure_short_circuits_and_redacts_token(
    monkeypatch, tmp_path
):
    token = "gho_patchsecret"
    tokenized_url = f"https://x-access-token:{token}@github.com/acme/widget.git"
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd == ["git", "apply", "--check", "-"]:
            return subprocess.CompletedProcess(
                cmd,
                1,
                stdout="",
                stderr=f"fatal: unable to access '{tokenized_url}/'",
            )
        raise AssertionError("git apply should not run after a failed preflight")

    monkeypatch.setattr(execute_node.subprocess, "run", fake_run)
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        repo_path=str(tmp_path),
        patch_content="malformed diff",
    )

    next_state = await execute_node.execute_fix(state)

    assert calls == [["git", "apply", "--check", "-"]]
    assert next_state.current_phase == new_agent.Phase.VERIFY
    assert len(next_state.fix_attempts) == 1
    attempt = next_state.fix_attempts[-1]
    assert attempt.test_result == "patch_apply_failed"
    assert attempt.failure_kind == "patch_apply_failed"
    assert token not in attempt.error_log
    assert tokenized_url not in attempt.error_log
    assert "https://x-access-token:<redacted>@github.com/acme/widget.git" in (
        attempt.error_log
    )


async def test_execute_fix_preflight_failure_records_distinct_stage(
    monkeypatch, tmp_path
):
    def fake_run(cmd, **kwargs):
        if cmd == ["git", "apply", "--check", "-"]:
            return subprocess.CompletedProcess(
                cmd,
                1,
                stdout="",
                stderr="error: No valid patches in input",
            )
        raise AssertionError("git apply should not run after a failed preflight")

    monkeypatch.setattr(execute_node.subprocess, "run", fake_run)
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        repo_path=str(tmp_path),
        patch_content="malformed diff",
    )

    next_state = await execute_node.execute_fix(state)

    attempt = next_state.fix_attempts[-1]
    assert attempt.test_result == "patch_apply_failed"
    assert attempt.failure_kind == "patch_apply_failed"
    assert "Patch preflight check failed" in attempt.error_log
    assert "error: No valid patches in input" in attempt.error_log


def test_repair_unified_diff_extracts_diff_and_recounts_hunk_lengths():
    raw_patch = """Here is the repaired patch:
```diff
diff --git a/src/app.py b/src/app.py
index 1234567..abcdefg 100644
--- a/src/app.py
+++ b/src/app.py
@@ -19,7 +19,7 @@
-old_value
+new_value
```
This fixes the issue.
"""

    repaired = repair_unified_diff(raw_patch)

    assert repaired.changed is True
    assert repaired.patch.startswith("diff --git a/src/app.py b/src/app.py\n")
    assert "Here is the repaired patch" not in repaired.patch
    assert "```" not in repaired.patch
    assert "index 1234567..abcdefg" not in repaired.patch
    assert "@@ -19,1 +19,1 @@" in repaired.patch
    assert repaired.patch.endswith("\n")
    assert "extracted_diff_block" in repaired.reasons
    assert "removed_invalid_index_line" in repaired.reasons
    assert "recounted_hunk_lengths" in repaired.reasons


async def test_execute_fix_uses_repaired_patch_when_preflight_repair_passes(
    monkeypatch, tmp_path
):
    original_patch = """Here is the repaired patch:
```diff
diff --git a/src/app.py b/src/app.py
index 1234567..abcdefg 100644
--- a/src/app.py
+++ b/src/app.py
@@ -19,7 +19,7 @@
-old_value
+new_value
```
"""
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs.get("input")))
        if cmd == ["git", "apply", "--check", "-"] and kwargs.get("input") == original_patch:
            return subprocess.CompletedProcess(
                cmd,
                1,
                stdout="",
                stderr="error: corrupt patch at line 9",
            )
        if cmd == ["git", "apply", "--check", "-"]:
            assert "@@ -19,1 +19,1 @@" in kwargs.get("input")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd == ["git", "apply", "-"]:
            assert "@@ -19,1 +19,1 @@" in kwargs.get("input")
            return subprocess.CompletedProcess(cmd, 0, stdout="applied", stderr="")
        raise AssertionError(f"Unexpected command: {cmd}")

    async def fake_run_pytest(repo_path, command=None):
        return {
            "command": "pytest -q",
            "returncode": 0,
            "stdout": "",
            "stderr": "",
            "success": True,
        }

    monkeypatch.setattr(execute_node.subprocess, "run", fake_run)
    monkeypatch.setattr(execute_node, "run_pytest", fake_run_pytest)
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        repo_path=str(tmp_path),
        patch_content=original_patch,
    )

    next_state = await execute_node.execute_fix(state)

    assert next_state.current_phase == new_agent.Phase.VERIFY
    assert next_state.fix_attempts[-1].success is True
    assert next_state.fix_attempts[-1].patch_content != original_patch
    assert next_state.patch_content == next_state.fix_attempts[-1].patch_content
    repair_diag = next(
        item
        for item in next_state.node_diagnostics
        if item["node"] == "execute_fix" and item["event"] == "patch_repair"
    )
    assert repair_diag["status"] == "success"
    assert repair_diag["repair_reasons"] == [
        "extracted_diff_block",
        "removed_invalid_index_line",
        "recounted_hunk_lengths",
    ]
    assert any(call.tool_name == "patch_repair" for call in next_state.tool_calls)
    assert [cmd for cmd, _ in calls] == [
        ["git", "apply", "--check", "-"],
        ["git", "apply", "--check", "-"],
        ["git", "apply", "-"],
    ]


async def test_execute_fix_records_repair_attempt_when_repaired_preflight_fails(
    monkeypatch, tmp_path
):
    original_patch = """```diff
diff --git a/src/app.py b/src/app.py
index 1234567..abcdefg 100644
--- a/src/app.py
+++ b/src/app.py
@@ -19,7 +19,7 @@
-old_value
+new_value
```
"""

    def fake_run(cmd, **kwargs):
        if cmd == ["git", "apply", "--check", "-"] and kwargs.get("input") == original_patch:
            return subprocess.CompletedProcess(
                cmd,
                1,
                stdout="",
                stderr="error: corrupt patch at line 9",
            )
        if cmd == ["git", "apply", "--check", "-"]:
            return subprocess.CompletedProcess(
                cmd,
                1,
                stdout="",
                stderr="error: patch failed: src/app.py:19",
            )
        raise AssertionError("git apply should not run after repaired preflight fails")

    monkeypatch.setattr(execute_node.subprocess, "run", fake_run)
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        repo_path=str(tmp_path),
        patch_content=original_patch,
    )

    next_state = await execute_node.execute_fix(state)

    attempt = next_state.fix_attempts[-1]
    assert attempt.success is False
    assert attempt.failure_kind == "patch_apply_failed"
    assert attempt.patch_content != original_patch
    assert "Patch repair attempted but preflight still failed" in attempt.error_log
    assert "error: corrupt patch at line 9" in attempt.error_log
    assert "error: patch failed: src/app.py:19" in attempt.error_log
    repair_diag = next(
        item
        for item in next_state.node_diagnostics
        if item["node"] == "execute_fix" and item["event"] == "patch_repair"
    )
    assert repair_diag["status"] == "error"
    assert repair_diag["repair_reasons"] == [
        "extracted_diff_block",
        "removed_invalid_index_line",
        "recounted_hunk_lengths",
    ]
    assert any(call.tool_name == "patch_repair" for call in next_state.tool_calls)
