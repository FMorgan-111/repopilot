"""EXECUTE phase: Apply the planned patch locally and run tests."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import tempfile
from typing import Any

from ..state import (
    AgentState,
    FixAttempt,
    Phase,
    _as_state,
    _is_budget_exceeded,
    _primary_patch_file,
    _record_tool,
)


async def git_clone(state: AgentState) -> str:
    """Clone the target repository to a temporary directory."""
    token = os.getenv("GITHUB_TOKEN", "")
    if token:
        repo_url = f"https://x-access-token:{token}@github.com/{state.owner}/{state.repo}.git"
    else:
        repo_url = f"https://github.com/{state.owner}/{state.repo}.git"
    target = tempfile.mkdtemp(prefix=f"repopilot-{state.owner}-{state.repo}-")
    result = subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, target],
        capture_output=True,
        text=True,
        timeout=180,
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout).strip())
    return target


async def apply_patch(repo_path: str, patch_content: str) -> tuple[bool, str]:
    """Apply a unified diff to the local clone."""
    result = subprocess.run(
        ["git", "apply", "-"],
        input=patch_content,
        cwd=repo_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    output = "\n".join(part for part in [result.stdout, result.stderr] if part)
    return result.returncode == 0, output


async def run_pytest(repo_path: str, command: str | None = None) -> dict[str, Any]:
    """Run the requested test command, defaulting to pytest."""
    if command:
        cmd = shlex.split(command)
    else:
        cmd = ["python3", "-m", "pytest", "-q"]
    result = subprocess.run(
        cmd,
        cwd=repo_path,
        capture_output=True,
        text=True,
        timeout=300,
    )
    return {
        "command": " ".join(cmd),
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "success": result.returncode == 0,
    }


async def execute_fix(state: AgentState | dict[str, Any]) -> AgentState:
    """Apply the planned patch locally and run tests."""
    state = _as_state(state)
    if _is_budget_exceeded(state):
        state.failure_reason = "Token budget exceeded before execution."
        state.current_phase = Phase.FAILURE
        return state

    patch = state.patch_content
    attempt = FixAttempt(patch_content=patch, file_path=_primary_patch_file(patch))
    try:
        if not state.repo_path:
            state.repo_path = await git_clone(state)
        applied, apply_output = await apply_patch(state.repo_path, patch)
        if not applied:
            attempt.test_result = "patch_apply_failed"
            attempt.error_log = apply_output
            attempt.success = False
            state.fix_attempts.append(attempt)
            state.current_phase = Phase.VERIFY
            return state

        test_result = await run_pytest(state.repo_path, state.test_command)
        attempt.test_result = json.dumps(
            {
                "command": test_result.get("command"),
                "returncode": test_result.get("returncode"),
                "success": test_result.get("success"),
            }
        )
        attempt.error_log = (
            (test_result.get("stdout") or "") + "\n" + (test_result.get("stderr") or "")
        )[-8000:]
        attempt.success = bool(test_result.get("success"))
        state.fix_attempts.append(attempt)
        _record_tool(state, "run_pytest", {"repo_path": state.repo_path}, test_result)

    except Exception as exc:
        attempt.test_result = "execution_error"
        attempt.error_log = str(exc)
        attempt.success = False
        state.fix_attempts.append(attempt)

    state.current_phase = Phase.VERIFY
    return state
