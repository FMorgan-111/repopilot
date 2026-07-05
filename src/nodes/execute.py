"""EXECUTE phase: Apply the planned patch locally and run tests."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..patch_repair import repair_unified_diff
from ..patch_match import (
    find_normalized_span,
    leading_spaces,
    locate_node_span,
    reindent,
    try_upgrade_to_node_target,
)
from ..state import (
    AgentState,
    FixAttempt,
    PatchEdit,
    Phase,
    _as_state,
    _is_budget_exceeded,
    _primary_patch_file,
    _record_node_diagnostic,
    _record_tool,
)

_TOKENIZED_GITHUB_URL_RE = re.compile(
    r"https://x-access-token:[^@\s'\"\]]+@github[.]com/"
)


@dataclass(frozen=True)
class PatchApplyResult:
    applied: bool
    output: str
    patch_content: str
    repaired: bool = False
    repair_reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PatchEditApplyResult:
    applied: bool
    output: str
    changed_files: list[str] = field(default_factory=list)


def _redact_sensitive_error_text(text: str) -> str:
    """Remove credentials from command/error text before persistence."""
    return _TOKENIZED_GITHUB_URL_RE.sub(
        "https://x-access-token:<redacted>@github.com/",
        text,
    )


def _repopilot_home() -> Path:
    configured = os.getenv("REPOPILOT_HOME")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".repopilot"


def _repo_cache_path(owner: str, repo: str) -> Path:
    safe_owner = owner.replace("/", "-")
    safe_repo = repo.replace("/", "-")
    return _repopilot_home() / "repos" / f"{safe_owner}-{safe_repo}"


def _repo_work_path(owner: str, repo: str) -> Path:
    """Stable per-repo work tree, reused across samples of the same repo so the
    sibling venv and editable install survive instead of being rebuilt each time.
    """
    safe_owner = owner.replace("/", "-")
    safe_repo = repo.replace("/", "-")
    return _repopilot_home() / "repos" / f"{safe_owner}-{safe_repo}-work"


def _repo_url(state: AgentState, *, include_token: bool) -> str:
    token = os.getenv("GITHUB_TOKEN", "") if include_token else ""
    if token:
        return f"https://x-access-token:{token}@github.com/{state.owner}/{state.repo}.git"
    return f"https://github.com/{state.owner}/{state.repo}.git"


def _clone_local_repo(cache_path: Path, target: str) -> None:  # pragma: no cover
    # Retained for backward-compat imports; the live path is the async variant.
    subprocess.run(
        ["git", "clone", "--local", "--no-hardlinks", str(cache_path), target],
        capture_output=True,
        text=True,
        timeout=180,
        check=True,
    )


class _ProcResult:
    __slots__ = ("returncode", "stdout", "stderr")

    def __init__(self, returncode: int, stdout: str, stderr: str) -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


async def _run_git_async(
    args: list[str], timeout: float, cwd: str | None = None
) -> _ProcResult:
    """Run git via an async subprocess so a hung clone does NOT pin the event
    loop. The subprocess is killed both on its own timeout and when the caller
    is cancelled (e.g. the execute_fix phase timeout), so neither can leave a
    git process running unbounded — the bug that made a stuck clone hang the
    whole run."""
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        proc.kill()
        await proc.wait()
        raise
    return _ProcResult(
        proc.returncode or 0,
        stdout.decode(errors="replace"),
        stderr.decode(errors="replace"),
    )


async def _clone_local_repo_async(cache_path: Path, target: str) -> None:
    res = await _run_git_async(
        ["git", "clone", "--local", "--no-hardlinks", str(cache_path), target],
        timeout=180,
    )
    if res.returncode != 0:
        raise subprocess.CalledProcessError(
            res.returncode, "git clone --local", res.stdout, res.stderr
        )
    if not await _worktree_is_healthy(target):
        # A "successful" clone that produced no checkout (empty/broken cache) —
        # treat as failure so the caller discards it and re-downloads.
        raise subprocess.CalledProcessError(
            1, "git clone --local", "", "clone produced an empty work tree"
        )


async def _worktree_is_healthy(work: str) -> bool:
    """A usable work tree resolves HEAD and has checked-out files. Guards against
    reusing an empty/broken clone (0 files, no HEAD) — which silently made every
    patch fail with 'target file was not found', masquerading as the model
    picking wrong paths."""
    if not (Path(work) / ".git").exists():
        return False
    head = await _run_git_async(["git", "-C", work, "rev-parse", "HEAD"], timeout=30)
    if head.returncode != 0:
        return False
    listed = await _run_git_async(["git", "-C", work, "ls-files"], timeout=60)
    return listed.returncode == 0 and bool(listed.stdout.strip())


async def _reset_work_tree_async(work: str) -> None:
    """Restore a reused work tree to pristine HEAD, dropping a prior sample's
    applied patch. ``clean -fd`` (no ``-x``) keeps git-ignored build artifacts
    like ``.egg-info`` so the editable install stays valid across reuse."""
    for args in (
        ["git", "-C", work, "reset", "--hard", "HEAD"],
        ["git", "-C", work, "clean", "-fd"],
    ):
        await _run_git_async(args, timeout=120)


async def git_clone(state: AgentState) -> str:
    """Clone the target repo, reusing a per-repo work tree across samples.

    - Reuse an existing work tree by resetting it to pristine HEAD — this skips
      a re-clone AND lets the sibling venv / editable install be reused.
    - Otherwise local-clone from the git-objects cache under
      ``repos/<owner-repo>`` (the network clone happens once); download to that
      cache on first sight.

    Initial-download fallbacks (best → fallback):
    1. --depth 1 --single-branch  (fast, blobful so it can serve --local)
    2. --depth 1  (shallow, all branches)
    3. full clone  (no flags)

    All git subprocesses run async (``_run_git_async``) so a hung clone is
    killable by its own timeout and by the execute_fix phase timeout.
    """
    repo_url = _repo_url(state, include_token=True)
    safe_repo_url = _repo_url(state, include_token=False)
    cache_path = _repo_cache_path(state.owner, state.repo)
    work = str(_repo_work_path(state.owner, state.repo))

    # Reuse an existing work tree ONLY if it is healthy (resolves HEAD, has
    # files). A broken/empty reuse is what made every patch fail with "target
    # file was not found" for whole repos. Discard an unhealthy tree and rebuild.
    if (Path(work) / ".git").exists():
        if await _worktree_is_healthy(work):
            await _reset_work_tree_async(work)
            return work
        shutil.rmtree(work, ignore_errors=True)

    # Git objects already cached: fast local clone into the work tree.
    if (cache_path / ".git").exists():
        try:
            await _clone_local_repo_async(cache_path, work)
            return work
        except (subprocess.CalledProcessError, asyncio.TimeoutError):
            # Corrupt/partial/empty cache (e.g. an older blobless clone that
            # cannot serve a local work tree, or one that checks out nothing).
            # Discard it and re-download from scratch.
            shutil.rmtree(cache_path, ignore_errors=True)
            shutil.rmtree(work, ignore_errors=True)

    strategies = [
        # NB: no --filter=blob:none — a blobless cache cannot serve a `git clone
        # --local` work tree (its blobs are missing and lazy-fetch is disabled),
        # which fails with "could not fetch ... from promisor remote".
        ["git", "clone", "--depth", "1", "--single-branch", repo_url, str(cache_path)],
        ["git", "clone", "--depth", "1", repo_url, str(cache_path)],
        ["git", "clone", repo_url, str(cache_path)],
    ]

    last_error = ""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    for cmd in strategies:
        try:
            # 300s per strategy; async so the execute_fix phase timeout can also
            # cancel a hung clone (a blocking subprocess.run could not be).
            result = await _run_git_async(cmd, timeout=300)
        except asyncio.TimeoutError:
            last_error = f"clone timed out: {' '.join(cmd[:4])}..."
            if cache_path.exists():
                shutil.rmtree(cache_path, ignore_errors=True)
            continue
        if result.returncode == 0:
            await _run_git_async(
                ["git", "-C", str(cache_path), "remote", "set-url", "origin", safe_repo_url],
                timeout=60,
            )
            await _clone_local_repo_async(cache_path, work)
            return work
        last_error = _redact_sensitive_error_text(
            (result.stderr or result.stdout).strip()
        )

    if cache_path.exists():
        shutil.rmtree(cache_path, ignore_errors=True)
    raise RuntimeError(last_error)


def _git_apply_check(repo_path: str, patch_content: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "apply", "--check", "-"],
        input=patch_content,
        cwd=repo_path,
        capture_output=True,
        text=True,
        timeout=60,
    )


def _git_apply(repo_path: str, patch_content: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "apply", "-"],
        input=patch_content,
        cwd=repo_path,
        capture_output=True,
        text=True,
        timeout=60,
    )


def _combined_process_output(result: subprocess.CompletedProcess) -> str:
    return "\n".join(part for part in [result.stdout, result.stderr] if part)


def _apply_patch_edits(
    repo_path: str, patch_edits: list[PatchEdit]
) -> PatchEditApplyResult:
    """Apply exact search/replace edits after validating every edit."""
    pending_writes: dict[Path, str] = {}
    changed_files: list[str] = []
    repo_root = Path(repo_path).resolve()

    for index, edit in enumerate(patch_edits, start=1):
        relative_path = Path(edit.file_path)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            return PatchEditApplyResult(
                applied=False,
                output=(
                    "Search/replace edit failed: "
                    f"edit {index} has unsafe file path {edit.file_path!r}."
                ),
            )

        file_path = (repo_root / relative_path).resolve()
        try:
            file_path.relative_to(repo_root)
        except ValueError:
            return PatchEditApplyResult(
                applied=False,
                output=(
                    "Search/replace edit failed: "
                    f"edit {index} escapes repository root: {edit.file_path!r}."
                ),
            )

        if not file_path.exists():
            return PatchEditApplyResult(
                applied=False,
                output=(
                    "Search/replace edit failed: "
                    f"edit {index} target file was not found: {edit.file_path}."
                ),
            )

        content = pending_writes.get(file_path)
        if content is None:
            content = file_path.read_text(encoding="utf-8")

        if edit.node_target:
            # AST-anchored replacement: locate the named def/class and replace
            # its whole span. No verbatim text anchoring, no line drift.
            span = locate_node_span(content, edit.node_target)
            if span is None:
                return PatchEditApplyResult(
                    applied=False,
                    output=(
                        f"Node-target edit failed: edit {index} could not locate a "
                        f"unique definition {edit.node_target!r} in {edit.file_path}."
                    ),
                )
            start, end, node_indent = span
            first = next((ln for ln in edit.replace.split("\n") if ln.strip()), "")
            replacement = reindent(edit.replace, node_indent - leading_spaces(first))
            if not replacement.endswith("\n"):
                replacement += "\n"
            updated = content[:start] + replacement + content[end:]
        else:
            match_count = content.count(edit.search)
            if match_count == 1:
                updated = content.replace(edit.search, edit.replace, 1)
            elif match_count > 1:
                if not edit.replace_all:
                    return PatchEditApplyResult(
                        applied=False,
                        output=(
                            "Search/replace edit failed: "
                            f"edit {index} search block matched {match_count} times in "
                            f"{edit.file_path}; set replace_all=true only when all matches "
                            "should change."
                        ),
                    )
                updated = content.replace(edit.search, edit.replace)
            else:
                # Exact search not found — the dominant Gemini failure is whitespace
                # drift (indent / trailing space). Retry with a normalized, unique
                # line match before giving up, reindenting the replacement to match.
                span = (
                    None if edit.replace_all else find_normalized_span(content, edit.search)
                )
                if span is None:
                    # Last resort: if this is really a whole-function rewrite whose
                    # search text drifted too far to match, re-anchor by AST node
                    # (the model won't do this itself). Size-gated so it can never
                    # truncate a partial edit. Python files only.
                    qualname = (
                        None
                        if edit.replace_all or not edit.file_path.endswith(".py")
                        else try_upgrade_to_node_target(content, edit.search, edit.replace)
                    )
                    node_span = (
                        locate_node_span(content, qualname) if qualname else None
                    )
                    if node_span is None:
                        # Diagnostic: why did the converter decline? (helps tune
                        # the gate). Reports the shape of this apply-failure.
                        if edit.file_path.endswith(".py") and not edit.replace_all:
                            from ..patch_match import diagnose_node_upgrade
                            print(
                                f"  [execute] node-upgrade declined edit {index} "
                                f"{edit.file_path}: {diagnose_node_upgrade(content, edit.search, edit.replace)}",
                                file=sys.stderr,
                                flush=True,
                            )
                        return PatchEditApplyResult(
                            applied=False,
                            output=(
                                "Search/replace edit failed: "
                                f"edit {index} search block was not found in {edit.file_path}."
                            ),
                        )
                    n_start, n_end, node_indent = node_span
                    first = next(
                        (ln for ln in edit.replace.split("\n") if ln.strip()), ""
                    )
                    body = reindent(edit.replace, node_indent - leading_spaces(first))
                    if not body.endswith("\n"):
                        body += "\n"
                    print(
                        f"  [execute] upgraded search->node_target {edit.file_path}:"
                        f"{qualname} (edit {index})",
                        file=sys.stderr,
                        flush=True,
                    )
                    updated = content[:n_start] + body + content[n_end:]
                else:
                    start, end, indent_delta = span
                    updated = content[:start] + reindent(edit.replace, indent_delta) + content[end:]

        pending_writes[file_path] = updated
        if edit.file_path not in changed_files:
            changed_files.append(edit.file_path)

    for file_path, content in pending_writes.items():
        file_path.write_text(content, encoding="utf-8")

    return PatchEditApplyResult(
        applied=True,
        output=f"Applied {len(patch_edits)} search/replace edit(s).",
        changed_files=changed_files,
    )


async def apply_patch_with_repair(
    repo_path: str, patch_content: str
) -> PatchApplyResult:
    """Apply a unified diff, attempting deterministic syntax repair once."""
    preflight = _git_apply_check(repo_path, patch_content)
    preflight_output = _combined_process_output(preflight)
    if preflight.returncode == 0:
        result = _git_apply(repo_path, patch_content)
        output = _combined_process_output(result)
        if result.returncode != 0:
            output = f"Patch apply failed after preflight passed:\n{output}"
        return PatchApplyResult(
            applied=result.returncode == 0,
            output=_redact_sensitive_error_text(output),
            patch_content=patch_content,
        )

    original_failure = f"Patch preflight check failed:\n{preflight_output}"
    repair = repair_unified_diff(patch_content)
    if not repair.changed:
        return PatchApplyResult(
            applied=False,
            output=_redact_sensitive_error_text(original_failure),
            patch_content=patch_content,
        )

    repaired_preflight = _git_apply_check(repo_path, repair.patch)
    repaired_preflight_output = _combined_process_output(repaired_preflight)
    if repaired_preflight.returncode != 0:
        output = (
            f"{original_failure}\n\n"
            "Patch repair attempted but preflight still failed "
            f"(reasons: {', '.join(repair.reasons)}):\n"
            f"{repaired_preflight_output}"
        )
        return PatchApplyResult(
            applied=False,
            output=_redact_sensitive_error_text(output),
            patch_content=repair.patch,
            repaired=True,
            repair_reasons=repair.reasons,
        )

    result = _git_apply(repo_path, repair.patch)
    output = _combined_process_output(result)
    prefix = f"Patch repaired before apply (reasons: {', '.join(repair.reasons)})."
    if result.returncode != 0:
        output = f"Patch apply failed after repaired preflight passed:\n{output}"
    output = f"{prefix}\n{output}".strip()
    return PatchApplyResult(
        applied=result.returncode == 0,
        output=_redact_sensitive_error_text(output),
        patch_content=repair.patch,
        repaired=True,
        repair_reasons=repair.reasons,
    )


async def apply_patch(repo_path: str, patch_content: str) -> tuple[bool, str]:
    """Apply a unified diff to the local clone."""
    result = await apply_patch_with_repair(repo_path, patch_content)
    return result.applied, result.output


# Best-effort editable install so the cloned package and its test deps are
# importable before pytest runs. Bounded and failure-tolerant: src/-layout repos
# (e.g. tox) die with "No module named X" / "command not found" without it, no
# matter how correct the patch is. We do NOT fail the attempt if install fails —
# pytest may still run for flat-layout / stdlib-only repos.
INSTALL_TIMEOUT_SECONDS = 240
VENV_CREATE_TIMEOUT_SECONDS = 120


def _venv_dir_for(repo_path: str) -> Path:
    """Sibling venv dir for a clone — kept outside the tree so pytest can't
    collect it, and so the clone's own files are untouched."""
    p = Path(repo_path)
    return p.parent / f"{p.name}-venv"


def _venv_python_path(repo_path: str) -> Path:
    return _venv_dir_for(repo_path) / "bin" / "python"


def _venv_ready_marker(repo_path: str) -> Path:
    return _venv_dir_for(repo_path) / ".repopilot-ready"


def _venv_is_ready(repo_path: str) -> bool:
    return _venv_python_path(repo_path).exists() and _venv_ready_marker(repo_path).exists()


def _mark_venv_ready(repo_path: str) -> None:
    _venv_ready_marker(repo_path).write_text("ready\n", encoding="utf-8")


def _create_venv(repo_path: str) -> dict[str, Any]:
    """Create an isolated venv for the clone, reusing system site-packages.

    The system Python is often PEP 668 externally-managed (pip install refused),
    and even when not, installing into it is unsafe. A venv sidesteps PEP 668
    entirely; --system-site-packages reuses already-present deps (pytest, etc.)
    so only the missing ones are installed. Returns a tracing record; never
    raises — callers fall back to system python3 on failure.
    """
    venv_dir = _venv_dir_for(repo_path)
    venv_python = _venv_python_path(repo_path)
    if _venv_is_ready(repo_path):
        return {"created": True, "python": str(venv_python), "reason": "exists"}
    if venv_dir.exists():
        shutil.rmtree(venv_dir, ignore_errors=True)
    try:
        result = subprocess.run(
            ["python3", "-m", "venv", "--system-site-packages",
             str(venv_dir)],
            capture_output=True,
            text=True,
            timeout=VENV_CREATE_TIMEOUT_SECONDS,
        )
    except Exception as exc:  # pragma: no cover - defensive
        stdlib_reason = str(exc)[:200]
    else:
        if result.returncode == 0 and venv_python.exists():
            _mark_venv_ready(repo_path)
            return {"created": True, "python": str(venv_python), "creator": "stdlib"}
        stdlib_reason = (result.stderr or "venv creation failed")[:200]

    shutil.rmtree(venv_dir, ignore_errors=True)
    try:
        uv_result = subprocess.run(
            ["uv", "venv", "--system-site-packages", str(venv_dir)],
            capture_output=True,
            text=True,
            timeout=VENV_CREATE_TIMEOUT_SECONDS,
        )
    except Exception as exc:  # pragma: no cover - defensive
        uv_reason = str(exc)[:200]
    else:
        if uv_result.returncode == 0 and venv_python.exists():
            _mark_venv_ready(repo_path)
            return {
                "created": True,
                "python": str(venv_python),
                "creator": "uv",
                "stdlib_reason": stdlib_reason,
            }
        uv_reason = (uv_result.stderr or "uv venv creation failed")[:200]

    shutil.rmtree(venv_dir, ignore_errors=True)
    return {
        "created": False,
        "python": None,
        "reason": stdlib_reason,
        "uv_reason": uv_reason,
    }


def _pip_install_editable(
    repo_path: str, python_exe: str = "python3"
) -> dict[str, Any]:
    """Try `pip install -e .[<extras>]`, falling back to a bare editable install.

    Installs with ``python_exe`` (the clone's venv interpreter when available).
    Returns a record for tracing. Never raises — install is best-effort.
    """
    has_metadata = any(
        (Path(repo_path) / name).exists()
        for name in ("pyproject.toml", "setup.py", "setup.cfg")
    )
    if not has_metadata:
        return {"attempted": False, "reason": "no_packaging_metadata"}

    candidates = [
        [python_exe, "-m", "pip", "install", "-e", ".[test]"],
        [python_exe, "-m", "pip", "install", "-e", ".[testing]"],
        [python_exe, "-m", "pip", "install", "-e", ".[dev]"],
        [python_exe, "-m", "pip", "install", "-e", "."],
    ]
    last: dict[str, Any] = {"attempted": True}
    for cmd in candidates:
        try:
            result = subprocess.run(
                cmd,
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=INSTALL_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            return {"attempted": True, "success": False, "reason": "timeout",
                    "command": " ".join(cmd)}
        except Exception as exc:  # pragma: no cover - defensive
            return {"attempted": True, "success": False, "reason": str(exc)[:200],
                    "command": " ".join(cmd)}
        last = {
            "attempted": True,
            "success": result.returncode == 0,
            "command": " ".join(cmd),
            "returncode": result.returncode,
        }
        if result.returncode == 0:
            return last
    return last


def _ensure_pytest_available(python_exe: str) -> dict[str, Any]:
    """Ensure the selected interpreter can run pytest with Scrapy's plugin args."""
    try:
        check = subprocess.run(
            [python_exe, "-c", "import pytest, pytest_twisted"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception as exc:  # pragma: no cover - defensive
        check = subprocess.CompletedProcess(
            [python_exe, "-c", "import pytest, pytest_twisted"],
            returncode=1,
            stdout="",
            stderr=str(exc),
        )
    if check.returncode == 0:
        return {"attempted": False, "reason": "pytest_available"}

    cmd = [python_exe, "-m", "pip", "install", "pytest", "pytest-twisted"]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=INSTALL_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return {"attempted": True, "success": False, "reason": "timeout",
                "command": " ".join(cmd)}
    except Exception as exc:  # pragma: no cover - defensive
        return {"attempted": True, "success": False, "reason": str(exc)[:200],
                "command": " ".join(cmd)}
    return {
        "attempted": True,
        "success": result.returncode == 0,
        "command": " ".join(cmd),
        "returncode": result.returncode,
    }


async def run_pytest(repo_path: str, command: str | None = None) -> dict[str, Any]:
    """Run the requested test command, defaulting to pytest.

    Uses the clone's venv interpreter when one exists (so the editable install
    is importable), falling back to system python3 otherwise. For an explicit
    command the venv's bin dir is prepended to PATH so bare `pytest`/`python`
    resolve to the venv.
    """
    venv_python = _venv_python_path(repo_path)
    has_venv = _venv_is_ready(repo_path)
    py = str(venv_python) if has_venv else "python3"

    env = os.environ.copy()
    if has_venv:
        env["PATH"] = f"{venv_python.parent}{os.pathsep}{env.get('PATH', '')}"
        env["VIRTUAL_ENV"] = str(_venv_dir_for(repo_path))

    if command:
        cmd = shlex.split(command)
        if has_venv and cmd and cmd[0] == "pytest":
            cmd = [py, "-m", "pytest", *cmd[1:]]
    else:
        cmd = [py, "-m", "pytest", "-q"]
    result = subprocess.run(
        cmd,
        cwd=repo_path,
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
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
    attempt = FixAttempt(
        patch_content=patch,
        patch_edits=state.patch_edits,
        file_path=(
            state.patch_edits[0].file_path
            if state.patch_edits
            else _primary_patch_file(patch)
        ),
    )
    try:
        if not state.repo_path:
            try:
                state.repo_path = await git_clone(state)
            except Exception as exc:
                attempt.test_result = "execution_error"
                attempt.failure_kind = "infra_error"
                attempt.error_log = _redact_sensitive_error_text(str(exc))
                attempt.success = False
                state.fix_attempts.append(attempt)
                state.current_phase = Phase.VERIFY
                return state
            # Build an isolated venv once, right after the fresh clone, then do
            # a best-effort editable install into it so the package and its test
            # deps import when pytest runs. The venv sidesteps PEP 668 on
            # externally-managed system pythons; install into venv pip, not
            # system pip.
            venv_record = _create_venv(state.repo_path)
            _record_tool(
                state, "create_venv", {"repo_path": state.repo_path}, venv_record
            )
            install_python = venv_record.get("python") or "python3"
            # Skip the (minutes-long) editable install when the venv was already
            # built for this repo on a prior sample — the reused work tree keeps
            # the editable install valid.
            if venv_record.get("reason") == "exists":
                install_record = {"attempted": False, "reason": "venv_cached"}
            else:
                install_record = _pip_install_editable(
                    state.repo_path, python_exe=install_python
                )
            _record_tool(
                state,
                "pip_install_editable",
                {"repo_path": state.repo_path, "python": install_python},
                install_record,
            )
            if install_python != "python3":
                pytest_record = _ensure_pytest_available(install_python)
                _record_tool(
                    state,
                    "ensure_pytest_available",
                    {"python": install_python},
                    pytest_record,
                )
        if state.patch_edits:
            edit_result = _apply_patch_edits(state.repo_path, state.patch_edits)
            _record_tool(
                state,
                "apply_patch_edits",
                {"edit_count": len(state.patch_edits)},
                {
                    "applied": edit_result.applied,
                    "changed_files": edit_result.changed_files,
                    "output": edit_result.output,
                },
            )
            if not edit_result.applied:
                attempt.test_result = "patch_apply_failed"
                attempt.failure_kind = "patch_apply_failed"
                attempt.error_log = _redact_sensitive_error_text(edit_result.output)
                attempt.success = False
                state.fix_attempts.append(attempt)
                state.current_phase = Phase.VERIFY
                return state
        else:
            apply_result = await apply_patch_with_repair(state.repo_path, patch)
            if apply_result.repaired:
                attempt.patch_content = apply_result.patch_content
                attempt.file_path = _primary_patch_file(apply_result.patch_content)
                state.patch_content = apply_result.patch_content
                _record_node_diagnostic(
                    state,
                    node="execute_fix",
                    event="patch_repair",
                    status="success" if apply_result.applied else "error",
                    elapsed_seconds=0.0,
                    repair_reasons=apply_result.repair_reasons,
                    original_patch_chars=len(patch),
                    repaired_patch_chars=len(apply_result.patch_content),
                    output_preview=apply_result.output[:500],
                )
                _record_tool(
                    state,
                    "patch_repair",
                    {"original_chars": len(patch)},
                    {
                        "changed": True,
                        "reasons": apply_result.repair_reasons,
                        "repaired_chars": len(apply_result.patch_content),
                    },
                )
            if not apply_result.applied:
                attempt.test_result = "patch_apply_failed"
                attempt.failure_kind = "patch_apply_failed"
                attempt.error_log = _redact_sensitive_error_text(apply_result.output)
                attempt.success = False
                state.fix_attempts.append(attempt)
                state.current_phase = Phase.VERIFY
                return state
        if state.patch_edits:
            _record_node_diagnostic(
                state,
                node="execute_fix",
                event="patch_edits",
                status="success",
                elapsed_seconds=0.0,
                edit_count=len(state.patch_edits),
            )

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
        attempt.error_log = _redact_sensitive_error_text(attempt.error_log)
        attempt.success = bool(test_result.get("success"))
        attempt.failure_kind = "" if attempt.success else "test_failed"
        state.fix_attempts.append(attempt)
        _record_tool(state, "run_pytest", {"repo_path": state.repo_path}, test_result)

    except Exception as exc:
        attempt.test_result = "execution_error"
        attempt.failure_kind = "execution_error"
        attempt.error_log = _redact_sensitive_error_text(str(exc))
        attempt.success = False
        state.fix_attempts.append(attempt)

    state.current_phase = Phase.VERIFY
    return state
