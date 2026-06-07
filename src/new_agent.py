"""RepoPilot v2 agent: graph-based issue fixing loop.

This module replaces the old linear tool loop with a state-machine agent:
UNDERSTAND -> LOCATE -> PLAN -> EXECUTE -> VERIFY -> COMMIT/DONE, with retry
edges from VERIFY back to PLAN. The implementation uses LangGraph when it is
installed and keeps a tiny compatible fallback runner for local tests.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import shlex
import subprocess
import tempfile
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Callable

import httpx
from pydantic import BaseModel, Field

from .agent import parse_issue_url
from .llm import llm_call
from .tools import GITHUB_API, _headers, read_file, read_issue, search_code
from .tracer import Tracer

try:  # pragma: no cover - exercised only when langgraph is installed.
    from langgraph.graph import END, StateGraph
except ImportError:  # pragma: no cover - fallback is covered by tests.
    END = "__end__"
    StateGraph = None


class Phase(str, Enum):
    UNDERSTAND = "UNDERSTAND"
    LOCATE = "LOCATE"
    PLAN = "PLAN"
    EXECUTE = "EXECUTE"
    VERIFY = "VERIFY"
    COMMIT = "COMMIT"
    FAILURE = "FAILURE"
    DONE = "DONE"
    FAILED = "FAILED"


class ConversationTurn(BaseModel):
    role: str
    content: str


class FileInfo(BaseModel):
    path: str
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""
    content: str = ""
    sha: str = ""


class FixAttempt(BaseModel):
    patch_content: str = ""
    file_path: str = ""
    test_result: str = ""
    error_log: str = ""
    success: bool = False


class ToolCall(BaseModel):
    tool_name: str
    args: dict[str, Any] = Field(default_factory=dict)
    result: Any = None
    error: str | None = None


class FinalReport(BaseModel):
    issue_url: str
    fix_applied: bool = False
    pr_url: str | None = None
    test_results: str = ""
    turns_taken: int = 0
    token_used: int = 0


class AgentState(BaseModel):
    issue_url: str
    issue_title: str = ""
    issue_body: str = ""
    current_phase: Phase = Phase.UNDERSTAND
    relevant_files: list[FileInfo] = Field(default_factory=list)
    fix_attempts: list[FixAttempt] = Field(default_factory=list)
    conversation_history: list[ConversationTurn] = Field(default_factory=list)
    token_usage: int = 0
    max_retries: int = 3
    token_budget: int = 50000
    retry_count: int = 0
    tool_calls: list[ToolCall] = Field(default_factory=list)
    owner: str = ""
    repo: str = ""
    issue_number: int = 0
    issue_type: str = "unknown"
    severity: str = "unknown"
    fix_plan: str = ""
    patch_content: str = ""
    test_command: str = ""
    repo_path: str = ""
    branch_name: str = ""
    base_branch: str = "main"
    pr_url: str | None = None
    failure_reason: str = ""
    trace_id: str = ""


NodeFn = Callable[[AgentState], Awaitable[AgentState]]


def _as_state(value: Any) -> AgentState:
    if isinstance(value, AgentState):
        return value
    if isinstance(value, dict):
        return AgentState.model_validate(value)
    return AgentState.model_validate(dict(value))


def _estimate_tokens(*parts: str) -> int:
    return max(1, sum(len(part or "") for part in parts) // 4)


def _remember(state: AgentState, role: str, content: str, max_turns: int = 12) -> None:
    state.conversation_history.append(ConversationTurn(role=role, content=content))
    if len(state.conversation_history) > max_turns:
        state.conversation_history = state.conversation_history[-max_turns:]


def _record_tool(
    state: AgentState,
    tool_name: str,
    args: dict[str, Any],
    result: Any = None,
    error: str | None = None,
) -> None:
    state.tool_calls.append(
        ToolCall(tool_name=tool_name, args=args, result=result, error=error)
    )


def _is_budget_exceeded(state: AgentState) -> bool:
    return state.token_usage >= state.token_budget


def _extract_json_object(data: Any) -> dict[str, Any]:
    if isinstance(data, dict):
        return data
    if not isinstance(data, str):
        return {}
    try:
        parsed = json.loads(data)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def _issue_search_terms(title: str, body: str) -> list[str]:
    text = f"{title} {body[:1200]}"
    code_terms = re.findall(r"`([^`]{2,120})`", text)
    words = re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", text)
    stop = {
        "the",
        "and",
        "for",
        "with",
        "when",
        "this",
        "that",
        "from",
        "issue",
        "error",
        "bug",
    }
    terms: list[str] = []
    for term in code_terms + words:
        normalized = term.strip().replace("/", " ")
        if normalized.lower() in stop:
            continue
        if normalized not in terms:
            terms.append(normalized)
        if len(terms) >= 6:
            break
    return terms or [title[:120]]


def _rank_reason(path: str, issue_title: str, issue_body: str) -> tuple[float, str]:
    haystack = f"{issue_title} {issue_body}".lower()
    path_lower = path.lower()
    filename = Path(path).name.lower()
    score = 0.35
    reasons = []
    if filename and filename.rsplit(".", 1)[0] in haystack:
        score += 0.25
        reasons.append("filename appears in issue text")
    if any(part in haystack for part in path_lower.split("/")):
        score += 0.15
        reasons.append("path components match issue terms")
    if path_lower.startswith(("src/", "lib/", "app/", "packages/")):
        score += 0.1
        reasons.append("source file")
    if path_lower.startswith("tests/") or "/tests/" in path_lower:
        score += 0.05
        reasons.append("test file")
    return min(score, 1.0), ", ".join(reasons) or "matched GitHub code search"


async def understand_issue(state: AgentState | dict[str, Any]) -> AgentState:
    """Read the GitHub issue, classify it, and seed conversation memory."""
    state = _as_state(state)
    try:
        owner, repo, issue_number = parse_issue_url(state.issue_url)
    except ValueError as exc:
        state.failure_reason = str(exc)
        state.current_phase = Phase.FAILURE
        return state

    state.owner = owner
    state.repo = repo
    state.issue_number = issue_number

    try:
        issue = await read_issue(owner, repo, issue_number)
        _record_tool(
            state,
            "read_issue",
            {"owner": owner, "repo": repo, "issue_number": issue_number},
            issue,
        )
    except Exception as exc:
        _record_tool(
            state,
            "read_issue",
            {"owner": owner, "repo": repo, "issue_number": issue_number},
            error=str(exc),
        )
        state.failure_reason = f"Failed to read issue: {exc}"
        state.current_phase = Phase.FAILURE
        return state

    state.issue_title = issue.get("title", "")
    state.issue_body = issue.get("body", "")
    labels = [str(label).lower() for label in issue.get("labels", [])]
    state.issue_type = "bug" if "bug" in labels else "feature" if "feature" in labels else "unknown"
    state.severity = "high" if {"security", "critical", "regression"} & set(labels) else "medium"

    system = (
        "You classify GitHub issues for an autonomous coding agent. "
        "Return JSON with keys: type, severity, summary, likely_modules."
    )
    user = (
        f"Title: {state.issue_title}\n\nBody:\n{state.issue_body[:4000]}\n"
        f"Labels: {labels}"
    )
    try:
        analysis = _extract_json_object(await llm_call(system, user))
        state.issue_type = analysis.get("type", state.issue_type)
        state.severity = analysis.get("severity", state.severity)
        _remember(state, "assistant", json.dumps(analysis))
        state.token_usage += _estimate_tokens(system, user, json.dumps(analysis))
    except Exception as exc:
        _remember(state, "assistant", f"Issue classification skipped: {exc}")
        state.token_usage += _estimate_tokens(system, user)

    _remember(state, "user", f"{state.issue_title}\n\n{state.issue_body[:2000]}")
    state.current_phase = Phase.LOCATE
    return state


async def locate_code(state: AgentState | dict[str, Any]) -> AgentState:
    """Search code and read the most relevant files into working memory."""
    state = _as_state(state)
    if _is_budget_exceeded(state):
        state.failure_reason = "Token budget exceeded before code location."
        state.current_phase = Phase.FAILURE
        return state

    by_path: dict[str, FileInfo] = {}
    for term in _issue_search_terms(state.issue_title, state.issue_body):
        try:
            results = await search_code(term, state.owner, state.repo)
            _record_tool(
                state,
                "search_code",
                {"query": term, "owner": state.owner, "repo": state.repo},
                {"count": len(results)},
            )
        except Exception as exc:
            _record_tool(
                state,
                "search_code",
                {"query": term, "owner": state.owner, "repo": state.repo},
                error=str(exc),
            )
            continue

        for result in results:
            path = result.get("path", "")
            if not path or path in by_path:
                continue
            score, reason = _rank_reason(path, state.issue_title, state.issue_body)
            by_path[path] = FileInfo(
                path=path,
                relevance_score=score,
                reason=reason,
                sha=result.get("sha", ""),
            )

    ranked = sorted(
        by_path.values(), key=lambda item: item.relevance_score, reverse=True
    )[:6]
    hydrated: list[FileInfo] = []
    for info in ranked:
        try:
            file_data = await read_file(state.owner, state.repo, info.path)
            info.content = file_data.get("content", "")
            info.sha = file_data.get("sha", info.sha)
            hydrated.append(info)
            _record_tool(
                state,
                "read_file",
                {"owner": state.owner, "repo": state.repo, "path": info.path},
                {"size": len(info.content), "sha": info.sha},
            )
        except Exception as exc:
            _record_tool(
                state,
                "read_file",
                {"owner": state.owner, "repo": state.repo, "path": info.path},
                error=str(exc),
            )

    state.relevant_files = hydrated
    state.token_usage += _estimate_tokens(
        state.issue_title,
        state.issue_body,
        "\n".join(f"{f.path}\n{f.content[:2000]}" for f in hydrated),
    )
    state.current_phase = Phase.PLAN if hydrated else Phase.FAILURE
    if not hydrated:
        state.failure_reason = "No relevant files could be located or read."
    return state


async def plan_fix(state: AgentState | dict[str, Any]) -> AgentState:
    """Ask the LLM for a concrete patch-oriented plan."""
    state = _as_state(state)
    if _is_budget_exceeded(state):
        state.failure_reason = "Token budget exceeded before planning."
        state.current_phase = Phase.FAILURE
        return state

    previous_failures = "\n\n".join(
        f"Attempt {idx + 1}: {attempt.test_result}\n{attempt.error_log[:2000]}"
        for idx, attempt in enumerate(state.fix_attempts)
    )
    files_context = "\n\n".join(
        f"FILE: {file.path}\nRELEVANCE: {file.relevance_score} - {file.reason}\n"
        f"CONTENT:\n{file.content[:8000]}"
        for file in state.relevant_files[:4]
    )
    system = (
        "You are RepoPilot's planning node. Return ONLY JSON with keys: "
        "plan (markdown string), patch (unified diff string), files (array of paths), "
        "test_command (string). The patch should be apply-able with git apply."
    )
    user = (
        f"Issue URL: {state.issue_url}\n"
        f"Title: {state.issue_title}\n\nBody:\n{state.issue_body[:4000]}\n\n"
        f"Relevant files:\n{files_context}\n\nPrevious failures:\n{previous_failures}"
    )

    try:
        response = _extract_json_object(await llm_call(system, user))
    except Exception as exc:
        state.failure_reason = f"Failed to generate fix plan: {exc}"
        state.current_phase = Phase.FAILURE
        return state

    state.fix_plan = response.get("plan", "")
    state.patch_content = response.get("patch", "")
    state.test_command = response.get("test_command", "")
    state.token_usage += _estimate_tokens(system, user, json.dumps(response))
    _remember(state, "assistant", state.fix_plan[:2000])
    state.current_phase = Phase.EXECUTE if state.patch_content else Phase.FAILURE
    if not state.patch_content:
        state.failure_reason = "Planner did not produce a patch."
    return state


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


def _primary_patch_file(patch_content: str) -> str:
    match = re.search(r"^\+\+\+ b/(.+)$", patch_content, re.MULTILINE)
    return match.group(1) if match else ""


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


def _same_failure_seen_twice(state: AgentState) -> bool:
    if len(state.fix_attempts) < 2:
        return False
    last = state.fix_attempts[-1]
    for previous in state.fix_attempts[:-1]:
        if (
            previous.patch_content == last.patch_content
            and previous.error_log == last.error_log
            and not previous.success
            and not last.success
        ):
            return True
    return False


async def verify_fix(state: AgentState | dict[str, Any]) -> AgentState:
    """Parse test output and route to COMMIT, retry PLAN, or FAILED."""
    state = _as_state(state)
    if not state.fix_attempts:
        state.failure_reason = "No fix attempt was recorded."
        state.current_phase = Phase.FAILURE
        return state

    latest = state.fix_attempts[-1]
    if latest.success:
        state.current_phase = Phase.COMMIT
        return state

    if _same_failure_seen_twice(state):
        state.failure_reason = "Same patch produced the same failure twice."
        state.current_phase = Phase.FAILURE
        return state

    if state.retry_count >= state.max_retries:
        state.failure_reason = f"Maximum retries reached: {state.max_retries}."
        state.current_phase = Phase.FAILURE
        return state

    if _is_budget_exceeded(state):
        state.failure_reason = "Token budget exceeded during verification."
        state.current_phase = Phase.FAILURE
        return state

    state.retry_count += 1
    state.current_phase = Phase.PLAN
    return state


async def _github_create_or_update_file(
    state: AgentState, path: str, content: str, branch: str, message: str, sha: str = ""
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "message": message,
        "content": base64.b64encode(content.encode("utf-8")).decode("ascii"),
        "branch": branch,
    }
    if sha:
        payload["sha"] = sha
    url = f"{GITHUB_API}/repos/{state.owner}/{state.repo}/contents/{path}"
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.put(url, headers=_headers(), json=payload)
    resp.raise_for_status()
    return resp.json()


async def _github_get_repo(state: AgentState) -> dict[str, Any]:
    url = f"{GITHUB_API}/repos/{state.owner}/{state.repo}"
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(url, headers=_headers())
    resp.raise_for_status()
    return resp.json()


async def _github_get_ref(state: AgentState, branch: str) -> dict[str, Any]:
    url = f"{GITHUB_API}/repos/{state.owner}/{state.repo}/git/ref/heads/{branch}"
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(url, headers=_headers())
    resp.raise_for_status()
    return resp.json()


async def _github_create_ref(state: AgentState, branch: str, sha: str) -> dict[str, Any]:
    url = f"{GITHUB_API}/repos/{state.owner}/{state.repo}/git/refs"
    payload = {"ref": f"refs/heads/{branch}", "sha": sha}
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(url, headers=_headers(), json=payload)
    if resp.status_code == 422:
        return {"ref": f"refs/heads/{branch}", "already_exists": True}
    resp.raise_for_status()
    return resp.json()


async def _github_get_file_sha(state: AgentState, path: str, branch: str) -> str:
    url = f"{GITHUB_API}/repos/{state.owner}/{state.repo}/contents/{path}"
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(url, headers=_headers(), params={"ref": branch})
    if resp.status_code == 404:
        return ""
    resp.raise_for_status()
    data = resp.json()
    return data.get("sha", "") if isinstance(data, dict) else ""


async def _github_create_pr(
    state: AgentState, title: str, body: str, head: str, base: str = "main"
) -> dict[str, Any]:
    url = f"{GITHUB_API}/repos/{state.owner}/{state.repo}/pulls"
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            url,
            headers=_headers(),
            json={"title": title, "body": body, "head": head, "base": base},
        )
    resp.raise_for_status()
    return resp.json()


async def _github_add_issue_comment(state: AgentState, body: str) -> dict[str, Any]:
    url = f"{GITHUB_API}/repos/{state.owner}/{state.repo}/issues/{state.issue_number}/comments"
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(url, headers=_headers(), json={"body": body})
    resp.raise_for_status()
    return resp.json()


async def push_files(state: AgentState) -> dict[str, Any]:
    """Push changed files through GitHub Contents API."""
    if not state.repo_path:
        raise RuntimeError("Cannot push files without a local repository path.")

    branch = state.branch_name or f"repopilot-fix-{state.issue_number}"
    state.branch_name = branch

    repo_info = await _github_get_repo(state)
    base_branch = repo_info.get("default_branch") or "main"
    state.base_branch = base_branch
    base_ref = await _github_get_ref(state, base_branch)
    base_sha = base_ref.get("object", {}).get("sha", "")
    if not base_sha:
        raise RuntimeError(f"Could not resolve base branch {base_branch}.")
    await _github_create_ref(state, branch, base_sha)

    changed = subprocess.run(
        ["git", "diff", "--name-only", "HEAD"],
        cwd=state.repo_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if changed.returncode != 0:
        raise RuntimeError(changed.stderr or changed.stdout)

    changed_paths = [line.strip() for line in changed.stdout.splitlines() if line.strip()]
    if not changed_paths:
        raise RuntimeError("Patch applied but produced no changed files.")

    results = []
    for path in changed_paths:
        full_path = Path(state.repo_path) / path
        if not full_path.exists():
            continue
        content = full_path.read_text(encoding="utf-8")
        sha = await _github_get_file_sha(state, path, base_branch)
        result = await _github_create_or_update_file(
            state,
            path=path,
            content=content,
            branch=branch,
            message=f"Fix #{state.issue_number}: update {path}",
            sha=sha,
        )
        results.append({"path": path, "result": result})

    return {"branch": branch, "base": base_branch, "files": results}


async def create_pr(state: AgentState) -> dict[str, Any]:
    body = (
        f"Fixes {state.issue_url}\n\n"
        f"## Plan\n{state.fix_plan}\n\n"
        f"## Tests\n{state.fix_attempts[-1].test_result if state.fix_attempts else 'Not run'}"
    )
    return await _github_create_pr(
        state,
        title=f"Fix #{state.issue_number}: {state.issue_title}",
        body=body,
        head=state.branch_name,
        base=state.base_branch,
    )


async def commit_fix(state: AgentState | dict[str, Any]) -> AgentState:
    """Push changes and create a PR through GitHub APIs/local git."""
    state = _as_state(state)
    if not state.repo_path:
        state.failure_reason = "Cannot commit without a local repository path."
        state.current_phase = Phase.FAILURE
        return state

    try:
        pushed = await push_files(state)
        _record_tool(state, "push_files", {"branch": state.branch_name}, pushed)
        pr = await create_pr(state)
        _record_tool(
            state,
            "create_pr",
            {"head": state.branch_name, "base": state.base_branch},
            pr,
        )
        state.pr_url = pr.get("html_url") or pr.get("url")
        state.current_phase = Phase.DONE
    except Exception as exc:
        _record_tool(state, "commit_fix", {}, error=str(exc))
        state.failure_reason = f"Failed to push or create PR: {exc}"
        state.current_phase = Phase.FAILURE
    return state


async def handle_failure(state: AgentState | dict[str, Any]) -> AgentState:
    """Gracefully report partial progress as an issue comment."""
    state = _as_state(state)
    files = "\n".join(f"- {file.path}: {file.reason}" for file in state.relevant_files[:6])
    attempts = "\n\n".join(
        f"Attempt {idx + 1}: success={attempt.success}\n"
        f"File: {attempt.file_path or 'unknown'}\n"
        f"Result: {attempt.test_result}\n"
        f"Error:\n{attempt.error_log[:1500]}"
        for idx, attempt in enumerate(state.fix_attempts)
    )
    body = (
        "RepoPilot v2 could not complete an automatic fix.\n\n"
        f"Reason: {state.failure_reason or 'unspecified failure'}\n\n"
        f"Relevant files:\n{files or 'None found'}\n\n"
        f"Attempts:\n{attempts or 'No patch attempts were made.'}\n\n"
        f"Token usage: {state.token_usage}/{state.token_budget}"
    )
    if state.owner and state.repo and state.issue_number:
        try:
            await _github_add_issue_comment(state, body)
            _record_tool(
                state,
                "add_issue_comment",
                {"issue_number": state.issue_number},
                {"ok": True},
            )
        except Exception as exc:
            _record_tool(
                state,
                "add_issue_comment",
                {"issue_number": state.issue_number},
                error=str(exc),
            )
    state.current_phase = Phase.FAILED
    return state


def route_from_state(state: AgentState | dict[str, Any]) -> str:
    current = _as_state(state).current_phase
    if current == Phase.UNDERSTAND:
        return "understand_issue"
    if current == Phase.LOCATE:
        return "locate_code"
    if current == Phase.PLAN:
        return "plan_fix"
    if current == Phase.EXECUTE:
        return "execute_fix"
    if current == Phase.VERIFY:
        return "verify_fix"
    if current == Phase.COMMIT:
        return "commit_fix"
    if current == Phase.FAILURE:
        return "handle_failure"
    return END


class FallbackCompiledGraph:
    """Minimal async graph runner matching the LangGraph node contract."""

    def __init__(self, nodes: dict[str, NodeFn], start: str):
        self.nodes = nodes
        self.start = start

    async def ainvoke(self, state: AgentState | dict[str, Any]) -> AgentState:
        current = self.start
        working = _as_state(state)
        guard = 0
        while current != END:
            guard += 1
            if guard > 64:
                working.failure_reason = "State graph guard limit reached."
                working.current_phase = Phase.FAILED
                return working
            node = self.nodes[current]
            working = _as_state(await node(working))
            current = route_from_state(working)
        return working


class FallbackStateGraph:
    def __init__(self) -> None:
        self.nodes: dict[str, NodeFn] = {}
        self.start = ""

    def add_node(self, name: str, fn: NodeFn) -> None:
        self.nodes[name] = fn

    def set_entry_point(self, name: str) -> None:
        self.start = name

    def compile(self) -> FallbackCompiledGraph:
        return FallbackCompiledGraph(self.nodes, self.start)


def build_agent_graph() -> Any:
    """Build the RepoPilot v2 state graph."""
    if StateGraph is None:
        graph = FallbackStateGraph()
        for name, fn in {
            "understand_issue": understand_issue,
            "locate_code": locate_code,
            "plan_fix": plan_fix,
            "execute_fix": execute_fix,
            "verify_fix": verify_fix,
            "commit_fix": commit_fix,
            "handle_failure": handle_failure,
        }.items():
            graph.add_node(name, fn)
        graph.set_entry_point("understand_issue")
        return graph.compile()

    graph = StateGraph(AgentState)
    graph.add_node("understand_issue", understand_issue)
    graph.add_node("locate_code", locate_code)
    graph.add_node("plan_fix", plan_fix)
    graph.add_node("execute_fix", execute_fix)
    graph.add_node("verify_fix", verify_fix)
    graph.add_node("commit_fix", commit_fix)
    graph.add_node("handle_failure", handle_failure)
    for node in [
        "understand_issue",
        "locate_code",
        "plan_fix",
        "execute_fix",
        "verify_fix",
        "commit_fix",
        "handle_failure",
    ]:
        graph.add_conditional_edges(
            node,
            route_from_state,
            {
                "understand_issue": "understand_issue",
                "locate_code": "locate_code",
                "plan_fix": "plan_fix",
                "execute_fix": "execute_fix",
                "verify_fix": "verify_fix",
                "commit_fix": "commit_fix",
                "handle_failure": "handle_failure",
                END: END,
            },
        )
    graph.set_entry_point("understand_issue")
    return graph.compile()


async def run_graph(graph: Any, state: AgentState) -> AgentState:
    result = await graph.ainvoke(state)
    return _as_state(result)


def final_report_from_state(state: AgentState, turns_taken: int) -> FinalReport:
    return FinalReport(
        issue_url=state.issue_url,
        fix_applied=state.current_phase == Phase.DONE,
        pr_url=state.pr_url,
        test_results=state.fix_attempts[-1].test_result if state.fix_attempts else "",
        turns_taken=turns_taken,
        token_used=state.token_usage,
    )


async def agent_v2(issue_url: str, max_retries: int = 3, token_budget: int = 50000) -> dict[str, Any]:
    tracer = Tracer()
    state = AgentState(
        issue_url=issue_url,
        max_retries=max_retries,
        token_budget=token_budget,
        trace_id=tracer.trace_id,
    )
    graph = build_agent_graph()
    final_state = await run_graph(graph, state)
    report = final_report_from_state(final_state, len(final_state.tool_calls))
    tracer.log(
        "agent_v2_done",
        {"issue_url": issue_url},
        {"phase": final_state.current_phase.value, "pr_url": final_state.pr_url},
        error=final_state.failure_reason or None,
    )
    payload = report.model_dump()
    payload.update(
        {
            "done": final_state.current_phase in {Phase.DONE, Phase.FAILED},
            "success": final_state.current_phase == Phase.DONE,
            "final_phase": final_state.current_phase.value,
            "trace_id": tracer.trace_id,
            "relevant_files": [file.model_dump() for file in final_state.relevant_files],
            "fix_attempts": [attempt.model_dump() for attempt in final_state.fix_attempts],
            "error": final_state.failure_reason or None,
        }
    )
    return payload


async def intelligent_analyze_issue(
    issue_url: str, max_retries: int = 3, token_budget: int = 50000
) -> dict[str, Any]:
    """Backward-compatible alias for the previous experimental endpoint."""
    return await agent_v2(issue_url, max_retries=max_retries, token_budget=token_budget)


if __name__ == "__main__":  # pragma: no cover
    print(
        asyncio.run(
            agent_v2(
                "https://github.com/example/repo/issues/1",
                max_retries=1,
                token_budget=10000,
            )
        )
    )
