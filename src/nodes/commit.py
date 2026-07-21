"""COMMIT phase: Push changes and create a PR through GitHub APIs."""

from __future__ import annotations

import base64
from typing import Any

import httpx

from ..memory import _fire_and_forget, get_store
from ..coverage_gate import LiveCoverageBinding, validate_live_coverage_binding
from ..state import AgentState, Phase, _as_state, _record_tool
from ..tools import GITHUB_API, _headers


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


async def push_files(
    state: AgentState,
    binding: LiveCoverageBinding | None = None,
) -> dict[str, Any]:
    """Push changed files through GitHub Contents API."""
    if not state.repo_path:
        raise RuntimeError("Cannot push files without a local repository path.")
    binding = binding or validate_live_coverage_binding(state)
    if not binding.approved_targets:
        raise RuntimeError("Patch applied but produced no approved changed files.")
    if any(target.change == "deleted" for target in binding.approved_targets):
        raise RuntimeError("Approved deletions are unsupported by the commit boundary.")

    branch = state.branch_name or f"repopilot-fix-{state.issue_number}"
    state.branch_name = branch

    repo_info = await _github_get_repo(state)
    base_branch = repo_info.get("default_branch") or "main"
    state.base_branch = base_branch
    base_ref = await _github_get_ref(state, base_branch)
    base_sha = base_ref.get("object", {}).get("sha", "")
    if base_sha != state.repo_ref:
        raise RuntimeError(f"Base branch {base_branch} drifted from the approved commit.")
    await _github_create_ref(state, branch, base_sha)

    results = []
    for target in binding.approved_targets:
        if target.content is None or target.content_sha256 is None:
            raise RuntimeError("Approved commit target has no exact content.")
        sha = target.base_blob_sha if target.change == "modified" else ""
        if target.change == "modified" and not sha:
            raise RuntimeError("Approved modified target lacks its base blob SHA.")
        result = await _github_create_or_update_file(
            state,
            path=target.path,
            content=target.content,
            branch=branch,
            message=f"Fix #{state.issue_number}: update {target.path}",
            sha=sha or "",
        )
        results.append({"path": target.path, "result": result})

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
        binding = validate_live_coverage_binding(state)
        pushed = await push_files(state, binding)
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

        # ── fire-and-forget memory recording ──
        store = get_store()
        for f in pushed.get("files", []):
            _fire_and_forget(
                store.record_file(state.owner, state.repo, f["path"])
            )
        _fire_and_forget(
            store.record_issue(
                state.owner, state.repo, state.issue_number, success=True
            )
        )
    except Exception as exc:
        _record_tool(state, "commit_fix", {}, error=str(exc))
        state.failure_reason = f"Failed to push or create PR: {exc}"
        state.current_phase = Phase.FAILURE
    return state
