"""COMMIT phase: Push changes and create a PR through GitHub APIs."""

from __future__ import annotations

import base64
import hashlib
import re
import secrets
from typing import Any
from urllib.parse import quote

import httpx

from ..coverage_gate import LiveCoverageBinding, validate_terminal_coverage_binding
from ..memory import _fire_and_forget, get_store
from ..state import AgentState, Phase, _as_state, _record_tool
from ..tools import GITHUB_API, _headers

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_REPAIR_BRANCH_RE = re.compile(
    r"^repopilot-fix-[0-9]+-[0-9a-f]{12}-[0-9a-f]{16}$"
)


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
    encoded = quote(branch, safe="")
    url = f"{GITHUB_API}/repos/{state.owner}/{state.repo}/git/ref/heads/{encoded}"
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(url, headers=_headers())
    resp.raise_for_status()
    return resp.json()


async def _github_get_commit(state: AgentState, sha: str) -> dict[str, Any]:
    url = f"{GITHUB_API}/repos/{state.owner}/{state.repo}/git/commits/{sha}"
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(url, headers=_headers())
    resp.raise_for_status()
    return resp.json()


async def _github_get_tree(state: AgentState, sha: str) -> dict[str, Any]:
    url = f"{GITHUB_API}/repos/{state.owner}/{state.repo}/git/trees/{sha}"
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(url, headers=_headers(), params={"recursive": "1"})
    resp.raise_for_status()
    return resp.json()


async def _github_get_blob(state: AgentState, sha: str) -> dict[str, Any]:
    url = f"{GITHUB_API}/repos/{state.owner}/{state.repo}/git/blobs/{sha}"
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


def _repair_branch_prefix(state: AgentState) -> str:
    trace_binding = hashlib.sha256(
        (
            f"{state.owner}/{state.repo}#{state.issue_number}:"
            f"{state.trace_id}"
        ).encode("utf-8")
    ).hexdigest()[:12]
    return f"repopilot-fix-{state.issue_number}-{trace_binding}"


def _repair_branch_name(state: AgentState) -> str:
    return f"{_repair_branch_prefix(state)}-{secrets.token_hex(8)}"


def _validated_repair_branch(state: AgentState) -> str:
    if not state.branch_name:
        state.branch_name = _repair_branch_name(state)
    expected_prefix = _repair_branch_prefix(state) + "-"
    if (
        not _REPAIR_BRANCH_RE.fullmatch(state.branch_name)
        or not state.branch_name.startswith(expected_prefix)
    ):
        raise RuntimeError("repair branch is not bound to this run")
    return state.branch_name


def _exact_ref_sha(payload: object, branch: str) -> str:
    if not isinstance(payload, dict) or payload.get("ref") != f"refs/heads/{branch}":
        raise RuntimeError("GitHub returned a mismatched branch reference")
    nested = payload.get("object")
    sha = nested.get("sha") if isinstance(nested, dict) else None
    if not isinstance(sha, str) or not _COMMIT_RE.fullmatch(sha.lower()):
        raise RuntimeError("GitHub returned an invalid branch head")
    return sha.lower()


def _git_blob_sha(content: bytes) -> str:
    header = f"blob {len(content)}\0".encode("ascii")
    # GitHub's Git object API identifies blobs by the repository's SHA-1 object
    # format. This is an identity check, not a new password/credential digest.
    return hashlib.sha1(header + content, usedforsecurity=False).hexdigest()


def _remote_tree_map(payload: object, expected_sha: str) -> dict[str, tuple[str, str, str]]:
    if (
        not isinstance(payload, dict)
        or payload.get("sha") != expected_sha
        or payload.get("truncated") is not False
        or not isinstance(payload.get("tree"), list)
    ):
        raise RuntimeError("remote branch tree response is incomplete")
    result: dict[str, tuple[str, str, str]] = {}
    for raw in payload["tree"]:
        if not isinstance(raw, dict):
            raise RuntimeError("remote branch tree response is malformed")
        path = raw.get("path")
        mode = raw.get("mode")
        kind = raw.get("type")
        sha = raw.get("sha")
        if kind == "tree":
            continue
        if (
            not isinstance(path, str)
            or not path
            or path in result
            or not isinstance(mode, str)
            or not isinstance(kind, str)
            or not isinstance(sha, str)
            or not _COMMIT_RE.fullmatch(sha.lower())
        ):
            raise RuntimeError("remote branch tree response is malformed")
        result[path] = (mode, kind, sha.lower())
    return result


def _commit_tree_sha(payload: object, expected_commit: str) -> str:
    if not isinstance(payload, dict) or payload.get("sha") != expected_commit:
        raise RuntimeError("remote commit identity mismatch")
    tree = payload.get("tree")
    sha = tree.get("sha") if isinstance(tree, dict) else None
    if not isinstance(sha, str) or not _COMMIT_RE.fullmatch(sha.lower()):
        raise RuntimeError("remote commit tree identity is invalid")
    return sha.lower()


def _decode_remote_blob(payload: object, expected_sha: str) -> bytes:
    if (
        not isinstance(payload, dict)
        or payload.get("sha") != expected_sha
        or payload.get("encoding") != "base64"
        or not isinstance(payload.get("content"), str)
        or not isinstance(payload.get("size"), int)
    ):
        raise RuntimeError("remote branch blob response is malformed")
    try:
        encoded = "".join(payload["content"].splitlines()).encode("ascii")
        content = base64.b64decode(encoded, validate=True)
    except (UnicodeEncodeError, ValueError) as exc:
        raise RuntimeError("remote branch blob response is malformed") from exc
    if len(content) != payload["size"] or _git_blob_sha(content) != expected_sha:
        raise RuntimeError("remote branch blob identity mismatch")
    return content


async def _verify_remote_branch(
    state: AgentState,
    binding: LiveCoverageBinding,
    base_sha: str,
    branch: str,
) -> str:
    """Verify the remote tree and target blobs exactly match the approval."""
    if base_sha.lower() != state.repo_ref.lower() or not _COMMIT_RE.fullmatch(
        base_sha.lower()
    ):
        raise RuntimeError("remote verification base does not match approval")
    if not _REPAIR_BRANCH_RE.fullmatch(branch):
        raise RuntimeError("remote verification branch is unsafe")
    head_ref = await _github_get_ref(state, branch)
    head_sha = _exact_ref_sha(head_ref, branch)
    base_commit = await _github_get_commit(state, base_sha)
    head_commit = await _github_get_commit(state, head_sha)
    base_tree_sha = _commit_tree_sha(base_commit, base_sha)
    head_tree_sha = _commit_tree_sha(head_commit, head_sha)
    base_tree = _remote_tree_map(
        await _github_get_tree(state, base_tree_sha), base_tree_sha
    )
    head_tree = _remote_tree_map(
        await _github_get_tree(state, head_tree_sha), head_tree_sha
    )

    expected_tree = dict(base_tree)
    for target in binding.approved_targets:
        if (
            target.change == "deleted"
            or target.content is None
            or target.content_sha256 is None
        ):
            raise RuntimeError("remote verification does not support deletions")
        base_entry = base_tree.get(target.path)
        if target.change == "added" and base_entry is not None:
            raise RuntimeError("remote base tree contradicts approved added target")
        if target.change == "modified" and (
            base_entry is None
            or base_entry[1] != "blob"
            or base_entry[2] != target.base_blob_sha
        ):
            raise RuntimeError("remote base tree contradicts approved modified target")
        content = target.content.encode("utf-8")
        expected_sha = _git_blob_sha(content)
        expected_tree[target.path] = (target.mode, "blob", expected_sha)
    if head_tree != expected_tree:
        raise RuntimeError("remote branch tree does not match approved patch manifest")

    for target in binding.approved_targets:
        if target.content is None or target.content_sha256 is None:
            raise RuntimeError("remote verification target content is incomplete")
        content = target.content.encode("utf-8")
        expected_sha = _git_blob_sha(content)
        remote = _decode_remote_blob(
            await _github_get_blob(state, expected_sha), expected_sha
        )
        if (
            remote != content
            or len(remote) != target.size
            or hashlib.sha256(remote).hexdigest() != target.content_sha256
        ):
            raise RuntimeError(
                "remote branch blob content does not match approved patch manifest"
            )
    return head_sha


async def push_files(
    state: AgentState,
    binding: LiveCoverageBinding | None = None,
) -> dict[str, Any]:
    """Push changed files through GitHub Contents API."""
    if not state.repo_path:
        raise RuntimeError("Cannot push files without a local repository path.")
    terminal_binding = validate_terminal_coverage_binding(state)
    if binding is not None and binding != terminal_binding:
        raise RuntimeError("Commit binding changed after terminal validation.")
    binding = terminal_binding
    if not binding.approved_targets:
        raise RuntimeError("Patch applied but produced no approved changed files.")
    if any(target.change == "deleted" for target in binding.approved_targets):
        raise RuntimeError("Approved deletions are unsupported by the commit boundary.")

    branch = _validated_repair_branch(state)

    repo_info = await _github_get_repo(state)
    base_branch = repo_info.get("default_branch") or "main"
    state.base_branch = base_branch
    base_ref = await _github_get_ref(state, base_branch)
    base_sha = _exact_ref_sha(base_ref, base_branch)
    if base_sha != state.repo_ref.lower():
        raise RuntimeError(f"Base branch {base_branch} drifted from the approved commit.")
    created = await _github_create_ref(state, branch, base_sha)
    branch_ref = await _github_get_ref(state, branch)
    branch_sha = _exact_ref_sha(branch_ref, branch)
    if branch_sha != base_sha:
        if isinstance(created, dict) and created.get("already_exists") is True:
            raise RuntimeError("repair branch collision points to an unrelated head")
        raise RuntimeError("created repair branch does not point to the approved base")

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

    head_sha = await _verify_remote_branch(state, binding, base_sha, branch)
    return {
        "branch": branch,
        "base": base_branch,
        "base_sha": base_sha,
        "head_sha": head_sha,
        "manifest_fingerprint": binding.manifest_fingerprint,
        "files": results,
    }


async def create_pr(
    state: AgentState,
    binding: LiveCoverageBinding | None = None,
) -> dict[str, Any]:
    terminal_binding = validate_terminal_coverage_binding(state)
    if binding is not None and binding != terminal_binding:
        raise RuntimeError("Commit binding changed before PR creation.")
    binding = terminal_binding
    base_ref = await _github_get_ref(state, state.base_branch)
    if _exact_ref_sha(base_ref, state.base_branch) != state.repo_ref.lower():
        raise RuntimeError("Base branch drifted before PR creation.")
    await _verify_remote_branch(
        state,
        binding,
        state.repo_ref,
        state.branch_name,
    )
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
        binding = validate_terminal_coverage_binding(state)
        pushed = await push_files(state, binding)
        _record_tool(state, "push_files", {"branch": state.branch_name}, pushed)
        pr = await create_pr(state, binding)
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
