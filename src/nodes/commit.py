"""COMMIT phase: Push changes and create a PR through GitHub APIs."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import re
import secrets
from collections.abc import Coroutine
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import httpx

from ..async_safety import CancellationDrainError, drain_task
from ..coverage_gate import LiveCoverageBinding, validate_terminal_coverage_binding
from ..memory import _fire_and_forget, get_store
from ..state import AgentState, Phase, _as_state, _record_tool
from ..tools import GITHUB_API, _headers

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_REPAIR_BRANCH_RE = re.compile(
    r"^repopilot-fix-[0-9]+-[0-9a-f]{12}-[0-9a-f]{16}$"
)
_PR_RECONCILE_DELAYS = (0.25, 0.75)


class PRCleanupError(RuntimeError):
    """A rejected PR could not be closed after validation failed."""

    def __init__(
        self,
        pr_number: int | None,
        validation_error: BaseException,
        cleanup_error: BaseException,
    ) -> None:
        label = str(pr_number) if pr_number is not None else "unknown"
        super().__init__(
            f"pull request {label} validation failed and cleanup failed"
        )
        self.pr_number = pr_number
        self.validation_error = validation_error
        self.cleanup_error = cleanup_error


class PRCancellationCleanupError(CancellationDrainError):
    """Cancellation was preserved, but PR cleanup also failed."""

    def __init__(
        self,
        pr_number: int | None,
        cancellation: asyncio.CancelledError,
        cleanup_error: BaseException,
        *,
        transaction_error: BaseException | None = None,
    ) -> None:
        label = str(pr_number) if pr_number is not None else "unknown"
        super().__init__(
            f"pull request {label} cleanup", cancellation, cleanup_error
        )
        self.pr_number = pr_number
        self.transaction_error = transaction_error


class PRCancellationTransactionError(CancellationDrainError):
    """Cancellation was preserved, but the in-flight PR transaction failed."""

    def __init__(
        self,
        pr_number: int | None,
        cancellation: asyncio.CancelledError,
        transaction_error: BaseException,
    ) -> None:
        label = str(pr_number) if pr_number is not None else "unknown"
        super().__init__(
            f"pull request {label} transaction",
            cancellation,
            transaction_error,
        )
        self.pr_number = pr_number
        self.transaction_error = transaction_error


def _repo_api_root(state: AgentState) -> str:
    owner = quote(state.owner, safe="")
    repo = quote(state.repo, safe="")
    return f"{GITHUB_API}/repos/{owner}/{repo}"


def _contents_url(state: AgentState, path: str) -> str:
    encoded_path = quote(path, safe="/")
    return f"{_repo_api_root(state)}/contents/{encoded_path}"


def _canonical_repo_identity(
    payload: object, state: AgentState
) -> dict[str, str]:
    full_name = payload.get("full_name") if isinstance(payload, dict) else None
    if not isinstance(full_name, str) or full_name.count("/") != 1:
        raise RuntimeError("GitHub repository identity is malformed")
    owner, repo = full_name.split("/", 1)
    if (
        not owner
        or not repo
        or owner.casefold() != state.owner.casefold()
        or repo.casefold() != state.repo.casefold()
    ):
        raise RuntimeError("GitHub repository identity does not match the issue")
    return {
        "owner": owner,
        "repo": repo,
        "full_name": full_name,
        "html_url": (
            "https://github.com/"
            f"{quote(owner, safe='')}/{quote(repo, safe='')}"
        ),
        "api_url": (
            f"{GITHUB_API}/repos/"
            f"{quote(owner, safe='')}/{quote(repo, safe='')}"
        ),
    }


def _validate_bound_repo_identity(
    identity: object, state: AgentState
) -> dict[str, str]:
    if not isinstance(identity, dict):
        raise RuntimeError("bound GitHub repository identity is malformed")
    rebound = _canonical_repo_identity(
        {"full_name": identity.get("full_name")}, state
    )
    if identity != rebound:
        raise RuntimeError("bound GitHub repository identity is malformed")
    return rebound


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
    url = _contents_url(state, path)
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.put(url, headers=_headers(), json=payload)
    resp.raise_for_status()
    return resp.json()


async def _github_get_repo(state: AgentState) -> dict[str, Any]:
    url = _repo_api_root(state)
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(url, headers=_headers())
    resp.raise_for_status()
    return resp.json()


async def _github_get_ref(state: AgentState, branch: str) -> dict[str, Any]:
    encoded = quote(branch, safe="")
    url = f"{_repo_api_root(state)}/git/ref/heads/{encoded}"
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(url, headers=_headers())
    resp.raise_for_status()
    return resp.json()


async def _github_get_commit(state: AgentState, sha: str) -> dict[str, Any]:
    url = f"{_repo_api_root(state)}/git/commits/{quote(sha, safe='')}"
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(url, headers=_headers())
    resp.raise_for_status()
    return resp.json()


async def _github_get_tree(state: AgentState, sha: str) -> dict[str, Any]:
    url = f"{_repo_api_root(state)}/git/trees/{quote(sha, safe='')}"
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(url, headers=_headers(), params={"recursive": "1"})
    resp.raise_for_status()
    return resp.json()


async def _github_get_blob(state: AgentState, sha: str) -> dict[str, Any]:
    url = f"{_repo_api_root(state)}/git/blobs/{quote(sha, safe='')}"
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(url, headers=_headers())
    resp.raise_for_status()
    return resp.json()


async def _github_create_ref(state: AgentState, branch: str, sha: str) -> dict[str, Any]:
    url = f"{_repo_api_root(state)}/git/refs"
    payload = {"ref": f"refs/heads/{branch}", "sha": sha}
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(url, headers=_headers(), json=payload)
    if resp.status_code == 422:
        return {"ref": f"refs/heads/{branch}", "already_exists": True}
    resp.raise_for_status()
    return resp.json()


async def _github_create_pr(
    state: AgentState, title: str, body: str, head: str, base: str = "main"
) -> dict[str, Any]:
    url = f"{_repo_api_root(state)}/pulls"
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            url,
            headers=_headers(),
            json={"title": title, "body": body, "head": head, "base": base},
        )
    resp.raise_for_status()
    return resp.json()


async def _github_get_pr(state: AgentState, number: int) -> dict[str, Any]:
    url = f"{_repo_api_root(state)}/pulls/{number}"
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(url, headers=_headers())
    resp.raise_for_status()
    return resp.json()


async def _github_list_open_prs(
    state: AgentState,
    repository_identity: dict[str, str],
    head: str,
    base: str,
) -> object:
    identity = _validate_bound_repo_identity(repository_identity, state)
    url = f"{identity['api_url']}/pulls"
    params = {
        "state": "open",
        "head": f"{identity['owner']}:{head}",
        "base": base,
        "per_page": "100",
    }
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(url, headers=_headers(), params=params)
    resp.raise_for_status()
    return resp.json()


async def _github_close_pr(state: AgentState, number: int) -> None:
    url = f"{_repo_api_root(state)}/pulls/{number}"
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.patch(url, headers=_headers(), json={"state": "closed"})
    resp.raise_for_status()


async def _github_add_issue_comment(state: AgentState, body: str) -> dict[str, Any]:
    url = f"{_repo_api_root(state)}/issues/{state.issue_number}/comments"
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


TreeEntry = tuple[str, str, str, int | None]

_TREE_MODE_TYPES = {
    ("040000", "tree"),
    ("100644", "blob"),
    ("100755", "blob"),
    ("120000", "blob"),
    ("160000", "commit"),
}


def _git_tree_sha(entries: list[tuple[str, str, str]]) -> str:
    ordered = sorted(
        entries,
        key=lambda item: (
            item[1] + ("/" if item[0] == "040000" else "")
        ).encode("utf-8"),
    )
    body = b"".join(
        mode.lstrip("0").encode("ascii")
        + b" "
        + name.encode("utf-8")
        + b"\0"
        + bytes.fromhex(sha)
        for mode, name, sha in ordered
    )
    header = f"tree {len(body)}\0".encode("ascii")
    return hashlib.sha1(header + body, usedforsecurity=False).hexdigest()


def _valid_tree_path(path: object) -> bool:
    if (
        not isinstance(path, str)
        or not path
        or path.startswith("/")
        or path.endswith("/")
        or "//" in path
        or "\\" in path
        or "\0" in path
    ):
        return False
    return all(part not in {"", ".", ".."} for part in path.split("/"))


def _tree_children(
    entries: dict[str, TreeEntry], parent: str
) -> list[tuple[str, str, str]]:
    children: list[tuple[str, str, str]] = []
    for path, (mode, _kind, sha, _size) in entries.items():
        entry_parent, _, name = path.rpartition("/")
        if entry_parent == parent:
            children.append((mode, name, sha))
    return children


def _prove_complete_tree(entries: dict[str, TreeEntry], expected_root: str) -> None:
    tree_paths = {path for path, entry in entries.items() if entry[1] == "tree"}
    for path in entries:
        parent, _, _name = path.rpartition("/")
        if parent and parent not in tree_paths:
            raise RuntimeError("remote branch tree response is malformed")
    for path in sorted(tree_paths, key=lambda value: value.count("/"), reverse=True):
        children = _tree_children(entries, path)
        if not children:
            raise RuntimeError("remote branch tree contains an extra empty tree")
        if _git_tree_sha(children) != entries[path][2]:
            raise RuntimeError("remote branch nested tree identity mismatch")
    if _git_tree_sha(_tree_children(entries, "")) != expected_root:
        raise RuntimeError("remote branch root tree identity mismatch")


def _remote_tree_map(payload: object, expected_sha: str) -> dict[str, TreeEntry]:
    if (
        not isinstance(payload, dict)
        or payload.get("sha") != expected_sha
        or payload.get("truncated") is not False
        or not isinstance(payload.get("tree"), list)
    ):
        raise RuntimeError("remote branch tree response is incomplete")
    result: dict[str, TreeEntry] = {}
    for raw in payload["tree"]:
        if not isinstance(raw, dict):
            raise RuntimeError("remote branch tree response is malformed")
        path = raw.get("path")
        mode = raw.get("mode")
        kind = raw.get("type")
        sha = raw.get("sha")
        size = raw.get("size")
        if (
            not _valid_tree_path(path)
            or path in result
            or (mode, kind) not in _TREE_MODE_TYPES
            or not isinstance(sha, str)
            or not _COMMIT_RE.fullmatch(sha)
            or (kind == "blob" and (type(size) is not int or size < 0))
            or (kind != "blob" and "size" in raw)
        ):
            raise RuntimeError("remote branch tree response is malformed")
        result[path] = (mode, kind, sha, size if kind == "blob" else None)
    _prove_complete_tree(result, expected_sha)
    return result


def _build_tree_from_leaves(
    leaves: dict[str, TreeEntry],
) -> tuple[dict[str, TreeEntry], str]:
    entries = dict(leaves)
    directories: set[str] = set()
    for path in leaves:
        parts = path.split("/")[:-1]
        for index in range(1, len(parts) + 1):
            directories.add("/".join(parts[:index]))
    for directory in sorted(
        directories, key=lambda value: value.count("/"), reverse=True
    ):
        children = _tree_children(entries, directory)
        if not children:
            raise RuntimeError("approved tree contains an empty directory")
        entries[directory] = ("040000", "tree", _git_tree_sha(children), None)
    return entries, _git_tree_sha(_tree_children(entries, ""))


def _commit_tree_sha(payload: object, expected_commit: str) -> str:
    if not isinstance(payload, dict) or payload.get("sha") != expected_commit:
        raise RuntimeError("remote commit identity mismatch")
    tree = payload.get("tree")
    sha = tree.get("sha") if isinstance(tree, dict) else None
    if not isinstance(sha, str) or not _COMMIT_RE.fullmatch(sha.lower()):
        raise RuntimeError("remote commit tree identity is invalid")
    return sha.lower()


def _commit_parents(payload: object, expected_commit: str) -> tuple[str, ...]:
    if not isinstance(payload, dict) or payload.get("sha") != expected_commit:
        raise RuntimeError("remote commit identity mismatch")
    parents = payload.get("parents")
    if not isinstance(parents, list):
        raise RuntimeError("remote commit ancestry is malformed")
    result: list[str] = []
    for parent in parents:
        sha = parent.get("sha") if isinstance(parent, dict) else None
        if not isinstance(sha, str) or not _COMMIT_RE.fullmatch(sha):
            raise RuntimeError("remote commit ancestry is malformed")
        result.append(sha)
    return tuple(result)


def _contents_commit_identity(payload: object, expected_parent: str) -> str:
    commit = payload.get("commit") if isinstance(payload, dict) else None
    if not isinstance(commit, dict):
        raise RuntimeError("GitHub Contents response omitted commit identity")
    sha = commit.get("sha")
    if not isinstance(sha, str) or not _COMMIT_RE.fullmatch(sha):
        raise RuntimeError("GitHub Contents response has invalid commit identity")
    if _commit_parents(commit, sha) != (expected_parent,):
        raise RuntimeError("GitHub Contents commit is not the exact linear successor")
    return sha


def _decode_remote_blob(payload: object, expected_sha: str) -> bytes:
    if (
        not isinstance(payload, dict)
        or payload.get("sha") != expected_sha
        or payload.get("encoding") != "base64"
        or not isinstance(payload.get("content"), str)
        or type(payload.get("size")) is not int
        or payload["size"] < 0
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


def _pr_number(payload: object) -> int | None:
    number = payload.get("number") if isinstance(payload, dict) else None
    if type(number) is int and number > 0:
        return number
    return None


def _validate_pr_identity(
    payload: object,
    state: AgentState,
    number: int,
    verified_head_sha: str,
    repository_identity: dict[str, str],
) -> dict[str, Any]:
    if not isinstance(payload, dict) or _pr_number(payload) != number:
        raise RuntimeError("GitHub pull request identity is malformed")
    identity = _validate_bound_repo_identity(repository_identity, state)
    expected_html_url = f"{identity['html_url']}/pull/{number}"
    expected_api_url = f"{identity['api_url']}/pulls/{number}"
    if (
        payload.get("html_url") != expected_html_url
        or payload.get("url") != expected_api_url
    ):
        raise RuntimeError("GitHub pull request identity is malformed")
    expected_repo = identity["full_name"]
    for side_name, expected_sha, expected_ref in (
        ("head", verified_head_sha, state.branch_name),
        ("base", state.repo_ref.lower(), state.base_branch),
    ):
        side = payload.get(side_name)
        repo = side.get("repo") if isinstance(side, dict) else None
        if (
            not isinstance(side, dict)
            or side.get("sha") != expected_sha
            or side.get("ref") != expected_ref
            or not isinstance(repo, dict)
            or repo.get("full_name") != expected_repo
        ):
            raise RuntimeError("GitHub pull request identity does not match verified refs")
    return payload


async def _assert_pr_refs_unchanged(
    state: AgentState, verified_head_sha: str
) -> None:
    base_sha = _exact_ref_sha(
        await _github_get_ref(state, state.base_branch), state.base_branch
    )
    head_sha = _exact_ref_sha(
        await _github_get_ref(state, state.branch_name), state.branch_name
    )
    if base_sha != state.repo_ref.lower() or head_sha != verified_head_sha:
        raise RuntimeError("base or repair branch moved during pull request creation")


def _exact_open_pr_numbers(
    payload: object,
    state: AgentState,
    repository_identity: dict[str, str],
    verified_head_sha: str,
) -> frozenset[int]:
    if not isinstance(payload, list) or len(payload) > 100:
        raise RuntimeError("GitHub pull request reconciliation is malformed")
    identity = _validate_bound_repo_identity(repository_identity, state)
    numbers: set[int] = set()
    for raw in payload:
        if not isinstance(raw, dict):
            raise RuntimeError("GitHub pull request reconciliation is malformed")
        head = raw.get("head")
        base = raw.get("base")
        head_repo = head.get("repo") if isinstance(head, dict) else None
        base_repo = base.get("repo") if isinstance(base, dict) else None
        is_exact_candidate = (
            isinstance(head, dict)
            and isinstance(base, dict)
            and isinstance(head_repo, dict)
            and isinstance(base_repo, dict)
            and head.get("sha") == verified_head_sha
            and head.get("ref") == state.branch_name
            and head_repo.get("full_name") == identity["full_name"]
            and base.get("sha") == state.repo_ref.lower()
            and base.get("ref") == state.base_branch
            and base_repo.get("full_name") == identity["full_name"]
        )
        if not is_exact_candidate:
            continue
        number = _pr_number(raw)
        if number is None:
            raise RuntimeError("GitHub pull request reconciliation is malformed")
        _validate_pr_identity(
            raw, state, number, verified_head_sha, identity
        )
        numbers.add(number)
    return frozenset(numbers)


@dataclass
class _PRTransactionOutcome:
    response: object | None = None
    number: int | None = None
    post_identity_validated: bool = False
    post_definitively_rejected: bool = False


@dataclass
class _ShieldedCleanupOutcome:
    error: BaseException | None = None
    delayed_cancellation: asyncio.CancelledError | None = None


async def _execute_pr_transaction(
    state: AgentState,
    title: str,
    body: str,
    verified_head_sha: str,
    repository_identity: dict[str, str],
    outcome: _PRTransactionOutcome,
) -> dict[str, Any]:
    try:
        outcome.response = await _github_create_pr(
            state,
            title=title,
            body=body,
            head=state.branch_name,
            base=state.base_branch,
        )
    except httpx.HTTPStatusError as exc:
        if 400 <= exc.response.status_code < 500:
            outcome.post_definitively_rejected = True
        raise
    outcome.number = _pr_number(outcome.response)
    if outcome.number is None:
        raise RuntimeError("GitHub pull request identity is malformed")
    _validate_pr_identity(
        outcome.response,
        state,
        outcome.number,
        verified_head_sha,
        repository_identity,
    )
    outcome.post_identity_validated = True
    confirmed = await _github_get_pr(state, outcome.number)
    _validate_pr_identity(
        confirmed,
        state,
        outcome.number,
        verified_head_sha,
        repository_identity,
    )
    await _assert_pr_refs_unchanged(state, verified_head_sha)
    return confirmed


async def _drain_shielded_task(task: asyncio.Task) -> object:
    outcome = await drain_task(task)
    if outcome.error is not None:
        raise outcome.error
    return outcome.result


async def _run_cleanup_shielded(
    coro: Coroutine[Any, Any, None],
) -> _ShieldedCleanupOutcome:
    task = asyncio.create_task(coro)
    outcome = await drain_task(task)
    return _ShieldedCleanupOutcome(
        outcome.error,
        outcome.delayed_cancellation,
    )


async def _cleanup_pr_transaction(
    state: AgentState,
    repository_identity: dict[str, str],
    verified_head_sha: str,
    preexisting_numbers: frozenset[int],
    outcome: _PRTransactionOutcome,
) -> None:
    if outcome.post_definitively_rejected:
        return
    if (
        outcome.post_identity_validated
        and outcome.number is not None
        and outcome.number not in preexisting_numbers
    ):
        numbers = {outcome.number}
    else:
        numbers: set[int] = set()
        for attempt in range(len(_PR_RECONCILE_DELAYS) + 1):
            current = _exact_open_pr_numbers(
                await _github_list_open_prs(
                    state,
                    repository_identity,
                    state.branch_name,
                    state.base_branch,
                ),
                state,
                repository_identity,
                verified_head_sha,
            )
            numbers = set(current - preexisting_numbers)
            if numbers or attempt == len(_PR_RECONCILE_DELAYS):
                break
            await asyncio.sleep(_PR_RECONCILE_DELAYS[attempt])
        if not numbers:
            raise RuntimeError(
                "GitHub pull request could not be reconciled after POST"
            )
    for number in sorted(numbers):
        await _github_close_pr(state, number)


async def _verify_remote_branch(
    state: AgentState,
    binding: LiveCoverageBinding,
    base_sha: str,
    branch: str,
    *,
    expected_head_sha: str = "",
    expected_commit_chain: tuple[str, ...] = (),
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
    if expected_head_sha and head_sha != expected_head_sha:
        raise RuntimeError("repair branch drifted from the verified commit")
    chain = expected_commit_chain or (head_sha,)
    if (
        len(chain) != len(binding.approved_targets)
        or len(set(chain)) != len(chain)
        or any(not _COMMIT_RE.fullmatch(sha) for sha in chain)
        or chain[-1] != head_sha
    ):
        raise RuntimeError("remote commit chain does not match approved writes")
    if state.base_branch:
        base_ref = await _github_get_ref(state, state.base_branch)
        if _exact_ref_sha(base_ref, state.base_branch) != base_sha:
            raise RuntimeError("base branch drifted during remote verification")
    base_commit = await _github_get_commit(state, base_sha)
    base_tree_sha = _commit_tree_sha(base_commit, base_sha)
    previous = base_sha
    head_commit: dict[str, Any] | None = None
    for sha in chain:
        commit = await _github_get_commit(state, sha)
        if _commit_parents(commit, sha) != (previous,):
            raise RuntimeError("remote commit chain is not the exact linear successor")
        previous = sha
        head_commit = commit
    if head_commit is None:
        raise RuntimeError("remote commit chain is empty")
    head_tree_sha = _commit_tree_sha(head_commit, head_sha)
    base_tree = _remote_tree_map(
        await _github_get_tree(state, base_tree_sha), base_tree_sha
    )
    head_tree = _remote_tree_map(
        await _github_get_tree(state, head_tree_sha), head_tree_sha
    )

    expected_leaves = {
        path: entry for path, entry in base_tree.items() if entry[1] != "tree"
    }
    for target in binding.approved_targets:
        if (
            target.change == "deleted"
            or target.content is None
            or target.content_sha256 is None
        ):
            raise RuntimeError("remote verification does not support deletions")
        base_entry = expected_leaves.get(target.path)
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
        expected_leaves[target.path] = (
            target.mode,
            "blob",
            expected_sha,
            target.size,
        )
    expected_tree, expected_root = _build_tree_from_leaves(expected_leaves)
    if expected_root != head_tree_sha or head_tree != expected_tree:
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
    final_head = _exact_ref_sha(await _github_get_ref(state, branch), branch)
    if final_head != head_sha:
        raise RuntimeError("repair branch moved during remote verification")
    if state.base_branch:
        final_base = _exact_ref_sha(
            await _github_get_ref(state, state.base_branch), state.base_branch
        )
        if final_base != base_sha:
            raise RuntimeError("base branch moved during remote verification")
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
    repository_identity = _canonical_repo_identity(repo_info, state)
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
    commit_chain: list[str] = []
    previous_commit = base_sha
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
        commit_sha = _contents_commit_identity(result, previous_commit)
        commit_chain.append(commit_sha)
        previous_commit = commit_sha
        results.append({"path": target.path, "result": result})

    head_sha = await _verify_remote_branch(
        state,
        binding,
        base_sha,
        branch,
        expected_head_sha=commit_chain[-1],
        expected_commit_chain=tuple(commit_chain),
    )
    return {
        "branch": branch,
        "base": base_branch,
        "base_sha": base_sha,
        "head_sha": head_sha,
        "commit_chain": commit_chain,
        "repository_identity": repository_identity,
        "manifest_fingerprint": binding.manifest_fingerprint,
        "files": results,
    }


async def create_pr(
    state: AgentState,
    binding: LiveCoverageBinding | None = None,
    *,
    verified_head_sha: str,
    commit_chain: tuple[str, ...],
    repository_identity: dict[str, str],
) -> dict[str, Any]:
    terminal_binding = validate_terminal_coverage_binding(state)
    if binding is not None and binding != terminal_binding:
        raise RuntimeError("Commit binding changed before PR creation.")
    binding = terminal_binding
    if not _COMMIT_RE.fullmatch(verified_head_sha):
        raise RuntimeError("PR creation requires the verified repair head")
    bound_identity = _validate_bound_repo_identity(repository_identity, state)
    current_identity = _canonical_repo_identity(
        await _github_get_repo(state), state
    )
    if current_identity != bound_identity:
        raise RuntimeError("GitHub repository identity changed before PR creation")
    reverified_head = await _verify_remote_branch(
        state,
        binding,
        state.repo_ref,
        state.branch_name,
        expected_head_sha=verified_head_sha,
        expected_commit_chain=commit_chain,
    )
    if reverified_head != verified_head_sha:
        raise RuntimeError("repair branch changed before PR creation")
    body = (
        f"Fixes {state.issue_url}\n\n"
        f"## Plan\n{state.fix_plan}\n\n"
        f"## Tests\n{state.fix_attempts[-1].test_result if state.fix_attempts else 'Not run'}"
    )
    preexisting_numbers = _exact_open_pr_numbers(
        await _github_list_open_prs(
            state,
            bound_identity,
            state.branch_name,
            state.base_branch,
        ),
        state,
        bound_identity,
        verified_head_sha,
    )
    outcome = _PRTransactionOutcome()
    transaction = asyncio.create_task(
        _execute_pr_transaction(
            state,
            f"Fix #{state.issue_number}: {state.issue_title}",
            body,
            verified_head_sha,
            bound_identity,
            outcome,
        )
    )
    try:
        return await asyncio.shield(transaction)
    except asyncio.CancelledError as cancellation:
        transaction_error: BaseException | None = None
        try:
            await _drain_shielded_task(transaction)
        except BaseException as error:
            if not isinstance(error, asyncio.CancelledError):
                transaction_error = error
        cleanup = await _run_cleanup_shielded(
            _cleanup_pr_transaction(
                state,
                bound_identity,
                verified_head_sha,
                preexisting_numbers,
                outcome,
            )
        )
        if cleanup.error is not None:
            raise PRCancellationCleanupError(
                outcome.number,
                cancellation,
                cleanup.error,
                transaction_error=transaction_error,
            ) from cleanup.error
        if transaction_error is not None:
            raise PRCancellationTransactionError(
                outcome.number,
                cancellation,
                transaction_error,
            ) from transaction_error
        raise cancellation
    except BaseException as validation_error:
        cleanup = await _run_cleanup_shielded(
            _cleanup_pr_transaction(
                state,
                bound_identity,
                verified_head_sha,
                preexisting_numbers,
                outcome,
            )
        )
        if cleanup.delayed_cancellation is not None:
            if cleanup.error is not None:
                raise PRCancellationCleanupError(
                    outcome.number,
                    cleanup.delayed_cancellation,
                    cleanup.error,
                    transaction_error=validation_error,
                ) from cleanup.error
            raise PRCancellationTransactionError(
                outcome.number,
                cleanup.delayed_cancellation,
                validation_error,
            ) from validation_error
        if cleanup.error is not None:
            raise PRCleanupError(
                outcome.number, validation_error, cleanup.error
            ) from cleanup.error
        raise


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
        pr = await create_pr(
            state,
            binding,
            verified_head_sha=pushed["head_sha"],
            commit_chain=tuple(pushed["commit_chain"]),
            repository_identity=pushed["repository_identity"],
        )
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
    except CancellationDrainError:
        raise
    except Exception as exc:
        _record_tool(state, "commit_fix", {}, error=str(exc))
        state.failure_reason = f"Failed to push or create PR: {exc}"
        state.current_phase = Phase.FAILURE
    return state
