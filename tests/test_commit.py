from __future__ import annotations

import asyncio
import base64
import hashlib
import re
import subprocess
from pathlib import Path

import httpx
import pytest

from src.async_safety import CancellationDrainError
from src.coverage_gate import validate_terminal_coverage_binding
from src.nodes import commit as commit_node
from src.nodes.commit import create_pr, push_files
from src.nodes.execute import git_diff
from src.state import (
    AgentState,
    CoverageProof,
    GeneratedTestApproval,
    SnapshotManifestEntry,
    ToolPatchApproval,
    tool_manifest_fingerprint,
)
from src.state import (
    TestRunFingerprint as RunFingerprint,
)


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _tree_sha(entries: list[tuple[str, str, str]]) -> str:
    body = b"".join(
        mode.lstrip("0").encode("ascii")
        + b" "
        + name.encode("utf-8")
        + b"\0"
        + bytes.fromhex(sha)
        for mode, name, sha in sorted(
            entries,
            key=lambda item: (item[1] + ("/" if item[0] == "040000" else "")).encode(),
        )
    )
    return hashlib.sha1(
        f"tree {len(body)}\0".encode("ascii") + body
    ).hexdigest()


def test_contents_api_url_encodes_repository_and_path_segments():
    state = AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        owner="acme org",
        repo="widget#fork",
    )

    assert commit_node._contents_url(state, "src/a b+#?.py") == (
        "https://api.github.com/repos/acme%20org/widget%23fork/contents/"
        "src/a%20b%2B%23%3F.py"
    )


def test_remote_tree_map_proves_every_nested_tree_object():
    leaf = _git_blob_sha(b"ok\n")
    forged_tree = "f" * 40
    root = _tree_sha([("040000", "src", forged_tree)])
    payload = {
        "sha": root,
        "truncated": False,
        "tree": [
            {"path": "src", "mode": "040000", "type": "tree", "sha": forged_tree},
            {
                "path": "src/app.py",
                "mode": "100644",
                "type": "blob",
                "sha": leaf,
                "size": 3,
            },
        ],
    }

    with pytest.raises(RuntimeError, match="tree identity"):
        commit_node._remote_tree_map(payload, root)


def test_remote_tree_map_rejects_omitted_descendants_and_extra_empty_trees():
    empty_tree = _tree_sha([])
    root = _tree_sha([("040000", "empty", empty_tree)])
    payload = {
        "sha": root,
        "truncated": False,
        "tree": [
            {"path": "empty", "mode": "040000", "type": "tree", "sha": empty_tree}
        ],
    }

    with pytest.raises(RuntimeError, match="empty tree"):
        commit_node._remote_tree_map(payload, root)


@pytest.mark.parametrize(
    "entry",
    [
        {"path": "../escape", "mode": "100644", "type": "blob", "size": 1},
        {"path": "src//app.py", "mode": "100644", "type": "blob", "size": 1},
        {"path": "app.py", "mode": "040000", "type": "blob", "size": 1},
        {"path": "app.py", "mode": "100644", "type": "tree", "size": 1},
        {"path": "app.py", "mode": "100644", "type": "blob", "size": True},
        {"path": "app.py", "mode": "100644", "type": "blob", "size": -1},
    ],
)
def test_remote_tree_map_rejects_invalid_paths_modes_and_sizes(entry):
    leaf = _git_blob_sha(b"x")
    candidate = {**entry, "sha": leaf}
    root = _tree_sha([("100644", "app.py", leaf)])

    with pytest.raises(RuntimeError, match="malformed"):
        commit_node._remote_tree_map(
            {"sha": root, "truncated": False, "tree": [candidate]}, root
        )


@pytest.mark.parametrize(
    "parents",
    [
        [{"sha": "b" * 40}],
        [{"sha": "a" * 40}, {"sha": "b" * 40}],
        [],
    ],
)
def test_contents_commit_must_be_exact_single_parent_successor(parents):
    with pytest.raises(RuntimeError, match="exact linear successor"):
        commit_node._contents_commit_identity(
            {"commit": {"sha": "c" * 40, "parents": parents}}, "a" * 40
        )

def _state(
    tmp_path: Path,
    *,
    add_test: bool = False,
    delete_source: bool = False,
) -> AgentState:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "test@example.invalid")
    _git(root, "config", "user.name", "Test")
    (root / "src").mkdir()
    (root / "tests").mkdir()
    source = root / "src" / "widget.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    (root / "tests" / "test_smoke.py").write_text(
        "def test_smoke(): assert True\n", encoding="utf-8"
    )
    _git(root, "add", "--all")
    _git(root, "commit", "-qm", "base")
    ref = _git(root, "rev-parse", "HEAD")
    entries: list[SnapshotManifestEntry] = []
    if delete_source:
        base = source.read_bytes()
        source.unlink()
        entries.append(
            SnapshotManifestEntry(
                path="src/widget.py",
                change="deleted",
                mode="100644",
                size=len(base),
            )
        )
    else:
        source.write_text("VALUE = 2\n", encoding="utf-8")
        content = source.read_bytes()
        entries.append(
            SnapshotManifestEntry(
                path="src/widget.py",
                change="modified",
                mode="100644",
                content_sha256=hashlib.sha256(content).hexdigest(),
                size=len(content),
            )
        )
    generated: list[GeneratedTestApproval] = []
    coverage_files: list[str] = []
    if add_test:
        test = root / "tests" / "test_generated.py"
        test.write_text("def test_generated(): assert True\n", encoding="utf-8")
        content = test.read_bytes()
        entries.append(
            SnapshotManifestEntry(
                path="tests/test_generated.py",
                change="added",
                mode="100644",
                content_sha256=hashlib.sha256(content).hexdigest(),
                size=len(content),
            )
        )
        coverage_files = ["tests/test_generated.py"]
    entries.sort(key=lambda item: item.path)
    patch = git_diff(str(root))
    fingerprint = "a" * 64
    if add_test:
        generated = [
            GeneratedTestApproval(
                path="tests/test_generated.py",
                content_sha256=next(
                    item.content_sha256
                    for item in entries
                    if item.path == "tests/test_generated.py"
                ),
                patch_gate_fingerprint=fingerprint,
            )
        ]
    return AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        issue_number=7,
        owner="acme",
        repo="widget",
        repo_path=str(root),
        repo_ref=ref,
        patch_content=patch,
        coverage_test_files=coverage_files,
        generated_test_approvals=generated,
        tool_patch_approval=ToolPatchApproval(
            base_ref=ref,
            patch_sha256=hashlib.sha256(patch.encode()).hexdigest(),
            patch_gate_fingerprint=fingerprint,
            changed_manifest=entries,
            manifest_fingerprint=tool_manifest_fingerprint(entries),
        ),
    )


def _mock_branch(monkeypatch: pytest.MonkeyPatch, state: AgentState) -> None:
    async def repo(_state):
        return {
            "default_branch": "main",
            "full_name": f"{state.owner}/{state.repo}",
        }

    async def ref(_state, branch):
        return {
            "ref": f"refs/heads/{branch}",
            "object": {"sha": state.repo_ref},
        }

    async def create_ref(_state, branch, sha):
        return {"ref": f"refs/heads/{branch}", "object": {"sha": sha}}

    async def verify(_state, _binding, _base_sha, _branch, **_kwargs):
        return "d" * 40

    monkeypatch.setattr("src.nodes.commit._github_get_repo", repo)
    monkeypatch.setattr("src.nodes.commit._github_get_ref", ref)
    monkeypatch.setattr("src.nodes.commit._github_create_ref", create_ref)
    monkeypatch.setattr("src.nodes.commit._verify_remote_branch", verify)


def _attach_proof(state: AgentState) -> None:
    approval = state.tool_patch_approval
    assert approval is not None
    generated = bool(state.generated_test_approvals)
    test_path = "tests/test_generated.py" if generated else "tests/test_smoke.py"
    status = "generated_verified" if generated else "existing_verified"
    source = "generated" if generated else "existing"
    state.coverage_status = status
    state.coverage_test_files = [test_path]
    state.coverage_proof = CoverageProof(
        source=source,
        status=status,
        test_files=[test_path],
        argv=["python", "-m", "pytest", test_path, "-q"],
        fixed_runs=[RunFingerprint(exit_code=0, outcome="pass", summary="pass")] * 2,
        base_runs=[
            RunFingerprint(
                exit_code=1,
                outcome="assertion_failure",
                failing_test_ids=[f"{test_path}::test_regression"],
                assertion_fingerprint="c" * 64,
                summary="assertion_failure",
            )
        ]
        * 2,
        base_ref=state.repo_ref,
        patch_sha256=approval.patch_sha256,
        patch_gate_fingerprint=approval.patch_gate_fingerprint,
        manifest_fingerprint=approval.manifest_fingerprint,
        test_content_digests={
            test_path: hashlib.sha256(
                (Path(state.repo_path) / test_path).read_bytes()
            ).hexdigest()
        },
    )


@pytest.mark.asyncio
async def test_commit_fix_reraises_create_pr_cancellation_cleanup_error(
    tmp_path,
    monkeypatch,
):
    state = _state(tmp_path)
    _attach_proof(state)
    cancellation = asyncio.CancelledError("cancel commit")
    cleanup_error = RuntimeError("PR cleanup failed")
    sentinel = commit_node.PRCancellationCleanupError(
        12,
        cancellation,
        cleanup_error,
    )

    async def pushed(_state, _binding):
        return {
            "head_sha": "d" * 40,
            "commit_chain": ["d" * 40],
            "repository_identity": _repo_identity(state),
            "files": [],
        }

    async def cancelled(*_args, **_kwargs):
        raise sentinel

    monkeypatch.setattr(commit_node, "push_files", pushed)
    monkeypatch.setattr(commit_node, "create_pr", cancelled)

    with pytest.raises(commit_node.PRCancellationCleanupError) as raised:
        await commit_node.commit_fix(state)

    assert raised.value is sentinel
    assert raised.value.cancellation is cancellation
    assert raised.value.cleanup_error is cleanup_error
    assert state.failure_reason == ""
    assert all(call.tool_name != "commit_fix" for call in state.tool_calls)


@pytest.mark.asyncio
async def test_push_files_uses_manifest_targets_and_base_blob_sha(
    tmp_path,
    monkeypatch,
):
    state = _state(tmp_path)
    _attach_proof(state)
    _mock_branch(monkeypatch, state)
    writes: list[dict[str, str]] = []

    async def write(_state, path, content, branch, message, sha=""):
        writes.append(
            {"path": path, "content": content, "branch": branch, "sha": sha}
        )
        return {
            "content": {"path": path},
            "commit": {
                "sha": "d" * 40,
                "parents": [{"sha": state.repo_ref}],
            },
        }

    monkeypatch.setattr("src.nodes.commit._github_create_or_update_file", write)
    result = await push_files(state)

    expected_blob = _git(Path(state.repo_path), "rev-parse", f"{state.repo_ref}:src/widget.py")
    assert [item["path"] for item in writes] == ["src/widget.py"]
    assert writes[0]["content"] == "VALUE = 2\n"
    assert writes[0]["sha"] == expected_blob
    assert result["files"][0]["path"] == "src/widget.py"
    assert result["repository_identity"]["full_name"] == "acme/widget"


@pytest.mark.asyncio
async def test_push_files_includes_approved_untracked_generated_test_without_sha(
    tmp_path,
    monkeypatch,
):
    state = _state(tmp_path, add_test=True)
    _attach_proof(state)
    _mock_branch(monkeypatch, state)
    writes: list[tuple[str, str]] = []
    commit_shas = iter(["d" * 40, "e" * 40])
    previous = state.repo_ref

    async def write(_state, path, content, branch, message, sha=""):
        nonlocal previous
        writes.append((path, sha))
        commit_sha = next(commit_shas)
        result = {
            "content": {"path": path},
            "commit": {"sha": commit_sha, "parents": [{"sha": previous}]},
        }
        previous = commit_sha
        return result

    monkeypatch.setattr("src.nodes.commit._github_create_or_update_file", write)
    await push_files(state)

    assert writes == [
        ("src/widget.py", _git(Path(state.repo_path), "rev-parse", f"{state.repo_ref}:src/widget.py")),
        ("tests/test_generated.py", ""),
    ]


@pytest.mark.asyncio
async def test_push_files_rejects_deletion_before_any_github_api(tmp_path, monkeypatch):
    state = _state(tmp_path, delete_source=True)
    _attach_proof(state)
    calls: list[str] = []

    async def forbidden(*_args, **_kwargs):
        calls.append("github")
        pytest.fail("unsupported deletion reached GitHub")

    monkeypatch.setattr("src.nodes.commit._github_get_repo", forbidden)
    with pytest.raises(RuntimeError, match="deletions are unsupported"):
        await push_files(state)
    assert calls == []


def test_repair_branch_names_are_collision_resistant_for_same_issue(monkeypatch):
    values = iter(["1" * 16, "2" * 16])
    monkeypatch.setattr(commit_node.secrets, "token_hex", lambda _size: next(values))
    first = AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        owner="acme",
        repo="widget",
        issue_number=7,
        trace_id="run-one",
    )
    second = AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        owner="acme",
        repo="widget",
        issue_number=7,
        trace_id="run-two",
    )

    first_branch = commit_node._repair_branch_name(first)
    second_branch = commit_node._repair_branch_name(second)

    pattern = r"^repopilot-fix-7-[0-9a-f]{12}-[0-9a-f]{16}$"
    assert re.fullmatch(pattern, first_branch)
    assert re.fullmatch(pattern, second_branch)
    assert first_branch != second_branch


@pytest.mark.asyncio
async def test_push_files_rejects_malicious_existing_branch_before_write(
    tmp_path, monkeypatch
):
    state = _state(tmp_path)
    _attach_proof(state)
    writes = []

    async def repo(_state):
        return {
            "default_branch": "main",
            "full_name": f"{state.owner}/{state.repo}",
        }

    async def get_ref(_state, branch):
        sha = state.repo_ref if branch == "main" else "f" * 40
        return {"ref": f"refs/heads/{branch}", "object": {"sha": sha}}

    async def create_ref(_state, branch, sha):
        return {"ref": f"refs/heads/{branch}", "already_exists": True}

    async def write(*_args, **_kwargs):
        writes.append("write")
        pytest.fail("unrelated branch must not be updated")

    monkeypatch.setattr(commit_node, "_github_get_repo", repo)
    monkeypatch.setattr(commit_node, "_github_get_ref", get_ref)
    monkeypatch.setattr(commit_node, "_github_create_ref", create_ref)
    monkeypatch.setattr(commit_node, "_github_create_or_update_file", write)

    with pytest.raises(RuntimeError, match="collision"):
        await push_files(state)
    assert writes == []


@pytest.mark.asyncio
async def test_push_files_reuses_colliding_branch_only_on_exact_base(
    tmp_path, monkeypatch
):
    state = _state(tmp_path)
    _attach_proof(state)
    writes = []

    async def repo(_state):
        return {
            "default_branch": "main",
            "full_name": f"{state.owner}/{state.repo}",
        }

    async def get_ref(_state, branch):
        return {
            "ref": f"refs/heads/{branch}",
            "object": {"sha": state.repo_ref},
        }

    async def create_ref(_state, branch, sha):
        return {"ref": f"refs/heads/{branch}", "already_exists": True}

    async def write(_state, path, content, branch, message, sha=""):
        writes.append(path)
        return {
            "content": {"path": path},
            "commit": {
                "sha": "d" * 40,
                "parents": [{"sha": state.repo_ref}],
            },
        }

    async def verify(_state, _binding, base_sha, branch, **kwargs):
        assert base_sha == state.repo_ref
        assert kwargs["expected_head_sha"] == "d" * 40
        return "d" * 40

    monkeypatch.setattr(commit_node, "_github_get_repo", repo)
    monkeypatch.setattr(commit_node, "_github_get_ref", get_ref)
    monkeypatch.setattr(commit_node, "_github_create_ref", create_ref)
    monkeypatch.setattr(commit_node, "_github_create_or_update_file", write)
    monkeypatch.setattr(commit_node, "_verify_remote_branch", verify, raising=False)

    result = await push_files(state)

    assert writes == ["src/widget.py"]
    assert result["head_sha"] == "d" * 40


def _git_blob_sha(content: bytes) -> str:
    header = f"blob {len(content)}\0".encode("ascii")
    return hashlib.sha1(header + content).hexdigest()


def _base_tree(root: Path, ref: str) -> list[dict[str, str]]:
    entries = []
    for line in _git(root, "ls-tree", "-r", ref).splitlines():
        metadata, path = line.split("\t", 1)
        mode, kind, sha = metadata.split()
        entries.append(
            {
                "path": path,
                "mode": mode,
                "type": kind,
                "sha": sha,
                "size": int(_git(root, "cat-file", "-s", sha)),
            }
        )
    return entries


def _complete_tree_entries(leaves):
    entries = {entry["path"]: dict(entry) for entry in leaves}
    directories = set()
    for path in entries:
        parts = path.split("/")[:-1]
        for index in range(1, len(parts) + 1):
            directories.add("/".join(parts[:index]))
    for directory in sorted(
        directories, key=lambda value: value.count("/"), reverse=True
    ):
        children = []
        for path, entry in entries.items():
            parent, _, name = path.rpartition("/")
            if parent == directory:
                children.append((entry["mode"], name, entry["sha"]))
        entries[directory] = {
            "path": directory,
            "mode": "040000",
            "type": "tree",
            "sha": _tree_sha(children),
        }
    root_children = []
    for path, entry in entries.items():
        parent, _, name = path.rpartition("/")
        if not parent:
            root_children.append((entry["mode"], name, entry["sha"]))
    return list(entries.values()), _tree_sha(root_children)


def _remote_tree_fixture(state: AgentState, *, unexpected=False):
    root = Path(state.repo_path)
    base_leaves = _base_tree(root, state.repo_ref)
    changed = b"VALUE = 2\n"
    head_leaves = [dict(entry) for entry in base_leaves]
    source = next(entry for entry in head_leaves if entry["path"] == "src/widget.py")
    source["sha"] = _git_blob_sha(changed)
    source["size"] = len(changed)
    if unexpected:
        head_leaves.append(
            {
                "path": "evil.txt",
                "mode": "100644",
                "type": "blob",
                "sha": _git_blob_sha(b"evil\n"),
                "size": 5,
            }
        )
    base_entries, base_root = _complete_tree_entries(base_leaves)
    head_entries, head_root = _complete_tree_entries(head_leaves)
    return base_entries, base_root, head_entries, head_root, changed


def _mock_remote_verification_api(
    monkeypatch,
    state,
    *,
    unexpected=False,
    blob_content=None,
    head_parent=None,
):
    base_entries, base_root, head_entries, head_root, changed = _remote_tree_fixture(
        state, unexpected=unexpected
    )
    branch = "repopilot-fix-7-aaaaaaaaaaaa-bbbbbbbbbbbbbbbb"
    head_sha = "d" * 40

    async def get_ref(_state, requested_branch):
        sha = state.repo_ref if requested_branch == "main" else head_sha
        return {
            "ref": f"refs/heads/{requested_branch}",
            "object": {"sha": sha},
        }

    async def get_commit(_state, sha):
        tree_sha = base_root if sha == state.repo_ref else head_root
        parents = (
            []
            if sha == state.repo_ref
            else [{"sha": head_parent or state.repo_ref}]
        )
        return {"sha": sha, "tree": {"sha": tree_sha}, "parents": parents}

    async def get_tree(_state, tree_sha):
        tree = base_entries if tree_sha == base_root else head_entries
        return {"sha": tree_sha, "truncated": False, "tree": tree}

    async def get_blob(_state, sha):
        assert sha == _git_blob_sha(changed)
        returned = changed if blob_content is None else blob_content
        return {
            "sha": sha,
            "encoding": "base64",
            "size": len(returned),
            "content": base64.b64encode(returned).decode("ascii"),
        }

    monkeypatch.setattr(commit_node, "_github_get_ref", get_ref)
    monkeypatch.setattr(commit_node, "_github_get_commit", get_commit, raising=False)
    monkeypatch.setattr(commit_node, "_github_get_tree", get_tree, raising=False)
    monkeypatch.setattr(commit_node, "_github_get_blob", get_blob, raising=False)
    return branch, head_sha


def _pr_payload(state, head_sha, *, number=12):
    return {
        "number": number,
        "html_url": f"https://github.com/{state.owner}/{state.repo}/pull/{number}",
        "url": f"https://api.github.com/repos/{state.owner}/{state.repo}/pulls/{number}",
        "head": {
            "sha": head_sha,
            "ref": state.branch_name,
            "repo": {"full_name": f"{state.owner}/{state.repo}"},
        },
        "base": {
            "sha": state.repo_ref,
            "ref": state.base_branch,
            "repo": {"full_name": f"{state.owner}/{state.repo}"},
        },
    }


def _mock_pr_repo(monkeypatch, state, *, full_name=None):
    async def get_repo(_state):
        return {"full_name": full_name or f"{state.owner}/{state.repo}"}

    async def list_prs(*_args, **_kwargs):
        return []

    monkeypatch.setattr(commit_node, "_github_get_repo", get_repo)
    monkeypatch.setattr(
        commit_node, "_github_list_open_prs", list_prs, raising=False
    )


def _repo_identity(state, *, full_name=None):
    return commit_node._canonical_repo_identity(
        {"full_name": full_name or f"{state.owner}/{state.repo}"}, state
    )


def test_pr_identity_rejects_valid_https_urls_for_the_wrong_resource(tmp_path):
    state = _state(tmp_path)
    state.branch_name = "repopilot-fix-7-aaaaaaaaaaaa-bbbbbbbbbbbbbbbb"
    state.base_branch = "main"
    payload = _pr_payload(state, "d" * 40)
    payload["html_url"] = "https://example.invalid/not-this-pr"

    with pytest.raises(RuntimeError, match="pull request identity"):
        commit_node._validate_pr_identity(
            payload, state, 12, "d" * 40, _repo_identity(state)
        )


def test_remote_blob_rejects_boolean_size():
    content = b"x"
    sha = _git_blob_sha(content)

    with pytest.raises(RuntimeError, match="blob response is malformed"):
        commit_node._decode_remote_blob(
            {
                "sha": sha,
                "encoding": "base64",
                "size": True,
                "content": base64.b64encode(content).decode("ascii"),
            },
            sha,
        )


@pytest.mark.asyncio
async def test_remote_branch_verification_accepts_exact_tree_and_blob(
    tmp_path, monkeypatch
):
    state = _state(tmp_path)
    _attach_proof(state)
    binding = validate_terminal_coverage_binding(state)
    branch, head_sha = _mock_remote_verification_api(monkeypatch, state)

    verified = await commit_node._verify_remote_branch(
        state, binding, state.repo_ref, branch
    )

    assert verified == head_sha


@pytest.mark.asyncio
async def test_remote_branch_verification_rejects_unapproved_tree_difference(
    tmp_path, monkeypatch
):
    state = _state(tmp_path)
    _attach_proof(state)
    binding = validate_terminal_coverage_binding(state)
    branch, _ = _mock_remote_verification_api(
        monkeypatch, state, unexpected=True
    )

    with pytest.raises(RuntimeError, match="remote branch tree"):
        await commit_node._verify_remote_branch(
            state, binding, state.repo_ref, branch
        )


@pytest.mark.asyncio
async def test_remote_branch_verification_rejects_mismatched_blob_content(
    tmp_path, monkeypatch
):
    state = _state(tmp_path)
    _attach_proof(state)
    binding = validate_terminal_coverage_binding(state)
    branch, _ = _mock_remote_verification_api(
        monkeypatch, state, blob_content=b"VALUE = 999\n"
    )

    with pytest.raises(RuntimeError, match="blob identity"):
        await commit_node._verify_remote_branch(
            state, binding, state.repo_ref, branch
        )


@pytest.mark.asyncio
async def test_remote_branch_verification_rejects_non_successor_commit(
    tmp_path, monkeypatch
):
    state = _state(tmp_path)
    _attach_proof(state)
    binding = validate_terminal_coverage_binding(state)
    branch, _ = _mock_remote_verification_api(
        monkeypatch, state, head_parent="b" * 40
    )

    with pytest.raises(RuntimeError, match="exact linear successor"):
        await commit_node._verify_remote_branch(
            state, binding, state.repo_ref, branch
        )


@pytest.mark.asyncio
async def test_create_pr_reverifies_remote_branch_before_api_call(
    tmp_path, monkeypatch
):
    state = _state(tmp_path)
    _attach_proof(state)
    state.branch_name = "repopilot-fix-7-aaaaaaaaaaaa-bbbbbbbbbbbbbbbb"
    state.base_branch = "main"
    calls = []
    _mock_pr_repo(monkeypatch, state)

    async def get_ref(_state, branch):
        assert branch == "main"
        return {"ref": "refs/heads/main", "object": {"sha": state.repo_ref}}

    async def reject_remote(*_args, **_kwargs):
        calls.append("verify")
        raise RuntimeError("remote branch tree mismatch")

    async def forbidden_pr(*_args, **_kwargs):
        calls.append("pr")
        pytest.fail("PR must not be created before remote verification")

    monkeypatch.setattr(commit_node, "_github_get_ref", get_ref)
    monkeypatch.setattr(commit_node, "_verify_remote_branch", reject_remote, raising=False)
    monkeypatch.setattr(commit_node, "_github_create_pr", forbidden_pr)

    with pytest.raises(RuntimeError, match="remote branch tree mismatch"):
        await create_pr(
            state,
            verified_head_sha="d" * 40,
            commit_chain=("d" * 40,),
            repository_identity=_repo_identity(state),
        )

    assert calls == ["verify"]


@pytest.mark.asyncio
async def test_create_pr_binds_post_get_and_refs_to_verified_head(
    tmp_path, monkeypatch
):
    state = _state(tmp_path)
    _attach_proof(state)
    state.branch_name = "repopilot-fix-7-aaaaaaaaaaaa-bbbbbbbbbbbbbbbb"
    state.base_branch = "main"
    head_sha = "d" * 40
    payload = _pr_payload(state, head_sha)
    calls = []
    _mock_pr_repo(monkeypatch, state)

    async def verify(*_args, **kwargs):
        assert kwargs["expected_head_sha"] == head_sha
        assert kwargs["expected_commit_chain"] == (head_sha,)
        calls.append("verify")
        return head_sha

    async def create(*_args, **_kwargs):
        calls.append("post")
        return payload

    async def get_pr(_state, number):
        assert number == 12
        calls.append("get")
        return payload

    async def get_ref(_state, branch):
        sha = state.repo_ref if branch == "main" else head_sha
        calls.append(f"ref:{branch}")
        return {"ref": f"refs/heads/{branch}", "object": {"sha": sha}}

    monkeypatch.setattr(commit_node, "_verify_remote_branch", verify)
    monkeypatch.setattr(commit_node, "_github_create_pr", create)
    monkeypatch.setattr(commit_node, "_github_get_pr", get_pr, raising=False)
    monkeypatch.setattr(commit_node, "_github_get_ref", get_ref)

    result = await create_pr(
        state,
        verified_head_sha=head_sha,
        commit_chain=(head_sha,),
        repository_identity=_repo_identity(state),
    )

    assert result == payload
    assert calls == ["verify", "post", "get", "ref:main", f"ref:{state.branch_name}"]


@pytest.mark.asyncio
async def test_create_pr_never_closes_unvalidated_post_response_number(
    tmp_path, monkeypatch
):
    state = _state(tmp_path)
    _attach_proof(state)
    state.branch_name = "repopilot-fix-7-aaaaaaaaaaaa-bbbbbbbbbbbbbbbb"
    state.base_branch = "main"
    head_sha = "d" * 40
    mismatched = _pr_payload(state, "e" * 40, number=42)
    closed = []
    _mock_pr_repo(monkeypatch, state)

    async def verify(*_args, **_kwargs):
        return head_sha

    async def create(*_args, **_kwargs):
        return mismatched

    async def close(_state, number):
        closed.append(number)

    async def sleep(_delay):
        return None

    monkeypatch.setattr(commit_node, "_verify_remote_branch", verify)
    monkeypatch.setattr(commit_node, "_github_create_pr", create)
    monkeypatch.setattr(commit_node, "_github_close_pr", close, raising=False)
    monkeypatch.setattr(commit_node.asyncio, "sleep", sleep)

    with pytest.raises(commit_node.PRCleanupError) as exc_info:
        await create_pr(
            state,
            verified_head_sha=head_sha,
            commit_chain=(head_sha,),
            repository_identity=_repo_identity(state),
        )

    assert closed == []
    assert exc_info.value.pr_number == 42
    assert isinstance(exc_info.value.validation_error, RuntimeError)
    assert isinstance(exc_info.value.cleanup_error, RuntimeError)
    assert "could not be reconciled" in str(exc_info.value.cleanup_error)


@pytest.mark.asyncio
async def test_create_pr_closes_pr_when_head_moves_after_verification(
    tmp_path, monkeypatch
):
    state = _state(tmp_path)
    _attach_proof(state)
    state.branch_name = "repopilot-fix-7-aaaaaaaaaaaa-bbbbbbbbbbbbbbbb"
    state.base_branch = "main"
    head_sha = "d" * 40
    payload = _pr_payload(state, head_sha)
    closed = []
    _mock_pr_repo(monkeypatch, state)

    async def verify(*_args, **_kwargs):
        return head_sha

    async def create(*_args, **_kwargs):
        return payload

    async def get_pr(*_args, **_kwargs):
        return payload

    async def get_ref(_state, branch):
        sha = state.repo_ref if branch == "main" else "e" * 40
        return {"ref": f"refs/heads/{branch}", "object": {"sha": sha}}

    async def close(_state, number):
        closed.append(number)

    monkeypatch.setattr(commit_node, "_verify_remote_branch", verify)
    monkeypatch.setattr(commit_node, "_github_create_pr", create)
    monkeypatch.setattr(commit_node, "_github_get_pr", get_pr, raising=False)
    monkeypatch.setattr(commit_node, "_github_get_ref", get_ref)
    monkeypatch.setattr(commit_node, "_github_close_pr", close, raising=False)

    with pytest.raises(RuntimeError, match="branch moved"):
        await create_pr(
            state,
            verified_head_sha=head_sha,
            commit_chain=(head_sha,),
            repository_identity=_repo_identity(state),
        )

    assert closed == [12]


@pytest.mark.asyncio
async def test_create_pr_surfaces_validation_and_cleanup_failures(
    tmp_path, monkeypatch
):
    state = _state(tmp_path)
    _attach_proof(state)
    state.branch_name = "repopilot-fix-7-aaaaaaaaaaaa-bbbbbbbbbbbbbbbb"
    state.base_branch = "main"
    head_sha = "d" * 40
    payload = _pr_payload(state, head_sha)
    mismatched = _pr_payload(state, "e" * 40)
    _mock_pr_repo(monkeypatch, state)

    async def verify(*_args, **_kwargs):
        return head_sha

    async def create(*_args, **_kwargs):
        return payload

    async def get_pr(*_args, **_kwargs):
        return mismatched

    async def close(*_args, **_kwargs):
        raise OSError("cleanup failed")

    monkeypatch.setattr(commit_node, "_verify_remote_branch", verify)
    monkeypatch.setattr(commit_node, "_github_create_pr", create)
    monkeypatch.setattr(commit_node, "_github_get_pr", get_pr)
    monkeypatch.setattr(commit_node, "_github_close_pr", close)

    with pytest.raises(commit_node.PRCleanupError) as exc_info:
        await create_pr(
            state,
            verified_head_sha=head_sha,
            commit_chain=(head_sha,),
            repository_identity=_repo_identity(state),
        )

    assert isinstance(exc_info.value.validation_error, RuntimeError)
    assert isinstance(exc_info.value.cleanup_error, OSError)
    assert exc_info.value.pr_number == 12


@pytest.mark.asyncio
async def test_create_pr_accepts_github_canonical_repository_casing(
    tmp_path, monkeypatch
):
    state = _state(tmp_path)
    _attach_proof(state)
    state.owner = "Acme"
    state.repo = "Widget"
    state.branch_name = "repopilot-fix-7-aaaaaaaaaaaa-bbbbbbbbbbbbbbbb"
    state.base_branch = "main"
    head_sha = "d" * 40
    payload = _pr_payload(state, head_sha)
    payload["html_url"] = "https://github.com/acme/widget/pull/12"
    payload["url"] = "https://api.github.com/repos/acme/widget/pulls/12"
    payload["head"]["repo"]["full_name"] = "acme/widget"
    payload["base"]["repo"]["full_name"] = "acme/widget"
    _mock_pr_repo(monkeypatch, state, full_name="acme/widget")

    async def verify(*_args, **_kwargs):
        return head_sha

    async def create(*_args, **_kwargs):
        return payload

    async def get_pr(*_args, **_kwargs):
        return payload

    async def get_ref(_state, branch):
        sha = state.repo_ref if branch == "main" else head_sha
        return {"ref": f"refs/heads/{branch}", "object": {"sha": sha}}

    monkeypatch.setattr(commit_node, "_verify_remote_branch", verify)
    monkeypatch.setattr(commit_node, "_github_create_pr", create)
    monkeypatch.setattr(commit_node, "_github_get_pr", get_pr)
    monkeypatch.setattr(commit_node, "_github_get_ref", get_ref)

    assert (
        await create_pr(
            state,
            verified_head_sha=head_sha,
            commit_chain=(head_sha,),
            repository_identity=_repo_identity(
                state, full_name="acme/widget"
            ),
        )
        == payload
    )


@pytest.mark.asyncio
async def test_create_pr_rejects_repository_identity_drift_before_post(
    tmp_path, monkeypatch
):
    state = _state(tmp_path)
    _attach_proof(state)
    state.branch_name = "repopilot-fix-7-aaaaaaaaaaaa-bbbbbbbbbbbbbbbb"
    state.base_branch = "main"
    posted = []

    _mock_pr_repo(monkeypatch, state, full_name="other/widget")

    async def forbidden(*_args, **_kwargs):
        posted.append(True)
        pytest.fail("PR POST must not run after repository identity drift")

    monkeypatch.setattr(commit_node, "_github_create_pr", forbidden)

    with pytest.raises(RuntimeError, match="repository identity"):
        await create_pr(
            state,
            verified_head_sha="d" * 40,
            commit_chain=("d" * 40,),
            repository_identity=_repo_identity(state),
        )

    assert posted == []


@pytest.mark.asyncio
async def test_create_pr_cancellation_during_post_drains_and_closes_remote_pr(
    tmp_path, monkeypatch
):
    state = _state(tmp_path)
    _attach_proof(state)
    state.branch_name = "repopilot-fix-7-aaaaaaaaaaaa-bbbbbbbbbbbbbbbb"
    state.base_branch = "main"
    head_sha = "d" * 40
    payload = _pr_payload(state, head_sha)
    post_created = asyncio.Event()
    release_post = asyncio.Event()
    closed = []
    _mock_pr_repo(monkeypatch, state)

    async def verify(*_args, **_kwargs):
        return head_sha

    async def create(*_args, **_kwargs):
        post_created.set()
        await release_post.wait()
        return payload

    async def get_pr(*_args, **_kwargs):
        return payload

    async def get_ref(_state, branch):
        sha = state.repo_ref if branch == "main" else head_sha
        return {"ref": f"refs/heads/{branch}", "object": {"sha": sha}}

    async def close(_state, number):
        closed.append(number)

    monkeypatch.setattr(commit_node, "_verify_remote_branch", verify)
    monkeypatch.setattr(commit_node, "_github_create_pr", create)
    monkeypatch.setattr(commit_node, "_github_get_pr", get_pr)
    monkeypatch.setattr(commit_node, "_github_get_ref", get_ref)
    monkeypatch.setattr(commit_node, "_github_close_pr", close)

    task = asyncio.create_task(
        create_pr(
            state,
            verified_head_sha=head_sha,
            commit_chain=(head_sha,),
            repository_identity=_repo_identity(state),
        )
    )
    await post_created.wait()
    task.cancel()
    release_post.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert closed == [12]


@pytest.mark.asyncio
async def test_create_pr_cancellation_during_confirmation_closes_remote_pr(
    tmp_path, monkeypatch
):
    state = _state(tmp_path)
    _attach_proof(state)
    state.branch_name = "repopilot-fix-7-aaaaaaaaaaaa-bbbbbbbbbbbbbbbb"
    state.base_branch = "main"
    head_sha = "d" * 40
    payload = _pr_payload(state, head_sha)
    confirmation_started = asyncio.Event()
    release_confirmation = asyncio.Event()
    closed = []
    _mock_pr_repo(monkeypatch, state)

    async def verify(*_args, **_kwargs):
        return head_sha

    async def get_pr(*_args, **_kwargs):
        confirmation_started.set()
        await release_confirmation.wait()
        return payload

    async def get_ref(_state, branch):
        sha = state.repo_ref if branch == "main" else head_sha
        return {"ref": f"refs/heads/{branch}", "object": {"sha": sha}}

    async def close(_state, number):
        closed.append(number)

    monkeypatch.setattr(commit_node, "_verify_remote_branch", verify)
    async def create(*_args, **_kwargs):
        return payload

    monkeypatch.setattr(commit_node, "_github_create_pr", create)
    monkeypatch.setattr(commit_node, "_github_get_pr", get_pr)
    monkeypatch.setattr(commit_node, "_github_get_ref", get_ref)
    monkeypatch.setattr(commit_node, "_github_close_pr", close)

    task = asyncio.create_task(
        create_pr(
            state,
            verified_head_sha=head_sha,
            commit_chain=(head_sha,),
            repository_identity=_repo_identity(state),
        )
    )
    await confirmation_started.wait()
    task.cancel()
    release_confirmation.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert closed == [12]


@pytest.mark.asyncio
async def test_create_pr_cancellation_during_final_ref_check_closes_remote_pr(
    tmp_path, monkeypatch
):
    state = _state(tmp_path)
    _attach_proof(state)
    state.branch_name = "repopilot-fix-7-aaaaaaaaaaaa-bbbbbbbbbbbbbbbb"
    state.base_branch = "main"
    head_sha = "d" * 40
    payload = _pr_payload(state, head_sha)
    ref_check_started = asyncio.Event()
    release_ref_check = asyncio.Event()
    closed = []
    _mock_pr_repo(monkeypatch, state)

    async def verify(*_args, **_kwargs):
        return head_sha

    async def create(*_args, **_kwargs):
        return payload

    async def get_pr(*_args, **_kwargs):
        return payload

    async def check_refs(*_args, **_kwargs):
        ref_check_started.set()
        await release_ref_check.wait()

    async def close(_state, number):
        closed.append(number)

    monkeypatch.setattr(commit_node, "_verify_remote_branch", verify)
    monkeypatch.setattr(commit_node, "_github_create_pr", create)
    monkeypatch.setattr(commit_node, "_github_get_pr", get_pr)
    monkeypatch.setattr(commit_node, "_assert_pr_refs_unchanged", check_refs)
    monkeypatch.setattr(commit_node, "_github_close_pr", close)

    task = asyncio.create_task(
        create_pr(
            state,
            verified_head_sha=head_sha,
            commit_chain=(head_sha,),
            repository_identity=_repo_identity(state),
        )
    )
    await ref_check_started.wait()
    task.cancel()
    release_ref_check.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert closed == [12]


@pytest.mark.asyncio
async def test_create_pr_cancellation_cleanup_failure_preserves_cancel_semantics(
    tmp_path, monkeypatch
):
    state = _state(tmp_path)
    _attach_proof(state)
    state.branch_name = "repopilot-fix-7-aaaaaaaaaaaa-bbbbbbbbbbbbbbbb"
    state.base_branch = "main"
    head_sha = "d" * 40
    payload = _pr_payload(state, head_sha)
    post_created = asyncio.Event()
    release_post = asyncio.Event()
    _mock_pr_repo(monkeypatch, state)

    async def verify(*_args, **_kwargs):
        return head_sha

    async def create(*_args, **_kwargs):
        post_created.set()
        await release_post.wait()
        return payload

    async def get_pr(*_args, **_kwargs):
        return payload

    async def get_ref(_state, branch):
        sha = state.repo_ref if branch == "main" else head_sha
        return {"ref": f"refs/heads/{branch}", "object": {"sha": sha}}

    async def close(*_args, **_kwargs):
        raise OSError("close failed")

    monkeypatch.setattr(commit_node, "_verify_remote_branch", verify)
    monkeypatch.setattr(commit_node, "_github_create_pr", create)
    monkeypatch.setattr(commit_node, "_github_get_pr", get_pr)
    monkeypatch.setattr(commit_node, "_github_get_ref", get_ref)
    monkeypatch.setattr(commit_node, "_github_close_pr", close)

    task = asyncio.create_task(
        create_pr(
            state,
            verified_head_sha=head_sha,
            commit_chain=(head_sha,),
            repository_identity=_repo_identity(state),
        )
    )
    await post_created.wait()
    task.cancel("cancel PR cleanup")
    release_post.set()

    with pytest.raises(commit_node.PRCancellationCleanupError) as exc_info:
        await task

    assert not isinstance(exc_info.value, asyncio.CancelledError)
    assert exc_info.value.pr_number == 12
    assert exc_info.value.cancellation.args == ("cancel PR cleanup",)
    assert isinstance(exc_info.value.cleanup_error, OSError)
    assert exc_info.value.__cause__ is exc_info.value.cleanup_error


@pytest.mark.asyncio
async def test_create_pr_cancellation_surfaces_transaction_failure_after_drain(
    tmp_path, monkeypatch
):
    state = _state(tmp_path)
    _attach_proof(state)
    state.branch_name = "repopilot-fix-7-aaaaaaaaaaaa-bbbbbbbbbbbbbbbb"
    state.base_branch = "main"
    head_sha = "d" * 40
    payload = _pr_payload(state, head_sha)
    transaction_blocked = asyncio.Event()
    release_transaction = asyncio.Event()
    transaction_error = OSError("transaction failed")
    closed = []
    _mock_pr_repo(monkeypatch, state)

    async def verify(*_args, **_kwargs):
        return head_sha

    async def create(*_args, **_kwargs):
        return payload

    async def get_pr(*_args, **_kwargs):
        transaction_blocked.set()
        await release_transaction.wait()
        raise transaction_error

    async def close(_state, number):
        closed.append(number)

    monkeypatch.setattr(commit_node, "_verify_remote_branch", verify)
    monkeypatch.setattr(commit_node, "_github_create_pr", create)
    monkeypatch.setattr(commit_node, "_github_get_pr", get_pr)
    monkeypatch.setattr(commit_node, "_github_close_pr", close)

    task = asyncio.create_task(
        create_pr(
            state,
            verified_head_sha=head_sha,
            commit_chain=(head_sha,),
            repository_identity=_repo_identity(state),
        )
    )
    await transaction_blocked.wait()
    task.cancel("cancel failing transaction")
    release_transaction.set()

    with pytest.raises(CancellationDrainError) as caught:
        await task

    error = caught.value
    assert isinstance(error, commit_node.PRCancellationTransactionError)
    assert error.cancellation.args == ("cancel failing transaction",)
    assert error.transaction_error is transaction_error
    assert error.cleanup_error is transaction_error
    assert error.__cause__ is transaction_error
    assert closed == [12]


@pytest.mark.asyncio
async def test_create_pr_cancellation_surfaces_transaction_and_cleanup_failures(
    tmp_path, monkeypatch
):
    state = _state(tmp_path)
    _attach_proof(state)
    state.branch_name = "repopilot-fix-7-aaaaaaaaaaaa-bbbbbbbbbbbbbbbb"
    state.base_branch = "main"
    head_sha = "d" * 40
    payload = _pr_payload(state, head_sha)
    transaction_blocked = asyncio.Event()
    release_transaction = asyncio.Event()
    transaction_error = OSError("transaction failed")
    cleanup_error = RuntimeError("cleanup failed")
    _mock_pr_repo(monkeypatch, state)

    async def verify(*_args, **_kwargs):
        return head_sha

    async def create(*_args, **_kwargs):
        return payload

    async def get_pr(*_args, **_kwargs):
        transaction_blocked.set()
        await release_transaction.wait()
        raise transaction_error

    async def close(*_args, **_kwargs):
        raise cleanup_error

    monkeypatch.setattr(commit_node, "_verify_remote_branch", verify)
    monkeypatch.setattr(commit_node, "_github_create_pr", create)
    monkeypatch.setattr(commit_node, "_github_get_pr", get_pr)
    monkeypatch.setattr(commit_node, "_github_close_pr", close)

    task = asyncio.create_task(
        create_pr(
            state,
            verified_head_sha=head_sha,
            commit_chain=(head_sha,),
            repository_identity=_repo_identity(state),
        )
    )
    await transaction_blocked.wait()
    task.cancel("cancel double failure")
    release_transaction.set()

    with pytest.raises(commit_node.PRCancellationCleanupError) as caught:
        await task

    error = caught.value
    assert error.cancellation.args == ("cancel double failure",)
    assert error.transaction_error is transaction_error
    assert error.cleanup_error is cleanup_error
    assert error.__cause__ is cleanup_error


@pytest.mark.asyncio
async def test_create_pr_transaction_failure_then_cancellation_preserves_both(
    tmp_path, monkeypatch
):
    state = _state(tmp_path)
    _attach_proof(state)
    state.branch_name = "repopilot-fix-7-aaaaaaaaaaaa-bbbbbbbbbbbbbbbb"
    state.base_branch = "main"
    head_sha = "d" * 40
    payload = _pr_payload(state, head_sha)
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()
    transaction_error = OSError("transaction validation failed")
    closed = []
    _mock_pr_repo(monkeypatch, state)

    async def verify(*_args, **_kwargs):
        return head_sha

    async def create(*_args, **_kwargs):
        return payload

    async def get_pr(*_args, **_kwargs):
        return payload

    async def fail_refs(*_args, **_kwargs):
        raise transaction_error

    async def close(_state, number):
        cleanup_started.set()
        await release_cleanup.wait()
        closed.append(number)

    monkeypatch.setattr(commit_node, "_verify_remote_branch", verify)
    monkeypatch.setattr(commit_node, "_github_create_pr", create)
    monkeypatch.setattr(commit_node, "_github_get_pr", get_pr)
    monkeypatch.setattr(commit_node, "_assert_pr_refs_unchanged", fail_refs)
    monkeypatch.setattr(commit_node, "_github_close_pr", close)

    task = asyncio.create_task(
        create_pr(
            state,
            verified_head_sha=head_sha,
            commit_chain=(head_sha,),
            repository_identity=_repo_identity(state),
        )
    )
    await cleanup_started.wait()
    task.cancel("cancel transaction cleanup")
    release_cleanup.set()

    with pytest.raises(commit_node.PRCancellationTransactionError) as caught:
        await task

    error = caught.value
    assert error.pr_number == 12
    assert error.cancellation.args == ("cancel transaction cleanup",)
    assert error.transaction_error is transaction_error
    assert error.cleanup_error is transaction_error
    assert error.__cause__ is transaction_error
    assert closed == [12]


@pytest.mark.asyncio
async def test_create_pr_transaction_failure_then_cancellation_keeps_cleanup_failure(
    tmp_path, monkeypatch
):
    state = _state(tmp_path)
    _attach_proof(state)
    state.branch_name = "repopilot-fix-7-aaaaaaaaaaaa-bbbbbbbbbbbbbbbb"
    state.base_branch = "main"
    head_sha = "d" * 40
    payload = _pr_payload(state, head_sha)
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()
    transaction_error = OSError("transaction validation failed")
    cleanup_error = RuntimeError("cleanup failed")
    _mock_pr_repo(monkeypatch, state)

    async def verify(*_args, **_kwargs):
        return head_sha

    async def create(*_args, **_kwargs):
        return payload

    async def get_pr(*_args, **_kwargs):
        return payload

    async def fail_refs(*_args, **_kwargs):
        raise transaction_error

    async def close(*_args, **_kwargs):
        cleanup_started.set()
        await release_cleanup.wait()
        raise cleanup_error

    monkeypatch.setattr(commit_node, "_verify_remote_branch", verify)
    monkeypatch.setattr(commit_node, "_github_create_pr", create)
    monkeypatch.setattr(commit_node, "_github_get_pr", get_pr)
    monkeypatch.setattr(commit_node, "_assert_pr_refs_unchanged", fail_refs)
    monkeypatch.setattr(commit_node, "_github_close_pr", close)

    task = asyncio.create_task(
        create_pr(
            state,
            verified_head_sha=head_sha,
            commit_chain=(head_sha,),
            repository_identity=_repo_identity(state),
        )
    )
    await cleanup_started.wait()
    task.cancel("cancel double cleanup failure")
    release_cleanup.set()

    with pytest.raises(commit_node.PRCancellationCleanupError) as caught:
        await task

    error = caught.value
    assert error.pr_number == 12
    assert error.cancellation.args == ("cancel double cleanup failure",)
    assert error.transaction_error is transaction_error
    assert error.cleanup_error is cleanup_error
    assert error.__cause__ is cleanup_error


@pytest.mark.asyncio
async def test_create_pr_second_cancellation_does_not_misreport_successful_cleanup(
    tmp_path, monkeypatch
):
    state = _state(tmp_path)
    _attach_proof(state)
    state.branch_name = "repopilot-fix-7-aaaaaaaaaaaa-bbbbbbbbbbbbbbbb"
    state.base_branch = "main"
    head_sha = "d" * 40
    payload = _pr_payload(state, head_sha)
    post_created = asyncio.Event()
    release_post = asyncio.Event()
    close_started = asyncio.Event()
    release_close = asyncio.Event()
    closed = []
    _mock_pr_repo(monkeypatch, state)

    async def verify(*_args, **_kwargs):
        return head_sha

    async def create(*_args, **_kwargs):
        post_created.set()
        await release_post.wait()
        return payload

    async def get_pr(*_args, **_kwargs):
        return payload

    async def get_ref(_state, branch):
        sha = state.repo_ref if branch == "main" else head_sha
        return {"ref": f"refs/heads/{branch}", "object": {"sha": sha}}

    async def close(_state, number):
        close_started.set()
        await release_close.wait()
        closed.append(number)

    monkeypatch.setattr(commit_node, "_verify_remote_branch", verify)
    monkeypatch.setattr(commit_node, "_github_create_pr", create)
    monkeypatch.setattr(commit_node, "_github_get_pr", get_pr)
    monkeypatch.setattr(commit_node, "_github_get_ref", get_ref)
    monkeypatch.setattr(commit_node, "_github_close_pr", close)

    task = asyncio.create_task(
        create_pr(
            state,
            verified_head_sha=head_sha,
            commit_chain=(head_sha,),
            repository_identity=_repo_identity(state),
        )
    )
    await post_created.wait()
    task.cancel()
    release_post.set()
    await close_started.wait()
    task.cancel()
    release_close.set()

    with pytest.raises(asyncio.CancelledError) as exc_info:
        await task

    assert not isinstance(
        exc_info.value, commit_node.PRCancellationCleanupError
    )
    assert closed == [12]


@pytest.mark.asyncio
async def test_create_pr_missing_number_reconciles_and_closes_exact_new_pr(
    tmp_path, monkeypatch
):
    state = _state(tmp_path)
    _attach_proof(state)
    state.branch_name = "repopilot-fix-7-aaaaaaaaaaaa-bbbbbbbbbbbbbbbb"
    state.base_branch = "main"
    head_sha = "d" * 40
    exact = _pr_payload(state, head_sha)
    malformed = dict(exact)
    malformed.pop("number")
    list_calls = 0
    closed = []
    _mock_pr_repo(monkeypatch, state)

    async def verify(*_args, **_kwargs):
        return head_sha

    async def list_prs(*_args, **_kwargs):
        nonlocal list_calls
        list_calls += 1
        return [] if list_calls == 1 else [exact]

    async def create(*_args, **_kwargs):
        return malformed

    async def close(_state, number):
        closed.append(number)

    monkeypatch.setattr(commit_node, "_verify_remote_branch", verify)
    monkeypatch.setattr(commit_node, "_github_list_open_prs", list_prs)
    monkeypatch.setattr(commit_node, "_github_create_pr", create)
    monkeypatch.setattr(commit_node, "_github_close_pr", close)

    with pytest.raises(RuntimeError, match="pull request identity"):
        await create_pr(
            state,
            verified_head_sha=head_sha,
            commit_chain=(head_sha,),
            repository_identity=_repo_identity(state),
        )

    assert closed == [12]


@pytest.mark.asyncio
async def test_create_pr_unknown_post_outcome_reconciles_only_new_exact_pr(
    tmp_path, monkeypatch
):
    state = _state(tmp_path)
    _attach_proof(state)
    state.branch_name = "repopilot-fix-7-aaaaaaaaaaaa-bbbbbbbbbbbbbbbb"
    state.base_branch = "main"
    head_sha = "d" * 40
    preexisting = _pr_payload(state, head_sha, number=10)
    created = _pr_payload(state, head_sha, number=12)
    unrelated = _pr_payload(state, "e" * 40, number=14)
    list_calls = 0
    closed = []
    _mock_pr_repo(monkeypatch, state)

    async def verify(*_args, **_kwargs):
        return head_sha

    async def list_prs(*_args, **_kwargs):
        nonlocal list_calls
        list_calls += 1
        return [preexisting] if list_calls == 1 else [preexisting, created, unrelated]

    async def create(*_args, **_kwargs):
        raise OSError("POST outcome unknown")

    async def close(_state, number):
        closed.append(number)

    monkeypatch.setattr(commit_node, "_verify_remote_branch", verify)
    monkeypatch.setattr(commit_node, "_github_list_open_prs", list_prs)
    monkeypatch.setattr(commit_node, "_github_create_pr", create)
    monkeypatch.setattr(commit_node, "_github_close_pr", close)

    with pytest.raises(OSError, match="outcome unknown"):
        await create_pr(
            state,
            verified_head_sha=head_sha,
            commit_chain=(head_sha,),
            repository_identity=_repo_identity(state),
        )

    assert closed == [12]


@pytest.mark.asyncio
async def test_create_pr_unknown_post_outcome_without_match_is_cleanup_failure(
    tmp_path, monkeypatch
):
    state = _state(tmp_path)
    _attach_proof(state)
    state.branch_name = "repopilot-fix-7-aaaaaaaaaaaa-bbbbbbbbbbbbbbbb"
    state.base_branch = "main"
    head_sha = "d" * 40
    closed = []
    _mock_pr_repo(monkeypatch, state)

    async def verify(*_args, **_kwargs):
        return head_sha

    async def create(*_args, **_kwargs):
        raise OSError("POST outcome unknown")

    async def close(_state, number):
        closed.append(number)

    async def sleep(_delay):
        return None

    monkeypatch.setattr(commit_node, "_verify_remote_branch", verify)
    monkeypatch.setattr(commit_node, "_github_create_pr", create)
    monkeypatch.setattr(commit_node, "_github_close_pr", close)
    monkeypatch.setattr(commit_node.asyncio, "sleep", sleep)

    with pytest.raises(commit_node.PRCleanupError) as exc_info:
        await create_pr(
            state,
            verified_head_sha=head_sha,
            commit_chain=(head_sha,),
            repository_identity=_repo_identity(state),
        )

    assert closed == []
    assert exc_info.value.pr_number is None
    assert isinstance(exc_info.value.validation_error, OSError)
    assert isinstance(exc_info.value.cleanup_error, RuntimeError)
    assert "could not be reconciled" in str(exc_info.value.cleanup_error)


@pytest.mark.asyncio
async def test_create_pr_definitive_4xx_rejection_is_not_cleanup_failure(
    tmp_path, monkeypatch
):
    state = _state(tmp_path)
    _attach_proof(state)
    state.branch_name = "repopilot-fix-7-aaaaaaaaaaaa-bbbbbbbbbbbbbbbb"
    state.base_branch = "main"
    head_sha = "d" * 40
    list_calls = 0
    _mock_pr_repo(monkeypatch, state)

    async def verify(*_args, **_kwargs):
        return head_sha

    async def list_prs(*_args, **_kwargs):
        nonlocal list_calls
        list_calls += 1
        return []

    request = httpx.Request(
        "POST", "https://api.github.com/repos/acme/widget/pulls"
    )
    response = httpx.Response(422, request=request)
    rejection = httpx.HTTPStatusError(
        "unprocessable entity", request=request, response=response
    )

    async def create(*_args, **_kwargs):
        raise rejection

    monkeypatch.setattr(commit_node, "_verify_remote_branch", verify)
    monkeypatch.setattr(commit_node, "_github_list_open_prs", list_prs)
    monkeypatch.setattr(commit_node, "_github_create_pr", create)

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        await create_pr(
            state,
            verified_head_sha=head_sha,
            commit_chain=(head_sha,),
            repository_identity=_repo_identity(state),
        )

    assert exc_info.value is rejection
    assert list_calls == 1


@pytest.mark.asyncio
async def test_create_pr_reconciliation_failure_is_not_swallowed(
    tmp_path, monkeypatch
):
    state = _state(tmp_path)
    _attach_proof(state)
    state.branch_name = "repopilot-fix-7-aaaaaaaaaaaa-bbbbbbbbbbbbbbbb"
    state.base_branch = "main"
    head_sha = "d" * 40
    list_calls = 0
    _mock_pr_repo(monkeypatch, state)

    async def verify(*_args, **_kwargs):
        return head_sha

    async def list_prs(*_args, **_kwargs):
        nonlocal list_calls
        list_calls += 1
        if list_calls == 1:
            return []
        raise OSError("reconciliation failed")

    async def create(*_args, **_kwargs):
        return {"html_url": "https://github.com/acme/widget/pull/unknown"}

    monkeypatch.setattr(commit_node, "_verify_remote_branch", verify)
    monkeypatch.setattr(commit_node, "_github_list_open_prs", list_prs)
    monkeypatch.setattr(commit_node, "_github_create_pr", create)

    with pytest.raises(commit_node.PRCleanupError) as exc_info:
        await create_pr(
            state,
            verified_head_sha=head_sha,
            commit_chain=(head_sha,),
            repository_identity=_repo_identity(state),
        )

    assert exc_info.value.pr_number is None
    assert isinstance(exc_info.value.cleanup_error, OSError)


@pytest.mark.asyncio
async def test_create_pr_reconciliation_never_closes_preexisting_or_unrelated_pr(
    tmp_path, monkeypatch
):
    state = _state(tmp_path)
    _attach_proof(state)
    state.branch_name = "repopilot-fix-7-aaaaaaaaaaaa-bbbbbbbbbbbbbbbb"
    state.base_branch = "main"
    head_sha = "d" * 40
    preexisting = _pr_payload(state, head_sha, number=10)
    unrelated = _pr_payload(state, "e" * 40, number=14)
    list_calls = 0
    closed = []
    _mock_pr_repo(monkeypatch, state)

    async def verify(*_args, **_kwargs):
        return head_sha

    async def list_prs(*_args, **_kwargs):
        nonlocal list_calls
        list_calls += 1
        return [preexisting] if list_calls == 1 else [preexisting, unrelated]

    async def create(*_args, **_kwargs):
        return {"html_url": "https://github.com/acme/widget/pull/unknown"}

    async def close(_state, number):
        closed.append(number)

    async def sleep(_delay):
        return None

    monkeypatch.setattr(commit_node, "_verify_remote_branch", verify)
    monkeypatch.setattr(commit_node, "_github_list_open_prs", list_prs)
    monkeypatch.setattr(commit_node, "_github_create_pr", create)
    monkeypatch.setattr(commit_node, "_github_close_pr", close)
    monkeypatch.setattr(commit_node.asyncio, "sleep", sleep)

    with pytest.raises(commit_node.PRCleanupError) as exc_info:
        await create_pr(
            state,
            verified_head_sha=head_sha,
            commit_chain=(head_sha,),
            repository_identity=_repo_identity(state),
        )

    assert list_calls == 4
    assert closed == []
    assert exc_info.value.pr_number is None
    assert isinstance(exc_info.value.cleanup_error, RuntimeError)
    assert "could not be reconciled" in str(exc_info.value.cleanup_error)
