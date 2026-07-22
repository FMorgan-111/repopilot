from __future__ import annotations

import base64
import hashlib
import re
import subprocess
from pathlib import Path

import pytest

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
        return {"default_branch": "main"}

    async def ref(_state, branch):
        return {
            "ref": f"refs/heads/{branch}",
            "object": {"sha": state.repo_ref},
        }

    async def create_ref(_state, branch, sha):
        return {"ref": f"refs/heads/{branch}", "object": {"sha": sha}}

    async def verify(_state, _binding, _base_sha, _branch):
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
        return {"content": {"path": path}}

    monkeypatch.setattr("src.nodes.commit._github_create_or_update_file", write)
    result = await push_files(state)

    expected_blob = _git(Path(state.repo_path), "rev-parse", f"{state.repo_ref}:src/widget.py")
    assert [item["path"] for item in writes] == ["src/widget.py"]
    assert writes[0]["content"] == "VALUE = 2\n"
    assert writes[0]["sha"] == expected_blob
    assert result["files"][0]["path"] == "src/widget.py"


@pytest.mark.asyncio
async def test_push_files_includes_approved_untracked_generated_test_without_sha(
    tmp_path,
    monkeypatch,
):
    state = _state(tmp_path, add_test=True)
    _attach_proof(state)
    _mock_branch(monkeypatch, state)
    writes: list[tuple[str, str]] = []

    async def write(_state, path, content, branch, message, sha=""):
        writes.append((path, sha))
        return {"content": {"path": path}}

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
        return {"default_branch": "main"}

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
        return {"default_branch": "main"}

    async def get_ref(_state, branch):
        return {
            "ref": f"refs/heads/{branch}",
            "object": {"sha": state.repo_ref},
        }

    async def create_ref(_state, branch, sha):
        return {"ref": f"refs/heads/{branch}", "already_exists": True}

    async def write(_state, path, content, branch, message, sha=""):
        writes.append(path)
        return {"content": {"path": path}}

    async def verify(_state, _binding, base_sha, branch):
        assert base_sha == state.repo_ref
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
        entries.append({"path": path, "mode": mode, "type": kind, "sha": sha})
    return entries


def _remote_tree_fixture(state: AgentState, *, unexpected=False):
    root = Path(state.repo_path)
    base_entries = _base_tree(root, state.repo_ref)
    changed = b"VALUE = 2\n"
    head_entries = [dict(entry) for entry in base_entries]
    source = next(entry for entry in head_entries if entry["path"] == "src/widget.py")
    source["sha"] = _git_blob_sha(changed)
    if unexpected:
        head_entries.append(
            {
                "path": "evil.txt",
                "mode": "100644",
                "type": "blob",
                "sha": _git_blob_sha(b"evil\n"),
            }
        )
    return base_entries, head_entries, changed


def _mock_remote_verification_api(
    monkeypatch, state, *, unexpected=False, blob_content=None
):
    base_entries, head_entries, changed = _remote_tree_fixture(
        state, unexpected=unexpected
    )
    branch = "repopilot-fix-7-aaaaaaaaaaaa-bbbbbbbbbbbbbbbb"
    head_sha = "d" * 40

    async def get_ref(_state, requested_branch):
        assert requested_branch == branch
        return {
            "ref": f"refs/heads/{branch}",
            "object": {"sha": head_sha},
        }

    async def get_commit(_state, sha):
        tree_sha = "a" * 40 if sha == state.repo_ref else "b" * 40
        return {"sha": sha, "tree": {"sha": tree_sha}}

    async def get_tree(_state, tree_sha):
        tree = base_entries if tree_sha == "a" * 40 else head_entries
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
async def test_create_pr_reverifies_remote_branch_before_api_call(
    tmp_path, monkeypatch
):
    state = _state(tmp_path)
    _attach_proof(state)
    state.branch_name = "repopilot-fix-7-aaaaaaaaaaaa-bbbbbbbbbbbbbbbb"
    state.base_branch = "main"
    calls = []

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
        await create_pr(state)

    assert calls == ["verify"]
