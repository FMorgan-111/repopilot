from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

from src.nodes.commit import push_files
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

    async def ref(_state, _branch):
        return {"object": {"sha": state.repo_ref}}

    async def create_ref(_state, branch, sha):
        return {"ref": branch, "sha": sha}

    monkeypatch.setattr("src.nodes.commit._github_get_repo", repo)
    monkeypatch.setattr("src.nodes.commit._github_get_ref", ref)
    monkeypatch.setattr("src.nodes.commit._github_create_ref", create_ref)


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
